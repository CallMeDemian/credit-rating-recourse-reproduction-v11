from __future__ import annotations

"""Static and contract-faithful synthetic verifier for C4R matched inference v3.

The verifier exercises:
* two independent backend cohorts;
* four budget arms per cohort;
* exact C4/C4R/C6 row and additive-identity contracts;
* primary three-Oracle and sensitivity nine-test Holm families;
* legacy v2 two-arm compatibility, including byte-identical CSV projections;
* optional regression against a real frozen v2 analysis directory.
"""

import argparse
import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.analysis.c4r_matched_inference import run_analysis as run_v2
from credit_recourse.analysis.c4r_matched_inference_v3 import (
    ArmSpec,
    BudgetSpec,
    _v2_compat_projection,
    run_analysis as run_v3,
)
from credit_recourse.verification.verify_c4r_journal_arm_contract import verify_arm
from credit_recourse.verification.verify_gemini_stage7_backend_contract import (
    verify as verify_gemini_backend,
)

SCHEMA_VERSION = "c4r_matched_v3_contract_v1"
BUDGETS: tuple[tuple[str, float | None], ...] = (
    ("0p75", 0.75),
    ("1p27", 1.27),
    ("2p00", 2.00),
    ("unbounded", None),
)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _effect_for_budget(budget_label: str, *, backend_scale: float) -> tuple[float, float]:
    self_map = {"0p75": -0.040, "1p27": -0.010, "2p00": 0.010, "unbounded": 0.030}
    ref_map = {"0p75": 0.030, "1p27": 0.045, "2p00": 0.055, "unbounded": 0.070}
    return self_map[budget_label] * backend_scale, ref_map[budget_label] * backend_scale


def _make_synthetic_archive(
    root: Path,
    *,
    label: str,
    run_role: str,
    backend_id: str,
    budget_label: str,
    budget: float | None,
    backend_scale: float = 1.0,
    live: bool = True,
) -> Path:
    run = root / label
    s7 = run / "stage7_llm_action_generation"
    s8 = run / "stage8_llm_multi_oracle_eval"
    s9 = run / "stage9_llm_rl_comparison"
    for path in (s7, s8, s9):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(
        run / "archive_manifest.json",
        {"run_label": label, "run_role": run_role, "file_count": 5},
    )
    budget_contract = (
        {
            "enabled": True,
            "l1_budget": float(budget),
            "budgeted_conditions": ["C4", "C4R", "C6"],
            "budgeted_modes": ["free_form_10d"],
        }
        if budget is not None
        else {"enabled": False}
    )
    payload_path = s7 / "llm_stage7_prompt_payloads.jsonl"
    payload_path.write_text(
        "".join(
            json.dumps({"row_id": rid, "condition": policy, "mode": "free_form_10d"}, sort_keys=True) + "\n"
            for rid in range(575) for policy in ("C4", "C4R", "C6")
        ),
        encoding="utf-8",
    )
    payload_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    _write_json(s7 / "llm_stage7_prompt_manifest.json", {
        "prompt_payload_archive": {
            "schema_version": "stage7_prompt_payload_archive_v1",
            "path": payload_path.name,
            "sha256": payload_sha,
            "record_count": 1725,
        }
    })
    _write_json(
        s7 / "metadata.json",
        {
            "status": "PASS" if live else "PASS_REPRODUCIBILITY_BACKEND",
            "backend_is_live": live,
            "backend_id": backend_id,
            "information_condition": "IC-b",
            "conditions": ["C4", "C4R", "C6"],
            "modes": ["free_form_10d"],
            "candidate_library_hash": "synthetic_p50_hash",
            "action_budget_contract": budget_contract,
            "prompt_payload_archive_contract": {
                "schema_version": "stage7_prompt_payload_archive_v1",
                "status": "FULL_PAYLOAD_ARCHIVED",
                "path": payload_path.name,
                "sha256": payload_sha,
                "record_count": 1725,
            },
        },
    )
    self_alpha, ref_alpha = _effect_for_budget(budget_label, backend_scale=backend_scale)
    score_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    revision_rows: list[dict[str, Any]] = []
    for rid in range(575):
        c4_alpha = (rid % 11) / 100.0
        jitter_self = ((rid % 5) - 2) / 10000.0
        jitter_ref = ((rid % 7) - 3) / 10000.0
        self_effect = self_alpha + jitter_self
        ref_effect = ref_alpha + jitter_ref
        values = {
            "alpha": (c4_alpha, c4_alpha + self_effect, c4_alpha + self_effect + ref_effect),
            "beta": (
                c4_alpha / 4.0,
                c4_alpha / 4.0 + self_effect / 2.0,
                c4_alpha / 4.0 + self_effect / 2.0 + ref_effect / 4.0,
            ),
            "gamma": (
                c4_alpha * 0.8,
                c4_alpha * 0.8 + self_effect * 0.8,
                c4_alpha * 0.8 + self_effect * 0.8 + ref_effect * 0.8,
            ),
        }
        for policy_index, policy in enumerate(("C4", "C4R", "C6")):
            rec: dict[str, Any] = {"row_id": rid, "policy": policy, "mode": "free_form_10d"}
            for oracle in ("alpha", "beta", "gamma"):
                rec[f"delta_R_score_{oracle}"] = values[oracle][policy_index]
            score_rows.append(rec)
            action_rec: dict[str, Any] = {
                "row_id": rid,
                "policy": policy,
                "candidate_id": "synthetic",
                "mode": "free_form_10d",
                "budgeted_condition_flag": budget is not None,
                "budget_compliant_raw": True if budget is not None else None,
                "budget_compliant_clipped": True if budget is not None else None,
                "budget_l1_target": budget,
                "budget_l1_raw": (0.5 * budget if budget is not None else None),
                "budget_l1_clipped": (0.5 * budget if budget is not None else None),
            }
            for axis_index, axis in enumerate((
                "action__ppe_pct", "action__inv_turnover_chg", "action__ar_turnover_chg",
                "action__ap_turnover_chg", "action__short_debt_pct", "action__long_debt_pct",
                "action__bond_pct", "action__revenue_growth", "action__cogs_ratio_chg",
                "action__sga_ratio_chg",
            )):
                action_rec[axis] = float((rid + axis_index + policy_index) % 7) / 100.0
            action_rows.append(action_rec)
        for policy in ("C4R", "C6"):
            revised_index = 1 if policy == "C4R" else 2
            rec = {
                "row_id": rid,
                "base_condition": "C4",
                "revision_condition": policy,
                "mode": "free_form_10d",
                "reference_source": "none" if policy == "C4R" else "rl",
                "metrics_defined": False if policy == "C4R" else True,
                "undefined_reason": "no_rl_reference" if policy == "C4R" else "",
            }
            for oracle in ("alpha", "beta", "gamma"):
                rec[f"revision_delta_R_score_{oracle}"] = values[oracle][revised_index] - values[oracle][0]
            revision_rows.append(rec)
    pd.DataFrame(action_rows).to_parquet(s7 / "llm_stage7_action_table.parquet", index=False)
    pd.DataFrame({"row_id": [], "failure_categories": []}).to_csv(
        s7 / "llm_stage7_failure_audit.csv", index=False
    )
    pd.DataFrame(score_rows).to_parquet(s8 / "llm_stage8_multi_oracle_scores.parquet", index=False)
    pd.DataFrame(revision_rows).to_csv(s9 / "llm_stage9_revision_metrics.csv", index=False)
    return run


def _compare_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    label: str,
    check_exact: bool = True,
    atol: float = 0.0,
) -> list[str]:
    errors: list[str] = []
    if list(left.columns) != list(right.columns):
        errors.append(f"{label} columns differ: left={list(left.columns)}, right={list(right.columns)}")
        return errors
    if left.shape != right.shape:
        errors.append(f"{label} shape differs: left={left.shape}, right={right.shape}")
        return errors
    try:
        pd.testing.assert_frame_equal(
            left, right, check_exact=check_exact, check_dtype=False, rtol=0.0, atol=atol
        )
    except AssertionError as exc:
        errors.append(f"{label} frame differs: {exc}")
    return errors


def _real_v2_output_regression(path: Path) -> dict[str, Any]:
    analysis = Path(path).resolve()
    firm_path = analysis / "c4r_matched_firm_frame.parquet"
    if not firm_path.is_file():
        csv_path = analysis / "c4r_matched_firm_frame.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"legacy v2 firm frame missing under {analysis}")
        firm = pd.read_csv(csv_path)
    else:
        firm = pd.read_parquet(firm_path)
    # Add only the cohort metadata required by the v3 projection helper.
    augmented = firm.copy()
    augmented.insert(0, "cohort_id", "frozen_e2")
    augmented.insert(1, "backend_id", "legacy_v2_unknown")
    augmented.insert(2, "run_role", "paper_c4r_matched_icb")
    _, generated_contrasts, generated_interactions = _v2_compat_projection(augmented)
    frozen_contrasts = pd.read_csv(analysis / "c4r_matched_contrasts.csv")
    frozen_interactions = pd.read_csv(analysis / "c4r_matched_interactions.csv")
    # The frozen CSV stores decimal text while the firm-frame parquet retains
    # binary float precision. Recomputing from parquet can differ below 1e-15;
    # this is not a statistical or contract difference. Synthetic v2/v3 CSVs
    # are still required to be byte-identical elsewhere in this verifier.
    errors = _compare_frames(
        frozen_contrasts, generated_contrasts, label="real v2 contrasts",
        check_exact=False, atol=1e-15
    )
    errors.extend(_compare_frames(
        frozen_interactions, generated_interactions, label="real v2 interactions",
        check_exact=False, atol=1e-15
    ))
    return {
        "status": "PASS" if not errors else "FAIL",
        "analysis_dir": str(analysis),
        "firm_rows": int(len(firm)),
        "contrast_rows": int(len(frozen_contrasts)),
        "interaction_rows": int(len(frozen_interactions)),
        "errors": errors,
    }


def verify(project_root: Path, *, legacy_v2_analysis_dir: Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_files = [
        root / "src" / "credit_recourse" / "analysis" / "c4r_matched_inference.py",
        root / "src" / "credit_recourse" / "analysis" / "c4r_matched_inference_v3.py",
        root / "src" / "credit_recourse" / "verification" / "verify_c4r_matched_v3_contract.py",
        root / "tools" / "run_c4r_journal_grid.ps1",
        root / "src" / "credit_recourse" / "configs" / "c4r_journal_extension_prereg_v1.json",
        root / "src" / "credit_recourse" / "configs" / "c4r_journal_extension_prereg_v2.json",
        root / "src" / "credit_recourse" / "configs" / "c4r_journal_extension_prereg_v3.json",
        root / "src" / "credit_recourse" / "verification" / "verify_gemini_stage7_backend_contract.py",
    ]
    errors: list[str] = []
    for path in source_files:
        if not path.is_file():
            errors.append(f"required C4R v3 artifact missing: {path}")
    v2_hash_before = None
    if source_files[0].is_file():
        import hashlib

        v2_hash_before = hashlib.sha256(source_files[0].read_bytes()).hexdigest()
    if source_files[1].is_file():
        source = source_files[1].read_text(encoding="utf-8-sig")
        for marker in (
            "cohort_finite_arm_contrast_across_three_oracles",
            "cohort_contrast_across_all_finite_arms_and_oracles",
            "Stage8/9 revision score mismatch",
            "_v2_compat_projection",
            "candidate-library hash mismatch across journal grid",
        ):
            if marker not in source:
                errors.append(f"C4R v3 source marker missing: {marker}")
    if source_files[3].is_file():
        runner_text = source_files[3].read_text(encoding="utf-8-sig")
        for marker in (
            "credit_recourse.analysis.c4r_matched_inference_v3",
            "credit_recourse.verification.verify_c4r_matched_v3_contract",
            "FULL_PAYLOAD_ARCHIVED",
            "c4r_journal_extension_prereg_v3.json",
            "verify_gemini_stage7_backend_contract",
            "ReuseGridManifest",
            "GEMINI_API_KEY",
            "PASS_REUSED",
            "C4,C4R,C6",
        ):
            if marker not in runner_text:
                errors.append(f"journal grid runner marker missing: {marker}")
        for forbidden in ('ErrorActionPreference = "Continue"', '"--skip-stage7"'):
            if forbidden in runner_text:
                errors.append(f"journal grid runner contains forbidden fallback/reuse marker: {forbidden}")

    gemini_contract: dict[str, Any] = {}
    try:
        gemini_contract = verify_gemini_backend(root)
        if gemini_contract.get("status") != "PASS":
            errors.append(
                "Gemini Stage7 backend contract did not PASS: "
                f"{gemini_contract.get('errors')}"
            )
    except Exception as exc:
        errors.append(
            f"Gemini Stage7 backend contract invocation failed: {type(exc).__name__}: {exc}"
        )

    synthetic_result: dict[str, Any] = {}
    legacy_regression: dict[str, Any] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="c4r_v3_contract_") as tmp:
            base = Path(tmp)
            arm_specs: list[ArmSpec] = []
            roles = {
                "gpt54mini": "paper_c4r_journal_ext_gpt54mini",
                "gemini31flashlite": "paper_c4r_journal_ext_gemini31flashlite",
            }
            for cohort, backend_id, scale in (
                ("gpt54mini", "openai_gpt-5.4-mini_responses_reasoning-low_maxout-1200", 1.0),
                (
                    "gemini31flashlite",
                    "google_gemini-3.1-flash-lite_generate-content_thinking-low_maxout-4096_json",
                    0.90,
                ),
            ):
                for budget_label, budget in BUDGETS:
                    archive = _make_synthetic_archive(
                        base,
                        label=f"{cohort}_{budget_label}",
                        run_role=roles[cohort],
                        backend_id=backend_id,
                        budget_label=budget_label,
                        budget=budget,
                        backend_scale=scale,
                    )
                    arm_specs.append(ArmSpec(cohort, budget_label, archive))
                    arm_qc = verify_arm(
                        archive_dir=archive,
                        cohort_id=cohort,
                        budget_label=budget_label,
                        l1_budget=budget,
                        expected_run_role=roles[cohort],
                        expected_backend_id=backend_id,
                        expected_firm_count=575,
                        require_live=True,
                    )
                    if arm_qc.get("status") != "PASS":
                        errors.append(
                            f"synthetic journal arm verifier failed cohort={cohort} budget={budget_label}: "
                            f"{arm_qc.get('errors')}"
                        )
            synthetic_result = run_v3(
                arm_specs=arm_specs,
                out_dir=base / "analysis_v3",
                budget_specs=[BudgetSpec(label, value) for label, value in BUDGETS],
                expected_run_roles=roles,
                expected_firm_count=575,
                require_live=True,
            )
            if synthetic_result.get("status") != "PASS":
                errors.append("C4R v3 synthetic 2-cohort/4-arm analysis did not PASS")
            expected_total = {"firm_frame_row_count": 4600, "contrast_row_count": 72, "interaction_row_count": 54}
            for key, expected in expected_total.items():
                if int(synthetic_result.get(key, -1)) != expected:
                    errors.append(f"C4R v3 synthetic {key} expected={expected}, observed={synthetic_result.get(key)}")
            cohort_summary = pd.read_csv(base / "analysis_v3" / "c4r_matched_v3_cohort_summary.csv")
            if len(cohort_summary) != 2:
                errors.append("C4R v3 cohort summary must contain two cohorts")
            else:
                for _, row in cohort_summary.iterrows():
                    if (int(row["firm_frame_row_count"]), int(row["contrast_row_count"]), int(row["interaction_row_count"])) != (2300, 36, 27):
                        errors.append(f"C4R v3 per-cohort row formula mismatch: {row.to_dict()}")
            interactions = pd.read_csv(base / "analysis_v3" / "c4r_matched_v3_interactions.csv")
            required_columns = {
                "cohort_id",
                "finite_budget_label",
                "contrast",
                "oracle_backend",
                "finite_minus_unbounded_interaction",
                "wilcoxon_p_holm_three_oracles",
                "wilcoxon_p_holm_component_all_finite_oracles",
            }
            missing = sorted(required_columns - set(interactions.columns))
            if missing:
                errors.append(f"C4R v3 interaction columns missing: {missing}")
            if not pd.to_numeric(interactions["finite_minus_unbounded_interaction"], errors="coerce").lt(0).all():
                errors.append("C4R v3 synthetic interactions must all be negative")
            if not pd.to_numeric(interactions["n_pairs"], errors="coerce").eq(575).all():
                errors.append("C4R v3 synthetic interactions must all use 575 firm pairs")

            # Exact v2 compatibility on a separately generated old-role two-arm cohort.
            finite = _make_synthetic_archive(
                base,
                label="legacy_finite_0p75",
                run_role="paper_c4r_matched_icb",
                backend_id="openai_gpt-5.4-mini_responses_reasoning-low_maxout-1200",
                budget_label="0p75",
                budget=0.75,
            )
            unbounded = _make_synthetic_archive(
                base,
                label="legacy_unbounded",
                run_role="paper_c4r_matched_icb",
                backend_id="openai_gpt-5.4-mini_responses_reasoning-low_maxout-1200",
                budget_label="unbounded",
                budget=None,
            )
            v2_out = base / "legacy_v2"
            v3_out = base / "legacy_v3"
            v2_result = run_v2(arm_dirs=[finite, unbounded], out_dir=v2_out)
            v3_result = run_v3(
                arm_specs=[ArmSpec("legacy", "0p75", finite), ArmSpec("legacy", "unbounded", unbounded)],
                out_dir=v3_out,
                budget_specs=[BudgetSpec("0p75", 0.75), BudgetSpec("unbounded", None)],
                expected_run_roles={"legacy": "paper_c4r_matched_icb"},
                expected_firm_count=575,
                require_live=True,
                write_v2_compat_projection=True,
            )
            compat_dir = v3_out / "v2_compat" / "legacy"
            byte_checks = {}
            for name in ("c4r_matched_contrasts.csv", "c4r_matched_interactions.csv"):
                left = (v2_out / name).read_bytes()
                right = (compat_dir / name).read_bytes()
                byte_checks[name] = left == right
                if left != right:
                    errors.append(f"v3 compatibility projection is not byte-identical to v2 for {name}")
            v2_firm = pd.read_parquet(v2_out / "c4r_matched_firm_frame.parquet")
            v3_firm = pd.read_parquet(compat_dir / "c4r_matched_firm_frame.parquet")
            firm_errors = _compare_frames(v2_firm, v3_firm, label="synthetic v2/v3 firm frame")
            errors.extend(firm_errors)
            legacy_regression = {
                "status": "PASS" if all(byte_checks.values()) and not firm_errors else "FAIL",
                "v2_status": v2_result.get("status"),
                "v3_status": v3_result.get("status"),
                "byte_identical_csv": byte_checks,
                "firm_frame_data_identical": not firm_errors,
            }
    except Exception as exc:
        errors.append(f"C4R v3 synthetic verification failed: {type(exc).__name__}: {exc}")

    real_regression: dict[str, Any] | None = None
    if legacy_v2_analysis_dir is not None:
        try:
            real_regression = _real_v2_output_regression(legacy_v2_analysis_dir)
            errors.extend(real_regression.get("errors") or [])
        except Exception as exc:
            real_regression = {
                "status": "FAIL",
                "analysis_dir": str(Path(legacy_v2_analysis_dir).resolve()),
                "errors": [f"{type(exc).__name__}: {exc}"],
            }
            errors.extend(real_regression["errors"])

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "source_files": [str(path) for path in source_files],
        "frozen_v2_source_sha256": v2_hash_before,
        "synthetic_v3": {
            "status": synthetic_result.get("status"),
            "cohort_count": synthetic_result.get("cohort_count"),
            "firm_frame_row_count": synthetic_result.get("firm_frame_row_count"),
            "contrast_row_count": synthetic_result.get("contrast_row_count"),
            "interaction_row_count": synthetic_result.get("interaction_row_count"),
            "interaction_inference_contract": synthetic_result.get("interaction_inference_contract"),
        },
        "gemini_stage7_backend_contract": {
            "status": gemini_contract.get("status"),
            "expected_model": gemini_contract.get("expected_model"),
            "expected_backend_id": gemini_contract.get("expected_backend_id"),
            "synthetic_runtime": gemini_contract.get("synthetic_runtime"),
        },
        "synthetic_v2_compatibility": legacy_regression,
        "real_v2_output_regression": real_regression,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--legacy-v2-analysis-dir", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(
        Path(args.project_root),
        legacy_v2_analysis_dir=(Path(args.legacy_v2_analysis_dir) if args.legacy_v2_analysis_dir else None),
    )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
