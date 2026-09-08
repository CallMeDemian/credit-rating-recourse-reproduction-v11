from __future__ import annotations

"""Static + contract-faithful synthetic verifier for the C4R matched runner."""

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.analysis.c4r_matched_inference import run_analysis
from credit_recourse.verification.smoke_stage7_c4r_contract import run_smoke

SCHEMA_VERSION = "c4r_matched_contract_v2"


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_synthetic_archive(root: Path, *, label: str, budget: float | None) -> Path:
    run = root / label
    s7 = run / "stage7_llm_action_generation"
    s8 = run / "stage8_llm_multi_oracle_eval"
    s9 = run / "stage9_llm_rl_comparison"
    for path in (s7, s8, s9):
        path.mkdir(parents=True, exist_ok=True)
    _write_json(run / "archive_manifest.json", {
        "run_label": label,
        "run_role": "paper_c4r_matched_icb",
        "file_count": 4,
    })
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
    _write_json(s7 / "metadata.json", {
        "status": "PASS",
        "backend_is_live": True,
        "backend_id": "openai_gpt-5.4-mini_responses_reasoning-low_maxout-1200",
        "information_condition": "IC-b",
        "conditions": ["C4", "C4R", "C6"],
        "modes": ["free_form_10d"],
        "candidate_library_hash": "synthetic_p50_hash",
        "action_budget_contract": budget_contract,
    })
    score_rows = []
    revision_rows = []
    for rid in range(575):
        # Deterministic, non-degenerate paired effects matching the intended
        # capacity-conditional pattern: both self-revision and shown-reference
        # increments are weaker under L1<=0.75 than under no cap.  Every
        # interaction is non-zero so the synthetic verifier exercises the
        # Wilcoxon/Holm producer rather than only checking table shape.
        c4_alpha = (rid % 11) / 100.0
        jitter_self = ((rid % 5) - 2) / 10000.0
        jitter_ref = ((rid % 7) - 3) / 10000.0
        self_alpha = (-0.04 if budget is not None else 0.03) + jitter_self
        ref_alpha = (0.03 if budget is not None else 0.07) + jitter_ref
        values = {
            "alpha": (c4_alpha, c4_alpha + self_alpha, c4_alpha + self_alpha + ref_alpha),
            "beta": (c4_alpha / 4.0, c4_alpha / 4.0 + self_alpha / 2.0, c4_alpha / 4.0 + self_alpha / 2.0 + ref_alpha / 4.0),
            "gamma": (c4_alpha * 0.8, c4_alpha * 0.8 + self_alpha * 0.8, c4_alpha * 0.8 + self_alpha * 0.8 + ref_alpha * 0.8),
        }
        for policy_index, policy in enumerate(("C4", "C4R", "C6")):
            rec = {"row_id": rid, "policy": policy, "mode": "free_form_10d"}
            for oracle in ("alpha", "beta", "gamma"):
                rec[f"delta_R_score_{oracle}"] = values[oracle][policy_index]
            score_rows.append(rec)
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
    # Preserve the production file contract.  When a parquet engine is
    # available, write a real parquet file; otherwise write the explicit CSV
    # sidecar that c4r_matched_inference._read_parquet_or_csv accepts.  Never
    # place CSV bytes under a .parquet suffix: with pyarrow installed pandas
    # correctly treats that as a corrupt parquet file and raises ArrowInvalid.
    score_frame = pd.DataFrame(score_rows)
    parquet_path = s8 / "llm_stage8_multi_oracle_scores.parquet"
    try:
        score_frame.to_parquet(parquet_path, index=False)
    except ImportError:
        score_frame.to_csv(parquet_path.with_suffix(".csv"), index=False)
    pd.DataFrame(revision_rows).to_csv(s9 / "llm_stage9_revision_metrics.csv", index=False)
    return run


def verify(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    source_files = [
        root / "src" / "credit_recourse" / "analysis" / "c4r_matched_inference.py",
        root / "src" / "credit_recourse" / "verification" / "smoke_stage7_c4r_contract.py",
        root / "tools" / "run_n5_budget_frontier_icb.ps1",
        root / "tools" / "run_thesis_repro.ps1",
    ]
    errors: list[str] = []
    for path in source_files:
        if not path.is_file():
            errors.append(f"required C4R implementation missing: {path}")
    inference_source = source_files[0]
    if inference_source.is_file():
        inference_text = inference_source.read_text(encoding="utf-8-sig")
        for marker in (
            "firm_paired_finite_minus_unbounded_DID",
            "contrast_across_three_oracles",
            "wilcoxon_p_holm_three_oracles",
            "_interaction_table(firm_frame)",
        ):
            if marker not in inference_text:
                errors.append(f"C4R paired-DID inference marker missing: {marker}")
    runner = source_files[-2]
    thesis_runner = source_files[-1]
    if runner.is_file():
        text = runner.read_text(encoding="utf-8-sig")
        required_markers = (
            '[switch]$C4RMatched',
            '$Conditions = if ($C4RMatched) { @("C4", "C4R", "C6") }',
            '$BudgetedConditionCsv = if ($C4RMatched) { "C4,C4R,C6" }',
            'same_batch_required = if ($C4RMatched) { $true }',
            'c4_reuse_from_prior_snapshot_forbidden = if ($C4RMatched) { $true }',
            'Assert-C4R-ArchivedContract',
            'Expected-BackendId',
            'credit_recourse.analysis.c4r_matched_inference',
            'credit_recourse.verification.smoke_stage7_c4r_contract',
        )
        for marker in required_markers:
            if marker not in text:
                errors.append(f"integrated C4R runner marker missing: {marker}")
        for forbidden in ('ErrorActionPreference = "Continue"', '"--skip-stage7"'):
            if forbidden in text:
                errors.append(f"integrated C4R runner contains forbidden fallback/reuse marker: {forbidden}")
    if thesis_runner.is_file():
        thesis_text = thesis_runner.read_text(encoding="utf-8-sig")
        for marker in ('[switch]$C4RMatched', '"-C4RMatched"', 'run_n5_budget_frontier_icb.ps1'):
            if marker not in thesis_text:
                errors.append(f"top-level C4R integration marker missing: {marker}")

    smoke_result = {}
    try:
        smoke_result = run_smoke()
        if smoke_result.get("status") != "PASS":
            errors.append(f"C4R prompt/revision smoke did not PASS: {smoke_result}")
    except Exception as exc:
        errors.append(f"C4R prompt/revision smoke failed: {type(exc).__name__}: {exc}")

    synthetic_result: dict = {}
    try:
        with tempfile.TemporaryDirectory(prefix="c4r_contract_") as tmp:
            base = Path(tmp)
            finite = _make_synthetic_archive(base, label="finite_0p75", budget=0.75)
            unbounded = _make_synthetic_archive(base, label="unbounded", budget=None)
            synthetic_result = run_analysis(
                arm_dirs=[finite, unbounded],
                out_dir=base / "analysis",
            )
            if synthetic_result.get("status") != "PASS":
                errors.append("C4R synthetic matched inference did not PASS")
            if int(synthetic_result.get("firm_frame_row_count", -1)) != 1150:
                errors.append("C4R synthetic firm frame must contain 1,150 rows")
            if int(synthetic_result.get("contrast_row_count", -1)) != 18:
                errors.append("C4R synthetic contrast table must contain 18 rows")
            if int(synthetic_result.get("interaction_row_count", -1)) != 9:
                errors.append("C4R synthetic interaction table must contain 9 rows")
            interaction_path = base / "analysis" / "c4r_matched_interactions.csv"
            if not interaction_path.is_file():
                errors.append("C4R synthetic interaction output is missing")
            else:
                interaction = pd.read_csv(interaction_path)
                required_columns = {
                    "oracle_backend", "contrast", "finite_budget_label",
                    "unbounded_budget_label", "n_pairs",
                    "finite_minus_unbounded_interaction", "wilcoxon_p_raw",
                    "wilcoxon_p_holm_three_oracles", "sig_holm", "holm_family",
                }
                missing_columns = sorted(required_columns - set(interaction.columns))
                if missing_columns:
                    errors.append(f"C4R synthetic interaction columns missing: {missing_columns}")
                if len(interaction) == 9:
                    expected_grid = {
                        (contrast, oracle)
                        for contrast in (
                            "self_revision",
                            "reference_content_conditional",
                            "reference_plus_revision_package",
                        )
                        for oracle in ("alpha", "beta", "gamma")
                    }
                    observed_grid = set(map(tuple, interaction[["contrast", "oracle_backend"]].astype(str).itertuples(index=False, name=None)))
                    if observed_grid != expected_grid:
                        errors.append("C4R synthetic interaction grid is not exact 3 components x 3 Oracles")
                    if not pd.to_numeric(interaction["n_pairs"], errors="coerce").eq(575).all():
                        errors.append("C4R synthetic interactions are not based on 575 paired firms")
                    if not pd.to_numeric(interaction["finite_minus_unbounded_interaction"], errors="coerce").lt(0).all():
                        errors.append("C4R synthetic finite-minus-unbounded interactions must all be negative")
                    if not pd.to_numeric(interaction["wilcoxon_p_holm_three_oracles"], errors="coerce").le(0.05).all():
                        errors.append("C4R synthetic paired-DID Holm results must all be significant")
                    if not interaction["holm_family"].astype(str).eq("contrast_across_three_oracles").all():
                        errors.append("C4R synthetic interaction Holm family mismatch")
                contract = synthetic_result.get("interaction_inference_contract") or {}
                if contract.get("unit") != "firm_paired_finite_minus_unbounded_DID":
                    errors.append("C4R manifest missing firm-paired DID unit")
                if contract.get("holm_family") != "contrast_across_three_oracles" or int(contract.get("holm_family_size", -1)) != 3:
                    errors.append("C4R manifest interaction Holm contract mismatch")
    except Exception as exc:
        errors.append(f"C4R synthetic matched inference failed: {type(exc).__name__}: {exc}")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "source_files": [str(path) for path in source_files],
        "prompt_revision_smoke": smoke_result,
        "synthetic_inference": {
            "status": synthetic_result.get("status"),
            "firm_frame_row_count": synthetic_result.get("firm_frame_row_count"),
            "contrast_row_count": synthetic_result.get("contrast_row_count"),
            "interaction_row_count": synthetic_result.get("interaction_row_count"),
            "interaction_inference_contract": synthetic_result.get("interaction_inference_contract"),
        },
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(Path(args.project_root))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
