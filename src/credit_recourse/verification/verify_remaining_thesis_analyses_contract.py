from __future__ import annotations

"""Synthetic regression and runner-contract checks for remaining thesis analyses."""

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.analysis.reference_quality_acceptance import (
    COHORT_CONTRACT,
    REFERENCE_LOOKUP_CONTRACT,
    _build_stage6_reference_lookup,
    run_analysis,
)
from credit_recourse.analysis.paper_output_layout import build_layout, ensure_layout
from credit_recourse.analysis.remaining_thesis_analyses import (
    _archive_stale_industry_cell,
    _run_frontier,
    _shuffle_cell_ok,
)
from credit_recourse.contracts.paper_reproduction import ProfileError, load_profile

SCHEMA_VERSION = "remaining_thesis_analyses_contract_v2"


def _write_revision_run(root: Path, *, n: int = 40) -> Path:
    run = root / "data" / "final_freeze" / "llm_runs" / "synthetic_primary"
    s7 = run / "stage7_llm_action_generation"
    s9 = run / "stage9_llm_rl_comparison"
    s7.mkdir(parents=True, exist_ok=True)
    s9.mkdir(parents=True, exist_ok=True)
    (s7 / "metadata.json").write_text(json.dumps({
        "status": "PASS",
        "run_label": "synthetic_primary",
        "information_condition": "IC-b",
    }), encoding="utf-8")

    stage7_rows = []
    stage9_rows = []
    for rid in range(n):
        ref = "A0_noop" if rid % 10 == 0 else ("REF_GOOD" if rid % 2 == 0 else "REF_WEAK")
        advantage = (rid + 1) / n if ref in {"A0_noop", "REF_GOOD"} else -(rid + 1) / (2 * n)
        adoption = 0.25 + 0.6 * ((advantage + 0.5) / 1.5)
        geometry_defined = rid % 13 != 0
        for mode in ("candidate_selection", "free_form_10d"):
            available = {"C4", "C6", "C6X", "C7"}
            if mode != "free_form_10d" or rid < n - 3:
                available.add("C5")
            for condition in sorted(available):
                stage7_rows.append({
                    "row_id": rid,
                    "policy": condition,
                    "mode": mode,
                })
            for condition, base in (("C6", "C4"), ("C6X", "C4"), ("C7", "C5")):
                if base not in available or condition not in available:
                    continue
                rec = {
                    "row_id": rid,
                    "base_condition": base,
                    "revision_condition": condition,
                    "mode": mode,
                    "rl_reference_candidate": ref,
                    "reference_source": "rl" if condition != "C6X" else "random",
                    "reference_draw_seed": 1,
                    "initial_candidate_id": "INIT",
                    "revised_candidate_id": ref,
                    "revision_changed_candidate_flag": True,
                    "revision_changed_active_dimensions": 2,
                    "revision_l1_distance": abs(adoption),
                    "revision_l2_distance": abs(adoption) / 2,
                    "revision_cosine_distance": 0.1,
                    "rl_adoption_ratio": adoption if geometry_defined else float("nan"),
                    "self_retention_ratio": (1.0 - adoption) if geometry_defined else float("nan"),
                    "orthogonal_drift": 0.05 if geometry_defined else float("nan"),
                    "u_norm_squared": 1.0 if geometry_defined else 0.0,
                    "metrics_defined": geometry_defined,
                    "undefined_reason": (
                        None if geometry_defined
                        else "rl_reference_equals_a0_in_normalized_space"
                    ),
                }
                for j, bk in enumerate(("alpha", "beta", "gamma")):
                    scale = 1.0 - j * 0.1
                    if ref == "A0_noop":
                        ref_value = 0.0
                        initial = -advantage * scale
                    else:
                        initial = 0.1 * j
                        ref_value = initial + advantage * scale
                    revised = initial + adoption * (ref_value - initial)
                    rec[f"initial_delta_R_score_{bk}"] = initial
                    rec[f"revised_delta_R_score_{bk}"] = revised
                    rec[f"revision_delta_R_score_{bk}"] = revised - initial
                stage9_rows.append(rec)
    pd.DataFrame(stage7_rows).to_parquet(s7 / "llm_stage7_action_table.parquet", index=False)
    pd.DataFrame(stage9_rows).to_csv(s9 / "llm_stage9_revision_metrics.csv", index=False)
    return run


def _write_stage6(root: Path, *, n: int = 40) -> Path:
    out = root / "data" / "final_freeze" / "stage6_candidate_selector_eval"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid in range(n):
        rows.append({
            "row_id": rid,
            "policy": "C0_noop",
            "candidate_id": "A0_noop",
            "delta_R_score_alpha": 0.0,
            "delta_R_score_beta": 0.0,
            "delta_R_score_gamma": 0.0,
        })
        # A row-conditional RL policy may select A0_noop too. It must not
        # create an ambiguous fixed-candidate lookup or replace the C0 row.
        rows.append({
            "row_id": rid,
            "policy": "C3_candidate_iql",
            "candidate_id": "A0_noop" if rid % 10 == 0 else "REF_GOOD",
            "delta_R_score_alpha": 0.0,
            "delta_R_score_beta": 0.0,
            "delta_R_score_gamma": 0.0,
        })
        for policy in ("REF_GOOD", "REF_WEAK"):
            sign = 1.0 if policy == "REF_GOOD" else -0.5
            magnitude = (rid + 1) / n
            rows.append({
                "row_id": rid,
                "policy": policy,
                "candidate_id": policy,
                "delta_R_score_alpha": sign * magnitude,
                "delta_R_score_beta": sign * magnitude * 0.9 + 0.1,
                "delta_R_score_gamma": sign * magnitude * 0.8 + 0.2,
            })
    path = out / "multi_oracle_policy_eval.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def verify(project_root: Path) -> dict:
    project_root = Path(project_root).resolve()
    errors: list[str] = []
    runner = project_root / "tools" / "run_thesis_repro.ps1"
    if not runner.is_file():
        errors.append(f"missing runner: {runner}")
    else:
        text = runner.read_text(encoding="utf-8-sig")
        for token in ("RemainingNoApi", "BudgetFrontier", "RemainingAll", "MatchedBudgetC4"):
            if token not in text:
                errors.append(f"run_thesis_repro missing Task token: {token}")
        if 'BudgetFrontier spends live OpenAI API calls' not in text:
            errors.append("BudgetFrontier is not guarded by explicit API-spend confirmation")
        if 'remaining_thesis_analyses' not in text:
            errors.append("run_thesis_repro does not invoke remaining_thesis_analyses")

    # Plan-only frontier validation must succeed before any live arm archive
    # exists. Partial archives remain a hard failure in the shared selector.
    with tempfile.TemporaryDirectory(prefix="verify_frontier_plan_only_") as tmp:
        plan_root = Path(tmp)
        profile_src = (
            project_root
            / "src"
            / "credit_recourse"
            / "configs"
            / "paper_reproduction_profile.json"
        )
        profile_dst = (
            plan_root
            / "src"
            / "credit_recourse"
            / "configs"
            / "paper_reproduction_profile.json"
        )
        profile_dst.parent.mkdir(parents=True, exist_ok=True)
        profile_dst.write_bytes(profile_src.read_bytes())
        profile = load_profile(plan_root)
        layout = build_layout(plan_root / profile["workspace"]["analysis_root"])
        ensure_layout(layout)
        planned_by_design: dict[str, dict] = {}
        expected_contracts = {
            "legacy_c6_only": {
                "profile_key": "n5_budget_frontier",
                "run_role": "paper_n5_budget_frontier_icb",
            },
            "matched_c4_c6": {
                "profile_key": "n5_matched_budget_frontier",
                "run_role": "paper_n5_matched_budget_frontier_icb",
            },
        }
        for frontier_design, expected in expected_contracts.items():
            planned = _run_frontier(
                root=plan_root,
                profile=profile,
                layout=layout,
                plan_only=True,
                frontier_design=frontier_design,
            )
            planned_by_design[frontier_design] = planned
            if planned.get("status") != "PLANNED":
                errors.append(
                    f"{frontier_design} frontier plan-only status={planned.get('status')}"
                )
            if planned.get("archive_state") != "ABSENT_NOT_YET_RUN":
                errors.append(
                    f"{frontier_design} frontier plan-only did not represent zero archives as "
                    f"ABSENT_NOT_YET_RUN: {planned}"
                )
            if int(planned.get("discovered_archive_count", -1)) != 0:
                errors.append(
                    f"{frontier_design} frontier plan-only archive count mismatch: {planned}"
                )
            if planned.get("expected_budgets") != [0.75, 1.27, 2.0, "unbounded"]:
                errors.append(
                    f"{frontier_design} frontier plan-only budget contract mismatch: {planned}"
                )
            if planned.get("frontier_design") != frontier_design:
                errors.append(
                    f"{frontier_design} frontier design marker mismatch: {planned}"
                )
            if planned.get("profile_key") != expected["profile_key"]:
                errors.append(
                    f"{frontier_design} profile key mismatch: {planned}"
                )
            if planned.get("run_role") != expected["run_role"]:
                errors.append(
                    f"{frontier_design} run role mismatch: {planned}"
                )

        partial = (
            plan_root
            / profile["workspace"]["llm_runs_root"]
            / "synthetic_frontier_partial_0p75"
        )
        stage7 = partial / "stage7_llm_action_generation"
        stage7.mkdir(parents=True, exist_ok=True)
        (partial / "archive_manifest.json").write_text(
            json.dumps({
                "run_role": "paper_n5_budget_frontier_icb",
                "extra": {
                    "reproduction_contract": {
                        "freeform_l1_budget": 0.75,
                    }
                },
            }),
            encoding="utf-8",
        )
        (stage7 / "metadata.json").write_text(
            json.dumps({
                "information_condition": "IC-b",
                "conditions": ["C4", "C6"],
                "modes": ["free_form_10d"],
                "action_budget_contract": {"l1_budget": 0.75},
            }),
            encoding="utf-8",
        )
        try:
            _run_frontier(
                root=plan_root,
                profile=profile,
                layout=layout,
                plan_only=True,
                frontier_design="legacy_c6_only",
            )
        except ProfileError as exc:
            if "found 1" not in str(exc):
                errors.append(f"frontier partial-plan wrong hard failure: {exc}")
        else:
            errors.append("frontier plan-only silently accepted a partial archive")

        matched_partial = (
            plan_root
            / profile["workspace"]["llm_runs_root"]
            / "synthetic_matched_frontier_partial_0p75"
        )
        matched_stage7 = matched_partial / "stage7_llm_action_generation"
        matched_stage7.mkdir(parents=True, exist_ok=True)
        (matched_partial / "archive_manifest.json").write_text(
            json.dumps({
                "run_role": "paper_n5_matched_budget_frontier_icb",
                "extra": {
                    "reproduction_contract": {
                        "freeform_l1_budget": 0.75,
                    }
                },
            }),
            encoding="utf-8",
        )
        (matched_stage7 / "metadata.json").write_text(
            json.dumps({
                "information_condition": "IC-b",
                "conditions": ["C4", "C6"],
                "modes": ["free_form_10d"],
                "action_budget_contract": {
                    "l1_budget": 0.75,
                    "budgeted_conditions": ["C4", "C6"],
                },
            }),
            encoding="utf-8",
        )
        try:
            _run_frontier(
                root=plan_root,
                profile=profile,
                layout=layout,
                plan_only=True,
                frontier_design="matched_c4_c6",
            )
        except ProfileError as exc:
            if "found 1" not in str(exc):
                errors.append(f"matched frontier partial-plan wrong hard failure: {exc}")
        else:
            errors.append("matched frontier plan-only silently accepted a partial archive")

    analysis_source = project_root / "src" / "credit_recourse" / "analysis" / "paper_repro_analysis.py"
    remaining_source = project_root / "src" / "credit_recourse" / "analysis" / "remaining_thesis_analyses.py"
    for path, markers in (
        (analysis_source, (
            "remaining no-API analysis backfill [resume]",
            "--allow-analysis-running",
        )),
        (remaining_source, (
            "--allow-analysis-running",
            "restricted to the noapi resume backfill path",
            "allow_analysis_running",
            "STALE_INDUSTRY_CONTRACT",
            "paper_repro_shuffle_invalidated",
            "allow_absent=plan_only",
            "ABSENT_NOT_YET_RUN",
            "--frontier-design",
            "matched_c4_c6",
        )),
    ):
        if not path.is_file():
            errors.append(f"missing integration source: {path}")
            continue
        source_text = path.read_text(encoding="utf-8-sig")
        for marker in markers:
            if marker not in source_text:
                errors.append(f"resume backfill integration marker missing in {path.name}: {marker}")

    with tempfile.TemporaryDirectory(prefix="verify_remaining_analyses_") as tmp:
        root = Path(tmp)
        stage6 = _write_stage6(root)
        run = _write_revision_run(root)
        out = root / "data" / "analysis" / "reference_quality"
        result = run_analysis(
            project_root=root,
            run_dirs=[run],
            output_dir=out,
            stage6_path=stage6,
            bootstrap_draws=50,
            bootstrap_seed=7,
        )
        if result.get("status") != "PASS":
            errors.append(f"synthetic reference-quality status={result.get('status')}")
        if result.get("schema_version") != "reference_quality_acceptance_v4":
            errors.append(f"synthetic reference-quality schema={result.get('schema_version')!r}")
        cohort = result.get("cohort_contract", {})
        if cohort.get("contract") != COHORT_CONTRACT:
            errors.append(f"synthetic cohort contract={cohort.get('contract')!r}")
        if int(cohort.get("target_row_count_per_run", -1)) != 40:
            errors.append(
                "synthetic target cohort rows per run="
                f"{cohort.get('target_row_count_per_run')}, expected=40"
            )
        if int(cohort.get("observed_cell_count", -1)) != 6:
            errors.append(f"synthetic observed paired cells={cohort.get('observed_cell_count')}, expected=6")
        if int(cohort.get("stage7_pair_eligible_row_count", -1)) != 237:
            errors.append(
                "synthetic Stage7 pair-eligible rows="
                f"{cohort.get('stage7_pair_eligible_row_count')}, expected=237"
            )
        if int(cohort.get("stage9_metric_row_count", -1)) != 237:
            errors.append(
                "synthetic Stage9 metric rows="
                f"{cohort.get('stage9_metric_row_count')}, expected=237"
            )
        if int(cohort.get("metrics_undefined_row_count", -1)) != 23:
            errors.append(
                "synthetic undefined geometry rows="
                f"{cohort.get('metrics_undefined_row_count')}, expected=23"
            )
        c7_free = [
            x for x in cohort.get("cells", [])
            if x.get("revision_condition") == "C7" and x.get("mode") == "free_form_10d"
        ]
        if len(c7_free) != 1 or int(c7_free[0].get("pair_eligible_count", -1)) != 37:
            errors.append(f"synthetic C7 free-form paired cohort mismatch: {c7_free}")
        lookup_meta = result.get("stage6_reference_lookup", {})
        if lookup_meta.get("contract") != REFERENCE_LOOKUP_CONTRACT:
            errors.append(f"synthetic reference lookup contract={lookup_meta.get('contract')!r}")
        if int(lookup_meta.get("c0_a0_alias_row_count", -1)) != 40:
            errors.append(
                f"synthetic C0/A0 alias rows={lookup_meta.get('c0_a0_alias_row_count')}, expected=40"
            )
        if int(result.get("shown_noop_reference_row_count", 0)) <= 0:
            errors.append("synthetic A0_noop shown references were not resolved")
        raw_stage6 = pd.read_parquet(stage6)
        lookup, direct_meta = _build_stage6_reference_lookup(raw_stage6)
        a0 = lookup[(lookup["row_id"].eq(0)) & lookup["reference_candidate_id"].eq("A0_noop")]
        if len(a0) != 1 or str(a0.iloc[0]["reference_policy"]) != "C0_noop":
            errors.append(f"C0/A0 alias did not resolve uniquely: {a0.to_dict('records')}")
        if direct_meta.get("excluded_nonfixed_policy_row_count") != 40:
            errors.append(
                "row-conditional C3 rows were not excluded from fixed-candidate lookup: "
                f"{direct_meta.get('excluded_nonfixed_policy_row_count')}"
            )
        required = [
            out / "reference_quality_acceptance_rows.parquet",
            out / "reference_quality_acceptance_summary.csv",
            out / "reference_quality_acceptance_quartiles.csv",
            out / "reference_quality_acceptance_primary_c6.csv",
            out / "reference_quality_acceptance_manifest.json",
        ]
        errors.extend(f"missing synthetic output: {p}" for p in required if not p.is_file())
        if (out / "reference_quality_acceptance_primary_c6.csv").is_file():
            primary = pd.read_csv(out / "reference_quality_acceptance_primary_c6.csv")
            if len(primary) != 6:
                errors.append(f"synthetic primary rows={len(primary)}, expected=6")
            if not pd.to_numeric(primary["n_rows"], errors="coerce").eq(40).all():
                errors.append("synthetic primary n_rows is not 40")
            if not pd.to_numeric(primary["n_metrics_defined"], errors="coerce").eq(36).all():
                errors.append("synthetic primary n_metrics_defined is not 36")
            if not pd.to_numeric(primary["n_metrics_undefined"], errors="coerce").eq(4).all():
                errors.append("synthetic primary n_metrics_undefined is not 4")
            defined_fraction = pd.to_numeric(primary["metrics_defined_fraction"], errors="coerce")
            if not np.allclose(defined_fraction.to_numpy(dtype=float), 0.9, atol=1e-12):
                errors.append("synthetic primary metrics_defined_fraction is not 0.9")
            rho = pd.to_numeric(primary["rho_reference_advantage_vs_adoption"], errors="coerce")
            if not np.isfinite(rho).all() or not (rho > 0).all():
                errors.append("synthetic quality-adoption relation was not recovered as positive")

        bad_stage6 = pd.read_parquet(stage6)
        bad_stage6.loc[
            bad_stage6["policy"].eq("C0_noop") & bad_stage6["row_id"].eq(0),
            "delta_R_score_alpha",
        ] = 0.01
        bad_stage6_path = stage6.with_name("multi_oracle_policy_eval_bad_noop.parquet")
        bad_stage6.to_parquet(bad_stage6_path, index=False)
        try:
            run_analysis(
                project_root=root,
                run_dirs=[run],
                output_dir=root / "bad_noop_out",
                stage6_path=bad_stage6_path,
                bootstrap_draws=10,
            )
        except ValueError as exc:
            if "no-op delta_R_score" not in str(exc):
                errors.append(f"wrong nonzero-noop failure: {exc}")
        else:
            errors.append("nonzero C0/A0 no-op score did not hard-fail")

        bad = pd.read_csv(run / "stage9_llm_rl_comparison" / "llm_stage9_revision_metrics.csv")
        bad.loc[0, "rl_reference_candidate"] = "MISSING_REFERENCE"
        bad.to_csv(run / "stage9_llm_rl_comparison" / "llm_stage9_revision_metrics.csv", index=False)
        try:
            run_analysis(
                project_root=root,
                run_dirs=[run],
                output_dir=root / "bad_out",
                stage6_path=stage6,
                bootstrap_draws=10,
            )
        except ValueError as exc:
            if "shown reference missing" not in str(exc):
                errors.append(f"wrong missing-reference failure: {exc}")
        else:
            errors.append("missing Stage6 reference did not hard-fail")

        pair_run = _write_revision_run(root / "pair_mismatch")
        pair_stage7 = pair_run / "stage7_llm_action_generation" / "llm_stage7_action_table.parquet"
        pair_frame = pd.read_parquet(pair_stage7)
        pair_frame = pair_frame[~(
            pair_frame["row_id"].eq(0)
            & pair_frame["policy"].eq("C4")
            & pair_frame["mode"].eq("candidate_selection")
        )].copy()
        pair_frame.to_parquet(pair_stage7, index=False)
        try:
            run_analysis(
                project_root=root,
                run_dirs=[pair_run],
                output_dir=root / "pair_mismatch_out",
                stage6_path=stage6,
                bootstrap_draws=10,
            )
        except ValueError as exc:
            if "Stage7/Stage9 paired-cohort contract failed" not in str(exc):
                errors.append(f"wrong Stage7/Stage9 pair mismatch failure: {exc}")
        else:
            errors.append("Stage9 row outside the Stage7 pair-eligible cohort did not hard-fail")

        shuffle_root = root / "shuffle_contract"
        stale = shuffle_root / "row_shuffle_vector_null_native_industry"
        stale.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"shuffle_stratum_count": [1]}).to_csv(stale / "shuffle_permutation_ci.csv", index=False)
        pd.DataFrame({"shuffle_seed": [1]}).to_csv(stale / "shuffle_per_draw_summary.csv", index=False)
        (stale / "metadata.json").write_text(json.dumps({
            "status": "PASS",
            "shuffle_draw_count": 1000,
            "shuffle_within": "industry",
            "stratum_count": 1,
        }), encoding="utf-8")
        stale_ok, stale_detail = _shuffle_cell_ok(stale, expected_within="industry", expected_draws=1000)
        if stale_ok or not stale_detail.startswith("STALE_INDUSTRY_CONTRACT:"):
            errors.append(f"legacy single-industry cell was not classified stale: {stale_ok}, {stale_detail}")
        archive_root = root / "repo"
        stale_under_repo = archive_root / "data" / "analysis" / "paper_repro" / stale.name
        stale_under_repo.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copytree(stale, stale_under_repo)
        archived = _archive_stale_industry_cell(
            root=archive_root,
            path=stale_under_repo,
            information_condition="IC-a",
            run_label="synthetic_run",
            reason=stale_detail,
        )
        if stale_under_repo.exists() or not archived.is_dir():
            errors.append("stale industry cell was not moved to the invalidation archive")
        if not (archived.parent / "invalidation_record.json").is_file():
            errors.append("industry invalidation archive lacks an audit record")

        valid = shuffle_root / "valid_industry"
        valid.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"shuffle_stratum_count": [8, 8, 8]}).to_csv(valid / "shuffle_permutation_ci.csv", index=False)
        pd.DataFrame({"shuffle_seed": [1, 2, 3]}).to_csv(valid / "shuffle_per_draw_summary.csv", index=False)
        (valid / "metadata.json").write_text(json.dumps({
            "status": "PASS",
            "shuffle_draw_count": 1000,
            "shuffle_within": "industry",
            "stratum_count": 8,
            "stratum_unique_known": 8,
            "stratum_known_fraction": 0.97,
            "stratum_resolution_status": "PASS",
            "stratum_contract_sha256": "a" * 64,
        }), encoding="utf-8")
        valid_ok, valid_detail = _shuffle_cell_ok(valid, expected_within="industry", expected_draws=1000)
        if not valid_ok:
            errors.append(f"valid informative industry cell failed validation: {valid_detail}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


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
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
