from __future__ import annotations

"""Synthetic verifier for N5 Table 7-10c Holm inference.

The verifier protects the canonical-C3 contract used by
``credit_recourse.analysis.n5_7_10c_holm_inference``:

- exact ``C3_candidate_iql`` rows are the only C3 reference;
- q_argmax/q_rerank diagnostic variants are not blended into C3;
- the exact C3 mean must agree with Stage6 ``final_policy_summary.csv``;
- the paper patch writes a C3 reference audit artifact.

It uses contract-faithful synthetic row-level Stage9 tables.  It does not call
LLM APIs, rerun RL, or mutate frozen artifacts.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.analysis.n5_7_10c_holm_inference import main as n5_holm_main


ICS = ("IC-a", "IC-b", "IC-c")
ORACLE_MEANS = {
    "alpha": {"C3": 0.630313, "C4": 0.775000, "C6": {"IC-a": 0.893000, "IC-b": 0.889000, "IC-c": 0.935000}},
    "beta": {"C3": 0.075830, "C4": 0.080000, "C6": {"IC-a": 0.204830, "IC-b": 0.204830, "IC-c": 0.204830}},
    "gamma": {"C3": 0.594781, "C4": 0.700000, "C6": {"IC-a": 0.758781, "IC-b": 0.763781, "IC-c": 0.763781}},
}
VARIANT_ALPHA_MEANS = {
    "C3_candidate_iql_q_argmax": 0.491039,
    "C3_candidate_iql_q_rerank_at_3": 0.592937,
    "C3_candidate_iql_q_rerank_at_5": 0.544472,
    "C3_candidate_iql_q_rerank_at_7": 0.512252,
    "C3_candidate_iql_q_rerank_at_9": 0.509939,
}


def _with_row_variation(mean: float, row_id: int) -> float:
    # Non-zero deterministic variation keeps Wilcoxon inputs realistic while
    # preserving the requested mean across row_ids 1..5.
    offsets = {1: -0.002, 2: -0.001, 3: 0.0, 4: 0.001, 5: 0.002}
    return float(mean + offsets[row_id])


def _write_stage6_summary(project_root: Path) -> Path:
    out = project_root / "data" / "final_freeze" / "stage6_candidate_selector_eval"
    out.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"policy": "C3_candidate_iql", "n": 5}
    for oracle in ["alpha", "beta", "gamma"]:
        row[f"mean_delta_R_score_{oracle}"] = ORACLE_MEANS[oracle]["C3"]
    # Include variants to ensure exact row selection in the summary is tested.
    rows = [row]
    for pol, val in VARIANT_ALPHA_MEANS.items():
        r: dict[str, Any] = {"policy": pol, "n": 5}
        r["mean_delta_R_score_alpha"] = val
        r["mean_delta_R_score_beta"] = 0.05
        r["mean_delta_R_score_gamma"] = 0.45
        rows.append(r)
    path = out / "final_policy_summary.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_stage9_run(project_root: Path, ic: str) -> Path:
    run_name = f"N5_C6_L1_1p27_{ic.replace('-', '')}_gpt54mini_p50_main_seed1_20260707"
    run_dir = project_root / "data" / "final_freeze" / "llm_runs" / run_name / "stage9_policy_comparison"
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for row_id in range(1, 6):
        base_c3 = {
            "row_id": row_id,
            "policy": "C3_candidate_iql",
            "candidate_id": "DL2_deleverage_moderate",
            "mode": "rl_native",
        }
        base_c4 = {
            "row_id": row_id,
            "policy": "C4",
            "candidate_id": "FREEFORM",
            "mode": "free_form_10d",
        }
        base_c6 = {
            "row_id": row_id,
            "policy": "C6",
            "candidate_id": "FREEFORM",
            "mode": "free_form_10d",
        }
        for oracle in ["alpha", "beta", "gamma"]:
            base_c3[f"delta_R_score_{oracle}"] = _with_row_variation(float(ORACLE_MEANS[oracle]["C3"]), row_id)
            base_c4[f"delta_R_score_{oracle}"] = _with_row_variation(float(ORACLE_MEANS[oracle]["C4"]), row_id)
            base_c6[f"delta_R_score_{oracle}"] = _with_row_variation(float(ORACLE_MEANS[oracle]["C6"][ic]), row_id)
        rows.extend([base_c3, base_c4, base_c6])

        for pol, alpha_mean in VARIANT_ALPHA_MEANS.items():
            v = {
                "row_id": row_id,
                "policy": pol,
                "candidate_id": "DIAGNOSTIC",
                "mode": "rl_native",
                "delta_R_score_alpha": _with_row_variation(alpha_mean, row_id),
                "delta_R_score_beta": _with_row_variation(0.05, row_id),
                "delta_R_score_gamma": _with_row_variation(0.45, row_id),
            }
            rows.append(v)
    path = run_dir / "llm_stage9_llm_rl_comparison.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def run_verification() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        project_root = Path(td)
        stage6_summary = _write_stage6_summary(project_root)
        for ic in ICS:
            _write_stage9_run(project_root, ic)
        out_dir = project_root / "data" / "analysis_n5" / "n5_7_10c_holm"
        n5_holm_main([
            "--project-root", str(project_root),
            "--out-dir", str(out_dir),
            "--stage6-summary", str(stage6_summary),
            "--holm-family", "oracle",
        ])
        patch = pd.read_csv(out_dir / "n5_7_10c_table_patch.csv")
        audit = pd.read_csv(out_dir / "n5_7_10c_c3_reference_audit.csv")
        alpha_audit = audit[audit["oracle_backend"].eq("alpha")].copy()
        expected_gaps = {
            "IC-a": ORACLE_MEANS["alpha"]["C6"]["IC-a"] - ORACLE_MEANS["alpha"]["C3"],
            "IC-b": ORACLE_MEANS["alpha"]["C6"]["IC-b"] - ORACLE_MEANS["alpha"]["C3"],
            "IC-c": ORACLE_MEANS["alpha"]["C6"]["IC-c"] - ORACLE_MEANS["alpha"]["C3"],
        }
        errors: list[str] = []
        observed: dict[str, Any] = {}
        for ic, exp in expected_gaps.items():
            row = patch[patch["information_condition"].eq(ic)]
            if len(row) != 1:
                errors.append(f"missing patch row for {ic}")
                continue
            got = float(row.iloc[0]["C6_minus_C3_alpha_gap"])
            observed[ic] = got
            if not np.isclose(got, exp, atol=1e-12):
                errors.append(f"{ic}: expected C6-C3 alpha gap {exp}, got {got}")
        if not (alpha_audit["stage9_exact_c3_mean_paired"].round(6) == round(ORACLE_MEANS["alpha"]["C3"], 6)).all():
            errors.append("exact C3 alpha audit mean does not match canonical Stage6 C3")
        if not (alpha_audit["stage9_blended_c3_variant_mean"] < alpha_audit["stage9_exact_c3_mean_paired"]).all():
            errors.append("synthetic blended C3 variant mean should be below exact C3 mean")
        if not (alpha_audit["c3_reference_audit_status"].astype(str).eq("PASS")).all():
            errors.append("C3 reference audit did not PASS")
        return {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "stage6_summary": str(stage6_summary),
            "observed_alpha_c6_minus_c3_gaps": observed,
            "alpha_audit": alpha_audit.to_dict(orient="records"),
        }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify N5 Table 7-10c Holm canonical-C3 contract with synthetic data")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    summary = run_verification()
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
