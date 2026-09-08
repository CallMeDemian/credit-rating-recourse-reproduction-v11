from __future__ import annotations

"""Synthetic verifier for optional N1/N3/N6/S2 closure analysis modules.

The verifier exercises the new optional analysis modules with contract-faithful
synthetic artifacts because the local container does not contain the user's
per-firm N5 Stage9 artifacts or TS2000 capital-change workbooks.  It does not
call live LLM APIs, does not run the simulator, and does not mutate frozen stage
outputs.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.analysis.action_budget_frontier_grid import build_frontier_plan, run_frontier_grid
from credit_recourse.analysis.action_space_coverage_audit import run_action_space_coverage_audit
from credit_recourse.analysis.winrate_heterogeneity import run_winrate_heterogeneity


def _check_winrate(tmp: Path, errors: list[str]) -> dict[str, Any]:
    run = tmp / "N5_C6_L1_1p27_ICa_gpt54mini_p50_main_seed1_20260707"
    stage9 = run / "stage9_policy_comparison"
    stage9.mkdir(parents=True)
    rows = []
    c3_scores = {1: 0.50, 2: 0.60, 3: 0.70, 4: 0.80}
    c6_scores = {1: 0.60, 2: 0.60, 3: 0.65, 4: 0.90}  # wins=2, tie=1, loss=1
    q_scores = {k: v - 0.2 for k, v in c3_scores.items()}
    for row_id in c3_scores:
        rows.append({"row_id": row_id, "policy": "C3_candidate_iql", "mode": "rl_native", "information_condition": "IC-a", "delta_R_score_alpha": c3_scores[row_id]})
        rows.append({"row_id": row_id, "policy": "C3_candidate_iql_q_argmax", "mode": "rl_native", "information_condition": "IC-a", "delta_R_score_alpha": q_scores[row_id]})
        rows.append({"row_id": row_id, "policy": "C6", "mode": "free_form_10d", "information_condition": "IC-a", "delta_R_score_alpha": c6_scores[row_id]})
    pd.DataFrame(rows).to_csv(stage9 / "llm_stage9_llm_rl_comparison.csv", index=False)
    panel = tmp / "panel.csv"
    pd.DataFrame({"row_id": [1, 2, 3, 4], "log_assets": [1.0, 2.0, 3.0, 4.0], "sector": ["A", "A", "B", "B"]}).to_csv(panel, index=False)
    out = tmp / "winrate_out"
    meta = run_winrate_heterogeneity(runs=[run], panel=panel, out_dir=out)
    wr = pd.read_csv(out / "win_rates_vs_C3.csv")
    row = wr[(wr["policy"] == "C6") & (wr["mode"] == "free_form_10d")].iloc[0]
    if int(row["n_pairs"]) != 4 or int(row["n_ties"]) != 1 or int(row["n_wins_excluding_ties"]) != 2:
        errors.append(f"winrate C6 counts mismatch: {row.to_dict()}")
    if not (out / "residual_heterogeneity_exploratory.csv").exists():
        errors.append("winrate heterogeneity output was not written")
    return {"metadata_status": meta.get("status"), "winrate_rows": int(len(wr))}


def _check_frontier_plan(tmp: Path, errors: list[str]) -> dict[str, Any]:
    run = tmp / "N5_C6_L1_1p27_ICb_gpt54mini_p50_main_seed1_20260707"
    s7 = run / "stage7_llm_action_generation"
    s7.mkdir(parents=True)
    # Plan mode only needs file discovery; contents are irrelevant in dry-run.
    (s7 / "llm_stage7_action_table.parquet").write_bytes(b"PARQUET_PLACEHOLDER_FOR_DISCOVERY_ONLY")
    out = tmp / "frontier_out"
    plan = build_frontier_plan(runs=[run], out_dir=out, grid=[1.27, 2.0], variants=["l1_rescale", "row_shuffle_vector_null"])
    if len(plan) != 4:
        errors.append(f"frontier plan expected 4 cells, got {len(plan)}")
    meta = run_frontier_grid(project_root=tmp, runs=[run], out_dir=out, grid=[1.27], variants=["l1_rescale"], dry_run=True)
    status = pd.read_csv(out / "frontier_grid_status.csv")
    if status.loc[0, "status"] != "DRY_RUN_NOT_EXECUTED":
        errors.append(f"frontier dry-run status mismatch: {status.to_dict(orient='records')}")
    return {"metadata_status": meta.get("status"), "dry_run": bool(meta.get("dry_run")), "plan_cells": int(meta.get("n_cells", -1))}


def _check_coverage(tmp: Path, errors: list[str]) -> dict[str, Any]:
    movers = tmp / "movers.csv"
    pd.DataFrame({
        "거래소코드": ["1", "000002", "3", "4"],
        "회계연도": [2024, 2024, 2024, 2024],
        "direction": [1, 1, -1, -1],
    }).to_csv(movers, index=False)
    capital = tmp / "capital.xlsx"
    pd.DataFrame({
        "종목코드": ["000001", "000003", "000099"],
        "변동일": ["2024-03-01", "2024-05-02", "2024-01-01"],
        "변동유형": ["유상증자", "단순변경", "감자"],
    }).to_excel(capital, index=False)
    struct = tmp / "struct.csv"
    pd.DataFrame({
        "corp_key": ["000002", "000004"],
        "fiscal_year": [2024, 2024],
        "has_structural_event": [True, False],
    }).to_csv(struct, index=False)
    out = tmp / "coverage_out"
    meta = run_action_space_coverage_audit(movers=movers, capital_files=[capital], struct_tags=struct, out_dir=out)
    tab = pd.read_csv(out / "action_space_coverage.csv")
    all_row = tab[tab["slice"].eq("ALL")].iloc[0]
    # outside rows: 000001 equity, 000002 structural => 2 / 4
    if abs(float(all_row["outside_action_space_share"]) - 0.5) > 1e-9:
        errors.append(f"coverage outside share mismatch: {all_row.to_dict()}")
    return {"metadata_status": meta.get("status"), "summary_rows": int(len(tab)), "n_movers": int(meta.get("n_movers", -1))}


def run_verification() -> dict[str, Any]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        checks = {
            "winrate_heterogeneity": _check_winrate(tmp, errors),
            "action_budget_frontier_grid_dryrun": _check_frontier_plan(tmp, errors),
            "action_space_coverage_audit": _check_coverage(tmp, errors),
        }
    return {"status": "PASS" if not errors else "FAIL", "checks": checks, "errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify optional closure analysis modules with synthetic artifacts")
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
