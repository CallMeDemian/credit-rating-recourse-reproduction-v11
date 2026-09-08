#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify Gamma-ML final 10-grade nonlinear regressor contract.

This verifier intentionally checks the *development backend output* contract:
- target is the 10-step AAA-D base scale, not legacy 7-grade/folded/notch
- R_grade_gamma is direct round/clip of predicted_rating_num_gamma
- R_score_gamma uses the expected/notch score formula
- no score-boundary remapping is present
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GRADE_ORDER_10 = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
NUM2GRADE_10 = {i + 1: g for i, g in enumerate(GRADE_ORDER_10)}

FORBIDDEN_OUTPUT_COLUMNS = {
    "grade_base_7",
    "rating_num_7",
    "grade_base_notch",
    "rating_num_notch",
    "grade_folded",
    "grade_folded_num",
}


def _load_output(path: Path) -> pd.DataFrame:
    pq = path / "benchmark_firm_year_output_gamma.parquet"
    csv = path / "benchmark_firm_year_output_gamma.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Gamma output not found under {path}")


def _round_clip(x: pd.Series) -> np.ndarray:
    return np.rint(np.clip(pd.to_numeric(x, errors="coerce").to_numpy(dtype=float), 1, 10)).astype(int)


def verify(project_root: Path) -> dict[str, Any]:
    gamma_dir = project_root / "data" / "final_freeze" / "stage1_oracle_backends" / "gamma"
    params_path = gamma_dir / "benchmark_gamma_params.json"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    with params_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    out = _load_output(gamma_dir)

    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any = None, required: bool = True):
        checks.append({"check": name, "pass": bool(passed), "required": bool(required), "detail": detail})

    add("target_grade_scale_10_step", "10" in str(params.get("target_grade_scale", "")) and "AAA" in str(params.get("target_grade_scale", "")) and "D" in str(params.get("target_grade_scale", "")), params.get("target_grade_scale"))
    add("grade_order_exact_10", params.get("grade_order") == GRADE_ORDER_10, params.get("grade_order"))
    add("modeled_grades_exact_10", params.get("modeled_grades") == GRADE_ORDER_10, params.get("modeled_grades"))
    add("modeled_grade_nums_exact_1_to_10", params.get("modeled_grade_nums") == list(range(1, 11)), params.get("modeled_grade_nums"))
    add("final_boundaries_empty", params.get("final_boundaries", None) == {}, params.get("final_boundaries"))
    add("grade_assignment_rule_round_clip", params.get("grade_assignment_rule") == "rounded_clipped_predicted_rating_num_10", params.get("grade_assignment_rule"))
    add("score_definition_no_boundary", "no score-boundary remapping" in str(params.get("score_definition", "")), params.get("score_definition"))

    forbidden_present = sorted(FORBIDDEN_OUTPUT_COLUMNS & set(out.columns))
    add("no_legacy_7grade_or_notch_or_folded_columns", len(forbidden_present) == 0, forbidden_present)

    required_cols = [
        "predicted_rating_num_gamma",
        "R_score_gamma",
        "R_grade_gamma",
        "R_grade_gamma_num",
        "grade_base_10",
        "rating_num_10",
    ]
    missing = [c for c in required_cols if c not in out.columns]
    add("required_output_columns_present", len(missing) == 0, missing)

    if not missing:
        pred_num = _round_clip(out["predicted_rating_num_gamma"])
        stored_num = pd.to_numeric(out["R_grade_gamma_num"], errors="coerce").to_numpy(dtype=float)
        num_mismatch = int(np.sum(~np.isclose(pred_num.astype(float), stored_num, equal_nan=False)))
        add("R_grade_gamma_num_round_clip_matches", num_mismatch == 0, {"mismatch": num_mismatch, "n": int(len(out))})

        expected_grade = pd.Series(pred_num).map(NUM2GRADE_10).to_numpy()
        stored_grade = out["R_grade_gamma"].astype(str).to_numpy()
        grade_mismatch = int(np.sum(expected_grade != stored_grade))
        add("R_grade_gamma_string_matches_num", grade_mismatch == 0, {"mismatch": grade_mismatch, "n": int(len(out))})

        expected_score = 100.0 * (10.0 - np.clip(pd.to_numeric(out["predicted_rating_num_gamma"], errors="coerce").to_numpy(dtype=float), 1, 10)) / 9.0
        stored_score = pd.to_numeric(out["R_score_gamma"], errors="coerce").to_numpy(dtype=float)
        diff = np.abs(expected_score - stored_score)
        add("R_score_gamma_formula_matches", bool(np.nanmax(diff) < 1e-9), {"max_abs_diff": float(np.nanmax(diff)), "mean_abs_diff": float(np.nanmean(diff))})

    status = "PASS" if all(c["pass"] for c in checks if c["required"]) else "FAIL"
    report = {
        "status": status,
        "gamma_dir": str(gamma_dir),
        "checks": checks,
    }
    ledger_dir = project_root / "data" / "final_freeze" / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    with (ledger_dir / "gamma_10grade_ml_contract_verification.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    report = verify(Path(args.project_root))
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
