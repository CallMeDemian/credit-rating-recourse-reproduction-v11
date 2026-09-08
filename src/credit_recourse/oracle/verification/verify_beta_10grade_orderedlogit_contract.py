
from __future__ import annotations
import json
from pathlib import Path
import sys
import pandas as pd

GRADE_ORDER = ["AAA","AA","A","BBB","BB","B","CCC","CC","C","D"]

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args(argv)
    root = Path(args.project_root)
    beta_dir = root / "data" / "final_freeze" / "stage1_oracle_backends" / "beta"
    params_path = beta_dir / "benchmark_beta_params.json"
    out_path = beta_dir / "benchmark_firm_year_output_beta.parquet"
    if not params_path.exists():
        raise FileNotFoundError(params_path)
    if not out_path.exists():
        raise FileNotFoundError(out_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    df = pd.read_parquet(out_path)
    prob_cols = [c for c in df.columns if c.startswith("prob_beta_grade_")]
    prob_grades = [c.replace("prob_beta_grade_", "") for c in prob_cols]
    argmax_grade = df[prob_cols].idxmax(axis=1).str.replace("prob_beta_grade_", "", regex=False)
    mismatch = int((argmax_grade != df["R_grade_beta"].astype(str)).sum())
    forbidden_cols = [c for c in ["grade_base_notch","rating_num_notch","grade_base_7","rating_num_7","grade_folded","grade_folded_num"] if c in df.columns]
    required = {
        "grade_order_10": params.get("grade_order") == GRADE_ORDER,
        "posterior_map": params.get("grade_assignment_rule") == "posterior_MAP_class_from_ordered_logit_probabilities",
        "no_final_boundaries": params.get("final_boundaries") == {},
        "has_imputation_map": "imputation_map" in params and isinstance(params["imputation_map"], dict),
        "has_C_prob": "prob_beta_grade_C" in prob_cols,
        "has_D_prob": "prob_beta_grade_D" in prob_cols,
        "posterior_map_mismatch_zero": mismatch == 0,
        "no_22_or_7_or_folded_cols": not forbidden_cols,
    }
    report = {
        "status": "PASS" if all(required.values()) else "FAIL",
        "checks": required,
        "modeled_grades": params.get("modeled_grades"),
        "probability_output_columns": params.get("probability_output_columns"),
        "probability_columns_in_output": prob_cols,
        "predicted_output_grades": sorted(df["R_grade_beta"].dropna().astype(str).unique(), key=lambda g: GRADE_ORDER.index(g) if g in GRADE_ORDER else 999),
        "posterior_map_mismatch_count": mismatch,
        "forbidden_columns_present": forbidden_cols,
    }
    out = root / "data" / "final_freeze" / "ledgers" / "beta_10grade_orderedlogit_contract_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0 if report["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
