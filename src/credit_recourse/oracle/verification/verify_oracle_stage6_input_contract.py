#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that Stage6 simulator inputs match the Stage1 Oracle variable contract.

This closes the gap that a bit-exact scorer test cannot close: Stage6 must feed
Alpha/Beta/Gamma with the same R-code definitions that Stage1 used for model
fitting.  The test is intentionally formula-contract focused.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.oracle_variables import compute_oracle_variables
from credit_recourse.simulator.oracle_variable_contract import (
    ORACLE_FINANCIAL_VARIABLE_CONTRACT,
    contract_as_records,
)
from credit_recourse.oracle.selected_variable_current_contract import (
    CURRENT_SELECTED_VARIABLE_IDS,
    verify_package_selected_variable_masters,
)

FIN_CODES = [v for v in CURRENT_SELECTED_VARIABLE_IDS if v.startswith("R")]
FORBIDDEN_BETA_COLUMNS = {
    "grade_base_notch", "rating_num_notch", "grade_base_7", "rating_num_7", "grade_folded"
}


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _maybe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8-sig")


def _find_formula_dictionary(root: Path) -> Path | None:
    candidates = [
        root / "data/final_freeze/stage1_oracle_inputs/stage00_02_financial_ratio_engineering/financial_ratio_formula_dictionary_final.csv",
        root / "data/final_freeze/stage1_oracle_inputs/stage00_02_financial_ratio_engineering/financial_ratio_formula_dictionary_draft.csv",
        root / "src/credit_recourse/oracle/stage1/stage00_02_ratio_engineering/candidate_ratio_master.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_formula_dictionary(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=0)
    return _maybe_read_csv(path)


def _lookup_formula_row(df: pd.DataFrame, code: str) -> dict[str, Any] | None:
    rid = int(code[1:])
    # Common schemas: No, ratio_id=R136, ratio_code=R136, variable_id=R136.
    for col in ["No", "no", "번호"]:
        if col in df.columns:
            m = df[pd.to_numeric(df[col], errors="coerce") == rid]
            if len(m):
                return m.iloc[0].to_dict()
    for col in ["ratio_id", "ratio_code", "variable_id", "R코드", "code"]:
        if col in df.columns:
            m = df[df[col].astype(str).str.strip().str.upper().eq(code)]
            if len(m):
                return m.iloc[0].to_dict()
    return None


def _contains_all(text: str, needles: list[str]) -> bool:
    text = str(text)
    return all(n in text for n in needles)


def check_formula_dictionary(root: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    warnings: list[str] = []
    p = _find_formula_dictionary(root)
    df = _load_formula_dictionary(p)
    if df is None:
        warnings.append("financial_ratio_formula_dictionary not found; formula dictionary cross-check skipped")
        return rows, errors, warnings

    for code, spec in ORACLE_FINANCIAL_VARIABLE_CONTRACT.items():
        r = _lookup_formula_row(df, code)
        if r is None:
            errors.append(f"{code}: missing from formula dictionary {p}")
            rows.append({"variable_id": code, "dictionary_found": False, **next(x for x in contract_as_records() if x["variable_id"] == code)})
            continue
        formula_text = " | ".join(str(r.get(c, "")) for c in ["후보비율명", "공식", "분자", "분모", "ratio_name", "formula", "numerator", "denominator"])
        ok = True
        if code == "R136":
            ok = _contains_all(formula_text, ["매입채무", "유동부채"])
            if not ok:
                errors.append(f"R136 formula dictionary is not 매입채무/유동부채: {formula_text}")
        elif code == "R006":
            ok = _contains_all(formula_text, ["법인세차감전", "매출액"])
        elif code == "R064":
            ok = _contains_all(formula_text, ["이익잉여금", "총자산"])
        elif code == "R085":
            ok = _contains_all(formula_text, ["금융비용", "매출액"])
        elif code == "R157":
            ok = _contains_all(formula_text, ["매출액", "자본금"])
        elif code == "R185":
            ok = _contains_all(formula_text, ["이익잉여금"])
        if not ok and code != "R136":
            warnings.append(f"{code}: formula dictionary text could not be fully matched: {formula_text}")
        rec = {
            "variable_id": code,
            "dictionary_found": True,
            "dictionary_source": str(p),
            "dictionary_formula_text": formula_text,
            "formula_dictionary_match": bool(ok),
        }
        rec.update(next(x for x in contract_as_records() if x["variable_id"] == code))
        rows.append(rec)
    return rows, errors, warnings


def check_compute_oracle_variables_unit() -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    prev = FirmState(
        firm_id="X", year=2022, sector="S", revenue=80.0, pretax_income=8.0,
        financial_cost=4.0, total_assets=180.0, retained_earnings=40.0,
        current_liabilities=50.0, payables=20.0, capital_stock=40.0,
        total_liabilities=120.0, operating_income=10.0, depreciation=4.0, amortization=1.0,
    )
    state = FirmState(
        firm_id="X", year=2023, sector="S", revenue=100.0, pretax_income=10.0,
        financial_cost=5.0, total_assets=200.0, retained_earnings=50.0,
        current_assets=150.0, current_liabilities=60.0, payables=30.0,
        capital_stock=60.0, total_liabilities=180.0,
        operating_income=20.0, depreciation=5.0, amortization=2.0,
    )
    out = compute_oracle_variables(state, prev_state=prev)
    expected = {
        "R006": 0.10,
        "R064": 0.25,
        "R085": 0.05,
        "R136": 0.50,  # payables/current liabilities; current ratio would be 2.50 and is forbidden.
        "R148": 100.0 / 150.0,
        "R174": 0.80,
        "R157": 2.00,
        "R185": 0.25,
    }
    observed = {k: out.get(k) for k in expected}
    for k, exp in expected.items():
        got = observed.get(k)
        if got is None or abs(float(got) - exp) > 1e-10:
            errors.append(f"compute_oracle_variables {k}: expected {exp}, got {got}")
    if abs(float(out.get("R136", float("nan"))) - 2.5) < 1e-10:
        errors.append("R136 is still current_assets/current_liabilities; expected payables/current_liabilities")
    return errors, {"expected": expected, "observed": observed}


def check_beta_artifact(root: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    beta_dir = root / "data/final_freeze/stage1_oracle_backends/beta"
    if not beta_dir.exists():
        # Allow running against an unpacked backend zip from current working dir.
        alt = root / "stage1_oracle_backends/beta"
        if alt.exists():
            beta_dir = alt
    params_path = beta_dir / "benchmark_beta_params.json"
    out_csv = beta_dir / "benchmark_firm_year_output_beta.csv"
    detail: dict[str, Any] = {"beta_dir": str(beta_dir), "params_exists": params_path.exists(), "output_csv_exists": out_csv.exists()}
    if not params_path.exists():
        warnings.append("Beta params not found; backend artifact checks skipped")
        return errors, warnings, detail
    params = _read_json(params_path)
    detail["target_grade_scale"] = params.get("target_grade_scale")
    detail["modeled_grades"] = params.get("modeled_grades")
    detail["modeled_grade_nums"] = params.get("modeled_grade_nums")
    detail["grade_assignment_rule"] = params.get("grade_assignment_rule")
    detail["has_imputation_map"] = "imputation_map" in params
    if "imputation_map" not in params:
        errors.append("benchmark_beta_params.json missing imputation_map; Beta is not self-contained")
    else:
        miss = [v for v in params.get("selected_variables", []) if v not in (params.get("imputation_map") or {})]
        if miss:
            errors.append(f"Beta imputation_map missing selected variables: {miss}")
    if params.get("final_boundaries") not in ({}, None):
        errors.append("Beta final_boundaries must be empty; ordered-logit grade must be posterior MAP, not boundary remapping")
    if params.get("grade_assignment_rule") != "posterior_MAP_class_from_ordered_logit_probabilities":
        errors.append("Beta grade_assignment_rule is not posterior_MAP_class_from_ordered_logit_probabilities")
    if any(int(x) > 10 for x in params.get("modeled_grade_nums", []) if str(x).lstrip('-').isdigit()):
        errors.append("Beta modeled_grade_nums contains >10 value; 22-notch leakage suspected")

    if out_csv.exists():
        out = _maybe_read_csv(out_csv)
        if out is not None:
            detail["output_rows"] = int(len(out))
            forbidden = sorted(FORBIDDEN_BETA_COLUMNS & set(out.columns))
            detail["forbidden_beta_columns_present"] = forbidden
            if forbidden:
                errors.append(f"Beta output contains forbidden 22-notch/7-grade/folded columns: {forbidden}")
            # R136 distribution sanity: Stage1 selected R136 is a bounded payables/current-liabilities ratio.
            if "R136" in out.columns:
                desc = pd.to_numeric(out["R136"], errors="coerce").describe().to_dict()
                detail["R136_distribution"] = {k: (None if pd.isna(v) else float(v)) for k, v in desc.items()}
                if desc.get("max", 0) > 5:
                    warnings.append("R136 max is >5; check whether Stage1 R136 scale changed from payables/current_liabilities")
    return errors, warnings, detail


def write_outputs(root: Path, report: dict[str, Any], contract_rows: list[dict[str, Any]]) -> None:
    ledgers = root / "data/final_freeze/ledgers"
    ledgers.mkdir(parents=True, exist_ok=True)
    with open(ledgers / "oracle_stage6_input_contract_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame(contract_rows).to_csv(ledgers / "oracle_stage6_financial_variable_contract.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--fail-on-warning", action="store_true")
    args = ap.parse_args()
    root = Path(args.project_root).resolve()

    contract_rows, errors, warnings = check_formula_dictionary(root)
    unit_errors, unit_detail = check_compute_oracle_variables_unit()
    errors.extend(unit_errors)
    beta_errors, beta_warnings, beta_detail = check_beta_artifact(root)
    errors.extend(beta_errors)
    warnings.extend(beta_warnings)
    package_root = Path(__file__).resolve().parents[2]
    selected_master_snapshot_sync = verify_package_selected_variable_masters(package_root)
    errors.extend(selected_master_snapshot_sync.get("errors", []))

    if not contract_rows:
        contract_rows = contract_as_records()

    report = {
        "status": "FAIL" if errors or (warnings and args.fail_on_warning) else "PASS",
        "purpose": "Verify Stage6 simulator R-code input definitions and Beta self-contained reproducibility artifact.",
        "errors": errors,
        "warnings": warnings,
        "financial_variable_contract": contract_rows,
        "compute_oracle_variables_unit_test": unit_detail,
        "beta_artifact_checks": beta_detail,
        "selected_variable_snapshot_sync": selected_master_snapshot_sync,
        "robustness_scope_note": (
            "Alpha/Beta/Gamma share the same 11 selected variables; this is functional-form robustness, "
            "not input-set or variable-selection robustness."
        ),
    }
    write_outputs(root, report, contract_rows)
    print(json.dumps({"status": report["status"], "errors": errors, "warnings": warnings}, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
