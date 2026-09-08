from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.oracle.contracts.rating_scale import GRADE_ORDER_10
from credit_recourse.utils.io_contract import read_json, read_csv_korean_safe, flatten_selected_variables, selected_variables_from_backend_params, write_json

REQUIRED_SELECTION_FILES = [
    "selected_variables_v2.json",
    "direction_encoding_v2.json",
    "weights_prior_v3.json",
    "selected_variable_master.csv",
    "simulator_ratio_alias_map.json",
]


def exists_nonempty(path: Path, errors: list[str], label: str | None = None) -> bool:
    if not path.exists():
        errors.append(f"missing {label or path}")
        return False
    if path.is_file() and path.stat().st_size == 0:
        errors.append(f"empty {label or path}")
        return False
    return True


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def selected_from_master(path: Path, errors: list[str]) -> list[str]:
    if not exists_nonempty(path, errors):
        return []
    df = read_csv_korean_safe(path)
    col = find_col(df, ["variable_id", "variable", "feature", "name", "id", "column"])
    if col is None:
        errors.append("selected_variable_master.csv has no dynamic variable id column")
        return []
    out = [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    if len(out) != len(set(out)):
        errors.append("selected_variable_master has duplicate selected variable ids")
    return out


def selected_from_json(path: Path) -> list[str]:
    if not path.exists():
        return []
    return flatten_selected_variables(read_json(path))


def verify_backend(name: str, params: Path, output: Path, selected_master: list[str], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    check: dict[str, Any] = {"backend": name}
    if not exists_nonempty(params, errors, f"{name} params"):
        return check
    obj = read_json(params)
    backend_vars = selected_variables_from_backend_params(params)
    check["selected_variables_count"] = len(backend_vars)
    if selected_master and backend_vars and set(backend_vars) != set(selected_master):
        errors.append(f"{name}: backend selected_variables differ from selected_variable_master")
    grade_order = obj.get("grade_order") or obj.get("GRADE_ORDER") or obj.get("grade_labels") or GRADE_ORDER_10
    if list(grade_order) != list(GRADE_ORDER_10):
        warnings.append(f"{name}: params do not expose explicit 10-grade order; found={grade_order}")
    fallback = obj.get("fallback_grade") or obj.get("fallback") or "D"
    if str(fallback) != "D":
        errors.append(f"{name}: fallback grade must be D, found {fallback}")
    if exists_nonempty(output, errors, f"{name} output"):
        try:
            df = pd.read_parquet(output)
            check["rows"] = int(len(df))
            score_cols = [c for c in df.columns if "score" in str(c).lower() or str(c).startswith("R_score")]
            if score_cols:
                bad = [c for c in score_cols if pd.to_numeric(df[c], errors="coerce").isna().all()]
                if bad:
                    errors.append(f"{name}: all-null score columns: {bad}")
            if df.duplicated([c for c in ["거래소코드", "year"] if c in df.columns]).any() if all(c in df.columns for c in ["거래소코드", "year"]) else False:
                errors.append(f"{name}: duplicate firm-year output rows")
        except Exception as exc:
            errors.append(f"{name}: cannot read output parquet: {exc!r}")
    return check


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--strict", action="store_true", help="Strict means dynamic consistency only; no fixed selected-variable list is enforced.")
    args = p.parse_args(argv)
    root = Path(args.project_root).resolve()
    final = root / "data" / "final_freeze"
    s4 = final / "stage1_oracle_inputs" / "stage00_04_variable_selection"
    b = final / "stage1_oracle_backends"
    cfg = final / "configs"
    ledgers = final / "ledgers"
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    for f in REQUIRED_SELECTION_FILES:
        exists_nonempty(s4 / f, errors)
    selected_master = selected_from_master(s4 / "selected_variable_master.csv", errors)
    selected_json = selected_from_json(s4 / "selected_variables_v2.json")
    checks["selected_variable_count"] = len(selected_master)
    checks["selected_variables_dynamic"] = True
    if selected_master and selected_json and set(selected_master) != set(selected_json):
        errors.append("selected_variables_v2.json and selected_variable_master.csv variable sets differ")
    if not selected_master:
        errors.append("no selected variables resolved from selected_variable_master.csv")
    if (s4 / "direction_encoding_v2.json").exists() and selected_master:
        dirs = read_json(s4 / "direction_encoding_v2.json")
        missing = [v for v in selected_master if isinstance(dirs, dict) and v not in dirs]
        if missing:
            errors.append(f"direction_encoding_v2.json missing selected variables: {missing[:20]}")
    ledger = ledgers / "stage1_oracle_backends_full_development.json"
    if exists_nonempty(ledger, errors):
        meta = read_json(ledger)
        checks["ledger_status"] = meta.get("status")
        if meta.get("status") != "PASS":
            errors.append(f"stage1 ledger status is not PASS: {meta.get('status')}")
        if not bool(meta.get("final_result_allowed")):
            errors.append("stage1 ledger final_result_allowed is false")
    reg = cfg / "oracle_backend_registry.yaml"
    if exists_nonempty(reg, errors):
        try:
            import yaml
            robj = yaml.safe_load(reg.read_text(encoding="utf-8")) or {}
            for name in ["alpha", "beta", "gamma"]:
                if name not in (robj.get("backends") or {}):
                    errors.append(f"oracle_backend_registry missing backend {name}")
        except Exception as exc:
            errors.append(f"cannot read registry: {exc!r}")
    checks["backends"] = {
        "alpha": verify_backend("alpha", b / "alpha" / "oracle_alpha_params.json", b / "alpha" / "oracle_firm_year_output_alpha.parquet", selected_master, errors, warnings),
        "beta": verify_backend("beta", b / "beta" / "benchmark_beta_params.json", b / "beta" / "benchmark_firm_year_output_beta.parquet", selected_master, errors, warnings),
        "gamma": verify_backend("gamma", b / "gamma" / "benchmark_gamma_params.json", b / "gamma" / "benchmark_firm_year_output_gamma.parquet", selected_master, errors, warnings),
    }
    bridge = final / "stage1_oracle_inputs" / "alpha_vanilla_input_candidate.parquet"
    bridge_meta = final / "stage1_oracle_inputs" / "alpha_vanilla_input_candidate_metadata.json"
    if exists_nonempty(bridge, errors, "Stage1→Stage2 bridge"):
        df = pd.read_parquet(bridge)
        for c in ["firm_id", "fiscal_year", "거래소코드", "year", "rating_num_10", "rating_num", "split", "selected_variables_all_complete"]:
            if c not in df.columns:
                errors.append(f"bridge missing required column {c}")
        missing = [v for v in selected_master if v not in df.columns]
        if missing:
            errors.append(f"bridge missing selected variables: {missing}")
        if all(c in df.columns for c in ["firm_id", "fiscal_year"]):
            dup = int(df.duplicated(["firm_id", "fiscal_year"]).sum())
            if dup:
                errors.append(f"bridge duplicate firm-year rows: {dup}")
        checks["bridge_rows"] = int(len(df))
    exists_nonempty(bridge_meta, errors, "Stage1→Stage2 bridge metadata")
    result = {"stage": "verify_stage1_outputs", "status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": warnings, "checks": checks}
    out = final / "ledgers" / "stage1_contract_verification.json"
    write_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
