from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _norm_var_record(x: Any) -> str | None:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s or None
    if isinstance(x, dict):
        for k in ["variable_id", "name", "variable", "feature", "column", "id"]:
            v = x.get(k)
            if v is not None and str(v).strip():
                return str(v).strip()
    return None


def unique_keep_order(values: list[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        s = _norm_var_record(v)
        if s and s not in seen:
            out.append(s); seen.add(s)
    return out


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def variables_from_selected_variables_json(path: Path) -> list[str]:
    obj = load_json(path)
    vals: list[Any] = []
    if isinstance(obj, dict):
        for key in ["selected_variables", "financial_variables", "nonfinancial_variables", "variables"]:
            v = obj.get(key)
            if isinstance(v, list):
                vals.extend(v)
        # Common Stage00_04 shape: category -> list/dict records.
        for v in obj.values():
            if isinstance(v, list):
                vals.extend(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    if isinstance(vv, list):
                        vals.extend(vv)
    elif isinstance(obj, list):
        vals.extend(obj)
    return unique_keep_order(vals)


def read_csv_korean_safe(path: Path, **kwargs) -> pd.DataFrame:
    last = None
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as e:
            last = e
    if last is not None:
        raise last
    return pd.read_csv(path, **kwargs)


def variables_from_master_csv(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected variable master: {path}")
    df = read_csv_korean_safe(path)
    for col in ["variable_id", "selected_variable", "name", "variable", "feature", "column"]:
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip().tolist()
            return unique_keep_order(vals)
    raise KeyError(f"selected_variable_master has no variable id column: {path}; columns={list(df.columns)}")


def variables_from_backend_params(path: Path) -> list[str]:
    obj = load_json(path)
    vals: list[Any] = []
    if isinstance(obj, dict):
        for key in ["selected_variables", "required_variables", "features", "feature_names"]:
            v = obj.get(key)
            if isinstance(v, list):
                vals.extend(v)
        if isinstance(obj.get("variables"), list):
            vals.extend(obj["variables"])
        if isinstance(obj.get("variable_params"), list):
            vals.extend(obj["variable_params"])
    return unique_keep_order(vals)


def resolve_stage1_selected_variables(stage1_inputs_dir: Path, backend_dir: Path | None = None) -> dict[str, Any]:
    """Resolve adaptive Stage00_04 variables without a fixed Alpha list.

    Source of truth order:
      1. Stage00_04 selected_variable_master.csv
      2. Stage00_04 selected_variables_v2.json
      3. Alpha/Beta/Gamma backend params, checked for consistency when present
    """
    s4 = stage1_inputs_dir / "stage00_04_variable_selection"
    master = s4 / "selected_variable_master.csv"
    selected_json = s4 / "selected_variables_v2.json"
    selected: list[str] = []
    source = None
    if master.exists():
        selected = variables_from_master_csv(master); source = str(master)
    if not selected and selected_json.exists():
        selected = variables_from_selected_variables_json(selected_json); source = str(selected_json)
    if not selected:
        raise ValueError(f"Could not resolve selected variables from {master} or {selected_json}")

    backend_vars: dict[str, list[str]] = {}
    if backend_dir is not None:
        paths = {
            "alpha": backend_dir / "alpha" / "oracle_alpha_params.json",
            "beta": backend_dir / "beta" / "benchmark_beta_params.json",
            "gamma": backend_dir / "gamma" / "benchmark_gamma_params.json",
        }
        for name, p in paths.items():
            if p.exists():
                backend_vars[name] = variables_from_backend_params(p)

    selected_set = set(selected)
    mismatches: dict[str, dict[str, list[str]]] = {}
    for name, vars_ in backend_vars.items():
        vset = set(vars_)
        if vset and vset != selected_set:
            # Robustness backends may intentionally use same selected universe; final contract requires no silent divergence.
            mismatches[name] = {
                "missing_from_backend": sorted(selected_set - vset),
                "extra_in_backend": sorted(vset - selected_set),
            }
    return {
        "selected_variables": selected,
        "selected_variable_source": source,
        "backend_selected_variables": backend_vars,
        "backend_mismatches": mismatches,
    }


def write_contract_metadata(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
