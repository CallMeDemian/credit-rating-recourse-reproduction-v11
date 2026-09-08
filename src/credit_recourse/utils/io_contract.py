from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

MOJIBAKE_TOKENS = ["�", "ï¿½", "ì", "ê", "í", "Ã"]


def configure_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use ASCII-escaped JSON for Windows PowerShell compatibility.  The object
    # semantics remain Unicode after json.loads, while Get-Content |
    # ConvertFrom-Json in legacy consoles no longer corrupts Korean keys/values
    # into malformed JSON.
    path.write_text(json.dumps(obj, ensure_ascii=True, indent=2, default=str), encoding="utf-8")


def read_json(path: Path) -> Any:
    # Accept both plain UTF-8 and UTF-8-with-BOM ledgers/checkpoint metadata.
    # Several historical artifacts were written with BOM; boundary verifiers
    # should read them without weakening any schema/hash/status gates.
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_csv_korean_safe(path: Path, **kwargs: Any) -> pd.DataFrame:
    last: Exception | None = None
    for enc in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError as exc:
            last = exc
    if last is not None:
        raise last
    return pd.read_csv(path, **kwargs)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_no_mojibake_columns(df: pd.DataFrame, where: str = "dataframe") -> None:
    bad_cols = [str(c) for c in df.columns if any(tok in str(c) for tok in MOJIBAKE_TOKENS)]
    if bad_cols:
        raise ValueError(f"Mojibake detected in {where} columns: {bad_cols[:20]}")
    # Do not reject legitimate placeholder strings globally, but reject obvious
    # question-mark replacement in column names.
    qbad = [str(c) for c in df.columns if re.search(r"\?{3,}", str(c))]
    if qbad:
        raise ValueError(f"Question-mark mojibake detected in {where} columns: {qbad[:20]}")


def flatten_selected_variables(obj: Any) -> list[str]:
    out: list[str] = []

    def add(x: Any) -> None:
        if x is None:
            return
        if isinstance(x, str):
            s = x.strip()
            if s and s not in out:
                out.append(s)
            return
        if isinstance(x, dict):
            for key in ["variable_id", "variable", "feature", "name", "id", "column", "raw_variable", "canonical_variable"]:
                if key in x:
                    add(x.get(key))
                    return
            for v in x.values():
                if isinstance(v, (list, tuple, dict)):
                    add(v)
            return
        if isinstance(x, (list, tuple, set)):
            for v in x:
                add(v)

    add(obj)
    return out


def selected_variables_from_master(master_csv: Path) -> list[str]:
    if not Path(master_csv).exists():
        return []
    df = read_csv_korean_safe(Path(master_csv))
    for col in ["variable_id", "variable", "feature", "name", "id", "column"]:
        if col in df.columns:
            return [str(x).strip() for x in df[col].dropna().tolist() if str(x).strip()]
    return []


def selected_variables_from_json(path: Path) -> list[str]:
    if not Path(path).exists():
        return []
    return flatten_selected_variables(read_json(Path(path)))


def selected_variables_from_backend_params(params_path: Path) -> list[str]:
    if not Path(params_path).exists():
        return []
    obj = read_json(Path(params_path))
    candidates = []
    if isinstance(obj, dict):
        for key in ["selected_variables", "required_variables", "features", "variables"]:
            if key in obj:
                candidates.extend(flatten_selected_variables(obj[key]))
    return list(dict.fromkeys(candidates))


def resolve_selected_variables(final_root: Path, backend_params: Path | None = None) -> dict[str, Any]:
    final_root = Path(final_root)
    s4 = final_root / "stage1_oracle_inputs" / "stage00_04_variable_selection"
    cfg = final_root / "configs"
    master_candidates = [s4 / "selected_variable_master.csv", cfg / "selected_variable_master.csv", s4 / "outputs" / "selected_variable_master.csv"]
    json_candidates = [s4 / "selected_variables_v2.json", s4 / "outputs" / "selected_variables_v2.json"]
    master_path = next((p for p in master_candidates if p.exists()), None)
    json_path = next((p for p in json_candidates if p.exists()), None)
    master_vars = selected_variables_from_master(master_path) if master_path else []
    json_vars = selected_variables_from_json(json_path) if json_path else []
    backend_vars = selected_variables_from_backend_params(backend_params) if backend_params else []
    selected = master_vars or backend_vars or json_vars
    selected = list(dict.fromkeys([v for v in selected if v]))
    return {
        "selected_variables": selected,
        "selected_variable_master_path": str(master_path) if master_path else None,
        "selected_variables_json_path": str(json_path) if json_path else None,
        "backend_params_path": str(backend_params) if backend_params else None,
        "master_variables": master_vars,
        "json_variables": json_vars,
        "backend_variables": backend_vars,
    }
