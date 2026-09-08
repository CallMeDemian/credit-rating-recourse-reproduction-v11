from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd
import yaml


def project_root_arg(path: str | Path) -> Path:
    return Path(path).resolve()


def final_root(project_root: Path) -> Path:
    return project_root / "data" / "final_freeze"


def config_root(project_root: Path) -> Path:
    return final_root(project_root) / "configs"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_safe(obj: Any) -> Any:
    """Convert pandas/numpy scalars, timestamps, NaN/Inf and non-string keys to JSON-safe objects."""
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat() if not pd.isna(obj) else None
    if isinstance(obj, (pd.Timedelta,)):
        return str(obj) if not pd.isna(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            sk = _json_safe(k)
            if sk is None:
                sk = "null"
            if not isinstance(sk, str):
                sk = str(sk)
            out[sk] = _json_safe(v)
        return out
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_json_safe(v) for v in obj.tolist()]
    if pd.isna(obj) if not isinstance(obj, (list, tuple, dict, set, np.ndarray)) else False:
        return None
    return obj


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(obj)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def read_parquet_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required parquet not found: {path}")
    df = pd.read_parquet(path)
    if len(df) == 0:
        raise ValueError(f"Required parquet is empty: {path}")
    return df
