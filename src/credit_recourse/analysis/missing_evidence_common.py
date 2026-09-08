from __future__ import annotations
import json, math
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from .claim_evidence_common import parse_table, repo_rel, write_csv, write_json

def find_tables(root: Path, patterns: Iterable[str]) -> list[Path]:
    out = []
    for pat in patterns:
        out.extend(root.glob(pat))
    return sorted({p.resolve(): p for p in out if p.is_file() and p.suffix.lower() in {'.csv', '.parquet'}}.values())

def load_first(root: Path, patterns: Iterable[str], required_cols: Iterable[str]=()) -> tuple[Path, pd.DataFrame]:
    cand = find_tables(root, patterns)
    failures = []
    for p in cand:
        try:
            d = parse_table(p)
            if all((c in d.columns for c in required_cols)):
                return (p, d)
            failures.append(f'{p}: missing {sorted(set(required_cols) - set(d.columns))}')
        except Exception as exc:
            failures.append(f'{p}: {exc}')
    raise RuntimeError('No compatible input table. ' + '; '.join(failures[:20]))

def col(df: pd.DataFrame, *names: str) -> str:
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f'none of columns present: {names}')

def action_cols(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise KeyError(f'no action columns for prefix {prefix}')
    return sorted(cols)
