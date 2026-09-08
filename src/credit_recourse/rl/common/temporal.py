from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd
import yaml

from credit_recourse.rl.common.io import config_root


@dataclass(frozen=True)
class TemporalContract:
    path: Path
    sha256: str
    eval_base_year: int
    inner_train_year_max: int
    inner_dev_year: int
    train_transition_year_max: int
    raw: dict[str, Any]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_temporal_contract(project_root: Path) -> TemporalContract:
    """Load the binding final-freeze temporal split contract.

    The contract is intentionally external to the package source and must live
    under data/final_freeze/configs so local runs cannot silently diverge from
    the frozen run configuration.
    """
    path = config_root(project_root) / 'temporal_split.yaml'
    if not path.exists():
        raise FileNotFoundError(f'Missing binding temporal split contract: {path}')
    raw = yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    def req_int(name: str) -> int:
        if name not in raw:
            raise ValueError(f'temporal_split.yaml missing required key: {name}')
        try:
            return int(raw[name])
        except Exception as exc:
            raise ValueError(f'temporal_split.yaml key {name} must be int-like: {raw.get(name)!r}') from exc
    contract = TemporalContract(
        path=path,
        sha256=_sha256_file(path),
        eval_base_year=req_int('eval_base_year'),
        inner_train_year_max=req_int('inner_train_year_max'),
        inner_dev_year=req_int('inner_dev_year'),
        train_transition_year_max=req_int('train_transition_year_max'),
        raw=raw,
    )
    if not (contract.inner_train_year_max < contract.inner_dev_year <= contract.train_transition_year_max < contract.eval_base_year):
        raise ValueError(
            'Invalid temporal split ordering: expected inner_train_year_max < inner_dev_year <= '
            'train_transition_year_max < eval_base_year, got '
            f'{contract.inner_train_year_max}, {contract.inner_dev_year}, '
            f'{contract.train_transition_year_max}, {contract.eval_base_year}'
        )
    return contract


def temporal_metadata(contract: TemporalContract, *, stage: str) -> dict[str, Any]:
    return {
        'stage': stage,
        'temporal_contract_path': str(contract.path),
        'temporal_contract_sha256': contract.sha256,
        'eval_base_year': contract.eval_base_year,
        'inner_train_year_max': contract.inner_train_year_max,
        'inner_dev_year': contract.inner_dev_year,
        'train_transition_year_max': contract.train_transition_year_max,
        'split_policy': 'inner_train_le_inner_train_year_max_dev_eq_inner_dev_year_final_refit_le_train_transition_year_max',
    }


def _year_series(df: pd.DataFrame) -> pd.Series:
    for c in ('fiscal_year', 'year'):
        if c in df.columns:
            y = pd.to_numeric(df[c], errors='coerce')
            if y.notna().any():
                return y
    raise ValueError('Temporal split requires fiscal_year or year column')


def temporal_train_dev_indices(df: pd.DataFrame, contract: TemporalContract, *, stage: str) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    y = _year_series(df)
    train_mask = y <= contract.inner_train_year_max
    dev_mask = y == contract.inner_dev_year
    if int(train_mask.sum()) <= 0:
        raise ValueError(f'{stage}: temporal inner-train partition is empty for fiscal_year <= {contract.inner_train_year_max}')
    if int(dev_mask.sum()) <= 0:
        raise ValueError(f'{stage}: temporal inner-dev partition is empty for fiscal_year == {contract.inner_dev_year}')
    tr_idx = np.flatnonzero(train_mask.to_numpy())
    dev_idx = np.flatnonzero(dev_mask.to_numpy())
    return tr_idx, dev_idx, temporal_metadata(contract, stage=stage) | {
        'train_rows': int(len(tr_idx)),
        'dev_rows': int(len(dev_idx)),
        'train_year_max_observed': int(y[train_mask].max()),
        'dev_year_observed': int(contract.inner_dev_year),
    }


def temporal_refit_indices(df: pd.DataFrame, contract: TemporalContract, *, stage: str) -> tuple[np.ndarray, dict[str, Any]]:
    y = _year_series(df)
    mask = y <= contract.train_transition_year_max
    if int(mask.sum()) <= 0:
        raise ValueError(f'{stage}: temporal refit partition is empty for fiscal_year <= {contract.train_transition_year_max}')
    idx = np.flatnonzero(mask.to_numpy())
    return idx, temporal_metadata(contract, stage=stage) | {
        'refit_rows': int(len(idx)),
        'refit_year_max_observed': int(y[mask].max()),
    }
