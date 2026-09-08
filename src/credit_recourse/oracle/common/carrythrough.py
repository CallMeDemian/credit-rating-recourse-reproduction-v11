#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Oracle firm-year context carry-through helpers.

These helpers preserve non-model context columns (market/sector/industry labels)
after Stage1 merges.  They are intentionally fail-fast when requested, because
all-UNKNOWN market/sector exports make downstream stratified diagnostics stale.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

UNKNOWN_MARKERS = {"", "UNKNOWN", "UNK", "NONE", "NULL", "NAN", "NA", "N/A", "<NA>", "미상"}
MARKET_VALUES = {"KOSPI", "KOSDAQ", "KONEX"}


def _clean_string_series(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip()


def _valid_mask(s: pd.Series, *, role: str) -> pd.Series:
    x = _clean_string_series(s)
    up = x.str.upper()
    mask = x.notna() & ~up.isin(UNKNOWN_MARKERS)
    if role == "sector":
        # Stage00_01 historically wrote sector_7 = 시장, which is not a real sector.
        mask = mask & ~up.isin(MARKET_VALUES)
    return mask.fillna(False)


def _first_valid(df: pd.DataFrame, columns: Sequence[str], *, role: str) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for col in columns:
        if col not in df.columns:
            continue
        s = _clean_string_series(df[col])
        m = out.isna() & _valid_mask(s, role=role)
        out.loc[m] = s.loc[m]
    return out


def coalesce_oracle_context(
    df: pd.DataFrame,
    *,
    market_cols: Sequence[str] = (
        "시장_n", "시장_y", "source_market", "source_market_n", "source_market_y",
        "market_n", "market_y", "시장", "시장_x", "market", "market_x",
    ),
    sector_cols: Sequence[str] = (
        "sector_7_n", "sector_7_y", "sector_7_nonfinancial", "sector_7",
        "sector_7_x", "industry_class", "sector",
    ),
    industry_code_cols: Sequence[str] = ("산업코드_n", "산업코드_y", "산업코드", "산업코드_x"),
    industry_name_cols: Sequence[str] = ("산업명_n", "산업명_y", "산업명", "산업명_x"),
    require_non_unknown: bool = False,
    label: str = "oracle panel",
) -> pd.DataFrame:
    """Return a copy with canonical 시장/sector_7/산업코드/산업명 columns.

    Priority intentionally prefers nonfinancial/market-bundle columns over the
    Stage00_01 base columns, because base ``sector_7`` may be a legacy market
    proxy and should not shadow the real Stage00_03 sector mapping.
    """
    out = df.copy()

    market = _first_valid(out, market_cols, role="market")
    sector = _first_valid(out, sector_cols, role="sector")
    industry_code = _first_valid(out, industry_code_cols, role="industry")
    industry_name = _first_valid(out, industry_name_cols, role="industry")

    out["시장"] = market.fillna("UNKNOWN").astype("string").str.upper()
    out["sector_7"] = sector.fillna("UNKNOWN").astype("string")
    if any(c in out.columns for c in industry_code_cols):
        out["산업코드"] = industry_code
    if any(c in out.columns for c in industry_name_cols):
        out["산업명"] = industry_name

    if require_non_unknown:
        market_unknown = int((~_valid_mask(out["시장"], role="market")).sum())
        sector_unknown = int((~_valid_mask(out["sector_7"], role="sector")).sum())
        n = int(len(out))
        if n and market_unknown == n:
            raise ValueError(f"{label}: 시장 is all UNKNOWN after context carry-through")
        if n and sector_unknown == n:
            raise ValueError(f"{label}: sector_7 is all UNKNOWN after context carry-through")
    return out


def context_distribution(df: pd.DataFrame) -> dict:
    """Small JSON-serializable market/sector diagnostic."""
    out: dict = {"rows": int(len(df))}
    for col in ["시장", "sector_7", "산업코드", "산업명"]:
        if col in df.columns:
            vc = df[col].astype("string").fillna("UNKNOWN").value_counts(dropna=False)
            out[col] = {"unique": int(vc.shape[0]), "top": {str(k): int(v) for k, v in vc.head(20).items()}}
            out[col]["unknown_count"] = int((~_valid_mask(df[col], role="sector" if col == "sector_7" else "market")).sum()) if col in ["시장", "sector_7"] else int(df[col].isna().sum())
    return out
