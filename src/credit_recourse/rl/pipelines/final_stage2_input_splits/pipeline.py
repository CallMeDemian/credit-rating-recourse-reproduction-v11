#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
Build final Stage2 input splits for candidate projection / BC / IQL.

This is the final Stage2 input-split builder for the final freeze contract.

Inputs:
  data/final_freeze/stage1_oracle_inputs/alpha_vanilla_input_candidate.parquet

Outputs:
  data/final_freeze/stage2_candidate_projection/input_splits/phase2_bc.parquet
  data/final_freeze/stage2_candidate_projection/input_splits/phase3_iql.parquet
  data/final_freeze/stage2_candidate_projection/input_splits/phase_eval.parquet
  data/final_freeze/stage2_candidate_projection/input_splits/metadata.json
  data/final_freeze/stage2_candidate_projection/input_splits/action_source_coverage.csv
  data/final_freeze/stage2_candidate_projection/input_splits/transition_gap_diagnostics.json

Design:
- Uses firm_id/fiscal_year consecutive one-year transitions.
- Uses final Alpha input complete rows.
- Consumes Stage2A direct raw-data action-source panel built from data/raw/raw_all Excel files.
- Engineered-ratio/R-code proxy fallback is forbidden.
- Row-level missing direct raw actions are flagged by action_observed__<dim>; dimension-level source gaps fail fast.
- Final default phase2_bc construction is seed-free: all broad training rows are used.
  Historical RNG subsetting is preserved only under explicit --bc-split-policy legacy_seeded.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from credit_recourse.rl.common.io import load_yaml, write_json as _write_json_safe
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_metadata
from credit_recourse.rl.contracts.stage2_avs256_enrichment import enrich_avs256_panel, write_feature_manifest


ACTION_DIMS = [
    "ppe_pct",
    "inv_turnover_chg",
    "ar_turnover_chg",
    "ap_turnover_chg",
    "short_debt_pct",
    "long_debt_pct",
    "bond_pct",
    "revenue_growth",
    "cogs_ratio_chg",
    "sga_ratio_chg",
]

BOUNDS = {
    "ppe_pct": (-0.50, 0.50),
    "inv_turnover_chg": (-3.00, 3.00),
    "ar_turnover_chg": (-3.00, 3.00),
    "ap_turnover_chg": (-3.00, 3.00),
    "short_debt_pct": (-1.00, 1.00),
    "long_debt_pct": (-0.50, 0.50),
    "bond_pct": (-0.50, 0.50),
    "revenue_growth": (-0.15, 0.15),
    "cogs_ratio_chg": (-0.03, 0.03),
    "sga_ratio_chg": (-0.02, 0.02),
}

# Fixed Alpha variable lists are forbidden in the final clean-run contract.
# The selected-variable universe is loaded dynamically from the Stage1→Stage2
# bridge metadata or selected_variable_master.csv.

# Stage2 sector-Φ components.  The split builder materializes and validates
# these columns so the downstream projection/reward stage is not dependent on
# accidental pass-through columns in alpha_vanilla_input_candidate.parquet.
PHI_COMPONENTS = [
    "derived__roa_proxy",
    "derived__operating_margin",
    "derived__cogs_to_revenue",
    "derived__sga_to_revenue",
    "derived__financial_cost_to_revenue",
    "derived__debt_to_assets",
]

KOREAN_PHI_ALIASES = {
    "derived__roa_proxy": ["derived__roa_proxy", "R006", "총자본순이익률(IFRS)", "ROA", "roa_proxy"],
    "derived__operating_margin": ["derived__operating_margin", "R136", "매출액정상영업이익률(IFRS)", "영업이익률", "operating_margin"],
    "derived__cogs_to_revenue": ["derived__cogs_to_revenue", "R157", "매출원가 대 매출액비율(IFRS)", "cogs_to_revenue"],
    "derived__sga_to_revenue": ["derived__sga_to_revenue", "R185", "영업비용 대 영업수익비율(IFRS)", "sga_to_revenue"],
    "derived__financial_cost_to_revenue": ["derived__financial_cost_to_revenue", "R085", "금융비용부담률(IFRS)", "financial_cost_to_revenue"],
    "derived__debt_to_assets": ["derived__debt_to_assets", "R064", "타인자본구성비율(IFRS)", "부채비율(IFRS)", "debt_to_assets"],
}

ACTION_COLUMNS = [f"action__{d}" for d in ACTION_DIMS]



def load_final_action_contract(root: Path) -> dict[str, Any]:
    cfg = root / "data/final_freeze/configs/final_action_contract.yaml"
    if not cfg.exists():
        raise FileNotFoundError(f"Missing final action contract: {cfg}")
    action = load_yaml(cfg)
    cols = list(action.get("action_columns") or [f"action__{x}" for x in action.get("canonical_action_order", [])])
    if cols != [
        "action__ppe_pct","action__inv_turnover_chg","action__ar_turnover_chg","action__ap_turnover_chg","action__short_debt_pct","action__long_debt_pct","action__bond_pct","action__revenue_growth","action__cogs_ratio_chg","action__sga_ratio_chg"
    ]:
        raise ValueError(f"Final action columns are not the v32 10D order: {cols}")
    raw_bounds = action.get("action_bounds") or {}
    bounds = {}
    for col in cols:
        raw = col.replace("action__", "")
        v = raw_bounds.get(raw) or raw_bounds.get(col)
        if v is None:
            raise ValueError(f"Missing action bound for {col}")
        bounds[raw] = (float(v[0]), float(v[1]))
    return {"path": str(cfg), "sha256": sha256_file(cfg), "action_columns": cols, "action_dims": [c.replace("action__", "") for c in cols], "bounds": bounds}

def activate_final_action_contract(root: Path) -> dict[str, Any]:
    global ACTION_DIMS, ACTION_COLUMNS, BOUNDS
    meta = load_final_action_contract(root)
    ACTION_DIMS = list(meta["action_dims"])
    ACTION_COLUMNS = list(meta["action_columns"])
    BOUNDS = dict(meta["bounds"])
    return meta

def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """JSON-safe writer for pandas/numpy diagnostics.

    Stage2 diagnostics may contain numpy scalar keys (np.int64) after
    groupby/value_counts.  The project-wide writer normalizes both keys and
    values, so this local wrapper must delegate to it rather than json.dumps
    directly.
    """
    _write_json_safe(path, obj)


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



def _norm_firm_id(x: Any) -> str | None:
    """Normalize Korean listed-firm identifiers for Stage2 joins.

    Raw statement panels may carry codes as ``A005930``, ``005930``,
    ``5930.0``, or with Korean/Excel decoration.  Stage1 bridge and Stage2A
    use six-digit exchange codes, so every Stage2 boundary merge must remove
    non-digits and zfill to six digits.  If a value has no digits, return the
    stripped value so downstream validation can fail visibly instead of silently
    manufacturing a key.
    """
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(?:\.0)?", s):
        s = str(int(float(s)))
    digits = re.sub(r"\D", "", s)
    return digits.zfill(6) if digits else s


def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _norm_col_name(x: Any) -> str:
    return re.sub(r"[\s_()/·ㆍ\-\[\]{}*]+", "", str(x).lower())


def _find_first_column(columns: list[Any], candidates: list[str]) -> str | None:
    """Find a physical column by exact or normalized alias match.

    Korean raw/cleaned panels have drifted between ``거래소코드``, ``종목코드``,
    ``회사코드`` and English bridge names.  Exact membership alone missed the
    current cleaned cash-flow panel's exchange-code key.  Normalized matching is
    intentionally alias-only: it broadens known key spellings without guessing
    arbitrary columns.
    """
    physical = [str(c) for c in columns]
    norm = {_norm_col_name(c): str(c) for c in physical}
    for cand in candidates:
        if cand in physical:
            return cand
        hit = norm.get(_norm_col_name(cand))
        if hit is not None:
            return hit
    return None


def _norm_fiscal_year(x: Any) -> int | None:
    """Normalize fiscal-year keys from integer, float, or NICE date strings.

    Raw NICE statement exports usually encode fiscal years as values like
    ``2017/12`` while downstream Stage2 contracts use integer calendar years.
    Returning ``None`` on unparseable values keeps the existing fail-fast/dropna
    behavior visible instead of silently manufacturing a year.
    """
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) and np.isfinite(x):
        return int(x)
    s = str(x).strip()
    if not s:
        return None
    m = re.search(r"(19\d{2}|20\d{2})", s)
    if m:
        return int(m.group(1))
    return None


def _coerce_fiscal_year_series(s: pd.Series) -> pd.Series:
    return s.map(_norm_fiscal_year).astype("Int64")


def _resolve_cash_flow_panel(root: Path, explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if not explicit_path.is_absolute():
            explicit_path = root / explicit_path
        candidates.append(explicit_path.resolve())
    candidates.extend([
        root / "data/final_freeze/stage1_oracle_inputs/stage00_01_rating_statement_integration/cleaned_statement_panels/현금흐름표_clean.parquet",
        root / "data/final_freeze/stage00_01_rating_statement_integration/cleaned_statement_panels/현금흐름표_clean.parquet",
        root / "data/final_freeze/stage1_oracle_inputs/cleaned_statement_panels/현금흐름표_clean.parquet",
        root / "data/final_freeze/stage1_oracle_inputs/현금흐름표_clean.parquet",
    ])
    seen: set[str] = set()
    ordered: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen:
            ordered.append(c)
            seen.add(key)
    for c in ordered:
        if c.exists():
            return c

    # Deterministic last-resort discovery inside final_freeze only.  This fixes
    # layout drift without masking a missing input: zero or multiple distinct
    # matches still fail with an explicit evidence payload.
    discovered = sorted((root / "data/final_freeze").glob("**/현금흐름표_clean.parquet"))
    if len(discovered) == 1:
        return discovered[0]
    if len(discovered) > 1:
        raise FileExistsError({
            "message": "Multiple cash-flow cleaned panels found; pass --cash-flow-panel explicitly.",
            "matches": [str(p) for p in discovered],
            "preferred_candidates": [str(p) for p in ordered],
        })
    raise FileNotFoundError({
        "message": "Missing cash-flow cleaned panel for --join-cash-flow-substrate.",
        "preferred_candidates": [str(c) for c in ordered],
        "recursive_search_root": str(root / "data/final_freeze"),
        "remedy": "Copy 현금흐름표_clean.parquet under one preferred path or pass --cash-flow-panel explicitly.",
    })



def _ocf_encoder_columns_to_scrub(columns: list[str]) -> list[str]:
    tokens = [
        "sim__operating_cf",
        "derived__ocf_to_total_debt",
        "cash_flow__[U01D100000000]",
        "U01D100000000",
    ]
    out: list[str] = []
    for c in columns:
        sc = str(c)
        if sc.startswith("reward_only__") or sc.startswith("next__reward_only__"):
            continue
        if any(tok in sc for tok in tokens):
            out.append(sc)
    return out


def _preserve_reward_only_cf_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    alias_pairs = [
        ("reward_only__sim__operating_cf", "sim__operating_cf"),
        ("reward_only__sim__capex", "sim__capex"),
        ("next__reward_only__sim__operating_cf", "next__sim__operating_cf"),
        ("next__reward_only__sim__capex", "next__sim__capex"),
    ]
    for alias, src in alias_pairs:
        if src in out.columns and alias not in out.columns:
            out[alias] = out[src]
    return out


def _apply_cash_flow_reward_only_encoder_guard(df: pd.DataFrame, *, enabled: bool) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta = {"enabled": bool(enabled), "scrubbed_columns": [], "policy": "not_applied"}
    if not enabled:
        return df, meta
    out = _preserve_reward_only_cf_aliases(df)
    scrub_cols = _ocf_encoder_columns_to_scrub(list(out.columns))
    for c in scrub_cols:
        out[c] = 0.0
    meta.update({
        "policy": "reward_only_cf_aliases_retained_for_reward_and_ocf_encoder_columns_zeroed_to_preserve_legacy_encoder_distribution",
        "scrubbed_columns": scrub_cols,
        "reward_alias_columns": [c for c in out.columns if str(c).startswith("reward_only__") or str(c).startswith("next__reward_only__")],
    })
    return out, meta

def join_cash_flow_substrate(root: Path, df: pd.DataFrame, *, cash_flow_panel: str | None = None, encoder_mode: str = "reward_only") -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge operating-CF and CF-statement capex levels onto the Stage1→Stage2 bridge.

    Default pipeline behavior is unchanged unless this helper is explicitly called
    by --join-cash-flow-substrate.  Column names intentionally contain U-codes so
    stage2_avs256_enrichment._find_by_code can materialize sim__operating_cf.
    """
    if encoder_mode not in {"reward_only", "full"}:
        raise ValueError(f"cash_flow_encoder_mode must be reward_only|full, got {encoder_mode!r}")
    cf_path = _resolve_cash_flow_panel(root, cash_flow_panel)
    cf = pd.read_parquet(cf_path)
    key_candidates = ["firm_id", "corp_code", "거래소코드", "종목코드", "회사코드", "회사번호", "stock_code"]
    year_candidates = ["fiscal_year", "year", "회계년도", "결산년도", "사업년도"]
    key = _find_first_column(list(cf.columns), key_candidates)
    year = _find_first_column(list(cf.columns), year_candidates)
    if key is None or year is None:
        raise ValueError({
            "message": "Cash-flow panel lacks firm/year key columns",
            "key": key,
            "year": year,
            "path": str(cf_path),
            "available_columns_sample": [str(c) for c in list(cf.columns)[:80]],
            "key_candidates": key_candidates,
            "year_candidates": year_candidates,
        })
    ocf_col = next((c for c in cf.columns if "U01D100000000" in str(c)), None)
    capex_col = next((c for c in cf.columns if "U01D206012400" in str(c)), None)
    if ocf_col is None:
        raise ValueError(f"Cash-flow panel missing operating CF U01D100000000: {cf_path}")
    use_cols = [key, year, ocf_col] + ([capex_col] if capex_col is not None else [])
    slim = cf[use_cols].copy()
    slim = slim.rename(columns={key: "firm_id", year: "fiscal_year", ocf_col: "cash_flow__[U01D100000000]영업활동현금흐름(IFRS)(천원)"})
    if capex_col is not None:
        slim = slim.rename(columns={capex_col: "cash_flow__[U01D206012400]유형자산취득(IFRS)(천원)"})
    slim["firm_id"] = slim["firm_id"].map(_norm_firm_id)
    slim["fiscal_year"] = _coerce_fiscal_year_series(slim["fiscal_year"])
    slim = slim.dropna(subset=["firm_id", "fiscal_year"]).drop_duplicates(["firm_id", "fiscal_year"], keep="first")
    out = df.copy()
    out["firm_id"] = out["firm_id"].map(_norm_firm_id)
    out["fiscal_year"] = _coerce_fiscal_year_series(out["fiscal_year"])
    drop = [c for c in ["cash_flow__[U01D100000000]영업활동현금흐름(IFRS)(천원)", "cash_flow__[U01D206012400]유형자산취득(IFRS)(천원)"] if c in out.columns]
    if drop:
        out = out.drop(columns=drop)
    out = out.merge(slim, on=["firm_id", "fiscal_year"], how="left", validate="many_to_one")
    out["reward_only__sim__operating_cf"] = pd.to_numeric(out["cash_flow__[U01D100000000]영업활동현금흐름(IFRS)(천원)"], errors="coerce")
    if "cash_flow__[U01D206012400]유형자산취득(IFRS)(천원)" in out.columns:
        out["reward_only__sim__capex"] = pd.to_numeric(out["cash_flow__[U01D206012400]유형자산취득(IFRS)(천원)"], errors="coerce").abs()
    ocf = pd.to_numeric(out["cash_flow__[U01D100000000]영업활동현금흐름(IFRS)(천원)"], errors="coerce")
    nonzero_rate = float((ocf.fillna(0.0).abs() > 1e-12).mean()) if len(out) else 0.0
    if nonzero_rate <= 0.05:
        raise ValueError(f"Cash-flow join produced degenerate OCF: nonzero_rate={nonzero_rate:.4f}; path={cf_path}")
    meta = {
        "cash_flow_substrate_joined": True,
        "cash_flow_source_path": _safe_relpath(cf_path, root),
        "cash_flow_source_sha256": sha256_file(cf_path),
        "cash_flow_encoder_mode": encoder_mode,
        "operating_cf_nonzero_rate_after_join": nonzero_rate,
        "capex_joined": capex_col is not None,
        "default_off_regression_policy": "no --join-cash-flow-substrate leaves prior substrate untouched",
    }
    return out, meta



def load_stage1_bridge_selected_variables(root: Path, input_path: Path, df: pd.DataFrame) -> tuple[list[str], dict[str, Any]]:
    """Load dynamic selected-variable contract for Stage2.

    Priority:
      1. alpha_vanilla_input_candidate_metadata.json emitted by Stage1 bridge
      2. data/final_freeze/configs/stage1_to_stage2_bridge_metadata.json
      3. Stage00_04 selected_variable_master.csv
    """
    candidates = [
        input_path.with_name("alpha_vanilla_input_candidate_metadata.json"),
        root / "data/final_freeze/configs/stage1_to_stage2_bridge_metadata.json",
        root / "data/final_freeze/stage1_oracle_inputs/stage00_04_variable_selection/selected_variable_master.csv",
    ]
    meta: dict[str, Any] = {"source_candidates": [str(x) for x in candidates]}
    selected: list[str] = []
    source = None
    for p in candidates[:2]:
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8"))
            selected = [str(x).strip() for x in obj.get("selected_variables_used", []) if str(x).strip()]
            if selected:
                source = str(p); meta["bridge_metadata"] = obj; break
    if not selected and candidates[2].exists():
        master = read_csv_korean_safe(candidates[2])
        for col in ["variable_id", "selected_variable", "name", "variable", "feature", "column"]:
            if col in master.columns:
                selected = [str(x).strip() for x in master[col].dropna().tolist() if str(x).strip()]
                source = str(candidates[2]); break
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("Stage2 split cannot resolve dynamic selected variables; fixed Alpha-variable fallback is forbidden")
    missing = [v for v in selected if v not in df.columns]
    if missing:
        raise KeyError("Stage2 input is missing dynamic selected variables: " + json.dumps(missing, ensure_ascii=False))
    meta.update({"selected_variables_used": selected, "selected_variable_source": source, "missing_selected_variables": missing})
    return selected, meta

def _norm_name(s: str) -> str:
    return re.sub(r"[\s_()/·ㆍ\-\[\]{}]+", "", str(s).lower())


def find_col(df: pd.DataFrame, any_groups: list[list[str]], numeric_only: bool = True) -> str | None:
    """Find first column whose normalized name contains all tokens in any group."""
    candidates = []
    for c in df.columns:
        if c in {"firm_id", "fiscal_year"}:
            continue
        if numeric_only and not pd.api.types.is_numeric_dtype(df[c]):
            continue
        nc = _norm_name(c)
        for group in any_groups:
            if all(_norm_name(tok) in nc for tok in group):
                candidates.append(c)
                break
    # Prefer shorter names to avoid source metadata columns with long names.
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: (len(str(x)), str(x)))[0]


def pct_to_unit(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce")
    # Most KIS ratios are in percent units. Convert to decimal unless already small.
    med = s.abs().median(skipna=True)
    if pd.notna(med) and med > 1.5:
        return s / 100.0
    return s


def clip_dim(s: pd.Series, dim: str) -> pd.Series:
    lo, hi = BOUNDS[dim]
    return pd.to_numeric(s, errors="coerce").clip(lo, hi)


def _series_or_none(df: pd.DataFrame, col: str | None) -> pd.Series | None:
    """Return a single Series even when a prior rename produced duplicate labels."""
    if col is None or col not in df.columns:
        return None
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        # Prefer the right-most duplicate, which is usually the next-state copy
        # after a suffix-stripping operation. This branch is guarded because
        # duplicate column labels otherwise make pandas arithmetic silently wrong.
        obj = obj.iloc[:, -1]
    return pd.to_numeric(obj, errors="coerce")


def _next_series(df_t: pd.DataFrame, df_tp1: pd.DataFrame, col: str | None) -> pd.Series | None:
    if col is None:
        return None
    for candidate in (f"next__{col}", f"{col}__next"):
        s = _series_or_none(df_t, candidate)
        if s is not None:
            return s
    return _series_or_none(df_tp1, col)


def ratio_level_delta(df_t: pd.DataFrame, df_tp1: pd.DataFrame, col: str | None, dim: str) -> tuple[pd.Series, pd.Series]:
    a_raw = _series_or_none(df_t, col)
    b_raw = _next_series(df_t, df_tp1, col)
    if a_raw is None or b_raw is None:
        return pd.Series(0.0, index=df_t.index), pd.Series(False, index=df_t.index)
    a = pct_to_unit(a_raw)
    b = pct_to_unit(b_raw)
    out = clip_dim(b - a, dim)
    mask = out.notna()
    return out.fillna(0.0), mask


def growth_rate_current(df_t: pd.DataFrame, col: str | None, dim: str) -> tuple[pd.Series, pd.Series]:
    if col is None:
        return pd.Series(0.0, index=df_t.index), pd.Series(False, index=df_t.index)
    out = clip_dim(pct_to_unit(df_t[col]), dim)
    mask = out.notna()
    return out.fillna(0.0), mask


def growth_rate_next(df_t: pd.DataFrame, df_tp1: pd.DataFrame, col: str | None, dim: str) -> tuple[pd.Series, pd.Series]:
    """Use t→t+1 growth semantics for pseudo-actions.

    KIS/NICE growth-ratio columns observed in year t usually encode t-1→t.
    For an RL transition (s_t, a_t, s_{t+1}), the pseudo-action must use the
    year t+1 observation. Prefer explicit next__/<col>__next columns and guard
    against duplicate-label DataFrame returns.
    """
    series = _next_series(df_t, df_tp1, col)
    if series is None:
        return pd.Series(0.0, index=df_t.index), pd.Series(False, index=df_t.index)
    out = clip_dim(pct_to_unit(series), dim)
    mask = out.notna()
    return out.fillna(0.0), mask


def log_amount_pct_change(df_t: pd.DataFrame, df_tp1: pd.DataFrame, col: str | None, dim: str) -> tuple[pd.Series, pd.Series]:
    a = _series_or_none(df_t, col)
    b = _next_series(df_t, df_tp1, col)
    if a is None or b is None:
        return pd.Series(0.0, index=df_t.index), pd.Series(False, index=df_t.index)
    # If already log amount, delta log approximates pct change. If raw amount, log first.
    if a.abs().median(skipna=True) > 100:
        a = np.log(a.where(a > 0))
    if b.abs().median(skipna=True) > 100:
        b = np.log(b.where(b > 0))
    out = clip_dim(b - a, dim)
    mask = out.notna()
    return out.fillna(0.0), mask


def build_source_map(df: pd.DataFrame) -> dict[str, str | None]:
    # These are intentionally broad and audited; exact Korean names vary across exports.
    return {
        "ppe_pct": find_col(df, [["유형", "자산", "증가"], ["tangible", "asset", "growth"], ["ppe", "growth"]]),
        "inv_turnover_chg": find_col(df, [["재고", "회전"], ["inventory", "turnover"]]),
        "ar_turnover_chg": find_col(df, [["매출채권", "회전"], ["receivable", "turnover"], ["ar", "turnover"]]),
        "ap_turnover_chg": find_col(df, [["매입채무", "회전"], ["payable", "turnover"], ["ap", "turnover"]]),
        "short_debt_pct": find_col(df, [["단기", "차입"], ["short", "debt"], ["short", "borrow"]]),
        "long_debt_pct": find_col(df, [["장기", "차입"], ["long", "debt"], ["long", "borrow"]]),
        "bond_pct": find_col(df, [["사채"], ["bond"]]),
        "revenue_growth": find_col(df, [["매출", "증가"], ["revenue", "growth"], ["sales", "growth"]]),
        "cogs_ratio_chg": find_col(df, [["매출원가", "매출"], ["cogs", "revenue"], ["cost", "sales"]]),
        "sga_ratio_chg": find_col(df, [["판매", "관리", "매출"], ["sga", "revenue"], ["selling", "admin"]]),
    }



def _find_existing_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    # Last-resort normalized-name match for Korean columns that sometimes carry
    # spacing or punctuation differences across exports.
    normalized = {_norm_name(c): c for c in df.columns}
    for name in names:
        hit = normalized.get(_norm_name(name))
        if hit is not None:
            return hit
    return None


def materialize_phi_contract_columns(pairs: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create canonical sector-Φ current and next columns, then hard-validate them.

    This closes the Stage1→Stage2 seam.  Stage2 no longer relies on ambiguous
    alpha input passthrough: if the canonical six Φ components and their
    one-year successors cannot be resolved here, the split builder fails before
    downstream training.
    """
    out = pairs.copy()
    mapping_rows: list[dict[str, Any]] = []
    missing_current: list[str] = []
    missing_next: list[str] = []
    for canonical, aliases in KOREAN_PHI_ALIASES.items():
        cur_col = _find_existing_column(out, aliases)
        if cur_col is None:
            missing_current.append(canonical)
        else:
            out[canonical] = pd.to_numeric(out[cur_col], errors="coerce")
        next_candidates: list[str] = []
        for a in aliases:
            next_candidates.extend([f"next__{a}", f"{a}__next"])
        nxt_col = _find_existing_column(out, next_candidates)
        if nxt_col is None:
            # If merge suffix created <canonical>__next after we made canonical,
            # try the pre-existing suffix candidates one more time.
            nxt_col = _find_existing_column(out, [f"{cur_col}__next"] if cur_col else [])
        if nxt_col is None:
            missing_next.append(canonical)
        else:
            out[f"next__{canonical}"] = pd.to_numeric(out[nxt_col], errors="coerce")
        mapping_rows.append({
            "component": canonical,
            "current_source_column": cur_col,
            "next_source_column": nxt_col,
            "current_non_null": int(out[canonical].notna().sum()) if canonical in out.columns else 0,
            "next_non_null": int(out[f"next__{canonical}"].notna().sum()) if f"next__{canonical}" in out.columns else 0,
        })
    if missing_current or missing_next:
        raise ValueError({
            "message": "Stage2 input split builder cannot materialize required sector-Φ components.",
            "missing_current_components": missing_current,
            "missing_next_components": missing_next,
            "required_components": PHI_COMPONENTS,
        })
    low_signal = [r for r in mapping_rows if r["current_non_null"] == 0 or r["next_non_null"] == 0]
    if low_signal:
        raise ValueError({
            "message": "Sector-Φ component columns were resolved but contain no usable values.",
            "low_signal_components": low_signal,
        })
    return out, {
        "sector_phi_components_materialized": PHI_COMPONENTS,
        "sector_phi_component_mapping": mapping_rows,
        "sector_phi_contract_enforced_in_builder": True,
    }

def derive_actions(df_t: pd.DataFrame, df_tp1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    src = build_source_map(df_t)
    out = pd.DataFrame(index=df_t.index)
    mask = pd.DataFrame(index=df_t.index)

    # Growth-like pseudo-actions must use t→t+1 semantics.
    out["ppe_pct"], mask["action_observed__ppe_pct"] = growth_rate_next(df_t, df_tp1, src["ppe_pct"], "ppe_pct")
    out["revenue_growth"], mask["action_observed__revenue_growth"] = growth_rate_next(df_t, df_tp1, src["revenue_growth"], "revenue_growth")

    # Ratio-level changes from t to t+1.
    for dim in ["inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "cogs_ratio_chg", "sga_ratio_chg"]:
        out[dim], mask[f"action_observed__{dim}"] = ratio_level_delta(df_t, df_tp1, src[dim], dim)

    # Debt/bond dimensions: prefer pct/log changes if amount/proxy columns exist.
    for dim in ["short_debt_pct", "long_debt_pct", "bond_pct"]:
        out[dim], mask[f"action_observed__{dim}"] = log_amount_pct_change(df_t, df_tp1, src[dim], dim)

    out = out[ACTION_DIMS].rename(columns={d: f"action__{d}" for d in ACTION_DIMS})
    mask = mask[[f"action_observed__{d}" for d in ACTION_DIMS]]
    coverage = []
    for d in ACTION_DIMS:
        m = mask[f"action_observed__{d}"]
        coverage.append({
            "action_dim": d,
            "source_column": src[d],
            "observed_count": int(m.sum()),
            "observed_rate": float(m.mean()),
            "mean": float(out[f"action__{d}"].mean()),
            "std": float(out[f"action__{d}"].std()) if len(out) > 1 else None,
            "min": float(out[f"action__{d}"].min()),
            "max": float(out[f"action__{d}"].max()),
        })
    meta = {
        "action_dims": ACTION_DIMS,
        "bounds": BOUNDS,
        "source_columns": src,
        "coverage": coverage,
        "missing_dims": [d for d in ACTION_DIMS if src[d] is None],
        "zero_imputation_policy": "Missing action source dimensions are imputed as 0 and flagged by action_observed__<dim>=False.",
    }
    return pd.concat([out, mask], axis=1), meta, pd.DataFrame(coverage)


def make_transitions(df: pd.DataFrame, selected_variables: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if "selected_variables_all_complete" in df.columns:
        use = df[df["selected_variables_all_complete"].astype(bool)].copy()
    elif "alpha_all_complete" in df.columns:
        # Backward-compatible name only; Stage1 bridge now defines it from dynamic selected variables.
        use = df[df["alpha_all_complete"].astype(bool)].copy()
    else:
        use = df[selected_variables].notna().all(axis=1)
        use = df[use].copy()

    use = use[use["firm_id"].notna() & use["fiscal_year"].notna()].copy()
    use["fiscal_year"] = use["fiscal_year"].astype(int)

    left = use.copy()
    right = use.copy()
    right["fiscal_year"] = right["fiscal_year"] - 1

    merge_cols = ["firm_id", "fiscal_year"]
    pairs = left.merge(right, on=merge_cols, how="inner", suffixes=("", "__next"))

    # Reconstruct next-year actual year for audit.
    pairs["fiscal_year_next"] = pairs["fiscal_year"] + 1

    # Reward: positive if rating_num_10 improves (lower 10-grade number is better).
    if "rating_num_10" in pairs.columns and "rating_num_10__next" in pairs.columns:
        pairs["reward"] = pd.to_numeric(pairs["rating_num_10"], errors="coerce") - pd.to_numeric(pairs["rating_num_10__next"], errors="coerce")
        pairs["rating_notch_delta"] = pairs["reward"]
    else:
        raise KeyError("Stage2 reward construction requires rating_num_10 and rating_num_10__next; legacy rating_num fallback is forbidden")

    # Next-state columns for ACD/BC/IQL convenience.
    # Preserve explicit next__ columns not only for Alpha variables but for every numeric
    # current-state column with an observed one-year successor. This closes the Stage1→2→3
    # seam: Stage3 ACD may only select current features that have next-state targets.
    generated_next_cols = []
    for col in list(pairs.columns):
        if not str(col).endswith("__next"):
            continue
        base = str(col)[:-6]
        if base in pairs.columns and f"next__{base}" not in pairs.columns:
            cur = pd.to_numeric(pairs[base], errors="coerce")
            nxt = pd.to_numeric(pairs[col], errors="coerce")
            if cur.notna().sum() >= max(10, int(len(pairs) * 0.01)) and nxt.notna().sum() >= max(10, int(len(pairs) * 0.01)):
                pairs[f"next__{base}"] = pairs[col]
                generated_next_cols.append(f"next__{base}")
    for v in selected_variables:
        if v in pairs.columns and f"{v}__next" in pairs.columns:
            pairs[f"next__{v}"] = pairs[f"{v}__next"]
            if f"next__{v}" not in generated_next_cols:
                generated_next_cols.append(f"next__{v}")

    pairs, phi_contract_meta = materialize_phi_contract_columns(pairs)
    for comp in PHI_COMPONENTS:
        ncol = f"next__{comp}"
        if ncol in pairs.columns and ncol not in generated_next_cols:
            generated_next_cols.append(ncol)

    # Pass the same transition frame as both current and next source; derive_actions
    # resolves next-state values through explicit next__/<col>__next columns.
    # This avoids duplicate column labels from suffix-stripping renames.
    actions, action_meta, coverage = derive_actions(pairs, pairs)
    pairs = pd.concat([pairs.reset_index(drop=True), actions.reset_index(drop=True)], axis=1)

    diagnostics = {
        "input_complete_rows": int(len(use)),
        "transition_rows": int(len(pairs)),
        "unique_firms": int(pairs["firm_id"].nunique()) if len(pairs) else 0,
        "year_min": int(pairs["fiscal_year"].min()) if len(pairs) else None,
        "year_max": int(pairs["fiscal_year"].max()) if len(pairs) else None,
        "reward_counts": pairs["reward"].value_counts(dropna=False).sort_index().to_dict() if len(pairs) else {},
        "action_meta": action_meta,
        "generated_next_state_columns": sorted(generated_next_cols),
        "generated_next_state_column_count": int(len(generated_next_cols)),
        **phi_contract_meta,
    }
    return pairs, {"diagnostics": diagnostics, "coverage_df": coverage}



def apply_direct_raw_action_source_panel(root: Path, transitions: pd.DataFrame, diagnostics_obj: dict[str, Any], action_panel_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace Stage2 pseudo-actions with the direct raw-data Stage2A panel.

    This is the production path.  Stage2 input split must not infer 10D action
    sources from Alpha-selected R-code columns or from engineered ratio proxies.
    The action-source panel is produced by
    credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline
    directly from data/raw/raw_all Excel files.
    """
    if action_panel_path is None:
        action_panel_path = root / "data/final_freeze/stage2_candidate_projection/action_sources/stage2_raw_action_source_panel.parquet"
    if not action_panel_path.exists():
        raise FileNotFoundError(
            "Missing direct raw-data Stage2 action source panel. Run "
            "credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline first: "
            f"{action_panel_path}"
        )
    raw = pd.read_parquet(action_panel_path)
    required = ["firm_id", "fiscal_year"] + ACTION_COLUMNS + [f"action_observed__{d}" for d in ACTION_DIMS]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Direct raw action source panel is missing required columns: {missing}")

    # Keep the broad SSL/BC feature namespace aligned with the rated IQL/eval
    # namespace.  Stage3 may pretrain on broad Stage2A raw features
    # (raw__*/next__raw__*), so the rated transition phases that later consume
    # the frozen encoder must carry the exact same raw feature columns.  Do not
    # use R-code or engineered-ratio proxy fallbacks here; these are direct
    # data/raw/raw_all statement-item values precomputed by Stage2A.
    raw_feature_cols = [
        c for c in raw.columns
        if str(c).startswith("raw__")
        or str(c).startswith("next__raw__")
        or str(c) in {"has_next_raw_any", "has_next_raw_all_core"}
    ]
    raw = raw[required + raw_feature_cols].copy()
    raw["firm_id"] = raw["firm_id"].map(_norm_firm_id)
    raw["fiscal_year"] = pd.to_numeric(raw["fiscal_year"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["firm_id", "fiscal_year"]).drop_duplicates(["firm_id", "fiscal_year"], keep="first")

    out = transitions.copy()
    out["firm_id"] = out["firm_id"].map(_norm_firm_id)
    out["fiscal_year"] = _coerce_fiscal_year_series(out["fiscal_year"])
    drop_cols = [c for c in ACTION_COLUMNS + [f"action_observed__{d}" for d in ACTION_DIMS] if c in out.columns]
    out = out.drop(columns=drop_cols)
    out = out.merge(raw, on=["firm_id", "fiscal_year"], how="left", validate="many_to_one")

    coverage_rows = []
    for dim, col in zip(ACTION_DIMS, ACTION_COLUMNS):
        obs_col = f"action_observed__{dim}"
        if col not in out.columns or obs_col not in out.columns:
            raise ValueError(f"Action source merge failed for {dim}: {col}/{obs_col} not found after merge")
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0).clip(*BOUNDS[dim])
        out[obs_col] = out[obs_col].fillna(False).astype(bool)
        obs = out[obs_col]
        coverage_rows.append({
            "action_dim": dim,
            "source_column": str(action_panel_path.relative_to(root)),
            "source_method": "direct_data_raw_all_xlsx_precompute",
            "observed_count": int(obs.sum()),
            "observed_rate": float(obs.mean()) if len(out) else 0.0,
            "nonzero_rate": float((out[col].abs() > 1e-12).mean()) if len(out) else 0.0,
            "mean": float(out[col].mean()) if len(out) else None,
            "std": float(out[col].std()) if len(out) > 1 else None,
            "min": float(out[col].min()) if len(out) else None,
            "max": float(out[col].max()) if len(out) else None,
        })
    # Rated phases must include the same raw feature schema used by broad SSL/BC.
    raw_feature_missing_after_merge = [c for c in raw_feature_cols if c not in out.columns]
    if raw_feature_missing_after_merge:
        raise ValueError({
            "message": "Rated IQL/eval transition panel lost Stage2A raw feature columns during merge; this would break Stage3→Stage5/6 schema compatibility.",
            "missing_raw_feature_columns": raw_feature_missing_after_merge[:50],
        })

    coverage = pd.DataFrame(coverage_rows)
    failed = coverage[coverage["observed_count"] <= 0][["action_dim", "observed_count", "observed_rate"]].to_dict(orient="records")
    if failed:
        raise ValueError({
            "message": "Direct raw-data action source has dead dimensions after merge; proxy fallback is forbidden.",
            "failed_dimensions": failed,
            "action_panel": str(action_panel_path),
        })
    diagnostics_obj["coverage_df"] = coverage
    diagnostics_obj.setdefault("diagnostics", {})["action_meta"] = {
        "action_dims": ACTION_DIMS,
        "bounds": BOUNDS,
        "source_policy": "direct_data_raw_all_xlsx_precompute_only",
        "raw_data_input": "data/raw/raw_all/*.xlsx",
        "stage0_statement_items_used": False,
        "engineered_ratio_proxy_used": False,
        "proxy_fallback_allowed": False,
        "action_panel": str(action_panel_path.relative_to(root)),
        "coverage": coverage.to_dict(orient="records"),
        "raw_feature_columns_passthrough_to_rated_phases": sorted(raw_feature_cols),
        "schema_alignment_policy": "broad_ssl_bc_and_rated_iql_eval_share_the_same_Stage2A_raw_feature_namespace",
        "zero_imputation_policy": "Only row-level unobserved direct raw actions are set to 0 with action_observed__<dim>=False; dimension-level source gaps fail fast.",
    }
    diagnostics_obj["diagnostics"]["direct_raw_action_source_panel"] = str(action_panel_path.relative_to(root))
    return out, diagnostics_obj

def split_transitions(transitions: pd.DataFrame, eval_start_year: int, bc_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Deprecated legacy splitter.

    The active final-freeze path is ``split_broad_and_rated`` followed by
    ``make_phase_eval_state_only``. This legacy helper used ``>= eval_start_year``
    for eval rows and must never be reintroduced into the active run path.
    """
    raise RuntimeError(
        "split_transitions() is deprecated and forbidden in the final-freeze path; "
        "use split_broad_and_rated() and make_phase_eval_state_only() instead."
    )



def _read_raw_action_panel_for_stage2(root: Path, action_panel_path: Path | None = None) -> pd.DataFrame:
    """Load Stage2A direct-raw action source panel.

    This panel is intentionally broad: it may include unrated firms/years and is
    the source for SSL pretraining and BC behavior cloning. Rated bridge rows are
    used only for IQL/eval reward-bearing phases.
    """
    if action_panel_path is None:
        action_panel_path = root / "data/final_freeze/stage2_candidate_projection/action_sources/stage2_raw_action_source_panel.parquet"
    if not action_panel_path.exists():
        raise FileNotFoundError(
            "Missing Stage2A raw action source panel. Run "
            "credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline first: "
            f"{action_panel_path}"
        )
    panel = pd.read_parquet(action_panel_path)
    required = ["firm_id", "fiscal_year"] + ACTION_COLUMNS
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"Stage2A raw action source panel missing columns: {missing}")
    out = panel.copy()
    out["firm_id"] = out["firm_id"].map(_norm_firm_id)
    out["fiscal_year"] = _coerce_fiscal_year_series(out["fiscal_year"])
    out = out[out["firm_id"].notna() & out["fiscal_year"].notna()].copy()
    out["year"] = out["fiscal_year"].astype(int)
    for col in ACTION_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    for dim in ACTION_DIMS:
        obs_col = f"action_observed__{dim}"
        if obs_col not in out.columns:
            out[obs_col] = pd.to_numeric(out[f"action__{dim}"], errors="coerce").fillna(0.0).abs() > 1e-12
        else:
            out[obs_col] = out[obs_col].fillna(False).astype(bool)
    # Ensure broad raw features have next-state companions for Stage3 ACD SSL.
    raw_features = [c for c in out.columns if c.startswith("raw__")]
    missing_next = [c for c in raw_features if f"next__{c}" not in out.columns]
    if missing_next:
        raise ValueError(
            "Stage2A raw action source panel is not broad-SSL ready; missing next-state raw columns: "
            f"{missing_next[:20]}"
        )
    if "has_next_raw_any" in out.columns:
        out = out[out["has_next_raw_any"].fillna(False).astype(bool)].copy()
    if out.duplicated(["firm_id", "fiscal_year"]).any():
        dup = int(out.duplicated(["firm_id", "fiscal_year"]).sum())
        raise ValueError(f"Stage2A raw action source panel has duplicate firm-year rows: {dup}")
    return out


def _raw_action_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    n = len(panel)
    rows = []
    for dim in ACTION_DIMS:
        col = f"action__{dim}"
        obs_col = f"action_observed__{dim}"
        obs = panel[obs_col].astype(bool) if obs_col in panel.columns else pd.to_numeric(panel[col], errors="coerce").fillna(0.0).abs() > 1e-12
        rows.append({
            "action_dim": dim,
            "source_column": col,
            "source_method": "direct_data_raw_all_xlsx_stage2a_broad_panel",
            "observed_count": int(obs.sum()),
            "observed_rate": float(obs.mean()) if n else 0.0,
            "nonzero_rate": float((pd.to_numeric(panel[col], errors="coerce").fillna(0.0).abs() > 1e-12).mean()) if n else 0.0,
        })
    cov = pd.DataFrame(rows)
    bad = cov[cov["observed_rate"] < 0.01]
    if len(bad):
        raise ValueError({
            "message": "Stage2A raw action source coverage failed; proxy fallback is forbidden.",
            "bad_dimensions": bad[["action_dim", "observed_rate", "observed_count"]].to_dict(orient="records"),
        })
    return cov



FORBIDDEN_EVAL_PREFIXES = ("action__", "action_observed__", "next__", "soft_cand_")
FORBIDDEN_EVAL_EXACT = {
    "candidate_id", "projection_distance", "out_of_library_flag", "near_tie_flag",
    "reward_raw_notch", "reward_raw", "phi_tplusH", "delta_phi", "delta_phi_clipped",
    "lambda_phi", "reward_aux_phi", "reward_total_raw", "reward_mean_train", "reward_std_train",
    "reward_train", "reward_original", "reward", "done",
}

def make_phase_eval_state_only(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return the final evaluation panel with outcome/action/reward columns removed.

    The binding Stage2 contract treats phase_eval as a state-only serving panel.
    Actual observed actions, next-state columns, projected labels, and reward
    fields are forbidden to prevent eval-time leakage into Stage6 selection.
    """
    forbidden = [c for c in df.columns if str(c).startswith(FORBIDDEN_EVAL_PREFIXES) or str(c) in FORBIDDEN_EVAL_EXACT]
    out = df.drop(columns=forbidden, errors="ignore").copy()
    remaining = [c for c in out.columns if str(c).startswith(FORBIDDEN_EVAL_PREFIXES) or str(c) in FORBIDDEN_EVAL_EXACT]
    if remaining:
        raise ValueError(f"phase_eval state-only contract failed; forbidden columns remain: {remaining[:20]}")
    return out, {"phase_eval_state_only": True, "dropped_forbidden_columns": sorted(forbidden), "forbidden_column_count": len(forbidden)}


def _complete_state_rows_for_selected_variables(df: pd.DataFrame, selected_variables: list[str]) -> pd.DataFrame:
    """Return current-state rows that satisfy the Stage1 selected-variable completeness contract.

    Unlike ``make_transitions()``, this does not require a t+1 rating. It is used
    only for the Stage6 evaluation base year where the simulator rolls
    s_{eval_base_year} forward to the rollout target year.
    """
    if "selected_variables_all_complete" in df.columns:
        use = df[df["selected_variables_all_complete"].astype(bool)].copy()
    elif "alpha_all_complete" in df.columns:
        use = df[df["alpha_all_complete"].astype(bool)].copy()
    else:
        missing = [c for c in selected_variables if c not in df.columns]
        if missing:
            raise KeyError(f"Stage2 eval-state construction missing selected variable columns: {missing[:20]}")
        use = df[df[selected_variables].notna().all(axis=1)].copy()
    use = use[use["firm_id"].notna() & use["fiscal_year"].notna()].copy()
    use["firm_id"] = use["firm_id"].map(_norm_firm_id)
    use["fiscal_year"] = pd.to_numeric(use["fiscal_year"], errors="coerce").astype("Int64")
    use = use.dropna(subset=["firm_id", "fiscal_year"]).copy()
    use["year"] = use["fiscal_year"].astype(int)
    return use


def make_eval_state_panel(
    df: pd.DataFrame,
    selected_variables: list[str],
    eval_start_year: int,
    root: Path,
    raw_action_panel_path: Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the state-only Stage6 evaluation base panel from current-year states.

    Phase-eval is intentionally a current-state panel, not a reward-bearing
    transition.  Requiring ``rated_transitions`` to contain eval_base_year would
    require an observed rating at eval_base_year+1, which contradicts the final
    freeze design where Stage6 simulates from s_eval_base_year to
    rollout_target_year.
    """
    states_all = _complete_state_rows_for_selected_variables(df, selected_variables)
    years_all = sorted(pd.to_numeric(states_all.get("fiscal_year"), errors="coerce").dropna().astype(int).unique().tolist())
    states = states_all[pd.to_numeric(states_all["fiscal_year"], errors="coerce") == int(eval_start_year)].copy()
    meta: dict[str, Any] = {
        "schema_version": "stage2_phase_eval_current_state_v1",
        "eval_base_year": int(eval_start_year),
        "source_policy": "rated_current_firm_states_at_eval_base_year_no_tplus1_rating_required",
        "stage1_complete_state_rows": int(len(states_all)),
        "stage1_complete_state_years": years_all,
        "phase_eval_current_state_rows_before_raw_merge": int(len(states)),
        "requires_tplus1_external_rating": False,
    }
    if states.empty:
        meta["status"] = "EMPTY"
        return states, meta

    raw_p = Path(raw_action_panel_path).resolve() if raw_action_panel_path else root / "data/final_freeze/stage2_candidate_projection/action_sources/stage2_raw_action_source_panel.parquet"
    if not raw_p.exists():
        raise FileNotFoundError(
            "Missing direct raw-data Stage2 action source panel for phase_eval state construction: "
            f"{raw_p}. Run final_stage2_raw_action_source_precompute first."
        )
    raw = pd.read_parquet(raw_p)
    required_keys = ["firm_id", "fiscal_year"]
    missing_keys = [c for c in required_keys if c not in raw.columns]
    if missing_keys:
        raise ValueError(f"Stage2A raw action source panel missing eval-state merge keys: {missing_keys}")
    raw = raw.copy()
    raw["firm_id"] = raw["firm_id"].map(_norm_firm_id)
    raw["fiscal_year"] = pd.to_numeric(raw["fiscal_year"], errors="coerce").astype("Int64")
    raw = raw.dropna(subset=["firm_id", "fiscal_year"]).drop_duplicates(["firm_id", "fiscal_year"], keep="first")
    raw_state_cols = [
        c for c in raw.columns
        if str(c).startswith("raw__")
        or str(c) in {"has_next_raw_any", "has_next_raw_all_core", "has_next_raw_avs_any"}
    ]
    if not raw_state_cols:
        raise ValueError("Stage2A raw action source panel has no raw__ state columns for phase_eval simulator input")
    raw_eval_years = sorted(pd.to_numeric(raw.get("fiscal_year"), errors="coerce").dropna().astype(int).unique().tolist())
    states = states.drop(columns=[c for c in raw_state_cols if c in states.columns], errors="ignore")
    merged = states.merge(raw[required_keys + raw_state_cols], on=required_keys, how="left", validate="many_to_one")
    raw_nonnull = merged[raw_state_cols].notna().any(axis=1)
    raw_match_rate = float(raw_nonnull.mean()) if len(merged) else 0.0
    if raw_match_rate <= 0.0:
        raise ValueError({
            "message": "Stage2 phase_eval current states could not be linked to Stage2A raw financial state columns",
            "eval_base_year": int(eval_start_year),
            "stage1_eval_state_rows": int(len(states)),
            "raw_panel_years": raw_eval_years,
            "raw_panel": str(raw_p),
        })
    meta.update({
        "status": "PASS",
        "raw_panel": str(raw_p.relative_to(root)) if raw_p.is_relative_to(root) else str(raw_p),
        "raw_panel_years": raw_eval_years,
        "raw_state_columns_merged": sorted(raw_state_cols),
        "raw_state_column_count": int(len(raw_state_cols)),
        "raw_state_match_rows": int(raw_nonnull.sum()),
        "raw_state_match_rate": raw_match_rate,
        "phase_eval_current_state_rows_after_raw_merge": int(len(merged)),
    })
    return merged, meta

def _stable_unit_hash(value: str, salt: str = "stage2_bc_v1") -> float:
    """Return a deterministic [0, 1) hash score independent of RNG state.

    This is used only for explicit ``--bc-split-policy hash``.  The final
    default is ``all`` so Stage2 does not depend on any model-training seed.
    """
    h = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(h[:16], 16) / float(16 ** 16)


def _build_phase2_bc(
    broad_train: pd.DataFrame,
    bc_frac: float,
    bc_split_policy: str,
    stage2_split_seed: int | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the BC universe without coupling Stage2 to model seeds.

    Policies:
      - all: use the full broad training transition universe. This is the
        final-freeze default and is intentionally seed-free.
      - hash: deterministic firm-level subset using a stable SHA-256 hash.
      - legacy_seeded: reproduce the historical RNG-based firm subset, but
        only when explicitly requested and with its own Stage2 split seed.

    The function never reads Stage3/Stage4/Stage5 seeds.  Action labeling and
    projection remain deterministic for a fixed input row universe.
    """
    policy = str(bc_split_policy or "all").strip().lower()
    if policy not in {"all", "hash", "legacy_seeded"}:
        raise ValueError(f"Unknown bc_split_policy={bc_split_policy!r}; expected all/hash/legacy_seeded")
    if broad_train.empty:
        return broad_train.copy(), {
            "bc_split_policy": policy,
            "bc_frac_requested": float(bc_frac),
            "stage2_split_seed_used": None,
            "selection_unit": "firm_id",
            "selected_firms": 0,
            "available_firms": 0,
            "seed_free": policy != "legacy_seeded",
        }
    if not (0.0 < float(bc_frac) <= 1.0):
        raise ValueError(f"bc_frac must be in (0, 1], got {bc_frac!r}")

    firm_series = broad_train["firm_id"].dropna().astype(str)
    firms = np.array(sorted(firm_series.unique()))
    if len(firms) == 0:
        raise ValueError("Cannot build phase2_bc: broad_train has rows but no valid firm_id values")

    if policy == "all":
        selected = set(firms.tolist())
        seed_used = None
        deterministic_detail = "all_broad_train_firms"
    elif policy == "hash":
        selected = {f for f in firms if _stable_unit_hash(f) < float(bc_frac)}
        if not selected:
            raise ValueError({
                "message": "Deterministic hash BC split selected zero firms; increase bc_frac or use all.",
                "bc_frac": float(bc_frac),
                "available_firms": int(len(firms)),
            })
        seed_used = None
        deterministic_detail = "sha256_firm_id_threshold"
    else:
        if stage2_split_seed is None:
            raise ValueError("legacy_seeded BC split requires --stage2-split-seed; do not use model seeds implicitly")
        rng = np.random.default_rng(int(stage2_split_seed))
        shuffled = firms.copy()
        rng.shuffle(shuffled)
        n_bc = max(1, int(round(len(shuffled) * float(bc_frac))))
        selected = set(shuffled[:n_bc].tolist())
        seed_used = int(stage2_split_seed)
        deterministic_detail = "legacy_numpy_default_rng_shuffle"

    phase2 = broad_train[broad_train["firm_id"].astype(str).isin(selected)].copy()
    if phase2.empty:
        raise ValueError({
            "message": "phase2_bc construction produced zero rows",
            "bc_split_policy": policy,
            "bc_frac": float(bc_frac),
            "available_firms": int(len(firms)),
            "selected_firms": int(len(selected)),
        })

    meta = {
        "bc_split_policy": policy,
        "bc_frac_requested": float(bc_frac),
        "stage2_split_seed_used": seed_used,
        "selection_unit": "firm_id",
        "available_firms": int(len(firms)),
        "selected_firms": int(len(selected)),
        "selected_firm_fraction": float(len(selected) / max(len(firms), 1)),
        "seed_free": policy != "legacy_seeded",
        "deterministic_detail": deterministic_detail,
        "model_training_seed_coupling_forbidden": True,
    }
    return phase2, meta


def split_broad_and_rated(
    broad: pd.DataFrame,
    rated: pd.DataFrame,
    eval_start_year: int,
    bc_frac: float,
    bc_split_policy: str = "all",
    stage2_split_seed: int | None = None,
    eval_states: pd.DataFrame | None = None,
    eval_state_meta: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create broad SSL/BC + rated IQL/eval split.

    phase1_pretrain: broad direct-raw transition universe, no external rating required.
    phase2_bc: broad direct-raw behavior cloning universe, no external rating required.
      The final default is the full broad train universe (seed-free).
      Historical random subsetting is preserved only behind explicit
      ``bc_split_policy='legacy_seeded'`` and a dedicated Stage2 split seed.
    phase3_iql: rated reward-bearing train transition universe.
    phase_eval: current firm-state serving universe at eval_base_year. It is
      intentionally not a reward-bearing transition and must not require an
      observed t+1 external rating.
    """
    broad_train = broad[pd.to_numeric(broad["fiscal_year"], errors="coerce") < eval_start_year].copy()
    rated_train = rated[pd.to_numeric(rated["fiscal_year"], errors="coerce") < eval_start_year].copy()
    rated_transition_eval = rated[pd.to_numeric(rated["fiscal_year"], errors="coerce") == eval_start_year].copy()
    if eval_states is not None and len(eval_states):
        phase_eval = eval_states.copy()
        phase_eval_source = "rated_current_state_panel"
    elif len(rated_transition_eval):
        # Rare compatibility path for datasets that actually contain t+1 ratings
        # beyond eval_base_year.  The normal final-freeze path should use
        # eval_states from current-year Stage1 rows.
        phase_eval = rated_transition_eval.copy()
        phase_eval_source = "rated_transition_panel_with_tplus1_rating_available"
    else:
        phase_eval = pd.DataFrame()
        phase_eval_source = "missing"
    if rated_train.empty:
        raise ValueError(f"Stage2 split contract failed: no rated training rows with fiscal_year < eval_base_year={eval_start_year}")
    if phase_eval.empty:
        rated_years = sorted(pd.to_numeric(rated.get("fiscal_year"), errors="coerce").dropna().astype(int).unique().tolist()) if "fiscal_year" in rated.columns else []
        broad_years = sorted(pd.to_numeric(broad.get("fiscal_year"), errors="coerce").dropna().astype(int).unique().tolist()) if "fiscal_year" in broad.columns else []
        eval_meta_status = (eval_state_meta or {}).get("status")
        eval_complete_years = (eval_state_meta or {}).get("stage1_complete_state_years")
        raise ValueError({
            "message": "Stage2 split contract failed: phase_eval current-state panel is empty",
            "eval_base_year": int(eval_start_year),
            "expected_policy": "phase_eval must be current firm-states at eval_base_year; it must not require eval_base_year+1 external ratings",
            "rated_transition_years": rated_years,
            "broad_transition_years_after_next_filter": broad_years,
            "eval_state_meta_status": eval_meta_status,
            "stage1_complete_state_years": eval_complete_years,
            "hint": "Ensure Stage1 bridge/input includes current firm-state rows for temporal_split.eval_base_year, or adjust temporal_split only as an explicit experimental contract change.",
        })
    if broad_train.empty:
        # If raw data starts late, do not silently fall back to rated bridge; use all broad rows before max eval fallback.
        max_y = int(pd.to_numeric(broad["fiscal_year"], errors="coerce").max())
        broad_train = broad[pd.to_numeric(broad["fiscal_year"], errors="coerce") < max_y].copy()

    phase2, phase2_split_meta = _build_phase2_bc(
        broad_train=broad_train,
        bc_frac=bc_frac,
        bc_split_policy=bc_split_policy,
        stage2_split_seed=stage2_split_seed,
    )

    phase1 = broad_train.copy()
    phase3 = rated_train.copy()

    meta = {
        "schema_version": "stage2_split_broad_ssl_bc_rated_iql_v2_seedless_phase2_default",
        "eval_start_year": int(eval_start_year),
        "bc_frac": float(bc_frac),
        "bc_split_policy": str(bc_split_policy),
        "stage2_split_seed": None if stage2_split_seed is None else int(stage2_split_seed),
        "stage2_seedless_default": str(bc_split_policy).strip().lower() == "all",
        "phase2_bc_split": phase2_split_meta,
        "universe_policy": {
            "phase1_pretrain": "broad_raw_financial_transition_universe_no_external_rating_required",
            "phase2_bc": "broad_raw_action_labeled_transition_universe_no_external_rating_required",
            "phase3_iql": "rated_reward_bearing_train_transition_universe",
            "phase_eval": "current_state_serving_universe_at_eval_base_year_no_tplus1_rating_required",
        },
        "phase_eval_source": phase_eval_source,
        "phase_eval_state_meta": eval_state_meta or {},
        "phase1_rows": int(len(phase1)),
        "phase2_rows": int(len(phase2)),
        "phase3_rows": int(len(phase3)),
        "phase_eval_rows": int(len(phase_eval)),
        "phase1_firms": int(phase1["firm_id"].nunique()) if len(phase1) else 0,
        "phase2_firms": int(phase2["firm_id"].nunique()) if len(phase2) else 0,
        "phase3_firms": int(phase3["firm_id"].nunique()) if len(phase3) else 0,
        "phase_eval_firms": int(phase_eval["firm_id"].nunique()) if len(phase_eval) else 0,
        "broad_to_rated_row_ratio_phase1_vs_phase3": float(len(phase1) / max(len(phase3), 1)),
        "broad_to_rated_row_ratio_phase2_vs_phase3": float(len(phase2) / max(len(phase3), 1)),
    }
    return phase1, phase2, phase3, phase_eval, meta


FAVORABLE_DIRECTION = {
    "action__ppe_pct": "negative",
    "action__inv_turnover_chg": "positive",
    "action__ar_turnover_chg": "positive",
    "action__ap_turnover_chg": "negative",
    "action__short_debt_pct": "negative",
    "action__long_debt_pct": "negative",
    "action__bond_pct": "negative",
    "action__revenue_growth": "positive",
    "action__cogs_ratio_chg": "negative",
    "action__sga_ratio_chg": "negative",
}

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _winsor_abs_favorable(series: pd.Series, direction: str) -> np.ndarray:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if direction == "negative":
        x = x[x < 0]
    elif direction == "positive":
        x = x[x > 0]
    else:
        x = x[np.abs(x) > 0]
    x = np.abs(x)
    if len(x) == 0:
        return x
    lo, hi = np.quantile(x, [0.01, 0.99])
    return np.clip(x, lo, hi)

def _candidate_percentile(mag: float, dist: np.ndarray) -> float | None:
    if dist is None or len(dist) == 0 or not np.isfinite(mag):
        return None
    return float(np.mean(dist <= abs(float(mag))))

def _magnitude_verdict(percentile: float | None, mag: float) -> tuple[str, bool, str]:
    if abs(float(mag)) < 1e-12:
        return "zero_dimension", False, "zero action dimension; not a magnitude-bearing lever"
    if percentile is None:
        return "no_empirical_support_or_zero", True, "explicit v32 design acceptance required: no favorable historical support"
    if percentile <= 0.25:
        return "too_weak", True, "explicit v32 design acceptance required: candidate is below P25 historical favorable move"
    if percentile <= 0.75:
        return "typical_or_moderate", False, "within P25-P75 historical favorable move"
    if percentile <= 0.95:
        return "aggressive_but_plausible", False, "within P75-P95 historical favorable move"
    return "too_aggressive", True, "explicit v32 design acceptance required: candidate exceeds P95 historical favorable move"

def write_action_magnitude_audit(root: Path, out_dir: Path, train_df: pd.DataFrame) -> dict[str, Any]:
    """Stage2.2 historical pseudo-action magnitude audit.

    This is audit-only by design. It does not silently recalibrate candidate YAML.
    If a nonzero main candidate dimension falls outside the historical favorable
    distribution, the row is flagged and accepted only with an explicit reason in
    the audit ledger.
    """
    import yaml
    cand_path = root / "data/final_freeze/configs/final_candidate_library.yaml"
    text = cand_path.read_text(encoding="utf-8")
    cand = yaml.safe_load(text) or {}
    fixed = cand.get("fixed_candidates", {}) or {}
    quant = {
        "schema_version": "action_magnitude_quantiles_v32",
        "created_utc": now_utc(),
        "source": "train_period_historical_pseudo_actions_phase2_plus_phase3",
        "eval_distribution_used": False,
        "winsorization": {"lower": 0.01, "upper": 0.99},
        "recalibration_policy": "audit_only_with_explicit_acceptance_reason",
        "automatic_recalibration_performed": False,
        "candidate_library_path": str(cand_path.relative_to(root)),
        "candidate_library_sha256": _sha256_text(text),
        "dimensions": {},
    }
    qrows=[]; audit_rows=[]; dist_cache={}
    for col in ACTION_COLUMNS:
        direction = FAVORABLE_DIRECTION.get(col, "any")
        dist = _winsor_abs_favorable(train_df[col] if col in train_df.columns else pd.Series(dtype=float), direction)
        dist_cache[col] = dist
        qs = {k: None for k in ["p25","p50","p65","p75","p85","p90","p95"]}
        if len(dist):
            vals = np.quantile(dist, [0.25,0.50,0.65,0.75,0.85,0.90,0.95])
            qs = dict(zip(qs.keys(), [float(v) for v in vals]))
        rec = {"action_dim": col, "improvement_direction": direction, "n": int(len(dist)), **qs}
        quant["dimensions"][col] = rec
        qrows.append(rec)
    for cid, raw in fixed.items():
        for col in ACTION_COLUMNS:
            val = float(raw.get(col, 0.0) or 0.0)
            if abs(val) < 1e-12:
                continue
            dist = dist_cache.get(col, np.array([]))
            pct = _candidate_percentile(abs(val), dist)
            verdict, req, reason = _magnitude_verdict(pct, val)
            accepted_reason = ""
            if req:
                accepted_reason = "audit_only_v32_pre_registered_controllability_ladder_candidate; report in appendix and rerun if committee requires recalibration"
            audit_rows.append({
                "candidate_id": cid,
                "action_dim": col,
                "candidate_value": val,
                "abs_candidate_magnitude": abs(val),
                "improvement_direction": FAVORABLE_DIRECTION.get(col, "any"),
                "train_n": int(len(dist)),
                "p25": quant["dimensions"][col]["p25"],
                "p50": quant["dimensions"][col]["p50"],
                "p65": quant["dimensions"][col]["p65"],
                "p75": quant["dimensions"][col]["p75"],
                "p85": quant["dimensions"][col]["p85"],
                "p90": quant["dimensions"][col]["p90"],
                "p95": quant["dimensions"][col]["p95"],
                "candidate_percentile": pct,
                "verdict": verdict,
                "recalibration_required": bool(req),
                "accepted_reason": accepted_reason,
                "recalibrated_value": np.nan,
                "final_candidate_value": val,
            })
    qdf = pd.DataFrame(qrows)
    adf = pd.DataFrame(audit_rows)
    qdf.to_csv(out_dir / "action_magnitude_quantiles.csv", index=False, encoding="utf-8-sig")
    adf.to_csv(out_dir / "candidate_magnitude_audit.csv", index=False, encoding="utf-8-sig")
    write_json(out_dir / "action_magnitude_quantiles.json", quant)
    requiring = int(adf["recalibration_required"].sum()) if not adf.empty else 0
    unaccepted = int(((adf.get("recalibration_required", False) == True) & (adf.get("accepted_reason", "").astype(str).str.len() == 0)).sum()) if not adf.empty else 0
    return {
        "action_magnitude_audit_required": True,
        "action_magnitude_quantiles_json": str((out_dir / "action_magnitude_quantiles.json").relative_to(root)),
        "candidate_magnitude_audit_csv": str((out_dir / "candidate_magnitude_audit.csv").relative_to(root)),
        "candidate_library_sha256": quant["candidate_library_sha256"],
        "automatic_recalibration_performed": False,
        "recalibration_policy": "audit_only_with_explicit_acceptance_reason",
        "main_candidate_rows_requiring_review": requiring,
        "unaccepted_recalibration_required_rows": unaccepted,
        "final_result_allowed": unaccepted == 0,
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--input", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--eval-start-year", type=int, default=None, help="Deprecated override; default comes from temporal_split.yaml eval_base_year")
    ap.add_argument("--bc-frac", type=float, default=1.0, help="BC firm fraction for hash/legacy_seeded policies; ignored by the seed-free all policy")
    ap.add_argument("--bc-split-policy", choices=["all", "hash", "legacy_seeded"], default="all", help="How to build phase2_bc. Default all is seed-free and uses the full broad training universe.")
    ap.add_argument("--stage2-split-seed", type=int, default=None, help="Dedicated Stage2 data split seed. Required only for --bc-split-policy legacy_seeded.")
    ap.add_argument("--seed", type=int, default=None, help="Deprecated legacy alias for --stage2-split-seed. Ignored unless --bc-split-policy legacy_seeded and --stage2-split-seed is omitted.")
    ap.add_argument("--raw-action-source-panel", default=None, help="Direct raw-data Stage2A action source panel; default uses final_freeze/stage2_candidate_projection/action_sources")
    ap.add_argument("--join-cash-flow-substrate", action="store_true", help="Default off. Merge cleaned CF statement OCF/capex levels into Stage2 substrate.")
    ap.add_argument("--cash-flow-encoder-mode", choices=["reward_only", "full"], default="reward_only", help="Metadata/audit mode for CF join; full means encoder feature distribution changes.")
    ap.add_argument("--cash-flow-panel", default=None, help="Optional explicit 현금흐름표_clean.parquet path for --join-cash-flow-substrate")
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    temporal_contract = load_temporal_contract(root)
    eval_start_year = int(args.eval_start_year) if args.eval_start_year is not None else int(temporal_contract.eval_base_year)
    action_contract_meta = activate_final_action_contract(root)
    inp = Path(args.input).resolve() if args.input else root / "data/final_freeze/stage1_oracle_inputs/alpha_vanilla_input_candidate.parquet"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "data/final_freeze/stage2_candidate_projection/input_splits"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not inp.exists():
        raise FileNotFoundError(inp)

    df = pd.read_parquet(inp)
    cash_flow_join_meta = {"cash_flow_substrate_joined": False, "cash_flow_encoder_mode": args.cash_flow_encoder_mode}
    if args.join_cash_flow_substrate:
        df, cash_flow_join_meta = join_cash_flow_substrate(root, df, cash_flow_panel=args.cash_flow_panel, encoder_mode=args.cash_flow_encoder_mode)
    selected_variables, selected_meta = load_stage1_bridge_selected_variables(root, inp, df)

    # Rated/reward-bearing universe from Stage1→Stage2 bridge.
    rated_transitions, diag_obj = make_transitions(df, selected_variables)
    raw_action_panel = Path(args.raw_action_source_panel).resolve() if args.raw_action_source_panel else None
    rated_transitions, diag_obj = apply_direct_raw_action_source_panel(root, rated_transitions, diag_obj, raw_action_panel)

    # Broad raw financial transition universe from Stage2A.  This is the source
    # for SSL encoder pretraining and BC imitation, and does not require external
    # credit ratings.  No R-code/ratio proxy fallback is allowed.
    broad_transitions = _read_raw_action_panel_for_stage2(root, raw_action_panel)
    coverage = _raw_action_coverage(broad_transitions)

    # AVS256_BLOCK_AWARE_ACD_V2 restoration: all split phases must carry the exact
    # 129 continuous state features, market/industry_class categoricals, and
    # next__ columns for the 118 ACD targets before Stage3 is allowed to run.
    broad_transitions, broad_avs_meta = enrich_avs256_panel(broad_transitions)
    rated_transitions, rated_avs_meta = enrich_avs256_panel(rated_transitions, reference_for_peer=broad_transitions)
    transition_meta = {
        "schema_version": "transition_proximity_metadata_v1",
        "status": "PASS" if broad_avs_meta.get("transition_proximity_status") == "computed" and rated_avs_meta.get("transition_proximity_status") == "computed" else "FAIL",
        "broad": broad_avs_meta.get("transition_proximity_metadata", {}),
        "rated": rated_avs_meta.get("transition_proximity_metadata", {}),
    }
    if transition_meta["status"] != "PASS":
        raise ValueError(f"Stage2 transition proximity must be computed for final runs: {transition_meta}")
    _write_json_safe(out_dir / "transition_proximity_metadata.json", transition_meta)
    pd.DataFrame([
        {"universe": "broad", **transition_meta["broad"]},
        {"universe": "rated", **transition_meta["rated"]},
    ]).to_parquet(out_dir / "transition_proximity_prototypes.parquet", index=False)

    diag_obj["diagnostics"]["selected_variable_contract"] = selected_meta
    diag_obj["diagnostics"]["cash_flow_substrate_join"] = cash_flow_join_meta
    diag_obj["diagnostics"]["split_universe_policy"] = "broad_ssl_bc_rated_iql"
    diag_obj["diagnostics"]["broad_raw_action_source_rows"] = int(len(broad_transitions))
    diag_obj["diagnostics"]["rated_reward_transition_rows"] = int(len(rated_transitions))

    phase_eval_states, phase_eval_state_source_meta = make_eval_state_panel(
        df, selected_variables, eval_start_year, root, raw_action_panel
    )
    diag_obj["diagnostics"]["phase_eval_current_state_source"] = phase_eval_state_source_meta

    legacy_seed = args.stage2_split_seed
    if legacy_seed is None and args.bc_split_policy == "legacy_seeded" and args.seed is not None:
        legacy_seed = int(args.seed)
    if args.bc_split_policy != "legacy_seeded" and args.seed is not None:
        # Keep --seed as a non-breaking CLI alias, but make the seed-free final
        # contract explicit in metadata rather than letting model seeds alter data.
        diag_obj.setdefault("diagnostics", {}).setdefault("warnings", []).append(
            "Deprecated --seed was provided but ignored because bc_split_policy is seed-free; "
            "use --bc-split-policy legacy_seeded --stage2-split-seed for historical replay."
        )

    phase1, phase2, phase3, phase_eval, split_meta = split_broad_and_rated(
        broad_transitions, rated_transitions, eval_start_year, args.bc_frac,
        bc_split_policy=args.bc_split_policy, stage2_split_seed=legacy_seed,
        eval_states=phase_eval_states, eval_state_meta=phase_eval_state_source_meta,
    )
    split_meta["deprecated_seed_arg_provided"] = args.seed is not None
    split_meta["deprecated_seed_arg_ignored"] = bool(args.seed is not None and args.bc_split_policy != "legacy_seeded")
    # RL-S2-009: phase_eval is state-only but still a serving input to the
    # frozen Stage3 encoder and Stage6 simulator.  Materialize the exact AVS256
    # state/categorical feature contract from current-state raw accounts, then
    # strip future/action/reward columns again so eval remains leakage-free.
    phase_eval, phase_eval_avs_meta = enrich_avs256_panel(
        phase_eval, reference_for_peer=broad_transitions, require_next_phi_critical=False
    )
    split_meta["phase_eval_avs256_enrichment"] = phase_eval_avs_meta
    split_meta["phase_eval_stage6_serving_features_materialized"] = True
    phase_eval, phase_eval_state_meta = make_phase_eval_state_only(phase_eval)
    split_meta.update(phase_eval_state_meta)

    cf_reward_only_encoder_meta = {"enabled": False, "policy": "not_applied"}
    if args.join_cash_flow_substrate and args.cash_flow_encoder_mode == "reward_only":
        phase1, m1 = _apply_cash_flow_reward_only_encoder_guard(phase1, enabled=True)
        phase2, m2 = _apply_cash_flow_reward_only_encoder_guard(phase2, enabled=True)
        phase3, m3 = _apply_cash_flow_reward_only_encoder_guard(phase3, enabled=True)
        phase_eval, m4 = _apply_cash_flow_reward_only_encoder_guard(phase_eval, enabled=True)
        cf_reward_only_encoder_meta = {"enabled": True, "phase1": m1, "phase2": m2, "phase3": m3, "phase_eval": m4}
        cash_flow_join_meta["reward_only_encoder_guard"] = cf_reward_only_encoder_meta

    paths = {
        "phase1_pretrain": out_dir / "phase1_pretrain.parquet",
        "phase2_bc": out_dir / "phase2_bc.parquet",
        "phase3_iql": out_dir / "phase3_iql.parquet",
        "phase_eval": out_dir / "phase_eval.parquet",
    }
    write_feature_manifest(out_dir.parent / "feature_manifest.json")
    phase1.to_parquet(paths["phase1_pretrain"], index=False)
    phase2.to_parquet(paths["phase2_bc"], index=False)
    phase3.to_parquet(paths["phase3_iql"], index=False)
    phase_eval.to_parquet(paths["phase_eval"], index=False)

    coverage.to_csv(out_dir / "action_source_coverage.csv", index=False, encoding="utf-8-sig")

    # Stage2.2 action magnitude audit must use the rated inner-train pseudo-action
    # distribution only.  Inner-dev/eval rows and broad SSL/BC rows are excluded
    # from the quantile reference to preserve the final temporal contract.
    phase3_year = pd.to_numeric(phase3.get("fiscal_year", phase3.get("year")), errors="coerce")
    phase3_inner_train = phase3.loc[phase3_year <= int(temporal_contract.inner_train_year_max)].copy()
    if phase3_inner_train.empty:
        raise ValueError(f"Stage2 magnitude audit failed: no phase3_iql inner-train rows <= {temporal_contract.inner_train_year_max}")
    magnitude_audit_meta = write_action_magnitude_audit(root, out_dir, phase3_inner_train)
    calibration_meta = {
        "schema_version": "magnitude_calibration_metadata_v32",
        "created_utc": now_utc(),
        "mode": "inner_dev_recalibrated",
        "policy": "quantile_libraries_materialized_in_stage2_candidate_projection_from_inner_train_pseudo_actions",
        "quantile_grid": [50, 65, 75, 85],
        "selected_quantile": None,
        "inner_train_year_max": int(temporal_contract.inner_train_year_max),
        "inner_dev_year": int(temporal_contract.inner_dev_year),
        "recalibration_method": "per_dimension_favorable_direction_abs_action_quantile",
        "ratio_preservation": "preserve_candidate_active_dimension_sign_pattern; replace nonzero magnitudes with selected quantile",
        "oot_used_for_calibration": False,
        "oracle_outcome_used_for_calibration": False,
        "automatic_recalibration_performed": True,
        "action_magnitude_quantiles_json": "action_magnitude_quantiles.json",
        "candidate_magnitude_audit_csv": "candidate_magnitude_audit.csv",
        "candidate_library_hash_per_quantile": {},
        "selected_candidate_library_hash": None,
        "final_candidate_values_source": "stage2_candidate_projection/final_candidate_library__P{q}.yaml",
    }
    write_json(out_dir / "magnitude_calibration_metadata.json", calibration_meta)

    diagnostics = diag_obj["diagnostics"]
    diagnostics["split"] = split_meta
    diagnostics["avs256_enrichment"] = {"broad": broad_avs_meta, "rated": rated_avs_meta}
    diagnostics["input"] = str(inp.relative_to(root))
    diagnostics["outputs"] = {k: str(v.relative_to(root)) for k, v in paths.items()}
    write_json(out_dir / "transition_gap_diagnostics.json", diagnostics)

    metadata = {
        "created_utc": now_utc(),
        "final_stage": "Stage2",
        "stage_role": "final Stage2 input split for candidate projection / BC / IQL",
        "builder_module": "credit_recourse.rl.pipelines.final_stage2_input_splits.pipeline",
        "split_builder_inside_stage2": False,
        "next_state_contract": "all selected Stage3 ACD features must have next__<feature> or <feature>__next",
        "schema_version": "FINAL_STAGE2_INPUT_SPLIT_V4_SEEDLESS_PHASE2_BC",
        **temporal_metadata(temporal_contract, stage="final_stage2_input_splits"),
        "input": str(inp.relative_to(root)),
        "transition_rows": int(len(rated_transitions)),
        "broad_transition_rows": int(len(broad_transitions)),
        "split": split_meta,
        "bc_split_policy": split_meta.get("bc_split_policy"),
        "stage2_seedless_default": bool(split_meta.get("stage2_seedless_default", False)),
        "stage2_split_seed": split_meta.get("stage2_split_seed"),
        "action_dims": ACTION_DIMS,
        "action_columns": ACTION_COLUMNS,
        "action_bounds": BOUNDS,
        "final_action_contract_hash": action_contract_meta["sha256"],
        "final_action_contract_path": "data/final_freeze/configs/final_action_contract.yaml",
        "magnitude_calibration_metadata_json": str((out_dir / "magnitude_calibration_metadata.json").relative_to(root)),
        "action_source_coverage": coverage.to_dict(orient="records"),
        "sector_phi_components_materialized": diagnostics.get("sector_phi_components_materialized", PHI_COMPONENTS),
        "sector_phi_contract_enforced_in_builder": bool(diagnostics.get("sector_phi_contract_enforced_in_builder", False)),
        "sector_phi_component_mapping": diagnostics.get("sector_phi_component_mapping", []),
        "feature_manifest": str((out_dir.parent / "feature_manifest.json").relative_to(root)),
        "cash_flow_substrate_join": cash_flow_join_meta,
        "avs256_enrichment": diagnostics.get("avs256_enrichment", {}),
        **magnitude_audit_meta,
        "warnings": [],
        "outputs": {
            name: {"path": str(path.relative_to(root)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    missing_dims = [r["action_dim"] for r in metadata["action_source_coverage"] if r["source_column"] is None]
    if missing_dims:
        metadata["warnings"].append(f"Some action dimensions have no source column and were zero-imputed: {missing_dims}")
    if split_meta.get("phase1_rows", 0) == 0 or split_meta["phase2_rows"] == 0 or split_meta["phase3_rows"] == 0 or split_meta["phase_eval_rows"] == 0:
        metadata["warnings"].append("One or more split phases are empty; check eval_start_year and data coverage.")

    write_json(out_dir / "metadata.json", metadata)

    print("[OK] wrote", paths["phase1_pretrain"])
    print("[OK] wrote", paths["phase2_bc"])
    print("[OK] wrote", paths["phase3_iql"])
    print("[OK] wrote", paths["phase_eval"])
    print("[OK] wrote", out_dir / "metadata.json")
    print(json.dumps({
        "transition_rows": int(len(rated_transitions)),
        "broad_transition_rows": int(len(broad_transitions)),
        "split": split_meta,
        "missing_action_source_dims": missing_dims,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
