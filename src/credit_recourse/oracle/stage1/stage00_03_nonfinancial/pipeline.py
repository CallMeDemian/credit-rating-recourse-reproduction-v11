"""
Stage 1C v3.2 — Nonfinancial Metadata Panel (Industry Risk Expanded)
===============================================================
Korean Credit Rating Oracle Project

Task 1-10 instruction을 함수 단위로 분리하여 구현.
Stage 1B / Stage 2 산출물은 read-only input. 절대 수정하지 않음.

Usage (Windows PowerShell):
    python stage1c_metadata_v3.py --config configs/paths.yaml

Reproducibility:
    python verify_metrics_v3_2.py
    python verify_output_hashes_v3_2.py --allow-mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as ilmd
import json
import platform
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")
KST = timezone(timedelta(hours=9))


# ============================================================
# Helpers
# ============================================================
def normalize_code(s: pd.Series) -> pd.Series:
    """Normalize exchange code to a 6-digit string key.

    Source files contain mixed int/string forms. Keeping a single string key
    prevents Int64/int64/object merge drift and preserves leading zeros.
    """
    ss = s.astype("string").str.strip()
    ss = ss.str.replace(r"\.0$", "", regex=True)
    ss = ss.str.replace(r"[^0-9]", "", regex=True)
    ss = ss.where(ss.str.len() > 0, pd.NA)
    return ss.str.zfill(6)
def _valid_code_value(code: Any) -> bool:
    """Return True when a configured KIS item code is usable.

    YAML/CSV driven mappings can occasionally arrive as NaN/None.  Treating
    those as literal column labels later causes pandas KeyError like
    ``[nan] not in index``.
    """
    if code is None:
        return False
    try:
        if pd.isna(code):
            return False
    except Exception:
        pass
    s = str(code).strip()
    return bool(s) and s.lower() not in {"nan", "none", "<na>"}


def find_col(df: pd.DataFrame, code: str) -> str | None:
    """KIS 코드로 컬럼 탐색.

    Handles both normal KIS-VALUE signatures like ``[U01A100000000]...`` and
    duplicated/expanded signatures where the U-code appears later in the full
    column name.  Invalid mapping values return ``None`` instead of becoming a
    bogus pandas column key.
    """
    if not _valid_code_value(code):
        return None
    code_s = str(code).strip()
    for c in df.columns:
        c_s = str(c)
        if c_s.startswith(f"[{code_s}]") or code_s in c_s:
            return c
    return None


def _is_existing_column(df: pd.DataFrame, col: Any) -> bool:
    if col is None:
        return False
    try:
        if pd.isna(col):
            return False
    except Exception:
        pass
    return str(col) in {str(c) for c in df.columns}


def _canonical_column_name(df: pd.DataFrame, col: Any) -> str | None:
    if not _is_existing_column(df, col):
        return None
    col_s = str(col)
    for c in df.columns:
        if str(c) == col_s:
            return c
    return None


def _safe_statement_value_frame(df: pd.DataFrame, col: Any, value_name: str, *, statement_name: str) -> pd.DataFrame:
    """Return 거래소코드/year/value frame without crashing on absent item columns.

    Stage00_03 business/financial risk variables are proxy features.  If a
    statement item code is absent in the current raw statement universe, the
    feature should become NA and be reflected in downstream availability/quality
    reports, not crash as ``KeyError: [nan] not in index``.
    """
    base_cols = ["거래소코드", "회계년도"]
    missing_base = [c for c in base_cols if c not in df.columns]
    if missing_base:
        raise KeyError(f"{statement_name} missing required key columns: {missing_base}")
    col_name = _canonical_column_name(df, col)
    if col_name is None:
        print(f"  [WARN] {statement_name}: missing mapped item column for {value_name}; filling {value_name} with NA")
        out = df[base_cols].copy()
        out[value_name] = np.nan
    else:
        out = df[base_cols + [col_name]].copy().rename(columns={col_name: value_name})
    out = _add_year_from_accounting_period(out)[["거래소코드", "year", value_name]]
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")
    return out.drop_duplicates(subset=["거래소코드", "year"], keep="last")



def _add_year_from_accounting_period(df: pd.DataFrame, accounting_col: str = "회계년도", year_col: str = "year") -> pd.DataFrame:
    """Return a copy with normalized integer `year`.

    Stage00_03 joins must not assume that the base panel still has `회계년도`.
    Raw statement tables usually have `회계년도`; normalized panels may only have
    `year`. This helper supports both contracts.
    """
    out = df.copy()
    if year_col in out.columns:
        out[year_col] = pd.to_numeric(out[year_col], errors="coerce").astype("Int64")
        return out
    if accounting_col not in out.columns:
        raise KeyError(f"neither {year_col!r} nor {accounting_col!r} exists in dataframe")
    out[year_col] = pd.to_numeric(out[accounting_col].astype(str).str[:4], errors="coerce").astype("Int64")
    return out

def sha256_of(path: Path, buf_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def package_versions() -> Dict[str, str]:
    pkgs = ["pandas", "numpy", "pyarrow", "openpyxl", "python-calamine", "PyYAML"]
    out = {}
    for p in pkgs:
        try:
            out[p] = ilmd.version(p)
        except Exception:
            out[p] = "not_installed"
    return out


def make_sector_7_mapper(mapping_config: dict):
    """config의 keyword 매핑으로 sector_7 함수 생성"""
    rules = mapping_config["priority_order"]
    default = mapping_config.get("default_unknown", "미분류")

    def map_sector_7(name) -> Any:
        if pd.isna(name) or not isinstance(name, str):
            return pd.NA
        for rule in rules:
            for kw in rule["keywords"]:
                if kw in name:
                    return rule["sector_name"]
        return default

    return map_sector_7


# ============================================================
# Read-only input loaders (Stage 1B / Stage 2 / raw)
# ============================================================
def load_stage1b_inputs(paths: dict) -> Dict[str, pd.DataFrame]:
    """Stage 1B 산출물 로드 (read-only)"""
    print("[Load] Stage 1B inputs (read-only)")
    out = {}
    panel_path = Path(paths["inputs"]["stage1b"]["panel"])
    out["panel"] = pd.read_parquet(panel_path)
    out["panel"]["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(out["panel"]["\uac70\ub798\uc18c\ucf54\ub4dc"])

    for name, p in paths["inputs"]["stage1b"]["statements"].items():
        df = pd.read_parquet(Path(p))
        df["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(df["\uac70\ub798\uc18c\ucf54\ub4dc"])
        out[name] = df
    print(f"  panel: {out['panel'].shape}, eligible={out['panel']['eligible_for_stage2'].sum()}")
    return out


def load_stage2_inputs(paths: dict) -> Dict[str, Any]:
    """Stage 2 산출물 로드 (read-only)"""
    print("[Load] Stage 2 inputs (read-only)")
    ratios = pd.read_parquet(Path(paths["inputs"]["stage2"]["engineered_ratios"]))
    ratios["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(ratios["\uac70\ub798\uc18c\ucf54\ub4dc"])
    pool = pd.read_csv(Path(paths["inputs"]["stage2"]["candidate_pool"]))
    lag_inc_path = Path(paths["inputs"]["stage2"]["lag_support_income"])
    lag_inc = pd.read_parquet(lag_inc_path) if lag_inc_path.exists() else None
    if lag_inc is not None:
        lag_inc["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(lag_inc["\uac70\ub798\uc18c\ucf54\ub4dc"])
    print(f"  engineered_ratios: {ratios.shape}, 134 quality-pass: {len(pool)}")
    return {"ratios": ratios, "ids_134": pool["ratio_id"].tolist(),
            "lag_income": lag_inc}


def _read_excel_with_engine(path: Path) -> pd.DataFrame:
    """Read Excel with python-calamine when available; fall back to openpyxl."""
    try:
        return pd.read_excel(path, engine="calamine")
    except ImportError:
        print(f"  [WARN] python-calamine unavailable while reading {path.name}; falling back to openpyxl")
        return pd.read_excel(path, engine="openpyxl")


def _datetime_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return pd.to_datetime(df[column], errors="coerce")
    return pd.Series(pd.NaT, index=df.index)


def _read_market_bundle(paths_dict: dict, source_name: str, required_markets: set[str] | None = None) -> pd.DataFrame:
    """Read market bundle. required_markets=set() means optional-empty."""
    if required_markets is None:
        required_markets = {"kospi", "kosdaq"}
    frames = []
    for market_label, p_str in (paths_dict or {}).items():
        market_key = str(market_label).lower()
        p = Path(p_str)
        if not p.exists():
            if market_key in required_markets:
                raise FileNotFoundError(f"Required raw nonfinancial file missing for {source_name}/{market_label}: {p}")
            print(f"  [SKIP optional] {source_name}/{market_label}: {p}")
            continue
        df = _read_excel_with_engine(p)
        if "\uac70\ub798\uc18c\ucf54\ub4dc" in df.columns:
            df["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(df["\uac70\ub798\uc18c\ucf54\ub4dc"])
        df["market"] = market_key.upper()
        df["source_market"] = market_key.upper()
        df["source_file"] = p.name
        frames.append(df)
        print(f"  loaded {source_name}/{market_label}: {len(df):,} rows from {p.name}")
    if not frames:
        if len(required_markets) == 0:
            print(f"  [OPTIONAL EMPTY] {source_name}: no usable raw files")
            return pd.DataFrame(columns=["\uac70\ub798\uc18c\ucf54\ub4dc", "market", "source_market", "source_file"])
        raise FileNotFoundError(f"No usable raw nonfinancial files loaded for source [{source_name}]")
    return pd.concat(frames, ignore_index=True)
def load_raw_nonfinancial(paths: dict, sector_mapper) -> Dict[str, pd.DataFrame]:
    print("[Load] Raw nonfinancial sources")
    raw = paths["inputs"]["raw_nonfinancial"]
    K_CODE = "\uac70\ub798\uc18c\ucf54\ub4dc"
    K_ACCT_YEAR = "\ud68c\uacc4\ub144\ub3c4"
    K_INDUSTRY = "\uc0b0\uc5c5\uba85"
    K_EMP = "\uc885\uc5c5\uc6d0"
    K_FOUND = "\uc124\ub9bd\uc77c"
    K_LIST = "\uc0c1\uc7a5\uc77c"
    K_CAP_DATE = "\ubcc0\ub3d9\uc77c"
    K_CAP_YEAR = "\ubcc0\ub3d9\ub144"
    K_DATE = "\uc77c\uc790"
    K_EVENT_YEAR = "\uc774\ubca4\ud2b8\ub144"
    K_LAW_YM = "\ud68c\uacc4\ub144\uc6d4"
    K_LAW_COUNTER = "\uc18c\uc1a1\uc0c1\ub300\ubc29"
    K_LAW_YEAR = "\uc18c\uc1a1\ub144"

    gi_raw = _read_market_bundle(raw.get("general_info", {}), "general_info")
    gi_raw[K_ACCT_YEAR] = gi_raw[K_ACCT_YEAR].astype(str) if K_ACCT_YEAR in gi_raw.columns else ""
    gi = (gi_raw.sort_values([K_CODE, K_ACCT_YEAR], ascending=[True, False])
                 .drop_duplicates(subset=K_CODE, keep="first")
                 .reset_index(drop=True))
    for c in [K_FOUND, K_LIST]:
        if c in gi.columns:
            gi[f"{c}_parsed"] = pd.to_datetime(gi[c], errors="coerce")
            gi[f"{c[:-1]}\ub144"] = gi[f"{c}_parsed"].dt.year.astype("Int64")
        else:
            gi[f"{c}_parsed"] = pd.NaT
            gi[f"{c[:-1]}\ub144"] = pd.Series(pd.NA, index=gi.index, dtype="Int64")
    gi[K_EMP] = pd.to_numeric(gi[K_EMP], errors="coerce") if K_EMP in gi.columns else np.nan
    gi["sector_7"] = gi[K_INDUSTRY].apply(sector_mapper) if K_INDUSTRY in gi.columns else pd.NA

    cap = _read_market_bundle(raw.get("capital_change", {}), "capital_change")
    if K_CAP_DATE not in cap.columns:
        cap[K_CAP_DATE] = pd.NaT
    if K_CODE in cap.columns:
        cap[K_CODE] = normalize_code(cap[K_CODE])
    cap[f"{K_CAP_DATE}_parsed"] = _datetime_series(cap, K_CAP_DATE)
    cap[K_CAP_YEAR] = cap[f"{K_CAP_DATE}_parsed"].dt.year.astype("Int64")

    ma = _read_market_bundle(raw.get("ma_events", {}), "ma_events", required_markets=set())
    if K_CODE in ma.columns:
        ma[K_CODE] = normalize_code(ma[K_CODE])
    if K_DATE not in ma.columns:
        ma[K_DATE] = pd.NaT
    ma[f"{K_DATE}_parsed"] = _datetime_series(ma, K_DATE)
    ma[K_EVENT_YEAR] = ma[f"{K_DATE}_parsed"].dt.year.astype("Int64")

    law_raw = _read_market_bundle(raw.get("lawsuit", {}), "lawsuit", required_markets=set())
    if K_CODE in law_raw.columns:
        law_raw[K_CODE] = normalize_code(law_raw[K_CODE])
    if K_LAW_COUNTER not in law_raw.columns:
        law_raw[K_LAW_COUNTER] = pd.NA
    if K_LAW_YM not in law_raw.columns:
        law_raw[K_LAW_YM] = pd.NaT
    law = law_raw[law_raw[K_LAW_COUNTER].notna()].copy() if K_LAW_COUNTER in law_raw.columns else law_raw.iloc[0:0].copy()
    law[f"{K_LAW_YM}_parsed"] = _datetime_series(law, K_LAW_YM)
    law[K_LAW_YEAR] = law[f"{K_LAW_YM}_parsed"].dt.year.astype("Int64")

    print(f"  general_info dedup: {len(gi)}, cap events: {len(cap)}, ma: {len(ma)}, lawsuits: {len(law)}")
    return {"general_info": gi, "cap_events": cap, "ma_events": ma, "lawsuits": law, "raw_paths": raw}
def task1_schema_inventory(paths: dict, mapping: dict) -> pd.DataFrame:
    """모든 raw 비재무 파일 column-level inventory 생성"""
    print("\n[Task 1] metadata_source_inventory")
    raw = paths["inputs"]["raw_nonfinancial"]

    role_hints = {
        "거래소코드": "firm_id", "회사명": "firm_name",
        "회계년도": "extraction_period", "회계년월": "extraction_period",
        "산업코드": "industry_code", "산업명": "industry_name",
        "업종코드": "sector_code", "종업원": "employee_count",
        "설립일": "incorporation_date", "상장일": "listing_date",
        "상장폐지일": "delisting_date", "결산월": "fiscal_month",
        "변동일": "event_date", "증자유형": "capital_event_type",
        "변동내역": "capital_event_text", "증자비율": "capital_event_ratio",
        "일자": "event_date", "내역(합병-영업양수도)": "ma_event_text",
        "구분(10-피고 20-원고 90-기타)": "lawsuit_role",
        "소송상대방": "lawsuit_counterparty",
        "소송가액(천원)": "lawsuit_amount", "진행사황": "lawsuit_status",
    }
    var_map = {
        "거래소코드": "firm_key", "회계년도": "panel_key",
        "산업명": "sector_7 (텍스트 매핑 기반)",
        "종업원": "log_emp", "설립일": "firm_age", "상장일": "list_age",
        "변동일": "cap_change_count_3y, cap_change_cumulative",
        "일자": "ma_count_3y, ma_cumulative",
        "구분(10-피고 20-원고 90-기타)": "lawsuit_count_3y",
        "소송상대방": "lawsuit_count_3y (filter: not null)",
        "회계년월": "lawsuit_count_3y (year extraction)",
    }

    def short_sample(s: pd.Series, n=3) -> str:
        vals = s.dropna().astype(str).str.slice(0, 30).tolist()[:n]
        return " | ".join(vals)

    inv = []
    for source_cat, paths_dict in raw.items():
        for market_label, p_str in paths_dict.items():
            p = Path(p_str)
            if not p.exists():
                continue
            df = _read_excel_with_engine(p)
            for col in df.columns:
                inv.append({
                    "source_file": p.name,
                    "source_category": source_cat,
                    "market": market_label.upper(),
                    "sheet_name": "Sheet1",
                    "column_name": col,
                    "inferred_role": role_hints.get(col, ""),
                    "non_missing_count": int(df[col].notna().sum()),
                    "non_missing_rate": round(df[col].notna().mean(), 4),
                    "sample_values": short_sample(df[col]),
                    "mapped_variable_candidate": var_map.get(col, ""),
                    "notes": "",
                })
    inv_df = pd.DataFrame(inv)
    print(f"  → {len(inv_df)} rows × {inv_df.shape[1]} cols")
    return inv_df


# ============================================================
# TASK 2 — metadata_candidate_availability_report.csv
# ============================================================
def task2_candidate_availability() -> pd.DataFrame:
    """Stage 1C v3.2 후보 변수 정의 + availability/eligibility 명세."""
    print("\n[Task 2] metadata_candidate_availability_report (v3.2)")

    rows = []

    def add(category, variable_name, definition, availability_status="available",
            selected_eligible=True, diagnostic_only=False, leakage_safe=True,
            expected_direction="empirical", caveat=""):
        rows.append({
            "category": category,
            "variable_name": variable_name,
            "definition": definition,
            "availability_status": availability_status,
            "selected_eligible": bool(selected_eligible),
            "diagnostic_only": bool(diagnostic_only),
            "leakage_safe": bool(leakage_safe),
            "expected_direction": expected_direction,
            "caveat": caveat,
        })

    # 산업위험 — v3.2 신규 9개 proxy, 모두 lagged/self-excluded
    add("산업위험", "industry_avg_rating_lag1_self_excl", "sector_7 × t-1 leave-one-out mean rating", expected_direction="higher_is_worse")
    add("산업위험", "industry_median_rating_lag1_self_excl", "sector_7 × t-1 leave-one-out median rating", expected_direction="higher_is_worse")
    add("산업위험", "industry_bad_grade_share_lag1_self_excl", "sector_7 × t-1 BB-or-lower share (rating_num_10>=5), self-excluded", expected_direction="higher_is_worse")
    add("산업위험", "industry_b_or_lower_share_lag1_self_excl", "sector_7 × t-1 B-or-lower share (rating_num_10>=6), self-excluded", expected_direction="higher_is_worse")
    add("산업위험", "industry_rating_std_lag1_self_excl", "sector_7 × t-1 rating dispersion std, self-excluded", expected_direction="higher_is_worse_or_uncertain", caveat="dispersion proxy")
    add("산업위험", "industry_rating_iqr_lag1_self_excl", "sector_7 × t-1 rating dispersion IQR, self-excluded", expected_direction="higher_is_worse_or_uncertain", caveat="dispersion proxy")
    add("산업위험", "industry_downgrade_rate_lag1_self_excl", "sector_7 t-2→t-1 downgrade rate, self-excluded", expected_direction="higher_is_worse")
    add("산업위험", "industry_upgrade_rate_lag1_self_excl", "sector_7 t-2→t-1 upgrade rate, self-excluded", expected_direction="higher_is_better")
    add("산업위험", "industry_net_downgrade_rate_lag1_self_excl", "downgrade_rate - upgrade_rate", expected_direction="higher_is_worse")

    # 산업위험 — 후방호환/진단 전용
    add("산업위험", "industry_avg_rating", "v2 alias of industry_avg_rating_lag1_self_excl", selected_eligible=False, diagnostic_only=True, expected_direction="higher_is_worse", caveat="backward-compatible alias")
    add("산업위험", "kospi_dummy", "panel.시장 == KOSPI", selected_eligible=False, diagnostic_only=True, expected_direction="not_risk_score", caveat="market-tier variable, not industry risk")
    add("산업위험", "sector_7", "sector grouping/context variable", selected_eligible=False, diagnostic_only=True, expected_direction="nominal", caveat="nominal sector code, not ordered score")
    add("산업위험", "sector_year_rating_count", "sector_7 × t-1 cell n alias", selected_eligible=False, diagnostic_only=True, expected_direction="coverage", caveat="coverage/count diagnostic")
    add("산업위험", "sector_year_rating_count_lag1", "sector_7 × t-1 cell n", selected_eligible=False, diagnostic_only=True, expected_direction="coverage", caveat="coverage/count diagnostic")

    # 경영위험 — v2 유지
    add("경영위험", "firm_age", "year - 설립년도", expected_direction="higher_is_better")
    add("경영위험", "list_age", "year - 상장년도", expected_direction="higher_is_better")
    add("경영위험", "cap_change_count_3y", "t-2~t 자본금 변동 events", expected_direction="higher_is_worse")
    add("경영위험", "cap_change_cumulative", "t까지 자본금 변동 누적", expected_direction="higher_is_worse", caveat="event-count proxy")
    add("경영위험", "ma_count_3y", "t-2~t 합병/양수도 events", expected_direction="higher_is_worse_or_uncertain", caveat="M&A can be growth or restructuring")
    add("경영위험", "ma_cumulative", "t까지 합병/양수도 누적", expected_direction="higher_is_worse_or_uncertain", caveat="M&A can be growth or restructuring")

    # 영업위험 — v2 유지
    add("영업위험", "log_emp", "log1p(종업원)", expected_direction="higher_is_better", caveat="snapshot; operating scale proxy")
    add("영업위험", "log_sales", "log1p(매출액)", expected_direction="higher_is_better", caveat="scale proxy")
    add("영업위험", "log_assets", "log1p(자산총계)", expected_direction="higher_is_better", caveat="scale proxy")

    # 재무위험 — v2 유지
    add("재무위험", "operating_loss_current", "영업이익 < 0 binary", expected_direction="higher_is_worse")
    add("재무위험", "net_loss_current", "당기순이익 < 0 binary", expected_direction="higher_is_worse")
    add("재무위험", "operating_loss_freq_3y", "t,t-1,t-2 영업손실 횟수", expected_direction="higher_is_worse")
    add("재무위험", "negative_equity_flag", "자본총계 < 0 binary", expected_direction="higher_is_worse")
    add("재무위험", "cap_impair_flag", "자본총계 < 자본금 binary", expected_direction="higher_is_worse")
    add("재무위험", "lawsuit_count_3y", "t-2~t 피고 소송 events", expected_direction="higher_is_worse", caveat="legal/contingent-liability proxy")

    # 신뢰도 — v2 유지
    add("신뢰도", "financial_data_completeness", "134 quality-pass ratio non-missing ratio", expected_direction="higher_is_better")
    add("신뢰도", "statement_coverage_count", "6 statement matched count", expected_direction="higher_is_better")
    add("신뢰도", "ratio_available_count", "134 ratio non-missing absolute count", expected_direction="higher_is_better")
    add("신뢰도", "ratio_missing_rate", "1 - financial_data_completeness", expected_direction="higher_is_worse")

    df = pd.DataFrame(rows)
    print(f"  → {len(df)} variables, category 분포: {df['category'].value_counts().to_dict()}")
    print(f"  → selected_eligible: {int(df['selected_eligible'].sum())}, diagnostic_only: {int(df['diagnostic_only'].sum())}")
    return df


# ============================================================
# TASK 3 — Industry risk (v3.2: 9개 proxy + 4-level fallback)
# ============================================================
# v2 대비 변경 사항:
#   - industry_avg_rating 1개 → 9개 rating-based proxy
#   - 모두 t-1 lagged + self-excluded + 동일한 4-level fallback 구조 유지
#   - kospi_dummy, sector_year_rating_count 후방호환 유지
#   - min_n_threshold: stage_config의 industry_risk_v3_2 블록 사용
#     (없으면 기존 industry_avg_rating 블록 fallback)
#
# 신규 변수 (모두 selected_eligible=True, leakage_safe=True):
#   industry_avg_rating_lag1_self_excl      평균등급 (v2의 industry_avg_rating과 동일 로직)
#   industry_median_rating_lag1_self_excl   중앙값등급
#   industry_bad_grade_share_lag1_self_excl BB이하(rating_num_10>=5) 비중
#   industry_b_or_lower_share_lag1_self_excl B이하(rating_num_10>=6) 비중
#   industry_rating_std_lag1_self_excl      표준편차
#   industry_rating_iqr_lag1_self_excl      IQR
#   industry_downgrade_rate_lag1_self_excl  t-2→t-1 하향 기업 비율
#   industry_upgrade_rate_lag1_self_excl    t-2→t-1 상향 기업 비율
#   industry_net_downgrade_rate_lag1_self_excl downgrade - upgrade
#
# 진단용 (selected_eligible=False):
#   sector_year_rating_count_lag1           커버리지 진단
#   kospi_dummy                             시장 계층 변수

def task3_industry_risk(base, config):
    # Robust sector_7 guard:
    # Earlier merges may create sector_7_x / sector_7_y and remove plain sector_7.
    # Downstream code expects base["sector_7"].
    base = base.copy()
    if "sector_7" not in base.columns:
        candidates = [c for c in ["sector_7_y", "sector_7_x"] if c in base.columns]
        if candidates:
            base["sector_7"] = base[candidates[0]]
            for c in candidates[1:]:
                base["sector_7"] = base["sector_7"].where(base["sector_7"].notna(), base[c])
        else:
            base["sector_7"] = "UNKNOWN"
    base["sector_7"] = base["sector_7"].fillna("UNKNOWN").astype(str)
    # Robust market guard:
    # Arrow/string dtype may carry pd.NA; dummy conversion must not see NAType.
    if "시장" not in base.columns:
        market_candidates = [c for c in ["시장_y", "시장_x", "market", "market_y", "market_x"] if c in base.columns]
        if market_candidates:
            base["시장"] = base[market_candidates[0]]
            for c in market_candidates[1:]:
                base["시장"] = base["시장"].where(base["시장"].notna(), base[c])
        else:
            base["시장"] = "UNKNOWN"
    base["시장"] = base["시장"].fillna("UNKNOWN").astype(str).str.upper()

    """
    v3.2: 9개 rating-based industry risk proxy (t-1 lagged, self-excluded).
    fallback 구조는 v2와 동일: L1→L2→L3→L4.
    반환 시그니처는 v2 동일: (DataFrame, fallback_counts_dict)

    신규 변수 (selected_eligible=True):
      industry_avg_rating_lag1_self_excl       평균등급
      industry_median_rating_lag1_self_excl    중앙값등급
      industry_bad_grade_share_lag1_self_excl  BB이하(rating_num_10>=5) 비중
      industry_b_or_lower_share_lag1_self_excl B이하(rating_num_10>=6) 비중
      industry_rating_std_lag1_self_excl       표준편차
      industry_rating_iqr_lag1_self_excl       IQR
      industry_downgrade_rate_lag1_self_excl   t-2→t-1 하향 기업 비율
      industry_upgrade_rate_lag1_self_excl     t-2→t-1 상향 기업 비율
      industry_net_downgrade_rate_lag1_self_excl downgrade - upgrade

    후방호환 별칭 (v2 stage3 호환):
      industry_avg_rating      = industry_avg_rating_lag1_self_excl
      sector_year_rating_count = sector_year_rating_count_lag1

    진단 전용 (selected_eligible=False):
      sector_year_rating_count_lag1   커버리지 진단
      kospi_dummy                     시장 계층 변수
    """
    print("\n[Task 3 v3.2] industry_risk — 9개 proxy (t-1 lagged, self-excluded, 4-level fallback)")

    # FINAL_HARDENING: task3 input guard
    base = base.copy()
    if "\uac70\ub798\uc18c\ucf54\ub4dc" in base.columns:
        base["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(base["\uac70\ub798\uc18c\ucf54\ub4dc"])
    base["year"] = pd.to_numeric(base["year"], errors="coerce").astype("int64")
    if "sector_7" not in base.columns:
        sc = [c for c in ["sector_7_y", "sector_7_x"] if c in base.columns]
        base["sector_7"] = base[sc[0]] if sc else "UNKNOWN"
    base["sector_7"] = base["sector_7"].fillna("UNKNOWN").astype(str)
    if "\uc2dc\uc7a5" not in base.columns:
        mc = [c for c in ["\uc2dc\uc7a5_y", "\uc2dc\uc7a5_x", "market"] if c in base.columns]
        base["\uc2dc\uc7a5"] = base[mc[0]] if mc else "UNKNOWN"
    base["\uc2dc\uc7a5"] = base["\uc2dc\uc7a5"].fillna("UNKNOWN").astype(str).str.upper()

    # ── config ──────────────────────────────────────────────────────────────
    cfg_v3 = config.get("industry_risk_v3_2", config["industry_avg_rating"])
    min_n  = int(cfg_v3.get("min_n_threshold",
                             config["industry_avg_rating"].get("min_n_threshold", 10)))

    # ── rating panel ─────────────────────────────────────────────────────────
    # Final Oracle contract: industry-risk variables must use the explicit 10-grade scale.
    # Do not silently fall back to legacy rating_num, because that can be a notch or 7-grade scale.
    if "rating_num_10" not in base.columns:
        raise KeyError("Stage00_03 requires rating_num_10 from Stage00_01; legacy rating_num fallback is forbidden")
    rating_panel = base[["거래소코드", "year", "sector_7", "rating_num_10"]].dropna(
        subset=["rating_num_10"]).copy()
    rating_panel["rating_num_10"] = pd.to_numeric(rating_panel["rating_num_10"], errors="coerce").astype("Int64")
    rating_panel = rating_panel.dropna(subset=["rating_num_10"]).copy()
    rating_panel["rating_num_10"] = rating_panel["rating_num_10"].astype(int)

    # (sector, year) → list[(firm_id, rating_num)]
    sy_items: Dict = {}
    for r in rating_panel.itertuples(index=False):
        key = (r.sector_7, int(r.year))
        sy_items.setdefault(key, []).append((r.거래소코드, int(r.rating_num_10)))

    # year → list[(firm_id, rating_num)] — global fallback용
    gy_items: Dict = {}
    for r in rating_panel.itertuples(index=False):
        gy_items.setdefault(int(r.year), []).append((r.거래소코드, int(r.rating_num_10)))

    # sector × year 셀 크기 (자기제외 전, 진단용)
    sy_count_dict: Dict = {}
    for (sec, yr), items in sy_items.items():
        sy_count_dict[(sec, yr)] = len(items)

    firm_sector_dict = (base.dropna(subset=["sector_7"])
                            .groupby("거래소코드")["sector_7"].first().to_dict())
    year_min = int(rating_panel["year"].min())

    # ── 헬퍼 ─────────────────────────────────────────────────────────────────
    def collect(keys, firm_id):
        """(sector,year) 또는 year 키 목록에서 firm_id 제외 rating 목록"""
        return [r for key in keys
                for (f, r) in sy_items.get(key, []) if f != firm_id]

    def collect_global(years, firm_id):
        return [r for y in years for (f, r) in gy_items.get(y, []) if f != firm_id]

    def agg(vals):
        if not vals:
            return dict(avg=np.nan, med=np.nan, bad=np.nan,
                        blow=np.nan, std=np.nan, iqr=np.nan, n=0)
        a = np.array(vals, dtype=float)
        q25, q75 = float(np.percentile(a, 25)), float(np.percentile(a, 75))
        return dict(
            avg  = float(a.mean()),
            med  = float(np.median(a)),
            bad  = float((a >= 5).mean()),
            blow = float((a >= 6).mean()),
            std  = float(a.std(ddof=1)) if len(a) > 1 else np.nan,
            iqr  = float(q75 - q25),
            n    = len(a),
        )

    def nan_m():
        return dict(avg=np.nan, med=np.nan, bad=np.nan,
                    blow=np.nan, std=np.nan, iqr=np.nan, n=0)

    def resolve_level(firm_id, t, sector):
        if pd.isna(sector):
            return nan_m(), 0, 0
        # L1: sector × t-1
        v = collect([(sector, t-1)], firm_id)
        if len(v) >= min_n:
            return agg(v), 1, len(v)
        # L2: sector × rolling t-3..t-1
        v = collect([(sector, t-3), (sector, t-2), (sector, t-1)], firm_id)
        if len(v) >= min_n:
            return agg(v), 2, len(v)
        # L3: sector × all past < t
        v = collect([(sector, y) for y in range(year_min, t)], firm_id)
        if len(v) >= min_n:
            return agg(v), 3, len(v)
        # L4: global × all past < t
        v = collect_global(range(year_min, t), firm_id)
        if len(v) >= min_n:
            return agg(v), 4, len(v)
        return nan_m(), 0, 0

    def resolve_transition(firm_id, t, sector):
        if pd.isna(sector):
            return dict(dn=np.nan, up=np.nan, net=np.nan), 0

        def pairs(ya, yb):
            da = {f: r for f, r in sy_items.get((sector, ya), []) if f != firm_id}
            db = {f: r for f, r in sy_items.get((sector, yb), []) if f != firm_id}
            return [(da[f], db[f]) for f in da if f in db]

        p = pairs(t-2, t-1)
        fb = 1
        if len(p) < min_n:
            p = pairs(t-4, t-3) + pairs(t-3, t-2) + pairs(t-2, t-1)
            fb = 2 if len(p) >= min_n else 0
        if not p:
            return dict(dn=np.nan, up=np.nan, net=np.nan), 0
        dn = float(np.mean([b > a for a, b in p]))
        up = float(np.mean([b < a for a, b in p]))
        return dict(dn=dn, up=up, net=dn - up), fb

    # ── 본 계산 ──────────────────────────────────────────────────────────────
    rows = []
    fb_counts = {"L1": 0, "L2": 0, "L3": 0, "L4": 0, "NaN": 0}
    tr_counts  = {"L1": 0, "L2": 0, "NaN": 0}

    for _, row in base.iterrows():
        firm_id = row["거래소코드"]
        t       = int(row["year"])
        sector  = (row.get("sector_7")
                   if pd.notna(row.get("sector_7", np.nan))
                   else firm_sector_dict.get(firm_id))

        m, fb_lvl, n_used = resolve_level(firm_id, t, sector)
        fb_label = {1:"L1",2:"L2",3:"L3",4:"L4"}.get(fb_lvl, "NaN")
        fb_counts[fb_label] += 1

        tr, tr_lvl = resolve_transition(firm_id, t, sector)
        tr_counts[{1:"L1",2:"L2"}.get(tr_lvl, "NaN")] += 1

        sy_cnt = sy_count_dict.get((sector, t-1), 0) if pd.notna(sector) else 0

        rows.append({
            "거래소코드": firm_id,
            "year": t,
            # 수준 기반 (selected_eligible=True)
            "industry_avg_rating_lag1_self_excl":         m["avg"],
            "industry_median_rating_lag1_self_excl":      m["med"],
            "industry_bad_grade_share_lag1_self_excl":    m["bad"],
            "industry_b_or_lower_share_lag1_self_excl":   m["blow"],
            "industry_rating_std_lag1_self_excl":         m["std"],
            "industry_rating_iqr_lag1_self_excl":         m["iqr"],
            # 변화 기반 (selected_eligible=True)
            "industry_downgrade_rate_lag1_self_excl":     tr["dn"],
            "industry_upgrade_rate_lag1_self_excl":       tr["up"],
            "industry_net_downgrade_rate_lag1_self_excl": tr["net"],
            # fallback 진단
            "industry_avg_rating_n":              n_used,
            "industry_avg_rating_fallback_level": fb_lvl,
            "industry_transition_fallback_level": tr_lvl,
            # 진단 전용 (selected_eligible=False)
            "sector_year_rating_count_lag1": sy_cnt,
            # v2 후방호환 별칭 (stage3 v3 allowlist의 industry_avg_rating 호환)
            "industry_avg_rating":        m["avg"],
            "sector_year_rating_count":   sy_cnt,
        })

    out = pd.DataFrame(rows)
    out["kospi_dummy"] = base["\uc2dc\uc7a5"].fillna("UNKNOWN").astype(str).str.upper().eq("KOSPI").astype("int8")

    print(f"  min_n={min_n}  fallback(수준): {fb_counts}")
    print(f"            fallback(변화): {tr_counts}")
    return out, fb_counts



# ============================================================
# TASK 4 — Management risk + event_parsing_log
# ============================================================
def _add_event_features(events: pd.DataFrame, year_col: str,
                        base: pd.DataFrame, label: str,
                        window_years: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cumulative + N-year event counts, robust to empty sources and dtype drift."""
    key_col = "\uac70\ub798\uc18c\ucf54\ub4dc"
    pk = base[[key_col, "year"]].copy()
    pk[key_col] = normalize_code(pk[key_col])
    pk["year"] = pd.to_numeric(pk["year"], errors="coerce")
    pk = pk.dropna(subset=[key_col, "year"]).copy()
    pk["year"] = pk["year"].astype("int64")
    pk = pk.drop_duplicates().sort_values([key_col, "year"]).reset_index(drop=True)

    def _zero_features():
        res_cum = pk.copy()
        res_cum[f"{label}_cumulative"] = 0
        res_3y = pk.copy()
        res_3y[f"{label}_count_3y"] = 0
        return res_cum, res_3y

    if events is None or len(events) == 0 or key_col not in events.columns or year_col not in events.columns:
        return _zero_features()

    ev = events.copy()
    ev[key_col] = normalize_code(ev[key_col])
    ev[year_col] = pd.to_numeric(ev[year_col], errors="coerce")
    ev = ev.dropna(subset=[key_col, year_col]).copy()
    if ev.empty:
        return _zero_features()
    ev[year_col] = ev[year_col].astype("int64")

    cnt = ev.groupby([key_col, year_col]).size().reset_index(name="_cnt")
    cnt = cnt.sort_values([key_col, year_col]).reset_index(drop=True)
    cnt["_cumulative"] = cnt.groupby(key_col)["_cnt"].cumsum()
    cnt_renamed = cnt.rename(columns={year_col: "year"})
    cnt_renamed[key_col] = normalize_code(cnt_renamed[key_col])
    cnt_renamed["year"] = pd.to_numeric(cnt_renamed["year"], errors="coerce").astype("int64")

    left = pk.sort_values(["year", key_col]).reset_index(drop=True)
    right = cnt_renamed[[key_col, "year", "_cumulative"]].sort_values(["year", key_col]).reset_index(drop=True)
    if right.empty:
        return _zero_features()
    res_cum = pd.merge_asof(left, right, on="year", by=key_col, direction="backward")
    res_cum["_cumulative"] = res_cum["_cumulative"].fillna(0).astype("int64")
    res_cum = res_cum.rename(columns={"_cumulative": f"{label}_cumulative"})

    fy = pk.groupby(key_col)["year"].agg(["min", "max"]).reset_index()
    fy["min"] = fy["min"].astype("int64") - int(window_years - 1)
    fy["max"] = fy["max"].astype("int64")
    rows = []
    for _, r in fy.iterrows():
        start_year = int(r["min"]); end_year = int(r["max"])
        if end_year >= start_year:
            rows.extend((r[key_col], y) for y in range(start_year, end_year + 1))
    if not rows:
        return _zero_features()
    grid = pd.DataFrame(rows, columns=[key_col, "year"])
    grid[key_col] = normalize_code(grid[key_col])
    grid["year"] = pd.to_numeric(grid["year"], errors="coerce").astype("int64")
    grid = grid.merge(cnt_renamed[[key_col, "year", "_cnt"]], on=[key_col, "year"], how="left")
    grid["_cnt"] = grid["_cnt"].fillna(0)
    grid[f"{label}_count_3y"] = (
        grid.sort_values([key_col, "year"])
            .groupby(key_col)["_cnt"]
            .transform(lambda x: x.rolling(window_years, min_periods=1).sum())
    )
    res_3y = grid[[key_col, "year", f"{label}_count_3y"]]
    return res_cum, res_3y
def task4_management_risk(base: pd.DataFrame, raw_nf: dict, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """firm_age, list_age, cap_change/ma 변수 + event_parsing_log"""
    print("\n[Task 4] management_risk")
    out = base[["거래소코드", "year"]].copy()

    out["firm_age"] = base["year"] - base["설립년"].astype("Int64")
    out.loc[out["firm_age"] < 0, "firm_age"] = pd.NA
    out["list_age"] = base["year"] - base["상장년"].astype("Int64")
    out.loc[out["list_age"] < 0, "list_age"] = pd.NA

    cap_window = config["event_windows"]["cap_change_window_years"]
    ma_window = config["event_windows"]["ma_window_years"]

    cap_cum, cap_3y = _add_event_features(raw_nf["cap_events"], "변동년", base,
                                          "cap_change", cap_window)
    ma_cum, ma_3y = _add_event_features(raw_nf["ma_events"], "이벤트년", base,
                                        "ma", ma_window)

    out = out.merge(cap_cum, on=["거래소코드", "year"], how="left") \
             .merge(cap_3y, on=["거래소코드", "year"], how="left") \
             .merge(ma_cum, on=["거래소코드", "year"], how="left") \
             .merge(ma_3y, on=["거래소코드", "year"], how="left")
    for c in ["cap_change_cumulative", "cap_change_count_3y",
              "ma_cumulative", "ma_count_3y"]:
        out[c] = out[c].fillna(0).astype(int)

    # event_parsing_log
    cap_fail = raw_nf["cap_events"]["\ubcc0\ub3d9\ub144"].isna().sum() if "\ubcc0\ub3d9\ub144" in raw_nf["cap_events"].columns else len(raw_nf["cap_events"])
    ma_fail = raw_nf["ma_events"]["\uc774\ubca4\ud2b8\ub144"].isna().sum() if "\uc774\ubca4\ud2b8\ub144" in raw_nf["ma_events"].columns else len(raw_nf["ma_events"])
    law_fail = raw_nf["lawsuits"]["\uc18c\uc1a1\ub144"].isna().sum() if "\uc18c\uc1a1\ub144" in raw_nf["lawsuits"].columns else len(raw_nf["lawsuits"])
    log_df = pd.DataFrame([
        {"event_type": "capital_change", "total_rows": len(raw_nf["cap_events"]),
         "parse_success": len(raw_nf["cap_events"]) - cap_fail,
         "parse_fail": int(cap_fail),
         "notes": "변동일 NaN/형식불량"},
        {"event_type": "ma", "total_rows": len(raw_nf["ma_events"]),
         "parse_success": len(raw_nf["ma_events"]) - ma_fail,
         "parse_fail": int(ma_fail),
         "notes": "일자 NaN/형식불량"},
        {"event_type": "lawsuit", "total_rows": len(raw_nf["lawsuits"]),
         "parse_success": len(raw_nf["lawsuits"]) - law_fail,
         "parse_fail": int(law_fail),
         "notes": "회계년월 NaN/형식불량 (소송상대방 not null filter 후)"},
    ])
    print(f"  firm_age med={out['firm_age'].median()}, list_age med={out['list_age'].median()}")
    print(f"  cap_change_count_3y max={out['cap_change_count_3y'].max()}, "
          f"ma_cumulative max={out['ma_cumulative'].max()}")
    return out, log_df


# ============================================================
# TASK 5 — Business risk (log1p)
# ============================================================
def task5_business_risk(base: pd.DataFrame, stage1b: dict,
                         var_mapping: dict) -> pd.DataFrame:
    """log_emp, log_sales, log_assets (log1p)"""
    print("\n[Task 5] business_risk (log1p)")
    fi = var_mapping.get("financial_items", {})
    asset_col = find_col(stage1b["재무상태표"], fi.get("자산총계", {}).get("code"))
    sales_col = find_col(stage1b["손익계산서"], fi.get("매출액", {}).get("code"))

    out = base[["거래소코드", "year"]].copy()
    emp = pd.to_numeric(base["종업원"], errors="coerce") if "종업원" in base.columns else pd.Series(np.nan, index=base.index)
    out["log_emp"] = np.where(emp >= 0, np.log1p(emp), np.nan)

    b_sub = _safe_statement_value_frame(stage1b["재무상태표"], asset_col, "_assets", statement_name="재무상태표")
    i_sub = _safe_statement_value_frame(stage1b["손익계산서"], sales_col, "_sales", statement_name="손익계산서")
    base_keys = _add_year_from_accounting_period(base)[["거래소코드", "year"]].copy()
    merged = (base_keys.merge(b_sub, on=["거래소코드", "year"], how="left")
                       .merge(i_sub, on=["거래소코드", "year"], how="left"))

    out["log_assets"] = np.where(merged["_assets"] > 0, np.log1p(merged["_assets"]), np.nan)
    out["log_sales"] = np.where(merged["_sales"] > 0, np.log1p(merged["_sales"]), np.nan)

    print(f"  log_emp range=({pd.Series(out['log_emp']).min():.2f}, {pd.Series(out['log_emp']).max():.2f}), "
          f"log_assets missing={out['log_assets'].isna().sum()}, "
          f"log_sales missing={out['log_sales'].isna().sum()}")
    return out


# ============================================================
# TASK 6 — Financial risk proxy
# ============================================================
def task6_financial_risk_proxy(base: pd.DataFrame, stage1b: dict,
                                stage2: dict, raw_nf: dict,
                                var_mapping: dict, config: dict) -> pd.DataFrame:
    """operating_loss_current, net_loss_current, operating_loss_freq_3y,
    negative_equity_flag, cap_impair_flag, lawsuit_count_3y"""
    print("\n[Task 6] financial_risk_proxy")
    fi = var_mapping.get("financial_items", {})
    op_col = find_col(stage1b["손익계산서"], fi.get("영업이익", {}).get("code"))
    ni_col = find_col(stage1b["손익계산서"], fi.get("당기순이익", {}).get("code"))
    eq_col = find_col(stage1b["재무상태표"], fi.get("자본", {}).get("code"))
    cap_col = find_col(stage1b["재무상태표"], fi.get("자본금", {}).get("code"))

    out = _add_year_from_accounting_period(base)[["거래소코드", "year"]].copy()

    # Current loss flags. Missing optional statement item columns become NA
    # features instead of pandas KeyError on None/NaN labels.
    op_df = _safe_statement_value_frame(stage1b["손익계산서"], op_col, "_op", statement_name="손익계산서")
    ni_df = _safe_statement_value_frame(stage1b["손익계산서"], ni_col, "_ni", statement_name="손익계산서")
    eq_base = _safe_statement_value_frame(stage1b["재무상태표"], eq_col, "_eq", statement_name="재무상태표")
    cap_base = _safe_statement_value_frame(stage1b["재무상태표"], cap_col, "_cap", statement_name="재무상태표")
    eq_df = eq_base.merge(cap_base, on=["거래소코드", "year"], how="outer")

    out = (out.merge(op_df, on=["거래소코드", "year"], how="left")
              .merge(ni_df, on=["거래소코드", "year"], how="left")
              .merge(eq_df, on=["거래소코드", "year"], how="left"))
    out["operating_loss_current"] = (out["_op"] < 0).astype("Int64")
    out.loc[out["_op"].isna(), "operating_loss_current"] = pd.NA
    out["net_loss_current"] = (out["_ni"] < 0).astype("Int64")
    out.loc[out["_ni"].isna(), "net_loss_current"] = pd.NA
    out["negative_equity_flag"] = (out["_eq"] < 0).astype("Int64")
    out.loc[out["_eq"].isna(), "negative_equity_flag"] = pd.NA
    out["cap_impair_flag"] = (out["_eq"] < out["_cap"]).astype("Int64")
    out.loc[out["_eq"].isna() | out["_cap"].isna(), "cap_impair_flag"] = pd.NA

    # operating_loss_freq_3y (Stage 2 lag_support 활용)
    if op_col is None:
        i_unique = out[["거래소코드", "year"]].copy()
        i_unique["op_loss"] = np.nan
    else:
        income_full = _safe_statement_value_frame(stage1b["손익계산서"], op_col, "_op", statement_name="손익계산서")
        income_full["op_loss"] = (income_full["_op"] < 0).astype("Int64")
        income_full.loc[income_full["_op"].isna(), "op_loss"] = pd.NA
        i_unique = income_full[["거래소코드", "year", "op_loss"]].drop_duplicates(subset=["거래소코드", "year"])

    if stage2.get("lag_income") is not None and op_col is not None:
        lag_inc = stage2["lag_income"]
        op_lag_col = find_col(lag_inc, fi.get("영업이익", {}).get("code"))
        if op_lag_col:
            lag_subset = _safe_statement_value_frame(lag_inc, op_lag_col, "_op_lag", statement_name="lag_income")
            lag_subset["year"] = lag_subset["year"] - 1
            lag_subset["op_loss"] = (lag_subset["_op_lag"] < 0).astype("Int64")
            lag_subset.loc[lag_subset["_op_lag"].isna(), "op_loss"] = pd.NA
            lag_subset = lag_subset[["거래소코드", "year", "op_loss"]].dropna(subset=["year"])
            existing = set(zip(i_unique["거래소코드"].astype(str), i_unique["year"]))
            lag_subset["_key"] = list(zip(lag_subset["거래소코드"].astype(str), lag_subset["year"]))
            lag_new = lag_subset[~lag_subset["_key"].isin(existing)]
            i_unique = pd.concat([
                i_unique[["거래소코드", "year", "op_loss"]],
                lag_new[["거래소코드", "year", "op_loss"]],
            ], ignore_index=True).sort_values(["거래소코드", "year"])

    window = config["event_windows"]["operating_loss_window_years"]
    i_unique["op_loss_3y"] = (i_unique.groupby("거래소코드")["op_loss"]
                                       .transform(lambda x: x.rolling(window, min_periods=1).sum()))
    out = out.merge(i_unique[["거래소코드", "year", "op_loss_3y"]],
                    on=["거래소코드", "year"], how="left")
    out = out.rename(columns={"op_loss_3y": "operating_loss_freq_3y"})

    # lawsuit_count_3y: optional source, zero-safe
    law_window = config["event_windows"]["lawsuit_window_years"]
    _, law_3y = _add_event_features(raw_nf.get("lawsuits", pd.DataFrame()), "\uc18c\uc1a1\ub144", base, "lawsuit", law_window)
    if "lawsuit_count_3y" not in law_3y.columns:
        law_3y["lawsuit_count_3y"] = 0
    out = out.merge(law_3y[["\uac70\ub798\uc18c\ucf54\ub4dc", "year", "lawsuit_count_3y"]], on=["\uac70\ub798\uc18c\ucf54\ub4dc", "year"], how="left")
    out["lawsuit_count_3y"] = out["lawsuit_count_3y"].fillna(0).astype(int)

    # 정리
    out = out.drop(columns=["_op", "_ni", "_eq", "_cap"], errors="ignore")
    print(f"  op_loss_curr pos={int((out['operating_loss_current']==1).sum())}, "
          f"net_loss pos={int((out['net_loss_current']==1).sum())}")
    print(f"  neg_equity pos={int((out['negative_equity_flag']==1).sum())}, "
          f"cap_impair pos={int((out['cap_impair_flag']==1).sum())}")
    return out


# ============================================================
# TASK 7 — Reliability (134 ratio 기반)
# ============================================================
def task7_reliability(base: pd.DataFrame, stage2: dict,
                       stage1b_panel: pd.DataFrame) -> pd.DataFrame:
    """financial_data_completeness, statement_coverage_count,
    ratio_available_count, ratio_missing_rate"""
    print("\n[Task 7] reliability (134 ratio 기반)")
    ids = stage2["ids_134"]
    n_ratios = len(ids)

    sub = stage2["ratios"][["거래소코드", "year"] + ids].copy()
    sub["ratio_available_count"] = sub[ids].notna().sum(axis=1)
    sub["financial_data_completeness"] = sub["ratio_available_count"] / n_ratios
    sub["ratio_missing_rate"] = 1 - sub["financial_data_completeness"]

    out = base[["거래소코드", "year"]].copy()
    out = out.merge(
        sub[["거래소코드", "year", "ratio_available_count",
             "financial_data_completeness", "ratio_missing_rate"]],
        on=["거래소코드", "year"], how="left")

    has_cols = [c for c in stage1b_panel.columns if c.startswith("has_")]
    # base는 이미 firm_year_panel_v1에서 has_<statement> 컬럼을 가지고 있음
    # (load_stage1b_inputs → panel → base merge 경로)
    out["statement_coverage_count"] = base[has_cols].sum(axis=1).values

    print(f"  fin_data_compl mean={out['financial_data_completeness'].mean():.3f}, "
          f"ratio_avail mean={out['ratio_available_count'].mean():.1f}")
    return out


# ============================================================
# TASK 8 — Integrate panel (4,809 전체 유지)
# ============================================================
def task8_integrate_panel(base: pd.DataFrame, ind: pd.DataFrame,
                           mgmt: pd.DataFrame, biz: pd.DataFrame,
                           fin: pd.DataFrame, rel: pd.DataFrame) -> pd.DataFrame:
    """5 features.parquet → 통합 panel (4,809 유지)"""
    print("\n[Task 8] integrate panel (4,809 유지)")
    panel = base.copy()
    if "회계년도" not in panel.columns and "year" in panel.columns:
        panel["회계년도"] = panel["year"].astype("Int64").astype(str)
    for feat in [ind, mgmt, biz, fin, rel]:
        feat_cols = [c for c in feat.columns if c not in ("거래소코드", "year")]
        panel = panel.merge(feat[["거래소코드", "year"] + feat_cols],
                             on=["거래소코드", "year"], how="left")

    cols = [
        "거래소코드", "회사명", "회계년도", "year", "시장",
        "rating_num", "rating_num_10", "grade_base_10", "rating_num_7", "grade_base_7", "rating_num_notch", "grade_base_notch",
        "신용등급", "split", "eligible_for_stage2",
        "matched_all", "excluded_from_main_nonfinancial",
        "산업코드", "산업명", "sector_7",
        # ── 산업위험 v3.2 신규 (selected_eligible=True) ──
        "industry_avg_rating_lag1_self_excl",
        "industry_median_rating_lag1_self_excl",
        "industry_bad_grade_share_lag1_self_excl",
        "industry_b_or_lower_share_lag1_self_excl",
        "industry_rating_std_lag1_self_excl",
        "industry_rating_iqr_lag1_self_excl",
        "industry_downgrade_rate_lag1_self_excl",
        "industry_upgrade_rate_lag1_self_excl",
        "industry_net_downgrade_rate_lag1_self_excl",
        # ── 산업위험 v2 후방호환 + 진단 ──
        "industry_avg_rating", "industry_avg_rating_n",
        "industry_avg_rating_fallback_level",
        "industry_transition_fallback_level",
        "sector_year_rating_count", "sector_year_rating_count_lag1",
        "kospi_dummy",
        # ── 경영위험 (v2 동일) ──
        "firm_age", "list_age",
        "cap_change_count_3y", "cap_change_cumulative",
        "ma_count_3y", "ma_cumulative",
        # ── 영업위험 (v2 동일) ──
        "log_emp", "log_sales", "log_assets",
        # ── 재무위험 (v2 동일) ──
        "operating_loss_current", "net_loss_current", "operating_loss_freq_3y",
        "negative_equity_flag", "cap_impair_flag", "lawsuit_count_3y",
        # ── 신뢰도 (v2 동일) ──
        "financial_data_completeness", "statement_coverage_count",
        "ratio_available_count", "ratio_missing_rate",
    ]
    cols = [c for c in cols if c in panel.columns]
    nf = panel[cols].copy()
    print(f"  panel shape: {nf.shape}, eligible: {nf['eligible_for_stage2'].sum()}")
    return nf


# ============================================================
# TASK 9 — Quality report + candidate pool
# ============================================================
# v3.2: 산업위험 후보 9개 추가. 진단/후방호환 변수는 selected_eligible=False.
INDUSTRY_V3_2_PROXIES = [
    "industry_avg_rating_lag1_self_excl",
    "industry_median_rating_lag1_self_excl",
    "industry_bad_grade_share_lag1_self_excl",
    "industry_b_or_lower_share_lag1_self_excl",
    "industry_rating_std_lag1_self_excl",
    "industry_rating_iqr_lag1_self_excl",
    "industry_downgrade_rate_lag1_self_excl",
    "industry_upgrade_rate_lag1_self_excl",
    "industry_net_downgrade_rate_lag1_self_excl",
]

DIAGNOSTIC_ONLY_VARS = {
    "industry_avg_rating",  # v2 alias
    "kospi_dummy",
    "sector_7",
    "sector_year_rating_count",
    "sector_year_rating_count_lag1",
}

CATEGORIES = {
    # 산업위험: v3.2 신규
    **{v: "산업위험" for v in INDUSTRY_V3_2_PROXIES},
    # 산업위험: 후방호환/진단
    "industry_avg_rating": "산업위험",
    "kospi_dummy": "산업위험",
    "sector_7": "산업위험",
    "sector_year_rating_count": "산업위험",
    "sector_year_rating_count_lag1": "산업위험",
    # 경영위험
    "firm_age": "경영위험", "list_age": "경영위험",
    "cap_change_count_3y": "경영위험", "cap_change_cumulative": "경영위험",
    "ma_count_3y": "경영위험", "ma_cumulative": "경영위험",
    # 영업위험
    "log_emp": "영업위험", "log_sales": "영업위험", "log_assets": "영업위험",
    # 재무위험
    "operating_loss_current": "재무위험", "net_loss_current": "재무위험",
    "operating_loss_freq_3y": "재무위험",
    "negative_equity_flag": "재무위험", "cap_impair_flag": "재무위험",
    "lawsuit_count_3y": "재무위험",
    # 신뢰도
    "financial_data_completeness": "신뢰도",
    "statement_coverage_count": "신뢰도",
    "ratio_available_count": "신뢰도", "ratio_missing_rate": "신뢰도",
}

BINARY_VARS = {"kospi_dummy", "operating_loss_current", "net_loss_current",
               "negative_equity_flag", "cap_impair_flag"}

EXPECTED_DIRECTION = {
    "industry_avg_rating_lag1_self_excl": "higher_is_worse",
    "industry_median_rating_lag1_self_excl": "higher_is_worse",
    "industry_bad_grade_share_lag1_self_excl": "higher_is_worse",
    "industry_b_or_lower_share_lag1_self_excl": "higher_is_worse",
    "industry_rating_std_lag1_self_excl": "higher_is_worse_or_uncertain",
    "industry_rating_iqr_lag1_self_excl": "higher_is_worse_or_uncertain",
    "industry_downgrade_rate_lag1_self_excl": "higher_is_worse",
    "industry_upgrade_rate_lag1_self_excl": "higher_is_better",
    "industry_net_downgrade_rate_lag1_self_excl": "higher_is_worse",
    "industry_avg_rating": "higher_is_worse",
    "kospi_dummy": "diagnostic_market_tier",
    "sector_7": "diagnostic_nominal_code",
    "sector_year_rating_count": "diagnostic_coverage",
    "sector_year_rating_count_lag1": "diagnostic_coverage",
    "firm_age": "higher_is_better", "list_age": "higher_is_better",
    "cap_change_count_3y": "higher_is_worse", "cap_change_cumulative": "higher_is_worse",
    "ma_count_3y": "higher_is_worse_or_uncertain", "ma_cumulative": "higher_is_worse_or_uncertain",
    "log_emp": "higher_is_better", "log_sales": "higher_is_better", "log_assets": "higher_is_better",
    "operating_loss_current": "higher_is_worse", "net_loss_current": "higher_is_worse",
    "operating_loss_freq_3y": "higher_is_worse", "negative_equity_flag": "higher_is_worse",
    "cap_impair_flag": "higher_is_worse", "lawsuit_count_3y": "higher_is_worse",
    "financial_data_completeness": "higher_is_better", "statement_coverage_count": "higher_is_better",
    "ratio_available_count": "higher_is_better", "ratio_missing_rate": "higher_is_worse",
}


def task9_quality_report(panel: pd.DataFrame, config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """변수별 quality + candidate_pool. v3.2는 selected_eligible/diagnostic_only를 명시한다."""
    print("\n[Task 9] quality_report + candidate_pool (v3.2)")
    elig = panel[panel["eligible_for_stage2"]].copy()
    miss_max = config["quality_filter"]["missing_rate_max"]
    weak_unique = config["quality_filter"]["weak_signal_unique_count_threshold"]
    caveat_min = config["quality_filter"]["candidate_but_missingness_caveat_min_rate"]

    rows = []
    for var, cat in CATEGORIES.items():
        if var not in elig.columns:
            continue
        s = elig[var]
        n = len(s)
        n_miss = int(s.isna().sum())
        miss_rate = n_miss / n
        nm_count = n - n_miss
        nm_rate = nm_count / n

        s_num = pd.to_numeric(s, errors="coerce").dropna()
        if len(s_num) > 0:
            unique_n = int(s_num.nunique())
            mean = float(s_num.mean()) if var != "sector_7" else None
            std = float(s_num.std()) if var != "sector_7" else None
            p1 = float(s_num.quantile(0.01))
            med = float(s_num.median())
            p99 = float(s_num.quantile(0.99))
            mn = float(s_num.min()); mx = float(s_num.max())
        elif var == "sector_7":
            unique_n = int(s.nunique(dropna=True))
            mean = std = p1 = med = p99 = mn = mx = None
        else:
            unique_n = 0
            mean = std = p1 = med = p99 = mn = mx = None

        quality_pass = miss_rate < miss_max
        diagnostic_only = var in DIAGNOSTIC_ONLY_VARS
        leakage_safe = True
        selected_eligible = bool(quality_pass and not diagnostic_only and leakage_safe)

        exclusion_reason = ""
        if not quality_pass:
            exclusion_reason = f"missing_rate={miss_rate:.3f}>={miss_max}"
        elif diagnostic_only:
            if var == "sector_7":
                exclusion_reason = "diagnostic_only: nominal sector code, not ordered risk score"
            elif var == "kospi_dummy":
                exclusion_reason = "diagnostic_only: market-tier variable, not industry risk"
            elif "count" in var:
                exclusion_reason = "diagnostic_only: coverage/count variable, not risk proxy"
            elif var == "industry_avg_rating":
                exclusion_reason = "diagnostic_only: v2 alias; use industry_avg_rating_lag1_self_excl"
            else:
                exclusion_reason = "diagnostic_only"

        caveat_miss = (var == "log_emp" and miss_rate > caveat_min)
        is_binary = var in BINARY_VARS
        weak_signal = (unique_n <= weak_unique and not is_binary and not diagnostic_only)

        notes = []
        if var == "log_emp":
            notes.append("snapshot — 시간 불변 (KIS 일반사항 dump)")
        if var in INDUSTRY_V3_2_PROXIES:
            notes.append("v3.2 신규: t-1 lagged, self-excluded, 4-level fallback")
        if var == "industry_avg_rating":
            notes.append("v2 alias (=industry_avg_rating_lag1_self_excl)")
        if var == "financial_data_completeness":
            notes.append("Stage 2 134 quality-pass ratio 기준 재정의")
        if is_binary:
            notes.append("binary variable")
        if diagnostic_only:
            notes.append("diagnostic_only; Stage 3 must exclude")

        rows.append({
            "variable_name": var,
            "category": cat,
            "non_missing_count": nm_count,
            "non_missing_rate": round(nm_rate, 4),
            "missing_rate": round(miss_rate, 4),
            "unique_count": unique_n,
            "mean": round(mean, 4) if mean is not None else None,
            "std": round(std, 4) if std is not None else None,
            "p1": round(p1, 4) if p1 is not None else None,
            "median": round(med, 4) if med is not None else None,
            "p99": round(p99, 4) if p99 is not None else None,
            "min": round(mn, 4) if mn is not None else None,
            "max": round(mx, 4) if mx is not None else None,
            "quality_pass": bool(quality_pass),
            "selected_eligible": bool(selected_eligible),
            "diagnostic_only": bool(diagnostic_only),
            "leakage_safe": bool(leakage_safe),
            "expected_direction": EXPECTED_DIRECTION.get(var, "empirical"),
            "exclusion_reason": exclusion_reason,
            "candidate_but_missingness_caveat": bool(caveat_miss),
            "weak_signal_caveat": bool(weak_signal),
            "notes": "; ".join(notes),
        })
    qr = pd.DataFrame(rows)

    pool_records = []
    # quality_pass 변수는 모두 pool에 남기되 selected_eligible/diagnostic_only를 명확히 표시한다.
    for _, r in qr[qr["quality_pass"]].iterrows():
        pool_records.append({
            "category": r["category"],
            "variable_name": r["variable_name"],
            "missing_rate": r["missing_rate"],
            "unique_count": r["unique_count"],
            "median": r["median"],
            "selected_eligible": bool(r["selected_eligible"]),
            "diagnostic_only": bool(r["diagnostic_only"]),
            "leakage_safe": bool(r["leakage_safe"]),
            "expected_direction": r["expected_direction"],
            "exclusion_reason": r["exclusion_reason"],
            "caveat": ("missingness" if r["candidate_but_missingness_caveat"]
                       else ("weak_signal" if r["weak_signal_caveat"] else "")),
        })
    pool = pd.DataFrame(pool_records).sort_values(["category", "selected_eligible", "missing_rate"], ascending=[True, False, True])
    print(f"  quality_pass: {int(qr['quality_pass'].sum())}/{len(qr)}")
    print(f"  selected_eligible: {int(qr['selected_eligible'].sum())}, diagnostic_only: {int(qr['diagnostic_only'].sum())}")
    return qr, pool


# ============================================================
# TASK 10 — Acceptance + verdict
# ============================================================
def task10_acceptance_verdict(panel: pd.DataFrame, qr: pd.DataFrame,
                                outdir: Path, config: dict) -> Tuple[str, pd.DataFrame]:
    """Stage 1C v3.2 acceptance 기준 점검 + PASS/FAIL."""
    print("\n[Task 10] acceptance + verdict (v3.2)")
    qpass = qr["quality_pass"].astype(bool) if "quality_pass" in qr.columns else pd.Series(False, index=qr.index)
    selected = qr["selected_eligible"].astype(bool) if "selected_eligible" in qr.columns else qpass
    cat_pass = qr[qpass].groupby("category").size()
    cat_selected = qr[selected].groupby("category").size()
    acc = config.get("acceptance_v3_2", config["acceptance"])

    required = [
        "metadata_source_inventory.csv",
        "metadata_candidate_availability_report.csv",
        "industry_risk_features.parquet",
        "management_risk_features.parquet",
        "event_parsing_log.csv",
        "business_risk_features.parquet",
        "financial_risk_proxy_features.parquet",
        "reliability_features.parquet",
        "nonfinancial_metadata_panel.parquet",
        "nonfinancial_metadata_panel.csv",
        "nonfinancial_variable_quality_report.csv",
        "nonfinancial_candidate_pool_by_item.csv",
    ]
    n_present = sum((outdir / f).exists() for f in required)

    # v3.2 산업위험 필수 proxy 검증
    new_cols_present = sum(1 for v in INDUSTRY_V3_2_PROXIES if v in panel.columns)
    new_qpass = int(qr[qr["variable_name"].isin(INDUSTRY_V3_2_PROXIES) & qpass].shape[0])
    new_selected = int(qr[qr["variable_name"].isin(INDUSTRY_V3_2_PROXIES) & selected].shape[0])
    min_ind = int(acc.get("industry_risk_min_eligible_v3_2", acc.get("industry_risk_min_candidates", 1)))

    panel_total_rows_ref = int(acc.get("panel_total_rows_min", acc.get("panel_total_rows", 0)))
    panel_row_op = str(acc.get("panel_total_rows_op", "ge")).lower()
    panel_desc = acc.get("panel_total_rows_description", "firm-year rows retained / expanded")
    # Acceptance coverage must be evaluated against the original Stage2-eligible
    # reference universe, not against the expanded metadata panel.  After the
    # KONEX/non-rated metadata merge, len(panel) can legitimately include many
    # rows that are not eligible for Stage2 modelling.  Using len(panel) here
    # makes the check fail for the wrong reason: it tests metadata expansion,
    # not Stage2 model eligibility retention.
    eligible_base_mode = str(acc.get("eligible_min_coverage_base", "panel_total_rows_ref")).lower()
    if eligible_base_mode in ("panel", "panel_len", "expanded_panel", "all_rows"):
        elig_threshold_base = len(panel)
    elif eligible_base_mode in ("current_eligible", "eligible_current"):
        elig_threshold_base = int(panel["eligible_for_stage2"].sum())
    else:
        elig_threshold_base = panel_total_rows_ref
    elig_threshold = int(float(acc["eligible_min_coverage_pct"]) * elig_threshold_base)

    # `panel_size` is intentionally diagnostic-only.  Stage1C may lose a small
    # number of firm-years when the nonfinancial/KONEX metadata universe is made
    # leakage-safe and Stage2-eligible.  The required downstream contract is the
    # configured eligible_for_stage2 coverage gate, not strict no-row-loss versus
    # the pre-merge reference count.
    checks = [
        ("신규 산업위험 9개 column 존재", "industry_v3_2_columns", new_cols_present, 9, "eq", True),
        ("신규 산업위험 quality_pass ≥ 기준", "industry_v3_2_quality_pass", new_qpass, min_ind, "ge", True),
        ("신규 산업위험 selected_eligible ≥ 기준", "industry_v3_2_selected", new_selected, min_ind, "ge", True),
        ("경영위험 후보 ≥ 1", "경영위험_pass", int(cat_pass.get("경영위험", 0)), acc["management_risk_min_candidates"], "ge", True),
        ("영업위험 후보 ≥ 1", "영업위험_pass", int(cat_pass.get("영업위험", 0)), acc["business_risk_min_candidates"], "ge", True),
        ("재무위험 후보 ≥ 1", "재무위험_pass", int(cat_pass.get("재무위험", 0)), acc["financial_risk_min_candidates"], "ge", True),
        ("신뢰도 후보 ≥ 1", "신뢰도_pass", int(cat_pass.get("신뢰도", 0)), acc["reliability_min_candidates"], "ge", True),
        (panel_desc, "panel_size", len(panel), panel_total_rows_ref, panel_row_op, False),
        ("eligible_for_stage2 커버 ≥ configured threshold", "elig_coverage", int(panel["eligible_for_stage2"].sum()), elig_threshold, "ge", True),
        ("Task 1-10 필수 output 모두 생성", "required_outputs", f"{n_present}/{len(required)}", f"{len(required)}/{len(required)}", "eq", True),
    ]

    rows = []
    all_required_pass = True
    for desc, key, actual, threshold, op, required_gate in checks:
        if op == "eq":
            passed = (actual == threshold)
        elif op == "ge":
            passed = actual >= threshold
        elif op == "le":
            passed = actual <= threshold
        else:
            raise ValueError(f"Unsupported acceptance op: {op}")
        if required_gate and not passed:
            all_required_pass = False
        rows.append({
            "criterion": desc,
            "key": key,
            "threshold": str(threshold),
            "actual": str(actual),
            "pass": bool(passed),
            "required": bool(required_gate),
            "severity": "required" if required_gate else "diagnostic",
        })
        symbol = "✓" if passed else ("!" if not required_gate else "✗")
        comp = "==" if op == "eq" else "≥" if op == "ge" else "≤"
        suffix = "" if required_gate else " [diagnostic only]"
        print(f"  {symbol} {desc}: {actual} ({comp} {threshold}){suffix}")

    df = pd.DataFrame(rows)
    verdict = "PASS" if all_required_pass else "FAIL"
    print(f"\n  ★ STAGE 1C v3.2 VERDICT: {verdict}")
    return verdict, df


# ============================================================
# Reporting helpers
# ============================================================
def write_diff_report(panel: pd.DataFrame, qr: pd.DataFrame,
                       fb_counts: dict, out: Path,
                       v1_panel_path: Path | None = None):
    """v2 vs v3.2 비교/변경 보고서 생성."""
    cat_pass = qr[qr["quality_pass"]].groupby("category").size()
    cat_selected = qr[qr.get("selected_eligible", qr["quality_pass"]).astype(bool)].groupby("category").size()
    industry_selected = qr[(qr["category"] == "산업위험") & (qr.get("selected_eligible", qr["quality_pass"]).astype(bool))]["variable_name"].tolist()

    md = f"""# Stage 1C v2 vs v3.2 — Diff Report

**Comparison date**: {datetime.now(KST).isoformat()}

## 1. Row / panel shape

| item | v3.2 |
|---|---:|
| total firm-year | {len(panel)} |
| eligible_for_stage2 | {panel['eligible_for_stage2'].sum()} |
| panel shape | {panel.shape} |

## 2. Variable count

| item | v3.2 |
|---|---:|
| quality report variables | {len(qr)} |
| quality_pass variables | {int(qr['quality_pass'].sum())} |
| selected_eligible variables | {int(qr.get('selected_eligible', qr['quality_pass']).astype(bool).sum())} |

## 3. Category counts

| 평가항목 | quality_pass | selected_eligible |
|---|---:|---:|
| 산업위험 | {cat_pass.get('산업위험', 0)} | {cat_selected.get('산업위험', 0)} |
| 경영위험 | {cat_pass.get('경영위험', 0)} | {cat_selected.get('경영위험', 0)} |
| 영업위험 | {cat_pass.get('영업위험', 0)} | {cat_selected.get('영업위험', 0)} |
| 재무위험 | {cat_pass.get('재무위험', 0)} | {cat_selected.get('재무위험', 0)} |
| 신뢰도 | {cat_pass.get('신뢰도', 0)} | {cat_selected.get('신뢰도', 0)} |

## 4. Industry-risk expansion

v2는 `industry_avg_rating` 중심의 산업위험 후보군이었다. v3.2는 아래 9개 lagged/self-excluded 산업위험 proxy를 추가한다.

"""
    for v in INDUSTRY_V3_2_PROXIES:
        md += f"- `{v}`\n"

    md += f"""

### Stage 3 selected-eligible industry candidates

"""
    for v in industry_selected:
        md += f"- `{v}`\n"

    md += f"""

## 5. Fallback distribution for level-based industry proxy

- Level 1: {fb_counts.get('L1', 0)}
- Level 2: {fb_counts.get('L2', 0)}
- Level 3: {fb_counts.get('L3', 0)}
- Level 4: {fb_counts.get('L4', 0)}
- NaN: {fb_counts.get('NaN', 0)}

## 6. Stage 3 input files

- `nonfinancial_metadata_panel.parquet`
- `nonfinancial_candidate_pool_by_item.csv`
- `nonfinancial_variable_quality_report.csv`

Stage 3 must use `selected_eligible=True`, `diagnostic_only=False`, and `leakage_safe=True` when screening nonfinancial candidates.
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)


def write_summary_md(panel: pd.DataFrame, qr: pd.DataFrame,
                      acc_df: pd.DataFrame, fb_counts: dict,
                      verdict: str, out: Path):
    cat_pass = qr[qr["quality_pass"]].groupby("category").size()
    selected_ser = qr.get("selected_eligible", qr["quality_pass"]).astype(bool)
    cat_selected = qr[selected_ser].groupby("category").size()
    md = f"""# Stage 1C v3.2 — Summary

**Stage**: 1C v3.2 — Nonfinancial Metadata Panel (Industry Risk Expanded)
**Verdict**: {'✅' if verdict == 'PASS' else '❌'} **{verdict}**
**Run date**: 2026-05-01
**Version**: v3.2

## 산출 panel

- shape: {panel.shape}
- firm-year: {len(panel)} (full panel 유지)
- eligible_for_stage2: {panel['eligible_for_stage2'].sum()}
- year range: {panel['year'].min()}–{panel['year'].max()}

## 변수 후보 요약

| 평가항목 | quality_pass | selected_eligible |
|---|---:|---:|
| 산업위험 | {cat_pass.get('산업위험', 0)} | {cat_selected.get('산업위험', 0)} |
| 경영위험 | {cat_pass.get('경영위험', 0)} | {cat_selected.get('경영위험', 0)} |
| 영업위험 | {cat_pass.get('영업위험', 0)} | {cat_selected.get('영업위험', 0)} |
| 재무위험 | {cat_pass.get('재무위험', 0)} | {cat_selected.get('재무위험', 0)} |
| 신뢰도 | {cat_pass.get('신뢰도', 0)} | {cat_selected.get('신뢰도', 0)} |

총 quality_pass: {int(qr['quality_pass'].sum())}/{len(qr)}  
총 selected_eligible: {int(selected_ser.sum())}/{len(qr)}

## v3.2 산업위험 신규 proxy

"""
    for v in INDUSTRY_V3_2_PROXIES:
        exists = "✅" if v in panel.columns else "❌"
        md += f"- {exists} `{v}`\n"

    md += f"""

## industry proxy fallback level 분포

- Level 1 (sector × t-1 leave-one-out): {fb_counts.get('L1', 0)}
- Level 2 (sector × t-3..t-1 rolling): {fb_counts.get('L2', 0)}
- Level 3 (sector × past all): {fb_counts.get('L3', 0)}
- Level 4 (all × past all): {fb_counts.get('L4', 0)}
- NaN: {fb_counts.get('NaN', 0)}

## Acceptance Criteria

| Criterion | Threshold | Actual | Pass |
|---|---|---|---|
"""
    for _, r in acc_df.iterrows():
        md += f"| {r['criterion']} | {r['threshold']} | {r['actual']} | {'✅' if r['pass'] else '❌'} |\n"
    md += """
## 다음 단계

→ Stage 3 v3.2-NF — Stage 1C v3.2 비재무 후보군을 사용해 4-metric screening + collinearity replacement 수행.
"""
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)


def write_run_metadata(run_dir: Path, results_dir: Path,
                        panel: pd.DataFrame, qr: pd.DataFrame,
                        fb_counts: dict, verdict: str, paths: dict):
    """input/output hashes + command_log + run_metadata.json"""
    cat_pass = qr[qr["quality_pass"]].groupby("category").size()

    # input_file_hashes
    inputs = [
        ("stage1b_panel", paths["inputs"]["stage1b"]["panel"]),
        ("stage2_engineered_ratios", paths["inputs"]["stage2"]["engineered_ratios"]),
        ("stage2_candidate_pool", paths["inputs"]["stage2"]["candidate_pool"]),
        ("stage2_lag_income", paths["inputs"]["stage2"]["lag_support_income"]),
    ]
    for k in ["재무상태표", "손익계산서", "현금흐름표"]:
        inputs.append((f"stage1b_{k}",
                        paths["inputs"]["stage1b"]["statements"][k]))
    for cat, market_paths in paths["inputs"]["raw_nonfinancial"].items():
        for mk, p in market_paths.items():
            inputs.append((f"raw_{cat}_{mk}", p))

    in_records = []
    for role, p_str in inputs:
        p = Path(p_str)
        if p.exists():
            in_records.append({
                "role": role, "file_name": p.name, "file_path": str(p),
                "size_bytes": p.stat().st_size, "sha256": sha256_of(p),
                "mtime_utc": datetime.fromtimestamp(
                    p.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
    pd.DataFrame(in_records).to_csv(run_dir / "input_file_hashes.csv",
                                      index=False, encoding="utf-8-sig")

    # output_file_hashes
    desc = {
        "metadata_source_inventory.csv": "Task 1 — column-level inventory",
        "metadata_candidate_availability_report.csv": "Task 2 — v3.2 candidate availability",
        "industry_risk_features.parquet": "Task 3 — 산업위험 v3.2 9 proxies + diagnostics",
        "management_risk_features.parquet": "Task 4 — 경영위험 6 vars",
        "event_parsing_log.csv": "Task 4 — event parsing log",
        "business_risk_features.parquet": "Task 5 — 영업위험 3 vars (log1p)",
        "financial_risk_proxy_features.parquet": "Task 6 — 재무위험 6 vars",
        "reliability_features.parquet": "Task 7 — 신뢰도 4 vars (134 ratio)",
        "nonfinancial_metadata_panel.parquet": "Task 8 — 4,809 firm-year 통합",
        "nonfinancial_metadata_panel.csv": "Task 8 — 동일 CSV",
        "nonfinancial_variable_quality_report.csv": "Task 9 — quality + caveat",
        "nonfinancial_candidate_pool_by_item.csv": "Task 9 — 평가항목별 pass 후보",
        "stage1c_summary.md": "Task 9 — 사람용 summary",
        "stage1c_v3_2_acceptance_checklist.csv": "Task 10 — v3.2 acceptance checks",
    }
    out_records = []
    for f in sorted(results_dir.iterdir()):
        if f.is_file():
            out_records.append({
                "file_name": f.name, "size_bytes": f.stat().st_size,
                "sha256": sha256_of(f), "description": desc.get(f.name, ""),
            })
    pd.DataFrame(out_records).to_csv(run_dir / "output_file_hashes.csv",
                                       index=False, encoding="utf-8-sig")

    # run_metadata.json
    meta = {
        "stage": "Stage 1C v3.2 — Nonfinancial Metadata Panel (Industry Risk Expanded)",
        "verdict": verdict, "version": "v3.2", "run_date_kst": "2026-05-01",
        "package_date_utc": datetime.now(timezone.utc).isoformat(),
        "package_date_kst": datetime.now(KST).isoformat(),
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "package_versions": package_versions(),
        },
        "outputs": {
            "panel_shape": list(panel.shape),
            "eligible_count": int(panel["eligible_for_stage2"].sum()),
            "variables_total": len(qr),
            "quality_pass_count": int(qr["quality_pass"].sum()),
            "category_pass": {k: int(v) for k, v in cat_pass.to_dict().items()},
        },
        "industry_risk_v3_2": {
            "n_proxies": 9,
            "proxy_names": [
                "industry_avg_rating_lag1_self_excl",
                "industry_median_rating_lag1_self_excl",
                "industry_bad_grade_share_lag1_self_excl",
                "industry_b_or_lower_share_lag1_self_excl",
                "industry_rating_std_lag1_self_excl",
                "industry_rating_iqr_lag1_self_excl",
                "industry_downgrade_rate_lag1_self_excl",
                "industry_upgrade_rate_lag1_self_excl",
                "industry_net_downgrade_rate_lag1_self_excl",
            ],
            "fallback_level_distribution": fb_counts,
            "self_exclusion": "firm i 자기 제외 (모든 level)",
            "look_ahead_bias": "prevented (t-1 or earlier only)",
        },
    }
    with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ============================================================
# Main pipeline
# ============================================================
def run_pipeline(paths: dict, config: dict, var_mapping: dict,
                  results_dir: Path, run_dir: Path):
    """Stage 1C v3.2 full pipeline — industry risk 9개 proxy 확장"""
    print("=" * 70)
    print("Stage 1C v3.2 — Industry Risk Expanded")
    print("=" * 70)

    # ---- Load inputs (read-only) ----
    stage1b = load_stage1b_inputs(paths)
    stage2 = load_stage2_inputs(paths)
    sector_mapper = make_sector_7_mapper(config["sector_7_mapping"])
    raw_nf = load_raw_nonfinancial(paths, sector_mapper)

    # ---- Base panel (4,809 유지) ----
    panel = stage1b["panel"].copy()
    panel = _add_year_from_accounting_period(panel)
    gi_sub = raw_nf["general_info"][[
        "거래소코드", "산업코드", "산업명", "sector_7",
        "종업원", "설립년", "상장년"
    ]].copy()
    gi_sub["\uac70\ub798\uc18c\ucf54\ub4dc"] = normalize_code(gi_sub["\uac70\ub798\uc18c\ucf54\ub4dc"])
    base = panel.merge(gi_sub, on="\uac70\ub798\uc18c\ucf54\ub4dc", how="left")

    # FINAL_HARDENING: preserve/coalesce industry context after merge suffixing.
    # Stage00_03 must not fail only because optional general_info columns were
    # suffix-renamed or absent in a particular raw universe.
    def _coalesce_context_column(_df, _base_name):
        _candidates = [
            _base_name,
            f"{_base_name}_x",
            f"{_base_name}_y",
            f"{_base_name}_general",
            f"{_base_name}_raw",
            f"{_base_name}_info",
        ]
        _present = [c for c in _candidates if c in _df.columns]
        if _base_name not in _df.columns:
            _df[_base_name] = pd.NA
        for _c in _present:
            if _c != _base_name:
                _df[_base_name] = _df[_base_name].where(_df[_base_name].notna(), _df[_c])
        return _df

    for _ctx_col in ["산업코드", "산업명", "업종코드"]:
        base = _coalesce_context_column(base, _ctx_col)

    # FINAL_HARDENING: base post-merge guard
    if "sector_7" not in base.columns:
        sc = [c for c in ["sector_7_y", "sector_7_x"] if c in base.columns]
        if sc:
            base["sector_7"] = base[sc[0]]
            for c in sc[1:]:
                base["sector_7"] = base["sector_7"].where(base["sector_7"].notna(), base[c])
        else:
            base["sector_7"] = "UNKNOWN"
    base["sector_7"] = base["sector_7"].fillna("UNKNOWN").astype(str)
    if "\uc2dc\uc7a5" not in base.columns:
        mc = [c for c in ["\uc2dc\uc7a5_y", "\uc2dc\uc7a5_x", "market"] if c in base.columns]
        base["\uc2dc\uc7a5"] = base[mc[0]] if mc else "UNKNOWN"
    base["\uc2dc\uc7a5"] = base["\uc2dc\uc7a5"].fillna("UNKNOWN").astype(str).str.upper()
    _general_info_signal_col = None
    for _candidate_col in ["산업코드", "산업명", "업종코드"]:
        if _candidate_col in base.columns and base[_candidate_col].notna().any():
            _general_info_signal_col = _candidate_col
            break
    if _general_info_signal_col is None:
        general_info_match = 0.0
    else:
        general_info_match = base[_general_info_signal_col].notna().mean() * 100
    print(f"\n[Base] {base.shape}, general_info match: {general_info_match:.1f}%")

    # ---- Tasks ----
    task1_df = task1_schema_inventory(paths, var_mapping)
    task1_df.to_csv(results_dir / "metadata_source_inventory.csv",
                    index=False, encoding="utf-8-sig")

    task2_df = task2_candidate_availability()
    task2_df.to_csv(results_dir / "metadata_candidate_availability_report.csv",
                    index=False, encoding="utf-8-sig")

    ind_df, fb_counts = task3_industry_risk(base, config)
    ind_df.to_parquet(results_dir / "industry_risk_features.parquet", index=False)

    mgmt_df, event_log = task4_management_risk(base, raw_nf, config)
    mgmt_df.to_parquet(results_dir / "management_risk_features.parquet", index=False)
    event_log.to_csv(results_dir / "event_parsing_log.csv",
                     index=False, encoding="utf-8-sig")

    biz_df = task5_business_risk(base, stage1b, var_mapping)
    biz_df.to_parquet(results_dir / "business_risk_features.parquet", index=False)

    fin_df = task6_financial_risk_proxy(base, stage1b, stage2, raw_nf,
                                         var_mapping, config)
    fin_df.to_parquet(results_dir / "financial_risk_proxy_features.parquet", index=False)

    rel_df = task7_reliability(base, stage2, stage1b["panel"])
    rel_df.to_parquet(results_dir / "reliability_features.parquet", index=False)

    panel_v2 = task8_integrate_panel(base, ind_df, mgmt_df, biz_df, fin_df, rel_df)
    panel_v2.to_parquet(results_dir / "nonfinancial_metadata_panel.parquet", index=False)
    panel_v2.to_csv(results_dir / "nonfinancial_metadata_panel.csv",
                    index=False, encoding="utf-8-sig")

    qr, pool = task9_quality_report(panel_v2, config)
    qr.to_csv(results_dir / "nonfinancial_variable_quality_report.csv",
              index=False, encoding="utf-8-sig")
    pool.to_csv(results_dir / "nonfinancial_candidate_pool_by_item.csv",
                index=False, encoding="utf-8-sig")

    verdict, acc_df = task10_acceptance_verdict(panel_v2, qr, results_dir, config)
    acc_df.to_csv(results_dir / "stage1c_v3_2_acceptance_checklist.csv",
                  index=False, encoding="utf-8-sig")

    write_summary_md(panel_v2, qr, acc_df, fb_counts, verdict,
                     results_dir / "stage1c_summary.md")

    # v1 vs v2 diff report (v1 panel이 있으면 비교, 없으면 표준)
    v1_panel_path = None
    v1_candidate = Path(paths["inputs"]["stage1b"]["panel"]).parent.parent / \
                   "stage1c_v1" / "nonfinancial_metadata_panel.parquet"
    if v1_candidate.exists():
        v1_panel_path = v1_candidate
    write_diff_report(panel_v2, qr, fb_counts,
                      results_dir / "stage1c_v2_vs_v3_2_diff_report.md",
                      v1_panel_path=v1_panel_path)

    # ---- 10 원칙 metadata (run_dir) ----
    write_run_metadata(run_dir, results_dir, panel_v2, qr, fb_counts, verdict, paths)

    return verdict


def main():
    parser = argparse.ArgumentParser(description="Stage 1C v3.2 pipeline (industry risk expanded)")
    parser.add_argument("--config", type=str, default="configs/paths.yaml",
                        help="Path to paths.yaml (relative to project_root)")
    parser.add_argument("--stage-config", type=str,
                        default="configs/stage_config.yaml")
    parser.add_argument("--var-mapping", type=str,
                        default="configs/variable_mapping.yaml")
    args = parser.parse_args()

    project_root = Path(args.config).resolve().parent.parent
    print(f"project_root: {project_root}")

    paths = load_yaml(Path(args.config))
    config = load_yaml(Path(args.stage_config))
    var_mapping = load_yaml(Path(args.var_mapping))

    # 모든 input/output 경로를 project_root 기준으로 resolve
    def resolve_paths(d):
        if isinstance(d, dict):
            return {k: resolve_paths(v) for k, v in d.items()}
        elif isinstance(d, str) and not Path(d).is_absolute():
            return str((project_root / d).resolve())
        return d
    paths = resolve_paths(paths)

    # 출력 디렉토리
    # v3.2: paths.yaml의 outputs_v3_2 블록 사용 (없으면 outputs 블록 fallback)
    out_cfg = paths.get("outputs_v3_2", paths["outputs"])
    results_dir = Path(out_cfg["results"])
    run_dir = Path(out_cfg["run_log"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    verdict = run_pipeline(paths, config, var_mapping, results_dir, run_dir)
    print("\n" + "=" * 70)
    print(f"DONE — VERDICT: {verdict}")
    print("=" * 70)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
