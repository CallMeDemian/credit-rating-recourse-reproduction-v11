#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage2A raw-data action-source precompute.

Purpose
-------
Build a small, Stage2-only 10D pseudo-action source panel directly from
``data/raw/raw_all/*.xlsx``.  This module intentionally does NOT use Oracle
selected variables, R-code engineered ratio proxies, or Stage0's large
``statement_items_panel.parquet`` as the action source.

The output is a slim firm-year panel consumed by the Stage2 input-split builder:

  data/final_freeze/stage2_candidate_projection/action_sources/
    stage2_raw_action_source_panel.parquet
    stage2_raw_action_source_coverage.csv
    stage2_raw_action_source_report.json
    stage2_raw_concept_values_panel.parquet

Design rules
------------
* Read raw Excel files column-selectively: key columns + matched raw item cols.
* Use KIS item-code exact matches first; raw Korean item-name fallback is allowed
  only as a raw-item alias, not as an engineered-ratio proxy.
* Never silently replace missing dimensions with R-code proxies.
* If any 10D action dimension has no raw source signal, fail fast.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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

ACTION_MIN_OBSERVED_RATE = {
    # Broad raw financial statements include many firms for which a specific
    # balance-sheet lever is structurally absent.  These are hard gates for
    # dead dimensions, not arbitrary performance thresholds.  Bond and long-debt
    # coverage are expected to be lower than income-statement ratios.
    "ppe_pct": 0.20,
    "inv_turnover_chg": 0.20,
    "ar_turnover_chg": 0.20,
    "ap_turnover_chg": 0.20,
    "short_debt_pct": 0.10,
    "long_debt_pct": 0.10,
    "bond_pct": 0.05,
    "revenue_growth": 0.20,
    "cogs_ratio_chg": 0.20,
    "sga_ratio_chg": 0.20,
}

# Raw statement concepts required to build the 10D action interface.  Codes are
# audited from NICE/KIS raw workbook headers.  Name aliases are only a raw-item
# fallback when a local export uses the same account under a nearby column code.
DEFAULT_CONCEPT_RULES: dict[str, dict[str, Any]] = {
    "total_assets": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A100000000"],
        "name_all": [["자산"]],
    },
    "current_assets": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A111038600"],
        "name_all": [["유동자산"]],
    },
    "cash": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A111050000", "U01A111050200", "U01A111050800", "U01A111052700"],
        "name_all": [["현금및현금성자산"], ["현금성자산"], ["현금"]],
    },
    "total_liabilities": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A800000000"],
        "name_all": [["부채"]],
    },
    "current_liabilities": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A811026000"],
        "name_all": [["유동부채"]],
    },
    "total_equity": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A600000000"],
        "name_all": [["자본"]],
    },
    "ppe": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A111000000", "U01A111051200"],
        "name_all": [["유형자산"]],
    },
    "short_debt": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A811026700", "U01A811027200", "U01A811037700"],
        "name_all": [["단기차입금"]],
    },
    "long_debt": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A811012800", "U01A811013300", "U01A811036800"],
        "name_all": [["장기차입금"]],
    },
    "bond": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A811000000", "U01A811010500"],
        "name_all": [["사채"]],
    },
    "inventory": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A111038700", "U01A111052200"],
        "name_all": [["재고자산"]],
    },
    "accounts_receivable": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A111045500", "U01A111045400", "U01A111052500"],
        "name_all": [["매출채권"], ["매출채권", "유동채권"]],
    },
    "accounts_payable": {
        "statement_types": ["재무상태표"],
        "codes": ["U01A811030800", "U01A811030700"],
        "name_all": [["매입채무"], ["매입채무", "유동채무"]],
    },
    "revenue": {
        "statement_types": ["손익계산서"],
        "codes": ["U01B100000000"],
        "name_all": [["매출액"]],
    },
    "cogs": {
        "statement_types": ["손익계산서"],
        "codes": ["U01B200000000"],
        "name_all": [["매출원가"]],
    },
    "sga": {
        "statement_types": ["손익계산서"],
        "codes": ["U01B350000000"],
        "name_all": [["판매비", "관리비"], ["판관비"]],
    },
    "operating_income": {
        "statement_types": ["손익계산서"],
        "codes": ["U01B400000000", "U01B430000000"],
        "name_all": [["영업이익"], ["영업", "손익"]],
    },
    "financial_cost": {
        "statement_types": ["손익계산서"],
        "codes": ["U01B550000000", "U01B500020000", "U01B500020100"],
        "name_all": [["금융원가"], ["이자비용"]],
    },
    "net_income": {
        "statement_types": ["손익계산서"],
        "codes": ["U01B840000000", "U01B800000000"],
        "name_all": [["당기순이익"], ["순이익"]],
    },
}

KEY_CANDIDATES = {
    "firm_id": ["거래소코드", "종목코드", "회사코드", "firm_id"],
    "firm_name": ["회사명", "기업명", "회사명칭"],
    "fiscal_year": ["회계년도", "결산년도", "fiscal_year", "year"],
}

BAD_TEXT_TOKENS = ["�", "ï¿½", "ì", "ê", "í", "Ã"]

# Broad AVS/raw-account feature support for Stage3 encoder.  The prior Stage2A
# implementation emitted only the 10 core action source accounts, which made
# Stage3 collapse to a 10-feature encoder.  These settings materialize a broad
# deterministic U-code account namespace as raw__avs__* with matching
# next__raw__avs__* columns for ACD pretraining.
AVS_CODE_RE = re.compile(r"^U01[AB][A-Z0-9]*$", re.IGNORECASE)
AVS_MAX_COLUMNS_PER_FILE = 180
AVS_MAX_TOTAL_FEATURES = 160


def configure_utf8_stdio() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(json_safe(k)): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        return x if np.isfinite(x) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _norm_text(s: Any) -> str:
    return re.sub(r"[\s_()/·ㆍ\-\[\]{}*]+", "", str(s).lower())


def _extract_code(col: str) -> str | None:
    m = re.search(r"\[([^\]]+)\]", str(col))
    if not m:
        return None
    return m.group(1).strip()


def _safe_avs_feature_name(col: str) -> str | None:
    """Stable raw AVS feature name from a NICE/KIS U-code column header."""
    code = _extract_code(col)
    if not code or not AVS_CODE_RE.match(code):
        return None
    label = re.sub(r"\[[^\]]+\]", "", str(col)).strip()
    label = re.sub(r"[^0-9A-Za-z가-힣]+", "_", label).strip("_")
    if len(label) > 36:
        label = label[:36].rstrip("_")
    return f"raw__avs__{code}" + (f"__{label}" if label else "")


def assert_no_mojibake_columns(cols: list[Any], where: str) -> None:
    bad = [str(c) for c in cols if any(tok in str(c) for tok in BAD_TEXT_TOKENS)]
    if bad:
        raise ValueError(f"Mojibake detected in {where}: {bad[:20]}")


def _find_key_col(columns: list[str], role: str) -> str | None:
    norm = {_norm_text(c): c for c in columns}
    for name in KEY_CANDIDATES[role]:
        if name in columns:
            return name
        hit = norm.get(_norm_text(name))
        if hit:
            return hit
    return None


def _parse_firm_id(x: Any) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if not s:
        return None
    if re.fullmatch(r"\d+(\.0)?", s):
        s = str(int(float(s)))
    digits = re.sub(r"\D", "", s)
    if not digits:
        return s
    return digits.zfill(6)


def _parse_year(x: Any) -> int | None:
    if pd.isna(x):
        return None
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)) and np.isfinite(x):
        return int(x)
    s = str(x)
    m = re.search(r"(19\d{2}|20\d{2})", s)
    return int(m.group(1)) if m else None


def _statement_type(path: Path) -> str | None:
    name = path.name
    if "재무상태표" in name:
        return "재무상태표"
    if "손익계산서" in name:
        return "손익계산서"
    return None


def load_user_action_map(root: Path) -> dict[str, dict[str, Any]]:
    """Load optional raw-action concept-rule overrides safely.

    Stage2A accepts only concept-rule dictionaries with statement_types/codes/name_all.
    Older action-source binding configs may contain string-valued entries; those
    are incompatible with this loader and are skipped with a warning instead of
    crashing or being treated as raw concept rules. DEFAULT_CONCEPT_RULES remain
    active, and missing 10D source coverage is still enforced later.
    """
    candidates = [
        root / "data/final_freeze/configs/stage2_action_source_map.json",
        root / "data/final_freeze/configs/stage2_raw_action_source_map.json",
    ]
    rules = json.loads(json.dumps(DEFAULT_CONCEPT_RULES, ensure_ascii=False))

    def _is_rule_dict(obj: Any) -> bool:
        return isinstance(obj, dict) and any(k in obj for k in ("statement_types", "codes", "name_all"))

    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if isinstance(value, set):
            return sorted(value, key=lambda x: str(x))
        return [value]

    for p in candidates:
        if not p.exists():
            continue
        payload = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            print(f"[Stage2A][WARN] Ignoring non-dict action source config: {p}", file=sys.stderr)
            continue
        extra = payload.get("concept_rules", payload)
        if not isinstance(extra, dict):
            print(f"[Stage2A][WARN] Ignoring action source config with non-dict concept_rules: {p}", file=sys.stderr)
            continue

        skipped: list[str] = []
        for concept, cfg in extra.items():
            if not _is_rule_dict(cfg):
                skipped.append(str(concept))
                continue
            base = rules.setdefault(str(concept), {"statement_types": [], "codes": [], "name_all": []})
            for key in ["statement_types", "codes"]:
                vals = [str(v).strip() for v in _as_list(cfg.get(key, [])) if str(v).strip()]
                base[key] = list(dict.fromkeys(list(base.get(key, [])) + vals))

            vals = _as_list(cfg.get("name_all", []))
            if vals and all(isinstance(v, str) for v in vals):
                vals = [vals]
            groups: list[list[str]] = []
            for group in vals:
                if isinstance(group, (list, tuple, set)):
                    g = [str(v).strip() for v in group if str(v).strip()]
                else:
                    g = [str(group).strip()] if str(group).strip() else []
                if g:
                    groups.append(g)
            seen = set()
            merged: list[list[str]] = []
            for group in list(base.get("name_all", [])) + groups:
                key_tuple = tuple(str(v) for v in group)
                if key_tuple not in seen:
                    seen.add(key_tuple)
                    merged.append(list(key_tuple))
            base["name_all"] = merged
        if skipped:
            print(
                f"[Stage2A][WARN] Ignored {len(skipped)} incompatible entries from {p.name}; "
                f"expected concept-rule dict values with statement_types/codes/name_all. "
                f"Examples: {skipped[:10]}",
                file=sys.stderr,
            )
    return rules


def _concept_matches_column(col: str, concept: str, rule: dict[str, Any]) -> tuple[bool, str, int]:
    code = _extract_code(col)
    codes = [str(x).strip() for x in rule.get("codes", []) if str(x).strip()]
    if code and code in codes:
        return True, f"code:{code}", codes.index(code)
    ncol = _norm_text(col)
    for i, group in enumerate(rule.get("name_all", []) or []):
        if all(_norm_text(tok) in ncol for tok in group):
            return True, "raw_name:" + "+".join(map(str, group)), 1000 + i
    return False, "", 999999


def _select_raw_columns(path: Path, rules: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    stype = _statement_type(path)
    if stype is None:
        return [], {}, {"skipped_reason": "not_balance_or_income_statement"}
    header = pd.read_excel(path, sheet_name=0, nrows=0, engine="openpyxl")
    cols = [str(c) for c in header.columns]
    assert_no_mojibake_columns(cols, f"raw Excel header {path}")
    firm_col = _find_key_col(cols, "firm_id")
    year_col = _find_key_col(cols, "fiscal_year")
    name_col = _find_key_col(cols, "firm_name")
    if firm_col is None or year_col is None:
        return [], {}, {"skipped_reason": "missing_key_columns", "firm_col": firm_col, "year_col": year_col}
    matches: dict[str, list[dict[str, Any]]] = {}
    for concept, rule in rules.items():
        if stype not in set(rule.get("statement_types", []) or []):
            continue
        hits = []
        for col in cols:
            ok, how, priority = _concept_matches_column(col, concept, rule)
            if ok:
                hits.append({"concept": concept, "column": col, "match": how, "priority": priority})
        if hits:
            matches[concept] = sorted(hits, key=lambda r: (int(r["priority"]), len(str(r["column"])), str(r["column"])))[:3]
    # Broad encoder feature candidates: deterministic U01A/U01B raw account columns.
    # Exclude key columns and already-selected concept columns.  We cap per file
    # to keep Excel reads tractable while preserving broad account coverage.
    concept_selected = {h["column"] for hits in matches.values() for h in hits}
    avs_hits = []
    for col in cols:
        if col in {firm_col, year_col, name_col} or col in concept_selected:
            continue
        fname = _safe_avs_feature_name(col)
        if fname:
            avs_hits.append({"column": col, "feature": fname, "code": _extract_code(col)})
    avs_hits = sorted(avs_hits, key=lambda r: (str(r["code"]), str(r["column"])))[:AVS_MAX_COLUMNS_PER_FILE]

    selected = [firm_col, year_col]
    if name_col:
        selected.append(name_col)
    for hits in matches.values():
        for h in hits:
            selected.append(h["column"])
    for h in avs_hits:
        selected.append(h["column"])
    selected = list(dict.fromkeys(selected))
    meta_out = {"statement_type": stype, "firm_col": firm_col, "year_col": year_col, "name_col": name_col, "avs_hits": avs_hits}
    return selected, matches, meta_out


def _first_non_null_frame(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    vals = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in cols if c in df.columns})
    if vals.empty:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return vals.bfill(axis=1).iloc[:, 0]


def read_raw_concept_values(root: Path, raw_all_dir: Path, rules: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = sorted(raw_all_dir.rglob("*.xlsx"))
    frames: list[pd.DataFrame] = []
    file_reports: list[dict[str, Any]] = []
    concept_sources: dict[str, list[dict[str, Any]]] = {k: [] for k in rules}
    for path in files:
        stype = _statement_type(path)
        if stype is None:
            continue
        try:
            selected, matches, meta = _select_raw_columns(path, rules)
            if not selected or not matches:
                file_reports.append({"file": str(path.relative_to(root)), "selected": False, **meta})
                continue
            print(f"[Stage2A raw] {path.name}: reading {len(selected)} selected columns ({sorted(matches)})", flush=True)
            df = pd.read_excel(path, sheet_name=0, usecols=selected, engine="openpyxl")
            assert_no_mojibake_columns([str(c) for c in df.columns], f"raw Excel selected columns {path}")
            firm_col = meta["firm_col"]
            year_col = meta["year_col"]
            name_col = meta.get("name_col")
            out = pd.DataFrame({
                "firm_id": df[firm_col].map(_parse_firm_id),
                "fiscal_year": df[year_col].map(_parse_year),
            })
            if name_col and name_col in df.columns:
                out["회사명"] = df[name_col].astype(str)
            for concept, hits in matches.items():
                cols = [h["column"] for h in hits]
                out[concept] = _first_non_null_frame(df, cols)
                nonnull = int(out[concept].notna().sum())
                concept_sources.setdefault(concept, []).append({
                    "file": str(path.relative_to(root)),
                    "statement_type": stype,
                    "columns": cols,
                    "matches": [h["match"] for h in hits],
                    "non_null": nonnull,
                })
            # Broad raw AVS account features for Stage3 encoder.  These are raw
            # financial statement accounts, not Oracle-selected ratios and not
            # engineered proxies.
            for h in meta.get("avs_hits", []) or []:
                src = h.get("column")
                dst = h.get("feature")
                if src in df.columns and dst:
                    out[dst] = pd.to_numeric(df[src], errors="coerce")
            out = out[out["firm_id"].notna() & out["fiscal_year"].notna()].copy()
            if len(out):
                frames.append(out)
            file_reports.append({
                "file": str(path.relative_to(root)),
                "selected": True,
                "statement_type": stype,
                "selected_columns": selected,
                "concepts": sorted(matches),
                "rows": int(len(df)),
                "usable_key_rows": int(len(out)),
            })
        except Exception as exc:
            file_reports.append({"file": str(path.relative_to(root)), "selected": False, "error": repr(exc)})
            raise
    if not frames:
        raise ValueError(f"No raw financial statement Excel files yielded Stage2 action concepts under {raw_all_dir}")
    raw = pd.concat(frames, ignore_index=True, sort=False)
    concept_cols = [c for c in rules if c in raw.columns]
    avs_cols_all = sorted([c for c in raw.columns if str(c).startswith("raw__avs__")])
    # Select the broad AVS columns with the most support, capped to keep Stage3
    # reproducible and bounded.
    avs_support = {c: int(raw[c].notna().sum()) for c in avs_cols_all}
    avs_cols = [c for c, _ in sorted(avs_support.items(), key=lambda kv: (-kv[1], kv[0]))[:AVS_MAX_TOTAL_FEATURES]]
    # Collapse split files/markets to one row per firm-year, taking the first non-null raw account value by concept/feature.
    agg_spec = {c: "first" for c in concept_cols + avs_cols}
    if "회사명" in raw.columns:
        agg_spec["회사명"] = "first"
    panel = raw.sort_values(["firm_id", "fiscal_year"]).groupby(["firm_id", "fiscal_year"], as_index=False).agg(agg_spec)
    report = {
        "raw_all_dir": str(raw_all_dir),
        "files_scanned": len(files),
        "files_with_action_concepts": int(sum(1 for r in file_reports if r.get("selected"))),
        "file_reports": file_reports,
        "concept_sources": concept_sources,
        "concept_non_null_counts": {c: int(panel[c].notna().sum()) for c in concept_cols},
        "broad_avs_feature_policy": {
            "enabled": True,
            "max_columns_per_file": AVS_MAX_COLUMNS_PER_FILE,
            "max_total_features": AVS_MAX_TOTAL_FEATURES,
            "selected_feature_count": int(len(avs_cols)),
            "selected_feature_sample": avs_cols[:30],
        },
        "broad_avs_non_null_counts": {c: int(panel[c].notna().sum()) for c in avs_cols[:200]},
        "concept_panel_rows": int(len(panel)),
    }
    required_concepts = sorted(DEFAULT_CONCEPT_RULES)
    missing = [c for c in required_concepts if c not in panel.columns or int(panel[c].notna().sum()) == 0]
    if missing:
        raise ValueError({
            "message": "Raw data Stage2 action source precompute could not resolve required raw statement concepts.",
            "missing_concepts": missing,
            "config_hint": "Add exact raw item codes/names to data/final_freeze/configs/stage2_action_source_map.json; proxy fallback is forbidden.",
        })
    return panel, report


def _clip(series: pd.Series, dim: str) -> pd.Series:
    lo, hi = BOUNDS[dim]
    return pd.to_numeric(series, errors="coerce").clip(lo, hi)


def _frac_change(cur: pd.Series, nxt: pd.Series, dim: str) -> tuple[pd.Series, pd.Series]:
    a = pd.to_numeric(cur, errors="coerce")
    b = pd.to_numeric(nxt, errors="coerce")
    den = a.abs()
    # Direct raw amount action with zero-base guard: if current is zero but next is
    # non-zero, scale by next amount so issuance/disposal is still observed.
    den = den.where(den > 1e-9, b.abs())
    val = (b - a) / den
    obs = a.notna() & b.notna() & den.notna() & (den > 1e-9)
    return _clip(val, dim).where(obs, 0.0).fillna(0.0), obs.fillna(False)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    n = pd.to_numeric(num, errors="coerce")
    d = pd.to_numeric(den, errors="coerce")
    return n / d.where(d.abs() > 1e-9)


def _ratio_delta(cur_num: pd.Series, cur_den: pd.Series, nxt_num: pd.Series, nxt_den: pd.Series, dim: str) -> tuple[pd.Series, pd.Series]:
    a = _ratio(cur_num, cur_den)
    b = _ratio(nxt_num, nxt_den)
    obs = a.notna() & b.notna()
    val = b - a
    return _clip(val, dim).where(obs, 0.0).fillna(0.0), obs.fillna(False)


def compute_action_panel(concepts: pd.DataFrame, min_observed_rate: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    cur = concepts.copy()
    nxt = concepts.copy()
    nxt["fiscal_year"] = pd.to_numeric(nxt["fiscal_year"], errors="coerce").astype("Int64") - 1
    merged = cur.merge(nxt, on=["firm_id", "fiscal_year"], how="left", suffixes=("", "__next"))
    # Keep only firm-years with at least one next raw concept; Stage2 transition builder will apply its own universe filter.
    concept_next_cols = [f"{c}__next" for c in DEFAULT_CONCEPT_RULES if f"{c}__next" in merged.columns]
    action = merged[["firm_id", "fiscal_year"]].copy()
    if "회사명" in merged.columns:
        action["회사명"] = merged["회사명"]

    action["raw__total_assets"] = pd.to_numeric(merged.get("total_assets"), errors="coerce")
    action["raw__current_assets"] = pd.to_numeric(merged.get("current_assets"), errors="coerce")
    action["raw__cash"] = pd.to_numeric(merged.get("cash"), errors="coerce")
    action["raw__total_liabilities"] = pd.to_numeric(merged.get("total_liabilities"), errors="coerce")
    action["raw__current_liabilities"] = pd.to_numeric(merged.get("current_liabilities"), errors="coerce")
    action["raw__total_equity"] = pd.to_numeric(merged.get("total_equity"), errors="coerce")
    action["raw__ppe"] = pd.to_numeric(merged["ppe"], errors="coerce")
    action["raw__short_debt"] = pd.to_numeric(merged["short_debt"], errors="coerce")
    action["raw__short_term_debt"] = action["raw__short_debt"]
    action["raw__long_debt"] = pd.to_numeric(merged["long_debt"], errors="coerce")
    action["raw__long_term_debt"] = action["raw__long_debt"]
    action["raw__bond"] = pd.to_numeric(merged["bond"], errors="coerce")
    action["raw__bonds"] = action["raw__bond"]
    action["raw__revenue"] = pd.to_numeric(merged["revenue"], errors="coerce")
    action["raw__cogs"] = pd.to_numeric(merged["cogs"], errors="coerce")
    action["raw__sga"] = pd.to_numeric(merged["sga"], errors="coerce")
    action["raw__operating_income"] = pd.to_numeric(merged["operating_income"], errors="coerce") if "operating_income" in merged.columns else np.nan
    action["raw__financial_cost"] = pd.to_numeric(merged["financial_cost"], errors="coerce") if "financial_cost" in merged.columns else np.nan
    action["raw__net_income"] = pd.to_numeric(merged["net_income"], errors="coerce") if "net_income" in merged.columns else np.nan
    action["raw__inventory"] = pd.to_numeric(merged["inventory"], errors="coerce")
    action["raw__accounts_receivable"] = pd.to_numeric(merged["accounts_receivable"], errors="coerce")
    action["raw__receivables"] = action["raw__accounts_receivable"]
    action["raw__accounts_payable"] = pd.to_numeric(merged["accounts_payable"], errors="coerce")
    action["raw__payables"] = action["raw__accounts_payable"]

    # Broad SSL/BC support: keep next-state raw features so Stage3 encoder can
    # pretrain on all raw financial transitions without depending on rated
    # Oracle bridge rows. Stage3 expects next-state columns as next__<feature>.
    next_raw_map = {
        "next__raw__total_assets": "total_assets__next",
        "next__raw__current_assets": "current_assets__next",
        "next__raw__cash": "cash__next",
        "next__raw__total_liabilities": "total_liabilities__next",
        "next__raw__current_liabilities": "current_liabilities__next",
        "next__raw__total_equity": "total_equity__next",
        "next__raw__ppe": "ppe__next",
        "next__raw__short_debt": "short_debt__next",
        "next__raw__short_term_debt": "short_debt__next",
        "next__raw__long_debt": "long_debt__next",
        "next__raw__long_term_debt": "long_debt__next",
        "next__raw__bond": "bond__next",
        "next__raw__bonds": "bond__next",
        "next__raw__revenue": "revenue__next",
        "next__raw__cogs": "cogs__next",
        "next__raw__sga": "sga__next",
        "next__raw__operating_income": "operating_income__next",
        "next__raw__financial_cost": "financial_cost__next",
        "next__raw__net_income": "net_income__next",
        "next__raw__inventory": "inventory__next",
        "next__raw__accounts_receivable": "accounts_receivable__next",
        "next__raw__receivables": "accounts_receivable__next",
        "next__raw__accounts_payable": "accounts_payable__next",
        "next__raw__payables": "accounts_payable__next",
    }
    for out_col, src_col in next_raw_map.items():
        action[out_col] = pd.to_numeric(merged[src_col], errors="coerce") if src_col in merged.columns else np.nan
    action["has_next_raw_any"] = action[list(next_raw_map)].notna().any(axis=1)
    action["has_next_raw_all_core"] = action[[
        "next__raw__ppe", "next__raw__short_debt", "next__raw__long_debt",
        "next__raw__bond", "next__raw__revenue", "next__raw__cogs",
        "next__raw__inventory", "next__raw__accounts_receivable",
        "next__raw__accounts_payable"
    ]].notna().all(axis=1)

    # Carry broad raw AVS features and their next-state companions for Stage3.
    avs_cols = sorted([c for c in concepts.columns if str(c).startswith("raw__avs__")])
    for c in avs_cols:
        action[c] = pd.to_numeric(merged[c], errors="coerce") if c in merged.columns else np.nan
        nc = f"{c}__next"
        action[f"next__{c}"] = pd.to_numeric(merged[nc], errors="coerce") if nc in merged.columns else np.nan
    if avs_cols:
        action["has_next_raw_avs_any"] = action[[f"next__{c}" for c in avs_cols]].notna().any(axis=1)
    else:
        action["has_next_raw_avs_any"] = False

    formulas: dict[str, str] = {}
    action["action__ppe_pct"], action["action_observed__ppe_pct"] = _frac_change(merged["ppe"], merged["ppe__next"], "ppe_pct")
    formulas["ppe_pct"] = "(유형자산_t+1 - 유형자산_t) / max(abs(유형자산_t), abs(유형자산_t+1 if zero-base))"
    action["action__short_debt_pct"], action["action_observed__short_debt_pct"] = _frac_change(merged["short_debt"], merged["short_debt__next"], "short_debt_pct")
    formulas["short_debt_pct"] = "(단기차입금_t+1 - 단기차입금_t) / max(abs(단기차입금_t), abs(단기차입금_t+1 if zero-base))"
    action["action__long_debt_pct"], action["action_observed__long_debt_pct"] = _frac_change(merged["long_debt"], merged["long_debt__next"], "long_debt_pct")
    formulas["long_debt_pct"] = "(장기차입금_t+1 - 장기차입금_t) / max(abs(장기차입금_t), abs(장기차입금_t+1 if zero-base))"
    action["action__bond_pct"], action["action_observed__bond_pct"] = _frac_change(merged["bond"], merged["bond__next"], "bond_pct")
    formulas["bond_pct"] = "(사채_t+1 - 사채_t) / max(abs(사채_t), abs(사채_t+1 if zero-base))"
    action["action__revenue_growth"], action["action_observed__revenue_growth"] = _frac_change(merged["revenue"], merged["revenue__next"], "revenue_growth")
    formulas["revenue_growth"] = "(매출액_t+1 - 매출액_t) / max(abs(매출액_t), abs(매출액_t+1 if zero-base))"

    action["action__inv_turnover_chg"], action["action_observed__inv_turnover_chg"] = _ratio_delta(merged["cogs"], merged["inventory"], merged["cogs__next"], merged["inventory__next"], "inv_turnover_chg")
    formulas["inv_turnover_chg"] = "(매출원가/재고자산)_t+1 - (매출원가/재고자산)_t"
    action["action__ar_turnover_chg"], action["action_observed__ar_turnover_chg"] = _ratio_delta(merged["revenue"], merged["accounts_receivable"], merged["revenue__next"], merged["accounts_receivable__next"], "ar_turnover_chg")
    formulas["ar_turnover_chg"] = "(매출액/매출채권)_t+1 - (매출액/매출채권)_t"
    action["action__ap_turnover_chg"], action["action_observed__ap_turnover_chg"] = _ratio_delta(merged["cogs"], merged["accounts_payable"], merged["cogs__next"], merged["accounts_payable__next"], "ap_turnover_chg")
    formulas["ap_turnover_chg"] = "(매출원가/매입채무)_t+1 - (매출원가/매입채무)_t"
    action["action__cogs_ratio_chg"], action["action_observed__cogs_ratio_chg"] = _ratio_delta(merged["cogs"], merged["revenue"], merged["cogs__next"], merged["revenue__next"], "cogs_ratio_chg")
    formulas["cogs_ratio_chg"] = "(매출원가/매출액)_t+1 - (매출원가/매출액)_t"
    action["action__sga_ratio_chg"], action["action_observed__sga_ratio_chg"] = _ratio_delta(merged["sga"], merged["revenue"], merged["sga__next"], merged["revenue__next"], "sga_ratio_chg")
    formulas["sga_ratio_chg"] = "(판관비/매출액)_t+1 - (판관비/매출액)_t"

    coverage_rows = []
    n = len(action)
    for dim in ACTION_DIMS:
        col = f"action__{dim}"
        obs_col = f"action_observed__{dim}"
        obs = action[obs_col].astype(bool)
        coverage_rows.append({
            "action_dim": dim,
            "action_column": col,
            "source_method": "direct_data_raw_all_xlsx_selected_columns",
            "formula": formulas[dim],
            "observed_count": int(obs.sum()),
            "observed_rate": float(obs.mean()) if n else 0.0,
            "nonzero_rate": float((pd.to_numeric(action[col], errors="coerce").fillna(0.0).abs() > 1e-12).mean()) if n else 0.0,
            "mean": float(pd.to_numeric(action[col], errors="coerce").mean()) if n else None,
            "std": float(pd.to_numeric(action[col], errors="coerce").std()) if n > 1 else None,
            "min": float(pd.to_numeric(action[col], errors="coerce").min()) if n else None,
            "max": float(pd.to_numeric(action[col], errors="coerce").max()) if n else None,
        })
    coverage = pd.DataFrame(coverage_rows)
    # Fail if a dimension is effectively dead.  Use action-specific gates because
    # direct raw accounting items have structural coverage differences: e.g., bond
    # balance exists for far fewer listed firms than revenue or SG&A.  This is still
    # stricter than the old 1% global gate and avoids silent zero-imputation of a
    # whole action dimension.
    failed = []
    for _, r in coverage.iterrows():
        dim = str(r["action_dim"])
        rate = float(r["observed_rate"])
        threshold = max(float(min_observed_rate), float(ACTION_MIN_OBSERVED_RATE.get(dim, min_observed_rate)))
        if rate < threshold:
            failed.append({
                "action_dim": dim,
                "observed_rate": rate,
                "observed_count": int(r["observed_count"]),
                "min_required_observed_rate": threshold,
            })
    if failed:
        raise ValueError({
            "message": "Raw-data Stage2 action source coverage below direct-item threshold; proxy fallback is forbidden.",
            "global_min_observed_rate": min_observed_rate,
            "action_min_observed_rate": ACTION_MIN_OBSERVED_RATE,
            "failed_dimensions": failed,
            "remedy": "Add/adjust exact raw item codes in data/final_freeze/configs/stage2_action_source_map.json or inspect raw exports.",
        })
    report = {
        "action_rows": int(len(action)),
        "min_observed_rate": float(min_observed_rate),
        "action_min_observed_rate": ACTION_MIN_OBSERVED_RATE,
        "formula_by_action_dim": formulas,
        "zero_base_policy": "for raw amount changes, use abs(next) denominator only when abs(current) is zero; no R-code proxy fallback",
        "source_policy": "direct data/raw/raw_all Excel selected columns only; engineered ratio proxy fallback forbidden",
        "broad_ssl_bc_ready": True,
        "next_state_raw_columns_included": True,
        "broad_avs_feature_count": int(len([c for c in action.columns if str(c).startswith("raw__avs__")])),
        "broad_avs_next_feature_count": int(len([c for c in action.columns if str(c).startswith("next__raw__avs__")])),
    }
    return action, coverage, report


def main() -> int:
    configure_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--raw-all-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--min-observed-rate", type=float, default=0.05)
    args = ap.parse_args()

    root = Path(args.project_root).resolve()
    raw_all = Path(args.raw_all_dir).resolve() if args.raw_all_dir else root / "data/raw/raw_all"
    out_dir = Path(args.out_dir).resolve() if args.out_dir else root / "data/final_freeze/stage2_candidate_projection/action_sources"
    if not raw_all.exists():
        raise FileNotFoundError(f"Missing raw_all directory: {raw_all}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rules = load_user_action_map(root)
    concept_panel, read_report = read_raw_concept_values(root, raw_all, rules)
    action_panel, coverage, action_report = compute_action_panel(concept_panel, args.min_observed_rate)

    concept_out = out_dir / "stage2_raw_concept_values_panel.parquet"
    action_out = out_dir / "stage2_raw_action_source_panel.parquet"
    coverage_out = out_dir / "stage2_raw_action_source_coverage.csv"
    report_out = out_dir / "stage2_raw_action_source_report.json"

    concept_panel.to_parquet(concept_out, index=False)
    action_panel.to_parquet(action_out, index=False)
    coverage.to_csv(coverage_out, index=False, encoding="utf-8-sig")
    report = {
        "stage_name": "stage2_raw_action_source_precompute",
        "contract_version": "stage2_raw_data_direct_action_source_v1",
        "status": "PASS",
        "created_utc": now_utc(),
        "input_paths": {"raw_all_dir": str(raw_all)},
        "output_paths": {
            "concept_panel": str(concept_out),
            "action_source_panel": str(action_out),
            "coverage_csv": str(coverage_out),
            "report_json": str(report_out),
        },
        "direct_raw_data_source": True,
        "stage0_statement_items_used": False,
        "engineered_ratio_proxy_used": False,
        "fallback_to_proxy_allowed": False,
        "read_report": read_report,
        "action_report": action_report,
        "coverage": coverage.to_dict(orient="records"),
    }
    write_json(report_out, report)
    print(json.dumps({
        "status": "PASS",
        "raw_all_dir": str(raw_all),
        "action_rows": int(len(action_panel)),
        "coverage": coverage[["action_dim", "observed_rate", "nonzero_rate"]].to_dict(orient="records"),
        "output": str(action_out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
