# -*- coding: utf-8 -*-
"""Oracle variable reconstruction from simulated FirmState.

Stage6 must not attach arbitrary R-codes to plausible ratios.  This module keeps
an explicit formula registry for the current Oracle-selected variables and
exports an audit helper so Stage6 can fail or warn before reporting scores when
backend params and simulator formulas drift.

Important: Stage2 sector-Φ aliases are *not* the source of truth for Oracle
selected-variable definitions.  The source of truth is Stage00_04 selected
variables + Stage1 backend params.  This registry covers the variables that can
be recomputed deterministically from the financial simulator; non-financial
variables are passed through from the base row/exogenous context.
"""
from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Optional

try:
    from .firm_state import FirmState
    from .ratio_alias import canonicalize_dict
except ImportError:  # direct script/local import fallback
    from firm_state import FirmState
    from ratio_alias import canonicalize_dict


def _safe_div(a: Any, b: Any, default: Optional[float] = None) -> Optional[float]:
    if a is None or b is None:
        return default
    try:
        aa = float(a)
        bb = float(b)
        if not math.isfinite(aa) or not math.isfinite(bb) or abs(bb) <= 1e-12:
            return default
        return aa / bb
    except (TypeError, ValueError, ZeroDivisionError):
        return default


def _avg(a: Any, b: Any) -> Optional[float]:
    vals = []
    for x in (a, b):
        try:
            if x is not None and math.isfinite(float(x)):
                vals.append(float(x))
        except Exception:
            pass
    if not vals:
        return None
    return sum(vals) / len(vals)


def _growth(curr: Any, prev: Any) -> Optional[float]:
    if curr is None or prev is None:
        return None
    try:
        c = float(curr)
        p = float(prev)
        if not math.isfinite(c) or not math.isfinite(p) or abs(p) <= 1e-12:
            return None
        return c / p - 1.0
    except Exception:
        return None


def _ebitda(state: Optional[FirmState]) -> Optional[float]:
    """EBITDA proxy used by Stage00_02 R174: operating income + D&A.

    FirmState carries depreciation and amortization separately.  Treat missing
    D&A components as zero only when operating income exists; if operating income
    itself is missing, EBITDA cannot be reconstructed deterministically.
    """
    if state is None or state.operating_income is None:
        return None
    try:
        op = float(state.operating_income)
        dep = 0.0 if state.depreciation is None else float(state.depreciation)
        amort = 0.0 if state.amortization is None else float(state.amortization)
        val = op + dep + amort
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


# Formula registry.  Update here only when Stage00_04/Stage1 selected-variable
# definitions change; do not infer definitions from Stage2 sector-Φ aliases.
# formula_name is intentionally Korean-readable for audit reports.
def _r006(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    return _safe_div(state.pretax_income, state.revenue)  # 세전이익률


def _r064(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    return _safe_div(state.retained_earnings, state.total_assets)  # 이익잉여금/총자산


def _r085(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    return _safe_div(state.financial_cost, state.revenue)  # 금융비용부담률


def _r116(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    # 당좌비율 = (유동자산 - 재고자산) / 유동부채.
    # This remains registered as a safe liquidity fallback after excluding R122/R136
    # from Stage00_04 liquidity selection.
    if state.current_assets is None:
        return None
    inventory = 0.0 if state.inventory is None else state.inventory
    return _safe_div(float(state.current_assets) - float(inventory), state.current_liabilities)


def _r133(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    # FCF/유동부채.  Stage00_02 defines FCF as operating cash flow minus CAPEX.
    # FinancialSimulator stores CAPEX as a positive outflow amount in FirmState.capex.
    if state.operating_cf is None:
        return None
    try:
        operating_cf = float(state.operating_cf)
        capex = 0.0 if state.capex is None else float(state.capex)
        fcf = operating_cf - capex
    except (TypeError, ValueError):
        return None
    return _safe_div(fcf, state.current_liabilities)


def _r136(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    # Deprecated for newly developed oracle liquidity selection: kept only to
    # score legacy artifacts until their selected-variable metadata is refit.
    return _safe_div(state.payables, state.current_liabilities)  # 매입채무/유동부채


def _r148(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    # 타인자본회전율 = 매출액 / 평균부채총계.
    # In Stage6, ``state`` is the simulated t+1 state and ``prev`` is the
    # audited base-year FirmState.  This preserves the Stage00_02 average
    # denominator semantics instead of using a current-year-only shortcut.
    avg_total_liabilities = _avg(
        getattr(prev, "total_liabilities", None) if prev is not None else None,
        state.total_liabilities,
    )
    return _safe_div(state.revenue, avg_total_liabilities)


def _r174(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    # EBITDA증가율 = EBITDA_t / EBITDA_t-1 - 1.
    # EBITDA proxy follows Stage00_02: operating_income + depreciation +
    # amortization.  Stage6 passes the base FirmState as ``prev``.
    return _growth(_ebitda(state), _ebitda(prev))


def _r157(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    avg_capital = _avg(getattr(prev, "capital_stock", None) if prev is not None else None, state.capital_stock)
    return _safe_div(state.revenue, avg_capital)  # 자본금회전율


def _r182(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    # 비유동자산증가율 = 비유동자산_t / 비유동자산_t-1 - 1.
    # Stage6 passes the audited base-year FirmState as prev and simulated t+1 as state.
    return _growth(state.non_current_assets, getattr(prev, "non_current_assets", None) if prev is not None else None)


def _r185(state: FirmState, prev: Optional[FirmState], exog: Dict[str, float]) -> Optional[float]:
    return _growth(state.retained_earnings, getattr(prev, "retained_earnings", None) if prev is not None else None)  # 이익잉여금증가율


ORACLE_FORMULA_REGISTRY: dict[str, dict[str, Any]] = {
    "R006": {"formula_name": "세전이익률", "callable": _r006, "source": "pretax_income/revenue"},
    "R064": {"formula_name": "이익잉여금/총자산", "callable": _r064, "source": "retained_earnings/total_assets"},
    "R085": {"formula_name": "금융비용부담률", "callable": _r085, "source": "financial_cost/revenue"},
    "R116": {"formula_name": "당좌비율", "callable": _r116, "source": "(current_assets-inventory)/current_liabilities"},
    "R133": {"formula_name": "FCF/유동부채", "callable": _r133, "source": "(operating_cf-capex)/current_liabilities"},
    "R136": {"formula_name": "매입채무/유동부채", "callable": _r136, "source": "payables/current_liabilities", "deprecated_for_new_oracle_development": True},
    "R148": {"formula_name": "타인자본회전율", "callable": _r148, "source": "revenue/average_total_liabilities"},
    "R174": {"formula_name": "EBITDA증가율", "callable": _r174, "source": "ebitda_t/ebitda_t-1 - 1"},
    "R157": {"formula_name": "자본금회전율", "callable": _r157, "source": "revenue/average_capital_stock"},
    "R182": {"formula_name": "비유동자산증가율", "callable": _r182, "source": "non_current_assets_t/non_current_assets_t-1 - 1"},
    "R185": {"formula_name": "이익잉여금증가율", "callable": _r185, "source": "retained_earnings_t/retained_earnings_t-1 - 1"},
}

NONFINANCIAL_PASSTHROUGH = [
    "industry_median_rating_lag1_self_excl",
    "industry_avg_rating_lag1_self_excl",
    "cap_change_count_3y",
    "log_assets",
    "nf_log_assets",
    "operating_loss_freq_3y",
    "financial_data_completeness",
    "nf_retained_earnings_negative_flag",
    "nf_ratio_missing_rate",
    "ratio_missing_rate",
]

_EXPECTED_NAME_TOKENS = {
    "R006": ["세전"],
    "R064": ["이익잉여", "총자산"],
    "R085": ["금융비용"],
    "R116": ["당좌", "비율"],
    "R133": ["FCF", "유동부채"],
    "R136": ["매입채무", "유동부채"],
    "R148": ["타인자본", "회전"],
    "R174": ["EBITDA", "증가"],
    "R157": ["자본", "회전"],
    "R182": ["비유동자산", "증가"],
    "R185": ["이익잉여", "증가"],
}


def _norm_text(x: Any) -> str:
    return re.sub(r"\s+", "", str(x or "").lower())


def audit_formula_registry(selected_records: list[Any] | None) -> dict[str, Any]:
    """Validate selected variable metadata against deterministic formulas.

    selected_records may be a list of strings, dicts with variable_id/name fields,
    or mixed records from alpha/beta/gamma params.  The audit is conservative:
    it hard-fails missing financial formulas and flags likely Korean-name drift;
    lack of descriptive names is reported as a warning rather than guessed.
    """
    selected_records = selected_records or []
    rows = []
    errors = []
    warnings = []
    selected_ids: list[str] = []
    for rec in selected_records:
        if isinstance(rec, str):
            vid = rec
            name = ""
        elif isinstance(rec, dict):
            vid = str(rec.get("variable_id") or rec.get("id") or rec.get("name") or rec.get("variable") or "")
            name = " ".join(str(rec.get(k, "")) for k in ["display_name", "korean_name", "label", "description", "category"])
        else:
            continue
        if not vid:
            continue
        selected_ids.append(vid)
        if vid.startswith("R"):
            if vid not in ORACLE_FORMULA_REGISTRY:
                errors.append(f"No deterministic simulator formula registered for selected financial variable {vid}")
                rows.append({"variable_id": vid, "status": "MISSING_FORMULA", "metadata_name": name})
                continue
            formula = ORACLE_FORMULA_REGISTRY[vid]
            tokens = _EXPECTED_NAME_TOKENS.get(vid, [])
            nt = _norm_text(name)
            token_hit = bool(nt) and all(_norm_text(t) in nt for t in tokens)
            if nt and not token_hit:
                warnings.append(f"Selected-variable metadata for {vid} may not match simulator formula: metadata={name!r}, formula={formula['formula_name']}")
            rows.append({
                "variable_id": vid,
                "status": "FORMULA_REGISTERED",
                "formula_name": formula["formula_name"],
                "formula_source": formula["source"],
                "metadata_name": name,
                "metadata_token_match": token_hit if nt else None,
            })
        else:
            rows.append({"variable_id": vid, "status": "NONFINANCIAL_PASSTHROUGH_OR_CONTEXT", "metadata_name": name})
    return {
        "status": "PASS" if not errors else "FAIL",
        "selected_variables": selected_ids,
        "formula_rows": rows,
        "errors": errors,
        "warnings": warnings,
        "stage2_phi_aliases_are_not_used_as_oracle_formula_source": True,
    }


def compute_oracle_variables(
    state: FirmState,
    prev_state: Optional[FirmState] = None,
    exogenous: Optional[Dict[str, float]] = None,
) -> Dict[str, Optional[float]]:
    """Compute deterministic Oracle variables from a simulated FirmState.

    Only variables with explicit formulas are recomputed.  Non-financial/context
    variables are passed through from exogenous/base-row context.  Missing
    selected-variable formulas are caught by audit_formula_registry in Stage6.
    """
    exog = exogenous or {}
    out: Dict[str, Optional[float]] = {}
    for vid, meta in ORACLE_FORMULA_REGISTRY.items():
        out[vid] = meta["callable"](state, prev_state, exog)

    if state.total_assets is not None and state.total_assets > 0:
        out["nf_log_assets"] = math.log(float(state.total_assets))
        out["log_assets"] = out["nf_log_assets"]
    else:
        out["nf_log_assets"] = None
        out["log_assets"] = None

    for var in NONFINANCIAL_PASSTHROUGH:
        if var in exog:
            out[var] = exog.get(var)

    return canonicalize_dict(out)


def update_history_variables(state_t1: FirmState, history_t: Dict[str, float]) -> Dict[str, float]:
    """t년 history → t+1 history."""
    new_hist = dict(history_t)
    olf = history_t.get("operating_loss_freq_3y", 0)
    if state_t1.operating_income is not None and state_t1.operating_income < 0:
        new_hist["operating_loss_freq_3y"] = min(3, (olf or 0) + 1)
    return new_hist
