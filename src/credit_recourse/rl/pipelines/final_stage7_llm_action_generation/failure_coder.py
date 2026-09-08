from __future__ import annotations

"""Stage 7/8 failure-coding helpers for the closed eight-type taxonomy.

This module is intentionally independent from Oracle scores.  It may inspect
LLM output text, the materialized/projection action vector, and simulator
feasibility diagnostics, but never downstream ``delta_R_score`` columns.  The
purpose is to make the proposal's error taxonomy executable without turning
low model scores into post-hoc "errors".
"""

from dataclasses import dataclass
import json
import math
import re
from typing import Any

FAILURE_CODER_VERSION = "failure_coder_v1"
ORACLE_SCORES_USED_FOR_FAILURE_CODING = False

ACTION_FAMILY_BY_CANDIDATE: dict[str, str] = {
    "A0_noop": "noop",
    "DL1_deleverage_mild": "debt",
    "DL2_deleverage_moderate": "debt",
    "RF1_short_debt_refinance": "debt",
    "CX1_capex_discipline": "capex",
    "WC1_working_capital_tightening": "working_capital",
    "WC2_supplier_financing": "working_capital",
    "OE1_cost_efficiency_mild": "cost",
    "OE2_cost_efficiency_moderate": "cost",
    "MX1_cost_and_deleverage": "mixed_debt_cost",
    "MX2_liquidity_rescue": "mixed_liquidity",
}

WEAKNESS_TO_ALLOWED_FAMILIES: dict[str, set[str]] = {
    "debt": {"debt", "mixed_debt_cost", "mixed_liquidity"},
    "leverage": {"debt", "mixed_debt_cost", "mixed_liquidity"},
    "liquidity": {"working_capital", "debt", "mixed_liquidity"},
    "working_capital": {"working_capital", "mixed_liquidity"},
    "cost": {"cost", "mixed_debt_cost"},
    "margin": {"cost", "mixed_debt_cost"},
    "capex": {"capex"},
}

WEAKNESS_PATTERNS: dict[str, tuple[str, ...]] = {
    "debt": ("debt", "borrowing", "leverage", "liabilit", "interest burden", "short term debt", "short-term debt", "bond"),
    "liquidity": ("liquidity", "current ratio", "cash", "working capital", "short term funding", "short-term funding"),
    "working_capital": ("inventory", "receivable", "payable", "working capital", "turnover"),
    "cost": ("cost", "cogs", "sga", "sg&a", "expense", "operating margin", "margin", "profitability"),
    "capex": ("capex", "ppe", "fixed asset", "capital expenditure", "investment"),
}

# Which action vector axes directly address each weakness.  Bare names, not
# action__ names.
WEAKNESS_ACTION_AXES: dict[str, set[str]] = {
    "debt": {"short_debt_pct", "long_debt_pct", "bond_pct"},
    "leverage": {"short_debt_pct", "long_debt_pct", "bond_pct"},
    "liquidity": {"inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg", "short_debt_pct"},
    "working_capital": {"inv_turnover_chg", "ar_turnover_chg", "ap_turnover_chg"},
    "cost": {"cogs_ratio_chg", "sga_ratio_chg"},
    "margin": {"cogs_ratio_chg", "sga_ratio_chg"},
    "capex": {"ppe_pct"},
}

DIRECTION_RULE_VERSION = "diagnosis_action_family_map_v1"
MAGNITUDE_RULE_VERSION = "bounds_projection_norm_v1"
FEASIBILITY_RULE_VERSION = "post_sim_accounting_feasibility_v3"


def _norm_text(value: Any) -> str:
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            parts.append(str(k))
            parts.append(_norm_text(v))
        value = " ".join(parts)
    elif isinstance(value, (list, tuple, set)):
        value = " ".join(_norm_text(v) for v in value)
    value = str(value or "").lower().replace("&", " and ")
    value = re.sub(r"[^a-z0-9가-힣]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_primary_weakness(diagnosis: Any, rationale: str | None) -> tuple[str | None, str]:
    """Conservatively infer the weakness named by the LLM itself.

    We do not use firm outcomes or Oracle scores.  If multiple domains are
    strongly present, return ``None`` and send the row to review rather than
    forcing a brittle label.
    """
    text = _norm_text(diagnosis) + " " + _norm_text(rationale)
    if not text.strip():
        return None, "no_diagnosis_or_rationale_text"
    hits: dict[str, int] = {}
    for weakness, pats in WEAKNESS_PATTERNS.items():
        hits[weakness] = sum(1 for p in pats if p in text)
    hits = {k: v for k, v in hits.items() if v > 0}
    if not hits:
        return None, "no_mapped_weakness_keywords"
    ordered = sorted(hits.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return None, f"ambiguous_weakness_hits={hits}"
    w = ordered[0][0]
    # Normalize overlapping debt/leverage to debt for action-family purposes.
    if w == "leverage":
        w = "debt"
    return w, f"mapped_from_llm_text hits={hits}"


def _vector_addresses_weakness(action: dict[str, float], weakness: str | None) -> bool:
    if not weakness:
        return False
    axes = WEAKNESS_ACTION_AXES.get(weakness, set())
    if not axes:
        return False
    for ax in axes:
        if abs(float(action.get(ax, 0.0) or 0.0)) > 1e-12:
            return True
    return False


def code_direction_error(*, diagnosis: Any, rationale: str | None, selected_candidate: str | None, materialized_action: dict[str, float]) -> dict[str, Any]:
    weakness, reason = extract_primary_weakness(diagnosis, rationale)
    family = ACTION_FAMILY_BY_CANDIDATE.get(str(selected_candidate or ""), "unknown")
    if weakness is None:
        return {"direction_error_auto": False, "direction_review_needed": True, "direction_error_reason": reason, "direction_rule_version": DIRECTION_RULE_VERSION}
    allowed = WEAKNESS_TO_ALLOWED_FAMILIES.get(weakness)
    if not allowed:
        return {"direction_error_auto": False, "direction_review_needed": True, "direction_error_reason": f"weakness_not_mapped:{weakness}", "direction_rule_version": DIRECTION_RULE_VERSION}
    if family in allowed:
        return {"direction_error_auto": False, "direction_review_needed": False, "direction_error_reason": f"candidate_family_aligned:{weakness}->{family}; {reason}", "direction_rule_version": DIRECTION_RULE_VERSION}
    if _vector_addresses_weakness(materialized_action, weakness):
        return {"direction_error_auto": False, "direction_review_needed": True, "direction_error_reason": f"label_family_mismatch_but_vector_addresses:{weakness}->{family}; {reason}", "direction_rule_version": DIRECTION_RULE_VERSION}
    # A0 can be a valid conservative refusal only if rationale explicitly says no action.
    if family == "noop" and any(tok in _norm_text(rationale) for tok in ["no action", "do nothing", "not recommend"]):
        return {"direction_error_auto": False, "direction_review_needed": True, "direction_error_reason": f"noop_with_conservative_rationale:{weakness}; {reason}", "direction_rule_version": DIRECTION_RULE_VERSION}
    return {"direction_error_auto": True, "direction_review_needed": False, "direction_error_reason": f"diagnosis={weakness}; selected_family={family}; allowed={sorted(allowed)}; {reason}", "direction_rule_version": DIRECTION_RULE_VERSION}


def code_magnitude_error(*, mode: str, bound_clipping: dict[str, Any] | None, projection_distance: float | None, out_of_library: bool | None, materialized_action: dict[str, float]) -> dict[str, Any]:
    clips = bound_clipping or {}
    clip_n = len(clips)
    max_bound_violation_abs = 0.0
    for detail in clips.values():
        try:
            max_bound_violation_abs = max(max_bound_violation_abs, abs(float(detail.get("raw_value")) - float(detail.get("clipped_value"))))
        except Exception:
            pass
    # Normalized action size uses bare vector values; thresholds are conservative
    # because contract bounds already perform hard clipping.
    vals = [abs(float(v or 0.0)) for v in (materialized_action or {}).values()]
    action_l1_norm = float(sum(vals))
    nonzero_dims = int(sum(1 for v in vals if v > 1e-12))
    pdist = None if projection_distance is None else float(projection_distance)
    hard = bool(clip_n > 0 or (out_of_library is True) or (pdist is not None and pdist > 0.50))
    review = bool((not hard) and (mode == "free_form_10d") and ((pdist is not None and pdist > 0.25) or nonzero_dims >= 6))
    reasons: list[str] = []
    if clip_n:
        reasons.append(f"bound_clipping_dims={clip_n}")
    if out_of_library is True:
        reasons.append("projected_out_of_library")
    if pdist is not None:
        reasons.append(f"projection_distance={pdist:.6g}")
    if nonzero_dims >= 6:
        reasons.append(f"many_nonzero_dims={nonzero_dims}")
    if not reasons:
        reasons.append("within_bounds_and_projection_thresholds")
    return {
        "magnitude_error_auto": hard,
        "magnitude_review_needed": review,
        "magnitude_error_reason": "; ".join(reasons),
        "max_bound_violation_abs": float(max_bound_violation_abs),
        "action_l1_norm": action_l1_norm,
        "action_nonzero_dim_count": nonzero_dims,
        "magnitude_rule_version": MAGNITUDE_RULE_VERSION,
    }


def _parse_accounting_check(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, str) and value.strip():
        try:
            obj = json.loads(value)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {"_parse_error": value[:200]}
    return {}


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip().lower()
    return text in {"", "nan", "none", "null", "na", "n/a"}


def _boolish_true(value: Any) -> bool:
    if _is_nullish(value):
        return False
    if value is True:
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(float(value) == 1.0)
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _status_token(value: Any) -> str:
    """Normalize simulator status-like fields without treating nulls as bad."""
    if _is_nullish(value):
        return ""
    return str(value).strip().lower()


def _status_bad(value: Any) -> bool:
    if _is_nullish(value):
        return False
    return _status_token(value) in {"false", "0", "fail", "failed", "error", "bad"}


def _sustainability_status(value: Any) -> str:
    """Normalize the simulator's 3-step sustainability contract.

    The financial simulator emits ``critical`` / ``fragile`` / ``ok``.  Older
    diagnostics may still contain boolean-ish or fail/pass strings, so keep
    those compatible while making the current producer contract explicit.
    """
    token = _status_token(value)
    if token in {"critical", "fragile", "ok"}:
        return token
    if token in {"false", "0", "fail", "failed", "error", "bad"}:
        return "critical"
    if token in {"true", "1", "pass", "passed", "good"}:
        return "ok"
    return token


def code_feasibility_violation(sim_row: dict[str, Any], *, plug_to_assets_hard: float = 0.10, plug_to_assets_review: float = 0.05) -> dict[str, Any]:
    """Code post-simulation feasibility violations without using Oracle scores.

    Version v3 intentionally aligns sustainability labels and separates *hard feasibility violations* from
    simulator closure diagnostics.  ``residual_negative_flag`` is emitted by the
    simulator when presentation residuals had to be clipped/repaired while
    constructing current/non-current residual accounts.  That flag is useful for
    audit review, but by itself it is not evidence that the LLM recommendation
    violates accounting identities after the simulator has produced a balanced,
    non-negative output frame.  Treating it as a hard failure caused every LLM
    row in the frozen 2026-07 runs to receive ``feasibility_violation``.

    Hard auto-feasibility failures are therefore limited to core post-sim
    contract breaks: explicit simulator/preflight failure, failed accounting
    identity check, negative final current balance, or explicit critical sustainability
    failure.  Fragile sustainability and large plug ratios are retained as review diagnostics.  Residual
    presentation repairs are retained as metadata-only diagnostics, not as
    automatic failure categories or review triggers.
    """

    def f(key: str, default: float = float("nan")) -> float:
        try:
            return float(sim_row.get(key, default))
        except Exception:
            return default

    raw_assets = f("total_assets_before", f("sim__total_assets_before", f("total_assets", 0.0)))
    assets = abs(raw_assets)
    plug_denominator_source = "total_assets_before"
    if not math.isfinite(assets) or assets <= 0:
        assets = max(abs(f("current_assets_before", 0.0)) + abs(f("current_liabilities_before", 0.0)), 1.0)
        plug_denominator_source = "current_assets_plus_current_liabilities_proxy"
    plug_amount = f("plug_amount", 0.0)
    plug_ratio = abs(plug_amount) / max(assets, 1.0)
    plug_review_exceeded = bool(math.isfinite(plug_ratio) and plug_ratio > plug_to_assets_review)
    plug_hard_exceeded = bool(math.isfinite(plug_ratio) and plug_ratio > plug_to_assets_hard)

    accounting = _parse_accounting_check(sim_row.get("accounting_check"))
    accounting_failed = False
    if accounting:
        if accounting.get("_parse_error"):
            accounting_failed = True
        # Accept common positive keys; any explicit false on an identity check fails.
        for k, v in accounting.items():
            key = str(k).lower()
            if key in {"ok", "pass", "passed", "identity_ok", "accounting_identity_ok", "balanced"} and v is False:
                accounting_failed = True
            if key.endswith("ok") and v is False:
                accounting_failed = True
            if key in {"check", "status", "result"} and str(v).strip().lower() in {"fail", "failed", "error", "bad"}:
                accounting_failed = True

    negative_balance = any(
        math.isfinite(f(k, 0.0)) and f(k, 0.0) < -1e-9
        for k in ["current_assets_after", "current_liabilities_after"]
    )
    sustainability_status = _sustainability_status(sim_row.get("sustainability"))
    sustainability_bad = sustainability_status == "critical"
    sustainability_review = sustainability_status == "fragile"
    residual_presentation_repair = _boolish_true(sim_row.get("residual_negative_flag"))
    preflight_raw = sim_row.get("simulator_preflight_status")
    preflight = "ok" if _is_nullish(preflight_raw) else str(preflight_raw).strip().lower()
    preflight_bad = preflight not in {"ok", "pass", "passed"}

    core_violation = bool(sustainability_bad or accounting_failed or negative_balance or preflight_bad)
    hard = core_violation
    # Residual presentation repair is carried as an audit field only.  It is often
    # a baseline decomposition artifact and becomes uninformative if it routes
    # every row to review.  Large accounting plugs remain review diagnostics.
    review = bool((not hard) and (sustainability_review or plug_review_exceeded))

    reasons: list[str] = []
    if sustainability_bad:
        reasons.append(f"sustainability={sustainability_status}")
    elif sustainability_review:
        reasons.append(f"sustainability={sustainability_status}_review")
    if accounting_failed:
        reasons.append("accounting_check_failed")
    if negative_balance:
        reasons.append("negative_current_balance")
    if preflight_bad:
        reasons.append(f"simulator_preflight_status={preflight}")
    if residual_presentation_repair:
        reasons.append("residual_presentation_repair_flag_not_hard")
    if plug_hard_exceeded:
        reasons.append(f"plug_to_assets_hard_exceeded>{plug_to_assets_hard:.6g}")
    elif plug_review_exceeded:
        reasons.append(f"plug_to_assets_review_exceeded>{plug_to_assets_review:.6g}")
    reasons.append(f"plug_to_assets={plug_ratio:.6g}")
    if not core_violation:
        reasons.append("no_core_post_sim_feasibility_violation")

    return {
        "feasibility_violation_auto": hard,
        "feasibility_review_needed": review,
        "feasibility_error_reason": "; ".join(reasons),
        "plug_to_assets": float(plug_ratio),
        "plug_denominator_source": plug_denominator_source,
        "accounting_check_failed": bool(accounting_failed),
        "negative_balance_flag": bool(negative_balance),
        "residual_presentation_repair_flag": bool(residual_presentation_repair),
        "plug_to_assets_review_exceeded": bool(plug_review_exceeded),
        "plug_to_assets_hard_exceeded": bool(plug_hard_exceeded),
        "feasibility_core_violation_flag": bool(core_violation),
        "feasibility_rule_version": FEASIBILITY_RULE_VERSION,
        "failure_coder_version": FAILURE_CODER_VERSION,
        "oracle_scores_used_for_failure_coding": ORACLE_SCORES_USED_FOR_FAILURE_CODING,
    }
