"""
Final Stage6 Oracle input-variable contract.

This module is the single source of truth for the financial R-code variables
that the simulator is allowed to recompute for Stage6 scoring.  The formulas
mirror Stage00_02's final ratio dictionary for the currently selected six
financial oracle variables.

Important design rule
---------------------
Alpha/Beta/Gamma use the same selected information set.  Cross-backend
agreement is therefore a functional-form robustness check, not a validation of
R-code formula correctness.  Formula correctness is validated separately by
``credit_recourse.oracle.verification.verify_oracle_stage6_input_contract``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

try:
    from .firm_state import FirmState
except ImportError:  # pragma: no cover
    from firm_state import FirmState


Number = Optional[float]


def safe_div(a: Number, b: Number, default: Number = None) -> Number:
    if a is None or b is None or b == 0:
        return default
    try:
        return float(a) / float(b)
    except (TypeError, ValueError, ZeroDivisionError):
        return default


@dataclass(frozen=True)
class OracleFinancialVariableSpec:
    variable_id: str
    category: str
    korean_name: str
    formula_korean: str
    formula_code: str
    numerator_fields: tuple[str, ...]
    denominator_fields: tuple[str, ...]
    expected_direction_from_dictionary: str
    selected_direction_is_data_driven: bool
    compute: Callable[[FirmState, Optional[FirmState]], Number]
    notes: str = ""


def _avg_capital_stock(state: FirmState, prev_state: Optional[FirmState]) -> Number:
    if prev_state is not None and prev_state.capital_stock is not None:
        return ((prev_state.capital_stock or 0.0) + (state.capital_stock or 0.0)) / 2.0
    return state.capital_stock


def _retained_earnings_growth(state: FirmState, prev_state: Optional[FirmState]) -> Number:
    if prev_state is None or prev_state.retained_earnings in (None, 0):
        return None
    if state.retained_earnings is None:
        return None
    return float(state.retained_earnings) / float(prev_state.retained_earnings) - 1.0


def _avg_total_liabilities(state: FirmState, prev_state: Optional[FirmState]) -> Number:
    vals: list[float] = []
    for value in (
        getattr(prev_state, "total_liabilities", None) if prev_state is not None else None,
        state.total_liabilities,
    ):
        if value is None:
            continue
        try:
            vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return sum(vals) / len(vals)


def _ebitda(state: Optional[FirmState]) -> Number:
    if state is None or state.operating_income is None:
        return None
    try:
        return float(state.operating_income) + float(state.depreciation or 0.0) + float(state.amortization or 0.0)
    except (TypeError, ValueError):
        return None


def _ebitda_growth(state: FirmState, prev_state: Optional[FirmState]) -> Number:
    prev_ebitda = _ebitda(prev_state)
    curr_ebitda = _ebitda(state)
    if prev_ebitda in (None, 0) or curr_ebitda is None:
        return None
    return float(curr_ebitda) / float(prev_ebitda) - 1.0


def _fcf_to_current_liabilities(state: FirmState, prev_state: Optional[FirmState]) -> Number:
    if state.operating_cf is None:
        return None
    try:
        capex = 0.0 if state.capex is None else float(state.capex)
        return safe_div(float(state.operating_cf) - capex, state.current_liabilities)
    except (TypeError, ValueError):
        return None


def _non_current_assets_growth(state: FirmState, prev_state: Optional[FirmState]) -> Number:
    if prev_state is None or prev_state.non_current_assets in (None, 0):
        return None
    if state.non_current_assets is None:
        return None
    try:
        return float(state.non_current_assets) / float(prev_state.non_current_assets) - 1.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


ORACLE_FINANCIAL_VARIABLE_CONTRACT: dict[str, OracleFinancialVariableSpec] = {
    "R006": OracleFinancialVariableSpec(
        variable_id="R006",
        category="수익성",
        korean_name="세전이익률",
        formula_korean="법인세차감전순이익 / 매출액",
        formula_code="pretax_income / revenue",
        numerator_fields=("pretax_income",),
        denominator_fields=("revenue",),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(s.pretax_income, s.revenue),
    ),
    "R064": OracleFinancialVariableSpec(
        variable_id="R064",
        category="안정성",
        korean_name="이익잉여금/총자산",
        formula_korean="이익잉여금 / 총자산",
        formula_code="retained_earnings / total_assets",
        numerator_fields=("retained_earnings",),
        denominator_fields=("total_assets",),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(s.retained_earnings, s.total_assets),
    ),
    "R085": OracleFinancialVariableSpec(
        variable_id="R085",
        category="부채상환능력",
        korean_name="금융비용부담률",
        formula_korean="금융비용 / 매출액",
        formula_code="financial_cost / revenue",
        numerator_fields=("financial_cost",),
        denominator_fields=("revenue",),
        expected_direction_from_dictionary="낮을수록 우량",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(s.financial_cost, s.revenue),
        notes="Legacy calculator may emit R086 first; simulator ratio_alias canonicalizes R086 -> R085.",
    ),
    "R116": OracleFinancialVariableSpec(
        variable_id="R116",
        category="유동성",
        korean_name="당좌비율",
        formula_korean="(유동자산 - 재고자산) / 유동부채",
        formula_code="(current_assets - inventory) / current_liabilities",
        numerator_fields=("current_assets", "inventory"),
        denominator_fields=("current_liabilities",),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(
            None if s.current_assets is None else float(s.current_assets) - float(s.inventory or 0.0),
            s.current_liabilities,
        ),
        notes=(
            "Liquidity replacement for newly developed oracle variables after excluding "
            "R122/R136 from Stage00_04 liquidity eligibility. R133 remains eligible by policy."
        ),
    ),
    "R133": OracleFinancialVariableSpec(
        variable_id="R133",
        category="유동성",
        korean_name="FCF/유동부채",
        formula_korean="FCF / 유동부채; FCF = 영업활동현금흐름 - CAPEX",
        formula_code="(operating_cf - capex) / current_liabilities",
        numerator_fields=("operating_cf", "capex"),
        denominator_fields=("current_liabilities",),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=_fcf_to_current_liabilities,
        notes=(
            "Selected by the 2026-06-03 oracle redevelopment after R122/R136 liquidity exclusions. "
            "CAPEX is interpreted as the simulator's positive investment outflow, matching Stage00_02 FCF = OCF - CAPEX."
        ),
    ),
    "R136": OracleFinancialVariableSpec(
        variable_id="R136",
        category="유동성",
        korean_name="매입채무/유동부채",
        formula_korean="매입채무 / 유동부채",
        formula_code="payables / current_liabilities",
        numerator_fields=("payables",),
        denominator_fields=("current_liabilities",),
        expected_direction_from_dictionary="상황별",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(s.payables, s.current_liabilities),
        notes=(
            "Deprecated for newly developed oracle liquidity selection. Kept for legacy artifact scoring "
            "until alpha/beta/gamma selected-variable metadata is refit."
        ),
    ),
    "R148": OracleFinancialVariableSpec(
        variable_id="R148",
        category="활동성",
        korean_name="타인자본회전율",
        formula_korean="매출액 / 평균부채총계",
        formula_code="revenue / avg(total_liabilities_t, total_liabilities_t_minus_1)",
        numerator_fields=("revenue",),
        denominator_fields=("total_liabilities", "prev_state.total_liabilities"),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(s.revenue, _avg_total_liabilities(s, p)),
        notes=(
            "Current Stage00_04 selection uses R148.  Stage6 computes the average denominator "
            "from simulated t+1 total liabilities and the base-year FirmState total liabilities."
        ),
    ),
    "R174": OracleFinancialVariableSpec(
        variable_id="R174",
        category="성장성",
        korean_name="EBITDA증가율",
        formula_korean="EBITDA_t / EBITDA_t-1 - 1; EBITDA = 영업이익 + 감가상각비 + 무형자산상각비",
        formula_code="(operating_income + depreciation + amortization)_t / (operating_income + depreciation + amortization)_t_minus_1 - 1",
        numerator_fields=("operating_income", "depreciation", "amortization"),
        denominator_fields=("prev_state.operating_income", "prev_state.depreciation", "prev_state.amortization"),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=_ebitda_growth,
        notes=(
            "Current Stage00_04 selection uses R174.  Stage6 uses simulator D&A fields for t+1 "
            "and base FirmState D&A fields for t to preserve the lag-growth semantics."
        ),
    ),
    "R157": OracleFinancialVariableSpec(
        variable_id="R157",
        category="활동성",
        korean_name="자본금회전율",
        formula_korean="매출액 / 평균자본금",
        formula_code="revenue / avg(capital_stock_t, capital_stock_t_minus_1)",
        numerator_fields=("revenue",),
        denominator_fields=("capital_stock", "prev_state.capital_stock"),
        expected_direction_from_dictionary="상황별",
        selected_direction_is_data_driven=True,
        compute=lambda s, p: safe_div(s.revenue, _avg_capital_stock(s, p)),
    ),
    "R182": OracleFinancialVariableSpec(
        variable_id="R182",
        category="성장성",
        korean_name="비유동자산증가율",
        formula_korean="비유동자산_t / 비유동자산_t-1 - 1",
        formula_code="non_current_assets_t / non_current_assets_t_minus_1 - 1",
        numerator_fields=("non_current_assets",),
        denominator_fields=("prev_state.non_current_assets",),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=_non_current_assets_growth,
        notes="Selected by the 2026-06-03 oracle redevelopment as the growth category variable.",
    ),
    "R185": OracleFinancialVariableSpec(
        variable_id="R185",
        category="성장성",
        korean_name="이익잉여금증가율",
        formula_korean="이익잉여금_t / 이익잉여금_t-1 - 1",
        formula_code="retained_earnings_t / retained_earnings_t_minus_1 - 1",
        numerator_fields=("retained_earnings",),
        denominator_fields=("prev_state.retained_earnings",),
        expected_direction_from_dictionary="높을수록 우량",
        selected_direction_is_data_driven=True,
        compute=_retained_earnings_growth,
    ),
}


def compute_contract_financial_variables(
    state: FirmState,
    prev_state: Optional[FirmState] = None,
) -> dict[str, Number]:
    """Compute the six selected financial oracle variables from FirmState."""
    return {
        k: spec.compute(state, prev_state)
        for k, spec in ORACLE_FINANCIAL_VARIABLE_CONTRACT.items()
    }


def contract_as_records() -> list[dict[str, object]]:
    return [
        {
            "variable_id": spec.variable_id,
            "category": spec.category,
            "korean_name": spec.korean_name,
            "formula_korean": spec.formula_korean,
            "formula_code": spec.formula_code,
            "numerator_fields": list(spec.numerator_fields),
            "denominator_fields": list(spec.denominator_fields),
            "expected_direction_from_dictionary": spec.expected_direction_from_dictionary,
            "selected_direction_is_data_driven": spec.selected_direction_is_data_driven,
            "notes": spec.notes,
        }
        for spec in ORACLE_FINANCIAL_VARIABLE_CONTRACT.values()
    ]
