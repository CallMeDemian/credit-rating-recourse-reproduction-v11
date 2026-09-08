"""
BusinessPlan — Simulator의 default 1년 사업계획.

Decision 1 (넓은 status quo) + firm 과거 평균 하이브리드:
- 매출 성장률 0% (action으로 변경 가능)
- 비용 비율 = firm 과거 평균
- CAPEX = firm 과거 평균 capex/revenue ratio
- 회전율 = firm 과거 평균
- 배당 = firm 과거 평균 payout ratio
"""
from dataclasses import dataclass, asdict
from typing import Optional, List
import statistics

from credit_recourse.simulator.firm_state import FirmState


# ---------------------------------------------------------------------
# Constants — Decision 3, 5, 11, 12 ; 등급 spread (Decision 4 확장)
# ---------------------------------------------------------------------
DEFAULT_TAX_RATE = 0.22
DEFAULT_DEPRECIATION_RATE = 0.08         # PP&E 대비
DEFAULT_AMORTIZATION_RATE = 0.10         # 무형자산 대비
DEFAULT_RECLASS_RATE = 0.10              # 장기차입금 → 유동성장기부채 매년 비율

# 등급 spread (annual %p added to base rate)
RATING_SPREAD = {
    "AAA": 0.000, "AA": 0.003, "A": 0.007,
    "BBB": 0.015, "BB": 0.030, "B": 0.050, "C": 0.080, "D": 0.100,
}

DEFAULT_RATES = {
    "short": 0.030,  # 단기차입금
    "long": 0.040,   # 장기차입금
    "bond": 0.045,   # 사채
}


def grade_to_spread(grade: Optional[str]) -> float:
    """등급 → spread 변환. 결측은 BBB로."""
    if grade is None:
        return RATING_SPREAD["BBB"]
    base = grade.rstrip("+-")
    return RATING_SPREAD.get(base, RATING_SPREAD["BBB"])


def effective_rate(debt_type: str, grade: Optional[str]) -> float:
    """차입 종류 + 등급 → 실효이자율."""
    return DEFAULT_RATES[debt_type] + grade_to_spread(grade)


# ---------------------------------------------------------------------
# BusinessPlan
# ---------------------------------------------------------------------
@dataclass
class BusinessPlan:
    """
    Default 1년 사업계획.
    값들은 firm 과거 평균에서 calibrate 후 action으로 perturb.
    """
    # 손익 비율 (대부분 매출 대비)
    revenue_growth: float = 0.0          # default 0% (status quo)
    cogs_ratio: float = 0.70             # COGS / Revenue
    sga_ratio: float = 0.15              # SG&A / Revenue
    non_op_income_ratio: float = 0.005   # (이자수익 + 배당수익) / Revenue
    tax_rate: float = DEFAULT_TAX_RATE

    # 회전율 (회 단위; days = 365/회전율)
    ar_turnover: float = 6.0             # 매출 / 평균매출채권
    inv_turnover: float = 8.0            # 매출원가 / 평균재고
    ap_turnover: float = 6.0             # 매출원가 / 평균매입채무

    # 자본 활동
    capex_to_revenue: float = 0.05       # 정기 CAPEX 비율
    depreciation_rate: float = DEFAULT_DEPRECIATION_RATE
    amortization_rate: float = DEFAULT_AMORTIZATION_RATE
    dividend_payout: float = 0.20        # net_income 대비 (양수일 때만)

    # 차입 활동 default
    debt_repayment_ratio: float = 0.0    # 행동 없을 때 차입 변동 0
    reclass_rate: float = DEFAULT_RECLASS_RATE

    # 이자율 (등급별 spread는 grade로 결정)
    rate_short: float = DEFAULT_RATES["short"]
    rate_long: float = DEFAULT_RATES["long"]
    rate_bond: float = DEFAULT_RATES["bond"]

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------
# Calibration — firm 과거 데이터에서 BusinessPlan default 추출
# ---------------------------------------------------------------------
def _safe_div(a, b, default=None):
    if a is None or b is None or b == 0:
        return default
    return a / b


def _avg(xs: list, default=None):
    valid = [x for x in xs if x is not None]
    if not valid:
        return default
    return statistics.mean(valid)


def calibrate_business_plan(
    firm_history: List[FirmState],
    grade: Optional[str] = None,
) -> BusinessPlan:
    """
    Firm 과거 (최대 3년) 재무제표에서 BusinessPlan default를 추출.

    history는 시간순. 가장 마지막이 t년.
    """
    if not firm_history:
        return BusinessPlan()

    # 비율 계산용 ratio list
    cogs_ratios = [
        _safe_div(s.cogs, s.revenue) for s in firm_history
    ]
    sga_ratios = [
        _safe_div(s.sga, s.revenue) for s in firm_history
    ]
    non_op_ratios = []
    for s in firm_history:
        ii = (s.interest_income or 0) + (s.dividend_income or 0)
        non_op_ratios.append(_safe_div(ii, s.revenue))

    # 회전율
    ar_turnovers = [
        _safe_div(s.revenue, s.receivables) for s in firm_history
    ]
    inv_turnovers = [
        _safe_div(s.cogs, s.inventory) for s in firm_history
    ]
    ap_turnovers = [
        _safe_div(s.cogs, s.payables) for s in firm_history
    ]

    # CAPEX intensity (capex / revenue, 양수 capex만)
    capex_ratios = []
    for s in firm_history:
        cx = abs(s.capex) if s.capex is not None else None
        capex_ratios.append(_safe_div(cx, s.revenue))

    # 감가상각률
    dep_rates = [
        _safe_div(s.depreciation, s.ppe) for s in firm_history
    ]
    amort_rates = [
        _safe_div(s.amortization, s.intangibles) for s in firm_history
    ]

    # 배당성향 (net_income > 0 한정)
    payout_ratios = []
    for s in firm_history:
        if s.net_income is not None and s.net_income > 0 and s.cash_dividends is not None:
            payout_ratios.append(s.cash_dividends / s.net_income)

    # Firm 자체 effective rate (financial_cost / total_debt) — 가능하면 사용
    eff_rates = []
    for s in firm_history:
        td = s.total_debt
        if td and td > 0 and s.financial_cost is not None:
            eff_rates.append(s.financial_cost / td)
    avg_eff_rate = _avg(eff_rates) if eff_rates else None

    if avg_eff_rate is not None:
        # Firm rate를 기본으로, 차입 종류별 spread는 firm rate를 가운데로 간격 조정
        # 단기 = avg-1%, 장기 = avg, 사채 = avg+1% (대략적 term structure)
        rate_short = max(0.005, avg_eff_rate - 0.005)
        rate_long = max(0.005, avg_eff_rate)
        rate_bond = max(0.005, avg_eff_rate + 0.005)
    else:
        # Fallback: 등급 기반 default
        rate_short = effective_rate("short", grade)
        rate_long = effective_rate("long", grade)
        rate_bond = effective_rate("bond", grade)

    bp = BusinessPlan(
        revenue_growth=0.0,
        cogs_ratio=_avg(cogs_ratios, default=0.70) or 0.70,
        sga_ratio=_avg(sga_ratios, default=0.15) or 0.15,
        non_op_income_ratio=_avg(non_op_ratios, default=0.005) or 0.005,
        tax_rate=DEFAULT_TAX_RATE,
        ar_turnover=_avg(ar_turnovers, default=6.0) or 6.0,
        inv_turnover=_avg(inv_turnovers, default=8.0) or 8.0,
        ap_turnover=_avg(ap_turnovers, default=6.0) or 6.0,
        capex_to_revenue=_avg(capex_ratios, default=0.05) or 0.05,
        depreciation_rate=_avg(dep_rates, default=DEFAULT_DEPRECIATION_RATE)
        or DEFAULT_DEPRECIATION_RATE,
        amortization_rate=_avg(amort_rates, default=DEFAULT_AMORTIZATION_RATE)
        or DEFAULT_AMORTIZATION_RATE,
        dividend_payout=_avg(payout_ratios, default=0.0) or 0.0,
        rate_short=rate_short,
        rate_long=rate_long,
        rate_bond=rate_bond,
    )
    return bp
