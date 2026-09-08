"""
FirmState — t년 시점의 firm 재무 상태.

Raw data의 41개 핵심 항목을 하나의 dataclass로 묶음.
모든 금액은 천원 단위 (raw data와 동일).
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------
# Item code / alias mapping (single source of truth)
# ---------------------------------------------------------------------
from credit_recourse.contracts.account_registry import (
    ITEM_CODE_MAP,
    REVERSE_ITEM_CODE_MAP,
    resolved_field_values,
)


@dataclass
class FirmState:
    """
    t년 firm 재무 상태. 모든 값 천원 단위.
    Missing은 None으로 표현 — simulator에서 NaN-safe하게 처리.
    """
    firm_id: str
    year: int
    sector: str  # KOSPI 제조업 등
    rating_num: Optional[int] = None  # 외부 등급 (있는 경우만)
    rating_grade: Optional[str] = None

    # ── 손익계산서 ──
    revenue: Optional[float] = None
    cogs: Optional[float] = None
    gross_profit: Optional[float] = None
    sga: Optional[float] = None
    operating_income: Optional[float] = None
    financial_cost: Optional[float] = None
    pretax_income: Optional[float] = None
    tax_expense: Optional[float] = None
    net_income: Optional[float] = None
    comprehensive_income: Optional[float] = None
    depreciation: Optional[float] = None
    amortization: Optional[float] = None
    interest_income: Optional[float] = None
    dividend_income: Optional[float] = None

    # ── 재무상태표 ──
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    non_current_assets: Optional[float] = None
    cash: Optional[float] = None
    short_term_investments: Optional[float] = None
    receivables: Optional[float] = None
    inventory: Optional[float] = None
    ppe: Optional[float] = None
    intangibles: Optional[float] = None
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    non_current_liabilities: Optional[float] = None
    short_term_debt: Optional[float] = None
    current_portion_long_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    bonds: Optional[float] = None
    payables: Optional[float] = None
    total_equity: Optional[float] = None
    capital_stock: Optional[float] = None
    retained_earnings: Optional[float] = None

    # ── 자본변동표 (자본 분해용) ──
    ending_capital_stock: Optional[float] = None
    ending_capital_surplus: Optional[float] = None
    ending_other_capital: Optional[float] = None
    ending_oci: Optional[float] = None
    ending_retained_earnings: Optional[float] = None

    # ── 이익잉여금처분계산서 ──
    cash_dividends: Optional[float] = None

    # ── 현금흐름표 ──
    operating_cf: Optional[float] = None
    investing_cf: Optional[float] = None
    financing_cf: Optional[float] = None
    capex: Optional[float] = None

    # ----------------------------------------------------------------
    # Convenience derived properties
    # ----------------------------------------------------------------
    @property
    def total_debt(self) -> Optional[float]:
        """총차입금 = 단기 + 유동성장기 + 장기 + 사채."""
        parts = [
            self.short_term_debt,
            self.current_portion_long_debt,
            self.long_term_debt,
            self.bonds,
        ]
        if all(p is None for p in parts):
            return None
        return sum(p or 0 for p in parts)

    @property
    def capital_surplus(self) -> Optional[float]:
        """자본잉여금 — 자본변동표에서. 없으면 자본총계 - 자본금 - 이익잉여금으로 plug."""
        if self.ending_capital_surplus is not None:
            return self.ending_capital_surplus
        if (
            self.total_equity is not None
            and self.capital_stock is not None
            and self.retained_earnings is not None
        ):
            return (
                self.total_equity - self.capital_stock - self.retained_earnings
            )
        return None

    def accounting_identity_check(self, tol: float = 1.0) -> dict:
        """자산 = 부채 + 자본 검증. tol = 천원."""
        if any(
            x is None
            for x in [self.total_assets, self.total_liabilities, self.total_equity]
        ):
            return {"check": "skipped", "reason": "missing"}
        diff = self.total_assets - (self.total_liabilities + self.total_equity)
        return {
            "check": "ok" if abs(diff) <= tol else "fail",
            "diff": diff,
            "rel_diff": diff / self.total_assets if self.total_assets else None,
        }

    def to_dict(self) -> dict:
        return asdict(self)


def load_firm_state_from_columns(
    column_dict: dict, firm_id: str, year: int, sector: str = "Unknown"
) -> FirmState:
    """Create FirmState from raw headers, U-codes, bare fields, or registry aliases.

    This loader is registry-aware and can read canonical Stage2/Stage6 columns
    such as sim__revenue, raw__revenue, raw__avs__U01..., and bracketed raw
    statement headers.  Existing behavior for full raw headers, item codes, and
    bare FirmState field names is preserved as a subset.
    """
    fs = FirmState(firm_id=firm_id, year=year, sector=sector)

    values = resolved_field_values(column_dict, next_state=False)
    # Preserve direct bare-field fallback for any field not registered.
    for key, value in column_dict.items():
        if isinstance(key, str) and hasattr(fs, key) and key not in values:
            values[key] = value

    for field_name, value in values.items():
        if value is None:
            continue
        try:
            if isinstance(value, float) and value != value:
                continue
            setattr(fs, field_name, float(value))
        except (TypeError, ValueError):
            pass

    return fs


def load_firm_state_from_registry(
    column_dict: dict, firm_id: str, year: int, sector: str = "Unknown"
) -> FirmState:
    """Alias for the registry-aware loader used by Stage6/counterfactual code."""
    return load_firm_state_from_columns(column_dict, firm_id=firm_id, year=year, sector=sector)
