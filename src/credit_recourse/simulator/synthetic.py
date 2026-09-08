"""
Synthetic firm fixtures for simulator, RL smoke tests, and prototype training.

These helpers are intentionally kept inside `src` because production/prototype
code must not import from `tests`. They are not empirical data generators for
paper claims; they only support deterministic smoke tests and simulator-level
RL prototypes.
"""
from __future__ import annotations

from typing import Tuple
import numpy as np

from credit_recourse.simulator.firm_state import FirmState


def synthetic_firm(name: str = "TestFirm") -> FirmState:
    """Realistic small KOSPI-style manufacturer fixture. Values are KRW thousand."""
    return FirmState(
        firm_id=name,
        year=2020,
        sector="제조업",
        rating_grade="BBB",
        rating_num=10,
        # IS
        revenue=100_000_000,
        cogs=70_000_000,
        gross_profit=30_000_000,
        sga=15_000_000,
        operating_income=15_000_000,
        financial_cost=2_000_000,
        pretax_income=13_000_000,
        tax_expense=2_860_000,
        net_income=10_140_000,
        comprehensive_income=10_140_000,
        depreciation=4_000_000,
        amortization=500_000,
        interest_income=200_000,
        dividend_income=100_000,
        # BS — assets = liabilities + equity
        total_assets=120_000_000,
        current_assets=50_000_000,
        non_current_assets=70_000_000,
        cash=10_000_000,
        short_term_investments=5_000_000,
        receivables=20_000_000,
        inventory=15_000_000,
        ppe=50_000_000,
        intangibles=5_000_000,
        total_liabilities=70_000_000,
        current_liabilities=35_000_000,
        non_current_liabilities=35_000_000,
        short_term_debt=20_000_000,
        current_portion_long_debt=5_000_000,
        long_term_debt=25_000_000,
        bonds=10_000_000,
        payables=10_000_000,
        total_equity=50_000_000,
        capital_stock=10_000_000,
        retained_earnings=30_000_000,
        # SOCE
        ending_capital_stock=10_000_000,
        ending_capital_surplus=5_000_000,
        ending_other_capital=5_000_000,
        ending_oci=0,
        ending_retained_earnings=30_000_000,
        # SORE
        cash_dividends=2_000_000,
        # CF
        operating_cf=12_000_000,
        investing_cf=-5_000_000,
        financing_cf=-3_000_000,
        capex=4_000_000,
    )


def make_synthetic_pair(ep: int, seed_mult: int = 1) -> Tuple[FirmState, FirmState]:
    """Generate a deterministic (t-1, t) firm pair with mild financial variation."""
    rng = np.random.default_rng(ep * seed_mult)

    firm_tm1 = synthetic_firm(name=f"Firm_{ep}")
    firm_tm1.year = 2019

    firm_t = synthetic_firm(name=f"Firm_{ep}")
    firm_t.year = 2020

    firm_t.revenue *= 1.0 + float(rng.uniform(-0.12, 0.12))
    firm_t.net_income *= 1.0 + float(rng.uniform(-0.20, 0.20))
    firm_t.retained_earnings = (
        firm_tm1.retained_earnings
        + firm_t.net_income
        - (firm_t.cash_dividends or 0)
    )

    debt_mult = float(rng.uniform(0.6, 1.6))
    firm_t.short_term_debt = (firm_t.short_term_debt or 0) * debt_mult
    firm_t.long_term_debt = (firm_t.long_term_debt or 0) * debt_mult

    return firm_tm1, firm_t
