from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AccountConcept:
    field: str
    codes: tuple[str, ...]
    aliases: tuple[str, ...] = ()


# Single source of truth for simulator/enrichment financial account concepts.
# Order matters: the first code is the canonical export code, following codes are
# accepted aliases observed in legacy/NICE raw tables.
ACCOUNT_REGISTRY: dict[str, AccountConcept] = {
    "revenue": AccountConcept("revenue", ("U01B100000000",), ("raw__revenue", "sim__revenue", "revenue")),
    "cogs": AccountConcept("cogs", ("U01B200000000",), ("raw__cogs", "sim__cogs", "cogs")),
    "gross_profit": AccountConcept("gross_profit", ("U01B201014400",), ("gross_profit",)),
    "sga": AccountConcept("sga", ("U01B350000000",), ("raw__sga", "sim__sga", "sga")),
    "operating_income": AccountConcept("operating_income", ("U01B400000000", "U01B430000000"), ("raw__operating_income", "sim__operating_income", "operating_income")),
    "financial_cost": AccountConcept("financial_cost", ("U01B500020000", "U01B500020100", "U01B550000000"), ("raw__financial_cost", "sim__financial_cost", "financial_cost")),
    "pretax_income": AccountConcept("pretax_income", ("U01B700000000",), ("pretax_income",)),
    "tax_expense": AccountConcept("tax_expense", ("U01B750000000",), ("tax_expense",)),
    "net_income": AccountConcept("net_income", ("U01B840000000", "U01B800000000"), ("raw__net_income", "sim__net_income", "net_income")),
    "comprehensive_income": AccountConcept("comprehensive_income", ("U01B900000000",), ("comprehensive_income",)),
    "depreciation": AccountConcept("depreciation", ("U01B350014100",), ("depreciation",)),
    "amortization": AccountConcept("amortization", ("U01B350014300",), ("amortization",)),
    "interest_income": AccountConcept("interest_income", ("U01B500010000",), ("interest_income",)),
    "dividend_income": AccountConcept("dividend_income", ("U01B500010100",), ("dividend_income",)),

    "total_assets": AccountConcept("total_assets", ("U01A100000000",), ("raw__total_assets", "sim__total_assets", "total_assets")),
    "current_assets": AccountConcept("current_assets", ("U01A120000000", "U01A111038600"), ("raw__current_assets", "sim__current_assets", "current_assets")),
    "non_current_assets": AccountConcept("non_current_assets", ("U01A110000000",), ("non_current_assets", "noncurrent_assets")),
    "cash": AccountConcept("cash", ("U01A111010000", "U01A121010000", "U01A111050000"), ("raw__cash", "sim__cash", "cash")),
    "short_term_investments": AccountConcept("short_term_investments", ("U01A111020000", "U01A121020000", "U01A111043200"), ("raw__short_term_investments", "sim__short_term_investments", "short_term_investments")),
    "receivables": AccountConcept("receivables", ("U01A111045500", "U01A111045400", "U01A111052500"), ("raw__accounts_receivable", "sim__receivables", "accounts_receivable", "receivables")),
    "inventory": AccountConcept("inventory", ("U01A111038700", "U01A111052200"), ("raw__inventory", "sim__inventory", "inventory")),
    "ppe": AccountConcept("ppe", ("U01A111000000", "U01A111051200"), ("raw__ppe", "sim__ppe", "ppe")),
    "intangibles": AccountConcept("intangibles", ("U01A111019000",), ("intangibles", "intangible")),
    "total_liabilities": AccountConcept("total_liabilities", ("U01A800000000",), ("raw__total_liabilities", "sim__total_liabilities", "total_liabilities")),
    "current_liabilities": AccountConcept("current_liabilities", ("U01A820000000", "U01A811026000"), ("raw__current_liabilities", "sim__current_liabilities", "current_liabilities")),
    "non_current_liabilities": AccountConcept("non_current_liabilities", ("U01A810000000",), ("non_current_liabilities", "noncurrent_liabilities")),
    "short_term_debt": AccountConcept("short_term_debt", ("U01A811026700", "U01A811027200", "U01A811037700"), ("raw__short_debt", "sim__short_term_debt", "short_debt", "short_term_debt")),
    "current_portion_long_debt": AccountConcept("current_portion_long_debt", ("U01A811027400",), ("current_portion_long_debt",)),
    "long_term_debt": AccountConcept("long_term_debt", ("U01A811012800", "U01A811013300", "U01A811036800"), ("raw__long_debt", "sim__long_term_debt", "long_debt", "long_term_debt")),
    "bonds": AccountConcept("bonds", ("U01A811000000", "U01A811010500"), ("raw__bond", "sim__bonds", "bond", "bonds")),
    "payables": AccountConcept("payables", ("U01A811030800", "U01A811030700"), ("raw__accounts_payable", "sim__payables", "accounts_payable", "payables")),
    "total_equity": AccountConcept("total_equity", ("U01A600000000",), ("raw__total_equity", "sim__total_equity", "total_equity")),
    "capital_stock": AccountConcept("capital_stock", ("U01A611000000",), ("capital_stock", "capital")),
    "retained_earnings": AccountConcept("retained_earnings", ("U01C500000000", "U01A617000000", "U01A615000000"), ("raw__retained_earnings", "sim__retained_earnings", "retained_earnings")),

    "operating_cf": AccountConcept("operating_cf", ("U01D100000000",), ("raw__operating_cf", "sim__operating_cf", "operating_cf")),
    "investing_cf": AccountConcept("investing_cf", ("U01D200000000",), ("investing_cf",)),
    "financing_cf": AccountConcept("financing_cf", ("U01D300000000",), ("financing_cf",)),
    "capex": AccountConcept("capex", ("U01D240000000", "U01D210000000", "U01D206012400"), ("raw__capex", "sim__capex", "capex")),
    "ending_capital_stock": AccountConcept("ending_capital_stock", ("U01C100000000",), ("ending_capital_stock",)),
    "ending_capital_surplus": AccountConcept("ending_capital_surplus", ("U01C200000000",), ("ending_capital_surplus",)),
    "ending_other_capital": AccountConcept("ending_other_capital", ("U01C300000000",), ("ending_other_capital",)),
    "ending_oci": AccountConcept("ending_oci", ("U01C400000000",), ("ending_oci",)),
    "ending_retained_earnings": AccountConcept("ending_retained_earnings", ("U01C500000000",), ("ending_retained_earnings",)),
    "cash_dividends": AccountConcept("cash_dividends", ("U01F340000000", "U01E300011000"), ("raw__cash_dividends", "sim__cash_dividends", "cash_dividends")),
}

ITEM_CODE_MAP: dict[str, str] = {code: concept.field for concept in ACCOUNT_REGISTRY.values() for code in concept.codes}
REVERSE_ITEM_CODE_MAP: dict[str, str] = {concept.field: concept.codes[0] for concept in ACCOUNT_REGISTRY.values()}


def aliases_for(concept_name: str) -> list[str]:
    concept = ACCOUNT_REGISTRY[concept_name]
    return [*concept.aliases, *concept.codes]


def concept_codes(concept_name: str) -> tuple[str, ...]:
    return ACCOUNT_REGISTRY[concept_name].codes


def _candidate_keys(token: str, *, next_state: bool = False) -> list[str]:
    token = str(token)
    if next_state:
        return [f"next__{token}", f"{token}__next"]
    return [token]


def _find_column(columns, token: str, *, next_state: bool = False) -> str | None:
    """Find a column by exact alias first, then by U-code substring.

    Handles Stage2/Stage6 naming forms such as sim__X, raw__X,
    next__sim__X, raw__avs__U01..., and bracketed statement headers.
    """
    cols = [str(c) for c in columns]
    original = {str(c): c for c in columns}
    for key in _candidate_keys(token, next_state=next_state):
        if key in original:
            return original[key]
    m = re.search(r"U01[A-Z0-9]+", str(token))
    if not m:
        return None
    code = m.group(0)
    for c in cols:
        if next_state and not (c.startswith("next__") or c.endswith("__next")):
            continue
        if (not next_state) and (c.startswith("next__") or c.endswith("__next")):
            continue
        if code in c:
            return original[c]
    return None


def resolve_series(df: pd.DataFrame, concept_name: str, *, next_state: bool = False, default=np.nan) -> pd.Series:
    """Return the first present numeric column for an account concept.

    The lookup order is the registry order: semantic aliases first, then U-code
    aliases.  Missing concepts return a Series filled with default.
    """
    for cand in aliases_for(concept_name):
        col = _find_column(df.columns, cand, next_state=next_state)
        if col is not None:
            obj = df[col]
            if isinstance(obj, pd.DataFrame):
                obj = obj.iloc[:, 0]
            return pd.to_numeric(obj, errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def resolve_mapping_value(column_dict: Mapping[Any, Any], concept_name: str, *, next_state: bool = False) -> Any:
    """Resolve one concept from a row/key-value mapping using registry aliases."""
    keys = list(column_dict.keys())
    for cand in aliases_for(concept_name):
        hit = _find_column(keys, cand, next_state=next_state)
        if hit is None:
            continue
        val = column_dict.get(hit)
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except Exception:
            pass
        return val
    return None


def resolved_field_values(column_dict: Mapping[Any, Any], *, next_state: bool = False) -> dict[str, Any]:
    """Resolve all registered concepts to canonical FirmState field names."""
    out: dict[str, Any] = {}
    for name, concept in ACCOUNT_REGISTRY.items():
        val = resolve_mapping_value(column_dict, name, next_state=next_state)
        if val is not None:
            out[concept.field] = val
    return out
