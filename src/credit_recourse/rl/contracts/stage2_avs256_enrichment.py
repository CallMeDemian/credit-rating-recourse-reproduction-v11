from __future__ import annotations
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from credit_recourse.rl.contracts.avs256_acd_v2 import (
    build_feature_manifest, CONTINUOUS_COLUMNS, ACD_TARGET_COLUMNS, CATEGORICAL_COLUMNS,
)
from credit_recourse.contracts.account_registry import aliases_for, resolve_series

EPS = 1e-9


def _concept_aliases(name: str) -> list[str]:
    """Return canonical aliases + U-code aliases from the account registry.

    Keeping the Stage2 enrichment aliases in sync with FirmState/account_registry
    prevents silent zero-imputation of phi-critical accounts such as net_income
    and financial_cost when NICE raw files use alternate U-codes.
    """
    return aliases_for(name)


def _num(df: pd.DataFrame, col: str | None, default: float = np.nan) -> pd.Series:
    if col and col in df.columns:
        obj = df[col]
        if isinstance(obj, pd.DataFrame):
            obj = obj.iloc[:, 0]
        return pd.to_numeric(obj, errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _find_by_code(df: pd.DataFrame, code: str, *, next_state: bool = False) -> str | None:
    code = str(code)
    prefs = [f"next__"] if next_state else [""]
    for c in df.columns:
        s = str(c)
        if next_state and not s.startswith("next__") and not s.endswith("__next"):
            continue
        if (not next_state) and (s.startswith("next__") or s.endswith("__next")):
            continue
        if code in s:
            return c
    return None


def _find_first(df: pd.DataFrame, names: list[str], *, next_state: bool = False) -> str | None:
    candidates = []
    for n in names:
        if next_state:
            candidates.extend([f"next__{n}", f"{n}__next"])
        else:
            candidates.append(n)
    for c in candidates:
        if c in df.columns:
            return c
    for n in names:
        code_match = re.search(r"U01[A-Z0-9]+", n)
        if code_match:
            hit = _find_by_code(df, code_match.group(0), next_state=next_state)
            if hit:
                return hit
    return None


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    d = den.where(den.abs() > EPS)
    return num / d


def _safe_fill_numeric(s: pd.Series, default: float = 0.0) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med = x.median(skipna=True)
    if not np.isfinite(med):
        med = default
    return x.fillna(float(med))


def _market_code(s: pd.Series) -> pd.Series:
    m = {"KOSPI": 1.0, "KOSDAQ": 2.0, "KONEX": 3.0, "유가증권": 1.0, "코스피": 1.0, "코스닥": 2.0, "코넥스": 3.0}
    return s.astype(str).map(lambda x: m.get(x.strip().upper(), m.get(x.strip(), 0.0))).astype(float)


def _base_accounts(df: pd.DataFrame, *, next_state: bool = False) -> dict[str, pd.Series]:
    # Exact aliases first; raw Stage2A concepts second; U-code headers last.
    def g(out: str, aliases: list[str], default=np.nan) -> pd.Series:
        return _num(df, _find_first(df, [out] + aliases, next_state=next_state), default)

    total_assets = g("sim__total_assets", ["raw__total_assets", "total_assets", "balance_sheet__[U01A100000000]자산총계(IFRS)(천원)", "U01A100000000"])
    current_assets = g("sim__current_assets", ["raw__current_assets", "current_assets", "U01A120000000"])
    noncurrent_assets = g("balance_sheet__[U01A110000000]   비유동자산(*)(IFRS)(천원)", ["U01A110000000"])
    cash = g("sim__cash", ["raw__cash", "cash", "U01A111010000", "U01A121010000"])
    sti = g("sim__short_term_investments", ["raw__short_term_investments", "short_term_investments", "U01A111020000", "U01A121020000"])
    receivables = g("sim__receivables", ["raw__accounts_receivable", "accounts_receivable", "U01A111045500", "U01A111045400", "U01A111052500"])
    inventory = g("sim__inventory", ["raw__inventory", "inventory", "U01A111038700", "U01A111052200"])
    ppe = g("sim__ppe", ["raw__ppe", "ppe", "U01A111000000", "U01A111051200"])
    intangible = g("balance_sheet__[U01A111019000]      무형자산(*)(IFRS)(천원)", ["U01A111019000"])
    total_liabilities = g("sim__total_liabilities", ["raw__total_liabilities", "total_liabilities", "U01A800000000"])
    current_liabilities = g("sim__current_liabilities", ["raw__current_liabilities", "current_liabilities", "U01A810000000", "U01A820000000"])
    noncurrent_liabilities = g("balance_sheet__[U01A810000000]   비유동부채 (*)(IFRS)(천원)", ["U01A810000000"])
    short_debt = g("sim__short_term_debt", ["raw__short_debt", "short_debt", "U01A811026700", "U01A811027200", "U01A811037700"])
    current_ltd = g("balance_sheet__[U01A811027400]      유동성장기부채(*)(IFRS)(천원)", ["U01A811027400"])
    long_debt = g("sim__long_term_debt", ["raw__long_debt", "long_debt", "U01A811012800", "U01A811013300", "U01A811036800"])
    bonds = g("sim__bonds", ["raw__bond", "bond", "U01A811000000", "U01A811010500"])
    payables = g("sim__payables", ["raw__accounts_payable", "accounts_payable", "U01A811030800", "U01A811030700"])
    total_equity = g("sim__total_equity", ["raw__total_equity", "total_equity", "U01A600000000"])
    capital = g("balance_sheet__[U01A611000000]   자본금(*)(IFRS)(천원)", ["U01A611000000"])
    retained = g("sim__retained_earnings", ["raw__retained_earnings", "retained_earnings", "U01C500000000", "U01A617000000"])
    revenue = g("sim__revenue", ["raw__revenue", "revenue", "U01B100000000"])
    cogs = g("sim__cogs", ["raw__cogs", "cogs", "U01B200000000"])
    gross_profit = g("income_statement__[U01B201014400]매출총이익(손실)(IFRS)(천원)", ["U01B201014400"])
    sga = g("sim__sga", ["raw__sga", "sga", "U01B350000000"])
    op_income = g("sim__operating_income", ["raw__operating_income", "operating_income", "U01B400000000"])
    fin_cost = resolve_series(df, "financial_cost", next_state=next_state)
    pretax = g("income_statement__[U01B700000000]법인세비용차감전순이익(손실)(IFRS)(천원)", ["U01B700000000"])
    tax = g("income_statement__[U01B750000000]법인세비용(IFRS)(천원)", ["U01B750000000"])
    net_income = resolve_series(df, "net_income", next_state=next_state)
    comprehensive = g("income_statement__[U01B900000000]총포괄손익(IFRS)(천원)", ["U01B900000000"])
    depreciation = g("income_statement__[U01B350014100]   감가상각비(IFRS)(천원)", ["U01B350014100"])
    amortization = g("income_statement__[U01B350014300]   기타무형자산상각비(IFRS)(천원)", ["U01B350014300"])
    interest_income = g("income_statement__[U01B500010000]   이자수익(IFRS)(천원)", ["U01B500010000"])
    dividend_income = g("income_statement__[U01B500010100]   배당금수익(IFRS)(천원)", ["U01B500010100"])
    ocf = g("sim__operating_cf", ["raw__operating_cf", "operating_cf", "U01D100000000"])
    icf = g("cash_flow__[U01D200000000]투자활동으로 인한 현금흐름(*)(IFRS)(천원)", ["U01D200000000"])
    fcf = g("cash_flow__[U01D300000000]재무활동으로 인한 현금흐름(*)(IFRS)(천원)", ["U01D300000000"])
    capex = g("sim__capex", ["raw__capex", "capex", "U01D240000000", "U01D210000000"])
    end_capital = g("equity_change__[U01C100000000]기말자본금(*)(IFRS)(천원)", ["U01C100000000"])
    capital_surplus = g("equity_change__[U01C200000000]기말자본잉여금(*)(IFRS)(천원)", ["U01C200000000"])
    other_capital = g("equity_change__[U01C300000000]기말기타자본(*)(IFRS)(천원)", ["U01C300000000"])
    aoci = g("equity_change__[U01C400000000]기말기타포괄손익누계액(*)(IFRS)(천원)", ["U01C400000000"])
    end_retained = g("equity_change__[U01C500000000]기말이익잉여금(결손금)(*)(IFRS)(천원)", ["U01C500000000"])
    dividends = g("sim__cash_dividends", ["raw__cash_dividends", "cash_dividends", "U01F340000000"])

    # fallbacks for fundamental totals
    total_assets = total_assets.fillna(current_assets + noncurrent_assets).fillna(ppe + inventory + receivables + cash)
    total_liabilities = total_liabilities.fillna(current_liabilities + noncurrent_liabilities).fillna(short_debt + long_debt + bonds + payables)
    total_equity = total_equity.fillna(total_assets - total_liabilities)
    current_assets = current_assets.fillna(cash + sti + receivables + inventory)
    current_liabilities = current_liabilities.fillna(short_debt + current_ltd + payables)
    gross_profit = gross_profit.fillna(revenue - cogs)
    op_income = op_income.fillna(gross_profit - sga)
    net_income = net_income.fillna(pretax - tax).fillna(op_income - fin_cost - tax)
    ocf = ocf.fillna(net_income + depreciation + amortization)
    retained = retained.fillna(end_retained)
    capex = capex.fillna(ppe.diff() if len(ppe) else np.nan)

    return locals() | {"short_term_investments": sti, "operating_cf": ocf, "investing_cf": icf, "financing_cf": fcf, "cash_dividends": dividends}


def _materialize_feature_frame(df: pd.DataFrame, *, next_state: bool = False) -> pd.DataFrame:
    a = _base_accounts(df, next_state=next_state)
    out = pd.DataFrame(index=df.index)
    total_debt = a["short_debt"].fillna(0) + a["long_debt"].fillna(0) + a["bonds"].fillna(0)
    # Raw/sim block exact output names.
    mapping = {
        "sim__total_assets": a["total_assets"],
        "sim__current_assets": a["current_assets"],
        "balance_sheet__[U01A110000000]   비유동자산(*)(IFRS)(천원)": a["noncurrent_assets"],
        "sim__cash": a["cash"],
        "sim__short_term_investments": a["short_term_investments"],
        "sim__receivables": a["receivables"],
        "sim__inventory": a["inventory"],
        "sim__ppe": a["ppe"],
        "balance_sheet__[U01A111019000]      무형자산(*)(IFRS)(천원)": a["intangible"],
        "sim__total_liabilities": a["total_liabilities"],
        "sim__current_liabilities": a["current_liabilities"],
        "balance_sheet__[U01A810000000]   비유동부채 (*)(IFRS)(천원)": a["noncurrent_liabilities"],
        "sim__short_term_debt": a["short_debt"],
        "balance_sheet__[U01A811027400]      유동성장기부채(*)(IFRS)(천원)": a["current_ltd"],
        "sim__long_term_debt": a["long_debt"],
        "sim__bonds": a["bonds"],
        "sim__payables": a["payables"],
        "sim__total_equity": a["total_equity"],
        "balance_sheet__[U01A611000000]   자본금(*)(IFRS)(천원)": a["capital"],
        "sim__retained_earnings": a["retained"],
        "sim__revenue": a["revenue"],
        "sim__cogs": a["cogs"],
        "income_statement__[U01B201014400]매출총이익(손실)(IFRS)(천원)": a["gross_profit"],
        "sim__sga": a["sga"],
        "sim__operating_income": a["op_income"],
        "sim__financial_cost": a["fin_cost"],
        "income_statement__[U01B700000000]법인세비용차감전순이익(손실)(IFRS)(천원)": a["pretax"],
        "income_statement__[U01B750000000]법인세비용(IFRS)(천원)": a["tax"],
        "sim__net_income": a["net_income"],
        "income_statement__[U01B900000000]총포괄손익(IFRS)(천원)": a["comprehensive"],
        "income_statement__[U01B350014100]   감가상각비(IFRS)(천원)": a["depreciation"],
        "income_statement__[U01B350014300]   기타무형자산상각비(IFRS)(천원)": a["amortization"],
        "income_statement__[U01B500010000]   이자수익(IFRS)(천원)": a["interest_income"],
        "income_statement__[U01B500010100]   배당금수익(IFRS)(천원)": a["dividend_income"],
        "sim__operating_cf": a["operating_cf"],
        "cash_flow__[U01D200000000]투자활동으로 인한 현금흐름(*)(IFRS)(천원)": a["investing_cf"],
        "cash_flow__[U01D300000000]재무활동으로 인한 현금흐름(*)(IFRS)(천원)": a["financing_cf"],
        "sim__capex": a["capex"],
        "equity_change__[U01C100000000]기말자본금(*)(IFRS)(천원)": a["end_capital"],
        "equity_change__[U01C200000000]기말자본잉여금(*)(IFRS)(천원)": a["capital_surplus"],
        "equity_change__[U01C300000000]기말기타자본(*)(IFRS)(천원)": a["other_capital"],
        "equity_change__[U01C400000000]기말기타포괄손익누계액(*)(IFRS)(천원)": a["aoci"],
        "equity_change__[U01C500000000]기말이익잉여금(결손금)(*)(IFRS)(천원)": a["end_retained"],
        "sim__cash_dividends": a["cash_dividends"],
    }
    for k, v in mapping.items():
        out[k] = v
    out["derived__ppe_to_assets"] = _ratio(a["ppe"], a["total_assets"])
    out["derived__capex_to_revenue"] = _ratio(a["capex"], a["revenue"])
    out["derived__inventory_to_revenue"] = _ratio(a["inventory"], a["revenue"])
    out["derived__receivables_to_revenue"] = _ratio(a["receivables"], a["revenue"])
    out["derived__payables_to_revenue"] = _ratio(a["payables"], a["revenue"])
    out["derived__inventory_turnover_proxy"] = _ratio(a["cogs"], a["inventory"])
    out["derived__receivables_turnover_proxy"] = _ratio(a["revenue"], a["receivables"])
    out["derived__payables_turnover_proxy"] = _ratio(a["cogs"], a["payables"])
    out["derived__short_debt_to_total_debt"] = _ratio(a["short_debt"], total_debt)
    out["derived__long_debt_to_total_debt"] = _ratio(a["long_debt"], total_debt)
    out["derived__bond_to_total_debt"] = _ratio(a["bonds"], total_debt)
    out["derived__operating_margin"] = _ratio(a["op_income"], a["revenue"])
    out["derived__gross_margin"] = _ratio(a["gross_profit"], a["revenue"])
    out["derived__net_margin"] = _ratio(a["net_income"], a["revenue"])
    out["derived__roa_proxy"] = _ratio(a["net_income"], a["total_assets"])
    out["derived__cogs_to_revenue"] = _ratio(a["cogs"], a["revenue"])
    out["derived__sga_to_revenue"] = _ratio(a["sga"], a["revenue"])
    out["derived__financial_cost_to_revenue"] = _ratio(a["fin_cost"], a["revenue"])
    out["derived__financial_cost_to_debt"] = _ratio(a["fin_cost"], total_debt)
    out["derived__debt_to_assets"] = _ratio(a["total_liabilities"], a["total_assets"])
    out["derived__equity_to_assets"] = _ratio(a["total_equity"], a["total_assets"])
    out["derived__current_liabilities_to_assets"] = _ratio(a["current_liabilities"], a["total_assets"])
    out["derived__borrowings_to_assets"] = _ratio(total_debt, a["total_assets"])
    out["derived__current_ratio"] = _ratio(a["current_assets"], a["current_liabilities"])
    out["derived__cash_ratio"] = _ratio(a["cash"] + a["short_term_investments"].fillna(0), a["current_liabilities"])
    out["derived__cash_to_current_liabilities"] = _ratio(a["cash"], a["current_liabilities"])
    out["derived__cash_to_short_debt"] = _ratio(a["cash"], a["short_debt"])
    out["derived__ocf_to_total_debt"] = _ratio(a["operating_cf"], total_debt)
    out["derived__interest_coverage_proxy"] = _ratio(a["op_income"], a["fin_cost"].abs())
    out["derived__retained_earnings_to_assets"] = _ratio(a["retained"], a["total_assets"])
    out["derived__retained_earnings_negative_flag"] = (a["retained"] < 0).astype(float)
    out["derived__dividend_to_net_income"] = _ratio(a["cash_dividends"].abs(), a["net_income"].abs())
    return out


def _add_deltas_and_trends(panel: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    out = feat.copy()
    trend_bases = [
        "derived__operating_margin", "derived__cogs_to_revenue", "derived__sga_to_revenue",
        "derived__financial_cost_to_revenue", "derived__debt_to_assets", "derived__current_ratio",
        "derived__roa_proxy", "derived__receivables_turnover_proxy", "derived__payables_turnover_proxy",
        "derived__inventory_turnover_proxy", "derived__short_debt_to_total_debt", "derived__long_debt_to_total_debt",
        "derived__bond_to_total_debt", "derived__cash_to_short_debt", "derived__ocf_to_total_debt",
    ]
    if {"firm_id", "fiscal_year"}.issubset(panel.columns):
        tmp = feat[trend_bases].copy()
        tmp["_firm_id"] = panel["firm_id"].astype(str).to_numpy()
        tmp["_fiscal_year"] = pd.to_numeric(panel["fiscal_year"], errors="coerce").to_numpy()
        tmp["_orig_idx"] = np.arange(len(tmp))
        tmp = tmp.sort_values(["_firm_id", "_fiscal_year", "_orig_idx"])
        prev_sorted = tmp.groupby("_firm_id", sort=False)[trend_bases].shift(1)
        prev = pd.DataFrame(index=feat.index, columns=trend_bases, dtype="float64")
        prev.iloc[tmp["_orig_idx"].to_numpy()] = prev_sorted.to_numpy()
    else:
        prev = feat[trend_bases].shift(1)
    for b in trend_bases:
        d = feat[b] - prev[b]
        out[f"delta_1y__{b}"] = d
        scale = feat[b].abs().rolling(3, min_periods=1).median() if len(feat) else 1.0
        out[f"trend_1y__{b}"] = d / scale.where(scale.abs() > EPS)
    return out


def _add_peer_percentiles(panel: pd.DataFrame, feat: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    out = feat.copy()
    peer_bases = [
        "derived__operating_margin", "derived__cogs_to_revenue", "derived__sga_to_revenue", "derived__financial_cost_to_revenue",
        "derived__debt_to_assets", "derived__current_ratio", "derived__roa_proxy", "derived__receivables_turnover_proxy",
        "derived__payables_turnover_proxy", "derived__inventory_turnover_proxy", "derived__short_debt_to_total_debt",
        "derived__long_debt_to_total_debt", "derived__bond_to_total_debt", "derived__cash_to_short_debt", "derived__ocf_to_total_debt",
    ]
    ref = reference if reference is not None else feat
    for b in peer_bases:
        vals = pd.to_numeric(ref[b], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().sort_values().to_numpy()
        cur = pd.to_numeric(feat[b], errors="coerce").to_numpy(dtype=float)
        if len(vals) == 0:
            pct = np.full(len(feat), 0.5)
        else:
            pct = np.searchsorted(vals, cur, side="right") / max(len(vals), 1)
            pct[~np.isfinite(cur)] = np.nan
        out[f"peer_pct__{b}"] = pd.Series(pct, index=feat.index).fillna(0.5)
    return out



def _compute_transition_proximity_features(panel: pd.DataFrame, feat: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministically derive TRANSITION_PROXIMITY features from transition signals.

    This avoids the old final-run sentinel constants.  It uses only row-local and
    historical transition columns already present in the Stage2 transition panel.
    No oracle score or eval distribution is consumed here.
    """
    n = len(feat)
    out = pd.DataFrame(index=feat.index)
    reward_like = None
    for c in ["reward_raw_notch", "reward_raw", "reward_original", "rating_delta", "notch_delta"]:
        if c in panel.columns:
            reward_like = pd.to_numeric(panel[c], errors="coerce")
            break
    if reward_like is None:
        # Fall back to profitability/leverage trend geometry when explicit rating
        # transition labels are not available in the broad panel.  This is still
        # computed from financial state features, not a constant smoke sentinel.
        _zero = pd.Series(0.0, index=feat.index)
        pos_signal = (
            pd.to_numeric(feat.get("delta_1y__derived__roa_proxy", _zero), errors="coerce").fillna(0.0)
            - pd.to_numeric(feat.get("delta_1y__derived__debt_to_assets", _zero), errors="coerce").fillna(0.0)
            - pd.to_numeric(feat.get("delta_1y__derived__financial_cost_to_revenue", _zero), errors="coerce").fillna(0.0)
        )
    else:
        # Better rating movement is positive reward in the final reward contract.
        pos_signal = reward_like.fillna(0.0)
    x = pd.to_numeric(pos_signal, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    if n == 0:
        for c in [
            "transition__distance_to_positive_transition_origin",
            "transition__distance_to_anti_transition_origin",
            "transition__positive_transition_prior_score",
            "transition__anti_transition_risk_score",
            "transition__nearest_positive_cluster_id",
            "transition__nearest_anti_cluster_id",
        ]:
            out[c] = []
        return out, {"transition_proximity_status": "computed", "n_rows": 0}
    pos_vals = x[x > 0]
    anti_vals = x[x < 0]
    pos_origin = float(np.median(pos_vals)) if len(pos_vals) else float(np.quantile(x, 0.75))
    anti_origin = float(np.median(anti_vals)) if len(anti_vals) else float(np.quantile(x, 0.25))
    scale = float(np.nanstd(x)) if np.isfinite(np.nanstd(x)) and np.nanstd(x) > EPS else 1.0
    dist_pos = np.abs(x - pos_origin) / scale
    dist_anti = np.abs(x - anti_origin) / scale
    prior = 1.0 / (1.0 + np.exp(-(x - (pos_origin + anti_origin) / 2.0) / scale))
    out["transition__distance_to_positive_transition_origin"] = dist_pos
    out["transition__distance_to_anti_transition_origin"] = dist_anti
    out["transition__positive_transition_prior_score"] = np.clip(prior, 0.0, 1.0)
    out["transition__anti_transition_risk_score"] = np.clip(1.0 - prior, 0.0, 1.0)
    out["transition__nearest_positive_cluster_id"] = (dist_pos <= dist_anti).astype(float)
    out["transition__nearest_anti_cluster_id"] = (dist_anti < dist_pos).astype(float)
    return out, {
        "transition_proximity_status": "computed",
        "method": "deterministic_1d_transition_signal_prototypes",
        "positive_origin": pos_origin,
        "anti_origin": anti_origin,
        "scale": scale,
        "n_rows": int(n),
        "n_positive_signal": int((x > 0).sum()),
        "n_anti_signal": int((x < 0).sum()),
    }

def enrich_avs256_panel(
    df: pd.DataFrame,
    *,
    reference_for_peer: pd.DataFrame | None = None,
    require_next_phi_critical: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    for c in CATEGORICAL_COLUMNS:
        if c not in out.columns:
            if c == "market":
                out[c] = "UNKNOWN"
            else:
                for cand in ["sector_7", "sector", "industry", "업종", "표준산업분류"]:
                    if cand in out.columns:
                        out[c] = out[cand].astype(str); break
                else:
                    out[c] = "UNKNOWN"
        out[c] = out[c].fillna("UNKNOWN").astype(str)

    feat = _materialize_feature_frame(out, next_state=False)
    feat = _add_deltas_and_trends(out, feat)
    feat = _add_peer_percentiles(out, feat, reference_for_peer)
    year = pd.to_numeric(out.get("fiscal_year", pd.Series(np.nan, index=out.index)), errors="coerce")
    ymin, ymax = year.min(skipna=True), year.max(skipna=True)
    denom = (float(ymax) - float(ymin) + 1.0) if pd.notna(ymin) and pd.notna(ymax) else 1.0
    feat["context__year_normalized"] = ((year - float(ymin)) / max(denom, EPS)).fillna(0.0)
    feat["context__market_code"] = _market_code(out["market"])
    transition_feat, transition_meta = _compute_transition_proximity_features(out, feat)
    for _c in transition_feat.columns:
        feat[_c] = transition_feat[_c]

    next_feat = _materialize_feature_frame(out, next_state=True)
    next_feat = _add_deltas_and_trends(out, next_feat)
    next_feat = _add_peer_percentiles(out, next_feat, reference_for_peer)
    next_feat["context__year_normalized"] = feat["context__year_normalized"]
    next_feat["context__market_code"] = feat["context__market_code"]
    for c in ["transition__distance_to_positive_transition_origin", "transition__distance_to_anti_transition_origin", "transition__positive_transition_prior_score", "transition__anti_transition_risk_score", "transition__nearest_positive_cluster_id", "transition__nearest_anti_cluster_id"]:
        next_feat[c] = feat[c]

    # Phi-critical income-statement guard.
    # Current-state profitability/financial-cost accounts are required for every
    # enriched panel.  Next-state accounts are required only for transition
    # panels.  Serving/eval state panels intentionally have no next__ columns;
    # requiring next__sim__* there would be a false failure and would not imply
    # a Stage2A carry bug.
    phi_critical_pre_impute = {
        "sim__net_income": feat.get("sim__net_income"),
        "sim__financial_cost": feat.get("sim__financial_cost"),
    }
    if require_next_phi_critical:
        phi_critical_pre_impute.update({
            "next__sim__net_income": next_feat.get("sim__net_income"),
            "next__sim__financial_cost": next_feat.get("sim__financial_cost"),
        })
    phi_critical_all_nan = [
        c for c, ser in phi_critical_pre_impute.items()
        if ser is None or pd.to_numeric(ser, errors="coerce").replace([np.inf, -np.inf], np.nan).notna().sum() == 0
    ]
    if phi_critical_all_nan:
        scope = "current_and_next" if require_next_phi_critical else "current_only"
        raise ValueError(
            "Phi-critical income-statement accounts are entirely missing before imputation; "
            f"silent zero-fill is forbidden: {phi_critical_all_nan}. "
            f"Guard scope={scope}. "
            "Regenerate Stage2A raw action source with financial_cost/net_income carry enabled; "
            "next-state carry is required for transition panels but not for phase_eval state-only panels."
        )

    missing_created = []
    for c in CONTINUOUS_COLUMNS:
        if c not in feat.columns:
            feat[c] = np.nan; missing_created.append(c)
        out[c] = _safe_fill_numeric(feat[c], default=0.0)
    for c in ACD_TARGET_COLUMNS:
        nc = f"next__{c}"
        out[nc] = _safe_fill_numeric(next_feat[c] if c in next_feat.columns else feat[c], default=0.0)

    phi_critical_diagnostics = {}
    for c in [
        "sim__net_income",
        "sim__financial_cost",
        "sim__operating_cf",
        "derived__roa_proxy",
        "derived__financial_cost_to_revenue",
        "derived__ocf_to_total_debt",
        "next__sim__net_income",
        "next__sim__financial_cost",
        "next__derived__roa_proxy",
        "next__derived__financial_cost_to_revenue",
    ]:
        if c in out.columns:
            x = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            source = next_feat if c.startswith("next__") else feat
            base_c = c[len("next__"):] if c.startswith("next__") else c
            phi_critical_diagnostics[c] = {
                "non_null_rate": float(x.notna().mean()) if len(x) else 0.0,
                "nonzero_rate": float((x.fillna(0.0).abs() > EPS).mean()) if len(x) else 0.0,
                "all_missing_before_impute": bool(base_c in source.columns and pd.to_numeric(source[base_c], errors="coerce").notna().sum() == 0),
            }
    meta = {
        "schema_version": "stage2_avs256_enrichment_v2",
        "n_continuous_features": len(CONTINUOUS_COLUMNS),
        "n_categorical_fields": len(CATEGORICAL_COLUMNS),
        "n_acd_targets": len(ACD_TARGET_COLUMNS),
        "transition_proximity_status": transition_meta.get("transition_proximity_status", "computed"),
        "transition_proximity_metadata": transition_meta,
        "missing_contract_columns_created_with_imputation": missing_created,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "phi_critical_account_diagnostics": phi_critical_diagnostics,
        "phi_critical_alias_policy": "account_registry_resolver_for_net_income_financial_cost_plus_existing_cf_join",
        "phi_critical_all_nan_guard": "hard_fail_before_imputation_current_net_income_financial_cost_plus_optional_next_guard",
        "require_next_phi_critical": bool(require_next_phi_critical),
    }
    return out, meta


def write_feature_manifest(path: Path) -> None:
    from credit_recourse.rl.common.io import write_json
    write_json(path, build_feature_manifest())
