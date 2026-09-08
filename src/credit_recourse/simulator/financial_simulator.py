"""
FinancialSimulator — 1-year forward financial projection.

Flow (Demian's spec):
  Step 1: 손익 → 영업활동현금흐름 (간접법) + 운전자본 변동
  Step 2: 투자활동현금흐름 + 재무활동현금흐름 (action 효과 포함)
  Step 3: 기말현금 + t+1 재무상태표 closing + 항등식 plug + sustainability check

References:
- Driver-based forecasting (Koller et al., Damodaran)
- IAS 7 cash flow structure
- K-IFRS reporting line item conventions
"""
from dataclasses import dataclass
from typing import Optional, Literal, Dict
import math

from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.business_plan import BusinessPlan
from credit_recourse.simulator.action import Action, clip_action


SustainabilityFlag = Literal["ok", "fragile", "critical"]


@dataclass
class SimulationResult:
    state_t1: FirmState
    diagnostics: Dict
    sustainability: SustainabilityFlag
    plug_used: str  # "cash" | "short_debt" | "both"
    plug_amount: float
    accounting_check: Dict


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_get(value, default=0.0):
    """None을 default로 변환."""
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def _avg(a, b):
    return (a + b) / 2.0


# ---------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------
class FinancialSimulator:
    """
    1-year forward simulator.

    Usage:
        sim = FinancialSimulator()
        result = sim.simulate(state_t, business_plan, action)
        # result.state_t1, result.diagnostics, result.sustainability, ...
    """

    def __init__(
        self,
        plug_priority: str = "cash",  # "cash" | "short_debt"
        sustainability_thresholds: Optional[Dict] = None,
        preserve_current_non_current_residual: bool = False,
    ):
        self.plug_priority = plug_priority
        self.preserve_current_non_current_residual = bool(preserve_current_non_current_residual)
        self.sustainability_thresholds = sustainability_thresholds or {
            "cash_to_cogs_ratio": 0.05,   # cash < 5% of COGS = fragile
            "current_ratio": 1.0,         # 유동비율 < 1 = fragile
            "interest_coverage": 1.0,     # 영업이익 / 이자비용 < 1 = fragile
        }

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def simulate(
        self,
        state_t: FirmState,
        bp: BusinessPlan,
        action: Action,
    ) -> SimulationResult:
        # 0. Apply action perturbations to business plan
        bp_eff = self._apply_action_to_plan(bp, action)
        action_clipped = clip_action(action)

        # 1. Income statement
        is_t1 = self._step1_income_statement(state_t, bp_eff, action_clipped)

        # 2. Operating cash flow (indirect method)
        ocf_components = self._step2_operating_cf(state_t, bp_eff, is_t1)

        # 3. Investing cash flow
        icf_components = self._step3_investing_cf(state_t, bp_eff, action_clipped)

        # 4. Financing cash flow
        fcf_components = self._step4_financing_cf(
            state_t, bp_eff, action_clipped, is_t1
        )

        # 5. Balance sheet closing + plug
        bs_t1, plug_used, plug_amount = self._step5_balance_sheet_closing(
            state_t, bp_eff, action_clipped,
            is_t1, ocf_components, icf_components, fcf_components,
        )

        # 6. Build t+1 FirmState
        state_t1 = self._build_state_t1(
            state_t, is_t1, ocf_components, icf_components, fcf_components, bs_t1
        )

        # 7. Sustainability + accounting checks
        sustainability = self._classify_sustainability(state_t1, is_t1)
        accounting = state_t1.accounting_identity_check(tol=10.0)

        diagnostics = {
            "is": is_t1,
            "ocf_components": ocf_components,
            "icf_components": icf_components,
            "fcf_components": fcf_components,
            "bp_effective": bp_eff.to_dict(),
            "action_clipped": action_clipped.to_dict(),
            "balance_sheet_closing": bs_t1,
            "residual_audit": bs_t1.get("residual_audit", {}),
        }
        return SimulationResult(
            state_t1=state_t1,
            diagnostics=diagnostics,
            sustainability=sustainability,
            plug_used=plug_used,
            plug_amount=plug_amount,
            accounting_check=accounting,
        )

    # ------------------------------------------------------------------
    # Step 0: Action → BusinessPlan perturbation
    # ------------------------------------------------------------------
    def _apply_action_to_plan(self, bp: BusinessPlan, action: Action) -> BusinessPlan:
        """Tier 2 actions override BusinessPlan parameters."""
        a = clip_action(action)
        bp_eff = BusinessPlan(**bp.to_dict())
        bp_eff.revenue_growth = bp.revenue_growth + a.revenue_growth
        bp_eff.cogs_ratio = bp.cogs_ratio + a.cogs_ratio_chg
        bp_eff.sga_ratio = bp.sga_ratio + a.sga_ratio_chg

        # Tier 1 turnover changes apply directly to BP
        bp_eff.ar_turnover = max(0.5, bp.ar_turnover + a.ar_turnover_chg)
        bp_eff.inv_turnover = max(0.5, bp.inv_turnover + a.inv_turnover_chg)
        bp_eff.ap_turnover = max(0.5, bp.ap_turnover + a.ap_turnover_chg)
        return bp_eff

    # ------------------------------------------------------------------
    # Step 1: Income Statement
    # ------------------------------------------------------------------
    def _step1_income_statement(
        self,
        state_t: FirmState,
        bp: BusinessPlan,
        action: Action,
    ) -> Dict:
        rev_t = _safe_get(state_t.revenue)
        revenue_t1 = rev_t * (1.0 + bp.revenue_growth)

        cogs_t1 = revenue_t1 * bp.cogs_ratio
        sga_t1 = revenue_t1 * bp.sga_ratio
        gross_profit = revenue_t1 - cogs_t1
        operating_income = gross_profit - sga_t1

        # Depreciation & Amortization (on t값 + action 효과 미반영)
        ppe_t = _safe_get(state_t.ppe)
        intang_t = _safe_get(state_t.intangibles)
        # 새 PP&E 평균잔액 (Decision 17: 기중 평균)
        new_ppe_action = ppe_t * action.ppe_pct
        avg_ppe = ppe_t + new_ppe_action / 2.0  # 기중 평균
        depreciation = max(0.0, avg_ppe * bp.depreciation_rate)
        amortization = max(0.0, intang_t * bp.amortization_rate)

        # Interest expense (이자비용; financial_cost proxy)
        # 기중 평균 차입금 사용
        sd_t = _safe_get(state_t.short_term_debt)
        ld_t = _safe_get(state_t.long_term_debt)
        bd_t = _safe_get(state_t.bonds)
        sd_t1_target = sd_t * (1.0 + action.short_debt_pct)
        ld_t1_target = ld_t * (1.0 + action.long_debt_pct)
        bd_t1_target = bd_t * (1.0 + action.bond_pct)

        avg_short = _avg(sd_t, sd_t1_target)
        avg_long = _avg(ld_t, ld_t1_target)
        avg_bond = _avg(bd_t, bd_t1_target)

        interest_expense = (
            avg_short * bp.rate_short
            + avg_long * bp.rate_long
            + avg_bond * bp.rate_bond
        )

        # 영업외수익
        non_op_income = revenue_t1 * bp.non_op_income_ratio

        pretax = operating_income - interest_expense + non_op_income
        # 손실시 법인세 = 0 (단순화; 이연법인세 무시)
        tax = max(0.0, pretax) * bp.tax_rate
        net_income = pretax - tax

        # 배당 (손실시 0, 이익일 때만 payout ratio 적용)
        dividends = max(0.0, net_income) * bp.dividend_payout

        return {
            "revenue": revenue_t1,
            "cogs": cogs_t1,
            "sga": sga_t1,
            "gross_profit": gross_profit,
            "operating_income": operating_income,
            "depreciation": depreciation,
            "amortization": amortization,
            "interest_expense": interest_expense,
            "non_op_income": non_op_income,
            "pretax_income": pretax,
            "tax_expense": tax,
            "net_income": net_income,
            "cash_dividends": dividends,
            # action-effected debt targets (used in Step 5)
            "_short_debt_t1": sd_t1_target,
            "_long_debt_t1": ld_t1_target,
            "_bond_t1": bd_t1_target,
            "_ppe_action_delta": new_ppe_action,
        }

    # ------------------------------------------------------------------
    # Step 2: Operating CF (indirect method)
    # ------------------------------------------------------------------
    def _step2_operating_cf(
        self,
        state_t: FirmState,
        bp: BusinessPlan,
        is_t1: Dict,
    ) -> Dict:
        # Working capital t+1 (회전율 기반 역산)
        revenue_t1 = is_t1["revenue"]
        cogs_t1 = is_t1["cogs"]

        ar_t1 = revenue_t1 / bp.ar_turnover if bp.ar_turnover > 0 else 0.0
        inv_t1 = cogs_t1 / bp.inv_turnover if bp.inv_turnover > 0 else 0.0
        ap_t1 = cogs_t1 / bp.ap_turnover if bp.ap_turnover > 0 else 0.0

        ar_change = ar_t1 - _safe_get(state_t.receivables)
        inv_change = inv_t1 - _safe_get(state_t.inventory)
        ap_change = ap_t1 - _safe_get(state_t.payables)

        # 운전자본 효과 (자산 ↑ → CF ↓, 부채 ↑ → CF ↑)
        wc_change = -(ar_change + inv_change) + ap_change

        operating_cf = (
            is_t1["net_income"]
            + is_t1["depreciation"]
            + is_t1["amortization"]
            + wc_change
        )

        return {
            "operating_cf": operating_cf,
            "ar_t1": ar_t1,
            "inv_t1": inv_t1,
            "ap_t1": ap_t1,
            "ar_change": ar_change,
            "inv_change": inv_change,
            "ap_change": ap_change,
            "wc_change": wc_change,
        }

    # ------------------------------------------------------------------
    # Step 3: Investing CF
    # ------------------------------------------------------------------
    def _step3_investing_cf(
        self,
        state_t: FirmState,
        bp: BusinessPlan,
        action: Action,
    ) -> Dict:
        rev_t = _safe_get(state_t.revenue)
        # Default capex (firm 과거 평균)
        default_capex = rev_t * bp.capex_to_revenue
        # Action 효과: PP&E 변경 = (default_capex - depreciation) + action delta
        ppe_t = _safe_get(state_t.ppe)
        action_capex = ppe_t * action.ppe_pct  # 양수: 매입, 음수: 매각
        total_capex = default_capex + action_capex  # 명목 capex (감가상각 제외)

        investing_cf = -total_capex  # 매입은 현금 유출
        # 매각인 경우 (action_capex < 0이고 절대값이 default_capex보다 큰 경우)
        # 단순화: investing_cf = -(default_capex + action_capex)
        # action_capex가 음수면 자동으로 inflow 됨

        return {
            "investing_cf": investing_cf,
            "default_capex": default_capex,
            "action_capex": action_capex,
            "total_capex": total_capex,
        }

    # ------------------------------------------------------------------
    # Step 4: Financing CF
    # ------------------------------------------------------------------
    def _step4_financing_cf(
        self,
        state_t: FirmState,
        bp: BusinessPlan,
        action: Action,
        is_t1: Dict,
    ) -> Dict:
        sd_t = _safe_get(state_t.short_term_debt)
        ld_t = _safe_get(state_t.long_term_debt)
        bd_t = _safe_get(state_t.bonds)
        cpld_t = _safe_get(state_t.current_portion_long_debt)

        sd_change = is_t1["_short_debt_t1"] - sd_t
        ld_change = is_t1["_long_debt_t1"] - ld_t
        bd_change = is_t1["_bond_t1"] - bd_t

        # 유동성장기부채(cpld) 만기 상환: t년 cpld는 cash 유출
        cpld_repayment = -cpld_t  # cash 유출

        dividends = is_t1["cash_dividends"]

        financing_cf = (
            sd_change + ld_change + bd_change + cpld_repayment - dividends
        )

        return {
            "financing_cf": financing_cf,
            "sd_change": sd_change,
            "ld_change": ld_change,
            "bd_change": bd_change,
            "cpld_repayment": cpld_repayment,
            "dividends": dividends,
        }

    # ------------------------------------------------------------------
    # Step 5: Balance sheet closing + plug
    # ------------------------------------------------------------------
    def _step5_balance_sheet_closing(
        self,
        state_t: FirmState,
        bp: BusinessPlan,
        action: Action,
        is_t1: Dict,
        ocf_comp: Dict,
        icf_comp: Dict,
        fcf_comp: Dict,
    ) -> tuple:
        # Cash
        cash_t = _safe_get(state_t.cash)
        cash_t1 = (
            cash_t
            + ocf_comp["operating_cf"]
            + icf_comp["investing_cf"]
            + fcf_comp["financing_cf"]
        )

        # PP&E
        ppe_t = _safe_get(state_t.ppe)
        ppe_t1 = (
            ppe_t
            + icf_comp["total_capex"]
            - is_t1["depreciation"]
        )
        ppe_t1 = max(0.0, ppe_t1)

        # Intangibles (action 없음, 자연 감소)
        intang_t = _safe_get(state_t.intangibles)
        intang_t1 = max(0.0, intang_t - is_t1["amortization"])

        # Working capital
        ar_t1 = ocf_comp["ar_t1"]
        inv_t1 = ocf_comp["inv_t1"]
        ap_t1 = ocf_comp["ap_t1"]

        # Debt
        sd_t1 = max(0.0, is_t1["_short_debt_t1"])
        ld_t1 = max(0.0, is_t1["_long_debt_t1"])
        bd_t1 = max(0.0, is_t1["_bond_t1"])

        # 유동성장기부채 reclass: t+1 = t+1 장기 × reclass_rate
        cpld_t1 = ld_t1 * bp.reclass_rate
        ld_t1_net = ld_t1 - cpld_t1  # 비유동 부분만

        # Other items (변화 없음 가정)
        sti_t1 = _safe_get(state_t.short_term_investments)

        if not self.preserve_current_non_current_residual:
            # Legacy/default presentation closing.  This deliberately preserves
            # the no-new-flag/no-behavior-change contract: residual assets are
            # pooled into non-current assets and residual liabilities are kept as
            # an audit/other-liabilities line without being allocated to current
            # or non-current liabilities.  Enable
            # preserve_current_non_current_residual=True to use the corrected
            # current/non-current residual split.
            # IMPORTANT: keep the default branch byte/semantic-compatible with
            # the legacy simulator used by the frozen baseline.  Do not clamp
            # negative residuals here.  The corrected preserve=True branch below
            # performs clipping and audit-flagging explicitly.
            other_assets_t = (_safe_get(state_t.total_assets)
                - _safe_get(state_t.cash)
                - _safe_get(state_t.short_term_investments)
                - _safe_get(state_t.receivables)
                - _safe_get(state_t.inventory)
                - _safe_get(state_t.ppe)
                - _safe_get(state_t.intangibles))
            other_liab_t = (_safe_get(state_t.total_liabilities)
                - _safe_get(state_t.short_term_debt)
                - _safe_get(state_t.current_portion_long_debt)
                - _safe_get(state_t.long_term_debt)
                - _safe_get(state_t.bonds)
                - _safe_get(state_t.payables))

            cap_stock_t = _safe_get(state_t.capital_stock)
            cs_val = state_t.capital_surplus
            cap_surplus_t = cs_val if cs_val is not None else 0.0
            re_t = _safe_get(state_t.retained_earnings)
            oth_eq_t = _safe_get(state_t.total_equity) - cap_stock_t - cap_surplus_t - re_t
            re_t1 = re_t + is_t1["net_income"] - is_t1["cash_dividends"]
            equity_t1 = cap_stock_t + cap_surplus_t + re_t1 + oth_eq_t

            assets_pre_plug = cash_t1 + sti_t1 + ar_t1 + inv_t1 + ppe_t1 + intang_t1 + other_assets_t
            liab_t1 = sd_t1 + cpld_t1 + ld_t1_net + bd_t1 + ap_t1 + other_liab_t
            gap = assets_pre_plug - (liab_t1 + equity_t1)
            plug_used = "cash"
            plug_amount = gap
            cash_t1_plugged = cash_t1 - gap
            if cash_t1_plugged < 0:
                shortfall = -cash_t1_plugged
                cash_t1_plugged = 0.0
                sd_t1 += shortfall
                plug_used = "both"

            assets_final = cash_t1_plugged + sti_t1 + ar_t1 + inv_t1 + ppe_t1 + intang_t1 + other_assets_t
            liab_final = sd_t1 + cpld_t1 + ld_t1_net + bd_t1 + ap_t1 + other_liab_t
            bs = {
                "cash": cash_t1_plugged,
                "short_term_investments": sti_t1,
                "receivables": ar_t1,
                "inventory": inv_t1,
                "ppe": ppe_t1,
                "intangibles": intang_t1,
                "other_assets": other_assets_t,
                "other_current_assets": 0.0,
                "other_non_current_assets": other_assets_t,
                "total_assets": assets_final,
                "current_assets": cash_t1_plugged + sti_t1 + ar_t1 + inv_t1,
                "non_current_assets": ppe_t1 + intang_t1 + other_assets_t,
                "short_term_debt": sd_t1,
                "current_portion_long_debt": cpld_t1,
                "long_term_debt": ld_t1_net,
                "bonds": bd_t1,
                "payables": ap_t1,
                "other_liabilities": other_liab_t,
                "other_current_liabilities": 0.0,
                "other_non_current_liabilities": other_liab_t,
                "total_liabilities": liab_final,
                "current_liabilities": sd_t1 + cpld_t1 + ap_t1,
                "non_current_liabilities": ld_t1_net + bd_t1,
                "capital_stock": cap_stock_t,
                "capital_surplus": cap_surplus_t,
                "retained_earnings": re_t1,
                "other_equity": oth_eq_t,
                "total_equity": equity_t1,
                "residual_audit": {
                    "presentation_residuals_preserved": False,
                    "residual_fallback_used": True,
                    "other_current_assets_raw": 0.0,
                    "other_non_current_assets_raw": other_assets_t,
                    "other_current_liabilities_raw": 0.0,
                    "other_non_current_liabilities_raw": other_liab_t,
                    "other_current_assets_clipped": False,
                    "other_non_current_assets_clipped": False,
                    "other_current_liabilities_clipped": False,
                    "other_non_current_liabilities_clipped": False,
                    # Default/legacy branch does not clamp negative residuals,
                    # but still records their presence for audit.  This keeps
                    # no-flag simulation levels legacy-compatible while avoiding
                    # silent diagnostic loss.
                    "residual_negative_flag": bool(other_assets_t < 0 or other_liab_t < 0),
                },
            }
            return bs, plug_used, plug_amount

        # Preserve current / non-current presentation residuals separately.
        # Earlier versions pooled all residual assets into non-current assets and
        # pooled residual liabilities only into total liabilities.  That preserved
        # A=L+E but distorted current_assets/current_liabilities denominators
        # used by liquidity ratios such as R136 = payables/current_liabilities.
        def _residual_nonnegative(raw: float) -> tuple[float, bool, float]:
            raw = float(raw) if math.isfinite(float(raw)) else 0.0
            return max(0.0, raw), bool(raw < -1e-9), raw

        other_current_assets_t, oca_clipped, oca_raw = _residual_nonnegative(
            _safe_get(state_t.current_assets)
            - _safe_get(state_t.cash)
            - _safe_get(state_t.short_term_investments)
            - _safe_get(state_t.receivables)
            - _safe_get(state_t.inventory)
        )
        other_non_current_assets_t, onca_clipped, onca_raw = _residual_nonnegative(
            _safe_get(state_t.non_current_assets)
            - _safe_get(state_t.ppe)
            - _safe_get(state_t.intangibles)
        )
        other_current_liabilities_t, ocl_clipped, ocl_raw = _residual_nonnegative(
            _safe_get(state_t.current_liabilities)
            - _safe_get(state_t.short_term_debt)
            - _safe_get(state_t.current_portion_long_debt)
            - _safe_get(state_t.payables)
        )
        other_non_current_liabilities_t, oncl_clipped, oncl_raw = _residual_nonnegative(
            _safe_get(state_t.non_current_liabilities)
            - _safe_get(state_t.long_term_debt)
            - _safe_get(state_t.bonds)
        )

        # Some legacy rows carry only total_assets / total_liabilities but not
        # current/non-current subtotals.  In that case, fall back to the prior
        # total residual while still exposing an explicit audit flag.
        residual_fallback_used = False
        if _safe_get(state_t.current_assets) == 0.0 and _safe_get(state_t.non_current_assets) == 0.0 and _safe_get(state_t.total_assets) != 0.0:
            residual_fallback_used = True
            pooled_assets = (
                _safe_get(state_t.total_assets)
                - _safe_get(state_t.cash)
                - _safe_get(state_t.short_term_investments)
                - _safe_get(state_t.receivables)
                - _safe_get(state_t.inventory)
                - _safe_get(state_t.ppe)
                - _safe_get(state_t.intangibles)
            )
            other_non_current_assets_t, onca_clipped, onca_raw = _residual_nonnegative(pooled_assets)
        if _safe_get(state_t.current_liabilities) == 0.0 and _safe_get(state_t.non_current_liabilities) == 0.0 and _safe_get(state_t.total_liabilities) != 0.0:
            residual_fallback_used = True
            pooled_liab = (
                _safe_get(state_t.total_liabilities)
                - _safe_get(state_t.short_term_debt)
                - _safe_get(state_t.current_portion_long_debt)
                - _safe_get(state_t.long_term_debt)
                - _safe_get(state_t.bonds)
                - _safe_get(state_t.payables)
            )
            other_non_current_liabilities_t, oncl_clipped, oncl_raw = _residual_nonnegative(pooled_liab)

        # Equity
        cap_stock_t = _safe_get(state_t.capital_stock)
        cap_surplus_t = _safe_get(state_t.capital_surplus, 0.0) if state_t.capital_surplus is not None else 0.0
        # capital_surplus property handles None → use safe access
        cs_val = state_t.capital_surplus
        cap_surplus_t = cs_val if cs_val is not None else 0.0
        re_t = _safe_get(state_t.retained_earnings)
        oth_eq_t = _safe_get(state_t.total_equity) - cap_stock_t - cap_surplus_t - re_t

        re_t1 = re_t + is_t1["net_income"] - is_t1["cash_dividends"]
        # 자본금/자본잉여금/기타자본 변경 없음 (action 제외 결정)
        equity_t1 = cap_stock_t + cap_surplus_t + re_t1 + oth_eq_t

        current_assets_pre = cash_t1 + sti_t1 + ar_t1 + inv_t1 + other_current_assets_t
        non_current_assets_pre = ppe_t1 + intang_t1 + other_non_current_assets_t
        current_liabilities_pre = sd_t1 + cpld_t1 + ap_t1 + other_current_liabilities_t
        non_current_liabilities_pre = ld_t1_net + bd_t1 + other_non_current_liabilities_t

        # Pre-plug totals
        assets_pre_plug = current_assets_pre + non_current_assets_pre
        liab_t1 = current_liabilities_pre + non_current_liabilities_pre

        gap = assets_pre_plug - (liab_t1 + equity_t1)
        # gap > 0: 자산이 더 많음 → 자본/부채에 plug or 자산 차감
        # gap < 0: 자산이 부족 → 자산 추가 or 부채/자본 차감
        # 단순화: cash로 자산 측 조정 (gap만큼 cash 감소 → 자산 = 부채 + 자본)

        plug_used = "cash"
        plug_amount = gap
        cash_t1_plugged = cash_t1 - gap

        # Cash 음수 fallback → 단기차입금으로 보충
        if cash_t1_plugged < 0:
            shortfall = -cash_t1_plugged
            cash_t1_plugged = 0.0
            sd_t1 += shortfall  # 단기차입금 plug
            plug_used = "both"

        # Final totals
        current_assets_final = cash_t1_plugged + sti_t1 + ar_t1 + inv_t1 + other_current_assets_t
        non_current_assets_final = ppe_t1 + intang_t1 + other_non_current_assets_t
        current_liabilities_final = sd_t1 + cpld_t1 + ap_t1 + other_current_liabilities_t
        non_current_liabilities_final = ld_t1_net + bd_t1 + other_non_current_liabilities_t
        assets_final = current_assets_final + non_current_assets_final
        liab_final = current_liabilities_final + non_current_liabilities_final

        bs = {
            "cash": cash_t1_plugged,
            "short_term_investments": sti_t1,
            "receivables": ar_t1,
            "inventory": inv_t1,
            "ppe": ppe_t1,
            "intangibles": intang_t1,
            "other_assets": other_current_assets_t + other_non_current_assets_t,
            "other_current_assets": other_current_assets_t,
            "other_non_current_assets": other_non_current_assets_t,
            "total_assets": assets_final,
            "current_assets": current_assets_final,
            "non_current_assets": non_current_assets_final,
            "short_term_debt": sd_t1,
            "current_portion_long_debt": cpld_t1,
            "long_term_debt": ld_t1_net,
            "bonds": bd_t1,
            "payables": ap_t1,
            "other_liabilities": other_current_liabilities_t + other_non_current_liabilities_t,
            "other_current_liabilities": other_current_liabilities_t,
            "other_non_current_liabilities": other_non_current_liabilities_t,
            "total_liabilities": liab_final,
            "current_liabilities": current_liabilities_final,
            "non_current_liabilities": non_current_liabilities_final,
            "capital_stock": cap_stock_t,
            "capital_surplus": cap_surplus_t,
            "retained_earnings": re_t1,
            "other_equity": oth_eq_t,
            "total_equity": equity_t1,
            "residual_audit": {
                "presentation_residuals_preserved": True,
                "residual_fallback_used": residual_fallback_used,
                "other_current_assets_raw": oca_raw,
                "other_non_current_assets_raw": onca_raw,
                "other_current_liabilities_raw": ocl_raw,
                "other_non_current_liabilities_raw": oncl_raw,
                "other_current_assets_clipped": oca_clipped,
                "other_non_current_assets_clipped": onca_clipped,
                "other_current_liabilities_clipped": ocl_clipped,
                "other_non_current_liabilities_clipped": oncl_clipped,
                "residual_negative_flag": bool(oca_clipped or onca_clipped or ocl_clipped or oncl_clipped),
            },
        }
        return bs, plug_used, plug_amount

    # ------------------------------------------------------------------
    # Step 6: Build FirmState_t+1
    # ------------------------------------------------------------------
    def _build_state_t1(
        self,
        state_t: FirmState,
        is_t1: Dict,
        ocf_comp: Dict,
        icf_comp: Dict,
        fcf_comp: Dict,
        bs_t1: Dict,
    ) -> FirmState:
        s = FirmState(
            firm_id=state_t.firm_id,
            year=state_t.year + 1,
            sector=state_t.sector,
            # 손익
            revenue=is_t1["revenue"],
            cogs=is_t1["cogs"],
            gross_profit=is_t1["gross_profit"],
            sga=is_t1["sga"],
            operating_income=is_t1["operating_income"],
            financial_cost=is_t1["interest_expense"],
            pretax_income=is_t1["pretax_income"],
            tax_expense=is_t1["tax_expense"],
            net_income=is_t1["net_income"],
            comprehensive_income=is_t1["net_income"],  # OCI 단순화: NI = CI
            depreciation=is_t1["depreciation"],
            amortization=is_t1["amortization"],
            interest_income=is_t1["non_op_income"] * 0.5,  # 분리 모름, 절반 assumption
            dividend_income=is_t1["non_op_income"] * 0.5,
            # 재무상태표
            total_assets=bs_t1["total_assets"],
            current_assets=bs_t1["current_assets"],
            non_current_assets=bs_t1["non_current_assets"],
            cash=bs_t1["cash"],
            short_term_investments=bs_t1["short_term_investments"],
            receivables=bs_t1["receivables"],
            inventory=bs_t1["inventory"],
            ppe=bs_t1["ppe"],
            intangibles=bs_t1["intangibles"],
            total_liabilities=bs_t1["total_liabilities"],
            current_liabilities=bs_t1["current_liabilities"],
            non_current_liabilities=bs_t1["non_current_liabilities"],
            short_term_debt=bs_t1["short_term_debt"],
            current_portion_long_debt=bs_t1["current_portion_long_debt"],
            long_term_debt=bs_t1["long_term_debt"],
            bonds=bs_t1["bonds"],
            payables=bs_t1["payables"],
            total_equity=bs_t1["total_equity"],
            capital_stock=bs_t1["capital_stock"],
            retained_earnings=bs_t1["retained_earnings"],
            # 자본변동표 매핑
            ending_capital_stock=bs_t1["capital_stock"],
            ending_capital_surplus=bs_t1["capital_surplus"],
            ending_other_capital=bs_t1["other_equity"],
            ending_oci=0.0,  # OCI 단순화
            ending_retained_earnings=bs_t1["retained_earnings"],
            cash_dividends=is_t1["cash_dividends"],
            # 현금흐름표
            operating_cf=ocf_comp["operating_cf"],
            investing_cf=icf_comp["investing_cf"],
            financing_cf=fcf_comp["financing_cf"],
            capex=icf_comp["total_capex"],
            rating_num=state_t.rating_num,
            rating_grade=state_t.rating_grade,
        )
        return s

    # ------------------------------------------------------------------
    # Step 7: Sustainability classification
    # ------------------------------------------------------------------
    def _classify_sustainability(
        self, state_t1: FirmState, is_t1: Dict
    ) -> SustainabilityFlag:
        """Decision 15 확장: critical / fragile / ok 3단계."""
        cash_t1 = _safe_get(state_t1.cash)
        cogs_t1 = _safe_get(is_t1["cogs"])
        cur_assets = _safe_get(state_t1.current_assets)
        cur_liab = _safe_get(state_t1.current_liabilities)
        op_income = _safe_get(is_t1["operating_income"])
        int_exp = _safe_get(is_t1["interest_expense"])

        # critical: 현금 < 0 (plug fallback 후엔 0이지만 originally < 0이었을 가능성)
        # → cash가 0인데 단기차입금 plug이 발생한 경우 critical
        # 단순 check: cash ≤ 0
        if cash_t1 <= 0:
            return "critical"

        # fragile checks
        thr = self.sustainability_thresholds
        if cogs_t1 > 0 and cash_t1 / cogs_t1 < thr["cash_to_cogs_ratio"]:
            return "fragile"
        if cur_liab > 0 and cur_assets / cur_liab < thr["current_ratio"]:
            return "fragile"
        if int_exp > 0 and op_income / int_exp < thr["interest_coverage"]:
            return "fragile"

        return "ok"
