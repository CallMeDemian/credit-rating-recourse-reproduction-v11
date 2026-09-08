"""Financial statement simulator package."""

from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.action import Action
from credit_recourse.simulator.business_plan import BusinessPlan, calibrate_business_plan
from credit_recourse.simulator.financial_simulator import FinancialSimulator
from credit_recourse.simulator.synthetic import synthetic_firm, make_synthetic_pair

__all__ = [
    "FirmState",
    "Action",
    "BusinessPlan",
    "calibrate_business_plan",
    "FinancialSimulator",
    "synthetic_firm",
    "make_synthetic_pair",
    "rollout_one_year",
]

def rollout_one_year(state_t_row, action_dict):
    """
    Stage 14 evaluation entrypoint.

    Contract:
        state_t_row: dict-like firm-year state
        action_dict: 10D action dict
        return: dict-like next-year state
    """
    from credit_recourse.simulator.firm_state import load_firm_state_from_columns
    from credit_recourse.simulator.action import Action, clip_action
    from credit_recourse.simulator.business_plan import BusinessPlan
    from credit_recourse.simulator.financial_simulator import FinancialSimulator

    row = dict(state_t_row)
    act = dict(action_dict)

    firm_id = str(
        row.get("firm_id")
        or row.get("거래소코드")
        or row.get("corp_code")
        or "UNKNOWN"
    )

    year_raw = row.get("year", row.get("회계년도", 0))
    try:
        year = int(year_raw)
    except Exception:
        year = 0

    sector = str(row.get("sector", row.get("industry", row.get("market", "Unknown"))))

    state_t = load_firm_state_from_columns(
        column_dict=row,
        firm_id=firm_id,
        year=year,
        sector=sector,
    )

    # Preserve rating metadata if present
    if "grade_num" in row:
        state_t.rating_num = row.get("grade_num")
    elif "rating_num" in row:
        state_t.rating_num = row.get("rating_num")

    if "grade_str" in row:
        state_t.rating_grade = row.get("grade_str")
    elif "신용등급" in row:
        state_t.rating_grade = row.get("신용등급")

    allowed = [
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

    clean_action = {}
    for k in allowed:
        try:
            clean_action[k] = float(act.get(k, 0.0))
        except Exception:
            clean_action[k] = 0.0

    action = clip_action(Action(**clean_action))

    # Minimal default plan. Later, this can be calibrated from firm history.
    plan = BusinessPlan()

    sim = FinancialSimulator()
    result = sim.simulate(state_t, plan, action)

    state_next = result.state_t1

    if hasattr(state_next, "to_dict"):
        out = state_next.to_dict()
    elif hasattr(state_next, "__dict__"):
        out = dict(state_next.__dict__)
    else:
        out = dict(row)

    out["firm_id"] = firm_id
    out["year"] = year + 1 if year else year
    out["grade_num"] = row.get("grade_num", row.get("rating_num", None))
    out["grade_str"] = row.get("grade_str", row.get("신용등급", None))

    if hasattr(result, "sustainability"):
        out["_sim_sustainability"] = result.sustainability
    if hasattr(result, "plug_used"):
        out["_sim_plug_used"] = result.plug_used
    if hasattr(result, "plug_amount"):
        out["_sim_plug_amount"] = result.plug_amount

    return out