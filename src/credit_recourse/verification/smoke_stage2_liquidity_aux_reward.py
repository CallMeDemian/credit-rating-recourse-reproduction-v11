from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline import (
    AUX_REWARD_COLUMNS,
    compute_aux_reward_stats,
    apply_merton_fcff_aux_reward,
)
from credit_recourse.rl.pipelines.final_stage2_counterfactual_transitions.pipeline import _standardize


def _synthetic_phase3_panel() -> pd.DataFrame:
    rows = []
    for i, (cid, cash_t, cash_n, sti_t, sti_n) in enumerate([
        ("DL2", 120.0, 60.0, 20.0, 10.0),
        ("MX2", 60.0, 120.0, 10.0, 20.0),
        ("A0", 80.0, 80.0, 10.0, 10.0),
        ("WC1", 90.0, 75.0, 15.0, 12.0),
    ]):
        assets_t = 1000.0 + i * 10.0
        assets_n = 1000.0 + i * 10.0
        rows.append({
            "firm_id": f"F{i:03d}",
            "fiscal_year": 2022,
            "candidate_id": cid,
            "reward_raw": 0.0,
            "sim__total_assets": assets_t,
            "next__sim__total_assets": assets_n,
            "sim__short_term_debt": 100.0,
            "sim__long_term_debt": 200.0,
            "sim__bonds": 50.0,
            "next__sim__short_term_debt": 95.0,
            "next__sim__long_term_debt": 190.0,
            "next__sim__bonds": 45.0,
            "sim__operating_cf": 80.0,
            "sim__capex": 20.0,
            "next__sim__operating_cf": 82.0,
            "next__sim__capex": 19.0,
            "sim__cash": cash_t,
            "sim__short_term_investments": sti_t,
            "next__sim__cash": cash_n,
            "next__sim__short_term_investments": sti_n,
        })
    return pd.DataFrame(rows)


def main() -> int:
    temporal = SimpleNamespace(inner_train_year_max=2023)

    # Disabled path must preserve a stable zero schema without touching simulator next-state inputs.
    disabled_input = pd.DataFrame({"firm_id": ["F0"], "fiscal_year": [2022], "reward_raw": [0.0]})
    disabled_stats = compute_aux_reward_stats(_synthetic_phase3_panel(), temporal, merton_lambda=0.0, fcff_lambda=0.0, liquidity_lambda=0.0)
    disabled_out, disabled_meta = apply_merton_fcff_aux_reward(disabled_input, disabled_stats, phase_name="disabled_schema_smoke")
    assert disabled_meta["liquidity_aux_enabled"] is False
    assert all(c in disabled_out.columns for c in AUX_REWARD_COLUMNS)
    assert float(disabled_out["reward_aux_liquidity"].abs().sum()) == 0.0

    # Enabled path must compute the intended sign: liquidity drain < 0, liquidity build > 0.
    enabled_stats = compute_aux_reward_stats(_synthetic_phase3_panel(), temporal, merton_lambda=0.0, fcff_lambda=0.0, liquidity_lambda=0.2)
    assert enabled_stats["liquidity_aux_enabled"] is True
    assert enabled_stats["reference_oracle_scores_used"] is False
    out, meta = apply_merton_fcff_aux_reward(_synthetic_phase3_panel(), enabled_stats, phase_name="enabled_liquidity_smoke")
    assert meta["liquidity_aux_enabled"] is True
    dl2 = float(out.loc[out["candidate_id"] == "DL2", "delta_liquid_capacity"].iloc[0])
    mx2 = float(out.loc[out["candidate_id"] == "MX2", "delta_liquid_capacity"].iloc[0])
    assert dl2 < 0.0, dl2
    assert mx2 > 0.0, mx2
    assert float(out.loc[out["candidate_id"] == "DL2", "reward_aux_liquidity"].iloc[0]) < 0.0
    assert float(out.loc[out["candidate_id"] == "MX2", "reward_aux_liquidity"].iloc[0]) > 0.0

    cf_stats = {
        "lambda_phi": 0.1,
        "reward_mean_train": 0.0,
        "reward_std_train": 1.0,
        "aux_reward_stats": enabled_stats,
    }
    cf_base = out.copy()
    cf_base["reward_raw_notch"] = 0.0
    cf_base["delta_phi"] = 0.0
    cf_base["is_observed_transition"] = False
    cf_disabled = _standardize(cf_base, cf_stats, reward_mode="phi_merton")
    assert float(cf_disabled["reward_aux_liquidity"].abs().sum()) == 0.0
    cf_enabled = _standardize(cf_base, cf_stats, reward_mode="phi_merton_liquidity")
    assert float(cf_enabled.loc[cf_enabled["candidate_id"] == "DL2", "reward_aux_liquidity"].iloc[0]) < 0.0
    assert float(cf_enabled.loc[cf_enabled["candidate_id"] == "MX2", "reward_aux_liquidity"].iloc[0]) > 0.0
    print("liquidity_aux_smoke PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
