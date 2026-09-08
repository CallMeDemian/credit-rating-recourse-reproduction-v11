from __future__ import annotations

# Final Stage2 raw pseudo-action source coverage thresholds.
# This is the single Python source of truth used by builders and verifiers.
# Dimensions are raw action names without the action__ prefix.
ACTION_MIN_OBSERVED_RATE: dict[str, float] = {
    "ppe_pct": 0.20,
    "inv_turnover_chg": 0.20,
    "ar_turnover_chg": 0.20,
    "ap_turnover_chg": 0.20,
    "short_debt_pct": 0.10,
    "long_debt_pct": 0.10,
    "bond_pct": 0.05,
    "revenue_growth": 0.20,
    "cogs_ratio_chg": 0.20,
    "sga_ratio_chg": 0.20,
}

GLOBAL_MIN_OBSERVED_RATE = 0.05
WARNING_REFERENCE_OBSERVED_RATE = 0.50
CONTRACT_VERSION = "action_source_coverage_thresholds_v1"


def min_required_observed_rate(action_dim: str, *, floor: float = GLOBAL_MIN_OBSERVED_RATE) -> float:
    return max(float(floor), float(ACTION_MIN_OBSERVED_RATE.get(str(action_dim), floor)))
