"""
Action — Simulator가 받는 10차원 continuous action vector.

GPT critique 수용:
- 무형자산 v1 main에서 제외 (10D)
- Tier 1: financial policy / working-capital levers (7)
- Tier 2: constrained operating-plan drivers (3)
- 회전율은 paper 표기는 days, 계산은 회전수
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal


FinancingPath = Literal[
    "cash",
    "short_debt",
    "long_debt",
    "bond",
    "working_capital_release",
    "capital_increase",  # 자본증자는 action으로는 안 받지만 LLM 자연어에서 등장 가능 → flag
    "unspecified",
]


@dataclass
class Action:
    """
    10차원 continuous action.
    각 차원의 default = 0 (no change).
    """
    # ── Tier 1: financial policy / working-capital levers (7) ──
    ppe_pct: float = 0.0                # 유형자산 % 변화. bound: [-0.50, +0.50]
    inv_turnover_chg: float = 0.0       # 재고회전율 변화 (회). bound: [-3.0, +3.0]
    ar_turnover_chg: float = 0.0        # 매출채권회전율 변화. bound: [-3.0, +3.0]
    ap_turnover_chg: float = 0.0        # 매입채무회전율 변화. bound: [-3.0, +3.0]
    short_debt_pct: float = 0.0         # 단기차입금 % 변화. bound: [-1.0, +1.0]
    long_debt_pct: float = 0.0          # 장기차입금 % 변화. bound: [-0.5, +0.5]
    bond_pct: float = 0.0               # 사채 % 변화. bound: [-0.5, +0.5]

    # ── Tier 2: constrained operating-plan drivers (3) ──
    revenue_growth: float = 0.0         # 매출 성장률. bound: [-0.15, +0.15]
    cogs_ratio_chg: float = 0.0         # 매출원가율 변화 (%pt). bound: [-0.03, +0.03]
    sga_ratio_chg: float = 0.0          # 판관비율 변화 (%pt). bound: [-0.02, +0.02]

    # ── Multi-action financing path (LLM에서 명시 요구) ──
    financing_path: FinancingPath = "unspecified"

    # ── Optional: confidence / source tracking ──
    source: Literal["RL", "LLM", "heuristic", "historical"] = "RL"
    confidence: float = 1.0  # pseudo-action labeling에서 활용
    rationale: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def is_no_op(self, tol: float = 1e-9) -> bool:
        """모든 차원이 0이면 True."""
        return all(
            abs(getattr(self, f)) < tol
            for f in [
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
        )


# Action bounds (calibration 전 default; 추후 historical P5/P95로 보정)
ACTION_BOUNDS = {
    # Tier 1
    "ppe_pct": (-0.50, 0.50),
    "inv_turnover_chg": (-3.0, 3.0),
    "ar_turnover_chg": (-3.0, 3.0),
    "ap_turnover_chg": (-3.0, 3.0),
    "short_debt_pct": (-1.0, 1.0),
    "long_debt_pct": (-0.5, 0.5),
    "bond_pct": (-0.5, 0.5),
    # Tier 2 (strict)
    "revenue_growth": (-0.15, 0.15),
    "cogs_ratio_chg": (-0.03, 0.03),
    "sga_ratio_chg": (-0.02, 0.02),
}


def clip_action(action: Action) -> Action:
    """Action을 bound 범위로 clip."""
    clipped = Action(**action.to_dict())
    for fname, (lo, hi) in ACTION_BOUNDS.items():
        v = getattr(clipped, fname)
        setattr(clipped, fname, max(lo, min(hi, v)))
    return clipped
