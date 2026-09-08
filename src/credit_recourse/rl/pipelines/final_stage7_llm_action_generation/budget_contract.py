"""Stage 7 free-form action-budget contract helpers.

This module implements the N5 budget-constrained generation contract without
introducing a new policy code.  The policy remains C6/C7/etc.; the budget
constraint is carried by prompt payload, Stage 7 metadata, and audit columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Any

ACTION_BUDGET_CONTRACT_SCHEMA_VERSION = "stage7_freeform_l1_budget_contract_v1"

BUDGET_AUDIT_COLUMNS = [
    "budget_contract_label",
    "budget_l1_target",
    "budget_l1_raw",
    "budget_l1_clipped",
    "budget_compliant_raw",
    "budget_compliant_clipped",
    "budgeted_condition_flag",
]


@dataclass(frozen=True)
class ActionBudgetContract:
    """Validated Stage 7 action-budget contract.

    The contract applies only to free-form generation for the configured policy
    conditions.  It is intentionally not a projection/evaluation post-hoc
    rescale; it is an LLM-facing output contract used at generation time.
    """

    label: str
    l1_budget: float
    budgeted_conditions: tuple[str, ...]
    budgeted_modes: tuple[str, ...] = ("free_form_10d",)
    tolerance: float = 1.0e-9

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ACTION_BUDGET_CONTRACT_SCHEMA_VERSION,
            "enabled": True,
            "label": self.label,
            "l1_budget": float(self.l1_budget),
            "budgeted_conditions": list(self.budgeted_conditions),
            "budgeted_modes": list(self.budgeted_modes),
            "tolerance": float(self.tolerance),
            "generation_semantics": (
                "LLM-facing free-form action-vector L1 contract; not a post-hoc "
                "rescale or projection constraint."
            ),
        }


def disabled_budget_contract() -> dict[str, Any]:
    return {
        "schema_version": ACTION_BUDGET_CONTRACT_SCHEMA_VERSION,
        "enabled": False,
    }


def make_action_budget_contract(
    *,
    l1_budget: float | None,
    budgeted_conditions: list[str] | tuple[str, ...] | None = None,
    label: str | None = None,
    budgeted_modes: list[str] | tuple[str, ...] | None = None,
    tolerance: float = 1.0e-9,
) -> ActionBudgetContract | None:
    """Validate user config and return a budget contract, or ``None``.

    Fail-fast on invalid positive budget/condition/mode values.  A missing
    ``l1_budget`` means the N5 path is disabled and existing Stage 7 behavior is
    preserved.
    """
    if l1_budget is None:
        return None
    budget = float(l1_budget)
    if not budget > 0:
        raise ValueError(f"free-form L1 budget must be positive; got {l1_budget!r}.")
    allowed_conditions = {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"}
    conditions = tuple(str(c).strip() for c in (budgeted_conditions or ["C6"]) if str(c).strip())
    if not conditions:
        raise ValueError("budgeted_conditions must contain at least one policy condition.")
    bad_conditions = sorted(set(conditions) - allowed_conditions)
    if bad_conditions:
        raise ValueError(f"Unsupported budgeted_conditions: {bad_conditions}")
    modes = tuple(str(m).strip() for m in (budgeted_modes or ["free_form_10d"]) if str(m).strip())
    bad_modes = sorted(set(modes) - {"free_form_10d"})
    if bad_modes:
        raise ValueError(
            f"Action-budget contracts currently apply only to free_form_10d; got {bad_modes}."
        )
    if not modes:
        raise ValueError("budgeted_modes must contain free_form_10d.")
    tol = float(tolerance)
    if tol < 0:
        raise ValueError(f"budget tolerance must be non-negative; got {tolerance!r}.")
    lab = (label or f"freeform_L1_le_{budget:g}").strip()
    if not lab or any(ch in lab for ch in "/\\\t\n\r"):
        raise ValueError(f"budget contract label must be a simple token; got {label!r}.")
    return ActionBudgetContract(
        label=lab,
        l1_budget=budget,
        budgeted_conditions=conditions,
        budgeted_modes=modes,
        tolerance=tol,
    )


def normalize_budget_contract(contract: ActionBudgetContract | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if contract is None:
        return None
    if isinstance(contract, ActionBudgetContract):
        return contract.to_dict()
    data = dict(contract)
    if not data.get("enabled"):
        return None
    required = ["label", "l1_budget", "budgeted_conditions", "budgeted_modes", "tolerance"]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"action_budget_contract missing required keys: {missing}")
    return make_action_budget_contract(
        l1_budget=float(data["l1_budget"]),
        budgeted_conditions=list(data["budgeted_conditions"]),
        label=str(data["label"]),
        budgeted_modes=list(data["budgeted_modes"]),
        tolerance=float(data.get("tolerance", 1.0e-9)),
    ).to_dict()


def budget_applies(
    contract: ActionBudgetContract | Mapping[str, Any] | None,
    *,
    condition: str,
    mode: str,
) -> bool:
    data = normalize_budget_contract(contract)
    if data is None:
        return False
    return str(condition) in set(map(str, data["budgeted_conditions"])) and str(mode) in set(map(str, data["budgeted_modes"]))


def action_l1(vector: Mapping[str, Any] | None) -> float | None:
    if vector is None:
        return None
    total = 0.0
    for v in vector.values():
        try:
            total += abs(float(v) if v is not None else 0.0)
        except (TypeError, ValueError):
            return None
    return float(total)


def empty_budget_audit() -> dict[str, Any]:
    return {
        "budget_contract_label": None,
        "budget_l1_target": None,
        "budget_l1_raw": None,
        "budget_l1_clipped": None,
        "budget_compliant_raw": None,
        "budget_compliant_clipped": None,
        "budgeted_condition_flag": False,
    }


def budget_audit_for_action(
    *,
    contract: ActionBudgetContract | Mapping[str, Any] | None,
    condition: str,
    mode: str,
    raw_action: Mapping[str, Any] | None,
    clipped_action: Mapping[str, Any] | None,
) -> dict[str, Any]:
    data = normalize_budget_contract(contract)
    if data is None or not budget_applies(data, condition=condition, mode=mode):
        return empty_budget_audit()
    target = float(data["l1_budget"])
    tol = float(data.get("tolerance", 1.0e-9))
    raw_l1 = action_l1(raw_action)
    clipped_l1 = action_l1(clipped_action)
    return {
        "budget_contract_label": str(data["label"]),
        "budget_l1_target": target,
        "budget_l1_raw": raw_l1,
        "budget_l1_clipped": clipped_l1,
        "budget_compliant_raw": (False if raw_l1 is None else bool(raw_l1 <= target + tol)),
        "budget_compliant_clipped": (False if clipped_l1 is None else bool(clipped_l1 <= target + tol)),
        "budgeted_condition_flag": True,
    }
