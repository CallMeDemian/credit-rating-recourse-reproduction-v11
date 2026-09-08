"""Current selected-variable contract for source snapshots and final-freeze configs.

This module intentionally captures the *current* Stage00_04 oracle selected
variable universe that RL archives must surface in source snapshots.  It does
not replace backend params as the scoring source of truth; instead it prevents
stale package-level config copies from reintroducing the deprecated R136/R185
oracle surface after the 2026-06-03 redevelopment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_SELECTED_VARIABLE_IDS: tuple[str, ...] = (
    "R006",
    "R064",
    "R085",
    "R133",
    "R157",
    "R182",
    "industry_avg_rating_lag1_self_excl",
    "cap_change_count_3y",
    "log_assets",
    "operating_loss_freq_3y",
    "ratio_missing_rate",
)

DEPRECATED_SELECTED_VARIABLE_IDS_FOR_CURRENT_ORACLE: frozenset[str] = frozenset(
    {
        "R136",
        "R185",
        "industry_median_rating_lag1_self_excl",
        "financial_data_completeness",
    }
)


def read_selected_variable_master(path: Path) -> pd.DataFrame:
    """Read selected_variable_master.csv with strict schema presence.

    The project writes this file with UTF-8 BOM in several places.  A missing
    variable_id column is a hard contract failure because downstream archive
    readers use that column to distinguish the current R133 oracle from stale
    R136 source snapshots.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"selected_variable_master missing: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "variable_id" not in df.columns:
        raise KeyError(f"selected_variable_master has no variable_id column: {path}; columns={list(df.columns)}")
    return df


def selected_variable_ids_from_master(path: Path) -> list[str]:
    df = read_selected_variable_master(path)
    values = [str(x).strip() for x in df["variable_id"].dropna().tolist() if str(x).strip()]
    if len(values) != len(set(values)):
        dupes = sorted({x for x in values if values.count(x) > 1})
        raise ValueError(f"selected_variable_master has duplicate variable_id values: {path}; duplicates={dupes}")
    return values


def verify_current_selected_variable_master(path: Path) -> dict[str, Any]:
    """Return a strict verification report for one selected-variable master."""
    ids = selected_variable_ids_from_master(path)
    expected = list(CURRENT_SELECTED_VARIABLE_IDS)
    errors: list[str] = []
    if ids != expected:
        errors.append(
            "selected_variable_master variable_id order/content mismatch: "
            f"expected={expected} found={ids} path={path}"
        )
    deprecated_present = sorted(set(ids) & set(DEPRECATED_SELECTED_VARIABLE_IDS_FOR_CURRENT_ORACLE))
    if deprecated_present:
        errors.append(f"deprecated current-oracle variable ids present in selected_variable_master: {deprecated_present} path={path}")
    return {
        "path": str(Path(path)),
        "status": "PASS" if not errors else "FAIL",
        "selected_variables": ids,
        "expected_selected_variables": expected,
        "deprecated_for_current_oracle_present": deprecated_present,
        "errors": errors,
    }


def verify_package_selected_variable_masters(package_root: Path) -> dict[str, Any]:
    """Verify both package-level config copies used by source snapshots."""
    package_root = Path(package_root)
    paths = [
        package_root / "configs" / "selected_variable_master.csv",
        package_root / "final_freeze_configs" / "configs" / "selected_variable_master.csv",
    ]
    checks = [verify_current_selected_variable_master(p) for p in paths]
    errors = [err for check in checks for err in check["errors"]]
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }
