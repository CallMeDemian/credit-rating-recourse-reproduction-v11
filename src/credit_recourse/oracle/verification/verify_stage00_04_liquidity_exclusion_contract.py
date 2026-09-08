from __future__ import annotations

"""Verify Stage00_04 liquidity exclusion policy.

This is a focused guard for oracle redevelopment: R122 and R136 must
not be eligible liquidity candidates.  R122 is also caught by the existing
revenue-term guard.  R133 must remain eligible so Stage00_04 can select
FCF/current-liabilities when its empirical score wins.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credit_recourse.rl.common.io import write_json

FORBIDDEN_LIQUIDITY_R_CODES = {"R122", "R136"}
REQUIRED_ELIGIBLE_LIQUIDITY_R_CODES = {"R133", "R116"}


def _load_pipeline_policy() -> set[str]:
    from credit_recourse.oracle.stage1.stage00_04_variable_selection.pipeline import (
        GAMEABLE_LIQUIDITY_EXCLUDE,
    )

    return {str(x) for x in GAMEABLE_LIQUIDITY_EXCLUDE}


def _synthetic_policy_application() -> dict[str, object]:
    policy = _load_pipeline_policy()
    rows = pd.DataFrame(
        [
            {"ratio_id": "R122", "category": "유동성", "ratio_name": "유동부채/매출액"},
            {"ratio_id": "R136", "category": "유동성", "ratio_name": "매입채무/유동부채"},
            {"ratio_id": "R133", "category": "유동성", "ratio_name": "FCF/유동부채"},
            {"ratio_id": "R116", "category": "유동성", "ratio_name": "당좌비율"},
            {"ratio_id": "R006", "category": "수익성", "ratio_name": "세전이익률"},
        ]
    )
    rows["selected_eligible_before"] = True
    rows["selected_eligible_after"] = ~(
        (rows["category"] == "유동성") & rows["ratio_id"].astype(str).isin(policy)
    )
    return {
        "active_policy": sorted(policy),
        "synthetic_rows": rows.to_dict(orient="records"),
        "forbidden_after": rows.loc[rows["ratio_id"].isin(FORBIDDEN_LIQUIDITY_R_CODES), "selected_eligible_after"].tolist(),
        "required_eligible_after": rows.loc[rows["ratio_id"].isin(REQUIRED_ELIGIBLE_LIQUIDITY_R_CODES), "selected_eligible_after"].tolist(),
        "r133_after": bool(rows.loc[rows["ratio_id"] == "R133", "selected_eligible_after"].iloc[0]),
        "r116_after": bool(rows.loc[rows["ratio_id"] == "R116", "selected_eligible_after"].iloc[0]),
    }


def run(project_root: Path | None = None) -> dict[str, object]:
    errors: list[str] = []
    synthetic = _synthetic_policy_application()
    active = set(synthetic["active_policy"])
    if not active.issuperset(FORBIDDEN_LIQUIDITY_R_CODES):
        errors.append(
            f"Stage00_04 GAMEABLE_LIQUIDITY_EXCLUDE missing required ids: "
            f"required={sorted(FORBIDDEN_LIQUIDITY_R_CODES)} active={sorted(active)}"
        )
    if any(bool(x) for x in synthetic["forbidden_after"]):
        errors.append("Synthetic policy application left at least one forbidden liquidity R-code eligible.")
    if not all(bool(x) for x in synthetic["required_eligible_after"]):
        errors.append("Synthetic policy application unexpectedly blocked R133 or R116 liquidity candidate.")
    report = {
        "stage": "stage00_04_liquidity_exclusion_contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "required_forbidden_liquidity_r_codes": sorted(FORBIDDEN_LIQUIDITY_R_CODES),
        "required_eligible_liquidity_r_codes": sorted(REQUIRED_ELIGIBLE_LIQUIDITY_R_CODES),
        "synthetic_policy_application": synthetic,
        "errors": errors,
    }
    if project_root is not None:
        out = project_root / "data" / "final_freeze" / "ledgers" / "stage00_04_liquidity_exclusion_contract_report.json"
        write_json(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify Stage00_04 liquidity exclusion policy.")
    parser.add_argument("--project-root", default=None)
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve() if args.project_root else None
    report = run(project_root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
