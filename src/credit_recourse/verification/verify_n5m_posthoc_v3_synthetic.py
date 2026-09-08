from __future__ import annotations

"""Contract-faithful synthetic execution for N5M post-hoc v4 extensions."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.analysis.n5m_posthoc import (
    ORACLES,
    _cross_oracle_local_q_gain,
    _oracle_consensus_by_budget,
    _reference_axis_anatomy,
    _wide_actions,
)
from credit_recourse.verification.smoke_stage7_n5_budget_contract import (
    ACTION_COLS,
    _make_space,
)

SCHEMA_VERSION = "n5m_posthoc_v4_synthetic_contract_v1"


def run_smoke() -> dict:
    space = _make_space()
    budgets = [("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", np.nan)]
    rows = []
    references = list(space.fixed_candidates)
    for budget_index, (label, value) in enumerate(budgets):
        for rid in range(12):
            reference = references[(rid + budget_index) % len(references)]
            rec = {
                "budget_label": label,
                "l1_budget": value,
                "row_id": rid,
                "rl_reference_candidate": reference,
            }
            ref_vec = np.asarray(space.candidate_vector(reference), dtype=float)
            for axis_index, axis in enumerate(ACTION_COLS):
                c4 = ((rid + axis_index) % 5 - 2) * 0.02
                c6 = c4 + ref_vec[axis_index] * 0.25 + ((rid + budget_index) % 3 - 1) * 0.005
                rec[f"c4__{axis}"] = float(c4)
                rec[f"c6__{axis}"] = float(c6)
            for oracle_index, oracle in enumerate(ORACLES):
                base = (rid - 5.5) * (oracle_index + 1) / 100.0
                budget_effect = (-0.04 + budget_index * 0.02) * (1.0 - oracle_index * 0.1)
                rec[f"reference_advantage_{oracle}"] = base
                rec[f"revision_delta_R_score_{oracle}"] = 0.4 * base + budget_effect
            rows.append(rec)
    frame = pd.DataFrame(rows)

    # Stage7-to-firm-frame handoff regression: projection diagnostics must
    # survive the C4/C6 wide conversion used by the Section 9.8 gate.
    action_rows = []
    for rid in range(12):
        for policy, distance in (("C4", 0.01 + rid * 0.001), ("C6", 0.02 + rid * 0.001)):
            row = {
                "budget_label": "0p75", "l1_budget": 0.75, "row_id": rid,
                "policy": policy, "projection_distance": distance,
                "projection_method": "nearest_candidate", "out_of_library": False,
                "budget_l1_target": 0.75, "budget_compliant_clipped": True,
            }
            row.update({axis: float((rid % 3 - 1) * 0.01) for axis in ACTION_COLS})
            action_rows.append(row)
    wide = _wide_actions(pd.DataFrame(action_rows), ACTION_COLS, np.ones(len(ACTION_COLS), dtype=float))
    assert "c4_projection_distance" in wide.columns
    assert "c6_projection_distance" in wide.columns
    assert np.allclose(wide.sort_values("row_id")["c4_projection_distance"], [0.01 + rid * 0.001 for rid in range(12)])

    cross = _cross_oracle_local_q_gain(frame)
    widths = np.asarray([space.bound_width(axis) for axis in ACTION_COLS], dtype=float)
    detail, summary = _reference_axis_anatomy(
        firm_frame=frame,
        action_columns=ACTION_COLS,
        action_widths=widths,
        space=space,
    )
    consensus = _oracle_consensus_by_budget(frame)

    assert len(cross) == 24
    assert not (cross["local_q_oracle"].astype(str) == cross["revision_gain_oracle"].astype(str)).any()
    assert cross["p_holm_24"].between(0.0, 1.0).all()
    assert len(detail) == 48
    assert len(summary) == 24
    assert set(summary["metric"].astype(str)) == {"removed_mass_off_reference_axes", "added_mass_on_reference_axes"}
    assert summary["p_holm_within_oracle_metric"].between(0.0, 1.0).all()
    assert len(consensus) == 4
    assert consensus["n_firms"].eq(12).all()
    for prefix in ("all_three_loss", "majority_loss", "all_three_win", "majority_win", "mixed_oracle_direction"):
        assert consensus[f"{prefix}_fraction"].between(0.0, 1.0).all()

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "synthetic_firm_frame_rows": int(len(frame)),
        "cross_oracle_rows": int(len(cross)),
        "reference_axis_detail_rows": int(len(detail)),
        "reference_axis_summary_rows": int(len(summary)),
        "oracle_consensus_rows": int(len(consensus)),
        "stage7_projection_handoff_rows": int(len(wide)),
        "stage7_projection_handoff_status": "PASS",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    try:
        result = run_smoke()
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
