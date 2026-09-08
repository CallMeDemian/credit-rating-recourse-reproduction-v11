from __future__ import annotations

"""Contract-faithful synthetic smoke for the reference-free C4R second pass.

No live LLM or project raw data are used.  The smoke exercises the active
prompt/budget contract and the Stage9 revision-metric construction with the
same ActionSpace and column names used by the Stage7 N5 smoke.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from credit_recourse.eval.final_stage9_llm_rl_comparison.revision_metrics import (
    build_revision_table,
)
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.budget_contract import (
    budget_applies,
    make_action_budget_contract,
)
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.prompt_builder import (
    build_prompt,
)
from credit_recourse.verification.smoke_stage7_n5_budget_contract import (
    ACTION_COLS,
    _make_panel,
    _make_space,
)

SCHEMA_VERSION = "stage7_c4r_contract_smoke_v1"


def run_smoke() -> dict:
    space = _make_space()
    panel = _make_panel()
    contract = make_action_budget_contract(
        l1_budget=0.75,
        budgeted_conditions=["C4", "C4R", "C6"],
        budgeted_modes=["free_form_10d"],
        label="C4R_MATCHED_SMOKE_L1_0p75",
    ).to_dict()
    initial_action = {
        "selected_candidate": "DL2_deleverage_moderate",
        "action_vector": {name: 0.0 for name in ACTION_COLS},
        "rationale": "synthetic paired C4 draft",
    }
    prompt = build_prompt(
        panel.iloc[0],
        condition="C4R",
        mode="free_form_10d",
        information_condition="IC-b",
        space=space,
        rl_reference_candidate=None,
        reference_source="none",
        reference_draw_seed=None,
        initial_action=initial_action,
        action_budget_contract=contract,
    )
    assert prompt["condition"] == "C4R"
    assert prompt["reference_source"] == "none"
    assert prompt["rl_reference_candidate"] is None
    assert prompt["reference_draw_seed"] is None
    assert prompt["initial_action"] == initial_action
    assert prompt["action_budget_contract"]["enabled"] is True
    assert budget_applies(contract, condition="C4R", mode="free_form_10d") is True
    joined = " ".join(map(str, prompt["instructions"]))
    assert "without any external reference" in joined
    assert "N5 action-budget contract" in joined

    # Two aligned Stage7 action rows and their Stage8 score rows.  C4R changes
    # one axis, but no RL reference exists, so only revision distance/score
    # deltas are defined; adoption geometry must remain NA.
    base = {name: 0.0 for name in ACTION_COLS}
    revised = dict(base)
    revised["action__short_debt_pct"] = -0.20
    action_table = pd.DataFrame(
        [
            {
                "row_id": 1,
                "policy": "C4",
                "candidate_id": "A0_noop",
                "mode": "free_form_10d",
                "reference_source": "none",
                "reference_draw_seed": None,
                "rl_reference_candidate": None,
                **base,
            },
            {
                "row_id": 1,
                "policy": "C4R",
                "candidate_id": "DL2_deleverage_moderate",
                "mode": "free_form_10d",
                "reference_source": "none",
                "reference_draw_seed": None,
                "rl_reference_candidate": None,
                **revised,
            },
        ]
    )
    stage8_scores = pd.DataFrame(
        [
            {"row_id": 1, "policy": "C4", "mode": "free_form_10d", "delta_R_score_alpha": 0.2, "delta_R_score_beta": 0.1, "delta_R_score_gamma": 0.3},
            {"row_id": 1, "policy": "C4R", "mode": "free_form_10d", "delta_R_score_alpha": 0.4, "delta_R_score_beta": 0.15, "delta_R_score_gamma": 0.35},
        ]
    )
    revision = build_revision_table(action_table=action_table, stage8_scores=stage8_scores, space=space)
    c4r = revision.loc[revision["revision_condition"].astype(str).eq("C4R")]
    assert len(c4r) == 1
    row = c4r.iloc[0]
    assert row["base_condition"] == "C4"
    assert row["reference_source"] == "none"
    assert bool(row["metrics_defined"]) is False
    assert row["undefined_reason"] == "no_rl_reference"
    assert pd.isna(row["rl_adoption_ratio"])
    assert float(row["revision_delta_R_score_alpha"]) == 0.2
    assert float(row["revision_l1_distance"]) > 0.0

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "c4r_prompt_reference_source": prompt["reference_source"],
        "c4r_budget_applies": True,
        "revision_row_count": int(len(c4r)),
        "revision_metrics_defined": bool(row["metrics_defined"]),
        "revision_undefined_reason": str(row["undefined_reason"]),
        "revision_delta_R_score_alpha": float(row["revision_delta_R_score_alpha"]),
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
