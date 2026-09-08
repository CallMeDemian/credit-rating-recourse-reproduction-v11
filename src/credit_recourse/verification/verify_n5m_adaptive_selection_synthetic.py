from __future__ import annotations

"""Contract-faithful synthetic execution for the thesis Section 9.8 patch.

The fixture preserves the production cardinality (575 firms x four budgets),
uses the canonical frozen contract, includes missing current-state values and a
category seen in only one firm, and carries protected audit columns that must
not enter the pre-action feature set.  It executes the producer and then the
independent output verifier.
"""

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.analysis.n5m_adaptive_selection import run_analysis
from credit_recourse.configs import __path__ as config_package_paths
from credit_recourse.verification.verify_n5m_adaptive_selection_contract import verify

SCHEMA_VERSION = "n5m_adaptive_selection_synthetic_contract_v1"
BUDGETS = ("0p75", "1p27", "2p00", "unbounded")


def _write_fixture(root: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(20260719)
    row_id = np.arange(575, dtype=int)
    x1 = rng.normal(size=575)
    x2 = rng.normal(size=575)
    x3 = rng.normal(size=575)
    x4 = rng.normal(size=575)
    size = 8.0 + 0.35 * x1 + rng.normal(scale=0.05, size=575)

    state = pd.DataFrame({
        "row_id": row_id,
        "derived__debt_to_assets": 0.45 + 0.10 * x1,
        "derived__current_ratio": 1.20 + 0.20 * x2,
        "derived__operating_margin": 0.08 + 0.04 * x3,
        "derived__cash_ratio": 0.20 + 0.05 * x4,
        "delta_1y__derived__debt_to_assets": 0.02 * (x2 - x1),
        # Looks like a derived feature but is deliberately absent from the
        # Stage7 IC-b prompt contract; the exact-name allow-list must exclude it.
        "derived__not_prompt_visible": 1000.0 + x4,
        "log_assets": size,
        "industry_class": np.where(row_id == 574, "UNSEEN_SINGLETON", np.char.add("IND_", (row_id % 8).astype(str))),
        "market": np.where(row_id % 3 == 0, "KOSPI", "KOSDAQ"),
        # Protected upstream audit/label fields: present but never selected.
        "candidate_id": ["PROTECTED"] * 575,
        "rating_num": np.full(575, 5),
        "delta_R_score_alpha": np.full(575, 999.0),
    })
    state.loc[state.index[::37], "derived__current_ratio"] = np.nan
    state.loc[state.index[::53], "industry_class"] = None

    budget_offsets = {
        "0p75": np.array([0.10, -0.08, -0.03, 0.01]),
        "1p27": np.array([0.03, 0.08, -0.02, -0.01]),
        "2p00": np.array([-0.02, 0.01, 0.09, -0.03]),
        "unbounded": np.array([-0.03, -0.02, 0.02, 0.10]),
    }
    rows: list[dict[str, Any]] = []
    for budget_index, budget in enumerate(BUDGETS):
        weights = budget_offsets[budget]
        c4 = (
            0.94
            + weights[0] * x1
            + weights[1] * x2
            + weights[2] * x3
            + weights[3] * x4
            + rng.normal(scale=0.015, size=575)
        )
        c4_l1 = 0.35 + 0.20 * budget_index + 0.10 * np.abs(x1) + 0.03 * np.abs(x2)
        projection = 0.02 + 0.025 * np.abs(x2) + 0.008 * budget_index
        active = np.clip(np.rint(3.0 + 0.7 * budget_index + 0.8 * np.abs(x3)), 1, 10)
        revision = (
            0.045
            + 0.075 * (c4_l1 - c4_l1.mean())
            - 0.55 * (projection - projection.mean())
            + 0.012 * (active - active.mean())
            + 0.015 * x4
            + rng.normal(scale=0.025, size=575)
        )
        c6 = c4 + revision
        for i in range(575):
            rows.append({
                "budget_label": budget,
                "l1_budget": np.nan if budget == "unbounded" else float((0.75, 1.27, 2.0)[budget_index]),
                "row_id": int(i),
                "initial_delta_R_score_alpha": float(c4[i]),
                "revised_delta_R_score_alpha": float(c6[i]),
                "revision_delta_R_score_alpha": float(revision[i]),
                "c4_final_l1": float(c4_l1[i]),
                "c4_projection_distance": float(projection[i]),
                "c4_active_dimensions": float(active[i]),
            })
    firm_frame = pd.DataFrame(rows)

    input_dir = root / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    firm_path = input_dir / "n5m_firm_frame.parquet"
    state_path = input_dir / "phase_eval_candidate.parquet"
    contract_path = root / "data" / "final_freeze" / "configs" / "n5m_adaptive_selection_contract.json"
    contract_path.parent.mkdir(parents=True, exist_ok=True)
    package_contract = Path(list(config_package_paths)[0]) / "n5m_adaptive_selection_contract.json"
    shutil.copy2(package_contract, contract_path)
    firm_frame.to_parquet(firm_path, index=False)
    state.to_parquet(state_path, index=False)
    return firm_path, state_path, contract_path


def run_smoke() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="n5m_selection_synthetic_") as td:
        root = Path(td) / "repo"
        firm_path, state_path, contract_path = _write_fixture(root)
        analysis_root = root / "data" / "analysis" / "paper_repro"
        output_dir = analysis_root / "03_output_contract_diagnostics" / "n5m_adaptive_selection"
        manifest = run_analysis(
            project_root=root,
            firm_frame_path=firm_path,
            state_panel_path=state_path,
            contract_path=contract_path,
            output_dir=output_dir,
        )
        verification = verify(root, analysis_root)
        if manifest.get("status") != "PASS" or verification.get("status") != "PASS":
            raise RuntimeError(f"producer/verifier failure: manifest={manifest.get('status')}, verification={verification}")

        feature_contract = json.loads((output_dir / "n5m_selection_feature_contract.json").read_text(encoding="utf-8"))
        selected = set(feature_contract["state_numeric_features"]) | set(feature_contract["state_categorical_features"])
        protected = {"candidate_id", "rating_num", "delta_R_score_alpha", "derived__not_prompt_visible"}
        leaked = sorted(selected & protected)
        if leaked:
            raise RuntimeError(f"protected columns entered synthetic selector: {leaked}")

        adaptive = pd.read_parquet(output_dir / "n5m_adaptive_budget_oof_predictions.parquet")
        gate = pd.read_parquet(output_dir / "n5m_postc4_gate_oof_predictions.parquet")
        assignments = pd.read_csv(output_dir / "n5m_selection_fold_assignments.csv")
        if adaptive["predicted_score"].isna().any() or gate["predicted_revision_gain"].isna().any():
            raise RuntimeError("synthetic OOF predictions contain missing values")
        if assignments.groupby(["analysis", "repeat"])["row_id"].nunique().ne(575).any():
            raise RuntimeError("synthetic grouped-fold assignment coverage failed")

        return {
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "data_source": "contract-faithful synthetic 575-firm x four-budget panel",
            "firm_frame_rows": 2300,
            "state_rows": 575,
            "adaptive_prediction_rows": int(len(adaptive)),
            "gate_prediction_rows": int(len(gate)),
            "selected_state_feature_count": int(len(selected)),
            "protected_columns_excluded": sorted(protected),
            "producer_status": manifest.get("status"),
            "verifier_status": verification.get("status"),
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
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
