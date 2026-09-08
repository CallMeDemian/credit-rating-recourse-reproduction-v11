from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import pandas as pd

from credit_recourse.rl.common.actions import load_action_space
from credit_recourse.rl.common.io import write_json
from credit_recourse.verification.stage_boundary_contracts import verify as verify_stage_boundary

from credit_recourse.rl.common.actions import ActionSpace
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.budget_contract import (
    BUDGET_AUDIT_COLUMNS,
    make_action_budget_contract,
)
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.llm_backends import (
    LLMRequest,
    ScriptedReproducibilityBackend,
)
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.pipeline import (
    _apply_stage7_row_selection,
)
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.prompt_builder import build_prompt
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.response_parser import (
    parse_response,
    project_free_form_batch,
    to_failure_audit_frame,
    to_policy_actions_frame,
)

ACTION_COLS = [
    "action__ppe_pct",
    "action__inv_turnover_chg",
    "action__ar_turnover_chg",
    "action__ap_turnover_chg",
    "action__short_debt_pct",
    "action__long_debt_pct",
    "action__bond_pct",
    "action__revenue_growth",
    "action__cogs_ratio_chg",
    "action__sga_ratio_chg",
]


def _make_space() -> ActionSpace:
    cols = list(ACTION_COLS)
    bounds = {c: (-2.0, 2.0) for c in cols}
    fixed = {
        "A0_noop": {c: 0.0 for c in cols},
        "DL1_deleverage_mild": {c: 0.0 for c in cols},
        "DL2_deleverage_moderate": {c: 0.0 for c in cols},
        "RF1_short_debt_refinance": {c: 0.0 for c in cols},
        "CX1_capex_discipline": {c: 0.0 for c in cols},
        "WC1_working_capital_tightening": {c: 0.0 for c in cols},
        "WC2_supplier_financing": {c: 0.0 for c in cols},
        "OE1_cost_efficiency_mild": {c: 0.0 for c in cols},
        "OE2_cost_efficiency_moderate": {c: 0.0 for c in cols},
        "MX1_cost_and_deleverage": {c: 0.0 for c in cols},
        "MX2_liquidity_rescue": {c: 0.0 for c in cols},
    }
    fixed["DL2_deleverage_moderate"].update({"action__short_debt_pct": -0.25, "action__long_debt_pct": -0.25})
    fixed["MX2_liquidity_rescue"].update({"action__short_debt_pct": -0.10, "action__inv_turnover_chg": 0.30})
    return ActionSpace(
        columns=cols,
        bounds=bounds,
        fixed_candidates=fixed,
        train_labels=list(fixed),
        row_conditional_baselines=[],
        final_rl_label="C3_candidate_iql",
        scenario_candidates={},
        diagnostic_candidates={},
        candidate_library_hash="synthetic",
        final_action_contract_hash="synthetic",
    )


def _make_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": list(range(12)),
            "firm_id": [f"F{i:03d}" for i in range(12)],
            "year": [2024] * 12,
            "sector_7": ["A", "A", "A", "B", "B", "B", "C", "C", "C", "C", "D", "D"],
            "grade_base_10": [1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5],
            "derived__debt_to_assets": [0.90, 0.85, 0.82, 0.20, 0.30, 0.40, 0.60, 0.65, 0.70, 0.20, 0.30, 0.40],
            "derived__debt_to_assets_sector_p75": [0.75] * 12,
            "derived__current_ratio": [0.8] * 12,
            "derived__current_ratio_sector_p25": [0.9] * 12,
            "derived__sga_ratio": [0.1] * 12,
            "derived__sga_ratio_sector_p75": [0.3] * 12,
            "derived__cogs_ratio": [0.3] * 12,
            "derived__cogs_ratio_sector_p75": [0.6] * 12,
            "derived__capex_to_assets": [0.1] * 12,
            "derived__capex_to_assets_sector_p75": [0.2] * 12,
            "derived__op_margin": [0.1] * 12,
            "derived__op_margin_sector_median": [0.05] * 12,
            "C3_candidate_iql": ["DL2_deleverage_moderate"] * 12,
        }
    )



def _write_synthetic_stage7_project(
    root: Path,
    *,
    action_df: pd.DataFrame,
    fail_df: pd.DataFrame,
    contract: dict,
) -> dict:
    """Materialize a minimal Stage7 output tree and run the real boundary verifier."""
    final = root / "data" / "final_freeze"
    cfg = final / "configs"
    stage2 = final / "stage2_candidate_projection"
    stage7 = final / "stage7_llm_action_generation"
    cfg.mkdir(parents=True, exist_ok=True)
    stage2.mkdir(parents=True, exist_ok=True)
    stage7.mkdir(parents=True, exist_ok=True)

    train_labels = [
        "A0_noop",
        "DL1_deleverage_mild",
        "DL2_deleverage_moderate",
        "RF1_short_debt_refinance",
        "CX1_capex_discipline",
        "WC1_working_capital_tightening",
        "WC2_supplier_financing",
        "OE1_cost_efficiency_mild",
        "OE2_cost_efficiency_moderate",
        "MX1_cost_and_deleverage",
        "MX2_liquidity_rescue",
    ]
    cols = list(ACTION_COLS)
    fixed = {name: {c: 0.0 for c in cols} for name in train_labels}
    fixed["DL2_deleverage_moderate"].update({"action__short_debt_pct": -0.25, "action__long_debt_pct": -0.25})
    fixed["MX2_liquidity_rescue"].update({"action__short_debt_pct": -0.10, "action__inv_turnover_chg": 0.30})
    action_contract = {
        "action_columns": cols,
        "action_bounds": {c: [-2.0, 2.0] for c in cols},
        "candidate_label_rule": {
            "train_labels": train_labels,
            "row_conditional_baselines": [],
            "final_rl_label": "C3_candidate_iql",
        },
    }
    cand_lib = {
        "main_train_labels": train_labels,
        "fixed_candidates": fixed,
        "scenario_candidates": {},
        "diagnostic_candidates": {},
        "final_rl_label": "C3_candidate_iql",
    }
    write_json(cfg / "final_action_contract.yaml", action_contract)
    write_json(cfg / "final_candidate_library.yaml", cand_lib)
    write_json(stage2 / "final_candidate_library__P50.yaml", cand_lib)
    space = load_action_space(root, candidate_library_path=stage2 / "final_candidate_library__P50.yaml")

    selected_rows = pd.DataFrame({"row_id": [int(action_df["row_id"].iloc[0])]})
    selected_rows.to_csv(stage7 / "llm_stage7_selected_row_ids.csv", index=False)
    parquet_engine_available = True
    try:
        action_df.to_parquet(stage7 / "llm_stage7_action_table.parquet", index=False)
        pd.DataFrame({
            "row_id": action_df["row_id"],
            "policy": action_df["policy"],
            "mode": action_df["mode"],
            "raw_response": ["{}"] * len(action_df),
        }).to_parquet(stage7 / "llm_stage7_response_log.parquet", index=False)
    except ImportError:
        # The production contract is parquet, but this smoke test must also run
        # in lean CI/sandbox environments without pyarrow/fastparquet.  Keep the
        # required filenames non-empty and monkeypatch pandas.read_parquet below
        # so the real Stage7 verifier logic is still exercised.
        parquet_engine_available = False
        action_df.to_csv(stage7 / "llm_stage7_action_table.parquet", index=False)
        pd.DataFrame({
            "row_id": action_df["row_id"],
            "policy": action_df["policy"],
            "mode": action_df["mode"],
            "raw_response": ["{}"] * len(action_df),
        }).to_csv(stage7 / "llm_stage7_response_log.parquet", index=False)
    fail_df.to_csv(stage7 / "llm_stage7_failure_audit.csv", index=False, encoding="utf-8-sig")
    prompt_payload_records = []
    for seq, row in action_df.reset_index(drop=True).iterrows():
        payload = {
            "instructions": ["synthetic Stage7 prompt archive contract smoke"],
            "row_id": int(row["row_id"]),
            "condition": str(row["policy"]),
            "mode": str(row["mode"]),
        }
        payload_text = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
        )
        prompt_payload_records.append({
            "schema_version": "stage7_prompt_payload_archive_v1",
            "request_seq_within_pass": int(seq),
            "request_fingerprint": hashlib.sha256(
                f"{row['row_id']}|{row['policy']}|{row['mode']}".encode("utf-8")
            ).hexdigest(),
            "row_id": int(row["row_id"]),
            "condition": str(row["policy"]),
            "mode": str(row["mode"]),
            "information_condition": "IC-a",
            "prompt_payload": payload,
            "prompt_sha256": hashlib.sha256(payload_text.encode("utf-8")).hexdigest(),
        })
    prompt_payload_path = stage7 / "llm_stage7_prompt_payloads.jsonl"
    prompt_payload_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in prompt_payload_records),
        encoding="utf-8",
    )
    prompt_payload_sha256 = hashlib.sha256(prompt_payload_path.read_bytes()).hexdigest()
    prompt_manifest = {
        "status": "PASS",
        "action_budget_contract": contract,
        "prompt_records": [],
        "prompt_payload_archive": {
            "schema_version": "stage7_prompt_payload_archive_v1",
            "path": prompt_payload_path.name,
            "sha256": prompt_payload_sha256,
            "record_count": len(prompt_payload_records),
            "preservation_policy": "full_structured_prompt_payload_per_request",
        },
    }
    write_json(stage7 / "llm_stage7_prompt_manifest.json", prompt_manifest)
    metadata = {
        "status": "PASS",
        "candidate_library_hash": space.candidate_library_hash,
        "candidate_library_path": str(stage2 / "final_candidate_library__P50.yaml"),
        "candidate_action_values_source": "stage2_recalibrated_candidate_library",
        "candidate_library_quantile": 50,
        "selected_recalibrated_candidate_library_hash": space.candidate_library_hash,
        "final_action_contract_hash": space.final_action_contract_hash,
        "backend_id": "scripted_reproducibility_v1",
        "backend_is_live": False,
        "final_paper_run_allowed": False,
        "reference_draw_seed": 1,
        "reference_source_policy": {"C6": "rl"},
        "information_condition": "IC-a",
        "action_budget_contract": contract,
        "action_budget_contract_schema_version": contract["schema_version"],
        "row_selection_contract": {
            "enabled": True,
            "selection_source": "row_id_file",
            "selected_row_ids_file": "llm_stage7_selected_row_ids.csv",
            "selected_row_count": int(len(selected_rows)),
            "input_panel_row_count_before_selection": int(len(selected_rows)),
            "input_panel_row_count_after_selection": int(len(selected_rows)),
        },
        "request_count": int(len(action_df)),
        "prompt_payload_archive_contract": {
            "schema_version": "stage7_prompt_payload_archive_v1",
            "status": "FULL_PAYLOAD_ARCHIVED",
            "path": prompt_payload_path.name,
            "sha256": prompt_payload_sha256,
            "record_count": len(prompt_payload_records),
        },
    }
    write_json(stage7 / "metadata.json", metadata)
    if parquet_engine_available:
        result = verify_stage_boundary(root, "stage7")
        parquet_mode = "native_parquet"
    else:
        old_read_parquet = pd.read_parquet
        pd.read_parquet = lambda path, *args, **kwargs: pd.read_csv(path)
        try:
            result = verify_stage_boundary(root, "stage7")
        finally:
            pd.read_parquet = old_read_parquet
        parquet_mode = "csv_backed_parquet_filenames_due_missing_engine"
    if result.get("status") != "PASS":
        raise AssertionError(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return {
        "stage_boundary_status": result.get("status"),
        "stage_boundary_warnings": result.get("warnings", []),
        "stage_boundary_parquet_mode": parquet_mode,
    }


def run_smoke() -> dict:
    space = _make_space()
    panel = _make_panel()
    legacy_contract = make_action_budget_contract(
        l1_budget=0.12,
        budgeted_conditions=["C6"],
        label="N5_SMOKE_L1_0p12",
        budgeted_modes=["free_form_10d"],
    ).to_dict()
    matched_contract = make_action_budget_contract(
        l1_budget=0.12,
        budgeted_conditions=["C4", "C6"],
        label="N5_MATCHED_SMOKE_L1_0p12",
        budgeted_modes=["free_form_10d"],
    ).to_dict()

    prompt_c6 = build_prompt(
        panel.iloc[0],
        condition="C6",
        mode="free_form_10d",
        information_condition="IC-a",
        space=space,
        rl_reference_candidate="DL2_deleverage_moderate",
        reference_source="rl",
        initial_action={"selected_candidate": "A0_noop"},
        action_budget_contract=legacy_contract,
    )
    assert prompt_c6["action_budget_contract"]["enabled"] is True
    assert any("N5 action-budget contract" in x for x in prompt_c6["instructions"])

    prompt_c4 = build_prompt(
        panel.iloc[0],
        condition="C4",
        mode="free_form_10d",
        information_condition="IC-a",
        space=space,
        action_budget_contract=legacy_contract,
    )
    assert prompt_c4["action_budget_contract"] is None
    assert not any("N5 action-budget contract" in x for x in prompt_c4["instructions"])

    matched_prompt_c4 = build_prompt(
        panel.iloc[0],
        condition="C4",
        mode="free_form_10d",
        information_condition="IC-a",
        space=space,
        action_budget_contract=matched_contract,
    )
    assert matched_prompt_c4["action_budget_contract"]["budgeted_conditions"] == ["C4", "C6"]
    assert any("N5 action-budget contract" in x for x in matched_prompt_c4["instructions"])
    matched_prompt_c6 = build_prompt(
        panel.iloc[0],
        condition="C6",
        mode="free_form_10d",
        information_condition="IC-a",
        space=space,
        rl_reference_candidate="DL2_deleverage_moderate",
        reference_source="rl",
        initial_action={"action_vector": {c.replace("action__", ""): 0.0 for c in ACTION_COLS}},
        action_budget_contract=matched_contract,
    )
    assert matched_prompt_c6["action_budget_contract"]["budgeted_conditions"] == ["C4", "C6"]

    backend = ScriptedReproducibilityBackend(seed=20260707)
    request = LLMRequest(
        row_id=0,
        condition="C6",
        mode="free_form_10d",
        information_condition="IC-a",
        prompt=prompt_c6,
        rl_reference_candidate="DL2_deleverage_moderate",
        reference_source="rl",
        initial_action={"selected_candidate": "A0_noop"},
    )
    parsed = parse_response(backend.generate(request), space)
    project_free_form_batch([parsed], space)
    assert parsed.budgeted_condition_flag is True
    assert parsed.budget_l1_target == 0.12
    assert parsed.budget_l1_raw is not None and parsed.budget_l1_raw <= 0.120000001
    assert parsed.budget_compliant_raw is True
    assert parsed.budget_compliant_clipped is True

    matched_parsed = []
    for condition, prompt, reference in [
        ("C4", matched_prompt_c4, None),
        ("C6", matched_prompt_c6, "DL2_deleverage_moderate"),
    ]:
        matched_request = LLMRequest(
            row_id=0,
            condition=condition,
            mode="free_form_10d",
            information_condition="IC-a",
            prompt=prompt,
            rl_reference_candidate=reference,
            reference_source=("rl" if condition == "C6" else "none"),
            initial_action=(matched_prompt_c6.get("initial_action") if condition == "C6" else None),
        )
        parsed_item = parse_response(backend.generate(matched_request), space)
        project_free_form_batch([parsed_item], space)
        assert parsed_item.budgeted_condition_flag is True
        assert parsed_item.budget_l1_target == 0.12
        assert parsed_item.budget_compliant_raw is True
        assert parsed_item.budget_compliant_clipped is True
        matched_parsed.append(parsed_item)

    action_df = to_policy_actions_frame([parsed], space)
    fail_df = to_failure_audit_frame([parsed])
    matched_action_df = to_policy_actions_frame(matched_parsed, space)
    matched_fail_df = to_failure_audit_frame(matched_parsed)
    for col in BUDGET_AUDIT_COLUMNS:
        assert col in action_df.columns, col
        assert col in fail_df.columns, col

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        row_file = out_dir / "n5_rows.csv"
        pd.DataFrame({"row_id": [0, 2, 4, 6]}).to_csv(row_file, index=False)
        selected, meta = _apply_stage7_row_selection(
            panel=panel,
            out_dir=out_dir,
            row_id_file=row_file,
            sample_size=None,
            sample_seed=20260707,
            sample_strata=None,
        )
        assert meta["enabled"] is True
        assert meta["selection_source"] == "row_id_file"
        assert meta["selected_row_count"] == 4
        assert selected["row_id"].tolist() == [0, 2, 4, 6]
        assert (out_dir / "llm_stage7_selected_row_ids.csv").exists()

        sampled, smeta = _apply_stage7_row_selection(
            panel=panel,
            out_dir=out_dir,
            row_id_file=None,
            sample_size=6,
            sample_seed=20260707,
            sample_strata=["sector_7", "grade_base_10"],
        )
        assert smeta["selection_source"] == "deterministic_stratified_sample"
        assert smeta["selected_row_count"] == 6
        assert len(sampled) == 6

    with tempfile.TemporaryDirectory() as td:
        boundary = _write_synthetic_stage7_project(
            Path(td), action_df=action_df, fail_df=fail_df, contract=legacy_contract
        )
    with tempfile.TemporaryDirectory() as td:
        matched_boundary = _write_synthetic_stage7_project(
            Path(td), action_df=matched_action_df, fail_df=matched_fail_df, contract=matched_contract
        )

    return {
        "status": "PASS",
        "stage_boundary_status": boundary["stage_boundary_status"],
        "stage_boundary_warning_count": len(boundary["stage_boundary_warnings"]),
        "stage_boundary_parquet_mode": boundary["stage_boundary_parquet_mode"],
        "matched_stage_boundary_status": matched_boundary["stage_boundary_status"],
        "matched_budgeted_policies": sorted(matched_action_df["policy"].astype(str).unique().tolist()),
        "budget_label": parsed.budget_contract_label,
        "budget_l1_raw": parsed.budget_l1_raw,
        "budget_l1_clipped": parsed.budget_l1_clipped,
        "row_selection_file_smoke_rows": 4,
        "row_selection_sample_smoke_rows": 6,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic smoke test for Stage7 N5 budget-contract patch.")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    args = ap.parse_args()
    result = run_smoke()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        print("N5 budget-contract smoke PASS")
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
