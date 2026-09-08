from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone

from credit_recourse.rl.common.io import write_json
from credit_recourse.rl.final_candidate.encoder import (
    FINAL_ENCODER_D_MODEL,
    FINAL_ENCODER_N_HEADS,
    FINAL_ENCODER_N_LAYERS,
    FINAL_ENCODER_FF_MULTIPLIER,
    FinalBlockAwareEncoder,
)
from credit_recourse.simulator.firm_state import FirmState
from credit_recourse.simulator.oracle_variables import compute_oracle_variables, ORACLE_FORMULA_REGISTRY
from credit_recourse.simulator.oracle_variable_contract import compute_contract_financial_variables
from credit_recourse.oracle.selected_variable_current_contract import verify_package_selected_variable_masters


EXPECTED_ENCODER_ARCH = {
    "d_model": 256,
    "n_heads": 8,
    "n_layers": 4,
    "ff_multiplier": 4,
}



def _expected_schema_artifact_path(project_root: Path) -> Path:
    materialized = project_root / "data" / "final_freeze" / "configs" / "final_expected_schema_artifacts.json"
    if materialized.exists():
        return materialized
    return Path(__file__).resolve().parents[1] / "configs" / "final_expected_schema_artifacts.json"


def _check_expected_schema_artifact_config(project_root: Path, errors: list[str]) -> dict:
    path = _expected_schema_artifact_path(project_root)
    if not path.exists():
        errors.append(f"missing expected schema artifact config: {path}")
        return {"path": str(path), "status": "MISSING"}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"failed to parse expected schema artifact config {path}: {exc}")
        return {"path": str(path), "status": "PARSE_ERROR"}
    arch = doc.get("architecture_required", {})
    expected = {
        "input_dim_continuous": 129,
        "input_dim_categorical": 2,
        "n_heads": FINAL_ENCODER_N_HEADS,
        "n_layers": FINAL_ENCODER_N_LAYERS,
        "ff_multiplier": FINAL_ENCODER_FF_MULTIPLIER,
    }
    found = {k: arch.get(k) for k in expected}
    for key, expected_value in expected.items():
        if arch.get(key) != expected_value:
            errors.append(
                f"expected schema artifact architecture mismatch for {key}: "
                f"expected={expected_value!r} found={arch.get(key)!r} path={path}"
            )
    hidden_allowed = arch.get("hidden_dim_allowed", [])
    if FINAL_ENCODER_D_MODEL not in hidden_allowed:
        errors.append(
            f"expected schema artifact hidden_dim_allowed does not include active d_model "
            f"{FINAL_ENCODER_D_MODEL}: found={hidden_allowed!r} path={path}"
        )
    return {
        "path": str(path),
        "status": "PASS" if all(found.get(k) == v for k, v in expected.items()) and FINAL_ENCODER_D_MODEL in hidden_allowed else "FAIL",
        "architecture_required_found": arch,
    }

def _check_encoder_arch(errors: list[str]) -> dict[str, int]:
    enc = FinalBlockAwareEncoder(n_features=129, block_ids=[0] * 129, direction_ids=[0] * 129, n_actions=10, n_acd_targets=118)
    found = {
        "d_model": int(FINAL_ENCODER_D_MODEL),
        "n_heads": int(FINAL_ENCODER_N_HEADS),
        "n_layers": int(FINAL_ENCODER_N_LAYERS),
        "ff_multiplier": int(FINAL_ENCODER_FF_MULTIPLIER),
    }
    attrs = {
        "d_model": int(getattr(enc, "d_model", -1)),
        "n_heads": int(getattr(enc, "n_heads", -1)),
        "n_layers": int(getattr(enc, "n_layers", -1)),
        "ff_multiplier": int(getattr(enc, "ff_multiplier", -1)),
    }
    if found != EXPECTED_ENCODER_ARCH:
        errors.append(f"encoder constants mismatch: expected={EXPECTED_ENCODER_ARCH} found={found}")
    if attrs != EXPECTED_ENCODER_ARCH:
        errors.append(f"encoder default attributes mismatch: expected={EXPECTED_ENCODER_ARCH} found={attrs}")
    return attrs


def _assert_close(errors: list[str], label: str, found: float | None, expected: float, tol: float = 1e-12) -> None:
    if found is None:
        errors.append(f"{label} missing: expected={expected}")
        return
    if abs(float(found) - expected) > tol:
        errors.append(f"{label} mismatch: expected={expected}, found={found}")


def _check_oracle_selected_variable_contract(errors: list[str]) -> dict[str, float | None]:
    prev = FirmState(
        firm_id="SMOKE",
        year=2023,
        sector="SMOKE",
        non_current_assets=200.0,
        capital_stock=50.0,
        total_liabilities=300.0,
        retained_earnings=100.0,
        operating_income=20.0,
        depreciation=5.0,
        amortization=1.0,
    )
    state = FirmState(
        firm_id="SMOKE",
        year=2024,
        sector="SMOKE",
        revenue=1000.0,
        pretax_income=100.0,
        retained_earnings=120.0,
        total_assets=600.0,
        financial_cost=20.0,
        current_assets=250.0,
        inventory=50.0,
        current_liabilities=100.0,
        payables=40.0,
        operating_cf=70.0,
        capex=20.0,
        capital_stock=50.0,
        non_current_assets=240.0,
        total_liabilities=320.0,
        operating_income=30.0,
        depreciation=6.0,
        amortization=2.0,
    )
    sim = compute_oracle_variables(state, prev_state=prev, exogenous={"ratio_missing_rate": 0.125})
    contract = compute_contract_financial_variables(state, prev)
    expected = {
        "R116": 2.0,
        "R133": 0.5,
        "R182": 0.2,
        "R136": 0.4,
    }
    _assert_close(errors, "R116 simulator formula", sim.get("R116"), expected["R116"])
    _assert_close(errors, "R116 contract formula", contract.get("R116"), expected["R116"])
    _assert_close(errors, "R133 simulator formula", sim.get("R133"), expected["R133"])
    _assert_close(errors, "R133 contract formula", contract.get("R133"), expected["R133"])
    _assert_close(errors, "R182 simulator formula", sim.get("R182"), expected["R182"])
    _assert_close(errors, "R182 contract formula", contract.get("R182"), expected["R182"])

    required_sources = {
        "R116": "(current_assets-inventory)/current_liabilities",
        "R133": "(operating_cf-capex)/current_liabilities",
        "R182": "non_current_assets_t/non_current_assets_t-1 - 1",
    }
    for vid, source in required_sources.items():
        meta = ORACLE_FORMULA_REGISTRY.get(vid, {})
        if meta.get("source") != source:
            errors.append(f"{vid} registry source mismatch: expected={source!r} found={meta.get('source')!r}")

    # Legacy R136 remains computable for old artifacts, but new Stage00_04
    # development excludes it from liquidity eligibility.
    _assert_close(errors, "legacy R136 simulator formula", sim.get("R136"), expected["R136"])
    if sim.get("ratio_missing_rate") != 0.125:
        errors.append(f"ratio_missing_rate passthrough mismatch: expected=0.125 found={sim.get('ratio_missing_rate')}")
    return {
        "simulator_R116": sim.get("R116"),
        "contract_R116": contract.get("R116"),
        "simulator_R133": sim.get("R133"),
        "contract_R133": contract.get("R133"),
        "simulator_R182": sim.get("R182"),
        "contract_R182": contract.get("R182"),
        "legacy_simulator_R136": sim.get("R136"),
        "passthrough_ratio_missing_rate": sim.get("ratio_missing_rate"),
    }



def _check_source_anchor_file(path: Path, anchors: list[str], errors: list[str], label: str) -> dict:
    if not path.exists():
        errors.append(f"missing {label} source file: {path}")
        return {"path": str(path), "status": "MISSING", "missing_anchors": anchors}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"cannot read {label} source file {path}: {exc}")
        return {"path": str(path), "status": "READ_ERROR", "error": str(exc), "missing_anchors": anchors}
    missing = [anchor for anchor in anchors if anchor not in text]
    if missing:
        errors.append(f"{label} source contract missing anchors: {missing}")
    return {
        "path": str(path),
        "status": "PASS" if not missing else "FAIL",
        "missing_anchors": missing,
        "checked_anchor_count": len(anchors),
    }


def _check_loopb2_b2gate_source_contract(project_root: Path, package_root: Path, errors: list[str]) -> dict:
    src_root = package_root.parent
    checks = {
        "stage1_shared_helpers": _check_source_anchor_file(
            src_root / "credit_recourse" / "oracle" / "verification" / "verify_stage1_substrate_validation.py",
            [
                "load_stage1_backend_score_panel",
                "compute_direction_agreement",
                "DIRECTION_AGREEMENT_CI_METHOD",
            ],
            errors,
            "Stage1 substrate shared helper",
        ),
        "stage2_loopb2_gate": _check_source_anchor_file(
            src_root / "credit_recourse" / "oracle" / "verification" / "verify_stage2_substrate_loopA_loopB2.py",
            [
                "_evaluate_loopb2_alpha_direction",
                "score_t_real",
                "delta_sim_alpha_score_t_to_pred_tplus1",
                "B2_MAX_B1_GAP",
                "B2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT",
                "B2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS",
                "threshold_min_b2_agreement_pct",
                "configured_minimum_agreement_50pct_diagnostic_after_검사3제거",
                "loopB2_oot_2020_2023",
                "gap_pp_missing_variable_diagnostics",
                "substrate_tier",
                "rule_status",
                "B2_PREV_STATE_UCODE_COLUMNS",
                "B2_PREV_STATE_JOIN_KEY_FIRM",
                "B2_NEXT_STATE_REQUIRED_FORMULA_FIELDS",
                "B2_SCORING_STATE_FIELD_GUARD_FIELDS",
                "B2_SELECTED_FORMULA_ID_PRESERVATION_GUARD_IDS",
                "B2_DIRECT_NEXT_RATIO_OVERLAY_POLICY",
                "B2_SCORING_POPULATION_POLICY",
                "B2_SIMULATOR_SCORE_RETENTION_POLICY",
                "B2_SIMULATOR_VALIDATION_TYPE",
                "B2_TRUE_SIMULATOR_ACTION_SOURCE",
                "B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY",
                "B2_TRUE_SIMULATOR_NO_NEXT_RATIO_OVERLAY_POLICY",
                "B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY",
                "B2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE",
                "_prepare_loopb2_actual_financial_source_frame",
                "cleaned_statement_panels",
                "재무상태표_clean.parquet",
                "financial_statement_items_raw.parquet",
                "long_statement_item_panel_pivoted_to_wide",
                "_load_loopb2_actual_financial_source",
                "_merge_loopb2_prev_state_inputs_from_actual_financial_panel",
                "stage1_actual_firm_year_financial_panel",
                "require_nonzero_source=True",
                "allow_observed_next_ratio_overlay=False",
                "B2_TRUE_SIMULATOR_ACTION_AUDIT_PREFIX",
                "sim_input__action__",
                "sim_input__action_observed__",
                "B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX",
                "B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX",
                "_load_loopb2_observed_action_value_source",
                "_merge_loopb2_observed_action_values_from_source",
                "observed_next_ratio_overlay_allowed",
                "B2_TRUE_SIMULATOR_ACTION_COLUMN_PREFIX",
                "_validate_loopb2_observed_action_source",
                "_observed_action_column_for_space_column",
                "_observed_action",
                "action__*",
                "action_observed__*",
                "C_obs_realized_action_values_from_stage2_raw_action_source_panel",
                "B2_REPLAY_DIAGNOSTIC_TYPE",
                "engineering_diagnostic_threshold_not_preregistered_research_gate",
                "configured_minimum_agreement_50pct_diagnostic_after_검사3제거",
                "threshold_min_b2_agreement_pct",
                "B2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS",
                "B2_STAGE2_DIAGNOSTIC_TIER",
                "B2_CURRENT_RESEARCH_GATE_POLICY",
                "_loopb2_target_row_alignment_for_lead_convention",
                "_build_loopb2_simulator_score_retention_frame",
                "_evaluate_loopb2_simulator_score_retention",
                "loopB2_simulator_score_retention.csv",
                "loopB2_replay_diagnostic",
                "simulator_financial_fidelity_gate",
                "historical_observed_action_resimulation",
                "_loopb2_observed_next_ratio_value",
                "_restrict_loopb2_scoring_population_to_observed",
                "direct_next_ratio_overlay_coverage",
                "scoring_population",
                "_find_base_state_ucode_input_column",
                "_add_loopb2_prev_state_join_keys",
                "_compute_loopb2_oracle_variables_preserving_selected_ids",
                "selected_formula_raw_id_preservation",
                "_merge_loopb2_prev_state_inputs_from_handoff",
                "_verify_loopb2_prev_state_ucode_loader_contract",
                "_registry_next_values",
                "_assert_loopb2_r157_r182_scoring_coverage",
                "normalised_loopb2_firm_key",
                "source_lookup_policy",
                "forced_firm_state_field_columns",
                "prev_state_loader_alias_precedence_guard",
                "state_field_coverage",
                "selected_formula_raw_id_preservation",
                "raw_formula_value_coverage",
                "direct_next_ratio_overlay_coverage",
                "scoring_population",
                "loopB2_simulator_score_retention_scoring_frame_meta_pre_guard.json",
                "r157_r182_nonnull_share_guard",
            ],
            errors,
            "Stage2 Loop B2 agreement gate",
        ),
        "stage_boundary_verifier": _check_source_anchor_file(
            src_root / "credit_recourse" / "verification" / "stage_boundary_contracts.py",
            [
                "verify_stage2_substrate_loopA_loopB2",
                "LOOPB2_REQUIRED_CSV_COLUMNS",
                "LOOPB2_EXPECTED_OBSERVED_TRANSITION_ROWS",
                "LOOPB2_MIN_R157_R182_NONNULL_SHARE",
                "LOOPB2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT",
                "LOOPB2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS",
                "threshold_min_b2_agreement_pct",
                "delta_sim_alpha_score_t_to_pred_tplus1",
                "score_t_loader",
                "prev_state_handoff_merge",
                "source_ucode_coverage",
                "forced_firm_state_field_columns",
                "normalised_loopb2_firm_key",
                "matched_share_any_required_field",
                "state_field_coverage",
                "LOOPB2_SCORING_POPULATION_POLICY_ANCHOR",
                "LOOPB2_DIRECT_NEXT_RATIO_OVERLAY_POLICY_ANCHOR",
                "LOOPB2_SIMULATOR_VALIDATION_TYPE",
                "LOOPB2_TRUE_SIMULATOR_ACTION_SOURCE",
                "LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX",
                "LOOPB2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX",
                "LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_AUDIT_PREFIX",
                "LOOPB2_TRUE_SIMULATOR_ACTION_FLAG_AUDIT_PREFIX",
                "LOOPB2_TRUE_SIMULATOR_PREV_STATE_SOURCE_ROLE",
                "LOOPB2_TRUE_SIMULATOR_PREV_STATE_POLICY_ANCHOR",
                "LOOPB2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE",
                "loopB2_prev_state_actual_financial_source_merge_for_loopA_driver",
                "actual_financial_source_contract",
                "prepared_source",
                "long_statement_item_panel_pivoted_to_wide",
                "wide_actual_financial_panel",
                "stage1_actual_firm_year_financial_panel",
                "require_nonzero_source",
                "nonzero_share",
                "sim_input__action__",
                "sim_input__action_observed__",
                "B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX",
                "B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX",
                "_load_loopb2_observed_action_value_source",
                "_merge_loopb2_observed_action_values_from_source",
                "observed_next_ratio_overlay_allowed",
                "LOOPB2_TRUE_SIMULATOR_ACTION_DIMS",
                "loopB2_observed_action_source_contract",
                "action__ppe_pct",
                "action_observed__ppe_pct",
                "action_source_policy",
                "LOOPB2_REPLAY_DIAGNOSTIC_TYPE",
                "LOOPB2_DIAGNOSTIC_RULE_ROLE",
                "configured_minimum_agreement_50pct_diagnostic_after_검사3제거",
                "LOOPB2_CURRENT_RESEARCH_GATE_POLICY_ANCHOR",
                "LOOPB2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS",
                "LOOPB2_STAGE2_DIAGNOSTIC_TIER",
                "loopB2_simulator_score_retention.csv",
                "simulator_financial_fidelity_gate",
                "valid_lead_aligned_rows",
                "selected_formula_raw_id_preservation",
                "raw_formula_value_coverage",
                "direct_next_ratio_overlay_coverage",
                "scoring_population",
                "loopB2_simulator_score_retention_scoring_frame_meta_pre_guard.json",
                "r157_r182_nonnull_share_guard",
            ],
            errors,
            "Loop B2 stage-boundary verifier",
        ),
        "runner_coverage": _check_source_anchor_file(
            project_root / "tools" / "internal" / "original" / "run_loopA_loopB2_stage2_extension.ps1",
            [
                "stage2_substrate_loopA_loopB2",
                "ConvertTo-BoolArg",
                "PreserveCurrentNonCurrentResidual",
                "Stage boundary verifier: Loop B2 B-gate contract",
            ],
            errors,
            "Loop A/B2 runner verifier coverage",
        ),
        "loopb2_b1_alignment_diagnostic": _check_source_anchor_file(
            src_root / "credit_recourse" / "oracle" / "verification" / "diagnose_loopb2_b1_alignment.py",
            [
                "Loop B2 vs Loop B1 sign/scale alignment diagnostic",
                "load_stage1_backend_score_panel",
                "score_t_real_minus_b1_score_t",
                "b2_rating_delta_minus_b1_improvement",
                "b2_current_delta_vs_rating_improvement",
                "b2_inverted_delta_vs_rating_improvement",
                "b1_lead_delta_vs_rating_improvement",
                "SYSTEMATIC_B2_SCORE_DELTA_INVERSION_SUSPECT",
                "loopB2_b1_b2_sign_alignment_report.json",
            ],
            errors,
            "Loop B2/B1 sign alignment diagnostic",
        ),
        "loopb2_b1_alignment_runner": _check_source_anchor_file(
            project_root / "tools" / "internal" / "original" / "run_loopB2_b1_b2_sign_diagnosis.ps1",
            [
                "diagnose_loopb2_b1_alignment",
                "LoopB2 vs Stage1 B1 sign/scale alignment diagnosis",
                "loopB2_b1_b2_sign_alignment_report.json",
            ],
            errors,
            "Loop B2/B1 sign alignment runner",
        ),
    }
    return {
        "status": "PASS" if all(v.get("status") == "PASS" for v in checks.values()) else "FAIL",
        "checks": checks,
    }


def run(project_root: Path) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    encoder_arch = _check_encoder_arch(errors)
    expected_schema_artifact = _check_expected_schema_artifact_config(project_root, errors)
    oracle_selected_variables = _check_oracle_selected_variable_contract(errors)
    package_root = Path(__file__).resolve().parents[1]
    selected_master_snapshot_sync = verify_package_selected_variable_masters(package_root)
    errors.extend(selected_master_snapshot_sync.get("errors", []))
    loopb2_b2gate_source_contract = _check_loopb2_b2gate_source_contract(project_root, package_root, errors)
    report = {
        "stage": "final_no_regression_contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "encoder_architecture_expected": EXPECTED_ENCODER_ARCH,
        "encoder_architecture_found": encoder_arch,
        "expected_schema_artifact_config": expected_schema_artifact,
        "oracle_selected_variable_smoke": oracle_selected_variables,
        "selected_variable_snapshot_sync": selected_master_snapshot_sync,
        "loopb2_b2gate_source_contract": loopb2_b2gate_source_contract,
        "errors": errors,
        "warnings": warnings,
    }
    out = project_root / "data" / "final_freeze" / "ledgers" / "final_no_regression_contract_report.json"
    write_json(out, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify final no-regression contract invariants.")
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args(argv)
    report = run(Path(args.project_root).resolve())
    print(report["status"])
    if report["errors"]:
        for err in report["errors"]:
            print("ERROR:", err)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
