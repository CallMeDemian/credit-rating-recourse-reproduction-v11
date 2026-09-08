from __future__ import annotations

"""Verify the complete canonical post-freeze paper-analysis tree."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.analysis.paper_output_layout import build_layout
from credit_recourse.contracts.paper_reproduction import analysis_output_dir, load_profile
from credit_recourse.verification.verify_n5m_adaptive_selection_contract import (
    verify as verify_n5m_adaptive_selection,
)

SCHEMA_VERSION = "paper_repro_output_contract_v4"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text.lower())


def _parse_bool_series(series: pd.Series) -> pd.Series:
    """Parse persisted CSV boolean values without truthiness shortcuts."""

    return series.astype(str).str.strip().str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )


def _verify_shuffle_cell(path: Path, *, expected_within: str, errors: list[str]) -> None:
    meta_path = path / "metadata.json"
    ci_path = path / "shuffle_permutation_ci.csv"
    draw_path = path / "shuffle_per_draw_summary.csv"
    for required in (meta_path, ci_path, draw_path):
        if not required.is_file():
            errors.append(f"shuffle cell missing required output: {required}")
            return
    try:
        meta = _read_json(meta_path)
    except Exception as exc:
        errors.append(f"shuffle metadata unreadable: {meta_path}: {exc!r}")
        return
    if meta.get("status") != "PASS":
        errors.append(f"shuffle cell status is not PASS: {path}: {meta.get('status')!r}")
    if str(meta.get("shuffle_within")) != expected_within:
        errors.append(
            f"shuffle cell mode mismatch: {path}: actual={meta.get('shuffle_within')!r}, expected={expected_within!r}"
        )
    if int(meta.get("shuffle_draw_count", -1)) != 1000:
        errors.append(f"shuffle cell draw count mismatch: {path}: {meta.get('shuffle_draw_count')!r}")
    try:
        stratum_count = int(meta.get("stratum_count", -1))
    except Exception:
        stratum_count = -1
    if expected_within == "none":
        if stratum_count != 1:
            errors.append(f"unconditional shuffle must have exactly one stratum: {path}: {stratum_count}")
    elif expected_within == "rating_band":
        if stratum_count < 2:
            errors.append(f"rating-band shuffle must have at least two strata: {path}: {stratum_count}")
    elif expected_within == "industry":
        try:
            unique_known = int(meta.get("stratum_unique_known", -1))
            known_fraction = float(meta.get("stratum_known_fraction", -1.0))
        except Exception:
            unique_known, known_fraction = -1, -1.0
        if meta.get("stratum_resolution_status") != "PASS":
            errors.append(f"industry stratum resolution did not PASS: {path}")
        if stratum_count < 2 or unique_known < 2:
            errors.append(
                f"industry shuffle is not informative: {path}: stratum_count={stratum_count}, unique_known={unique_known}"
            )
        if known_fraction < 0.90:
            errors.append(f"industry shuffle coverage below 0.90: {path}: {known_fraction}")
        if not _is_sha256(meta.get("stratum_contract_sha256")):
            errors.append(f"industry shuffle lacks a valid stratum contract SHA-256: {path}")
    try:
        ci = pd.read_csv(ci_path)
    except Exception as exc:
        errors.append(f"shuffle CI unreadable: {ci_path}: {exc!r}")
        return
    if "shuffle_stratum_count" not in ci.columns:
        errors.append(f"shuffle CI missing shuffle_stratum_count: {ci_path}")
    else:
        observed = pd.to_numeric(ci["shuffle_stratum_count"], errors="coerce")
        if observed.isna().any() or not observed.astype(int).eq(stratum_count).all():
            errors.append(
                f"shuffle CI stratum count does not match metadata: {ci_path}: expected={stratum_count}"
            )


def _verify_frontier_contract(
    source_dir: Path,
    *,
    matched: bool,
    required: bool,
    errors: list[str],
) -> dict[str, Any]:
    manifest_path = source_dir / "n5_budget_frontier_holm_manifest.json"
    if not manifest_path.is_file():
        if required:
            errors.append(f"required frontier manifest missing: {manifest_path}")
        return {"status": "MISSING", "source_dir": str(source_dir)}
    try:
        meta = _read_json(manifest_path)
    except Exception as exc:
        errors.append(f"frontier manifest unreadable: {manifest_path}: {exc!r}")
        return {"status": "FAIL", "source_dir": str(source_dir)}
    expected_role = "paper_n5_matched_budget_frontier_icb" if matched else "paper_n5_budget_frontier_icb"
    expected_design = "matched_c4_c6" if matched else "legacy_c6_only"
    schema = meta.get("schema_version")
    if meta.get("status") != "PASS":
        errors.append(f"frontier manifest not PASS: {manifest_path}")
    if meta.get("run_role") != expected_role:
        errors.append(f"frontier role mismatch: {manifest_path}: {meta.get('run_role')!r}")
    if matched:
        if schema != "n5_generation_budget_frontier_holm_v2" or meta.get("design") != expected_design:
            errors.append(f"matched N5M frontier must be v2/matched_c4_c6: {manifest_path}")
        audits = [
            source_dir / "n5_budget_frontier_control_alignment_audit.csv",
            source_dir / "n5_budget_frontier_budget_contract_audit.csv",
        ]
    elif schema == "n5_generation_budget_frontier_holm_v2":
        if meta.get("design") != expected_design:
            errors.append(f"supplementary N5F v2 design mismatch: {manifest_path}")
        audits = [
            source_dir / "n5_budget_frontier_control_alignment_audit.csv",
            source_dir / "n5_budget_frontier_budget_contract_audit.csv",
        ]
    elif schema == "n5_generation_budget_frontier_holm_v1":
        audits = [source_dir / "n5_budget_frontier_c4_stability_audit.csv"]
    else:
        errors.append(f"unsupported frontier schema: {manifest_path}: {schema!r}")
        audits = []
    required_paths = [source_dir / "n5_budget_frontier_table_patch.csv", *audits]
    for path in required_paths:
        if not path.is_file():
            errors.append(f"frontier output missing: {path}")
    table_path = source_dir / "n5_budget_frontier_table_patch.csv"
    if table_path.is_file():
        table = pd.read_csv(table_path)
        if len(table) != 4 or set(table.get("budget_label", pd.Series(dtype=str)).astype(str)) != {
            "0p75", "1p27", "2p00", "unbounded"
        }:
            errors.append(f"frontier table does not contain exact four-arm grid: {table_path}")
        if matched or schema == "n5_generation_budget_frontier_holm_v2":
            required_columns = {
                "design", "oracle_backend", "budget_label", "l1_budget", "mean_C4", "mean_C6",
                "mean_C6_minus_C4", "C6_minus_C4_p_holm",
                "interaction_or_package_DID_gap", "interaction_or_package_DID_p_holm",
                "interaction_or_package_DID_sig",
            }
            missing_columns = sorted(required_columns - set(table.columns))
            if missing_columns:
                errors.append(f"frontier v2 table missing columns: {table_path}: {missing_columns}")
    return {
        "status": meta.get("status"),
        "schema_version": schema,
        "design": meta.get("design") or expected_design,
        "run_role": meta.get("run_role"),
        "source_dir": str(source_dir),
    }


def _verify_n5m_posthoc(layout, errors: list[str]) -> dict[str, Any]:
    source_dir = layout.n5m_posthoc
    manifest_path = source_dir / "n5m_posthoc_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"required N5M post-hoc manifest missing: {manifest_path}")
        return {"status": "MISSING"}
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "n5m_posthoc_v4" or manifest.get("status") != "PASS":
        errors.append(f"N5M post-hoc manifest contract mismatch: {manifest_path}")
    expected_counts = {
        "run_count": 4,
        "firm_count": 575,
        "stage7_action_row_count": 4600,
        "stage9_revision_row_count": 2300,
        "stage8_feasibility_row_count": 4600,
        "sustainability_critical_row_count": 55,
        "sustainability_critical_unique_firm_count": 12,
    }
    for key, expected in expected_counts.items():
        try:
            actual = int(manifest.get(key, -1))
        except Exception:
            actual = -1
        if actual != expected:
            errors.append(f"N5M post-hoc {key}={actual}, expected={expected}")
    main_contract = manifest.get("main_harness_backend_decomposition_contract", {})
    if main_contract.get("included") is not False or set(main_contract.get("excluded_run_families", [])) != {"N5", "N5F", "N5M"}:
        errors.append("N5M post-hoc does not enforce exclusion from main harness-vs-backend decomposition")
    if manifest.get("auxiliary_variance_contract", {}).get("status") != "SUPPLEMENTARY_ONLY":
        errors.append("N5M auxiliary variance is not marked SUPPLEMENTARY_ONLY")
    required_files = [
        "n5m_alignment_audit.csv",
        "n5m_win_tie_loss_by_budget_oracle.csv",
        "n5m_reference_quality_by_budget.csv",
        "n5m_reference_quality_quartiles.csv",
        "n5m_firm_frame.parquet",
        "n5m_cross_oracle_local_q_gain.csv",
        "n5m_reference_axis_anatomy.parquet",
        "n5m_reference_axis_outcome_summary.csv",
        "n5m_oracle_consensus_by_budget.csv",
        "n5m_action_axis_c4_c6_differences.csv",
        "n5m_dimension_overlap_summary.csv",
        "n5m_reallocation_by_outcome_group.csv",
        "n5m_revision_distance_score_relationship.csv",
        "n5m_feasibility_by_budget_policy.csv",
        "n5m_feasibility_unique_firms.csv",
        "n5m_auxiliary_within_firm_variance.csv",
        "n5m_score_budget_auditability_operating_points.csv",
        "n5m_vs_c3_paired_holm.csv",
        "n5m_adoption_score_relationship.csv",
        "n5m_adoption_quartiles.csv",
    ]
    for name in required_files:
        if not (source_dir / name).is_file():
            errors.append(f"N5M post-hoc output missing: {source_dir / name}")
    win_path = source_dir / "n5m_win_tie_loss_by_budget_oracle.csv"
    if win_path.is_file():
        win = pd.read_csv(win_path)
        if len(win) != 12 or not pd.to_numeric(win.get("n_firms"), errors="coerce").eq(575).all():
            errors.append("N5M win/tie/loss must contain 4 budgets x 3 oracles x 575 firms")
        alpha = win[win.get("oracle_backend", "").astype(str).eq("alpha")].copy()
        expected = {"0p75": 0.2740740740740741, "1p27": 0.5555555555555556, "2p00": 0.56875, "unbounded": 0.8284023668639053}
        observed = dict(zip(alpha["budget_label"].astype(str), pd.to_numeric(alpha["c6_win_rate_non_tie"], errors="coerce")))
        for budget, value in expected.items():
            if budget not in observed or not np.isclose(observed[budget], value, atol=1e-12, rtol=0):
                errors.append(f"N5M alpha non-tie win rate mismatch for {budget}: {observed.get(budget)!r}")
    aux_path = source_dir / "n5m_auxiliary_within_firm_variance.csv"
    if aux_path.is_file():
        aux = pd.read_csv(aux_path)
        if len(aux) != 3:
            errors.append("N5M auxiliary variance must contain three Oracle rows")
        if not aux.get("supplementary_only", pd.Series(dtype=bool)).astype(bool).all():
            errors.append("N5M auxiliary variance rows are not all supplementary_only")
        if not aux.get("excluded_from_main_harness_backend_decomposition", pd.Series(dtype=bool)).astype(bool).all():
            errors.append("N5M auxiliary variance rows are not excluded from main decomposition")
    vs_c3_contract = manifest.get("vs_c3_contrast_contract", {})
    if vs_c3_contract.get("canonical_reference_policy") != "C3_candidate_iql":
        errors.append("N5M versus-C3 contract does not pin exact C3_candidate_iql")
    if int(vs_c3_contract.get("row_count", -1) or -1) != 24:
        errors.append("N5M versus-C3 contract must contain 24 contrasts")
    if vs_c3_contract.get("holm_family") != "policy_x_oracle_across_four_budgets":
        errors.append("N5M versus-C3 Holm family mismatch")
    firm_contract = manifest.get("firm_frame_contract", {})
    if int(firm_contract.get("row_count", -1) or -1) != 2300:
        errors.append("N5M firm-frame contract must contain 2,300 rows")
    if firm_contract.get("post_c4_gate_feature_status") != "AVAILABLE":
        errors.append("N5M firm-frame does not expose the frozen post-C4 gate feature contract")
    if list(firm_contract.get("post_c4_gate_feature_columns", [])) != [
        "c4_final_l1",
        "c4_projection_distance",
        "c4_active_dimensions",
    ]:
        errors.append("N5M firm-frame post-C4 gate feature list mismatch")
    cross_contract = manifest.get("cross_oracle_shared_term_sensitivity_contract", {})
    if int(cross_contract.get("row_count", -1) or -1) != 24:
        errors.append("N5M cross-Oracle local-Q contract must contain 24 off-diagonal rows")
    if int(cross_contract.get("holm_family_size", -1) or -1) != 24:
        errors.append("N5M cross-Oracle Holm family size must be 24")
    axis_contract = manifest.get("reference_axis_anatomy_contract", {})
    if int(axis_contract.get("detail_row_count", -1) or -1) != 2300:
        errors.append("N5M reference-axis detail must contain 2,300 rows")
    if int(axis_contract.get("summary_row_count", -1) or -1) != 24:
        errors.append("N5M reference-axis summary must contain 24 rows")
    consensus_contract = manifest.get("oracle_consensus_contract", {})
    if int(consensus_contract.get("row_count", -1) or -1) != 4:
        errors.append("N5M Oracle-consensus contract must contain four budget rows")

    adoption_contract = manifest.get("adoption_diagnostics_contract", {})
    if int(adoption_contract.get("relationship_row_count", -1) or -1) != 12:
        errors.append("N5M adoption relationship contract must contain 12 rows")
    if int(adoption_contract.get("quartile_row_count", -1) or -1) != 48:
        errors.append("N5M adoption quartile contract must contain 48 rows")

    vs_c3_path = source_dir / "n5m_vs_c3_paired_holm.csv"
    if vs_c3_path.is_file():
        vs_c3 = pd.read_csv(vs_c3_path)
        required = {
            "budget_label", "policy", "oracle_backend", "n_pairs",
            "mean_gap", "wilcoxon_p_raw", "wilcoxon_p_holm", "sig_holm",
            "reference_policy",
        }
        missing = sorted(required - set(vs_c3.columns))
        if missing:
            errors.append(f"N5M versus-C3 table missing columns: {missing}")
        if len(vs_c3) != 24:
            errors.append(f"N5M versus-C3 table rows={len(vs_c3)}, expected=24")
        if "reference_policy" in vs_c3.columns and not vs_c3["reference_policy"].astype(str).eq("C3_candidate_iql").all():
            errors.append("N5M versus-C3 table contains non-canonical C3 variants")
        if "n_pairs" in vs_c3.columns and not pd.to_numeric(vs_c3["n_pairs"], errors="coerce").eq(575).all():
            errors.append("N5M versus-C3 contrasts are not based on 575 paired firms")
        if {"budget_label", "policy", "oracle_backend"}.issubset(vs_c3.columns):
            expected_grid = {
                (budget, policy, oracle)
                for budget in ("0p75", "1p27", "2p00", "unbounded")
                for policy in ("C4", "C6")
                for oracle in ("alpha", "beta", "gamma")
            }
            observed_grid = set(map(tuple, vs_c3[["budget_label", "policy", "oracle_backend"]].astype(str).itertuples(index=False, name=None)))
            if observed_grid != expected_grid:
                errors.append("N5M versus-C3 contrast grid is not exact 4 budgets x 2 policies x 3 Oracles")
        if "mean_gap" in vs_c3.columns and not pd.to_numeric(vs_c3["mean_gap"], errors="coerce").gt(0).all():
            errors.append("N5M versus-C3 contains a non-positive mean gap")
        if "wilcoxon_p_holm" in vs_c3.columns and not pd.to_numeric(vs_c3["wilcoxon_p_holm"], errors="coerce").le(0.05).all():
            errors.append("N5M versus-C3 contains a non-significant Holm-adjusted contrast")

    cross_path = source_dir / "n5m_cross_oracle_local_q_gain.csv"
    if cross_path.is_file():
        cross = pd.read_csv(cross_path)
        required = {"budget_label", "local_q_oracle", "revision_gain_oracle", "n_complete", "spearman_rho", "p_raw", "p_holm_24", "sig_holm_24"}
        missing = sorted(required - set(cross.columns))
        if missing:
            errors.append(f"N5M cross-Oracle local-Q table missing columns: {missing}")
        if len(cross) != 24:
            errors.append(f"N5M cross-Oracle local-Q rows={len(cross)}, expected=24")
        if {"local_q_oracle", "revision_gain_oracle"}.issubset(cross.columns) and (cross["local_q_oracle"].astype(str) == cross["revision_gain_oracle"].astype(str)).any():
            errors.append("N5M cross-Oracle local-Q table contains on-diagonal rows")

    axis_summary_path = source_dir / "n5m_reference_axis_outcome_summary.csv"
    if axis_summary_path.is_file():
        axis = pd.read_csv(axis_summary_path)
        required = {"budget_label", "oracle_backend", "metric", "n_complete", "spearman_rho_vs_revision_gain", "p_raw", "p_holm_within_oracle_metric", "sig_holm"}
        missing = sorted(required - set(axis.columns))
        if missing:
            errors.append(f"N5M reference-axis summary missing columns: {missing}")
        if len(axis) != 24:
            errors.append(f"N5M reference-axis summary rows={len(axis)}, expected=24")

    consensus_path = source_dir / "n5m_oracle_consensus_by_budget.csv"
    if consensus_path.is_file():
        consensus = pd.read_csv(consensus_path)
        required = {"budget_label", "n_firms", "all_three_loss_fraction", "majority_loss_fraction", "all_three_win_fraction", "majority_win_fraction", "mixed_oracle_direction_fraction"}
        missing = sorted(required - set(consensus.columns))
        if missing:
            errors.append(f"N5M Oracle-consensus table missing columns: {missing}")
        if len(consensus) != 4 or not pd.to_numeric(consensus.get("n_firms"), errors="coerce").eq(575).all():
            errors.append("N5M Oracle-consensus table must contain four 575-firm budget rows")

    adoption_path = source_dir / "n5m_adoption_score_relationship.csv"
    if adoption_path.is_file():
        adoption = pd.read_csv(adoption_path)
        if len(adoption) != 12:
            errors.append(f"N5M adoption relationship rows={len(adoption)}, expected=12")
        required = {
            "budget_label", "oracle_backend", "n_firms", "n_metrics_defined",
            "n_complete", "spearman_adoption_vs_revision_gain", "p_raw", "p_holm", "sig_holm",
        }
        missing = sorted(required - set(adoption.columns))
        if missing:
            errors.append(f"N5M adoption relationship table missing columns: {missing}")
        alpha_075 = adoption[
            adoption.get("budget_label", pd.Series(dtype=str)).astype(str).eq("0p75")
            & adoption.get("oracle_backend", pd.Series(dtype=str)).astype(str).eq("alpha")
        ]
        if len(alpha_075) != 1:
            errors.append("N5M adoption relationship lacks unique alpha/0p75 cell")
        else:
            rho = float(pd.to_numeric(alpha_075["spearman_adoption_vs_revision_gain"], errors="coerce").iloc[0])
            if not np.isclose(rho, -0.4501731165305016, atol=1e-12, rtol=0):
                errors.append(f"N5M alpha/0p75 adoption-gain Spearman mismatch: {rho}")

    adoption_quartiles_path = source_dir / "n5m_adoption_quartiles.csv"
    if adoption_quartiles_path.is_file():
        quartiles = pd.read_csv(adoption_quartiles_path)
        if len(quartiles) != 48:
            errors.append(f"N5M adoption quartile rows={len(quartiles)}, expected=48")
        alpha_075_q4 = quartiles[
            quartiles.get("budget_label", pd.Series(dtype=str)).astype(str).eq("0p75")
            & quartiles.get("oracle_backend", pd.Series(dtype=str)).astype(str).eq("alpha")
            & quartiles.get("adoption_quartile", pd.Series(dtype=str)).astype(str).eq("Q4")
        ]
        if len(alpha_075_q4) != 1:
            errors.append("N5M adoption quartiles lack unique alpha/0p75/Q4 cell")
        else:
            mean_gain = float(pd.to_numeric(alpha_075_q4["mean_revision_gain"], errors="coerce").iloc[0])
            if not np.isclose(mean_gain, -0.3526592985915261, atol=1e-12, rtol=0):
                errors.append(f"N5M alpha/0p75/Q4 mean revision gain mismatch: {mean_gain}")
    return {"status": manifest.get("status"), "schema_version": manifest.get("schema_version")}



def _verify_main_harness_backend_decomposition(layout, errors: list[str]) -> dict[str, Any]:
    source_dir = layout.main_harness_backend_decomposition
    manifest_path = source_dir / "main_harness_backend_decomposition_manifest.json"
    if not manifest_path.is_file():
        errors.append(f"required main harness-backend decomposition manifest missing: {manifest_path}")
        return {"status": "MISSING"}
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != "main_harness_backend_decomposition_v4":
        errors.append("main harness-backend decomposition schema mismatch")
    if manifest.get("status") != "PASS":
        errors.append("main harness-backend decomposition manifest is not PASS")
    if manifest.get("panel_contract") != "common_crossed_icb_harness_backend_panel_v3":
        errors.append("main harness-backend panel contract mismatch")
    if manifest.get("solver_contract") != "exact_group_projection_plus_small_additive_lstsq_v1":
        errors.append("main harness-backend solver contract mismatch")
    solver = manifest.get("solver_diagnostics")
    if not isinstance(solver, dict):
        errors.append("main harness-backend solver diagnostics are missing")
        solver = {}
    if solver.get("dense_firm_dummy_lstsq_used") is not False:
        errors.append("main harness-backend dense firm-dummy least squares must remain disabled")
    if solver.get("firm_fe_solver") != "exact_group_mean_projection":
        errors.append("main harness-backend firm-FE solver mismatch")
    if solver.get("one_way_solver") != "exact_group_mean_projection":
        errors.append("main harness-backend one-way solver mismatch")
    if solver.get("additive_solver") != "numpy_lstsq_small_design":
        errors.append("main harness-backend additive solver mismatch")
    expected_scalars = {"firm_count": 575, "backend_count": 3, "harness_cell_count": 12}
    for key, expected in expected_scalars.items():
        try:
            actual = int(manifest.get(key, -1))
        except Exception:
            actual = -1
        if actual != expected:
            errors.append(f"main harness-backend {key}={actual}, expected={expected}")
    if manifest.get("n5_n5f_n5m_included") is not False:
        errors.append("main harness-backend decomposition includes a forbidden N5 family")
    if set(manifest.get("forbidden_run_families", [])) != {"N5", "N5F", "N5M"}:
        errors.append("main harness-backend forbidden run family contract mismatch")
    expected_roles = {"paper_primary_gpt54", "paper_supplementary_gpt41", "paper_supplementary_haiku45"}
    if set(manifest.get("run_role_backend_labels", {})) != expected_roles:
        errors.append("main harness-backend run role set mismatch")
    role_contract = manifest.get("run_role_resolution_contract")
    allowed_role_sources = {
        "archive_manifest.run_role",
        "stage7.metadata.run_role",
        "paper_reproduction.inspect_archived_run.legacy_role_bridge",
    }
    if not isinstance(role_contract, dict):
        errors.append("main harness-backend run-role resolution contract is missing")
        role_contract = {}
    if set(role_contract.get("allowed_sources", [])) != allowed_role_sources:
        errors.append("main harness-backend run-role allowed-source contract mismatch")
    if role_contract.get("resolver") != "credit_recourse.contracts.paper_reproduction.inspect_archived_run":
        errors.append("main harness-backend run-role resolver mismatch")
    if role_contract.get("fresh_runs_must_write_explicit_run_role") is not True:
        errors.append("main harness-backend fresh-run explicit-role policy is not enforced")
    if role_contract.get("archive_manifest_optional_only_for_legacy_bridge") is not True:
        errors.append("main harness-backend legacy archive-manifest exception policy is missing")
    archive_presence = manifest.get("archive_manifest_presence_contract")
    if not isinstance(archive_presence, dict):
        errors.append("main harness-backend archive-manifest presence contract is missing")
        archive_presence = {}
    if archive_presence.get("required_for_fresh_or_explicit_role_runs") is not True:
        errors.append("main harness-backend fresh/explicit archive-manifest requirement is not enforced")
    if archive_presence.get("allowed_missing_role_source") != (
        "paper_reproduction.inspect_archived_run.legacy_role_bridge"
    ):
        errors.append("main harness-backend allowed missing archive-manifest source mismatch")
    try:
        explicit_count = int(role_contract.get("explicit_role_count", -1))
        legacy_count = int(role_contract.get("legacy_bridge_count", -1))
    except Exception:
        explicit_count = legacy_count = -1
    if explicit_count < 0 or legacy_count < 0 or explicit_count + legacy_count != 3:
        errors.append(
            "main harness-backend run-role resolution counts must be nonnegative and sum to three"
        )
    try:
        archive_present_count = int(archive_presence.get("present_count", -1))
        archive_missing_count = int(archive_presence.get("missing_legacy_count", -1))
    except Exception:
        archive_present_count = archive_missing_count = -1
    if (
        archive_present_count < 0
        or archive_missing_count < 0
        or archive_present_count + archive_missing_count != 3
    ):
        errors.append(
            "main harness-backend archive-manifest presence counts must be nonnegative and sum to three"
        )
    required_files = [
        "main_harness_backend_panel.parquet",
        "main_harness_backend_decomposition.csv",
        "main_harness_backend_cell_means.csv",
        "main_harness_backend_swing_summary.csv",
        "main_harness_backend_alignment_audit.csv",
        "main_harness_backend_missing_cell_audit.csv",
        "main_harness_backend_input_files.csv",
    ]
    for name in required_files:
        if not (source_dir / name).is_file():
            errors.append(f"main harness-backend output missing: {source_dir / name}")
    decomposition_path = source_dir / "main_harness_backend_decomposition.csv"
    if decomposition_path.is_file():
        decomp = pd.read_csv(decomposition_path)
        required = {
            "oracle_backend", "n_firms", "n_backends", "n_harness_cells", "n_observations",
            "firm_fe_r2_total_variance", "harness_main_effect_r2_within_firm",
            "backend_main_effect_r2_within_firm", "partial_harness_given_backend_r2_within_firm",
            "partial_backend_plus_interaction_given_harness_r2_within_firm",
            "solver_contract", "firm_fe_solver", "one_way_solver", "additive_solver",
        }
        missing = sorted(required - set(decomp.columns))
        if missing:
            errors.append(f"main harness-backend decomposition missing columns: {missing}")
        if len(decomp) != 3 or set(decomp.get("oracle_backend", pd.Series(dtype=str)).astype(str)) != {"alpha", "beta", "gamma"}:
            errors.append("main harness-backend decomposition must contain exactly alpha/beta/gamma")
        if "n_firms" in decomp.columns and not pd.to_numeric(decomp["n_firms"], errors="coerce").eq(575).all():
            errors.append("main harness-backend decomposition does not use 575 firms")
        if "n_backends" in decomp.columns and not pd.to_numeric(decomp["n_backends"], errors="coerce").eq(3).all():
            errors.append("main harness-backend decomposition does not use three backends")
        if "n_harness_cells" in decomp.columns and not pd.to_numeric(decomp["n_harness_cells"], errors="coerce").eq(12).all():
            errors.append("main harness-backend decomposition does not use 12 harness cells")
        if "solver_contract" in decomp.columns and not decomp["solver_contract"].astype(str).eq(
            "exact_group_projection_plus_small_additive_lstsq_v1"
        ).all():
            errors.append("main harness-backend decomposition row solver contract mismatch")
        if "firm_fe_solver" in decomp.columns and not decomp["firm_fe_solver"].astype(str).eq(
            "exact_group_mean_projection"
        ).all():
            errors.append("main harness-backend decomposition firm-FE row solver mismatch")
    alignment_path = source_dir / "main_harness_backend_alignment_audit.csv"
    if alignment_path.is_file():
        alignment = pd.read_csv(alignment_path)
        if len(alignment) != 3:
            errors.append("main harness-backend alignment audit must contain three runs")
        if set(alignment.get("run_role", pd.Series(dtype=str)).astype(str)) != expected_roles:
            errors.append("main harness-backend alignment run roles mismatch")
        required_role_columns = {
            "run_role_source",
            "run_role_explicit",
            "archive_manifest_present",
        }
        missing_role_columns = sorted(required_role_columns - set(alignment.columns))
        if missing_role_columns:
            errors.append(
                f"main harness-backend alignment lacks run-role provenance columns: {missing_role_columns}"
            )
        else:
            sources = set(alignment["run_role_source"].astype(str))
            if not sources.issubset(allowed_role_sources):
                errors.append(f"main harness-backend alignment has forbidden role sources: {sorted(sources)}")
            explicit_flags = _parse_bool_series(alignment["run_role_explicit"])
            archive_flags = _parse_bool_series(alignment["archive_manifest_present"])
            if explicit_flags.isna().any():
                errors.append("main harness-backend alignment has invalid run_role_explicit values")
            else:
                legacy_mask = alignment["run_role_source"].astype(str).eq(
                    "paper_reproduction.inspect_archived_run.legacy_role_bridge"
                )
                if not explicit_flags.eq(~legacy_mask).all():
                    errors.append(
                        "main harness-backend run_role_explicit flags disagree with role provenance"
                    )
                if int(explicit_flags.sum()) != explicit_count or int((~explicit_flags).sum()) != legacy_count:
                    errors.append(
                        "main harness-backend alignment role counts disagree with manifest"
                    )
            if archive_flags.isna().any():
                errors.append("main harness-backend alignment has invalid archive_manifest_present values")
            else:
                missing_archive = ~archive_flags
                if (missing_archive & ~legacy_mask).any():
                    errors.append(
                        "main harness-backend missing archive manifests are not restricted to the legacy bridge"
                    )
                if (missing_archive & explicit_flags.fillna(True)).any():
                    errors.append(
                        "main harness-backend explicit-role runs may not omit archive_manifest.json"
                    )
                if int(archive_flags.sum()) != archive_present_count or int(missing_archive.sum()) != archive_missing_count:
                    errors.append(
                        "main harness-backend alignment archive-manifest counts disagree with manifest"
                    )
        coverage = pd.to_numeric(alignment.get("observation_coverage"), errors="coerce")
        if coverage.isna().any() or not coverage.ge(0.99).all():
            errors.append("main harness-backend backend coverage below 0.99")
        if not alignment.get("stage7_stage8_key_alignment", pd.Series(dtype=str)).astype(str).eq("PASS").all():
            errors.append("main harness-backend Stage7/Stage8 key alignment is not PASS")
    input_files_path = source_dir / "main_harness_backend_input_files.csv"
    if input_files_path.is_file():
        input_files = pd.read_csv(input_files_path, keep_default_na=False)
        required_input_columns = {
            "run_label",
            "run_role",
            "run_role_source",
            "artifact",
            "path",
            "artifact_present",
            "absence_policy",
            "sha256",
            "row_count",
        }
        missing_input_columns = sorted(required_input_columns - set(input_files.columns))
        if missing_input_columns:
            errors.append(
                f"main harness-backend input-file audit missing columns: {missing_input_columns}"
            )
        else:
            if len(input_files) != 9:
                errors.append("main harness-backend input-file audit must contain three artifacts per run")
            for role in expected_roles:
                role_rows = input_files.loc[input_files["run_role"].astype(str).eq(role)]
                if set(role_rows["artifact"].astype(str)) != {
                    "stage7_action_table",
                    "stage8_multi_oracle_scores",
                    "archive_manifest",
                }:
                    errors.append(
                        f"main harness-backend input-file artifact set mismatch for role={role}"
                    )
            present_flags = _parse_bool_series(input_files["artifact_present"])
            if present_flags.isna().any():
                errors.append("main harness-backend input-file audit has invalid artifact_present values")
            else:
                required_rows = input_files["artifact"].astype(str).isin(
                    ["stage7_action_table", "stage8_multi_oracle_scores"]
                )
                if not present_flags.loc[required_rows].all():
                    errors.append("main harness-backend required Stage7/Stage8 inputs may not be absent")
                archive_rows = input_files.loc[input_files["artifact"].astype(str).eq("archive_manifest")].copy()
                archive_row_flags = _parse_bool_series(archive_rows["artifact_present"])
                if int(archive_row_flags.sum()) != archive_present_count or int((~archive_row_flags).sum()) != archive_missing_count:
                    errors.append("main harness-backend input-file archive counts disagree with manifest")
                missing_rows = archive_rows.loc[~archive_row_flags]
                if not missing_rows.empty:
                    if not missing_rows["run_role_source"].astype(str).eq(
                        "paper_reproduction.inspect_archived_run.legacy_role_bridge"
                    ).all():
                        errors.append("main harness-backend absent archive rows use a forbidden role source")
                    if not missing_rows["absence_policy"].astype(str).eq(
                        "ABSENT_LEGACY_ALLOWED"
                    ).all():
                        errors.append("main harness-backend absent archive rows lack the legacy absence policy")
                    if missing_rows["sha256"].astype(str).str.strip().ne("").any():
                        errors.append("main harness-backend absent archive rows must not fabricate hashes")
                present_rows = input_files.loc[present_flags]
                if not present_rows["sha256"].map(_is_sha256).all():
                    errors.append("main harness-backend present input files lack valid SHA-256 values")
    return {"status": manifest.get("status"), "schema_version": manifest.get("schema_version")}

def verify(project_root: Path, analysis_dir: Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = load_profile(root)
    expected_root = analysis_output_dir(root, profile).resolve()
    actual_root = Path(analysis_dir).resolve() if analysis_dir else expected_root
    errors: list[str] = []
    if actual_root != expected_root:
        errors.append(f"analysis_dir must equal canonical profile path: actual={actual_root}, expected={expected_root}")
    layout = build_layout(actual_root)

    analysis_manifest_path = layout.manifest / "paper_repro_analysis_manifest.json"
    analysis_manifest = _read_json(analysis_manifest_path) if analysis_manifest_path.is_file() else {}
    selected_runs = analysis_manifest.get("selected_runs", {})
    selected_frontier = selected_runs.get("n5_budget_frontier", [])
    selected_matched = selected_runs.get("n5_matched_budget_frontier", [])
    if not isinstance(selected_frontier, list):
        errors.append("analysis manifest selected_runs.n5_budget_frontier must be a list")
        selected_frontier = []
    if not isinstance(selected_matched, list):
        errors.append("analysis manifest selected_runs.n5_matched_budget_frontier must be a list")
        selected_matched = []
    if len(selected_frontier) not in {0, 4}:
        errors.append(f"analysis manifest has partial supplementary N5F selection: count={len(selected_frontier)}")
    if len(selected_matched) != 4:
        errors.append(f"analysis manifest must select exactly four canonical N5M arms: count={len(selected_matched)}")

    frontier_source_files = (
        sorted(path for path in layout.n5_budget_frontier_holm.glob("*") if path.is_file())
        if layout.n5_budget_frontier_holm.is_dir()
        else []
    )
    frontier_expected = len(selected_frontier) == 4 or bool(frontier_source_files)

    required_files = [
        layout.manifest / "paper_repro_analysis_manifest.json",
        layout.manifest / "archived_llm_run_catalog.json",
        layout.manifest / "archived_llm_run_catalog.csv",
        layout.b2_gap / "b2_gap_decomposition_report.json",
        layout.test3 / "test3_counterfactual_fidelity_report.json",
        layout.test3 / "test3_property_summary.csv",
        layout.structural_slice / "b2_structural_event_slice_summary.csv",
        layout.holm / "holm_H1_score.csv",
        layout.holm / "holm_H3_reference.csv",
        layout.reference_quality_acceptance / "reference_quality_acceptance_manifest.json",
        layout.reference_quality_acceptance / "reference_quality_acceptance_summary.csv",
        layout.reference_quality_acceptance / "reference_quality_acceptance_quartiles.csv",
        layout.reference_quality_acceptance / "reference_quality_acceptance_primary_c6.csv",
        layout.n5_holm / "n5_7_10c_holm_manifest.json",
        layout.n5_holm / "n5_7_10c_table_patch.csv",
        layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_holm_manifest.json",
        layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_table_patch.csv",
        layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_control_alignment_audit.csv",
        layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_budget_contract_audit.csv",
        layout.n5m_posthoc / "n5m_posthoc_manifest.json",
        layout.n5m_posthoc / "n5m_vs_c3_paired_holm.csv",
        layout.n5m_posthoc / "n5m_adoption_score_relationship.csv",
        layout.n5m_posthoc / "n5m_adoption_quartiles.csv",
        layout.n5m_adaptive_selection / "n5m_adaptive_selection_manifest.json",
        layout.n5m_adaptive_selection / "n5m_adaptive_selection_input_files.csv",
        layout.n5m_adaptive_selection / "n5m_selection_feature_contract.json",
        layout.n5m_adaptive_selection / "n5m_selection_fold_assignments.csv",
        layout.n5m_adaptive_selection / "n5m_adaptive_budget_oof_predictions.parquet",
        layout.n5m_adaptive_selection / "n5m_postc4_gate_oof_predictions.parquet",
        layout.n5m_adaptive_selection / "n5m_adaptive_budget_summary.csv",
        layout.n5m_adaptive_selection / "n5m_postc4_gate_summary.csv",
        layout.n5m_adaptive_selection / "n5m_selection_fold_metrics.csv",
        layout.n5m_adaptive_selection / "n5m_selection_coefficients.csv",
        layout.main_harness_backend_decomposition / "main_harness_backend_decomposition_manifest.json",
        layout.main_harness_backend_decomposition / "main_harness_backend_decomposition.csv",
        layout.main_harness_backend_decomposition / "main_harness_backend_cell_means.csv",
        layout.main_harness_backend_decomposition / "main_harness_backend_swing_summary.csv",
        layout.main_harness_backend_decomposition / "main_harness_backend_alignment_audit.csv",
        layout.frontier / "frontier_grid_raw.csv",
        layout.frontier / "frontier_grid_status.csv",
        layout.winrate / "win_rates_vs_C3.csv",
        layout.winrate / "residual_heterogeneity_exploratory.csv",
        layout.icc_probe / "icc_probe_analysis_manifest.json",
        layout.icc_probe / "icc_probe_channel_summary.csv",
        layout.paper_assets / "paper_assets_manifest.json",
        layout.tables / "freeform_contrast_ladder_alpha.csv",
        layout.tables / "reference_quality_acceptance_primary_c6.csv",
        layout.tables / "n1_frontier_alpha_mean_by_budget.csv",
        layout.tables / "icc_probe_channel_summary.csv",
        layout.tables / "n5m_matched_budget_frontier_alpha.csv",
        layout.tables / "n5m_win_tie_loss_by_budget_oracle.csv",
        layout.tables / "n5m_reference_quality_by_budget.csv",
        layout.tables / "n5m_cross_oracle_local_q_gain.csv",
        layout.tables / "n5m_reference_axis_outcome_summary.csv",
        layout.tables / "n5m_oracle_consensus_by_budget.csv",
        layout.tables / "n5m_dimension_overlap_summary.csv",
        layout.tables / "n5m_feasibility_by_budget_policy.csv",
        layout.tables / "n5m_feasibility_unique_firms.csv",
        layout.tables / "n5m_auxiliary_within_firm_variance.csv",
        layout.tables / "n5m_vs_c3_paired_holm.csv",
        layout.tables / "n5m_adoption_score_relationship.csv",
        layout.tables / "n5m_adoption_quartiles.csv",
        layout.tables / "n5m_adaptive_budget_summary.csv",
        layout.tables / "n5m_postc4_gate_summary.csv",
        layout.tables / "main_harness_backend_decomposition.csv",
        layout.tables / "main_harness_backend_cell_means.csv",
        layout.tables / "main_harness_backend_swing_summary.csv",
        layout.tables / "main_harness_backend_alignment_audit.csv",
        layout.figures / "freeform_contrast_ladder_alpha.png",
        layout.figures / "n1_frontier_alpha_curve.png",
        layout.figures / "n3_winrate_vs_c3.png",
        layout.figures / "n6_log_assets_heterogeneity.png",
        layout.figures / "signflip_mean_null_alpha.png",
        layout.figures / "icc_probe_channel_rates.png",
        layout.figures / "n5m_matched_budget_frontier_alpha.png",
        layout.figures / "n5m_win_tie_loss_alpha.png",
    ]
    if frontier_expected:
        required_files.extend([
            layout.n5_budget_frontier_holm / "n5_budget_frontier_holm_manifest.json",
            layout.n5_budget_frontier_holm / "n5_budget_frontier_table_patch.csv",
            layout.tables / "n5f_legacy_budget_frontier_alpha.csv",
        ])
    else:
        stale_frontier_asset = layout.tables / "n5f_legacy_budget_frontier_alpha.csv"
        if stale_frontier_asset.exists():
            errors.append(
                "stale supplementary N5F paper table exists without a frontier source manifest: "
                f"{stale_frontier_asset}"
            )

    missing = [str(path) for path in required_files if not path.is_file()]
    errors.extend(f"missing required output: {path}" for path in missing)

    for ic in ("IC-a", "IC-b", "IC-c"):
        sign_hits = list((layout.signflip / ic).glob("*/ablation_policy_summary.csv"))
        if len(sign_hits) != 1:
            errors.append(f"sign-flip contract for {ic}: expected 1 summary, found {len(sign_hits)}")
        ablation_hits = list((layout.ablation / ic).glob("*/*/ablation_policy_summary.csv"))
        if len(ablation_hits) != 8:
            errors.append(f"ablation contract for {ic}: expected 8 summaries, found {len(ablation_hits)}")
        shuffle_cells = {
            "none": list((layout.ablation / ic).glob("*/row_shuffle_vector_null_native")),
            "industry": list((layout.ablation / ic).glob("*/row_shuffle_vector_null_native_industry")),
            "rating_band": list((layout.ablation / ic).glob("*/row_shuffle_vector_null_native_rating_band")),
        }
        for expected_within, hits in shuffle_cells.items():
            if len(hits) != 1:
                errors.append(
                    f"shuffle permutation contract for {ic}/{expected_within}: expected 1 cell, found {len(hits)}"
                )
                continue
            _verify_shuffle_cell(hits[0], expected_within=expected_within, errors=errors)

    manifest_statuses: dict[str, Any] = {}
    manifest_paths = {
        "analysis": layout.manifest / "paper_repro_analysis_manifest.json",
        "reference_quality_acceptance": layout.reference_quality_acceptance / "reference_quality_acceptance_manifest.json",
        "n5_holm": layout.n5_holm / "n5_7_10c_holm_manifest.json",
        "icc_probe": layout.icc_probe / "icc_probe_analysis_manifest.json",
        "paper_assets": layout.paper_assets / "paper_assets_manifest.json",
        "n5_matched_budget_frontier_holm": layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_holm_manifest.json",
        "n5m_posthoc": layout.n5m_posthoc / "n5m_posthoc_manifest.json",
        "n5m_adaptive_selection": layout.n5m_adaptive_selection / "n5m_adaptive_selection_manifest.json",
        "main_harness_backend_decomposition": (
            layout.main_harness_backend_decomposition / "main_harness_backend_decomposition_manifest.json"
        ),
    }
    if frontier_expected:
        manifest_paths["n5_budget_frontier_holm"] = (
            layout.n5_budget_frontier_holm / "n5_budget_frontier_holm_manifest.json"
        )
    for name, path in manifest_paths.items():
        if path.exists():
            data = _read_json(path)
            manifest_statuses[name] = data.get("status")
            allowed = {"PASS"}
            if data.get("status") not in allowed:
                errors.append(f"manifest status not acceptable: {name}={data.get('status')!r}")
            if name == "reference_quality_acceptance":
                if data.get("schema_version") != "reference_quality_acceptance_v4":
                    errors.append(
                        "reference-quality manifest schema must be reference_quality_acceptance_v4; "
                        f"found {data.get('schema_version')!r}"
                    )
                lookup = data.get("stage6_reference_lookup", {})
                if lookup.get("contract") != "stage6_fixed_candidate_id_with_c0_noop_alias_v1":
                    errors.append(
                        "reference-quality Stage6 lookup contract mismatch: "
                        f"{lookup.get('contract')!r}"
                    )
                if int(lookup.get("c0_a0_alias_row_count", 0) or 0) != int(
                    lookup.get("stage6_unique_row_count", -1) or -1
                ):
                    errors.append(
                        "reference-quality C0_noop/A0_noop alias coverage does not equal "
                        "the Stage6 evaluation row count"
                    )
                cohort = data.get("cohort_contract", {})
                if cohort.get("contract") != "stage7_pair_eligible_cohort_complete_case_geometry_v1":
                    errors.append(
                        "reference-quality cohort contract mismatch: "
                        f"{cohort.get('contract')!r}"
                    )
                cells = cohort.get("cells", [])
                try:
                    target_rows = int(cohort.get("target_row_count_per_run", -1))
                    observed_runs = int(cohort.get("observed_run_count", -1))
                    observed_cells = int(cohort.get("observed_cell_count", -1))
                    pair_rows = int(cohort.get("stage7_pair_eligible_row_count", -1))
                    stage9_rows = int(cohort.get("stage9_metric_row_count", -1))
                    defined_rows = int(cohort.get("metrics_defined_row_count", -1))
                    undefined_rows = int(cohort.get("metrics_undefined_row_count", -1))
                    manifest_row_count = int(data.get("row_count", -1))
                except Exception:
                    target_rows = observed_runs = observed_cells = pair_rows = stage9_rows = -1
                    defined_rows = undefined_rows = manifest_row_count = -1
                if target_rows != 575:
                    errors.append(
                        "reference-quality frozen evaluator universe must contain 575 firms: "
                        f"{target_rows}"
                    )
                if observed_runs != 3 or observed_cells != 18:
                    errors.append(
                        "reference-quality paired cohort must contain 3 runs and 18 cells "
                        f"(3 conditions x 2 modes): runs={observed_runs}, cells={observed_cells}"
                    )
                if not isinstance(cells, list) or len(cells) != observed_cells:
                    errors.append(
                        "reference-quality paired-cohort cell audit is missing or has the wrong size: "
                        f"cells={len(cells) if isinstance(cells, list) else 'INVALID'}, expected={observed_cells}"
                    )
                else:
                    for cell in cells:
                        try:
                            target = int(cell.get("target_firm_count", -1))
                            base_n = int(cell.get("base_available_count", -1))
                            revision_n = int(cell.get("revision_available_count", -1))
                            pair_n = int(cell.get("pair_eligible_count", -1))
                            metric_n = int(cell.get("stage9_metric_row_count", -1))
                            cell_defined = int(cell.get("n_metrics_defined", -1))
                            cell_undefined = int(cell.get("n_metrics_undefined", -1))
                            coverage = float(cell.get("pair_coverage_rate", float("nan")))
                        except Exception:
                            errors.append(f"reference-quality malformed paired-cohort cell: {cell}")
                            continue
                        if target != target_rows or min(base_n, revision_n, pair_n, metric_n) < 0:
                            errors.append(f"reference-quality invalid paired-cohort counts: {cell}")
                        if pair_n > min(base_n, revision_n) or metric_n != pair_n:
                            errors.append(f"reference-quality Stage7/Stage9 pair count mismatch: {cell}")
                        if cell_defined + cell_undefined != metric_n:
                            errors.append(f"reference-quality geometry counts do not reconcile in cell: {cell}")
                        if not np.isfinite(coverage) or abs(coverage - pair_n / target_rows) > 1e-12:
                            errors.append(f"reference-quality pair coverage rate is inconsistent: {cell}")
                        if str(cell.get("revision_condition")) == "C6" and pair_n != 575:
                            errors.append(f"reference-quality primary C6 paired cohort is not 575 firms: {cell}")
                if pair_rows != stage9_rows or stage9_rows != manifest_row_count:
                    errors.append(
                        "reference-quality Stage7 pair, Stage9 metric, and manifest row counts do not reconcile: "
                        f"pair={pair_rows}, stage9={stage9_rows}, row_count={manifest_row_count}"
                    )
                if defined_rows < 0 or undefined_rows < 0 or defined_rows + undefined_rows != manifest_row_count:
                    errors.append(
                        "reference-quality defined/undefined geometry counts do not reconcile "
                        f"with row_count: defined={defined_rows}, undefined={undefined_rows}, "
                        f"row_count={manifest_row_count}"
                    )

    paper_assets_manifest_path = layout.paper_assets / "paper_assets_manifest.json"
    if paper_assets_manifest_path.is_file():
        paper_assets_manifest = _read_json(paper_assets_manifest_path)
        frontier_asset_state = (
            paper_assets_manifest.get("optional_inputs", {})
            .get("n5_generation_budget_frontier", {})
            .get("status")
        )
        expected_asset_state = "AVAILABLE_PASS" if frontier_expected else "NOT_AVAILABLE_OPTIONAL"
        if frontier_asset_state != expected_asset_state:
            errors.append(
                "paper-assets optional frontier state mismatch: "
                f"actual={frontier_asset_state!r}, expected={expected_asset_state!r}"
            )
        if paper_assets_manifest.get("schema_version") != "paper_repro_assets_v5":
            errors.append("paper-assets manifest must use paper_repro_assets_v5")
        matched_state = paper_assets_manifest.get("required_inputs", {}).get("n5_matched_budget_frontier", {})
        if matched_state.get("status") != "AVAILABLE_PASS" or matched_state.get("paper_use") != "primary":
            errors.append("paper-assets manifest does not require the canonical N5M matched frontier")
        notes = " ".join(map(str, paper_assets_manifest.get("notes", [])))
        if "excluded from the main crossed harness-vs-backend variance decomposition" not in notes:
            errors.append("paper-assets manifest lacks N5/N5F/N5M main-decomposition exclusion note")

    reference_primary = layout.reference_quality_acceptance / "reference_quality_acceptance_primary_c6.csv"
    if reference_primary.exists():
        ref = pd.read_csv(reference_primary)
        required_columns = {
            "run_label", "information_condition", "mode", "oracle_backend", "n_rows",
            "n_metrics_defined", "n_metrics_undefined", "metrics_defined_fraction",
            "n_reference_advantage_defined",
            "reference_better_fraction", "n_spearman_adoption",
            "rho_reference_advantage_vs_adoption",
            "p_reference_advantage_vs_adoption_holm",
            "n_spearman_revision_gain",
            "rho_reference_advantage_vs_revision_gain",
            "p_reference_advantage_vs_revision_gain_holm",
        }
        missing_columns = sorted(required_columns - set(ref.columns))
        if missing_columns:
            errors.append(f"reference-quality primary table missing columns: {missing_columns}")
        if len(ref) != 18:
            errors.append(f"reference-quality primary C6 rows={len(ref)}, expected=18 (3 IC x 2 modes x 3 oracles)")
        if "n_rows" in ref.columns and not pd.to_numeric(ref["n_rows"], errors="coerce").eq(575).all():
            errors.append("reference-quality primary table is not based on 575 firms per cell")
        if {"n_rows", "n_metrics_defined", "n_metrics_undefined"}.issubset(ref.columns):
            n_rows = pd.to_numeric(ref["n_rows"], errors="coerce")
            n_defined = pd.to_numeric(ref["n_metrics_defined"], errors="coerce")
            n_undefined = pd.to_numeric(ref["n_metrics_undefined"], errors="coerce")
            if n_rows.isna().any() or n_defined.isna().any() or n_undefined.isna().any():
                errors.append("reference-quality primary cohort counts contain non-numeric values")
            elif not (n_defined.ge(0) & n_undefined.ge(0) & (n_defined + n_undefined).eq(n_rows)).all():
                errors.append(
                    "reference-quality primary defined/undefined geometry counts do not sum to n_rows"
                )
        if {"n_rows", "n_metrics_defined", "metrics_defined_fraction"}.issubset(ref.columns):
            n_rows = pd.to_numeric(ref["n_rows"], errors="coerce")
            n_defined = pd.to_numeric(ref["n_metrics_defined"], errors="coerce")
            fraction = pd.to_numeric(ref["metrics_defined_fraction"], errors="coerce")
            expected_fraction = n_defined / n_rows
            if fraction.isna().any() or not (fraction - expected_fraction).abs().le(1e-12).all():
                errors.append(
                    "reference-quality primary metrics_defined_fraction is inconsistent with counts"
                )
        if {"n_spearman_adoption", "n_metrics_defined"}.issubset(ref.columns):
            n_spearman = pd.to_numeric(ref["n_spearman_adoption"], errors="coerce")
            n_defined = pd.to_numeric(ref["n_metrics_defined"], errors="coerce")
            if n_spearman.isna().any() or not n_spearman.le(n_defined).all():
                errors.append(
                    "reference-quality adoption correlation uses more rows than have defined geometry"
                )

    matched_frontier_status = _verify_frontier_contract(
        layout.n5_matched_budget_frontier_holm, matched=True, required=True, errors=errors
    )
    supplementary_frontier_status = _verify_frontier_contract(
        layout.n5_budget_frontier_holm, matched=False, required=frontier_expected, errors=errors
    ) if frontier_expected else {"status": "NOT_AVAILABLE_OPTIONAL"}
    n5m_posthoc_status = _verify_n5m_posthoc(layout, errors)
    n5m_adaptive_selection_status = verify_n5m_adaptive_selection(root, actual_root)
    if n5m_adaptive_selection_status.get("status") != "PASS":
        errors.extend(
            f"Section 9.8 adaptive-selection contract: {message}"
            for message in n5m_adaptive_selection_status.get("errors", [])
        )
    main_harness_backend_status = _verify_main_harness_backend_decomposition(layout, errors)

    if (layout.frontier / "frontier_grid_status.csv").exists():
        status = pd.read_csv(layout.frontier / "frontier_grid_status.csv")
        expected_cells = 3 * len(profile["analysis"]["frontier_grid"]) * len(profile["analysis"]["frontier_variants"])
        if len(status) != expected_cells:
            errors.append(f"frontier cells={len(status)}, expected={expected_cells}")
        if "status" not in status.columns or not status["status"].astype(str).eq("PASS").all():
            errors.append("frontier contains non-PASS cells")

    if (layout.icc_probe / "icc_probe_channel_summary.csv").exists():
        channel = pd.read_csv(layout.icc_probe / "icc_probe_channel_summary.csv")
        if not set(channel.get("channel", pd.Series(dtype=str))) >= {
            "firm_recognition", "numeric_debt_ratio_recall", "numeric_contamination_flag", "parse_failure"
        }:
            errors.append("IC-c probe channel summary is incomplete")
        if not channel.empty and not channel["n"].astype(int).eq(575).all():
            errors.append("IC-c probe channel summary is not based on 575 firms")
        recognized = channel[channel["channel"].astype(str).eq("firm_recognition")]
        if len(recognized) != 1:
            errors.append("IC-c probe must contain exactly one firm_recognition row")
        else:
            count = int(pd.to_numeric(recognized["count"], errors="coerce").iloc[0])
            rate = float(pd.to_numeric(recognized["rate"], errors="coerce").iloc[0])
            if count != 444 or not np.isclose(rate, 444 / 575, atol=1e-12, rtol=0):
                errors.append(f"IC-c probe canonical recognition mismatch: count={count}, rate={rate}")

    catalog_path = layout.manifest / "archived_llm_run_catalog.csv"
    if catalog_path.is_file():
        catalog = pd.read_csv(catalog_path)
        if "paper_use" not in catalog.columns:
            errors.append("archived LLM run catalog missing paper_use")
        else:
            matched_rows = catalog[catalog.get("run_role", "").astype(str).eq("paper_n5_matched_budget_frontier_icb")]
            legacy_rows = catalog[catalog.get("run_role", "").astype(str).eq("paper_n5_budget_frontier_icb")]
            if len(matched_rows) != 4 or not matched_rows["paper_use"].astype(str).eq("primary").all():
                errors.append("N5M four arms are not catalogued as primary")
            if not legacy_rows.empty and not legacy_rows["paper_use"].astype(str).eq("supplementary").all():
                errors.append("N5F arms are not catalogued as supplementary")

    output_files: list[dict[str, Any]] = []
    if actual_root.exists():
        for path in sorted(p for p in actual_root.rglob("*") if p.is_file()):
            # Exclude this verifier's own output to avoid recursive hash instability.
            if path.name in {"verify_paper_repro_output_contract.json", "paper_repro_analysis_manifest.json"}:
                continue
            output_files.append({
                "relative_path": path.relative_to(actual_root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": _sha256(path),
            })

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _now(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "analysis_dir": str(actual_root),
        "manifest_statuses": manifest_statuses,
        "optional_generation_frontier": {
            "expected": frontier_expected,
            "selected_run_count": len(selected_frontier),
            "source_file_count": len(frontier_source_files),
            "status": "AVAILABLE_REQUIRED" if frontier_expected else "NOT_AVAILABLE_OPTIONAL",
        },
        "canonical_n5m": {
            "selected_run_count": len(selected_matched),
            "frontier": matched_frontier_status,
            "posthoc": n5m_posthoc_status,
            "adaptive_selection": n5m_adaptive_selection_status,
            "main_harness_backend_decomposition_included": False,
        },
        "main_harness_backend_decomposition": main_harness_backend_status,
        "supplementary_n5f": supplementary_frontier_status,
        "required_file_count": len(required_files),
        "output_file_count": len(output_files),
        "output_files": output_files,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(Path(args.project_root), Path(args.analysis_dir) if args.analysis_dir else None)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
