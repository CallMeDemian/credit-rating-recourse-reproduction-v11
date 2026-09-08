from __future__ import annotations

"""Synthetic verifier for canonical paper-facing tables and figures."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.analysis.paper_output_layout import build_layout, ensure_layout
from credit_recourse.analysis.paper_repro_assets import run_assets


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _ablation_cell(path: Path, alpha: float) -> None:
    _csv(path / "ablation_policy_summary.csv", [{
        "policy": "C6", "mode": "free_form_10d", "mean_delta_R_score_alpha": alpha,
        "positive_fraction_alpha": 0.5,
    }])
    paired = [{
        "policy": "C6", "mode": "free_form_10d", "oracle_backend": "alpha",
        "mean_gap_vs_reference": alpha - 0.630313, "sig_holm": "***",
    }]
    _csv(path / "ablation_paired_vs_reference.csv", paired)
    _csv(path / "ablation_paired_vs_original_target.csv", [{**paired[0], "mean_gap_vs_reference": alpha - 1.0}])


def _populate(root: Path, *, include_frontier: bool) -> None:
    layout = build_layout(root)
    ensure_layout(layout)
    _json(layout.b2_gap / "b2_gap_decomposition_report.json", {"status": "PASS", "ledger": {"B1": 0.72, "B2": 0.55}})
    _csv(layout.test3 / "test3_property_summary.csv", [
        {"property": "P2_rank", "metric": "spearman", "split": "oot", "value": 0.903, "n": 657},
        {"property": "P3_direction", "metric": "agreement", "split": "oot", "value": 0.717, "n": 657},
    ])
    _csv(layout.structural_slice / "b2_structural_event_slice_summary.csv", [
        {"slice": "full_sample", "n_movers": 1628, "n_agree": 1169, "agreement": 0.718},
        {"slice": "no_structural_event", "n_movers": 1111, "n_agree": 787, "agreement": 0.708},
        {"slice": "structural_event", "n_movers": 517, "n_agree": 382, "agreement": 0.739},
    ])
    _csv(layout.holm / "holm_H1_score.csv", [{"contrast": "C5-C4", "sig_holm": "***"}])
    _csv(layout.holm / "holm_H3_reference.csv", [{"contrast": "C6-C6X", "sig_holm": "***"}])
    _csv(layout.reference_quality_acceptance / "reference_quality_acceptance_primary_c6.csv", [
        {
            "run_label": f"paper_{ic}", "information_condition": ic,
            "mode": mode, "oracle_backend": oracle, "n_rows": 575,
            "n_metrics_defined": 560, "n_metrics_undefined": 15,
            "metrics_defined_fraction": 560 / 575,
            "n_reference_advantage_defined": 575,
            "reference_better_fraction": 0.55, "mean_reference_advantage": 0.12,
            "n_spearman_adoption": 560,
            "rho_reference_advantage_vs_adoption": 0.18,
            "rho_adoption_ci_lo": 0.08, "rho_adoption_ci_hi": 0.28,
            "p_reference_advantage_vs_adoption_holm": 0.01,
            "sig_reference_advantage_vs_adoption_holm": "**",
            "n_spearman_revision_gain": 575,
            "rho_reference_advantage_vs_revision_gain": 0.25,
            "rho_revision_gain_ci_lo": 0.15, "rho_revision_gain_ci_hi": 0.35,
            "p_reference_advantage_vs_revision_gain_holm": 0.001,
            "sig_reference_advantage_vs_revision_gain_holm": "***",
            "adoption_group_gap": 0.10, "revision_gain_group_gap": 0.08,
        }
        for ic in ("IC-a", "IC-b", "IC-c")
        for mode in ("candidate_selection", "free_form_10d")
        for oracle in ("alpha", "beta", "gamma")
    ])

    for i, ic in enumerate(("IC-a", "IC-b", "IC-c")):
        run = f"paper_{ic}"
        _ablation_cell(layout.signflip / ic / run, -1.4 + 0.1 * i)
        _ablation_cell(layout.ablation / ic / run / "global_mean_vector_null_native", 1.1 - 0.08 * i)
    _csv(layout.n5_holm / "n5_7_10c_table_patch.csv", [
        {"information_condition": ic, "mean_C6": 0.89 + i * 0.02, "C6_minus_C3_alpha_gap": 0.26 + i * 0.02, "C6_minus_C3_sig": "***"}
        for i, ic in enumerate(("IC-a", "IC-b", "IC-c"))
    ])
    # The matched N5M frontier is canonical and required in every paper bundle.
    _csv(layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_table_patch.csv", [
        {
            "budget_label": label,
            "l1_budget": budget,
            "oracle_backend": "alpha",
            "mean_C4": mean_c4,
            "mean_C6": mean_c6,
            "mean_C6_minus_C4": mean_c6 - mean_c4,
            "C6_minus_C4_p_holm": 0.001,
            "C6_minus_C4_sig": "***",
            "C6_minus_unbounded_gap": gap,
            "C6_minus_unbounded_p_holm": 0.01 if gap is not None else None,
            "C6_minus_unbounded_sig": "**" if gap is not None else "reference",
            "interaction_or_package_DID_gap": gap,
            "interaction_or_package_DID_p_holm": 0.01 if gap is not None else None,
            "interaction_or_package_DID_sig": "**" if gap is not None else "reference",
        }
        for label, budget, mean_c4, mean_c6, gap in (
            ("0p75", 0.75, 0.94, 0.84, -0.31),
            ("1p27", 1.27, 0.92, 0.94, -0.16),
            ("2p00", 2.00, 1.11, 1.14, -0.17),
            ("unbounded", None, 0.89, 1.10, None),
        )
    ])
    _csv(layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_control_alignment_audit.csv", [
        {"oracle_backend": oracle, "arm": arm, "n_rows": 575, "status": "PASS"}
        for oracle in ("alpha", "beta", "gamma")
        for arm in ("0p75", "1p27", "2p00", "unbounded")
    ])
    _csv(layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_budget_contract_audit.csv", [
        {"arm": arm, "n_rows": 1150, "budgeted_conditions": "C4;C6", "status": "PASS"}
        for arm in ("0p75", "1p27", "2p00", "unbounded")
    ])
    _json(layout.n5_matched_budget_frontier_holm / "n5_budget_frontier_holm_manifest.json", {
        "schema_version": "n5_generation_budget_frontier_holm_v2",
        "status": "PASS",
        "design": "matched_c4_c6",
        "run_role": "paper_n5_matched_budget_frontier_icb",
        "information_condition": "IC-b",
    })

    # N5M post-hoc is a separate supplementary budget-conditioned analysis.
    # It is explicitly excluded from the common crossed harness-vs-backend panel.
    _csv(layout.n5m_posthoc / "n5m_win_tie_loss_by_budget_oracle.csv", [
        {
            "budget_label": arm, "l1_budget": budget, "oracle_backend": "alpha",
            "n_firms": 575, "n_c6_win": wins, "n_tie": ties,
            "n_c6_loss": 575 - wins - ties, "c6_win_rate_all": wins / 575,
            "tie_rate": ties / 575, "c6_loss_rate_all": (575 - wins - ties) / 575,
            "n_non_tie": 575 - ties, "c6_win_rate_non_tie": wins / (575 - ties),
            "wilson_95_lo_non_tie": 0.20, "wilson_95_hi_non_tie": 0.90,
            "mean_C6_minus_C4": gap, "median_C6_minus_C4": 0.0,
        }
        for arm, budget, wins, ties, gap in (
            ("0p75", 0.75, 37, 440, -0.100),
            ("1p27", 1.27, 75, 440, 0.020),
            ("2p00", 2.00, 91, 415, 0.030),
            ("unbounded", None, 140, 406, 0.210),
        )
    ])
    _csv(layout.n5m_posthoc / "n5m_reference_quality_by_budget.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "oracle_backend": "alpha",
        "n_firms": 575, "rho_reference_advantage_vs_revision_gain": 0.40,
    }])
    _csv(layout.n5m_posthoc / "n5m_reference_quality_quartiles.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "oracle_backend": "alpha",
        "reference_advantage_quartile": "Q1", "n_firms": 144,
    }])
    (layout.n5m_posthoc / "n5m_firm_frame.parquet").write_text("synthetic parquet placeholder for existence-only contract\n", encoding="utf-8")
    (layout.n5m_posthoc / "n5m_reference_axis_anatomy.parquet").write_text("synthetic parquet placeholder for existence-only contract\n", encoding="utf-8")
    _csv(layout.n5m_posthoc / "n5m_cross_oracle_local_q_gain.csv", [
        {
            "budget_label": arm, "l1_budget": budget,
            "local_q_oracle": qo, "revision_gain_oracle": go,
            "n_complete": 575, "spearman_rho": 0.15, "p_raw": 0.001,
            "p_holm_24": 0.024, "sig_holm_24": "*",
            "shared_term_status": "OFF_DIAGONAL_ORACLE_SENSITIVITY",
            "evidence_tier": "EVALUATOR_ONLY_POSTHOC",
        }
        for arm, budget in (("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", None))
        for qo in ("alpha", "beta", "gamma")
        for go in ("alpha", "beta", "gamma") if qo != go
    ])
    _csv(layout.n5m_posthoc / "n5m_reference_axis_outcome_summary.csv", [
        {
            "budget_label": arm, "l1_budget": budget, "oracle_backend": oracle,
            "metric": metric, "n_complete": 575,
            "spearman_rho_vs_revision_gain": -0.15 if metric.startswith("removed") else 0.20,
            "p_raw": 0.001, "p_holm_within_oracle_metric": 0.004, "sig_holm": "**",
            "mean_all": 0.1, "n_loss": 98, "mean_loss": 0.2,
            "n_win": 37, "mean_win": 0.05, "evidence_tier": "EVALUATOR_ONLY_POSTHOC",
        }
        for arm, budget in (("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", None))
        for oracle in ("alpha", "beta", "gamma")
        for metric in ("removed_mass_off_reference_axes", "added_mass_on_reference_axes")
    ])
    _csv(layout.n5m_posthoc / "n5m_oracle_consensus_by_budget.csv", [
        {
            "budget_label": arm, "l1_budget": budget, "n_firms": 575,
            "all_three_loss_count": 40, "all_three_loss_fraction": 40/575,
            "majority_loss_count": 150, "majority_loss_fraction": 150/575,
            "all_three_win_count": 11, "all_three_win_fraction": 11/575,
            "majority_win_count": 70, "majority_win_fraction": 70/575,
            "all_three_tie_count": 200, "all_three_tie_fraction": 200/575,
            "mixed_oracle_direction_count": 50, "mixed_oracle_direction_fraction": 50/575,
            "evidence_tier": "EVALUATOR_ONLY_POSTHOC",
        }
        for arm, budget in (("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", None))
    ])
    _csv(layout.n5m_posthoc / "n5m_action_axis_c4_c6_differences.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "action_axis": "action__ppe_pct",
        "n_firms": 575, "mean_C6_minus_C4": 0.01,
    }])
    _csv(layout.n5m_posthoc / "n5m_dimension_overlap_summary.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "n_firms": 575,
        "mean_revision_l1_over_budget": 0.83,
    }])
    _csv(layout.n5m_posthoc / "n5m_reallocation_by_outcome_group.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "oracle_backend": "alpha",
        "outcome_group": "C6_loss", "n_firms": 98,
    }])
    _csv(layout.n5m_posthoc / "n5m_revision_distance_score_relationship.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "oracle_backend": "alpha",
        "n_complete": 575, "spearman_revision_l1_vs_score_gain": -0.20,
    }])
    _csv(layout.n5m_posthoc / "n5m_feasibility_by_budget_policy.csv", [{
        "budget_label": "0p75", "l1_budget": 0.75, "policy": "C4", "n_rows": 575,
        "sustainability_critical_count": 5, "accounting_check_failure_count": 0,
        "negative_balance_count": 0, "preflight_failure_count": 0,
    }])
    _csv(layout.n5m_posthoc / "n5m_feasibility_unique_firms.csv", [{
        "row_id": 39, "critical_row_count": 8, "affected_budget_count": 4,
        "affected_policy_count": 2,
    }])
    _csv(layout.n5m_posthoc / "n5m_auxiliary_within_firm_variance.csv", [
        {
            "oracle_backend": oracle,
            "n_firms": 575,
            "n_cells_per_firm": 8,
            "n_observations": 4600,
            "budget_main_effect_r2_within_firm": budget_r2,
            "reference_main_effect_r2_within_firm": ref_r2,
            "budget_policy_cell_r2_within_firm": budget_r2 + ref_r2,
            "supplementary_only": True,
            "excluded_from_main_harness_backend_decomposition": True,
        }
        for oracle, budget_r2, ref_r2 in (
            ("alpha", 0.036, 0.002), ("beta", 0.158, 0.001), ("gamma", 0.061, 0.003)
        )
    ])
    _csv(layout.n5m_posthoc / "n5m_score_budget_auditability_operating_points.csv", [{
        "budget_label": "2p00", "l1_budget": 2.0, "policy": "C6", "n_firms": 575,
        "mean_final_l1": 1.0, "mean_projection_distance": 0.14,
        "budget_violation_rate": 0.0, "mean_delta_R_score_alpha": 1.14,
        "point_estimate_nondominated_within_policy_alpha": True,
    }])
    _csv(layout.n5m_posthoc / "n5m_vs_c3_paired_holm.csv", [
        {
            "budget_label": arm, "l1_budget": budget, "policy": policy,
            "oracle_backend": oracle, "reference_policy": "C3_candidate_iql",
            "n_pairs": 575, "n_nonzero_pairs": 500,
            "mean_policy_score": 0.9, "mean_C3_score": 0.63,
            "mean_gap": 0.2, "median_gap": 0.0,
            "positive_fraction": 0.8, "zero_fraction": 0.1, "negative_fraction": 0.1,
            "wilcoxon_p_raw": 0.001, "wilcoxon_p_holm": 0.004, "sig_holm": "**",
            "contrast": f"{policy}_minus_C3__{arm}",
            "holm_family": f"N5M_{policy}_minus_C3_{oracle}",
        }
        for arm, budget in (("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", None))
        for policy in ("C4", "C6")
        for oracle in ("alpha", "beta", "gamma")
    ])
    _csv(layout.n5m_posthoc / "n5m_adoption_score_relationship.csv", [
        {
            "budget_label": arm, "l1_budget": budget, "oracle_backend": oracle,
            "n_firms": 575, "n_metrics_defined": 560, "n_complete": 560,
            "spearman_adoption_vs_revision_gain": (-0.4501731165305016 if arm == "0p75" and oracle == "alpha" else -0.1),
            "p_raw": 0.001, "p_holm": 0.004, "sig_holm": "**",
        }
        for arm, budget in (("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", None))
        for oracle in ("alpha", "beta", "gamma")
    ])
    _csv(layout.n5m_posthoc / "n5m_adoption_quartiles.csv", [
        {
            "budget_label": arm, "l1_budget": budget, "oracle_backend": oracle,
            "adoption_quartile": q, "n_firms": 140,
            "mean_adoption_ratio": 0.1 * qi,
            "median_adoption_ratio": 0.1 * qi,
            "mean_revision_gain": (-0.3526592985915261 if arm == "0p75" and oracle == "alpha" and q == "Q4" else 0.01),
            "median_revision_gain": 0.0, "score_decline_fraction": 0.2,
            "mean_removed_initial_l1": 0.1, "mean_added_revised_l1": 0.1,
            "mean_zero_sum_reallocation_share": 0.5,
            "mean_revision_l1_normalized": 0.6, "mean_revision_l1_over_budget": 0.8,
            "mean_c4_final_l1_normalized": 0.4, "mean_c6_final_l1_normalized": 0.4,
            "mean_added_dimensions": 1.0, "mean_removed_dimensions": 1.0,
            "mean_sign_flipped_dimensions": 0.0,
        }
        for arm, budget in (("0p75", 0.75), ("1p27", 1.27), ("2p00", 2.0), ("unbounded", None))
        for oracle in ("alpha", "beta", "gamma")
        for qi, q in enumerate(("Q1", "Q2", "Q3", "Q4"), start=1)
    ])

    _json(layout.n5m_posthoc / "n5m_posthoc_manifest.json", {
        "schema_version": "n5m_posthoc_v4",
        "status": "PASS",
        "run_role": "paper_n5_matched_budget_frontier_icb",
        "information_condition": "IC-b",
        "run_count": 4, "firm_count": 575,
        "stage7_action_row_count": 4600, "stage9_revision_row_count": 2300,
        "stage8_feasibility_row_count": 4600,
        "sustainability_critical_row_count": 55,
        "sustainability_critical_unique_firm_count": 12,
        "vs_c3_contrast_contract": {
            "canonical_reference_policy": "C3_candidate_iql",
            "row_count": 24, "expected_row_count": 24,
            "holm_family": "policy_x_oracle_across_four_budgets",
        },
        "adoption_diagnostics_contract": {
            "relationship_row_count": 12, "quartile_row_count": 48,
            "complete_case_policy": "metrics_defined_and_finite_adoption_and_revision_gain",
        },
        "firm_frame_contract": {
            "row_count": 2300,
            "unique_firm_count": 575,
            "key": ["budget_label", "l1_budget", "row_id"],
            "post_c4_gate_feature_columns": [
                "c4_final_l1", "c4_projection_distance", "c4_active_dimensions"
            ],
            "post_c4_gate_feature_status": "AVAILABLE",
        },
        "cross_oracle_shared_term_sensitivity_contract": {"row_count": 24, "expected_row_count": 24, "holm_family_size": 24, "status": "EVALUATOR_ONLY_POSTHOC"},
        "reference_axis_anatomy_contract": {"detail_row_count": 2300, "summary_row_count": 24, "interpretation": "descriptive action-mass decomposition; not axis-level causal attribution"},
        "oracle_consensus_contract": {"row_count": 4, "expected_row_count": 4},
        "main_harness_backend_decomposition_contract": {
            "included": False,
            "excluded_run_families": ["N5", "N5F", "N5M"],
        },
        "auxiliary_variance_contract": {"status": "SUPPLEMENTARY_ONLY"},
    })

    # Section 9.8 paper-facing tables are copied from the canonical selector/gate
    # producer.  The independent Section 9.8 verifier checks the firm-level OOF
    # predictions, group folds, feature contract, hashes, and summary arithmetic.
    _csv(layout.n5m_adaptive_selection / "n5m_adaptive_budget_summary.csv", [
        {
            "analysis": "adaptive_budget", "summary_level": "aggregate",
            "policy": policy, "repeats": 5, "n_firms": 575,
            "selector_mean_score": selector, "selector_repeat_sd": 0.01,
            "frozen_fixed_budget_label": "2p00",
            "frozen_fixed_mean_score": fixed,
            "gap_vs_frozen_fixed": selector - fixed,
            "hindsight_budget_oracle_mean_score": oracle,
            "hindsight_headroom": oracle - fixed,
            "headroom_capture_fraction": (selector - fixed) / (oracle - fixed),
        }
        for policy, selector, fixed, oracle in (
            ("C4", 0.94, 0.95, 1.17),
            ("C6", 0.97, 1.00, 1.20),
        )
    ])
    _csv(layout.n5m_adaptive_selection / "n5m_postc4_gate_summary.csv", [{
        "analysis": "post_c4_gate", "summary_level": "aggregate",
        "repeats": 1, "n_firm_budget_rows": 2300, "n_firms": 575,
        "c4_mean_score": 0.91, "c6_mean_score": 0.96,
        "best_unconditional_policy": "C6", "best_unconditional_mean_score": 0.96,
        "gate_mean_score": 0.99, "gate_repeat_sd": 0.0,
        "gap_vs_best_unconditional": 0.03,
        "hindsight_oracle_mean_score": 1.06, "hindsight_headroom": 0.10,
        "headroom_capture_fraction": 0.30,
        "c6_selection_rate": 0.45,
    }])

    # Canonical common-crossed harness-vs-backend decomposition.
    _csv(layout.main_harness_backend_decomposition / "main_harness_backend_decomposition.csv", [
        {
            "oracle_backend": oracle, "n_firms": 575, "n_backends": 3,
            "n_harness_cells": 12, "n_observations": 20682,
            "firm_fe_r2_total_variance": 0.5,
            "harness_main_effect_r2_within_firm": 0.14,
            "backend_main_effect_r2_within_firm": 0.01,
            "additive_harness_backend_r2_within_firm": 0.16,
            "full_backend_by_harness_cell_r2_within_firm": 0.18,
            "partial_harness_given_backend_r2_within_firm": 0.15,
            "partial_backend_main_given_harness_r2_within_firm": 0.02,
            "partial_backend_plus_interaction_given_harness_r2_within_firm": 0.04,
            "solver_contract": "exact_group_projection_plus_small_additive_lstsq_v1",
            "firm_fe_solver": "exact_group_mean_projection",
            "one_way_solver": "exact_group_mean_projection",
            "additive_solver": "numpy_lstsq_small_design",
        } for oracle in ("alpha", "beta", "gamma")
    ])
    _csv(layout.main_harness_backend_decomposition / "main_harness_backend_cell_means.csv", [
        {"oracle_backend": oracle, "backend_label": backend, "harness_cell": cell,
         "n_firms": 575, "mean_score": 0.5, "median_score": 0.5}
        for oracle in ("alpha", "beta", "gamma")
        for backend in ("GPT-5.4-mini", "GPT-4.1-mini", "Haiku-4.5")
        for cell in ("C4__candidate_selection", "C4__free_form_10d", "C5__candidate_selection",
                     "C5__free_form_10d", "C6__candidate_selection", "C6__free_form_10d",
                     "C6X__candidate_selection", "C6X__free_form_10d", "C7__candidate_selection",
                     "C7__free_form_10d", "C8__candidate_selection", "C8__free_form_10d")
    ])
    _csv(layout.main_harness_backend_decomposition / "main_harness_backend_swing_summary.csv", [
        {"oracle_backend": oracle, "min_harness_swing_backend_fixed": 0.5,
         "max_harness_swing_backend_fixed": 0.8, "mean_harness_swing_backend_fixed": 0.65,
         "mean_backend_swing_harness_fixed": 0.1, "max_backend_swing_harness_fixed": 0.2,
         "n_backend_swing_cells": 12}
        for oracle in ("alpha", "beta", "gamma")
    ])
    roles = (("paper_primary_gpt54", "GPT-5.4-mini", 6900),
             ("paper_supplementary_gpt41", "GPT-4.1-mini", 6900),
             ("paper_supplementary_haiku45", "Haiku-4.5", 6882))
    _csv(layout.main_harness_backend_decomposition / "main_harness_backend_alignment_audit.csv", [
        {"run_label": role, "run_role": role, "backend_label": backend, "backend_id": backend,
         "run_role_source": "paper_reproduction.inspect_archived_run.legacy_role_bridge",
         "run_role_explicit": False,
         "archive_manifest_present": role != "paper_supplementary_gpt41",
         "information_condition": "IC-b", "firm_count": 575,
         "expected_harness_cell_count": 12, "observed_harness_cell_count": 12,
         "expected_observation_count": 6900, "observed_observation_count": count,
         "missing_observation_count": 6900-count, "observation_coverage": count/6900,
         "stage7_stage8_key_alignment": "PASS"}
        for role, backend, count in roles
    ])
    _csv(layout.main_harness_backend_decomposition / "main_harness_backend_missing_cell_audit.csv", [
        {"run_label": role, "run_role": role, "backend_label": backend,
         "run_role_source": "paper_reproduction.inspect_archived_run.legacy_role_bridge",
         "run_role_explicit": False,
         "archive_manifest_present": role != "paper_supplementary_gpt41",
         "harness_cell": cell,
         "expected_rows": 575, "observed_rows": 575, "missing_rows": 0, "coverage": 1.0}
        for role, backend, _ in roles for cell in ("C4__candidate_selection", "C4__free_form_10d",
          "C5__candidate_selection", "C5__free_form_10d", "C6__candidate_selection",
          "C6__free_form_10d", "C6X__candidate_selection", "C6X__free_form_10d",
          "C7__candidate_selection", "C7__free_form_10d", "C8__candidate_selection", "C8__free_form_10d")
    ])
    input_file_rows = []
    for role, _, _ in roles:
        for artifact in ("stage7_action_table", "stage8_multi_oracle_scores", "archive_manifest"):
            archive_missing = artifact == "archive_manifest" and role == "paper_supplementary_gpt41"
            input_file_rows.append({
                "run_label": role,
                "run_role": role,
                "run_role_source": "paper_reproduction.inspect_archived_run.legacy_role_bridge",
                "artifact": artifact,
                "path": f"/{role}/{artifact}",
                "artifact_present": not archive_missing,
                "absence_policy": "ABSENT_LEGACY_ALLOWED" if archive_missing else "",
                "sha256": "" if archive_missing else "0" * 64,
                "row_count": "" if artifact == "archive_manifest" else 6900,
            })
    _csv(
        layout.main_harness_backend_decomposition / "main_harness_backend_input_files.csv",
        input_file_rows,
    )
    panel_fixture = pd.DataFrame({"row_id": [0], "backend_label": ["GPT-5.4-mini"], "harness_cell": ["C4__free_form_10d"]})
    panel_path = layout.main_harness_backend_decomposition / "main_harness_backend_panel.parquet"
    try:
        panel_fixture.to_parquet(panel_path, index=False)
    except ImportError:
        # Existence-only fixture for lean contract environments without a
        # parquet engine; production outputs remain real parquet artifacts.
        panel_fixture.to_csv(panel_path, index=False)
    _json(layout.main_harness_backend_decomposition / "main_harness_backend_decomposition_manifest.json", {
        "schema_version": "main_harness_backend_decomposition_v4", "status": "PASS",
        "panel_contract": "common_crossed_icb_harness_backend_panel_v3",
        "solver_contract": "exact_group_projection_plus_small_additive_lstsq_v1",
        "solver_diagnostics": {
            "dense_firm_dummy_lstsq_used": False,
            "firm_fe_solver": "exact_group_mean_projection",
            "one_way_solver": "exact_group_mean_projection",
            "additive_solver": "numpy_lstsq_small_design",
            "maximum_dense_additive_columns": 64,
        },
        "information_condition": "IC-b", "firm_count": 575, "backend_count": 3,
        "harness_cell_count": 12, "observation_count": 20682,
        "expected_balanced_observation_count": 20700, "missing_observation_count": 18,
        "run_role_backend_labels": {role: backend for role, backend, _ in roles},
        "run_role_resolution_contract": {
            "allowed_sources": [
                "archive_manifest.run_role",
                "stage7.metadata.run_role",
                "paper_reproduction.inspect_archived_run.legacy_role_bridge",
            ],
            "resolver": "credit_recourse.contracts.paper_reproduction.inspect_archived_run",
            "fresh_runs_must_write_explicit_run_role": True,
            "legacy_bridge_scope": "frozen canonical main-backend archives selected by the paper profile",
            "archive_manifest_optional_only_for_legacy_bridge": True,
            "explicit_role_count": 0,
            "legacy_bridge_count": 3,
            "runs": [
                {
                    "run_label": role,
                    "run_role": role,
                    "backend_label": backend,
                    "run_role_source": "paper_reproduction.inspect_archived_run.legacy_role_bridge",
                    "run_role_explicit": False,
                    "archive_manifest_present": role != "paper_supplementary_gpt41",
                }
                for role, backend, _ in roles
            ],
        },
        "archive_manifest_presence_contract": {
            "required_for_fresh_or_explicit_role_runs": True,
            "allowed_missing_role_source": "paper_reproduction.inspect_archived_run.legacy_role_bridge",
            "present_count": 2,
            "missing_legacy_count": 1,
            "missing_runs": [
                {
                    "run_label": "paper_supplementary_gpt41",
                    "run_role": "paper_supplementary_gpt41",
                    "run_role_source": "paper_reproduction.inspect_archived_run.legacy_role_bridge",
                }
            ],
        },
        "forbidden_run_families": ["N5", "N5F", "N5M"], "n5_n5f_n5m_included": False,
    })

    # The legacy N5F generation frontier remains supplementary and optional.
    if include_frontier:
        _csv(layout.n5_budget_frontier_holm / "n5_budget_frontier_table_patch.csv", [
            {
                "budget_label": label, "l1_budget": budget, "mean_C4": 0.72,
                "mean_C6": mean_c6, "mean_C6_minus_C4": mean_c6 - 0.72,
                "C6_minus_C4_p_holm": 0.001, "C6_minus_C4_sig": "***",
                "C6_minus_unbounded_gap": gap, "C6_minus_unbounded_p_holm": 0.01,
                "C6_minus_unbounded_sig": "**",
                "budget_DID_gap": gap, "budget_DID_p_holm": 0.01,
                "budget_DID_sig": "**",
            }
            for label, budget, mean_c6, gap in (
                ("0p75", 0.75, 0.81, -0.09), ("1p27", 1.27, 0.86, -0.04),
                ("2p00", 2.00, 0.89, -0.01), ("unbounded", None, 0.90, None),
            )
        ])
        _csv(layout.n5_budget_frontier_holm / "n5_budget_frontier_c4_stability_audit.csv", [
            {"oracle_backend": oracle, "reference_arm": "unbounded", "arm": arm,
             "n_rows": 575, "mean_C4_drift": 0.0, "mean_abs_C4_drift": 0.0,
             "max_abs_C4_drift": 0.0, "wilcoxon_p_raw": 1.0, "status": "PASS_EXACT"}
            for oracle in ("alpha", "beta", "gamma")
            for arm in ("0p75", "1p27", "2p00", "unbounded")
        ])
        _json(layout.n5_budget_frontier_holm / "n5_budget_frontier_holm_manifest.json", {
            "schema_version": "n5_generation_budget_frontier_holm_v1",
            "status": "PASS", "run_role": "paper_n5_budget_frontier_icb",
            "information_condition": "IC-b",
        })
    _csv(layout.winrate / "win_rates_vs_C3.csv", [
        {"information_condition": ic, "policy": "C6", "mode": "free_form_10d", "oracle_backend": "alpha", "n_pairs": 575,
         "n_ties": 260, "n_non_tie_pairs": 315, "win_rate_excluding_ties": 0.82 + i * 0.01,
         "win_rate_wilson_lo": 0.78, "win_rate_wilson_hi": 0.86,
         "positive_fraction_including_ties": 0.45, "zero_fraction": 0.45,
         "mean_gap_vs_c3": 0.26 + i * 0.02, "median_gap_vs_c3": 0.0}
        for i, ic in enumerate(("IC-a", "IC-b", "IC-c"))
    ])
    _csv(layout.winrate / "residual_heterogeneity_exploratory.csv", [
        {"information_condition": ic, "slice": q, "mean_gap": 0.2 + 0.05 * qi}
        for ic in ("IC-a", "IC-b", "IC-c") for qi, q in enumerate(("Q1", "Q2", "Q3", "Q4"))
    ])
    _csv(layout.frontier / "frontier_grid_raw.csv", [
        {"policy": "C6", "mode": "free_form_10d", "frontier_budget": b, "frontier_variant": v, "mean_delta_R_score_alpha": 0.5 + b * 0.4}
        for b in (0.5, 1.27, 2.0) for v in ("l1_rescale", "global_mean_vector_null", "row_shuffle_vector_null")
    ])
    _csv(layout.icc_probe / "icc_probe_channel_summary.csv", [
        {"channel": "firm_recognition", "count": 444, "n": 575, "rate": 444/575, "wilson_95_lo": 0.73615, "wilson_95_hi": 0.80458},
        {"channel": "numeric_debt_ratio_recall", "count": 0, "n": 575, "rate": 0.0, "wilson_95_lo": 0.0, "wilson_95_hi": 0.0066},
        {"channel": "numeric_contamination_flag", "count": 0, "n": 575, "rate": 0.0, "wilson_95_lo": 0.0, "wilson_95_hi": 0.0066},
        {"channel": "parse_failure", "count": 0, "n": 575, "rate": 0.0, "wilson_95_lo": 0.0, "wilson_95_hi": 0.0066},
    ])
    _csv(layout.icc_probe / "icc_probe_familiarity_distribution.csv", [
        {"familiarity": x, "count": c, "n": 575, "rate": c/575}
        for x, c in ((0,89),(1,111),(2,374),(3,1))
    ])


def verify(out_json: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    cases: dict[str, Any] = {}
    for include_frontier in (False, True):
        label = "with_frontier" if include_frontier else "without_frontier"
        with tempfile.TemporaryDirectory(prefix=f"paper_assets_{label}_") as td:
            root = Path(td) / "paper_repro"
            _populate(root, include_frontier=include_frontier)
            manifest = run_assets(root, strict=True)
            layout = build_layout(root)
            required = [
                layout.paper_assets / "paper_assets_manifest.json",
                layout.tables / "freeform_contrast_ladder_alpha.csv",
                layout.tables / "test3_property_summary.csv",
                layout.tables / "reference_quality_acceptance_primary_c6.csv",
                layout.figures / "freeform_contrast_ladder_alpha.png",
                layout.figures / "structural_event_slice.png",
                layout.figures / "icc_probe_channel_rates.png",
            ]
            matched_table = layout.tables / "n5m_matched_budget_frontier_alpha.csv"
            n5m_win_table = layout.tables / "n5m_win_tie_loss_by_budget_oracle.csv"
            required.extend([
                matched_table, n5m_win_table,
                layout.tables / "n5m_cross_oracle_local_q_gain.csv",
                layout.tables / "n5m_reference_axis_outcome_summary.csv",
                layout.tables / "n5m_oracle_consensus_by_budget.csv",
                layout.tables / "n5m_vs_c3_paired_holm.csv",
                layout.tables / "n5m_adoption_score_relationship.csv",
                layout.tables / "n5m_adoption_quartiles.csv",
                layout.tables / "n5m_adaptive_budget_summary.csv",
                layout.tables / "n5m_postc4_gate_summary.csv",
                layout.tables / "main_harness_backend_decomposition.csv",
                layout.tables / "main_harness_backend_cell_means.csv",
                layout.tables / "main_harness_backend_swing_summary.csv",
                layout.tables / "main_harness_backend_alignment_audit.csv",
            ])
            frontier_table = layout.tables / "n5f_legacy_budget_frontier_alpha.csv"
            if include_frontier:
                required.append(frontier_table)
            elif frontier_table.exists():
                errors.append(f"{label}: optional frontier table was emitted without source frontier")
            errors.extend(f"{label}: missing output: {path}" for path in required if not path.exists())
            copied_reference = layout.tables / "reference_quality_acceptance_primary_c6.csv"
            if copied_reference.is_file():
                copied = pd.read_csv(copied_reference)
                required_reference_columns = {
                    "n_rows", "n_metrics_defined", "n_metrics_undefined",
                    "metrics_defined_fraction", "n_spearman_adoption",
                    "n_spearman_revision_gain",
                }
                missing_reference_columns = sorted(required_reference_columns - set(copied.columns))
                if missing_reference_columns:
                    errors.append(
                        f"{label}: copied reference-quality table lost cohort columns: "
                        f"{missing_reference_columns}"
                    )
            state = (
                manifest.get("optional_inputs", {})
                .get("n5_generation_budget_frontier", {})
                .get("status")
            )
            expected_state = "AVAILABLE_PASS" if include_frontier else "NOT_AVAILABLE_OPTIONAL"
            if state != expected_state:
                errors.append(f"{label}: optional frontier state={state!r}, expected={expected_state!r}")
            matched_state = (
                manifest.get("required_inputs", {})
                .get("n5_matched_budget_frontier", {})
                .get("status")
            )
            if matched_state != "AVAILABLE_PASS":
                errors.append(f"{label}: required N5M matched frontier state={matched_state!r}")
            notes = " ".join(str(x) for x in manifest.get("notes", []))
            if "excluded from the main crossed harness-vs-backend variance decomposition" not in notes:
                errors.append(f"{label}: N5/N5F/N5M main-decomposition exclusion note is missing")
            cases[label] = {
                "manifest_status": manifest.get("status"),
                "frontier_state": state,
                "matched_frontier_state": matched_state,
                "tables": sorted(manifest.get("tables", {})),
                "figures": sorted(manifest.get("figures", {})),
            }

    with tempfile.TemporaryDirectory(prefix="paper_assets_partial_frontier_") as td:
        root = Path(td) / "paper_repro"
        _populate(root, include_frontier=False)
        layout = build_layout(root)
        _csv(layout.n5_budget_frontier_holm / "n5_budget_frontier_table_patch.csv", [{"budget_label": "0p75"}])
        try:
            run_assets(root, strict=True)
        except FileNotFoundError:
            cases["partial_frontier_hard_fail"] = "PASS"
        else:
            errors.append("partial frontier directory did not hard-fail")

    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "cases": cases,
    }
    if out_json:
        _json(out_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(Path(args.out_json) if args.out_json else None)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
