from __future__ import annotations

"""Synthetic verifier for the lightweight input check and paper-analysis plan."""

import argparse
import hashlib
import io
import json
import shutil
import tempfile
from contextlib import redirect_stderr
from argparse import Namespace
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.analysis.paper_repro_analysis import (
    CANONICAL_TASK_ORDER,
    _find_frontier_panel,
    _validate_frontier_resume_state,
    main as paper_repro_main,
    run_analysis,
)
from credit_recourse.configs import __path__ as config_package_paths


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _make_standard_run(
    root: Path,
    *,
    label: str,
    role: str,
    ic: str,
    backend: str,
    conditions: list[str],
    modes: list[str],
    budget: float | None = None,
    budgeted_conditions: list[str] | None = None,
    seed_location: str = "metadata",
    budget_location: str = "metadata_contract",
    persist_role: bool = True,
) -> None:
    run = root / "data" / "final_freeze" / "llm_runs" / label
    s7 = run / "stage7_llm_action_generation"
    s8 = run / "stage8_llm_multi_oracle_eval"
    s9 = run / "stage9_llm_rl_comparison"
    for path in (s7, s8, s9):
        path.mkdir(parents=True, exist_ok=True)
    (s7 / "llm_stage7_action_table.parquet").write_bytes(b"synthetic-plan-placeholder")
    meta = {
        "status": "PASS",
        "information_condition": ic,
        "backend_id": backend.replace(":", "_"),
        "backend_is_live": True,
        "conditions": conditions,
        "modes": modes,
        "reference_draw_seed": 1,
        "candidate_library_quantile": 50,
        "row_count": 575,
        "request_count": 575 * len(conditions) * len(modes),
    }
    if persist_role:
        meta["run_role"] = role
    if seed_location == "metadata":
        meta["seed"] = 1
    elif seed_location not in {"archive", "legacy_label"}:
        raise ValueError(f"Unsupported seed_location: {seed_location}")
    if budget is not None:
        if budget_location == "metadata_contract":
            meta["action_budget_contract"] = {
                "schema_version": "stage7_freeform_l1_budget_contract_v1",
                "enabled": True,
                "l1_budget": budget,
                "label": "synthetic_n5_budget",
                "budgeted_conditions": list(budgeted_conditions or ["C6"]),
                "budgeted_modes": ["free_form_10d"],
                "tolerance": 1.0e-9,
            }
        elif budget_location == "legacy_top_level":
            meta["freeform_l1_budget"] = budget
        elif budget_location != "archive":
            raise ValueError(f"Unsupported budget_location: {budget_location}")
    _json(s7 / "metadata.json", meta)
    provider, model = backend.split(":", 1)
    _json(s7 / "llm_stage7_prompt_manifest.json", {
        "backend": {"model": model, "provider_options": {"provider": provider}}
    })
    _json(s8 / "metadata.json", {"status": "PASS", "final_paper_run_allowed": True})
    _json(s9 / "metadata.json", {"status": "PASS", "final_paper_run_allowed": True})
    pd.DataFrame([{
        "row_id": 1,
        "policy": "C6",
        "mode": "free_form_10d",
        "oracle_backend": "alpha",
        "delta_R_score": 0.0,
    }]).to_csv(s9 / "llm_stage9_llm_rl_comparison.csv", index=False)
    pd.DataFrame([{
        "row_id": 1, "base_condition": "C4", "revision_condition": "C6",
        "mode": "free_form_10d", "rl_reference_candidate": "REF",
        "rl_adoption_ratio": 0.5, "orthogonal_drift": 0.0,
        "u_norm_squared": 1.0, "metrics_defined": True,
        "initial_delta_R_score_alpha": 0.0, "revised_delta_R_score_alpha": 0.1, "revision_delta_R_score_alpha": 0.1,
        "initial_delta_R_score_beta": 0.0, "revised_delta_R_score_beta": 0.1, "revision_delta_R_score_beta": 0.1,
        "initial_delta_R_score_gamma": 0.0, "revised_delta_R_score_gamma": 0.1, "revision_delta_R_score_gamma": 0.1,
    }]).to_csv(s9 / "llm_stage9_revision_metrics.csv", index=False)
    archive_manifest = {
        "schema_version": "llm_run_archive_manifest_v2",
        "run_label": label,
        "run_type": "stage7_8_9",
    }
    if persist_role:
        archive_manifest["run_role"] = role
    reproduction_contract: dict[str, Any] = {}
    if seed_location == "archive":
        reproduction_contract.update({
            "schema_version": "llm_run_reproduction_contract_v1",
            "seed": 1,
            "reference_draw_seed": 1,
            "candidate_library_quantile": 50,
        })
    if budget is not None and budget_location == "archive":
        reproduction_contract.update({
            "schema_version": "llm_run_reproduction_contract_v1",
            "freeform_l1_budget": budget,
        })
    if reproduction_contract:
        archive_manifest["extra"] = {"reproduction_contract": reproduction_contract}
    _json(run / "archive_manifest.json", archive_manifest)


def _populate(root: Path) -> None:
    package_config_root = Path(list(config_package_paths)[0])
    package_config = package_config_root / "paper_reproduction_profile.json"
    dest_config_root = root / "src" / "credit_recourse" / "configs"
    dest_config = dest_config_root / "paper_reproduction_profile.json"
    dest_config_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(package_config, dest_config)
    selection_contract_source = package_config_root / "n5m_adaptive_selection_contract.json"
    shutil.copy2(
        selection_contract_source,
        dest_config_root / "n5m_adaptive_selection_contract.json",
    )
    frozen_config_root = root / "data" / "final_freeze" / "configs"
    frozen_config_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        selection_contract_source,
        frozen_config_root / "n5m_adaptive_selection_contract.json",
    )
    (root / "tools").mkdir(parents=True, exist_ok=True)

    full_conditions = ["C4", "C5", "C6", "C6X", "C7", "C8"]
    full_modes = ["candidate_selection", "free_form_10d"]
    primary_seed_locations = {
        "IC-a": "legacy_label",
        "IC-b": "archive",
        "IC-c": "metadata",
    }
    for ic in ("IC-a", "IC-b", "IC-c"):
        suffix = "_seed1" if ic == "IC-a" else ""
        _make_standard_run(
            root, label=f"primary_{ic}{suffix}", role="paper_primary_gpt54", ic=ic,
            backend="openai:gpt-5.4-mini", conditions=full_conditions, modes=full_modes,
            seed_location=primary_seed_locations[ic],
        )
    _make_standard_run(
        root, label="supp_gpt41_IC-b", role="paper_supplementary_gpt41", ic="IC-b",
        backend="openai:gpt-4.1-mini", conditions=full_conditions, modes=full_modes,
    )
    _make_standard_run(
        root, label="supp_haiku_IC-b", role="paper_supplementary_haiku45", ic="IC-b",
        backend="anthropic:claude-haiku-4-5-20251001", conditions=full_conditions, modes=full_modes,
    )
    _make_standard_run(
        root,
        label="N5_C6_L1_1p27_ICa_gpt54mini_p50_main_seed1_20260707",
        role="paper_n5_l1_1p27",
        ic="IC-a",
        backend="openai:gpt-5.4-mini",
        conditions=["C4", "C6"],
        modes=["free_form_10d"],
        budget=1.27,
        budget_location="metadata_contract",
        persist_role=False,
    )
    _make_standard_run(
        root, label="n5_IC-b", role="paper_n5_l1_1p27", ic="IC-b",
        backend="openai:gpt-5.4-mini", conditions=["C4", "C6"], modes=["free_form_10d"],
        budget=1.27, budget_location="archive",
    )
    _make_standard_run(
        root, label="n5_IC-c", role="paper_n5_l1_1p27", ic="IC-c",
        backend="openai:gpt-5.4-mini", conditions=["C4", "C6"], modes=["free_form_10d"],
        budget=1.27, budget_location="legacy_top_level",
    )
    for label, budget in (
        ("n5f_IC-b_0p75", 0.75),
        ("n5f_IC-b_1p27", 1.27),
        ("n5f_IC-b_2p00", 2.00),
        ("n5f_IC-b_unbounded", None),
    ):
        _make_standard_run(
            root, label=label, role="paper_n5_budget_frontier_icb", ic="IC-b",
            backend="openai:gpt-5.4-mini", conditions=["C4", "C6"], modes=["free_form_10d"],
            budget=budget, budget_location="metadata_contract",
        )

    for label, budget in (
        ("n5m_IC-b_0p75", 0.75),
        ("n5m_IC-b_1p27", 1.27),
        ("n5m_IC-b_2p00", 2.00),
        ("n5m_IC-b_unbounded", None),
    ):
        _make_standard_run(
            root,
            label=label,
            role="paper_n5_matched_budget_frontier_icb",
            ic="IC-b",
            backend="openai:gpt-5.4-mini",
            conditions=["C4", "C6"],
            modes=["free_form_10d"],
            budget=budget,
            budget_location="metadata_contract",
            budgeted_conditions=["C4", "C6"],
        )

    probe = root / "data" / "final_freeze" / "llm_runs" / "probe_icc"
    stage = probe / "stage7_icc_probe"
    stage.mkdir(parents=True, exist_ok=True)
    primary_icc = root / "data" / "final_freeze" / "llm_runs" / "primary_IC-c"
    base_stage7 = primary_icc / "stage7_llm_action_generation"
    _json(stage / "icc_probe_summary.json", {
        "schema_version": "icc_probe_only_summary_v2",
        "status": "PASS",
        "run_label": "probe_icc",
        "run_role": "paper_icc_probe",
        "execution_mode": "probe_only_from_archived_stage7",
        "information_condition": "IC-c",
        "base_run_label": "primary_IC-c",
        "base_stage7_metadata_sha256": _sha256(base_stage7 / "metadata.json"),
        "base_stage7_action_table_sha256": _sha256(base_stage7 / "llm_stage7_action_table.parquet"),
        "probe_schema_version": "icc_probe_numeric_recall_v2",
        "probe_target_feature": "derived__debt_to_assets",
        "probe_tolerance_relative": 0.20,
        "probe_row_count": 575,
        "main_stage7_api_calls": 0,
        "main_checkpoint_used": False,
        "stage8_stage9_rerun": False,
        "backend": {"backend_id": "openai_gpt-5.4-mini", "model": "gpt-5.4-mini"},
    })
    (stage / "icc_probe_firm_level.csv").write_text("row_id\n", encoding="utf-8")
    (stage / "llm_stage7_icc_probe_checkpoint.jsonl").write_text("", encoding="utf-8")
    _json(probe / "archive_manifest.json", {
        "schema_version": "llm_run_archive_manifest_v2", "run_label": "probe_icc",
        "run_role": "paper_icc_probe", "run_type": "icc_probe_only",
    })

    summary = root / "data" / "final_freeze" / "stage6_candidate_selector_eval" / "final_policy_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "policy": "C3_candidate_iql", "n": 575,
        "mean_delta_R_score_alpha": 0.630313,
        "mean_delta_R_score_beta": 0.076,
        "mean_delta_R_score_gamma": 0.595,
    }]).to_csv(summary, index=False)
    pd.DataFrame([{
        "row_id": 1, "policy": "REF", "candidate_id": "REF",
        "delta_R_score_alpha": 0.2, "delta_R_score_beta": 0.2, "delta_R_score_gamma": 0.2,
    }]).to_parquet(summary.parent / "multi_oracle_policy_eval.parquet", index=False)

    b2 = root / "data" / "final_freeze" / "stage2_substrate_loopA_loopB2"
    b2.mkdir(parents=True, exist_ok=True)
    _json(b2 / "substrate_loopA_loopB2_report.json", {"status": "PASS"})

    # The active canonical plan now requires the frozen Stage2 evaluation state
    # panel used by the Section 9.8 pre-action selector.  Include protected
    # audit columns as a regression fixture: the selector must ignore them, not
    # weaken the allow-list or reject the whole upstream panel.
    stage2 = root / "data" / "final_freeze" / "stage2_candidate_projection"
    stage2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": list(range(575)),
        "derived__debt_to_assets": [0.20 + (i % 17) * 0.01 for i in range(575)],
        "derived__current_ratio": [0.80 + (i % 19) * 0.03 for i in range(575)],
        "derived__operating_margin": [-0.05 + (i % 23) * 0.01 for i in range(575)],
        "delta_1y__derived__debt_to_assets": [((i % 11) - 5) * 0.005 for i in range(575)],
        "log_assets": [8.0 + (i % 29) * 0.08 for i in range(575)],
        "industry_class": [f"IND_{i % 7}" for i in range(575)],
        "candidate_id": ["PROTECTED_NOT_A_FEATURE"] * 575,
        "rating_num": [5] * 575,
    }).to_parquet(stage2 / "phase_eval_candidate.parquet", index=False)
    (root / "data" / "raw" / "raw_nonfinancial").mkdir(parents=True, exist_ok=True)


def verify(out_json: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []

    runner_path = Path(__file__).resolve().parents[3] / "tools" / "run_postfreeze_analysis.ps1"
    if not runner_path.is_file():
        errors.append(f"canonical PowerShell runner is missing: {runner_path}")
    else:
        runner_text = runner_path.read_text(encoding="utf-8-sig")
        for marker in (
            "function Invoke-PythonCaptured",
            '$ErrorActionPreference = "Continue"',
            "--failure-report",
            "--concise-errors",
            "Write-StructuredPythonFailure",
            "[switch]$PlanOnly",
            "[string]$ResumeFailedAnalysisPath",
            "$ResumeInPlace",
            "canonical partial paper analysis resumed in place",
            "--resume-after-frontier",
            "postfreeze_import_smoke_",
            "[System.IO.File]::WriteAllText",
            "New-Object System.Text.UTF8Encoding -ArgumentList $false",
        ):
            if marker not in runner_text:
                errors.append(f"PowerShell error-capture contract marker missing: {marker}")
        if runner_text.count("& $script:Py @Arguments 2>&1") != 1:
            errors.append("PowerShell runner must have exactly one native Python invocation inside Invoke-PythonCaptured")
        if "& $Py @Arguments" in runner_text or "| & $Py -" in runner_text:
            errors.append("PowerShell runner contains a direct Python invocation outside the capture helper")
        if '@("-c", $ImportSmoke)' in runner_text or "@('-c', $ImportSmoke)" in runner_text:
            errors.append(
                "PowerShell runner passes multiline import smoke source through python -c; "
                "Windows PowerShell 5.1 can corrupt embedded quotes"
            )
        for forbidden in (
            "credit_recourse.verification.verify_paper_repro_analysis_runner",
            "compileall",
            "frozen-artifact plan",
        ):
            if forbidden in runner_text:
                errors.append(f"heavy preflight remains in active post-freeze runner: {forbidden}")
        if "--check-inputs-only" not in runner_text:
            errors.append("active post-freeze runner does not call the lightweight required-input check")

    thesis_runner = Path(__file__).resolve().parents[3] / "tools" / "run_thesis_repro.ps1"
    if not thesis_runner.is_file():
        errors.append(f"canonical thesis runner is missing: {thesis_runner}")
    else:
        thesis_text = thesis_runner.read_text(encoding="utf-8-sig")
        for marker in (
            '"AnalysisResume"',
            "Auto-selected canonical partial analysis for in-place resume",
            '$canonicalStatus -in @("FAIL", "RUNNING")',
            '"-ResumeFailedAnalysisPath", $FailedAnalysisPath',
        ):
            if marker not in thesis_text:
                errors.append(f"canonical in-place resume runner marker missing: {marker}")

    with tempfile.TemporaryDirectory(prefix="paper_repro_plan_") as td:
        root = Path(td) / "repo"
        _populate(root)
        out = root / "data" / "analysis" / "paper_repro"
        input_check = run_analysis(Namespace(
            project_root=str(root), analysis_dir=str(out), stage6_summary=None,
            plan_only=False, check_inputs_only=True,
        ))
        if input_check.get("status") != "PASS":
            errors.append(f"required-input check failed: {input_check}")
        if input_check.get("required_input_count", 0) < 10:
            errors.append(f"required-input check covered too few files: {input_check}")

        args = Namespace(
            project_root=str(root), analysis_dir=str(out), stage6_summary=None,
            plan_only=True, check_inputs_only=False,
        )
        manifest = run_analysis(args)
        if manifest.get("status") != "PLANNED":
            errors.append(f"unexpected manifest status: {manifest.get('status')}")
        labels = [step.get("label") for step in manifest.get("steps", [])]
        if any(str(label).startswith("Stage8 feasibility contract") for label in labels):
            errors.append("Stage8 version/feasibility verifier remains in the active analysis plan")
        if "fresh-workspace and raw-input contract" in labels:
            errors.append("workspace/hash verifier remains in the active analysis plan")
        for expected in (
            "B2 gap decomposition", "Test3 counterfactual-device properties",
            "H1/H3 Holm inference", "reference-quality acceptance response",
            "N5 Table 7-10c Holm and C3 audit",
            "N5 generation-time budget frontier Holm",
            "N5M matched generation-time budget frontier Holm",
            "N5M matched-budget post-hoc diagnostics",
            "N5M Section 9.8 adaptive-budget selector and post-C4 gate",
            "N1 full budget frontier", "N3 win rate and N6 heterogeneity",
            "IC-c probe import and channel summary", "paper-facing tables and figures",
            "final paper-analysis output contract",
        ):
            if expected not in labels:
                errors.append(f"missing planned step: {expected}")
        if not (out / "00_manifest" / "paper_repro_analysis_manifest.json").exists():
            errors.append("canonical manifest path missing")
        selected = manifest.get("selected_runs", {})
        if (
            len(selected.get("primary", [])) != 3
            or len(selected.get("n5", [])) != 3
            or len(selected.get("n5_budget_frontier", [])) != 4
            or len(selected.get("n5_matched_budget_frontier", [])) != 4
            or len(selected.get("holm", [])) != 3
        ):
            errors.append(f"unexpected selected run counts: {selected}")
        frontier_budgets = sorted(
            ("unbounded" if row.get("freeform_l1_budget") is None else f"{float(row['freeform_l1_budget']):.2f}")
            for row in selected.get("n5_budget_frontier", [])
        )
        if frontier_budgets != ["0.75", "1.27", "2.00", "unbounded"]:
            errors.append(f"unexpected frontier budget selection: {frontier_budgets}")
        matched_frontier_budgets = sorted(
            ("unbounded" if row.get("freeform_l1_budget") is None else f"{float(row['freeform_l1_budget']):.2f}")
            for row in selected.get("n5_matched_budget_frontier", [])
        )
        if matched_frontier_budgets != ["0.75", "1.27", "2.00", "unbounded"]:
            errors.append(f"unexpected matched frontier budget selection: {matched_frontier_budgets}")
        if any(row.get("paper_use") != "primary" for row in selected.get("n5_matched_budget_frontier", [])):
            errors.append("canonical N5M runs are not marked paper_use=primary")
        if any(row.get("paper_use") != "supplementary" for row in selected.get("n5_budget_frontier", [])):
            errors.append("legacy N5F runs are not marked paper_use=supplementary")

        frontier_backup = root / "_frontier_contract_backup"
        frontier_backup.mkdir(parents=True, exist_ok=True)
        frontier_run_dirs = [Path(row["run_dir"]) for row in selected.get("n5_budget_frontier", [])]
        for run_dir in frontier_run_dirs:
            shutil.copytree(run_dir, frontier_backup / run_dir.name)
            shutil.rmtree(run_dir)

        absent_out = root / "data" / "analysis" / "paper_repro_optional_frontier_absent"
        absent_manifest = run_analysis(Namespace(
            project_root=str(root), analysis_dir=str(absent_out), stage6_summary=None,
            plan_only=True, check_inputs_only=False,
        ))
        absent_selected = absent_manifest.get("selected_runs", {}).get("n5_budget_frontier", [])
        if absent_selected != []:
            errors.append(f"optional absent frontier selected unexpected runs: {absent_selected}")
        absent_steps = {step.get("label"): step for step in absent_manifest.get("steps", [])}
        absent_frontier_step = absent_steps.get("N5 generation-time budget frontier Holm", {})
        if absent_frontier_step.get("status") != "SKIPPED_OPTIONAL_NOT_AVAILABLE":
            errors.append(f"optional absent frontier was not explicitly skipped: {absent_frontier_step}")

        one_frontier = frontier_run_dirs[0]
        shutil.copytree(frontier_backup / one_frontier.name, one_frontier)
        partial_out = root / "data" / "analysis" / "paper_repro_partial_frontier"
        try:
            run_analysis(Namespace(
                project_root=str(root), analysis_dir=str(partial_out), stage6_summary=None,
                plan_only=True, check_inputs_only=False,
            ))
        except Exception as exc:
            if type(exc).__name__ != "ProfileError" or "found 1" not in str(exc):
                errors.append(f"partial frontier raised unexpected error: {type(exc).__name__}: {exc}")
        else:
            errors.append("partial generation-time frontier did not hard-fail")
        shutil.rmtree(one_frontier)

        primary_by_ic = {row["information_condition"]: row for row in selected.get("primary", [])}
        expected_seed_sources = {
            "IC-a": "legacy_run_label.explicit_seed_token",
            "IC-b": "archive_manifest.extra.reproduction_contract.seed",
            "IC-c": "stage7.metadata.seed",
        }
        for ic, expected_source in expected_seed_sources.items():
            row = primary_by_ic.get(ic)
            if row is None:
                errors.append(f"missing primary run for provenance test: {ic}")
                continue
            if row.get("seed") != 1 or row.get("seed_source") != expected_source:
                errors.append(
                    f"seed provenance mismatch for {ic}: "
                    f"seed={row.get('seed')}, source={row.get('seed_source')}, "
                    f"expected seed=1 source={expected_source}"
                )

        n5_by_ic = {row["information_condition"]: row for row in selected.get("n5", [])}
        expected_budget_sources = {
            "IC-a": "stage7.metadata.action_budget_contract.l1_budget",
            "IC-b": "archive_manifest.extra.reproduction_contract.freeform_l1_budget",
            "IC-c": "stage7.metadata.freeform_l1_budget",
        }
        for ic, expected_source in expected_budget_sources.items():
            row = n5_by_ic.get(ic)
            if row is None:
                errors.append(f"missing N5 run for budget provenance test: {ic}")
                continue
            if (
                row.get("freeform_l1_budget") != 1.27
                or row.get("freeform_l1_budget_source") != expected_source
            ):
                errors.append(
                    f"budget provenance mismatch for {ic}: "
                    f"budget={row.get('freeform_l1_budget')}, "
                    f"source={row.get('freeform_l1_budget_source')}, "
                    f"expected budget=1.27 source={expected_source}"
                )
        probe_selected = selected.get("icc_probe", {})
        if probe_selected.get("source_kind") != "canonical_llm_archive":
            errors.append(f"unexpected canonical probe source selection: {probe_selected}")

        # Regression test for the 2026-07-11 frontier panel identity bug. The
        # frontier contains all three N5 information conditions, so selecting by
        # the primary IC-a label must fail while the exact N5 IC-a label resolves.
        frontier_root = out / "03_output_contract_diagnostics" / "budget_frontier"
        frontier_rows: list[dict[str, Any]] = []
        for row in selected.get("n5", []):
            cell = (
                frontier_root
                / row["run_label"]
                / "b1p27"
                / "l1_rescale"
            )
            cell.mkdir(parents=True, exist_ok=True)
            (cell / "simulated_oracle_input_frame.parquet").write_bytes(b"synthetic-frontier-panel")
            frontier_rows.append({
                "run_label": row["run_label"],
                "budget": 1.27,
                "variant": "l1_rescale",
                "status": "PASS",
                "output_dir": str(cell),
            })
        _json(frontier_root / "metadata.json", {
            "status": "PASS", "dry_run": False, "n_cells": len(frontier_rows)
        })
        pd.DataFrame(frontier_rows).to_csv(
            frontier_root / "frontier_grid_status.csv", index=False
        )
        n5_ic_a_label = n5_by_ic["IC-a"]["run_label"]
        selected_panel = _validate_frontier_resume_state(frontier_root, n5_ic_a_label)
        if selected_panel.parents[2].name != n5_ic_a_label:
            errors.append(f"frontier panel resolved to wrong run: {selected_panel}")
        try:
            _find_frontier_panel(frontier_root, primary_by_ic["IC-a"]["run_label"])
        except FileNotFoundError:
            pass
        else:
            errors.append(
                "frontier panel selector accepted the primary IC-a label instead of the exact N5 IC-a run label"
            )

        canonical_probe = root / "data" / "final_freeze" / "llm_runs" / "probe_icc"
        legacy_probe = (
            root
            / "data"
            / "final_freeze"
            / "icc_probe_runs"
            / "probe_icc_legacy"
        )
        legacy_probe.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(canonical_probe / "stage7_icc_probe", legacy_probe)
        shutil.rmtree(canonical_probe)

        legacy_out = root / "data" / "analysis" / "paper_repro_legacy_probe"
        legacy_manifest = run_analysis(
            Namespace(
                project_root=str(root),
                analysis_dir=str(legacy_out),
                stage6_summary=None,
                plan_only=True,
                check_inputs_only=False,
            )
        )
        legacy_selected = legacy_manifest.get("selected_runs", {}).get("icc_probe", {})
        if legacy_selected.get("source_kind") != "legacy_final_freeze_probe_root":
            errors.append(f"legacy probe compatibility selection failed: {legacy_selected}")
        if legacy_selected.get("base_run_label") != "primary_IC-c":
            errors.append(f"legacy probe base-run contract mismatch: {legacy_selected}")

        # Historical Move-Item operations can add an extra directory level.
        # The resolver must remain strict about supported roots while accepting
        # this nested legacy layout.
        nested_root = (
            root
            / "data"
            / "archive"
            / "icc_probe_runs_legacy_20260711"
            / "icc_probe_runs"
            / "probe_icc_nested"
        )
        nested_root.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(legacy_probe, nested_root)
        shutil.rmtree(root / "data" / "final_freeze" / "icc_probe_runs")

        nested_out = root / "data" / "analysis" / "paper_repro_nested_legacy_probe"
        nested_manifest = run_analysis(
            Namespace(
                project_root=str(root),
                analysis_dir=str(nested_out),
                stage6_summary=None,
                plan_only=True,
                check_inputs_only=False,
            )
        )
        nested_selected = nested_manifest.get("selected_runs", {}).get("icc_probe", {})
        if nested_selected.get("source_kind") != "legacy_archived_probe_root":
            errors.append(f"nested legacy probe selection failed: {nested_selected}")

        # Exercise the exact no-probe branch so an undefined ProfileError symbol
        # cannot survive compile/import checks again.
        shutil.rmtree(root / "data" / "archive" / "icc_probe_runs_legacy_20260711")
        no_probe_out = root / "data" / "analysis" / "paper_repro_no_probe"
        try:
            run_analysis(
                Namespace(
                    project_root=str(root),
                    analysis_dir=str(no_probe_out),
                    stage6_summary=None,
                    plan_only=True,
                )
            )
        except FileNotFoundError as exc:
            if "Required IC-c probe files were not found" not in str(exc):
                errors.append(f"unexpected no-probe FileNotFoundError message: {exc}")
        except Exception as exc:
            errors.append(
                "no-probe branch raised the wrong exception type: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            errors.append("no-probe branch did not hard-fail")

        # The PowerShell runner uses the concise CLI path so Windows PowerShell
        # does not replace the actual Python failure with NativeCommandError.
        failure_report = root / "data" / "diagnostics" / "paper_repro_failure.json"
        concise_stderr = io.StringIO()
        with redirect_stderr(concise_stderr):
            concise_rc = paper_repro_main([
                "--project-root",
                str(root),
                "--analysis-dir",
                str(root / "data" / "analysis" / "paper_repro_cli_failure"),
                "--check-inputs-only",
                "--failure-report",
                str(failure_report),
                "--concise-errors",
            ])
        concise_text = concise_stderr.getvalue()
        if concise_rc != 1:
            errors.append(f"concise CLI failure returned rc={concise_rc}, expected=1")
        if "FileNotFoundError: Required IC-c probe files were not found" not in concise_text:
            errors.append(f"concise CLI message is incomplete: {concise_text!r}")
        if "Traceback (most recent call last)" in concise_text:
            errors.append("concise CLI leaked a traceback to stderr")
        if not failure_report.is_file():
            errors.append("concise CLI did not write its structured failure report")
        else:
            failure = json.loads(failure_report.read_text(encoding="utf-8-sig"))
            if failure.get("exception_type") != "FileNotFoundError":
                errors.append(f"failure report exception type mismatch: {failure}")
            if "Required IC-c probe files were not found" not in str(failure.get("exception_message")):
                errors.append(f"failure report message mismatch: {failure}")
            if "Traceback (most recent call last)" not in str(failure.get("traceback")):
                errors.append("failure report does not retain the full traceback")

        result = {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "canonical_task_order": list(CANONICAL_TASK_ORDER),
            "planned_step_count": len(labels),
            "planned_labels": labels,
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
