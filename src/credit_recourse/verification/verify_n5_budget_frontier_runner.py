from __future__ import annotations

"""Verify the canonical IC-b generation-time N5 budget-frontier runner contract."""

import argparse
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from credit_recourse.contracts.paper_reproduction import load_profile
from credit_recourse.contracts.stage_paths import stage_dir
from credit_recourse.utils.llm_run_archive import (
    archive_path_diagnostics,
    archive_standard_stages,
)

SCHEMA_VERSION = "n5_budget_frontier_runner_contract_v3"


def _allocate_short_same_volume_project_root(project_root: Path) -> Path:
    """Create a short verifier project root beside the real repository.

    The previous verifier used ``tempfile.TemporaryDirectory`` under the user's
    long ``%TEMP%`` path.  With the frontier run label repeated in the
    checkpoint filename, that synthetic-only root could push the archived file
    above legacy Windows MAX_PATH even though the real repository path was
    safe.  A short sibling stays on the same volume and mirrors the production
    path-length contract instead of testing an artificially longer root.
    """
    parent = Path(project_root).resolve().parent
    for _ in range(32):
        candidate = parent / f".vfa-{uuid.uuid4().hex[:6]}"
        if candidate.exists():
            continue
        candidate.mkdir(parents=False, exist_ok=False)
        return candidate
    raise RuntimeError(f"Could not allocate a short same-volume verifier root under {parent}")


def verify(project_root: Path, *, design: str = "legacy_c6_only") -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = load_profile(root)
    script = root / "tools" / "run_n5_budget_frontier_icb.ps1"
    thesis = root / "tools" / "run_thesis_repro.ps1"
    errors: list[str] = []
    if not script.is_file():
        errors.append(f"missing frontier runner: {script}")
        text = ""
    else:
        text = script.read_text(encoding="utf-8-sig")
    if not thesis.is_file():
        errors.append(f"missing thesis runner: {thesis}")
        thesis_text = ""
    else:
        thesis_text = thesis.read_text(encoding="utf-8-sig")


    design_specs = {
        "legacy_c6_only": {
            "profile_key": "n5_budget_frontier",
            "run_role": "paper_n5_budget_frontier_icb",
            "budgeted_conditions": ["C6"],
            "run_prefix": "N5F",
        },
        "matched_c4_c6": {
            "profile_key": "n5_matched_budget_frontier",
            "run_role": "paper_n5_matched_budget_frontier_icb",
            "budgeted_conditions": ["C4", "C6"],
            "run_prefix": "N5M",
        },
    }
    if design not in design_specs:
        errors.append(f"unsupported frontier design: {design!r}")
        spec = design_specs["legacy_c6_only"]
    else:
        spec = design_specs[design]
    cfg = profile["llm"][spec["profile_key"]]
    if cfg.get("run_role") != spec["run_role"]:
        errors.append(f"profile frontier run_role mismatch: {cfg.get('run_role')!r}")
    if cfg.get("information_condition") != "IC-b":
        errors.append(f"profile frontier IC mismatch: {cfg.get('information_condition')!r}")
    if cfg.get("conditions") != ["C4", "C6"] or cfg.get("modes") != ["free_form_10d"]:
        errors.append("profile frontier condition/mode contract mismatch")
    if cfg.get("budgeted_conditions") != spec["budgeted_conditions"]:
        errors.append(
            f"profile frontier budgeted_conditions mismatch: {cfg.get('budgeted_conditions')!r}"
        )
    expected_required = design == "matched_c4_c6"
    if cfg.get("required_for_canonical_analysis") is not expected_required:
        errors.append(
            "frontier canonical-analysis requirement mismatch: "
            f"design={design!r}, expected={expected_required}, "
            f"actual={cfg.get('required_for_canonical_analysis')!r}"
        )
    budgets = cfg.get("l1_budgets")
    if budgets != [0.75, 1.27, 2.0, None]:
        errors.append(f"profile frontier budget order mismatch: {budgets!r}")

    required_markers = (
        '[switch]$PlanOnly',
        'paper_n5_budget_frontier_icb',
        'paper_n5_matched_budget_frontier_icb',
        '[switch]$BudgetC4',
        '[ValidateSet("canonical", "replication")][string]$ExperimentClass',
        '[string]$ReplicationGroupId',
        'function Backend-Slug',
        'function Expected-BackendId',
        'function Assert-ArchivedBackend',
        '@(0.75, 1.27, 2.00, $null)',
        '$Conditions = if ($C4RMatched) { @("C4", "C4R", "C6") } else { @("C4", "C6") }',
        '"--conditions", $ConditionCsv',
        '"--modes", "free_form_10d"',
        '$BudgetedConditionCsv = if ($C4RMatched) { "C4,C4R,C6" } elseif ($BudgetC4) { "C4,C6" } else { "C6" }',
        'if ($null -ne $budget)',
        '$RunPrefix = if ($C4RMatched) { "C4R" } elseif ($ExperimentClass -eq "replication") { "N5MR" } elseif ($BudgetC4) { "N5M" } else { "N5F" }',
        'replication_n5m_icb_',
        'same_date_batch = $true',
        'within_arm_control = if ($C4RMatched)',
        '[switch]$C4RMatched',
        '${ManifestStem}_${DateTag}.json',
        'backend_spec = $Backend',
        'backend_label = $BackendSlug',
        'expected_backend_id = $ExpectedBackendId',
        'credit_recourse.analysis.n5_budget_frontier_holm_inference',
        'replication_analysis_command',
        'replication_analysis_status',
        'n5_budget_frontier_holm_manifest.json',
        'smoke_stage7_n5_budget_contract',
        'verify_n5_budget_frontier_runner',
    )
    for marker in required_markers:
        if marker not in text:
            errors.append(f"frontier runner contract marker missing: {marker}")
    for marker in (
        'if (-not $BudgetC4) { throw "Replication frontier requires -BudgetC4 matched design." }',
        'if ($RunRole -eq $DefaultRunRole) { throw "Replication runs must not use the canonical paper run role." }',
        'ReplicationGroupId is required',
        'if (-not $PlanOnly -and $ExperimentClass -eq "replication")',
        'if ([string]$analysisMeta.status -ne "PASS" -or [string]$analysisMeta.run_role -ne $RunRole -or [string]$analysisMeta.design -ne "matched_c4_c6")',
    ):
        if marker not in text:
            errors.append(f"frontier replication-isolation marker missing: {marker}")
    for marker in (
        "Preserve-CheckpointAcrossCanonicalCleanup",
        "Restore-CheckpointAfterCanonicalCleanup",
        "checkpoint_recovery",
        "PASS (existing immutable archive)",
        "Assert-ArchivedBackend -ArchiveDir",
        'if ($Backend -match "^openai:"',
        'if ($Backend -match "^anthropic:"',
    ):
        if marker not in text:
            errors.append(f"frontier resume/checkpoint marker missing: {marker}")
    if "BudgetFrontierDateTag" not in thesis_text:
        errors.append("run_thesis_repro.ps1 lacks explicit BudgetFrontierDateTag resume support")
    for marker in ("ReplicationGroupId", '"-ExperimentClass", "replication"', '"-ReplicationGroupId"'):
        if marker not in thesis_text:
            errors.append(f"top-level replication integration marker missing: {marker}")
    for marker in (
        'if (-not $PlanOnly -and -not $ConfirmLiveApiSpend) { throw "BudgetFrontier spends live OpenAI API calls.',
        'if (-not $PlanOnly -and $PromptApiKeys -and $ExperimentBackend -match "^openai:"',
        'if (-not $PlanOnly -and $PromptApiKeys -and $ExperimentBackend -match "^anthropic:"',
        'if (-not $PlanOnly -and -not $ConfirmLiveApiSpend) { throw "RemainingAll includes the live budget frontier.',
        'if (-not $PlanOnly -and $PromptApiKeys -and [string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY))',
    ):
        if marker not in thesis_text:
            errors.append(f"top-level plan-only live-spend guard marker missing: {marker}")
    for marker in (
        'if (-not $ConfirmLiveApiSpend) { throw "BudgetFrontier spends live OpenAI API calls.',
        'if (-not $ConfirmLiveApiSpend) { throw "RemainingAll includes the live budget frontier.',
    ):
        if marker in thesis_text:
            errors.append(f"top-level plan-only path is incorrectly blocked by unconditional spend gate: {marker}")

    sample_tag = "20260712_083213"
    sample_run_label = (
        f"{spec['run_prefix']}_C4C6_L1_0p75_ICb_gpt54mini_p50_main_seed1_" + sample_tag
    )
    sample_checkpoint = f"checkpoint_{sample_run_label}.jsonl"
    path_diagnostics = archive_path_diagnostics(
        project_root=root,
        run_label=sample_run_label,
        artifact_dir_name="stage7_llm_action_generation",
        artifact_file_name=sample_checkpoint,
    )
    if not path_diagnostics["final_path_within_legacy_limit"]:
        errors.append(
            "frontier immutable archive path exceeds the conservative Windows legacy limit: "
            f"{path_diagnostics['final_path_length']}"
        )
    if not path_diagnostics["transaction_path_within_legacy_limit"]:
        errors.append(
            "frontier transaction path exceeds the conservative Windows legacy limit: "
            f"{path_diagnostics['transaction_path_length']}"
        )
    if path_diagnostics["transaction_path_length"] >= path_diagnostics["final_path_length"]:
        errors.append("archive transaction path must be shorter than the immutable final path")

    synthetic_archive_status = "NOT_RUN"
    synthetic_archive_path = None
    synthetic_project_root = None
    synthetic_final_path_length = None
    synthetic_cleanup_status = "NOT_RUN"
    try:
        synthetic_root = _allocate_short_same_volume_project_root(root)
        synthetic_project_root = str(synthetic_root)
        synthetic_diagnostics = archive_path_diagnostics(
            project_root=synthetic_root,
            run_label=sample_run_label,
            artifact_dir_name="stage7_llm_action_generation",
            artifact_file_name=sample_checkpoint,
        )
        synthetic_final_path_length = synthetic_diagnostics["final_path_length"]
        if not synthetic_diagnostics["final_path_within_legacy_limit"]:
            raise RuntimeError(
                "short same-volume verifier root still exceeds the legacy Windows path limit: "
                f"{synthetic_diagnostics['final_path_length']}"
            )
        for key in ("stage7", "stage8", "stage9"):
            out = stage_dir(synthetic_root, key)
            out.mkdir(parents=True, exist_ok=True)
            (out / "metadata.json").write_text(
                json.dumps({"stage": key, "status": "PASS"}) + "\n",
                encoding="utf-8",
            )
        checkpoint = stage_dir(synthetic_root, "stage7") / sample_checkpoint
        checkpoint.write_text('{"status":"ok"}\n', encoding="utf-8")
        archived = archive_standard_stages(
            project_root=synthetic_root,
            run_label=sample_run_label,
            run_role=spec["run_role"],
            skip_stage7=False,
            skip_stage8=False,
            skip_stage9=False,
            reproduction_contract={"schema_version": "synthetic_frontier_archive_v1"},
        )
        synthetic_archive_path = archived["path"]
        archived_checkpoint = (
            Path(archived["path"])
            / "stage7_llm_action_generation"
            / sample_checkpoint
        )
        if not archived_checkpoint.is_file():
            raise FileNotFoundError(
                f"synthetic archive checkpoint missing: {archived_checkpoint}"
            )
        synthetic_archive_status = "PASS"
    except Exception as exc:
        synthetic_archive_status = "FAIL"
        errors.append(f"synthetic long-label archive transaction failed: {type(exc).__name__}: {exc}")
    finally:
        if synthetic_project_root is not None:
            try:
                shutil.rmtree(Path(synthetic_project_root))
                synthetic_cleanup_status = "PASS"
            except Exception as exc:
                synthetic_cleanup_status = "FAIL"
                errors.append(
                    "synthetic archive verifier cleanup failed: "
                    f"{type(exc).__name__}: {exc}"
                )

    forbidden = (
        '"--freeform-l1-budget", "None"',
        '"--freeform-l1-budget", "unbounded"',
        'SilentlyContinue',
        'ErrorActionPreference = "Continue"',
    )
    for marker in forbidden:
        if marker in text:
            errors.append(f"frontier runner contains forbidden fallback/encoding: {marker}")
    if 'run_n5_budget_frontier_icb.ps1' not in thesis_text:
        errors.append("run_thesis_repro.ps1 does not invoke the frontier runner")
    for marker in (
        'N5M IC-b matched generation-time budget frontier',
        'IC-b C4/C6 generation-time budget frontier',
    ):
        if marker not in thesis_text:
            errors.append(f"run_thesis_repro.ps1 lacks the named frontier step: {marker}")
    for marker in (
        "Invoke-FrontierPaperAssetRefresh",
        "refresh paper assets after generation-time frontier",
        "refresh final paper-analysis output contract",
    ):
        if marker not in thesis_text:
            errors.append(f"frontier post-analysis refresh marker missing: {marker}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "runner": str(script),
        "design": design,
        "profile_key": spec["profile_key"],
        "profile_run_role": cfg.get("run_role"),
        "profile_budgets": budgets,
        "same_date_conditions": cfg.get("conditions"),
        "budgeted_conditions": cfg.get("budgeted_conditions"),
        "archive_path_diagnostics": path_diagnostics,
        "synthetic_archive_transaction_status": synthetic_archive_status,
        "synthetic_archive_path": synthetic_archive_path,
        "synthetic_project_root": synthetic_project_root,
        "synthetic_final_path_length": synthetic_final_path_length,
        "synthetic_cleanup_status": synthetic_cleanup_status,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--design", choices=["legacy_c6_only", "matched_c4_c6"], default="legacy_c6_only")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(Path(args.project_root), design=args.design)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
