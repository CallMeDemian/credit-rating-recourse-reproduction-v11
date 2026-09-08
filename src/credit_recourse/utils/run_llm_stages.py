from __future__ import annotations

"""Sequential Stage 7 → 8 → 9 runner.

Per the Stage 7-9 LLM contract §1, the LLM stages consume the *frozen*
Stage 6 substrate (simulator, Oracle backend artifacts, candidate library,
temporal split, RL winner) and produce their own outputs in canonical
``data/final_freeze/stage{7,8,9}_*`` directories.  This script provides a
single CLI entry that:

1. Runs Stage 7 LLM action generation with the configured backend, optional
   request-level checkpoint/resume, bounded concurrency, and retry policy.
2. Runs Stage 8 LLM multi-oracle evaluation.
3. Runs Stage 9 LLM-RL comparison + revision analysis.
4. Invokes the Stage 7/8/9 boundary verifiers and writes ledgers.

The runner is fail-fast: any stage that exits non-zero or any verifier that
reports errors aborts the run.  This matches the existing Stage 0-6
orchestrator's error policy.

The runner does **not** rerun Stage 0-6.  Those must already be present
under ``data/final_freeze/`` (Stage 1 backend registry, Stage 2
``phase_eval_candidate.parquet``, Stage 6 ``policy_actions.parquet`` and
``oracle_scores_*.parquet``).
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import (
    run_stage8,
)
from credit_recourse.eval.final_stage9_llm_rl_comparison.pipeline import (
    run_stage9,
)
from credit_recourse.eval.final_stage9_statistical_inference import run as run_stage9_inference
from credit_recourse.rl.common.io import write_json
from credit_recourse.utils.llm_run_archive import archive_standard_stages
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.pipeline import (
    run_stage7,
)
from credit_recourse.verification import stage_boundary_contracts as sbc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verify_or_fail(root: Path, stage: str) -> dict:
    """Run the stage-boundary verifier and write its ledger; raise on fail."""
    result = sbc.verify(root, stage)
    ledger_path = final_root(root) / "ledgers" / f"verify_{stage}.json"
    write_json(ledger_path, result)
    if result["status"] != "PASS":
        raise RuntimeError(
            f"{stage} verifier failed.  Ledger: {ledger_path}.  "
            f"Errors: {result.get('errors', [])}"
        )
    return result


def _archive_llm_run(
    *,
    project_root: Path,
    run_label: str,
    run_role: str,
    skip_stage7: bool,
    skip_stage8: bool,
    skip_stage9: bool,
    reproduction_contract: dict,
) -> dict:
    """Delegate immutable Stage7/8/9 archiving to the shared LLM archive contract."""
    return archive_standard_stages(
        project_root=project_root,
        run_label=run_label,
        run_role=run_role,
        skip_stage7=skip_stage7,
        skip_stage8=skip_stage8,
        skip_stage9=skip_stage9,
        reproduction_contract=reproduction_contract,
    )

def run_llm_stages(
    *,
    project_root: Path,
    backend_spec: str,
    information_condition: str,
    conditions: list[str],
    modes: list[str],
    seed: int,
    reference_draw_seed: int | None = None,
    candidate_library_quantile: int = 50,
    max_concurrency: int = 1,
    max_retries: int = 2,
    retry_sleep_seconds: float = 1.0,
    resume_stage7: bool = True,
    checkpoint_path: Path | None = None,
    openai_api_mode: str | None = None,
    openai_reasoning_effort: str | None = None,
    openai_max_output_tokens: int | None = None,
    anthropic_temperature: float | None = None,
    anthropic_max_tokens: int | None = None,
    anthropic_thinking_budget_tokens: int | None = None,
    gemini_thinking_level: str | None = None,
    gemini_max_output_tokens: int | None = None,
    gemini_response_mime_type: str | None = None,
    gemini_timeout_seconds: float | None = None,
    icc_probe: bool = False,
    firm_name_lookup_path: Path | None = None,
    icc_probe_tolerance: float = 0.20,
    run_label: str | None = None,
    run_role: str = "unclassified_llm_run",
    skip_stage7: bool = False,
    skip_stage8: bool = False,
    skip_stage9: bool = False,
    run_inference: bool = False,
    llm_runs_dir: Path | None = None,
    mde_freeze: bool = False,
    row_id_file: Path | None = None,
    sample_size: int | None = None,
    sample_seed: int = 20260707,
    sample_strata: list[str] | None = None,
    freeform_l1_budget: float | None = None,
    budgeted_conditions: list[str] | None = None,
    budget_contract_label: str | None = None,
    budget_tolerance: float = 1.0e-9,
) -> dict:
    """Execute the LLM stages 7 → 8 → 9 sequentially with verification.

    Each stage's metadata and verifier ledger are written to disk regardless
    of whether subsequent stages succeed.  Returns a roll-up dict that names
    every stage's status.
    """
    project_root = Path(project_root).resolve()
    final_root(project_root).mkdir(parents=True, exist_ok=True)
    ledger_dir = final_root(project_root) / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    rollup: dict = {
        "runner": "run_llm_stages",
        "created_utc": _now(),
        "project_root": str(project_root),
        "backend_spec": backend_spec,
        "run_role": run_role,
        "information_condition": information_condition,
        "conditions": conditions,
        "modes": modes,
        "seed": seed,
        "reference_draw_seed": int(reference_draw_seed if reference_draw_seed is not None else seed),
        "candidate_library_quantile": int(candidate_library_quantile),
        "stage7_checkpoint": {
            "resume": bool(resume_stage7),
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "max_concurrency": int(max_concurrency),
            "max_retries": int(max_retries),
            "retry_sleep_seconds": float(retry_sleep_seconds),
        },
        "openai_options": {
            "api_mode": openai_api_mode,
            "reasoning_effort": openai_reasoning_effort,
            "max_output_tokens": openai_max_output_tokens,
        },
        "anthropic_options": {
            "temperature": anthropic_temperature,
            "max_tokens": anthropic_max_tokens,
            "thinking_budget_tokens": anthropic_thinking_budget_tokens,
        },
        "gemini_options": {
            "thinking_level": gemini_thinking_level,
            "max_output_tokens": gemini_max_output_tokens,
            "response_mime_type": gemini_response_mime_type,
            "timeout_seconds": gemini_timeout_seconds,
        },
        "icc_probe": bool(icc_probe),
        "icc_probe_tolerance": float(icc_probe_tolerance),
        "firm_name_lookup_path": str(firm_name_lookup_path) if firm_name_lookup_path is not None else None,
        "run_label": run_label,
        "run_inference": bool(run_inference),
        "llm_runs_dir": str(llm_runs_dir) if llm_runs_dir is not None else None,
        "mde_freeze": bool(mde_freeze),
        "row_id_file": str(row_id_file) if row_id_file is not None else None,
        "sample_size": int(sample_size) if sample_size is not None else None,
        "sample_seed": int(sample_seed),
        "sample_strata": list(sample_strata or []),
        "freeform_l1_budget": float(freeform_l1_budget) if freeform_l1_budget is not None else None,
        "budgeted_conditions": list(budgeted_conditions or []),
        "budget_contract_label": budget_contract_label,
        "provider_options": {
            "openai": {
                "api_mode": openai_api_mode,
                "reasoning_effort": openai_reasoning_effort,
                "max_output_tokens": openai_max_output_tokens,
            },
            "anthropic": {
                "temperature": anthropic_temperature,
                "max_tokens": anthropic_max_tokens,
                "thinking_budget_tokens": anthropic_thinking_budget_tokens,
            },
            "gemini": {
                "thinking_level": gemini_thinking_level,
                "max_output_tokens": gemini_max_output_tokens,
                "response_mime_type": gemini_response_mime_type,
                "timeout_seconds": gemini_timeout_seconds,
            },
        },
        "budget_tolerance": float(budget_tolerance),
        "stages": {},
    }

    reproduction_contract = {
        "schema_version": "llm_run_reproduction_contract_v1",
        "backend_spec": str(backend_spec),
        "information_condition": str(information_condition),
        "conditions": list(conditions),
        "modes": list(modes),
        "seed": int(seed),
        "reference_draw_seed": int(
            reference_draw_seed if reference_draw_seed is not None else seed
        ),
        "candidate_library_quantile": int(candidate_library_quantile),
        "row_id_file": str(row_id_file) if row_id_file is not None else None,
        "sample_size": int(sample_size) if sample_size is not None else None,
        "sample_seed": int(sample_seed),
        "sample_strata": list(sample_strata or []),
        "freeform_l1_budget": (
            float(freeform_l1_budget) if freeform_l1_budget is not None else None
        ),
        "budgeted_conditions": list(budgeted_conditions or []),
        "budget_contract_label": budget_contract_label,
        "provider_options": {
            "openai": {
                "api_mode": openai_api_mode,
                "reasoning_effort": openai_reasoning_effort,
                "max_output_tokens": openai_max_output_tokens,
            },
            "anthropic": {
                "temperature": anthropic_temperature,
                "max_tokens": anthropic_max_tokens,
                "thinking_budget_tokens": anthropic_thinking_budget_tokens,
            },
            "gemini": {
                "thinking_level": gemini_thinking_level,
                "max_output_tokens": gemini_max_output_tokens,
                "response_mime_type": gemini_response_mime_type,
                "timeout_seconds": gemini_timeout_seconds,
            },
        },
    }
    rollup["reproduction_contract"] = reproduction_contract

    if not skip_stage7:
        backend_kwargs: dict = {}
        if backend_spec == "scripted":
            backend_kwargs["seed"] = seed
        elif backend_spec.strip().lower().startswith("openai:"):
            if openai_api_mode is not None:
                backend_kwargs["api_mode"] = openai_api_mode
            if openai_reasoning_effort is not None:
                backend_kwargs["reasoning_effort"] = openai_reasoning_effort
            if openai_max_output_tokens is not None:
                backend_kwargs["max_output_tokens"] = int(openai_max_output_tokens)
        elif backend_spec.strip().lower().startswith("anthropic:"):
            if anthropic_temperature is not None:
                backend_kwargs["temperature"] = float(anthropic_temperature)
            if anthropic_max_tokens is not None:
                backend_kwargs["max_tokens"] = int(anthropic_max_tokens)
            if anthropic_thinking_budget_tokens is not None:
                backend_kwargs["thinking_budget_tokens"] = int(anthropic_thinking_budget_tokens)
        elif backend_spec.strip().lower().startswith("gemini:"):
            if gemini_thinking_level is not None:
                backend_kwargs["thinking_level"] = str(gemini_thinking_level)
            if gemini_max_output_tokens is not None:
                backend_kwargs["max_output_tokens"] = int(gemini_max_output_tokens)
            if gemini_response_mime_type is not None:
                backend_kwargs["response_mime_type"] = str(gemini_response_mime_type)
            if gemini_timeout_seconds is not None:
                backend_kwargs["timeout_seconds"] = float(gemini_timeout_seconds)
        s7_meta = run_stage7(
            project_root=project_root,
            backend_spec=backend_spec,
            information_condition=information_condition,
            conditions=conditions,
            modes=modes,
            reference_draw_seed=int(reference_draw_seed if reference_draw_seed is not None else seed),
            candidate_library_quantile=int(candidate_library_quantile),
            backend_kwargs=backend_kwargs,
            max_concurrency=int(max_concurrency),
            resume=bool(resume_stage7),
            checkpoint_path=checkpoint_path,
            max_retries=int(max_retries),
            retry_sleep_seconds=float(retry_sleep_seconds),
            icc_probe=bool(icc_probe),
            firm_name_lookup_path=firm_name_lookup_path,
            icc_probe_tolerance=float(icc_probe_tolerance),
            row_id_file=row_id_file,
            sample_size=sample_size,
            sample_seed=int(sample_seed),
            sample_strata=list(sample_strata or []),
            freeform_l1_budget=freeform_l1_budget,
            budgeted_conditions=list(budgeted_conditions or []),
            budget_contract_label=budget_contract_label,
            budget_tolerance=float(budget_tolerance),
        )
        rollup["stages"]["stage7"] = {"metadata": s7_meta}
        rollup["stages"]["stage7"]["verifier"] = _verify_or_fail(project_root, "stage7")

    if not skip_stage8:
        s8_meta = run_stage8(project_root=project_root)
        rollup["stages"]["stage8"] = {"metadata": s8_meta}
        rollup["stages"]["stage8"]["verifier"] = _verify_or_fail(project_root, "stage8")

    if not skip_stage9:
        s9_meta = run_stage9(project_root=project_root)
        rollup["stages"]["stage9"] = {"metadata": s9_meta}
        rollup["stages"]["stage9"]["verifier"] = _verify_or_fail(project_root, "stage9")

    rollup["final_paper_run_allowed"] = bool(
        rollup["stages"].get("stage9", {}).get("metadata", {}).get("final_paper_run_allowed")
    )
    if run_label:
        rollup["archive"] = _archive_llm_run(
            project_root=project_root,
            run_label=str(run_label),
            run_role=str(run_role),
            skip_stage7=skip_stage7,
            skip_stage8=skip_stage8,
            skip_stage9=skip_stage9,
            reproduction_contract=reproduction_contract,
        )

    if run_inference:
        inference_runs_dir = Path(llm_runs_dir).resolve() if llm_runs_dir is not None else (final_root(project_root) / "llm_runs")
        inf = run_stage9_inference(
            project_root,
            mde_freeze=bool(mde_freeze),
            llm_runs_dir=inference_runs_dir if inference_runs_dir.exists() else None,
            out_dir=final_root(project_root) / "stage9_statistical_inference",
        )
        rollup["stages"]["stage9_statistical_inference"] = {"metadata": inf}

    rollup["completed_utc"] = _now()
    write_json(final_root(project_root) / "ledgers" / "run_llm_stages_summary.json", rollup)
    return rollup


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sequential Stage 7→8→9 runner")
    ap.add_argument("--project-root", required=True)
    ap.add_argument(
        "--backend",
        default="scripted",
        help="Backend spec: 'scripted', 'openai:<model>', 'anthropic:<model>', or 'gemini:<model>'.",
    )
    ap.add_argument(
        "--information-condition",
        default="IC-a",
        choices=["IC-a", "IC-b", "IC-c"],
    )
    ap.add_argument(
        "--conditions",
        default="C4,C5,C6,C6X,C7,C8",
        help="Comma-separated subset of {C4,C4R,C5,C6,C6X,C7,C8}.",
    )
    ap.add_argument(
        "--modes",
        default="candidate_selection,free_form_10d",
        help="Comma-separated subset of {candidate_selection, free_form_10d}.",
    )
    ap.add_argument("--seed", type=int, default=20260523)
    ap.add_argument("--reference-draw-seed", type=int, default=None)
    ap.add_argument(
        "--candidate-library-quantile",
        type=int,
        default=50,
        choices=[50, 65, 75, 85],
        help="Stage2 materialized candidate-library quantile for Stage7/8/9 LLM action vectors; default P50 aligns LLM with RL.",
    )
    ap.add_argument("--max-concurrency", type=int, default=1)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    ap.add_argument("--no-resume-stage7", action="store_true")
    ap.add_argument("--checkpoint-path", default=None)
    ap.add_argument(
        "--openai-api-mode",
        choices=["chat", "responses"],
        default=None,
        help=(
            "OpenAI API mode.  Default is legacy chat unless "
            "--openai-reasoning-effort is supplied, in which case the "
            "OpenAI backend uses Responses API."
        ),
    )
    ap.add_argument(
        "--openai-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=None,
        help="Enable OpenAI Responses API reasoning with the selected effort.",
    )
    ap.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=None,
        help="Optional Responses API max_output_tokens budget for reasoning/output tokens.",
    )
    ap.add_argument("--anthropic-max-tokens", type=int, default=None,
                    help="Anthropic max_tokens budget; default preserves the backend default (1024).")
    ap.add_argument("--anthropic-temperature", type=float, default=None,
                    help="Anthropic sampling temperature; omit when extended thinking is enabled.")
    ap.add_argument("--anthropic-thinking-budget-tokens", type=int, default=None,
                    help="Enable Anthropic manual extended thinking with this token budget; must be >=1024 and < --anthropic-max-tokens.")
    ap.add_argument(
        "--gemini-thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default=None,
        help="Gemini 3.x thinking level; journal amendment uses low.",
    )
    ap.add_argument(
        "--gemini-max-output-tokens",
        type=int,
        default=None,
        help="Gemini generateContent maxOutputTokens.",
    )
    ap.add_argument(
        "--gemini-response-mime-type",
        choices=["application/json", "text/plain"],
        default=None,
        help="Gemini generateContent response MIME type; journal amendment uses application/json.",
    )
    ap.add_argument(
        "--gemini-timeout-seconds",
        type=float,
        default=None,
        help="Per-request Gemini HTTP timeout in seconds; outer Stage7 retry policy remains authoritative.",
    )
    ap.add_argument("--icc-probe", action="store_true",
                    help="Run the IC-c prior-knowledge probe after generation (LLM789-009). IC-c only.")
    ap.add_argument("--firm-name-lookup", default=None,
                    help="Explicit firm_id→회사명 lookup (parquet/csv) for IC-c firm-name exposure (LLM789-008).")
    ap.add_argument("--icc-probe-tolerance", type=float, default=0.20,
                    help="Pre-registered relative tolerance for the v2 numeric-recall contamination flag (LLM789-009 v2).")
    ap.add_argument("--run-label", default=None,
                    help="Archive this run under data/final_freeze/llm_runs/<label>/ after verifiers pass (LLM789-010).")
    ap.add_argument("--run-role", default="unclassified_llm_run",
                    help="Stable reproduction role recorded in the archive manifest; analysis selects runs by this role, not by timestamps.")
    ap.add_argument("--skip-stage7", action="store_true")
    ap.add_argument("--skip-stage8", action="store_true")
    ap.add_argument("--skip-stage9", action="store_true")
    ap.add_argument("--run-inference", action="store_true",
                    help="Run Stage9 statistical inference after Stage9/archive, producing H1-H5/MDE tables.")
    ap.add_argument("--llm-runs-dir", default=None,
                    help="Optional archived LLM runs directory for H5; default data/final_freeze/llm_runs.")
    ap.add_argument("--mde-freeze", action="store_true",
                    help="Freeze/update Stage9 MDE table during inference.")
    ap.add_argument("--row-id-file", default=None,
                    help="Optional pre-registered row_id subset file (csv/tsv/txt/parquet) for pilot runs such as N5.")
    ap.add_argument("--sample-size", type=int, default=None,
                    help="Optional deterministic row sample size when no --row-id-file is provided.")
    ap.add_argument("--sample-seed", type=int, default=20260707,
                    help="Seed for deterministic Stage7 row sampling; default 20260707.")
    ap.add_argument("--sample-strata", default=None,
                    help="Comma-separated panel columns for proportional stratified sampling, e.g. sector_7,grade_base_10.")
    ap.add_argument("--freeform-l1-budget", type=float, default=None,
                    help="N5-style generation-time L1 budget for free_form_10d action vectors.")
    ap.add_argument("--budgeted-conditions", default="C6",
                    help="Comma-separated policy conditions receiving --freeform-l1-budget; default C6.")
    ap.add_argument("--budget-contract-label", default=None,
                    help="Directory-safe label recorded in prompt/metadata/audit for the budget contract.")
    ap.add_argument("--budget-tolerance", type=float, default=1.0e-9,
                    help="Numerical tolerance for budget-compliance audit; default 1e-9.")
    args = ap.parse_args(argv)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    sample_strata = [c.strip() for c in (args.sample_strata or "").split(",") if c.strip()]
    budgeted_conditions = [c.strip() for c in (args.budgeted_conditions or "").split(",") if c.strip()]
    try:
        result = run_llm_stages(
            project_root=Path(args.project_root),
            backend_spec=args.backend,
            information_condition=args.information_condition,
            conditions=conditions,
            modes=modes,
            seed=args.seed,
            reference_draw_seed=(args.reference_draw_seed if args.reference_draw_seed is not None else args.seed),
            candidate_library_quantile=int(args.candidate_library_quantile),
            max_concurrency=args.max_concurrency,
            max_retries=args.max_retries,
            retry_sleep_seconds=args.retry_sleep_seconds,
            resume_stage7=not args.no_resume_stage7,
            checkpoint_path=(Path(args.checkpoint_path) if args.checkpoint_path else None),
            openai_api_mode=args.openai_api_mode,
            openai_reasoning_effort=args.openai_reasoning_effort,
            openai_max_output_tokens=args.openai_max_output_tokens,
            anthropic_temperature=args.anthropic_temperature,
            anthropic_max_tokens=args.anthropic_max_tokens,
            anthropic_thinking_budget_tokens=args.anthropic_thinking_budget_tokens,
            gemini_thinking_level=args.gemini_thinking_level,
            gemini_max_output_tokens=args.gemini_max_output_tokens,
            gemini_response_mime_type=args.gemini_response_mime_type,
            gemini_timeout_seconds=args.gemini_timeout_seconds,
            icc_probe=bool(args.icc_probe),
            firm_name_lookup_path=(Path(args.firm_name_lookup) if args.firm_name_lookup else None),
            icc_probe_tolerance=float(args.icc_probe_tolerance),
            run_label=args.run_label,
            run_role=args.run_role,
            skip_stage7=args.skip_stage7,
            skip_stage8=args.skip_stage8,
            skip_stage9=args.skip_stage9,
            run_inference=bool(args.run_inference),
            llm_runs_dir=(Path(args.llm_runs_dir) if args.llm_runs_dir else None),
            mde_freeze=bool(args.mde_freeze),
            row_id_file=(Path(args.row_id_file) if args.row_id_file else None),
            sample_size=args.sample_size,
            sample_seed=int(args.sample_seed),
            sample_strata=sample_strata,
            freeform_l1_budget=args.freeform_l1_budget,
            budgeted_conditions=budgeted_conditions,
            budget_contract_label=args.budget_contract_label,
            budget_tolerance=float(args.budget_tolerance),
        )
    except Exception as exc:
        print(f"LLM stages runner failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in result.items() if k != "stages"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
