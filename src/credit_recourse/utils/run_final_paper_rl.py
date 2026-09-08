from __future__ import annotations

"""Sequential Stage 2A → 6 runner pinned to the frozen paper profile (RL-RUN-007).

This is the in-repo replacement for the external PowerShell paper runner: it
loads ``final_run_profile.yaml`` (the frozen 2026-06-06 configuration), builds
the exact per-stage argv, executes each stage module, and runs the matching
stage-boundary verifier.  Any sentinel value left in the profile hard-fails
before any stage executes (see ``rl.common.run_profile``).

Stage 2 reward defaults are aligned with the frozen paper profile as a defense
against accidental direct invocation.  The explicit profile remains the
authoritative full-pipeline contract and pins every stage argument.

``--dry-run`` prints every command without executing, so the argv set can be
diffed against the frozen PowerShell runner before first use.
"""

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from credit_recourse.contracts.stage_paths import final_root
from credit_recourse.rl.common.io import write_json
from credit_recourse.rl.common.run_profile import (
    FinalRunProfile,
    assert_profile_resolved,
    load_final_run_profile,
    stage_argv,
)
from credit_recourse.utils.materialize_final_freeze_configs import materialize
from credit_recourse.verification import stage_boundary_contracts as sbc


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# (stage_key_in_profile, python_module, verifier_keys_after)
STAGE_SEQUENCE: list[tuple[str, str, list[str]]] = [
    (
        "stage2_raw_action_source",
        "credit_recourse.rl.pipelines.final_stage2_raw_action_source_precompute.pipeline",
        [],
    ),
    (
        "stage2_input_splits",
        "credit_recourse.rl.pipelines.final_stage2_input_splits.pipeline",
        ["stage2_input"],
    ),
    (
        "stage2_candidate_projection",
        "credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline",
        [],
    ),
    (
        "stage2_counterfactual_transitions",
        "credit_recourse.rl.pipelines.final_stage2_counterfactual_transitions.pipeline",
        ["stage2"],
    ),
    (
        "stage3",
        "credit_recourse.rl.pipelines.final_stage3_acd_ssl.pipeline",
        ["stage3"],
    ),
    (
        "stage4",
        "credit_recourse.rl.pipelines.final_stage4_candidate_bc.pipeline",
        ["stage4"],
    ),
    (
        "stage5",
        "credit_recourse.rl.pipelines.final_stage5_candidate_iql.pipeline",
        ["stage5"],
    ),
    (
        "stage6_multi_oracle",
        "credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline",
        ["stage6_actions"],
    ),
    (
        "stage6_statistical_inference",
        "credit_recourse.eval.final_stage6_statistical_inference",
        [],
    ),
    (
        "stage6_derived_comparators",
        "credit_recourse.eval.final_stage6_derived_comparators",
        ["stage6"],
    ),
]

ALL_STAGE_KEYS = [k for k, _, _ in STAGE_SEQUENCE]


def _verify_or_fail(root: Path, stage: str) -> dict:
    result = sbc.verify(root, stage)
    ledger_path = final_root(root) / "ledgers" / f"verify_{stage}.json"
    write_json(ledger_path, result)
    if result["status"] != "PASS":
        raise RuntimeError(
            f"{stage} verifier failed.  Ledger: {ledger_path}.  "
            f"Errors: {result.get('errors', [])}"
        )
    return result


def run_final_paper_rl(
    *,
    project_root: Path,
    stages: list[str] | None = None,
    dry_run: bool = False,
    skip_verifiers: bool = False,
) -> dict:
    project_root = Path(project_root).resolve()
    requested = stages or list(ALL_STAGE_KEYS)
    unknown = sorted(set(requested) - set(ALL_STAGE_KEYS))
    if unknown:
        raise ValueError(f"Unknown stage keys: {unknown}. Valid: {ALL_STAGE_KEYS}")

    # 0) materialize embedded configs so the profile itself is on disk.
    mat = materialize(project_root, overwrite=False)

    profile: FinalRunProfile = load_final_run_profile(project_root)
    # Fail fast on any unresolved sentinel BEFORE any stage runs.
    assert_profile_resolved(profile, requested)

    ledger_dir = final_root(project_root) / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    rollup: dict = {
        "runner": "run_final_paper_rl",
        "created_utc": _now(),
        "project_root": str(project_root),
        "profile_name": profile.raw.get("profile_name"),
        "profile_path": str(profile.path),
        "profile_sha256": profile.sha256,
        "frozen_run_reference": profile.raw.get("frozen_run_reference"),
        "dry_run": bool(dry_run),
        "skip_verifiers": bool(skip_verifiers),
        "requested_stages": requested,
        "config_materialization": {
            "status": mat.get("status"),
            "files": len(mat.get("files", [])) if isinstance(mat.get("files"), list) else None,
        },
        "stages": {},
    }

    for stage_key, module_name, verifier_keys in STAGE_SEQUENCE:
        if stage_key not in requested:
            continue
        argv_profile, unresolved = stage_argv(profile, stage_key)
        assert not unresolved  # guaranteed by assert_profile_resolved above
        argv = ["--project-root", str(project_root)] + argv_profile
        command_repr = f"{sys.executable} -m {module_name} " + " ".join(argv)
        entry: dict = {"module": module_name, "argv": argv, "command": command_repr}
        if dry_run:
            entry["status"] = "DRY_RUN"
            print(command_repr)
        else:
            mod = importlib.import_module(module_name)
            rc = mod.main(argv)
            if rc not in (0, None):
                entry["status"] = "FAIL"
                rollup["stages"][stage_key] = entry
                write_json(ledger_dir / "run_final_paper_rl_summary.json", rollup)
                raise RuntimeError(f"{stage_key} ({module_name}) exited with code {rc}.")
            entry["status"] = "PASS"
            if not skip_verifiers:
                entry["verifiers"] = {}
                for vk in verifier_keys:
                    entry["verifiers"][vk] = _verify_or_fail(project_root, vk)["status"]
        rollup["stages"][stage_key] = entry

    rollup["completed_utc"] = _now()
    write_json(ledger_dir / "run_final_paper_rl_summary.json", rollup)
    return rollup


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Frozen paper-profile Stage 2A→6 runner (final_run_profile.yaml)"
    )
    ap.add_argument("--project-root", required=True)
    ap.add_argument(
        "--stages",
        default=",".join(ALL_STAGE_KEYS),
        help=f"Comma-separated subset of {ALL_STAGE_KEYS}",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact per-stage commands without executing. Diff these "
        "against the frozen PowerShell runner before first use.",
    )
    ap.add_argument("--skip-verifiers", action="store_true")
    args = ap.parse_args(argv)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    try:
        rollup = run_final_paper_rl(
            project_root=Path(args.project_root),
            stages=stages,
            dry_run=args.dry_run,
            skip_verifiers=args.skip_verifiers,
        )
    except Exception as exc:
        print(f"run_final_paper_rl failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: v for k, v in rollup.items() if k != "stages"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
