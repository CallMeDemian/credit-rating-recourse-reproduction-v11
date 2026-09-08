from __future__ import annotations

"""Stage7 IC-c probe-only runner.

This runner belongs to the LLM generation layer, not the post-freeze analysis
layer.  It reads one completed archived IC-c Stage7 run for provenance, calls
only the identity/numeric-recall probe (one API request per firm), and archives
its result under ``data/final_freeze/llm_runs/<label>/stage7_icc_probe``.
It never reconstructs or replays the 6,900 main C4/C5/C6/C6X/C7/C8 requests.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.contracts.paper_reproduction import (
    discover_archived_runs,
    load_profile,
    select_exact_role,
)
from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.rl.common.io import write_json
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.llm_backends import make_backend
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.pipeline import (
    _inject_firm_names,
    _load_firm_name_lookup,
    _run_icc_probe_pass,
)
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.prompt_builder import (
    ICC_PROBE_SCHEMA_VERSION,
    ICC_PROBE_TARGET_FEATURE,
)
from credit_recourse.utils.llm_run_archive import (
    sha256_file,
    update_llm_runs_matrix_manifest,
    write_archive_manifest,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return obj


def _validate_base_run(base_run: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    stage7 = base_run / "stage7_llm_action_generation"
    meta = _read_json(stage7 / "metadata.json")
    prompt = _read_json(stage7 / "llm_stage7_prompt_manifest.json")
    action_path = stage7 / "llm_stage7_action_table.parquet"
    if meta.get("information_condition") != "IC-c":
        raise ValueError(f"Probe base run must be IC-c; got {meta.get('information_condition')!r}")
    if int(meta.get("row_count", 0)) != 575 or int(meta.get("request_count", 0)) != 6900:
        raise ValueError(
            "Probe base run must be the full 575-firm/6,900-request IC-c grid; "
            f"got row_count={meta.get('row_count')}, request_count={meta.get('request_count')}"
        )
    if not action_path.exists():
        raise FileNotFoundError(f"Archived Stage7 action table missing: {action_path}")
    return stage7, meta, prompt


def run_probe_only(
    *,
    project_root: Path,
    run_label: str,
    run_role: str,
    base_run_label: str | None,
    base_run_role: str,
    tolerance: float,
    sample_size: int | None,
    sample_seed: int,
    max_retries: int,
    retry_sleep_seconds: float,
    firm_name_lookup: Path | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = load_profile(root)
    runs_root = final_root(root) / "llm_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    if base_run_label:
        base_run = runs_root / base_run_label
        if not base_run.is_dir():
            raise FileNotFoundError(f"Probe base run not found: {base_run}")
    else:
        records = discover_archived_runs(root, profile)
        base_run = select_exact_role(
            records,
            run_role=base_run_role,
            information_condition="IC-c",
        ).run_dir

    base_stage7, base_meta, base_prompt = _validate_base_run(base_run)
    backend_info = base_prompt.get("backend") or {}
    provider_options = backend_info.get("provider_options") or {}
    provider = str(provider_options.get("provider") or "")
    model = str(backend_info.get("model") or "")
    if provider != "openai" or not model:
        raise ValueError(f"Expected archived OpenAI base backend; provider={provider!r}, model={model!r}")
    backend_kwargs = {
        "api_mode": provider_options.get("api_mode"),
        "reasoning_effort": provider_options.get("reasoning_effort"),
        "max_output_tokens": provider_options.get("max_output_tokens"),
    }
    backend_kwargs = {k: v for k, v in backend_kwargs.items() if v is not None}
    backend = make_backend(f"openai:{model}", **backend_kwargs)

    state_panel_path = stage_dir(root, "stage2") / "phase_eval_candidate.parquet"
    if not state_panel_path.exists():
        raise FileNotFoundError(f"Stage2 serving panel missing: {state_panel_path}")
    panel = pd.read_parquet(state_panel_path)
    if "row_id" not in panel.columns:
        panel = panel.reset_index(drop=True).copy()
        panel["row_id"] = panel.index
    panel["row_id"] = pd.to_numeric(panel["row_id"], errors="raise").astype(int)
    if panel["row_id"].duplicated().any():
        raise ValueError("Stage2 serving panel contains duplicate row_id values")
    forbidden = [
        c
        for c in panel.columns
        if str(c).startswith("next__")
        or str(c).startswith("action__")
        or str(c) in {"reward_train", "reward_raw"}
    ]
    if forbidden:
        raise ValueError(f"Stage2 serving panel leaks forbidden columns: {forbidden}")

    base_ids = set(
        pd.to_numeric(
            pd.read_parquet(base_stage7 / "llm_stage7_action_table.parquet", columns=["row_id"])["row_id"],
            errors="raise",
        ).astype(int).unique()
    )
    panel_ids = set(panel["row_id"].tolist())
    if base_ids != panel_ids or len(base_ids) != 575:
        raise ValueError("Archived IC-c Stage7 row_id set does not match current Stage2 serving panel")

    lookup, lookup_meta = _load_firm_name_lookup(root, firm_name_lookup)
    panel, firm_name_meta = _inject_firm_names(panel, lookup, lookup_meta)
    if ICC_PROBE_TARGET_FEATURE not in panel.columns:
        raise ValueError(f"Probe target feature missing from Stage2 panel: {ICC_PROBE_TARGET_FEATURE}")

    if sample_size is not None:
        if sample_size <= 0 or sample_size > len(panel):
            raise ValueError(f"sample_size must be in [1,{len(panel)}], got {sample_size}")
        panel = (
            panel.sample(n=sample_size, random_state=sample_seed, replace=False)
            .sort_values("row_id")
            .reset_index(drop=True)
        )

    run_dir = runs_root / run_label
    archive_manifest = run_dir / "archive_manifest.json"
    if archive_manifest.exists():
        raise FileExistsError(f"Completed probe archive already exists: {run_dir}")
    probe_dir = run_dir / "stage7_icc_probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    probe_map, probe_meta = _run_icc_probe_pass(
        backend=backend,
        panel=panel,
        information_condition="IC-c",
        out_dir=probe_dir,
        resume=True,
        max_retries=max_retries,
        retry_sleep_seconds=retry_sleep_seconds,
        tolerance=tolerance,
    )
    rows = [{"row_id": int(rid), **probe_map[rid]} for rid in sorted(probe_map)]
    firm_path = probe_dir / "icc_probe_firm_level.csv"
    pd.DataFrame(rows).to_csv(firm_path, index=False, encoding="utf-8-sig")

    n = len(rows)
    summary = {
        "schema_version": "icc_probe_only_summary_v2",
        "created_utc": _now(),
        "status": "PASS",
        "run_label": run_label,
        "run_role": run_role,
        "run_type": "icc_probe_only",
        "execution_mode": "probe_only_from_archived_stage7",
        "information_condition": "IC-c",
        "base_run_label": base_run.name,
        "base_run_role": base_run_role,
        "base_stage7_dir": str(base_stage7),
        "base_stage7_metadata_sha256": sha256_file(base_stage7 / "metadata.json"),
        "base_stage7_action_table_sha256": sha256_file(base_stage7 / "llm_stage7_action_table.parquet"),
        "stage2_state_panel": str(state_panel_path),
        "stage2_state_panel_sha256": sha256_file(state_panel_path),
        "backend": backend.manifest(),
        "probe_schema_version": ICC_PROBE_SCHEMA_VERSION,
        "probe_target_feature": ICC_PROBE_TARGET_FEATURE,
        "probe_tolerance_relative": float(tolerance),
        "sample_size": sample_size,
        "sample_seed": sample_seed if sample_size is not None else None,
        "panel_row_count": n,
        "probe_row_count": n,
        "probe_resumed_count": int(probe_meta["icc_probe_resumed_count"]),
        "probe_parse_failures": int(probe_meta["icc_probe_parse_failures"]),
        "probe_recognized_count": int(probe_meta["icc_probe_recognized_count"]),
        "probe_recall_null_count": int(probe_meta["icc_probe_recall_null_count"]),
        "probe_panel_value_missing_count": int(probe_meta["icc_probe_panel_value_missing_count"]),
        "probe_contaminated_count": int(probe_meta["icc_probe_contaminated_count"]),
        "numeric_recall_count": n
        - int(probe_meta["icc_probe_recall_null_count"])
        - int(probe_meta["icc_probe_parse_failures"]),
        "firm_name_contract": firm_name_meta,
        "main_stage7_api_calls": 0,
        "main_checkpoint_used": False,
        "stage8_stage9_rerun": False,
        "interpretation_boundary": (
            "A close numeric recall is channel-level evidence under the identity-only prompt and tolerance. "
            "A negative result does not prove the complete absence of every form of pretrained firm knowledge."
        ),
    }
    write_json(probe_dir / "icc_probe_summary.json", summary)
    persisted_result = {
        "status": "PASS",
        "run_label": run_label,
        "run_role": run_role,
        "run_dir": str(run_dir),
        "probe_dir": str(probe_dir),
        "probe_row_count": n,
        "main_stage7_api_calls": 0,
        "archive_manifest": str(archive_manifest),
    }
    # Persist the run summary before computing the immutable file catalog so the
    # archive manifest covers every result file except itself.
    write_json(probe_dir / "run_summary.json", persisted_result)
    manifest = write_archive_manifest(
        run_dir,
        run_label=run_label,
        run_role=run_role,
        artifact_dirs=[probe_dir.name],
        run_type="icc_probe_only",
        extra={
            "base_run_label": base_run.name,
            "base_run_role": base_run_role,
            "probe_row_count": n,
            "main_stage7_api_calls": 0,
        },
    )
    matrix = update_llm_runs_matrix_manifest(root)
    return {
        **persisted_result,
        "archive_file_count": manifest["file_count"],
        "matrix_run_count": matrix["run_count"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--run-label", required=True)
    ap.add_argument("--run-role", default="paper_icc_probe")
    ap.add_argument("--base-run-label", default=None)
    ap.add_argument("--base-run-role", default="paper_primary_gpt54")
    ap.add_argument("--icc-probe-tolerance", type=float, default=0.20)
    ap.add_argument("--sample-size", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=20260710)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--retry-sleep-seconds", type=float, default=20.0)
    ap.add_argument("--firm-name-lookup", default=None)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_probe_only(
        project_root=Path(args.project_root),
        run_label=args.run_label,
        run_role=args.run_role,
        base_run_label=args.base_run_label,
        base_run_role=args.base_run_role,
        tolerance=float(args.icc_probe_tolerance),
        sample_size=args.sample_size,
        sample_seed=int(args.sample_seed),
        max_retries=int(args.max_retries),
        retry_sleep_seconds=float(args.retry_sleep_seconds),
        firm_name_lookup=Path(args.firm_name_lookup).resolve() if args.firm_name_lookup else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
