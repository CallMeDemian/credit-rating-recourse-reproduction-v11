from __future__ import annotations

"""Immutable archive and catalog utilities for all LLM-originated runs.

Standard Stage7/8/9 grids, N5 runs, and IC-c probe-only runs share one archive
root: ``data/final_freeze/llm_runs``.  This module keeps their provenance and
catalog format consistent without duplicating filesystem/hash logic in runners.
"""

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.rl.common.io import write_json


ARCHIVE_STAGING_PREFIX = ".txn-"
WINDOWS_LEGACY_MAX_PATH = 259


def _new_staging_root(runs_root: Path) -> Path:
    """Return a short same-volume transaction directory for atomic archiving.

    The run label is deliberately excluded from the staging directory name.
    Long N5/frontier labels are retained in the immutable destination, but
    repeating them in both the transaction directory and checkpoint filename
    can exceed the legacy Windows MAX_PATH boundary during ``copytree``.
    """
    runs_root = Path(runs_root)
    for _ in range(32):
        candidate = runs_root / f"{ARCHIVE_STAGING_PREFIX}{uuid.uuid4().hex}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique LLM archive transaction directory under {runs_root}")


def archive_path_diagnostics(
    *,
    project_root: Path,
    run_label: str,
    artifact_dir_name: str,
    artifact_file_name: str,
) -> dict[str, Any]:
    """Preview destination and transaction path lengths for a named artifact."""
    runs_root = final_root(project_root) / "llm_runs"
    final_path = runs_root / run_label / artifact_dir_name / artifact_file_name
    transaction_path = (
        runs_root
        / f"{ARCHIVE_STAGING_PREFIX}{'0' * 32}"
        / artifact_dir_name
        / artifact_file_name
    )
    return {
        "schema_version": "llm_archive_path_diagnostics_v1",
        "staging_strategy": "short_same_volume_transaction_v1",
        "final_path": str(final_path),
        "final_path_length": len(str(final_path)),
        "transaction_path": str(transaction_path),
        "transaction_path_length": len(str(transaction_path)),
        "windows_legacy_max_path": WINDOWS_LEGACY_MAX_PATH,
        "final_path_within_legacy_limit": len(str(final_path)) <= WINDOWS_LEGACY_MAX_PATH,
        "transaction_path_within_legacy_limit": len(str(transaction_path)) <= WINDOWS_LEGACY_MAX_PATH,
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return obj if isinstance(obj, dict) else {}


def collect_file_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "archive_manifest.json"):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return rows


def write_archive_manifest(
    run_dir: Path,
    *,
    run_label: str,
    run_role: str,
    artifact_dirs: Iterable[str],
    run_type: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    files = collect_file_manifest(run_dir)
    manifest: dict[str, Any] = {
        "schema_version": "llm_run_archive_manifest_v2",
        "run_label": run_label,
        "run_role": run_role,
        "run_type": run_type,
        "created_utc": _now(),
        "artifact_dirs": list(artifact_dirs),
        "file_count": len(files),
        "files": files,
    }
    if extra:
        manifest["extra"] = extra
    write_json(run_dir / "archive_manifest.json", manifest)
    return manifest


def update_llm_runs_matrix_manifest(project_root: Path) -> dict[str, Any]:
    runs_root = final_root(project_root) / "llm_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        s7_meta = _read_json(run_dir / "stage7_llm_action_generation" / "metadata.json")
        s8_meta = _read_json(run_dir / "stage8_llm_multi_oracle_eval" / "metadata.json")
        s9_meta = _read_json(run_dir / "stage9_llm_rl_comparison" / "metadata.json")
        probe_meta = _read_json(run_dir / "stage7_icc_probe" / "icc_probe_summary.json")
        archive = _read_json(run_dir / "archive_manifest.json")
        archive_extra = archive.get("extra") if isinstance(archive.get("extra"), dict) else {}
        reproduction_contract = (
            archive_extra.get("reproduction_contract")
            if isinstance(archive_extra.get("reproduction_contract"), dict)
            else {}
        )
        info_condition = s7_meta.get("information_condition") or probe_meta.get("information_condition")
        rec = {
            "run_label": run_dir.name,
            "path": str(run_dir),
            "run_role": archive.get("run_role") or s7_meta.get("run_role") or probe_meta.get("run_role"),
            "run_type": archive.get("run_type") or ("icc_probe_only" if probe_meta else "stage7_8_9"),
            "information_condition": info_condition,
            "seed": (
                reproduction_contract.get("seed")
                if reproduction_contract.get("seed") is not None
                else s7_meta.get("seed")
            ),
            "reference_draw_seed": (
                reproduction_contract.get("reference_draw_seed")
                if reproduction_contract.get("reference_draw_seed") is not None
                else s7_meta.get("reference_draw_seed")
            ),
            "candidate_library_quantile": (
                reproduction_contract.get("candidate_library_quantile")
                if reproduction_contract.get("candidate_library_quantile") is not None
                else s7_meta.get("candidate_library_quantile")
            ),
            "backend_id": s7_meta.get("backend_id") or (probe_meta.get("backend") or {}).get("backend_id"),
            "backend_is_live": s7_meta.get("backend_is_live") if s7_meta else (probe_meta.get("backend") or {}).get("is_live"),
            "final_paper_run_allowed": bool(
                (s7_meta.get("final_paper_run_allowed") and s8_meta.get("final_paper_run_allowed") and s9_meta.get("final_paper_run_allowed"))
                if s7_meta else probe_meta.get("status") == "PASS"
            ),
            "stage7_status": s7_meta.get("status"),
            "stage8_status": s8_meta.get("status"),
            "stage9_status": s9_meta.get("status"),
            "probe_status": probe_meta.get("status"),
            "file_count": archive.get("file_count"),
        }
        records.append(rec)
    manifest = {
        "schema_version": "llm_runs_matrix_manifest_v2",
        "created_utc": _now(),
        "runs_root": str(runs_root),
        "run_count": len(records),
        "run_roles_found": sorted({str(r["run_role"]) for r in records if r.get("run_role")}),
        "information_conditions_found": sorted({str(r["information_condition"]) for r in records if r.get("information_condition")}),
        "runs": records,
    }
    write_json(runs_root / "llm_runs_matrix_manifest.json", manifest)
    return manifest


def archive_standard_stages(
    *,
    project_root: Path,
    run_label: str,
    run_role: str,
    skip_stage7: bool,
    skip_stage8: bool,
    skip_stage9: bool,
    reproduction_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not run_label or any(ch in run_label for ch in "/\\ \t"):
        raise ValueError(f"run_label must be a simple directory-safe token; got {run_label!r}")
    if not run_role or any(ch in run_role for ch in "/\\ \t"):
        raise ValueError(f"run_role must be a simple directory-safe token; got {run_role!r}")
    runs_root = final_root(project_root) / "llm_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    dest_root = runs_root / run_label
    if dest_root.exists():
        raise FileExistsError(
            f"Archive label already exists: {dest_root}. Archived runs are immutable; choose a new run label."
        )
    staging_root = _new_staging_root(runs_root)
    copied: list[str] = []
    try:
        staging_root.mkdir(parents=False, exist_ok=False)
        for key, skipped in (("stage7", skip_stage7), ("stage8", skip_stage8), ("stage9", skip_stage9)):
            if skipped:
                continue
            src = stage_dir(project_root, key)
            if not src.exists():
                raise FileNotFoundError(f"Cannot archive {key}: expected canonical directory {src}")
            dst = staging_root / src.name
            shutil.copytree(src, dst)
            copied.append(src.name)
        manifest = write_archive_manifest(
            staging_root,
            run_label=run_label,
            run_role=run_role,
            artifact_dirs=copied,
            run_type="stage7_8_9",
            extra=(
                {"reproduction_contract": dict(reproduction_contract)}
                if reproduction_contract is not None
                else None
            ),
        )
        staging_root.rename(dest_root)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root, ignore_errors=True)
        raise
    matrix = update_llm_runs_matrix_manifest(project_root)
    return {
        "run_label": run_label,
        "run_role": run_role,
        "path": str(dest_root),
        "copied_stage_dirs": copied,
        "file_count": manifest["file_count"],
        "manifest": str(dest_root / "archive_manifest.json"),
        "matrix_manifest": str(final_root(project_root) / "llm_runs" / "llm_runs_matrix_manifest.json"),
        "matrix_run_count": matrix["run_count"],
    }
