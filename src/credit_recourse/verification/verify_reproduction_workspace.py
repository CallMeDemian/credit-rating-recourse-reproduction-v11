from __future__ import annotations

"""Validate a fresh thesis reproduction workspace and freeze raw-input provenance.

The intended workspace contains only ``src/``, ``tools/``, and ``data/raw/``
before execution.  This verifier confirms that contract, checks the required raw
layout without silently substituting alternate paths, and writes a SHA-256
manifest under ``data/reproduction/``.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit_recourse.contracts.paper_reproduction import load_profile, source_profile_path


REQUIRED_TOOLS = (
    "setup_env.ps1",
    "run_thesis_repro.ps1",
    "run_oracle_stage0_stage1.ps1",
    "run_rl_unified_stage3456.ps1",
    "run_loopA_loopB2_stage2_extension.ps1",
    "run_llm789_fresh_all_single_repo.ps1",
    "run_n5_main_gpt54mini_icab.ps1",
    "run_n5_budget_frontier_icb.ps1",
    "run_postfreeze_analysis.ps1",
    "README_REPRO_TOOLS.md",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_workspace(project_root: Path, *, write_manifest: bool = True) -> dict[str, Any]:
    root = Path(project_root).resolve()
    profile = load_profile(root)
    errors: list[str] = []
    warnings: list[str] = []

    src_dir = root / profile["workspace"]["required_source_dir"]
    tools_dir = root / profile["workspace"]["required_tools_dir"]
    raw_root = root / profile["workspace"]["raw_root"]
    if not src_dir.is_dir():
        errors.append(f"missing source package directory: {src_dir}")
    if not tools_dir.is_dir():
        errors.append(f"missing tools directory: {tools_dir}")
    if not raw_root.is_dir():
        errors.append(f"missing raw data root: {raw_root}")

    for name in REQUIRED_TOOLS:
        p = tools_dir / name
        if not p.is_file():
            errors.append(f"missing required tool: {p}")

    raw_contract = profile["raw_contract"]
    for rel in raw_contract["required_directories"]:
        p = root / rel
        if not p.is_dir():
            errors.append(f"missing required raw directory: {p}")
    for rel in raw_contract.get("optional_directories", []):
        p = root / rel
        if not p.is_dir():
            warnings.append(f"optional raw directory absent: {p}")
    for rel, minimum in raw_contract["minimum_xlsx_counts"].items():
        p = root / rel
        count = len(list(p.rglob("*.xlsx"))) if p.exists() else 0
        if count < int(minimum):
            errors.append(f"raw workbook count below contract: {p} has {count}, requires >= {minimum}")

    raw_files: list[dict[str, Any]] = []
    if raw_root.exists():
        for path in sorted(p for p in raw_root.rglob("*") if p.is_file() and p.name != ".gitkeep"):
            raw_files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )
    if not raw_files:
        errors.append(f"no raw input files found under {raw_root}")

    source_files: list[dict[str, Any]] = []
    for base in (root / "src", tools_dir):
        if not base.exists():
            continue
        for path in sorted(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts and path_suffix_ok(p)):
            source_files.append(
                {
                    "relative_path": path.relative_to(root).as_posix(),
                    "size_bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
            )

    result: dict[str, Any] = {
        "schema_version": "reproduction_workspace_manifest_v1",
        "created_utc": _now(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "profile_path": str(source_profile_path(root)),
        "profile_name": profile.get("profile_name"),
        "raw_root": str(raw_root),
        "raw_file_count": len(raw_files),
        "raw_total_bytes": sum(int(x["size_bytes"]) for x in raw_files),
        "raw_files": raw_files,
        "source_tool_file_count": len(source_files),
        "source_tool_files": source_files,
        "errors": errors,
        "warnings": warnings,
    }
    if write_manifest:
        out = root / "data" / "reproduction" / "workspace_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["manifest_path"] = str(out)
    return result


def path_suffix_ok(path: Path) -> bool:
    return path.suffix.lower() not in {".pyc", ".pyo"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--no-write-manifest", action="store_true")
    args = ap.parse_args(argv)
    result = verify_workspace(Path(args.project_root), write_manifest=not args.no_write_manifest)
    if args.out_json:
        p = Path(args.out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
