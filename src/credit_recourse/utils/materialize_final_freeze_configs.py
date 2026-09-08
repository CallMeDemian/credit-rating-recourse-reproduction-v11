from __future__ import annotations

"""Materialize embedded final-freeze configuration files into ``data/final_freeze/configs``.

This module is intentionally small and fail-fast.  The active runners call it
before Stage0/Stage1/Stage2 so that runtime pipelines consume a single on-disk
configuration tree under ``data/final_freeze/configs`` while the repository keeps
versioned source-of-truth templates inside the Python package.
"""

import argparse
import hashlib
import json
import shutil
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    PACKAGE_ROOT / "final_freeze_configs" / "configs",
    PACKAGE_ROOT / "configs",
)
SKIP_DIR_NAMES = {"__pycache__", ".pytest_cache"}
SKIP_SUFFIXES = {".pyc", ".pyo"}
GENERATED_CONFIG_PATTERNS = ("*.generated.yaml", "*.final_freeze.generated.yaml")
QUARANTINED_ACTIVE_CONFIGS = {
    "final_candidate_library_calibrated.yaml": (
        PACKAGE_ROOT
        / "final_freeze_configs"
        / "configs"
        / "deprecated_nonbinding"
        / "final_candidate_library_calibrated.NONBINDING.yaml"
    ),
}

def is_generated_runtime_config(path: Path) -> bool:
    return any(fnmatch.fnmatch(path.name, pat) for pat in GENERATED_CONFIG_PATTERNS)


@dataclass(frozen=True)
class MaterializedFile:
    source: Path
    destination: Path
    relative_path: str
    sha256: str
    status: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_config_files(source_dirs: Iterable[Path]) -> list[tuple[Path, Path]]:
    files: dict[str, Path] = {}
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source_dir)
            if any(part in SKIP_DIR_NAMES for part in rel.parts):
                continue
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if is_generated_runtime_config(path):
                continue
            if rel.as_posix() in QUARANTINED_ACTIVE_CONFIGS:
                continue
            key = rel.as_posix()
            # Later source dirs intentionally override earlier ones.  This lets
            # credit_recourse/configs provide surgical current-code overrides
            # such as stage2_action_source_map.json while preserving the broader
            # final_freeze_configs tree.
            files[key] = path
    return [(Path(rel), src) for rel, src in sorted(files.items())]


def materialize(project_root: Path, *, overwrite: bool = False) -> dict:
    project_root = project_root.resolve()
    dest_root = project_root / "data" / "final_freeze" / "configs"
    dest_root.mkdir(parents=True, exist_ok=True)

    pairs = iter_config_files(SOURCE_DIRS)
    if not pairs:
        searched = [str(p) for p in SOURCE_DIRS]
        raise FileNotFoundError(
            "No embedded final-freeze config files found. Searched: " + json.dumps(searched, ensure_ascii=False)
        )

    quarantined_removed: list[dict[str, str]] = []
    for relative_path, quarantine_source in QUARANTINED_ACTIVE_CONFIGS.items():
        stale = dest_root / relative_path
        if not stale.exists():
            continue
        if not quarantine_source.exists():
            raise FileNotFoundError(
                f"Quarantine reference is missing for stale active config {stale}: {quarantine_source}"
            )
        stale_hash = sha256_file(stale)
        quarantine_hash = sha256_file(quarantine_source)
        if stale_hash != quarantine_hash:
            raise FileExistsError(
                "Refusing to remove a stale active config whose content differs from the "
                f"registered non-binding quarantine copy: {stale} "
                f"(active={stale_hash}, quarantine={quarantine_hash})"
            )
        stale.unlink()
        quarantined_removed.append({
            "relative_path": relative_path,
            "removed_path": str(stale),
            "sha256": stale_hash,
            "quarantine_source": str(quarantine_source),
            "status": "removed_stale_active_copy",
        })

    materialized: list[MaterializedFile] = []
    copied = 0
    skipped = 0
    overwritten = 0
    for rel, src in pairs:
        dst = dest_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        src_hash = sha256_file(src)
        status = "copied"
        if dst.exists():
            dst_hash = sha256_file(dst)
            if dst_hash == src_hash:
                status = "unchanged"
                skipped += 1
            elif overwrite:
                shutil.copy2(src, dst)
                status = "overwritten"
                copied += 1
                overwritten += 1
            else:
                raise FileExistsError(
                    f"Config already exists with different content: {dst}. "
                    "Re-run with --overwrite to intentionally refresh materialized configs."
                )
        else:
            shutil.copy2(src, dst)
            copied += 1
        materialized.append(MaterializedFile(src, dst, rel.as_posix(), src_hash, status))

    required = [
        "final_action_contract.yaml",
        "final_candidate_library.yaml",
        "final_oracle_rl_contract.json",
        "stage2_action_source_map.json",
        "temporal_split.yaml",
        "oracle_components/stage00_02/paths.yaml",
        "oracle_components/stage00_03/paths.yaml",
        "oracle_components/stage00_04/stage_config.yaml",
        "paper_reproduction_profile.json",
        "n5m_adaptive_selection_contract.json",
    ]
    missing = [rel for rel in required if not (dest_root / rel).exists()]
    if missing:
        raise FileNotFoundError("Materialized config tree is missing required files: " + ", ".join(missing))

    ledger = {
        "status": "PASS",
        "project_root": str(project_root),
        "destination_root": str(dest_root),
        "source_roots": [str(p) for p in SOURCE_DIRS if p.exists()],
        "copied_or_overwritten": copied,
        "overwritten": overwritten,
        "unchanged": skipped,
        "file_count": len(materialized),
        "quarantined_active_configs_removed": quarantined_removed,
        "files": [
            {
                "relative_path": m.relative_path,
                "source": str(m.source),
                "destination": str(m.destination),
                "sha256": m.sha256,
                "status": m.status,
            }
            for m in materialized
        ],
    }
    ledger_path = project_root / "data" / "final_freeze" / "ledgers" / "config_materialization_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    return ledger


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    ledger = materialize(args.project_root, overwrite=bool(args.overwrite))
    print(
        "[OK] materialized final-freeze configs: "
        f"files={ledger['file_count']} copied_or_overwritten={ledger['copied_or_overwritten']} "
        f"unchanged={ledger['unchanged']}"
    )
    print(f"[OK] ledger: {args.project_root / 'data' / 'final_freeze' / 'ledgers' / 'config_materialization_ledger.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
