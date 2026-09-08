from __future__ import annotations

import argparse
import csv
import fnmatch
import glob
import hashlib
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from .docx_inventory import (
    dumps_inventory,
    extract_thesis_inventory,
    sha256_file,
    validate_canonical_inventory,
)


ACCEPTED_RUN_STATUS = {
    "PASS",
    "PASSED",
    "PASS_WITH_DOCUMENTED_GAPS",
    "COMPLETE",
    "COMPLETED",
    "SUCCESS",
    "SUCCEEDED",
}
FROZEN_REPLAY_PROVENANCE = "PRESERVED_EVIDENCE/FROZEN_REPLAY_INPUT"
FORBIDDEN_COMPUTE_PARTS = {
    "frozen_outputs",
    "table_values",
    "thesis_printed_values",
    "04_paper_assets",
    "06_thesis_lineage",
}
DEFAULT_THESIS_NAME = "★ 석사학위논문_조종선_A67035_20260809_인사이트통합최종본.docx"
MAX_FILES_PER_ITEM = 80
MAX_ROWS_PER_SOURCE = 10000
MAX_PARQUET_PREVIEW_ROWS = 250
MAX_SOURCE_TABLE_CELLS = 120_000


@dataclass(frozen=True)
class SelectedRun:
    run_id: str
    manifest_path: Path
    manifest: dict[str, Any]
    status: str
    mode: str
    final_freeze_root: Path
    analysis_root: Path
    final_freeze_overlay_root: Path | None = None
    documented_gaps: tuple[str, ...] = ()

    @property
    def is_frozen_replay(self) -> bool:
        return _is_frozen_replay_mode(self.mode)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _first_string(payload: dict[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def resolve_thesis_docx(project_root: Path, explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("THESIS_DOCX"):
        candidates.append(Path(os.environ["THESIS_DOCX"]))
    candidates.extend(
        [
            project_root / "docs" / "thesis" / "canonical_thesis.docx",
            project_root / "docs" / "original" / DEFAULT_THESIS_NAME,
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Canonical thesis DOCX not found. Pass --thesis-docx or set THESIS_DOCX. "
        f"Tried: {[str(path) for path in candidates]}"
    )


def _run_manifest_candidates(run_dir: Path) -> list[Path]:
    preferred = [run_dir / "run_manifest.json", run_dir / "manifest.json"]
    found = [path for path in preferred if path.is_file()]
    if found:
        return found
    return sorted(
        (
            path
            for path in run_dir.rglob("*.json")
            if path.name.lower() in {"run_manifest.json", "manifest.json"}
        ),
        key=lambda path: (len(path.parts), path.as_posix().lower()),
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _manifest_status(payload: dict[str, Any]) -> str:
    direct = _first_string(payload, ("status", "run_status", "result", "state"))
    if direct:
        return direct.upper()
    verification = payload.get("verification")
    if isinstance(verification, dict):
        nested = _first_string(verification, ("status", "result", "state"))
        if nested:
            return nested.upper()
    stages = payload.get("stages")
    if isinstance(stages, list) and stages:
        stage_statuses = {
            str(stage.get("status", "")).upper()
            for stage in stages
            if isinstance(stage, dict) and stage.get("status")
        }
        if stage_statuses and stage_statuses.issubset(ACCEPTED_RUN_STATUS | {"SKIP", "SKIPPED"}):
            return "PASS"
    return "UNKNOWN"


def _manifest_mode(payload: dict[str, Any]) -> str:
    mode = _first_string(payload, ("mode", "run_mode", "reproduction_mode"))
    return mode or "UNKNOWN"


def _manifest_documented_gaps(payload: dict[str, Any]) -> tuple[str, ...]:
    for key in ("documented_gaps", "known_gaps", "gaps"):
        value = payload.get(key)
        if isinstance(value, list):
            return tuple(
                json.dumps(item, ensure_ascii=False, sort_keys=True)
                if isinstance(item, (dict, list))
                else str(item)
                for item in value
            )
        if isinstance(value, dict):
            return tuple(
                f"{name}={json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, (dict, list)) else item}"
                for name, item in sorted(value.items())
            )
        if isinstance(value, str) and value.strip():
            return (value.strip(),)
    return ()


def _is_frozen_replay_mode(mode: str) -> bool:
    """Accept cosmetic spelling differences without weakening mode isolation."""
    return re.sub(r"[^a-z0-9]", "", str(mode).lower()) == "frozenreplay"


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _find_root_value(payload: dict[str, Any], names: Sequence[str]) -> str:
    for container_name in ("output_roots", "roots", "paths", "outputs"):
        container = payload.get(container_name)
        if isinstance(container, dict):
            value = _first_string(container, names)
            if value:
                return value
    return _first_string(payload, names)


def _resolve_manifest_root(project_root: Path, value: str, fallback: Path) -> Path:
    if not value:
        return fallback.resolve()
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def select_run(project_root: Path, run_id: str | None) -> SelectedRun:
    runs_root = project_root / "data" / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(f"Run registry does not exist: {runs_root}")

    # The final kit deliberately has no V11 manifest framework.  Its public
    # runner writes one small human-readable RUN_INFO.txt after each real run.
    # Select that run and read the canonical working outputs directly.
    info_paths = (
        [runs_root / run_id / "RUN_INFO.txt"]
        if run_id
        else sorted(runs_root.glob("*/RUN_INFO.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
    )
    for info_path in info_paths:
        if not info_path.is_file():
            continue
        values: dict[str, str] = {}
        for line in info_path.read_text(encoding="utf-8-sig").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        if values.get("exit_status") != "0":
            continue
        selected_id = values.get("run_id") or info_path.parent.name
        mode = values.get("mode") or "UNKNOWN"
        manifest = {
            "run_id": selected_id,
            "mode": mode,
            "status": "PASS",
            "final_freeze": values.get("final_freeze", "data/final_freeze"),
            "analysis": values.get("analysis", "data/analysis"),
        }
        final_freeze_root = _resolve_manifest_root(
            project_root,
            manifest["final_freeze"],
            project_root / "data" / "final_freeze",
        )
        analysis_root = _resolve_manifest_root(
            project_root,
            manifest["analysis"],
            project_root / "data" / "analysis",
        )
        for label, root in (("final_freeze", final_freeze_root), ("analysis", analysis_root)):
            try:
                root.relative_to(project_root / "data")
            except ValueError as exc:
                raise RuntimeError(
                    f"Selected run {label} root is outside project data/: {root}"
                ) from exc
        return SelectedRun(
            run_id=selected_id,
            manifest_path=info_path.resolve(),
            manifest=manifest,
            status="PASS",
            mode=mode,
            final_freeze_root=final_freeze_root,
            analysis_root=analysis_root,
            final_freeze_overlay_root=None,
            documented_gaps=(),
        )

    if run_id:
        run_dirs = [runs_root / run_id]
        if not run_dirs[0].is_dir():
            # Permit one level of mode grouping, but never search outside data/runs.
            run_dirs = sorted(path for path in runs_root.glob(f"*/{run_id}") if path.is_dir())
    else:
        run_dirs = [path for path in runs_root.rglob("*") if path.is_dir()]

    candidates: list[tuple[float, Path, dict[str, Any], str]] = []
    rejected: list[str] = []
    for run_dir in run_dirs:
        for manifest_path in _run_manifest_candidates(run_dir):
            try:
                payload = _load_json_object(manifest_path)
            except Exception as exc:  # preserve the exact failure in the message
                rejected.append(f"{manifest_path}: unreadable ({exc})")
                continue
            status = _manifest_status(payload)
            if status not in ACCEPTED_RUN_STATUS:
                rejected.append(f"{manifest_path}: status={status}")
                continue
            candidates.append((manifest_path.stat().st_mtime, manifest_path, payload, status))
            break
    if not candidates:
        suffix = f" for run_id={run_id!r}" if run_id else ""
        detail = "; ".join(rejected[-8:]) or "no run manifest was found"
        raise RuntimeError(f"No successful selected-run manifest{suffix}: {detail}")
    _, manifest_path, manifest, status = max(candidates, key=lambda row: row[0])
    manifest_run_id = _first_string(manifest, ("run_id", "id")) or manifest_path.parent.name
    final_freeze_root = _resolve_manifest_root(
        project_root,
        _find_root_value(
            manifest,
            ("final_freeze", "final_freeze_root", "oracle_rl_root", "active_final_freeze"),
        ),
        project_root / "data" / "final_freeze",
    )
    analysis_root = _resolve_manifest_root(
        project_root,
        _find_root_value(manifest, ("analysis", "analysis_root", "active_analysis")),
        project_root / "data" / "analysis",
    )
    overlay_value = _find_root_value(
        manifest, ("final_freeze_overlay", "final_freeze_overlay_root")
    )
    final_freeze_overlay_root = (
        _resolve_manifest_root(project_root, overlay_value, project_root / "data" / "runs")
        if overlay_value
        else None
    )
    mode = _manifest_mode(manifest)
    frozen_replay_root = (project_root / "frozen_outputs" / "final_freeze").resolve()
    for label, root in (("final_freeze", final_freeze_root), ("analysis", analysis_root)):
        under_frozen = "frozen_outputs" in {part.lower() for part in root.parts}
        if under_frozen:
            permitted_frozen_input = (
                label == "final_freeze"
                and _is_frozen_replay_mode(mode)
                and _is_within(root, frozen_replay_root)
            )
            if not permitted_frozen_input:
                raise RuntimeError(
                    "Only mode=FrozenReplay may use frozen_outputs/final_freeze as the "
                    f"final_freeze input; selected {label} root is forbidden: {root}"
                )
            continue
        try:
            root.relative_to(project_root / "data")
        except ValueError as exc:
            raise RuntimeError(f"Selected run {label} root is outside project data/: {root}") from exc
    if final_freeze_overlay_root is not None:
        if "frozen_outputs" in {part.lower() for part in final_freeze_overlay_root.parts}:
            raise RuntimeError(
                f"Selected run final_freeze overlay cannot be frozen_outputs: {final_freeze_overlay_root}"
            )
        if not _is_within(final_freeze_overlay_root, project_root / "data" / "runs"):
            raise RuntimeError(
                f"Selected run final_freeze overlay must remain under data/runs: {final_freeze_overlay_root}"
            )
    return SelectedRun(
        run_id=manifest_run_id,
        manifest_path=manifest_path.resolve(),
        manifest=manifest,
        status=status,
        mode=mode,
        final_freeze_root=final_freeze_root,
        analysis_root=analysis_root,
        final_freeze_overlay_root=final_freeze_overlay_root,
        documented_gaps=_manifest_documented_gaps(manifest),
    )


def _forbidden_compute_path(path: Path, selected_run: SelectedRun | None = None) -> str:
    allow_frozen_replay_input = (
        selected_run is not None
        and selected_run.is_frozen_replay
        and _is_within(path, selected_run.final_freeze_root)
    )
    lowered_parts = {part.lower() for part in path.parts}
    for forbidden in FORBIDDEN_COMPUTE_PARTS:
        if forbidden in lowered_parts:
            if forbidden == "frozen_outputs" and allow_frozen_replay_input:
                # The selected root itself was strictly validated by select_run.
                continue
            return forbidden
    lowered = path.as_posix().lower()
    for forbidden in FORBIDDEN_COMPUTE_PARTS:
        if f"/{forbidden}/" in lowered:
            if forbidden == "frozen_outputs" and allow_frozen_replay_input:
                continue
            return forbidden
    return ""


def _map_patterns(pattern: str, project_root: Path, selected_run: SelectedRun) -> list[str]:
    normalized = pattern.replace("\\", "/")
    if normalized == "data/final_freeze":
        roots = [selected_run.final_freeze_overlay_root, selected_run.final_freeze_root]
        return [str(root) for root in roots if root is not None]
    if normalized.startswith("data/final_freeze/"):
        suffix = normalized[len("data/final_freeze/") :]
        roots = [selected_run.final_freeze_overlay_root, selected_run.final_freeze_root]
        return [str(root / Path(*suffix.split("/"))) for root in roots if root is not None]
    if normalized == "data/analysis":
        return [str(selected_run.analysis_root)]
    if normalized.startswith("data/analysis/"):
        suffix = normalized[len("data/analysis/") :]
        # ``paper_repro`` is the canonical repository-level directory name,
        # but v11 run manifests point ``analysis_root`` at the producer's
        # output directory itself.  In that run-local layout the first child
        # is ``00_manifest``/``01_substrate_validation`` rather than another
        # ``paper_repro`` directory.  Resolve the optional namespace by the
        # files that actually belong to the selected run; retain the nested
        # spelling for canonical/legacy clean layouts.
        if suffix == "paper_repro" or suffix.startswith("paper_repro/"):
            direct_suffix = suffix[len("paper_repro") :].lstrip("/")
            direct = (
                selected_run.analysis_root
                if not direct_suffix
                else selected_run.analysis_root / Path(*direct_suffix.split("/"))
            )
            nested = selected_run.analysis_root / Path(*suffix.split("/"))

            def has_file_match(candidate: Path) -> bool:
                return any(
                    Path(value).is_file()
                    for value in glob.iglob(str(candidate), recursive=True)
                )

            direct_has_files = has_file_match(direct)
            nested_has_files = has_file_match(nested)
            if direct_has_files and not nested_has_files:
                return [str(direct)]
            if nested_has_files:
                return [str(nested)]

            # Successful producer output always has a manifest directory.  It
            # also disambiguates a pattern whose particular optional artifact
            # is absent, so NOT_FOUND is reported against the right layout.
            if (selected_run.analysis_root / "00_manifest").is_dir():
                return [str(direct)]
            return [str(nested)]
        return [str(selected_run.analysis_root / Path(*suffix.split("/")))]
    return [str(project_root / Path(*normalized.split("/")))]


def _logical_source_path(
    path: Path, project_root: Path, selected_run: SelectedRun
) -> str:
    """Return the run-relative logical path used by selector contracts.

    A FrozenReplay overlay and the immutable frozen fallback can contain the
    same logical file.  Selectors must address that file as
    ``data/final_freeze/...`` regardless of which physical layer supplied it.
    """
    resolved = path.resolve()
    roots: list[tuple[Path | None, str]] = [
        (selected_run.final_freeze_overlay_root, "data/final_freeze"),
        (selected_run.final_freeze_root, "data/final_freeze"),
        (selected_run.analysis_root, "data/analysis"),
    ]
    for root, prefix in roots:
        if root is None or not _is_within(resolved, root):
            continue
        relative = resolved.relative_to(root.resolve()).as_posix()
        if (
            prefix == "data/analysis"
            and (root / "00_manifest" / "paper_repro_analysis_manifest.json").is_file()
        ):
            # Preserve the canonical selector namespace even when the selected
            # run's analysis_root is already the paper-reproduction output
            # directory.  This is manifest-driven rather than a hard-coded
            # chapter-directory whitelist, so canonical extensions such as
            # 05_extension_e3_e4 keep the same namespace.
            relative = f"paper_repro/{relative}"
        return f"{prefix}/{relative}" if relative else prefix
    return _relative(resolved, project_root)


def _portable_source_path(
    path: Path, project_root: Path, selected_run: SelectedRun
) -> str:
    """Return a reviewer-portable display path without changing computation.

    Build-time derived CSVs are embedded in the finished workbook and removed
    after publication. FrozenReplay working copies may later be moved under a
    run archive. Neither implementation detail should leave a dead host path
    in a reviewer-facing workbook.
    """
    relative = _relative(path, project_root).replace("\\", "/")
    build_prefix = f"data/thesis_outputs/_build/{selected_run.run_id}/"
    if relative.startswith(build_prefix):
        return "embedded://thesis-output-calculation/" + relative[len(build_prefix) :]

    if not selected_run.is_frozen_replay:
        return relative

    logical = _logical_source_path(path, project_root, selected_run).replace("\\", "/")
    frozen_candidates: list[tuple[str, str]] = [
        ("data/final_freeze", "frozen_outputs/final_freeze"),
        ("data/analysis", "frozen_outputs/analysis"),
    ]
    for logical_root, frozen_root in frozen_candidates:
        if logical != logical_root and not logical.startswith(logical_root + "/"):
            continue
        tail = logical[len(logical_root) :].lstrip("/")
        frozen_relative = f"{frozen_root}/{tail}" if tail else frozen_root
        frozen_path = project_root / Path(frozen_relative)
        if frozen_path.is_file() and sha256_file(frozen_path) == sha256_file(path):
            return frozen_relative
        return "embedded://selected-run-output/" + logical
    return relative


def _portable_embedded_value(
    value: Any, project_root: Path, selected_run: SelectedRun
) -> Any:
    """Remove machine-specific absolute paths from embedded source previews."""
    if not isinstance(value, str):
        return value

    project_prefix = project_root.resolve().as_posix().rstrip("/") + "/"

    def portable_line(line: str) -> str:
        normalized = line.replace("\\", "/")
        relative = ""
        if normalized.lower().startswith(project_prefix.lower()):
            relative = normalized[len(project_prefix) :]
        elif re.match(r"^[A-Za-z]:/", normalized):
            lowered = normalized.lower()
            for anchor in (
                "data/final_freeze/",
                "data/analysis/",
                "data/raw/",
                "src/",
                "contracts/",
                "docs/",
            ):
                marker = "/" + anchor
                index = lowered.find(marker)
                if index >= 0:
                    relative = normalized[index + 1 :]
                    break
        if not relative:
            return line

        build_prefix = f"data/thesis_outputs/_build/{selected_run.run_id}/"
        if relative.startswith(build_prefix):
            return "embedded://thesis-output-calculation/" + relative[len(build_prefix) :]
        if selected_run.is_frozen_replay:
            for logical_root, frozen_root in (
                ("data/final_freeze", "frozen_outputs/final_freeze"),
                ("data/analysis", "frozen_outputs/analysis"),
            ):
                if relative == logical_root or relative.startswith(logical_root + "/"):
                    tail = relative[len(logical_root) :].lstrip("/")
                    frozen_relative = f"{frozen_root}/{tail}" if tail else frozen_root
                    if (project_root / Path(frozen_relative)).exists():
                        return frozen_relative
                    return "embedded://historical-source/" + relative
        return relative

    return "\n".join(portable_line(line) for line in value.splitlines())


def resolve_sources(
    patterns: Sequence[str], project_root: Path, selected_run: SelectedRun
) -> tuple[list[Path], list[dict[str, str]]]:
    sources: list[Path] = []
    audit: list[dict[str, str]] = []
    for pattern in patterns:
        mapped_candidates = _map_patterns(pattern, project_root, selected_run)
        mapped = mapped_candidates[-1]
        layered_files: list[tuple[Path, str, int]] = []
        seen_logical: set[str] = set()
        for layer_index, candidate in enumerate(mapped_candidates):
            candidate_matches = [
                Path(value).resolve() for value in glob.glob(candidate, recursive=True)
            ]
            candidate_files = sorted(
                (path for path in candidate_matches if path.is_file()),
                key=lambda value: value.as_posix().lower(),
            )
            if candidate_files:
                mapped = candidate
            for path in candidate_files:
                logical_path = _logical_source_path(path, project_root, selected_run)
                if logical_path in seen_logical:
                    audit.append(
                        {
                            "pattern": pattern,
                            "mapped_pattern": candidate,
                            "status": "SHADOWED_BY_EARLIER_LAYER",
                            "path": str(path),
                            "logical_path": logical_path,
                            "reason": "overlay path has precedence for this exact logical relative path",
                        }
                    )
                    continue
                seen_logical.add(logical_path)
                layered_files.append((path, candidate, layer_index))
        if not layered_files:
            audit.append(
                {
                    "pattern": pattern,
                    "mapped_pattern": mapped,
                    "status": "NOT_FOUND",
                    "path": "",
                    "logical_path": "",
                    "reason": "",
                }
            )
            continue
        for path, candidate, layer_index in layered_files:
            reason = _forbidden_compute_path(path, selected_run)
            if reason:
                audit.append(
                    {
                        "pattern": pattern,
                        "mapped_pattern": candidate,
                        "status": "REJECTED_FORBIDDEN_COMPUTE_PARENT",
                        "path": str(path),
                        "logical_path": _logical_source_path(path, project_root, selected_run),
                        "reason": reason,
                    }
                )
                continue
            if path not in sources:
                sources.append(path)
                audit.append(
                    {
                        "pattern": pattern,
                        "mapped_pattern": candidate,
                        "status": (
                            "ACCEPTED_FROZEN_REPLAY_INPUT"
                            if "frozen_outputs" in {part.lower() for part in path.parts}
                            else "ACCEPTED"
                        ),
                        "path": str(path),
                        "logical_path": _logical_source_path(path, project_root, selected_run),
                        "reason": (
                            "overlay layer"
                            if layer_index == 0 and len(mapped_candidates) > 1
                            else "fallback layer for a logical path absent from overlay"
                            if layer_index > 0
                            else ""
                        ),
                    }
                )
            if len(sources) >= MAX_FILES_PER_ITEM:
                audit.append(
                    {
                        "pattern": pattern,
                        "mapped_pattern": mapped,
                        "status": "TRUNCATED_FILE_MATCHES",
                        "path": "",
                        "logical_path": "",
                        "reason": f"MAX_FILES_PER_ITEM={MAX_FILES_PER_ITEM}",
                    }
                )
                return sources, audit
    return sources, audit


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+", text) and not (len(text.lstrip("+-")) > 1 and text.lstrip("+-").startswith("0")):
        try:
            return int(text)
        except ValueError:
            pass
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[Ee][+-]?\d+)?", text):
        try:
            return float(text)
        except ValueError:
            pass
    return text


def _flatten_json(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten_json(nested, path)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _flatten_json(nested, f"{prefix}[{index}]")
    else:
        yield prefix or "$", value


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _source_table(path: Path) -> tuple[list[str], list[list[Any]], str, str]:
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter=delimiter)
                raw_rows = []
                truncated = False
                for index, row in enumerate(reader):
                    if index > MAX_ROWS_PER_SOURCE:
                        truncated = True
                        break
                    raw_rows.append(row)
            if not raw_rows:
                return ["empty"], [], "EMPTY", ""
            width = max(len(row) for row in raw_rows)
            header = [str(value).strip() or f"column_{i + 1}" for i, value in enumerate(raw_rows[0])]
            header.extend(f"column_{i + 1}" for i in range(len(header), width))
            rows = [
                [_coerce_scalar(value) for value in row + [""] * (width - len(row))]
                for row in raw_rows[1:]
            ]
            status = "TRUNCATED" if truncated else "OK"
            return header, rows, status, ""
        if suffix == ".json":
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            flat = list(_flatten_json(value))[:MAX_ROWS_PER_SOURCE]
            return ["json_path", "value"], [[key, _coerce_scalar(item)] for key, item in flat], "OK", ""
        if suffix in {".yaml", ".yml"}:
            value = _load_yaml(path)
            if value is not None:
                flat = list(_flatten_json(value))[:MAX_ROWS_PER_SOURCE]
                return ["config_path", "value"], [[key, _coerce_scalar(item)] for key, item in flat], "OK", ""
        if suffix == ".parquet":
            try:
                import pandas as pd  # type: ignore

                frame = pd.read_parquet(path)
                truncated = len(frame) > MAX_PARQUET_PREVIEW_ROWS
                frame = frame.head(MAX_PARQUET_PREVIEW_ROWS)
                header = [str(value) for value in frame.columns]
                rows = [
                    [_coerce_scalar(value) for value in row]
                    for row in frame.astype(object).where(frame.notna(), None).itertuples(index=False, name=None)
                ]
                return header, rows, "TRUNCATED" if truncated else "OK", ""
            except Exception as exc:
                return ["parquet_path"], [], "UNREADABLE_PARQUET", str(exc)
        if suffix == ".docx":
            return ["reference_path"], [], "REFERENCE_ONLY_NOT_READ_AS_SOURCE", ""
        text = path.read_text(encoding="utf-8-sig", errors="replace")
        lines = text.splitlines()
        rows = [[index + 1, line] for index, line in enumerate(lines[:MAX_ROWS_PER_SOURCE])]
        return ["line_number", "text"], rows, "TRUNCATED" if len(lines) > MAX_ROWS_PER_SOURCE else "OK", ""
    except Exception as exc:
        return ["error"], [], "READ_ERROR", str(exc)


def _source_role(path: Path, project_root: Path, selected_run: SelectedRun) -> str:
    if (
        selected_run.is_frozen_replay
        and "frozen_outputs" in {part.lower() for part in path.parts}
        and _is_within(path, selected_run.final_freeze_root)
    ):
        return FROZEN_REPLAY_PROVENANCE
    relative = _relative(path, project_root).lower()
    if relative.startswith(("src/", "contracts/", "docs/")):
        return "DESIGN_CONFIG_EVIDENCE"
    return "SELECTED_RUN_EMPIRICAL_EVIDENCE"


def _source_table_payload(
    path: Path, project_root: Path, selected_run: SelectedRun
) -> dict[str, Any]:
    header, rows, status, error = _source_table(path)
    # Keep the JSON payload bounded.  Source files remain available in full at
    # their hashed path even when the workbook preview is truncated.
    if len(header) * max(1, len(rows)) > MAX_SOURCE_TABLE_CELLS:
        limit = max(1, MAX_SOURCE_TABLE_CELLS // max(1, len(header)))
        rows = rows[:limit]
        status = "TRUNCATED"
    portable_header = [
        _portable_embedded_value(value, project_root, selected_run) for value in header
    ]
    portable_rows = [
        [_portable_embedded_value(value, project_root, selected_run) for value in row]
        for row in rows
    ]
    return {
        "path": _portable_source_path(path, project_root, selected_run),
        "logical_path": _logical_source_path(path, project_root, selected_run),
        "absolute_path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "source_role": _source_role(path, project_root, selected_run),
        "lineage_provenance": (
            FROZEN_REPLAY_PROVENANCE
            if "frozen_outputs" in {part.lower() for part in path.parts}
            else "SELECTED_RUN_OUTPUT_OR_DESIGN_EVIDENCE"
        ),
        "read_status": status,
        "read_error": error,
        "columns": portable_header,
        "rows": portable_rows,
        "row_count_in_workbook": len(portable_rows),
    }


def _selector_filter_matches(
    row: Sequence[Any], columns: Sequence[str], filters: Sequence[dict[str, Any]]
) -> bool:
    """Apply only contract-declared exact filters.

    Deliberately unsupported: substring, fuzzy, nearest-value, regex, and
    thesis-display-value matching.
    """
    for condition in filters:
        column = str(condition.get("column", ""))
        if column not in columns:
            return False
        index = columns.index(column)
        actual = row[index] if index < len(row) else None
        operator = str(condition.get("op", "equals"))
        if operator == "equals":
            if actual != _coerce_scalar(condition.get("value")):
                return False
        elif operator == "in":
            expected = [_coerce_scalar(value) for value in condition.get("values", [])]
            if actual not in expected:
                return False
        elif operator == "not_equals":
            if actual == _coerce_scalar(condition.get("value")):
                return False
        else:
            raise ValueError(f"Unsupported selector filter operator: {operator}")
    return True


def _selector_sort_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return (2, "")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (0, float(value))
    return (1, str(value))


def _build_selector_output(
    item_id: str,
    spec: dict[str, Any],
    source_tables: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selector_id = str(spec.get("selector_id", "")).strip()
    source_pattern = str(spec.get("source_pattern", "")).replace("\\", "/")
    raw_column_specs = spec.get("columns", [])
    column_specs = [
        {"name": str(column), "source": str(column), "kind": "source"}
        if isinstance(column, str)
        else column
        for column in raw_column_specs
    ] if isinstance(raw_column_specs, list) else []
    base = {
        "item_id": item_id,
        "selector_id": selector_id,
        "source_pattern": source_pattern,
        "filters": spec.get("filters", []),
        "row_keys": spec.get("row_keys", []),
        "value_columns": [str(value) for value in spec.get("value_columns", [])],
        "aggregation": str(spec.get("aggregation", "NONE")),
        "reshape": str(spec.get("reshape", "LONG")),
        "sort_by": spec.get("sort_by", []),
        "row_count": spec.get("row_count", {}),
        "actual_row_count": 0,
        "key_policy": str(spec.get("key_policy", "UNIQUE")),
        "expected_duplicate_count": spec.get("expected_duplicate_count", 0),
        "duplicate_key_count": 0,
        "source_paths": [],
        "columns": [],
        "rows": [],
        "cell_bindings": [],
    }
    if not selector_id or not source_pattern:
        return {**base, "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT", "reason": "selector_id and source_pattern are required"}
    if base["aggregation"] not in {
        "NONE_SOURCE_ROWS",
        "PRODUCER_AGGREGATE_NO_REAGGREGATION",
        "PRODUCER_MEAN_AFTER_SAME_FIRM_PAIRING_VERIFIED",
    }:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_UNSUPPORTED_AGGREGATION",
            "reason": f"unsupported aggregation {base['aggregation']}",
        }
    if base["key_policy"] not in {
        "UNIQUE",
        "ALLOW_DUPLICATES_ACROSS_SOURCE_FILES",
        "ALLOW_DUPLICATES",
    }:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": f"unsupported key_policy {base['key_policy']}",
        }
    if base["key_policy"] != "UNIQUE" and "expected_duplicate_count" not in spec:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": "an explicit expected_duplicate_count is required when duplicates are allowed",
        }
    matched = [
        (index, table)
        for index, table in enumerate(source_tables)
        if fnmatch.fnmatchcase(str(table.get("logical_path", "")), source_pattern)
    ]
    file_count = spec.get("file_count", {}) if isinstance(spec.get("file_count"), dict) else {}
    minimum = int(file_count.get("min", 1))
    maximum = int(file_count.get("max", minimum))
    if len(matched) < minimum or len(matched) > maximum:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_FILE_CARDINALITY",
            "reason": f"matched {len(matched)} files; expected {minimum}..{maximum}",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
        }
    if not isinstance(column_specs, list) or not column_specs:
        return {**base, "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT", "reason": "columns are required"}
    output_columns = [str(column.get("name", "")) for column in column_specs if isinstance(column, dict)]
    if len(output_columns) != len(column_specs) or any(not value for value in output_columns):
        return {**base, "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT", "reason": "every column needs a name"}
    if not base["value_columns"]:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": "value_columns must explicitly identify reproduced values",
        }
    missing_value_columns = sorted(set(base["value_columns"]) - set(output_columns))
    if missing_value_columns:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": f"value_columns are not selected columns: {missing_value_columns}",
        }
    selected: list[tuple[list[Any], list[dict[str, Any]]]] = []
    schema_errors: list[str] = []
    for table_index, table in matched:
        columns = [str(value) for value in table.get("columns", [])]
        required = {
            str(column.get("source", ""))
            for column in column_specs
            if isinstance(column, dict) and column.get("kind", "source") == "source"
        }
        filter_columns = {
            str(condition.get("column", ""))
            for condition in spec.get("filters", [])
            if isinstance(condition, dict)
        }
        missing = sorted((required | filter_columns) - set(columns))
        if missing:
            schema_errors.append(f"{table.get('logical_path', '')}: missing {missing}")
            continue
        for source_row_index, source_row in enumerate(table.get("rows", [])):
            if not isinstance(source_row, list):
                continue
            if not _selector_filter_matches(source_row, columns, spec.get("filters", [])):
                continue
            values: list[Any] = []
            bindings: list[dict[str, Any]] = []
            for column in column_specs:
                kind = str(column.get("kind", "source"))
                if kind == "logical_path":
                    values.append(str(table.get("logical_path", "")))
                    bindings.append(
                        {
                            "kind": "CONTRACT_METADATA",
                            "value": str(table.get("logical_path", "")),
                            "source_table_index": table_index,
                            "source_data_row_index": "",
                            "source_column_index": "",
                        }
                    )
                    continue
                if kind != "source":
                    raise ValueError(f"Unsupported selector column kind: {kind}")
                source_column = str(column.get("source", ""))
                source_column_index = columns.index(source_column)
                values.append(source_row[source_column_index] if source_column_index < len(source_row) else None)
                bindings.append(
                    {
                        "kind": "DIRECT_SOURCE_CELL",
                        "value": values[-1],
                        "source_table_index": table_index,
                        "source_data_row_index": source_row_index,
                        "source_column_index": source_column_index,
                        "source_path": table.get("path", ""),
                        "logical_path": table.get("logical_path", ""),
                        "source_row": source_row_index + 2,
                        "source_column": source_column,
                    }
                )
            selected.append((values, bindings))
    if schema_errors:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_SCHEMA_MISMATCH",
            "reason": "; ".join(schema_errors),
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
        }
    for sort_spec in reversed(spec.get("sort_by", [])):
        sort_column = str(sort_spec.get("column", ""))
        if sort_column not in output_columns:
            return {
                **base,
                "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
                "reason": f"sort column not selected: {sort_column}",
                "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
                "columns": output_columns,
            }
        sort_index = output_columns.index(sort_column)
        selected.sort(
            key=lambda pair: _selector_sort_key(pair[0][sort_index]),
            reverse=str(sort_spec.get("order", "asc")).lower() == "desc",
        )
    row_keys = [str(value) for value in spec.get("row_keys", [])]
    missing_row_keys = sorted(set(row_keys) - set(output_columns))
    if missing_row_keys:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": f"row_keys are not selected columns: {missing_row_keys}",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
        }
    overlapping_key_values = sorted(set(row_keys) & set(base["value_columns"]))
    if overlapping_key_values:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": f"row_keys and value_columns overlap: {overlapping_key_values}",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
        }
    row_count = spec.get("row_count")
    if not isinstance(row_count, dict):
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": "row_count must declare integer min/max cardinality",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
        }
    try:
        row_minimum = int(row_count["min"])
        row_maximum = int(row_count["max"])
    except (KeyError, TypeError, ValueError):
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": "row_count must declare integer min/max cardinality",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
        }
    if row_minimum < 0 or row_maximum < row_minimum:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": f"invalid row_count range {row_minimum}..{row_maximum}",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
        }
    if len(selected) < row_minimum or len(selected) > row_maximum:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_ROW_CARDINALITY",
            "reason": f"exact filters selected {len(selected)} rows; expected {row_minimum}..{row_maximum}",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
        }
    duplicate_key_count = 0
    if row_keys:
        key_indices = [output_columns.index(value) for value in row_keys]
        seen_keys: dict[tuple[Any, ...], set[str]] = {}
        for values, bindings in selected:
            key = tuple(values[index] for index in key_indices)
            if any(value is None or value == "" for value in key):
                return {
                    **base,
                    "status": "UNRESOLVED_SELECTOR_NULL_ROW_KEY",
                    "reason": f"null/blank row key for declared composite key {row_keys}: {key}",
                    "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
                    "columns": output_columns,
                }
            source_paths = {
                str(binding.get("logical_path", ""))
                for binding in bindings
                if binding.get("logical_path")
            }
            if key in seen_keys:
                duplicate_key_count += 1
                policy = str(spec.get("key_policy", "UNIQUE"))
                if policy == "ALLOW_DUPLICATES_ACROSS_SOURCE_FILES":
                    if seen_keys[key] & source_paths:
                        return {
                            **base,
                            "status": "UNRESOLVED_SELECTOR_DUPLICATE_ROW_KEY",
                            "reason": f"duplicate key within one source file for {row_keys}: {key}",
                            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
                            "columns": output_columns,
                            "duplicate_key_count": duplicate_key_count,
                        }
                elif policy != "ALLOW_DUPLICATES":
                    return {
                        **base,
                        "status": "UNRESOLVED_SELECTOR_DUPLICATE_ROW_KEY",
                        "reason": f"duplicate declared row key {row_keys}: {key}",
                        "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
                        "columns": output_columns,
                        "duplicate_key_count": duplicate_key_count,
                    }
                seen_keys[key].update(source_paths)
            else:
                seen_keys[key] = set(source_paths)
    try:
        expected_duplicate_count = int(spec.get("expected_duplicate_count", 0))
    except (TypeError, ValueError):
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_INVALID_CONTRACT",
            "reason": "expected_duplicate_count must be an integer",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
            "duplicate_key_count": duplicate_key_count,
        }
    if duplicate_key_count != expected_duplicate_count:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_DUPLICATE_CARDINALITY",
            "reason": (
                f"observed {duplicate_key_count} duplicate keys; "
                f"expected {expected_duplicate_count}"
            ),
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
            "actual_row_count": len(selected),
            "duplicate_key_count": duplicate_key_count,
        }
    max_rows = int(spec.get("max_rows", MAX_ROWS_PER_SOURCE))
    selected = selected[:max_rows]
    if not selected:
        return {
            **base,
            "status": "UNRESOLVED_SELECTOR_ZERO_ROWS",
            "reason": "exact contract filters selected zero rows",
            "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
            "columns": output_columns,
        }
    return {
        **base,
        "status": "SELECTOR_COMPUTED",
        "reason": "",
        "source_paths": [str(table.get("logical_path", "")) for _, table in matched],
        "columns": output_columns,
        "rows": [values for values, _ in selected],
        "cell_bindings": [bindings for _, bindings in selected],
        "actual_row_count": len(selected),
        "duplicate_key_count": duplicate_key_count,
    }


def _build_selector_outputs(
    item_id: str,
    selector_rule: dict[str, Any],
    source_tables: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    if str(selector_rule.get("status", "UNRESOLVED")) != "ACTIVE":
        return []
    selectors = selector_rule.get("selectors", [])
    if not isinstance(selectors, list):
        raise ValueError(f"Selector contract for {item_id} has non-list selectors")
    return [
        _build_selector_output(item_id, spec, source_tables)
        for spec in selectors
        if isinstance(spec, dict)
    ]


def _policy_value_checks(source_tables: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for table in source_tables:
        columns = [str(value) for value in table.get("columns", [])]
        rows = table.get("rows", [])
        if "policy" not in columns or not isinstance(rows, list):
            continue
        policy_index = columns.index("policy")
        no_op = next(
            (
                row
                for row in rows
                if isinstance(row, list)
                and len(row) > policy_index
                and "noop" in str(row[policy_index]).lower()
            ),
            None,
        )
        if not isinstance(no_op, list):
            continue
        for oracle in ("alpha", "beta", "gamma"):
            score_name = f"mean_R_score_{oracle}"
            delta_name = f"mean_delta_R_score_{oracle}"
            if score_name not in columns or delta_name not in columns:
                continue
            score_index = columns.index(score_name)
            delta_index = columns.index(delta_name)
            try:
                no_op_score = float(no_op[score_index])
            except (TypeError, ValueError, IndexError):
                continue
            for row_number, row in enumerate(rows, start=2):
                if not isinstance(row, list) or len(row) <= max(policy_index, score_index, delta_index):
                    continue
                try:
                    action_score = float(row[score_index])
                    reported_delta = float(row[delta_index])
                except (TypeError, ValueError):
                    continue
                recomputed = action_score - no_op_score
                checks.append(
                    {
                        "source_path": table.get("path", ""),
                        "source_row": row_number,
                        "policy": row[policy_index],
                        "oracle": oracle,
                        "action_score": action_score,
                        "same_firm_noop_mean_score": no_op_score,
                        "recomputed_delta": recomputed,
                        "reported_delta": reported_delta,
                        "absolute_error": abs(recomputed - reported_delta),
                        "check_status": (
                            "PASS_AGGREGATE_IDENTITY_ONLY"
                            if abs(recomputed - reported_delta) <= 1e-9
                            else "FAIL_AGGREGATE_IDENTITY"
                        ),
                        "check_scope": "AGGREGATE_IDENTITY_ONLY",
                        "pairing_verification": "SAME_FIRM_PAIRING_DEFINED_BY_PRODUCER; NOT_PROVEN_BY_SUMMARY_ROW",
                        "formula": "mean(action score_i - no-op score_i); aggregate only after same-firm pairing",
                    }
                )
    return checks


def _same_firm_row_level_checks(path: Path, project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the policy-value identity at the firm row before aggregation."""
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Row-level policy-value verification requires pandas and pyarrow from requirements.txt"
        ) from exc
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot read required row-level policy evidence {path}; install requirements.txt"
        ) from exc
    required = {"row_id", "policy"}
    missing_required = sorted(required - set(frame.columns))
    if missing_required:
        raise RuntimeError(f"Row-level policy evidence lacks {missing_required}: {path}")
    relative = _relative(path, project_root)
    is_stage8 = path.name == "llm_stage8_multi_oracle_scores.parquet"
    composite = ["row_id", "policy"] + (["mode"] if "mode" in frame.columns else [])
    duplicate_count = int(frame.duplicated(composite, keep=False).sum())
    missing_noop_count = 0
    duplicate_noop_count = 0
    if is_stage8:
        for oracle in ("alpha", "beta", "gamma"):
            expected = {f"R_score_{oracle}", f"noop_R_score_{oracle}", f"delta_R_score_{oracle}"}
            missing = sorted(expected - set(frame.columns))
            if missing:
                raise RuntimeError(f"Stage8 row-level evidence lacks {missing}: {path}")
        paired = frame.copy()
        pairing_method = "EXPLICIT_NOOP_SCORE_SAME_ROW"
    else:
        for oracle in ("alpha", "beta", "gamma"):
            expected = {f"R_score_{oracle}", f"delta_R_score_{oracle}"}
            missing = sorted(expected - set(frame.columns))
            if missing:
                raise RuntimeError(f"Stage6 row-level evidence lacks {missing}: {path}")
        noop_mask = frame["policy"].astype(str).str.lower().str.contains("noop", regex=False)
        noop_rows = frame.loc[noop_mask].copy()
        all_row_ids = set(frame["row_id"].tolist())
        noop_counts = noop_rows.groupby("row_id", dropna=False).size()
        missing_noop_count = len(all_row_ids - set(noop_counts.index.tolist()))
        duplicate_noop_count = int((noop_counts > 1).sum())
        if missing_noop_count or duplicate_noop_count:
            raise RuntimeError(
                "Same-firm no-op pairing failed for "
                f"{path}: missing_row_ids={missing_noop_count}, duplicate_row_ids={duplicate_noop_count}"
            )
        noop_columns = ["row_id"] + [f"R_score_{oracle}" for oracle in ("alpha", "beta", "gamma")]
        noop_rows = noop_rows[noop_columns].rename(
            columns={f"R_score_{oracle}": f"noop_R_score_{oracle}" for oracle in ("alpha", "beta", "gamma")}
        )
        paired = frame.merge(noop_rows, on="row_id", how="left", validate="many_to_one")
        pairing_method = "ROW_ID_TO_UNIQUE_NOOP_POLICY_JOIN"
    if duplicate_count:
        raise RuntimeError(
            f"Row-level policy evidence has {duplicate_count} duplicate composite-key rows {composite}: {path}"
        )
    row_checks: list[dict[str, Any]] = []
    failure_count = 0
    tolerance = 1e-9
    for dataframe_index, record in paired.iterrows():
        for oracle in ("alpha", "beta", "gamma"):
            try:
                action_score = float(record[f"R_score_{oracle}"])
                noop_score = float(record[f"noop_R_score_{oracle}"])
                reported_delta = float(record[f"delta_R_score_{oracle}"])
            except (TypeError, ValueError, KeyError):
                absolute_error = math.inf
                recomputed_delta = None
                check_status = "FAIL_SAME_FIRM_NON_NUMERIC_OR_MISSING"
            else:
                recomputed_delta = action_score - noop_score
                absolute_error = abs(recomputed_delta - reported_delta)
                check_status = (
                    "PASS_SAME_FIRM_ROW_IDENTITY"
                    if math.isfinite(absolute_error) and absolute_error <= tolerance
                    else "FAIL_SAME_FIRM_ROW_IDENTITY"
                )
            if check_status.startswith("FAIL"):
                failure_count += 1
            row_checks.append(
                {
                    "check_kind": "SAME_FIRM_ROW_LEVEL_IDENTITY",
                    "source_path": relative,
                    "source_row": int(dataframe_index) + 1 if isinstance(dataframe_index, int) else str(dataframe_index),
                    "row_id": record.get("row_id", ""),
                    "policy": record.get("policy", ""),
                    "mode": record.get("mode", ""),
                    "oracle": oracle,
                    "action_score": record.get(f"R_score_{oracle}", ""),
                    "same_firm_noop_score": record.get(f"noop_R_score_{oracle}", ""),
                    "recomputed_delta": recomputed_delta,
                    "reported_delta": record.get(f"delta_R_score_{oracle}", ""),
                    "absolute_error": absolute_error if math.isfinite(absolute_error) else "",
                    "check_status": check_status,
                    "check_scope": "ROW_LEVEL",
                    "pairing_verification": pairing_method,
                    "formula": f"R_score_{oracle}[row_id] - noop_R_score_{oracle}[same row_id]",
                }
            )
    if failure_count:
        raise RuntimeError(
            f"Same-firm policy-value identity failed for {failure_count} row-oracle cells: {path}"
        )
    summary = {
        "source_path": relative,
        "row_count": int(len(paired)),
        "oracle_check_count": len(row_checks),
        "duplicate_composite_key_rows": duplicate_count,
        "missing_noop_row_ids": missing_noop_count,
        "duplicate_noop_row_ids": duplicate_noop_count,
        "pairing_method": pairing_method,
        "status": "SAME_FIRM_PAIRING_VERIFIED",
        "tolerance": tolerance,
    }
    return summary, row_checks


_DISPLAY_NUMBER_RE = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[+-]?\.\d+")


def _display_token_spec(token: str) -> dict[str, Any]:
    token_text = (
        str(token)
        .replace("−", "-")
        .replace("–", "-")
        .replace("﹣", "-")
        .replace("＋", "+")
    )
    raw_components = _DISPLAY_NUMBER_RE.findall(token_text)
    values: list[Decimal] = []
    decimals: list[int] = []
    for raw in raw_components:
        normalized = raw.replace(",", "")
        try:
            values.append(Decimal(normalized))
        except InvalidOperation:
            continue
        decimals.append(len(normalized.partition(".")[2]))

    if "%" in token_text:
        unit = "PERCENT"
    elif "차원" in token_text:
        unit = "DIMENSION"
    elif "년" in token_text:
        unit = "YEAR"
    elif "개월" in token_text:
        unit = "MONTH"
    elif "개" in token_text:
        unit = "COUNT"
    elif "점" in token_text:
        unit = "POINT"
    else:
        unit = "UNSPECIFIED"
    return {
        "values": values,
        "decimals": decimals,
        "unit": unit,
        "rounding": "; ".join(
            f"component_{index + 1}=ROUND_HALF_UP({digits}dp)"
            for index, digits in enumerate(decimals)
        ),
    }


def _reference_cell_role(item_class: str, row_index: int, value: str) -> str:
    numeric = bool(re.search(r"\d", value or ""))
    if not numeric:
        return "LABEL"
    if row_index == 0:
        return "DESIGN_CONFIG"
    if item_class == "NON_EMPIRICAL":
        return "DESIGN_CONFIG"
    if item_class == "EMPIRICAL":
        return "COMPUTED"
    design_cues = ("조건", "정의", "상한", "모형", "단계", "기간", "설정", "변수")
    empirical_cues = ("평균", "차이", "비율", "p", "신뢰", "결과", "정책가치", "준수")
    if any(cue in value for cue in design_cues):
        return "DESIGN_CONFIG"
    if any(cue.lower() in value.lower() for cue in empirical_cues):
        return "COMPUTED"
    return "MIXED_REVIEW_REQUIRED"


def _item_status(
    item_class: str,
    calculation: str,
    source_tables: Sequence[dict[str, Any]],
    provenance: str,
) -> str:
    if provenance == "PRESERVED_EVIDENCE_PRODUCER_SNAPSHOT_GAP":
        return provenance
    readable = any(
        table.get("read_status") in {"OK", "TRUNCATED", "REFERENCE_ONLY_NOT_READ_AS_SOURCE"}
        for table in source_tables
    )
    if not readable:
        return "UNRESOLVED_NO_ALLOWED_SOURCE"
    if item_class == "NON_EMPIRICAL":
        return "DESIGN_CONFIG_EVIDENCE"
    if item_class == "MIXED":
        return "MIXED_CELL_OR_SERIES_ROLES"
    if calculation == "UNRESOLVED":
        return "UNRESOLVED_NO_ALLOWED_SOURCE"
    source_roles = {str(table.get("source_role", "")) for table in source_tables}
    if FROZEN_REPLAY_PROVENANCE in source_roles:
        if "SELECTED_RUN_EMPIRICAL_EVIDENCE" in source_roles:
            return "RECOMPUTED_DOWNSTREAM_FROM_FROZEN_REPLAY_INPUT"
        return FROZEN_REPLAY_PROVENANCE
    return "RECOMPUTED_FROM_SELECTED_RUN"


def _claim_rule_index(contract: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    """Expand explicit semantic claim rules into unique claim-token keys.

    The paragraph hash is deliberately part of every expanded rule and is
    verified at resolution time.  Neither the thesis token text nor its
    numeric value participates in this index.
    """
    if contract.get("schema_version") != "numeric_claim_selectors_v11":
        raise RuntimeError("Unsupported numeric claim selector contract")
    paragraph_hashes = contract.get("paragraph_text_sha256")
    rules = contract.get("rules")
    if not isinstance(paragraph_hashes, dict) or not isinstance(rules, list):
        raise RuntimeError("Numeric claim selector contract needs paragraph hashes and rules")
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for rule in rules:
        if not isinstance(rule, dict):
            raise RuntimeError("Numeric claim selector rules must be objects")
        rule_id = str(rule.get("rule_id", "")).strip()
        item_id = str(rule.get("item_id", "")).strip()
        selector_id = str(rule.get("selector_id", "")).strip()
        value_column = str(rule.get("value_column", "")).strip()
        formula = str(rule.get("formula", "")).strip()
        if not all((rule_id, item_id, selector_id, value_column, formula)):
            raise RuntimeError(f"Incomplete numeric claim selector rule: {rule_id or '<unnamed>'}")
        transform = str(rule.get("transform", "IDENTITY"))
        if transform not in {"IDENTITY", "MULTIPLY_100", "ABS"}:
            raise RuntimeError(f"Unsupported numeric claim transform {transform}: {rule_id}")
        try:
            decimals = int(rule.get("rounding_decimals", 0))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid rounding_decimals: {rule_id}") from exc
        if decimals < 0 or decimals > 12:
            raise RuntimeError(f"rounding_decimals outside 0..12: {rule_id}")
        targets = rule.get("targets")
        if not isinstance(targets, list) or not targets:
            raise RuntimeError(f"Numeric claim selector has no targets: {rule_id}")
        for target in targets:
            if not isinstance(target, dict):
                raise RuntimeError(f"Numeric claim selector target must be an object: {rule_id}")
            claim_id = str(target.get("claim_id", "")).strip()
            try:
                token_index = int(target.get("token_index"))
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Invalid token index in {rule_id}") from exc
            paragraph_hash = str(paragraph_hashes.get(claim_id, "")).strip().lower()
            if not re.fullmatch(r"[0-9a-f]{64}", paragraph_hash):
                raise RuntimeError(f"Missing paragraph hash for {claim_id}: {rule_id}")
            key = (claim_id, token_index)
            if key in index:
                raise RuntimeError(f"Duplicate numeric claim selector key: {claim_id}-{token_index:02d}")
            row_match = target.get("row_match", rule.get("row_match", {}))
            if not isinstance(row_match, dict) or not row_match:
                raise RuntimeError(f"Numeric claim selector needs exact row_match: {rule_id}")
            target_value_column = str(target.get("value_column", value_column)).strip()
            if not target_value_column:
                raise RuntimeError(f"Numeric claim selector target has no value column: {rule_id}")
            index[key] = {
                **rule,
                "claim_id": claim_id,
                "token_index": token_index,
                "paragraph_text_sha256": paragraph_hash,
                "row_match": row_match,
                "value_column": target_value_column,
            }
    return index


def _rounded_claim_value(value: Any, transform: str, decimals: int) -> int | float:
    try:
        numeric = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"selected source cell is not numeric: {value!r}") from exc
    if transform == "MULTIPLY_100":
        numeric *= Decimal("100")
    elif transform == "ABS":
        numeric = abs(numeric)
    elif transform != "IDENTITY":
        raise ValueError(f"unsupported transform: {transform}")
    quantum = Decimal(1).scaleb(-decimals)
    rounded = numeric.quantize(quantum, rounding=ROUND_HALF_UP)
    return int(rounded) if decimals == 0 else float(rounded)


def _unresolved_claim_rule(
    rule: dict[str, Any],
    item: dict[str, Any] | None,
    reason: str,
) -> dict[str, Any]:
    source_tables = list(item.get("source_tables", [])) if isinstance(item, dict) else []
    return {
        "computed_value": "",
        "unit": str(rule.get("unit", "UNSPECIFIED")),
        "rounding": f"ROUND_HALF_UP({int(rule.get('rounding_decimals', 0))}dp)",
        "formula": str(rule.get("formula", "")),
        "source_path": "; ".join(str(table.get("path", "")) for table in source_tables),
        "source_sha256": "; ".join(str(table.get("sha256", "")) for table in source_tables),
        "source_row": "",
        "source_column": str(rule.get("value_column", "")),
        "source_json_or_config_key": "; ".join(
            f"{key}={value}" for key, value in rule.get("row_match", {}).items()
        ),
        "source_role": "REGISTERED_EXPLICIT_SELECTOR_NOT_RESOLVED",
        "producer": str(item.get("producer", "")) if isinstance(item, dict) else "",
        "evidence_status": reason,
        "source_match_count": 0,
        "source_binding_kind": "",
        "source_data_row_index": "",
        "source_column_index": "",
        "source_table_index": "",
        "display_transform": str(rule.get("transform", "IDENTITY")),
        "resolution_rule": f"EXPLICIT_CLAIM_SELECTOR:{rule.get('rule_id', '')}",
    }


def _resolve_claim_rule(
    claim: dict[str, Any],
    rule: dict[str, Any],
    item_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    actual_hash = hashlib.sha256(str(claim.get("text", "")).encode("utf-8")).hexdigest()
    item = item_map.get(str(rule.get("item_id", "")))
    if actual_hash != str(rule.get("paragraph_text_sha256", "")):
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_PARAGRAPH_HASH_DRIFT")
    if item is None:
        return _unresolved_claim_rule(rule, None, "UNRESOLVED_CLAIM_ITEM_NOT_FOUND")
    selector_id = str(rule.get("selector_id", ""))
    outputs = [
        output
        for output in item.get("selector_outputs", [])
        if str(output.get("selector_id", "")) == selector_id
    ]
    if len(outputs) != 1:
        return _unresolved_claim_rule(
            rule, item, "UNRESOLVED_CLAIM_SELECTOR_MISSING_OR_DUPLICATE"
        )
    output = outputs[0]
    if output.get("status") != "SELECTOR_COMPUTED":
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_SELECTOR_NOT_COMPUTED")
    columns = [str(value) for value in output.get("columns", [])]
    row_match = dict(rule.get("row_match", {}))
    declared_row_keys = [str(value) for value in output.get("row_keys", [])]
    if list(row_match) != declared_row_keys:
        return _unresolved_claim_rule(
            rule, item, "UNRESOLVED_CLAIM_SELECTOR_ROW_KEY_CONTRACT_MISMATCH"
        )
    value_column = str(rule.get("value_column", ""))
    declared_value_columns = [str(value) for value in output.get("value_columns", [])]
    if value_column not in declared_value_columns:
        return _unresolved_claim_rule(
            rule, item, "UNRESOLVED_CLAIM_SELECTOR_VALUE_COLUMN_NOT_DECLARED"
        )
    missing_keys = sorted(set(row_match) - set(columns))
    if missing_keys or value_column not in columns:
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_SELECTOR_KEY_MISSING")
    matching_indices: list[int] = []
    for row_index, row in enumerate(output.get("rows", [])):
        if not isinstance(row, list):
            continue
        if all(
            (row[columns.index(key)] if columns.index(key) < len(row) else None)
            == _coerce_scalar(expected)
            for key, expected in row_match.items()
        ):
            matching_indices.append(row_index)
    if len(matching_indices) != 1:
        return _unresolved_claim_rule(
            rule,
            item,
            "UNRESOLVED_CLAIM_SELECTOR_ROW_MISSING"
            if not matching_indices
            else "UNRESOLVED_CLAIM_SELECTOR_ROW_DUPLICATE",
        )
    row_index = matching_indices[0]
    column_index = columns.index(value_column)
    row = output.get("rows", [])[row_index]
    if column_index >= len(row):
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_VALUE_CELL_MISSING")
    bindings = output.get("cell_bindings", [])
    if row_index >= len(bindings) or column_index >= len(bindings[row_index]):
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_BINDING_MISSING")
    binding = bindings[row_index][column_index]
    if not isinstance(binding, dict) or binding.get("kind") != "DIRECT_SOURCE_CELL":
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_BINDING_NOT_DIRECT")
    try:
        table_index = int(binding.get("source_table_index"))
        source_table = item.get("source_tables", [])[table_index]
    except (TypeError, ValueError, IndexError):
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_SOURCE_TABLE_MISSING")
    try:
        computed_value = _rounded_claim_value(
            row[column_index],
            str(rule.get("transform", "IDENTITY")),
            int(rule.get("rounding_decimals", 0)),
        )
    except ValueError:
        return _unresolved_claim_rule(rule, item, "UNRESOLVED_CLAIM_SOURCE_NOT_NUMERIC")
    source_role = str(source_table.get("source_role", ""))
    evidence_status = (
        "PRESERVED_EVIDENCE_EXACT_CELL"
        if source_role == FROZEN_REPLAY_PROVENANCE
        else "RECOMPUTED_EXACT_SELECTED_RUN_SOURCE_CELL"
    )
    return {
        "computed_value": computed_value,
        "unit": str(rule.get("unit", "UNSPECIFIED")),
        "rounding": f"ROUND_HALF_UP({int(rule.get('rounding_decimals', 0))}dp)",
        "formula": str(rule.get("formula", "")),
        "source_path": str(binding.get("source_path") or source_table.get("path", "")),
        "source_sha256": str(source_table.get("sha256", "")),
        "source_row": binding.get("source_row", ""),
        "source_column": str(binding.get("source_column", value_column)),
        "source_json_or_config_key": "; ".join(
            f"{key}={value}" for key, value in row_match.items()
        ),
        "source_role": source_role,
        "producer": str(item.get("producer", "")),
        "evidence_status": evidence_status,
        "source_match_count": 1,
        "source_binding_kind": str(binding.get("kind", "")),
        "source_data_row_index": binding.get("source_data_row_index", ""),
        "source_column_index": binding.get("source_column_index", ""),
        "source_table_index": table_index,
        "display_transform": (
            f"{rule.get('transform', 'IDENTITY')}; ROUND_HALF_UP({int(rule.get('rounding_decimals', 0))}dp)"
        ),
        "resolution_rule": f"EXPLICIT_CLAIM_SELECTOR:{rule.get('rule_id', '')}",
    }


def _claim_token_rows(
    claim: dict[str, Any],
    item_map: dict[str, dict[str, Any]],
    claim_rules: dict[tuple[str, int], dict[str, Any]],
    run_id: str,
    documented_gaps: Sequence[str],
) -> list[dict[str, Any]]:
    linked_ids = [str(value) for value in claim.get("nearest_item_ids", []) if value in item_map]
    linked = [item_map[value] for value in linked_ids]
    classes = {str(item.get("class", "")) for item in linked}
    force_non_empirical = (
        str(claim.get("classification_hint", "")) == "NON_EMPIRICAL"
        or bool(linked and classes == {"NON_EMPIRICAL"})
    )
    tokens = [str(value) for value in claim.get("numeric_tokens", [])]
    rows: list[dict[str, Any]] = []
    for token_index, token in enumerate(tokens, start=1):
        base = {
            "claim_token_id": f"{claim.get('claim_id', '')}-{token_index:02d}",
            "claim_id": claim.get("claim_id", ""),
            "token_index": token_index,
            "paragraph_index": claim.get("paragraph_index", ""),
            "pdf_page": claim.get("pdf_page", ""),
            "printed_page": claim.get("printed_page", ""),
            "section": claim.get("section", ""),
            "text": claim.get("text", ""),
            "classification_hint": claim.get("classification_hint", ""),
            "nearest_item_ids": linked_ids,
            "run_id": run_id,
            "run_documented_gaps": "; ".join(documented_gaps),
            "resolution_rule": "EXPLICIT_CLAIM_SELECTOR_ONLY; THESIS_VALUE_LOOKUP_PROHIBITED",
        }
        if not claim.get("include_in_ledger", True):
            rows.append(
                {
                    **base,
                    "thesis_display_value": token,
                    "computed_value": "",
                    "unit": _display_token_spec(token)["unit"],
                    "rounding": _display_token_spec(token)["rounding"],
                    "formula": "",
                    "source_path": "",
                    "source_sha256": "",
                    "source_row": "",
                    "source_column": "",
                    "source_json_or_config_key": "",
                    "source_role": "",
                    "producer": "",
                    "evidence_status": "EXCLUDED_LITERATURE_CONTEXT",
                    "source_match_count": 0,
                    "source_binding_kind": "",
                    "source_data_row_index": "",
                    "source_column_index": "",
                    "source_table_index": "",
                    "display_transform": "",
                }
            )
            continue
        explicit_rule = claim_rules.get((str(claim.get("claim_id", "")), token_index))
        if explicit_rule is not None:
            rows.append(
                {
                    **base,
                    "thesis_display_value": token,
                    **_resolve_claim_rule(claim, explicit_rule, item_map),
                }
            )
            continue
        linked_source_tables = [
            table
            for item in linked
            for table in item.get("source_tables", [])
            if isinstance(table, dict)
        ]
        linked_formulas = list(
            dict.fromkeys(str(item.get("formula", "")) for item in linked if item.get("formula"))
        )
        linked_producers = list(
            dict.fromkeys(str(item.get("producer", "")) for item in linked if item.get("producer"))
        )
        if not linked_ids:
            evidence_status = "UNRESOLVED_NO_SEMANTIC_SOURCE_SCOPE"
        elif force_non_empirical:
            evidence_status = "NON_EMPIRICAL_CONFIG_REGISTRY"
        else:
            evidence_status = "UNRESOLVED_NO_EXPLICIT_CLAIM_SELECTOR"
        semantic_sources = [
            table
            for table in linked_source_tables
            if not force_non_empirical or table.get("source_role") == "DESIGN_CONFIG_EVIDENCE"
        ]
        rows.append(
            {
                **base,
                "thesis_display_value": token,
                "computed_value": "",
                "unit": _display_token_spec(token)["unit"],
                "rounding": _display_token_spec(token)["rounding"],
                "formula": (
                    "CONFIG_REGISTRY; NO_EMPIRICAL_CALCULATION"
                    if force_non_empirical
                    else "; ".join(linked_formulas)
                ),
                "source_path": "; ".join(str(table.get("path", "")) for table in semantic_sources),
                "source_sha256": "; ".join(str(table.get("sha256", "")) for table in semantic_sources),
                "source_row": "",
                "source_column": "",
                "source_json_or_config_key": "",
                "source_role": (
                    "DESIGN_CONFIG_EVIDENCE"
                    if force_non_empirical
                    else "REGISTERED_SEMANTIC_SCOPE_NOT_CELL_RESOLVED"
                ),
                "producer": "; ".join(linked_producers),
                "evidence_status": evidence_status,
                "source_match_count": len(semantic_sources),
                "source_binding_kind": "",
                "source_data_row_index": "",
                "source_column_index": "",
                "source_table_index": "",
                "display_transform": "",
            }
        )
    return rows


def _is_exact_computed_claim(row: dict[str, Any]) -> bool:
    status = str(row.get("evidence_status", ""))
    return status.endswith("EXACT_SELECTED_RUN_SOURCE_CELL") or status.endswith("_EXACT_CELL")


def _computed_value_matches_thesis_display(row: dict[str, Any]) -> bool | None:
    """Compare a recalculated value with the number as printed in the thesis."""
    if not _is_exact_computed_claim(row):
        return None
    display = _display_token_spec(str(row.get("thesis_display_value", "")))
    paper_values = list(display["values"])
    computed = row.get("computed_value", "")
    computed_values = computed if isinstance(computed, list) else [computed]
    if len(paper_values) != len(computed_values) or not paper_values:
        return False
    try:
        numeric_values = [Decimal(str(value)) for value in computed_values]
    except (InvalidOperation, ValueError):
        return False
    token = str(row.get("thesis_display_value", "")).strip()
    if len(paper_values) == 1:
        if token.startswith(("<", "＜")):
            return numeric_values[0] < paper_values[0]
        if token.startswith((">", "＞")):
            return numeric_values[0] > paper_values[0]
        if token.startswith(("≤", "≦")):
            return numeric_values[0] <= paper_values[0]
        if token.startswith(("≥", "≧")):
            return numeric_values[0] >= paper_values[0]
    decimals = list(display["decimals"])
    for index, (paper, value) in enumerate(zip(paper_values, numeric_values)):
        digits = decimals[index] if index < len(decimals) else 12
        quantum = Decimal(1).scaleb(-digits)
        if value.quantize(quantum, rounding=ROUND_HALF_UP) != paper:
            return False
    return True


def _explicit_claim_target_failures(
    claim_rows: Sequence[dict[str, Any]],
    claim_rules: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return exactly one failure record for every unresolved/missing contract target."""
    rows_by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in claim_rows:
        try:
            key = (str(row.get("claim_id", "")), int(row.get("token_index")))
        except (TypeError, ValueError):
            continue
        if key in claim_rules:
            rows_by_key.setdefault(key, []).append(row)

    failures: list[dict[str, Any]] = []
    for claim_id, token_index in sorted(claim_rules):
        matching = rows_by_key.get((claim_id, token_index), [])
        claim_token_id = f"{claim_id}-{token_index:02d}"
        if len(matching) != 1:
            failures.append(
                {
                    "claim_token_id": claim_token_id,
                    "evidence_status": (
                        "UNRESOLVED_EXPLICIT_CLAIM_LEDGER_ROW_MISSING"
                        if not matching
                        else "UNRESOLVED_EXPLICIT_CLAIM_LEDGER_ROW_DUPLICATE"
                    ),
                    "ledger_row_count": len(matching),
                }
            )
            continue
        row = matching[0]
        if not _is_exact_computed_claim(row):
            failures.append(
                {
                    "claim_token_id": claim_token_id,
                    "evidence_status": str(row.get("evidence_status", "UNRESOLVED")),
                    "ledger_row_count": 1,
                }
            )
    return failures


def _reviewer_core_claim_rows(
    reviewer_contract: dict[str, Any],
    claim_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse selected numeric tokens into thesis-claim rows a reviewer can read."""
    if reviewer_contract.get("schema_version") != "reviewer_core_claims_v11":
        raise RuntimeError("Unsupported reviewer core claim contract")
    definitions = reviewer_contract.get("claims")
    if not isinstance(definitions, list) or not definitions:
        raise RuntimeError("Reviewer core claim contract needs a non-empty claims list")
    rows_by_token: dict[str, dict[str, Any]] = {}
    for row in claim_rows:
        token_id = str(row.get("claim_token_id", ""))
        if token_id in rows_by_token:
            raise RuntimeError(f"Duplicate thesis numeric token row: {token_id}")
        rows_by_token[token_id] = row

    output: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    seen_tokens: set[str] = set()
    for order, definition in enumerate(definitions, start=1):
        if not isinstance(definition, dict):
            raise RuntimeError("Reviewer core claim entries must be objects")
        claim_key = str(definition.get("claim_key", "")).strip()
        category = str(definition.get("category", "")).strip()
        title = str(definition.get("title", "")).strip()
        token_ids = definition.get("token_ids")
        if not all((claim_key, category, title)) or not isinstance(token_ids, list) or not token_ids:
            raise RuntimeError(f"Incomplete reviewer core claim: {claim_key or order}")
        if claim_key in seen_keys:
            raise RuntimeError(f"Duplicate reviewer core claim key: {claim_key}")
        seen_keys.add(claim_key)
        normalized_ids = [str(value).strip() for value in token_ids]
        missing = [value for value in normalized_ids if value not in rows_by_token]
        if missing:
            raise RuntimeError(f"Reviewer core claim tokens are absent from thesis: {missing}")
        duplicates = [value for value in normalized_ids if value in seen_tokens]
        if duplicates:
            raise RuntimeError(f"Reviewer core claim token reused across claims: {duplicates}")
        seen_tokens.update(normalized_ids)
        selected = [rows_by_token[value] for value in normalized_ids]
        exact = [_is_exact_computed_claim(row) for row in selected]
        agreements = [_computed_value_matches_thesis_display(row) for row in selected]
        computed_count = sum(exact)
        agreement_count = sum(value is True for value in agreements)
        disagreement_count = sum(value is False for value in agreements)
        if computed_count == len(selected) and disagreement_count == 0:
            result = "논문값과 일치"
            unavailable_reason = ""
        elif computed_count == len(selected):
            result = f"차이 있음 ({agreement_count}/{len(selected)} 일치)"
            unavailable_reason = "실행 산출물에서 다시 계산한 값과 논문 표시값이 다름"
        elif computed_count:
            result = (
                f"일부 계산됨 ({computed_count}/{len(selected)}); "
                f"계산된 값 중 {agreement_count}/{computed_count} 일치"
            )
            reasons = ["일부 수치의 원 실행 산출물이 현재 선택 실행에 없음"]
            if disagreement_count:
                reasons.append("계산 가능한 값 중 논문 표시값과 다른 수치가 있음")
            unavailable_reason = "; ".join(reasons)
        else:
            result = "현재 산출물로 계산할 수 없음"
            unavailable_reason = "해당 수치를 직접 계산한 원 실행 산출물이 보존되지 않았거나 현재 선택 실행에 포함되지 않음"

        paper_values: list[str] = []
        computed_values: list[str] = []
        differences: list[str] = []
        for row, is_exact in zip(selected, exact):
            paper = str(row.get("thesis_display_value", ""))
            paper_values.append(paper)
            if not is_exact:
                computed_values.append("확인 불가")
                differences.append("—")
                continue
            computed = row.get("computed_value", "")
            if isinstance(computed, list):
                computed_values.append(" / ".join(str(value) for value in computed))
            else:
                computed_values.append(str(computed))
            display_numbers = _display_token_spec(paper)["values"]
            if len(display_numbers) == 1 and not isinstance(computed, list):
                try:
                    gap = Decimal(str(computed)) - display_numbers[0]
                    differences.append(format(gap.normalize(), "f") if gap else "0")
                except (InvalidOperation, ValueError):
                    differences.append("—")
            elif isinstance(computed, list) and len(display_numbers) == len(computed):
                try:
                    gaps = [
                        Decimal(str(value)) - paper
                        for value, paper in zip(computed, display_numbers)
                    ]
                    differences.append(
                        " / ".join(format(gap.normalize(), "f") if gap else "0" for gap in gaps)
                    )
                except (InvalidOperation, ValueError):
                    differences.append("—")
            else:
                differences.append("—")

        paragraphs = list(dict.fromkeys(str(row.get("text", "")) for row in selected))
        pages = list(
            dict.fromkeys(
                str(row.get("printed_page", ""))
                for row in selected
                if str(row.get("printed_page", "")).strip()
            )
        )
        formulas = list(
            dict.fromkeys(
                str(row.get("formula", ""))
                for row, is_exact in zip(selected, exact)
                if is_exact and str(row.get("formula", "")).strip()
            )
        )
        sources: list[str] = []
        for row, is_exact in zip(selected, exact):
            if not is_exact:
                continue
            for field in ("input_source_paths", "source_path"):
                for value in str(row.get(field, "")).splitlines():
                    value = value.strip()
                    if value and value not in sources:
                        sources.append(value)
        producers = list(
            dict.fromkeys(
                str(row.get("producer", ""))
                for row, is_exact in zip(selected, exact)
                if is_exact and str(row.get("producer", "")).strip()
            )
        )
        output.append(
            {
                "order": order,
                "claim_key": claim_key,
                "category": category,
                "title": title,
                "printed_page": ", ".join(pages),
                "thesis_claim": "\n\n".join(paragraphs),
                "thesis_values": " | ".join(paper_values),
                "recomputed_values": " | ".join(computed_values),
                "differences": " | ".join(differences),
                "result": result,
                "calculation": "\n".join(formulas),
                "source_outputs": "\n".join(sources),
                "producer": "\n".join(producers),
                "run_id": str(selected[0].get("run_id", "")),
                "unavailable_reason": unavailable_reason,
                "token_ids": "; ".join(normalized_ids),
                "computed_value_count": computed_count,
                "value_count": len(selected),
            }
        )
    return output


def _sample_flow_value(
    *,
    stage: str,
    thesis_value: int | None,
    actual_value: int | None,
    meaning: str,
    calculation: str,
    source_outputs: Sequence[str],
    producer: str,
    note: str = "",
    result_override: str = "",
) -> dict[str, Any]:
    if thesis_value is None:
        difference: int | None = None
        result = "논문 표의 중간 연결값"
    elif actual_value is None:
        difference = None
        result = "현재 자료로 계산할 수 없음"
    else:
        difference = actual_value - thesis_value
        result = "확인됨" if difference == 0 else f"차이 있음 ({difference:+,})"
    if result_override:
        result = result_override
    return {
        "stage": stage,
        "thesis_value": thesis_value,
        "actual_value": actual_value,
        "difference": difference,
        "result": result,
        "meaning": meaning,
        "calculation": calculation,
        "source_outputs": "\n".join(value for value in source_outputs if value),
        "producer": producer,
        "note": note,
    }


def _first_selected_source(
    pattern: str, project_root: Path, selected_run: SelectedRun
) -> Path | None:
    sources, _ = resolve_sources([pattern], project_root, selected_run)
    return sources[0] if len(sources) == 1 else None


def _build_sample_flow(
    project_root: Path,
    selected_run: SelectedRun,
    build_dir: Path,
) -> dict[str, Any]:
    """Recompute the sample flow only from outputs made by the original producers."""
    stage0_path = _first_selected_source(
        "data/final_freeze/stage0_oracle_foundation/stage0_manifest.json",
        project_root,
        selected_run,
    )
    raw_count: int | None = None
    raw_source_rows: list[dict[str, Any]] = []
    raw_details: dict[str, Any] = {}
    stage0: dict[str, Any] = {}
    if stage0_path is not None:
        stage0 = _load_json_object(stage0_path)
        files = stage0.get("statement_items", {}).get("files", [])
        counts_by_statement: dict[str, int] = {}
        if isinstance(files, list):
            for source in files:
                if not isinstance(source, dict):
                    continue
                statement_type = str(source.get("statement_type", "")).strip()
                try:
                    rows_raw = int(source.get("rows_raw", 0))
                    rows_long = int(source.get("rows_long", 0))
                except (TypeError, ValueError):
                    continue
                if not statement_type:
                    continue
                counts_by_statement[statement_type] = (
                    counts_by_statement.get(statement_type, 0) + rows_raw
                )
                raw_source_rows.append(
                    {
                        "statement_type": statement_type,
                        "source_path": str(source.get("file", "")),
                        "source_rows": rows_raw,
                        "valid_key_rows": rows_raw,
                        "unique_keys_in_file": "",
                    }
                )
        statement_counts = set(counts_by_statement.values())
        if len(counts_by_statement) == 5 and len(statement_counts) == 1:
            raw_count = next(iter(statement_counts))
        raw_details = {
            "calculation_grain": "one source row per firm-year within each statement family",
            "producer_manifest": _relative(stage0_path, project_root),
            "statement_family_row_counts": counts_by_statement,
            "statement_items_long_rows": stage0.get("row_counts", {}).get(
                "statement_items_panel"
            ),
            "population_rule": (
                "exact Stage0 producer input selection; no additional industry filter "
                "is imposed by the thesis-output builder"
            ),
        }
    raw_note = (
        "원본 Stage0 producer가 실제로 읽은 다섯 재무제표 유형은 각각 "
        f"{raw_count:,}개 행입니다. "
        if raw_count is not None
        else "원본 Stage0 기록에는 다섯 재무제표 유형의 공통 행수 정보가 부족합니다. "
    )
    raw_note += (
        "원본 producer는 기업 단위 금융업 제외 전의 표본을 사용하며, "
        "이 Excel도 같은 표본 구성을 따릅니다."
    )
    derived_dir = build_dir / "derived"
    _write_csv(
        derived_dir / "sample_flow_raw_sources.csv",
        [
            "statement_type",
            "source_path",
            "source_rows",
            "valid_key_rows",
            "unique_keys_in_file",
        ],
        raw_source_rows,
    )

    stage1_metadata_path = _first_selected_source(
        "data/final_freeze/stage1_oracle_inputs/stage00_01_rating_statement_integration/stage00_01_metadata.json",
        project_root,
        selected_run,
    )
    sample_report_path = _first_selected_source(
        "data/final_freeze/stage1_oracle_backends/alpha/sample_filter_report.csv",
        project_root,
        selected_run,
    )
    transition_path = _first_selected_source(
        "data/final_freeze/stage2_candidate_projection/counterfactual_transitions_metadata__P50.json",
        project_root,
        selected_run,
    )
    split_path = _first_selected_source(
        "data/final_freeze/stage1_oracle_backends/alpha/split_reconciliation_stage4.csv",
        project_root,
        selected_run,
    )
    final_policy_path = _first_selected_source(
        "data/final_freeze/stage6_multi_oracle_eval/final_policy_summary.csv",
        project_root,
        selected_run,
    )

    stage0_rating_count: int | None = None
    if stage0_path is not None:
        try:
            stage0_rating_count = int(
                stage0["row_counts"].get(
                    "rating_panel", stage0["row_counts"].get("canonical_panel")
                )
            )
        except (KeyError, TypeError, ValueError):
            stage0_rating_count = None
    stage1_join_count: int | None = None
    financial_exclusion_rows: int | None = None
    financial_exclusion_firms: int | None = None
    has_explicit_financial_exclusion = False
    if stage1_metadata_path is not None:
        stage1 = _load_json_object(stage1_metadata_path)
        try:
            stage1_join_count = int(stage1["firm_year_rows"])
        except (KeyError, TypeError, ValueError):
            stage1_join_count = None
        exclusion = stage1.get("financial_industry_exclusion")
        if isinstance(exclusion, dict):
            try:
                financial_exclusion_rows = int(exclusion["rows_excluded"])
                financial_exclusion_firms = int(exclusion["firms_excluded"])
                has_explicit_financial_exclusion = True
            except (KeyError, TypeError, ValueError):
                has_explicit_financial_exclusion = False
    rating_count = stage1_join_count
    if has_explicit_financial_exclusion:
        rating_note = (
            f"Stage0 외부등급 후보 {stage0_rating_count:,} 기업-연도 중 금융업 "
            f"{financial_exclusion_rows:,}행({financial_exclusion_firms:,}개사)을 제외한 뒤의 Stage1 행수."
            if None not in (
                stage0_rating_count,
                financial_exclusion_rows,
                financial_exclusion_firms,
            )
            else "Stage1에서 실제 금융업 제외 내역을 보존한 뒤 계산한 행수."
        )
        rating_result_override = ""
    else:
        rating_note = (
            "이 FrozenReplay의 과거 Stage1 산출물은 금융업 분류 항목이 미기록 상태입니다. "
            "4,924는 등급 결합 표본으로 표기합니다."
        )
        rating_result_override = (
            "숫자는 일치하나 금융업 제외 여부는 확인할 수 없음"
            if rating_count == 4_924
            else "금융업 제외 여부를 확인할 수 없음"
        )

    modeling_count: int | None = None
    if sample_report_path is not None:
        with sample_report_path.open("r", encoding="utf-8-sig", newline="") as handle:
            report_rows = list(csv.DictReader(handle))
        if len(report_rows) == 1:
            try:
                modeling_count = int(report_rows[0]["final_modeling_rows"])
            except (KeyError, TypeError, ValueError):
                modeling_count = None

    transition_count: int | None = None
    if transition_path is not None:
        transition = _load_json_object(transition_path)
        try:
            transition_count = int(transition["base_rows"])
        except (KeyError, TypeError, ValueError):
            transition_count = None

    split_counts: dict[str, int] = {}
    if split_path is not None:
        with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                split_name = str(row.get("split_stage4", "")).strip().lower()
                try:
                    count = int(row.get("n", ""))
                except (TypeError, ValueError):
                    continue
                split_counts[split_name] = split_counts.get(split_name, 0) + count
    dev_count = split_counts.get("dev")
    oot_count = split_counts.get("oot")
    if (
        modeling_count is not None
        and dev_count is not None
        and oot_count is not None
        and dev_count + oot_count != modeling_count
    ):
        dev_count = None
        oot_count = None

    final_count: int | None = None
    if final_policy_path is not None:
        with final_policy_path.open("r", encoding="utf-8-sig", newline="") as handle:
            policy_rows = list(csv.DictReader(handle))
        try:
            values = {int(row["n"]) for row in policy_rows if str(row.get("n", "")).strip()}
            final_count = next(iter(values)) if len(values) == 1 else None
        except (KeyError, TypeError, ValueError):
            final_count = None

    raw_source_registry = _relative(
        derived_dir / "sample_flow_raw_sources.csv", project_root
    )
    rows = [
        _sample_flow_value(
            stage="① 원본 Stage0 재무제표 패널",
            thesis_value=70_777,
            actual_value=raw_count,
            meaning="원본 Stage0 producer가 실제로 선택한 KOSPI·KOSDAQ·KONEX 재무제표 파일의 기업-연도 행. 다섯 재무제표 유형별 공통 분모.",
            calculation="원본 Stage0 기록의 파일별 rows_raw를 재무제표 유형별로 합산하고 다섯 유형의 합계를 확인. 표본 범위는 원본 Stage0 producer 입력과 동일.",
            source_outputs=[
                _relative(stage0_path, project_root) if stage0_path else "",
                raw_source_registry,
            ],
            producer="src/credit_recourse/oracle/stage0/build_stage0_foundation_from_raw.py",
            note=raw_note,
        ),
        _sample_flow_value(
            stage="② 원본 Stage1 외부등급-재무 연결",
            thesis_value=4_924,
            actual_value=rating_count,
            meaning="원본 Stage0의 허용 외부등급을 기업-연도별 1개로 정리하고 원본 Stage1이 재무자료와 결합한 행수.",
            calculation="원본 Stage1 metadata의 firm_year_rows를 읽음. 표본 범위는 Stage1 producer의 결합 결과와 동일.",
            source_outputs=[
                _relative(stage0_path, project_root) if stage0_path else "",
                _relative(stage1_metadata_path, project_root) if stage1_metadata_path else "",
            ],
            producer="src/credit_recourse/oracle/stage0/build_stage0_foundation_from_raw.py; src/credit_recourse/oracle/stage1/stage00_01_rating_statement/final_stage0_adapter.py",
            note=rating_note,
            result_override=rating_result_override,
        ),
        _sample_flow_value(
            stage="②-1 Oracle 모델링 가능 기업-연도",
            thesis_value=None,
            actual_value=modeling_count,
            meaning="4,924개 등급 기업-연도 중 Oracle 입력 재무정보가 남은 3,822개. DEV와 OOT의 공통 분모.",
            calculation="Stage1 Oracle 입력에서 재무정보가 모두 결측인 행을 제외.",
            source_outputs=[
                _relative(sample_report_path, project_root) if sample_report_path else ""
            ],
            producer="src/credit_recourse/oracle/backends/alpha/_pipeline_impl.py",
        ),
        _sample_flow_value(
            stage="③ 등급 전이",
            thesis_value=3_159,
            actual_value=transition_count,
            meaning="연속된 두 기업-연도를 연결해 만든 실제 t→t+1 전이. Candidate-IQL 학습자료의 분석단위.",
            calculation="기업 내 연도순으로 연속 2개년을 연결하고 실제 다음 해 등급이 있는 전이만 집계.",
            source_outputs=[
                _relative(transition_path, project_root) if transition_path else ""
            ],
            producer="src/credit_recourse/rl/pipelines/final_stage2_candidate_projection/pipeline.py",
        ),
        _sample_flow_value(
            stage="④ Oracle 개발표본 (DEV, 2002–2019)",
            thesis_value=2_032,
            actual_value=dev_count,
            meaning="Oracle 모형 개발에 사용한 2002–2019년 기업-연도.",
            calculation="Stage1 최종 모델링 표본을 회계연도 2019년까지로 합산.",
            source_outputs=[_relative(split_path, project_root) if split_path else ""],
            producer="src/credit_recourse/oracle/backends/alpha/_pipeline_impl.py",
        ),
        _sample_flow_value(
            stage="⑤ Oracle 시간외 검증표본 (OOT, 2020–2023)",
            thesis_value=1_790,
            actual_value=oot_count,
            meaning="Oracle 개발에 사용하지 않은 시간외 기업-연도.",
            calculation="Stage1 최종 모델링 표본을 회계연도 2020–2023으로 합산.",
            source_outputs=[_relative(split_path, project_root) if split_path else ""],
            producer="src/credit_recourse/oracle/backends/alpha/_pipeline_impl.py",
        ),
        _sample_flow_value(
            stage="⑥ 최종 정책 비교",
            thesis_value=575,
            actual_value=final_count,
            meaning="2024년 기준 등급을 보유한 평가 전용 기업. 정책별·Oracle별 비교의 공통 기업 분모.",
            calculation="Stage6 최종 정책요약의 모든 정책 행이 공유하는 기업수 n을 확인.",
            source_outputs=[
                _relative(final_policy_path, project_root) if final_policy_path else ""
            ],
            producer="src/credit_recourse/rl/pipelines/final_stage6_candidate_iql_multi_oracle_eval/pipeline.py",
        ),
    ]
    result = {
        "run_id": selected_run.run_id,
        "run_mode": selected_run.mode,
        "rows": rows,
        "raw_panel_details": raw_details,
    }
    _write_json(derived_dir / "sample_flow_recomputation.json", result)
    _write_csv(
        derived_dir / "sample_flow_recomputation.csv",
        [
            "stage",
            "thesis_value",
            "actual_value",
            "difference",
            "result",
            "meaning",
            "calculation",
            "source_outputs",
            "producer",
            "note",
        ],
        rows,
    )
    table_3_1_rows = [row for row in rows if row["thesis_value"] is not None]
    _write_csv(
        derived_dir / "table_3_1_sample_flow.csv",
        [
            "선정 단계",
            "논문 표시값",
            "재계산값",
            "차이",
            "확인 결과",
            "분모와 의미",
            "계산 방법",
            "실제 입력·산출물",
            "생산 코드",
            "주의할 점",
        ],
        [
            {
                "선정 단계": row["stage"],
                "논문 표시값": row["thesis_value"],
                "재계산값": row["actual_value"],
                "차이": row["difference"],
                "확인 결과": row["result"],
                "분모와 의미": row["meaning"],
                "계산 방법": row["calculation"],
                "실제 입력·산출물": row["source_outputs"],
                "생산 코드": row["producer"],
                "주의할 점": row["note"],
            }
            for row in table_3_1_rows
        ],
    )
    return result


def _derive_action_semantics_claim_values(
    selected_run: SelectedRun,
    build_dir: Path,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """Recompute proposed/applied action claims from preserved Stage-7 evidence.

    The raw response supplies the proposed 10D vector.  The Stage-7 action
    table supplies the vector actually sent to the simulator.  Keeping those
    two objects separate is essential: total-L1 instruction compliance and
    per-variable bound clipping answer different questions.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return {}

    action_names = [
        "ppe_pct",
        "inv_turnover_chg",
        "ar_turnover_chg",
        "ap_turnover_chg",
        "short_debt_pct",
        "long_debt_pct",
        "bond_pct",
        "revenue_growth",
        "cogs_ratio_chg",
        "sga_ratio_chg",
    ]
    run_specs = [
        ("GPT-5.4-mini", "0p75", "C4RXJ_C4C4RC6_L1_0p75_ICb_gpt54mini_p50_seed1_*"),
        ("GPT-5.4-mini", "unbounded", "C4RXJ_C4C4RC6_L1_unbounded_ICb_gpt54mini_p50_seed1_*"),
        ("Gemini 3.1 Flash-Lite", "0p75", "C4RXJ_C4C4RC6_L1_0p75_ICb_gemini31flashlite_p50_seed1_*"),
        ("Gemini 3.1 Flash-Lite", "unbounded", "C4RXJ_C4C4RC6_L1_unbounded_ICb_gemini31flashlite_p50_seed1_*"),
    ]
    roots = [
        value
        for value in (selected_run.final_freeze_overlay_root, selected_run.final_freeze_root)
        if value is not None and value.exists()
    ]

    summaries: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    decoder = json.JSONDecoder()
    for model, budget, run_pattern in run_specs:
        matches: list[Path] = []
        for root in roots:
            matches = [
                candidate
                for candidate in sorted((root / "llm_runs").glob(run_pattern))
                if (
                    candidate
                    / "stage7_llm_action_generation"
                    / "llm_stage7_action_table.parquet"
                ).is_file()
                and (
                    candidate
                    / "stage7_llm_action_generation"
                    / "llm_stage7_response_log.parquet"
                ).is_file()
            ]
            if matches:
                break
        if len(matches) != 1:
            continue
        stage7 = matches[0] / "stage7_llm_action_generation"
        action_path = stage7 / "llm_stage7_action_table.parquet"
        response_path = stage7 / "llm_stage7_response_log.parquet"
        if not action_path.is_file() or not response_path.is_file():
            continue
        actions = pd.read_parquet(action_path)
        responses = pd.read_parquet(response_path)
        required_action = {
            "row_id",
            "policy",
            "mode",
            *(f"action__{name}" for name in action_names),
        }
        if not required_action.issubset(actions.columns) or not {
            "row_id", "policy", "mode", "raw_response"
        }.issubset(responses.columns):
            continue
        actions = actions[
            actions["policy"].astype(str).eq("C4")
            & actions["mode"].astype(str).eq("free_form_10d")
        ].copy()
        responses = responses[
            responses["policy"].astype(str).eq("C4")
            & responses["mode"].astype(str).eq("free_form_10d")
        ].copy()
        merged = actions.merge(
            responses[["row_id", "policy", "mode", "raw_response"]],
            on=["row_id", "policy", "mode"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != 575:
            continue
        proposed_vectors: list[list[float]] = []
        parse_ok = True
        for raw_response in merged["raw_response"]:
            text = str(raw_response).strip()
            object_start = text.find("{")
            if object_start < 0:
                parse_ok = False
                break
            try:
                payload, _ = decoder.raw_decode(text[object_start:])
                vector = payload["action_vector"]
                proposed_vectors.append(
                    [float(vector.get(name, 0.0) or 0.0) for name in action_names]
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                parse_ok = False
                break
        if not parse_ok:
            continue
        proposed_l1: list[float] = []
        applied_l1: list[float] = []
        applied_vectors: list[list[float]] = []
        clipped_flags: list[bool] = []
        for row_index, proposed in enumerate(proposed_vectors):
            applied = [
                float(merged.iloc[row_index][f"action__{name}"] or 0.0)
                for name in action_names
            ]
            proposed_l1.append(sum(abs(value) for value in proposed))
            applied_l1.append(sum(abs(value) for value in applied))
            applied_vectors.append(applied)
            clipped_flags.append(
                any(abs(raw - final) > 1.0e-12 for raw, final in zip(proposed, applied))
            )
        if budget == "0p75":
            raw_audit = pd.to_numeric(merged.get("budget_l1_raw"), errors="coerce")
            clipped_audit = pd.to_numeric(merged.get("budget_l1_clipped"), errors="coerce")
            if raw_audit.isna().any() or clipped_audit.isna().any():
                continue
            if max(abs(float(value) - audit) for value, audit in zip(proposed_l1, raw_audit)) > 1e-9:
                continue
            if max(abs(float(value) - audit) for value, audit in zip(applied_l1, clipped_audit)) > 1e-9:
                continue
        summaries.append(
            {
                "model": model,
                "budget": budget,
                "n_firms": len(merged),
                "mean_proposed_l1": sum(proposed_l1) / len(proposed_l1),
                "mean_applied_l1": sum(applied_l1) / len(applied_l1),
                "proposed_l1_over_0p75_pct": 100.0 * sum(value > 0.750000001 for value in proposed_l1) / len(proposed_l1),
                "any_variable_clipping_pct": 100.0 * sum(clipped_flags) / len(clipped_flags),
                "applied_l1_at_least_95pct_of_0p75_pct": 100.0 * sum(value >= 0.75 * 0.95 for value in applied_l1) / len(applied_l1),
                "stage7_action_table": _relative(action_path, project_root),
                "stage7_raw_response_log": _relative(response_path, project_root),
            }
        )
        for row_index, (proposed, applied) in enumerate(
            zip(proposed_vectors, applied_vectors)
        ):
            detail = {
                "model": model,
                "budget": budget,
                "source_run": matches[0].name,
                "row_id": merged.iloc[row_index]["row_id"],
                "proposed_l1": proposed_l1[row_index],
                "applied_l1": applied_l1[row_index],
                "proposed_l1_over_0p75": int(proposed_l1[row_index] > 0.750000001),
                "any_variable_clipping": int(clipped_flags[row_index]),
                "applied_l1_at_least_95pct_of_0p75": int(
                    applied_l1[row_index] >= 0.75 * 0.95
                ),
            }
            for action_index, action_name in enumerate(action_names):
                detail[f"proposed__{action_name}"] = proposed[action_index]
                detail[f"applied__{action_name}"] = applied[action_index]
            detail_rows.append(detail)
    if len(summaries) != len(run_specs):
        return {}

    derived_dir = build_dir / "derived"
    derived_path = derived_dir / "action_semantics_summary.csv"
    fields = [
        "model",
        "budget",
        "n_firms",
        "mean_proposed_l1",
        "mean_applied_l1",
        "proposed_l1_over_0p75_pct",
        "any_variable_clipping_pct",
        "applied_l1_at_least_95pct_of_0p75_pct",
        "stage7_action_table",
        "stage7_raw_response_log",
    ]
    _write_csv(derived_path, fields, summaries)
    detail_path = derived_dir / "action_semantics_firm_rows.csv"
    detail_fields = [
        "model",
        "budget",
        "source_run",
        "row_id",
        *[f"proposed__{name}" for name in action_names],
        *[f"applied__{name}" for name in action_names],
        "proposed_l1",
        "applied_l1",
        "proposed_l1_over_0p75",
        "any_variable_clipping",
        "applied_l1_at_least_95pct_of_0p75",
    ]
    _write_csv(detail_path, detail_fields, detail_rows)
    derived_sha256 = sha256_file(derived_path)
    metric_specs = [
        ("NC0197-03", "GPT-5.4-mini", "0p75", "mean_proposed_l1", 3, "기업별 제안행동 10개 절댓값 합을 구한 뒤 575개 기업 평균"),
        ("NC0197-04", "GPT-5.4-mini", "unbounded", "mean_proposed_l1", 3, "기업별 제안행동 10개 절댓값 합을 구한 뒤 575개 기업 평균"),
        ("NC0197-06", "Gemini 3.1 Flash-Lite", "0p75", "mean_proposed_l1", 3, "기업별 제안행동 10개 절댓값 합을 구한 뒤 575개 기업 평균"),
        ("NC0197-07", "Gemini 3.1 Flash-Lite", "unbounded", "mean_proposed_l1", 3, "기업별 제안행동 10개 절댓값 합을 구한 뒤 575개 기업 평균"),
        ("NC0197-10", "GPT-5.4-mini", "0p75", "proposed_l1_over_0p75_pct", 2, "제안행동 L1이 0.75를 넘는 기업 비율"),
        ("NC0197-12", "Gemini 3.1 Flash-Lite", "0p75", "proposed_l1_over_0p75_pct", 2, "제안행동 L1이 0.75를 넘는 기업 비율"),
        ("NC0198-04", "GPT-5.4-mini", "0p75", "any_variable_clipping_pct", 2, "제안행동과 변수별 한계 적용 후 행동이 하나라도 다른 기업 비율"),
        ("NC0198-06", "Gemini 3.1 Flash-Lite", "0p75", "any_variable_clipping_pct", 2, "제안행동과 변수별 한계 적용 후 행동이 하나라도 다른 기업 비율"),
        ("NC0199-04", "GPT-5.4-mini", "0p75", "applied_l1_at_least_95pct_of_0p75_pct", 2, "적용행동 L1이 0.75의 95% 이상인 기업 비율"),
        ("NC0199-06", "Gemini 3.1 Flash-Lite", "0p75", "applied_l1_at_least_95pct_of_0p75_pct", 2, "적용행동 L1이 0.75의 95% 이상인 기업 비율"),
    ]
    by_key = {(row["model"], row["budget"]): (index, row) for index, row in enumerate(summaries)}
    overrides: dict[str, dict[str, Any]] = {}
    for token_id, model, budget, metric, decimals, formula in metric_specs:
        source_row, summary = by_key[(model, budget)]
        value = _rounded_claim_value(summary[metric], "IDENTITY", decimals)
        overrides[token_id] = {
            "computed_value": value,
            "unit": "PERCENT" if metric.endswith("_pct") else "L1_NORM",
            "rounding": f"ROUND_HALF_UP({decimals}dp)",
            "formula": formula,
            "source_path": _relative(derived_path, project_root),
            "input_source_paths": "\n".join(
                [
                    str(summary["stage7_raw_response_log"]),
                    str(summary["stage7_action_table"]),
                ]
            ),
            "source_sha256": derived_sha256,
            "source_row": source_row + 2,
            "source_column": metric,
            "source_json_or_config_key": f"model={model}; budget={budget}",
            "source_role": "RECOMPUTED_FROM_SELECTED_RUN_STAGE7_EVIDENCE",
            "producer": "src/credit_recourse/reproduction/thesis_outputs/prepare.py::_derive_action_semantics_claim_values",
            "evidence_status": "RECOMPUTED_EXACT_SELECTED_RUN_SOURCE_CELL",
            "source_match_count": 1,
            "source_binding_kind": "DIRECT_SOURCE_CELL",
            "source_data_row_index": source_row,
            "source_column_index": fields.index(metric),
            "source_table_index": 0,
            "display_transform": f"IDENTITY; ROUND_HALF_UP({decimals}dp)",
            "resolution_rule": "DIRECT_REVIEWER_RECOMPUTATION:ACTION_SEMANTICS_FROM_PROPOSED_AND_APPLIED_VECTORS",
        }
    return overrides


def _derive_direct_reviewer_claim_values(
    selected_run: SelectedRun,
    build_dir: Path,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """Recalculate reviewer-facing thesis numbers from selected-run row data.

    These calculations deliberately start from the actual analysis or model
    output rows.  They do not read values printed in the thesis, legacy paper
    tables, or a previously prepared claim ledger.
    """
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return {}

    def exact_source(pattern: str) -> Path | None:
        paths, _ = resolve_sources([pattern], project_root, selected_run)
        return paths[0] if len(paths) == 1 else None

    records: list[dict[str, Any]] = []
    candidate_detail_rows: list[dict[str, Any]] = []
    repeatability_detail_rows: list[dict[str, Any]] = []
    e2_detail_rows: list[dict[str, Any]] = []
    e2_e3_detail_rows: list[dict[str, Any]] = []

    # Oracle counterfactual fidelity: count OOT firm-year rows whose empirical
    # and simulated changes have the same direction.
    test3_path = exact_source(
        "data/analysis/paper_repro/01_substrate_validation/"
        "test3_counterfactual_fidelity/test3_rows.csv"
    )
    if test3_path is not None:
        frame = pd.read_csv(test3_path)
        required = {"split", "direction_match"}
        if required.issubset(frame.columns):
            eligible = frame.loc[
                frame["split"].astype(str).str.lower().eq("oot")
                & frame["direction_match"].notna()
            ].copy()
            if not eligible.empty:
                direction_match = pd.to_numeric(
                    eligible["direction_match"], errors="coerce"
                )
                if direction_match.notna().all():
                    records.append(
                        {
                            "claim_token_id": "NC0120-03",
                            "metric": "oot_direction_match_count",
                            "computed_value": int(direction_match.eq(1).sum()),
                            "unit": "FIRM_YEAR_ROWS",
                            "rounding_decimals": 0,
                            "formula": (
                                "OOT 행 중 방향 판정이 가능한 행을 고른 뒤 "
                                "direction_match = 1인 기업-연도 행의 개수"
                            ),
                            "input_source_paths": _relative(test3_path, project_root),
                            "source_note": "선택 실행의 Oracle 반사실 적합도 행자료에서 재계산",
                        }
                    )

    # Candidate-IQL versus the deterministic weakest-component heuristic.
    # Recreate the paired estimand from the 575 firm rows; the producer's
    # policy_paired_inference CSV is comparison evidence, not the compute parent.
    stage6_rows_path = exact_source(
        "data/final_freeze/stage6_candidate_selector_eval/"
        "multi_oracle_policy_eval.parquet"
    )
    if stage6_rows_path is not None:
        stage6_rows = pd.read_parquet(stage6_rows_path)
        required = {
            "row_id",
            "policy",
            "delta_R_score_alpha",
            "delta_R_score_beta",
            "delta_R_score_gamma",
        }
        policies = {"C3_candidate_iql", "C2_weakest_component_rule"}
        if required.issubset(stage6_rows.columns):
            selected_rows = stage6_rows.loc[
                stage6_rows["policy"].astype(str).isin(policies)
            ].copy()
            if not selected_rows.duplicated(["row_id", "policy"]).any():
                candidate_by_row: dict[Any, dict[str, Any]] = {}
                for token_id, oracle in (
                    ("NC0127-04", "alpha"),
                    ("NC0127-05", "beta"),
                    ("NC0127-06", "gamma"),
                ):
                    wide = selected_rows.pivot(
                        index="row_id",
                        columns="policy",
                        values=f"delta_R_score_{oracle}",
                    )
                    if set(wide.columns) != policies or len(wide) != 575:
                        continue
                    wide = wide.apply(pd.to_numeric, errors="coerce")
                    if wide.isna().any().any():
                        continue
                    paired_difference = (
                        wide["C3_candidate_iql"]
                        - wide["C2_weakest_component_rule"]
                    )
                    for row_id, difference in paired_difference.items():
                        detail = candidate_by_row.setdefault(
                            row_id, {"row_id": row_id}
                        )
                        detail[f"candidate_iql_{oracle}"] = float(
                            wide.loc[row_id, "C3_candidate_iql"]
                        )
                        detail[f"weakest_component_{oracle}"] = float(
                            wide.loc[row_id, "C2_weakest_component_rule"]
                        )
                        detail[f"difference_{oracle}"] = float(difference)
                    records.append(
                        {
                            "claim_token_id": token_id,
                            "metric": f"candidate_iql_minus_heuristic_{oracle}",
                            "computed_value": float(paired_difference.mean()),
                            "unit": "ORACLE_SCORE_POINTS",
                            "rounding_decimals": 3,
                            "formula": (
                                "575개 동일 기업 각각에서 Candidate-IQL 정책가치에서 "
                                f"최약요인 휴리스틱 정책가치를 뺀 뒤 평균 ({oracle})"
                            ),
                            "input_source_paths": _relative(stage6_rows_path, project_root),
                            "source_note": "선택 실행 Stage6의 575개 기업 행을 row_id로 직접 대응해 재계산",
                        }
                    )
                expected_detail_columns = {
                    f"{prefix}_{oracle}"
                    for prefix in ("candidate_iql", "weakest_component", "difference")
                    for oracle in ("alpha", "beta", "gamma")
                }
                if (
                    len(candidate_by_row) == 575
                    and all(
                        expected_detail_columns.issubset(detail)
                        for detail in candidate_by_row.values()
                    )
                ):
                    candidate_detail_rows.extend(candidate_by_row.values())

    # Section 5.5 repeatability.  The three same-model C4 executions and the
    # three cross-model C4 executions are preserved at row level.  Recreate
    # the three pairwise comparisons from candidate_id and same-firm policy
    # values rather than reading the printed Table 5-8 numbers.
    def load_c4_repeat_run(run_pattern: str) -> tuple[Any, list[str]] | None:
        action_path = exact_source(
            "data/final_freeze/llm_runs/"
            f"{run_pattern}/stage7_llm_action_generation/llm_stage7_action_table.parquet"
        )
        score_path = exact_source(
            "data/final_freeze/llm_runs/"
            f"{run_pattern}/stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores.parquet"
        )
        if action_path is None or score_path is None:
            return None
        actions = pd.read_parquet(action_path)
        scores = pd.read_parquet(score_path)
        required_actions = {"row_id", "policy", "mode", "candidate_id"}
        required_scores = {
            "row_id",
            "policy",
            "mode",
            "delta_R_score_alpha",
            "delta_R_score_beta",
            "delta_R_score_gamma",
        }
        if not required_actions.issubset(actions.columns) or not required_scores.issubset(scores.columns):
            return None
        actions = actions.loc[
            actions["policy"].astype(str).eq("C4")
            & actions["mode"].astype(str).eq("free_form_10d"),
            ["row_id", "candidate_id"],
        ].copy()
        scores = scores.loc[
            scores["policy"].astype(str).eq("C4")
            & scores["mode"].astype(str).eq("free_form_10d"),
            ["row_id", "delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma"],
        ].copy()
        if (
            len(actions) != 575
            or len(scores) != 575
            or actions["row_id"].duplicated().any()
            or scores["row_id"].duplicated().any()
        ):
            return None
        frame = actions.merge(scores, on="row_id", how="inner", validate="one_to_one")
        if len(frame) != 575:
            return None
        return frame, [_relative(action_path, project_root), _relative(score_path, project_root)]

    repeat_groups = {
        "같은 파운데이션 모델 반복": {
            "기준 실행": "ICb_gpt54mini_p50_live_seed1",
            "독립 실행 2": "N5M_C4C6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_*",
            "독립 실행 3": "C4R_C4C4RC6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_*",
        },
        "서로 다른 파운데이션 모델": {
            "GPT-5.4-mini": "ICb_gpt54mini_p50_live_seed1",
            "GPT-4.1-mini": "ICb_gpt41mini_p50_live_seed1_postpatch_*",
            "Claude Haiku 4.5": "ICb_haiku45_p50_live_seed1_postpatch_*",
        },
    }
    pair_rows: list[dict[str, Any]] = []
    repeat_summaries: dict[str, dict[str, float]] = {}
    repeat_input_paths: list[str] = []
    for group_name, run_patterns in repeat_groups.items():
        loaded: dict[str, Any] = {}
        for label, pattern in run_patterns.items():
            result = load_c4_repeat_run(pattern)
            if result is None:
                loaded = {}
                break
            loaded[label] = result[0]
            repeat_input_paths.extend(result[1])
        if len(loaded) != 3:
            continue
        labels = list(loaded)
        group_metrics: list[dict[str, float]] = []
        for left_index in range(len(labels)):
            for right_index in range(left_index + 1, len(labels)):
                left_label = labels[left_index]
                right_label = labels[right_index]
                paired = loaded[left_label].merge(
                    loaded[right_label],
                    on="row_id",
                    how="inner",
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
                if len(paired) != 575:
                    group_metrics = []
                    break
                metrics = {
                    "nearest_candidate_match_pct": 100.0
                    * float(
                        paired["candidate_id_left"].astype(str).eq(
                            paired["candidate_id_right"].astype(str)
                        ).mean()
                    ),
                    "rank_corr_alpha": float(
                        paired["delta_R_score_alpha_left"].corr(
                            paired["delta_R_score_alpha_right"], method="spearman"
                        )
                    ),
                    "rank_corr_beta": float(
                        paired["delta_R_score_beta_left"].corr(
                            paired["delta_R_score_beta_right"], method="spearman"
                        )
                    ),
                    "rank_corr_gamma": float(
                        paired["delta_R_score_gamma_left"].corr(
                            paired["delta_R_score_gamma_right"], method="spearman"
                        )
                    ),
                }
                if not all(math.isfinite(value) for value in metrics.values()):
                    group_metrics = []
                    break
                for firm in paired.to_dict(orient="records"):
                    detail = {
                        "comparison_group": group_name,
                        "pair": f"{left_label} vs {right_label}",
                        "row_id": firm["row_id"],
                        "candidate_id_left": firm["candidate_id_left"],
                        "candidate_id_right": firm["candidate_id_right"],
                        "nearest_candidate_match": int(
                            str(firm["candidate_id_left"])
                            == str(firm["candidate_id_right"])
                        ),
                    }
                    for oracle in ("alpha", "beta", "gamma"):
                        left_value = float(firm[f"delta_R_score_{oracle}_left"])
                        right_value = float(firm[f"delta_R_score_{oracle}_right"])
                        detail[f"policy_value_{oracle}_left"] = left_value
                        detail[f"policy_value_{oracle}_right"] = right_value
                        detail[f"policy_value_{oracle}_difference"] = (
                            left_value - right_value
                        )
                    repeatability_detail_rows.append(detail)
                group_metrics.append(metrics)
                pair_rows.append(
                    {
                        "comparison_group": group_name,
                        "row_kind": "실행쌍",
                        "pair": f"{left_label} vs {right_label}",
                        "n_firms": len(paired),
                        **metrics,
                    }
                )
            if not group_metrics:
                break
        if len(group_metrics) == 3:
            mean_metrics = {
                key: sum(row[key] for row in group_metrics) / len(group_metrics)
                for key in group_metrics[0]
            }
            repeat_summaries[group_name] = mean_metrics
            for label, reducer in (
                ("평균", lambda values: sum(values) / len(values)),
                ("최솟값", min),
                ("최댓값", max),
            ):
                pair_rows.append(
                    {
                        "comparison_group": group_name,
                        "row_kind": "요약",
                        "pair": label,
                        "n_firms": 575,
                        **{
                            key: reducer([row[key] for row in group_metrics])
                            for key in group_metrics[0]
                        },
                    }
                )

    if len(repeat_summaries) == 2:
        repeat_pair_path = build_dir / "derived" / "repeatability_pair_calculations.csv"
        repeat_pair_fields = [
            "comparison_group",
            "row_kind",
            "pair",
            "n_firms",
            "nearest_candidate_match_pct",
            "rank_corr_alpha",
            "rank_corr_beta",
            "rank_corr_gamma",
        ]
        _write_csv(repeat_pair_path, repeat_pair_fields, pair_rows)
        figure_run_specs = {
            "C4 자유작성": [
                ("실행 1", "ICb_gpt54mini_p50_live_seed1", "C4"),
                ("실행 2", "N5M_C4C6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_*", "C4"),
                ("실행 3", "C4R_C4C4RC6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_*", "C4"),
            ],
            "C6 행동크기 무지시": [
                ("실행 1", "ICb_gpt54mini_p50_live_seed1", "C6"),
                ("실행 2", "N5F_C4C6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_*", "C6"),
                ("실행 3", "N5M_C4C6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_*", "C6"),
            ],
            "C6 행동크기 2.00": [
                ("실행 1", "N5_C6_L1_2p00_ICb_gpt54mini_p50_main_seed1_*", "C6"),
                ("실행 2", "N5F_C4C6_L1_2p00_ICb_gpt54mini_p50_main_seed1_*", "C6"),
                ("실행 3", "N5M_C4C6_L1_2p00_ICb_gpt54mini_p50_main_seed1_*", "C6"),
            ],
            "C6 행동크기 1.27": [
                ("실행 1", "N5_C6_L1_1p27_ICb_gpt54mini_p50_main_seed1_*", "C6"),
                ("실행 2", "N5F_C4C6_L1_1p27_ICb_gpt54mini_p50_main_seed1_*", "C6"),
                ("실행 3", "N5M_C4C6_L1_1p27_ICb_gpt54mini_p50_main_seed1_*", "C6"),
            ],
        }
        figure_rows: list[dict[str, Any]] = []
        figure_sources: list[str] = []
        for condition, run_specs in figure_run_specs.items():
            condition_values: list[float] = []
            condition_rows: list[dict[str, Any]] = []
            for execution, run_pattern, policy in run_specs:
                score_path = exact_source(
                    "data/final_freeze/llm_runs/"
                    f"{run_pattern}/stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores.parquet"
                )
                if score_path is None:
                    condition_rows = []
                    break
                frame = pd.read_parquet(score_path)
                required = {"row_id", "policy", "mode", "delta_R_score_alpha"}
                if not required.issubset(frame.columns):
                    condition_rows = []
                    break
                selected = frame.loc[
                    frame["policy"].astype(str).eq(policy)
                    & frame["mode"].astype(str).eq("free_form_10d")
                ]
                if len(selected) != 575 or selected["row_id"].duplicated().any():
                    condition_rows = []
                    break
                value = float(pd.to_numeric(selected["delta_R_score_alpha"], errors="coerce").mean())
                if not math.isfinite(value):
                    condition_rows = []
                    break
                condition_values.append(value)
                condition_rows.append(
                    {
                        "condition": condition,
                        "execution": execution,
                        "condition_execution": f"{condition} · {execution}",
                        "n_firms": len(selected),
                        "mean_policy_value_alpha": value,
                        "source_run": score_path.parents[1].name,
                    }
                )
                figure_sources.append(_relative(score_path, project_root))
            if len(condition_rows) != 3:
                figure_rows = []
                break
            sample_mean = sum(condition_values) / len(condition_values)
            sample_sd = math.sqrt(
                sum((value - sample_mean) ** 2 for value in condition_values)
                / (len(condition_values) - 1)
            )
            for row in condition_rows:
                row["condition_mean"] = sample_mean
                row["condition_sd"] = sample_sd
            figure_rows.extend(condition_rows)
        if len(figure_rows) == 12:
            _write_csv(
                build_dir / "derived" / "repeatability_mean_policy_values.csv",
                [
                    "condition",
                    "execution",
                    "condition_execution",
                    "n_firms",
                    "mean_policy_value_alpha",
                    "condition_mean",
                    "condition_sd",
                    "source_run",
                ],
                figure_rows,
            )
        same = repeat_summaries["같은 파운데이션 모델 반복"]
        different = repeat_summaries["서로 다른 파운데이션 모델"]
        repeat_claim_specs = [
            ("NC0167-01", same["nearest_candidate_match_pct"], "PERCENT", 1, "세 동일모델 실행쌍의 최근접 후보행동 일치율 평균"),
            ("NC0167-02", same["rank_corr_alpha"], "SPEARMAN_RHO", 3, "세 동일모델 실행쌍의 기업별 정책가치 Spearman 순위상관 평균 (alpha)"),
            ("NC0167-03", same["rank_corr_beta"], "SPEARMAN_RHO", 3, "세 동일모델 실행쌍의 기업별 정책가치 Spearman 순위상관 평균 (beta)"),
            ("NC0167-04", same["rank_corr_gamma"], "SPEARMAN_RHO", 3, "세 동일모델 실행쌍의 기업별 정책가치 Spearman 순위상관 평균 (gamma)"),
            ("NC0168-01", different["nearest_candidate_match_pct"], "PERCENT", 1, "세 이종모델 실행쌍의 최근접 후보행동 일치율 평균"),
            ("NC0168-02", different["rank_corr_alpha"], "SPEARMAN_RHO", 3, "세 이종모델 실행쌍의 기업별 정책가치 Spearman 순위상관 평균 (alpha)"),
            ("NC0168-03", different["rank_corr_beta"], "SPEARMAN_RHO", 3, "세 이종모델 실행쌍의 기업별 정책가치 Spearman 순위상관 평균 (beta)"),
            ("NC0168-04", different["rank_corr_gamma"], "SPEARMAN_RHO", 3, "세 이종모델 실행쌍의 기업별 정책가치 Spearman 순위상관 평균 (gamma)"),
            (
                "NC0247-02",
                [
                    min(same["rank_corr_alpha"], same["rank_corr_beta"], same["rank_corr_gamma"]),
                    max(same["rank_corr_alpha"], same["rank_corr_beta"], same["rank_corr_gamma"]),
                ],
                "SPEARMAN_RHO_RANGE",
                3,
                "동일모델 세 실행쌍의 Oracle별 평균 순위상관 가운데 최솟값과 최댓값",
            ),
        ]
        repeat_sources_text = "\n".join(dict.fromkeys(repeat_input_paths))
        for token_id, value, unit, decimals, formula in repeat_claim_specs:
            records.append(
                {
                    "claim_token_id": token_id,
                    "metric": f"repeatability_{token_id.lower()}",
                    "computed_value": value,
                    "unit": unit,
                    "rounding_decimals": decimals,
                    "formula": formula,
                    "input_source_paths": repeat_sources_text,
                    "source_note": (
                        "각 실행의 C4 자유작성 575개 기업을 row_id로 대응하고 "
                        "행동은 candidate_id, 정책가치는 행동 점수-동일기업 무행동 점수로 계산"
                    ),
                }
            )

    # E2 is a preserved historical LLM experiment.  Recompute its contrasts
    # from the row-level Stage8 scores, while explicitly avoiding any claim
    # that the Stage7 API responses were freshly regenerated.
    e2_specs = (
        (
            "0p75",
            {
                "NC0245-02": ("self_review", "C4R", "C4"),
                "NC0245-04": ("reference_increment", "C6", "C4R"),
                "NC0245-06": ("total", "C6", "C4"),
            },
        ),
        (
            "unbounded",
            {
                "NC0246-01": ("self_review", "C4R", "C4"),
                "NC0246-02": ("reference_increment", "C6", "C4R"),
                "NC0246-03": ("total", "C6", "C4"),
            },
        ),
    )
    for budget, token_specs in e2_specs:
        score_path = exact_source(
            "data/final_freeze/llm_runs/"
            f"C4R_C4C4RC6_L1_{budget}_ICb_gpt54mini_p50_main_seed1_*/"
            "stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores.parquet"
        )
        if score_path is None:
            continue
        frame = pd.read_parquet(score_path)
        required = {"row_id", "policy", "mode", "delta_R_score_alpha"}
        if not required.issubset(frame.columns):
            continue
        frame = frame.loc[
            frame["policy"].astype(str).isin(["C4", "C4R", "C6"])
            & frame["mode"].astype(str).eq("free_form_10d")
        ].copy()
        if frame.duplicated(["row_id", "policy"]).any():
            continue
        wide = frame.pivot(
            index="row_id", columns="policy", values="delta_R_score_alpha"
        )
        if set(wide.columns) != {"C4", "C4R", "C6"} or wide.isna().any().any():
            continue
        # All three policies must be paired on the same firm universe.  The
        # identity is checked on every firm before taking any mean.
        self_review = wide["C4R"] - wide["C4"]
        reference_increment = wide["C6"] - wide["C4R"]
        total = wide["C6"] - wide["C4"]
        if (total - (self_review + reference_increment)).abs().max() > 1.0e-12:
            continue
        contrasts = {
            "self_review": self_review,
            "reference_increment": reference_increment,
            "total": total,
        }
        for row_id in wide.index:
            e2_detail_rows.append(
                {
                    "budget": budget,
                    "row_id": row_id,
                    "policy_value_C4_alpha": float(wide.loc[row_id, "C4"]),
                    "policy_value_C4R_alpha": float(wide.loc[row_id, "C4R"]),
                    "policy_value_C6_alpha": float(wide.loc[row_id, "C6"]),
                    "self_review_C4R_minus_C4": float(self_review.loc[row_id]),
                    "reference_increment_C6_minus_C4R": float(
                        reference_increment.loc[row_id]
                    ),
                    "total_C6_minus_C4": float(total.loc[row_id]),
                    "identity_error": float(
                        total.loc[row_id]
                        - self_review.loc[row_id]
                        - reference_increment.loc[row_id]
                    ),
                    "source_run": score_path.parents[1].name,
                }
            )
        for token_id, (metric, minuend, subtrahend) in token_specs.items():
            records.append(
                {
                    "claim_token_id": token_id,
                    "metric": f"e2_{budget}_{metric}_alpha",
                    "computed_value": float(contrasts[metric].mean()),
                    "unit": "ORACLE_SCORE_POINTS",
                    "rounding_decimals": 3,
                    "formula": (
                        f"575개 동일 기업 각각에서 {minuend} 정책가치 - "
                        f"{subtrahend} 정책가치를 계산한 뒤 평균 (alpha)"
                    ),
                    "input_source_paths": _relative(score_path, project_root),
                    "source_note": (
                        "보존된 E2 Stage8 점수 행에서 재계산; "
                        "새 LLM 응답을 생성했다는 뜻이 아님"
                    ),
                }
            )

    # The thesis describes NC0247-03 specifically as the E2-versus-E3
    # firm-level rank correlation of the reference increment C6-C4R.  Apply
    # that estimand literally for the two action-size conditions.  This is
    # intentionally kept separate from correlations for C4R-C4 or C6-C4,
    # because mixing the three contrasts changes the reported range.
    repeat_correlations: list[float] = []
    repeat_sources: list[str] = []
    for budget in ("0p75", "unbounded"):
        e2_path = exact_source(
            "data/final_freeze/llm_runs/"
            f"C4R_C4C4RC6_L1_{budget}_ICb_gpt54mini_p50_main_seed1_*/"
            "stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores.parquet"
        )
        e3_path = exact_source(
            "data/final_freeze/llm_runs/"
            f"C4RXJ_C4C4RC6_L1_{budget}_ICb_gpt54mini_p50_seed1_*/"
            "stage8_llm_multi_oracle_eval/llm_stage8_multi_oracle_scores.parquet"
        )
        if e2_path is None or e3_path is None:
            repeat_correlations = []
            break
        increments: list[Any] = []
        valid = True
        for path in (e2_path, e3_path):
            frame = pd.read_parquet(path)
            required = {"row_id", "policy", "mode", "delta_R_score_alpha"}
            if not required.issubset(frame.columns):
                valid = False
                break
            frame = frame.loc[
                frame["policy"].astype(str).isin(["C4R", "C6"])
                & frame["mode"].astype(str).eq("free_form_10d")
            ].copy()
            if frame.duplicated(["row_id", "policy"]).any():
                valid = False
                break
            wide = frame.pivot(
                index="row_id", columns="policy", values="delta_R_score_alpha"
            )
            if set(wide.columns) != {"C4R", "C6"} or wide.isna().any().any():
                valid = False
                break
            increments.append((wide["C6"] - wide["C4R"]).rename(path.name))
        if not valid:
            repeat_correlations = []
            break
        paired = pd.concat(increments, axis=1, join="inner").dropna()
        if len(paired) != 575:
            repeat_correlations = []
            break
        correlation = paired.iloc[:, 0].corr(paired.iloc[:, 1], method="spearman")
        if pd.isna(correlation):
            repeat_correlations = []
            break
        for row_id in paired.index:
            e2_e3_detail_rows.append(
                {
                    "budget": budget,
                    "row_id": row_id,
                    "e2_reference_increment_C6_minus_C4R": float(
                        paired.loc[row_id].iloc[0]
                    ),
                    "e3_reference_increment_C6_minus_C4R": float(
                        paired.loc[row_id].iloc[1]
                    ),
                    "e2_minus_e3": float(
                        paired.loc[row_id].iloc[0] - paired.loc[row_id].iloc[1]
                    ),
                    "e2_source_run": e2_path.parents[1].name,
                    "e3_source_run": e3_path.parents[1].name,
                }
            )
        repeat_correlations.append(float(correlation))
        repeat_sources.extend(
            [_relative(e2_path, project_root), _relative(e3_path, project_root)]
        )
    if len(repeat_correlations) == 2:
        records.append(
            {
                "claim_token_id": "NC0247-03",
                "metric": "e2_e3_reference_increment_rank_correlation_range_alpha",
                "computed_value": [min(repeat_correlations), max(repeat_correlations)],
                "unit": "SPEARMAN_RHO_RANGE",
                "rounding_decimals": 2,
                "formula": (
                    "행동크기 0.75와 무지시 조건별로 575개 동일 기업의 "
                    "C6-C4R 정책가치를 E2와 E3 사이에서 Spearman 순위상관으로 "
                    "계산한 뒤 두 상관계수의 최솟값과 최댓값"
                ),
                "input_source_paths": "\n".join(dict.fromkeys(repeat_sources)),
                "source_note": (
                    "논문 문장에 적힌 외부참조 증분효과(C6-C4R) 정의를 그대로 적용"
                ),
            }
        )

    if not records:
        return {}
    derived_dir = build_dir / "derived"
    if len(candidate_detail_rows) == 575:
        _write_csv(
            derived_dir / "candidate_iql_heuristic_firm_rows.csv",
            [
                "row_id",
                "candidate_iql_alpha",
                "weakest_component_alpha",
                "difference_alpha",
                "candidate_iql_beta",
                "weakest_component_beta",
                "difference_beta",
                "candidate_iql_gamma",
                "weakest_component_gamma",
                "difference_gamma",
            ],
            candidate_detail_rows,
        )
    if repeatability_detail_rows:
        _write_csv(
            derived_dir / "repeatability_firm_pair_rows.csv",
            [
                "comparison_group",
                "pair",
                "row_id",
                "candidate_id_left",
                "candidate_id_right",
                "nearest_candidate_match",
                "policy_value_alpha_left",
                "policy_value_alpha_right",
                "policy_value_alpha_difference",
                "policy_value_beta_left",
                "policy_value_beta_right",
                "policy_value_beta_difference",
                "policy_value_gamma_left",
                "policy_value_gamma_right",
                "policy_value_gamma_difference",
            ],
            repeatability_detail_rows,
        )
    if e2_detail_rows:
        _write_csv(
            derived_dir / "e2_c4_c4r_c6_firm_rows.csv",
            [
                "budget",
                "row_id",
                "policy_value_C4_alpha",
                "policy_value_C4R_alpha",
                "policy_value_C6_alpha",
                "self_review_C4R_minus_C4",
                "reference_increment_C6_minus_C4R",
                "total_C6_minus_C4",
                "identity_error",
                "source_run",
            ],
            e2_detail_rows,
        )
    if e2_e3_detail_rows:
        _write_csv(
            derived_dir / "e2_e3_reference_increment_firm_rows.csv",
            [
                "budget",
                "row_id",
                "e2_reference_increment_C6_minus_C4R",
                "e3_reference_increment_C6_minus_C4R",
                "e2_minus_e3",
                "e2_source_run",
                "e3_source_run",
            ],
            e2_e3_detail_rows,
        )
    derived_path = derived_dir / "reviewer_core_direct_calculations.csv"
    fields = [
        "claim_token_id",
        "metric",
        "computed_value",
        "unit",
        "rounding_decimals",
        "formula",
        "input_source_paths",
        "source_note",
    ]
    _write_csv(derived_path, fields, records)
    derived_sha256 = sha256_file(derived_path)
    overrides: dict[str, dict[str, Any]] = {}
    for source_row, record in enumerate(records):
        token_id = str(record["claim_token_id"])
        decimals = int(record["rounding_decimals"])
        raw_computed_value = record["computed_value"]
        rounded_computed_value = (
            [
                _rounded_claim_value(value, "IDENTITY", decimals)
                for value in raw_computed_value
            ]
            if isinstance(raw_computed_value, list)
            else _rounded_claim_value(raw_computed_value, "IDENTITY", decimals)
        )
        overrides[token_id] = {
            "computed_value": rounded_computed_value,
            "unit": record["unit"],
            "rounding": f"ROUND_HALF_UP({decimals}dp)",
            "formula": f"{record['formula']}; {record['source_note']}",
            "source_path": _relative(derived_path, project_root),
            "input_source_paths": record["input_source_paths"],
            "source_sha256": derived_sha256,
            "source_row": source_row + 2,
            "source_column": "computed_value",
            "source_json_or_config_key": f"claim_token_id={token_id}",
            "source_role": "RECOMPUTED_FROM_ACTUAL_SELECTED_RUN_ROWS",
            "producer": (
                "src/credit_recourse/reproduction/thesis_outputs/prepare.py::"
                "_derive_direct_reviewer_claim_values"
            ),
            "evidence_status": "RECOMPUTED_EXACT_SELECTED_RUN_SOURCE_CELL",
            "source_match_count": 1,
            "source_binding_kind": "DIRECT_SOURCE_CELL",
            "source_data_row_index": source_row,
            "source_column_index": fields.index("computed_value"),
            "source_table_index": 0,
            "display_transform": f"IDENTITY; ROUND_HALF_UP({decimals}dp)",
            "resolution_rule": "DIRECT_REVIEWER_RECOMPUTATION:ACTUAL_ROW_LEVEL_OUTPUTS",
        }
    return overrides


def _attach_repeatability_outputs_to_thesis_items(
    item_payloads: Sequence[dict[str, Any]],
    selected_run: SelectedRun,
    build_dir: Path,
    project_root: Path,
) -> None:
    """Attach freshly calculated Section 5.5 table/figure data to their workbooks."""

    def selector_output(
        selector_id: str,
        table: dict[str, Any],
        *,
        row_keys: Sequence[str],
        value_columns: Sequence[str],
    ) -> dict[str, Any]:
        columns = [str(value) for value in table.get("columns", [])]
        rows = list(table.get("rows", []))
        bindings: list[list[dict[str, Any]]] = []
        for row_index, values in enumerate(rows):
            bindings.append(
                [
                    {
                        "kind": "DIRECT_SOURCE_CELL",
                        "value": values[column_index] if column_index < len(values) else None,
                        "source_table_index": 0,
                        "source_data_row_index": row_index,
                        "source_column_index": column_index,
                        "source_path": table.get("path", ""),
                        "logical_path": table.get("logical_path", ""),
                        "source_row": row_index + 2,
                        "source_column": column,
                    }
                    for column_index, column in enumerate(columns)
                ]
            )
        return {
            "selector_id": selector_id,
            "status": "SELECTOR_COMPUTED",
            "reason": "",
            "source_pattern": table.get("path", ""),
            "source_paths": [table.get("path", "")],
            "columns": columns,
            "rows": rows,
            "cell_bindings": bindings,
            "row_keys": list(row_keys),
            "value_columns": list(value_columns),
            "filters": [],
            "actual_row_count": len(rows),
            "duplicate_key_count": 0,
        }

    by_id = {str(item.get("item_id", "")): item for item in item_payloads}
    pair_path = build_dir / "derived" / "repeatability_pair_calculations.csv"
    if pair_path.is_file() and "T5-8" in by_id:
        table = _source_table_payload(pair_path, project_root, selected_run)
        item = by_id["T5-8"]
        item.update(
            {
                "calculation": "PAIRWISE_ROW_LEVEL_RECOMPUTATION",
                "formula": (
                    "575개 기업 C4 자유작성 결과를 실행쌍별 row_id로 대응; "
                    "최근접 후보행동 일치율과 Oracle별 정책가치 Spearman 순위상관을 계산"
                ),
                "producer": (
                    "src/credit_recourse/reproduction/thesis_outputs/prepare.py::"
                    "_derive_direct_reviewer_claim_values"
                ),
                "source_tables": [table],
                "selector_status": "ACTIVE",
                "selector_reason": "",
                "selector_outputs": [
                    selector_output(
                        "repeatability_pair_metrics",
                        table,
                        row_keys=("comparison_group", "row_kind", "pair"),
                        value_columns=(
                            "nearest_candidate_match_pct",
                            "rank_corr_alpha",
                            "rank_corr_beta",
                            "rank_corr_gamma",
                        ),
                    )
                ],
                "paper_selector": "repeatability_pair_metrics",
                "evidence_status": "RECOMPUTED_FROM_ACTUAL_RUN_ROWS",
                "provenance": "PRESERVED_STAGE7_EVIDENCE_RECALCULATED",
            }
        )

    figure_path = build_dir / "derived" / "repeatability_mean_policy_values.csv"
    if figure_path.is_file() and "F6" in by_id:
        table = _source_table_payload(figure_path, project_root, selected_run)
        item = by_id["F6"]
        item.update(
            {
                "calculation": "RUN_LEVEL_MEAN_AND_SAMPLE_SD",
                "formula": (
                    "각 실행에서 575개 기업의 Oracle-alpha 정책가치를 평균하고, "
                    "같은 조건의 세 독립 실행 사이 표본 표준편차를 계산"
                ),
                "producer": (
                    "src/credit_recourse/reproduction/thesis_outputs/prepare.py::"
                    "_derive_direct_reviewer_claim_values"
                ),
                "source_tables": [table],
                "selector_status": "ACTIVE",
                "selector_reason": "",
                "selector_outputs": [
                    selector_output(
                        "repeatability_mean_policy_values",
                        table,
                        row_keys=("condition", "execution"),
                        value_columns=(
                            "mean_policy_value_alpha",
                            "condition_mean",
                            "condition_sd",
                        ),
                    )
                ],
                "paper_selector": "repeatability_mean_policy_values",
                "chart_spec": {
                    "status": "NATIVE_CHART",
                    "selector_id": "repeatability_mean_policy_values",
                    "x": "condition_execution",
                    "series": ["mean_policy_value_alpha"],
                    "chart_type": "bar",
                    "max_rows": 12,
                },
                "evidence_status": "RECOMPUTED_FROM_ACTUAL_RUN_ROWS",
                "provenance": "PRESERVED_STAGE7_EVIDENCE_RECALCULATED",
            }
        )


def _attach_sample_flow_output_to_thesis_items(
    item_payloads: Sequence[dict[str, Any]],
    selected_run: SelectedRun,
    build_dir: Path,
    project_root: Path,
) -> None:
    """Make Table 3-1 use the raw/selected-run recalculation, not printed values."""
    by_id = {str(item.get("item_id", "")): item for item in item_payloads}
    item = by_id.get("T3-1")
    source_path = build_dir / "derived" / "table_3_1_sample_flow.csv"
    if item is None or not source_path.is_file():
        return
    table = _source_table_payload(source_path, project_root, selected_run)
    columns = [str(value) for value in table.get("columns", [])]
    rows = list(table.get("rows", []))
    bindings = [
        [
            {
                "kind": "DIRECT_SOURCE_CELL",
                "value": values[column_index] if column_index < len(values) else None,
                "source_table_index": 0,
                "source_data_row_index": row_index,
                "source_column_index": column_index,
                "source_path": table.get("path", ""),
                "logical_path": table.get("logical_path", ""),
                "source_row": row_index + 2,
                "source_column": column,
            }
            for column_index, column in enumerate(columns)
        ]
        for row_index, values in enumerate(rows)
    ]
    selector_id = "table_3_1_sample_flow"
    item.update(
        {
            "calculation": "SOURCE_TABLE_RESHAPE",
            "formula": (
                "원본 Stage0 manifest에서 producer가 실제로 읽은 재무제표 행을 세고, "
                "선택 실행의 Stage1·전이·DEV·OOT·Stage6 산출물에서 각 분모를 다시 계산. "
                "표본 범위는 각 원본 producer의 산출물과 동일"
            ),
            "producer": (
                "src/credit_recourse/reproduction/thesis_outputs/prepare.py::"
                "_build_sample_flow"
            ),
            "source_tables": [table],
            "selector_status": "ACTIVE",
            "selector_reason": "",
            "selector_outputs": [
                {
                    "selector_id": selector_id,
                    "status": "SELECTOR_COMPUTED",
                    "reason": "",
                    "source_pattern": table.get("path", ""),
                    "source_paths": [table.get("path", "")],
                    "columns": columns,
                    "rows": rows,
                    "cell_bindings": bindings,
                    "row_keys": ["선정 단계"],
                    "value_columns": ["논문 표시값", "재계산값", "차이"],
                    "filters": [],
                    "actual_row_count": len(rows),
                    "duplicate_key_count": 0,
                }
            ],
            "paper_selector": selector_id,
            "evidence_status": "RECOMPUTED_FROM_ORIGINAL_PRODUCER_OUTPUTS",
            "provenance": "SELECTED_RUN_RECALCULATED",
        }
    )

def _computed_claim_lineage_errors(row: dict[str, Any]) -> list[str]:
    required_text = (
        "formula",
        "source_path",
        "source_sha256",
        "source_row",
        "source_column",
        "producer",
        "run_id",
        "resolution_rule",
    )
    errors = [field for field in required_text if not str(row.get(field, "")).strip()]
    source_sha256 = str(row.get("source_sha256", "")).strip().lower()
    if source_sha256 and not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        errors.append("source_sha256_invalid")
    if type(row.get("source_match_count")) is not int or row.get("source_match_count") != 1:
        errors.append("source_match_count_not_one")
    for field in ("source_data_row_index", "source_column_index", "source_table_index"):
        value = row.get(field, "")
        valid_index = (
            isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ) or (
            isinstance(value, str) and re.fullmatch(r"0|[1-9][0-9]*", value.strip()) is not None
        )
        if not valid_index:
            errors.append(field)
    if row.get("source_binding_kind") != "DIRECT_SOURCE_CELL":
        errors.append("direct_source_binding_evidence")
    if not str(row.get("resolution_rule", "")).startswith(
        ("EXPLICIT_CLAIM_SELECTOR:", "DIRECT_REVIEWER_RECOMPUTATION:")
    ):
        errors.append("explicit_resolution_rule")
    return list(dict.fromkeys(errors))


def prepare_build(
    project_root: Path,
    thesis_docx: Path,
    contract_path: Path,
    selected_run: SelectedRun,
    output_root: Path,
) -> dict[str, Any]:
    inventory = extract_thesis_inventory(thesis_docx)
    errors = validate_canonical_inventory(inventory)
    if errors:
        raise RuntimeError("Canonical thesis inventory failed: " + "; ".join(errors))
    contract = _load_json_object(contract_path)
    contract_items = contract.get("items")
    if not isinstance(contract_items, dict):
        raise ValueError(f"Contract has no items object: {contract_path}")
    inventory_ids = {str(item["item_id"]) for item in inventory["items"]}
    contract_ids = set(contract_items)
    if inventory_ids != contract_ids:
        raise RuntimeError(
            "Contract/inventory IDs differ; "
            f"missing_contract={sorted(inventory_ids - contract_ids)}, "
            f"extra_contract={sorted(contract_ids - inventory_ids)}"
        )
    selector_contract_value = str(contract.get("selector_contract_path", "")).strip()
    if not selector_contract_value:
        raise RuntimeError("thesis_output_contract.json must declare selector_contract_path")
    selector_contract_path = (
        project_root / Path(*selector_contract_value.replace("\\", "/").split("/"))
    ).resolve()
    selector_contract = _load_json_object(selector_contract_path)
    if selector_contract.get("schema_version") != "thesis_output_selectors_v11":
        raise RuntimeError(f"Unsupported selector contract: {selector_contract_path}")
    selector_items = selector_contract.get("items")
    if not isinstance(selector_items, dict):
        raise RuntimeError(f"Selector contract has no items object: {selector_contract_path}")
    claim_contract_value = str(contract.get("numeric_claim_selector_contract_path", "")).strip()
    if not claim_contract_value:
        raise RuntimeError(
            "thesis_output_contract.json must declare numeric_claim_selector_contract_path"
        )
    claim_contract_path = (
        project_root / Path(*claim_contract_value.replace("\\", "/").split("/"))
    ).resolve()
    claim_contract = _load_json_object(claim_contract_path)
    claim_rules = _claim_rule_index(claim_contract)
    reviewer_contract_value = str(contract.get("reviewer_claim_contract_path", "")).strip()
    if not reviewer_contract_value:
        raise RuntimeError(
            "thesis_output_contract.json must declare reviewer_claim_contract_path"
        )
    reviewer_contract_path = (
        project_root / Path(*reviewer_contract_value.replace("\\", "/").split("/"))
    ).resolve()
    reviewer_contract = _load_json_object(reviewer_contract_path)
    empirical_or_mixed_ids = {
        item_id
        for item_id, rule in contract_items.items()
        if str(rule.get("class", "")) in {"EMPIRICAL", "MIXED"}
    }
    if set(selector_items) != empirical_or_mixed_ids:
        raise RuntimeError(
            "Selector contract must cover every and only EMPIRICAL/MIXED terminal; "
            f"missing={sorted(empirical_or_mixed_ids - set(selector_items))}, "
            f"extra={sorted(set(selector_items) - empirical_or_mixed_ids)}"
        )

    build_dir = output_root / "_build" / selected_run.run_id
    registry_dir = build_dir / "registry"
    build_dir.mkdir(parents=True, exist_ok=True)
    sample_flow = _build_sample_flow(project_root, selected_run, build_dir)
    item_payloads: list[dict[str, Any]] = []
    source_file_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    role_rows: list[dict[str, Any]] = []
    calculation_rows: list[dict[str, Any]] = []
    row_level_calculation_rows: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    row_level_check_cache: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    non_empirical_rows: list[dict[str, Any]] = []
    selector_registry_rows: list[dict[str, Any]] = []

    for item in inventory["items"]:
        item_id = str(item["item_id"])
        rule = contract_items[item_id]
        item_class = str(rule.get("class", item.get("evidence_class_hint", "MIXED")))
        calculation = str(rule.get("calculation", "UNRESOLVED"))
        provenance = str(rule.get("provenance", ""))
        selector_rule = selector_items.get(item_id, {})
        selector_patterns = [
            str(spec.get("source_pattern", ""))
            for spec in selector_rule.get("selectors", [])
            if isinstance(spec, dict) and spec.get("source_pattern")
        ]
        source_patterns = list(
            dict.fromkeys(
                [str(value) for value in rule.get("sources", [])] + selector_patterns
            )
        )
        sources, resolution_audit = resolve_sources(
            source_patterns, project_root, selected_run
        )
        source_tables = [
            _source_table_payload(path, project_root, selected_run) for path in sources
        ]
        lineage_provenance = sorted(
            {
                str(table.get("lineage_provenance", ""))
                for table in source_tables
                if table.get("lineage_provenance")
            }
        )
        status = _item_status(item_class, calculation, source_tables, provenance)
        formula = str(contract.get("calculation_vocabulary", {}).get(calculation, calculation))
        selector_outputs = _build_selector_outputs(item_id, selector_rule, source_tables)
        selector_status = str(selector_rule.get("status", "UNRESOLVED"))
        selector_reason = str(selector_rule.get("reason", ""))
        if selector_status == "ACTIVE":
            selector_failures = [
                output for output in selector_outputs if output.get("status") != "SELECTOR_COMPUTED"
            ]
            if not selector_outputs or selector_failures:
                status = "UNRESOLVED_EXPLICIT_SELECTOR_FAILED"
                selector_reason = "; ".join(
                    str(output.get("reason", "selector did not compute"))
                    for output in selector_failures
                ) or "active selector produced no output"
            else:
                selector_status = "SELECTOR_COMPUTED"
        elif item_class in {"EMPIRICAL", "MIXED"} and provenance != "PRESERVED_EVIDENCE_PRODUCER_SNAPSHOT_GAP":
            status = "UNRESOLVED_EXPLICIT_SELECTOR"
        if selector_outputs:
            for output in selector_outputs:
                selector_registry_rows.append(
                    {
                        "item_id": item_id,
                        "selector_id": output.get("selector_id", ""),
                        "contract_status": selector_rule.get("status", ""),
                        "execution_status": output.get("status", ""),
                        "source_pattern": output.get("source_pattern", ""),
                        "source_paths": "; ".join(output.get("source_paths", [])),
                        "row_keys": "; ".join(output.get("row_keys", [])),
                        "row_count_contract": json.dumps(output.get("row_count", {}), sort_keys=True),
                        "actual_row_count": output.get("actual_row_count", len(output.get("rows", []))),
                        "key_policy": output.get("key_policy", ""),
                        "expected_duplicate_count": output.get("expected_duplicate_count", 0),
                        "duplicate_key_count": output.get("duplicate_key_count", 0),
                        "value_columns": "; ".join(output.get("value_columns", [])),
                        "filters": json.dumps(output.get("filters", []), ensure_ascii=False, sort_keys=True),
                        "aggregation": output.get("aggregation", ""),
                        "reshape": output.get("reshape", ""),
                        "selected_row_count": len(output.get("rows", [])),
                        "reason": output.get("reason", ""),
                        "producer": rule.get("producer", ""),
                        "run_id": selected_run.run_id,
                    }
                )
        else:
            selector_registry_rows.append(
                {
                    "item_id": item_id,
                    "selector_id": "",
                    "contract_status": selector_rule.get("status", ""),
                    "execution_status": "UNRESOLVED_EXPLICIT_SELECTOR",
                    "source_pattern": "",
                    "source_paths": "",
                    "row_keys": "",
                    "row_count_contract": "",
                    "actual_row_count": 0,
                    "key_policy": "",
                    "expected_duplicate_count": 0,
                    "duplicate_key_count": 0,
                    "value_columns": "",
                    "filters": "",
                    "aggregation": "",
                    "reshape": "",
                    "selected_row_count": 0,
                    "reason": selector_rule.get("reason", ""),
                    "producer": rule.get("producer", ""),
                    "run_id": selected_run.run_id,
                }
            )
        policy_checks = _policy_value_checks(source_tables) if calculation == "POLICY_VALUE_SAME_FIRM" else []
        same_firm_pairing_summaries: list[dict[str, Any]] = []
        if calculation == "POLICY_VALUE_SAME_FIRM":
            for table in source_tables:
                absolute_path = Path(str(table.get("absolute_path", "")))
                if absolute_path.name not in {
                    "multi_oracle_policy_eval.parquet",
                    "llm_stage8_multi_oracle_scores.parquet",
                }:
                    continue
                cache_key = str(absolute_path.resolve())
                if cache_key not in row_level_check_cache:
                    row_level_check_cache[cache_key] = _same_firm_row_level_checks(
                        absolute_path, project_root
                    )
                pairing_summary, row_checks = row_level_check_cache[cache_key]
                same_firm_pairing_summaries.append(pairing_summary)
                for check in row_checks:
                    key = (
                        str(check.get("source_path", "")),
                        str(check.get("row_id", "")),
                        str(check.get("policy", "")),
                        str(check.get("mode", "")),
                        str(check.get("oracle", "")),
                    )
                    if key not in row_level_calculation_rows:
                        row_level_calculation_rows[key] = {
                            **check,
                            "run_id": selected_run.run_id,
                            "item_ids": {item_id},
                        }
                    else:
                        row_level_calculation_rows[key]["item_ids"].add(item_id)
        if policy_checks and any(
            str(check["check_status"]).startswith("FAIL") for check in policy_checks
        ):
            status = "UNRESOLVED_POLICY_VALUE_IDENTITY_FAILED"
        cell_bindings: list[dict[str, Any]] = []
        for row_index, row in enumerate(item.get("reference_cells") or []):
            for column_index, value in enumerate(row):
                text_value = str(value)
                role = _reference_cell_role(item_class, row_index, text_value)
                if role == "LABEL":
                    binding = {
                        "thesis_display_value": text_value,
                        "computed_value": text_value,
                        "unit": "LABEL",
                        "rounding": "",
                        "formula": "LABEL_SCAFFOLD_FROM_CANONICAL_THESIS",
                        "source_path": "",
                        "source_sha256": "",
                        "source_row": "",
                        "source_column": "",
                        "source_json_or_config_key": "",
                        "source_role": "THESIS_LABEL_SCAFFOLD",
                        "producer": "canonical thesis OOXML inventory",
                        "evidence_status": "LABEL_FROM_THESIS_REFERENCE",
                        "source_match_count": 0,
                        "source_data_row_index": "",
                        "source_column_index": "",
                        "source_table_index": "",
                        "display_transform": "",
                    }
                else:
                    binding = {
                        "thesis_display_value": text_value,
                        "computed_value": "",
                        "unit": _display_token_spec(text_value)["unit"],
                        "rounding": _display_token_spec(text_value)["rounding"],
                        "formula": "NO_VALUE_BASED_REVERSE_MATCH; SEE EXPLICIT SELECTOR OUTPUT",
                        "source_path": "",
                        "source_sha256": "",
                        "source_row": "",
                        "source_column": "",
                        "source_json_or_config_key": "",
                        "source_role": "THESIS_COMPARISON_REFERENCE_ONLY",
                        "producer": "",
                        "evidence_status": "UNRESOLVED_NO_EXPLICIT_PAPER_CELL_SELECTOR",
                        "source_match_count": 0,
                        "source_data_row_index": "",
                        "source_column_index": "",
                        "source_table_index": "",
                        "display_transform": "",
                    }
                cell_binding = {
                    "item_id": item_id,
                    "row": row_index + 1,
                    "column": column_index + 1,
                    "role": role,
                    "run_id": selected_run.run_id,
                    **binding,
                }
                cell_bindings.append(cell_binding)
                role_rows.append(
                    {
                        "item_id": item_id,
                        "row": row_index + 1,
                        "column": column_index + 1,
                        "thesis_reference_value": text_value,
                        "role": role,
                        "resolution": binding["evidence_status"],
                        "reproduced_value": binding["computed_value"],
                        "source_path": binding["source_path"],
                        "source_row": binding["source_row"],
                        "source_column": binding["source_column"],
                        "formula": binding["formula"],
                        "run_id": selected_run.run_id,
                    }
                )
        payload = {
            **item,
            "class": item_class,
            "calculation": calculation,
            "formula": formula,
            "producer": str(rule.get("producer", "")),
            "contract_note": str(rule.get("note", "")),
            "provenance": provenance or "; ".join(lineage_provenance) or status,
            "lineage_provenance": lineage_provenance,
            "evidence_status": status,
            "source_tables": source_tables,
            "source_resolution_audit": resolution_audit,
            "selector_status": selector_status,
            "selector_reason": selector_reason,
            "selector_outputs": selector_outputs,
            "paper_selector": str(selector_rule.get("paper_selector", "")),
            "policy_value_checks": policy_checks,
            "same_firm_pairing_summaries": same_firm_pairing_summaries,
            "paper_table_cell_bindings": cell_bindings,
            "chart_spec": (
                selector_rule.get("chart")
                if item["kind"] == "figure" and selector_rule.get("chart")
                else (
                    {
                        "status": "EDITABLE_SPEC_ONLY",
                        "reason": "Editable diagram item based on the listed evidence paths and layout specification.",
                    }
                    if item["kind"] == "figure"
                    else None
                )
            ),
            "run_id": selected_run.run_id,
            "run_mode": selected_run.mode,
            "run_documented_gaps": list(selected_run.documented_gaps),
        }
        item_payloads.append(payload)
        evidence_rows.append(
            {
                "item_id": item_id,
                "kind": item["kind"],
                "caption": item["caption"],
                "class": item_class,
                "calculation": calculation,
                "formula": formula,
                "producer": rule.get("producer", ""),
                "run_id": selected_run.run_id,
                "run_mode": selected_run.mode,
                "documented_gaps": "; ".join(selected_run.documented_gaps),
                "status": status,
                "provenance": provenance or "; ".join(lineage_provenance) or status,
                "source_count": len(source_tables),
                "source_paths": "; ".join(str(table["path"]) for table in source_tables),
                "note": rule.get("note", ""),
            }
        )
        for table in source_tables:
            source_file_rows.append(
                {
                    "item_id": item_id,
                    "path": table["path"],
                    "logical_path": table["logical_path"],
                    "sha256": table["sha256"],
                    "size_bytes": table["size_bytes"],
                    "source_role": table["source_role"],
                    "lineage_provenance": table["lineage_provenance"],
                    "read_status": table["read_status"],
                    "read_error": table["read_error"],
                    "columns": "; ".join(str(value) for value in table["columns"]),
                    "row_count_in_workbook": table["row_count_in_workbook"],
                }
            )
        if item["kind"] == "figure":
            role_rows.append(
                {
                    "item_id": item_id,
                    "row": "",
                    "column": "",
                    "thesis_reference_value": item["caption"],
                    "role": "SERIES_" + item_class,
                    "resolution": status,
                    "reproduced_value": "",
                    "source_path": "; ".join(str(table["path"]) for table in source_tables),
                    "formula": formula,
                    "run_id": selected_run.run_id,
                }
            )
        for check in policy_checks:
            calculation_rows.append({"item_id": item_id, "run_id": selected_run.run_id, **check})
        if item_class in {"NON_EMPIRICAL", "MIXED"}:
            non_empirical_rows.append(
                {
                    "record_type": "TERMINAL_ITEM",
                    "item_id": item_id,
                    "claim_id": "",
                    "token_index": "",
                    "caption": item["caption"],
                    "class": item_class,
                    "design_config_sources": "; ".join(
                        str(table["path"])
                        for table in source_tables
                        if table["source_role"] == "DESIGN_CONFIG_EVIDENCE"
                    ),
                    "producer": rule.get("producer", ""),
                    "basis": rule.get("note", ""),
                    "status": status,
                    "empirical_claim": "NO" if item_class == "NON_EMPIRICAL" else "CELL_OR_SERIES_DEPENDENT",
                }
            )

    selector_registry_rows = [
        row for row in selector_registry_rows if row.get("item_id") in selector_items
    ]
    calculation_rows.extend(
        {
            **row,
            "item_ids": "; ".join(sorted(row["item_ids"])),
        }
        for row in row_level_calculation_rows.values()
    )
    _attach_sample_flow_output_to_thesis_items(
        item_payloads, selected_run, build_dir, project_root
    )
    item_map = {str(item["item_id"]): item for item in item_payloads}
    claim_rows = [
        row
        for claim in inventory["numeric_claim_candidates"]
        for row in _claim_token_rows(
            claim,
            item_map,
            claim_rules,
            selected_run.run_id,
            selected_run.documented_gaps,
        )
    ]
    claim_value_overrides = _derive_action_semantics_claim_values(
        selected_run, build_dir, project_root
    )
    claim_value_overrides.update(
        _derive_direct_reviewer_claim_values(selected_run, build_dir, project_root)
    )
    reviewer_calculation_tables: list[dict[str, Any]] = []
    for filename, sheet_name, description in (
        (
            "candidate_iql_heuristic_firm_rows.csv",
            "CANDIDATE_IQL_ROWS",
            "575개 동일기업의 Candidate-IQL·휴리스틱 정책가치와 기업별 차이",
        ),
        (
            "action_semantics_firm_rows.csv",
            "ACTION_ROWS",
            "Stage7 원응답 제안행동과 실제 적용행동의 기업별 L1·clipping 계산",
        ),
        (
            "repeatability_firm_pair_rows.csv",
            "REPEATABILITY_ROWS",
            "실행쌍별 575개 동일기업의 후보행동 일치와 정책가치 대응",
        ),
        (
            "e2_c4_c4r_c6_firm_rows.csv",
            "C4_C4R_C6_ROWS",
            "두 행동크기 조건의 C4·C4R·C6 동일기업 대비",
        ),
        (
            "e2_e3_reference_increment_firm_rows.csv",
            "E2_E3_ROWS",
            "E2와 E3의 동일기업 C6-C4R 외부참조 증분 대응",
        ),
    ):
        path = build_dir / "derived" / filename
        if not path.is_file():
            continue
        table = _source_table_payload(path, project_root, selected_run)
        table["sheet_name"] = sheet_name
        table["description"] = description
        reviewer_calculation_tables.append(table)
    _attach_repeatability_outputs_to_thesis_items(
        item_payloads, selected_run, build_dir, project_root
    )
    for claim_row in claim_rows:
        override = claim_value_overrides.get(str(claim_row.get("claim_token_id", "")))
        if override is not None:
            claim_row.update(override)
    included_claim_rows = [
        row for row in claim_rows if row.get("evidence_status") != "EXCLUDED_LITERATURE_CONTEXT"
    ]
    reviewer_core_claims = _reviewer_core_claim_rows(
        reviewer_contract, included_claim_rows
    )
    reviewer_token_ids = {
        str(token_id)
        for definition in reviewer_contract.get("claims", [])
        if isinstance(definition, dict)
        for token_id in definition.get("token_ids", [])
    }
    for item in item_payloads:
        item_id = str(item.get("item_id", ""))
        linked_rows: list[dict[str, Any]] = []
        for row in included_claim_rows:
            if (
                str(row.get("claim_token_id", "")) not in reviewer_token_ids
                or item_id not in row.get("nearest_item_ids", [])
            ):
                continue
            match = _computed_value_matches_thesis_display(row)
            linked_rows.append(
                {
                    "claim_token_id": row.get("claim_token_id", ""),
                    "thesis_claim": row.get("text", ""),
                    "thesis_value": row.get("thesis_display_value", ""),
                    "recomputed_value": row.get("computed_value", "") if match is not None else "계산 불가",
                    "result": (
                        "논문값과 일치"
                        if match is True
                        else "차이 있음"
                        if match is False
                        else "현재 산출물로 계산할 수 없음"
                    ),
                    "calculation": row.get("formula", ""),
                    "source_outputs": "\n".join(
                        value
                        for value in dict.fromkeys(
                            line.strip()
                            for field in ("input_source_paths", "source_path")
                            for line in str(row.get(field, "")).splitlines()
                        )
                        if value
                    ),
                    "producer": row.get("producer", ""),
                    "run_id": row.get("run_id", ""),
                }
            )
        item["reviewer_claim_rows"] = linked_rows
    non_empirical_rows.extend(
        {
            "record_type": "NUMERIC_CLAIM_TOKEN",
            "item_id": "; ".join(row.get("nearest_item_ids", [])),
            "claim_id": row.get("claim_id", ""),
            "token_index": row.get("token_index", ""),
            "caption": row.get("text", ""),
            "class": "NON_EMPIRICAL",
            "design_config_sources": row.get("source_path", ""),
            "producer": row.get("producer", ""),
            "basis": row.get("formula", "CONFIG_REGISTRY; NO_EMPIRICAL_CALCULATION"),
            "status": row.get("evidence_status", ""),
            "empirical_claim": "NO",
        }
        for row in included_claim_rows
        if row.get("evidence_status") == "NON_EMPIRICAL_CONFIG_REGISTRY"
    )

    table_workbook_count = sum(
        item.get("kind") in {"table", "unnumbered_table"}
        for item in item_payloads
    )
    figure_workbook_count = sum(
        item.get("kind") == "figure" for item in item_payloads
    )

    payload = {
        "schema_version": "thesis_output_build_payload_v11",
        "created_at_utc": _utc_now(),
        "project_root": str(project_root),
        "output_root": str(output_root),
        "build_dir": str(build_dir),
        "thesis": inventory["summary"],
        "selected_run": {
            "run_id": selected_run.run_id,
            "mode": selected_run.mode,
            "status": selected_run.status,
            "manifest_path": _relative(selected_run.manifest_path, project_root),
            "manifest_sha256": sha256_file(selected_run.manifest_path),
            "final_freeze_root": _relative(selected_run.final_freeze_root, project_root),
            "final_freeze_overlay_root": (
                _relative(selected_run.final_freeze_overlay_root, project_root)
                if selected_run.final_freeze_overlay_root is not None
                else None
            ),
            "analysis_root": _relative(selected_run.analysis_root, project_root),
            "documented_gaps": list(selected_run.documented_gaps),
        },
        "contract": {
            "path": _relative(contract_path, project_root),
            "sha256": sha256_file(contract_path),
            "schema_version": contract.get("schema_version", ""),
            "selector_path": _relative(selector_contract_path, project_root),
            "selector_sha256": sha256_file(selector_contract_path),
            "selector_schema_version": selector_contract.get("schema_version", ""),
            "numeric_claim_selector_path": _relative(claim_contract_path, project_root),
            "numeric_claim_selector_sha256": sha256_file(claim_contract_path),
            "numeric_claim_selector_schema_version": claim_contract.get("schema_version", ""),
            "reviewer_claim_path": _relative(reviewer_contract_path, project_root),
            "reviewer_claim_sha256": sha256_file(reviewer_contract_path),
            "reviewer_claim_schema_version": reviewer_contract.get("schema_version", ""),
        },
        "workbook_plan": {
            "index_workbooks": 2,
            "table_workbooks": table_workbook_count,
            "figure_workbooks": figure_workbook_count,
            "total_workbooks": 2 + table_workbook_count + figure_workbook_count,
        },
        "items": item_payloads,
        "numeric_claims": included_claim_rows,
        "reviewer_core_claims": reviewer_core_claims,
        "reviewer_calculation_tables": reviewer_calculation_tables,
        "sample_flow": sample_flow,
        "excluded_numeric_context": [
            row for row in claim_rows if row.get("evidence_status") == "EXCLUDED_LITERATURE_CONTEXT"
        ],
        "non_empirical_registry": non_empirical_rows,
        "cell_series_role_registry": role_rows,
    }
    _write_json(build_dir / "build_payload.json", payload)
    _write_json(registry_dir / "THESIS_INVENTORY.json", inventory)
    _write_json(registry_dir / "ITEM_EVIDENCE_REGISTRY.json", item_payloads)
    _write_json(registry_dir / "NUMERIC_CLAIMS.json", included_claim_rows)
    _write_json(registry_dir / "CORE_THESIS_CLAIMS.json", reviewer_core_claims)
    _write_json(registry_dir / "SAMPLE_FLOW.json", sample_flow)

    inventory_rows = [
        {key: value for key, value in item.items() if key != "reference_cells"}
        for item in inventory["items"]
    ]
    _write_csv(
        registry_dir / "THESIS_INVENTORY.csv",
        [
            "item_id",
            "kind",
            "thesis_number",
            "caption",
            "title",
            "pdf_page",
            "printed_page",
            "section",
            "paragraph_index",
            "body_index",
            "body_table_index",
            "table_rows",
            "table_columns",
            "reference_role",
            "evidence_class_hint",
        ],
        inventory_rows,
    )
    claim_fields = [
        "claim_token_id",
        "claim_id",
        "token_index",
        "paragraph_index",
        "pdf_page",
        "printed_page",
        "section",
        "text",
        "thesis_display_value",
        "computed_value",
        "unit",
        "rounding",
        "classification_hint",
        "nearest_item_ids",
        "run_id",
        "run_documented_gaps",
        "evidence_status",
        "resolution_rule",
        "formula",
        "source_path",
        "source_sha256",
        "source_row",
        "source_column",
        "source_json_or_config_key",
        "source_role",
        "producer",
        "source_match_count",
        "source_binding_kind",
        "source_data_row_index",
        "source_column_index",
        "source_table_index",
        "display_transform",
    ]
    serialized_claims = [
        {
            **row,
            "nearest_item_ids": "; ".join(row.get("nearest_item_ids", [])),
            "computed_value": (
                "; ".join(str(value) for value in row.get("computed_value", []))
                if isinstance(row.get("computed_value"), list)
                else row.get("computed_value", "")
            ),
        }
        for row in included_claim_rows
    ]
    _write_csv(registry_dir / "NUMERIC_CLAIMS.csv", claim_fields, serialized_claims)
    _write_csv(
        registry_dir / "CORE_THESIS_CLAIMS.csv",
        [
            "order",
            "claim_key",
            "category",
            "title",
            "printed_page",
            "thesis_claim",
            "thesis_values",
            "recomputed_values",
            "differences",
            "result",
            "calculation",
            "source_outputs",
            "producer",
            "run_id",
            "unavailable_reason",
            "token_ids",
            "computed_value_count",
            "value_count",
        ],
        reviewer_core_claims,
    )
    _write_csv(
        registry_dir / "ITEM_EVIDENCE_REGISTRY.csv",
        [
            "item_id",
            "kind",
            "caption",
            "class",
            "calculation",
            "formula",
            "producer",
            "run_id",
            "run_mode",
            "documented_gaps",
            "status",
            "provenance",
            "source_count",
            "source_paths",
            "note",
        ],
        evidence_rows,
    )
    _write_csv(
        registry_dir / "SOURCE_FILE_REGISTRY.csv",
        [
            "item_id",
            "path",
            "logical_path",
            "sha256",
            "size_bytes",
            "source_role",
            "lineage_provenance",
            "read_status",
            "read_error",
            "columns",
            "row_count_in_workbook",
        ],
        source_file_rows,
    )
    _write_csv(
        registry_dir / "CELL_SERIES_ROLE_REGISTRY.csv",
        [
            "item_id",
            "row",
            "column",
            "thesis_reference_value",
            "role",
            "resolution",
            "reproduced_value",
            "source_path",
            "source_row",
            "source_column",
            "formula",
            "run_id",
        ],
        role_rows,
    )
    _write_csv(
        registry_dir / "NON_EMPIRICAL_REGISTRY.csv",
        [
            "record_type",
            "item_id",
            "claim_id",
            "token_index",
            "caption",
            "class",
            "design_config_sources",
            "producer",
            "basis",
            "status",
            "empirical_claim",
        ],
        non_empirical_rows,
    )
    _write_csv(
        registry_dir / "CALCULATION_REGISTRY.csv",
        [
            "item_id",
            "item_ids",
            "run_id",
            "check_kind",
            "source_path",
            "source_row",
            "row_id",
            "policy",
            "mode",
            "oracle",
            "action_score",
            "same_firm_noop_mean_score",
            "same_firm_noop_score",
            "recomputed_delta",
            "reported_delta",
            "absolute_error",
            "check_status",
            "check_scope",
            "pairing_verification",
            "formula",
        ],
        calculation_rows,
    )
    _write_csv(
        registry_dir / "SELECTOR_REGISTRY.csv",
        [
            "item_id",
            "selector_id",
            "contract_status",
            "execution_status",
            "source_pattern",
            "source_paths",
            "row_keys",
            "row_count_contract",
            "actual_row_count",
            "key_policy",
            "expected_duplicate_count",
            "duplicate_key_count",
            "value_columns",
            "filters",
            "aggregation",
            "reshape",
            "selected_row_count",
            "reason",
            "producer",
            "run_id",
        ],
        selector_registry_rows,
    )
    status_counts: dict[str, int] = {}
    for row in evidence_rows:
        status_counts[str(row["status"])] = status_counts.get(str(row["status"]), 0) + 1
    computed_claim_rows = [
        row
        for row in included_claim_rows
        if _is_exact_computed_claim(row)
    ]
    for row in computed_claim_rows:
        lineage_errors = _computed_claim_lineage_errors(row)
        if lineage_errors:
            raise RuntimeError(
                f"Computed claim {row.get('claim_token_id')} lacks required lineage: "
                f"{lineage_errors}"
            )
    unresolved_claim_count = sum(
        str(row.get("evidence_status", "")).startswith("UNRESOLVED")
        for row in included_claim_rows
    )
    non_empirical_claim_count = sum(
        row.get("evidence_status") == "NON_EMPIRICAL_CONFIG_REGISTRY"
        for row in included_claim_rows
    )
    if len(computed_claim_rows) + unresolved_claim_count + non_empirical_claim_count != len(
        included_claim_rows
    ):
        raise RuntimeError("Numeric claim ledger statuses do not reconcile")
    active_selector_failures = [
        row
        for row in selector_registry_rows
        if row.get("contract_status") == "ACTIVE"
        and row.get("execution_status") != "SELECTOR_COMPUTED"
    ]
    explicit_claim_target_failures = _explicit_claim_target_failures(
        included_claim_rows, claim_rules
    )
    if active_selector_failures and explicit_claim_target_failures:
        prepare_status = "FAIL_ACTIVE_SELECTOR_AND_EXPLICIT_CLAIM_TARGET"
    elif active_selector_failures:
        prepare_status = "FAIL_ACTIVE_SELECTOR"
    elif explicit_claim_target_failures:
        prepare_status = "FAIL_EXPLICIT_CLAIM_TARGET"
    else:
        prepare_status = "PASS"
    manifest = {
        "schema_version": "thesis_output_prepare_manifest_v11",
        "created_at_utc": payload["created_at_utc"],
        "run_id": selected_run.run_id,
        "run_mode": selected_run.mode,
        "thesis_sha256": inventory["summary"]["thesis_sha256"],
        "contract_sha256": payload["contract"]["sha256"],
        "inventory_counts": {
            key: inventory["summary"][key]
            for key in (
                "numbered_table_count",
                "figure_count",
                "unnumbered_table_count",
                "terminal_item_count",
                "numeric_prose_candidate_count",
            )
        },
        "workbook_plan": payload["workbook_plan"],
        "item_status_counts": status_counts,
        "numeric_claims_in_ledger": len(included_claim_rows),
        "numeric_claim_paragraph_candidates": inventory["summary"]["numeric_prose_candidate_count"],
        "numeric_claim_exact_source_cells": len(computed_claim_rows),
        "numeric_claim_unresolved_rows": unresolved_claim_count,
        "numeric_claim_non_empirical_rows": non_empirical_claim_count,
        "numeric_claim_ledger_reconciled": True,
        "numeric_claim_explicit_contract_targets": len(claim_rules),
        "explicit_claim_target_failure_count": len(explicit_claim_target_failures),
        "explicit_claim_target_failures": explicit_claim_target_failures,
        "non_empirical_registry_rows": len(non_empirical_rows),
        "cell_series_role_rows": len(role_rows),
        "policy_value_identity_checks": len(calculation_rows),
        "same_firm_row_level_identity_checks": len(row_level_calculation_rows),
        "same_firm_pairing_verified_sources": len(row_level_check_cache),
        "selector_contract_items": len(selector_items),
        "selector_contract_active_items": sum(
            str(rule.get("status", "")) == "ACTIVE" for rule in selector_items.values()
        ),
        "selector_contract_unresolved_items": sum(
            str(rule.get("status", "")) == "UNRESOLVED" for rule in selector_items.values()
        ),
        "selector_outputs_computed": sum(
            row.get("execution_status") == "SELECTOR_COMPUTED" for row in selector_registry_rows
        ),
        "selector_outputs_failed_or_unresolved": sum(
            row.get("execution_status") != "SELECTOR_COMPUTED" for row in selector_registry_rows
        ),
        "policy_value_identity_failures": sum(
            str(row.get("check_status", "")).startswith("FAIL") for row in calculation_rows
        ),
        "compute_parent_policy": (
            "FrozenReplay: preserved frozen_outputs/final_freeze may be read only as "
            "PRESERVED_EVIDENCE/FROZEN_REPLAY_INPUT, while rerun Stage8/9/analysis outputs "
            "take priority; clean modes: selected-run active outputs only and frozen_outputs "
            "hard-failed; TABLE_VALUES, printed CSV, and paper assets rejected in every mode"
            if selected_run.is_frozen_replay
            else "clean selected-run active outputs only; frozen_outputs, TABLE_VALUES, printed CSV, and paper assets rejected"
        ),
        "prepare_status": prepare_status,
        "active_selector_failure_count": len(active_selector_failures),
        "build_payload": _relative(build_dir / "build_payload.json", project_root),
    }
    _write_json(build_dir / "PREPARE_MANIFEST.json", manifest)
    return manifest


def _inventory_only(project_root: Path, thesis_docx: Path, output: Path | None) -> int:
    inventory = extract_thesis_inventory(thesis_docx)
    errors = validate_canonical_inventory(inventory)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(dumps_inventory(inventory), encoding="utf-8")
    print(json.dumps(inventory["summary"], ensure_ascii=False, indent=2))
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare reviewer-facing thesis workbooks from one selected run.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--thesis-docx")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--inventory-output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = args.project_root.resolve()
    thesis_docx = resolve_thesis_docx(project_root, args.thesis_docx)
    if args.inventory_only:
        return _inventory_only(project_root, thesis_docx, args.inventory_output)
    contract_path = (
        args.contract
        or project_root
        / "analysis"
        / "thesis_output_mappings"
        / "thesis_output_plan.json"
    ).resolve()
    output_root = (args.output_root or project_root / "data" / "thesis_outputs").resolve()
    if "frozen_outputs" in {part.lower() for part in output_root.parts}:
        raise RuntimeError(f"Thesis outputs cannot be written under frozen_outputs: {output_root}")
    selected_run = select_run(project_root, args.run_id)
    manifest = prepare_build(project_root, thesis_docx, contract_path, selected_run, output_root)
    plan = manifest.get("workbook_plan", {})
    print(f"선택 실행: {selected_run.run_id} ({selected_run.mode})")
    print(
        "논문 DOCX에서 읽은 항목: "
        f"표 {plan.get('table_workbooks', 0)}개, "
        f"그림 {plan.get('figure_workbooks', 0)}개"
    )
    print(f"계산 준비 경로: {output_root / '_build' / selected_run.run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
