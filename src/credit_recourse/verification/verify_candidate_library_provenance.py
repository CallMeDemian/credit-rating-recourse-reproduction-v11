from __future__ import annotations

"""Verify selected/P50 candidate-library provenance across the final pipeline.

The thesis proposal describes the 11 actions as P50-calibrated.  The active
base YAML is a design source, while a frozen run may materialize a calibrated
``final_candidate_library__P50.yaml``.  This verifier prevents accidental
confusion between base config and the selected runtime library by checking
metadata/hash propagation and exporting the thesis table source.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.rl.common.actions import load_action_space, active_config_hashes, resolve_candidate_library_path
from credit_recourse.rl.common.io import write_json

SCHEMA_VERSION = "candidate_library_provenance_v1"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        obj = yaml.safe_load(fh) or {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def _candidate_table_from_yaml(path: Path, action_cols: list[str]) -> pd.DataFrame:
    obj = _load_yaml(path)
    fixed = obj.get("fixed_candidates") or {}
    rows = []
    for cid, payload in fixed.items():
        rec = {"candidate_id": cid}
        if isinstance(payload, dict):
            for col in action_cols:
                rec[col] = float(payload.get(col, 0.0) or 0.0)
            for k in ["label", "summary", "paper_role"]:
                if k in payload:
                    rec[k] = payload.get(k)
        rows.append(rec)
    return pd.DataFrame(rows)


def _find_selected_candidate_libraries(final: Path) -> list[dict[str, Any]]:
    rows = []
    search_roots = [
        ("configs", final / "configs"),
        ("stage2_candidate_projection", final / "stage2_candidate_projection"),
    ]
    seen: set[Path] = set()
    for location, root in search_roots:
        if not root.exists():
            continue
        for p in sorted(root.glob("final_candidate_library*.yaml")):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            rows.append({
                "path": str(p),
                "name": p.name,
                "location": location,
                "sha256": _sha(p),
                "exists": p.exists(),
            })
    return rows


def _metadata_hash_refs(project_root: Path) -> list[dict[str, Any]]:
    rows = []
    for sk in ["stage2", "stage3", "stage4", "stage5", "stage6", "stage7", "stage8", "stage9"]:
        d = stage_dir(project_root, sk)
        for name in ["metadata.json", "multi_oracle_metadata.json", "candidate_projection_metadata.json", "magnitude_calibration_metadata.json"]:
            p = d / name
            if not p.exists():
                continue
            try:
                meta = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key in [
                "candidate_library_hash",
                "candidate_library_path",
                "candidate_action_values_source",
                "candidate_library_quantile",
                "selected_candidate_library_hash",
                "selected_recalibrated_candidate_library_hash",
                "selected_recalibrated_candidate_library_path",
                "base_candidate_library_hash",
                "final_candidate_library_sha256",
            ]:
                if key in meta:
                    rows.append({"stage": sk, "metadata_path": str(p), "key": key, "value": str(meta.get(key))})
            cfg_hashes = meta.get("config_hashes") if isinstance(meta.get("config_hashes"), dict) else {}
            for key in ["candidate_library_hash", "selected_candidate_library_hash", "selected_recalibrated_candidate_library_hash"]:
                if key in cfg_hashes:
                    rows.append({"stage": sk, "metadata_path": str(p), "key": f"config_hashes.{key}", "value": str(cfg_hashes.get(key))})
    return rows



def _values_for(meta_refs: list[dict[str, Any]], stage_set: set[str], key: str) -> set[str]:
    return {r["value"] for r in meta_refs if r.get("stage") in stage_set and r.get("key") == key and r.get("value") not in {None, "None", ""}}


def _lineage_status(*, name: str, stages: set[str], meta_refs: list[dict[str, Any]], expected_hash: str | None) -> dict[str, Any]:
    selected_values = _values_for(meta_refs, stages, "selected_recalibrated_candidate_library_hash")
    candidate_values = _values_for(meta_refs, stages, "candidate_library_hash")
    source_values = _values_for(meta_refs, stages, "candidate_action_values_source")
    quantile_values = _values_for(meta_refs, stages, "candidate_library_quantile")
    values_to_check = selected_values or candidate_values
    status = "NOT_OBSERVED"
    if values_to_check and expected_hash is not None:
        status = "PASS" if values_to_check == {expected_hash} else "FAIL"
    elif values_to_check:
        status = "OBSERVED_NO_EXPECTED_HASH"
    return {
        "track": name,
        "stages": sorted(stages),
        "status": status,
        "candidate_library_hash_values": sorted(candidate_values),
        "selected_recalibrated_candidate_library_hash_values": sorted(selected_values),
        "candidate_action_values_source_values": sorted(source_values),
        "candidate_library_quantile_values": sorted(quantile_values),
        "expected_selected_hash": expected_hash,
    }

def verify(
    project_root: Path,
    final_freeze: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    final = Path(final_freeze).resolve() if final_freeze is not None else final_root(project_root)
    if not final.exists():
        raise FileNotFoundError(f"final_freeze root does not exist: {final}")
    active = active_config_hashes(project_root)
    # Base action contract supplies the canonical column order/bounds.
    base_space = load_action_space(project_root)
    libraries = _find_selected_candidate_libraries(final)
    meta_refs = _metadata_hash_refs(project_root)

    active_path = Path(active["candidate_library_path"]).resolve()
    active_sha = active["candidate_library_hash"]
    p50_path = resolve_candidate_library_path(project_root, magnitude_quantile=50)
    p50_sha = _sha(p50_path)
    load_action_space(project_root, candidate_library_path=p50_path)  # fail-fast schema/sign guard

    # Export the thesis table from the selected P50 library, not from active base.
    # The active base YAML is still reported separately to avoid mutating the config freeze.
    selected_mode = "P50"
    selected_path = p50_path
    selected_sha = p50_sha

    table = _candidate_table_from_yaml(selected_path, list(base_space.columns))
    out_dir = Path(output_dir).resolve() if output_dir is not None else final / "ledgers"
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "candidate_library_table_for_thesis.csv"
    table.to_csv(table_path, index=False, encoding="utf-8-sig")

    hash_values = {r["value"] for r in meta_refs}
    p50_propagated_count = sum(1 for v in hash_values if v == selected_sha)
    active_propagated_count = sum(1 for v in hash_values if v == active_sha)
    track_lineage = [
        _lineage_status(name="rl_train", stages={"stage4", "stage5"}, meta_refs=meta_refs, expected_hash=selected_sha),
        _lineage_status(name="llm", stages={"stage7", "stage8", "stage9"}, meta_refs=meta_refs, expected_hash=selected_sha),
    ]
    errors = []
    warnings = []
    if active_sha == selected_sha:
        warnings.append("Active base candidate library hash equals selected P50 hash; unusual but allowed.")
    if active_sha != selected_sha:
        warnings.append(
            "Active base final_candidate_library.yaml differs from selected P50 library. "
            "This is expected after P50 alignment; do not infer runtime vectors from active base alone."
        )
    observed_llm = [t for t in track_lineage if t["track"] == "llm"][0]
    if observed_llm["status"] == "FAIL":
        errors.append("LLM track candidate-library lineage does not match selected P50 hash")
    elif observed_llm["status"] == "NOT_OBSERVED":
        warnings.append("No Stage7/8/9 metadata observed yet; run LLM stages to confirm P50 lineage.")
    observed_rl = [t for t in track_lineage if t["track"] == "rl_train"][0]
    if observed_rl["status"] == "FAIL":
        errors.append("RL Stage4/5 selected_recalibrated candidate-library lineage does not match P50 hash")
    elif observed_rl["status"] == "NOT_OBSERVED":
        warnings.append("No Stage4/5 selected_recalibrated metadata observed; cannot confirm RL P50 lineage from metadata.")
    if p50_propagated_count == 0:
        warnings.append("No stage metadata hash reference to selected P50 candidate library found yet; rerun relevant stages to create propagation evidence.")
    status = "PASS" if selected_sha and not errors else "FAIL"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _now(),
        "status": status,
        "project_root": str(project_root),
        "final_freeze": str(final),
        "active_candidate_library_path": str(active_path),
        "active_candidate_library_hash": active_sha,
        "selected_candidate_library_path": str(selected_path),
        "selected_candidate_library_hash": selected_sha,
        "selected_mode": selected_mode,
        "candidate_libraries_found": libraries,
        "metadata_hash_refs": meta_refs,
        "track_lineage": track_lineage,
        "active_hash_propagated_ref_count": int(active_propagated_count),
        "selected_p50_hash_propagated_ref_count": int(p50_propagated_count),
        "errors": errors,
        "warnings": warnings,
        "outputs": {"candidate_library_table_for_thesis": str(table_path)},
        "row_count": int(len(table)),
    }
    write_json(out_dir / "candidate_library_provenance_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify candidate-library provenance and export thesis action table")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--final-freeze", default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    try:
        report = verify(
            Path(args.project_root),
            Path(args.final_freeze) if args.final_freeze else None,
            Path(args.output_dir) if args.output_dir else None,
        )
    except Exception as exc:
        print(f"verify_candidate_library_provenance failed: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps({k: v for k, v in report.items() if k not in {"candidate_libraries_found", "metadata_hash_refs"}}, ensure_ascii=False, indent=2, default=str))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
