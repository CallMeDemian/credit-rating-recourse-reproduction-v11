from __future__ import annotations

"""Synthetic contract verifier for the post-freeze IC-c probe analysis."""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.analysis.icc_probe_analysis import import_completed_probe


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _populate_source(source: Path, n: int = 6) -> None:
    source.mkdir(parents=True, exist_ok=True)
    rows = []
    checkpoint = []
    for rid in range(n):
        recognized = rid < 4
        raw = json.dumps(
            {
                "recalled_debt_ratio": None,
                "recognized": recognized,
                "familiarity": 2 if recognized else 0,
                "known_facts": ["synthetic fact"] if recognized else [],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        rows.append(
            {
                "row_id": rid,
                "icc_probe_response_raw": raw,
                "icc_probe_value": None,
                "icc_contamination_flag": None,
                "icc_probe_parse_error": None,
                "icc_probe_rel_err": None,
                "icc_probe_panel_value": 0.25 + rid * 0.01,
            }
        )
        checkpoint.append(
            {
                "backend_id": "synthetic_backend",
                "created_utc": "2026-07-11T00:00:00+00:00",
                "icc_probe_familiarity": 2 if recognized else 0,
                "icc_probe_panel_value": 0.25 + rid * 0.01,
                "icc_probe_parse_error": None,
                "icc_probe_recognized": recognized,
                "icc_probe_rel_err": None,
                "icc_probe_response_raw": raw,
                "icc_probe_value": None,
                "probe_fingerprint": f"fp-{rid}",
                "probe_schema_version": "icc_probe_numeric_recall_v2",
                "row_id": rid,
            }
        )
    pd.DataFrame(rows).to_csv(source / "icc_probe_firm_level.csv", index=False, encoding="utf-8-sig")
    with (source / "llm_stage7_icc_probe_checkpoint.jsonl").open("w", encoding="utf-8") as fh:
        for rec in checkpoint:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    _write_json(
        source / "icc_probe_summary.json",
        {
            "schema_version": "icc_probe_only_summary_v1",
            "created_utc": "2026-07-11T00:00:00+00:00",
            "status": "PASS",
            "execution_mode": "probe_only_from_archived_stage7",
            "probe_row_count": n,
            "probe_recognized_count": 4,
            "probe_recall_null_count": n,
            "probe_parse_failures": 0,
            "probe_contaminated_count": 0,
            "numeric_recall_count": 0,
            "main_stage7_api_calls": 0,
            "interpretation_boundary": "synthetic boundary",
        },
    )


def verify(out_json: Path | None = None) -> dict[str, Any]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="icc_probe_analysis_verify_") as td:
        root = Path(td)
        source = root / "source"
        out = root / "analysis" / "icc_probe"
        _populate_source(source)
        manifest = import_completed_probe(source_dir=source, out_dir=out, expected_full_rows=6)
        required = [
            out / "icc_probe_analysis_manifest.json",
            out / "icc_probe_summary.json",
            out / "icc_probe_firm_level.csv",
            out / "icc_probe_firm_level_enriched.csv",
            out / "icc_probe_channel_summary.csv",
            out / "icc_probe_familiarity_distribution.csv",
            out / "llm_stage7_icc_probe_checkpoint.jsonl",
        ]
        errors.extend(f"missing expected output: {p}" for p in required if not p.exists())
        channel = pd.read_csv(out / "icc_probe_channel_summary.csv") if (out / "icc_probe_channel_summary.csv").exists() else pd.DataFrame()
        recognition = channel.loc[channel.get("channel", pd.Series(dtype=str)).astype(str).eq("firm_recognition")]
        numeric = channel.loc[channel.get("channel", pd.Series(dtype=str)).astype(str).eq("numeric_debt_ratio_recall")]
        if recognition.empty or int(recognition.iloc[0]["count"]) != 4:
            errors.append(f"unexpected recognition row: {recognition.to_dict(orient='records')}")
        if numeric.empty or int(numeric.iloc[0]["count"]) != 0:
            errors.append(f"unexpected numeric recall row: {numeric.to_dict(orient='records')}")
        if manifest.get("main_stage7_api_calls") != 0:
            errors.append("import mode must record main_stage7_api_calls=0")
        result = {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "outputs": [str(p) for p in required],
            "manifest_status": manifest.get("status"),
        }
    if out_json is not None:
        _write_json(out_json, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify post-freeze IC-c probe analysis contract.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)
    result = verify(Path(args.out_json) if args.out_json else None)
    return 0 if result.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
