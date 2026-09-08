from __future__ import annotations

"""Verify the canonical common-crossed harness-vs-backend decomposition contract."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from credit_recourse.analysis.paper_output_layout import build_layout
from credit_recourse.contracts.paper_reproduction import analysis_output_dir, load_profile
from credit_recourse.verification.verify_paper_repro_output_contract import (
    _verify_main_harness_backend_decomposition,
)

SCHEMA_VERSION = "main_harness_backend_decomposition_contract_v4"


def verify(project_root: Path, analysis_dir: Path | None = None) -> dict:
    root = Path(project_root).resolve()
    profile = load_profile(root)
    expected = analysis_output_dir(root, profile).resolve()
    actual = Path(analysis_dir).resolve() if analysis_dir is not None else expected
    errors: list[str] = []
    if actual != expected:
        errors.append(f"analysis_dir must equal canonical profile path: actual={actual}, expected={expected}")
    layout = build_layout(actual)
    detail = _verify_main_harness_backend_decomposition(layout, errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "analysis_dir": str(actual),
        "detail": detail,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--analysis-dir", default=None)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(Path(args.project_root), Path(args.analysis_dir) if args.analysis_dir else None)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
