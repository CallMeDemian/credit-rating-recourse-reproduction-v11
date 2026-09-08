from __future__ import annotations

"""Verify Stage 8 feasibility-coding contract for LLM failure audits.

This verifier is intentionally narrow: it checks that post-simulation
``feasibility_violation`` categories are driven by core feasibility failures
(accounting identity failure, negative final balance, bad preflight, or explicit
sustainability failure) rather than by residual presentation-repair diagnostics
or large plug ratios alone.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.failure_coder import (
    FEASIBILITY_RULE_VERSION,
    ORACLE_SCORES_USED_FOR_FAILURE_CODING,
)

LEGACY_COMPATIBLE_FEASIBILITY_RULE_VERSIONS = {"post_sim_accounting_feasibility_v2"}

REQUIRED_COLUMNS = {
    "failure_categories",
    "feasibility_violation_auto",
    "feasibility_review_needed",
    "plug_to_assets",
    "plug_denominator_source",
    "accounting_check_failed",
    "negative_balance_flag",
    "residual_presentation_repair_flag",
    "plug_to_assets_review_exceeded",
    "plug_to_assets_hard_exceeded",
    "feasibility_core_violation_flag",
    "feasibility_rule_version",
    "oracle_scores_used_for_failure_coding",
}


def _load_bool(s: pd.Series) -> pd.Series:
    return s.fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def verify_stage8_feasibility_rule(stage8_dir: Path, *, allow_legacy_compatible_rule_version: bool = False) -> dict[str, Any]:
    stage8_dir = Path(stage8_dir)
    path = stage8_dir / "llm_stage8_failure_audit_enriched.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing Stage8 enriched failure audit: {path}")
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    errors: list[str] = []
    if missing:
        errors.append(f"Missing required feasibility audit columns: {missing}")
    if df.empty:
        errors.append("Stage8 enriched failure audit is empty")

    artifact_rule_version: str | None = None
    version_status = "NOT_CHECKED"
    allowed_versions = {FEASIBILITY_RULE_VERSION}
    if allow_legacy_compatible_rule_version:
        allowed_versions |= LEGACY_COMPATIBLE_FEASIBILITY_RULE_VERSIONS

    if not missing and not df.empty:
        version_values = set(df["feasibility_rule_version"].dropna().astype(str).unique())
        if len(version_values) != 1:
            errors.append(
                f"Stage8 feasibility audit must contain exactly one feasibility_rule_version; observed {sorted(version_values)}"
            )
        else:
            artifact_rule_version = next(iter(version_values))
            if artifact_rule_version == FEASIBILITY_RULE_VERSION:
                version_status = "CURRENT"
            elif artifact_rule_version in LEGACY_COMPATIBLE_FEASIBILITY_RULE_VERSIONS and allow_legacy_compatible_rule_version:
                version_status = "LEGACY_COMPATIBLE"
            else:
                expected = sorted(allowed_versions)
                errors.append(
                    f"Unexpected feasibility_rule_version values: {sorted(version_values)}; expected one of {expected}"
                )
                version_status = "UNEXPECTED"
        if _load_bool(df["oracle_scores_used_for_failure_coding"]).any() or ORACLE_SCORES_USED_FOR_FAILURE_CODING is not False:
            errors.append("Feasibility failure coding must not use Oracle scores")

        auto = _load_bool(df["feasibility_violation_auto"])
        core = _load_bool(df["feasibility_core_violation_flag"])
        residual = _load_bool(df["residual_presentation_repair_flag"])
        plug_review = _load_bool(df["plug_to_assets_review_exceeded"])
        if (auto != core).any():
            errors.append("feasibility_violation_auto must equal feasibility_core_violation_flag")
        if auto.all() and not core.all():
            errors.append("All rows are auto-feasibility failures without all rows being core violations")
        if auto.any() and not core.any():
            errors.append("Auto-feasibility failures exist although no core violation exists")
        if residual.all() and auto.all() and not core.all():
            errors.append("Residual presentation repair alone appears to be driving all feasibility failures")
        if plug_review.any() and (df.loc[plug_review, "plug_to_assets"].astype(float) <= 0.0).any():
            errors.append("plug_to_assets_review_exceeded has non-positive plug_to_assets values")

    manifest_path = stage8_dir / "failure_coder_manifest.json"
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_version = manifest.get("feasibility_rule_version")
        if artifact_rule_version is not None and manifest_version != artifact_rule_version:
            errors.append(
                f"failure_coder_manifest feasibility_rule_version={manifest_version} does not match artifact feasibility_rule_version={artifact_rule_version}"
            )
        elif artifact_rule_version is None and manifest_version not in allowed_versions:
            errors.append(
                f"failure_coder_manifest feasibility_rule_version={manifest_version} expected one of {sorted(allowed_versions)}"
            )
        if manifest.get("oracle_scores_used_for_failure_coding") is not False:
            errors.append("failure_coder_manifest must record oracle_scores_used_for_failure_coding=False")
    else:
        errors.append(f"Missing failure_coder_manifest.json: {manifest_path}")

    summary = {
        "status": "PASS" if not errors else "FAIL",
        "stage8_dir": str(stage8_dir),
        "row_count": int(len(df)),
        "expected_feasibility_rule_version": FEASIBILITY_RULE_VERSION,
        "allowed_legacy_compatible_feasibility_rule_versions": sorted(LEGACY_COMPATIBLE_FEASIBILITY_RULE_VERSIONS) if allow_legacy_compatible_rule_version else [],
        "artifact_feasibility_rule_version": artifact_rule_version,
        "version_status": version_status,
        "legacy_compatibility_mode": bool(allow_legacy_compatible_rule_version),
        "auto_feasibility_violation_count": int(_load_bool(df["feasibility_violation_auto"]).sum()) if "feasibility_violation_auto" in df.columns else None,
        "core_feasibility_violation_count": int(_load_bool(df["feasibility_core_violation_flag"]).sum()) if "feasibility_core_violation_flag" in df.columns else None,
        "residual_presentation_repair_count": int(_load_bool(df["residual_presentation_repair_flag"]).sum()) if "residual_presentation_repair_flag" in df.columns else None,
        "plug_to_assets_review_exceeded_count": int(_load_bool(df["plug_to_assets_review_exceeded"]).sum()) if "plug_to_assets_review_exceeded" in df.columns else None,
        "plug_to_assets_hard_exceeded_count": int(_load_bool(df["plug_to_assets_hard_exceeded"]).sum()) if "plug_to_assets_hard_exceeded" in df.columns else None,
        "errors": errors,
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify Stage8 feasibility failure-coding contract")
    ap.add_argument("--stage8-dir", required=True, type=Path)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument(
        "--allow-legacy-compatible-rule-version",
        action="store_true",
        help=(
            "Accept frozen Stage8 artifacts coded with a documented legacy-compatible feasibility "
            "rule version (currently v2) while still enforcing the semantic feasibility contract. "
            "Without this flag, only the current FEASIBILITY_RULE_VERSION is accepted."
        ),
    )
    args = ap.parse_args()
    summary = verify_stage8_feasibility_rule(
        args.stage8_dir,
        allow_legacy_compatible_rule_version=args.allow_legacy_compatible_rule_version,
    )
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
