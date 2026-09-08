from __future__ import annotations

"""Fail-fast verifier for one journal-extension C4/C4R/C6 archive arm."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.analysis.c4r_matched_inference_v3 import (
    ArmSpec,
    CONDITIONS,
    MODE,
    _arm_firm_frame,
    _inner_archive_dir,
    _load_arm,
    _read_parquet_or_csv,
)

SCHEMA_VERSION = "c4r_journal_arm_contract_v1"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_arm(
    *,
    archive_dir: Path,
    cohort_id: str,
    budget_label: str,
    l1_budget: float | None,
    expected_run_role: str,
    expected_backend_id: str,
    expected_firm_count: int = 575,
    require_live: bool = True,
    aggregate_raw_compliance_min: float = 0.95,
    condition_raw_compliance_min: float = 0.90,
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    budget_map = {budget_label: l1_budget}
    loaded_arm: dict[str, Any] = {}
    frame_rows = None
    try:
        loaded_arm, scores, revision = _load_arm(
            ArmSpec(cohort_id=cohort_id, budget_label=budget_label, path=archive_dir),
            budget_specs=budget_map,
            expected_role=expected_run_role,
            information_condition="IC-b",
            require_live=require_live,
        )
        if str(loaded_arm.get("backend_id")) != expected_backend_id:
            errors.append(
                f"backend_id mismatch: expected={expected_backend_id!r}, observed={loaded_arm.get('backend_id')!r}"
            )
        frame = _arm_firm_frame(
            loaded_arm,
            scores,
            revision,
            expected_firm_count=expected_firm_count,
        )
        frame_rows = int(len(frame))
    except Exception as exc:
        errors.append(f"archive/stage8/stage9 contract failed: {type(exc).__name__}: {exc}")
        scores = pd.DataFrame()
        revision = pd.DataFrame()

    run_dir = None
    action_rows = None
    prompt_payload_rows = None
    prompt_payload_hash_match = None
    aggregate_compliance = None
    condition_compliance: dict[str, float] = {}
    parsing_failure_rows = None
    try:
        run_dir = _inner_archive_dir(archive_dir)
        stage7 = run_dir / "stage7_llm_action_generation"
        meta_path = stage7 / "metadata.json"
        prompt_manifest_path = stage7 / "llm_stage7_prompt_manifest.json"
        action_path = stage7 / "llm_stage7_action_table.parquet"
        failure_path = stage7 / "llm_stage7_failure_audit.csv"
        for required in (meta_path, prompt_manifest_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        prompt_contract = meta.get("prompt_payload_archive_contract") or {}
        if prompt_contract.get("status") != "FULL_PAYLOAD_ARCHIVED":
            errors.append("prompt payload archive status is not FULL_PAYLOAD_ARCHIVED")
        payload_name = str(prompt_contract.get("path") or "")
        payload_path = stage7 / payload_name
        if not payload_name or not payload_path.is_file():
            errors.append(f"prompt payload file missing: {payload_path}")
        else:
            actual_hash = _sha256(payload_path)
            prompt_payload_hash_match = actual_hash == str(prompt_contract.get("sha256"))
            if not prompt_payload_hash_match:
                errors.append(
                    f"prompt payload sha256 mismatch: expected={prompt_contract.get('sha256')}, observed={actual_hash}"
                )
            with payload_path.open("r", encoding="utf-8") as fh:
                prompt_payload_rows = sum(1 for line in fh if line.strip())
            if prompt_payload_rows != int(prompt_contract.get("record_count", -1)):
                errors.append(
                    f"prompt payload record count mismatch: metadata={prompt_contract.get('record_count')}, observed={prompt_payload_rows}"
                )
        actions = _read_parquet_or_csv(action_path)
        required_action_columns = {
            "row_id",
            "policy",
            "mode",
            "budgeted_condition_flag",
            "budget_compliant_raw",
        }
        missing = sorted(required_action_columns - set(actions.columns))
        if missing:
            errors.append(f"Stage7 action table missing columns: {missing}")
        selected = actions.loc[
            actions["policy"].astype(str).isin(CONDITIONS)
            & actions["mode"].astype(str).eq(MODE)
        ].copy()
        action_rows = int(len(selected))
        expected_action_rows = expected_firm_count * len(CONDITIONS)
        if action_rows != expected_action_rows:
            errors.append(f"Stage7 expected {expected_action_rows} action rows, observed={action_rows}")
        if selected.duplicated(["row_id", "policy", "mode"]).any():
            errors.append("Stage7 action table has duplicate row_id/policy/mode keys")
        if l1_budget is not None and not selected.empty:
            applies = selected["budgeted_condition_flag"].fillna(False).astype(bool)
            if not applies.all():
                errors.append("finite arm contains C4/C4R/C6 rows without budgeted_condition_flag")
            compliance = selected["budget_compliant_raw"].fillna(False).astype(bool)
            aggregate_compliance = float(compliance.mean())
            if aggregate_compliance < aggregate_raw_compliance_min:
                errors.append(
                    f"aggregate raw budget compliance {aggregate_compliance:.4f} < {aggregate_raw_compliance_min:.4f}"
                )
            for condition, group in selected.groupby(selected["policy"].astype(str)):
                value = float(group["budget_compliant_raw"].fillna(False).astype(bool).mean())
                condition_compliance[str(condition)] = value
                if value < condition_raw_compliance_min:
                    errors.append(
                        f"condition={condition} raw budget compliance {value:.4f} < {condition_raw_compliance_min:.4f}"
                    )
        elif l1_budget is None and not selected.empty:
            if selected["budgeted_condition_flag"].fillna(False).astype(bool).any():
                errors.append("unbounded arm contains budgeted_condition_flag=True")
        if failure_path.is_file():
            failure = pd.read_csv(failure_path)
            parsing_failure_rows = int(
                failure.astype(str).apply(
                    lambda col: col.str.contains("translational_failure", regex=False, na=False)
                ).any(axis=1).sum()
            )
        else:
            warnings.append(f"failure audit not found for parser-failure count: {failure_path}")
    except Exception as exc:
        errors.append(f"stage7 payload/action contract failed: {type(exc).__name__}: {exc}")

    stage8_rows = None
    stage9_rows = None
    if not scores.empty:
        stage8_rows = int(
            scores.loc[
                scores["policy"].astype(str).isin(CONDITIONS)
                & scores["mode"].astype(str).eq(MODE)
            ].shape[0]
        )
    if not revision.empty:
        stage9_rows = int(
            revision.loc[
                revision["revision_condition"].astype(str).isin(["C4R", "C6"])
                & revision["mode"].astype(str).eq(MODE)
            ].shape[0]
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "archive_dir": str(Path(archive_dir).resolve()),
        "resolved_run_dir": str(run_dir) if run_dir else None,
        "cohort_id": cohort_id,
        "budget_label": budget_label,
        "l1_budget": l1_budget,
        "expected_run_role": expected_run_role,
        "expected_backend_id": expected_backend_id,
        "require_live": require_live,
        "candidate_library_hash": loaded_arm.get("candidate_library_hash"),
        "observed_backend_id": loaded_arm.get("backend_id"),
        "stage7_action_rows": action_rows,
        "stage8_score_rows": stage8_rows,
        "stage9_revision_rows": stage9_rows,
        "firm_frame_rows": frame_rows,
        "prompt_payload_rows": prompt_payload_rows,
        "prompt_payload_hash_match": prompt_payload_hash_match,
        "aggregate_raw_budget_compliance": aggregate_compliance,
        "condition_raw_budget_compliance": condition_compliance,
        "translational_failure_rows": parsing_failure_rows,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", required=True)
    parser.add_argument("--cohort-id", required=True)
    parser.add_argument("--budget-label", required=True)
    parser.add_argument("--l1-budget", default="unbounded")
    parser.add_argument("--expected-run-role", required=True)
    parser.add_argument("--expected-backend-id", required=True)
    parser.add_argument("--expected-firm-count", type=int, default=575)
    parser.add_argument("--allow-nonlive", action="store_true")
    parser.add_argument("--aggregate-raw-compliance-min", type=float, default=0.95)
    parser.add_argument("--condition-raw-compliance-min", type=float, default=0.90)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    raw_budget = str(args.l1_budget).strip().lower()
    budget = None if raw_budget in {"unbounded", "none", "null"} else float(raw_budget)
    result = verify_arm(
        archive_dir=Path(args.archive_dir),
        cohort_id=args.cohort_id,
        budget_label=args.budget_label,
        l1_budget=budget,
        expected_run_role=args.expected_run_role,
        expected_backend_id=args.expected_backend_id,
        expected_firm_count=args.expected_firm_count,
        require_live=not args.allow_nonlive,
        aggregate_raw_compliance_min=args.aggregate_raw_compliance_min,
        condition_raw_compliance_min=args.condition_raw_compliance_min,
    )
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
