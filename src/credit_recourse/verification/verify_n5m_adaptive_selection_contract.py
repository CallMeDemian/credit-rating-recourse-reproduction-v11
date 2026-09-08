from __future__ import annotations

"""Verify the persisted Section 9.8 N5M selector/gate contract."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.analysis.paper_output_layout import build_layout
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.prompt_builder import (
    IC_A_DERIVED_FEATURES,
    IC_B_CONTEXT_FEATURES,
)

SCHEMA_VERSION = "verify_n5m_adaptive_selection_contract_v1"
EXPECTED_BUDGETS = {"0p75", "1p27", "2p00", "unbounded"}
EXPECTED_FIRMS = 575
EXPECTED_ROWS = 2300
REQUIRED_FILES = (
    "n5m_adaptive_selection_manifest.json",
    "n5m_adaptive_selection_input_files.csv",
    "n5m_selection_feature_contract.json",
    "n5m_selection_fold_assignments.csv",
    "n5m_adaptive_budget_oof_predictions.parquet",
    "n5m_postc4_gate_oof_predictions.parquet",
    "n5m_adaptive_budget_summary.csv",
    "n5m_postc4_gate_summary.csv",
    "n5m_selection_fold_metrics.csv",
    "n5m_selection_coefficients.csv",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _close(actual: float, expected: float, *, atol: float = 1.0e-12) -> bool:
    return bool(np.isclose(float(actual), float(expected), atol=atol, rtol=0.0, equal_nan=True))


def verify(project_root: Path, analysis_dir: Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    analysis_root = (
        Path(analysis_dir).resolve()
        if analysis_dir is not None
        else root / "data" / "analysis" / "paper_repro"
    )
    source_dir = build_layout(analysis_root).n5m_adaptive_selection
    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (source_dir / name).is_file():
            errors.append(f"required Section 9.8 artifact missing: {source_dir / name}")
    if errors:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_utc": _now(),
            "status": "FAIL",
            "analysis_dir": str(analysis_root),
            "source_dir": str(source_dir),
            "errors": errors,
        }

    manifest = _read_json(source_dir / "n5m_adaptive_selection_manifest.json")
    feature_contract = _read_json(source_dir / "n5m_selection_feature_contract.json")
    if manifest.get("schema_version") != "n5m_adaptive_selection_v1" or manifest.get("status") != "PASS":
        errors.append("Section 9.8 manifest schema/status mismatch")
    if feature_contract.get("schema_version") != "n5m_selection_feature_contract_v1" or feature_contract.get("status") != "PASS":
        errors.append("Section 9.8 feature contract schema/status mismatch")
    if int(manifest.get("firm_count", -1)) != EXPECTED_FIRMS:
        errors.append(f"Section 9.8 firm_count={manifest.get('firm_count')}, expected={EXPECTED_FIRMS}")
    if int(manifest.get("firm_budget_row_count", -1)) != EXPECTED_ROWS:
        errors.append(f"Section 9.8 firm_budget_row_count={manifest.get('firm_budget_row_count')}, expected={EXPECTED_ROWS}")
    if list(map(str, manifest.get("budget_order", []))) != ["0p75", "1p27", "2p00", "unbounded"]:
        errors.append("Section 9.8 budget order/grid mismatch")
    if manifest.get("evidence_tier") != "EVALUATOR_ONLY_EXPLORATORY_OOF":
        errors.append("Section 9.8 evidence tier is not explicitly exploratory OOF")

    required_gate_features = ["c4_final_l1", "c4_projection_distance", "c4_active_dimensions"]
    if list(feature_contract.get("post_c4_gate_features", [])) != required_gate_features:
        errors.append("Post-C4 gate feature contract differs from the thesis Section 9.8 feature set")
    state_features = list(feature_contract.get("state_numeric_features", [])) + list(
        feature_contract.get("state_categorical_features", [])
    )
    if len(state_features) < 5:
        errors.append("Pre-action state feature contract contains fewer than five features")
    forbidden_tokens = ("next__", "action__", "reward", "oracle", "revision_delta", "revised_delta", "initial_delta")
    leaked = [feature for feature in state_features if any(token in str(feature).lower() for token in forbidden_tokens)]
    if leaked:
        errors.append(f"Pre-action feature contract contains leakage-prone fields: {leaked}")
    feature_payload = json.dumps(state_features, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if feature_contract.get("state_feature_order_hash") != hashlib.sha256(feature_payload.encode("utf-8")).hexdigest():
        errors.append("Pre-action state feature order hash mismatch")

    inputs = pd.read_csv(source_dir / "n5m_adaptive_selection_input_files.csv")
    expected_input_roles = {"n5m_firm_frame", "phase_eval_state_panel", "selection_contract"}
    if set(inputs.get("input_role", pd.Series(dtype=str)).astype(str)) != expected_input_roles:
        errors.append("Section 9.8 input ledger does not contain the exact three canonical input roles")
    input_paths: dict[str, Path] = {}
    for _, row in inputs.iterrows():
        role = str(row.get("input_role", ""))
        path = Path(str(row["path"]))
        input_paths[role] = path
        if not path.is_file():
            errors.append(f"Section 9.8 input recorded in manifest no longer exists: {path}")
            continue
        if _sha256(path) != str(row["sha256"]):
            errors.append(f"Section 9.8 input SHA mismatch: {path}")
    contract_path = input_paths.get("selection_contract")
    if contract_path and contract_path.is_file():
        frozen_contract = _read_json(contract_path)
        if frozen_contract.get("schema_version") != "n5m_adaptive_selection_contract_v1":
            errors.append("Section 9.8 frozen selection contract schema mismatch")
        if frozen_contract.get("status") != "ACTIVE":
            errors.append("Section 9.8 frozen selection contract is not ACTIVE")
        if manifest.get("contract_sha256") != _sha256(contract_path):
            errors.append("Section 9.8 manifest contract SHA does not match the input ledger")
        if manifest.get("adaptive_budget_contract") != frozen_contract.get("adaptive_budget"):
            errors.append("Section 9.8 adaptive-budget manifest contract differs from the frozen contract")
        if manifest.get("post_c4_gate_contract") != frozen_contract.get("post_c4_gate"):
            errors.append("Section 9.8 post-C4 gate manifest contract differs from the frozen contract")

        adaptive_cfg = frozen_contract.get("adaptive_budget", {})
        if adaptive_cfg.get("feature_visibility_contract") != "stage7_prompt_ic_b":
            errors.append("Section 9.8 pre-action feature visibility contract is not Stage7 IC-b")
        if adaptive_cfg.get("numeric_prefixes") not in ([], None):
            errors.append("Section 9.8 pre-action numeric feature contract must be exact-name only")
        numeric_allowlist = [str(value) for value in adaptive_cfg.get("numeric_exact_candidates", [])]
        categorical_allowlist = [str(value) for value in adaptive_cfg.get("categorical_candidates", [])]
        expected_context_numeric = [
            feature for feature in IC_B_CONTEXT_FEATURES
            if feature in {"year", "fiscal_year", "log_assets", "nf_log_assets"}
        ]
        expected_context_categorical = [
            feature for feature in IC_B_CONTEXT_FEATURES
            if feature in {"industry_class", "market", "sector_7"}
        ]
        expected_numeric_allowlist = [*IC_A_DERIVED_FEATURES, *expected_context_numeric]
        if numeric_allowlist != expected_numeric_allowlist:
            errors.append(
                "Section 9.8 numeric allow-list differs from the active Stage7 IC-b prompt contract"
            )
        if categorical_allowlist != expected_context_categorical:
            errors.append(
                "Section 9.8 categorical allow-list differs from the active Stage7 IC-b prompt contract"
            )
        if not numeric_allowlist or not categorical_allowlist:
            errors.append("Section 9.8 frozen IC-b feature allow-list is empty or incomplete")
        selected_numeric = [str(value) for value in feature_contract.get("state_numeric_features", [])]
        selected_categorical = [str(value) for value in feature_contract.get("state_categorical_features", [])]
        if not set(selected_numeric).issubset(set(numeric_allowlist)):
            errors.append(
                "Section 9.8 selected numeric features exceed the frozen Stage7 IC-b allow-list: "
                f"{sorted(set(selected_numeric) - set(numeric_allowlist))}"
            )
        if not set(selected_categorical).issubset(set(categorical_allowlist)):
            errors.append(
                "Section 9.8 selected categorical features exceed the frozen Stage7 IC-b allow-list: "
                f"{sorted(set(selected_categorical) - set(categorical_allowlist))}"
            )
        if feature_contract.get("state_feature_visibility_contract") != adaptive_cfg.get("feature_visibility_contract"):
            errors.append("Section 9.8 feature artifact visibility contract differs from the frozen contract")
        if feature_contract.get("state_feature_visibility_source") != adaptive_cfg.get("feature_visibility_source"):
            errors.append("Section 9.8 feature artifact visibility source differs from the frozen contract")
        if feature_contract.get("state_numeric_allowlist") != numeric_allowlist:
            errors.append("Section 9.8 feature artifact numeric allow-list differs from the frozen contract")
        if feature_contract.get("state_categorical_allowlist") != categorical_allowlist:
            errors.append("Section 9.8 feature artifact categorical allow-list differs from the frozen contract")

    adaptive = pd.read_parquet(source_dir / "n5m_adaptive_budget_oof_predictions.parquet")
    gate = pd.read_parquet(source_dir / "n5m_postc4_gate_oof_predictions.parquet")
    assignments = pd.read_csv(source_dir / "n5m_selection_fold_assignments.csv")
    coefficients = pd.read_csv(source_dir / "n5m_selection_coefficients.csv")
    adaptive_summary = pd.read_csv(source_dir / "n5m_adaptive_budget_summary.csv")
    gate_summary = pd.read_csv(source_dir / "n5m_postc4_gate_summary.csv")

    adaptive_repeats = int(manifest.get("adaptive_budget_contract", {}).get("repeats", -1))
    gate_repeats = int(manifest.get("post_c4_gate_contract", {}).get("repeats", -1))
    expected_adaptive_rows = adaptive_repeats * 2 * EXPECTED_ROWS
    expected_gate_rows = gate_repeats * EXPECTED_ROWS
    if len(adaptive) != expected_adaptive_rows:
        errors.append(f"adaptive OOF rows={len(adaptive)}, expected={expected_adaptive_rows}")
    if len(gate) != expected_gate_rows:
        errors.append(f"post-C4 gate OOF rows={len(gate)}, expected={expected_gate_rows}")
    if adaptive.duplicated(["repeat", "policy", "row_id", "budget_label"]).any():
        errors.append("adaptive OOF predictions contain duplicate keys")
    if gate.duplicated(["repeat", "row_id", "budget_label"]).any():
        errors.append("post-C4 gate OOF predictions contain duplicate keys")
    if set(adaptive["budget_label"].astype(str)) != EXPECTED_BUDGETS or set(gate["budget_label"].astype(str)) != EXPECTED_BUDGETS:
        errors.append("Section 9.8 prediction budget grid mismatch")
    if not adaptive.groupby(["repeat", "policy"])["selected_flag"].sum().eq(EXPECTED_FIRMS).all():
        errors.append("adaptive selector does not choose exactly one budget per firm and repeat")
    if gate.groupby("repeat")["row_id"].nunique().ne(EXPECTED_FIRMS).any():
        errors.append("post-C4 gate does not cover all 575 firms in every repeat")
    adaptive_per_group = adaptive.groupby(["repeat", "policy", "row_id"], sort=False)
    if adaptive_per_group.size().ne(4).any():
        errors.append("adaptive selector does not retain all four budget candidates per firm/policy/repeat")
    if adaptive_per_group["selected_flag"].sum().ne(1).any():
        errors.append("adaptive selector does not choose exactly one budget within each firm/policy/repeat")
    if gate.groupby(["repeat", "row_id"], sort=False).size().ne(4).any():
        errors.append("post-C4 gate does not retain all four budget rows per firm/repeat")
    for frame_name, frame, columns in (
        ("adaptive", adaptive, ["predicted_score", "observed_score", "selected_observed_score", "frozen_fixed_observed_score", "hindsight_budget_oracle_score"]),
        ("post-C4 gate", gate, ["predicted_revision_gain", "observed_revision_gain", "c4_score", "c6_score", "selected_score", "hindsight_oracle_score"]),
    ):
        for column in columns:
            values = pd.to_numeric(frame.get(column), errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
                errors.append(f"{frame_name} OOF column contains non-finite values: {column}")
    if {"select_c6", "c4_score", "c6_score", "selected_score"}.issubset(gate.columns):
        expected_selected = np.where(gate["select_c6"].astype(bool), gate["c6_score"], gate["c4_score"])
        if not np.allclose(pd.to_numeric(gate["selected_score"], errors="coerce"), expected_selected, atol=1e-12, rtol=0):
            errors.append("post-C4 gate selected_score is inconsistent with the persisted decision")

    for analysis, repeats in (("adaptive_budget", adaptive_repeats), ("post_c4_gate", gate_repeats)):
        sub = assignments.loc[assignments["analysis"].astype(str).eq(analysis)].copy()
        if len(sub) != repeats * EXPECTED_FIRMS:
            errors.append(f"{analysis} fold assignment rows={len(sub)}, expected={repeats * EXPECTED_FIRMS}")
        if sub.duplicated(["analysis", "repeat", "row_id"]).any():
            errors.append(f"{analysis} fold assignments contain duplicate group keys")
        if sub.groupby("repeat")["row_id"].nunique().ne(EXPECTED_FIRMS).any():
            errors.append(f"{analysis} fold assignments do not cover all firms")
        predictions = adaptive if analysis == "adaptive_budget" else gate
        merged = predictions.merge(
            sub[["analysis", "repeat", "row_id", "fold"]],
            on=["analysis", "repeat", "row_id"],
            how="left",
            validate="many_to_one",
            suffixes=("_prediction", "_assignment"),
        )
        if merged["fold_assignment"].isna().any():
            errors.append(f"{analysis} predictions are not fully covered by the fold ledger")
        elif not pd.to_numeric(merged["fold_prediction"], errors="coerce").eq(
            pd.to_numeric(merged["fold_assignment"], errors="coerce")
        ).all():
            errors.append(f"{analysis} prediction folds differ from the persisted fold ledger")
    if coefficients.empty or not {"adaptive_budget", "post_c4_gate"}.issubset(set(coefficients["analysis"].astype(str))):
        errors.append("Section 9.8 coefficient ledger is empty or incomplete")

    # Recalculate the aggregate post-C4 gate claims from firm-level OOF rows.
    gate_aggregate = gate_summary.loc[gate_summary["summary_level"].astype(str).eq("aggregate")]
    if len(gate_aggregate) != 1:
        errors.append("Post-C4 gate summary must contain exactly one aggregate row")
    else:
        row = gate_aggregate.iloc[0]
        per_repeat: list[dict[str, float]] = []
        for _, group in gate.groupby("repeat", sort=True):
            c4 = float(pd.to_numeric(group["c4_score"], errors="raise").mean())
            c6 = float(pd.to_numeric(group["c6_score"], errors="raise").mean())
            selected = float(pd.to_numeric(group["selected_score"], errors="raise").mean())
            oracle = float(pd.to_numeric(group["hindsight_oracle_score"], errors="raise").mean())
            best = max(c4, c6)
            headroom = oracle - best
            per_repeat.append({
                "c4": c4,
                "c6": c6,
                "best": best,
                "gate": selected,
                "oracle": oracle,
                "headroom": headroom,
                "capture": (selected - best) / headroom if headroom > 0 else np.nan,
                "rate": float(group["select_c6"].astype(bool).mean()),
            })
        recalculated = {
            "c4_mean_score": np.mean([x["c4"] for x in per_repeat]),
            "c6_mean_score": np.mean([x["c6"] for x in per_repeat]),
            "best_unconditional_mean_score": np.mean([x["best"] for x in per_repeat]),
            "gate_mean_score": np.mean([x["gate"] for x in per_repeat]),
            "hindsight_oracle_mean_score": np.mean([x["oracle"] for x in per_repeat]),
            "hindsight_headroom": np.mean([x["headroom"] for x in per_repeat]),
            "headroom_capture_fraction": np.mean([x["capture"] for x in per_repeat]),
            "c6_selection_rate": np.mean([x["rate"] for x in per_repeat]),
        }
        for column, expected in recalculated.items():
            if column not in gate_aggregate.columns or not _close(row[column], expected):
                errors.append(f"Post-C4 gate aggregate is not reproducible from OOF rows: {column}")

    adaptive_aggregate = adaptive_summary.loc[adaptive_summary["summary_level"].astype(str).eq("aggregate")]
    if set(adaptive_aggregate["policy"].astype(str)) != {"C4", "C6"}:
        errors.append("Adaptive-budget summary lacks one aggregate row for each policy")
    else:
        for policy in ("C4", "C6"):
            source = adaptive.loc[(adaptive["policy"].astype(str).eq(policy)) & adaptive["selected_flag"].astype(bool)]
            summaries = []
            for _, group in source.groupby("repeat", sort=True):
                selector = float(group["selected_observed_score"].mean())
                fixed = float(group["frozen_fixed_observed_score"].mean())
                oracle = float(group["hindsight_budget_oracle_score"].mean())
                summaries.append((selector, fixed, oracle))
            row = adaptive_aggregate.loc[adaptive_aggregate["policy"].astype(str).eq(policy)].iloc[0]
            expected_values = {
                "selector_mean_score": np.mean([x[0] for x in summaries]),
                "frozen_fixed_mean_score": np.mean([x[1] for x in summaries]),
                "gap_vs_frozen_fixed": np.mean([x[0] - x[1] for x in summaries]),
                "hindsight_budget_oracle_mean_score": np.mean([x[2] for x in summaries]),
            }
            for column, expected in expected_values.items():
                if column not in row.index or not _close(row[column], expected):
                    errors.append(f"Adaptive-budget aggregate is not reproducible from OOF rows: {policy}/{column}")

    outputs = manifest.get("outputs", {})
    for key, payload in outputs.items():
        path = Path(str(payload.get("path", "")))
        if not path.is_file():
            errors.append(f"Section 9.8 manifest output missing: {key} -> {path}")
        elif str(payload.get("sha256")) != _sha256(path):
            errors.append(f"Section 9.8 output SHA mismatch: {key} -> {path}")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _now(),
        "status": "PASS" if not errors else "FAIL",
        "analysis_dir": str(analysis_root),
        "source_dir": str(source_dir),
        "firm_count": EXPECTED_FIRMS,
        "firm_budget_row_count": EXPECTED_ROWS,
        "adaptive_prediction_rows": int(len(adaptive)),
        "gate_prediction_rows": int(len(gate)),
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
        path = Path(args.out_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
