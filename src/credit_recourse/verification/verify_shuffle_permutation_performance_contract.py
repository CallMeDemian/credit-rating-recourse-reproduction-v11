from __future__ import annotations

"""Synthetic regression verifier for repeated row-shuffle performance semantics.

The verifier does not run the financial simulator. It checks the exact contract
that caused the production slowdown:

* only target policy/mode rows are scored after the first auditable draw;
* worker scoring suppresses temporary parquet/audit writes;
* partial checkpoints are resumable only under an identical input contract;
* completed draw rows preserve the requested seed set without duplicates.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import credit_recourse.analysis.llm_action_budget_ablation as ab


class _Space:
    columns = ["action__x", "action__y"]
    bounds = {"action__x": (-10.0, 10.0), "action__y": (-10.0, 10.0)}

    def bound_width(self, col: str) -> float:
        lo, hi = self.bounds[col]
        return float(hi - lo)


def _action_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_id in range(4):
        for policy, mode in [("C6", "free_form_10d"), ("C4", "free_form_10d"), ("C6", "candidate_selection")]:
            rows.append({
                "row_id": row_id,
                "policy": policy,
                "mode": mode,
                "candidate_id": f"{policy}_{mode}",
                "projection_distance": 0.0,
                "projection_method": "synthetic",
                "out_of_library": mode == "free_form_10d",
                "action__x": float(row_id + 1),
                "action__y": float(-(row_id + 1)),
            })
    return pd.DataFrame(rows)


def _score_frame(table: pd.DataFrame, offset: float) -> pd.DataFrame:
    out = table[["row_id", "policy", "mode", "candidate_id"]].copy()
    base = pd.to_numeric(out["row_id"], errors="raise").astype(float) + float(offset)
    for bk, delta in [("alpha", 0.0), ("beta", 1.0), ("gamma", 2.0)]:
        out[f"delta_R_score_{bk}"] = base + delta
    return out


def _check_target_rows_only(errors: list[str]) -> dict[str, Any]:
    table = _action_table()
    target = ab._target_filter(table, ["C6"], ["free_form_10d"])
    original = _score_frame(target, 10.0)
    reference = _score_frame(target, 8.0)

    observed: dict[str, Any] = {}
    old_score = ab._score_action_table
    try:
        def fake_score_action_table(*, project_root, action_table, out, space, output_prefix="ablation", write_outputs=True):
            observed["n_scored_rows"] = int(len(action_table))
            observed["write_outputs"] = bool(write_outputs)
            scored = _score_frame(action_table, 9.0)
            return scored, pd.DataFrame(), pd.DataFrame(), {"identity": "synthetic"}

        ab._score_action_table = fake_score_action_table
        ab._SHUFFLE_WORKER_STATE = {
            "project_root_path": Path("."),
            "space": _Space(),
            "action_table": table,
            "shuffle_strata": pd.Series("__ALL__", index=table.index, dtype="object"),
            "original_scores": original,
            "reference_scores": reference,
            "policies": ["C6"],
            "modes": ["free_form_10d"],
            "variant": "row_shuffle_vector_null",
            "target_budget": "native",
            "active_eps": 1e-9,
            "shuffle_within": "none",
            "resolved_reference": "C3_candidate_iql",
            "simulator_identity": {"identity": "synthetic"},
        }
        records = ab._score_shuffle_seed_worker(2)
    finally:
        ab._score_action_table = old_score
        ab._SHUFFLE_WORKER_STATE = None

    expected = len(target)
    if observed.get("n_scored_rows") != expected:
        errors.append(f"worker scored {observed.get('n_scored_rows')} rows; expected target-only {expected}")
    if observed.get("write_outputs") is not False:
        errors.append("worker did not suppress temporary scoring outputs")
    if len(records) != 3:
        errors.append(f"worker summary expected 3 oracle rows, got {len(records)}")
    return {
        "full_stage7_rows": int(len(table)),
        "target_rows": int(expected),
        "observed_scored_rows": observed.get("n_scored_rows"),
        "write_outputs": observed.get("write_outputs"),
        "summary_rows": len(records),
    }


def _check_checkpoint_contract(tmp: Path, errors: list[str]) -> dict[str, Any]:
    source = tmp / "stage7.parquet"
    source.write_bytes(b"synthetic-stage7")
    partial = tmp / "shuffle_per_draw_summary.partial.csv"
    checkpoint = tmp / "shuffle_permutation_checkpoint.json"
    signature = ab._checkpoint_signature(
        stage7_action_table=source,
        variant="row_shuffle_vector_null",
        target_budget="native",
        policies=["C6"],
        modes=["free_form_10d"],
        shuffle_within="industry",
        seeds=[1, 2, 3],
        reference_policy="C3",
        stratum_contract_sha256="a" * 64,
    )
    frame = pd.DataFrame({
        "shuffle_seed": [1, 2],
        "policy": ["C6", "C6"],
        "mode": ["free_form_10d", "free_form_10d"],
        "oracle_backend": ["alpha", "alpha"],
        "reference_label": ["C3_candidate_iql", "C3_candidate_iql"],
        "shuffle_within": ["industry", "industry"],
        "mean_original_score": [1.0, 1.0],
        "mean_shuffled_score": [0.8, 0.9],
        "mean_matching_gain_original_minus_shuffle": [0.2, 0.1],
        "mean_gap_shuffle_vs_reference": [0.1, 0.2],
        "shuffle_fixed_point_count": [0, 0],
        "shuffle_singleton_row_count": [0, 0],
        "shuffle_stratum_count": [2, 2],
    })
    written = ab._write_shuffle_checkpoint(
        frames=[frame],
        partial_path=partial,
        checkpoint_path=checkpoint,
        signature=signature,
    )
    loaded = ab._load_shuffle_checkpoint(
        partial_path=partial,
        checkpoint_path=checkpoint,
        expected_signature=signature,
    )
    if len(written) != 2 or len(loaded) != 2:
        errors.append("checkpoint roundtrip did not preserve draw rows")

    wrong = dict(signature)
    wrong["signature_sha256"] = "0" * 64
    mismatch_failed = False
    try:
        ab._load_shuffle_checkpoint(
            partial_path=partial,
            checkpoint_path=checkpoint,
            expected_signature=wrong,
        )
    except RuntimeError:
        mismatch_failed = True
    if not mismatch_failed:
        errors.append("checkpoint signature mismatch did not hard-fail")

    final_path = tmp / "shuffle_per_draw_summary.csv"
    loaded.to_csv(final_path, index=False)
    final_meta = dict(signature)
    final_meta.update({"status": "PASS", "completed_seed_count": 2})
    checkpoint.write_text(json.dumps(final_meta), encoding="utf-8")
    partial.unlink()
    completed_reload = ab._load_shuffle_checkpoint(
        partial_path=partial,
        checkpoint_path=checkpoint,
        expected_signature=signature,
    )
    if len(completed_reload) != 2:
        errors.append("PASS checkpoint did not reload the final per-draw summary")
    return {
        "roundtrip_rows": int(len(loaded)),
        "completed_seeds": sorted(set(loaded["shuffle_seed"].astype(int))),
        "mismatch_hard_fail": mismatch_failed,
        "pass_checkpoint_final_reload_rows": int(len(completed_reload)),
    }



def _check_informative_industry_strata(tmp: Path, errors: list[str]) -> dict[str, Any]:
    action = _action_table().copy()
    names = {
        0: "전자 장비 제조업",
        1: "화학 제품 제조업",
        2: "자동차 부품 제조업",
        3: "정보통신 서비스업",
    }
    action["sector_7"] = "UNKNOWN"
    action["산업명"] = action["row_id"].map(names)
    direct_values, direct_meta = ab._resolve_shuffle_strata(tmp, action, shuffle_within="industry")
    if direct_meta.get("stratum_column") != "산업명":
        errors.append(f"informative industry-name column was not selected: {direct_meta}")
    if int(direct_meta.get("stratum_unique_known", 0)) < 2:
        errors.append("industry-name derivation did not yield multiple known strata")
    if direct_values.astype(str).eq("UNKNOWN").any():
        errors.append("all-UNKNOWN sector_7 leaked through despite informative industry names")

    # Raw recovery regression: active Stage2 carries only UNKNOWN sector_7, while
    # the frozen general-information workbook carries the actual industry field.
    raw_root = tmp / "raw_case"
    stage2 = ab.stage_dir(raw_root, "stage2")
    stage2.mkdir(parents=True, exist_ok=True)
    identity = pd.DataFrame({
        # Exact production regression: Stage7 derives row_id from stable serving
        # panel order when Stage2 does not physically store a row_id column.
        "firm_id": ["000001", "000002", "000003", "000004"],
        "sector_7": ["UNKNOWN"] * 4,
    })
    identity.to_parquet(stage2 / "phase_eval_candidate.parquet", index=False)
    raw_dir = raw_root / "data" / "raw" / "raw_nonfinancial" / "kospi_kosdaq"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "거래소코드": ["000001", "000002"],
        "회계년도": ["2025/12", "2025/12"],
        "산업코드": [32001, 20001],
        "산업명": ["전자 장비 제조업", "화학 제품 제조업"],
    }).to_excel(raw_dir / ab.RAW_GENERAL_CANONICAL_FILENAMES["kospi"], index=False)
    pd.DataFrame({
        "거래소코드": ["000003", "000004"],
        "회계년도": ["2025/12", "2025/12"],
        "산업코드": [30001, 58001],
        "산업명": ["자동차 부품 제조업", "정보통신 서비스업"],
    }).to_excel(raw_dir / ab.RAW_GENERAL_CANONICAL_FILENAMES["kosdaq"], index=False)
    # A stale extra workbook must not enter the Stage00-03 source contract.
    stale_dir = raw_root / "data" / "raw" / "raw_all"
    stale_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "거래소코드": ["000001"],
        "회계년도": ["2099/12"],
        "산업코드": [20001],
        "산업명": ["화학 제품 제조업"],
    }).to_excel(stale_dir / "stale_snapshot_일반사항.xlsx", index=False)
    unknown_action = action.drop(columns=["산업명"]).copy()
    raw_values, raw_meta = ab._resolve_shuffle_strata(raw_root, unknown_action, shuffle_within="industry")
    if raw_meta.get("stratum_source") != "raw_general_information_recovery":
        errors.append(f"raw industry fallback was not selected: {raw_meta}")
    if int(raw_meta.get("stratum_unique_known", 0)) < 2 or float(raw_meta.get("stratum_known_fraction", 0.0)) < 0.90:
        errors.append(f"raw industry fallback is not informative: {raw_meta}")
    if not isinstance(raw_meta.get("stratum_contract_sha256"), str) or len(raw_meta["stratum_contract_sha256"]) != 64:
        errors.append("raw industry fallback did not emit a stratum contract SHA-256")
    identity_contract = raw_meta.get("stratum_identity_contract") or {}
    if identity_contract.get("identity_row_id_derivation") != "stage7_stable_row_order_index":
        errors.append(f"Stage2 serving-panel index row_id contract was not used: {identity_contract}")

    # Frozen Stage6 fallback regression: an old Stage2 serving panel can lack a
    # usable firm key while Stage6's canonical simulated frame still records the
    # exact row_id→firm_id relation. Repeated policy rows must agree.
    fallback_root = tmp / "stage6_fallback_case"
    fallback_stage2 = ab.stage_dir(fallback_root, "stage2")
    fallback_stage2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sector_7": ["UNKNOWN"] * 4}).to_parquet(
        fallback_stage2 / "phase_eval_candidate.parquet", index=False
    )
    fallback_stage6 = ab.stage_dir(fallback_root, "stage6")
    fallback_stage6.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": [0, 0, 1, 1, 2, 2, 3, 3],
        "firm_id": ["000001", "000001", "000002", "000002", "000003", "000003", "000004", "000004"],
        "policy": ["C0", "C3"] * 4,
    }).to_parquet(fallback_stage6 / "simulated_oracle_input_frame.parquet", index=False)
    fallback_raw = fallback_root / "data" / "raw" / "raw_nonfinancial" / "kospi_kosdaq"
    fallback_raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "거래소코드": ["000001", "000002"],
        "회계년도": ["2025/12", "2025/12"],
        "산업코드": [32001, 20001],
        "산업명": ["전자 장비 제조업", "화학 제품 제조업"],
    }).to_excel(fallback_raw / ab.RAW_GENERAL_CANONICAL_FILENAMES["kospi"], index=False)
    pd.DataFrame({
        "거래소코드": ["000003", "000004"],
        "회계년도": ["2025/12", "2025/12"],
        "산업코드": [30001, 58001],
        "산업명": ["자동차 부품 제조업", "정보통신 서비스업"],
    }).to_excel(fallback_raw / ab.RAW_GENERAL_CANONICAL_FILENAMES["kosdaq"], index=False)
    fallback_values, fallback_meta = ab._resolve_shuffle_strata(
        fallback_root, unknown_action, shuffle_within="industry"
    )
    fallback_identity = fallback_meta.get("stratum_identity_contract") or {}
    if fallback_identity.get("identity_source") != str(
        fallback_stage6 / "simulated_oracle_input_frame.parquet"
    ):
        errors.append(f"Stage6 identity fallback was not selected: {fallback_identity}")
    if int(fallback_values.nunique()) < 2:
        errors.append("Stage6 identity fallback did not recover informative industries")

    conflict_root = tmp / "stage6_conflict_case"
    conflict_stage2 = ab.stage_dir(conflict_root, "stage2")
    conflict_stage2.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sector_7": ["UNKNOWN"] * 4}).to_parquet(
        conflict_stage2 / "phase_eval_candidate.parquet", index=False
    )
    conflict_stage6 = ab.stage_dir(conflict_root, "stage6")
    conflict_stage6.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "row_id": [0, 0, 1, 2, 3],
        "firm_id": ["000001", "999999", "000002", "000003", "000004"],
    }).to_parquet(conflict_stage6 / "simulated_oracle_input_frame.parquet", index=False)
    conflict_failed = False
    try:
        ab._resolve_action_identity_source(conflict_root, unknown_action)
    except ValueError:
        conflict_failed = True
    if not conflict_failed:
        errors.append("conflicting Stage6 row_id→firm_id identity did not hard-fail")

    # Canonical raw-source conflict resolution: newest accounting period wins;
    # equal-period cross-market ties follow configured KOSPI -> KOSDAQ order.
    source_contract_root = tmp / "raw_source_contract_case"
    contract_raw = source_contract_root / "data" / "raw" / "raw_nonfinancial" / "kospi_kosdaq"
    contract_raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "거래소코드": ["000001", "000002", "000003", "000003"],
        "회계년도": ["2025/12", "2025/12", "2025/12", "2025/12"],
        "산업명": ["전자 장비 제조업", "자동차 부품 제조업", "전자 장비 제조업", "화학 제품 제조업"],
    }).to_excel(contract_raw / ab.RAW_GENERAL_CANONICAL_FILENAMES["kospi"], index=False)
    pd.DataFrame({
        "거래소코드": ["000001", "000002"],
        "회계년도": ["2024/12", "2025/12"],
        "산업명": ["화학 제품 제조업", "정보통신 서비스업"],
    }).to_excel(contract_raw / ab.RAW_GENERAL_CANONICAL_FILENAMES["kosdaq"], index=False)
    ab._raw_general_industry_lookup_cached.cache_clear()
    same_precedence_failed = False
    try:
        ab._raw_general_industry_lookup_cached(str(source_contract_root.resolve()))
    except ValueError:
        same_precedence_failed = True
    if not same_precedence_failed:
        errors.append("same-period same-market conflicting sectors did not hard-fail")

    # Remove the unresolved same-source duplicate, then verify deterministic
    # cross-year and cross-market resolution and stale-workbook exclusion.
    pd.DataFrame({
        "거래소코드": ["000001", "000002"],
        "회계년도": ["2025/12", "2025/12"],
        "산업명": ["전자 장비 제조업", "자동차 부품 제조업"],
    }).to_excel(contract_raw / ab.RAW_GENERAL_CANONICAL_FILENAMES["kospi"], index=False)
    stale_contract = source_contract_root / "data" / "raw" / "raw_all"
    stale_contract.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "거래소코드": ["000001"],
        "회계년도": ["2099/12"],
        "산업명": ["건설업"],
    }).to_excel(stale_contract / "stale_일반사항.xlsx", index=False)
    ab._raw_general_industry_lookup_cached.cache_clear()
    contract_lookup, contract_meta = ab._raw_general_industry_lookup_cached(str(source_contract_root.resolve()))
    contract_map = contract_lookup.set_index("firm_id")["sector_7_recovered"].to_dict()
    if contract_map.get("000001") != "금속·기계·전자":
        errors.append(f"latest-year or stale-source exclusion failed for 000001: {contract_map}")
    if contract_map.get("000002") != "자동차·운송장비":
        errors.append(f"configured market-priority resolution failed for 000002: {contract_map}")
    if contract_meta.get("resolution_policy") != "stage00_03_latest_accounting_period_then_configured_market_priority_v1":
        errors.append(f"raw source resolution policy metadata is missing: {contract_meta}")
    if int(contract_meta.get("conflicting_sector_firm_count", 0)) < 2:
        errors.append(f"resolved conflict audit was not recorded: {contract_meta}")

    no_raw_root = tmp / "no_raw_case"
    stage2_no_raw = ab.stage_dir(no_raw_root, "stage2")
    stage2_no_raw.mkdir(parents=True, exist_ok=True)
    identity.to_parquet(stage2_no_raw / "phase_eval_candidate.parquet", index=False)
    hard_failed = False
    try:
        ab._resolve_shuffle_strata(no_raw_root, unknown_action, shuffle_within="industry")
    except (FileNotFoundError, ValueError):
        hard_failed = True
    if not hard_failed:
        errors.append("uninformative industry sources without raw recovery did not hard-fail")

    return {
        "direct_source": direct_meta.get("stratum_source"),
        "direct_column": direct_meta.get("stratum_column"),
        "direct_unique_known": direct_meta.get("stratum_unique_known"),
        "raw_source": raw_meta.get("stratum_source"),
        "raw_unique_known": raw_meta.get("stratum_unique_known"),
        "raw_known_fraction": raw_meta.get("stratum_known_fraction"),
        "raw_identity_row_id_derivation": identity_contract.get("identity_row_id_derivation"),
        "stage6_fallback_identity_source": fallback_identity.get("identity_source"),
        "stage6_conflict_hard_fail": conflict_failed,
        "same_precedence_raw_conflict_hard_fail": same_precedence_failed,
        "raw_resolution_policy": contract_meta.get("resolution_policy"),
        "raw_resolved_conflict_count": contract_meta.get("conflicting_sector_firm_count"),
        "no_raw_hard_fail": hard_failed,
    }


def _check_selective_checkpoint_invalidation(tmp: Path, errors: list[str]) -> dict[str, Any]:
    source = tmp / "signature_stage7.parquet"
    source.write_bytes(b"signature-stage7")
    common = dict(
        stage7_action_table=source,
        variant="row_shuffle_vector_null",
        target_budget="native",
        policies=["C6"],
        modes=["free_form_10d"],
        seeds=[1, 2, 3],
        reference_policy="C3",
    )
    none_sig = ab._checkpoint_signature(shuffle_within="none", **common)
    rating_sig = ab._checkpoint_signature(shuffle_within="rating_band", **common)
    industry_a = ab._checkpoint_signature(
        shuffle_within="industry", stratum_contract_sha256="a" * 64, **common
    )
    industry_b = ab._checkpoint_signature(
        shuffle_within="industry", stratum_contract_sha256="b" * 64, **common
    )
    if "industry_stratum_contract_sha256" in none_sig or "industry_stratum_contract_sha256" in rating_sig:
        errors.append("non-industry checkpoint signatures were unexpectedly changed by the industry fix")
    if industry_a["signature_sha256"] == industry_b["signature_sha256"]:
        errors.append("industry checkpoint signature is not bound to the resolved stratum contract")
    if industry_a.get("industry_stratum_contract_version") != "informative_industry_strata_v1":
        errors.append("industry checkpoint contract version is missing")
    return {
        "none_preserves_legacy_shape": "industry_stratum_contract_sha256" not in none_sig,
        "rating_preserves_legacy_shape": "industry_stratum_contract_sha256" not in rating_sig,
        "industry_signature_changes_with_strata": industry_a["signature_sha256"] != industry_b["signature_sha256"],
    }

def run_verification() -> dict[str, Any]:
    errors: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        checks = {
            "target_rows_only_worker": _check_target_rows_only(errors),
            "checkpoint_contract": _check_checkpoint_contract(tmp, errors),
            "informative_industry_strata": _check_informative_industry_strata(tmp, errors),
            "selective_checkpoint_invalidation": _check_selective_checkpoint_invalidation(tmp, errors),
        }
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify repeated shuffle performance/resume contract")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args(argv)
    result = run_verification()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
