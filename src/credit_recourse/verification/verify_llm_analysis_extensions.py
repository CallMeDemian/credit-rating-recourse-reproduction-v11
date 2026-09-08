from __future__ import annotations

"""Synthetic contract verifier for non-N5 LLM analysis extensions.

Checks four surgical contracts introduced after the main freeze:
  1. feasibility v3 maps sustainability=critical to hard and fragile to review;
  2. sign_flip_mean_vector is a true sign-reversal null, not row shuffle;
  3. row-shuffle permutations preserve multivariate vectors within frozen strata and derange non-singletons;
  4. paired TOST requires row-level paired data and can establish equivalence under an explicit margin.
"""

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.analysis.llm_action_budget_ablation import _aggregate_shuffle_draws, _parse_seed_spec, _variant_rows
from credit_recourse.analysis.llm_tost_equivalence import run_tost_equivalence
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.failure_coder import (
    FEASIBILITY_RULE_VERSION,
    code_feasibility_violation,
)


class _SyntheticActionSpace:
    columns = ["ppe_pct", "short_debt_pct", "cogs_ratio_chg"]
    bounds = {c: (-10.0, 10.0) for c in columns}
    train_labels = ["A0_noop", "DL2_deleverage_moderate"]
    final_rl_label = "C3_candidate_iql"

    def candidate_vector(self, name: str) -> np.ndarray:
        if name == "A0_noop":
            return np.array([0.0, 0.0, 0.0], dtype=float)
        if name == "DL2_deleverage_moderate":
            return np.array([0.0, -2.0, 0.0], dtype=float)
        raise KeyError(name)

    def bound_width(self, col: str) -> float:
        lo, hi = self.bounds[col]
        return float(hi - lo)


def _fail(errors: list[str], message: str) -> None:
    errors.append(message)


def _check_feasibility_contract(errors: list[str]) -> dict[str, Any]:
    base = {
        "total_assets_before": 100.0,
        "plug_amount": 0.0,
        "current_assets_after": 10.0,
        "current_liabilities_after": 5.0,
        "accounting_check": {"check": "ok"},
        "simulator_preflight_status": "ok",
        "residual_negative_flag": False,
    }
    ok = code_feasibility_violation({**base, "sustainability": "ok"})
    fragile = code_feasibility_violation({**base, "sustainability": "fragile"})
    critical = code_feasibility_violation({**base, "sustainability": "critical"})
    accounting_fail = code_feasibility_violation({**base, "sustainability": "ok", "accounting_check": {"check": "fail"}})
    residual_only = code_feasibility_violation({**base, "sustainability": "ok", "residual_negative_flag": True})

    if ok["feasibility_violation_auto"] or ok["feasibility_review_needed"]:
        _fail(errors, "sustainability=ok should not hard-fail or review")
    if fragile["feasibility_violation_auto"] or not fragile["feasibility_review_needed"]:
        _fail(errors, "sustainability=fragile should be review-only")
    if not critical["feasibility_violation_auto"] or not critical["feasibility_core_violation_flag"]:
        _fail(errors, "sustainability=critical should be a hard core feasibility violation")
    if not accounting_fail["accounting_check_failed"] or not accounting_fail["feasibility_violation_auto"]:
        _fail(errors, "accounting_check={check: fail} should hard-fail")
    if residual_only["feasibility_violation_auto"] or residual_only["feasibility_review_needed"]:
        _fail(errors, "residual_negative_flag alone should remain metadata-only")
    return {
        "rule_version": FEASIBILITY_RULE_VERSION,
        "ok": {"hard": ok["feasibility_violation_auto"], "review": ok["feasibility_review_needed"]},
        "fragile": {"hard": fragile["feasibility_violation_auto"], "review": fragile["feasibility_review_needed"]},
        "critical": {"hard": critical["feasibility_violation_auto"], "review": critical["feasibility_review_needed"]},
        "accounting_fail": {"hard": accounting_fail["feasibility_violation_auto"]},
        "residual_only": {"hard": residual_only["feasibility_violation_auto"], "review": residual_only["feasibility_review_needed"]},
    }


def _check_sign_flip_contract(errors: list[str]) -> dict[str, Any]:
    space = _SyntheticActionSpace()
    rows = []
    vectors = [
        (1, 1.0, -2.0, 0.5),
        (2, 3.0, -4.0, 1.5),
        (3, 2.0, -6.0, 1.0),
    ]
    for row_id, ppe, debt, cogs in vectors:
        rows.append({
            "row_id": row_id,
            "policy": "C6",
            "mode": "free_form_10d",
            "candidate_id": "ORIGINAL",
            "projection_distance": 0.1,
            "projection_method": "original",
            "out_of_library": True,
            "ppe_pct": ppe,
            "short_debt_pct": debt,
            "cogs_ratio_chg": cogs,
        })
    # Candidate-selection rows make budget modes resolvable, though this verifier uses native budget.
    for row_id in [1, 2, 3]:
        rows.append({
            "row_id": row_id,
            "policy": "C6",
            "mode": "candidate_selection",
            "candidate_id": "A0_noop",
            "projection_distance": 0.0,
            "projection_method": "candidate",
            "out_of_library": False,
            "ppe_pct": 0.0,
            "short_debt_pct": -1.0,
            "cogs_ratio_chg": 0.0,
        })
    action_table = pd.DataFrame(rows)
    transformed, diag = _variant_rows(
        action_table,
        space=space,
        policies=["C6"],
        modes=["free_form_10d"],
        variant="sign_flip_mean_vector",
        target_budget="native",
        active_eps=1e-9,
        random_seed=123,
    )
    target = transformed[transformed["mode"].eq("free_form_10d")].sort_values("row_id")
    original = action_table[action_table["mode"].eq("free_form_10d")]
    mean_vec = original[space.columns].mean(axis=0).to_numpy(dtype=float)
    expected = -mean_vec
    got = target[space.columns].to_numpy(dtype=float)
    if not np.allclose(got, np.tile(expected, (len(target), 1))):
        _fail(errors, f"sign_flip_mean_vector did not apply the negative group mean; expected={expected.tolist()} got={got.tolist()}")
    if not set(target["projection_method"].astype(str)) == {"sign_flip_mean_vector"}:
        _fail(errors, "sign_flip_mean_vector did not stamp projection_method correctly")
    if not (diag["sign_multiplier"].astype(float) == -1.0).all():
        _fail(errors, "sign_flip diagnostics must record sign_multiplier=-1")
    return {
        "expected_negative_mean": expected.tolist(),
        "observed_first_row": got[0].tolist() if len(got) else [],
        "n_rows": int(len(target)),
    }


def _check_shuffle_permutation_contract(errors: list[str]) -> dict[str, Any]:
    space = _SyntheticActionSpace()
    rows = []
    strata = []
    for row_id, industry, vector in [
        (1, "A", (1.0, -1.0, 0.1)),
        (2, "A", (2.0, -2.0, 0.2)),
        (3, "A", (3.0, -3.0, 0.3)),
        (4, "B", (4.0, -4.0, 0.4)),
        (5, "B", (5.0, -5.0, 0.5)),
        (6, "B", (6.0, -6.0, 0.6)),
    ]:
        rows.append({
            "row_id": row_id,
            "policy": "C6",
            "mode": "free_form_10d",
            "candidate_id": "ORIGINAL",
            "projection_distance": 0.1,
            "projection_method": "original",
            "out_of_library": True,
            "ppe_pct": vector[0],
            "short_debt_pct": vector[1],
            "cogs_ratio_chg": vector[2],
        })
        strata.append(industry)
    action_table = pd.DataFrame(rows)
    stratum_series = pd.Series(strata, index=action_table.index)
    source_industry = dict(zip(action_table.index, strata))
    for seed in _parse_seed_spec("1:5"):
        transformed, diag = _variant_rows(
            action_table,
            space=space,
            policies=["C6"],
            modes=["free_form_10d"],
            variant="row_shuffle_vector_null",
            target_budget="native",
            active_eps=1e-9,
            random_seed=seed,
            shuffle_within="industry",
            shuffle_strata=stratum_series,
        )
        if not (diag["shuffle_within"].astype(str) == "industry").all():
            _fail(errors, "conditional shuffle diagnostics lost shuffle_within=industry")
        if bool(diag.loc[diag["shuffle_group_size"].astype(int) > 1, "shuffle_kept_same_row"].any()):
            _fail(errors, f"conditional shuffle seed={seed} retained a fixed point in a non-singleton group")
        for _, drow in diag.iterrows():
            source_idx = int(drow["null_source_index"])
            if source_industry[source_idx] != str(drow["shuffle_stratum"]):
                _fail(errors, f"conditional shuffle seed={seed} crossed industry strata")
        for industry in sorted(set(strata)):
            original = action_table.loc[stratum_series.eq(industry), space.columns].to_numpy(dtype=float)
            shuffled = transformed.loc[stratum_series.eq(industry), space.columns].to_numpy(dtype=float)
            original_multiset = sorted(map(tuple, np.round(original, 12)))
            shuffled_multiset = sorted(map(tuple, np.round(shuffled, 12)))
            if original_multiset != shuffled_multiset:
                _fail(errors, f"conditional shuffle seed={seed} changed the vector multiset in industry={industry}")

    per_draw = pd.DataFrame({
        "policy": ["C6"] * 5,
        "mode": ["free_form_10d"] * 5,
        "oracle_backend": ["alpha"] * 5,
        "reference_label": ["C3_candidate_iql"] * 5,
        "shuffle_within": ["industry"] * 5,
        "mean_original_score": [1.0] * 5,
        "mean_shuffled_score": [0.7, 0.8, 0.9, 0.75, 0.85],
        "mean_matching_gain_original_minus_shuffle": [0.3, 0.2, 0.1, 0.25, 0.15],
        "mean_gap_shuffle_vs_reference": [0.1, 0.2, 0.3, 0.15, 0.25],
        "shuffle_fixed_point_count": [0] * 5,
        "shuffle_singleton_row_count": [0] * 5,
        "shuffle_stratum_count": [2] * 5,
    })
    aggregate = _aggregate_shuffle_draws(per_draw)
    if int(aggregate.loc[0, "n_shuffle_draws"]) != 5:
        _fail(errors, "shuffle aggregation did not retain all draws")
    if float(aggregate.loc[0, "matching_gain_mean"]) <= 0:
        _fail(errors, "shuffle aggregation produced a non-positive matching gain in the positive synthetic case")
    return {
        "seeds": _parse_seed_spec("1:5"),
        "shuffle_within": "industry",
        "stratum_count": 2,
        "aggregate_matching_gain_mean": float(aggregate.loc[0, "matching_gain_mean"]),
        "all_non_singleton_groups_deranged": bool(aggregate.loc[0, "all_non_singleton_groups_deranged"]),
    }


def _check_tost_contract(errors: list[str]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        row_ids = list(range(1, 31))
        reference = pd.DataFrame({
            "row_id": row_ids,
            "policy": ["C3"] * len(row_ids),
            "mode": ["free_form_10d"] * len(row_ids),
            "delta_R_score_alpha": np.linspace(0.50, 0.79, len(row_ids)),
        })
        target = reference.copy()
        target["policy"] = "projection"
        target["delta_R_score_alpha"] = target["delta_R_score_alpha"] + 0.01
        ref_path = td_path / "reference.csv"
        target_path = td_path / "target.csv"
        reference.to_csv(ref_path, index=False)
        target.to_csv(target_path, index=False)
        meta = run_tost_equivalence(
            target_scores=target_path,
            reference_scores=ref_path,
            out_dir=td_path / "tost_out",
            equivalence_margin=0.05,
            alpha=0.05,
            target_policy="projection",
            target_mode="free_form_10d",
            reference_policy="C3",
            reference_mode="free_form_10d",
            backends=["alpha"],
        )
        result = pd.read_csv(td_path / "tost_out" / "tost_equivalence_results.csv")
        equivalent = bool(result.loc[0, "equivalent"])
        if not equivalent:
            _fail(errors, "TOST synthetic near-zero gap should be equivalent within explicit margin=0.05")
        return {
            "metadata_status": meta.get("status"),
            "p_tost": float(result.loc[0, "p_tost"]),
            "equivalent": equivalent,
            "n_pairs": int(result.loc[0, "n_pairs"]),
        }


def run_verification() -> dict[str, Any]:
    errors: list[str] = []
    checks = {
        "feasibility_contract": _check_feasibility_contract(errors),
        "sign_flip_mean_vector": _check_sign_flip_contract(errors),
        "shuffle_permutation": _check_shuffle_permutation_contract(errors),
        "tost_equivalence": _check_tost_contract(errors),
    }
    return {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify non-N5 LLM analysis extension contracts with synthetic data")
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()
    summary = run_verification()
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
