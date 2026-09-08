"""Stage 6 derived comparators: C_fix, per-firm oracle-best, and C1 random-uniform.

Post-processing only. Reads the already-scored Stage 6 output and derives:
  * C_fix: the single fixed v32 candidate with the highest mean no-op-adjusted score.
  * oracle_best: the per-firm-year maximum no-op-adjusted score over fixed v32 candidates.
  * C1_random_uniform: per-firm mean (and std) over all fixed v32 candidates = expected
    score of a uniform random draw across the candidate library. Fulfils the H1 reference
    baseline (research design §3.3.4: "11개 표준행동의 평균 점수 = 무작위 기대값").
    Derived from already-scored rows; no simulator call.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.common.io import final_root, write_json
from credit_recourse.rl.common.actions import load_action_space

BACKENDS = ("alpha", "beta", "gamma")
# Whitelist note: _fixed_policy_labels iterates space.train_labels, so only
# v32 candidate IDs can ever enter the fixed set — ladder/baseline policy
# codes cannot leak in by construction.  The names below are belt-and-
# suspenders documentation of that intent (incl. the DERIVED C1 baseline,
# which is never a scored policy row; EVAL-P4-001 update 2026-07-04).
NON_FIXED_POLICIES = {"C0_noop", "C1_random_uniform", "C2_weakest_component_rule", "C_obs"}


def _safe_mean(s: pd.Series) -> float | None:
    x = pd.to_numeric(s, errors="coerce").dropna()
    return float(x.mean()) if len(x) else None


def _fixed_policy_labels(root: Path) -> list[str]:
    space = load_action_space(root)
    labels: list[str] = []
    for c in list(space.train_labels):
        if c == "A0_noop":
            continue
        if c in NON_FIXED_POLICIES or c == space.final_rl_label:
            continue
        labels.append(c)
    return labels


def _read_backend_scores(selector_out: Path, backend: str) -> pd.DataFrame:
    path = selector_out / f"oracle_scores_{backend}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    merged = selector_out / "multi_oracle_policy_eval.parquet"
    if merged.exists():
        df = pd.read_parquet(merged)
        col = f"R_score_{backend}"
        needed = ["row_id", "policy", "candidate_id", col]
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise ValueError(f"{merged} missing columns for {backend}: {missing}")
        return df[needed].copy()
    raise FileNotFoundError(f"Missing Stage6 scores for {backend}: {path} or {merged}")


def _ensure_delta(scores: pd.DataFrame, backend: str) -> pd.DataFrame:
    score_col = f"R_score_{backend}"
    delta_col = f"delta_R_score_{backend}"
    out = scores.copy()
    if delta_col in out.columns:
        out["adj_score"] = pd.to_numeric(out[delta_col], errors="coerce")
        return out
    if score_col not in out.columns:
        raise ValueError(f"Missing {score_col}")
    if "C0_noop" not in set(out["policy"].astype(str)):
        raise ValueError(f"{backend}: C0_noop rows absent; cannot derive no-op-adjusted comparator")
    noop = out[out["policy"].astype(str) == "C0_noop"][["row_id", score_col]].copy()
    if noop["row_id"].duplicated().any():
        raise ValueError(f"{backend}: duplicate C0_noop row_id values")
    noop = noop.rename(columns={score_col: "noop_score"})
    out = out.merge(noop, on="row_id", how="left")
    if out["noop_score"].isna().any():
        raise ValueError(f"{backend}: missing noop baseline score for some rows")
    out["adj_score"] = pd.to_numeric(out[score_col], errors="coerce") - pd.to_numeric(out["noop_score"], errors="coerce")
    return out


def _evaluate_backend(scores: pd.DataFrame, backend: str, fixed_labels: list[str], final_rl_label: str) -> dict[str, Any]:
    adj = _ensure_delta(scores, backend)
    present = set(adj["policy"].astype(str))
    fixed_present = [p for p in fixed_labels if p in present]
    if not fixed_present:
        return {"backend": backend, "status": "FAIL", "error": "no fixed v32 candidate policies present"}
    fixed = adj[adj["policy"].astype(str).isin(fixed_present)].dropna(subset=["adj_score"]).copy()
    if fixed.empty:
        return {"backend": backend, "status": "FAIL", "error": "fixed candidate adjusted scores are empty"}

    means = fixed.groupby("policy")["adj_score"].mean().sort_values(ascending=False)
    c_fix_label = str(means.index[0])
    c_fix_mean = float(means.iloc[0])

    # Oracle-best ceiling: best fixed candidate per firm-year.
    per_row_best = fixed.groupby("row_id")["adj_score"].max()
    oracle_best_mean = float(per_row_best.mean()) if len(per_row_best) else None

    # C1 random-uniform: per-firm mean over all fixed candidates (= uniform random draw expected value).
    # Per research design §3.3.4: report mean + std of per-firm scores across the 11 candidates.
    # Uses all fixed_present candidates so the denominator matches what the LLM is also choosing from.
    per_row_c1_mean = fixed.groupby("row_id")["adj_score"].mean()  # per-firm uniform mean
    per_row_c1_std  = fixed.groupby("row_id")["adj_score"].std(ddof=1).fillna(0.0)  # per-firm candidate std
    c1_mean = float(per_row_c1_mean.mean()) if len(per_row_c1_mean) else None
    c1_std_mean = float(per_row_c1_std.mean()) if len(per_row_c1_std) else None  # mean within-firm std

    policy_means = adj.dropna(subset=["adj_score"]).groupby("policy")["adj_score"].mean()
    policy_table = {
        str(k): {
            "mean_noop_adjusted_score": float(v),
            "gap_to_c_fix": float(v - c_fix_mean),
            "gap_to_oracle_best": (float(v - oracle_best_mean) if oracle_best_mean is not None else None),
        }
        for k, v in policy_means.items()
    }

    return {
        "backend": backend,
        "status": "PASS",
        "metric": f"delta_R_score_{backend}",
        "c_fix_label": c_fix_label,
        "c_fix_mean_noop_adjusted_score": c_fix_mean,
        "oracle_best_mean_noop_adjusted_score": oracle_best_mean,
        "selection_headroom_oracle_best_minus_c_fix": (float(oracle_best_mean - c_fix_mean) if oracle_best_mean is not None else None),
        "c1_random_uniform_mean_noop_adjusted_score": c1_mean,
        "c1_random_uniform_within_firm_std_mean": c1_std_mean,
        "n_fixed_candidate_labels": int(len(fixed_present)),
        "fixed_candidate_labels": fixed_present,
        "rl_policy_label": final_rl_label,
        "rl_gap_to_c_fix": policy_table.get(final_rl_label, {}).get("gap_to_c_fix"),
        "c_obs_gap_to_c_fix": policy_table.get("C_obs", {}).get("gap_to_c_fix"),
        "policy_means": policy_table,
    }


def derive_stage6_comparators(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    final = final_root(root)
    selector_out = final / "stage6_candidate_selector_eval"
    space = load_action_space(root)
    fixed_labels = _fixed_policy_labels(root)
    errors: list[str] = []
    per_backend: dict[str, Any] = {}
    for backend in BACKENDS:
        try:
            scores = _read_backend_scores(selector_out, backend)
            per_backend[backend] = _evaluate_backend(scores, backend, fixed_labels, space.final_rl_label)
            if per_backend[backend].get("status") != "PASS":
                errors.append(f"{backend}: {per_backend[backend].get('error', 'failed')}")
        except Exception as exc:
            per_backend[backend] = {"backend": backend, "status": "FAIL", "error": str(exc)}
            errors.append(f"{backend}: {exc}")

    result = {
        "stage": "final_stage6_derived_comparators",
        "status": "PASS" if not errors else "FAIL",
        "metric": "no-op-adjusted Oracle score change",
        "comparators": {
            "C_fix": "single fixed v32 candidate with the highest mean no-op-adjusted score",
            "oracle_best": "per-firm-year maximum no-op-adjusted score over fixed v32 candidates",
            "C1_random_uniform": "per-firm mean noop-adjusted score over all fixed v32 candidates (= uniform random draw expected value; std = within-firm candidate dispersion)",
        },
        "note": "C_fix/oracle_best are derived from already-scored fixed candidate policies; no simulator call is made here.",
        "fixed_candidate_policy_set": fixed_labels,
        "per_backend": per_backend,
        "errors": errors,
    }
    out = final / "ledgers" / "stage6_derived_comparators.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Derive Stage6 C_fix and oracle-best comparators")
    p.add_argument("--project-root", required=True)
    args = p.parse_args(argv)
    res = derive_stage6_comparators(args.project_root)
    print(json.dumps(res, ensure_ascii=False, indent=2, default=str))
    return 0 if res.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
