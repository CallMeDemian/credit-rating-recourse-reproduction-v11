from __future__ import annotations

"""Stage 9 — LLM–RL Comparison and Revision Analysis.

Reads Stage 6 (RL/candidate/baseline scores) and Stage 8 (LLM scores),
produces the paired comparison table that places every LLM condition on
the reference ladder, computes revision-behavior metrics for C6/C7 (with
C8 explicitly NA per LLM contract §8), and aggregates the Stage 7 failure
audit.

Outputs (under ``data/final_freeze/stage9_llm_rl_comparison/``):

* ``llm_stage9_llm_rl_comparison.csv`` — paired delta_R_score per row per
  policy across Alpha/Beta/Gamma, plus per-policy summary.
* ``llm_stage9_revision_metrics.csv`` — RL Adoption / Self-Retention /
  Orthogonal Drift + per-backend before/after deltas for each C6 and C7
  (and C8 marked NA).
* ``llm_stage9_failure_audit.csv`` — aggregated failure taxonomy counts by
  condition and mode (sourced from Stage 7's per-row failure audit).
* ``metadata.json`` — stage status and the hashes that bind Stage 9 to the
  Stage 6 + Stage 8 substrate.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.contracts.stage_paths import stage_dir, final_root
from credit_recourse.rl.common.actions import (
    active_config_hashes,
    load_action_space,
    resolve_candidate_library_path,
)
from credit_recourse.rl.common.io import read_parquet_required, write_json

from .revision_metrics import build_identity_contrast_table, build_revision_table


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_stage6_delta_R_by_row(project_root: Path) -> pd.DataFrame:
    """Read the Stage 6 multi-oracle eval and return the per-row delta_R_score
    frame across Alpha/Beta/Gamma plus policy/candidate id columns."""
    p = stage_dir(project_root, "stage6") / "multi_oracle_policy_eval.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Stage 9 requires Stage 6 multi_oracle_policy_eval.parquet at {p}."
        )
    df = read_parquet_required(p)
    needed = {"row_id", "policy", "candidate_id"}
    miss = needed - set(df.columns)
    if miss:
        raise ValueError(f"Stage 6 multi_oracle_policy_eval missing columns: {miss}")
    return df


def _load_stage8_delta_R_by_row(project_root: Path) -> pd.DataFrame:
    p = stage_dir(project_root, "stage8") / "llm_stage8_multi_oracle_scores.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Stage 9 requires Stage 8 llm_stage8_multi_oracle_scores.parquet at {p}."
        )
    return read_parquet_required(p)


def _derive_fixed_and_oracle_best(stage6: pd.DataFrame, space) -> dict:
    """Compute the C_fix (best single fixed candidate by mean delta_R_score_alpha
    over candidate-uniform rows) and the per-firm oracle-best ceiling.

    The Stage 6 multi_oracle_policy_eval contains every fixed candidate
    evaluated uniformly across rows (one row per (row_id, policy)).  This
    helper distills the two derived comparators.
    """
    # Restrict to the fixed 11 candidate labels by positive whitelist.  Do not
    # infer fixed candidates through a blacklist: C1_random_uniform, C2, C3,
    # C_obs, LLM rows, scenario candidates, or diagnostic labels must never
    # enter C_fix/ceiling even if upstream names change.
    fixed_labels = set(space.train_labels)
    cand_df = stage6[stage6["policy"].astype(str).isin(fixed_labels)].copy()
    if cand_df.empty:
        return {"C_fix": None, "C_fix_mean_alpha": None, "oracle_best_ceiling_mean_alpha": None}

    # C_fix = single fixed candidate with highest mean delta_R_score_alpha.
    if "delta_R_score_alpha" not in cand_df.columns:
        return {"C_fix": None, "C_fix_mean_alpha": None, "oracle_best_ceiling_mean_alpha": None}
    grp = cand_df.groupby("policy")["delta_R_score_alpha"].mean().sort_values(ascending=False)
    if grp.empty:
        return {"C_fix": None, "C_fix_mean_alpha": None, "oracle_best_ceiling_mean_alpha": None}
    c_fix = str(grp.index[0])
    c_fix_mean = float(grp.iloc[0])

    # Oracle-best ceiling = per row, pick the highest delta_R_score_alpha
    # across all fixed candidates; average those picks.
    per_row_best = cand_df.groupby("row_id")["delta_R_score_alpha"].max()
    oracle_best = float(per_row_best.mean()) if not per_row_best.empty else float("nan")

    return {
        "C_fix": c_fix,
        "C_fix_mean_alpha": c_fix_mean,
        "oracle_best_ceiling_mean_alpha": oracle_best,
    }


def _stack_for_comparison(
    stage6: pd.DataFrame, stage8: pd.DataFrame
) -> pd.DataFrame:
    """Stack Stage 6 and Stage 8 delta_R_score rows into one comparison frame.

    Stage 8 rows carry a ``mode`` column (``candidate_selection`` /
    ``free_form_10d``); Stage 6 rows have no ``mode``.  The stacked frame
    keeps ``mode`` and fills the Stage 6 rows with ``"rl_native"`` so a
    single comparison can be filtered or grouped by mode without losing
    Stage 6 reference rows.
    """
    cols_keep = ["row_id", "policy", "candidate_id"] + (
        ["mode"] if "mode" in stage8.columns else []
    ) + [f"delta_R_score_{bk}" for bk in ["alpha", "beta", "gamma"]]
    s6 = stage6[[c for c in cols_keep if c in stage6.columns]].copy()
    s8 = stage8[[c for c in cols_keep if c in stage8.columns]].copy()
    if "mode" not in s6.columns:
        s6["mode"] = "rl_native"
    if "mode" not in s8.columns:
        s8["mode"] = "unspecified"
    s6["source"] = "stage6"
    s8["source"] = "stage8"
    return pd.concat([s6, s8], ignore_index=True, sort=False)


def _policy_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # Group on (policy, mode) so per-mode statistics are reported separately
    # for LLM stages that exercise both modes.
    group_keys = ["policy"] + (["mode"] if "mode" in comparison.columns else [])
    for keys, g in comparison.groupby(group_keys):
        if isinstance(keys, tuple):
            pol = str(keys[0]); mode_val = str(keys[1]) if len(keys) > 1 else ""
        else:
            pol = str(keys); mode_val = ""
        rec: dict = {
            "policy": pol,
            "mode": mode_val,
            "n_rows": int(g["row_id"].nunique()),
            "source": str(g["source"].iloc[0]) if "source" in g.columns else "",
        }
        for bk in ["alpha", "beta", "gamma"]:
            col = f"delta_R_score_{bk}"
            if col not in g.columns:
                continue
            s = pd.to_numeric(g[col], errors="coerce")
            rec[f"mean_delta_R_score_{bk}"] = float(s.mean()) if s.notna().any() else float("nan")
            rec[f"median_delta_R_score_{bk}"] = float(s.median()) if s.notna().any() else float("nan")
            rec[f"std_delta_R_score_{bk}"] = float(s.std()) if s.notna().any() else float("nan")
            rec[f"positive_fraction_{bk}"] = float((s > 0).mean()) if s.notna().any() else float("nan")
            rec[f"valid_fraction_{bk}"] = float(s.notna().mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        sort_cols = ["policy"] + (["mode"] if "mode" in out.columns else [])
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def _paired_against(comparison: pd.DataFrame, reference_policy: str) -> pd.DataFrame:
    """Row-wise paired delta against a single reference policy (typically
    ``C0_noop``, ``C_fix``, ``C_obs``, or the RL policy).

    Reference (Stage 6) rows are joined per-row_id; the LLM policy rows
    carry ``mode`` and are paired against the single mode-agnostic reference.
    """
    ref = comparison[comparison["policy"] == reference_policy].copy()
    if ref.empty:
        return pd.DataFrame(columns=[
            "row_id", "policy", "mode", "reference_policy",
            "gap_alpha", "gap_beta", "gap_gamma",
        ])
    keep = ["row_id"] + [f"delta_R_score_{bk}" for bk in ["alpha", "beta", "gamma"] if f"delta_R_score_{bk}" in ref.columns]
    ref_keep = ref[keep].drop_duplicates(subset=["row_id"]).rename(columns={
        f"delta_R_score_{bk}": f"ref_delta_R_score_{bk}" for bk in ["alpha", "beta", "gamma"]
    })
    other = comparison[comparison["policy"] != reference_policy].copy()
    merged = other.merge(ref_keep, on="row_id", how="inner")
    for bk in ["alpha", "beta", "gamma"]:
        c_pol = f"delta_R_score_{bk}"
        c_ref = f"ref_delta_R_score_{bk}"
        if c_pol in merged.columns and c_ref in merged.columns:
            merged[f"gap_{bk}"] = pd.to_numeric(merged[c_pol], errors="coerce") - pd.to_numeric(merged[c_ref], errors="coerce")
    merged["reference_policy"] = reference_policy
    keep_cols = ["row_id", "policy", "mode", "candidate_id", "reference_policy", "source"]
    keep_cols += [f"gap_{bk}" for bk in ["alpha", "beta", "gamma"] if f"gap_{bk}" in merged.columns]
    return merged[[c for c in keep_cols if c in merged.columns]].copy()


def _aggregate_failure_audit(project_root: Path) -> pd.DataFrame:
    """Aggregate per-row failure audit into condition-level counts.

    Prefer Stage 8's enriched audit when present because feasibility_violation
    is a post-simulation category.  Fall back to Stage 7 only for legacy runs.
    """
    p = stage_dir(project_root, "stage8") / "llm_stage8_failure_audit_enriched.csv"
    if not p.exists():
        p = stage_dir(project_root, "stage7") / "llm_stage7_failure_audit.csv"
    if not p.exists():
        return pd.DataFrame(columns=[
            "policy", "mode", "information_condition",
            "n_rows", "n_routed_to_simulator", "n_with_any_failure",
            "fail_translational_failure", "fail_structural_out_of_scope",
            "fail_direction_error", "fail_magnitude_error",
            "fail_feasibility_violation", "fail_liquidity_destructive_recourse",
            "fail_anchoring_or_confirmation_failure", "fail_ungrounded_judgment",
        ])
    df = pd.read_csv(p)
    df["failure_categories_list"] = (
        df["failure_categories"].fillna("").astype(str).apply(lambda s: [t for t in s.split(",") if t])
    )
    df["has_any_failure"] = df["failure_categories_list"].apply(lambda lst: len(lst) > 0)
    cats = [
        "translational_failure", "structural_out_of_scope",
        "direction_error", "magnitude_error",
        "feasibility_violation", "liquidity_destructive_recourse",
        "anchoring_or_confirmation_failure", "ungrounded_judgment",
    ]
    for c in cats:
        df[f"has_{c}"] = df["failure_categories_list"].apply(lambda lst: c in lst)
    rows = []
    for (pol, mode, ic), g in df.groupby(["policy", "mode", "information_condition"]):
        rec = {
            "policy": str(pol),
            "mode": str(mode),
            "information_condition": str(ic),
            "n_rows": int(len(g)),
            "n_routed_to_simulator": int(g["routed_to_simulator"].sum()),
            "n_with_any_failure": int(g["has_any_failure"].sum()),
        }
        for c in cats:
            rec[f"fail_{c}"] = int(g[f"has_{c}"].sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["policy", "mode", "information_condition"]).reset_index(drop=True)


def run_stage9(
    *,
    project_root: Path,
) -> dict:
    """Execute Stage 9 end-to-end and write all outputs."""
    project_root = Path(project_root).resolve()
    out = stage_dir(project_root, "stage9")
    out.mkdir(parents=True, exist_ok=True)

    base_hashes = active_config_hashes(project_root)
    stage8_meta_path_for_space = stage_dir(project_root, "stage8") / "metadata.json"
    if not stage8_meta_path_for_space.exists():
        raise FileNotFoundError(f"Missing Stage 8 metadata for candidate-library lineage: {stage8_meta_path_for_space}")
    stage8_meta_for_space = json.loads(stage8_meta_path_for_space.read_text(encoding="utf-8"))
    q_raw = stage8_meta_for_space.get("candidate_library_quantile")
    if q_raw is None:
        raise ValueError(
            "Stage 9 requires Stage 8 metadata candidate_library_quantile; "
            "rerun Stage 7/8 with --candidate-library-quantile 50."
        )
    selected_candidate_library_path = resolve_candidate_library_path(project_root, magnitude_quantile=int(q_raw))
    space = load_action_space(project_root, candidate_library_path=selected_candidate_library_path)
    hashes = dict(base_hashes)
    hashes["candidate_library_hash"] = space.candidate_library_hash
    hashes["candidate_library_path"] = str(selected_candidate_library_path)
    if stage8_meta_for_space.get("candidate_library_hash") != space.candidate_library_hash:
        raise ValueError(
            f"Stage 9 candidate_library_hash mismatch with Stage 8 selected library: "
            f"stage8={stage8_meta_for_space.get('candidate_library_hash')} selected={space.candidate_library_hash}"
        )

    # --- Stage 6 + Stage 8 substrate ---
    stage6 = _load_stage6_delta_R_by_row(project_root)
    stage8 = _load_stage8_delta_R_by_row(project_root)
    comparison = _stack_for_comparison(stage6, stage8)
    comparison.to_parquet(out / "llm_stage9_llm_rl_comparison.parquet", index=False)
    comparison.to_csv(out / "llm_stage9_llm_rl_comparison.csv", index=False, encoding="utf-8-sig")

    summary = _policy_summary(comparison)
    summary.to_csv(out / "llm_stage9_policy_summary.csv", index=False, encoding="utf-8-sig")

    # --- Paired ladder views ---
    derived = _derive_fixed_and_oracle_best(stage6, space)
    for reference in ["C0_noop", "C_obs"]:
        paired = _paired_against(comparison, reference)
        paired.to_csv(out / f"llm_stage9_paired_vs_{reference}.csv", index=False, encoding="utf-8-sig")
    if derived["C_fix"] is not None:
        paired_cfix = _paired_against(comparison, derived["C_fix"])
        paired_cfix.to_csv(out / f"llm_stage9_paired_vs_C_fix_{derived['C_fix']}.csv", index=False, encoding="utf-8-sig")
    paired_rl = _paired_against(comparison, space.final_rl_label)
    paired_rl.to_csv(out / f"llm_stage9_paired_vs_{space.final_rl_label}.csv", index=False, encoding="utf-8-sig")

    # --- Revision metrics ---
    s7_action_table_path = stage_dir(project_root, "stage7") / "llm_stage7_action_table.parquet"
    if not s7_action_table_path.exists():
        raise FileNotFoundError(
            f"Stage 9 requires Stage 7 llm_stage7_action_table.parquet at {s7_action_table_path}."
        )
    action_table = read_parquet_required(s7_action_table_path)
    revision_df = build_revision_table(
        action_table=action_table,
        stage8_scores=stage8,
        space=space,
    )
    revision_df.to_csv(out / "llm_stage9_revision_metrics.csv", index=False, encoding="utf-8-sig")
    identity_df = build_identity_contrast_table(revision_df)
    identity_df.to_csv(out / "llm_stage9_identity_contrast.csv", index=False, encoding="utf-8-sig")

    # Enforce LLM contract §8 fail-fast: no C8 row may have a defined
    # within-case before/after revision metric.
    c8_rows = revision_df[revision_df["base_condition"] == "C8"]
    if (c8_rows["metrics_defined"].fillna(False).astype(bool)).any():
        raise ValueError(
            "Stage 9 fail-fast: C8 row reported metrics_defined=True; "
            "within-case revision metrics must be NA for C8."
        )

    # --- Failure audit aggregation ---
    failure_audit = _aggregate_failure_audit(project_root)
    failure_audit.to_csv(out / "llm_stage9_failure_audit.csv", index=False, encoding="utf-8-sig")

    # --- Metadata ---
    stage7_meta_path = stage_dir(project_root, "stage7") / "metadata.json"
    stage8_meta_path = stage_dir(project_root, "stage8") / "metadata.json"
    stage7_meta = json.loads(stage7_meta_path.read_text(encoding="utf-8")) if stage7_meta_path.exists() else {}
    stage8_meta = json.loads(stage8_meta_path.read_text(encoding="utf-8")) if stage8_meta_path.exists() else {}

    meta = {
        "stage": "final_stage9_llm_rl_comparison",
        "status": "PASS",
        "created_utc": _now(),
        "candidate_library_hash": space.candidate_library_hash,
        "candidate_library_path": str(selected_candidate_library_path),
        "candidate_action_values_source": "stage2_recalibrated_candidate_library",
        "candidate_library_quantile": int(q_raw),
        "selected_recalibrated_candidate_library_hash": space.candidate_library_hash,
        "selected_recalibrated_candidate_library_path": str(selected_candidate_library_path),
        "base_candidate_library_hash": base_hashes["candidate_library_hash"],
        "base_candidate_library_path": base_hashes["candidate_library_path"],
        "final_action_contract_hash": base_hashes["final_action_contract_hash"],
        "stage6_multi_oracle_eval_consumed": str(stage_dir(project_root, "stage6") / "multi_oracle_policy_eval.parquet"),
        "stage8_multi_oracle_scores_consumed": str(stage_dir(project_root, "stage8") / "llm_stage8_multi_oracle_scores.parquet"),
        "stage7_backend_is_live": bool(stage7_meta.get("backend_is_live")),
        "final_paper_run_allowed": bool(stage7_meta.get("final_paper_run_allowed") and stage8_meta.get("final_paper_run_allowed")),
        "derived_comparators": derived,
        "row_count_compared": int(comparison["row_id"].nunique()),
        "policies_compared": sorted(set(comparison["policy"].astype(str).unique())),
        "revision_metrics_row_count": int(len(revision_df)),
        "identity_contrast_row_count": int(len(identity_df)),
        "c6x_identity_contrast_defined": bool(len(identity_df) > 0) if "C6X" in set(action_table.get("policy", pd.Series(dtype=str)).astype(str)) else None,
        "c8_within_case_revision_metrics_undefined": True,
        "outputs": {
            "llm_rl_comparison_parquet": "llm_stage9_llm_rl_comparison.parquet",
            "llm_rl_comparison_csv": "llm_stage9_llm_rl_comparison.csv",
            "policy_summary_csv": "llm_stage9_policy_summary.csv",
            "revision_metrics_csv": "llm_stage9_revision_metrics.csv",
            "identity_contrast_csv": "llm_stage9_identity_contrast.csv",
            "failure_audit_csv": "llm_stage9_failure_audit.csv",
        },
    }
    write_json(out / "metadata.json", meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 9 — LLM–RL Comparison and Revision Analysis")
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args(argv)
    meta = run_stage9(project_root=Path(args.project_root))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
