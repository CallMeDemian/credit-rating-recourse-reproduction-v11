from __future__ import annotations

"""Stage 8 — LLM Multi-Oracle Evaluation.

Consumes the Stage 7 LLM action table and evaluates it through the **exact
same** simulator + Alpha/Beta/Gamma scoring substrate Stage 6 uses for the
RL/candidate ladder.  No re-implementation: the simulator function
(``simulate_policy_states``) and scorer functions (``score_alpha``,
``score_beta_ordered_logit_params``, ``score_gamma_model``) are imported
directly from Stage 6.

Per the Stage 7-9 LLM contract §6 (Evaluation rule):

::

    (s_t, a_t) -> financial simulator -> ŝ_{t+1} -> Alpha/Beta/Gamma R_score
    delta_R_score_backend(policy) = R_score_backend(policy) - R_score_backend(A0_noop)

The no-op baseline (``A0_noop``) is **not** re-simulated.  It is read from
Stage 6's ``oracle_scores_{backend}.parquet`` so the delta_R_score baseline
is identical between RL and LLM stacks.  This guarantees Stage 9 comparisons
are paired on the same no-op-adjusted scale.

Fail-fast rules enforced (per Stage 7-9 LLM contract §9):

* Stage 8 reads the **same** backend params Stage 6 used (no re-train, no
  new artifacts).
* Stage 8 scores simulator output, not real ``s_{t+1}``.
* LLM rows are not inserted into Stage 6 policy ladder; Stage 8 writes to a
  separate directory.
* The Stage 6 substrate hashes are recorded and Stage 8 fails if Stage 6
  metadata is unavailable.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.contracts.stage_paths import stage_dir, final_root
from credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline import (
    load_registry,
    resolve_backend_artifact,
    score_alpha,
    score_beta_ordered_logit_params,
    score_gamma_model,
    simulate_policy_states,
)
from credit_recourse.rl.common.actions import (
    active_config_hashes,
    load_action_space,
    resolve_candidate_library_path,
)
from credit_recourse.rl.common.io import read_parquet_required, write_json
from .failure_enrichment import enrich_failure_audit


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_stage6_noop_scores(
    project_root: Path,
) -> dict[str, pd.DataFrame]:
    """Read the Stage 6 per-backend score frames and extract C0_noop rows.

    Returns ``{backend: DataFrame(row_id, R_score_<backend>)}`` for
    Alpha/Beta/Gamma.
    """
    s6 = stage_dir(project_root, "stage6")
    out: dict[str, pd.DataFrame] = {}
    for backend in ["alpha", "beta", "gamma"]:
        path = s6 / f"oracle_scores_{backend}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Stage 8 requires Stage 6 oracle_scores_{backend}.parquet at {path}."
            )
        df = read_parquet_required(path)
        noop = df[df["policy"].astype(str) == "C0_noop"][
            ["row_id", f"R_score_{backend}"]
        ].copy()
        if noop.empty:
            raise ValueError(
                f"Stage 6 oracle_scores_{backend} has no C0_noop rows; "
                f"cannot compute Stage 8 delta_R_score baseline."
            )
        out[backend] = noop.rename(columns={f"R_score_{backend}": f"noop_R_score_{backend}"})
    return out


def _stage6_substrate_hashes(project_root: Path) -> dict:
    """Record the Stage 6 substrate identity (backend registry + each backend's
    params artifact hash) so Stage 8 metadata can prove the same substrate
    was used."""
    final = final_root(project_root)
    reg_path = final / "configs" / "oracle_backend_registry.yaml"
    if not reg_path.exists():
        raise FileNotFoundError(f"Missing oracle backend registry: {reg_path}")
    reg = load_registry(reg_path)
    backends = reg.get("backends", {})
    hashes: dict[str, str] = {
        "oracle_backend_registry_sha256": _sha256_file(reg_path),
    }
    for bk in ["alpha", "beta", "gamma"]:
        bp = resolve_backend_artifact(project_root, final, backends.get(bk, {}).get("params", ""))
        if not bp.exists():
            raise FileNotFoundError(f"Missing {bk} params: {bp}")
        hashes[f"{bk}_params_sha256"] = _sha256_file(bp)
        model_val = backends.get(bk, {}).get("model", "")
        if model_val:
            mp = resolve_backend_artifact(project_root, final, model_val)
            if mp.exists():
                hashes[f"{bk}_model_sha256"] = _sha256_file(mp)
    return hashes




def _load_stage6_simulator_identity(project_root: Path) -> dict[str, object]:
    """Load the exact simulator/substrate identity recorded by Stage 6.

    Stage 8 must score LLM actions under the same deterministic simulator
    settings used to create Stage 6 no-op scores.  Do not fall back to
    simulate_policy_states defaults: default/False/None would silently make
    delta_R_score invalid because the numerator and no-op denominator would
    be different substrates.
    """
    s6 = stage_dir(project_root, "stage6")
    meta_path = s6 / "multi_oracle_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Stage 8 requires Stage 6 multi_oracle_metadata.json to preserve "
            f"simulator identity: {meta_path}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    required = [
        "sim_business_plan_mode",
        "preserve_current_non_current_residual",
        "predicted_fiscal_year",
    ]
    missing = [k for k in required if k not in meta or meta.get(k) is None]
    if missing:
        raise ValueError(
            "Stage 6 metadata lacks simulator identity keys required by Stage 8: "
            f"{missing}. Refusing to use simulate_policy_states defaults."
        )
    return {
        "stage6_metadata_path": str(meta_path),
        "sim_business_plan_mode": str(meta["sim_business_plan_mode"]),
        "preserve_current_non_current_residual": bool(meta["preserve_current_non_current_residual"]),
        "predicted_fiscal_year": int(meta["predicted_fiscal_year"]),
    }


def _validate_llm_action_table(
    df: pd.DataFrame, space, allowed_conditions: set[str]
) -> None:
    """Stage 8 input contract checks."""
    required = {"row_id", "policy", "candidate_id", "reference_source", "reference_draw_seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"LLM action table missing required columns: {missing}")

    action_cols = [c for c in df.columns if c.startswith("action__")]
    if action_cols != list(space.columns):
        raise ValueError(
            f"LLM action table action column order differs from final_action_contract: "
            f"got {action_cols}, expected {space.columns}"
        )

    pols = set(df["policy"].astype(str).unique())
    forbidden_pols = pols - allowed_conditions
    if forbidden_pols:
        raise ValueError(
            f"LLM action table contains forbidden policy codes (must be in "
            f"{sorted(allowed_conditions)}): {sorted(forbidden_pols)}"
        )


    bad_sources = set(df["reference_source"].dropna().astype(str).unique()) - {"rl", "random", "none"}
    if bad_sources:
        raise ValueError(f"LLM action table contains invalid reference_source values: {sorted(bad_sources)}")
    c6x = df[df["policy"].astype(str) == "C6X"]
    if not c6x.empty:
        if (c6x["reference_source"].astype(str) != "random").any():
            raise ValueError("C6X rows must carry reference_source='random'.")
        if c6x["reference_draw_seed"].isna().any():
            raise ValueError("C6X rows must carry non-null reference_draw_seed.")
    rl_ref_rows = df[df["policy"].astype(str).isin(["C6", "C7", "C8"])]
    if not rl_ref_rows.empty and (rl_ref_rows["reference_source"].astype(str) != "rl").any():
        raise ValueError("C6/C7/C8 rows must carry reference_source='rl'.")

    candidates = set(df["candidate_id"].astype(str).unique())
    bad = candidates - set(space.train_labels)
    if bad:
        raise ValueError(
            f"LLM action table contains candidates outside v32 main_train_labels: {sorted(bad)}"
        )

    # Fail-fast: no bound violation may exist post-clipping.  The Stage 7
    # parser is supposed to clip; this is the contract check at the Stage 8
    # boundary.
    for col in action_cols:
        raw_name = col.replace("action__", "")
        bkey = col if col in space.bounds else raw_name
        lo, hi = space.bounds[bkey]
        s = pd.to_numeric(df[col], errors="coerce")
        if (s < lo - 1e-9).any() or (s > hi + 1e-9).any():
            offending = df.loc[(s < lo - 1e-9) | (s > hi + 1e-9), ["row_id", "policy", col]].head(5)
            raise ValueError(
                f"Stage 8 LLM action {col} out of bounds [{lo},{hi}]; offending rows:\n{offending}"
            )


def _per_policy_summary(merged: pd.DataFrame) -> pd.DataFrame:
    """Compact policy-level summary across backends."""
    rows = []
    for pol, g in merged.groupby("policy"):
        rec: dict = {"policy": str(pol), "n_rows": int(len(g))}
        for bk in ["alpha", "beta", "gamma"]:
            col = f"delta_R_score_{bk}"
            if col not in g.columns:
                continue
            s = pd.to_numeric(g[col], errors="coerce")
            rec[f"mean_delta_R_score_{bk}"] = float(s.mean()) if s.notna().any() else float("nan")
            rec[f"median_delta_R_score_{bk}"] = float(s.median()) if s.notna().any() else float("nan")
            rec[f"std_delta_R_score_{bk}"] = float(s.std()) if s.notna().any() else float("nan")
            rec[f"valid_fraction_{bk}"] = float(s.notna().mean())
        rows.append(rec)
    return pd.DataFrame(rows)


def run_stage8(
    *,
    project_root: Path,
) -> dict:
    """Execute Stage 8 end-to-end and write all outputs."""
    project_root = Path(project_root).resolve()
    final = final_root(project_root)
    s7 = stage_dir(project_root, "stage7")
    out = stage_dir(project_root, "stage8")
    out.mkdir(parents=True, exist_ok=True)

    # --- base contract artifacts (action bounds + immutable config hash) ---
    base_hashes = active_config_hashes(project_root)

    # --- Stage 7 input ---
    action_table_path = s7 / "llm_stage7_action_table.parquet"
    if not action_table_path.exists():
        raise FileNotFoundError(
            f"Stage 8 requires Stage 7 output {action_table_path}."
        )
    stage7_meta_path = s7 / "metadata.json"
    if not stage7_meta_path.exists():
        raise FileNotFoundError(f"Missing Stage 7 metadata: {stage7_meta_path}")
    stage7_meta = json.loads(stage7_meta_path.read_text(encoding="utf-8"))

    # Load the exact candidate vectors used by Stage 7. This must be P50 by
    # default so LLM simulation is aligned with the Stage4/5/6 RL track; the
    # base active YAML is retained only as immutable provenance.
    q_raw = stage7_meta.get("candidate_library_quantile")
    if q_raw is None:
        raise ValueError(
            "Stage 8 requires Stage 7 metadata candidate_library_quantile; "
            "rerun Stage 7 with --candidate-library-quantile 50."
        )
    selected_candidate_library_path = resolve_candidate_library_path(
        project_root, magnitude_quantile=int(q_raw)
    )
    space = load_action_space(project_root, candidate_library_path=selected_candidate_library_path)
    hashes = dict(base_hashes)
    hashes["candidate_library_hash"] = space.candidate_library_hash
    hashes["candidate_library_path"] = str(selected_candidate_library_path)

    if stage7_meta.get("candidate_library_hash") != space.candidate_library_hash:
        raise ValueError(
            f"Stage 8 candidate_library_hash mismatch with Stage 7 selected library: "
            f"stage7={stage7_meta.get('candidate_library_hash')} "
            f"selected={space.candidate_library_hash} path={selected_candidate_library_path}"
        )
    if stage7_meta.get("final_action_contract_hash") != base_hashes["final_action_contract_hash"]:
        raise ValueError(
            f"Stage 8 final_action_contract_hash mismatch with Stage 7."
        )

    action_table = read_parquet_required(action_table_path)
    _validate_llm_action_table(
        action_table, space, allowed_conditions={"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"}
    )

    # --- shared state base from Stage 2 (same as Stage 6) ---
    base_path = stage_dir(project_root, "stage2") / "phase_eval_candidate.parquet"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing Stage 2 phase_eval_candidate.parquet at {base_path}.")
    base = read_parquet_required(base_path)

    # --- backend registry (identical to Stage 6) ---
    reg_path = final / "configs" / "oracle_backend_registry.yaml"
    reg = load_registry(reg_path)
    backends = reg.get("backends", {})
    if reg.get("final_result_allowed") is not True or reg.get("status") != "generated_by_stage1_oracle_development_verified":
        raise ValueError(
            "Stage 8 refuses to score: Oracle backend registry is not "
            "final/generated_by_stage1_oracle_development_verified."
        )

    # --- run the Stage 6 simulator on the LLM actions ---
    # Contract v4: substrate identity is non-negotiable.  The settings below
    # are read from Stage 6 metadata and passed explicitly; using the Stage 6
    # function defaults would silently score LLM actions on a different
    # simulator substrate than the C0_noop baseline consumed below.
    simulator_identity = _load_stage6_simulator_identity(project_root)
    sim_state, audit = simulate_policy_states(
        base,
        action_table,
        space,
        out,
        predicted_fiscal_year=int(simulator_identity["predicted_fiscal_year"]),
        preserve_current_non_current_residual=bool(
            simulator_identity["preserve_current_non_current_residual"]
        ),
        sim_business_plan_mode=str(simulator_identity["sim_business_plan_mode"]),
    )

    if "sim_business_plan_mode" in audit.columns:
        got_modes = set(audit["sim_business_plan_mode"].astype(str).dropna().unique())
        expected_mode = str(simulator_identity["sim_business_plan_mode"])
        if got_modes and got_modes != {expected_mode}:
            raise ValueError(
                f"Stage 8 simulator audit mode mismatch: got {sorted(got_modes)}, "
                f"expected {expected_mode}"
            )
    if "preserve_current_non_current_residual" in audit.columns:
        got_preserve = set(audit["preserve_current_non_current_residual"].dropna().astype(bool).unique())
        expected_preserve = bool(simulator_identity["preserve_current_non_current_residual"])
        if got_preserve and got_preserve != {expected_preserve}:
            raise ValueError(
                f"Stage 8 simulator audit preserve flag mismatch: got {sorted(got_preserve)}, "
                f"expected {expected_preserve}"
            )

    # --- post-simulation feasibility failure coding ---
    stage7_failure_path = s7 / "llm_stage7_failure_audit.csv"
    if not stage7_failure_path.exists():
        raise FileNotFoundError(
            f"Stage 8 requires Stage 7 failure audit for taxonomy enrichment: {stage7_failure_path}"
        )
    stage7_failure_audit = pd.read_csv(stage7_failure_path)
    enriched_failure_audit, failure_coder_manifest = enrich_failure_audit(
        stage7_failure_audit=stage7_failure_audit,
        simulated_state=sim_state,
        action_effect_audit=audit,
        out_dir=out,
    )

    # --- score with the Stage 6 scorers (Alpha/Beta/Gamma) ---
    # The LLM action table can contain two rows for the same (row_id, policy)
    # when both modes (candidate_selection, free_form_10d) are exercised in
    # one Stage 7 run.  Stage 8 keeps ``mode`` in the score-frame primary
    # key so the per-mode rows do not silently fan out under the three-way
    # outer merge.  Stage 9's revision-metric and comparison logic already
    # carries ``mode`` so this preserves the row-level pairing end to end.
    has_mode_col = "mode" in action_table.columns
    score_key_cols = ["row_id", "policy"] + (["mode"] if has_mode_col else []) + ["candidate_id"]

    all_scores = []
    for backend in ["alpha", "beta", "gamma"]:
        b = backends[backend]
        params = resolve_backend_artifact(project_root, final, b.get("params", ""))
        if not params.exists():
            raise FileNotFoundError(f"Missing {backend} params: {params}")
        if backend == "alpha":
            score = score_alpha(sim_state, params)
            col = "R_score_alpha"
        elif backend == "beta":
            score = score_beta_ordered_logit_params(sim_state, params)
            col = "R_score_beta"
        else:
            model = resolve_backend_artifact(project_root, final, b.get("model", ""))
            if not model.exists():
                raise FileNotFoundError(f"Missing gamma model artifact: {model}")
            score = score_gamma_model(sim_state, params, model)
            col = "R_score_gamma"
        scored = action_table[score_key_cols + list(space.columns)].copy()
        scored[col] = score.to_numpy()
        scored.to_parquet(out / f"llm_oracle_scores_{backend}.parquet", index=False)
        all_scores.append(scored[score_key_cols + [col]])

    merged = all_scores[0]
    for s in all_scores[1:]:
        merged = merged.merge(s, on=score_key_cols, how="outer")

    # --- pull C0_noop scores from Stage 6 and compute delta_R_score ---
    noop_by_backend = _load_stage6_noop_scores(project_root)
    for backend in ["alpha", "beta", "gamma"]:
        merged = merged.merge(noop_by_backend[backend], on="row_id", how="left")
        merged[f"delta_R_score_{backend}"] = (
            pd.to_numeric(merged[f"R_score_{backend}"], errors="coerce")
            - pd.to_numeric(merged[f"noop_R_score_{backend}"], errors="coerce")
        )

    merged.to_parquet(out / "llm_stage8_multi_oracle_scores.parquet", index=False)

    summary = _per_policy_summary(merged)
    summary.to_csv(out / "llm_stage8_policy_summary.csv", index=False, encoding="utf-8-sig")

    # --- metadata ---
    substrate_hashes = _stage6_substrate_hashes(project_root)
    meta = {
        "stage": "final_stage8_llm_multi_oracle_eval",
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
        "stage7_metadata_consumed": str(stage7_meta_path),
        "stage7_backend_is_live": bool(stage7_meta.get("backend_is_live")),
        "final_paper_run_allowed": bool(stage7_meta.get("final_paper_run_allowed")),
        "stage6_substrate_hashes": substrate_hashes,
        "stage6_simulator_identity": simulator_identity,
        "sim_business_plan_mode": simulator_identity["sim_business_plan_mode"],
        "preserve_current_non_current_residual": simulator_identity["preserve_current_non_current_residual"],
        "predicted_fiscal_year": simulator_identity["predicted_fiscal_year"],
        "no_separate_oracle_used": True,
        "scored_via_simulator_only": True,
        "delta_R_score_baseline_source": "stage6_oracle_scores_<backend>.parquet (C0_noop rows)",
        "row_count_evaluated": int(merged["row_id"].nunique()),
        "policies_evaluated": sorted(set(merged["policy"].astype(str).unique())),
        "outputs": {
            "multi_oracle_scores": "llm_stage8_multi_oracle_scores.parquet",
            "policy_summary": "llm_stage8_policy_summary.csv",
            "simulated_oracle_input_frame": "simulated_oracle_input_frame.parquet",
            "action_effect_audit": "action_effect_audit.parquet",
            "failure_audit_enriched": "llm_stage8_failure_audit_enriched.csv",
            "failure_coder_manifest": "failure_coder_manifest.json",
        },
        "failure_coder_manifest": failure_coder_manifest,
        "failure_audit_enriched_rows": int(len(enriched_failure_audit)),
    }
    write_json(out / "metadata.json", meta)
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 8 — LLM Multi-Oracle Evaluation")
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args(argv)
    meta = run_stage8(project_root=Path(args.project_root))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
