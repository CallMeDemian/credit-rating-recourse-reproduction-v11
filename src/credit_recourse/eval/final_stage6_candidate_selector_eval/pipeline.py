from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

torch.set_num_threads(1)

from credit_recourse.rl.common.io import final_root, write_json, read_parquet_required
from credit_recourse.rl.common.actions import load_action_space, assert_hashes_match
from credit_recourse.rl.common.torch_utils import transform_with_missing_mask, transform_categorical
from credit_recourse.rl.final_candidate.common import DiscreteIQL
from credit_recourse.rl.final_candidate.encoder import load_stage3_encoder_payload
from credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline import (
    PHI_COMPONENTS,
    LOWER_GOOD,
    materialize_phi_aliases,
    sector_col,
)
from credit_recourse.simulator.firm_state import ITEM_CODE_MAP
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_metadata


def _num(r, name, default=0.0):
    try:
        v = r.get(name, default)
        return float(v) if pd.notna(v) else default
    except Exception:
        return default


def _validated_checkpoint_action_frame(ckpt: dict, space, *, include_stage6_extras: bool) -> tuple[pd.DataFrame, dict]:
    """Validate the Stage2-recalibrated action payload embedded in Stage5.

    Stage6 is the first point where policy labels become simulator vectors. Any
    missing label/column, duplicate candidate ID, unknown candidate, non-finite
    value, or stale payload source must therefore hard-fail instead of silently
    reverting to the base YAML action vectors.
    """
    source = ckpt.get("candidate_action_values_source")
    if source != "stage2_recalibrated_candidate_library":
        raise ValueError(
            "Stage6 requires candidate_action_values_source="
            f"'stage2_recalibrated_candidate_library'; got {source!r}"
        )
    rows = ckpt.get("candidate_action_values")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Stage6 requires a non-empty candidate_action_values list in the Stage5 checkpoint")
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Stage6 candidate_action_values must contain only object rows")

    payload = pd.DataFrame(rows)
    if "candidate_id" not in payload.columns:
        raise ValueError("Stage6 candidate_action_values is missing candidate_id")
    payload["candidate_id"] = payload["candidate_id"].astype(str)
    duplicate_ids = sorted(payload.loc[payload["candidate_id"].duplicated(keep=False), "candidate_id"].unique().tolist())
    if duplicate_ids:
        raise ValueError(f"Stage6 candidate_action_values contains duplicate candidate_id values: {duplicate_ids}")

    all_known = space.frame(include_stage6_extras=True)["candidate_id"].astype(str).tolist()
    allowed = set(all_known)
    unknown = sorted(set(payload["candidate_id"]) - allowed)
    if unknown:
        raise ValueError(f"Stage6 candidate_action_values contains unknown candidates: {unknown}")

    required_ids = space.frame(include_stage6_extras=include_stage6_extras)["candidate_id"].astype(str).tolist()
    missing_ids = [candidate_id for candidate_id in required_ids if candidate_id not in set(payload["candidate_id"])]
    if missing_ids:
        raise ValueError(f"Stage6 candidate_action_values missing required candidates: {missing_ids}")

    missing_cols = [col for col in space.columns if col not in payload.columns]
    if missing_cols:
        raise ValueError(f"Stage6 candidate_action_values missing action columns: {missing_cols}")
    for col in space.columns:
        numeric = pd.to_numeric(payload[col], errors="coerce")
        bad_mask = ~np.isfinite(numeric.to_numpy(dtype=float))
        if bad_mask.any():
            bad_ids = payload.loc[bad_mask, "candidate_id"].tolist()
            raise ValueError(f"Stage6 candidate_action_values has non-finite {col} for candidates: {bad_ids}")
        payload[col] = numeric.astype(float)

    selected = payload.set_index("candidate_id").loc[required_ids].reset_index()
    canonical = selected[["candidate_id", *space.columns]].to_dict(orient="records")
    payload_hash = hashlib.sha256(
        json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = {
        "candidate_action_values_source": source,
        "candidate_action_value_count": int(len(payload)),
        "candidate_action_required_count": int(len(required_ids)),
        "candidate_action_values_sha256": payload_hash,
        "candidate_action_values_include_stage6_extras": bool(include_stage6_extras),
        "candidate_action_payload_validation": "PASS_FAIL_FAST_NO_BASE_FALLBACK",
    }
    return payload, audit


def _percentile(values: pd.Series, ref: np.ndarray) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    ref = np.asarray(ref, dtype=float)
    ref = ref[np.isfinite(ref)]
    if len(ref) == 0:
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    pct = np.searchsorted(np.sort(ref), arr, side="right") / max(len(ref), 1)
    pct[~np.isfinite(arr)] = np.nan
    return pd.Series(pct, index=values.index)


def _build_sector_refs(train_df: pd.DataFrame) -> tuple[dict[str, dict[str, np.ndarray]], str | None]:
    train_df = materialize_phi_aliases(train_df, next_state=False)
    missing = [c for c in PHI_COMPONENTS if c not in train_df.columns]
    if missing:
        raise ValueError(f"C2 sector-phi weakest rule requires frozen reference components; missing {missing}")
    sec = sector_col(train_df)
    refs: dict[str, dict[str, np.ndarray]] = {}
    for comp in PHI_COMPONENTS:
        refs[comp] = {}
        global_vals = pd.to_numeric(train_df[comp], errors="coerce").dropna().to_numpy(dtype=float)
        refs[comp]["__GLOBAL__"] = np.sort(global_vals)
        if sec and sec in train_df.columns:
            for key, g in train_df.groupby(sec, dropna=False):
                vals = pd.to_numeric(g[comp], errors="coerce").dropna().to_numpy(dtype=float)
                refs[comp][str(key)] = np.sort(vals) if len(vals) >= 20 else refs[comp]["__GLOBAL__"]
    return refs, sec


def c2_row_conditional(df: pd.DataFrame, space, reference_df: pd.DataFrame, fixed_actions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Row-conditional C2 baseline: frozen sector-relative weakest component top-1 rule.

    fixed_actions carries the active Stage2-recalibrated candidate magnitudes
    embedded in the selected Stage5 checkpoint. C2 is a row-conditional
    comparator, not a train label, but it must use the same active candidate
    vectors as the other Stage6 policies.
    """
    df = materialize_phi_aliases(df.copy(), next_state=False)
    reference_df = materialize_phi_aliases(reference_df.copy(), next_state=False)
    if fixed_actions is None or "candidate_id" not in fixed_actions.columns:
        raise ValueError("C2 requires the validated Stage5 checkpoint candidate action frame")
    active_fixed = fixed_actions.set_index("candidate_id")
    refs, sec = _build_sector_refs(reference_df)
    missing = [c for c in PHI_COMPONENTS if c not in df.columns]
    if missing:
        raise ValueError(f"C2 sector-phi weakest rule requires phase_eval components; missing {missing}")
    mapping = {
        "derived__debt_to_assets": "DL1_deleverage_mild",
        "derived__financial_cost_to_revenue": "DL1_deleverage_mild",
        "derived__cogs_to_revenue": "OE2_cost_efficiency_moderate",
        "derived__sga_to_revenue": "OE2_cost_efficiency_moderate",
        "derived__operating_margin": "OE2_cost_efficiency_moderate",
        "derived__roa_proxy": "OE2_cost_efficiency_moderate",
    }
    quality = pd.DataFrame(index=df.index)
    for comp in PHI_COMPONENTS:
        pct = pd.Series(index=df.index, dtype=float)
        if sec and sec in df.columns:
            for key, idx in df.groupby(sec, dropna=False).groups.items():
                ref = refs[comp].get(str(key), refs[comp]["__GLOBAL__"])
                pct.loc[idx] = _percentile(df.loc[idx, comp], ref)
        else:
            pct = _percentile(df[comp], refs[comp]["__GLOBAL__"])
        quality[comp] = (1.0 - pct if comp in LOWER_GOOD else pct).fillna(0.5)
    weakness = 1.0 - quality
    rows = []
    for idx, r in weakness.iterrows():
        component = str(r.astype(float).idxmax())
        name = mapping[component]
        if name not in active_fixed.index:
            raise ValueError(f"C2 mapped candidate missing from validated checkpoint action payload: {name}")
        vec = {col: float(active_fixed.loc[name, col]) for col in space.columns}
        vec["candidate_id"] = "C2_weakest_component_rule"
        vec["c2_mapped_candidate"] = name
        vec["c2_rule_id"] = "frozen_sector_phi_percentile_weakest_component_top1_rule"
        vec["c2_weakest_component"] = component
        vec["c2_weakest_component_quality_pct"] = float(quality.loc[idx, component])
        vec["c2_weakest_component_weakness"] = float(weakness.loc[idx, component])
        rows.append(vec)
    return pd.DataFrame(rows)



STAGE6_FIRMSTATE_FIELD_ALIASES = {
    "receivables": ["accounts_receivable"],
    "payables": ["accounts_payable"],
    "short_term_debt": ["short_debt"],
    "long_term_debt": ["long_debt"],
    "bonds": ["bond"],
}

def _field_alias_matches(column: str, field: str, code: str | None = None) -> bool:
    """Return whether a Stage6 phase_eval column can supply a FirmState field.

    Stage2 phase_eval is a current-state serving panel. Depending on the
    upstream materialization path, financial state variables can arrive as
    canonical simulator names (sim__revenue), raw state names (raw__sim__revenue
    or raw__revenue), AVS raw names (raw__avs__...), or U-code bearing Korean
    raw headers.  The audit must use the same alias tolerance as the simulator
    loader; otherwise valid state-only inputs are incorrectly reported as 0%
    coverage.
    """
    c = str(column)
    aliases = [field] + list(STAGE6_FIRMSTATE_FIELD_ALIASES.get(field, []))
    for alias in aliases:
        if c == alias or c == f"sim__{alias}":
            return True
        if c == f"raw__{alias}" or c == f"raw__sim__{alias}" or c == f"avs__{alias}" or c == f"raw__avs__{alias}":
            return True
        if c.endswith(f"__{alias}"):
            return True
    if code and code in c:
        return True
    return False

def _firm_state_input_audit(df: pd.DataFrame, out: Path) -> dict:
    required = [
        "revenue", "cogs", "sga", "total_assets", "current_assets", "current_liabilities", "cash",
        "inventory", "receivables", "payables", "ppe", "short_term_debt", "long_term_debt", "bonds",
        "total_liabilities", "total_equity",
    ]
    rows = []
    for field in required:
        code = next((k for k, v in ITEM_CODE_MAP.items() if v == field), None)
        matches = [c for c in df.columns if _field_alias_matches(str(c), field, code)]
        if matches:
            cov = max(float(pd.to_numeric(df[c], errors="coerce").notna().mean()) for c in matches)
        else:
            cov = 0.0
        rows.append({"field": field, "matched_columns": matches[:20], "coverage": cov, "status": "PASS" if cov >= 0.50 else "LOW_COVERAGE"})
    status = "PASS" if all(r["status"] == "PASS" for r in rows) else "FAIL"
    payload = {"schema_version": "stage6_firm_state_input_audit_v2_alias_aware", "status": status, "rows": rows}
    write_json(out / "firm_state_input_audit.json", payload)
    pd.DataFrame(rows).to_csv(out / "firm_state_input_audit.csv", index=False, encoding="utf-8-sig")
    if status != "PASS":
        bad = [r for r in rows if r["status"] != "PASS"]
        raise ValueError(f"Stage6 FirmState input audit failed; low coverage fields: {bad[:8]}")
    return payload


def _pairing_audit(pa: pd.DataFrame, out: Path) -> dict:
    all_rows = set(pd.to_numeric(pa["row_id"], errors="raise").astype(int).tolist())
    noop_rows = set(pd.to_numeric(pa.loc[pa["policy"].astype(str) == "C0_noop", "row_id"], errors="raise").astype(int).tolist())
    missing_noop = sorted(all_rows - noop_rows)
    policies = sorted(pa["policy"].astype(str).unique().tolist())
    payload = {
        "schema_version": "stage6_policy_pairing_audit_v1",
        "status": "PASS" if not missing_noop else "FAIL",
        "n_row_ids": len(all_rows),
        "n_noop_row_ids": len(noop_rows),
        "missing_noop_row_id_count": len(missing_noop),
        "missing_noop_row_id_sample": missing_noop[:20],
        "policies": policies,
        "c_obs_present": "C_obs" in policies,
    }
    # C_obs is forbidden on the primary phase_eval panel; it is only expected
    # on the secondary inner-dev diagnostic panel.
    write_json(out / "policy_pairing_audit.json", payload)
    if payload["status"] != "PASS":
        raise ValueError(f"Stage6 policy pairing audit failed: {payload}")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--include-stage6-extras", action="store_true", help="Opt-in scenario/diagnostic candidates for robustness only")
    args = ap.parse_args(argv)
    root = Path(args.project_root).resolve()
    temporal_contract = load_temporal_contract(root)
    final = final_root(root)
    space = load_action_space(root)
    out = final / "stage6_candidate_selector_eval"
    out.mkdir(parents=True, exist_ok=True)

    df = read_parquet_required(final / "stage2_candidate_projection" / "phase_eval_candidate.parquet")
    fy = pd.to_numeric(df.get('fiscal_year', df.get('year')), errors='coerce')
    vals = sorted([int(x) for x in fy.dropna().unique().tolist()])
    if vals != [int(temporal_contract.eval_base_year)]:
        raise ValueError({'message':'Stage6 phase_eval temporal contract failed','expected_eval_base_year':int(temporal_contract.eval_base_year),'observed_fiscal_years':vals})
    _firm_state_input_audit(df, out)

    # FINAL_FREEZE_STRICT_FINAL_REFIT_IQL_2026_05_24
    # Stage6 final evaluation must consume the full-train final-refit IQL checkpoint only.
    # Do not silently fall back to candidate_iql_policy.pt, which may be an inner-dev selection alias.
    ckpt_path = final / "stage5_candidate_iql" / "stage5_candidate_iql_final_refit_fulltrain.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError({
            "message": "Missing Stage5 final-refit IQL checkpoint required by Stage6",
            "required": str(ckpt_path),
            "forbidden_fallback": str(final / "stage5_candidate_iql" / "candidate_iql_policy.pt"),
            "required_stage5_invocation": "python -m credit_recourse.rl.pipelines.final_stage5_candidate_iql.pipeline --project-root <ROOT> --train-mode final_refit",
        })
    ckpt = torch.load(ckpt_path, map_location="cpu")
    assert_hashes_match(ckpt, root, context="Stage6 selector loading Stage5 IQL")
    vocab = list(ckpt.get("action_vocabulary") or ckpt.get("candidate_ids", []))
    if vocab != list(space.train_labels):
        raise ValueError(f"Stage6 checkpoint action vocabulary differs from active main_train_labels: {vocab} vs {space.train_labels}")

    features = ckpt["features"]
    stats = ckpt["preprocess_stats"]
    X_np, M_np = transform_with_missing_mask(df, features, stats)
    schema = ckpt.get("stage3_encoder", {}).get("schema", {})
    C_np, categorical_oov_counts = transform_categorical(df, list(schema.get("categorical_columns") or []), schema.get("categorical_vocab") or {})
    X = torch.tensor(X_np, dtype=torch.float32)
    M = torch.tensor(M_np, dtype=torch.bool)
    C = torch.tensor(C_np, dtype=torch.long)
    encoder = load_stage3_encoder_payload(ckpt["stage3_encoder"], strict=True)
    critic_head_arch = str(ckpt.get("critic_head_arch", "linear"))
    actor_head_arch = str(ckpt.get("actor_head_arch", "linear"))
    checkpoint_actions, checkpoint_action_audit = _validated_checkpoint_action_frame(
        ckpt, space, include_stage6_extras=bool(args.include_stage6_extras)
    )
    checkpoint_actions_by_id = checkpoint_actions.set_index("candidate_id")
    candidate_action_vectors = None
    if critic_head_arch in ("cross_attention", "cross_attention_film") or actor_head_arch == "action_conditioned":
        cav_np = checkpoint_actions_by_id.loc[space.train_labels, space.columns].to_numpy(dtype=np.float32)
        candidate_action_vectors = torch.tensor(cav_np, dtype=torch.float32)
    model = DiscreteIQL(
        encoder,
        len(vocab),
        d_model=int(encoder.d_model),
        critic_head_arch=critic_head_arch,
        actor_head_arch=actor_head_arch,
        candidate_action_vectors=candidate_action_vectors,
        n_attn_blocks=int(ckpt.get("cross_attn_blocks") or 2),
        n_attn_heads=int(ckpt.get("cross_attn_heads") or 4),
        attn_dropout=float(ckpt.get("cross_attn_dropout") or 0.1),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        logits = model.logits(X, missing_mask=M, cat=C)
        probs = torch.softmax(logits, dim=1).numpy()
        q_min = model.q_min(X, missing_mask=M, cat=C).numpy()
        pred_idx = probs.argmax(axis=1)
    pred_names = [vocab[i] for i in pred_idx]
    fixed = space.frame(include_stage6_extras=args.include_stage6_extras).set_index("candidate_id")
    required_fixed_ids = fixed.index.astype(str).tolist()
    fixed.loc[required_fixed_ids, space.columns] = checkpoint_actions_by_id.loc[
        required_fixed_ids, space.columns
    ].to_numpy(dtype=float)

    rows = []
    for i, name in enumerate(pred_names):
        row = {"row_id": i, "policy": space.final_rl_label, "candidate_id": name}
        row.update(fixed.loc[name].to_dict())
        rows.append(row)

    c2_ref_path = final / "stage2_candidate_projection" / "phase3_iql_candidate.parquet"
    if not c2_ref_path.exists():
        raise FileNotFoundError({
            "message": "Missing Stage2 Phase3 IQL panel required to build C2 weakest-component comparator",
            "required": str(c2_ref_path),
            "why": "Stage6 metadata and multi-oracle geometry require C2_weakest_component_rule on the primary eval panel",
        })
    c2_ref = read_parquet_required(c2_ref_path)
    c2_ref_year = pd.to_numeric(c2_ref.get("fiscal_year", c2_ref.get("year")), errors="coerce")
    c2_ref = c2_ref[c2_ref_year <= int(temporal_contract.train_transition_year_max)].copy()
    if c2_ref.empty:
        raise ValueError("C2 weakest-component reference panel is empty after temporal train-transition filter")

    c2_actions = c2_row_conditional(df, space, c2_ref, fixed.reset_index())
    if len(c2_actions) != len(df):
        raise ValueError(f"C2 row count mismatch: {len(c2_actions)} actions for {len(df)} eval rows")

    for i, (_, r) in enumerate(c2_actions.reset_index(drop=True).iterrows()):
        row = {"row_id": i, "policy": "C2_weakest_component_rule"}
        row.update(r.to_dict())
        row["candidate_id"] = "C2_weakest_component_rule"
        rows.append(row)

    baseline_labels = ["C0_noop"] + [c for c in space.train_labels if c != "A0_noop"]
    if args.include_stage6_extras:
        baseline_labels += list(space.scenario_candidates.keys()) + list(space.diagnostic_candidates.keys())
    for pol in baseline_labels:
        cand = "A0_noop" if pol == "C0_noop" else pol
        vec = fixed.loc[cand].to_dict()
        for i in range(len(df)):
            row = {"row_id": i, "policy": pol, "candidate_id": cand}
            row.update(vec)
            rows.append(row)

    # Primary phase_eval is state-only by design; C_obs is not emitted here.
    pa = pd.DataFrame(rows)
    _pairing_audit(pa, out)
    pa.to_parquet(out / "policy_actions.parquet", index=False)
    pa.to_csv(out / "policy_actions.csv", index=False, encoding="utf-8-sig")
    prob = pd.DataFrame(probs, columns=[f"prob__{c}" for c in vocab])
    prob.to_parquet(out / "candidate_probabilities.parquet", index=False)
    qdf = pd.DataFrame(q_min, columns=[f"q_min__{c}" for c in vocab])
    qdf.to_parquet(out / "candidate_q_values.parquet", index=False)
    sel = pd.DataFrame({"row_id": range(len(df)), "policy": space.final_rl_label, "candidate_id": pred_names})
    sel.to_parquet(out / "candidate_selection.parquet", index=False)
    sel["candidate_id"].value_counts().rename_axis("candidate_id").reset_index(name="count").to_csv(out / "candidate_distribution_c3_iql.csv", index=False)
    pd.DataFrame({"metric": ["status"], "value": ["pending_multi_oracle_scoring"]}).to_csv(out / "rl_vs_c2_action_geometry.csv", index=False)
    meta = {
        "stage": "final_stage6_candidate_selector_eval",
        "status": "PASS_ACTIONS_EXPORTED_PENDING_MULTI_ORACLE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "final_rl_label": space.final_rl_label,
        "evaluated_policies": ["C0_noop", "v32 main/scenario/diagnostic fixed candidates", "C2_weakest_component_rule row conditional", space.final_rl_label],
        "h9_evaluated": False,
        "c2_projection_label": False,
        "c2_semantics": "frozen_sector_phi_percentile_weakest_component_top1_rule",
        "c_obs_semantics": "not_available_on_primary_state_only_phase_eval; secondary_inner_dev_only",
        **temporal_metadata(temporal_contract, stage="final_stage6_candidate_selector_eval"),
        "candidate_library_version": "v32_controllability_ladder",
        "scenario_and_diagnostic_evaluated": bool(args.include_stage6_extras),
        "candidate_library_hash": ckpt.get("candidate_library_hash"),
        "final_action_contract_hash": ckpt.get("final_action_contract_hash"),
        "stage3_encoder_class": "FinalBlockAwareEncoder",
        "stage3_encoder_loaded_strict": True,
        "serving_missing_mask_used": True,
        "action_vocabulary": vocab,
        "candidate_probabilities": "candidate_probabilities.parquet",
        "candidate_q_values": "candidate_q_values.parquet",
        "q_value_basis": "IQL critic q_min = min(q1, q2), computed before Stage6 oracle scoring",
        "critic_head_arch": critic_head_arch,
        "actor_head_arch": actor_head_arch,
        "cross_attn_blocks": ckpt.get("cross_attn_blocks"),
        "cross_attn_heads": ckpt.get("cross_attn_heads"),
        "cross_attn_dropout": ckpt.get("cross_attn_dropout"),
        **checkpoint_action_audit,
    }
    
    # Secondary inner-dev diagnostic panel: observed action is allowed here because
    # it is not the primary evaluation/serving panel.
    inner_path = final / "stage2_candidate_projection" / "phase3_iql_candidate.parquet"
    if inner_path.exists():
        inner = read_parquet_required(inner_path)
        y = pd.to_numeric(inner.get("fiscal_year", inner.get("year")), errors="coerce")
        inner = inner[y == int(temporal_contract.inner_dev_year)].copy()
        inner_rows=[]
        if len(inner):
            fixed_inner = fixed
            for i, (_, r) in enumerate(inner.reset_index(drop=True).iterrows()):
                row={"row_id":i,"policy":"C_obs","candidate_id":"C_obs_projected_inner_dev","observed_projected_candidate_id":str(r.get("candidate_id",""))}
                for col in space.columns:
                    row[col]=_num(r,col,0.0)
                inner_rows.append(row)
                noop={"row_id":i,"policy":"C0_noop","candidate_id":"A0_noop"}; noop.update(fixed_inner.loc["A0_noop"].to_dict()); inner_rows.append(noop)
            inner_pa=pd.DataFrame(inner_rows)
            inner_pa.to_parquet(out/"stage6_policy_actions_inner_dev.parquet", index=False)
            inner.to_parquet(out/"stage6_inner_dev_state_panel.parquet", index=False)
            meta["secondary_inner_dev_panel"]={"status":"PASS","policy_actions":"stage6_policy_actions_inner_dev.parquet","state_panel":"stage6_inner_dev_state_panel.parquet","rows":int(len(inner)),"c_obs_present":True}
        else:
            meta["secondary_inner_dev_panel"]={"status":"EMPTY_INNER_DEV","rows":0,"c_obs_present":False}
    else:
        meta["secondary_inner_dev_panel"]={"status":"MISSING_PHASE3_IQL_CANDIDATE","rows":0,"c_obs_present":False}

    write_json(out / "metadata.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
