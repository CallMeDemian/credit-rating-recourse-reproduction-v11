from __future__ import annotations
import argparse, json, hashlib, math, time, random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch.utils.data import DataLoader, TensorDataset

from credit_recourse.rl.common.io import final_root, write_json, read_parquet_required
from credit_recourse.rl.common.actions import load_action_space, active_config_hashes
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_metadata
from credit_recourse.rl.final_candidate.encoder import FinalBlockAwareEncoder
from credit_recourse.rl.contracts.avs256_acd_v2 import (
    SCHEMA_VERSION, CONTRACT_VERSION, CONTINUOUS_COLUMNS, ACD_TARGET_COLUMNS,
    CATEGORICAL_COLUMNS, DIRECTION_VOCAB, EXPECTED_BLOCK_COUNTS, FEATURES,
    build_feature_manifest, validate_manifest, feature_block_ids, feature_direction_ids,
)

# Forbidden leakage identifiers are intentionally split into prefix and exact
# checks.  A naive substring check incorrectly flags legitimate AVS256 columns
# such as ``transition__positive_transition_prior_score`` because it contains
# the text ``r_score`` inside ``prior_score``.  The manifest itself is binding,
# so leakage detection must be precise enough not to reject contract features.
FORBIDDEN_PREFIXES = [
    "oracle_", "alpha_score", "beta_score", "gamma_score", "r_score_", "R_score_",
    "policy", "policy_value", "post_intervention", "action__", "action_observed__",
    "reward", "reward_", "phi_t", "phi_t1", "phi_diff", "delta_phi",
    "aux_", "soft_cand_", "next__",
]
FORBIDDEN_EXACT = {"pv", "PV", "r_score", "R_score"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_leakage(c: str) -> bool:
    s = str(c)
    lc = s.lower()
    if s in FORBIDDEN_EXACT or lc in {x.lower() for x in FORBIDDEN_EXACT}:
        return True
    if lc.startswith("next__") or lc.endswith("__next"):
        return True
    for tok in FORBIDDEN_PREFIXES:
        t = tok.lower()
        if t.endswith("_") or t.endswith("__"):
            if lc.startswith(t):
                return True
        elif lc == t or lc.startswith(t + "_"):
            return True
    return False


def load_feature_manifest(stage2_root: Path) -> dict[str, Any]:
    p = stage2_root / "feature_manifest.json"
    if not p.exists():
        raise FileNotFoundError(f"AVS256 manifest required; legacy keyword heuristic disabled: {p}")
    manifest = json.loads(p.read_text(encoding="utf-8"))
    validate_manifest(manifest)
    return manifest


def ensure_manifest_on_disk(stage2_root: Path) -> dict[str, Any]:
    # Stage2 must write this, but writing the canonical object here gives a clearer
    # failure path for older test copies before the Stage2 patch is run.
    p = stage2_root / "feature_manifest.json"
    if p.exists():
        return load_feature_manifest(stage2_root)
    raise FileNotFoundError(f"AVS256 manifest required; legacy keyword heuristic disabled: {p}")


def fit_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, dict[str, float]]:
    stats = {"median": {}, "iqr": {}, "lo": {}, "hi": {}}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        lo = float(s.quantile(0.005)); hi = float(s.quantile(0.995))
        med = float(s.median()); q1 = float(s.quantile(.25)); q3 = float(s.quantile(.75))
        iqr = max(q3 - q1, 1e-6) if math.isfinite(q3 - q1) else 1.0
        stats["median"][c] = med if math.isfinite(med) else 0.0
        stats["iqr"][c] = iqr
        stats["lo"][c] = lo if math.isfinite(lo) else -1e6
        stats["hi"][c] = hi if math.isfinite(hi) else 1e6
    return stats


def _as_float_series(df: pd.DataFrame, c: str, fill_value: float) -> pd.Series:
    if c not in df.columns:
        raise KeyError(f"Missing Stage3 feature column: {c}")
    return pd.to_numeric(df[c], errors="coerce").astype("float64").fillna(float(fill_value))


def transform(df: pd.DataFrame, cols: list[str], stats: dict[str, dict[str, float]]) -> np.ndarray:
    xs = []
    for c in cols:
        med = float(stats["median"][c]); iqr = max(float(stats["iqr"][c]), 1e-6)
        lo = float(stats["lo"][c]); hi = float(stats["hi"][c])
        if not math.isfinite(lo): lo = -1e6
        if not math.isfinite(hi): hi = 1e6
        if lo > hi: lo, hi = hi, lo
        s = _as_float_series(df, c, med).clip(lower=lo, upper=hi)
        xs.append(((s - med) / iqr).clip(-5, 5).to_numpy(dtype=np.float32))
    return np.vstack(xs).T if xs else np.zeros((len(df), 0), dtype=np.float32)


def missing_mask(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.vstack([pd.to_numeric(df[c], errors="coerce").isna().to_numpy(dtype=np.bool_) for c in cols]).T


def categorical_vocab(df: pd.DataFrame, cats: list[str]) -> tuple[dict[str, dict[str, int]], dict[str, int], int]:
    vocabs: dict[str, dict[str, int]] = {}; offsets: dict[str, int] = {}; next_id = 1
    for c in cats:
        vals = sorted(str(x) for x in df[c].dropna().astype(str).unique()) if c in df.columns else []
        offsets[c] = next_id
        vocabs[c] = {v: i + next_id for i, v in enumerate(vals)}
        next_id += len(vals)
    return vocabs, offsets, next_id


def transform_cat(df: pd.DataFrame, cats: list[str], vocabs: dict[str, dict[str, int]]) -> np.ndarray:
    arr = []
    for c in cats:
        vocab = vocabs.get(c, {})
        arr.append(df[c].astype(str).map(vocab).fillna(0).astype(np.int64).to_numpy() if c in df.columns else np.zeros(len(df), dtype=np.int64))
    return np.vstack(arr).T if arr else np.zeros((len(df), 0), dtype=np.int64)


def transform_next_acd(df: pd.DataFrame, acd_cols: list[str], stats: dict[str, dict[str, float]]) -> np.ndarray:
    data = {}
    missing = []
    for c in acd_cols:
        nc = f"next__{c}"
        alt = f"{c}__next"
        if nc in df.columns:
            data[c] = df[nc]
        elif alt in df.columns:
            data[c] = df[alt]
        else:
            missing.append(nc)
    if missing:
        raise ValueError(f"Stage3 ACD requires exactly 118 next-state target columns; missing {len(missing)}: {missing[:20]}")
    return transform(pd.DataFrame(data, index=df.index), acd_cols, stats)


def transform_action(df: pd.DataFrame, action_cols: list[str], bounds: dict[str, tuple[float, float]]) -> np.ndarray:
    arr = []
    for c in action_cols:
        if c not in df.columns:
            raise ValueError(f"Missing ACD action column {c}")
        lo, hi = bounds[c]
        scale = max(abs(float(lo)), abs(float(hi)), 1e-12)
        arr.append((pd.to_numeric(df[c], errors="coerce").fillna(0.0) / scale).clip(-1, 1).to_numpy(dtype=np.float32))
    return np.vstack(arr).T


def nt_xent(z: torch.Tensor, pair_ids: torch.Tensor, temp: float = 0.1) -> torch.Tensor:
    z = torch.nn.functional.normalize(z, dim=1)
    sim = (z @ z.T) / temp
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    same = (pair_ids[:, None] == pair_ids[None, :]) & (~eye)
    if same.sum() == 0:
        return z.new_tensor(0.0)
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(logp[same].mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--val-batch-size", type=int, default=None)
    ap.add_argument("--pretraining-epochs", type=int, default=None)
    ap.add_argument("--masking-ratio", type=float, default=0.15)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--phase-alpha-sweep", action="store_true")
    ap.add_argument("--train-mode", choices=["selection", "final_refit"], default="selection")
    ap.add_argument("--torch-deterministic", action="store_true", help="Enable strict torch/CUDA determinism (RL-DET-001). Default off preserves the frozen-run training path; enabling changes numerics and cannot recover the frozen s2 policy.")
    args = ap.parse_args(argv)
    if args.pretraining_epochs is not None:
        args.epochs = int(args.pretraining_epochs)
    if float(args.learning_rate) <= 0.0:
        raise ValueError({"message": "Stage3 learning_rate must be positive", "learning_rate": float(args.learning_rate)})
    if float(args.weight_decay) < 0.0:
        raise ValueError({"message": "Stage3 weight_decay must be non-negative", "weight_decay": float(args.weight_decay)})
    if args.torch_deterministic:
        from credit_recourse.rl.common.determinism import enable_torch_determinism
        torch_determinism_meta = enable_torch_determinism()
    else:
        from credit_recourse.rl.common.determinism import torch_determinism_disabled_metadata
        torch_determinism_meta = torch_determinism_disabled_metadata()
    set_seed(args.seed)
    t0 = time.time()
    root = Path(args.project_root).resolve()
    final = final_root(root)
    stage2_root = final / "stage2_candidate_projection"
    in_root = stage2_root / "input_splits"
    out = final / "stage3_acd_ssl"
    out.mkdir(parents=True, exist_ok=True)
    cfg_hashes = active_config_hashes(root)
    space = load_action_space(root)

    manifest = ensure_manifest_on_disk(stage2_root)
    features = [f["name"] for f in manifest["features"]]
    acd_targets = [f["name"] for f in manifest["features"] if f["used_for_acd"]]
    cats = CATEGORICAL_COLUMNS
    if features != CONTINUOUS_COLUMNS or acd_targets != ACD_TARGET_COLUMNS:
        raise ValueError("Loaded feature_manifest.json does not match binding AVS256 v2 feature/ACD order")

    phase1_path = in_root / "phase1_pretrain.parquet"
    df = read_parquet_required(phase1_path)
    temporal_contract = load_temporal_contract(root)
    year = pd.to_numeric(df.get("fiscal_year", df.get("year")), errors="coerce")
    max_train_year = int(temporal_contract.inner_train_year_max if args.train_mode == "selection" else temporal_contract.train_transition_year_max)
    before_rows = int(len(df))
    df = df.loc[year <= max_train_year].copy()
    if df.empty:
        raise ValueError(f"Stage3 {args.train_mode} training window is empty: fiscal_year <= {max_train_year}")
    if args.smoke_test:
        df = df.head(512).copy()

    missing_features = [c for c in features if c not in df.columns]
    missing_cats = [c for c in cats if c not in df.columns]
    missing_next = [f"next__{c}" for c in acd_targets if f"next__{c}" not in df.columns and f"{c}__next" not in df.columns]
    if missing_features or missing_cats or missing_next:
        raise ValueError({
            "message": "Stage3 input does not satisfy AVS256_BLOCK_AWARE_ACD_V2",
            "missing_features_count": len(missing_features), "missing_features_sample": missing_features[:30],
            "missing_categorical_columns": missing_cats,
            "missing_acd_next_targets_count": len(missing_next), "missing_acd_next_targets_sample": missing_next[:30],
        })
    selected_leak = [c for c in features + cats if is_leakage(c)]
    if selected_leak:
        raise ValueError(f"Feature manifest contains leakage features: {selected_leak}")

    forbidden_cols = [str(c) for c in df.columns if is_leakage(str(c))]
    leakage_audit = {
        "schema_version": "feature_leakage_audit_avs256_acd_v2",
        "forbidden_prefixes": FORBIDDEN_PREFIXES,
        "forbidden_exact": sorted(FORBIDDEN_EXACT),
        "forbidden_columns_seen_in_input": forbidden_cols,
        "selected_forbidden_feature_hits": selected_leak,
        "n_forbidden_columns_seen": len(forbidden_cols),
        "n_selected_forbidden_feature_hits": len(selected_leak),
    }
    write_json(out / "feature_leakage_audit.json", leakage_audit)

    stats = fit_stats(df, features)
    X = transform(df, features, stats)
    X_missing = missing_mask(df, features)
    Xn_acd = transform_next_acd(df, acd_targets, stats)
    A = transform_action(df, space.columns, space.bounds)
    pair = df["firm_id"].astype(str).factorize()[0] if "firm_id" in df.columns else np.arange(len(df))

    block_ids = feature_block_ids()
    direction_ids = feature_direction_ids()
    cat_vocabs, cat_offsets, n_cat_tokens = categorical_vocab(df, cats)
    C = transform_cat(df, cats, cat_vocabs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinalBlockAwareEncoder(
        n_features=129,
        block_ids=block_ids,
        direction_ids=direction_ids,
        d_model=(16 if args.smoke_test else int(args.hidden_dim)),
        n_heads=(2 if args.smoke_test else 8),
        n_layers=(1 if args.smoke_test else 4),
        dropout=0.1,
        n_actions=len(space.columns),
        n_categorical_tokens=n_cat_tokens,
        n_categorical_fields=2,
        n_acd_targets=118,
        action_emb_dim=64,
    ).to(device)
    learning_rate = float(args.learning_rate)
    weight_decay = float(args.weight_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer_name = "AdamW"
    lr_scheduler = "none"
    warmup_steps = 0
    total_steps_mode = "epoch_based"
    optimizer_steps = 0

    arr = torch.tensor(X, dtype=torch.float32)
    arr_missing = torch.tensor(X_missing, dtype=torch.bool)
    arr_next = torch.tensor(Xn_acd, dtype=torch.float32)
    arr_a = torch.tensor(A, dtype=torch.float32)
    arr_pair = torch.tensor(pair, dtype=torch.long)
    arr_cat = torch.tensor(C, dtype=torch.long)
    n = len(arr)
    idx = np.arange(n); np.random.shuffle(idx)
    val_n = max(1, int(n * .1)); val_idx = idx[:val_n]; tr_idx = idx[val_n:]
    train_batch_size = int(args.batch_size)
    val_batch_size = int(args.val_batch_size or args.batch_size)
    if train_batch_size <= 0 or val_batch_size <= 0:
        raise ValueError({"message": "Stage3 batch sizes must be positive", "batch_size": train_batch_size, "val_batch_size": val_batch_size})
    train = DataLoader(TensorDataset(arr[tr_idx], arr_missing[tr_idx], arr_next[tr_idx], arr_a[tr_idx], arr_pair[tr_idx], arr_cat[tr_idx]), batch_size=train_batch_size, shuffle=True)
    val = DataLoader(TensorDataset(arr[val_idx], arr_missing[val_idx], arr_next[val_idx], arr_a[val_idx], arr_pair[val_idx], arr_cat[val_idx]), batch_size=val_batch_size)

    log = []; best = float("inf"); best_state = None; patience = 5; bad_epochs = 0
    epochs = 2 if args.smoke_test else int(args.epochs)
    for ep in range(1, epochs + 1):
        model.train(); losses=[]; mcm_losses=[]; acd_losses=[]; con_losses=[]
        for xb, mb, xnb, ab, pb, cb in train:
            xb=xb.to(device); mb=mb.to(device); xnb=xnb.to(device); ab=ab.to(device); pb=pb.to(device); cb=cb.to(device)
            mask=(torch.rand_like(xb)<float(args.masking_ratio)) & (~mb)
            _, recon, acd_pred, proj = model.forward_pretrain(xb, ab, missing_mask=mb, mcm_mask=mask, cat=cb)
            mcm = ((recon-xb)[mask]**2).mean() if mask.any() else ((recon-xb)**2).mean()
            acd_loss = ((acd_pred-xnb)**2).mean()
            con = nt_xent(proj, pb, temp=0.1)
            loss = mcm + 0.5*acd_loss + 0.3*con
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); optimizer_steps += 1
            losses.append(float(loss.item())); mcm_losses.append(float(mcm.item())); acd_losses.append(float(acd_loss.item())); con_losses.append(float(con.item()))
        model.eval(); vloss=[]; vmcm=[]; vacd=[]; vcon=[]
        with torch.no_grad():
            for xb, mb, xnb, ab, pb, cb in val:
                xb=xb.to(device); mb=mb.to(device); xnb=xnb.to(device); ab=ab.to(device); pb=pb.to(device); cb=cb.to(device)
                mask=(torch.rand_like(xb)<float(args.masking_ratio)) & (~mb)
                _, recon, acd_pred, proj = model.forward_pretrain(xb, ab, missing_mask=mb, mcm_mask=mask, cat=cb)
                mcm = ((recon-xb)[mask]**2).mean() if mask.any() else ((recon-xb)**2).mean()
                acd_loss = ((acd_pred-xnb)**2).mean()
                con = nt_xent(proj, pb, temp=0.1)
                total = mcm + 0.5*acd_loss + 0.3*con
                vloss.append(float(total.item())); vmcm.append(float(mcm.item())); vacd.append(float(acd_loss.item())); vcon.append(float(con.item()))
        va = float(np.mean(vloss)); tr = float(np.mean(losses))
        log.append({
            "epoch": ep, "train_loss": tr, "val_loss": va,
            "train_L_MCM": float(np.mean(mcm_losses)), "train_L_ACD": float(np.mean(acd_losses)), "train_L_con": float(np.mean(con_losses)),
            "val_L_MCM": float(np.mean(vmcm)), "val_L_ACD": float(np.mean(vacd)), "val_L_con": float(np.mean(vcon)),
            "optimizer": optimizer_name, "learning_rate": float(learning_rate), "weight_decay": float(weight_decay),
            "lr_scheduler": lr_scheduler, "warmup_steps": int(warmup_steps),
            "total_steps_mode": total_steps_mode, "optimizer_steps": int(optimizer_steps),
        })
        if va < best:
            best = va; bad_epochs = 0; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
        if not args.smoke_test and ep >= 10 and bad_epochs >= patience:
            break
    pd.DataFrame(log).to_csv(out / "training_log.csv", index=False)

    flat_stats = {c: {"median": stats["median"][c], "iqr": stats["iqr"][c], "lo": stats["lo"][c], "hi": stats["hi"][c]} for c in features}
    manifest_hash = hashlib.sha256(json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    schema_hash = hashlib.sha256(json.dumps({"schema_version": SCHEMA_VERSION, "features": features, "acd_targets": acd_targets, "categorical": cats, "manifest_hash": manifest_hash}, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]
    blocks = {f["name"]: f["block"] for f in FEATURES}
    dirs = {f["name"]: f["direction"] for f in FEATURES}
    schema = {
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "contract_hash": schema_hash,
        "feature_schema_hash": schema_hash,
        "feature_manifest_hash": manifest_hash,
        "candidate_library_hash": cfg_hashes["candidate_library_hash"],
        "final_action_contract_hash": cfg_hashes["final_action_contract_hash"],
        "encoder_class": "FinalBlockAwareEncoder",
        "encoder_architecture": {
            "d_model": (16 if args.smoke_test else int(args.hidden_dim)),
            "n_heads": (2 if args.smoke_test else 8),
            "n_layers": (1 if args.smoke_test else 4),
            "ff_multiplier": 4,
            "final_non_smoke_contract": {"d_model": int(args.hidden_dim), "n_heads": 8, "n_layers": 4, "ff_multiplier": 4},
        },
        "continuous_columns": features,
        "categorical_columns": cats,
        "categorical_vocab": cat_vocabs,
        "categorical_offsets": cat_offsets,
        "n_categorical_tokens": n_cat_tokens,
        "n_categorical_fields": 2,
        "n_continuous_features": 129,
        "n_acd_targets": 118,
        "acd_target_columns": acd_targets,
        "acd_target_indices_in_feature_list": [features.index(c) for c in acd_targets],
        "feature_blocks": blocks,
        "feature_directions": dirs,
        "feature_block_ids": block_ids,
        "feature_direction_ids": direction_ids,
        "block_realized_counts": EXPECTED_BLOCK_COUNTS,
        "direction_vocab": DIRECTION_VOCAB,
        "categorical_fields": manifest["categorical_fields"],
        "action_columns": space.columns,
        "action_bounds": {k: list(v) for k, v in space.bounds.items()},
        "preprocess_stats": flat_stats,
        "direction_embedding_used": True,
        "categorical_embedding_used": True,
        "true_missing_mask_used": True,
        "ssl_objectives": ["masked_cell_modeling", "action_conditional_forward_dynamics_with_interaction", "contrastive_same_firm"],
        "mcm_weight": 1.0,
        "acd_weight": 0.5,
        "contrastive_weight": 0.3,
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "lr_scheduler": lr_scheduler,
        "warmup_steps": int(warmup_steps),
        "total_steps_mode": total_steps_mode,
        "optimizer_steps": int(optimizer_steps),
        "acd_head_class": "ActionConditionalForwardHead",
        "acd_uses_interaction": True,
        "acd_action_emb_dim": 64,
        "forbidden_feature_hits": selected_leak,
        "strict_downstream_load_required": True,
    }
    model_config = {
        "class": "FinalBlockAwareEncoder",
        "d_model": (16 if args.smoke_test else int(args.hidden_dim)),
        "n_heads": (2 if args.smoke_test else 8),
        "n_layers": (1 if args.smoke_test else 4),
        "ff_multiplier": 4,
        "dropout": 0.1,
        "block_ids": block_ids,
        "direction_ids": direction_ids,
        "n_actions": len(space.columns),
        "action_columns": space.columns,
        "n_categorical_tokens": n_cat_tokens,
        "n_categorical_fields": 2,
        "n_acd_targets": 118,
        "acd_target_columns": acd_targets,
        "action_emb_dim": 64,
        "acd_head_class": "ActionConditionalForwardHead",
        "acd_uses_interaction": True,
    }
    write_json(out / "feature_schema.json", schema)
    write_json(out / "preprocess_stats.json", flat_stats)
    payload = {
        "encoder_class": "FinalBlockAwareEncoder",
        "candidate_library_hash": cfg_hashes["candidate_library_hash"],
        "final_action_contract_hash": cfg_hashes["final_action_contract_hash"],
        "encoder_state_dict": best_state,
        "features": features,
        "schema": schema,
        "preprocess_stats": flat_stats,
        "model_config": model_config,
        "val_loss": best,
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "lr_scheduler": lr_scheduler,
        "warmup_steps": int(warmup_steps),
        "total_steps_mode": total_steps_mode,
        "optimizer_steps": int(optimizer_steps),
        "epochs": int(args.epochs),
        "train_mode": args.train_mode,
        "training_max_year": int(max_train_year),
        "is_backward_compat_alias": False,
    }
    torch.save(payload, out / "ssl_encoder.pt")
    stage3_alias = "stage3_encoder_avs256_innerdev_winner.pt" if args.train_mode == "selection" else "stage3_encoder_avs256_final_refit_fulltrain.pt"
    torch.save(payload, out / stage3_alias)
    if args.train_mode == "selection" and not (out / "stage3_encoder_avs256_final_refit_fulltrain.pt").exists():
        # Backward-compatible local-run alias only. Downstream final_refit stages
        # must reject this alias unless Stage3 is rerun with --train-mode final_refit.
        alias_payload = dict(payload)
        alias_payload["is_backward_compat_alias"] = True
        alias_payload["alias_target"] = "stage3_encoder_avs256_final_refit_fulltrain.pt"
        alias_payload["alias_source"] = stage3_alias
        torch.save(alias_payload, out / "stage3_encoder_avs256_final_refit_fulltrain.pt")
    meta = {
        "stage": "final_stage3_acd_ssl",
        "status": "PASS",
        "created_utc": now(),
        "elapsed_seconds": round(time.time()-t0, 2),
        "smoke_test": bool(args.smoke_test),
        "schema_version": SCHEMA_VERSION,
        "contract_version": CONTRACT_VERSION,
        "feature_schema_hash": schema_hash,
        "feature_manifest_hash": manifest_hash,
        "candidate_library_hash": cfg_hashes["candidate_library_hash"],
        "final_action_contract_hash": cfg_hashes["final_action_contract_hash"],
        "candidate_library_path": cfg_hashes.get("candidate_library_path"),
        "final_action_contract_path": cfg_hashes.get("final_action_contract_path"),
        "n_continuous_features": 129,
        "n_categorical_fields": 2,
        "n_acd_targets": 118,
        "batch_size": int(train_batch_size),
        "val_batch_size": int(val_batch_size),
        "epochs": int(args.epochs),
        "pretraining_epochs": int(args.epochs),
        "optimizer": optimizer_name,
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "lr_scheduler": lr_scheduler,
        "warmup_steps": int(warmup_steps),
        "total_steps_mode": total_steps_mode,
        "optimizer_steps": int(optimizer_steps),
        "n_train_rows": int(len(tr_idx)),
        "n_val_rows": int(len(val_idx)),
        "train_batches_per_epoch": int(len(train)),
        "val_batches_per_epoch": int(len(val)),
        "encoder_architecture": model_config,
        "acd_head_class": "ActionConditionalForwardHead",
        "acd_uses_interaction": True,
        "ssl_objectives": schema["ssl_objectives"],
        "device": str(device),
        "best_val_loss": best,
        "seed": args.seed,
        "train_mode": args.train_mode,
        "training_max_year": int(max_train_year),
        "stage3_backward_compat_final_refit_alias_written": bool(args.train_mode == "selection"),
        "torch_determinism": torch_determinism_meta,
    }
    alpha_grid = []
    for hidden in (256, 384):
        for ep in (30, 50, 80):
            for mask_ratio in (0.15, 0.30, 0.50):
                alpha_grid.append({
                    "hidden_dim": hidden,
                    "pretraining_epochs": ep,
                    "masking_ratio": mask_ratio,
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "selection_metric": "acd_holdout_loss",
                    "status": "SELECTED_RUN" if (hidden == int(args.hidden_dim) and abs(mask_ratio - float(args.masking_ratio)) < 1e-12 and ep == int(args.epochs)) else "REGISTERED_NOT_RUN",
                })
    pd.DataFrame(alpha_grid).to_csv(out / "stage3_sensitivity_phase_alpha.csv", index=False)
    meta["phase_alpha_grid_size"] = len(alpha_grid)
    meta["phase_alpha_grid_artifact"] = "stage3_sensitivity_phase_alpha.csv"
    meta["phase_alpha_selection_metric"] = "acd_holdout_loss"
    meta["phase_alpha_selected_hidden_dim"] = int(args.hidden_dim)
    meta["phase_alpha_selected_pretraining_epochs"] = int(args.epochs)
    meta["phase_alpha_selected_masking_ratio"] = float(args.masking_ratio)
    meta["stage3_encoder_avs256_final_refit_fulltrain"] = "ssl_encoder.pt"
    write_json(out / "metadata.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
