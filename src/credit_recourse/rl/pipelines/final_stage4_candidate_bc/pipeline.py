from __future__ import annotations
import argparse, csv, json, random, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import torch
torch.set_num_threads(1)
from credit_recourse.rl.common.io import final_root, write_json, read_parquet_required
from credit_recourse.rl.common.actions import load_action_space, assert_training_labels_allowed, active_config_hashes, assert_hashes_match
from credit_recourse.rl.common.torch_utils import transform_with_missing_mask, transform_categorical
from credit_recourse.rl.final_candidate.common import CandidatePolicy
from credit_recourse.rl.final_candidate.encoder import load_stage3_encoder_payload
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_metadata

SOFT_ID_COLS=["soft_cand_id_1","soft_cand_id_2","soft_cand_id_3"]
SOFT_P_COLS=["soft_cand_prob_1","soft_cand_prob_2","soft_cand_prob_3"]

# Stage4 family balance is deliberately action-contract based, not oracle based.
# The mapping groups candidates by economic intervention mechanism so that the
# BC warm-start is not dominated by broad historical pseudo-action families.
#
# Important: this is NOT an oracle-derived mapping.  It is defined only from
# candidate action semantics and must match `space.train_labels` exactly.
ACTION_FAMILY_BY_CANDIDATE = {
    "A0_noop": "noop",
    "DL1_deleverage_mild": "deleveraging",
    "DL2_deleverage_moderate": "deleveraging",
    "RF1_short_debt_refinance": "refinance",
    "CX1_capex_discipline": "capex_working_capital",
    "WC1_working_capital_tightening": "capex_working_capital",
    "WC2_supplier_financing": "liquidity_working_capital",
    "OE1_cost_efficiency_mild": "cost_efficiency",
    "OE2_cost_efficiency_moderate": "cost_efficiency",
    "MX1_cost_and_deleverage": "mixed_restructuring",
    "MX2_liquidity_rescue": "liquidity_working_capital",
}
FAMILY_ORDER = [
    "noop",
    "deleveraging",
    "refinance",
    "capex_working_capital",
    "liquidity_working_capital",
    "cost_efficiency",
    "mixed_restructuring",
]

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _assert_stage3_encoder_train_mode(enc_payload: dict, expected_mode: str, enc_path: Path) -> None:
    actual = enc_payload.get('train_mode')
    alias = bool(enc_payload.get('is_backward_compat_alias', False))
    if actual != expected_mode or alias:
        raise ValueError(
            f"Stage4 {expected_mode} requires a real Stage3 encoder trained with train_mode={expected_mode}; "
            f"got train_mode={actual!r}, is_backward_compat_alias={alias} from {enc_path}. "
            "Rerun Stage3 with the matching --train-mode before Stage4."
        )

def build_soft_targets(df: pd.DataFrame, labels: list[str], allow_hard_fallback: bool=False) -> tuple[torch.Tensor,bool]:
    label_to_idx={n:i for i,n in enumerate(labels)}
    y=torch.zeros((len(df),len(labels)), dtype=torch.float32)
    has_soft=all(c in df.columns for c in SOFT_ID_COLS+SOFT_P_COLS)
    if has_soft:
        for i, (_, r) in enumerate(df.iterrows()):
            s=0.0
            for idc,pc in zip(SOFT_ID_COLS,SOFT_P_COLS):
                cid=str(r.get(idc,''))
                if cid and cid != 'nan':
                    if cid not in label_to_idx: raise ValueError(f'Forbidden soft target candidate {cid}')
                    p=float(r.get(pc,0.0) or 0.0)
                    if p<0: raise ValueError('Negative soft target probability')
                    y[i,label_to_idx[cid]] += p; s += p
            if s <= 0:
                cid=str(r['candidate_id'])
                y[i,label_to_idx[cid]]=1.0
            else:
                y[i] /= y[i].sum().clamp_min(1e-12)
        return y, False
    if not allow_hard_fallback:
        raise ValueError('Final Stage4 requires soft candidate target columns. Use --allow-hard-target-fallback only for debug/non-final runs.')
    for i,cid in enumerate(df['candidate_id'].astype(str)):
        y[i,label_to_idx[cid]]=1.0
    return y, True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_selected_recalibrated_candidate_actions(stage2_dir: Path, q: int, labels: list[str], action_columns: list[str], *, include_stage6_extras: bool = True) -> tuple[list[dict], dict]:
    """Load the exact per-quantile candidate action vectors used by Stage2 projection.

    Stage4/5 checkpoints are classifiers, but Stage6 applies simulator actions.
    Therefore the checkpoint payload must embed the selected recalibrated action
    vectors, not the base final_candidate_library.yaml values.
    """
    lib_path = stage2_dir / f"final_candidate_library__P{int(q)}.yaml"
    if not lib_path.exists():
        raise FileNotFoundError({
            "message": "Missing selected recalibrated candidate library required by Stage4",
            "required": str(lib_path),
            "hint": "Rerun Stage2 Candidate Projection after magnitude recalibration patch.",
        })
    payload = yaml.safe_load(lib_path.read_text(encoding="utf-8")) or {}
    fixed = payload.get("fixed_candidates", {}) or {}
    missing = [x for x in labels if x not in fixed]
    if missing:
        raise ValueError({
            "message": "Selected recalibrated candidate library missing train labels",
            "path": str(lib_path),
            "missing": missing,
        })
    sections = [("fixed", fixed)]
    if include_stage6_extras:
        sections += [
            ("scenario", payload.get("scenario_candidates", {}) or {}),
            ("diagnostic", payload.get("diagnostic_candidates", {}) or {}),
        ]
    rows = []
    seen_vectors: dict[tuple, str] = {}
    duplicates = []
    for section, mapping in sections:
        for cid, raw in mapping.items():
            row = {"candidate_id": str(cid), "candidate_section": section}
            for col in action_columns:
                row[col] = float((raw or {}).get(col, 0.0) or 0.0)
            for extra in ["tier", "paper_role", "intent"]:
                if isinstance(raw, dict) and extra in raw:
                    row[extra] = raw.get(extra)
            rows.append(row)
            if section == "fixed":
                vec = tuple(round(float(row[c]), 12) for c in action_columns)
                if vec in seen_vectors:
                    duplicates.append((seen_vectors[vec], str(cid)))
                else:
                    seen_vectors[vec] = str(cid)
    if duplicates:
        raise ValueError({
            "message": "Duplicate fixed candidate action vectors in selected recalibrated library",
            "path": str(lib_path),
            "duplicates": duplicates,
        })
    audit = {
        "candidate_action_values_source": "stage2_recalibrated_candidate_library",
        "selected_magnitude_quantile": int(q),
        "selected_recalibrated_candidate_library_path": str(lib_path),
        "selected_recalibrated_candidate_library_hash": _sha256_file(lib_path),
        "selected_recalibrated_fixed_candidate_count": len(fixed),
        "selected_recalibrated_action_value_count": len(rows),
        "selected_recalibrated_unique_fixed_action_vector_count": len(seen_vectors),
        "selected_recalibrated_duplicate_fixed_action_vector_count": len(duplicates),
    }
    return rows, audit

def _training_device_metadata() -> tuple[torch.device, dict]:
    cuda_available = bool(torch.cuda.is_available())
    device = torch.device('cuda') if cuda_available else torch.device('cpu')
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else ''
    return device, {
        'training_device': str(device),
        'cuda_available': cuda_available,
        'cuda_device_name': cuda_name,
    }


def _weighted_mean_one(weights: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    counts = counts.detach().float().clamp_min(0.0)
    denom = counts.sum().clamp_min(1e-12)
    return (weights.float() * counts).sum() / denom


def _normalize_by_soft_mass(weights: torch.Tensor, counts: torch.Tensor, *, cap: float | None = None) -> torch.Tensor:
    out = weights.float() / _weighted_mean_one(weights.float(), counts).clamp_min(1e-12)
    if cap is not None and float(cap) > 0:
        # Preserve the user-visible cap as a hard maximum.  We deliberately do
        # not renormalize after capping because that could push rare-candidate
        # weights above the cap again.
        out = out.clamp(max=float(cap))
    return out


def _class_balance_audit(y_soft: torch.Tensor, labels: list[str], *, enabled: bool, beta: float, weight_cap: float) -> tuple[torch.Tensor, dict, pd.DataFrame]:
    """Return per-class BC loss weights and an auditable class-balance table.

    The default path returns all-ones weights so legacy Stage4 behaviour is
    unchanged unless --class-balanced is explicitly enabled.  When enabled,
    weights are computed from the effective number of soft-target samples
    (Cui et al., 2019), normalized to soft-mass weighted mean 1, and capped to
    prevent rare-class noise from dominating the BC anchor.
    """
    if not (0.0 <= float(beta) < 1.0):
        raise ValueError(f"class_balance_beta must be in [0, 1); got {beta}")
    if float(weight_cap) <= 0.0:
        raise ValueError(f"class_weight_cap must be positive; got {weight_cap}")

    counts = y_soft.detach().float().sum(dim=0).clamp_min(1.0)
    if enabled:
        beta_t = torch.full_like(counts, float(beta))
        eff_num = (1.0 - torch.pow(beta_t, counts)) / max(1e-12, (1.0 - float(beta)))
        weights = (1.0 / eff_num.clamp_min(1e-12))
        weights = (weights / weights.mean().clamp_min(1e-12)).clamp(max=float(weight_cap))
    else:
        weights = torch.ones_like(counts)

    rows = []
    total = float(counts.sum().item())
    for label, count, weight in zip(labels, counts.cpu().tolist(), weights.cpu().tolist()):
        rows.append({
            'candidate_id': str(label),
            'action_family': ACTION_FAMILY_BY_CANDIDATE.get(str(label), 'unknown'),
            'soft_target_count': float(count),
            'soft_target_share': float(count / total) if total > 0 else 0.0,
            'class_weight': float(weight),
        })
    audit_df = pd.DataFrame(rows)
    audit = {
        'class_balanced_loss': bool(enabled),
        'class_balance_beta': float(beta),
        'class_weight_cap': float(weight_cap),
        'class_weight_formula': 'effective_number_inverse_soft_mass_mean1_capped',
        'class_weights_by_candidate': {str(r['candidate_id']): float(r['class_weight']) for _, r in audit_df.iterrows()},
        'soft_target_counts_by_candidate': {str(r['candidate_id']): float(r['soft_target_count']) for _, r in audit_df.iterrows()},
    }
    return weights, audit, audit_df


def _family_balance_audit(y_soft: torch.Tensor, labels: list[str], *, enabled: bool, power: float, weight_cap: float) -> tuple[torch.Tensor, dict, pd.DataFrame]:
    """Return per-candidate family-balance weights.

    Family balancing operates at the economic action-family level rather than
    at the individual candidate level.  This is intentionally oracle-free: the
    family map comes from the action contract semantics.  The goal is to keep
    the BC warm-start from being dominated by broad historical pseudo-actions
    such as capex/WC and cost-efficiency families.
    """
    if float(power) < 0.0:
        raise ValueError(f"family_balance_power must be non-negative; got {power}")
    if float(weight_cap) <= 0.0:
        raise ValueError(f"family_weight_cap must be positive; got {weight_cap}")
    unknown = [x for x in labels if x not in ACTION_FAMILY_BY_CANDIDATE]
    if unknown:
        raise ValueError(f"Stage4 family-balanced BC has no family mapping for candidates: {unknown}")

    counts = y_soft.detach().float().sum(dim=0).clamp_min(1.0)
    family_counts: dict[str, float] = {fam: 0.0 for fam in FAMILY_ORDER}
    for label, count in zip(labels, counts.cpu().tolist()):
        fam = ACTION_FAMILY_BY_CANDIDATE[str(label)]
        family_counts[fam] = float(family_counts.get(fam, 0.0) + float(count))

    raw = []
    for label in labels:
        fam = ACTION_FAMILY_BY_CANDIDATE[str(label)]
        fc = max(float(family_counts.get(fam, 0.0)), 1.0)
        raw.append(fc ** (-float(power)) if enabled else 1.0)
    weights = torch.tensor(raw, dtype=torch.float32)
    if enabled:
        weights = _normalize_by_soft_mass(weights, counts, cap=float(weight_cap))
    else:
        weights = torch.ones_like(counts)

    rows = []
    total = float(counts.sum().item())
    family_weight_by_family = {}
    for fam in FAMILY_ORDER:
        idxs = [i for i, lab in enumerate(labels) if ACTION_FAMILY_BY_CANDIDATE.get(str(lab)) == fam]
        if idxs:
            family_weight_by_family[fam] = float(np.mean([float(weights[i].item()) for i in idxs]))
        else:
            family_weight_by_family[fam] = 0.0
    for label, count, weight in zip(labels, counts.cpu().tolist(), weights.cpu().tolist()):
        fam = ACTION_FAMILY_BY_CANDIDATE[str(label)]
        rows.append({
            'candidate_id': str(label),
            'action_family': fam,
            'soft_target_count': float(count),
            'soft_target_share': float(count / total) if total > 0 else 0.0,
            'family_soft_target_count': float(family_counts.get(fam, 0.0)),
            'family_soft_target_share': float(family_counts.get(fam, 0.0) / total) if total > 0 else 0.0,
            'family_weight': float(weight),
        })
    audit_df = pd.DataFrame(rows)
    audit = {
        'family_balanced_loss': bool(enabled),
        'family_balance_power': float(power),
        'family_weight_cap': float(weight_cap),
        'family_weight_formula': 'inverse_family_soft_count_power_soft_mass_mean1_capped' if enabled else 'disabled_all_ones',
        'action_family_by_candidate': {str(x): ACTION_FAMILY_BY_CANDIDATE[str(x)] for x in labels},
        'family_soft_counts': {str(k): float(v) for k, v in family_counts.items()},
        'family_weights_by_family': {str(k): float(v) for k, v in family_weight_by_family.items()},
        'family_weights_by_candidate': {str(r['candidate_id']): float(r['family_weight']) for _, r in audit_df.iterrows()},
    }
    return weights, audit, audit_df


def _combine_stage4_loss_weights(
    class_weights: torch.Tensor,
    family_weights: torch.Tensor,
    y_soft: torch.Tensor,
    *,
    combined_weight_cap: float,
) -> torch.Tensor:
    if float(combined_weight_cap) <= 0.0:
        raise ValueError(f"combined_weight_cap must be positive; got {combined_weight_cap}")
    counts = y_soft.detach().float().sum(dim=0).clamp_min(1.0)
    weights = class_weights.detach().float() * family_weights.detach().float()
    weights = _normalize_by_soft_mass(weights, counts, cap=float(combined_weight_cap))
    return weights


def _stage4_label_quality_audit(
    df: pd.DataFrame,
    y_soft: torch.Tensor,
    labels: list[str],
    class_df: pd.DataFrame,
    family_df: pd.DataFrame,
    final_weights: torch.Tensor,
) -> pd.DataFrame:
    """Build an auditable candidate/family target-quality table for Stage4.

    This audit is deliberately best-effort for optional diagnostic columns:
    if projection diagnostics are not present in the input frame, their fields
    are emitted as NaN rather than silently inventing data.  Required target
    support fields are always computed from the actual Stage4 y_soft tensor.
    """
    rows = []
    y_np = y_soft.detach().cpu().numpy().astype(float)
    total_rows = max(1, int(len(df)))
    soft_counts = y_np.sum(axis=0)
    soft_total = float(max(1e-12, soft_counts.sum()))
    sorted_probs = np.sort(y_np, axis=1)
    top1_prob = y_np.max(axis=1)
    top1_minus_top2 = sorted_probs[:, -1] - sorted_probs[:, -2] if y_np.shape[1] >= 2 else np.ones(len(df))
    soft_entropy = -(y_np * np.log(np.clip(y_np, 1e-12, 1.0))).sum(axis=1)

    class_by = class_df.set_index('candidate_id').to_dict(orient='index') if not class_df.empty else {}
    fam_by = family_df.set_index('candidate_id').to_dict(orient='index') if not family_df.empty else {}
    hard = df['candidate_id'].astype(str) if 'candidate_id' in df.columns else pd.Series([''] * len(df))

    optional_numeric_cols = [
        'projection_distance', 'distance_to_candidate', 'candidate_projection_distance',
        'projection_margin', 'top1_minus_top2', 'soft_entropy', 'delta_phi', 'reward_train', 'reward_raw',
    ]
    optional_bool_cols = ['near_tie', 'near_tie_flag', 'out_of_library', 'out_of_library_flag', 'high_distance']

    for i, label in enumerate(labels):
        mask = (hard == str(label)).to_numpy()
        row = {
            'candidate_id': str(label),
            'action_family': ACTION_FAMILY_BY_CANDIDATE.get(str(label), 'unknown'),
            'hard_count': int(mask.sum()),
            'hard_share': float(mask.sum() / total_rows),
            'soft_target_count': float(soft_counts[i]),
            'soft_target_share': float(soft_counts[i] / soft_total),
            'class_weight': float((class_by.get(str(label)) or {}).get('class_weight', np.nan)),
            'family_weight': float((fam_by.get(str(label)) or {}).get('family_weight', np.nan)),
            'final_loss_weight': float(final_weights[i].detach().cpu().item()),
            'final_effective_count': float(soft_counts[i] * final_weights[i].detach().cpu().item()),
        }
        # Soft-target quality by hard top-1 candidate bucket.
        if mask.any():
            row.update({
                'mean_soft_top1_prob': float(np.nanmean(top1_prob[mask])),
                'mean_soft_top1_minus_top2': float(np.nanmean(top1_minus_top2[mask])),
                'mean_soft_entropy': float(np.nanmean(soft_entropy[mask])),
            })
            for col in optional_numeric_cols:
                if col in df.columns:
                    vals = pd.to_numeric(df.loc[mask, col], errors='coerce')
                    row[f'mean_{col}'] = float(vals.mean()) if vals.notna().any() else np.nan
                    row[f'median_{col}'] = float(vals.median()) if vals.notna().any() else np.nan
            for col in optional_bool_cols:
                if col in df.columns:
                    vals = df.loc[mask, col]
                    if vals.dtype == bool:
                        b = vals.astype(float)
                    else:
                        b = pd.to_numeric(vals, errors='coerce')
                    row[f'{col}_rate'] = float(b.mean()) if b.notna().any() else np.nan
        else:
            row.update({'mean_soft_top1_prob': np.nan, 'mean_soft_top1_minus_top2': np.nan, 'mean_soft_entropy': np.nan})
        rows.append(row)
    audit = pd.DataFrame(rows)
    eff_total = float(audit['final_effective_count'].sum()) if 'final_effective_count' in audit.columns else 0.0
    audit['final_effective_share'] = audit['final_effective_count'] / max(eff_total, 1e-12)
    return audit


def train_one(df, X, M, C, y_soft, enc_payload, labels, seed:int, epochs:int, out_dir:Path, cfg_hashes:dict, batch_size:int=512, oov_counts:dict|None=None, encoder_finetune:bool=False, class_balanced:bool=False, class_balance_beta:float=0.999, class_weight_cap:float=5.0, family_balanced:bool=False, family_balance_power:float=0.5, family_weight_cap:float=3.0, combined_weight_cap:float=5.0):
    set_seed(seed)
    device, device_meta = _training_device_metadata()
    print(f"[Stage4] training_device={device_meta['training_device']} cuda_available={device_meta['cuda_available']} cuda_name={device_meta['cuda_device_name'] or 'CPU'}", flush=True)
    Xd=X.to(device); Md=M.to(device); Cd=C.to(device); Yd=y_soft.to(device)
    class_weights_cpu, class_balance_audit, class_balance_df = _class_balance_audit(
        y_soft, labels, enabled=bool(class_balanced), beta=float(class_balance_beta), weight_cap=float(class_weight_cap)
    )
    family_weights_cpu, family_balance_audit, family_balance_df = _family_balance_audit(
        y_soft, labels, enabled=bool(family_balanced), power=float(family_balance_power), weight_cap=float(family_weight_cap)
    )
    loss_weights_cpu = _combine_stage4_loss_weights(
        class_weights_cpu, family_weights_cpu, y_soft, combined_weight_cap=float(combined_weight_cap)
    )
    label_quality_df = _stage4_label_quality_audit(
        df, y_soft, labels, class_balance_df, family_balance_df, loss_weights_cpu
    )
    loss_weight_audit = {
        'combined_weight_cap': float(combined_weight_cap),
        'combined_weight_formula': 'normalize_mean1_cap_after_class_weight_times_family_weight',
        'loss_weights_by_candidate': {
            str(label): float(weight) for label, weight in zip(labels, loss_weights_cpu.cpu().tolist())
        },
        'final_effective_counts_by_candidate': {
            str(r['candidate_id']): float(r['final_effective_count']) for _, r in label_quality_df.iterrows()
        },
        'final_effective_share_by_candidate': {
            str(r['candidate_id']): float(r['final_effective_share']) for _, r in label_quality_df.iterrows()
        },
    }
    loss_weights = loss_weights_cpu.to(device)
    if class_balanced:
        print(
            f"[Stage4] class_balanced_loss=True beta={float(class_balance_beta)} weight_cap={float(class_weight_cap)}",
            flush=True,
        )
    if family_balanced:
        print(
            f"[Stage4] family_balanced_loss=True power={float(family_balance_power)} family_weight_cap={float(family_weight_cap)} combined_weight_cap={float(combined_weight_cap)}",
            flush=True,
        )
    encoder=load_stage3_encoder_payload(enc_payload, strict=True)
    for p in encoder.parameters(): p.requires_grad=bool(encoder_finetune)
    model=CandidatePolicy(encoder, len(labels)).to(device)
    enc_params=[]; head_params=[]
    for name,p in model.named_parameters():
        (enc_params if name.startswith("encoder.") else head_params).append(p)
    groups=[{"params":[p for p in head_params if p.requires_grad],"lr":1e-3}]
    if encoder_finetune:
        groups.append({"params":[p for p in enc_params if p.requires_grad],"lr":1e-4})
    opt=torch.optim.AdamW(groups, weight_decay=1e-4)
    idx=np.arange(len(X)); np.random.shuffle(idx); val_n=max(1,int(len(idx)*0.2)); val_idx=idx[:val_n]; tr_idx=idx[val_n:]
    val_idx_t=torch.as_tensor(val_idx, dtype=torch.long, device=device)
    log=[]; best=float('inf'); best_state=None
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        np.random.shuffle(tr_idx)
        for start in range(0, len(tr_idx), batch_size):
            bi=tr_idx[start:start+batch_size]
            bi_t=torch.as_tensor(bi, dtype=torch.long, device=device)
            opt.zero_grad()
            logits=model(Xd[bi_t], missing_mask=Md[bi_t], cat=Cd[bi_t]); logp=torch.log_softmax(logits, dim=1)
            loss=-(loss_weights.unsqueeze(0)*Yd[bi_t]*logp).sum(dim=1).mean(); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        train_loss=float(np.mean(losses)) if losses else 0.0
        model.eval()
        with torch.no_grad():
            vl=-(loss_weights.unsqueeze(0)*Yd[val_idx_t]*torch.log_softmax(model(Xd[val_idx_t], missing_mask=Md[val_idx_t], cat=Cd[val_idx_t]),dim=1)).sum(dim=1).mean()
        val_loss=float(vl.detach().cpu())
        log.append({'seed':seed,'epoch':epoch,'train_loss':train_loss,'val_loss':val_loss, **device_meta})
        if epoch == 1 or epoch == epochs or epoch % max(1, min(5, epochs)) == 0:
            print(f"[Stage4][seed={seed}][{device_meta['training_device']}] epoch {epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}", flush=True)
        if val_loss < best:
            best=val_loss; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if seed == 0 or not (out_dir/'class_balance_audit.csv').exists():
        class_balance_df.to_csv(out_dir/'class_balance_audit.csv', index=False, encoding='utf-8-sig')
        family_balance_df.to_csv(out_dir/'family_balance_audit.csv', index=False, encoding='utf-8-sig')
        label_quality_df.to_csv(out_dir/'stage4_label_quality_audit.csv', index=False, encoding='utf-8-sig')
    payload={'model_state_dict':best_state,'candidate_ids':labels,'action_vocabulary':labels,'features':enc_payload['features'],'preprocess_stats':enc_payload['preprocess_stats'],'stage3_encoder':enc_payload,'stage3_encoder_class':'FinalBlockAwareEncoder','stage3_feature_schema_hash':enc_payload.get('schema',{}).get('feature_schema_hash') or enc_payload.get('schema',{}).get('contract_hash'),'candidate_library_hash':cfg_hashes['candidate_library_hash'],'final_action_contract_hash':cfg_hashes['final_action_contract_hash'],'stage3_schema_hash':enc_payload.get('schema',{}).get('feature_schema_hash') or enc_payload.get('schema',{}).get('contract_hash'),'stage3_encoder_loaded_strict':True,'loaded_stage3_encoder_path':enc_payload.get('loaded_stage3_encoder_path'),'loaded_stage3_encoder_sha256':enc_payload.get('loaded_stage3_encoder_sha256'),'stage3_encoder_missing_keys':0,'stage3_encoder_unexpected_keys':0,'stage3_lineage_preserved':True,'soft_target_bc':True,'serving_missing_mask_used':True,'categorical_embedding_used':True,'categorical_oov_counts':(oov_counts or {}),'minibatch_training':True,'batch_size':batch_size,'encoder_finetune':bool(encoder_finetune),'encoder_lr_scale':0.1 if encoder_finetune else 0.0,'candidate_action_values': None,'seed':seed,'best_val_loss':best, **class_balance_audit, **family_balance_audit, **loss_weight_audit, **device_meta}
    torch.save(payload,out_dir/f'candidate_bc_policy_seed{seed}.pt')
    return payload, log

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root', required=True); ap.add_argument('--epochs', type=int, default=30); ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--allow-hard-target-fallback', action='store_true'); ap.add_argument('--batch-size', type=int, default=512); ap.add_argument('--magnitude-quantile', type=int, default=50, choices=[50,65,75,85]); ap.add_argument('--encoder-finetune', action='store_true'); ap.add_argument('--phase-gamma-sweep', action='store_true'); ap.add_argument('--train-mode', choices=['selection','final_refit'], default='final_refit'); ap.add_argument('--class-balanced', action='store_true', help='Use effective-number class-balanced weights for Stage4 soft-target BC loss. Default is legacy unweighted BC.'); ap.add_argument('--class-balance-beta', type=float, default=0.999); ap.add_argument('--class-weight-cap', type=float, default=5.0); ap.add_argument('--family-balanced', action='store_true', help='Use action-family-balanced Stage4 BC loss weights in addition to optional class-balanced weights. Oracle-free; families are defined from candidate action semantics.'); ap.add_argument('--family-balance-power', type=float, default=0.5, help='Family balance exponent. 0.5 = inverse-sqrt family balancing.'); ap.add_argument('--family-weight-cap', type=float, default=3.0); ap.add_argument('--combined-weight-cap', type=float, default=5.0); ap.add_argument('--torch-deterministic', action='store_true', help='Enable strict torch/CUDA determinism (RL-DET-001); default off preserves frozen-run training path.')
    args=ap.parse_args(argv)
    if args.torch_deterministic:
        from credit_recourse.rl.common.determinism import enable_torch_determinism
        torch_determinism_meta = enable_torch_determinism()
    else:
        from credit_recourse.rl.common.determinism import torch_determinism_disabled_metadata
        torch_determinism_meta = torch_determinism_disabled_metadata()
    if not (0.0 <= float(args.class_balance_beta) < 1.0):
        raise ValueError(f"--class-balance-beta must be in [0, 1); got {args.class_balance_beta}")
    if float(args.class_weight_cap) <= 0.0:
        raise ValueError(f"--class-weight-cap must be positive; got {args.class_weight_cap}")
    if float(args.family_balance_power) < 0.0:
        raise ValueError(f"--family-balance-power must be non-negative; got {args.family_balance_power}")
    if float(args.family_weight_cap) <= 0.0:
        raise ValueError(f"--family-weight-cap must be positive; got {args.family_weight_cap}")
    if float(args.combined_weight_cap) <= 0.0:
        raise ValueError(f"--combined-weight-cap must be positive; got {args.combined_weight_cap}")
    root=Path(args.project_root).resolve(); cfg_hashes=active_config_hashes(root); final=final_root(root); space=load_action_space(root); temporal_contract=load_temporal_contract(root); s2=final/'stage2_candidate_projection'; s3=final/'stage3_acd_ssl'; out=final/'stage4_candidate_bc'; out.mkdir(parents=True, exist_ok=True)
    selected_candidate_action_values, selected_candidate_action_audit = _load_selected_recalibrated_candidate_actions(s2, args.magnitude_quantile, space.train_labels, space.columns, include_stage6_extras=True)
    phase2_path = s2/f'phase2_bc_candidate__P{args.magnitude_quantile}.parquet'
    if not phase2_path.exists(): phase2_path = s2/'phase2_bc_candidate.parquet'
    df=read_parquet_required(phase2_path); assert_training_labels_allowed(df, space, 'candidate_id')
    year=pd.to_numeric(df.get('fiscal_year', df.get('year')), errors='coerce')
    max_train_year=int(temporal_contract.inner_train_year_max if args.train_mode=='selection' else temporal_contract.train_transition_year_max)
    before_rows=int(len(df)); df=df.loc[year <= max_train_year].copy()
    if df.empty: raise ValueError(f'Stage4 {args.train_mode} training window is empty: fiscal_year <= {max_train_year}')
    enc_name='stage3_encoder_avs256_innerdev_winner.pt' if args.train_mode=='selection' else 'stage3_encoder_avs256_final_refit_fulltrain.pt'
    enc_path=s3/enc_name
    if not enc_path.exists(): enc_path=s3/'ssl_encoder.pt'
    enc_payload=torch.load(enc_path, map_location='cpu'); assert_hashes_match(enc_payload, root, context='Stage4 loading Stage3 encoder'); _assert_stage3_encoder_train_mode(enc_payload, args.train_mode, enc_path); enc_sha=_sha256_file(enc_path); enc_payload=dict(enc_payload); enc_payload['loaded_stage3_encoder_path']=str(enc_path); enc_payload['loaded_stage3_encoder_sha256']=enc_sha; features=enc_payload['features']; stats=enc_payload['preprocess_stats']; schema=enc_payload.get('schema',{}); X_np,M_np=transform_with_missing_mask(df,features,stats); C_np,oov_counts=transform_categorical(df, list(schema.get('categorical_columns') or []), schema.get('categorical_vocab') or {}); X=torch.tensor(X_np, dtype=torch.float32); M=torch.tensor(M_np, dtype=torch.bool); C=torch.tensor(C_np, dtype=torch.long)
    missing_soft=[c for c in SOFT_ID_COLS+SOFT_P_COLS if c not in df.columns]
    if missing_soft:
        raise ValueError(f'Final Stage4 requires soft target projection columns; hard-label fallback is forbidden: {missing_soft}')
    y_soft, hard_fallback=build_soft_targets(df, space.train_labels, False)
    seeds=[int(x) for x in str(args.seeds).split(',') if str(x).strip()!='']
    logs=[]; summaries=[]; best_payload=None; best_seed=None; best_loss=float('inf')
    for seed in seeds:
        payload, log = train_one(df, X, M, C, y_soft, enc_payload, space.train_labels, seed, args.epochs, out, cfg_hashes, args.batch_size, oov_counts, encoder_finetune=args.encoder_finetune, class_balanced=bool(args.class_balanced), class_balance_beta=float(args.class_balance_beta), class_weight_cap=float(args.class_weight_cap), family_balanced=bool(args.family_balanced), family_balance_power=float(args.family_balance_power), family_weight_cap=float(args.family_weight_cap), combined_weight_cap=float(args.combined_weight_cap))
        payload['candidate_action_values'] = selected_candidate_action_values
        payload.update(selected_candidate_action_audit)
        payload['base_candidate_library_hash'] = cfg_hashes['candidate_library_hash']
        payload['train_mode'] = args.train_mode
        payload['training_max_year'] = max_train_year
        logs.extend(log); summaries.append({'seed':seed,'best_val_loss':payload['best_val_loss'],'checkpoint':f'candidate_bc_policy_seed{seed}.pt'})
        if payload['best_val_loss'] < best_loss:
            best_loss=payload['best_val_loss']; best_seed=seed; best_payload=payload
    torch.save(best_payload, out/'candidate_bc_policy.pt')
    suffix = f'P{args.magnitude_quantile}__' + ('lr_scale_0.1' if args.encoder_finetune else 'frozen')
    if args.train_mode == 'selection':
        torch.save(best_payload, out/f'stage4_bc_selection__{suffix}.pt')
        if args.class_balanced:
            torch.save(best_payload, out/f'stage4_bc_selection__{suffix}__class_balanced.pt')
        if args.family_balanced:
            torch.save(best_payload, out/f'stage4_bc_selection__{suffix}__family_balanced.pt')
        if args.class_balanced and args.family_balanced:
            torch.save(best_payload, out/f'stage4_bc_selection__{suffix}__class_family_balanced.pt')
    else:
        torch.save(best_payload, out/f'stage4_bc_final_refit__{suffix}.pt')
        torch.save(best_payload, out/'stage4_bc_final_refit_fulltrain.pt')
        if args.class_balanced:
            torch.save(best_payload, out/f'stage4_bc_final_refit__{suffix}__class_balanced.pt')
        if args.family_balanced:
            torch.save(best_payload, out/f'stage4_bc_final_refit__{suffix}__family_balanced.pt')
        if args.class_balanced and args.family_balanced:
            torch.save(best_payload, out/f'stage4_bc_final_refit__{suffix}__class_family_balanced.pt')
    pd.DataFrame(logs).to_csv(out/'training_log.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'seed_sweep_summary.csv',index=False)
    dist=df['candidate_id'].value_counts().rename_axis('candidate_id').reset_index(name='count'); dist.to_csv(out/'candidate_distribution.csv',index=False,encoding='utf-8-sig')
    class_balance_audit = {k: best_payload.get(k) for k in ['class_balanced_loss','class_balance_beta','class_weight_cap','class_weight_formula','class_weights_by_candidate','soft_target_counts_by_candidate','family_balanced_loss','family_balance_power','family_weight_cap','family_weight_formula','action_family_by_candidate','family_soft_counts','family_weights_by_family','family_weights_by_candidate','combined_weight_cap','combined_weight_formula','loss_weights_by_candidate','final_effective_counts_by_candidate','final_effective_share_by_candidate']}
    metrics={'action_vocabulary':space.train_labels,'best_seed':best_seed,'best_val_loss':best_loss,'n_train':int(len(df)),'seeds':seeds,'soft_target_bc':True,'hard_target_fallback_used':False, **class_balance_audit}; write_json(out/'validation_metrics.json', metrics)
    meta={
        'stage':'final_stage4_candidate_bc','train_mode':args.train_mode,'training_max_year':max_train_year,'rows_before_temporal_filter':before_rows,'rows_after_temporal_filter':int(len(df)),
        'status':'PASS',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'allowed_train_labels':space.train_labels,
        'action_vocabulary':space.train_labels,
        'forbidden_labels_hard_fail':True,
        'soft_target_bc':True,
        'soft_target_columns':SOFT_ID_COLS+SOFT_P_COLS,
        'hard_target_fallback_used':False,
        'torch_determinism': torch_determinism_meta,
        **class_balance_audit,
        'class_balance_audit_file':'class_balance_audit.csv',
        'family_balance_audit_file':'family_balance_audit.csv',
        'stage4_label_quality_audit_file':'stage4_label_quality_audit.csv',
        'final_paper_run_allowed':True,
        'seeds':seeds,
        'selected_main_seed':best_seed,
        'candidate_library_hash':cfg_hashes['candidate_library_hash'],
        'base_candidate_library_hash':cfg_hashes['candidate_library_hash'],
        'selected_recalibrated_candidate_library_hash':selected_candidate_action_audit['selected_recalibrated_candidate_library_hash'],
        'selected_recalibrated_candidate_library_path':selected_candidate_action_audit['selected_recalibrated_candidate_library_path'],
        'candidate_action_values_source':selected_candidate_action_audit['candidate_action_values_source'],
        'selected_recalibrated_unique_fixed_action_vector_count':selected_candidate_action_audit['selected_recalibrated_unique_fixed_action_vector_count'],
        'selected_recalibrated_duplicate_fixed_action_vector_count':selected_candidate_action_audit['selected_recalibrated_duplicate_fixed_action_vector_count'],
        'final_action_contract_hash':cfg_hashes['final_action_contract_hash'],
        'stage3_encoder_class':'FinalBlockAwareEncoder',
        'stage3_feature_schema_hash':enc_payload.get('schema',{}).get('feature_schema_hash') or enc_payload.get('schema',{}).get('contract_hash'),
        'stage3_schema_hash':enc_payload.get('schema',{}).get('feature_schema_hash') or enc_payload.get('schema',{}).get('contract_hash'),
        'stage3_encoder_loaded_strict':True,
        'loaded_stage3_encoder_path':str(enc_path),
        'loaded_stage3_encoder_sha256':enc_sha,
        'stage3_encoder_missing_keys':0,
        'stage3_encoder_unexpected_keys':0,
        'stage3_lineage_preserved':True,
        'serving_missing_mask_used':True,'categorical_embedding_used':True,'categorical_oov_counts':(oov_counts or {}),
        'minibatch_training':True,
        'batch_size':args.batch_size,
        'epochs':int(args.epochs),
        'magnitude_quantile':args.magnitude_quantile,
        'encoder_finetune':bool(args.encoder_finetune),
        'encoder_lr_scale':0.1 if args.encoder_finetune else 0.0,
        'selection_checkpoint':f'stage4_bc_selection__P{args.magnitude_quantile}__' + ('lr_scale_0.1' if args.encoder_finetune else 'frozen') + '.pt',
        'final_refit_checkpoint':f'stage4_bc_final_refit__P{args.magnitude_quantile}__' + ('lr_scale_0.1' if args.encoder_finetune else 'frozen') + '.pt',
        'class_balanced_named_checkpoint': (f"stage4_bc_{args.train_mode}__P{args.magnitude_quantile}__" + ('lr_scale_0.1' if args.encoder_finetune else 'frozen') + '__class_balanced.pt') if args.class_balanced else '',
        'family_balanced_named_checkpoint': (f"stage4_bc_{args.train_mode}__P{args.magnitude_quantile}__" + ('lr_scale_0.1' if args.encoder_finetune else 'frozen') + '__family_balanced.pt') if args.family_balanced else '',
        'class_family_balanced_named_checkpoint': (f"stage4_bc_{args.train_mode}__P{args.magnitude_quantile}__" + ('lr_scale_0.1' if args.encoder_finetune else 'frozen') + '__class_family_balanced.pt') if (args.class_balanced and args.family_balanced) else ''
    }
    
    _device, _device_meta = _training_device_metadata()
    meta.update(_device_meta)

    grid=[]
    for q in [50,65,75,85]:
        for ft in [False, True]:
            grid.append({'magnitude_quantile':q,'encoder_finetune':ft,'encoder_lr_scale':0.1 if ft else 0.0,'checkpoint':f'stage4_bc_selection__P{q}__' + ('lr_scale_0.1.pt' if ft else 'frozen.pt'),'status':'SELECTED_RUN' if (q==args.magnitude_quantile and ft==bool(args.encoder_finetune)) else 'REGISTERED_NOT_RUN'})
    pd.DataFrame(grid).to_csv(out/'stage4_bc_sensitivity_grid.csv', index=False)
    meta['phase_gamma_bc_grid_size']=len(grid)
    meta['phase_gamma_bc_grid_artifact']='stage4_bc_sensitivity_grid.csv'

    write_json(out/'metadata.json', meta); print(json.dumps(meta,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
