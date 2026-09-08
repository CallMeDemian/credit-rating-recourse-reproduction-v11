from __future__ import annotations
import argparse, csv, json, random, hashlib
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from credit_recourse.rl.common.io import final_root, write_json, read_parquet_required
from credit_recourse.rl.common.actions import load_action_space, assert_training_labels_allowed, active_config_hashes, assert_hashes_match
from credit_recourse.rl.common.torch_utils import transform, transform_with_missing_mask, transform_categorical
from credit_recourse.rl.final_candidate.common import DiscreteIQL
from credit_recourse.rl.final_candidate.encoder import load_stage3_encoder_payload
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_train_dev_indices, temporal_refit_indices, temporal_metadata

def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fmt_float_for_id(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _assert_stage4_recalibrated_action_payload(bc: dict, expected_q: int, bc_path: Path) -> None:
    src = bc.get('candidate_action_values_source')
    rows = bc.get('candidate_action_values')
    selected_hash = bc.get('selected_recalibrated_candidate_library_hash')
    selected_q = bc.get('selected_magnitude_quantile')
    if src != 'stage2_recalibrated_candidate_library':
        raise ValueError(f"Stage5 requires Stage4 checkpoint action payload from selected recalibrated Stage2 library; got source={src!r} from {bc_path}")
    if int(selected_q) != int(expected_q):
        raise ValueError(f"Stage5 magnitude quantile mismatch: CLI P{expected_q}, Stage4 checkpoint P{selected_q} from {bc_path}")
    if not selected_hash:
        raise ValueError(f"Stage5 requires selected_recalibrated_candidate_library_hash in Stage4 checkpoint: {bc_path}")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Stage5 requires non-empty candidate_action_values in Stage4 checkpoint: {bc_path}")
    fixed_rows = [r for r in rows if isinstance(r, dict) and str(r.get('candidate_id', '')) and str(r.get('candidate_id')) in set(bc.get('action_vocabulary') or bc.get('candidate_ids') or [])]
    if len(fixed_rows) != len(list(bc.get('action_vocabulary') or bc.get('candidate_ids') or [])):
        raise ValueError(f"Stage5 Stage4 action payload does not cover all train candidates: rows={len(fixed_rows)} checkpoint={bc_path}")

def _assert_stage4_bc_train_mode(bc_payload: dict, expected_mode: str, bc_path: Path) -> None:
    actual = bc_payload.get('train_mode')
    if actual != expected_mode:
        raise ValueError(
            f"Stage5 {expected_mode} requires a Stage4 BC checkpoint trained with train_mode={expected_mode}; "
            f"got train_mode={actual!r} from {bc_path}. "
            "Rerun Stage4 with the matching --train-mode before Stage5."
        )

def _training_device_metadata() -> tuple[torch.device, dict]:
    cuda_available = bool(torch.cuda.is_available())
    device = torch.device('cuda') if cuda_available else torch.device('cpu')
    cuda_name = torch.cuda.get_device_name(0) if cuda_available else ''
    return device, {
        'training_device': str(device),
        'cuda_available': cuda_available,
        'cuda_device_name': cuda_name,
    }




def _resolve_sector_phi_rho(df: pd.DataFrame) -> float:
    if "lambda_phi" not in df.columns:
        raise ValueError("Stage5 requires lambda_phi to record the actual sector-Phi reward weight")
    values = pd.to_numeric(df["lambda_phi"], errors="coerce")
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Stage5 lambda_phi contains missing or non-finite values")
    unique = np.unique(np.round(values.to_numpy(dtype=float), 12))
    if len(unique) != 1:
        raise ValueError(f"Stage5 requires a single frozen lambda_phi value; observed={unique.tolist()}")
    return float(unique[0])

def expectile_loss(diff, tau):
    return torch.mean(torch.where(diff>0, tau*diff.pow(2), (1-tau)*diff.pow(2)))

def evaluate_policy_value_proxy(model, X, M=None, C=None, labels=None, batch_size: int = 128):
    """Report actor-policy and critic-greedy diagnostics without changing IQL semantics.

    Historical ``critic_proxy_mean_q_argmax_pi`` is retained as a deprecated alias
    for mean Q(s, actor_argmax).  The true critic-greedy proxy is exposed
    separately as ``critic_value_greedy_proxy``.

    Cross-attention critics/actors are memory-heavy when evaluated on the full
    validation tensor at once.  Keep the metric definition identical, but stream
    it in mini-batches so Stage5 does not OOM on 8GB GPUs.
    """
    model.eval()
    n = int(X.shape[0])
    if n <= 0:
        raise ValueError("evaluate_policy_value_proxy received an empty evaluation tensor")
    bs = max(1, int(batch_size))
    n_actions = len(labels) if labels is not None else 0
    actor_counts = None
    q_counts = None
    actor_sum = 0.0
    greedy_sum = 0.0
    ent_sum = 0.0
    agreement_sum = 0.0
    total = 0

    with torch.no_grad():
        for start in range(0, n, bs):
            end = min(start + bs, n)
            xb = X[start:end]
            mb = M[start:end] if M is not None else None
            cb = C[start:end] if C is not None else None

            logits = model.logits(xb, missing_mask=mb, cat=cb)
            prob = torch.softmax(logits, dim=-1)
            actor_argmax = prob.argmax(dim=-1)
            qmin = model.q_min(xb, missing_mask=mb, cat=cb)
            q_argmax = qmin.argmax(dim=1)
            actor_vals = qmin.gather(1, actor_argmax[:, None]).squeeze(1)
            greedy_vals = qmin.max(dim=1).values
            ent = (-(prob * torch.log(prob + 1e-12)).sum(dim=-1))

            if n_actions <= 0:
                n_actions = int(qmin.shape[1])
            actor_batch_counts = torch.bincount(actor_argmax.detach().cpu(), minlength=n_actions).numpy().astype(int)
            q_batch_counts = torch.bincount(q_argmax.detach().cpu(), minlength=n_actions).numpy().astype(int)
            if actor_counts is None:
                actor_counts = actor_batch_counts
                q_counts = q_batch_counts
            else:
                actor_counts += actor_batch_counts
                q_counts += q_batch_counts

            bsz = int(end - start)
            actor_sum += float(actor_vals.sum().detach().cpu())
            greedy_sum += float(greedy_vals.sum().detach().cpu())
            ent_sum += float(ent.sum().detach().cpu())
            agreement_sum += float((actor_argmax == q_argmax).float().sum().detach().cpu())
            total += bsz

            del logits, prob, actor_argmax, qmin, q_argmax, actor_vals, greedy_vals, ent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    labs = list(labels) if labels is not None else [str(i) for i in range(n_actions)]
    actor_dist = {labs[i] if i < len(labs) else str(i): int(actor_counts[i]) for i in range(n_actions)}
    q_dist = {labs[i] if i < len(labs) else str(i): int(q_counts[i]) for i in range(n_actions)}
    mx2_idx = next((i for i, lab in enumerate(labs) if str(lab).startswith('MX2')), None)
    mx2_share = float(actor_counts[mx2_idx] / max(int(actor_counts.sum()), 1)) if mx2_idx is not None else 0.0
    actor_policy_q = float(actor_sum / max(total, 1))
    return {
        'actor_policy_q_proxy': actor_policy_q,
        'critic_proxy_mean_q_argmax_pi': actor_policy_q,
        'critic_value_greedy_proxy': float(greedy_sum / max(total, 1)),
        'actor_qargmax_agreement': float(agreement_sum / max(total, 1)),
        'policy_entropy_mean': float(ent_sum / max(total, 1)),
        'action_entropy_mean': float(ent_sum / max(total, 1)),
        'mx2_share': mx2_share,
        'actor_argmax_distribution': actor_dist,
        'actor_action_distribution': actor_dist,
        'q_argmax_distribution': q_dist,
    }

def evaluate_policy_value_proxy_indexed(model, X, M=None, C=None, row_idx=None, labels=None, batch_size: int = 64):
    """Memory-safe validation proxy over row indices.

    This preserves the same diagnostic semantics as ``evaluate_policy_value_proxy``
    while avoiding full validation materialization such as ``Xd[val_idx_t]``.
    Only the current mini-batch slice is gathered on the target device.
    """
    if row_idx is None:
        return evaluate_policy_value_proxy(model, X, M, C, labels=labels, batch_size=batch_size)
    if isinstance(row_idx, torch.Tensor):
        idx_np = row_idx.detach().cpu().numpy().astype(int)
    else:
        idx_np = np.asarray(row_idx, dtype=int)
    if idx_np.size <= 0:
        raise ValueError("evaluate_policy_value_proxy_indexed received an empty row_idx")
    model.eval()
    device = X.device
    bs = max(1, int(batch_size))
    n_actions = len(labels) if labels is not None else 0
    actor_counts = None
    q_counts = None
    actor_sum = 0.0
    greedy_sum = 0.0
    ent_sum = 0.0
    agreement_sum = 0.0
    total = 0
    with torch.no_grad():
        for start in range(0, len(idx_np), bs):
            chunk = idx_np[start:start + bs]
            chunk_t = torch.as_tensor(chunk, dtype=torch.long, device=device)
            xb = X.index_select(0, chunk_t)
            mb = M.index_select(0, chunk_t) if M is not None else None
            cb = C.index_select(0, chunk_t) if C is not None else None
            logits = model.logits(xb, missing_mask=mb, cat=cb)
            prob = torch.softmax(logits, dim=-1)
            actor_argmax = prob.argmax(dim=-1)
            qmin = model.q_min(xb, missing_mask=mb, cat=cb)
            q_argmax = qmin.argmax(dim=1)
            actor_vals = qmin.gather(1, actor_argmax[:, None]).squeeze(1)
            greedy_vals = qmin.max(dim=1).values
            ent = (-(prob * torch.log(prob + 1e-12)).sum(dim=-1))
            if n_actions <= 0:
                n_actions = int(qmin.shape[1])
            actor_batch_counts = torch.bincount(actor_argmax.detach().cpu(), minlength=n_actions).numpy().astype(int)
            q_batch_counts = torch.bincount(q_argmax.detach().cpu(), minlength=n_actions).numpy().astype(int)
            if actor_counts is None:
                actor_counts = actor_batch_counts
                q_counts = q_batch_counts
            else:
                actor_counts += actor_batch_counts
                q_counts += q_batch_counts
            bsz = int(len(chunk))
            actor_sum += float(actor_vals.sum().detach().cpu())
            greedy_sum += float(greedy_vals.sum().detach().cpu())
            ent_sum += float(ent.sum().detach().cpu())
            agreement_sum += float((actor_argmax == q_argmax).float().sum().detach().cpu())
            total += bsz
            del chunk_t, xb, mb, cb, logits, prob, actor_argmax, qmin, q_argmax, actor_vals, greedy_vals, ent
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    labs = list(labels) if labels is not None else [str(i) for i in range(n_actions)]
    actor_dist = {labs[i] if i < len(labs) else str(i): int(actor_counts[i]) for i in range(n_actions)}
    q_dist = {labs[i] if i < len(labs) else str(i): int(q_counts[i]) for i in range(n_actions)}
    mx2_idx = next((i for i, lab in enumerate(labs) if str(lab).startswith('MX2')), None)
    mx2_share = float(actor_counts[mx2_idx] / max(int(actor_counts.sum()), 1)) if mx2_idx is not None else 0.0
    actor_policy_q = float(actor_sum / max(total, 1))
    entropy_mean = float(ent_sum / max(total, 1))
    return {
        'actor_policy_q_proxy': actor_policy_q,
        'critic_proxy_mean_q_argmax_pi': actor_policy_q,
        'critic_value_greedy_proxy': float(greedy_sum / max(total, 1)),
        'actor_qargmax_agreement': float(agreement_sum / max(total, 1)),
        'policy_entropy_mean': entropy_mean,
        'action_entropy_mean': entropy_mean,
        'mx2_share': mx2_share,
        'actor_argmax_distribution': actor_dist,
        'actor_action_distribution': actor_dist,
        'q_argmax_distribution': q_dist,
    }


def _population_entropy_from_distribution(distribution: dict) -> float:
    """Entropy of the population-level argmax action distribution.

    This is intentionally different from ``action_entropy_mean`` /
    ``policy_entropy_mean`` logged by ``evaluate_policy_value_proxy*``.  Those
    fields are per-firm softmax entropies averaged across rows.  Pareto-knee
    checkpoint selection needs the cross-firm distribution of actor argmax
    actions, so it must be computed from ``actor_argmax_distribution``.
    """
    if not isinstance(distribution, dict) or not distribution:
        raise ValueError("Population entropy requires a non-empty action distribution dict")
    counts = np.asarray([float(v) for v in distribution.values()], dtype=np.float64)
    if np.any(~np.isfinite(counts)) or np.any(counts < 0):
        raise ValueError(f"Invalid action distribution counts for entropy: {distribution!r}")
    total = float(counts.sum())
    if total <= 0.0:
        raise ValueError(f"Population entropy requires positive total count: {distribution!r}")
    probs = counts[counts > 0.0] / total
    return float(-(probs * np.log(probs)).sum())


def _snapshot_model_state(model) -> dict:
    """Clone a CPU snapshot so later optimizer steps cannot mutate it in-place."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _is_pareto_dominated(candidate: dict, other: dict) -> bool:
    """Return True when ``other`` weakly improves both objectives and strictly one."""
    return (
        float(other['q']) >= float(candidate['q'])
        and float(other['entropy']) >= float(candidate['entropy'])
        and (float(other['q']) > float(candidate['q']) or float(other['entropy']) > float(candidate['entropy']))
    )


def _pareto_prune_frontier(frontier: list[dict]) -> list[dict]:
    pruned: list[dict] = []
    for i, cand in enumerate(frontier):
        if any(_is_pareto_dominated(cand, other) for j, other in enumerate(frontier) if i != j):
            continue
        pruned.append(cand)
    pruned.sort(key=lambda row: (float(row['q']), float(row['entropy']), int(row['epoch'])))
    return pruned


def _normalise_objective(value: float, lo: float, hi: float) -> float:
    if not np.isfinite(value):
        raise ValueError(f"Cannot normalise non-finite objective value: {value!r}")
    if hi <= lo:
        return 1.0
    return float((value - lo) / (hi - lo))


def _select_pareto_knee(frontier: list[dict]) -> dict:
    """Select a no-knob knee from a max-Q / max-population-entropy frontier.

    The selected checkpoint maximises the maximin distance to the dominated
    corner in normalised (Q, population-entropy) space.  This keeps legacy
    behaviour under degenerate objectives: if entropy is constant, the Q tie
    break selects max-Q; if Q is constant, the entropy tie break selects max
    population diversity.
    """
    if not frontier:
        raise ValueError("Pareto-knee selection requires at least one checkpoint candidate")
    qs = np.asarray([float(row['q']) for row in frontier], dtype=np.float64)
    ents = np.asarray([float(row['entropy']) for row in frontier], dtype=np.float64)
    if np.any(~np.isfinite(qs)) or np.any(~np.isfinite(ents)):
        raise ValueError("Pareto-knee frontier contains non-finite objectives")
    q_lo, q_hi = float(qs.min()), float(qs.max())
    e_lo, e_hi = float(ents.min()), float(ents.max())
    best = None
    for row in frontier:
        q_norm = _normalise_objective(float(row['q']), q_lo, q_hi)
        e_norm = _normalise_objective(float(row['entropy']), e_lo, e_hi)
        knee_score = min(q_norm, e_norm)
        row['selection_score'] = float(knee_score)
        row['q_norm'] = float(q_norm)
        row['entropy_norm'] = float(e_norm)
        tie_key = (float(knee_score), float(q_norm + e_norm), float(row['q']), float(row['entropy']), -int(row['epoch']))
        if best is None or tie_key > best[0]:
            best = (tie_key, row)
    selected = dict(best[1])
    selected['frontier_size'] = int(len(frontier))
    selected['q_min'] = q_lo
    selected['q_max'] = q_hi
    selected['entropy_min'] = e_lo
    selected['entropy_max'] = e_hi
    return selected


def _actor_parameter_list(model) -> list[torch.nn.Parameter]:
    """Return actor-only parameters for both legacy and action-conditioned actors.

    The legacy actor exposes ``model.pi``. The optional action-conditioned
    actor intentionally has no ``pi`` module; its trainable actor parameters are
    the action projection, actor cross-attention blocks, residual action query,
    and scalar output head. Keeping this helper local avoids touching the IQL
    update equation and prevents critic/Q/V/encoder parameters from being
    unfrozen during actor-only q_argmax distillation.
    """
    if getattr(model, "actor_head_arch", "linear") == "linear":
        if not hasattr(model, "pi"):
            raise AttributeError("linear actor_head_arch requires model.pi")
        return list(model.pi.parameters())
    if getattr(model, "actor_head_arch", None) == "action_conditioned":
        params: list[torch.nn.Parameter] = []
        for name in ("actor_action_proj", "actor_blocks", "actor_out"):
            module = getattr(model, name, None)
            if module is None:
                raise AttributeError(f"action_conditioned actor missing {name}")
            params.extend(list(module.parameters()))
        if not hasattr(model, "actor_residual"):
            raise AttributeError("action_conditioned actor missing actor_residual")
        params.append(model.actor_residual)
        return params
    raise ValueError(f"Unsupported actor_head_arch={getattr(model, 'actor_head_arch', None)!r}")

def _actor_only_qargmax_finetune(model, Xd, Md, Cd, train_idx, *, steps: int, batch_size: int, learning_rate: float) -> int:
    """Freeze encoder/Q/V and fine-tune only the actor head toward critic q_argmax."""
    if int(steps) <= 0:
        return 0
    device = Xd.device
    for p in model.parameters():
        p.requires_grad = False
    actor_params = _actor_parameter_list(model)
    for p in actor_params:
        p.requires_grad = True
    opt = torch.optim.AdamW(actor_params, lr=float(learning_rate), weight_decay=0.0)
    tr_idx = np.asarray(train_idx, dtype=int).copy()
    done_steps = 0
    model.train()
    while done_steps < int(steps):
        np.random.shuffle(tr_idx)
        for start in range(0, len(tr_idx), batch_size):
            if done_steps >= int(steps):
                break
            bi = tr_idx[start:start + batch_size]
            if len(bi) == 0:
                continue
            bi_t = torch.as_tensor(bi, dtype=torch.long, device=device)
            with torch.no_grad():
                target = model.q_min(Xd[bi_t], missing_mask=Md[bi_t], cat=Cd[bi_t]).argmax(dim=1)
            logits = model.logits(Xd[bi_t], missing_mask=Md[bi_t], cat=Cd[bi_t])
            loss = torch.nn.functional.cross_entropy(logits, target)
            opt.zero_grad(); loss.backward(); opt.step()
            done_steps += 1
    for p in model.parameters():
        p.requires_grad = True
    return done_steps

def build_next_X(df, features, stats, allow_fallback=False):
    dfn=df.copy(); matched=0
    for f in features:
        for nc in (f'next__{f}', f'{f}__next'):
            if nc in dfn.columns:
                dfn[f]=dfn[nc]; matched+=1; break
    if matched == 0:
        if not allow_fallback:
            raise ValueError('Stage5 final run requires next-state feature columns matching Stage3 schema. Fallback is debug-only.')
        return torch.tensor(transform(df,features,stats), dtype=torch.float32), True, matched
    if matched < max(1,int(0.5*len(features))):
        raise ValueError(f'Insufficient next-state feature coverage for IQL target: matched {matched}/{len(features)}')
    return torch.tensor(transform(dfn,features,stats), dtype=torch.float32), False, matched

def train_one(df, X, M, C, Xn, Mn, Cn, a, r, done, bc, labels, seed:int, gamma:float, tau:float, beta:float, cql_alpha:float, epochs:int, out_dir:Path, batch_size:int=256, train_idx=None, dev_idx=None, setting_id="base", learning_rate:float=3e-4, weight_decay:float=1e-4, actor_distill_mode:str="none", actor_distill_lambda:float=0.0, actor_distill_margin_min:float=0.0, actor_distill_temperature:float=1.0, critic_head_arch:str="linear", actor_head_arch:str="linear", candidate_action_vectors:torch.Tensor|None=None, cross_attn_blocks:int=2, cross_attn_heads:int=4, cross_attn_dropout:float=0.1, selection_metric:str="actor_policy_q", actor_extraction_mode:str="awr", actor_finetune_steps:int=0, sector_phi_rho_used:float|None=None) :
    set_seed(seed)
    device, device_meta = _training_device_metadata()
    print(f"[Stage5] training_device={device_meta['training_device']} cuda_available={device_meta['cuda_available']} cuda_name={device_meta['cuda_device_name'] or 'CPU'}", flush=True)
    Xd=X.to(device); Md=M.to(device); Cd=C.to(device)
    Xnd=Xn.to(device); Mnd=Mn.to(device); Cnd=Cn.to(device)
    ad=a.to(device); rd=r.to(device); doned=done.to(device)
    encoder=load_stage3_encoder_payload(bc['stage3_encoder'], strict=True)
    for p in encoder.parameters(): p.requires_grad=False
    model=DiscreteIQL(encoder, len(labels), d_model=int(encoder.d_model), critic_head_arch=critic_head_arch, actor_head_arch=actor_head_arch, candidate_action_vectors=candidate_action_vectors, n_attn_blocks=cross_attn_blocks, n_attn_heads=cross_attn_heads, attn_dropout=cross_attn_dropout).to(device)
    st=bc.get('model_state_dict',{})
    # Legacy Stage4 BC warm-start is only valid for the legacy linear actor.
    # The optional action-conditioned actor has no ``pi`` module and cannot
    # reuse Stage4's linear head weights; it starts from its own initialized
    # action-conditioned parameters while preserving the same frozen encoder.
    if actor_head_arch == 'linear' and hasattr(model, 'pi') and 'head.weight' in st and tuple(st['head.weight'].shape)==tuple(model.pi.weight.shape):
        model.pi.weight.data.copy_(st['head.weight'].to(device)); model.pi.bias.data.copy_(st['head.bias'].to(device))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(learning_rate), weight_decay=float(weight_decay))
    if train_idx is None or dev_idx is None:
        idx=np.arange(len(X)); np.random.shuffle(idx); val_n=max(1,int(len(idx)*0.2)); val_idx=idx[:val_n]; tr_idx=idx[val_n:]
    else:
        tr_idx=np.asarray(train_idx, dtype=int); val_idx=np.asarray(dev_idx, dtype=int)
    val_idx_t=torch.as_tensor(val_idx, dtype=torch.long, device=device)
    log=[]; best_proxy=-1e18; best_state=None; best_epoch=0; latest_proxy=None; optimizer_steps=0; frontier=[]; selected_selection_report={}
    for epoch in range(1,epochs+1):
        model.train(); q_losses=[]; v_losses=[]; actor_losses=[]; actor_iql_losses=[]; actor_distill_losses=[]; actor_distill_active_rates=[]; cql_losses=[]
        np.random.shuffle(tr_idx)
        for start in range(0, len(tr_idx), batch_size):
            bi=tr_idx[start:start+batch_size]
            bi_t=torch.as_tensor(bi, dtype=torch.long, device=device)
            q1,q2=model.q_values(Xd[bi_t], missing_mask=Md[bi_t], cat=Cd[bi_t]); q1_sa=q1.gather(1,ad[bi_t].view(-1,1)).squeeze(1); q2_sa=q2.gather(1,ad[bi_t].view(-1,1)).squeeze(1)
            with torch.no_grad():
                # CRITICAL FINAL FORMULA: target = r + gamma * (1 - done) * V(s_next)
                target = rd[bi_t] + gamma * (1.0 - doned[bi_t]) * model.v_value(Xnd[bi_t], missing_mask=Mnd[bi_t], cat=Cnd[bi_t])
            q_loss=torch.nn.functional.smooth_l1_loss(q1_sa,target) + torch.nn.functional.smooth_l1_loss(q2_sa,target)
            qmin_sa=torch.minimum(q1_sa, q2_sa)
            v=model.v_value(Xd[bi_t], missing_mask=Md[bi_t], cat=Cd[bi_t]); adv_det=qmin_sa.detach()-v; v_loss=expectile_loss(adv_det,tau)
            logits=model.logits(Xd[bi_t], missing_mask=Md[bi_t], cat=Cd[bi_t]); adv=(qmin_sa.detach()-v.detach()); weights=torch.exp(beta*adv).clamp(max=20.0); actor_iql_loss=torch.mean(weights*torch.nn.functional.cross_entropy(logits,ad[bi_t],reduction='none'))
            actor_distill_loss=torch.tensor(0.0, device=device); actor_distill_active_rate=torch.tensor(0.0, device=device)
            if actor_distill_mode != 'none' and actor_distill_lambda > 0:
                # Critic-to-actor distillation: teach the actor the learned critic
                # q_min ranking on the same state batch. q_min is detached so the
                # distillation term updates the actor head only through logits, not the critic.
                with torch.no_grad():
                    qmin_all=torch.minimum(q1, q2).detach()
                    if qmin_all.shape[1] < 2:
                        eligible=torch.ones(qmin_all.shape[0], dtype=torch.bool, device=device)
                    else:
                        top2=torch.topk(qmin_all, k=2, dim=1).values
                        eligible=(top2[:,0]-top2[:,1]) >= float(actor_distill_margin_min)
                    q_argmax=qmin_all.argmax(dim=1)
                if bool(eligible.any().item()):
                    actor_distill_active_rate=eligible.float().mean()
                    if actor_distill_mode == 'ce':
                        actor_distill_loss=torch.nn.functional.cross_entropy(logits[eligible], q_argmax[eligible])
                    elif actor_distill_mode == 'kl':
                        temp=max(float(actor_distill_temperature), 1e-6)
                        q_soft=torch.softmax(qmin_all[eligible] / temp, dim=1)
                        logp=torch.nn.functional.log_softmax(logits[eligible], dim=1)
                        actor_distill_loss=torch.nn.functional.kl_div(logp, q_soft, reduction='batchmean')
                    else:
                        raise ValueError(f'Unsupported actor_distill_mode={actor_distill_mode!r}')
            actor_loss=actor_iql_loss + float(actor_distill_lambda) * actor_distill_loss
            cql_loss=torch.tensor(0.0, device=device)
            if cql_alpha > 0:
                cql_loss = cql_alpha*((torch.logsumexp(q1,dim=1)-q1_sa).mean() + (torch.logsumexp(q2,dim=1)-q2_sa).mean())
            loss=q_loss+v_loss+actor_loss+cql_loss; opt.zero_grad(); loss.backward(); opt.step(); optimizer_steps += 1
            q_losses.append(float(q_loss.detach().cpu())); v_losses.append(float(v_loss.detach().cpu())); actor_losses.append(float(actor_loss.detach().cpu())); actor_iql_losses.append(float(actor_iql_loss.detach().cpu())); actor_distill_losses.append(float(actor_distill_loss.detach().cpu())); actor_distill_active_rates.append(float(actor_distill_active_rate.detach().cpu())); cql_losses.append(float(cql_loss.detach().cpu()))
        q_loss=float(np.mean(q_losses)) if q_losses else 0.0; v_loss=float(np.mean(v_losses)) if v_losses else 0.0; actor_loss=float(np.mean(actor_losses)) if actor_losses else 0.0; actor_iql_loss=float(np.mean(actor_iql_losses)) if actor_iql_losses else actor_loss; actor_distill_loss=float(np.mean(actor_distill_losses)) if actor_distill_losses else 0.0; actor_distill_active_rate=float(np.mean(actor_distill_active_rates)) if actor_distill_active_rates else 0.0; cql_loss=float(np.mean(cql_losses)) if cql_losses else 0.0
        proxy=evaluate_policy_value_proxy_indexed(model, Xd, Md, Cd, val_idx, labels=labels, batch_size=min(batch_size, 64))
        actor_population_entropy = _population_entropy_from_distribution(proxy['actor_argmax_distribution'])
        if selection_metric == 'critic_value_greedy':
            metric_key = 'critic_value_greedy_proxy'
        else:
            metric_key = 'actor_policy_q_proxy'
        latest_proxy=proxy[metric_key]
        log.append({'seed':seed,'epoch':epoch,'q_loss':q_loss,'v_loss':v_loss,'actor_loss':actor_loss,'actor_iql_loss':actor_iql_loss,'actor_distill_loss':actor_distill_loss,'actor_distill_active_rate':actor_distill_active_rate,'actor_distill_mode':actor_distill_mode,'actor_distill_lambda':float(actor_distill_lambda),'actor_distill_margin_min':float(actor_distill_margin_min),'actor_distill_temperature':float(actor_distill_temperature),'cql_loss':cql_loss,'learning_rate':float(learning_rate),'weight_decay':float(weight_decay),'optimizer':'AdamW','lr_scheduler':'none','selection_metric':selection_metric,'selected_proxy_value':latest_proxy,'actor_policy_q_proxy':proxy['actor_policy_q_proxy'],'critic_proxy_mean_q_argmax_pi':proxy['critic_proxy_mean_q_argmax_pi'],'critic_value_greedy_proxy':proxy['critic_value_greedy_proxy'],'actor_qargmax_agreement':proxy['actor_qargmax_agreement'],'policy_entropy_mean':proxy['policy_entropy_mean'],'action_entropy_mean':proxy['action_entropy_mean'],'actor_argmax_population_entropy':actor_population_entropy,'mx2_share':proxy['mx2_share'],'actor_argmax_distribution_json':json.dumps(proxy['actor_argmax_distribution'], ensure_ascii=False),'q_argmax_distribution_json':json.dumps(proxy['q_argmax_distribution'], ensure_ascii=False), **device_meta})
        if epoch == 1 or epoch == epochs or epoch % max(1, min(5, epochs)) == 0:
            print(f"[Stage5][seed={seed}][{device_meta['training_device']}] epoch {epoch}/{epochs} q={q_loss:.6f} v={v_loss:.6f} actor={actor_loss:.6f} distill={actor_distill_loss:.6f} active={actor_distill_active_rate:.3f} proxy={latest_proxy:.6f}", flush=True)
        if selection_metric == 'q_pareto_knee':
            frontier.append({
                'epoch': int(epoch),
                'q': float(proxy['actor_policy_q_proxy']),
                'entropy': float(actor_population_entropy),
                'proxy': dict(proxy),
                'state': _snapshot_model_state(model),
            })
            frontier = _pareto_prune_frontier(frontier)
        elif latest_proxy > best_proxy:
            best_proxy=latest_proxy; best_epoch=epoch; best_state=_snapshot_model_state(model)
    if selection_metric == 'q_pareto_knee':
        selected_knee = _select_pareto_knee(frontier)
        best_proxy = float(selected_knee['selection_score'])
        best_epoch = int(selected_knee['epoch'])
        best_state = selected_knee['state']
        latest_proxy = best_proxy
        selected_selection_report = {k: v for k, v in selected_knee.items() if k not in ('state', 'proxy')}
    finetune_steps_done = 0
    selected_metrics = {}
    if best_state is not None:
        model.load_state_dict(best_state)
        if actor_extraction_mode == 'distill_only_finetune':
            finetune_steps_done = _actor_only_qargmax_finetune(model, Xd, Md, Cd, tr_idx, steps=int(actor_finetune_steps), batch_size=batch_size, learning_rate=learning_rate)
            best_state = _snapshot_model_state(model)
        selected_metrics = evaluate_policy_value_proxy_indexed(model, Xd, Md, Cd, val_idx, labels=labels, batch_size=min(batch_size, 64))

    payload={'model_state_dict':best_state,'candidate_ids':labels,'action_vocabulary':labels,'features':bc['features'],'preprocess_stats':bc['preprocess_stats'],'stage4_bc_checkpoint':bc,'stage3_encoder':bc['stage3_encoder'],'stage3_encoder_class':'FinalBlockAwareEncoder','stage3_feature_schema_hash':bc.get('stage3_encoder',{}).get('schema',{}).get('feature_schema_hash') or bc.get('stage3_encoder',{}).get('schema',{}).get('contract_hash'),
             'stage3_schema_hash':bc.get('stage3_encoder',{}).get('schema',{}).get('feature_schema_hash') or bc.get('stage3_encoder',{}).get('schema',{}).get('contract_hash'),'candidate_library_hash':bc.get('candidate_library_hash'),'base_candidate_library_hash':bc.get('base_candidate_library_hash') or bc.get('candidate_library_hash'),'selected_recalibrated_candidate_library_hash':bc.get('selected_recalibrated_candidate_library_hash'),'selected_magnitude_quantile':bc.get('selected_magnitude_quantile'),'candidate_action_values_source':bc.get('candidate_action_values_source'),'final_action_contract_hash':bc.get('final_action_contract_hash'),'stage4_checkpoint_hash':bc.get('stage4_checkpoint_hash'),'stage4_bc_hash':bc.get('stage4_bc_hash'),'stage3_encoder_loaded_strict':True,'loaded_stage3_encoder_path':bc.get('loaded_stage3_encoder_path'),'loaded_stage3_encoder_sha256':bc.get('loaded_stage3_encoder_sha256'),'stage3_lineage_preserved':True,'target_formula':'target = r + gamma * (1 - done) * V(s_next)','reward_column':'reward_train','reward_is_prestandardized_by_stage2':True,'sector_phi_columns_required':True,'sector_phi_rho_used':float(sector_phi_rho_used) if sector_phi_rho_used is not None else None,'gamma':gamma,'expectile_tau':tau,'beta':beta,'cql_alpha':cql_alpha,'actor_distill_mode':actor_distill_mode,'actor_distill_lambda':float(actor_distill_lambda),'actor_distill_margin_min':float(actor_distill_margin_min),'actor_distill_temperature':float(actor_distill_temperature),'seed':seed,'checkpoint_selection_rule':f'{selection_metric}_for_selection_then_final_refit_fulltrain_for_stage6','best_epoch_by_critic_proxy':best_epoch,'best_critic_proxy_value':best_proxy,'inner_dev_q_mean':best_proxy,'actor_policy_q_proxy':selected_metrics.get('actor_policy_q_proxy'),'critic_value_greedy_proxy':selected_metrics.get('critic_value_greedy_proxy'),'actor_qargmax_agreement':selected_metrics.get('actor_qargmax_agreement'),'mx2_share':selected_metrics.get('mx2_share'),'actor_argmax_distribution':selected_metrics.get('actor_argmax_distribution'),'q_argmax_distribution':selected_metrics.get('q_argmax_distribution'),'actor_argmax_population_entropy':_population_entropy_from_distribution(selected_metrics['actor_argmax_distribution']) if selected_metrics.get('actor_argmax_distribution') else None,'selection_report':selected_selection_report,'selection_metric':selection_metric,'actor_extraction_mode':actor_extraction_mode,'actor_finetune_steps':int(actor_finetune_steps),'actor_finetune_steps_done':int(finetune_steps_done),'setting_id':setting_id,'latest_epoch':epochs,'latest_critic_proxy_value':latest_proxy,'d_model':int(load_stage3_encoder_payload(bc['stage3_encoder'], strict=True).d_model),'serving_missing_mask_used':True,'categorical_embedding_used':True,'categorical_oov_counts':bc.get('categorical_oov_counts',{}),'minibatch_training':True,'batch_size':batch_size,'optimizer':'AdamW','learning_rate':float(learning_rate),'weight_decay':float(weight_decay),'lr_scheduler':'none','warmup_steps':0,'total_steps_mode':'epoch_based','optimizer_steps':int(optimizer_steps),'critic_head_arch':critic_head_arch,'actor_head_arch':actor_head_arch,'cross_attn_blocks':int(cross_attn_blocks) if (critic_head_arch in ('cross_attention', 'cross_attention_film') or actor_head_arch == 'action_conditioned') else None,'cross_attn_heads':int(cross_attn_heads) if (critic_head_arch in ('cross_attention', 'cross_attention_film') or actor_head_arch == 'action_conditioned') else None,'cross_attn_dropout':float(cross_attn_dropout) if (critic_head_arch in ('cross_attention', 'cross_attention_film') or actor_head_arch == 'action_conditioned') else None, **device_meta}
    ckpt=out_dir/f'candidate_iql_policy__{setting_id}__seed{seed}.pt'; torch.save(payload, ckpt)
    return payload, log

def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root', required=True); ap.add_argument('--train-mode', choices=['selection','final_refit'], default='final_refit'); ap.add_argument('--epochs', type=int, default=60); ap.add_argument('--gamma', type=float, default=0.5); ap.add_argument('--expectile-tau', type=float, default=0.7); ap.add_argument('--beta', type=float, default=3.0); ap.add_argument('--cql-alpha', type=float, default=0.0); ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--allow-current-as-next-fallback', action='store_true'); ap.add_argument('--batch-size', type=int, default=256); ap.add_argument('--magnitude-quantile', type=int, default=50, choices=[50,65,75,85]); ap.add_argument('--encoder-finetune', action='store_true'); ap.add_argument('--phase-gamma-sweep', action='store_true'); ap.add_argument('--actor-distill-mode', choices=['none','ce','kl'], default='none'); ap.add_argument('--actor-distill-lambda', type=float, default=0.0); ap.add_argument('--actor-distill-margin-min', type=float, default=0.0); ap.add_argument('--actor-distill-temperature', type=float, default=1.0); ap.add_argument('--learning-rate', type=float, default=3e-4); ap.add_argument('--weight-decay', type=float, default=1e-4); ap.add_argument('--critic-head-arch', choices=['linear','cross_attention','cross_attention_film'], default='linear'); ap.add_argument('--actor-head-arch', choices=['linear','action_conditioned'], default='linear'); ap.add_argument('--cross-attn-blocks', type=int, default=2); ap.add_argument('--cross-attn-heads', type=int, default=4); ap.add_argument('--cross-attn-dropout', type=float, default=0.1); ap.add_argument('--transition-source', choices=['observed','counterfactual'], default='observed'); ap.add_argument('--selection-metric', choices=['actor_policy_q','critic_value_greedy','q_pareto_knee'], default='actor_policy_q'); ap.add_argument('--actor-extraction-mode', choices=['awr','distill_only_finetune'], default='awr'); ap.add_argument('--actor-finetune-steps', type=int, default=200); ap.add_argument('--torch-deterministic', action='store_true', help='Enable strict torch/CUDA determinism (RL-DET-001); default off preserves frozen-run training path.')
    args=ap.parse_args(argv)
    if args.torch_deterministic:
        from credit_recourse.rl.common.determinism import enable_torch_determinism
        torch_determinism_meta = enable_torch_determinism()
    else:
        from credit_recourse.rl.common.determinism import torch_determinism_disabled_metadata
        torch_determinism_meta = torch_determinism_disabled_metadata()
    root=Path(args.project_root).resolve(); temporal_contract=load_temporal_contract(root); cfg_hashes=active_config_hashes(root); final=final_root(root); space=load_action_space(root);
    s2=final/'stage2_candidate_projection'; s3=final/'stage3_acd_ssl'; s4=final/'stage4_candidate_bc'; out=final/'stage5_candidate_iql'; out.mkdir(parents=True, exist_ok=True)
    phase3_base = 'phase3_iql_counterfactual_candidate' if args.transition_source == 'counterfactual' else 'phase3_iql_candidate'
    phase3_path=s2/f'{phase3_base}__P{args.magnitude_quantile}.parquet'
    if not phase3_path.exists():
        phase3_path=s2/f'{phase3_base}.parquet'
    if args.transition_source == 'counterfactual' and not phase3_path.exists():
        raise FileNotFoundError(f'Stage5 --transition-source counterfactual requested but missing {phase3_path}')
    df=read_parquet_required(phase3_path); assert_training_labels_allowed(df, space, 'candidate_id')
    year=pd.to_numeric(df.get('fiscal_year', df.get('year')), errors='coerce')
    # RL-S5-008: selection mode must keep the 2023 inner-dev rows in the
    # in-memory Stage5 frame. The actual train/dev split is still produced by
    # temporal_train_dev_indices(): train <= inner_train_year_max and
    # dev == inner_dev_year. Filtering selection to <= inner_train_year_max
    # before that split makes inner-dev empty and breaks the tau/beta sweep.
    max_train_year=int(temporal_contract.train_transition_year_max)
    before_rows=int(len(df)); df=df.loc[year <= max_train_year].copy()
    if df.empty: raise ValueError(f'Stage5 {args.train_mode} training/refit window is empty: fiscal_year <= {max_train_year}')
    required_reward=['reward_train','reward_raw','reward_aux_phi','phi_t','phi_tplusH','delta_phi','lambda_phi']
    miss=[c for c in required_reward if c not in df.columns]
    if miss: raise ValueError(f'Stage5 requires sector-phi reward columns: {miss}')
    sector_phi_rho_used = _resolve_sector_phi_rho(df)
    suffix=('lr_scale_0.1.pt' if args.encoder_finetune else 'frozen.pt')
    if args.train_mode == 'selection':
        bc_path = s4 / f"stage4_bc_selection__P{args.magnitude_quantile}__{suffix}"
    else:
        bc_path = s4 / f"stage4_bc_final_refit__P{args.magnitude_quantile}__{suffix}"
        if not bc_path.exists(): bc_path=s4/'stage4_bc_final_refit_fulltrain.pt'
    if not bc_path.exists():
        bc_path=s4/'candidate_bc_policy.pt'
    bc=torch.load(bc_path, map_location='cpu')
    assert_hashes_match(bc, root, context='Stage5 loading Stage4 BC')
    _assert_stage4_bc_train_mode(bc, args.train_mode, bc_path)
    if list(bc.get('action_vocabulary') or bc.get('candidate_ids', [])) != list(space.train_labels):
        raise ValueError(f"Stage5 BC action vocabulary differs from active main_train_labels: {bc.get('action_vocabulary') or bc.get('candidate_ids')} vs {space.train_labels}")
    _assert_stage4_recalibrated_action_payload(bc, args.magnitude_quantile, bc_path)
    candidate_action_vectors = None
    if args.critic_head_arch in ('cross_attention', 'cross_attention_film') or args.actor_head_arch == 'action_conditioned':
        cav_rows = bc.get('candidate_action_values') or []
        rows_by_id = {str(r.get('candidate_id')): r for r in cav_rows if isinstance(r, dict) and r.get('candidate_id') is not None}
        missing_cav = [lab for lab in space.train_labels if lab not in rows_by_id]
        if missing_cav:
            raise ValueError(f"Stage5 action-conditioned/cross_attention heads require candidate_action_values for all train labels; missing={missing_cav}")
        missing_cols = {lab: [col for col in space.columns if col not in rows_by_id[lab]] for lab in space.train_labels}
        missing_cols = {lab: cols for lab, cols in missing_cols.items() if cols}
        if missing_cols:
            raise ValueError(
                "Stage5 action-conditioned/cross_attention heads require complete 10D candidate_action_values; "
                f"missing columns by label={missing_cols}"
            )
        cav_np = np.array([[float(rows_by_id[lab].get(col, 0.0) or 0.0) for col in space.columns] for lab in space.train_labels], dtype=np.float32)
        candidate_action_vectors = torch.tensor(cav_np, dtype=torch.float32)
    bc['stage4_checkpoint_hash']=hashlib.sha256(bc_path.read_bytes()).hexdigest(); bc['stage4_bc_hash']=bc['stage4_checkpoint_hash'];
    if not bc.get('loaded_stage3_encoder_sha256'):
        enc_name='stage3_encoder_avs256_innerdev_winner.pt' if args.train_mode=='selection' else 'stage3_encoder_avs256_final_refit_fulltrain.pt'
        enc_path=s3/enc_name
        if not enc_path.exists(): enc_path=s3/'ssl_encoder.pt'
        if enc_path.exists():
            bc['loaded_stage3_encoder_path']=str(enc_path)
            bc['loaded_stage3_encoder_sha256']=_sha256_file(enc_path)
    if not bc.get('loaded_stage3_encoder_sha256'):
        raise ValueError(f"Stage5 requires loaded_stage3_encoder_sha256 in Stage4 checkpoint or a resolvable Stage3 encoder file; got neither for {bc_path}")
    features=bc['features']; stats=bc['preprocess_stats']; schema=bc.get('stage3_encoder',{}).get('schema',{}); X_np,M_np=transform_with_missing_mask(df,features,stats); C_np,oov_counts=transform_categorical(df, list(schema.get('categorical_columns') or []), schema.get('categorical_vocab') or {}); X=torch.tensor(X_np, dtype=torch.float32); M=torch.tensor(M_np, dtype=torch.bool); C=torch.tensor(C_np, dtype=torch.long)
    Xn,fallback_used,next_feature_count=build_next_X(df, features, stats, args.allow_current_as_next_fallback); Mn=torch.zeros_like(M, dtype=torch.bool); Cn=C
    label_to_idx={n:i for i,n in enumerate(space.train_labels)}; a=torch.tensor([label_to_idx[str(x)] for x in df['candidate_id']], dtype=torch.long)
    # Stage2 already standardizes reward_total_raw into reward_train/reward using
    # phase2+phase3 statistics. Stage5 must consume that prestandardized signal
    # directly; re-standardizing only phase3 would violate the v32 reward contract.
    r_raw=pd.to_numeric(df['reward_train'],errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    r_mean=float(pd.to_numeric(df.get('reward_mean_train', pd.Series([0.0]*len(df))), errors='coerce').dropna().iloc[0]) if 'reward_mean_train' in df.columns and pd.to_numeric(df['reward_mean_train'], errors='coerce').notna().any() else 0.0
    r_std=float(pd.to_numeric(df.get('reward_std_train', pd.Series([1.0]*len(df))), errors='coerce').dropna().iloc[0]) if 'reward_std_train' in df.columns and pd.to_numeric(df['reward_std_train'], errors='coerce').notna().any() else 1.0
    r=torch.tensor(r_raw, dtype=torch.float32)
    done=torch.tensor(pd.to_numeric(df.get('done',0),errors='coerce').fillna(0.0).to_numpy() if 'done' in df.columns else [0.0]*len(df), dtype=torch.float32)
    if args.train_mode == 'selection':
        train_idx, dev_idx, temporal_split_meta = temporal_train_dev_indices(df, temporal_contract, stage='final_stage5_candidate_iql')
    else:
        refit_idx, temporal_split_meta = temporal_refit_indices(df, temporal_contract, stage='final_stage5_candidate_iql')
        # Final-refit intentionally trains on the full <= train_transition_year_max
        # transition window. Reuse the refit rows as the proxy/evaluation slice
        # only for best-epoch bookkeeping; Stage6 remains the sole 2024 evaluation.
        train_idx = refit_idx
        dev_idx = refit_idx
    seeds=[int(x) for x in str(args.seeds).split(',') if str(x).strip()!='']
    setting_id=f'P{args.magnitude_quantile}__tau{args.expectile_tau:g}__beta{args.beta:g}__' + ('ft' if args.encoder_finetune else 'frozen')
    if args.actor_distill_mode != 'none' and args.actor_distill_lambda > 0:
        setting_id += f'__distill_{args.actor_distill_mode}_l{args.actor_distill_lambda:g}_m{args.actor_distill_margin_min:g}_t{args.actor_distill_temperature:g}'
    setting_id += f'__lr{_fmt_float_for_id(args.learning_rate)}_wd{_fmt_float_for_id(args.weight_decay)}'
    if args.critic_head_arch != 'linear':
        setting_id += f'__head_{args.critic_head_arch}_b{args.cross_attn_blocks}_h{args.cross_attn_heads}'
    if args.actor_head_arch != 'linear':
        setting_id += f'__actorhead_{args.actor_head_arch}_b{args.cross_attn_blocks}_h{args.cross_attn_heads}'
    if args.transition_source != 'observed':
        setting_id += f'__src_{args.transition_source}'
    if args.selection_metric != 'actor_policy_q':
        setting_id += f'__sel_{args.selection_metric}'
    if args.actor_extraction_mode != 'awr':
        setting_id += f'__actor_{args.actor_extraction_mode}_s{args.actor_finetune_steps}'
    logs=[]; summaries=[]; best_payload=None; best_seed=None; best_proxy=-1e18
    for seed in seeds:
        payload, log = train_one(df,X,M,C,Xn,Mn,Cn,a,r,done,bc,space.train_labels,seed,args.gamma,args.expectile_tau,args.beta,args.cql_alpha,args.epochs,out,args.batch_size,train_idx=train_idx,dev_idx=dev_idx,setting_id=setting_id,learning_rate=args.learning_rate,weight_decay=args.weight_decay,actor_distill_mode=args.actor_distill_mode,actor_distill_lambda=args.actor_distill_lambda,actor_distill_margin_min=args.actor_distill_margin_min,actor_distill_temperature=args.actor_distill_temperature,critic_head_arch=args.critic_head_arch,actor_head_arch=args.actor_head_arch,candidate_action_vectors=candidate_action_vectors,cross_attn_blocks=args.cross_attn_blocks,cross_attn_heads=args.cross_attn_heads,cross_attn_dropout=args.cross_attn_dropout,selection_metric=args.selection_metric,actor_extraction_mode=args.actor_extraction_mode,actor_finetune_steps=args.actor_finetune_steps,sector_phi_rho_used=sector_phi_rho_used)
        logs.extend(log); summaries.append({'seed':seed,'best_epoch_by_critic_proxy':payload['best_epoch_by_critic_proxy'],'best_critic_proxy_value':payload['best_critic_proxy_value'],'actor_policy_q_proxy':payload.get('actor_policy_q_proxy'),'critic_value_greedy_proxy':payload.get('critic_value_greedy_proxy'),'actor_qargmax_agreement':payload.get('actor_qargmax_agreement'),'mx2_share':payload.get('mx2_share'),'actor_argmax_population_entropy':payload.get('actor_argmax_population_entropy'),'selection_report_json':json.dumps(payload.get('selection_report') or {}, ensure_ascii=False),'selection_metric':args.selection_metric,'learning_rate':float(args.learning_rate),'weight_decay':float(args.weight_decay),'checkpoint':f'candidate_iql_policy__{setting_id}__seed{seed}.pt'})
        if payload['best_critic_proxy_value'] > best_proxy:
            best_proxy=payload['best_critic_proxy_value']; best_seed=seed; best_payload=payload
    best_payload['train_mode']=args.train_mode
    best_payload['training_max_year']=max_train_year
    best_payload['candidate_action_values']=bc.get('candidate_action_values')
    if args.train_mode == 'selection':
        torch.save(best_payload,out/'stage5_candidate_iql_innerdev_winner.pt')
    else:
        torch.save(best_payload,out/'stage5_candidate_iql_final_refit_fulltrain.pt')
    torch.save(best_payload,out/'candidate_iql_policy.pt')
    torch.save(best_payload,out/f'candidate_iql_policy__{setting_id}.pt')
    pd.DataFrame(logs).to_csv(out/'training_log.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'seed_sweep_summary.csv',index=False)
    write_json(out/'validation_metrics.json', {'n_train':int(len(df)),'seeds':seeds,'best_seed':best_seed,'best_critic_proxy_value':best_proxy,'selected_actor_argmax_population_entropy':best_payload.get('actor_argmax_population_entropy'),'selection_report':best_payload.get('selection_report') or {},'selection_metric':args.selection_metric,'transition_source':args.transition_source,'actor_extraction_mode':args.actor_extraction_mode,'optimizer':'AdamW','learning_rate':float(args.learning_rate),'weight_decay':float(args.weight_decay),'lr_scheduler':'none'})
    write_json(out/'checkpoint_selection_report.json', {
        'train_mode': args.train_mode,
        'selected_seed': best_seed,
        'best_critic_proxy_value': best_proxy,
        'selected_actor_argmax_population_entropy': best_payload.get('actor_argmax_population_entropy'),
        'selection_report': best_payload.get('selection_report') or {},
        'optimizer': 'AdamW',
        'learning_rate': float(args.learning_rate),
        'weight_decay': float(args.weight_decay),
        'lr_scheduler': 'none',
        'selection_checkpoint_rule': f'{args.selection_metric} over Phase gamma grid setting/seed',
        'final_eval_checkpoint_rule': 'Stage6 must load stage5_candidate_iql_final_refit_fulltrain.pt; candidate_iql_policy.pt is compatibility alias only',
        'main_checkpoint': 'stage5_candidate_iql_final_refit_fulltrain.pt' if args.train_mode == 'final_refit' else 'stage5_candidate_iql_innerdev_winner.pt',
        'legacy_alias': 'candidate_iql_policy.pt',
    })
    pd.DataFrame({'metric':['current_as_next_fallback_used','next_feature_count','n_train'],'value':[fallback_used,next_feature_count,int(len(df))]}).to_csv(out/'q_value_summary.csv',index=False)
    cfg_candidate_hash = cfg_hashes['candidate_library_hash']
    cfg_action_hash = cfg_hashes['final_action_contract_hash']
    meta={
        'stage':'final_stage5_candidate_iql','train_mode':args.train_mode,'training_max_year':max_train_year,'rows_before_temporal_filter':before_rows,'rows_after_temporal_filter':int(len(df)),'temporal_fit_rows':int(len(train_idx)),'temporal_proxy_rows':int(len(dev_idx)),
        'status':'PASS_WITH_SMOKE_FALLBACK' if fallback_used else 'PASS',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'setting_id':setting_id,
        'magnitude_quantile':args.magnitude_quantile,
        'epochs':int(args.epochs),
        'encoder_finetune':bool(args.encoder_finetune),
        **temporal_metadata(temporal_contract, stage='final_stage5_candidate_iql'),
        'gamma':args.gamma,
        'expectile_tau':args.expectile_tau,
        'beta':args.beta,
        'cql_alpha':args.cql_alpha,
        'optimizer':'AdamW',
        'learning_rate':float(args.learning_rate),
        'weight_decay':float(args.weight_decay),
        'lr_scheduler':'none',
        'warmup_steps':0,
        'total_steps_mode':'epoch_based',
        'optimizer_steps':int(best_payload.get('optimizer_steps', 0)),
        'actor_distill_mode':args.actor_distill_mode,
        'actor_distill_lambda':float(args.actor_distill_lambda),
        'actor_distill_margin_min':float(args.actor_distill_margin_min),
        'actor_distill_temperature':float(args.actor_distill_temperature),
        'actor_distillation_enabled':bool(args.actor_distill_mode != 'none' and args.actor_distill_lambda > 0),
        'transition_source':args.transition_source,'transition_source_path':str(phase3_path.relative_to(root)),'transition_source_sha256':_sha256_file(phase3_path),'selection_metric':args.selection_metric,'actor_extraction_mode':args.actor_extraction_mode,'actor_finetune_steps':int(args.actor_finetune_steps),'critic_head_arch':args.critic_head_arch,'actor_head_arch':args.actor_head_arch,
        'cross_attn_blocks':args.cross_attn_blocks if (args.critic_head_arch in ('cross_attention', 'cross_attention_film') or args.actor_head_arch == 'action_conditioned') else None,
        'cross_attn_heads':args.cross_attn_heads if (args.critic_head_arch in ('cross_attention', 'cross_attention_film') or args.actor_head_arch == 'action_conditioned') else None,
        'cross_attn_dropout':args.cross_attn_dropout if (args.critic_head_arch in ('cross_attention', 'cross_attention_film') or args.actor_head_arch == 'action_conditioned') else None,
        'actor_distillation_target':'critic_q_min_argmax_or_soft_qmin_distribution',
        'actor_distillation_uses_stage6_oracle_scores':False,
        'seeds':seeds,
        'target_formula':'target = r + gamma * (1 - done) * V(s_next)',
        'forbidden_target_formula':'target = r + gamma * V(s)',
        'checkpoint_selection_rule':f'{args.selection_metric}_for_selection_then_final_refit_fulltrain_for_stage6',
        'selected_main_seed':best_seed,
        'best_critic_proxy_value':best_proxy,
        'selected_actor_argmax_population_entropy':best_payload.get('actor_argmax_population_entropy'),
        'selection_report':best_payload.get('selection_report') or {},
        'reward_column':'reward_train',
        'reward_standardization_stats':{'mean':r_mean,'std':r_std,'source':'Stage2_phase3_iql_inner_train_only'},
        'reward_restandardized_in_stage5':False,
        'action_vocabulary':space.train_labels,
        # Active config hashes are recorded explicitly. Stage4 embedded hashes are
        # recorded under separate keys so duplicate dict keys cannot silently
        # overwrite lineage metadata.
        'candidate_library_hash':cfg_candidate_hash,
        'final_action_contract_hash':cfg_action_hash,
        'stage4_embedded_candidate_library_hash':bc.get('candidate_library_hash'),
        'stage4_embedded_base_candidate_library_hash':bc.get('base_candidate_library_hash') or bc.get('candidate_library_hash'),
        'selected_recalibrated_candidate_library_hash':bc.get('selected_recalibrated_candidate_library_hash'),
        'selected_magnitude_quantile':bc.get('selected_magnitude_quantile'),
        'candidate_action_values_source':bc.get('candidate_action_values_source'),
        'stage4_embedded_final_action_contract_hash':bc.get('final_action_contract_hash'),
        'stage4_checkpoint_hash':bc.get('stage4_checkpoint_hash'),
        'stage4_bc_hash':bc.get('stage4_bc_hash'),
        'sector_phi_columns_required':True,'sector_phi_rho_used':sector_phi_rho_used,
        'stage3_encoder_class':'FinalBlockAwareEncoder',
        'stage3_feature_schema_hash':bc.get('stage3_encoder',{}).get('schema',{}).get('feature_schema_hash') or bc.get('stage3_encoder',{}).get('schema',{}).get('contract_hash'),
             'stage3_schema_hash':bc.get('stage3_encoder',{}).get('schema',{}).get('feature_schema_hash') or bc.get('stage3_encoder',{}).get('schema',{}).get('contract_hash'),
        'stage3_encoder_loaded_strict':True,
        'loaded_stage3_encoder_path':bc.get('loaded_stage3_encoder_path'),
        'loaded_stage3_encoder_sha256':bc.get('loaded_stage3_encoder_sha256'),
        'stage3_lineage_preserved':True,
        'current_as_next_fallback_used':fallback_used,
        'final_paper_run_allowed':not fallback_used,
        'serving_missing_mask_used':True,'categorical_embedding_used':True,'categorical_oov_counts':bc.get('categorical_oov_counts',{}),
        'minibatch_training':True,
        'batch_size':args.batch_size,
        'torch_determinism': torch_determinism_meta,
        'fallback_warning':'current-as-next fallback is smoke/debug only and must not be used for final paper tables' if fallback_used else ''
    }
    _device, _device_meta = _training_device_metadata()
    meta.update(_device_meta)
    
    gamma_rows=[]
    for q in [50,65,75,85]:
        for tau in [0.5,0.7,0.8,0.9]:
            for beta_v in [1.0,3.0,10.0]:
                for ft in [False, True]:
                    gamma_rows.append({'magnitude_quantile':q,'expectile_tau':tau,'beta':beta_v,'encoder_finetune':ft,'learning_rate':float(args.learning_rate),'weight_decay':float(args.weight_decay),'selection_metric':args.selection_metric,'status':'SELECTED_RUN' if (q==args.magnitude_quantile and abs(tau-args.expectile_tau)<1e-12 and abs(beta_v-args.beta)<1e-12 and ft==bool(args.encoder_finetune)) else 'REGISTERED_NOT_RUN'})
    pd.DataFrame(gamma_rows).to_csv(out/'stage5_sensitivity_phase_gamma.csv', index=False)
    meta['phase_gamma_grid_size']=len(gamma_rows)
    meta['phase_gamma_grid_artifact']='stage5_sensitivity_phase_gamma.csv'
    meta['inner_dev_q_mean']=best_proxy
    meta['actor_policy_q_proxy']=best_payload.get('actor_policy_q_proxy')
    meta['critic_value_greedy_proxy']=best_payload.get('critic_value_greedy_proxy')
    meta['actor_qargmax_agreement']=best_payload.get('actor_qargmax_agreement')
    meta['mx2_share']=best_payload.get('mx2_share')
    meta['actor_argmax_distribution']=best_payload.get('actor_argmax_distribution')
    meta['q_argmax_distribution']=best_payload.get('q_argmax_distribution')

    write_json(out/'metadata.json', meta); print(json.dumps(meta,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
