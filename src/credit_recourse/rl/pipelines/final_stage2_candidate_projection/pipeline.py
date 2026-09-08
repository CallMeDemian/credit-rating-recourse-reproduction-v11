from __future__ import annotations
import argparse, json, shutil, hashlib
import yaml
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from credit_recourse.rl.common.io import final_root, write_json, read_parquet_required
from credit_recourse.rl.common.actions import (
    ActionSpace,
    load_action_space,
    validate_action_columns,
    project_actions_to_candidates,
    assert_training_labels_allowed,
    resolve_weighted_l1_weights,
    projection_distance_description,
)
from credit_recourse.rl.contracts.stage2_avs256_enrichment import write_feature_manifest
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_metadata

PHI_COMPONENTS=["derived__roa_proxy","derived__operating_margin","derived__cogs_to_revenue","derived__sga_to_revenue","derived__financial_cost_to_revenue","derived__debt_to_assets"]
LOWER_GOOD={"derived__cogs_to_revenue","derived__sga_to_revenue","derived__financial_cost_to_revenue","derived__debt_to_assets"}
REQ_REWARD=["reward_raw_notch","reward_raw","phi_t","phi_tplusH","delta_phi","delta_phi_clipped","lambda_phi","reward_aux_phi","reward_total_raw","reward_mean_train","reward_std_train","reward_train","reward_original","reward"]

MERTON_AUX_COMPONENTS = ["sim__total_assets", "sim__short_term_debt", "sim__long_term_debt", "sim__bonds"]
FCFF_AUX_COMPONENTS = ["sim__total_assets", "sim__operating_cf", "sim__capex"]
LIQUIDITY_AUX_COMPONENTS = ["sim__cash", "sim__short_term_investments", "sim__total_assets"]
AUX_REWARD_COLUMNS = [
    "merton_default_point_t", "merton_default_point_tplusH", "merton_badness_t", "merton_badness_tplusH",
    "delta_merton_badness", "delta_merton_badness_scaled", "lambda_merton", "reward_aux_merton",
    "fcff_capacity_t", "fcff_capacity_tplusH", "delta_fcff_capacity", "delta_fcff_capacity_scaled",
    "lambda_fcff", "reward_aux_fcff",
    "liquid_capacity_t", "liquid_capacity_tplusH", "delta_liquid_capacity",
    "delta_liquid_capacity_scaled", "lambda_liquidity", "reward_aux_liquidity",
]
KOREAN_PHI_ALIASES={
 "derived__roa_proxy":["총자본순이익률(IFRS)","ROA","roa_proxy"],
 "derived__operating_margin":["매출액정상영업이익률(IFRS)","영업이익률","operating_margin"],
 "derived__cogs_to_revenue":["매출원가 대 매출액비율(IFRS)","cogs_to_revenue"],
 "derived__sga_to_revenue":["영업비용 대 영업수익비율(IFRS)","sga_to_revenue"],
 "derived__financial_cost_to_revenue":["금융비용부담률(IFRS)","financial_cost_to_revenue"],
 "derived__debt_to_assets":["타인자본구성비율(IFRS)","부채비율(IFRS)","debt_to_assets"],
}
def now(): return datetime.now(timezone.utc).isoformat()
def materialize_phi_aliases(df:pd.DataFrame, *, next_state:bool=False)->pd.DataFrame:
    out=df.copy(); prefix='next__' if next_state else ''; suffix='__next' if next_state else ''
    for canonical, aliases in KOREAN_PHI_ALIASES.items():
        target=prefix+canonical if next_state else canonical
        if target in out.columns: continue
        found=None
        for a in aliases:
            for cand in (prefix+a, a+suffix):
                if cand in out.columns: found=cand; break
            if found: break
        if found is not None: out[target]=pd.to_numeric(out[found], errors='coerce')
    return out
def sector_col(df):
    for c in ['sector_7','industry_class','sector','industry']:
        if c in df.columns: return c
    return None
def _vals(s): return pd.to_numeric(s,errors='coerce').dropna().to_numpy(dtype=float)
def build_frozen_phi_cdf(train_df:pd.DataFrame, out_dir:Path)->dict:
    train_df=materialize_phi_aliases(train_df,next_state=False); missing=[c for c in PHI_COMPONENTS if c not in train_df.columns]
    if missing: raise ValueError(f'Cannot build frozen sector-phi CDF; missing components: {missing}')
    sec=sector_col(train_df); rows=[]; ref={}
    for comp in PHI_COMPONENTS:
        ref[comp]={}; global_vals=np.sort(_vals(train_df[comp]))
        if len(global_vals)==0: raise ValueError(f'No non-null values for sector-phi component {comp}')
        ref[comp]['__GLOBAL__']=global_vals.tolist()
        group_iter=train_df.groupby(sec, dropna=False) if sec else [('__GLOBAL__', train_df)]
        for key,g in group_iter:
            vals=np.sort(_vals(g[comp])); used_fallback=False
            if len(vals)<20: vals=global_vals; used_fallback=True
            ref[comp][str(key)]=vals.tolist(); q=np.quantile(vals,[.01,.05,.10,.25,.50,.75,.90,.95,.99])
            rows.append({'component':comp,'sector':str(key),'n':int(len(vals)),'used_global_fallback':bool(used_fallback),'p01':q[0],'p05':q[1],'p10':q[2],'p25':q[3],'p50':q[4],'p75':q[5],'p90':q[6],'p95':q[7],'p99':q[8],'lower_good':comp in LOWER_GOOD})
    out_dir.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_parquet(out_dir/'sector_phi_breakpoints.parquet', index=False)
    meta={'schema_version':'sector_phi_frozen_cdf_v28','created_utc':now(),'phi_cdf_source':'rated_phase3_iql_inner_train_only','eval_distribution_used_for_cdf':False,'cdf_frozen':True,'components':PHI_COMPONENTS,'lower_good_components':sorted(LOWER_GOOD),'sector_column':sec,'n_reference_rows':int(len(train_df)),'reference_phases':['phase3_iql'],'excluded_from_cdf':['phase_eval']}
    (out_dir/'sector_phi_cdf_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    return {'reference':ref,'sector_column':sec,'metadata':meta}
def _pct(values, ref_vals):
    arr=pd.to_numeric(values,errors='coerce').to_numpy(dtype=float); ref_vals=np.asarray(ref_vals,dtype=float); ref_vals=ref_vals[np.isfinite(ref_vals)]
    if len(ref_vals)==0: return pd.Series(np.full(len(values),.5),index=values.index)
    pct=np.searchsorted(ref_vals, arr, side='right')/max(len(ref_vals),1); pct[~np.isfinite(arr)]=np.nan
    return pd.Series(pct,index=values.index)
def compute_phi_with_frozen_cdf(df:pd.DataFrame, frozen:dict)->pd.Series:
    miss=[c for c in PHI_COMPONENTS if c not in df.columns]
    if miss: raise ValueError(f'Cannot compute sector-phi with frozen CDF; missing components: {miss}')
    sec=frozen.get('sector_column'); qcols=[]
    for comp in PHI_COMPONENTS:
        comp_ref=frozen['reference'][comp]; pct=pd.Series(index=df.index,dtype=float)
        if sec and sec in df.columns:
            for key,idx in df.groupby(sec,dropna=False).groups.items(): pct.loc[idx]=_pct(df.loc[idx,comp], np.asarray(comp_ref.get(str(key)) or comp_ref['__GLOBAL__'],dtype=float))
        else: pct=_pct(df[comp], np.asarray(comp_ref['__GLOBAL__'],dtype=float))
        qcols.append((1-pct if comp in LOWER_GOOD else pct).fillna(pct.median()).fillna(.5))
    q=pd.concat(qcols,axis=1); return .4*q.iloc[:,[0,1]].mean(axis=1)+.3*q.iloc[:,[2,3,4]].mean(axis=1)+.3*q.iloc[:,5]
def _required_next_columns(cols: list[str]) -> list[str]:
    return [f"next__{c}" for c in cols]


def _validate_no_r_code_aux_columns(df: pd.DataFrame, columns: list[str], context: str) -> None:
    bad = [c for c in columns if str(c).startswith("R") and len(str(c)) == 4 and str(c)[1:].isdigit()]
    bad += [c for c in columns if str(c).startswith("next__R") and str(c)[6:].isdigit()]
    if bad:
        raise ValueError(f"{context}: R-code/oracle scorecard columns are forbidden in Merton/FCFF/liquidity auxiliary reward path: {bad}")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required non-oracle simulator columns for Merton/FCFF/liquidity auxiliary reward: {missing}")


def _numeric_checked(df: pd.DataFrame, col: str, context: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"{context}: missing required column {col}")
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        raise ValueError(f"{context}: duplicate column label {col!r} would make auxiliary reward ambiguous")
    x = pd.to_numeric(obj, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if x.isna().any():
        raise ValueError(f"{context}: non-finite values in required auxiliary reward column {col}: n={int(x.isna().sum())}")
    return x.astype(float)


def _numeric_checked_any(df: pd.DataFrame, cols: list[str], context: str) -> pd.Series:
    """Return the first finite numeric Series among equivalent source columns.

    Used for default-vs-reward_only simulator substrate aliases, e.g.
    reward_only__sim__operating_cf should be preferred when present while
    retaining the legacy sim__operating_cf fallback. This helper is intentionally
    strict: it only falls back when a candidate column is absent; if a present
    column is duplicated or contains non-finite values, fail fast rather than
    silently changing reward semantics.
    """
    missing: list[str] = []
    for col in cols:
        if col not in df.columns:
            missing.append(col)
            continue
        return _numeric_checked(df, col, context)
    raise ValueError(f"{context}: missing all equivalent required columns {cols}; absent={missing}")


def _compute_aux_reward_raw_deltas(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Compute no-oracle Merton/KMV and OCF-Capex auxiliary raw deltas.

    This helper intentionally consumes only simulator/accounting primitive columns
    already present in the Stage2 AVS256 state contract. It does not resolve R###
    scorecard aliases and does not import or call any reference oracle backend.
    """
    required = (
        list(MERTON_AUX_COMPONENTS) + _required_next_columns(MERTON_AUX_COMPONENTS)
        + list(FCFF_AUX_COMPONENTS) + _required_next_columns(FCFF_AUX_COMPONENTS)
        + list(LIQUIDITY_AUX_COMPONENTS) + _required_next_columns(LIQUIDITY_AUX_COMPONENTS)
    )
    _validate_no_r_code_aux_columns(df, required, context)
    out = df.copy()
    assets_t = _numeric_checked(out, "sim__total_assets", context)
    assets_n = _numeric_checked(out, "next__sim__total_assets", context)
    bad_assets = (assets_t <= 0) | (assets_n <= 0)
    if bad_assets.any():
        raise ValueError(f"{context}: total assets must be positive for Merton/FCFF/liquidity auxiliary reward: n={int(bad_assets.sum())}")
    short_t = _numeric_checked(out, "sim__short_term_debt", context)
    long_t = _numeric_checked(out, "sim__long_term_debt", context)
    bonds_t = _numeric_checked(out, "sim__bonds", context)
    short_n = _numeric_checked(out, "next__sim__short_term_debt", context)
    long_n = _numeric_checked(out, "next__sim__long_term_debt", context)
    bonds_n = _numeric_checked(out, "next__sim__bonds", context)
    debt_components = [short_t, long_t, bonds_t, short_n, long_n, bonds_n]
    neg_debt = np.zeros(len(out), dtype=bool)
    for x in debt_components:
        neg_debt |= (x < -1e-9).to_numpy(dtype=bool)
    if bool(neg_debt.any()):
        raise ValueError(f"{context}: debt components must be non-negative for Merton default-point proxy: n={int(neg_debt.sum())}")
    default_t = short_t + 0.5 * long_t + bonds_t
    default_n = short_n + 0.5 * long_n + bonds_n
    out["merton_default_point_t"] = default_t
    out["merton_default_point_tplusH"] = default_n
    out["merton_badness_t"] = default_t / assets_t
    out["merton_badness_tplusH"] = default_n / assets_n
    out["delta_merton_badness"] = out["merton_badness_t"] - out["merton_badness_tplusH"]

    ocf_t = _numeric_checked_any(out, ["reward_only__sim__operating_cf", "sim__operating_cf"], context)
    capex_t = _numeric_checked_any(out, ["reward_only__sim__capex", "sim__capex"], context)
    ocf_n = _numeric_checked_any(out, ["next__reward_only__sim__operating_cf", "next__sim__operating_cf"], context)
    capex_n = _numeric_checked_any(out, ["next__reward_only__sim__capex", "next__sim__capex"], context)
    out["fcff_capacity_t"] = (ocf_t - capex_t) / assets_t
    out["fcff_capacity_tplusH"] = (ocf_n - capex_n) / assets_n
    out["delta_fcff_capacity"] = out["fcff_capacity_tplusH"] - out["fcff_capacity_t"]

    cash_t = _numeric_checked(out, "sim__cash", context)
    sti_t = _numeric_checked(out, "sim__short_term_investments", context)
    cash_n = _numeric_checked(out, "next__sim__cash", context)
    sti_n = _numeric_checked(out, "next__sim__short_term_investments", context)
    liquid_components = [cash_t, sti_t, cash_n, sti_n]
    neg_liquid = np.zeros(len(out), dtype=bool)
    for x in liquid_components:
        neg_liquid |= (x < -1e-9).to_numpy(dtype=bool)
    if bool(neg_liquid.any()):
        raise ValueError(f"{context}: liquid asset components must be non-negative for liquidity-capacity auxiliary reward: n={int(neg_liquid.sum())}")
    out["liquid_capacity_t"] = (cash_t + sti_t) / assets_t
    out["liquid_capacity_tplusH"] = (cash_n + sti_n) / assets_n
    out["delta_liquid_capacity"] = out["liquid_capacity_tplusH"] - out["liquid_capacity_t"]
    return out


def _robust_p95_abs(s: pd.Series, *, label: str) -> float:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().abs()
    if x.empty:
        raise ValueError(f"Cannot compute auxiliary reward robust scale for {label}: no finite values")
    scale = float(x.quantile(0.95))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError(f"Cannot compute auxiliary reward robust scale for {label}: p95_abs={scale}")
    return scale


def compute_aux_reward_stats(train: pd.DataFrame, temporal_contract, *, merton_lambda: float, fcff_lambda: float, liquidity_lambda: float = 0.0) -> dict:
    y = pd.to_numeric(train.get('fiscal_year', train.get('year')), errors='coerce')
    inner = train[y <= int(temporal_contract.inner_train_year_max)].copy()
    if inner.empty:
        raise ValueError(f'No phase3_iql inner-train rows for Merton/FCFF/liquidity auxiliary reward scaling <= {temporal_contract.inner_train_year_max}')
    lam_m = float(merton_lambda)
    lam_f = float(fcff_lambda)
    lam_l = float(liquidity_lambda)
    base = {
        'lambda_merton': lam_m,
        'lambda_fcff': lam_f,
        'lambda_liquidity': lam_l,
        'merton_aux_enabled': bool(lam_m != 0.0),
        'fcff_aux_enabled': bool(lam_f != 0.0),
        'liquidity_aux_enabled': bool(lam_l != 0.0),
        'merton_aux_clip': [-1.0, 1.0],
        'fcff_aux_clip': [-0.5, 0.5],
        'liquidity_aux_clip': [-0.5, 0.5],
        'aux_reward_stats_source': 'phase3_iql_inner_train_only',
        'aux_reward_inner_train_rows': int(len(inner)),
        'aux_reward_theory': 'Merton/KMV structural credit risk with OCF-minus-Capex cash-flow capacity proxy and liquid-asset capacity proxy',
        'merton_default_point_proxy': 'sim__short_term_debt + 0.5 * sim__long_term_debt + sim__bonds',
        'merton_asset_proxy': 'sim__total_assets',
        'fcff_capacity_proxy': '(sim__operating_cf - sim__capex) / sim__total_assets',
        'liquidity_capacity_proxy': '(sim__cash + sim__short_term_investments) / sim__total_assets',
        'reference_oracle_scores_used': False,
        'reference_oracle_variables_used': False,
        'r_code_fallback_allowed': False,
    }
    if lam_m == 0.0 and lam_f == 0.0 and lam_l == 0.0:
        return {
            **base,
            'merton_aux_scale_p95_abs': 1.0,
            'fcff_aux_scale_p95_abs': 1.0,
            'liquidity_aux_scale_p95_abs': 1.0,
        }
    aux = _compute_aux_reward_raw_deltas(inner, context='phase3_iql_inner_train_aux_stats')
    return {
        **base,
        'merton_aux_scale_p95_abs': _robust_p95_abs(aux['delta_merton_badness'], label='delta_merton_badness'),
        'fcff_aux_scale_p95_abs': _robust_p95_abs(aux['delta_fcff_capacity'], label='delta_fcff_capacity'),
        'liquidity_aux_scale_p95_abs': _robust_p95_abs(aux['delta_liquid_capacity'], label='delta_liquid_capacity'),
    }


def apply_merton_fcff_aux_reward(df: pd.DataFrame, stats: dict, *, phase_name: str) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    lam_m = float(stats.get('lambda_merton', 0.0) or 0.0)
    lam_f = float(stats.get('lambda_fcff', 0.0) or 0.0)
    lam_l = float(stats.get('lambda_liquidity', 0.0) or 0.0)
    # Preserve a stable schema even when disabled; do not require simulator next-state columns unless used.
    if lam_m == 0.0 and lam_f == 0.0 and lam_l == 0.0:
        # Overwrite any stale pre-existing aux columns as well as filling absent ones;
        # this preserves lambda=0 no-op semantics and prevents reward-mode gates from
        # leaking previously computed auxiliary rewards into counterfactual transitions.
        for c in AUX_REWARD_COLUMNS:
            out[c] = 0.0
        return out, {'phase': phase_name, 'merton_aux_enabled': False, 'fcff_aux_enabled': False, 'liquidity_aux_enabled': False}
    out = _compute_aux_reward_raw_deltas(out, context=f'{phase_name}_merton_fcff_liquidity_aux')
    m_scale = float(stats['merton_aux_scale_p95_abs'])
    f_scale = float(stats['fcff_aux_scale_p95_abs'])
    l_scale = float(stats.get('liquidity_aux_scale_p95_abs', 1.0) or 1.0)
    out['delta_merton_badness_scaled'] = (pd.to_numeric(out['delta_merton_badness'], errors='coerce') / m_scale).clip(-1.0, 1.0)
    out['delta_fcff_capacity_scaled'] = (pd.to_numeric(out['delta_fcff_capacity'], errors='coerce') / f_scale).clip(-0.5, 0.5)
    out['delta_liquid_capacity_scaled'] = (pd.to_numeric(out['delta_liquid_capacity'], errors='coerce') / l_scale).clip(-0.5, 0.5)
    out['lambda_merton'] = lam_m
    out['lambda_fcff'] = lam_f
    out['lambda_liquidity'] = lam_l
    out['reward_aux_merton'] = lam_m * out['delta_merton_badness_scaled'].fillna(0.0)
    out['reward_aux_fcff'] = lam_f * out['delta_fcff_capacity_scaled'].fillna(0.0)
    out['reward_aux_liquidity'] = lam_l * out['delta_liquid_capacity_scaled'].fillna(0.0)
    meta = {
        'phase': phase_name,
        'merton_aux_enabled': bool(lam_m != 0.0),
        'fcff_aux_enabled': bool(lam_f != 0.0),
        'liquidity_aux_enabled': bool(lam_l != 0.0),
        'merton_aux_mean': float(out['reward_aux_merton'].mean()),
        'fcff_aux_mean': float(out['reward_aux_fcff'].mean()),
        'liquidity_aux_mean': float(out['reward_aux_liquidity'].mean()),
        'merton_aux_nonzero_rate': float((out['reward_aux_merton'].abs() > 1e-12).mean()),
        'fcff_aux_nonzero_rate': float((out['reward_aux_fcff'].abs() > 1e-12).mean()),
        'liquidity_aux_nonzero_rate': float((out['reward_aux_liquidity'].abs() > 1e-12).mean()),
        'reference_oracle_scores_used': False,
        'reference_oracle_variables_used': False,
        'r_code_fallback_allowed': False,
    }
    return out, meta

def ensure_sector_phi(df,rho,phase_name,frozen,allow_current_phi_fallback=False, lambda_phi_override=None):
    out=materialize_phi_aliases(materialize_phi_aliases(df.copy(),next_state=False),next_state=True)
    # v32 reward naming: reward_raw_notch/reward_raw are the base sparse/dense components.
    if 'reward_raw_notch' not in out.columns:
        if 'rating_notch_delta' in out.columns: out['reward_raw_notch']=pd.to_numeric(out['rating_notch_delta'],errors='coerce')
        elif 'reward_original' in out.columns: out['reward_raw_notch']=pd.to_numeric(out['reward_original'],errors='coerce')
        elif 'reward' in out.columns: out['reward_raw_notch']=pd.to_numeric(out['reward'],errors='coerce')
        else: raise ValueError('Missing reward_raw_notch/reward_original/reward column')
    if 'reward_raw' not in out.columns:
        out['reward_raw']=pd.to_numeric(out['reward_raw_notch'],errors='coerce')
    phi=compute_phi_with_frozen_cdf(out,frozen)
    out['phi_t']=pd.to_numeric(out.get('phi_t',phi),errors='coerce')
    next_cols=[f'next__{c}' for c in PHI_COMPONENTS]; alt=[f'{c}__next' for c in PHI_COMPONENTS]; used=False
    if all(c in out.columns for c in next_cols):
        nx=out.copy(); [nx.__setitem__(dst,out[src]) for src,dst in zip(next_cols,PHI_COMPONENTS)]; out['phi_tplusH']=compute_phi_with_frozen_cdf(nx,frozen); policy='computed_phi_t_and_phi_tplusH_from_next_components_with_frozen_train_dev_cdf'
    elif all(c in out.columns for c in alt):
        nx=out.copy(); [nx.__setitem__(dst,out[src]) for src,dst in zip(alt,PHI_COMPONENTS)]; out['phi_tplusH']=compute_phi_with_frozen_cdf(nx,frozen); policy='computed_phi_t_and_phi_tplusH_from_alt_next_components_with_frozen_train_dev_cdf'
    elif 'phi_tplusH' in out.columns and 'delta_phi' in out.columns: policy='existing_v32_phi_columns_verified_against_required_contract'
    elif allow_current_phi_fallback: out['phi_tplusH']=phi; used=True; policy='DEBUG_ONLY_current_phi_fallback_phi_tplusH_equals_phi_t'
    else: raise ValueError('Final Stage2 requires next-state sector-phi components or precomputed phi_tplusH/delta_phi. Use --allow-current-phi-fallback only for smoke/debug runs.')
    out['delta_phi']=pd.to_numeric(out.get('delta_phi',out['phi_tplusH']-out['phi_t']),errors='coerce')
    missing=int(out['delta_phi'].isna().sum())
    if missing: out['delta_phi']=out['delta_phi'].fillna(0.0); missing_policy='aux_zero_for_missing_delta_phi'
    else: missing_policy='none_missing'
    out['delta_phi_clipped']=out['delta_phi'].clip(-1.0,1.0)
    std_r=float(pd.to_numeric(out['reward_raw'],errors='coerce').std()) or 1.0
    std_phi=float(pd.to_numeric(out['delta_phi_clipped'],errors='coerce').std()) or 1.0
    lam=float(lambda_phi_override) if lambda_phi_override is not None else rho*std_r/std_phi
    out['lambda_phi']=float(lam)
    out['reward_aux_phi']=lam*out['delta_phi_clipped'].fillna(0.0)
    out['reward_total_raw']=pd.to_numeric(out['reward_raw'],errors='coerce').fillna(0.0)+out['reward_aux_phi'].fillna(0.0)
    # reward_train/reward are standardized later using phase2+phase3 only.
    out['reward_original']=out['reward_raw']
    out['aux_reward_sector_phi']=out['reward_aux_phi']
    out['phi_t1']=out['phi_tplusH']
    out['phi_diff']=out['delta_phi']
    return out, {'phase':phase_name,'phi_policy':policy,'used_current_phi_fallback':used,'final_paper_run_allowed':not used,'phi_missing_count_before_policy':missing,'phi_missing_policy':missing_policy,'rho':rho,'lambda_phi':float(lam),'phi_cdf_source':frozen['metadata']['phi_cdf_source'],'eval_distribution_used_for_cdf':False,'cdf_frozen':True}
def process_phase(path,out_path,phase,space,rho,do_project,frozen,allow_current_phi_fallback=False,require_reward=True, lambda_phi_override=None, projection_kwargs=None, aux_reward_stats=None):
    df=read_parquet_required(path)
    if phase == 'phase_eval':
        forbidden=[c for c in df.columns if str(c).startswith(('action__','action_observed__','next__','soft_cand_')) or str(c) in set(REQ_REWARD+['candidate_id','projection_distance','out_of_library_flag','near_tie_flag','done'])]
        if forbidden:
            raise ValueError(f'phase_eval must be state-only before Stage2 projection; forbidden columns present: {forbidden[:20]}')
        meta={'phase':phase,'phi_policy':'not_required_for_state_only_eval_phase','reward_required':False,'projection_required':False,'phase_eval_state_only':True,'final_paper_run_allowed':True}
        out_path.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(out_path,index=False)
        return {'phase':phase,'input':str(path),'output':str(out_path),'rows':int(len(df)),**meta}
    validate_action_columns(df,space)
    if do_project or 'candidate_id' not in df.columns: df=project_actions_to_candidates(df, space, **(projection_kwargs or {}))
    else:
        bad=set(df['candidate_id'].astype(str).unique())-set(space.train_labels)
        if bad: raise ValueError(f'Forbidden projection labels in {phase}: {sorted(bad)}')
    if require_reward:
        df,meta=ensure_sector_phi(df,rho,phase,frozen,allow_current_phi_fallback,lambda_phi_override=lambda_phi_override)
        if aux_reward_stats is not None:
            df, aux_meta = apply_merton_fcff_aux_reward(df, aux_reward_stats, phase_name=phase)
            df['reward_total_raw'] = pd.to_numeric(df['reward_total_raw'], errors='coerce').fillna(0.0) + pd.to_numeric(df['reward_aux_merton'], errors='coerce').fillna(0.0) + pd.to_numeric(df['reward_aux_fcff'], errors='coerce').fillna(0.0) + pd.to_numeric(df.get('reward_aux_liquidity', 0.0), errors='coerce').fillna(0.0)
            meta.update({'aux_reward_merton_fcff': aux_meta})
    else:
        meta={
            'phase': phase,
            'phi_policy': 'not_required_for_broad_ssl_bc_phase',
            'reward_required': False,
            'universe': 'broad_raw_financial_transition_universe_no_external_rating_required',
            'final_paper_run_allowed': True,
        }
    out_path.parent.mkdir(parents=True,exist_ok=True); df.to_parquet(out_path,index=False)
    return {'phase':phase,'input':str(path),'output':str(out_path),'rows':int(len(df)),**meta}

def compute_inner_train_reward_stats(train: pd.DataFrame, temporal_contract, rho: float, aux_reward_stats: dict | None = None) -> dict:
    y = pd.to_numeric(train.get('fiscal_year', train.get('year')), errors='coerce')
    inner = train[y <= int(temporal_contract.inner_train_year_max)].copy()
    if inner.empty:
        raise ValueError(f'No phase3_iql inner-train rows for reward standardization <= {temporal_contract.inner_train_year_max}')
    std_r = float(pd.to_numeric(inner['reward_raw'], errors='coerce').std()) or 1.0
    std_phi = float(pd.to_numeric(inner['delta_phi_clipped'], errors='coerce').std()) or 1.0
    lam = float(rho) * std_r / std_phi
    total = pd.to_numeric(inner['reward_raw'], errors='coerce').fillna(0.0) + lam * pd.to_numeric(inner['delta_phi_clipped'], errors='coerce').fillna(0.0)
    if aux_reward_stats is not None and (float(aux_reward_stats.get('lambda_merton', 0.0) or 0.0) != 0.0 or float(aux_reward_stats.get('lambda_fcff', 0.0) or 0.0) != 0.0 or float(aux_reward_stats.get('lambda_liquidity', 0.0) or 0.0) != 0.0):
        aux_inner, _aux_meta = apply_merton_fcff_aux_reward(inner, aux_reward_stats, phase_name='phase3_iql_inner_train_reward_stats')
        total = total + pd.to_numeric(aux_inner['reward_aux_merton'], errors='coerce').fillna(0.0) + pd.to_numeric(aux_inner['reward_aux_fcff'], errors='coerce').fillna(0.0) + pd.to_numeric(aux_inner.get('reward_aux_liquidity', 0.0), errors='coerce').fillna(0.0)
    mean = float(total.mean()) if np.isfinite(float(total.mean())) else 0.0
    std = float(total.std()) if np.isfinite(float(total.std())) and float(total.std()) > 1e-12 else 1.0
    return {'lambda_phi': lam, 'reward_mean_train': mean, 'reward_std_train': std, 'stats_source': 'phase3_iql_inner_train_only', 'inner_train_year_max': int(temporal_contract.inner_train_year_max), 'inner_train_rows': int(len(inner))}

def standardize_rewards_and_write(paths: dict[str, Path], stats: dict, out_dir: Path) -> dict:
    # Broad BC/SSL and state-only eval phases intentionally have no external-rating reward.
    for ph in ['phase3_iql']:
        path = paths[ph]
        df = pd.read_parquet(path)
        df['lambda_phi'] = float(stats['lambda_phi'])
        df['reward_aux_phi'] = df['lambda_phi'] * pd.to_numeric(df['delta_phi_clipped'], errors='coerce').fillna(0.0)
        df['reward_total_raw'] = pd.to_numeric(df['reward_raw'], errors='coerce').fillna(0.0) + df['reward_aux_phi'].fillna(0.0) + pd.to_numeric(df.get('reward_aux_merton', 0.0), errors='coerce').fillna(0.0) + pd.to_numeric(df.get('reward_aux_fcff', 0.0), errors='coerce').fillna(0.0) + pd.to_numeric(df.get('reward_aux_liquidity', 0.0), errors='coerce').fillna(0.0)
        df['reward_mean_train'] = float(stats['reward_mean_train'])
        df['reward_std_train'] = float(stats['reward_std_train'])
        df['reward_train'] = (pd.to_numeric(df['reward_total_raw'], errors='coerce').fillna(stats['reward_mean_train']) - stats['reward_mean_train']) / stats['reward_std_train']
        df['reward'] = df['reward_train']
        df['reward_original'] = df['reward_raw']
        req = [c for c in REQ_REWARD if c not in df.columns]
        if req:
            raise ValueError(f'{ph} missing required v32 reward columns after standardization: {req}')
        df.to_parquet(path, index=False)
    p2 = pd.read_parquet(paths['phase2_bc'])
    p2['reward_available'] = False
    p2['reward_unavailable_reason'] = 'broad_bc_phase_no_external_rating_required'
    p2.to_parquet(paths['phase2_bc'], index=False)
    write_json(out_dir/'reward_standardization_stats.json', {**stats, 'aux_reward_stats': stats.get('aux_reward_stats', {})})
    return {'reward_mean_train': float(stats['reward_mean_train']), 'reward_std_train': float(stats['reward_std_train']), 'lambda_phi': float(stats['lambda_phi']), 'reward_standardized_using': 'phase3_iql_inner_train_only', 'reward_standardization_stats_json': 'reward_standardization_stats.json'}

def write_projection_artifacts(root: Path, out_dir: Path, paths: dict[str, Path], space, projection_settings: dict | None = None) -> dict:
    projection_settings = dict(projection_settings or {})
    frames=[]
    for ph,path in paths.items():
        if ph == 'phase_eval':
            continue
        df=pd.read_parquet(path)
        assert_training_labels_allowed(df, space, 'candidate_id')
        bad=[x for x in df['candidate_id'].dropna().astype(str).unique() if x not in set(space.train_labels)]
        if bad:
            raise ValueError(f'Forbidden projected training labels in {ph}: {bad}')
        tmp=df.copy(); tmp['phase']=ph; frames.append(tmp)
    all_df=pd.concat(frames, ignore_index=True, sort=False)
    support=all_df.groupby(['phase','candidate_id']).size().reset_index(name='n')
    support.to_csv(out_dir/'projection_support_by_candidate.csv', index=False, encoding='utf-8-sig')
    diag_rows=[]
    for ph,g in all_df.groupby('phase'):
        diag_rows.append({
            'phase': ph,
            'n': int(len(g)),
            'mean_projection_distance': float(pd.to_numeric(g['projection_distance'], errors='coerce').mean()),
            'p95_projection_distance': float(pd.to_numeric(g['projection_distance'], errors='coerce').quantile(.95)),
            'out_of_library_rate': float(pd.to_numeric(g['out_of_library_flag'], errors='coerce').fillna(0).mean()),
            'near_tie_rate': float(pd.to_numeric(g['near_tie_flag'], errors='coerce').fillna(0).mean()),
            'n_labels': int(g['candidate_id'].nunique()),
            'projection_mode': str(projection_settings.get('projection_mode', 'active_intent')),
            'a0_policy': str(projection_settings.get('a0_policy', 'allow_nearest')),
            'a0_margin': float(projection_settings.get('a0_margin', 0.0) or 0.0),
            'a0_raw_best_rate': float(pd.to_numeric(g.get('projection_a0_raw_best', pd.Series(False, index=g.index)), errors='coerce').fillna(0).mean()) if 'projection_a0_raw_best' in g.columns else 0.0,
            'a0_override_rate': float(pd.to_numeric(g.get('projection_a0_override_from_noop', pd.Series(False, index=g.index)), errors='coerce').fillna(0).mean()) if 'projection_a0_override_from_noop' in g.columns else 0.0,
            'a0_primary_rate': float((g['candidate_id'].astype(str) == 'A0_noop').mean()),
        })
    # Candidate diversity guard.  A0/OE-only collapse means projection is not
    # using the raw action source space correctly and must not be fed to BC/IQL.
    collapsed = [r for r in diag_rows if r['phase'] in {'phase2_bc', 'phase3_iql'} and int(r['n_labels']) < 5]
    if collapsed:
        raise ValueError({
            'message': 'Stage2 projection label collapse detected before BC/IQL; active-action-first projection did not produce enough candidate diversity.',
            'min_train_phase_labels': 5,
            'collapsed_phases': collapsed,
            'remedy': 'Inspect action_source_coverage.csv and projection_support_by_candidate.csv; do not continue to Stage3-6 with A0/OE-only labels.',
        })
    diag_df=pd.DataFrame(diag_rows)
    diag_df.to_csv(out_dir/'candidate_projection_diagnostics.csv', index=False, encoding='utf-8-sig')
    hi=all_df[pd.to_numeric(all_df['projection_distance'], errors='coerce') > 0.50].copy()
    hi.to_parquet(out_dir/'high_distance_projection_rows.parquet', index=False)
    cand_meta={
        'schema_version':'candidate_library_metadata_v32',
        'candidate_library_hash':space.candidate_library_hash,
        'final_action_contract_hash':space.final_action_contract_hash,
        'action_columns':space.columns,
        'main_train_labels':space.train_labels,
        'scenario_candidates_train_label_allowed':False,
        'diagnostic_candidates_train_label_allowed':False,
        'c2_projection_label':False,
        'fixed_candidate_count':len(space.fixed_candidates),
        'scenario_candidate_count':len(space.scenario_candidates),
        'diagnostic_candidate_count':len(space.diagnostic_candidates),
    }
    write_json(out_dir/'candidate_library_metadata.json', cand_meta)
    diag_json={
        'schema_version':'candidate_projection_diagnostics_v32',
        'created_utc':now(),
        'projection_distance': projection_distance_description(str(projection_settings.get('projection_mode', 'active_intent')), a0_policy=str(projection_settings.get('a0_policy', 'allow_nearest'))),
        'projection_mode': str(projection_settings.get('projection_mode', 'active_intent')),
        'a0_policy': str(projection_settings.get('a0_policy', 'allow_nearest')),
        'a0_margin': float(projection_settings.get('a0_margin', 0.0) or 0.0),
        'weighted_l1_preset': str(projection_settings.get('weighted_l1_preset', '')),
        'weighted_l1_weights': projection_settings.get('weighted_l1_weights', {}),
        'candidate_library_hash':space.candidate_library_hash,
        'final_action_contract_hash':space.final_action_contract_hash,
        'diagnostics':diag_rows,
        'artifacts':{
            'candidate_projection_diagnostics_csv':'candidate_projection_diagnostics.csv',
            'projection_support_by_candidate_csv':'projection_support_by_candidate.csv',
            'high_distance_projection_rows_parquet':'high_distance_projection_rows.parquet',
            'candidate_library_metadata_json':'candidate_library_metadata.json',
        }
    }
    write_json(out_dir/'candidate_projection_diagnostics.json', diag_json)
    return {'projection_artifacts':diag_json['artifacts'], 'projection_phase_diagnostics':diag_rows}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def _quantile_magnitudes(train_df: pd.DataFrame, columns: list[str], q: int) -> dict[str, float]:
    out = {}
    for col in columns:
        vals = pd.to_numeric(train_df.get(col), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        vals = np.abs(vals[np.abs(vals) > 1e-12])
        if len(vals) == 0:
            out[col] = 0.0
        else:
            out[col] = float(np.quantile(vals, float(q) / 100.0))
    return out

def _action_vector_key(vec: dict[str, float], columns: list[str], ndigits: int = 12) -> tuple[float, ...]:
    return tuple(round(float(vec.get(c, 0.0) or 0.0), ndigits) for c in columns)


def _candidate_pairwise_l1(fixed: dict[str, dict[str, float]], columns: list[str]) -> tuple[float, str]:
    names = list(fixed.keys())
    best = float('inf')
    best_pair = ''
    for i, a in enumerate(names):
        va = np.array([float(fixed[a].get(c, 0.0) or 0.0) for c in columns], dtype=float)
        for b in names[i+1:]:
            vb = np.array([float(fixed[b].get(c, 0.0) or 0.0) for c in columns], dtype=float)
            d = float(np.abs(va - vb).sum())
            if d < best:
                best = d
                best_pair = f'{a}::{b}'
    if not np.isfinite(best):
        best = 0.0
    return best, best_pair


def _assert_unique_action_vectors(fixed: dict[str, dict[str, float]], columns: list[str], *, context: str) -> dict:
    seen: dict[tuple[float, ...], str] = {}
    duplicates = []
    for cid, vec in fixed.items():
        key = _action_vector_key(vec, columns)
        if key in seen:
            duplicates.append({'first': seen[key], 'second': cid})
        else:
            seen[key] = cid
    if duplicates:
        raise ValueError(f'{context}: duplicate fixed candidate action vectors after recalibration: {duplicates}')
    min_l1, min_pair = _candidate_pairwise_l1(fixed, columns)
    return {'unique_action_vector_count': len(seen), 'duplicate_action_vector_count': 0, 'min_pairwise_l1_distance': min_l1, 'min_pairwise_l1_pair': min_pair}


def _recalibrated_action_space(space: ActionSpace, train_df: pd.DataFrame, q: int) -> tuple[ActionSpace, dict[str, float]]:
    mags = _quantile_magnitudes(train_df, space.columns, q)
    fixed = {}
    fallback_dims = []
    clipped = []
    base_max_abs = {}
    for col in space.columns:
        vals = [abs(float(vec.get(col, 0.0) or 0.0)) for vec in space.fixed_candidates.values()]
        vals = [v for v in vals if v > 1e-12]
        base_max_abs[col] = max(vals) if vals else 0.0
    for cid, vec in space.fixed_candidates.items():
        new_vec = dict(vec)
        for col in space.columns:
            val = float(vec.get(col, 0.0) or 0.0)
            if abs(val) <= 1e-12:
                new_vec[col] = 0.0
                continue
            mag = float(mags.get(col, 0.0) or 0.0)
            denom = float(base_max_abs.get(col, 0.0) or 0.0)
            if mag <= 1e-12 or denom <= 1e-12:
                new_abs = abs(val)
                fallback_dims.append(col)
            lo, hi = space.bounds[col]
            sign = float(np.sign(val))
            bound_abs = abs(hi) if sign > 0 else abs(lo)
            if mag <= 1e-12 or denom <= 1e-12:
                new_abs = min(abs(val), bound_abs)
                fallback_dims.append(col)
            else:
                # Preserve mild-vs-moderate and mixed-candidate tier geometry.
                # Previous logic assigned the same per-dimension quantile magnitude
                # to every nonzero candidate, which collapsed DL1==DL2 and OE1==OE2.
                # The tier ratio is applied after considering the feasible bound so
                # smaller tiers do not collapse into the same clipped boundary value.
                tier_ratio = min(1.0, abs(val) / denom)
                feasible_anchor = min(mag, bound_abs)
                new_abs = feasible_anchor * tier_ratio
            unclipped = float(sign * new_abs)
            new_val = float(np.clip(unclipped, lo, hi))
            if abs(new_val - unclipped) > 1e-12:
                clipped.append({'candidate_id': cid, 'column': col, 'unclipped': unclipped, 'clipped': new_val})
            new_vec[col] = new_val
        fixed[cid] = new_vec
    uniqueness = _assert_unique_action_vectors(fixed, space.columns, context=f'P{q} recalibrated candidate library')
    recal = ActionSpace(space.columns, space.bounds, fixed, space.train_labels, space.row_conditional_baselines, space.final_rl_label, space.scenario_candidates, space.diagnostic_candidates, space.candidate_library_hash, space.final_action_contract_hash)
    return recal, {'quantile': int(q), 'magnitudes': mags, 'fallback_to_base_dimension_count': len(set(fallback_dims)), 'fallback_to_base_dimensions': sorted(set(fallback_dims)), 'tier_geometry_preservation': 'scale empirical per-dimension quantile by base candidate abs(value)/max_abs(value) before clipping', 'clipped_recalibrated_value_count': len(clipped), 'clipped_recalibrated_values_preview': clipped[:20], **uniqueness}

def _write_recalibrated_candidate_yaml(base_path: Path, out_path: Path, recal_space: ActionSpace, q: int, quantile_meta: dict) -> str:
    data = yaml.safe_load(base_path.read_text(encoding='utf-8')) if base_path.exists() else {}
    data['magnitude_recalibration'] = {
        'mode': 'inner_train_quantile_recalibrated',
        'selected_quantile': int(q),
        'method': 'tier_preserving_per_dimension_inner_train_abs_action_quantile',
        'tier_geometry_preservation': quantile_meta.get('tier_geometry_preservation'),
        'unique_action_vector_count': quantile_meta.get('unique_action_vector_count'),
        'fallback_to_base_dimensions': quantile_meta.get('fallback_to_base_dimensions', []),
        'oot_used_for_calibration': False,
        'oracle_outcome_used_for_calibration': False,
    }
    data['fixed_candidates'] = recal_space.fixed_candidates
    out_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return _sha256_file(out_path)

def _write_quantile_projected_phase(src_path: Path, dst_path: Path, phase: str, recal_space: ActionSpace, projection_kwargs: dict | None = None) -> None:
    df = read_parquet_required(src_path)
    if phase == 'phase_eval':
        df.to_parquet(dst_path, index=False)
        return
    validate_action_columns(df, recal_space)
    df = project_actions_to_candidates(df, recal_space, **(projection_kwargs or {}))
    assert_training_labels_allowed(df, recal_space, 'candidate_id')
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst_path, index=False)

def main(argv=None):
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root',required=True)
    ap.add_argument('--sector-phi',action='store_true')
    ap.add_argument('--rho',type=float,default=0.0, help='Sector-Phi shaping weight; canonical frozen-paper default is 0.0')
    ap.add_argument('--merton-lambda', type=float, default=0.05, help='Merton/KMV default-point-burden auxiliary reward weight; canonical frozen-paper default is 0.05')
    ap.add_argument('--fcff-lambda', type=float, default=0.45, help='OCF-minus-Capex FCFF-capacity auxiliary reward weight; canonical frozen-paper default is 0.45')
    ap.add_argument('--liquidity-lambda', type=float, default=0.45, help='Liquid-asset capacity auxiliary reward weight; canonical frozen-paper default is 0.45')
    ap.add_argument('--no-project',action='store_true')
    ap.add_argument('--allow-current-phi-fallback',action='store_true')
    ap.add_argument('--projection-mode', choices=['active_intent','l1_best','weighted_l1'], default='active_intent')
    ap.add_argument('--a0-policy', choices=['allow_nearest','margin'], default='allow_nearest')
    ap.add_argument('--a0-margin', type=float, default=0.0)
    ap.add_argument('--weighted-l1-preset', default='credit_recourse_v1')
    ap.add_argument('--weighted-l1-weights-json', default='', help='Optional JSON object overriding weighted-L1 per action__ column weights')
    args=ap.parse_args(argv); root=Path(args.project_root).resolve(); temporal_contract=load_temporal_contract(root); final=final_root(root); space=load_action_space(root); in_dir=final/'stage2_candidate_projection'/'input_splits'; out_dir=final/'stage2_candidate_projection'; out_dir.mkdir(parents=True,exist_ok=True); write_feature_manifest(out_dir/'feature_manifest.json')
    weighted_overrides = json.loads(args.weighted_l1_weights_json) if str(args.weighted_l1_weights_json or '').strip() else None
    weighted_l1_weights = resolve_weighted_l1_weights(space, preset=args.weighted_l1_preset, overrides=weighted_overrides) if args.projection_mode == 'weighted_l1' else None
    projection_kwargs = {
        'projection_mode': args.projection_mode,
        'a0_policy': args.a0_policy,
        'a0_margin': float(args.a0_margin),
        'weighted_l1_weights': weighted_l1_weights,
    }
    if args.no_project and args.projection_mode != 'active_intent':
        raise ValueError('--no-project cannot be combined with projection-mode l1_best/weighted_l1 because the ablation must rewrite projection labels')
    if args.projection_mode == 'active_intent' and args.a0_policy != 'allow_nearest':
        raise ValueError('--a0-policy margin is only valid with --projection-mode l1_best or weighted_l1')
    projection_settings = {
        'projection_mode': args.projection_mode,
        'projection_distance': projection_distance_description(args.projection_mode, a0_policy=args.a0_policy),
        'a0_policy': args.a0_policy,
        'a0_margin': float(args.a0_margin),
        'weighted_l1_preset': args.weighted_l1_preset if args.projection_mode == 'weighted_l1' else '',
        'weighted_l1_weights': weighted_l1_weights or {},
    }
    raw3=materialize_phi_aliases(read_parquet_required(in_dir/'phase3_iql.parquet'))
    y3=pd.to_numeric(raw3.get('fiscal_year', raw3.get('year')), errors='coerce')
    raw3_inner=raw3[y3 <= int(temporal_contract.inner_train_year_max)].copy()
    if raw3_inner.empty: raise ValueError('Cannot build sector-phi CDF: phase3_iql inner-train partition is empty')
    frozen=build_frozen_phi_cdf(raw3_inner,out_dir)
    phases=[('phase1_pretrain','phase1_pretrain_candidate.parquet',False),('phase2_bc','phase2_bc_candidate.parquet',False),('phase3_iql','phase3_iql_candidate.parquet',True),('phase_eval','phase_eval_candidate.parquet',False)]
    paths={ph:out_dir/outname for ph,outname,_ in phases}
    # First process phase3 with a temporary lambda to derive frozen inner-train stats, then rewrite using the frozen stats.
    tmp_phase3, _tmp_meta = ensure_sector_phi(raw3.copy(), args.rho, 'phase3_iql_inner_train_stats_probe', frozen, args.allow_current_phi_fallback)
    aux_reward_stats = compute_aux_reward_stats(tmp_phase3, temporal_contract, merton_lambda=args.merton_lambda, fcff_lambda=args.fcff_lambda, liquidity_lambda=args.liquidity_lambda)
    stats = compute_inner_train_reward_stats(tmp_phase3, temporal_contract, args.rho, aux_reward_stats=aux_reward_stats)
    stats['aux_reward_stats'] = aux_reward_stats
    metas=[process_phase(in_dir/f'{ph}.parquet',paths[ph],ph,space,args.rho,not args.no_project,frozen,args.allow_current_phi_fallback,require_reward=req,lambda_phi_override=stats['lambda_phi'],projection_kwargs=projection_kwargs, aux_reward_stats=aux_reward_stats) for ph,outname,req in phases]
    reward_std_meta=standardize_rewards_and_write(paths, stats, out_dir)
    
    # Inner-train quantile-recalibrated magnitude libraries.  This replaces the
    # previous copyfile-only placeholder: for each P{q}, nonzero candidate action
    # dimensions keep their sign pattern but receive the empirical q-th quantile
    # magnitude from phase3_iql inner-train pseudo-actions.
    magnitude_quantiles=[50,65,75,85]
    quantile_artifacts=[]
    base_lib=root/'data/final_freeze/configs/final_candidate_library.yaml'
    if not base_lib.exists():
        raise FileNotFoundError(f'Missing active final candidate library for recalibration: {base_lib}')
    candidate_library_hash_per_quantile={}
    for q in magnitude_quantiles:
        recal_space, qmeta = _recalibrated_action_space(space, raw3_inner, q)
        lib_out=out_dir/f'final_candidate_library__P{q}.yaml'
        lib_hash=_write_recalibrated_candidate_yaml(base_lib, lib_out, recal_space, q, qmeta)
        candidate_library_hash_per_quantile[f'P{q}']=lib_hash
        for ph in ['phase1_pretrain','phase2_bc','phase3_iql']:
            src=paths[ph]; dst=out_dir/f'{ph}_candidate__P{q}.parquet'
            _write_quantile_projected_phase(src, dst, ph, recal_space, projection_kwargs)
        quantile_artifacts.append({'magnitude_quantile':q,'candidate_library':f'final_candidate_library__P{q}.yaml','candidate_library_hash':lib_hash,'status':'COMPUTED_FROM_PHASE3_IQL_INNER_TRAIN_QUANTILES',**qmeta})
    write_json(out_dir/'magnitude_recalibrated_libraries_metadata.json',{'schema_version':'inner_dev_recalibrated_magnitude_libraries_v2','mode':'inner_dev_recalibrated','quantile_grid':magnitude_quantiles,'inner_train_year_max':int(temporal_contract.inner_train_year_max),'inner_dev_year':int(temporal_contract.inner_dev_year),'recalibration_method':'tier_preserving_per_dimension_favorable_direction_abs_action_quantile','ratio_preservation':'preserve active dimension sign pattern and relative mild/moderate tier geometry; scale quantile magnitudes by base candidate abs(value)/dimension max_abs before clipping','oot_used_for_calibration':False,'oracle_outcome_used_for_calibration':False,'candidate_library_hash_per_quantile':candidate_library_hash_per_quantile,'quantiles':quantile_artifacts})

    projection_meta=write_projection_artifacts(root,out_dir,paths,space, projection_settings)
    write_json(out_dir/'input_split_provenance.json',{'stage':'final_stage2_input_split_provenance','created_utc':now(),'input_splits_dir':str(in_dir),'phase_files':{ph:str(in_dir/f'{ph}.parquet') for ph,_,_ in phases},'split_builder_inside_stage2':False,'input_split_builder_module':'credit_recourse.rl.pipelines.final_stage2_input_splits.pipeline','note':'Stage2 consumes input_splits materialized by final_stage2_input_splits.pipeline and validates/projects them.'})
    meta={'stage':'final_stage2_candidate_projection','created_utc':now(),'status':'PASS','input_splits':str(in_dir),'output_dir':str(out_dir),'action_columns':space.columns,'projection_distance': projection_distance_description(str(projection_settings.get('projection_mode', 'active_intent')), a0_policy=str(projection_settings.get('a0_policy', 'allow_nearest'))),
        'projection_mode': str(projection_settings.get('projection_mode', 'active_intent')),
        'a0_policy': str(projection_settings.get('a0_policy', 'allow_nearest')),
        'a0_margin': float(projection_settings.get('a0_margin', 0.0) or 0.0),
        'weighted_l1_preset': str(projection_settings.get('weighted_l1_preset', '')),
        'weighted_l1_weights': projection_settings.get('weighted_l1_weights', {}),'soft_target_columns':['soft_cand_id_1','soft_cand_id_2','soft_cand_id_3','soft_cand_prob_1','soft_cand_prob_2','soft_cand_prob_3'],'c2_projection_label':False,'fixed_train_labels':space.train_labels,'sector_phi_required':True,'sector_phi_components_materialized':PHI_COMPONENTS,'korean_nice_alias_materialization':True,'phi_cdf_source':'rated_phase3_iql_inner_train_only','eval_distribution_used_for_phi_cdf':False, **temporal_metadata(temporal_contract, stage='final_stage2_candidate_projection'),'cdf_frozen':True,'candidate_library_hash_per_quantile':candidate_library_hash_per_quantile,'rho_main':args.rho,'rho_used':args.rho,'rho_cli_default':0.0,'rho_default_expected':0.0,'merton_lambda_cli_default':0.05,'fcff_lambda_cli_default':0.45,'liquidity_lambda_cli_default':0.45,'reward_default_contract':'frozen_paper_profile_0p0_0p05_0p45_0p45','reward_column_contract':'v32_reward_train_plus_optional_merton_fcff_liquidity_aux','required_reward_columns_for_rated_phases':REQ_REWARD, 'optional_aux_reward_columns':AUX_REWARD_COLUMNS, 'aux_reward_stats':aux_reward_stats,'allow_current_phi_fallback':bool(args.allow_current_phi_fallback),'current_as_next_fallback_used':bool(args.allow_current_phi_fallback),'final_paper_run_allowed':not bool(args.allow_current_phi_fallback),'candidate_library_hash':space.candidate_library_hash,'final_action_contract_hash':space.final_action_contract_hash,**reward_std_meta,**projection_meta,'phases':metas}
    write_json(out_dir/'metadata.json',meta); write_json(out_dir/'sector_phi_repair_v28_metadata.json',meta); print(json.dumps(meta,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
