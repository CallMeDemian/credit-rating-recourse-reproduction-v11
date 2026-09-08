from __future__ import annotations
import argparse, json, math, sys
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from credit_recourse.rl.common.io import final_root, write_json, read_parquet_required
from credit_recourse.rl.common.actions import load_action_space
from credit_recourse.rl.common.temporal import load_temporal_contract, temporal_metadata
from credit_recourse.simulator.firm_state import FirmState, load_firm_state_from_registry
from credit_recourse.simulator.action import Action, clip_action
from credit_recourse.simulator.business_plan import BusinessPlan, calibrate_business_plan
from credit_recourse.simulator.financial_simulator import FinancialSimulator
from credit_recourse.simulator.oracle_variables import compute_oracle_variables, audit_formula_registry

def now(): return datetime.now(timezone.utc).isoformat()


def resolve_stage6_rollout_target_year(temporal_contract) -> int:
    """Resolve the fiscal year of Stage6 simulated oracle inputs.

    Stage6 policy selection consumes the state-only eval base year, but the
    financial-statement simulator rolls those states one year forward.  The
    resulting oracle input frame must therefore be stamped with the rollout
    target year, not the policy selection base year.
    """
    raw = getattr(temporal_contract, 'raw', {}) or {}
    for key in ('rollout_target_year', 'predicted_fiscal_year', 'target_year'):
        val = raw.get(key)
        if val is not None and str(val).strip() != '':
            return int(val)
    attr = getattr(temporal_contract, 'rollout_target_year', None)
    if attr is not None and str(attr).strip() != '':
        return int(attr)
    return int(temporal_contract.eval_base_year) + 1
def load_registry(path: Path) -> dict[str,Any]:
    if not path.exists(): raise FileNotFoundError(f"Missing oracle backend registry: {path}")
    text=path.read_text(encoding='utf-8')
    try: return json.loads(text)
    except Exception: return yaml.safe_load(text)


def resolve_backend_artifact(root: Path, final: Path, value: Any) -> Path:
    """Resolve backend artifact paths from portable registry entries.

    Supports:
      - existing absolute paths on the local machine,
      - project-root relative paths such as data/final_freeze/..., and
      - stale absolute paths from another checkout by recovering the
        data/final_freeze/... or stage1_oracle_backends/... suffix.
    """
    if value is None or str(value).strip() == "":
        return Path("")
    raw = str(value).strip()
    p = Path(raw)
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append((root / p).resolve())
        candidates.append((final / p).resolve())

    normalized = raw.replace('\\', '/')
    marker = 'data/final_freeze/'
    if marker in normalized:
        suffix = normalized[normalized.index(marker):]
        candidates.append((root / suffix).resolve())
    marker2 = 'stage1_oracle_backends/'
    if marker2 in normalized:
        suffix = normalized[normalized.index(marker2):]
        candidates.append((final / suffix).resolve())

    for cand in candidates:
        if cand.exists():
            return cand
    return candidates[0] if candidates else p


def _logistic_cdf(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _transform_ordered_logit_thresholds(raw_threshold_params: list[float]) -> np.ndarray:
    """Statsmodels OrderedModel threshold transform.

    The first threshold is direct; later threshold parameters are exponentiated
    increments and cumulatively summed to enforce monotonic cutpoints.
    """
    raw = np.asarray(raw_threshold_params, dtype=float)
    if raw.size == 0:
        return raw
    increments = np.concatenate([raw[:1], np.exp(raw[1:])])
    return np.cumsum(increments)

def _first(row, names, default=None):
    for n in names:
        if n in row and pd.notna(row[n]): return row[n]
    return default


STAGE6_FIRMSTATE_FIELD_ALIASES = {
    "accounts_receivable": "receivables",
    "accounts_payable": "payables",
    "short_debt": "short_term_debt",
    "long_debt": "long_term_debt",
    "bond": "bonds",
}

def _stage6_state_alias_keys(column: str) -> list[str]:
    """Alias keys for Stage6 FirmState loading from state-only phase_eval rows.

    The phase_eval panel can carry financial accounts under raw provenance
    prefixes.  Preserve original columns while adding canonical aliases so that
    load_firm_state_from_columns can recover the simulator fields without
    weakening the state-only/no-leakage contract.
    """
    c = str(column)
    keys = [c]
    prefixes = ("raw__sim__", "raw__", "raw__avs__", "avs__", "sim__")
    for prefix in prefixes:
        if c.startswith(prefix):
            keys.append(c[len(prefix):])
    if "__" in c:
        keys.append(c.rsplit("__", 1)[-1])
    for k in list(keys):
        if k in STAGE6_FIRMSTATE_FIELD_ALIASES:
            keys.append(STAGE6_FIRMSTATE_FIELD_ALIASES[k])
    # de-duplicate preserving order
    out: list[str] = []
    seen: set[str] = set()
    for k in keys:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

def _row_to_firm_state(row: pd.Series) -> FirmState:
    d={}
    for k,v in row.items():
        for alias in _stage6_state_alias_keys(str(k)):
            if alias not in d or pd.isna(d.get(alias)):
                d[alias] = v
    firm_id=str(_first(row, ['firm_id','corp_code','company_id','醫낅ぉ肄붾뱶'], 'UNKNOWN'))
    year=int(float(_first(row, ['year','fiscal_year','?ъ뾽?곕룄'], 0)))
    sector=str(_first(row, ['sector','sector_7','industry_class','market'], 'Unknown'))
    fs=load_firm_state_from_registry(d, firm_id=firm_id, year=year, sector=sector)
    fs.rating_num=_first(row, ['rating_num','rating_numeric'], None)
    fs.rating_grade=_first(row, ['rating_grade','grade','credit_rating'], None)
    return fs

def _action_from_row(row: pd.Series, space) -> Action:
    vals={c.replace('action__',''): float(row.get(c, 0.0) or 0.0) for c in space.columns}
    return clip_action(Action(**vals))

def prepare_stage6_history_lookup(base: pd.DataFrame) -> dict[str, list[tuple[int, FirmState]]]:
    """Precompute firm histories for repeated evaluator-only simulation draws.

    The canonical implementation formerly filtered the full base DataFrame once
    per action row. Repeated permutation analysis therefore performed an
    avoidable O(n^2) history lookup on every draw. This lookup preserves the
    exact same <=year, sort, and tail(3) semantics while materializing each base
    FirmState once per worker process.
    """
    if 'firm_id' not in base.columns and 'corp_code' not in base.columns:
        return {}
    fid_col = 'firm_id' if 'firm_id' in base.columns else 'corp_code'
    y_col = 'fiscal_year' if 'fiscal_year' in base.columns else ('year' if 'year' in base.columns else None)
    if y_col is None:
        return {}
    lookup: dict[str, list[tuple[int, FirmState]]] = {}
    ordered = base.copy()
    ordered['_history_year'] = pd.to_numeric(ordered[y_col], errors='coerce')
    ordered = ordered.loc[ordered['_history_year'].notna()].sort_values([fid_col, '_history_year'])
    for firm_id, group in ordered.groupby(fid_col, dropna=False, sort=False):
        records: list[tuple[int, FirmState]] = []
        for _, row in group.drop(columns=['_history_year']).iterrows():
            try:
                fs = _row_to_firm_state(row)
                records.append((int(float(row[y_col])), fs))
            except Exception:
                continue
        lookup[str(firm_id)] = records
    return lookup


def _build_stage6_history(
    base: pd.DataFrame,
    firm_id: str,
    year: int,
    *,
    history_lookup: dict[str, list[tuple[int, FirmState]]] | None = None,
) -> list[FirmState]:
    lookup = history_lookup if history_lookup is not None else prepare_stage6_history_lookup(base)
    records = lookup.get(str(firm_id), [])
    return [state for hist_year, state in records if int(hist_year) <= int(year)][-3:]


def _select_stage6_business_plan(mode: str, hist: list[FirmState], rating_grade=None) -> BusinessPlan:
    if mode == 'default':
        return BusinessPlan()
    if mode == 'calibrated':
        return calibrate_business_plan(hist, grade=rating_grade) if hist else BusinessPlan()
    raise ValueError(f'Unsupported sim_business_plan_mode: {mode}')


def _r136_from_state(state: FirmState) -> float:
    denom = float(getattr(state, 'current_liabilities', 0.0) or 0.0)
    if abs(denom) <= 1e-12:
        return float('nan')
    return float(getattr(state, 'payables', 0.0) or 0.0) / denom


def _selected_oracle_audit_from_state(state: FirmState, prev_state: FirmState | None = None) -> dict[str, Any]:
    """Small audit payload for current and legacy selected financial variables.

    These columns are diagnostic only.  Scoring uses ``compute_oracle_variables``
    and backend-selected variables, while formula coverage is guarded by
    ``audit_formula_registry`` before scoring.
    """
    return compute_oracle_variables(state, prev_state=prev_state)


def _state_to_frame_dict(sim_vars: dict[str,Any], base_row: pd.Series) -> dict[str,Any]:
    out=dict(sim_vars)
    # RL-S6-007: preserve Stage1-selected nonfinancial/context variables
    # required by Alpha/Beta/Gamma backend params. These are not simulator
    # formulas and must be supplied from the original phase_eval row.
    passthrough_cols = [
        'industry_median_rating_lag1_self_excl', 'industry_avg_rating_lag1_self_excl',
        'industry_bad_grade_share_lag1_self_excl',
        'cap_change_count_3y', 'log_assets', 'nf_log_assets',
        'operating_loss_freq_3y', 'financial_data_completeness',
        'ratio_missing_rate', 'nf_ratio_missing_rate',
        'nf_retained_earnings_negative_flag',
        'firm_id', 'year', 'fiscal_year', 'sector_7', 'industry_class'
    ]
    for c in passthrough_cols:
        if c in base_row and c not in out:
            out[c]=base_row[c]
    if 'industry_bad_grade_share_lag1_self_excl' not in out and 'alpha__industry_bad_grade_share_lag1_self_excl' in base_row:
        out['industry_bad_grade_share_lag1_self_excl'] = base_row['alpha__industry_bad_grade_share_lag1_self_excl']
    if 'ratio_missing_rate' not in out and 'nf_ratio_missing_rate' in out:
        out['ratio_missing_rate'] = out['nf_ratio_missing_rate']
    return out

def simulate_policy_states(base: pd.DataFrame, policy_actions: pd.DataFrame, space, out: Path, *, predicted_fiscal_year: int | None = None, preserve_current_non_current_residual: bool = False, sim_business_plan_mode: str = 'default', write_outputs: bool = True, collect_audit: bool = True, history_lookup: dict[str, list[tuple[int, FirmState]]] | None = None) -> tuple[pd.DataFrame,pd.DataFrame]:
    sim=FinancialSimulator(preserve_current_non_current_residual=preserve_current_non_current_residual)
    rows=[]; audits=[]
    if history_lookup is None:
        history_lookup = prepare_stage6_history_lookup(base)
    base_idx=base.reset_index(drop=True).reset_index().rename(columns={'index':'row_id'})
    pa=policy_actions.merge(base_idx, on='row_id', how='left', suffixes=('','__base'))
    if pa.isna().all(axis=1).any(): raise ValueError('Policy actions contain row_id not found in phase_eval base')
    for _,r in pa.iterrows():
        fs=_row_to_firm_state(r); action=_action_from_row(r, space)
        hist=_build_stage6_history(base, str(fs.firm_id), int(fs.year), history_lookup=history_lookup)
        bp=_select_stage6_business_plan(sim_business_plan_mode, hist, rating_grade=getattr(fs, 'rating_grade', None))
        result=sim.simulate(fs, bp, action)
        residual_audit = result.diagnostics.get('residual_audit', {}) or {}
        r136_before = _r136_from_state(fs)
        r136_after = _r136_from_state(result.state_t1)
        selected_audit_before = _selected_oracle_audit_from_state(fs)
        selected_audit_after = _selected_oracle_audit_from_state(result.state_t1, prev_state=fs)
        sim_vars=compute_oracle_variables(result.state_t1, prev_state=fs, exogenous={
            'industry_median_rating_lag1_self_excl': r.get('industry_median_rating_lag1_self_excl'),
            'industry_avg_rating_lag1_self_excl': r.get('industry_avg_rating_lag1_self_excl'),
            'cap_change_count_3y': r.get('cap_change_count_3y'),
            'log_assets': r.get('log_assets'),
            'nf_log_assets': r.get('nf_log_assets'),
            'operating_loss_freq_3y': r.get('operating_loss_freq_3y'),
            'financial_data_completeness': r.get('financial_data_completeness'),
            'nf_retained_earnings_negative_flag': r.get('nf_retained_earnings_negative_flag'),
            'nf_ratio_missing_rate': r.get('nf_ratio_missing_rate'),
            'ratio_missing_rate': r.get('ratio_missing_rate'),
        })
        row=_state_to_frame_dict(sim_vars, r)
        row.update({'row_id':r['row_id'],'policy':r['policy'],'candidate_id':r['candidate_id'],'sustainability':result.sustainability,'plug_used':result.plug_used,'plug_amount':result.plug_amount,
                    'mode': r.get('mode') if 'mode' in r.index else None,
                    'sim_business_plan_mode': sim_business_plan_mode, 'preserve_current_non_current_residual': bool(preserve_current_non_current_residual),
                    'total_assets_before': float(getattr(fs, 'total_assets', np.nan) or 0.0),
                    'current_liabilities_before': float(getattr(fs, 'current_liabilities', np.nan) or 0.0),
                    'current_liabilities_after': float(getattr(result.state_t1, 'current_liabilities', np.nan) or 0.0),
                    'current_assets_before': float(getattr(fs, 'current_assets', np.nan) or 0.0),
                    'current_assets_after': float(getattr(result.state_t1, 'current_assets', np.nan) or 0.0),
                    'R136_before': r136_before, 'R136_after': r136_after,
                    'R133_before': selected_audit_before.get('R133'), 'R133_after': selected_audit_after.get('R133'),
                    'R182_before': selected_audit_before.get('R182'), 'R182_after': selected_audit_after.get('R182'),
                    'residual_negative_flag': bool(any(bool(residual_audit.get(k, False)) for k in ['other_current_assets_clipped','other_non_current_assets_clipped','other_current_liabilities_clipped','other_non_current_liabilities_clipped'])),
                    'other_current_assets_t': float(residual_audit.get('other_current_assets_raw', np.nan)),
                    'other_non_current_assets_t': float(residual_audit.get('other_non_current_assets_raw', np.nan)),
                    'other_current_liab_t': float(residual_audit.get('other_current_liabilities_raw', np.nan)),
                    'other_non_current_liab_t': float(residual_audit.get('other_non_current_liabilities_raw', np.nan))})
        if predicted_fiscal_year is not None:
            row['predicted_fiscal_year'] = int(predicted_fiscal_year)
        rows.append(row)
        if collect_audit:
            # Per-action effect audit: record the financial account/ratio affected by each nonzero action.
            # This is not a proxy evaluator; it is an audit trail around the full simulator result.
            audit_targets = {
                'action__ppe_pct': [('ppe', getattr(fs, 'ppe', np.nan), getattr(result.state_t1, 'ppe', np.nan))],
                'action__inv_turnover_chg': [('inventory', getattr(fs, 'inventory', np.nan), getattr(result.state_t1, 'inventory', np.nan))],
                'action__ar_turnover_chg': [('receivables', getattr(fs, 'receivables', np.nan), getattr(result.state_t1, 'receivables', np.nan))],
                'action__ap_turnover_chg': [('payables', getattr(fs, 'payables', np.nan), getattr(result.state_t1, 'payables', np.nan))],
                'action__short_debt_pct': [('short_term_debt', getattr(fs, 'short_term_debt', np.nan), getattr(result.state_t1, 'short_term_debt', np.nan))],
                'action__long_debt_pct': [('long_term_debt', getattr(fs, 'long_term_debt', np.nan), getattr(result.state_t1, 'long_term_debt', np.nan))],
                'action__bond_pct': [('bonds', getattr(fs, 'bonds', np.nan), getattr(result.state_t1, 'bonds', np.nan))],
                'action__revenue_growth': [('revenue', getattr(fs, 'revenue', np.nan), getattr(result.state_t1, 'revenue', np.nan))],
                'action__cogs_ratio_chg': [('cogs', getattr(fs, 'cogs', np.nan), getattr(result.state_t1, 'cogs', np.nan))],
                'action__sga_ratio_chg': [('sga', getattr(fs, 'sga', np.nan), getattr(result.state_t1, 'sga', np.nan))],
            }
            for ac in space.columns:
                val=float(r.get(ac,0.0) or 0.0)
                if val==0.0: continue
                raw_name=ac.replace('action__','')
                clipped_val=float(getattr(_action_from_row(r, space), raw_name))
                clipped=abs(clipped_val-val)>1e-12
                for target,before,after in audit_targets.get(ac, [('financial_statement_simulator', np.nan, np.nan)]):
                    try:
                        delta=float(after)-float(before)
                    except Exception:
                        delta=np.nan
                    audits.append({'firm_id':fs.firm_id,'year':fs.year,'row_id':r['row_id'],'policy':r['policy'],'candidate_id':r['candidate_id'],'mode':r.get('mode') if 'mode' in r.index else None,'action_dim':ac,'action_value':val,'target_variable':target,'affected_account_or_ratio':target,'before_value':before,'after_value':after,'delta_value':delta,'clipped':bool(clipped),'clip_reason':'action_bound_clip' if clipped else 'not_clipped','adapter_rule_id':'full_financial_statement_simulator','oracle_backend':'pending','score_before':np.nan,'score_after':np.nan,'score_delta':np.nan,'simulator_preflight_status':'ok','sustainability':result.sustainability,'fallback_metrics':json.dumps(result.diagnostics.get('fallback_metrics',{}),ensure_ascii=False),'plug_used':result.plug_used,'accounting_check':json.dumps(result.accounting_check,ensure_ascii=False),'sim_business_plan_mode':sim_business_plan_mode,'preserve_current_non_current_residual':bool(preserve_current_non_current_residual),'total_assets_before':float(getattr(fs,'total_assets',np.nan) or 0.0),'current_liabilities_before':float(getattr(fs,'current_liabilities',np.nan) or 0.0),'current_liabilities_after':float(getattr(result.state_t1,'current_liabilities',np.nan) or 0.0),'current_assets_before':float(getattr(fs,'current_assets',np.nan) or 0.0),'current_assets_after':float(getattr(result.state_t1,'current_assets',np.nan) or 0.0),'R136_before':r136_before,'R136_after':r136_after,'R133_before':selected_audit_before.get('R133'),'R133_after':selected_audit_after.get('R133'),'R182_before':selected_audit_before.get('R182'),'R182_after':selected_audit_after.get('R182'),'residual_negative_flag':bool(any(bool(residual_audit.get(k, False)) for k in ['other_current_assets_clipped','other_non_current_assets_clipped','other_current_liabilities_clipped','other_non_current_liabilities_clipped']))})
    state=pd.DataFrame(rows); audit=pd.DataFrame(audits)
    if write_outputs:
        state.to_parquet(out/'simulated_oracle_input_frame.parquet', index=False)
        audit.to_parquet(out/'action_effect_audit.parquet', index=False)
    return state,audit


@lru_cache(maxsize=16)
def _load_json_artifact_cached(path_text: str) -> dict[str, Any]:
    path = Path(path_text)
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=8)
def _load_gamma_model_cached(path_text: str):
    import joblib
    return joblib.load(Path(path_text))


@lru_cache(maxsize=8)
def _load_alpha_scorer_cached(path_text: str):
    from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer
    params = _load_json_artifact_cached(path_text)
    return build_alpha_scorer(params)

def score_alpha(df: pd.DataFrame, params_path: Path) -> pd.Series:
    params=_load_json_artifact_cached(str(Path(params_path).resolve()))
    vars=params.get('selected_variables') or [v.get('variable_id') for v in params.get('variables',[]) if isinstance(v,dict)]
    vars=[v for v in vars if v]
    if not vars: raise ValueError('Alpha params missing selected_variables')
    missing=[v for v in vars if v not in df.columns]
    if missing: raise KeyError(f'Alpha scoring missing variables: {missing[:20]}')
    scorer=_load_alpha_scorer_cached(str(Path(params_path).resolve()))
    vals=[]
    for _, row in df.iterrows():
        vals.append(float(scorer({v: row.get(v) for v in vars})['R_score']))
    return pd.Series(vals, index=df.index, name='R_score_alpha')

def score_beta_ordered_logit_params(df: pd.DataFrame, params_path: Path) -> pd.Series:
    params=_load_json_artifact_cached(str(Path(params_path).resolve()))
    if 'ordered' not in str(params.get('model_name','')).lower() and 'ordered' not in str(params.get('version','')).lower():
        raise ValueError(f'Beta params are not ordered-logit backend params: {params_path}')
    vars=params.get('selected_variables') or []
    std=params.get('standardization_params') or {}; coef_records=params.get('coefficients') or []
    coef={str(r.get('variable')): float(r.get('coefficient')) for r in coef_records if str(r.get('variable')) in vars and r.get('coefficient') is not None}
    missing=[v for v in vars if v not in df.columns or v not in std or v not in coef]
    if missing: raise KeyError(f'Beta ordered-logit scoring missing exported feature inputs: {missing[:20]}')

    grade_nums = [int(x) for x in (params.get('modeled_grade_nums') or params.get('probability_output_grade_nums') or [])]
    if len(grade_nums) < 2:
        raise ValueError('Beta ordered-logit params missing modeled_grade_nums/probability_output_grade_nums')

    finite_cutpoints = params.get('ordered_logit_finite_cutpoints') or params.get('finite_cutpoints')
    if finite_cutpoints is None:
        raw_thresholds = params.get('ordered_logit_threshold_raw_params') or []
        if isinstance(raw_thresholds, list) and raw_thresholds and isinstance(raw_thresholds[0], dict):
            raw_thresholds = [r.get('coefficient') for r in raw_thresholds]
        if not raw_thresholds:
            # Backward-compatible fallback for older beta params: threshold rows are
            # the coefficient records whose variable names are not selected features.
            raw_thresholds = [r.get('coefficient') for r in coef_records if str(r.get('variable')) not in set(vars) and r.get('coefficient') is not None]
        finite_cutpoints = _transform_ordered_logit_thresholds([float(x) for x in raw_thresholds]).tolist()

    finite_cutpoints = np.asarray(finite_cutpoints, dtype=float)
    if len(finite_cutpoints) != len(grade_nums) - 1:
        raise ValueError(f'Beta cutpoint/class mismatch: {len(finite_cutpoints)} cutpoints for {len(grade_nums)} classes')

    xb=np.zeros(len(df), dtype=float)
    for v in vars:
        mean=float(std[v].get('mean',0.0)); scale=float(std[v].get('std',1.0)) or 1.0
        x=pd.to_numeric(df[v], errors='coerce').fillna(mean)
        xb += ((x-mean)/scale).to_numpy(dtype=float) * coef[v]

    thresholds = np.concatenate([[-np.inf], finite_cutpoints, [np.inf]])
    upper = _logistic_cdf(thresholds[1:][None, :] - xb[:, None])
    lower = _logistic_cdf(thresholds[:-1][None, :] - xb[:, None])
    upper[:, -1] = 1.0
    lower[:, 0] = 0.0
    probs = np.clip(upper - lower, 0.0, 1.0)
    denom = probs.sum(axis=1, keepdims=True)
    probs = np.divide(probs, denom, out=np.full_like(probs, 1.0 / probs.shape[1]), where=denom > 1e-12)
    expected_rating = probs @ np.asarray(grade_nums, dtype=float)
    score = np.clip(100.0 * (10.0 - expected_rating) / 9.0, 0.0, 100.0)
    return pd.Series(score, index=df.index, name='R_score_beta')

def score_gamma_model(df: pd.DataFrame, params_path: Path, model_path: Path) -> pd.Series:
    params=_load_json_artifact_cached(str(Path(params_path).resolve()))
    vars=params.get('selected_variables') or []
    if not vars: raise ValueError('Gamma params missing selected_variables')
    missing=[v for v in vars if v not in df.columns]
    if missing: raise KeyError(f'Gamma model scoring missing features: {missing[:20]}')
    model=_load_gamma_model_cached(str(Path(model_path).resolve()))
    pred=np.asarray(model.predict(df[vars].apply(pd.to_numeric, errors='coerce')), dtype=float)
    return pd.Series((100.0*(1.0-(pred-1.0)/9.0)).clip(0,100), index=df.index, name='R_score_gamma')

def summarize(scores: pd.DataFrame, backend: str, out: Path) -> None:
    score_col=f'R_score_{backend}'
    scores.groupby('policy')[score_col].agg(['count','mean','median','std']).reset_index().to_csv(out/f'summary_by_policy_{backend}.csv', index=False, encoding='utf-8-sig')
    piv=scores.pivot_table(index='row_id', columns='policy', values=score_col, aggfunc='first')
    rows=[]
    for ref in ['C0_noop','C2_weakest_component_rule']:
        if ref in piv.columns:
            for pol in piv.columns:
                if pol==ref: continue
                d=(piv[pol]-piv[ref]).dropna()
                if len(d): rows.append({'policy':pol,'reference':ref,'n':len(d),'mean_diff':float(d.mean()),'median_diff':float(d.median())})
    pd.DataFrame(rows).to_csv(out/f'paired_tests_{backend}.csv', index=False, encoding='utf-8-sig')

def rl_vs_c2_geometry(policy_actions: pd.DataFrame, space, out: Path) -> None:
    cols=space.columns; rl=policy_actions[policy_actions['policy']==space.final_rl_label].sort_values('row_id'); c2=policy_actions[policy_actions['policy']=='C2_weakest_component_rule'].sort_values('row_id')
    if rl.empty or c2.empty: raise ValueError('Missing RL or C2 policy actions for geometry audit')
    widths=np.array([space.bound_width(c) for c in cols], dtype=float); A=rl[cols].to_numpy(dtype=float)/widths; B=c2[cols].to_numpy(dtype=float)/widths
    denom=np.linalg.norm(A,axis=1)*np.linalg.norm(B,axis=1); cos=np.divide((A*B).sum(axis=1), denom, out=np.zeros_like(denom), where=denom>1e-12)
    pd.DataFrame({'row_id':rl['row_id'].to_numpy(), 'cosine_rl_c2':cos}).to_csv(out/'policy_geometry_rl_vs_c2.csv', index=False)
    pd.DataFrame({'metric':['cosine_mean','cosine_median'], 'value':[float(np.mean(cos)), float(np.median(cos))]}).to_csv(out/'rl_vs_c2_action_geometry.csv', index=False)


def validate_noop_pairing(policy_actions: pd.DataFrame) -> dict[str, Any]:
    if "row_id" not in policy_actions.columns or "policy" not in policy_actions.columns:
        raise ValueError("policy_actions must contain row_id and policy")
    all_ids = set(pd.to_numeric(policy_actions["row_id"], errors="raise").astype(int).tolist())
    noop_ids = set(pd.to_numeric(policy_actions.loc[policy_actions["policy"].astype(str) == "C0_noop", "row_id"], errors="raise").astype(int).tolist())
    missing = sorted(all_ids - noop_ids)
    if missing:
        raise ValueError(f"Stage6 no-op pairing violation: missing C0_noop for row_ids sample={missing[:20]} count={len(missing)}")
    c_obs_present = "C_obs" in set(policy_actions["policy"].astype(str))
    return {"status": "PASS", "n_row_ids": len(all_ids), "n_noop_row_ids": len(noop_ids), "c_obs_present": c_obs_present, "c_obs_policy": "secondary_inner_dev_only"}


def add_noop_deltas(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    for backend in ["alpha", "beta", "gamma"]:
        col = f"R_score_{backend}"
        if col not in out.columns:
            continue
        base = out[out["policy"].astype(str) == "C0_noop"].set_index("row_id")[col].to_dict()
        dcol = f"delta_R_score_{backend}"
        out[dcol] = pd.to_numeric(out[col], errors="coerce") - out["row_id"].map(base).astype(float)
        if out[dcol].isna().any():
            bad = out.loc[out[dcol].isna(), ["row_id", "policy"]].head(20).to_dict("records")
            raise ValueError(f"Failed to compute {dcol}; likely missing no-op pair or score. sample={bad}")
    return out


def write_final_policy_summary(merged: pd.DataFrame, out: Path, *, deploy_qargmax_as_policy: bool = False, qargmax_policy_name: str = "C3_candidate_iql_q_argmax") -> None:
    rows=[]
    for pol, g in merged.groupby("policy", dropna=False):
        row={"policy": str(pol), "n": int(len(g))}
        row["headline_policy"] = bool(deploy_qargmax_as_policy and str(pol) == qargmax_policy_name)
        row["headline_policy_reason"] = "deployed_iql_critic_q_argmax_no_oracle_selection" if row["headline_policy"] else ""
        for backend in ["alpha", "beta", "gamma"]:
            rcol=f"R_score_{backend}"; dcol=f"delta_R_score_{backend}"
            if rcol in g.columns:
                row[f"mean_{rcol}"]=float(pd.to_numeric(g[rcol], errors="coerce").mean())
            if dcol in g.columns:
                row[f"mean_{dcol}"]=float(pd.to_numeric(g[dcol], errors="coerce").mean())
                row[f"median_{dcol}"]=float(pd.to_numeric(g[dcol], errors="coerce").median())
                row[f"positive_share_{dcol}"]=float((pd.to_numeric(g[dcol], errors="coerce")>0).mean())
        rows.append(row)
    pd.DataFrame(rows).sort_values("policy").to_csv(out/"final_policy_summary.csv", index=False, encoding="utf-8-sig")


def rewrite_backend_summaries_from_merged(merged: pd.DataFrame, out: Path) -> None:
    """Rewrite backend summaries/paired tests after derived diagnostics are appended.

    Stage6 first scores real simulator policy rows. Derived diagnostics, such as
    IQL critic-Q rerank@k and Q-argmax, select actions without using oracle
    scores and are evaluated by copying the already-scored fixed-candidate rows
    after selection. They must be summarized after being appended to the merged
    score table. This function intentionally reuses the same summary/paired-test
    schema as the raw backend scoring outputs.
    """
    for backend in ["alpha", "beta", "gamma"]:
        col = f"R_score_{backend}"
        if col not in merged.columns:
            continue
        scores = merged[["row_id", "policy", "candidate_id", col]].dropna(subset=[col]).copy()
        summarize(scores, backend, out)



def append_iql_q_diagnostics(merged: pd.DataFrame, out: Path, space, ks: tuple[int, ...] = (3, 5, 7, 9)) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append RL-internal IQL critic diagnostics: Q-rerank@k and Q-argmax.

    No Alpha/Beta/Gamma oracle score is used to select actions. The Stage6
    selector writes actor probabilities and critic q_min=min(q1,q2) for the
    same evaluation states before simulator/oracle scoring begins.

    Derived policies:
    - C3_candidate_iql_q_rerank_at_k: actor top-k pool, choose max q_min.
    - C3_candidate_iql_q_argmax: ignore actor pool, choose max q_min across the
      active action vocabulary. This is a critic-only deployable diagnostic; it
      can be more OOD-prone than actor top-k, so it must be reported separately.
    """
    prob_path = out / "candidate_probabilities.parquet"
    q_path = out / "candidate_q_values.parquet"
    if not prob_path.exists():
        raise FileNotFoundError(f"Missing Stage6 actor probabilities required for IQL Q diagnostics: {prob_path}")
    if not q_path.exists():
        raise FileNotFoundError(f"Missing Stage6 critic Q-values required for IQL Q diagnostics: {q_path}")
    probs = pd.read_parquet(prob_path)
    qvals = pd.read_parquet(q_path)
    prob_cols = [c for c in probs.columns if str(c).startswith("prob__")]
    q_cols = [c for c in qvals.columns if str(c).startswith("q_min__")]
    if not prob_cols:
        raise ValueError("candidate_probabilities.parquet has no prob__* columns")
    if not q_cols:
        raise ValueError("candidate_q_values.parquet has no q_min__* columns")
    prob_names = [c.replace("prob__", "") for c in prob_cols]
    q_names = [c.replace("q_min__", "") for c in q_cols]
    if prob_names != q_names:
        raise ValueError({
            "message": "Stage6 Q diagnostic candidate order mismatch between probabilities and q-values",
            "probability_candidates": prob_names,
            "q_value_candidates": q_names,
        })

    source = merged.copy()
    source["policy"] = source["policy"].astype(str)
    source["candidate_id"] = source["candidate_id"].astype(str)
    candidate_source = source[source["policy"].isin(list(space.train_labels)) | source["policy"].eq("C0_noop")].copy()
    candidate_source = candidate_source.drop_duplicates(["row_id", "candidate_id"], keep="first")
    if candidate_source.empty:
        raise ValueError("No fixed-candidate source rows available for IQL Q diagnostic evaluation")

    row_ids = sorted(pd.to_numeric(source["row_id"], errors="raise").astype(int).unique().tolist())
    if len(probs) != len(row_ids) or len(qvals) != len(row_ids):
        raise ValueError({
            "message": "Stage6 Q diagnostic row count mismatch",
            "probability_rows": int(len(probs)),
            "q_value_rows": int(len(qvals)),
            "score_row_ids": int(len(row_ids)),
        })

    score_cols = [c for c in ["R_score_alpha", "R_score_beta", "R_score_gamma"] if c in merged.columns]
    lookup = candidate_source.set_index(["row_id", "candidate_id"])
    diag_rows: list[dict[str, Any]] = []
    derived_rows: list[dict[str, Any]] = []
    candidate_names = {c: c.replace("prob__", "") for c in prob_cols}

    def append_selected(row_id: int, policy: str, cid: str, base: dict[str, Any]) -> None:
        key = (int(row_id), str(cid))
        if key not in lookup.index:
            raise ValueError(f"Missing scored candidate row for Q diagnostic evaluation: row_id={row_id} candidate_id={cid}")
        src = lookup.loc[key]
        if isinstance(src, pd.DataFrame):
            src = src.iloc[0]
        out_row: dict[str, Any] = {
            "row_id": int(row_id),
            "policy": policy,
            "candidate_id": str(cid),
            **base,
        }
        for col in score_cols:
            out_row[col] = float(src[col])
        derived_rows.append(out_row)
        diag_rows.append(out_row.copy())

    for row_pos, row_id in enumerate(row_ids):
        p_row = probs.iloc[row_pos][prob_cols].astype(float)
        q_row = qvals.iloc[row_pos][q_cols].astype(float)
        ranked_cols = list(p_row.sort_values(ascending=False).index)

        # Actor top-k proposal + critic q_min rerank.
        for k in ks:
            top_cols = ranked_cols[:min(int(k), len(ranked_cols))]
            best: dict[str, Any] | None = None
            for rank, prob_col in enumerate(top_cols, start=1):
                cid = candidate_names[prob_col]
                q_col = f"q_min__{cid}"
                q_value = float(q_row[q_col])
                rec = {
                    "rerank_k": int(k),
                    "candidate_id": cid,
                    "actor_rank": int(rank),
                    "actor_probability": float(p_row[prob_col]),
                    "critic_q_min": q_value,
                }
                if best is None or q_value > float(best["critic_q_min"]):
                    best = rec
            if best is None:
                raise ValueError(f"No candidates available for Q-rerank row_id={row_id} k={k}")
            append_selected(
                int(row_id),
                f"{space.final_rl_label}_q_rerank_at_{int(k)}",
                str(best["candidate_id"]),
                {
                    "derived_policy_type": "iql_critic_q_rerank_at_k_no_oracle_selection",
                    "rerank_k": int(k),
                    "actor_rank": int(best["actor_rank"]),
                    "actor_probability": float(best["actor_probability"]),
                    "critic_q_min": float(best["critic_q_min"]),
                },
            )

        # Critic-only q_min argmax over the full active action vocabulary.
        best_q_col = str(q_row.idxmax())
        best_cid = best_q_col.replace("q_min__", "")
        actor_rank_map = {candidate_names[col]: rank for rank, col in enumerate(ranked_cols, start=1)}
        append_selected(
            int(row_id),
            f"{space.final_rl_label}_q_argmax",
            best_cid,
            {
                "derived_policy_type": "iql_critic_q_argmax_no_oracle_selection",
                "rerank_k": 0,
                "actor_rank": int(actor_rank_map.get(best_cid, -1)),
                "actor_probability": float(p_row[f"prob__{best_cid}"]),
                "critic_q_min": float(q_row[best_q_col]),
                "q_argmax_scope": "all_active_iql_action_vocabulary_candidates",
            },
        )

    diag = pd.DataFrame(diag_rows)
    diag.to_csv(out / "iql_q_diagnostic_selection.csv", index=False, encoding="utf-8-sig")
    if not diag.empty:
        dist = (
            diag.groupby(["policy", "candidate_id"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["policy", "count"], ascending=[True, False])
        )
        dist.to_csv(out / "iql_q_diagnostic_candidate_distribution.csv", index=False, encoding="utf-8-sig")
        # Backward-compatible aliases for the already-introduced Q-rerank artifacts.
        diag[diag["policy"].astype(str).str.contains("_q_rerank_at_", regex=False)].to_csv(
            out / "iql_q_rerank_selection_diagnostics.csv", index=False, encoding="utf-8-sig"
        )
        dist[dist["policy"].astype(str).str.contains("_q_rerank_at_", regex=False)].to_csv(
            out / "iql_q_rerank_candidate_distribution.csv", index=False, encoding="utf-8-sig"
        )

    derived = pd.DataFrame(derived_rows)
    result = pd.concat([merged, derived], ignore_index=True, sort=False) if not derived.empty else merged
    payload = {
        "status": "PASS",
        "schema_version": "stage6_iql_q_diagnostic_v2_rerank_and_argmax_k4",
        "policy_type": "rl_internal_critic_q_diagnostics_no_oracle_selection",
        "deployability": "selection_requires_encoder_critic_and_candidate_library_only; oracle_scores_used_only_for_posthoc_evaluation",
        "ks": [int(k) for k in ks],
        "q_value_basis": "candidate_q_values.parquet q_min__* from Stage5 IQL critic min(q1,q2)",
        "actor_pool_basis": "candidate_probabilities.parquet prob__* top-k pool for q_rerank only; q_argmax ignores actor pool",
        "selection_rule_q_rerank": "within actor top-k candidates, choose max IQL critic q_min; do not use Alpha/Beta/Gamma oracle scores for selection",
        "selection_rule_q_argmax": "choose max IQL critic q_min over all active action-vocabulary candidates; do not use Alpha/Beta/Gamma oracle scores for selection",
        "diagnostic_file": "iql_q_diagnostic_selection.csv",
        "distribution_file": "iql_q_diagnostic_candidate_distribution.csv",
        "q_rerank_compat_files": ["iql_q_rerank_selection_diagnostics.csv", "iql_q_rerank_candidate_distribution.csv"],
        "derived_policies": sorted(derived["policy"].astype(str).unique().tolist()) if not derived.empty else [],
        "n_rows_per_policy": int(len(row_ids)),
    }
    write_json(out / "iql_q_diagnostic_metadata.json", payload)
    # Backward-compatible metadata alias for the already-introduced Q-rerank artifact.
    write_json(out / "iql_q_rerank_diagnostic_metadata.json", payload)
    return result, payload


def write_variable_supply_manifest(sim_state: pd.DataFrame, backends: dict[str, Any], root: Path, final: Path, out: Path) -> None:
    manifest={"schema_version":"stage6_variable_supply_manifest_v1", "missing_required_variables_by_backend":{}, "required_variables_by_backend":{}}
    for bk in ["alpha","beta","gamma"]:
        params=resolve_backend_artifact(root, final, backends.get(bk,{}).get("params", ""))
        required=[]
        if params.exists():
            try:
                payload=json.loads(params.read_text(encoding="utf-8"))
                required = payload.get("selected_variables") or [v.get("variable_id") for v in payload.get("variables",[]) if isinstance(v,dict) and v.get("variable_id")]
            except Exception:
                required=[]
        required=[str(v) for v in required if v]
        manifest["required_variables_by_backend"][bk]=required
        manifest["missing_required_variables_by_backend"][bk]=[v for v in required if v not in sim_state.columns]
    manifest["status"]="PASS" if not any(manifest["missing_required_variables_by_backend"].values()) else "FAIL"
    write_json(out/"variable_supply_manifest.json", manifest)
    if manifest["status"] != "PASS":
        raise ValueError(f"Stage6 variable supply manifest failed: {manifest['missing_required_variables_by_backend']}")


def mirror_stage6_outputs(selector_out: Path, mirror_out: Path) -> None:
    mirror_out.mkdir(parents=True, exist_ok=True)
    for name in [
        "multi_oracle_policy_eval.parquet", "final_policy_summary.csv", "variable_supply_manifest.json",
        "simulated_oracle_input_frame.parquet", "action_effect_audit.parquet",
        "multi_oracle_metadata.json", "backend_scoring_equivalence_report.json", "oracle_variable_formula_audit.json",
        "rl_vs_c2_action_geometry.csv", "policy_geometry_rl_vs_c2.csv",
        "summary_by_policy_alpha.csv", "summary_by_policy_beta.csv", "summary_by_policy_gamma.csv",
        "paired_tests_alpha.csv", "paired_tests_beta.csv", "paired_tests_gamma.csv",
        "iql_q_diagnostic_selection.csv", "iql_q_diagnostic_candidate_distribution.csv",
        "iql_q_diagnostic_metadata.json",
        "iql_q_rerank_selection_diagnostics.csv", "iql_q_rerank_candidate_distribution.csv",
        "iql_q_rerank_diagnostic_metadata.json",
    ]:
        src=selector_out/name
        if src.exists():
            (mirror_out/name).write_bytes(src.read_bytes())

def main(argv=None) -> int:
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root', required=True); ap.add_argument('--allow-unscored', action='store_true'); ap.add_argument('--preserve-current-non-current-residual', action='store_true'); ap.add_argument('--sim-business-plan-mode', choices=['default','calibrated'], default='default'); ap.add_argument('--deploy-qargmax-as-policy', action='store_true')
    args=ap.parse_args(argv); root=Path(args.project_root).resolve(); temporal_contract=load_temporal_contract(root); final=final_root(root); out=final/'stage6_candidate_selector_eval'; out.mkdir(parents=True, exist_ok=True)
    reg_path=final/'configs'/'oracle_backend_registry.yaml'
    try:
        reg=load_registry(reg_path); backends=reg.get('backends',{})
        if reg.get('final_result_allowed') is not True or reg.get('status') != 'generated_by_stage1_oracle_development_verified': raise ValueError('Oracle backend registry is not final/generated_by_stage1_oracle_development_verified')
        allowed_types={'alpha':'alpha_vanilla_isotonic_scorecard','beta':'beta_ordered_logit','gamma':'gamma_ml_tree_boosting'}
        for k,v in allowed_types.items():
            if backends.get(k,{}).get('backend_type') != v: raise ValueError(f'Unexpected backend_type for {k}: {backends.get(k,{}).get("backend_type")} expected {v}')
        pa=read_parquet_required(out/'policy_actions.parquet'); base=read_parquet_required(final/'stage2_candidate_projection'/'phase_eval_candidate.parquet')
        fy=pd.to_numeric(base.get('fiscal_year', base.get('year')), errors='coerce')
        observed_years=sorted([int(x) for x in fy.dropna().unique().tolist()])
        if observed_years != [int(temporal_contract.eval_base_year)]:
            raise ValueError({'message':'Stage6 multi-oracle phase_eval temporal contract failed','expected_eval_base_year':int(temporal_contract.eval_base_year),'observed_fiscal_years':observed_years})
        space=load_action_space(root)
        pairing_audit = validate_noop_pairing(pa)
        write_json(out/'policy_pairing_audit_runtime.json', pairing_audit)
        # Formula-registry audit: Stage6 simulator must not silently attach the
        # wrong formula to an R-code.  Check all selected variables exported by
        # backend params before scoring.
        formula_audits=[]
        for bk in ['alpha','beta','gamma']:
            bp = resolve_backend_artifact(root, final, backends.get(bk, {}).get('params', ''))
            if bp.exists():
                try:
                    bparams=json.loads(bp.read_text(encoding='utf-8'))
                except Exception:
                    bparams={}
                selected_records = bparams.get('variables') or bparams.get('selected_variables') or []
                audit = audit_formula_registry(selected_records)
                audit['backend'] = bk
                audit['params'] = str(bp)
                formula_audits.append(audit)
        write_json(out/'oracle_variable_formula_audit.json', {'status':'PASS' if all(a.get('status')=='PASS' for a in formula_audits) else 'FAIL', 'audits': formula_audits})
        if any(a.get('status') != 'PASS' for a in formula_audits):
            raise ValueError('Oracle variable formula audit failed; refusing to score with potentially mismapped R-code formulas. See oracle_variable_formula_audit.json')
        if space.final_rl_label not in set(pa['policy'].astype(str)): raise ValueError(f'Missing final RL label {space.final_rl_label}')
        if any(pa['policy'].astype(str).str.contains('H9', na=False)): raise ValueError('H9 policy found in Stage6 actions')
        predicted_fiscal_year = resolve_stage6_rollout_target_year(temporal_contract)
        if int(predicted_fiscal_year) <= int(temporal_contract.eval_base_year):
            raise ValueError({'message':'Stage6 rollout target year must be after eval_base_year', 'eval_base_year':int(temporal_contract.eval_base_year), 'predicted_fiscal_year':int(predicted_fiscal_year)})
        sim_state, audit=simulate_policy_states(base, pa, space, out, predicted_fiscal_year=int(predicted_fiscal_year), preserve_current_non_current_residual=args.preserve_current_non_current_residual, sim_business_plan_mode=args.sim_business_plan_mode)
        write_variable_supply_manifest(sim_state, backends, root, final, out)
        all_scores=[]
        backend_score_maps={}
        for backend in ['alpha','beta','gamma']:
            b=backends[backend]; params=resolve_backend_artifact(root, final, b.get('params',''))
            if not params.exists(): raise FileNotFoundError(f'Missing {backend} params: {params}')
            if backend=='alpha': score=score_alpha(sim_state, params); col='R_score_alpha'
            elif backend=='beta': score=score_beta_ordered_logit_params(sim_state, params); col='R_score_beta'
            else:
                model=resolve_backend_artifact(root, final, b.get('model',''))
                if not model.exists(): raise FileNotFoundError(f'Missing gamma model artifact: {model}')
                score=score_gamma_model(sim_state, params, model); col='R_score_gamma'
            scored=pa[['row_id','policy','candidate_id']+space.columns].copy(); scored[col]=score.to_numpy(); scored.to_parquet(out/f'oracle_scores_{backend}.parquet', index=False); summarize(scored, backend, out); all_scores.append(scored[['row_id','policy','candidate_id',col]])
            backend_score_maps[backend]=scored[['row_id','policy',col]].copy()
        merged=all_scores[0]
        for s in all_scores[1:]: merged=merged.merge(s,on=['row_id','policy','candidate_id'],how='outer')
        merged, iql_q_meta = append_iql_q_diagnostics(merged, out, space, ks=(3, 5, 7, 9))
        merged=add_noop_deltas(merged)
        merged.to_parquet(out/'multi_oracle_policy_eval.parquet', index=False)
        write_final_policy_summary(merged, out, deploy_qargmax_as_policy=bool(args.deploy_qargmax_as_policy), qargmax_policy_name=f"{space.final_rl_label}_q_argmax")
        if args.deploy_qargmax_as_policy:
            write_json(out/'headline_policy_metadata.json', {
                'status':'PASS',
                'headline_policy': f"{space.final_rl_label}_q_argmax",
                'selection_basis':'iql_critic_q_argmax_no_oracle_selection',
                'oracle_scores_used_for_selection': False,
                'source_file':'iql_q_diagnostic_selection.csv',
            })
        rewrite_backend_summaries_from_merged(merged, out)
        rl_vs_c2_geometry(pa, space, out)
        # update action-effect audit with per-backend score deltas versus C0_noop by row_id.
        if not audit.empty:
            expanded=[]
            for backend, sdf in backend_score_maps.items():
                col=f'R_score_{backend}'
                base_scores=sdf[sdf['policy']=='C0_noop'].set_index('row_id')[col].to_dict()
                pol_scores=sdf.set_index(['row_id','policy'])[col].to_dict()
                tmp=audit.copy()
                tmp['oracle_backend']=backend
                tmp['score_before']=tmp['row_id'].map(base_scores)
                tmp['score_after']=[pol_scores.get((rid,pol), np.nan) for rid,pol in zip(tmp['row_id'],tmp['policy'])]
                tmp['score_delta']=tmp['score_after']-tmp['score_before']
                expanded.append(tmp)
            audit=pd.concat(expanded, ignore_index=True)
            audit.to_parquet(out/'action_effect_audit.parquet', index=False)
        # scorer equivalence report: exact backend scoring functions are used; Stage1-vs-Stage6 row equivalence is checked when Stage1 benchmark rows overlap with simulated inputs.
        equiv={'status':'PASS_RUNTIME_SCORERS_USED','checked_backends':['alpha','beta','gamma'],'note':'Stage6 uses backend scorer functions/artifacts directly; row-level Stage1 equivalence requires overlapping benchmark rows and is audited during full data run.'}
        write_json(out/'backend_scoring_equivalence_report.json', equiv)
        try:
            from credit_recourse.eval.final_stage6_derived_comparators import derive_stage6_comparators
            derived = derive_stage6_comparators(root)
            write_json(out/'derived_comparators_runtime.json', derived)
        except Exception as exc_derived:
            raise ValueError(f'Stage6 derived comparator generation failed: {exc_derived}') from exc_derived
        meta={'stage':'final_stage6_multi_oracle_eval','status':'PASS_SCORED_REAL_BACKENDS_FULL_SIMULATOR','created_utc':now(),'unscored_pending':False,'final_result_allowed':True,'registry':str(reg_path),'no_placeholder_scoring':True,'generic_proxy_scoring':False,'intervention_mode':'full_financial_statement_simulator','full_statement_simulator_used':True,'preserve_current_non_current_residual':bool(args.preserve_current_non_current_residual),'sim_business_plan_mode':args.sim_business_plan_mode,'deploy_qargmax_as_policy':bool(args.deploy_qargmax_as_policy),'oracle_proxy_scoring':False,'simulator_package':'credit_recourse.simulator.FinancialSimulator','eval_base_year':int(temporal_contract.eval_base_year),'predicted_fiscal_year':int(predicted_fiscal_year),'rollout_target_year':int(predicted_fiscal_year),'stage6_temporal_rule':'select policy on eval_base_year state; simulate one-year rollout and score predicted fiscal year','backend_scoring_equivalence_report':'backend_scoring_equivalence_report.json','oracle_variable_formula_audit':'oracle_variable_formula_audit.json','derived_comparators':'data/final_freeze/ledgers/stage6_derived_comparators.json','simulated_oracle_input_frame':'simulated_oracle_input_frame.parquet','iql_q_diagnostic':iql_q_meta}
    except Exception as exc:
        if not args.allow_unscored: raise
        meta={'stage':'final_stage6_multi_oracle_eval','status':'UNSCORED_PENDING_BACKENDS','created_utc':now(),'unscored_pending':True,'final_result_allowed':False,'error':repr(exc),'no_placeholder_scoring':True,'generic_proxy_scoring':False,'intervention_mode':'full_financial_statement_simulator','full_statement_simulator_used':True,'preserve_current_non_current_residual':bool(args.preserve_current_non_current_residual),'sim_business_plan_mode':args.sim_business_plan_mode,'deploy_qargmax_as_policy':bool(args.deploy_qargmax_as_policy),'oracle_proxy_scoring':False}
    write_json(out/'multi_oracle_metadata.json', meta)
    mirror_stage6_outputs(out, final/'stage6_multi_oracle_eval')
    print(json.dumps(meta,ensure_ascii=False,indent=2)); return 0 if meta.get('final_result_allowed') else 2
if __name__=='__main__': raise SystemExit(main())

