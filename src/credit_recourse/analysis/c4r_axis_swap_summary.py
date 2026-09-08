"""Combine C4R axis-swap OAT/Shapley outputs into journal-ready ledgers."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
SCHEMA_VERSION = 'c4r_axis_swap_summary_v1'
EXPECTED_AXES = 10
EXPECTED_ORACLES = {'alpha', 'beta', 'gamma'}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse_spec(spec: str) -> tuple[str, str, Path]:
    left, sep, raw_path = str(spec).partition('=')
    if sep != '=' or not left.strip() or (not raw_path.strip()):
        raise ValueError(f'--input must use cohort|policy_pair=axis_output_dir; got {spec!r}')
    cohort, sep2, pair = left.partition('|')
    if sep2 != '|' or not cohort.strip() or (not pair.strip()):
        raise ValueError(f'--input must use cohort|policy_pair=axis_output_dir; got {spec!r}')
    return (cohort.strip(), pair.strip(), Path(raw_path.strip()).resolve())

def _summarize_oat(df: pd.DataFrame) -> pd.DataFrame:
    needed = {'row_id', 'axis', 'oracle', 'oat_marginal'}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f'OAT table missing columns: {missing}')
    return df.groupby(['axis', 'oracle'], as_index=False)['oat_marginal'].agg(n_firms='count', mean='mean', median='median', std='std')

def _summarize_shapley(df: pd.DataFrame) -> pd.DataFrame:
    needed = {'row_id', 'axis', 'oracle', 'phi'}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f'Shapley table missing columns: {missing}')
    out = df.groupby(['axis', 'oracle'], as_index=False)['phi'].agg(n_firms='count', mean='mean', median='median', std='std')
    abs_mean = df.assign(abs_phi=pd.to_numeric(df['phi'], errors='coerce').abs()).groupby(['axis', 'oracle'], as_index=False)['abs_phi'].mean().rename(columns={'abs_phi': 'mean_abs'})
    return out.merge(abs_mean, on=['axis', 'oracle'], validate='one_to_one')

def run(*, inputs: list[str], out: Path) -> dict:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    oat_frames: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for spec in inputs:
        cohort, pair, root = _parse_spec(spec)
        key = (cohort, pair)
        if key in seen:
            raise ValueError(f'Duplicate axis input: {key}')
        seen.add(key)
        manifest_path = root / 'intervention_manifest.json'
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest.get('base_policy') is None or manifest.get('target_policy') is None:
            raise ValueError(f'Axis manifest lacks policy pair: {manifest_path}')
        policy_pair_recorded = f"{manifest['base_policy']}_to_{manifest['target_policy']}"
        if policy_pair_recorded != pair:
            raise ValueError(f'Policy-pair mismatch: spec={pair}, manifest={policy_pair_recorded}, path={root}')
        arm_labels = sorted((manifest.get('arms') or {}).keys())
        if set(arm_labels) != {'0p75', 'unbounded'}:
            raise ValueError(f'{cohort}/{pair}: expected arms 0p75,unbounded; got {arm_labels}')
        for budget in arm_labels:
            arm_dir = root / budget
            oat_path = arm_dir / 'oat_marginals.parquet'
            shap_path = arm_dir / 'shapley_phi.parquet'
            if not oat_path.exists() or not shap_path.exists():
                raise FileNotFoundError(f'Missing OAT/Shapley output under {arm_dir}')
            oat = pd.read_parquet(oat_path)
            shap = pd.read_parquet(shap_path)
            if oat['axis'].nunique() != EXPECTED_AXES or set(oat['oracle'].astype(str)) != EXPECTED_ORACLES:
                raise ValueError(f'{cohort}/{pair}/{budget}: malformed OAT axis/oracle universe')
            if shap['axis'].nunique() != EXPECTED_AXES or set(shap['oracle'].astype(str)) != EXPECTED_ORACLES:
                raise ValueError(f'{cohort}/{pair}/{budget}: malformed Shapley axis/oracle universe')
            osum = _summarize_oat(oat)
            osum.insert(0, 'budget_label', budget)
            osum.insert(0, 'policy_pair', pair)
            osum.insert(0, 'cohort_id', cohort)
            oat_frames.append(osum)
            ssum = _summarize_shapley(shap)
            ssum.insert(0, 'budget_label', budget)
            ssum.insert(0, 'policy_pair', pair)
            ssum.insert(0, 'cohort_id', cohort)
            shap_frames.append(ssum)
            sources.append({'cohort_id': cohort, 'policy_pair': pair, 'budget_label': budget, 'oat_path': str(oat_path), 'shapley_path': str(shap_path)})
    oat_all = pd.concat(oat_frames, ignore_index=True)
    shap_all = pd.concat(shap_frames, ignore_index=True)
    alpha_ranked = shap_all[shap_all['oracle'].astype(str).eq('alpha')].copy()
    alpha_ranked['rank_abs_mean'] = alpha_ranked.groupby(['cohort_id', 'policy_pair', 'budget_label'])['mean_abs'].rank(method='first', ascending=False).astype(int)
    alpha_ranked = alpha_ranked.sort_values(['cohort_id', 'policy_pair', 'budget_label', 'rank_abs_mean'])
    cross_backend = shap_all[shap_all['oracle'].astype(str).eq('alpha')].pivot_table(index=['policy_pair', 'budget_label', 'axis'], columns='cohort_id', values='mean', aggfunc='first').reset_index()
    if {'gpt54mini', 'gemini31flashlite'}.issubset(cross_backend.columns):
        cross_backend['gemini_minus_gpt_mean_phi'] = cross_backend['gemini31flashlite'] - cross_backend['gpt54mini']
    oat_path = out / 'c4r_axis_oat_summary.csv'
    shap_path = out / 'c4r_axis_shapley_summary.csv'
    rank_path = out / 'c4r_axis_alpha_ranked.csv'
    cross_path = out / 'c4r_axis_alpha_backend_difference.csv'
    oat_all.to_csv(oat_path, index=False, encoding='utf-8-sig')
    shap_all.to_csv(shap_path, index=False, encoding='utf-8-sig')
    alpha_ranked.to_csv(rank_path, index=False, encoding='utf-8-sig')
    cross_backend.to_csv(cross_path, index=False, encoding='utf-8-sig')
    meta = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'created_utc': _now(), 'input_count': len(inputs), 'sources': sources, 'oat_summary_rows': int(len(oat_all)), 'shapley_summary_rows': int(len(shap_all)), 'alpha_ranked_rows': int(len(alpha_ranked)), 'outputs': {oat_path.name: '0', shap_path.name: '0', rank_path.name: '0', cross_path.name: '0'}, 'interpretation_boundary': 'Axis attributions are evaluator-only decompositions of the frozen base-to-target policy pair. OAT effects need not add because the simulator is nonlinear; Shapley values are used for efficient attribution.'}
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Summarize C4R axis-swap OAT/Shapley outputs')
    ap.add_argument('--input', action='append', required=True, help='cohort|Base_to_Target=axis_output_dir (repeatable)')
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args(argv)
    meta = run(inputs=args.input, out=args.out)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
