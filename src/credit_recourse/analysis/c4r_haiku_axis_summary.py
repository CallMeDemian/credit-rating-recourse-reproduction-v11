"""Summarize Haiku thinking/non-thinking 0.75 axis-swap outputs."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
SCHEMA_VERSION = 'c4r_haiku_axis_summary_v1'
EXPECTED_AXES = 10
EXPECTED_ORACLES = {'alpha', 'beta', 'gamma'}
EXPECTED_ARM = '0p75'

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse_spec(spec: str) -> tuple[str, str, Path]:
    left, sep, raw = str(spec).partition('=')
    if sep != '=' or not left.strip() or (not raw.strip()):
        raise ValueError(f'--input must use protocol|policy_pair=axis_output_dir; got {spec!r}')
    protocol, sep2, pair = left.partition('|')
    if sep2 != '|' or not protocol.strip() or (not pair.strip()):
        raise ValueError(f'--input must use protocol|policy_pair=axis_output_dir; got {spec!r}')
    return (protocol.strip(), pair.strip(), Path(raw.strip()).resolve())

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

def run(*, inputs: list[str], out: Path, expected_n: int) -> dict:
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    oat_frames: list[pd.DataFrame] = []
    shap_frames: list[pd.DataFrame] = []
    sources: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for spec in inputs:
        protocol, pair, root = _parse_spec(spec)
        key = (protocol, pair)
        if key in seen:
            raise ValueError(f'duplicate input: {key}')
        seen.add(key)
        manifest_path = root / 'intervention_manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        recorded_pair = f"{manifest.get('base_policy')}_to_{manifest.get('target_policy')}"
        if recorded_pair != pair:
            raise ValueError(f'policy-pair mismatch: spec={pair}, manifest={recorded_pair}')
        if int(manifest.get('expected_n', -1)) != expected_n:
            raise ValueError(f'{protocol}/{pair}: expected_n mismatch in manifest')
        arms = manifest.get('arms') or {}
        if set(arms) != {EXPECTED_ARM}:
            raise ValueError(f'{protocol}/{pair}: expected only {EXPECTED_ARM}; got {sorted(arms)}')
        if int(arms[EXPECTED_ARM].get('n_firms', -1)) != expected_n:
            raise ValueError(f'{protocol}/{pair}: arm n_firms mismatch')
        arm_dir = root / EXPECTED_ARM
        oat_path = arm_dir / 'oat_marginals.parquet'
        shap_path = arm_dir / 'shapley_phi.parquet'
        if not oat_path.is_file() or not shap_path.is_file():
            raise FileNotFoundError(f'missing OAT/Shapley outputs under {arm_dir}')
        oat = pd.read_parquet(oat_path)
        shap = pd.read_parquet(shap_path)
        for name, frame in (('OAT', oat), ('Shapley', shap)):
            if frame['axis'].nunique() != EXPECTED_AXES:
                raise ValueError(f'{protocol}/{pair}: malformed {name} axis universe')
            if set(frame['oracle'].astype(str)) != EXPECTED_ORACLES:
                raise ValueError(f'{protocol}/{pair}: malformed {name} oracle universe')
            if frame['row_id'].nunique() != expected_n:
                raise ValueError(f'{protocol}/{pair}: malformed {name} firm count')
        osum = _summarize_oat(oat)
        osum.insert(0, 'policy_pair', pair)
        osum.insert(0, 'protocol', protocol)
        oat_frames.append(osum)
        ssum = _summarize_shapley(shap)
        ssum.insert(0, 'policy_pair', pair)
        ssum.insert(0, 'protocol', protocol)
        shap_frames.append(ssum)
        sources.append({'protocol': protocol, 'policy_pair': pair, 'manifest_path': str(manifest_path), 'oat_path': str(oat_path), 'shapley_path': str(shap_path)})
    expected_keys = {('nonthinking', 'C4_to_C4R'), ('nonthinking', 'C4R_to_C6'), ('thinking', 'C4_to_C4R'), ('thinking', 'C4R_to_C6')}
    if seen != expected_keys:
        raise ValueError(f'expected exact 2x2 protocol/pair grid; got {sorted(seen)}')
    oat_all = pd.concat(oat_frames, ignore_index=True)
    shap_all = pd.concat(shap_frames, ignore_index=True)
    alpha_ranked = shap_all[shap_all['oracle'].astype(str).eq('alpha')].copy()
    alpha_ranked['rank_abs_mean'] = alpha_ranked.groupby(['protocol', 'policy_pair'])['mean_abs'].rank(method='first', ascending=False).astype(int)
    alpha_ranked = alpha_ranked.sort_values(['protocol', 'policy_pair', 'rank_abs_mean'])
    protocol_diff = shap_all[shap_all['oracle'].astype(str).eq('alpha')].pivot_table(index=['policy_pair', 'axis'], columns='protocol', values='mean', aggfunc='first').reset_index()
    if not {'thinking', 'nonthinking'}.issubset(protocol_diff.columns):
        raise ValueError('protocol-difference table lacks thinking/nonthinking columns')
    protocol_diff['thinking_minus_nonthinking_mean_phi'] = protocol_diff['thinking'] - protocol_diff['nonthinking']
    oat_out = out / 'haiku_axis_oat_summary.csv'
    shap_out = out / 'haiku_axis_shapley_summary.csv'
    rank_out = out / 'haiku_axis_alpha_ranked.csv'
    diff_out = out / 'haiku_axis_alpha_protocol_difference.csv'
    oat_all.to_csv(oat_out, index=False, encoding='utf-8-sig')
    shap_all.to_csv(shap_out, index=False, encoding='utf-8-sig')
    alpha_ranked.to_csv(rank_out, index=False, encoding='utf-8-sig')
    protocol_diff.to_csv(diff_out, index=False, encoding='utf-8-sig')
    meta = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'created_utc': _now(), 'expected_n': expected_n, 'input_count': len(inputs), 'sources': sources, 'outputs': {oat_out.name: '0', shap_out.name: '0', rank_out.name: '0', diff_out.name: '0'}, 'evidence_tier': 'EXPLORATORY_COMPLETE_CASE_EVALUATOR_ONLY', 'interpretation_boundary': 'Both Haiku protocols are decomposed on the same 571-firm complete-case cohort. The non-thinking run failed the provider/translation and raw-budget feasibility gates, and the thinking run failed the preregistered raw-budget QC. Therefore these outputs characterize mechanisms of the realized recommendations; they are not confirmatory contract-compliant model rankings.'}
    (out / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Summarize Haiku thinking/non-thinking axis decomposition')
    ap.add_argument('--input', action='append', required=True, help='protocol|Base_to_Target=axis_output_dir')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--expected-n', type=int, default=571)
    args = ap.parse_args(argv)
    meta = run(inputs=args.input, out=args.out, expected_n=args.expected_n)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
