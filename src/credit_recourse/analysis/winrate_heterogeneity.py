from __future__ import annotations
'N3 win-rate and N6 residual-heterogeneity diagnostics against exact C3.\n\nThis module is read-only with respect to frozen Stage 7/8/9 artifacts.  It\ncomputes row-level policy win rates versus the exact canonical Stage 6 C3 label\n``C3_candidate_iql`` and, optionally, exploratory heterogeneity slices for C6\nfree-form residual gaps after joining a serving/evaluation panel.\n\nThe diagnostic is intentionally outside the canonical thesis pipeline.  It does\nnot call an LLM API, rerun a simulator, or retrain RL.  It fails fast on missing\nrow-level keys, duplicate exact C3 rows, ambiguous score columns, and missing\npanel join keys.\n'
import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from credit_recourse.rl.common.io import write_json
CANONICAL_C3_POLICY = 'C3_candidate_iql'
C3_DIAGNOSTIC_PREFIX = CANONICAL_C3_POLICY + '_'
ORACLES = ('alpha', 'beta', 'gamma')
ROW_KEY_CANDIDATES = ('row_id', 'firm_year_id', 'case_id')
COMPARISON_RELATIVE_CANDIDATES = ('stage9_policy_comparison/llm_stage9_llm_rl_comparison.parquet', 'stage9_policy_comparison/llm_stage9_llm_rl_comparison.csv', 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.parquet', 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.csv', 'stage9_policy_comparison/llm_stage9_paired_vs_C0_noop.parquet', 'stage9_policy_comparison/llm_stage9_paired_vs_C0_noop.csv', 'stage9_llm_rl_comparison/llm_stage9_paired_vs_C0_noop.parquet', 'stage9_llm_rl_comparison/llm_stage9_paired_vs_C0_noop.csv')
WINRATE_SCHEMA_VERSION = 'winrate_heterogeneity_v1'

@dataclass(frozen=True)
class RunInput:
    run_dir: Path
    comparison_file: Path
    run_label: str
    information_condition: str

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Input table not found: {path}')
    suf = path.suffix.lower()
    if suf == '.parquet':
        return pd.read_parquet(path)
    if suf == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table extension for {path}; expected .csv or .parquet')

def _find_first_existing(root: Path, rels: Iterable[str]) -> Path | None:
    for rel in rels:
        p = root / rel
        if p.exists() and p.is_file():
            return p
    return None

def find_comparison_file(run_dir_or_file: Path) -> Path:
    """Find a row-level Stage9 comparison table under a run directory.

    Both canonical archive names (``stage9_llm_rl_comparison``) and the N5 paper
    alias (``stage9_policy_comparison``) are supported.  The function returns a
    single best file or raises with the searched patterns.
    """
    p = Path(run_dir_or_file).resolve()
    if p.is_file():
        return p
    if not p.exists():
        raise FileNotFoundError(f'Run directory does not exist: {p}')
    exact = _find_first_existing(p, COMPARISON_RELATIVE_CANDIDATES)
    if exact is not None:
        return exact
    patterns = ('**/llm_stage9_llm_rl_comparison.parquet', '**/llm_stage9_llm_rl_comparison.csv', '**/llm_stage9_paired_vs_C0_noop.parquet', '**/llm_stage9_paired_vs_C0_noop.csv', '**/*llm_rl_comparison*.parquet', '**/*llm_rl_comparison*.csv')
    candidates: list[Path] = []
    for pat in patterns:
        candidates.extend(p.glob(pat))
    candidates = sorted(set(candidates), key=lambda x: (x.suffix.lower() != '.parquet', len(str(x)), str(x)))
    if candidates:
        return candidates[0]
    raise FileNotFoundError({'message': 'No row-level Stage9 comparison file found under run directory.', 'run_dir': str(p), 'exact_candidates': list(COMPARISON_RELATIVE_CANDIDATES), 'glob_patterns': list(patterns)})

def parse_information_condition(run_dir: Path, df: pd.DataFrame | None=None) -> str:
    if df is not None and 'information_condition' in df.columns:
        vals = [str(v) for v in df['information_condition'].dropna().unique().tolist()]
        if len(vals) == 1:
            return vals[0]
        if len(vals) > 1:
            return ';'.join(sorted(vals))
    s = Path(run_dir).name
    m = re.search('IC[-_]?([abcABC])', s)
    if m:
        return f'IC-{m.group(1).lower()}'
    return s

def _detect_row_key(df: pd.DataFrame, *, label: str) -> str:
    for col in ROW_KEY_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError({'message': f'{label} is missing a recognized row key.', 'accepted': list(ROW_KEY_CANDIDATES), 'available_columns_sample': list(df.columns)[:100]})

def detect_score_column(df: pd.DataFrame, oracle: str) -> str:
    candidates = (f'delta_R_score_{oracle}', f'gap_{oracle}', f'score_{oracle}')
    for col in candidates:
        if col in df.columns:
            return col
    aggregate = f'mean_delta_R_score_{oracle}'
    if aggregate in df.columns:
        raise ValueError(f'Input appears to be an aggregate summary because it contains {aggregate}; win-rate diagnostics require row-level policy scores.')
    raise ValueError({'message': f'Could not locate a row-level score column for oracle={oracle}.', 'accepted': list(candidates), 'available_columns_sample': list(df.columns)[:100]})

def _wilson(k: int, n: int, z: float=1.96) -> tuple[float, float, float]:
    if n <= 0:
        return (math.nan, math.nan, math.nan)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (float(p), float(center - half), float(center + half))

def _validate_policy_table(df: pd.DataFrame, *, source: Path) -> tuple[pd.DataFrame, str, dict[str, str]]:
    if 'policy' not in df.columns:
        raise ValueError(f'Stage9 comparison table missing policy column: {source}')
    row_key = _detect_row_key(df, label=str(source))
    score_cols = {oracle: detect_score_column(df, oracle) for oracle in ORACLES if any((c in df.columns for c in (f'delta_R_score_{oracle}', f'gap_{oracle}', f'score_{oracle}', f'mean_delta_R_score_{oracle}')))}
    if 'alpha' not in score_cols:
        raise ValueError(f'Stage9 comparison table must contain an alpha row-level score column: {source}')
    out = df.copy()
    if row_key != 'row_id':
        out = out.rename(columns={row_key: 'row_id'})
    out['policy'] = out['policy'].astype(str)
    return (out, row_key, score_cols)

def _reference_rows(df: pd.DataFrame, score_col: str, *, run_label: str) -> pd.DataFrame:
    exact = df[df['policy'].eq(CANONICAL_C3_POLICY)][['row_id', score_col]].copy()
    if exact.empty:
        c3_like = sorted(df.loc[df['policy'].str.startswith('C3', na=False), 'policy'].unique().tolist())
        raise ValueError({'message': f'Exact canonical C3 rows absent for run {run_label}; loose matching is forbidden.', 'required_policy': CANONICAL_C3_POLICY, 'c3_like_available': c3_like})
    dup = exact['row_id'].duplicated(keep=False)
    if dup.any():
        raise ValueError({'message': f'Exact canonical C3 has duplicate row_id values in run {run_label}.', 'duplicate_sample': exact.loc[dup, 'row_id'].head(20).tolist(), 'n_duplicate_rows': int(dup.sum())})
    return exact.rename(columns={score_col: 'c3_score'})

def _policy_mode_groups(df: pd.DataFrame) -> list[tuple[tuple[str, str], pd.DataFrame]]:
    if 'mode' in df.columns:
        grouped = df.groupby(['policy', 'mode'], dropna=False)
        return [((str(pol), str(mode)), g.copy()) for (pol, mode), g in grouped]
    grouped = df.groupby('policy', dropna=False)
    return [((str(pol), ''), g.copy()) for pol, g in grouped]

def _is_c3_diagnostic(policy: str) -> bool:
    return policy == CANONICAL_C3_POLICY or policy.startswith(C3_DIAGNOSTIC_PREFIX)

def _mode_is_freeform(mode: str) -> bool:
    return 'free' in str(mode).lower()

def _resolve_panel_columns(panel: pd.DataFrame) -> tuple[str, list[str]]:
    row_key = _detect_row_key(panel, label='heterogeneity panel')
    covariates: list[str] = []
    preferred = ['rating_grade', 'rating', 'grade', 'sector', 'sector_7', 'industry', 'industry_code', 'total_assets', 'log_assets', 'raw__total_assets', 'current_ratio', 'leverage', 'revenue']
    for col in preferred:
        if col in panel.columns and col not in covariates:
            covariates.append(col)
    if not covariates:
        raise ValueError({'message': 'No supported heterogeneity covariates found in panel.', 'looked_for': preferred, 'available_columns_sample': list(panel.columns)[:100]})
    return (row_key, covariates)

def _normalise_join_key(s: pd.Series) -> pd.Series:
    """Normalise row keys for robust panel joins without silently coalescing rows.

    N5/Stage9 artifacts may store ``row_id`` as an integer while some panel
    candidates store the same key as a string.  A direct pandas merge can then
    collapse all covariates to missing even though the key namespace is the same.
    The normalisation keeps integer-like keys canonical (``1.0`` -> ``1``) and
    otherwise falls back to stripped strings.
    """
    numeric = pd.to_numeric(s, errors='coerce')
    out = s.astype('string').str.strip()
    finite = numeric.notna() & np.isfinite(numeric)
    integer_like = finite & np.isclose(numeric, np.round(numeric))
    out.loc[integer_like] = numeric.loc[integer_like].round().astype('int64').astype('string')
    out = out.fillna('<MISSING_ROW_KEY>')
    return out

def _is_unknown_like(s: pd.Series) -> pd.Series:
    txt = s.astype('string').str.strip().str.upper()
    return txt.isna() | txt.isin({'', 'NA', 'N/A', 'NONE', 'NULL', 'NAN', 'UNKNOWN', 'UNK', '<NA>'})

def _panel_preference_order(panel: pd.DataFrame) -> pd.Series:
    """Prefer state rows closest to the C6 free-form evaluation when a panel has duplicates."""
    score = pd.Series(0, index=panel.index, dtype='int64')
    if 'policy' in panel.columns:
        pol = panel['policy'].astype(str)
        score += pol.eq('C6').astype('int64') * 100
        score += pol.eq('C4').astype('int64') * 10
    if 'mode' in panel.columns:
        mode = panel['mode'].astype(str).str.lower()
        score += mode.str.contains('free', na=False).astype('int64')
    return score

def _prepare_panel_for_join(panel: pd.DataFrame, *, hetero_keys: pd.Series) -> tuple[pd.DataFrame, str, list[str], dict]:
    original_key, covariates = _resolve_panel_columns(panel)
    p = panel.copy()
    if original_key != 'row_id':
        p = p.rename(columns={original_key: 'row_id'})
    p['__join_row_id'] = _normalise_join_key(p['row_id'])
    hkeys = _normalise_join_key(hetero_keys)
    p['__panel_pref'] = _panel_preference_order(p)
    p = p.sort_values(['__join_row_id', '__panel_pref'], ascending=[True, False])
    panel_small = p[['__join_row_id'] + covariates].drop_duplicates('__join_row_id', keep='first')
    hset = set(hkeys.astype(str))
    pset = set(panel_small['__join_row_id'].astype(str))
    intersection = hset & pset
    diagnostics = {'original_panel_key': original_key, 'panel_rows': int(len(panel)), 'panel_key_nunique': int(panel_small['__join_row_id'].nunique()), 'heterogeneity_key_nunique': int(hkeys.nunique()), 'join_intersection_nunique': int(len(intersection)), 'join_coverage': float(len(intersection) / hkeys.nunique()) if hkeys.nunique() else 0.0, 'covariates_detected': list(covariates), 'covariate_diagnostics': {}}
    if diagnostics['join_coverage'] < 0.95:
        raise ValueError({'message': 'Panel join coverage is below the required 95% threshold.', **diagnostics, 'heterogeneity_key_sample': hkeys.head(20).tolist(), 'panel_key_sample': panel_small['__join_row_id'].head(20).tolist()})
    return (panel_small, original_key, covariates, diagnostics)

def _slice_covariate(sub: pd.DataFrame, cov: str) -> tuple[pd.Series | None, dict]:
    raw = sub[cov]
    diag = {'merged_nonnull': int(raw.notna().sum()), 'merged_nunique_raw': int(raw.dropna().nunique()), 'usable': False, 'reason': None}
    numeric = pd.to_numeric(raw, errors='coerce')
    numeric_nonnull = int(numeric.notna().sum())
    numeric_nunique = int(numeric.dropna().nunique())
    diag['numeric_nonnull'] = numeric_nonnull
    diag['numeric_nunique'] = numeric_nunique
    if numeric_nonnull >= max(30, int(0.5 * len(raw))) and numeric_nunique > 1:
        try:
            sliced = pd.qcut(numeric, q=4, duplicates='drop').astype(str)
        except ValueError as exc:
            diag['reason'] = f'qcut_failed: {exc}'
            return (None, diag)
        if sliced.dropna().nunique() <= 1:
            diag['reason'] = 'numeric_slice_collapsed'
            return (None, diag)
        sliced.loc[numeric.isna()] = 'MISSING'
        diag['usable'] = True
        diag['kind'] = 'numeric_qcut'
        diag['slice_nunique'] = int(sliced.dropna().nunique())
        return (sliced, diag)
    unknown = _is_unknown_like(raw)
    meaningful = raw[~unknown]
    meaningful_nunique = int(meaningful.dropna().nunique())
    diag['meaningful_nonnull'] = int((~unknown).sum())
    diag['meaningful_nunique'] = meaningful_nunique
    if meaningful_nunique <= 1:
        diag['reason'] = 'categorical_missing_unknown_or_constant'
        return (None, diag)
    vc = raw.astype('string').fillna('MISSING')
    top = set(vc[~unknown].value_counts().head(12).index.tolist())
    sliced = vc.where(vc.isin(top), other='OTHER')
    diag['usable'] = True
    diag['kind'] = 'categorical_top12'
    diag['slice_nunique'] = int(sliced.dropna().nunique())
    return (sliced, diag)

def run_winrate_heterogeneity(*, runs: Iterable[Path], out_dir: Path, panel: Path | None=None, oracle: str='alpha', include_c3_diagnostics: bool=False) -> dict:
    if oracle not in ORACLES:
        raise ValueError(f'Unsupported oracle {oracle!r}; supported={ORACLES}')
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    run_inputs: list[RunInput] = []
    win_rows: list[dict] = []
    heterogeneity_frames: list[pd.DataFrame] = []
    input_records: list[dict] = []
    for run in runs:
        run_dir = Path(run).resolve()
        comp = find_comparison_file(run_dir)
        raw = _read_table(comp)
        df, original_row_key, score_cols = _validate_policy_table(raw, source=comp)
        score_col = score_cols[oracle]
        ic = parse_information_condition(run_dir, df)
        run_input = RunInput(run_dir=run_dir if run_dir.is_dir() else comp.parent, comparison_file=comp, run_label=run_dir.name, information_condition=ic)
        run_inputs.append(run_input)
        input_records.append({'run_label': run_input.run_label, 'run_dir': str(run_input.run_dir), 'comparison_file': str(comp), 'information_condition': ic, 'original_row_key': original_row_key, 'score_column': score_col, 'n_rows': int(len(df))})
        ref = _reference_rows(df, score_col, run_label=run_input.run_label)
        for (policy, mode), g in _policy_mode_groups(df):
            if not include_c3_diagnostics and _is_c3_diagnostic(policy):
                continue
            if policy == CANONICAL_C3_POLICY:
                continue
            if g['row_id'].duplicated(keep=False).any():
                raise ValueError({'message': 'Target policy/mode has duplicate row_id values; win-rate requires paired row-level scores.', 'run': run_input.run_label, 'policy': policy, 'mode': mode, 'duplicate_sample': g.loc[g['row_id'].duplicated(keep=False), 'row_id'].head(20).tolist()})
            merged = g[['row_id', score_col]].rename(columns={score_col: 'target_score'}).merge(ref, on='row_id', how='inner')
            if merged.empty:
                continue
            gap = pd.to_numeric(merged['target_score'], errors='coerce') - pd.to_numeric(merged['c3_score'], errors='coerce')
            finite = gap[np.isfinite(gap)]
            n_pairs = int(len(finite))
            n_ties = int((finite == 0).sum())
            n_non_ties = int(n_pairs - n_ties)
            n_wins = int((finite > 0).sum())
            wr, wr_lo, wr_hi = _wilson(n_wins, n_non_ties)
            win_rows.append({'run': run_input.run_label, 'information_condition': ic, 'policy': policy, 'mode': mode, 'oracle_backend': oracle, 'reference_policy': CANONICAL_C3_POLICY, 'n_pairs': n_pairs, 'n_ties': n_ties, 'n_non_tie_pairs': n_non_ties, 'n_wins_excluding_ties': n_wins, 'win_rate_excluding_ties': wr, 'win_rate_wilson_lo': wr_lo, 'win_rate_wilson_hi': wr_hi, 'positive_fraction_including_ties': float((finite > 0).mean()) if n_pairs else math.nan, 'zero_fraction': float((finite == 0).mean()) if n_pairs else math.nan, 'mean_gap_vs_c3': float(finite.mean()) if n_pairs else math.nan, 'median_gap_vs_c3': float(finite.median()) if n_pairs else math.nan})
            if policy.startswith('C6') and _mode_is_freeform(mode):
                h = merged[['row_id']].copy()
                h['gap_vs_c3'] = finite.to_numpy(dtype=float)
                h['run'] = run_input.run_label
                h['information_condition'] = ic
                h['policy'] = policy
                h['mode'] = mode
                heterogeneity_frames.append(h)
    if not win_rows:
        raise ValueError('No policy rows could be paired against exact C3; no win-rate output produced.')
    win_df = pd.DataFrame(win_rows).sort_values(['run', 'policy', 'mode']).reset_index(drop=True)
    win_df.to_csv(out_dir / 'win_rates_vs_C3.csv', index=False, encoding='utf-8-sig')
    heterogeneity_output = None
    panel_diagnostics = None
    if panel is not None:
        if not heterogeneity_frames:
            raise ValueError('--panel was provided but no C6 free-form rows were found for heterogeneity analysis.')
        panel_df = _read_table(Path(panel))
        hetero_base = pd.concat(heterogeneity_frames, ignore_index=True)
        panel_small, original_panel_key, covariates, panel_diagnostics = _prepare_panel_for_join(panel_df, hetero_keys=hetero_base['row_id'])
        hetero_base['__join_row_id'] = _normalise_join_key(hetero_base['row_id'])
        hetero = hetero_base.merge(panel_small, on='__join_row_id', how='left')
        rows: list[pd.DataFrame] = []
        usable_covariates: list[str] = []
        for cov in covariates:
            sub = hetero.copy()
            sliced, diag = _slice_covariate(sub, cov)
            panel_diagnostics['covariate_diagnostics'][cov] = diag
            if sliced is None:
                continue
            usable_covariates.append(cov)
            sub['_slice'] = sliced
            grouped = sub.groupby(['run', 'information_condition', 'policy', 'mode', '_slice'], dropna=False)['gap_vs_c3']
            agg = grouped.agg(mean_gap='mean', median_gap='median', n='count', positive_fraction=lambda s: float((s > 0).mean()), zero_fraction=lambda s: float((s == 0).mean()), negative_fraction=lambda s: float((s < 0).mean()))
            agg = agg.reset_index().rename(columns={'_slice': 'slice'})
            agg['covariate'] = cov
            rows.append(agg)
        if not rows:
            raise ValueError({'message': 'Panel was provided, but no heterogeneity covariate remained usable after join/slice diagnostics.', 'panel_diagnostics': panel_diagnostics})
        panel_diagnostics['usable_covariates'] = usable_covariates
        hetero_out = pd.concat(rows, ignore_index=True)
        hetero_out = hetero_out[['run', 'information_condition', 'policy', 'mode', 'covariate', 'slice', 'n', 'mean_gap', 'median_gap', 'positive_fraction', 'zero_fraction', 'negative_fraction']]
        hetero_out.to_csv(out_dir / 'residual_heterogeneity_exploratory.csv', index=False, encoding='utf-8-sig')
        heterogeneity_output = 'residual_heterogeneity_exploratory.csv'
    meta = {'schema_version': WINRATE_SCHEMA_VERSION, 'created_utc': _now(), 'status': 'PASS', 'oracle_backend': oracle, 'canonical_c3_policy': CANONICAL_C3_POLICY, 'include_c3_diagnostics': bool(include_c3_diagnostics), 'inputs': input_records, 'panel': str(Path(panel).resolve()) if panel else None, 'outputs': {'win_rates_vs_C3': 'win_rates_vs_C3.csv', 'residual_heterogeneity_exploratory': heterogeneity_output}, 'panel_diagnostics': panel_diagnostics if panel is not None else None, 'interpretation_note': 'Win rate excludes ties for the Wilson interval; heterogeneity slices are exploratory and unadjusted.'}
    write_json(out_dir / 'metadata.json', meta)
    return meta

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='N3 win-rate and N6 residual heterogeneity diagnostics versus exact C3')
    ap.add_argument('--runs', nargs='+', required=True, help='Run directories or row-level Stage9 comparison files')
    ap.add_argument('--panel', default=None, help='Optional phase_eval/serving panel for exploratory heterogeneity')
    ap.add_argument('--out-dir', '--out', dest='out_dir', required=True)
    ap.add_argument('--oracle', default='alpha', choices=list(ORACLES))
    ap.add_argument('--include-c3-diagnostics', action='store_true', help='Include C3 q-diagnostic variants as target policies instead of excluding them')
    args = ap.parse_args(argv)
    meta = run_winrate_heterogeneity(runs=[Path(p) for p in args.runs], panel=Path(args.panel) if args.panel else None, out_dir=Path(args.out_dir), oracle=args.oracle, include_c3_diagnostics=args.include_c3_diagnostics)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
