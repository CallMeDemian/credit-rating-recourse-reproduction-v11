from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from credit_recourse.contracts.paper_reproduction import discover_archived_runs, load_profile, select_exact_role
from .claim_evidence_common import repo_rel, write_csv

def _holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues), dtype=float)
    running = 0.0
    m = len(pvalues)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * float(pvalues[index]))
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()

def _bootstrap_ci(values: np.ndarray, seed: int, draws: int=10000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(draws, dtype=float)
    for start in range(0, draws, 500):
        batch = min(500, draws - start)
        indices = rng.integers(0, n, size=(batch, n))
        means[start:start + batch] = values[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return (float(lo), float(hi))

def _stage9_path(run_dir: Path) -> tuple[Path, Path | None]:
    parquet_candidates = [run_dir / 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.parquet', run_dir / 'stage9_policy_comparison/llm_stage9_llm_rl_comparison.parquet']
    parquet = [path for path in parquet_candidates if path.is_file()]
    if len(parquet) != 1:
        raise RuntimeError(f'exactly one canonical Stage9 parquet ledger required, found {parquet}')
    csv_candidates = [parquet[0].with_suffix('.csv')]
    csv_matches = [path for path in csv_candidates if path.is_file()]
    if len(csv_matches) > 1:
        raise RuntimeError(f'multiple Stage9 CSV mirrors found: {csv_matches}')
    return (parquet[0], csv_matches[0] if csv_matches else None)

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    profile = load_profile(root)
    records = discover_archived_runs(root, profile)
    primary = select_exact_role(records, run_role=str(profile['llm']['primary']['run_role']), information_condition='IC-b')
    if not (primary.has_stage7 and primary.has_stage8 and primary.has_stage9):
        raise RuntimeError(f'selected C8 run is incomplete: {primary.run_label}')
    path, csv_mirror = _stage9_path(primary.run_dir)
    frame = pd.read_parquet(path)
    if csv_mirror is not None:
        mirror = pd.read_csv(csv_mirror, encoding='utf-8-sig')
        comparison_columns = [column for column in frame.columns if column in mirror.columns]
        if {'row_id', 'policy', 'mode'} - set(comparison_columns) or len(mirror) != len(frame):
            raise RuntimeError('Stage9 CSV mirror schema/row count differs from canonical parquet')
        left = frame[comparison_columns].sort_values(['row_id', 'policy', 'mode']).reset_index(drop=True)
        right = mirror[comparison_columns].sort_values(['row_id', 'policy', 'mode']).reset_index(drop=True)
        for column in comparison_columns:
            if pd.api.types.is_numeric_dtype(left[column]):
                a = pd.to_numeric(left[column], errors='coerce').to_numpy(float)
                b = pd.to_numeric(right[column], errors='coerce').to_numpy(float)
                if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
                    raise RuntimeError(f'Stage9 CSV mirror numeric mismatch in {column}')
            elif not left[column].fillna('').astype(str).eq(right[column].fillna('').astype(str)).all():
                raise RuntimeError(f'Stage9 CSV mirror text mismatch in {column}')
    required = {'row_id', 'policy', 'mode', 'delta_R_score_alpha', 'delta_R_score_beta', 'delta_R_score_gamma'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'Stage9 C8 input missing columns: {missing}')
    selected = frame.loc[frame['mode'].astype(str).eq('free_form_10d') & frame['policy'].astype(str).isin(['C6', 'C8'])].copy()
    if selected.duplicated(['row_id', 'policy']).any():
        raise RuntimeError('C8 timing input has duplicate row_id/policy keys')
    counts = selected.groupby('policy')['row_id'].nunique().to_dict()
    if counts != {'C6': 575, 'C8': 575}:
        raise RuntimeError(f'C8 timing requires 575 paired firms per policy, got {counts}')
    rows: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []
    for oracle_index, oracle in enumerate(('alpha', 'beta', 'gamma')):
        value = f'delta_R_score_{oracle}'
        wide = selected.pivot(index='row_id', columns='policy', values=value)
        if list(wide.columns.sort_values()) != ['C6', 'C8'] or len(wide.dropna()) != 575:
            raise RuntimeError(f'{oracle}: C8/C6 pairing failed')
        gap = (wide['C8'] - wide['C6']).to_numpy(dtype=float)
        if not np.isfinite(gap).all():
            raise RuntimeError(f'{oracle}: non-finite paired gaps')
        if np.allclose(gap, 0.0, rtol=0.0, atol=0.0):
            p_raw = 1.0
        else:
            p_raw = float(wilcoxon(gap, zero_method='pratt', alternative='two-sided', method='auto').pvalue)
        ci_low, ci_high = _bootstrap_ci(gap, seed=20260719 + oracle_index)
        raw_pvalues.append(p_raw)
        rows.append({'backend': 'gpt54mini', 'run_role': primary.run_role, 'run_label': primary.run_label, 'information_condition': 'IC-b', 'mode': 'free_form_10d', 'contrast': 'C8-C6', 'oracle_backend': oracle, 'n_pairs': len(gap), 'mean_diff': float(np.mean(gap)), 'median_diff': float(np.median(gap)), 'ci_low': ci_low, 'ci_high': ci_high, 'test': 'paired_wilcoxon_pratt_two_sided', 'p_raw': p_raw, 'holm_family': 'C8_minus_C6_across_three_oracles', 'source_path': repo_rel(root, path)})
    adjusted = _holm(raw_pvalues)
    for row, p_holm in zip(rows, adjusted):
        row['p_holm'] = p_holm
        row['decision'] = 'SIGNIFICANT' if p_holm < 0.05 else 'NOT_SIGNIFICANT'
    output = root / 'data/analysis/paper_repro/07_claim_sources/c8_timing_contrast.csv'
    write_csv(output, rows)
    return {'status': 'PASS', 'run_label': primary.run_label, 'rows': len(rows), 'n_pairs_per_oracle': 575, 'holm_family': 'C8_minus_C6_across_three_oracles', 'source_parquet': {'path': repo_rel(root, path)}, 'csv_mirror': {'path': repo_rel(root, csv_mirror)} if csv_mirror else None}

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
