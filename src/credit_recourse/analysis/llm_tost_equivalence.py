from __future__ import annotations
'Paired TOST equivalence checks for LLM/RL or ablation score tables.\n\nThis module is intentionally evaluator-only. It reads already-scored row-level\npolicy tables, joins target and reference rows by ``row_id``, and performs a\npaired mean two-one-sided-test (TOST) for a user-declared equivalence margin.\nIt does not call an LLM API, does not rerun the simulator, and does not retrain\nRL.  The equivalence margin is required on the CLI so the test cannot silently\nintroduce an undocumented practical-equivalence threshold.\n'
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
try:
    from scipy import stats
except Exception:
    stats = None
from credit_recourse.rl.common.io import write_json
SCORE_COLUMN_CANDIDATES = {'alpha': ('delta_R_score_alpha', 'gap_alpha', 'mean_delta_R_score_alpha'), 'beta': ('delta_R_score_beta', 'gap_beta', 'mean_delta_R_score_beta'), 'gamma': ('delta_R_score_gamma', 'gap_gamma', 'mean_delta_R_score_gamma')}
ROW_KEY = 'row_id'
TOST_SCHEMA_VERSION = 'llm_tost_equivalence_v1'

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Input score table does not exist: {path}')
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported input extension for {path}; expected .csv or .parquet')

def _filter_table(df: pd.DataFrame, *, policy: str | None, mode: str | None, label: str) -> pd.DataFrame:
    out = df.copy()
    if ROW_KEY not in out.columns:
        raise ValueError(f'{label} table missing required key column: {ROW_KEY}')
    if policy is not None:
        if 'policy' not in out.columns:
            raise ValueError(f'{label} policy filter requested but table has no policy column')
        out = out[out['policy'].astype(str).eq(str(policy))].copy()
    if mode is not None:
        if 'mode' not in out.columns:
            raise ValueError(f'{label} mode filter requested but table has no mode column')
        out = out[out['mode'].astype(str).eq(str(mode))].copy()
    if out.empty:
        raise ValueError({'message': f'{label} filter produced zero rows', 'policy': policy, 'mode': mode})
    dup = out[ROW_KEY].duplicated(keep=False)
    if dup.any():
        sample = out.loc[dup, ROW_KEY].head(20).tolist()
        raise ValueError({'message': f'{label} table has duplicate row_id after filtering; paired TOST requires one row per firm.', 'duplicate_sample': sample, 'n_duplicate_rows': int(dup.sum())})
    return out

def _resolve_score_column(df: pd.DataFrame, backend: str, *, label: str) -> str:
    for col in SCORE_COLUMN_CANDIDATES[backend]:
        if col in df.columns:
            if col.startswith('mean_'):
                raise ValueError(f'{label} table appears to be an aggregate summary because it contains {col}; paired TOST requires row_id-level scores.')
            return col
    raise ValueError({'message': f'Could not find a row-level score column for backend={backend} in {label} table.', 'accepted_columns': SCORE_COLUMN_CANDIDATES[backend], 'available_columns_sample': list(df.columns)[:80]})

def paired_tost(gap: Iterable[float], *, margin: float, alpha: float=0.05) -> dict:
    """Return paired mean TOST statistics for H1: -margin < mean(gap) < margin."""
    if margin <= 0 or not math.isfinite(float(margin)):
        raise ValueError(f'Equivalence margin must be a positive finite number; got {margin}')
    if not 0.0 < float(alpha) < 0.5:
        raise ValueError(f'alpha must be in (0, 0.5); got {alpha}')
    x = np.asarray(list(gap), dtype=float)
    x = x[np.isfinite(x)]
    n = int(len(x))
    if n < 2:
        raise ValueError(f'At least two finite paired gaps are required for TOST; got n={n}')
    if stats is None:
        raise RuntimeError('scipy is required for paired TOST equivalence checks. Install scipy in the thesis venv.')
    mean = float(np.mean(x))
    sd = float(np.std(x, ddof=1))
    df = n - 1
    if sd == 0.0:
        se = 0.0
        t_lower = math.inf if mean > -margin else -math.inf
        t_upper = -math.inf if mean < margin else math.inf
        p_lower = 0.0 if mean > -margin else 1.0
        p_upper = 0.0 if mean < margin else 1.0
        ci_low = mean
        ci_high = mean
    else:
        se = sd / math.sqrt(n)
        t_lower = (mean + margin) / se
        t_upper = (mean - margin) / se
        p_lower = float(stats.t.sf(t_lower, df=df))
        p_upper = float(stats.t.cdf(t_upper, df=df))
        tcrit = float(stats.t.ppf(1.0 - float(alpha), df=df))
        ci_low = mean - tcrit * se
        ci_high = mean + tcrit * se
    p_tost = max(float(p_lower), float(p_upper))
    return {'n_pairs': n, 'mean_gap': mean, 'sd_gap': sd, 'se_gap': float(se), 'equivalence_margin': float(margin), 'alpha': float(alpha), 't_lower': float(t_lower), 'p_lower_mean_gt_minus_margin': float(p_lower), 't_upper': float(t_upper), 'p_upper_mean_lt_plus_margin': float(p_upper), 'p_tost': p_tost, 'ci_level': float(1.0 - 2.0 * float(alpha)), 'ci_low': float(ci_low), 'ci_high': float(ci_high), 'equivalent': bool(p_tost < float(alpha)), 'ci_within_margin': bool(ci_low > -float(margin) and ci_high < float(margin))}

def run_tost_equivalence(*, target_scores: Path, reference_scores: Path, out_dir: Path, equivalence_margin: float, alpha: float=0.05, target_policy: str | None=None, target_mode: str | None=None, reference_policy: str | None=None, reference_mode: str | None=None, backends: Iterable[str]=('alpha', 'beta', 'gamma')) -> dict:
    target_scores = Path(target_scores).resolve()
    reference_scores = Path(reference_scores).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _filter_table(_read_table(target_scores), policy=target_policy, mode=target_mode, label='target')
    reference = _filter_table(_read_table(reference_scores), policy=reference_policy, mode=reference_mode, label='reference')
    requested_backends = [str(b).strip().lower() for b in backends if str(b).strip()]
    bad = sorted(set(requested_backends) - set(SCORE_COLUMN_CANDIDATES))
    if bad:
        raise ValueError(f'Unsupported backend(s): {bad}; supported={sorted(SCORE_COLUMN_CANDIDATES)}')
    rows: list[dict] = []
    for backend in requested_backends:
        tcol = _resolve_score_column(target, backend, label='target')
        rcol = _resolve_score_column(reference, backend, label='reference')
        joined = target[[ROW_KEY, tcol]].rename(columns={tcol: 'target_score'}).merge(reference[[ROW_KEY, rcol]].rename(columns={rcol: 'reference_score'}), on=ROW_KEY, how='inner')
        if joined.empty:
            raise ValueError({'message': 'No overlapping row_id values between target and reference after filtering.', 'backend': backend})
        gap = pd.to_numeric(joined['target_score'], errors='coerce') - pd.to_numeric(joined['reference_score'], errors='coerce')
        stats_row = paired_tost(gap.to_numpy(), margin=float(equivalence_margin), alpha=float(alpha))
        stats_row.update({'backend': backend, 'target_score_column': tcol, 'reference_score_column': rcol, 'target_policy': target_policy or '<unfiltered>', 'target_mode': target_mode or '<unfiltered>', 'reference_policy': reference_policy or '<unfiltered>', 'reference_mode': reference_mode or '<unfiltered>', 'n_target_rows': int(len(target)), 'n_reference_rows': int(len(reference))})
        rows.append(stats_row)
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / 'tost_equivalence_results.csv', index=False, encoding='utf-8-sig')
    meta = {'schema_version': TOST_SCHEMA_VERSION, 'created_utc': _now(), 'target_scores': str(target_scores), 'reference_scores': str(reference_scores), 'target_policy': target_policy, 'target_mode': target_mode, 'reference_policy': reference_policy, 'reference_mode': reference_mode, 'backends': requested_backends, 'equivalence_margin': float(equivalence_margin), 'alpha': float(alpha), 'row_key': ROW_KEY, 'output': 'tost_equivalence_results.csv', 'status': 'PASS'}
    write_json(out_dir / 'metadata.json', meta)
    return meta

def _parse_backends(value: str) -> list[str]:
    return [x.strip().lower() for x in str(value).split(',') if x.strip()]

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Paired TOST equivalence for row-level LLM/RL or ablation score tables')
    ap.add_argument('--target-scores', required=True, help='Row-level target score table (.csv/.parquet)')
    ap.add_argument('--reference-scores', required=True, help='Row-level reference score table (.csv/.parquet)')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--equivalence-margin', type=float, required=True, help='Positive practical-equivalence margin on the score scale')
    ap.add_argument('--alpha', type=float, default=0.05)
    ap.add_argument('--target-policy', default=None)
    ap.add_argument('--target-mode', default=None)
    ap.add_argument('--reference-policy', default=None)
    ap.add_argument('--reference-mode', default=None)
    ap.add_argument('--backends', default='alpha,beta,gamma', help='Comma-separated subset of alpha,beta,gamma')
    args = ap.parse_args(argv)
    meta = run_tost_equivalence(target_scores=Path(args.target_scores), reference_scores=Path(args.reference_scores), out_dir=Path(args.out_dir), equivalence_margin=float(args.equivalence_margin), alpha=float(args.alpha), target_policy=args.target_policy, target_mode=args.target_mode, reference_policy=args.reference_policy, reference_mode=args.reference_mode, backends=_parse_backends(args.backends))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
