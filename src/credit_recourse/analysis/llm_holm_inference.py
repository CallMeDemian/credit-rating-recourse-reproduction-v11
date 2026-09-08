"""
Compute paired Wilcoxon contrasts and Holm-adjusted p-values for LLM Stage9 runs.

Robust input support:
  - llm_stage9_llm_rl_comparison.csv/parquet
  - llm_stage9_paired_vs_C0_noop.csv/parquet
  - any nested file matching *llm_rl_comparison* or *paired_vs*C0*

Main paper use:
  H1: C5-C4 and C7-C6, by mode/backend/run.
  H3: C6-C6X, by mode/backend/run.

Default Holm family: hypothesis_mode
  - H1_score:candidate_selection
  - H1_score:free_form_10d
  - H3_reference:candidate_selection
  - H3_reference:free_form_10d
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

@dataclass(frozen=True)
class ContrastSpec:
    hypothesis: str
    contrast: str
    policy_a: str
    policy_b: str
    mode: str
    backend_label: str
    run_dir: Path

def _file_priority(p: Path) -> tuple[int, int, str]:
    name = p.name.lower()
    s = str(p).lower()
    if name == 'llm_stage9_llm_rl_comparison.csv':
        pri = 0
    elif name == 'llm_stage9_llm_rl_comparison.parquet':
        pri = 1
    elif name == 'llm_stage9_paired_vs_c0_noop.csv':
        pri = 2
    elif name == 'llm_stage9_paired_vs_c0_noop.parquet':
        pri = 3
    elif 'llm_rl_comparison' in name and p.suffix.lower() in {'.csv', '.parquet'}:
        pri = 4
    elif 'paired_vs' in name and 'c0' in name and (p.suffix.lower() in {'.csv', '.parquet'}):
        pri = 5
    else:
        pri = 99
    return (pri, 0 if 'stage9' in s else 1, len(str(p)))

def find_stage9_file(run_dir: Path) -> Path:
    if run_dir.is_file():
        return run_dir
    candidates: list[Path] = []
    exact_names = ['llm_stage9_llm_rl_comparison.csv', 'llm_stage9_llm_rl_comparison.parquet', 'llm_stage9_paired_vs_C0_noop.csv', 'llm_stage9_paired_vs_C0_noop.parquet']
    for nm in exact_names:
        p = run_dir / nm
        if p.exists():
            candidates.append(p)
    for pat in ['**/llm_stage9_llm_rl_comparison.csv', '**/llm_stage9_llm_rl_comparison.parquet', '**/llm_stage9_paired_vs_C0_noop.csv', '**/llm_stage9_paired_vs_C0_noop.parquet', '**/*llm_rl_comparison*.csv', '**/*llm_rl_comparison*.parquet', '**/*paired_vs*C0*.csv', '**/*paired_vs*C0*.parquet']:
        candidates.extend(run_dir.glob(pat))
    candidates = sorted(set(candidates), key=_file_priority)
    if candidates:
        return candidates[0]
    stage9_like = sorted([p for p in run_dir.rglob('*stage9*') if p.is_file()])[:50]
    msg = f'No usable Stage9 per-row file found under {run_dir}.\n'
    msg += 'Expected one of: llm_stage9_llm_rl_comparison.csv/parquet or llm_stage9_paired_vs_C0_noop.csv/parquet.'
    if stage9_like:
        msg += '\nStage9-like files found:\n' + '\n'.join((str(p) for p in stage9_like))
    else:
        msg += '\nNo *stage9* files found below this path. Check whether RunDirs points to the extracted run folder, not only a summary/archive parent.'
    raise FileNotFoundError(msg)

def read_stage9_table(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix.lower() == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f'Unsupported input file extension: {path}')
    if 'delta_R_score_alpha' in df.columns:
        score_col = 'delta_R_score_alpha'
    elif 'gap_alpha' in df.columns:
        score_col = 'gap_alpha'
    elif 'mean_delta_R_score_alpha' in df.columns:
        raise ValueError(f'{path} looks like a policy summary, not a per-row file. Need row_id-level llm_stage9_llm_rl_comparison or paired_vs_C0_noop.')
    else:
        raise ValueError(f'Cannot find alpha score column in {path}. Available columns: {list(df.columns)[:80]}')
    return (df, score_col)

def normalize_policy_values(df: pd.DataFrame) -> pd.DataFrame:
    """Add a compact policy key if policy labels have suffixes like C4_direct."""
    out = df.copy()
    if 'policy' not in out.columns:
        raise ValueError('Input table missing policy column.')
    pol = out['policy'].astype(str)
    extracted = pol.str.extract('^(C6X|C[0-9]+)', expand=False)
    out['policy_key'] = extracted.fillna(pol)
    return out

def holm_adjust(pvals: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvals), dtype=float)
    n = len(p)
    out = np.full(n, np.nan, dtype=float)
    ok = np.isfinite(p)
    idx = np.where(ok)[0]
    if len(idx) == 0:
        return out.tolist()
    order = idx[np.argsort(p[idx])]
    prev = 0.0
    for rank, i in enumerate(order, start=1):
        adj = (len(order) - rank + 1) * p[i]
        prev = max(prev, adj)
        out[i] = min(prev, 1.0)
    return out.tolist()

def p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def signed_wilcoxon_p(x: np.ndarray, alternative: str='two-sided') -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[np.abs(x) > 0]
    if len(x) == 0:
        return math.nan
    if wilcoxon is None:
        raise RuntimeError('scipy is required for Wilcoxon p-values. Install scipy in the thesis venv.')
    return float(wilcoxon(x, alternative=alternative, zero_method='wilcox', correction=False).pvalue)

def paired_contrast(df: pd.DataFrame, spec: ContrastSpec, score_col: str) -> dict:
    needed = {'row_id', 'policy_key', 'mode', score_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'{spec.run_dir}: missing columns {missing}')
    d = df[df['mode'].astype(str).eq(spec.mode)].copy()
    a = d[d['policy_key'].astype(str).eq(spec.policy_a)][['row_id', score_col]].rename(columns={score_col: 'score_a'})
    b = d[d['policy_key'].astype(str).eq(spec.policy_b)][['row_id', score_col]].rename(columns={score_col: 'score_b'})
    m = a.merge(b, on='row_id', how='inner')
    gap = pd.to_numeric(m['score_b'], errors='coerce') - pd.to_numeric(m['score_a'], errors='coerce')
    gap = gap[np.isfinite(gap)]
    p = signed_wilcoxon_p(gap.to_numpy())
    return {'hypothesis': spec.hypothesis, 'contrast': spec.contrast, 'policy_a': spec.policy_a, 'policy_b': spec.policy_b, 'mode': spec.mode, 'backend_label': spec.backend_label, 'run_dir': str(spec.run_dir), 'n_pairs': int(len(gap)), 'n_nonzero_pairs': int((np.abs(gap) > 0).sum()), 'mean_gap': float(gap.mean()) if len(gap) else math.nan, 'median_gap': float(gap.median()) if len(gap) else math.nan, 'wilcoxon_p_raw': p}

def build_specs(run_dir: Path, backend_label: str, include_candidate_c7_c6: bool) -> list[ContrastSpec]:
    modes = ['candidate_selection', 'free_form_10d']
    specs: list[ContrastSpec] = []
    for mode in modes:
        specs.append(ContrastSpec('H1_score', 'C5-C4', 'C4', 'C5', mode, backend_label, run_dir))
        if mode == 'free_form_10d' or include_candidate_c7_c6:
            specs.append(ContrastSpec('H1_score', 'C7-C6', 'C6', 'C7', mode, backend_label, run_dir))
        specs.append(ContrastSpec('H3_reference', 'C6-C6X', 'C6X', 'C6', mode, backend_label, run_dir))
    return specs

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', required=True, help='Stage9, run/archive directories, or explicit per-row files')
    ap.add_argument('--backend-labels', nargs='*', default=None, help='Optional labels, one per run')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--holm-family', choices=['hypothesis', 'hypothesis_mode', 'hypothesis_mode_contrast'], default='hypothesis_mode', help='Paper default: hypothesis_mode. Use hypothesis for more conservative all-H1/all-H3 correction.')
    ap.add_argument('--include-candidate-c7-c6', action='store_true', help='Also include candidate-selection C7-C6 in H1.')
    args = ap.parse_args(argv)
    run_paths = [Path(p).resolve() for p in args.runs]
    if args.backend_labels and len(args.backend_labels) != len(run_paths):
        raise ValueError('--backend-labels must have the same length as --runs')
    labels = args.backend_labels or [p.name if p.is_dir() else p.parent.name for p in run_paths]
    rows = []
    input_records = []
    for run_path, label in zip(run_paths, labels):
        stage9_file = find_stage9_file(run_path)
        df, score_col = read_stage9_table(stage9_file)
        df = normalize_policy_values(df)
        input_records.append({'run_arg': str(run_path), 'backend_label': label, 'selected_input_file': str(stage9_file), 'score_col': score_col, 'n_rows': len(df), 'policies': ','.join(sorted(df['policy_key'].astype(str).unique())), 'modes': ','.join(sorted(df['mode'].astype(str).unique())) if 'mode' in df.columns else ''})
        for spec in build_specs(run_dir=stage9_file.parent, backend_label=label, include_candidate_c7_c6=args.include_candidate_c7_c6):
            try:
                rows.append(paired_contrast(df, spec, score_col))
            except Exception as e:
                rows.append({'hypothesis': spec.hypothesis, 'contrast': spec.contrast, 'policy_a': spec.policy_a, 'policy_b': spec.policy_b, 'mode': spec.mode, 'backend_label': spec.backend_label, 'run_dir': str(spec.run_dir), 'error': repr(e), 'n_pairs': 0, 'n_nonzero_pairs': 0, 'mean_gap': math.nan, 'median_gap': math.nan, 'wilcoxon_p_raw': math.nan})
    out = pd.DataFrame(rows)
    if args.holm_family == 'hypothesis':
        out['holm_family'] = out['hypothesis']
    elif args.holm_family == 'hypothesis_mode':
        out['holm_family'] = out['hypothesis'] + ':' + out['mode']
    else:
        out['holm_family'] = out['hypothesis'] + ':' + out['mode'] + ':' + out['contrast']
    out['wilcoxon_p_holm'] = np.nan
    for fam, idx in out.groupby('holm_family').groups.items():
        out.loc[idx, 'wilcoxon_p_holm'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].to_list())
    out['sig_holm'] = out['wilcoxon_p_holm'].apply(p_to_stars)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'holm_pairwise_contrasts.csv'
    out.to_csv(out_path, index=False, encoding='utf-8-sig')
    pd.DataFrame(input_records).to_csv(out_dir / 'holm_input_files.csv', index=False, encoding='utf-8-sig')
    for hyp in ['H1_score', 'H3_reference']:
        h = out[out['hypothesis'].eq(hyp)].copy()
        if not h.empty:
            h.to_csv(out_dir / f'holm_{hyp}.csv', index=False, encoding='utf-8-sig')
    cols = ['hypothesis', 'mode', 'backend_label', 'contrast', 'n_pairs', 'mean_gap', 'wilcoxon_p_raw', 'wilcoxon_p_holm', 'sig_holm']
    print(f"Selected input files written to {out_dir / 'holm_input_files.csv'}")
    print(f'Wrote {out_path}')
    print(out[[c for c in cols if c in out.columns]].to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
