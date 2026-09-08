"""
N5 Table 7-10c paired Wilcoxon + Holm inference.

Purpose
-------
Compute paper-ready significance markers for the N5 generation-time budget
constraint block, especially Table 7-10c:
  - N5 C6 free-form minus the canonical Stage6 C3 reference policy
  - N5 C6 free-form minus N5 C4 free-form

This script is intentionally read-only with respect to frozen run artifacts. It
writes analysis outputs under --out-dir.

Critical contract
-----------------
The canonical RL reference is the exact policy label ``C3_candidate_iql``.
Diagnostic variants such as ``C3_candidate_iql_q_argmax`` and
``C3_candidate_iql_q_rerank_at_3`` must never be collapsed into C3.  A previous
version used a broad ``candidate_iql`` substring match and blended those rows,
which lowered the apparent C3 mean and inflated the C6-C3 gap.  Frozen replay
hard-checks the exact C3 rows against the active Stage6 summary. Fresh
retraining hard-checks them against the immutable thesis reference and records
the active fresh Stage6 difference as advisory checkpoint drift.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping
import numpy as np
import pandas as pd
from credit_recourse.analysis.reference_reproduction_comparison import REFERENCE_REL, resolve_mode as resolve_rl_validation_mode
try:
    from scipy.stats import wilcoxon
except Exception as exc:
    wilcoxon = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None
ORACLES = ('alpha', 'beta', 'gamma')
CONTRASTS = (('C6_minus_C3', 'C6', 'C3'), ('C6_minus_C4', 'C6', 'C4'))
CANONICAL_C3_POLICY = 'C3_candidate_iql'
C3_VARIANT_PREFIX = CANONICAL_C3_POLICY + '_'

@dataclass(frozen=True)
class InputRun:
    information_condition: str
    run_dir: Path
    comparison_file: Path

def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == '.parquet':
        return pd.read_parquet(path)
    if suf == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table extension: {path}')

def _find_first_existing(root: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        p = root / name
        if p.exists() and p.is_file():
            return p
    return None

def find_comparison_file(run_dir: Path) -> Path:
    """Find a row-level Stage9 comparison file inside an N5 run directory."""
    run_dir = run_dir.resolve()
    if run_dir.is_file():
        return run_dir
    exact_rel = ['stage9_policy_comparison/llm_stage9_llm_rl_comparison.parquet', 'stage9_policy_comparison/llm_stage9_llm_rl_comparison.csv', 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.parquet', 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.csv', 'stage9_policy_comparison/llm_stage9_paired_vs_C0_noop.parquet', 'stage9_policy_comparison/llm_stage9_paired_vs_C0_noop.csv', 'stage9_llm_rl_comparison/llm_stage9_paired_vs_C0_noop.parquet', 'stage9_llm_rl_comparison/llm_stage9_paired_vs_C0_noop.csv']
    p = _find_first_existing(run_dir, exact_rel)
    if p is not None:
        return p
    candidates: list[Path] = []
    patterns = ['**/llm_stage9_llm_rl_comparison.parquet', '**/llm_stage9_llm_rl_comparison.csv', '**/llm_stage9_paired_vs_C0_noop.parquet', '**/llm_stage9_paired_vs_C0_noop.csv', '**/*llm_rl_comparison*.parquet', '**/*llm_rl_comparison*.csv']
    for pat in patterns:
        candidates.extend(run_dir.glob(pat))
    candidates = sorted(set(candidates), key=lambda x: ('paired_vs' in x.name.lower(), x.suffix != '.parquet', len(str(x))))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f'No row-level Stage9 comparison file found under {run_dir}. Expected llm_stage9_llm_rl_comparison.csv/parquet or paired_vs_C0_noop.')

def discover_n5_runs(project_root: Path) -> list[Path]:
    llm_root = project_root / 'data' / 'final_freeze' / 'llm_runs'
    if not llm_root.exists():
        raise FileNotFoundError(f'LLM run root not found: {llm_root}')
    runs = sorted(llm_root.glob('N5_C6_L1_1p27_IC*_gpt54mini_p50_main_seed1_20260707'))
    if not runs:
        runs = sorted(llm_root.glob('N5*C6*L1*1p27*IC*'))
    if not runs:
        raise FileNotFoundError(f'No N5 run directories found under {llm_root}')
    return runs

def parse_ic_label(run_dir: Path) -> str:
    s = run_dir.name
    m = re.search('IC[-_]?([abcABC])', s)
    if not m:
        return s
    return f'IC-{m.group(1).lower()}'

def policy_key(policy: object, *, c3_policy: str=CANONICAL_C3_POLICY) -> str:
    """Normalize policy labels without collapsing C3 diagnostics into C3.

    Only the exact canonical label ``C3_candidate_iql`` maps to ``C3``.  C3
    variants are returned as their own labels so they cannot contaminate paired
    C3 inference.
    """
    s = str(policy)
    if s == c3_policy:
        return 'C3'
    if s.startswith(c3_policy + '_'):
        return s
    m = re.match('^(C6X|C\\d+)', s)
    if m:
        return m.group(1)
    return s

def detect_score_column(df: pd.DataFrame, oracle: str) -> str:
    candidates = [f'delta_R_score_{oracle}', f'gap_{oracle}', f'score_{oracle}', f'mean_delta_R_score_{oracle}']
    for c in candidates:
        if c in df.columns:
            if c.startswith('mean_'):
                raise ValueError(f'Input looks like an aggregated summary, not a row-level table. Found {c}; need per-row scores to run paired Wilcoxon.')
            return c
    raise ValueError(f'No {oracle} score column found. Columns: {list(df.columns)[:80]}')

def standardize_comparison(df: pd.DataFrame, *, c3_policy: str=CANONICAL_C3_POLICY) -> pd.DataFrame:
    if 'row_id' not in df.columns:
        raise ValueError('Input comparison table is missing row_id.')
    if 'policy' not in df.columns:
        raise ValueError('Input comparison table is missing policy.')
    out = df.copy()
    out['policy_raw'] = out['policy'].astype(str)
    out['policy_key'] = out['policy_raw'].map(lambda x: policy_key(x, c3_policy=c3_policy))
    if 'mode' not in out.columns:
        out['mode'] = 'unspecified'
    return out

def holm_adjust(pvals: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok_idx = np.where(np.isfinite(p))[0]
    if len(ok_idx) == 0:
        return out.tolist()
    order = ok_idx[np.argsort(p[ok_idx])]
    prev = 0.0
    m = len(order)
    for rank, idx in enumerate(order, start=1):
        adj = (m - rank + 1) * p[idx]
        prev = max(prev, adj)
        out[idx] = min(prev, 1.0)
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

def wilcoxon_paired_p(gap: pd.Series) -> float:
    if wilcoxon is None:
        raise RuntimeError(f'scipy is required for Wilcoxon inference: {_SCIPY_IMPORT_ERROR!r}')
    x = pd.to_numeric(gap, errors='coerce').to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    x = x[np.abs(x) > 0]
    if len(x) == 0:
        return math.nan
    return float(wilcoxon(x, alternative='two-sided', zero_method='wilcox', correction=False).pvalue)

def _policy_score_frame(df: pd.DataFrame, policy: str, score_col: str, *, c3_policy: str=CANONICAL_C3_POLICY) -> pd.DataFrame:
    if policy == 'C3':
        d = df[df['policy_raw'].eq(c3_policy)].copy()
        if d.empty:
            variants = sorted(df.loc[df['policy_raw'].str.startswith(c3_policy + '_', na=False), 'policy_raw'].unique())
            raise ValueError(f'No exact canonical C3 policy rows found for {c3_policy!r}. Detected C3 variant rows: {variants[:20]}. Variants must not be used as C3.')
    else:
        d = df[df['policy_key'].astype(str).eq(policy)].copy()
    if policy in {'C4', 'C5', 'C6', 'C6X', 'C7', 'C8'}:
        d = d[d['mode'].astype(str).eq('free_form_10d')]
    if d.empty:
        raise ValueError(f'No rows found for policy {policy} and score {score_col}')
    d[score_col] = pd.to_numeric(d[score_col], errors='coerce')
    return d.groupby('row_id', as_index=False)[score_col].mean().rename(columns={score_col: f'score_{policy}'})

def _c3_reference_audit(df: pd.DataFrame, score_col: str, *, c3_policy: str=CANONICAL_C3_POLICY) -> dict:
    exact = df[df['policy_raw'].eq(c3_policy)].copy()
    variants = df[df['policy_raw'].str.startswith(c3_policy + '_', na=False)].copy()
    family = pd.concat([exact, variants], ignore_index=True, sort=False)

    def _per_row_mean(frame: pd.DataFrame) -> float:
        if frame.empty:
            return float('nan')
        s = frame.copy()
        s[score_col] = pd.to_numeric(s[score_col], errors='coerce')
        return float(s.groupby('row_id')[score_col].mean().mean())
    return {'exact_c3_policy': c3_policy, 'n_exact_rows': int(len(exact)), 'n_exact_row_ids': int(exact['row_id'].nunique()) if not exact.empty else 0, 'n_c3_variant_rows': int(len(variants)), 'n_c3_variant_row_ids': int(variants['row_id'].nunique()) if not variants.empty else 0, 'c3_variant_policies': ';'.join(sorted(variants['policy_raw'].unique())) if not variants.empty else '', 'stage9_exact_c3_mean_paired': _per_row_mean(exact), 'stage9_blended_c3_variant_mean': _per_row_mean(family), 'blended_minus_exact': _per_row_mean(family) - _per_row_mean(exact) if not exact.empty and (not family.empty) else float('nan')}

def _candidate_stage6_summary_paths(project_root: Path) -> list[Path]:
    return [project_root / 'data' / 'final_freeze' / 'stage6_candidate_selector_eval' / 'final_policy_summary.csv', project_root / 'data' / 'final_freeze' / 'stage6_multi_oracle_eval' / 'final_policy_summary.csv', project_root / 'stage6_candidate_selector_eval' / 'final_policy_summary.csv', project_root / 'stage6_multi_oracle_eval' / 'final_policy_summary.csv']

def find_stage6_summary(project_root: Path, explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f'Explicit --stage6-summary path not found: {p}')
        return p
    for p in _candidate_stage6_summary_paths(project_root):
        if p.exists() and p.is_file():
            return p.resolve()
    return None

def load_stage6_canonical_means(stage6_summary: Path, *, c3_policy: str=CANONICAL_C3_POLICY) -> dict[str, float | int | str]:
    df = pd.read_csv(stage6_summary)
    if 'policy' not in df.columns:
        raise ValueError(f'Stage6 summary missing policy column: {stage6_summary}')
    row = df[df['policy'].astype(str).eq(c3_policy)].copy()
    if len(row) != 1:
        matches = sorted(df.loc[df['policy'].astype(str).str.startswith(c3_policy, na=False), 'policy'].astype(str).unique())
        raise ValueError(f'Stage6 summary must contain exactly one canonical C3 row {c3_policy!r}; found {len(row)}. Matching policies: {matches}')
    rec: dict[str, float | int | str] = {'stage6_summary_file': str(stage6_summary), 'stage6_c3_policy': c3_policy}
    if 'n' in row.columns:
        rec['stage6_c3_n'] = int(row.iloc[0]['n'])
    for oracle in ORACLES:
        col = f'mean_delta_R_score_{oracle}'
        if col not in row.columns:
            raise ValueError(f'Stage6 summary missing {col}: {stage6_summary}')
        rec[f'stage6_canonical_c3_mean_{oracle}'] = float(row.iloc[0][col])
    return rec

def load_historical_stage6_canonical_means(project_root: Path, *, c3_policy: str=CANONICAL_C3_POLICY) -> dict[str, float | int | str]:
    """Load the immutable thesis Candidate-IQL point reference.

    Archived Stage7--9 runs were evaluated against the historical frozen thesis
    Candidate-IQL checkpoint.  During ``fresh_retrain`` analysis, the active
    Stage6 checkpoint is intentionally different and therefore cannot be the
    hard reference for those archived Stage9 C3 rows.
    """
    reference_path = (Path(project_root).resolve() / REFERENCE_REL).resolve()
    if not reference_path.is_file():
        raise FileNotFoundError(f'Historical RL reference config not found: {reference_path}')
    payload = json.loads(reference_path.read_text(encoding='utf-8-sig'))
    contract = payload.get('stage6_final_candidate_iql')
    if not isinstance(contract, Mapping):
        raise ValueError(f'Historical RL reference config lacks stage6_final_candidate_iql: {reference_path}')
    rec: dict[str, float | int | str] = {'stage6_summary_file': str(reference_path), 'stage6_c3_policy': c3_policy, 'stage6_reference_kind': 'historical_thesis_reference'}
    if 'expected_firm_count' not in contract:
        raise ValueError(f'Historical RL reference lacks expected_firm_count: {reference_path}')
    rec['stage6_c3_n'] = int(contract['expected_firm_count'])
    for oracle in ORACLES:
        col = f'mean_delta_R_score_{oracle}'
        if col not in contract:
            raise ValueError(f'Historical RL reference lacks {col}: {reference_path}')
        rec[f'stage6_canonical_c3_mean_{oracle}'] = float(contract[col])
    return rec

def compute_run_contrasts(input_run: InputRun, *, c3_policy: str=CANONICAL_C3_POLICY) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = _read_table(input_run.comparison_file)
    df = standardize_comparison(raw, c3_policy=c3_policy)
    rows: list[dict] = []
    score_rows: list[dict] = []
    audit_rows: list[dict] = []
    for oracle in ORACLES:
        score_col = detect_score_column(df, oracle)
        c6 = _policy_score_frame(df, 'C6', score_col, c3_policy=c3_policy)
        c4 = _policy_score_frame(df, 'C4', score_col, c3_policy=c3_policy)
        c3 = _policy_score_frame(df, 'C3', score_col, c3_policy=c3_policy)
        scores = c6.merge(c4, on='row_id', how='inner').merge(c3, on='row_id', how='inner')
        if scores.empty:
            raise ValueError(f'No common paired rows for {input_run.run_dir} {oracle}')
        for contrast, left, right in CONTRASTS:
            gap = scores[f'score_{left}'] - scores[f'score_{right}']
            rows.append({'information_condition': input_run.information_condition, 'oracle_backend': oracle, 'contrast': contrast, 'left_policy': left, 'right_policy': right, 'n_pairs': int(gap.notna().sum()), 'n_nonzero_pairs': int((gap.abs() > 0).sum()), 'mean_gap': float(gap.mean()), 'median_gap': float(gap.median()), 'wilcoxon_p_raw': wilcoxon_paired_p(gap), 'comparison_file': str(input_run.comparison_file), 'run_dir': str(input_run.run_dir)})
        score_rows.append({'information_condition': input_run.information_condition, 'oracle_backend': oracle, 'n_pairs': int(len(scores)), 'mean_C6': float(scores['score_C6'].mean()), 'mean_C4': float(scores['score_C4'].mean()), 'mean_C3': float(scores['score_C3'].mean()), 'mean_C6_minus_C3': float((scores['score_C6'] - scores['score_C3']).mean()), 'mean_C6_minus_C4': float((scores['score_C6'] - scores['score_C4']).mean())})
        audit_rows.append({'information_condition': input_run.information_condition, 'oracle_backend': oracle, **_c3_reference_audit(df, score_col, c3_policy=c3_policy)})
    return (pd.DataFrame(rows), pd.DataFrame(score_rows), pd.DataFrame(audit_rows))

def attach_stage6_c3_audit(audit: pd.DataFrame, stage6_means: dict[str, float | int | str] | None, *, tolerance: float, require_stage6: bool, audit_reference_kind: str='active_stage6_summary', active_stage6_means: dict[str, float | int | str] | None=None) -> pd.DataFrame:
    """Attach the hard C3 audit and, when needed, active fresh-RL diagnostics.

    ``stage6_means`` is the hard reference for the archived Stage9 rows.  Under
    ``frozen_replay`` it is the active Stage6 summary.  Under ``fresh_retrain``
    it is the immutable historical thesis reference, while
    ``active_stage6_means`` is recorded only as advisory checkpoint drift.
    """
    out = audit.copy()
    out['c3_audit_reference_kind'] = str(audit_reference_kind)
    if stage6_means is None:
        if require_stage6:
            raise FileNotFoundError('No Stage6 canonical reference was resolved. Use --stage6-summary for frozen replay, or provide the immutable historical reference config for fresh retraining.')
        out['stage6_summary_file'] = ''
        out['stage6_canonical_c3_mean'] = np.nan
        out['exact_minus_stage6'] = np.nan
        out['c3_reference_audit_status'] = 'SKIPPED_STAGE6_SUMMARY_MISSING'
        return out
    out['stage6_summary_file'] = str(stage6_means['stage6_summary_file'])
    out['stage6_c3_policy'] = str(stage6_means['stage6_c3_policy'])
    if 'stage6_c3_n' in stage6_means:
        out['stage6_c3_n'] = int(stage6_means['stage6_c3_n'])
        out['stage9_minus_stage6_n'] = out['n_exact_row_ids'].astype(int) - int(stage6_means['stage6_c3_n'])
    else:
        out['stage6_c3_n'] = np.nan
        out['stage9_minus_stage6_n'] = np.nan
    errors: list[str] = []
    means: list[float] = []
    diffs: list[float] = []
    for _, r in out.iterrows():
        oracle = str(r['oracle_backend'])
        stage6_mean = float(stage6_means[f'stage6_canonical_c3_mean_{oracle}'])
        exact_mean = float(r['stage9_exact_c3_mean_paired'])
        diff = exact_mean - stage6_mean
        means.append(stage6_mean)
        diffs.append(diff)
        if np.isfinite(diff) and abs(diff) > tolerance:
            errors.append(f"{r['information_condition']} {oracle}: exact Stage9 C3 mean {exact_mean:.12g} differs from {audit_reference_kind} C3 mean {stage6_mean:.12g} by {diff:.12g}")
        if 'stage6_c3_n' in stage6_means and int(r['n_exact_row_ids']) != int(stage6_means['stage6_c3_n']):
            errors.append(f"{r['information_condition']} {oracle}: Stage9 exact C3 row_id count {int(r['n_exact_row_ids'])} differs from reference n {int(stage6_means['stage6_c3_n'])}")
    out['stage6_canonical_c3_mean'] = means
    out['exact_minus_stage6'] = diffs
    out['c3_reference_audit_status'] = ['PASS' if np.isfinite(x) and abs(x) <= tolerance else 'FAIL' for x in diffs]
    if active_stage6_means is not None:
        out['active_stage6_summary_file'] = str(active_stage6_means['stage6_summary_file'])
        if 'stage6_c3_n' in active_stage6_means:
            active_n = int(active_stage6_means['stage6_c3_n'])
            out['active_stage6_c3_n'] = active_n
            out['stage9_minus_active_stage6_n'] = out['n_exact_row_ids'].astype(int) - active_n
        active_means: list[float] = []
        active_diffs: list[float] = []
        active_status: list[str] = []
        for _, r in out.iterrows():
            oracle = str(r['oracle_backend'])
            active_mean = float(active_stage6_means[f'stage6_canonical_c3_mean_{oracle}'])
            exact_mean = float(r['stage9_exact_c3_mean_paired'])
            diff = exact_mean - active_mean
            active_means.append(active_mean)
            active_diffs.append(diff)
            active_status.append('PASS' if np.isfinite(diff) and abs(diff) <= tolerance else 'ADVISORY_FRESH_CHECKPOINT_DRIFT')
        out['active_stage6_c3_mean'] = active_means
        out['stage9_exact_minus_active_stage6'] = active_diffs
        out['active_stage6_comparison_status'] = active_status
    if errors:
        raise ValueError('Canonical C3 audit failed. Diagnostic C3 variants must not replace exact C3. ' + ' | '.join(errors))
    return out

def attach_holm(df: pd.DataFrame, family_mode: str) -> pd.DataFrame:
    out = df.copy()
    if family_mode == 'oracle':
        out['holm_family'] = 'N5_7_10c_' + out['oracle_backend'].astype(str)
    elif family_mode == 'oracle_contrast':
        out['holm_family'] = 'N5_7_10c_' + out['oracle_backend'].astype(str) + '_' + out['contrast'].astype(str)
    elif family_mode == 'all':
        out['holm_family'] = 'N5_7_10c_all_oracles'
    else:
        raise ValueError(f'Unknown family mode: {family_mode}')
    out['wilcoxon_p_holm'] = np.nan
    for _, idx in out.groupby('holm_family').groups.items():
        out.loc[idx, 'wilcoxon_p_holm'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].to_list())
    out['sig_holm'] = out['wilcoxon_p_holm'].apply(p_to_stars)
    return out

def fmt_num(x: float, nd: int=3, signed: bool=True) -> str:
    if not np.isfinite(x):
        return ''
    prefix = '+' if signed and x >= 0 else ''
    return f'{prefix}{x:.{nd}f}'

def build_table_patch(scores: pd.DataFrame, infer: pd.DataFrame) -> pd.DataFrame:
    alpha = scores[scores['oracle_backend'].eq('alpha')].copy()
    sig = infer[infer['oracle_backend'].eq('alpha')][['information_condition', 'contrast', 'mean_gap', 'wilcoxon_p_raw', 'wilcoxon_p_holm', 'sig_holm']].copy()
    c3 = sig[sig['contrast'].eq('C6_minus_C3')].rename(columns={'mean_gap': 'C6_minus_C3_alpha_gap', 'wilcoxon_p_raw': 'C6_minus_C3_p_raw', 'wilcoxon_p_holm': 'C6_minus_C3_p_holm', 'sig_holm': 'C6_minus_C3_sig'}).drop(columns=['contrast'])
    c4 = sig[sig['contrast'].eq('C6_minus_C4')].rename(columns={'mean_gap': 'C6_minus_C4_alpha_gap', 'wilcoxon_p_raw': 'C6_minus_C4_p_raw', 'wilcoxon_p_holm': 'C6_minus_C4_p_holm', 'sig_holm': 'C6_minus_C4_sig'}).drop(columns=['contrast'])
    out = alpha[['information_condition', 'n_pairs', 'mean_C6', 'mean_C4', 'mean_C3']].merge(c3, on='information_condition').merge(c4, on='information_condition')
    out['paper_C6_alpha_delta'] = out['mean_C6'].map(lambda x: fmt_num(x))
    out['paper_C3_gap_with_star'] = out.apply(lambda r: f"{fmt_num(r['C6_minus_C3_alpha_gap'])} ({r['C6_minus_C3_sig']})", axis=1)
    out['paper_C6_minus_C4_with_star'] = out.apply(lambda r: f"{fmt_num(r['C6_minus_C4_alpha_gap'])} ({r['C6_minus_C4_sig']})", axis=1)
    return out.sort_values('information_condition').reset_index(drop=True)

def write_markdown_patch(table: pd.DataFrame, path: Path) -> None:
    cols = ['information_condition', 'paper_C6_alpha_delta', 'paper_C3_gap_with_star', 'paper_C6_minus_C4_with_star', 'C6_minus_C3_p_holm', 'C6_minus_C4_p_holm']
    t = table[cols].copy()
    t = t.rename(columns={'information_condition': '정보조건', 'paper_C6_alpha_delta': 'N5 C6 α Δnoop', 'paper_C3_gap_with_star': 'C3 대비 α gap', 'paper_C6_minus_C4_with_star': 'C6-C4 α gap', 'C6_minus_C3_p_holm': 'C6-C3 Holm p', 'C6_minus_C4_p_holm': 'C6-C4 Holm p'})
    lines = []
    lines.append('| ' + ' | '.join(t.columns) + ' |')
    lines.append('|' + '|'.join(['---'] * len(t.columns)) + '|')
    for _, r in t.iterrows():
        vals = []
        for c in t.columns:
            v = r[c]
            if isinstance(v, float):
                vals.append(f'{v:.3e}' if v < 0.001 else f'{v:.6f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    lines.append('')
    lines.append('주: 괄호 안은 Holm 보정 후 유의성이다(*** p<0.001, ** p<0.01, * p<0.05, n.s. 비유의).')
    lines.append('C3 대비 gap은 Stage6 canonical policy=C3_candidate_iql의 exact row만 사용한다. q_argmax/q_rerank 진단 정책은 C3에 섞지 않는다.')
    lines.append('기존 native C6 대비는 서로 다른 live API snapshot bridge 비교이므로 이 표의 Holm family에 넣지 않는다.')
    path.write_text('\n'.join(lines), encoding='utf-8')

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Compute N5 Table 7-10c Wilcoxon/Holm significance markers.')
    ap.add_argument('--project-root', default='.', help='thesis_repo root. Default: current directory')
    ap.add_argument('--run-dirs', nargs='*', default=None, help='Explicit N5 run directories or comparison files')
    ap.add_argument('--out-dir', default=None, help='Output directory. Default: data/analysis_n5/n5_7_10c_holm')
    ap.add_argument('--stage6-summary', default=None, help='Explicit Stage6 final_policy_summary.csv. Default: discover canonical Stage6 path under project-root')
    ap.add_argument('--rl-validation-mode', choices=['auto', 'frozen_replay', 'fresh_retrain'], default='auto', help='Resolve the C3 audit reference. frozen_replay hard-checks archived Stage9 against the active Stage6 summary; fresh_retrain hard-checks against the immutable thesis reference and records the active fresh Stage6 difference as advisory.')
    ap.add_argument('--c3-policy', default=CANONICAL_C3_POLICY, help='Canonical exact C3 policy label. Default: C3_candidate_iql')
    ap.add_argument('--c3-audit-tolerance', type=float, default=1e-09, help='Max allowed absolute difference between Stage9 exact C3 mean and Stage6 canonical mean')
    ap.add_argument('--allow-missing-stage6-summary', action='store_true', help='Allow running without Stage6 canonical summary. Not recommended for paper tables')
    ap.add_argument('--holm-family', choices=['oracle', 'oracle_contrast', 'all'], default='oracle', help='Default oracle: alpha table adjusts 6 tests together; beta/gamma separately. all is more conservative.')
    args = ap.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    run_paths = [Path(p).resolve() for p in args.run_dirs] if args.run_dirs else discover_n5_runs(project_root)
    inputs: list[InputRun] = []
    for p in run_paths:
        run_dir = p if p.is_dir() else p.parent
        inputs.append(InputRun(parse_ic_label(run_dir), run_dir, find_comparison_file(p)))
    resolved_rl_mode, rl_mode_source = resolve_rl_validation_mode(project_root, args.rl_validation_mode)
    stage6_summary = find_stage6_summary(project_root, args.stage6_summary)
    active_stage6_means = None if stage6_summary is None else load_stage6_canonical_means(stage6_summary, c3_policy=args.c3_policy)
    if resolved_rl_mode == 'fresh_retrain':
        stage6_means = load_historical_stage6_canonical_means(project_root, c3_policy=args.c3_policy)
        audit_reference_kind = 'historical_thesis_reference'
        advisory_active_stage6_means = active_stage6_means
    else:
        stage6_means = active_stage6_means
        audit_reference_kind = 'active_frozen_stage6_summary'
        advisory_active_stage6_means = None
    all_rows: list[pd.DataFrame] = []
    all_scores: list[pd.DataFrame] = []
    all_audits: list[pd.DataFrame] = []
    for inp in inputs:
        contrasts, scores, audit = compute_run_contrasts(inp, c3_policy=args.c3_policy)
        all_rows.append(contrasts)
        all_scores.append(scores)
        all_audits.append(audit)
    infer = pd.concat(all_rows, ignore_index=True)
    scores = pd.concat(all_scores, ignore_index=True)
    c3_audit = pd.concat(all_audits, ignore_index=True)
    c3_audit = attach_stage6_c3_audit(c3_audit, stage6_means, tolerance=float(args.c3_audit_tolerance), require_stage6=not bool(args.allow_missing_stage6_summary), audit_reference_kind=audit_reference_kind, active_stage6_means=advisory_active_stage6_means)
    infer = attach_holm(infer, args.holm_family)
    patch = build_table_patch(scores, infer)
    out_dir = Path(args.out_dir).resolve() if args.out_dir else project_root / 'data' / 'analysis_n5' / 'n5_7_10c_holm'
    out_dir.mkdir(parents=True, exist_ok=True)
    infer_path = out_dir / 'n5_7_10c_pairwise_holm.csv'
    scores_path = out_dir / 'n5_7_10c_policy_means.csv'
    c3_audit_path = out_dir / 'n5_7_10c_c3_reference_audit.csv'
    patch_path = out_dir / 'n5_7_10c_table_patch.csv'
    md_path = out_dir / 'n5_7_10c_table_patch.md'
    input_path = out_dir / 'n5_7_10c_input_files.csv'
    manifest_path = out_dir / 'n5_7_10c_holm_manifest.json'
    infer.to_csv(infer_path, index=False, encoding='utf-8-sig')
    scores.to_csv(scores_path, index=False, encoding='utf-8-sig')
    c3_audit.to_csv(c3_audit_path, index=False, encoding='utf-8-sig')
    patch.to_csv(patch_path, index=False, encoding='utf-8-sig')
    write_markdown_patch(patch, md_path)
    manifest = {'status': 'PASS', 'script': Path(__file__).name, 'project_root': str(project_root), 'holm_family': args.holm_family, 'canonical_c3_policy': args.c3_policy, 'rl_validation_mode': resolved_rl_mode, 'rl_validation_mode_source': rl_mode_source, 'c3_audit_reference_kind': audit_reference_kind, 'stage6_summary': str(stage6_summary) if stage6_summary else None, 'historical_reference_path': str(project_root / REFERENCE_REL) if resolved_rl_mode == 'fresh_retrain' else None, 'c3_audit_tolerance': float(args.c3_audit_tolerance), 'outputs': {'pairwise_holm': str(infer_path), 'policy_means': str(scores_path), 'c3_reference_audit': str(c3_audit_path), 'table_patch_csv': str(patch_path), 'table_patch_md': str(md_path), 'input_files': str(input_path)}, 'note': 'Table 7-10c paper markers should use alpha rows from n5_7_10c_table_patch.*. C3 is exact C3_candidate_iql only; q_argmax/q_rerank diagnostic variants are audited and excluded. Native full-run bridge is intentionally not tested here. Frozen replay audits archived C3 against active Stage6; fresh retraining audits against the immutable thesis reference and records active Stage6 drift as advisory.'}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote: {infer_path}')
    print(f'Wrote: {scores_path}')
    print(f'Wrote: {c3_audit_path}')
    print(f'Wrote: {patch_path}')
    print(f'Wrote: {md_path}')
    print('')
    alpha_audit = c3_audit[c3_audit['oracle_backend'].eq('alpha')][['information_condition', 'stage9_exact_c3_mean_paired', 'stage6_canonical_c3_mean', 'stage9_blended_c3_variant_mean', 'exact_minus_stage6', 'blended_minus_exact', 'c3_reference_audit_status']]
    print('[Alpha C3 reference audit]')
    print(alpha_audit.to_string(index=False))
    print('')
    print(patch[['information_condition', 'paper_C6_alpha_delta', 'paper_C3_gap_with_star', 'paper_C6_minus_C4_with_star', 'C6_minus_C3_p_holm', 'C6_minus_C4_p_holm']].to_string(index=False))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
