from __future__ import annotations
"Reference-quality response analysis for Stage 9 revision conditions.\n\nThis post-freeze analysis asks a deliberately narrower question than the H3\nC6-vs-C6X mean contrast: when the shown reference is better than the model's\ninitial action on the frozen evaluator, does the revision move further toward\nthat reference and produce a larger score gain?\n\nThe module is read-only with respect to ``data/final_freeze``.  It combines:\n\n* Stage 7 frozen action availability to define each pair-eligible cohort,\n* Stage 9 revision metrics (C6/C6X/C7, paired to C4/C4/C5), and\n* Stage 6 row-by-candidate multi-oracle values for the shown reference.\n\nIt never calls an LLM API, re-simulates an action, or refits an Oracle.  The\noutputs are descriptive/associational and do not turn H3 into a causal claim.\n"
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
SCHEMA_VERSION = 'reference_quality_acceptance_v4'
COHORT_CONTRACT = 'stage7_pair_eligible_cohort_complete_case_geometry_v1'
REFERENCE_LOOKUP_CONTRACT = 'stage6_fixed_candidate_id_with_c0_noop_alias_v1'
NOOP_POLICY = 'C0_noop'
NOOP_CANDIDATE = 'A0_noop'
NOOP_SCORE_TOLERANCE = 1e-09
ORACLES = ('alpha', 'beta', 'gamma')
REVISION_CONDITIONS = ('C6', 'C6X', 'C7')
BASE_CONDITION_BY_REVISION = {'C6': 'C4', 'C6X': 'C4', 'C7': 'C5'}
MIN_GROUP_N = 30

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _normalise_ic(value: str | None) -> str:
    text = str(value or '').lower().replace('_', '-')
    if text in {'ic-a', 'ica'}:
        return 'IC-a'
    if text in {'ic-b', 'icb'}:
        return 'IC-b'
    if text in {'ic-c', 'icc'}:
        return 'IC-c'
    return str(value or 'unknown')

def _find_exact(run_dir: Path, candidates: Iterable[str], label: str) -> Path:
    hits = [run_dir / rel for rel in candidates if (run_dir / rel).is_file()]
    if not hits:
        names = {Path(x).name for x in candidates}
        hits = [p for p in run_dir.rglob('*') if p.is_file() and p.name in names]
    resolved = sorted({p.resolve() for p in hits})
    if len(resolved) != 1:
        raise FileNotFoundError(f'Expected exactly one {label} under {run_dir}; found {[str(x) for x in resolved]}')
    return resolved[0]

def _find_revision_metrics(run_dir: Path) -> Path:
    return _find_exact(run_dir, ('stage9_llm_rl_comparison/llm_stage9_revision_metrics.csv', 'stage9_policy_comparison/llm_stage9_revision_metrics.csv', 'stage9_llm_rl_comparison.csv/llm_stage9_revision_metrics.csv'), 'Stage9 revision metrics')

def _find_stage7_metadata(run_dir: Path) -> Path:
    return _find_exact(run_dir, ('stage7_llm_action_generation/metadata.json', 'stage7_action_generation/metadata.json'), 'Stage7 metadata')

def _find_stage7_action_table(run_dir: Path) -> Path:
    return _find_exact(run_dir, ('stage7_llm_action_generation/llm_stage7_action_table.parquet', 'stage7_action_generation/llm_stage7_action_table.parquet'), 'Stage7 action table')

def _read_stage6(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Stage6 multi-oracle row table missing: {path}')
    df = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path)
    required = {'row_id', 'policy', 'candidate_id', *[f'delta_R_score_{x}' for x in ORACLES]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Stage6 reference table missing columns: {missing}')
    out = df[['row_id', 'policy', 'candidate_id', *[f'delta_R_score_{x}' for x in ORACLES]]].copy()
    out['row_id'] = pd.to_numeric(out['row_id'], errors='raise').astype(int)
    for col in ('policy', 'candidate_id'):
        out[col] = out[col].astype(str).str.strip()
        if out[col].eq('').any():
            sample = out.loc[out[col].eq(''), ['row_id', 'policy', 'candidate_id']].head(10)
            raise ValueError(f"Stage6 reference table has blank {col}: {sample.to_dict('records')}")
    if out.duplicated(['row_id', 'policy']).any():
        dup = out.loc[out.duplicated(['row_id', 'policy'], keep=False), ['row_id', 'policy', 'candidate_id']].head(10)
        raise ValueError(f"Stage6 reference table has duplicate row_id/policy keys: {dup.to_dict('records')}")
    return out

def _build_stage6_reference_lookup(stage6: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve shown candidate ids to their fixed Stage6 evaluation rows.

    Stage 9 stores the candidate shown to the LLM in
    ``rl_reference_candidate``.  For ordinary fixed candidates the Stage6
    policy label and candidate id are identical.  The no-op is the deliberate
    exception: Stage6 exposes it as policy ``C0_noop`` with candidate id
    ``A0_noop``.  RL-policy rows can carry the same candidate id and must not
    be used as candidate-uniform reference scores.
    """
    score_cols = [f'delta_R_score_{x}' for x in ORACLES]
    bad_noop_policy = stage6[stage6['policy'].eq(NOOP_POLICY) & ~stage6['candidate_id'].eq(NOOP_CANDIDATE)]
    if not bad_noop_policy.empty:
        raise ValueError(f"Stage6 no-op policy alias is inconsistent: {bad_noop_policy[['row_id', 'policy', 'candidate_id']].head(10).to_dict('records')}")
    identity_rows = stage6['policy'].eq(stage6['candidate_id'])
    noop_alias_rows = stage6['policy'].eq(NOOP_POLICY) & stage6['candidate_id'].eq(NOOP_CANDIDATE)
    eligible = stage6.loc[identity_rows | noop_alias_rows].copy()
    if eligible.empty:
        raise ValueError('Stage6 contains no fixed-candidate reference rows')
    noop = eligible.loc[noop_alias_rows.loc[eligible.index]].copy()
    expected_row_ids = set(stage6['row_id'].astype(int).unique().tolist())
    noop_row_ids = set(noop['row_id'].astype(int).unique().tolist())
    if noop_row_ids != expected_row_ids:
        missing = sorted(expected_row_ids - noop_row_ids)[:20]
        extra = sorted(noop_row_ids - expected_row_ids)[:20]
        raise ValueError(f'Stage6 C0_noop/A0_noop alias must cover every evaluation row: missing={missing}, extra={extra}')
    noop_scores = noop[score_cols].apply(pd.to_numeric, errors='coerce')
    bad_noop_scores = ~np.isfinite(noop_scores.to_numpy(dtype=float)) | (np.abs(noop_scores.to_numpy(dtype=float)) > NOOP_SCORE_TOLERANCE)
    if bad_noop_scores.any():
        bad_rows = noop.loc[bad_noop_scores.any(axis=1), ['row_id', 'policy', 'candidate_id', *score_cols]].head(10)
        raise ValueError(f"Stage6 no-op delta_R_score must be finite and exactly zero within tolerance={NOOP_SCORE_TOLERANCE}: {bad_rows.to_dict('records')}")
    eligible['reference_candidate_id'] = eligible['candidate_id']
    eligible['reference_policy'] = eligible['policy']
    duplicate_key = eligible.duplicated(['row_id', 'reference_candidate_id'], keep=False)
    if duplicate_key.any():
        dup = eligible.loc[duplicate_key, ['row_id', 'reference_candidate_id', 'reference_policy', 'candidate_id']].head(20)
        raise ValueError(f"Stage6 fixed-candidate lookup has ambiguous row_id/candidate keys: {dup.to_dict('records')}")
    lookup = eligible[['row_id', 'reference_candidate_id', 'reference_policy', *score_cols]].rename(columns={f'delta_R_score_{bk}': f'reference_delta_R_score_{bk}' for bk in ORACLES})
    metadata = {'contract': REFERENCE_LOOKUP_CONTRACT, 'join_key': ['row_id', 'rl_reference_candidate->reference_candidate_id'], 'stage6_row_count': int(len(stage6)), 'stage6_unique_row_count': int(stage6['row_id'].nunique()), 'eligible_lookup_row_count': int(len(lookup)), 'eligible_reference_candidate_count': int(lookup['reference_candidate_id'].nunique()), 'identity_policy_candidate_row_count': int(identity_rows.sum()), 'c0_a0_alias_row_count': int(noop_alias_rows.sum()), 'excluded_nonfixed_policy_row_count': int((~(identity_rows | noop_alias_rows)).sum()), 'noop_score_tolerance': float(NOOP_SCORE_TOLERANCE), 'reference_candidates': sorted(lookup['reference_candidate_id'].astype(str).unique().tolist())}
    return (lookup, metadata)

@dataclass(frozen=True)
class RunInput:
    run_dir: Path
    run_label: str
    information_condition: str
    revision_metrics: Path
    stage7_metadata: Path
    stage7_action_table: Path

def _resolve_run(run_dir: Path) -> RunInput:
    run_dir = Path(run_dir).resolve()
    revision = _find_revision_metrics(run_dir)
    meta_path = _find_stage7_metadata(run_dir)
    action_table = _find_stage7_action_table(run_dir)
    meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
    run_label = str(meta.get('run_label') or run_dir.name)
    ic = _normalise_ic(meta.get('information_condition'))
    return RunInput(run_dir, run_label, ic, revision, meta_path, action_table)

def _build_rows(lookup: pd.DataFrame, runs: list[RunInput]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    required_revision = {'row_id', 'base_condition', 'revision_condition', 'mode', 'rl_reference_candidate', 'rl_adoption_ratio', 'orthogonal_drift', 'u_norm_squared', 'metrics_defined', *[f'initial_delta_R_score_{x}' for x in ORACLES], *[f'revised_delta_R_score_{x}' for x in ORACLES], *[f'revision_delta_R_score_{x}' for x in ORACLES]}
    for run in runs:
        rev = pd.read_csv(run.revision_metrics)
        missing = sorted(required_revision - set(rev.columns))
        if missing:
            raise ValueError(f'{run.run_label}: revision metrics missing columns: {missing}')
        rev = rev[rev['revision_condition'].astype(str).isin(REVISION_CONDITIONS)].copy()
        if rev.empty:
            raise ValueError(f'{run.run_label}: no C6/C6X/C7 revision rows')
        rev['row_id'] = pd.to_numeric(rev['row_id'], errors='raise').astype(int)
        rev['metrics_defined'] = rev['metrics_defined'].fillna(False).astype(bool)
        rev['rl_reference_candidate'] = rev['rl_reference_candidate'].astype(str).str.strip()
        blank_reference = rev['rl_reference_candidate'].isin({'', 'None', 'nan'})
        if blank_reference.any():
            sample = rev.loc[blank_reference, ['row_id', 'revision_condition', 'mode', 'metrics_defined', 'undefined_reason']].head(10)
            raise ValueError(f"{run.run_label}: reference-quality rows require an explicit shown reference candidate: {sample.to_dict('records')}")
        if rev.duplicated(['row_id', 'revision_condition', 'mode']).any():
            raise ValueError(f'{run.run_label}: duplicate row/condition/mode revision metrics')
        merged = rev.merge(lookup, left_on=['row_id', 'rl_reference_candidate'], right_on=['row_id', 'reference_candidate_id'], how='left', validate='many_to_one', indicator=True)
        misses = merged['_merge'].ne('both')
        if misses.any():
            sample = merged.loc[misses, ['row_id', 'revision_condition', 'rl_reference_candidate']].head(10)
            raise ValueError(f"{run.run_label}: shown reference missing from Stage6 row-candidate lookup: {sample.to_dict('records')}")
        merged = merged.drop(columns=['_merge', 'reference_candidate_id'])
        merged.insert(0, 'run_label', run.run_label)
        merged.insert(1, 'information_condition', run.information_condition)
        merged['reference_distance'] = np.sqrt(pd.to_numeric(merged['u_norm_squared'], errors='coerce').clip(lower=0.0))
        for bk in ORACLES:
            merged[f'reference_advantage_{bk}'] = pd.to_numeric(merged[f'reference_delta_R_score_{bk}'], errors='coerce') - pd.to_numeric(merged[f'initial_delta_R_score_{bk}'], errors='coerce')
            merged[f'reference_better_{bk}'] = merged[f'reference_advantage_{bk}'] > 0
        rows.append(merged)
    out = pd.concat(rows, ignore_index=True, sort=False)
    if out.empty:
        raise ValueError('Reference-quality analysis produced no rows')
    return out

def _read_stage7_availability(run: RunInput, *, expected_row_ids: set[int]) -> pd.DataFrame:
    path = run.stage7_action_table
    frame = pd.read_parquet(path) if path.suffix.lower() == '.parquet' else pd.read_csv(path)
    required = {'row_id', 'policy', 'mode'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'{run.run_label}: Stage7 action table missing columns: {missing}')
    out = frame[['row_id', 'policy', 'mode']].copy()
    out['row_id'] = pd.to_numeric(out['row_id'], errors='raise').astype(int)
    for col in ('policy', 'mode'):
        out[col] = out[col].astype(str).str.strip()
        if out[col].isin({'', 'None', 'nan'}).any():
            sample = out.loc[out[col].isin({'', 'None', 'nan'}), ['row_id', 'policy', 'mode']].head(10)
            raise ValueError(f"{run.run_label}: Stage7 action table has blank {col}: {sample.to_dict('records')}")
    tracked = set(BASE_CONDITION_BY_REVISION) | set(BASE_CONDITION_BY_REVISION.values())
    out = out[out['policy'].isin(tracked)].copy()
    if out.empty:
        raise ValueError(f'{run.run_label}: Stage7 action table has no tracked revision/base conditions')
    duplicate = out.duplicated(['row_id', 'policy', 'mode'], keep=False)
    if duplicate.any():
        sample = out.loc[duplicate, ['row_id', 'policy', 'mode']].head(20)
        raise ValueError(f"{run.run_label}: duplicate Stage7 row/condition/mode keys: {sample.to_dict('records')}")
    stage7_ids = set(out['row_id'].tolist())
    extra = sorted(stage7_ids - expected_row_ids)
    if extra:
        raise ValueError(f'{run.run_label}: Stage7 row ids fall outside the frozen Stage6 evaluation universe: {extra[:20]}')
    return out

def _audit_stage7_paired_cohort(rows: pd.DataFrame, *, runs: list[RunInput], expected_row_ids: set[int]) -> dict[str, Any]:
    """Verify Stage9 against the pair-eligible cohort in each frozen Stage7 run.

    Stage7 availability determines whether a revision comparison exists.  For
    example, C7 requires both C5 and C7 for the same row and mode.  Stage6
    remains the frozen evaluator universe and reference-score lookup, but it
    does not manufacture missing LLM base/revision pairs.  Within every valid
    pair, ``metrics_defined`` controls only geometric complete-case analyses.
    """
    if not expected_row_ids:
        raise ValueError('Reference-quality cohort contract received no Stage6 row ids')
    records: list[dict[str, Any]] = []
    expected_cells: set[tuple[str, str, str]] = set()
    observed_cells: set[tuple[str, str, str]] = set()
    for run in runs:
        stage7 = _read_stage7_availability(run, expected_row_ids=expected_row_ids)
        run_rows = rows[rows['run_label'].astype(str).eq(run.run_label)].copy()
        if run_rows.empty:
            raise ValueError(f'{run.run_label}: no Stage9 revision rows survived reference lookup')
        observed_run_cells = {(str(c), str(m)) for c, m in run_rows[['revision_condition', 'mode']].drop_duplicates().itertuples(index=False, name=None)}
        stage7_modes = sorted(stage7['mode'].astype(str).unique().tolist())
        for revision_condition in REVISION_CONDITIONS:
            base_condition = BASE_CONDITION_BY_REVISION[revision_condition]
            for mode in stage7_modes:
                base_ids = set(stage7.loc[stage7['policy'].eq(base_condition) & stage7['mode'].eq(mode), 'row_id'].astype(int).tolist())
                revision_ids = set(stage7.loc[stage7['policy'].eq(revision_condition) & stage7['mode'].eq(mode), 'row_id'].astype(int).tolist())
                if not revision_ids:
                    continue
                pair_ids = base_ids & revision_ids
                cell_key = (run.run_label, revision_condition, mode)
                expected_cells.add(cell_key)
                if not pair_ids:
                    raise ValueError(f'{run.run_label}: Stage7 has no pair-eligible rows for base={base_condition}, revision={revision_condition}, mode={mode}')
                group = run_rows[run_rows['revision_condition'].astype(str).eq(revision_condition) & run_rows['mode'].astype(str).eq(mode)].copy()
                stage9_ids = set(pd.to_numeric(group['row_id'], errors='raise').astype(int).tolist())
                duplicate_count = int(group['row_id'].duplicated().sum())
                missing = sorted(pair_ids - stage9_ids)
                extra = sorted(stage9_ids - pair_ids)
                base_values = sorted(group['base_condition'].astype(str).unique().tolist())
                if duplicate_count or missing or extra or (len(group) != len(pair_ids)):
                    raise ValueError(f'Reference-quality Stage7/Stage9 paired-cohort contract failed for run={run.run_label}, base={base_condition}, revision={revision_condition}, mode={mode}: pair_eligible={len(pair_ids)}, stage9_rows={len(group)}, duplicates={duplicate_count}, missing={missing[:20]}, extra={extra[:20]}')
                if base_values != [base_condition]:
                    raise ValueError(f'{run.run_label}: Stage9 base_condition mismatch for revision={revision_condition}, mode={mode}: {base_values}, expected={[base_condition]}')
                observed_cells.add(cell_key)
                defined = group['metrics_defined'].fillna(False).astype(bool)
                target_count = len(expected_row_ids)
                records.append({'run_label': run.run_label, 'information_condition': run.information_condition, 'base_condition': base_condition, 'revision_condition': revision_condition, 'mode': mode, 'target_firm_count': int(target_count), 'base_available_count': int(len(base_ids)), 'revision_available_count': int(len(revision_ids)), 'pair_eligible_count': int(len(pair_ids)), 'stage9_metric_row_count': int(len(group)), 'missing_base_count': int(target_count - len(base_ids)), 'missing_revision_count': int(target_count - len(revision_ids)), 'missing_both_count': int(target_count - len(base_ids | revision_ids)), 'base_only_count': int(len(base_ids - revision_ids)), 'revision_only_count': int(len(revision_ids - base_ids)), 'pair_coverage_rate': float(len(pair_ids) / target_count), 'n_metrics_defined': int(defined.sum()), 'n_metrics_undefined': int((~defined).sum()), 'geometry_complete_case_rate': float(defined.mean())})
        run_expected_cells = {(condition, mode) for label, condition, mode in expected_cells if label == run.run_label}
        unexpected = sorted(observed_run_cells - run_expected_cells)
        if unexpected:
            raise ValueError(f'{run.run_label}: Stage9 contains cells not supported by Stage7: {unexpected}')
    if expected_cells != observed_cells:
        missing_cells = sorted(expected_cells - observed_cells)
        extra_cells = sorted(observed_cells - expected_cells)
        raise ValueError(f'Reference-quality Stage7/Stage9 cell inventory mismatch: missing={missing_cells}, extra={extra_cells}')
    if not records:
        raise ValueError('Reference-quality paired-cohort audit produced no cells')
    audit = pd.DataFrame(records)
    return {'contract': COHORT_CONTRACT, 'target_row_count_per_run': int(len(expected_row_ids)), 'observed_run_count': int(len(runs)), 'observed_cell_count': int(len(audit)), 'min_pair_eligible_count': int(audit['pair_eligible_count'].min()), 'max_pair_eligible_count': int(audit['pair_eligible_count'].max()), 'stage7_pair_eligible_row_count': int(audit['pair_eligible_count'].sum()), 'stage9_metric_row_count': int(audit['stage9_metric_row_count'].sum()), 'metrics_defined_row_count': int(audit['n_metrics_defined'].sum()), 'metrics_undefined_row_count': int(audit['n_metrics_undefined'].sum()), 'base_condition_by_revision': dict(BASE_CONDITION_BY_REVISION), 'cells': records}

def _holm_adjust(p_values: pd.Series) -> pd.Series:
    vals = pd.to_numeric(p_values, errors='coerce').to_numpy(dtype=float)
    out = np.full(len(vals), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(vals))
    if not len(valid):
        return pd.Series(out, index=p_values.index)
    order = valid[np.argsort(vals[valid], kind='mergesort')]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        adjusted = min(1.0, vals[idx] * (m - rank))
        running = max(running, adjusted)
        out[idx] = running
    return pd.Series(out, index=p_values.index)

def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, *, seed: int, draws: int) -> tuple[float, float]:
    if draws <= 0 or len(x) < 4:
        return (float('nan'), float('nan'))
    rng = np.random.default_rng(seed)
    stats = np.empty(draws, dtype=float)
    n = len(x)
    for i in range(draws):
        idx = rng.integers(0, n, n)
        rho = spearmanr(x[idx], y[idx], nan_policy='omit').statistic
        stats[i] = float(rho) if np.isfinite(rho) else np.nan
    stats = stats[np.isfinite(stats)]
    if not len(stats):
        return (float('nan'), float('nan'))
    return (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))

def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[int, float, float, np.ndarray, np.ndarray]:
    frame = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': pd.to_numeric(y, errors='coerce')}).dropna()
    if len(frame) < MIN_GROUP_N or frame['x'].nunique() < 2 or frame['y'].nunique() < 2:
        return (len(frame), float('nan'), float('nan'), frame['x'].to_numpy(), frame['y'].to_numpy())
    result = spearmanr(frame['x'], frame['y'], nan_policy='omit')
    return (len(frame), float(result.statistic), float(result.pvalue), frame['x'].to_numpy(), frame['y'].to_numpy())

def _summarise(rows: pd.DataFrame, *, bootstrap_draws: int, bootstrap_seed: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_cols = ['run_label', 'information_condition', 'revision_condition', 'mode']
    for group_index, (keys, group) in enumerate(rows.groupby(group_cols, sort=True)):
        run_label, ic, condition, mode = keys
        adoption = pd.to_numeric(group['rl_adoption_ratio'], errors='coerce')
        for oracle_index, bk in enumerate(ORACLES):
            advantage = pd.to_numeric(group[f'reference_advantage_{bk}'], errors='coerce')
            gain = pd.to_numeric(group[f'revision_delta_R_score_{bk}'], errors='coerce')
            n_adopt, rho_adopt, p_adopt, xa, ya = _safe_spearman(advantage, adoption)
            n_gain, rho_gain, p_gain, xg, yg = _safe_spearman(advantage, gain)
            seed_base = int(bootstrap_seed + group_index * 101 + oracle_index * 10007)
            adopt_lo, adopt_hi = _bootstrap_spearman(xa, ya, seed=seed_base, draws=bootstrap_draws)
            gain_lo, gain_hi = _bootstrap_spearman(xg, yg, seed=seed_base + 1, draws=bootstrap_draws)
            valid = pd.DataFrame({'advantage': advantage, 'adoption': adoption, 'gain': gain, 'distance': pd.to_numeric(group['reference_distance'], errors='coerce')}).dropna(subset=['advantage'])
            better = valid['advantage'] > 0
            metrics_defined = group['metrics_defined'].fillna(False).astype(bool)
            records.append({'run_label': run_label, 'information_condition': ic, 'revision_condition': condition, 'mode': mode, 'oracle_backend': bk, 'n_rows': int(len(group)), 'n_metrics_defined': int(metrics_defined.sum()), 'n_metrics_undefined': int((~metrics_defined).sum()), 'metrics_defined_fraction': float(metrics_defined.mean()), 'n_reference_advantage_defined': int(valid['advantage'].notna().sum()), 'n_reference_better': int(better.sum()), 'reference_better_fraction': float(better.mean()) if len(valid) else float('nan'), 'mean_reference_advantage': float(valid['advantage'].mean()) if len(valid) else float('nan'), 'median_reference_advantage': float(valid['advantage'].median()) if len(valid) else float('nan'), 'mean_reference_distance': float(valid['distance'].mean()) if valid['distance'].notna().any() else float('nan'), 'mean_adoption_reference_better': float(valid.loc[better, 'adoption'].mean()) if valid.loc[better, 'adoption'].notna().any() else float('nan'), 'mean_adoption_reference_not_better': float(valid.loc[~better, 'adoption'].mean()) if valid.loc[~better, 'adoption'].notna().any() else float('nan'), 'adoption_group_gap': float(valid.loc[better, 'adoption'].mean() - valid.loc[~better, 'adoption'].mean()) if valid.loc[better, 'adoption'].notna().any() and valid.loc[~better, 'adoption'].notna().any() else float('nan'), 'mean_revision_gain_reference_better': float(valid.loc[better, 'gain'].mean()) if valid.loc[better, 'gain'].notna().any() else float('nan'), 'mean_revision_gain_reference_not_better': float(valid.loc[~better, 'gain'].mean()) if valid.loc[~better, 'gain'].notna().any() else float('nan'), 'revision_gain_group_gap': float(valid.loc[better, 'gain'].mean() - valid.loc[~better, 'gain'].mean()) if valid.loc[better, 'gain'].notna().any() and valid.loc[~better, 'gain'].notna().any() else float('nan'), 'n_spearman_adoption': int(n_adopt), 'rho_reference_advantage_vs_adoption': rho_adopt, 'rho_adoption_ci_lo': adopt_lo, 'rho_adoption_ci_hi': adopt_hi, 'p_reference_advantage_vs_adoption_raw': p_adopt, 'n_spearman_revision_gain': int(n_gain), 'rho_reference_advantage_vs_revision_gain': rho_gain, 'rho_revision_gain_ci_lo': gain_lo, 'rho_revision_gain_ci_hi': gain_hi, 'p_reference_advantage_vs_revision_gain_raw': p_gain})
    summary = pd.DataFrame(records)
    if summary.empty:
        raise ValueError('Reference-quality summary is empty')
    for outcome in ('adoption', 'revision_gain'):
        raw = f'p_reference_advantage_vs_{outcome}_raw'
        holm = f'p_reference_advantage_vs_{outcome}_holm'
        summary[holm] = np.nan
        for _, idx in summary.groupby(['revision_condition', 'mode'], sort=False).groups.items():
            summary.loc[idx, holm] = _holm_adjust(summary.loc[idx, raw]).to_numpy()
        summary[f'sig_reference_advantage_vs_{outcome}_holm'] = summary[holm].map(lambda p: '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else 'n.s.' if pd.notna(p) else 'NA')
    return summary.sort_values(group_cols + ['oracle_backend']).reset_index(drop=True)

def _make_bins(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    group_cols = ['run_label', 'information_condition', 'revision_condition', 'mode']
    for keys, group in rows.groupby(group_cols, sort=True):
        for bk in ORACLES:
            frame = pd.DataFrame({'advantage': pd.to_numeric(group[f'reference_advantage_{bk}'], errors='coerce'), 'adoption': pd.to_numeric(group['rl_adoption_ratio'], errors='coerce'), 'gain': pd.to_numeric(group[f'revision_delta_R_score_{bk}'], errors='coerce'), 'distance': pd.to_numeric(group['reference_distance'], errors='coerce')}).dropna(subset=['advantage'])
            if len(frame) < MIN_GROUP_N:
                continue
            ranks = frame['advantage'].rank(method='first')
            frame['reference_quality_quartile'] = pd.qcut(ranks, q=4, labels=[1, 2, 3, 4]).astype(int)
            for quartile, cell in frame.groupby('reference_quality_quartile', sort=True):
                records.append({'run_label': keys[0], 'information_condition': keys[1], 'revision_condition': keys[2], 'mode': keys[3], 'oracle_backend': bk, 'reference_quality_quartile': int(quartile), 'n_rows': int(len(cell)), 'mean_reference_advantage': float(cell['advantage'].mean()), 'median_reference_advantage': float(cell['advantage'].median()), 'mean_rl_adoption_ratio': float(cell['adoption'].mean()), 'median_rl_adoption_ratio': float(cell['adoption'].median()), 'mean_revision_gain': float(cell['gain'].mean()), 'median_revision_gain': float(cell['gain'].median()), 'mean_reference_distance': float(cell['distance'].mean())})
    return pd.DataFrame(records)

def _primary_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = ['run_label', 'information_condition', 'mode', 'oracle_backend', 'n_rows', 'n_metrics_defined', 'n_metrics_undefined', 'metrics_defined_fraction', 'n_reference_advantage_defined', 'reference_better_fraction', 'mean_reference_advantage', 'n_spearman_adoption', 'rho_reference_advantage_vs_adoption', 'rho_adoption_ci_lo', 'rho_adoption_ci_hi', 'p_reference_advantage_vs_adoption_holm', 'sig_reference_advantage_vs_adoption_holm', 'n_spearman_revision_gain', 'rho_reference_advantage_vs_revision_gain', 'rho_revision_gain_ci_lo', 'rho_revision_gain_ci_hi', 'p_reference_advantage_vs_revision_gain_holm', 'sig_reference_advantage_vs_revision_gain_holm', 'adoption_group_gap', 'revision_gain_group_gap']
    out = summary[summary['revision_condition'].astype(str).eq('C6')].copy()
    return out[[c for c in cols if c in out.columns]].reset_index(drop=True)

def run_analysis(*, project_root: Path, run_dirs: Iterable[Path], output_dir: Path, stage6_path: Path | None=None, bootstrap_draws: int=1000, bootstrap_seed: int=20260711) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage6_path = Path(stage6_path).resolve() if stage6_path else root / 'data' / 'final_freeze' / 'stage6_candidate_selector_eval' / 'multi_oracle_policy_eval.parquet'
    resolved_runs = [_resolve_run(Path(x)) for x in run_dirs]
    if not resolved_runs:
        raise ValueError('At least one --run-dir is required')
    labels = [x.run_label for x in resolved_runs]
    if len(set(labels)) != len(labels):
        raise ValueError(f'Duplicate run labels: {labels}')
    manifest_path = output_dir / 'reference_quality_acceptance_manifest.json'
    manifest: dict[str, Any] = {'schema_version': SCHEMA_VERSION, 'created_utc': _now(), 'status': 'RUNNING', 'project_root': str(root), 'output_dir': str(output_dir), 'stage6_path': str(stage6_path), 'bootstrap_draws': int(bootstrap_draws), 'bootstrap_seed': int(bootstrap_seed), 'interpretation_boundary': 'Associational response analysis; does not identify causal quality discrimination.', 'runs': [{'run_dir': str(x.run_dir), 'run_label': x.run_label, 'information_condition': x.information_condition, 'revision_metrics': str(x.revision_metrics), 'stage7_metadata': str(x.stage7_metadata), 'stage7_action_table': str(x.stage7_action_table)} for x in resolved_runs]}
    _write_json(manifest_path, manifest)
    try:
        stage6 = _read_stage6(stage6_path)
        reference_lookup, lookup_metadata = _build_stage6_reference_lookup(stage6)
        manifest['stage6_reference_lookup'] = lookup_metadata
        rows = _build_rows(reference_lookup, resolved_runs)
        cohort_audit = _audit_stage7_paired_cohort(rows, runs=resolved_runs, expected_row_ids=set(stage6['row_id'].astype(int).unique().tolist()))
        manifest['cohort_contract'] = cohort_audit
        summary = _summarise(rows, bootstrap_draws=int(bootstrap_draws), bootstrap_seed=int(bootstrap_seed))
        bins = _make_bins(rows)
        primary = _primary_table(summary)
        row_path = output_dir / 'reference_quality_acceptance_rows.parquet'
        summary_path = output_dir / 'reference_quality_acceptance_summary.csv'
        bins_path = output_dir / 'reference_quality_acceptance_quartiles.csv'
        primary_path = output_dir / 'reference_quality_acceptance_primary_c6.csv'
        rows.to_parquet(row_path, index=False)
        summary.to_csv(summary_path, index=False, encoding='utf-8-sig')
        bins.to_csv(bins_path, index=False, encoding='utf-8-sig')
        primary.to_csv(primary_path, index=False, encoding='utf-8-sig')
        manifest.update({'status': 'PASS', 'completed_utc': _now(), 'row_count': int(len(rows)), 'summary_row_count': int(len(summary)), 'quartile_row_count': int(len(bins)), 'primary_c6_row_count': int(len(primary)), 'revision_conditions': sorted(rows['revision_condition'].astype(str).unique().tolist()), 'modes': sorted(rows['mode'].astype(str).unique().tolist()), 'oracles': list(ORACLES), 'shown_reference_candidate_counts': {str(k): int(v) for k, v in rows['rl_reference_candidate'].astype(str).value_counts().sort_index().items()}, 'shown_reference_policy_counts': {str(k): int(v) for k, v in rows['reference_policy'].astype(str).value_counts().sort_index().items()}, 'shown_noop_reference_row_count': int(rows['rl_reference_candidate'].astype(str).eq(NOOP_CANDIDATE).sum()), 'metrics_defined_row_count': int(rows['metrics_defined'].fillna(False).astype(bool).sum()), 'metrics_undefined_row_count': int((~rows['metrics_defined'].fillna(False).astype(bool)).sum()), 'undefined_reason_counts': {str(k): int(v) for k, v in rows.loc[~rows['metrics_defined'].fillna(False).astype(bool), 'undefined_reason'].fillna('UNSPECIFIED').astype(str).value_counts().sort_index().items()}, 'outputs': {'rows': row_path.name, 'summary': summary_path.name, 'quartiles': bins_path.name, 'primary_c6': primary_path.name}})
    except Exception as exc:
        manifest.update({'status': 'FAIL', 'completed_utc': _now(), 'error': repr(exc)})
        raise
    finally:
        _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--run-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--stage6-path', default=None)
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20260711)
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_analysis(project_root=Path(args.project_root), run_dirs=[Path(x) for x in args.run_dirs], output_dir=Path(args.output_dir), stage6_path=Path(args.stage6_path) if args.stage6_path else None, bootstrap_draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
