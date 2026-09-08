from __future__ import annotations
'Post-hoc diagnostics for the canonical N5M matched-budget C4/C6 frontier.\n\nThis module is deliberately separate from the common-backend harness-vs-model\nvariance decomposition.  N5M varies budget only inside GPT-5.4-mini, IC-b,\nfree-form C4/C6, seed 1; it therefore cannot identify a backend main effect or\nbe pooled into the crossed backend/harness panel.  The optional within-panel\nvariance output is descriptive and supplementary only.\n\nThe module is read-only with respect to ``data/final_freeze``.  It consumes the\nfour frozen N5M archives plus the frozen Stage6 fixed-candidate score table and\nwrites analysis artifacts under ``data/analysis/paper_repro``.\n'
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from credit_recourse.analysis.n5_7_10c_holm_inference import CANONICAL_C3_POLICY, ORACLES, holm_adjust, p_to_stars, wilcoxon_paired_p
from credit_recourse.analysis.n5_budget_frontier_holm_inference import FrontierArm, validate_frontier_runs
from credit_recourse.analysis.reference_quality_acceptance import _build_stage6_reference_lookup, _read_stage6
from credit_recourse.rl.common.actions import load_action_space, resolve_candidate_library_path
from credit_recourse.rl.common.io import load_yaml
SCHEMA_VERSION = 'n5m_posthoc_v4'
RUN_ROLE = 'paper_n5_matched_budget_frontier_icb'
INFORMATION_CONDITION = 'IC-b'
EXPECTED_BUDGETS: tuple[float | None, ...] = (0.75, 1.27, 2.0, None)
MODE = 'free_form_10d'
POLICIES = ('C4', 'C6')
ACTION_TOLERANCE = 1e-12
SCORE_TIE_TOLERANCE = 1e-12

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _budget_label(value: float | None) -> str:
    return 'unbounded' if value is None else f'{float(value):.2f}'.replace('.', 'p')

def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    return pd.read_parquet(path)

def _wilson(k: int, n: int, z: float=1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))

def _spearman(x: pd.Series, y: pd.Series) -> tuple[int, float, float]:
    frame = pd.DataFrame({'x': pd.to_numeric(x, errors='coerce'), 'y': pd.to_numeric(y, errors='coerce')}).dropna()
    if len(frame) < 3 or frame['x'].nunique() < 2 or frame['y'].nunique() < 2:
        return (int(len(frame)), float('nan'), float('nan'))
    result = spearmanr(frame['x'].to_numpy(float), frame['y'].to_numpy(float))
    return (int(len(frame)), float(result.statistic), float(result.pvalue))

def _numeric_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors='coerce').dropna()
    return float(numeric.mean()) if not numeric.empty else float('nan')

def _numeric_median(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors='coerce').dropna()
    return float(numeric.median()) if not numeric.empty else float('nan')

def _concat_with_observed_columns(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate arm frames without carrying columns that are globally all-null.

    Frozen Stage7/8/9 tables contain optional probe columns that are intentionally
    null for every N5M row.  Dropping only columns that are all-null across every
    arm preserves all observed information and avoids pandas' deprecated dtype
    inference for all-null join units.
    """
    if not frames:
        raise ValueError('Cannot concatenate an empty N5M frame list')
    column_order: list[str] = []
    for frame in frames:
        for column in frame.columns:
            if column not in column_order:
                column_order.append(column)
    keep = [column for column in column_order if any((column in frame.columns and frame[column].notna().any() for frame in frames))]
    records: list[dict[str, Any]] = []
    for frame in frames:
        records.extend(frame.reindex(columns=keep).to_dict(orient='records'))
    return pd.DataFrame.from_records(records, columns=keep)

def _read_arm_inputs(arm: FrontierArm) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stage7 = _read_parquet_or_csv(arm.action_table)
    revision_path = arm.run_dir / 'stage9_llm_rl_comparison' / 'llm_stage9_revision_metrics.csv'
    feasibility_path = arm.run_dir / 'stage8_llm_multi_oracle_eval' / 'llm_stage8_failure_audit_enriched.csv'
    if not revision_path.is_file():
        raise FileNotFoundError(f'N5M revision metrics missing: {revision_path}')
    if not feasibility_path.is_file():
        raise FileNotFoundError(f'N5M feasibility audit missing: {feasibility_path}')
    revision = pd.read_csv(revision_path)
    feasibility = pd.read_csv(feasibility_path)
    return (stage7, revision, feasibility)

def _validate_and_build_panel(arms: list[FrontierArm]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[str]]:
    row_universe: set[int] | None = None
    action_columns: list[str] | None = None
    action_rows: list[pd.DataFrame] = []
    revision_rows: list[pd.DataFrame] = []
    feasibility_rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for arm in arms:
        stage7, revision, feasibility = _read_arm_inputs(arm)
        required_stage7 = {'row_id', 'policy', 'mode', 'projection_distance', 'budget_l1_target', 'budget_compliant_clipped'}
        missing_stage7 = sorted(required_stage7 - set(stage7.columns))
        if missing_stage7:
            raise ValueError(f'{arm.run_label}: Stage7 missing columns: {missing_stage7}')
        current_actions = [c for c in stage7.columns if str(c).startswith('action__')]
        if not current_actions:
            raise ValueError(f'{arm.run_label}: Stage7 contains no action__ columns')
        if action_columns is None:
            action_columns = current_actions
        elif current_actions != action_columns:
            raise ValueError(f'{arm.run_label}: action column order differs across N5M arms: {current_actions} != {action_columns}')
        s7 = stage7.loc[stage7['policy'].astype(str).isin(POLICIES) & stage7['mode'].astype(str).eq(MODE)].copy()
        s7['row_id'] = pd.to_numeric(s7['row_id'], errors='raise').astype(int)
        if len(s7) != 1150 or s7.duplicated(['row_id', 'policy', 'mode']).any():
            raise ValueError(f'{arm.run_label}: expected 1,150 unique C4/C6 free-form Stage7 rows, found {len(s7)}')
        sets = {policy: set(s7.loc[s7['policy'].eq(policy), 'row_id'].tolist()) for policy in POLICIES}
        if any((len(values) != 575 for values in sets.values())) or sets['C4'] != sets['C6']:
            raise ValueError(f'{arm.run_label}: Stage7 C4/C6 row alignment failed')
        if row_universe is None:
            row_universe = sets['C4']
        elif sets['C4'] != row_universe:
            raise ValueError(f'{arm.run_label}: N5M arm row universe differs from the other arms')
        required_revision = {'row_id', 'base_condition', 'revision_condition', 'mode', 'rl_reference_candidate', 'revision_l1_distance', 'rl_adoption_ratio', 'metrics_defined', *[f'initial_delta_R_score_{oracle}' for oracle in ORACLES], *[f'revised_delta_R_score_{oracle}' for oracle in ORACLES], *[f'revision_delta_R_score_{oracle}' for oracle in ORACLES]}
        missing_revision = sorted(required_revision - set(revision.columns))
        if missing_revision:
            raise ValueError(f'{arm.run_label}: Stage9 revision metrics missing columns: {missing_revision}')
        rev = revision.loc[revision['base_condition'].astype(str).eq('C4') & revision['revision_condition'].astype(str).eq('C6') & revision['mode'].astype(str).eq(MODE)].copy()
        rev['row_id'] = pd.to_numeric(rev['row_id'], errors='raise').astype(int)
        if len(rev) != 575 or rev.duplicated(['row_id', 'revision_condition', 'mode']).any():
            raise ValueError(f'{arm.run_label}: expected 575 unique C4->C6 Stage9 rows, found {len(rev)}')
        if set(rev['row_id'].tolist()) != row_universe:
            raise ValueError(f'{arm.run_label}: Stage9 row ids do not match the Stage7 C4/C6 pair')
        required_feasibility = {'row_id', 'policy', 'mode', 'sustainability', 'feasibility_core_violation_flag', 'accounting_check_failed', 'negative_balance_flag', 'simulator_preflight_status'}
        missing_feasibility = sorted(required_feasibility - set(feasibility.columns))
        if missing_feasibility:
            raise ValueError(f'{arm.run_label}: Stage8 feasibility audit missing columns: {missing_feasibility}')
        feas = feasibility.loc[feasibility['policy'].astype(str).isin(POLICIES) & feasibility['mode'].astype(str).eq(MODE)].copy()
        feas['row_id'] = pd.to_numeric(feas['row_id'], errors='raise').astype(int)
        if len(feas) != 1150 or feas.duplicated(['row_id', 'policy', 'mode']).any():
            raise ValueError(f'{arm.run_label}: expected 1,150 unique Stage8 feasibility rows, found {len(feas)}')
        if set(feas['row_id'].tolist()) != row_universe:
            raise ValueError(f'{arm.run_label}: Stage8 feasibility row ids do not match N5M row universe')
        label = arm.budget_label
        budget_value = np.nan if arm.budget is None else float(arm.budget)
        for frame in (s7, rev, feas):
            frame.insert(0, 'run_label', arm.run_label)
            frame.insert(1, 'budget_label', label)
            frame.insert(2, 'l1_budget', budget_value)
        action_rows.append(s7)
        revision_rows.append(rev)
        feasibility_rows.append(feas)
        audit.append({'run_label': arm.run_label, 'budget_label': label, 'l1_budget': arm.budget, 'stage7_row_count': int(len(s7)), 'stage7_unique_firm_count': int(s7['row_id'].nunique()), 'stage9_revision_row_count': int(len(rev)), 'stage8_feasibility_row_count': int(len(feas)), 'row_alignment_status': 'PASS'})
    assert action_columns is not None and row_universe is not None
    return (_concat_with_observed_columns(action_rows), _concat_with_observed_columns(revision_rows), _concat_with_observed_columns(feasibility_rows), audit, action_columns)

def _load_action_widths(project_root: Path, action_columns: list[str]) -> np.ndarray:
    contract_path = project_root / 'data' / 'final_freeze' / 'configs' / 'final_action_contract.yaml'
    if not contract_path.is_file():
        raise FileNotFoundError(f'Frozen final action contract missing: {contract_path}')
    payload = load_yaml(contract_path)
    raw_bounds = payload.get('action_bounds') or {}
    widths: list[float] = []
    for column in action_columns:
        key = str(column)[len('action__'):] if str(column).startswith('action__') else str(column)
        if key not in raw_bounds:
            raise ValueError(f'Frozen action contract missing bound for {column}: {contract_path}')
        bounds = raw_bounds[key]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f'Invalid frozen action bound for {column}: {bounds!r}')
        width = max(abs(float(bounds[0])), abs(float(bounds[1])))
        if not width > 0:
            raise ValueError(f'Non-positive frozen action bound width for {column}: {bounds!r}')
        widths.append(width)
    return np.asarray(widths, dtype=float)

def _wide_actions(action_panel: pd.DataFrame, action_columns: list[str], action_widths: np.ndarray) -> pd.DataFrame:
    stage7_diagnostic_columns = ['projection_distance', 'projection_method', 'out_of_library', 'budget_l1_target', 'budget_compliant_clipped']
    observed_stage7_diagnostics = [column for column in stage7_diagnostic_columns if column in action_panel.columns]
    c4 = action_panel.loc[action_panel['policy'].eq('C4'), ['budget_label', 'l1_budget', 'row_id', *observed_stage7_diagnostics, *action_columns]].copy()
    c6 = action_panel.loc[action_panel['policy'].eq('C6'), ['budget_label', 'l1_budget', 'row_id', *observed_stage7_diagnostics, *action_columns]].copy()
    c4 = c4.rename(columns={column: f'c4_{column}' for column in observed_stage7_diagnostics})
    c6 = c6.rename(columns={column: f'c6_{column}' for column in observed_stage7_diagnostics})
    c4 = c4.rename(columns={col: f'c4__{col}' for col in action_columns})
    c6 = c6.rename(columns={col: f'c6__{col}' for col in action_columns})
    wide = c4.merge(c6, on=['budget_label', 'l1_budget', 'row_id'], how='inner', validate='one_to_one')
    c4_values = wide[[f'c4__{col}' for col in action_columns]].to_numpy(float)
    c6_values = wide[[f'c6__{col}' for col in action_columns]].to_numpy(float)
    c4_abs, c6_abs = (np.abs(c4_values), np.abs(c6_values))
    c4_active, c6_active = (c4_abs > ACTION_TOLERANCE, c6_abs > ACTION_TOLERANCE)
    same_sign = c4_active & c6_active & (np.sign(c4_values) == np.sign(c6_values))
    retained = np.where(same_sign, np.minimum(c4_abs, c6_abs), 0.0).sum(axis=1)
    c4_l1, c6_l1 = (c4_abs.sum(axis=1), c6_abs.sum(axis=1))
    revision_l1_raw = np.abs(c6_values - c4_values).sum(axis=1)
    revision_l1_normalized = np.abs((c6_values - c4_values) / action_widths).sum(axis=1)
    c4_l1_normalized = np.abs(c4_values / action_widths).sum(axis=1)
    c6_l1_normalized = np.abs(c6_values / action_widths).sum(axis=1)
    removed_l1, added_l1 = (c4_l1 - retained, c6_l1 - retained)
    denom = removed_l1 + added_l1
    reallocation_share = np.divide(2.0 * np.minimum(removed_l1, added_l1), denom, out=np.full_like(denom, np.nan, dtype=float), where=denom > ACTION_TOLERANCE)
    wide['c4_final_l1'] = c4_l1
    wide['c6_final_l1'] = c6_l1
    wide['revision_l1_raw_recomputed'] = revision_l1_raw
    wide['revision_l1_normalized_recomputed'] = revision_l1_normalized
    wide['c4_final_l1_normalized'] = c4_l1_normalized
    wide['c6_final_l1_normalized'] = c6_l1_normalized
    wide['c4_active_dimensions'] = c4_active.sum(axis=1)
    wide['c6_active_dimensions'] = c6_active.sum(axis=1)
    wide['overlap_active_dimensions'] = (c4_active & c6_active).sum(axis=1)
    wide['added_dimensions'] = (~c4_active & c6_active).sum(axis=1)
    wide['removed_dimensions'] = (c4_active & ~c6_active).sum(axis=1)
    wide['sign_flipped_dimensions'] = (c4_active & c6_active & (np.sign(c4_values) != np.sign(c6_values))).sum(axis=1)
    wide['retained_same_sign_l1'] = retained
    wide['removed_initial_l1'] = removed_l1
    wide['added_revised_l1'] = added_l1
    wide['zero_sum_reallocation_share'] = reallocation_share
    budget = pd.to_numeric(wide['l1_budget'], errors='coerce')
    wide['revision_l1_over_budget'] = np.where(budget.notna(), revision_l1_normalized / budget, np.nan)
    wide['c4_final_l1_over_budget'] = np.where(budget.notna(), c4_l1 / budget, np.nan)
    wide['c6_final_l1_over_budget'] = np.where(budget.notna(), c6_l1 / budget, np.nan)
    return wide

def _win_tie_loss(revision: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in revision.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for oracle in ORACLES:
            gap = pd.to_numeric(group[f'revision_delta_R_score_{oracle}'], errors='coerce')
            if gap.isna().any():
                raise ValueError(f'N5M revision score gap contains NaN: budget={budget_label}, oracle={oracle}')
            wins = int((gap > SCORE_TIE_TOLERANCE).sum())
            losses = int((gap < -SCORE_TIE_TOLERANCE).sum())
            ties = int(len(gap) - wins - losses)
            non_tie = wins + losses
            lo, hi = _wilson(wins, non_tie)
            rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'n_firms': int(len(gap)), 'n_c6_win': wins, 'n_tie': ties, 'n_c6_loss': losses, 'c6_win_rate_all': wins / len(gap), 'tie_rate': ties / len(gap), 'c6_loss_rate_all': losses / len(gap), 'n_non_tie': non_tie, 'c6_win_rate_non_tie': wins / non_tie if non_tie else float('nan'), 'wilson_95_lo_non_tie': lo, 'wilson_95_hi_non_tie': hi, 'mean_C6_minus_C4': float(gap.mean()), 'median_C6_minus_C4': float(gap.median())})
    return pd.DataFrame(rows)

def _action_axis_summary(action_panel: pd.DataFrame, action_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in action_panel.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for action in action_columns:
            c4 = pd.to_numeric(group.loc[group['policy'].eq('C4'), action], errors='raise')
            c6 = pd.to_numeric(group.loc[group['policy'].eq('C6'), action], errors='raise')
            c4.index = group.loc[group['policy'].eq('C4'), 'row_id'].astype(int).to_numpy()
            c6.index = group.loc[group['policy'].eq('C6'), 'row_id'].astype(int).to_numpy()
            diff = c6.sort_index() - c4.sort_index()
            rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'action_axis': action, 'n_firms': int(len(diff)), 'mean_C4': float(c4.mean()), 'mean_C6': float(c6.mean()), 'mean_C6_minus_C4': float(diff.mean()), 'mean_abs_C6_minus_C4': float(diff.abs().mean()), 'positive_change_fraction': float((diff > ACTION_TOLERANCE).mean()), 'negative_change_fraction': float((diff < -ACTION_TOLERANCE).mean()), 'zero_change_fraction': float((diff.abs() <= ACTION_TOLERANCE).mean())})
    return pd.DataFrame(rows)

def _reallocation_outputs(wide: pd.DataFrame, revision: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    joined = wide.merge(revision[['budget_label', 'l1_budget', 'row_id', 'revision_l1_distance', *[f'revision_delta_R_score_{oracle}' for oracle in ORACLES]]], on=['budget_label', 'l1_budget', 'row_id'], how='inner', validate='one_to_one')
    mismatch = (pd.to_numeric(joined['revision_l1_distance'], errors='coerce') - pd.to_numeric(joined['revision_l1_normalized_recomputed'], errors='coerce')).abs()
    if mismatch.isna().any() or float(mismatch.max()) > 1e-09:
        raise ValueError(f'N5M Stage9 revision L1 does not match Stage7 action vectors; max_abs_diff={mismatch.max()}')
    metrics = ['c4_final_l1', 'c6_final_l1', 'revision_l1_raw_recomputed', 'c4_final_l1_normalized', 'c6_final_l1_normalized', 'revision_l1_normalized_recomputed', 'c4_active_dimensions', 'c6_active_dimensions', 'overlap_active_dimensions', 'added_dimensions', 'removed_dimensions', 'sign_flipped_dimensions', 'retained_same_sign_l1', 'removed_initial_l1', 'added_revised_l1', 'zero_sum_reallocation_share', 'revision_l1_over_budget', 'c4_final_l1_over_budget', 'c6_final_l1_over_budget']
    summary_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    corr_rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in joined.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        row: dict[str, Any] = {'budget_label': budget_label, 'l1_budget': l1_budget, 'n_firms': int(len(group))}
        for metric in metrics:
            row[f'mean_{metric}'] = _numeric_mean(group[metric])
            row[f'median_{metric}'] = _numeric_median(group[metric])
        summary_rows.append(row)
        for oracle in ORACLES:
            gap = pd.to_numeric(group[f'revision_delta_R_score_{oracle}'], errors='raise')
            outcome = np.where(gap > SCORE_TIE_TOLERANCE, 'C6_win', np.where(gap < -SCORE_TIE_TOLERANCE, 'C6_loss', 'tie'))
            temp = group.copy()
            temp['outcome_group'] = outcome
            for outcome_group, out_group in temp.groupby('outcome_group', sort=False):
                out: dict[str, Any] = {'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'outcome_group': outcome_group, 'n_firms': int(len(out_group)), 'mean_revision_score_gain': _numeric_mean(out_group[f'revision_delta_R_score_{oracle}'])}
                for metric in metrics:
                    out[f'mean_{metric}'] = _numeric_mean(out_group[metric])
                outcome_rows.append(out)
            n, rho, p = _spearman(group['revision_l1_normalized_recomputed'], gap)
            corr_rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'n_complete': n, 'spearman_revision_l1_vs_score_gain': rho, 'p_raw': p})
    corr = pd.DataFrame(corr_rows)
    corr['p_holm'] = np.nan
    for oracle, idx in corr.groupby('oracle_backend', sort=False).groups.items():
        vals = corr.loc[idx, 'p_raw'].tolist()
        adjusted = holm_adjust([1.0 if not np.isfinite(v) else float(v) for v in vals])
        corr.loc[idx, 'p_holm'] = adjusted
    corr['sig_holm'] = corr['p_holm'].map(p_to_stars)
    return (pd.DataFrame(summary_rows), pd.DataFrame(outcome_rows), corr)

def _reference_quality(revision: pd.DataFrame, lookup: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = revision.copy()
    merged['row_id'] = pd.to_numeric(merged['row_id'], errors='raise').astype(int)
    merged['rl_reference_candidate'] = merged['rl_reference_candidate'].astype(str).str.strip()
    merged = merged.merge(lookup, left_on=['row_id', 'rl_reference_candidate'], right_on=['row_id', 'reference_candidate_id'], how='left', validate='many_to_one', indicator=True)
    if merged['_merge'].ne('both').any():
        sample = merged.loc[merged['_merge'].ne('both'), ['budget_label', 'row_id', 'rl_reference_candidate']].head(10)
        raise ValueError(f"N5M shown reference missing from Stage6 fixed-candidate lookup: {sample.to_dict('records')}")
    merged = merged.drop(columns=['_merge', 'reference_candidate_id'])
    rows: list[dict[str, Any]] = []
    quartiles: list[dict[str, Any]] = []
    for oracle in ORACLES:
        merged[f'reference_advantage_{oracle}'] = pd.to_numeric(merged[f'reference_delta_R_score_{oracle}'], errors='coerce') - pd.to_numeric(merged[f'initial_delta_R_score_{oracle}'], errors='coerce')
    for (budget_label, l1_budget), group in merged.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for oracle in ORACLES:
            advantage = pd.to_numeric(group[f'reference_advantage_{oracle}'], errors='coerce')
            gain = pd.to_numeric(group[f'revision_delta_R_score_{oracle}'], errors='coerce')
            adoption = pd.to_numeric(group['rl_adoption_ratio'], errors='coerce')
            metrics_defined = group['metrics_defined'].fillna(False).astype(bool)
            n_gain, rho_gain, p_gain = _spearman(advantage, gain)
            n_adoption, rho_adoption, p_adoption = _spearman(advantage[metrics_defined], adoption[metrics_defined])
            good = advantage > 0
            worse = advantage < 0
            changed = pd.to_numeric(group['revision_l1_distance'], errors='coerce') > ACTION_TOLERANCE
            rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'n_firms': int(len(group)), 'n_reference_advantage_defined': int(advantage.notna().sum()), 'reference_better_fraction': float(good.mean()), 'reference_worse_fraction': float(worse.mean()), 'mean_reference_advantage': float(advantage.mean()), 'median_reference_advantage': float(advantage.median()), 'mean_revision_gain': float(gain.mean()), 'n_spearman_revision_gain': n_gain, 'rho_reference_advantage_vs_revision_gain': rho_gain, 'p_reference_advantage_vs_revision_gain_raw': p_gain, 'n_spearman_adoption': n_adoption, 'rho_reference_advantage_vs_adoption': rho_adoption, 'p_reference_advantage_vs_adoption_raw': p_adoption, 'reference_worse_but_revision_changed_fraction': float(changed[worse].mean()) if worse.any() else float('nan'), 'good_reference_but_score_declined_fraction': float((gain[good] < -SCORE_TIE_TOLERANCE).mean()) if good.any() else float('nan')})
            rank = advantage.rank(method='first')
            try:
                q = pd.qcut(rank, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            except ValueError as exc:
                raise ValueError(f'N5M reference-quality quartiles cannot be formed: budget={budget_label}, oracle={oracle}: {exc}') from exc
            temp = pd.DataFrame({'quartile': q, 'advantage': advantage, 'gain': gain, 'adoption': adoption.where(metrics_defined)})
            for quartile, cell in temp.groupby('quartile', observed=True, sort=True):
                quartiles.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'reference_advantage_quartile': str(quartile), 'n_firms': int(len(cell)), 'mean_reference_advantage': float(cell['advantage'].mean()), 'mean_revision_gain': float(cell['gain'].mean()), 'mean_adoption_ratio': float(cell['adoption'].mean()), 'score_decline_fraction': float((cell['gain'] < -SCORE_TIE_TOLERANCE).mean())})
    summary = pd.DataFrame(rows)
    for outcome in ('revision_gain', 'adoption'):
        raw_col = f'p_reference_advantage_vs_{outcome}_raw'
        holm_col = f'p_reference_advantage_vs_{outcome}_holm'
        summary[holm_col] = np.nan
        for oracle, idx in summary.groupby('oracle_backend', sort=False).groups.items():
            values = summary.loc[idx, raw_col].tolist()
            adjusted = holm_adjust([1.0 if not np.isfinite(v) else float(v) for v in values])
            summary.loc[idx, holm_col] = adjusted
        summary[f'sig_reference_advantage_vs_{outcome}_holm'] = summary[holm_col].map(p_to_stars)
    row_columns = ['budget_label', 'l1_budget', 'row_id', 'rl_reference_candidate', 'reference_policy', 'metrics_defined', 'rl_adoption_ratio', 'revision_l1_distance', *[f'reference_advantage_{oracle}' for oracle in ORACLES], *[f'revision_delta_R_score_{oracle}' for oracle in ORACLES]]
    return (summary, pd.DataFrame(quartiles), merged[row_columns].copy())

def _load_n5m_action_space(project_root: Path, arms: list[FrontierArm]):
    """Load the exact Stage2 candidate-library quantile declared by every arm.

    Frozen archives may contain absolute paths from the original Windows host,
    so the active project-root path is resolved from the recorded quantile and
    then cross-checked against the archived path name and source declaration.
    """
    quantiles = {int(arm.stage7_metadata.get('candidate_library_quantile')) for arm in arms if arm.stage7_metadata.get('candidate_library_quantile') is not None}
    if len(quantiles) != 1:
        raise ValueError(f'N5M arms must declare one common candidate-library quantile; got {sorted(quantiles)}')
    quantile = next(iter(quantiles))
    candidate_path = resolve_candidate_library_path(project_root, magnitude_quantile=quantile)
    space = load_action_space(project_root, candidate_library_path=candidate_path)
    archived_paths = {
        str(arm.stage7_metadata.get('selected_recalibrated_candidate_library_path') or arm.stage7_metadata.get('candidate_library_path') or '')
        for arm in arms
    }
    if '' in archived_paths or {Path(value).name for value in archived_paths} != {candidate_path.name}:
        raise ValueError(f'N5M arms must identify {candidate_path.name}; got {sorted(archived_paths)}')
    source_values = {str(arm.stage7_metadata.get('candidate_action_values_source') or '') for arm in arms}
    if source_values != {'stage2_recalibrated_candidate_library'}:
        raise ValueError(f'N5M arms must use Stage2 recalibrated candidate values; got {sorted(source_values)}')
    return (space, candidate_path, quantile)

def _build_firm_frame(*, revision: pd.DataFrame, wide: pd.DataFrame, reference_rows: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical one-row-per-firm×budget mechanism frame."""
    keys = ['budget_label', 'l1_budget', 'row_id']
    rev_cols = keys + ['run_label', 'rl_reference_candidate', 'reference_source', 'metrics_defined', 'rl_adoption_ratio', 'revision_l1_distance', 'revision_l2_distance', 'revision_cosine_distance', 'revision_changed_active_dimensions', *[f'initial_delta_R_score_{oracle}' for oracle in ORACLES], *[f'revised_delta_R_score_{oracle}' for oracle in ORACLES], *[f'revision_delta_R_score_{oracle}' for oracle in ORACLES]]
    rev_cols = [column for column in rev_cols if column in revision.columns]
    wide_cols = keys + [column for column in wide.columns if column not in keys and (column.startswith('c4__action__') or column.startswith('c6__action__') or column.startswith('c4_projection_') or column.startswith('c6_projection_') or (column in {'c4_out_of_library', 'c6_out_of_library', 'c4_budget_l1_target', 'c6_budget_l1_target', 'c4_budget_compliant_clipped', 'c6_budget_compliant_clipped'}) or (column in {'c4_final_l1', 'c6_final_l1', 'c4_final_l1_normalized', 'c6_final_l1_normalized', 'revision_l1_raw_recomputed', 'revision_l1_normalized_recomputed', 'removed_initial_l1', 'added_revised_l1', 'zero_sum_reallocation_share', 'c4_active_dimensions', 'c6_active_dimensions', 'overlap_active_dimensions', 'added_dimensions', 'removed_dimensions', 'sign_flipped_dimensions'}))]
    ref_cols = keys + ['rl_reference_candidate', 'reference_policy', *[f'reference_advantage_{oracle}' for oracle in ORACLES]]
    frame = revision[rev_cols].merge(wide[wide_cols], on=keys, how='inner', validate='one_to_one')
    frame = frame.merge(reference_rows[[column for column in ref_cols if column in reference_rows.columns]], on=keys + ['rl_reference_candidate'], how='inner', validate='one_to_one')
    expected = len(revision)
    if len(frame) != expected or frame.duplicated(keys).any():
        raise ValueError(f'N5M firm-frame alignment failed: expected={expected}, observed={len(frame)}')
    return frame.sort_values(['l1_budget', 'budget_label', 'row_id'], na_position='last').reset_index(drop=True)

def _cross_oracle_local_q_gain(firm_frame: pd.DataFrame) -> pd.DataFrame:
    """Shared-term-resistant sensitivity: Q from one Oracle, gain from another."""
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in firm_frame.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for q_oracle in ORACLES:
            q = pd.to_numeric(group[f'reference_advantage_{q_oracle}'], errors='coerce')
            for gain_oracle in ORACLES:
                if gain_oracle == q_oracle:
                    continue
                gain = pd.to_numeric(group[f'revision_delta_R_score_{gain_oracle}'], errors='coerce')
                n, rho, p_raw = _spearman(q, gain)
                rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'local_q_oracle': q_oracle, 'revision_gain_oracle': gain_oracle, 'n_complete': n, 'spearman_rho': rho, 'p_raw': p_raw, 'shared_term_status': 'OFF_DIAGONAL_ORACLE_SENSITIVITY', 'evidence_tier': 'EVALUATOR_ONLY_POSTHOC'})
    out = pd.DataFrame(rows)
    out['p_holm_24'] = holm_adjust([1.0 if not np.isfinite(v) else float(v) for v in out['p_raw']])
    out['sig_holm_24'] = out['p_holm_24'].map(p_to_stars)
    return out.sort_values(['p_holm_24', 'budget_label', 'local_q_oracle', 'revision_gain_oracle']).reset_index(drop=True)

def _reference_axis_anatomy(*, firm_frame: pd.DataFrame, action_columns: list[str], action_widths: np.ndarray, space) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose revision mass on and off the shown reference's active axes.

    This is an evaluator-only descriptive decomposition.  It measures action
    mass movement and must not be described as an axis-level causal effect.
    """
    rows: list[dict[str, Any]] = []
    widths = np.asarray(action_widths, dtype=float)
    for _, row in firm_frame.iterrows():
        reference = str(row['rl_reference_candidate'])
        if reference not in space.fixed_candidates:
            raise ValueError(f'N5M shown reference is absent from active candidate space: {reference}')
        c4 = np.asarray([float(row[f'c4__{column}']) for column in action_columns], dtype=float) / widths
        c6 = np.asarray([float(row[f'c6__{column}']) for column in action_columns], dtype=float) / widths
        ref = np.asarray(space.candidate_vector(reference), dtype=float) / widths
        ref_active = np.abs(ref) > ACTION_TOLERANCE
        c4_abs, c6_abs = (np.abs(c4), np.abs(c6))
        ref_direction_match = np.sign(c6) == np.sign(ref)
        add = np.maximum(c6_abs - c4_abs, 0.0)
        remove = np.maximum(c4_abs - c6_abs, 0.0)
        rec = {'budget_label': row['budget_label'], 'l1_budget': row['l1_budget'], 'row_id': int(row['row_id']), 'rl_reference_candidate': reference, 'reference_active_axis_count': int(ref_active.sum()), 'added_mass_on_reference_axes': float(add[ref_active & ref_direction_match].sum()), 'removed_mass_on_reference_axes': float(remove[ref_active].sum()), 'added_mass_off_reference_axes': float(add[~ref_active].sum()), 'removed_mass_off_reference_axes': float(remove[~ref_active].sum()), 'evidence_tier': 'EVALUATOR_ONLY_POSTHOC'}
        for oracle in ORACLES:
            rec[f'revision_gain_{oracle}'] = float(row[f'revision_delta_R_score_{oracle}'])
        rows.append(rec)
    detail = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in detail.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for oracle in ORACLES:
            gain = group[f'revision_gain_{oracle}']
            for metric in ('removed_mass_off_reference_axes', 'added_mass_on_reference_axes'):
                n, rho, p_raw = _spearman(group[metric], gain)
                loss = group.loc[gain < -SCORE_TIE_TOLERANCE, metric]
                win = group.loc[gain > SCORE_TIE_TOLERANCE, metric]
                summary_rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'metric': metric, 'n_complete': n, 'spearman_rho_vs_revision_gain': rho, 'p_raw': p_raw, 'mean_all': _numeric_mean(group[metric]), 'n_loss': int(len(loss)), 'mean_loss': _numeric_mean(loss), 'n_win': int(len(win)), 'mean_win': _numeric_mean(win), 'evidence_tier': 'EVALUATOR_ONLY_POSTHOC'})
    summary = pd.DataFrame(summary_rows)
    summary['p_holm_within_oracle_metric'] = np.nan
    for _, idx in summary.groupby(['oracle_backend', 'metric'], sort=False).groups.items():
        summary.loc[idx, 'p_holm_within_oracle_metric'] = holm_adjust([1.0 if not np.isfinite(v) else float(v) for v in summary.loc[idx, 'p_raw']])
    summary['sig_holm'] = summary['p_holm_within_oracle_metric'].map(p_to_stars)
    return (detail, summary)

def _oracle_consensus_by_budget(firm_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in firm_frame.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        gains = group[[f'revision_delta_R_score_{oracle}' for oracle in ORACLES]].to_numpy(float)
        loss_count = (gains < -SCORE_TIE_TOLERANCE).sum(axis=1)
        win_count = (gains > SCORE_TIE_TOLERANCE).sum(axis=1)
        tie_count = (np.abs(gains) <= SCORE_TIE_TOLERANCE).sum(axis=1)
        rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'n_firms': int(len(group)), 'all_three_loss_count': int((loss_count == 3).sum()), 'all_three_loss_fraction': float((loss_count == 3).mean()), 'majority_loss_count': int((loss_count >= 2).sum()), 'majority_loss_fraction': float((loss_count >= 2).mean()), 'all_three_win_count': int((win_count == 3).sum()), 'all_three_win_fraction': float((win_count == 3).mean()), 'majority_win_count': int((win_count >= 2).sum()), 'majority_win_fraction': float((win_count >= 2).mean()), 'all_three_tie_count': int((tie_count == 3).sum()), 'all_three_tie_fraction': float((tie_count == 3).mean()), 'mixed_oracle_direction_count': int(((loss_count > 0) & (win_count > 0)).sum()), 'mixed_oracle_direction_fraction': float(((loss_count > 0) & (win_count > 0)).mean()), 'evidence_tier': 'EVALUATOR_ONLY_POSTHOC'})
    return pd.DataFrame(rows)

def _vs_c3_contrasts(revision: pd.DataFrame, stage6: pd.DataFrame) -> pd.DataFrame:
    required = {'row_id', 'policy', *[f'delta_R_score_{oracle}' for oracle in ORACLES]}
    missing = sorted(required - set(stage6.columns))
    if missing:
        raise ValueError(f'Stage6 table missing canonical C3 contrast columns: {missing}')
    c3 = stage6.loc[stage6['policy'].astype(str).eq(CANONICAL_C3_POLICY)].copy()
    c3['row_id'] = pd.to_numeric(c3['row_id'], errors='raise').astype(int)
    if len(c3) != 575 or c3['row_id'].nunique() != 575 or c3.duplicated('row_id').any():
        raise ValueError(f"Canonical C3 must contain exactly 575 unique row ids; rows={len(c3)}, unique={c3['row_id'].nunique()}")
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in revision.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        if len(group) != 575 or group['row_id'].nunique() != 575:
            raise ValueError(f'N5M vs C3 requires 575 paired firms per budget; budget={budget_label}')
        for policy, score_prefix in (('C4', 'initial'), ('C6', 'revised')):
            for oracle in ORACLES:
                score_col = f'{score_prefix}_delta_R_score_{oracle}'
                left = group[['row_id', score_col]].rename(columns={score_col: 'policy_score'})
                right = c3[['row_id', f'delta_R_score_{oracle}']].rename(columns={f'delta_R_score_{oracle}': 'c3_score'})
                paired = left.merge(right, on='row_id', how='outer', validate='one_to_one', indicator=True)
                if not paired['_merge'].eq('both').all():
                    sample = paired.loc[paired['_merge'].ne('both'), ['row_id', '_merge']].head(10)
                    raise ValueError(f"N5M {policy} vs C3 row alignment failed: budget={budget_label}, oracle={oracle}, sample={sample.to_dict('records')}")
                gap = pd.to_numeric(paired['policy_score'], errors='raise') - pd.to_numeric(paired['c3_score'], errors='raise')
                rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'policy': policy, 'reference_policy': CANONICAL_C3_POLICY, 'contrast': f'{policy}_minus_C3__{budget_label}', 'holm_family': f'N5M_{policy}_minus_C3_{oracle}', 'n_pairs': int(len(gap)), 'n_nonzero_pairs': int((gap.abs() > SCORE_TIE_TOLERANCE).sum()), 'mean_policy_score': float(paired['policy_score'].mean()), 'mean_C3_score': float(paired['c3_score'].mean()), 'mean_gap': float(gap.mean()), 'median_gap': float(gap.median()), 'positive_fraction': float((gap > SCORE_TIE_TOLERANCE).mean()), 'zero_fraction': float((gap.abs() <= SCORE_TIE_TOLERANCE).mean()), 'negative_fraction': float((gap < -SCORE_TIE_TOLERANCE).mean()), 'wilcoxon_p_raw': wilcoxon_paired_p(gap)})
    out = pd.DataFrame(rows)
    out['wilcoxon_p_holm'] = np.nan
    for _, idx in out.groupby('holm_family', sort=False).groups.items():
        out.loc[idx, 'wilcoxon_p_holm'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].tolist())
    out['sig_holm'] = out['wilcoxon_p_holm'].map(p_to_stars)
    return out.sort_values(['policy', 'oracle_backend', 'l1_budget'], na_position='last').reset_index(drop=True)

def _adoption_diagnostics(wide: pd.DataFrame, revision: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    reallocation_columns = ['budget_label', 'l1_budget', 'row_id', 'c4_final_l1_normalized', 'c6_final_l1_normalized', 'revision_l1_normalized_recomputed', 'revision_l1_over_budget', 'removed_initial_l1', 'added_revised_l1', 'zero_sum_reallocation_share', 'added_dimensions', 'removed_dimensions', 'sign_flipped_dimensions']
    joined = revision.merge(wide[reallocation_columns], on=['budget_label', 'l1_budget', 'row_id'], how='inner', validate='one_to_one')
    if len(joined) != len(revision):
        raise ValueError(f'N5M adoption diagnostics lost rows: revision={len(revision)}, joined={len(joined)}')
    relation_rows: list[dict[str, Any]] = []
    quartile_rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in joined.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        defined = group['metrics_defined'].fillna(False).astype(bool)
        for oracle in ORACLES:
            adoption = pd.to_numeric(group.loc[defined, 'rl_adoption_ratio'], errors='coerce')
            gain = pd.to_numeric(group.loc[defined, f'revision_delta_R_score_{oracle}'], errors='coerce')
            complete = pd.DataFrame({'adoption': adoption, 'gain': gain}).dropna()
            n_complete, rho, p_raw = _spearman(complete['adoption'], complete['gain'])
            relation_rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'n_firms': int(len(group)), 'n_metrics_defined': int(defined.sum()), 'n_complete': n_complete, 'spearman_adoption_vs_revision_gain': rho, 'p_raw': p_raw})
            if len(complete) < 4:
                raise ValueError(f'Insufficient adoption complete cases: budget={budget_label}, oracle={oracle}')
            complete = complete.join(group.loc[complete.index, ['removed_initial_l1', 'added_revised_l1', 'zero_sum_reallocation_share', 'revision_l1_normalized_recomputed', 'revision_l1_over_budget', 'c4_final_l1_normalized', 'c6_final_l1_normalized', 'added_dimensions', 'removed_dimensions', 'sign_flipped_dimensions']])
            try:
                complete['adoption_quartile'] = pd.qcut(complete['adoption'].rank(method='first'), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            except ValueError as exc:
                raise ValueError(f'N5M adoption quartiles cannot be formed: budget={budget_label}, oracle={oracle}: {exc}') from exc
            for quartile, cell in complete.groupby('adoption_quartile', observed=True, sort=True):
                quartile_rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'adoption_quartile': str(quartile), 'n_firms': int(len(cell)), 'mean_adoption_ratio': float(cell['adoption'].mean()), 'median_adoption_ratio': float(cell['adoption'].median()), 'mean_revision_gain': float(cell['gain'].mean()), 'median_revision_gain': float(cell['gain'].median()), 'score_decline_fraction': float((cell['gain'] < -SCORE_TIE_TOLERANCE).mean()), 'mean_removed_initial_l1': _numeric_mean(cell['removed_initial_l1']), 'mean_added_revised_l1': _numeric_mean(cell['added_revised_l1']), 'mean_zero_sum_reallocation_share': _numeric_mean(cell['zero_sum_reallocation_share']), 'mean_revision_l1_normalized': _numeric_mean(cell['revision_l1_normalized_recomputed']), 'mean_revision_l1_over_budget': _numeric_mean(cell['revision_l1_over_budget']), 'mean_c4_final_l1_normalized': _numeric_mean(cell['c4_final_l1_normalized']), 'mean_c6_final_l1_normalized': _numeric_mean(cell['c6_final_l1_normalized']), 'mean_added_dimensions': _numeric_mean(cell['added_dimensions']), 'mean_removed_dimensions': _numeric_mean(cell['removed_dimensions']), 'mean_sign_flipped_dimensions': _numeric_mean(cell['sign_flipped_dimensions'])})
    relation = pd.DataFrame(relation_rows)
    relation['p_holm'] = np.nan
    for _, idx in relation.groupby('oracle_backend', sort=False).groups.items():
        relation.loc[idx, 'p_holm'] = holm_adjust([1.0 if not np.isfinite(value) else float(value) for value in relation.loc[idx, 'p_raw']])
    relation['sig_holm'] = relation['p_holm'].map(p_to_stars)
    return (relation, pd.DataFrame(quartile_rows))

def _feasibility(feasibility: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    critical = feasibility['sustainability'].astype(str).str.lower().eq('critical')
    for (budget_label, l1_budget, policy), group in feasibility.groupby(['budget_label', 'l1_budget', 'policy'], dropna=False, sort=False):
        critical_group = group['sustainability'].astype(str).str.lower().eq('critical')
        rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'policy': policy, 'n_rows': int(len(group)), 'sustainability_critical_count': int(critical_group.sum()), 'sustainability_critical_rate': float(critical_group.mean()), 'feasibility_core_violation_count': int(group['feasibility_core_violation_flag'].fillna(False).astype(bool).sum()), 'accounting_check_failure_count': int(group['accounting_check_failed'].fillna(False).astype(bool).sum()), 'negative_balance_count': int(group['negative_balance_flag'].fillna(False).astype(bool).sum()), 'preflight_failure_count': int((group['simulator_preflight_status'].astype(str).str.lower() != 'ok').sum())})
    unique = feasibility.loc[critical, ['row_id', 'budget_label', 'l1_budget', 'policy', 'sustainability']].copy()
    firm = unique.groupby('row_id', as_index=False).agg(critical_row_count=('sustainability', 'size'), affected_budget_count=('budget_label', 'nunique'), affected_policy_count=('policy', 'nunique'), affected_budgets=('budget_label', lambda s: ';'.join(sorted(set(map(str, s))))), affected_policies=('policy', lambda s: ';'.join(sorted(set(map(str, s)))))).sort_values(['critical_row_count', 'row_id'], ascending=[False, True])
    return (pd.DataFrame(rows), firm)

def _auxiliary_variance(revision: pd.DataFrame) -> pd.DataFrame:
    panel_rows: list[pd.DataFrame] = []
    for oracle in ORACLES:
        c4 = revision[['budget_label', 'l1_budget', 'row_id', f'initial_delta_R_score_{oracle}']].copy()
        c4['policy'] = 'C4'
        c4['score'] = pd.to_numeric(c4.pop(f'initial_delta_R_score_{oracle}'), errors='raise')
        c6 = revision[['budget_label', 'l1_budget', 'row_id', f'revised_delta_R_score_{oracle}']].copy()
        c6['policy'] = 'C6'
        c6['score'] = pd.to_numeric(c6.pop(f'revised_delta_R_score_{oracle}'), errors='raise')
        for frame in (c4, c6):
            frame['oracle_backend'] = oracle
        panel_rows.extend([c4, c6])
    panel = pd.concat(panel_rows, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for oracle, group in panel.groupby('oracle_backend', sort=False):
        work = group.copy()
        work['within_score'] = work['score'] - work.groupby('row_id')['score'].transform('mean')
        sst = float(np.square(work['within_score']).sum())
        if not sst > 0:
            raise ValueError(f'N5M within-firm score variance is zero for oracle={oracle}')
        budget_hat = work.groupby('budget_label')['within_score'].transform('mean')
        policy_hat = work.groupby('policy')['within_score'].transform('mean')
        cell_hat = work.groupby(['budget_label', 'policy'])['within_score'].transform('mean')
        rows.append({'oracle_backend': oracle, 'n_firms': int(work['row_id'].nunique()), 'n_cells_per_firm': 8, 'n_observations': int(len(work)), 'budget_main_effect_r2_within_firm': float(np.square(budget_hat).sum() / sst), 'reference_main_effect_r2_within_firm': float(np.square(policy_hat).sum() / sst), 'budget_policy_cell_r2_within_firm': float(np.square(cell_hat).sum() / sst), 'supplementary_only': True, 'excluded_from_main_harness_backend_decomposition': True, 'interpretation': 'Descriptive N5M-only within-firm decomposition. Not crossed with backend; must not be pooled into the common-panel harness-vs-model decomposition.'})
    return pd.DataFrame(rows)

def _operating_point_table(action_panel: pd.DataFrame, revision: pd.DataFrame, feasibility_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget, policy), group in action_panel.groupby(['budget_label', 'l1_budget', 'policy'], dropna=False, sort=False):
        rev = revision.loc[revision['budget_label'].eq(budget_label) & revision['l1_budget'].fillna(-1.0).eq(-1.0 if pd.isna(l1_budget) else float(l1_budget))]
        if policy == 'C4':
            score_cols = {oracle: f'initial_delta_R_score_{oracle}' for oracle in ORACLES}
        else:
            score_cols = {oracle: f'revised_delta_R_score_{oracle}' for oracle in ORACLES}
        action_cols = [c for c in group.columns if c.startswith('action__')]
        row = {'budget_label': budget_label, 'l1_budget': l1_budget, 'policy': policy, 'n_firms': int(len(group)), 'mean_final_l1': float(group[action_cols].abs().sum(axis=1).mean()), 'mean_projection_distance': _numeric_mean(group['projection_distance']), 'budget_violation_rate': float((~group['budget_compliant_clipped'].astype('boolean').fillna(True).astype(bool)).mean())}
        for oracle, col in score_cols.items():
            row[f'mean_delta_R_score_{oracle}'] = _numeric_mean(rev[col])
        feas = feasibility_summary.loc[feasibility_summary['budget_label'].eq(budget_label) & feasibility_summary['policy'].eq(policy)]
        if len(feas) != 1:
            raise ValueError(f'N5M feasibility summary missing operating-point cell: {budget_label}/{policy}')
        row['sustainability_critical_rate'] = float(feas['sustainability_critical_rate'].iloc[0])
        rows.append(row)
    out = pd.DataFrame(rows)
    out['point_estimate_nondominated_within_policy_alpha'] = False
    criteria_min = ['mean_final_l1', 'mean_projection_distance', 'budget_violation_rate', 'sustainability_critical_rate']
    for policy, idx in out.groupby('policy', sort=False).groups.items():
        subset = out.loc[idx]
        for i, row in subset.iterrows():
            dominated = False
            for j, other in subset.iterrows():
                if i == j:
                    continue
                no_worse = other['mean_delta_R_score_alpha'] >= row['mean_delta_R_score_alpha'] - 1e-12
                no_worse = no_worse and all((other[c] <= row[c] + 1e-12 for c in criteria_min))
                strictly = other['mean_delta_R_score_alpha'] > row['mean_delta_R_score_alpha'] + 1e-12
                strictly = strictly or any((other[c] < row[c] - 1e-12 for c in criteria_min))
                if no_worse and strictly:
                    dominated = True
                    break
            out.loc[i, 'point_estimate_nondominated_within_policy_alpha'] = not dominated
    out['interpretation_boundary'] = 'Point-estimate screen only; not a statistical Pareto dominance claim and not an optimal-budget declaration.'
    return out

def run_analysis(*, project_root: Path, run_dirs: Iterable[Path], output_dir: Path, expected_budgets: tuple[float | None, ...]=EXPECTED_BUDGETS) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    arms = validate_frontier_runs([Path(path).resolve() for path in run_dirs], run_role=RUN_ROLE, information_condition=INFORMATION_CONDITION, expected_budgets=expected_budgets, design='matched_c4_c6')
    action_panel, revision, feasibility, alignment_audit, action_columns = _validate_and_build_panel(arms)
    action_widths = _load_action_widths(project_root, action_columns)
    space, candidate_library_path, candidate_library_quantile = _load_n5m_action_space(project_root, arms)
    if list(space.columns) != list(action_columns):
        raise ValueError(f'N5M action column order differs from candidate space: {action_columns} != {list(space.columns)}')
    wide = _wide_actions(action_panel, action_columns, action_widths)
    win_tie_loss = _win_tie_loss(revision)
    action_axis = _action_axis_summary(action_panel, action_columns)
    overlap, by_outcome, distance_corr = _reallocation_outputs(wide, revision)
    stage6_path = project_root / 'data' / 'final_freeze' / 'stage6_candidate_selector_eval' / 'multi_oracle_policy_eval.parquet'
    stage6 = _read_stage6(stage6_path)
    lookup, lookup_metadata = _build_stage6_reference_lookup(stage6)
    reference_summary, reference_quartiles, reference_rows = _reference_quality(revision, lookup)
    firm_frame = _build_firm_frame(revision=revision, wide=wide, reference_rows=reference_rows)
    cross_oracle = _cross_oracle_local_q_gain(firm_frame)
    reference_axis_detail, reference_axis_summary = _reference_axis_anatomy(firm_frame=firm_frame, action_columns=action_columns, action_widths=action_widths, space=space)
    oracle_consensus = _oracle_consensus_by_budget(firm_frame)
    vs_c3 = _vs_c3_contrasts(revision, stage6)
    adoption_relationship, adoption_quartiles = _adoption_diagnostics(wide, revision)
    feasibility_summary, feasibility_firms = _feasibility(feasibility)
    auxiliary_variance = _auxiliary_variance(revision)
    operating_points = _operating_point_table(action_panel, revision, feasibility_summary)
    outputs: dict[str, Path] = {'alignment_audit': output_dir / 'n5m_alignment_audit.csv', 'win_tie_loss': output_dir / 'n5m_win_tie_loss_by_budget_oracle.csv', 'reference_quality': output_dir / 'n5m_reference_quality_by_budget.csv', 'reference_quality_quartiles': output_dir / 'n5m_reference_quality_quartiles.csv', 'reference_quality_rows': output_dir / 'n5m_reference_quality_rows.parquet', 'firm_frame': output_dir / 'n5m_firm_frame.parquet', 'cross_oracle_local_q_gain': output_dir / 'n5m_cross_oracle_local_q_gain.csv', 'reference_axis_anatomy': output_dir / 'n5m_reference_axis_anatomy.parquet', 'reference_axis_outcome_summary': output_dir / 'n5m_reference_axis_outcome_summary.csv', 'oracle_consensus_by_budget': output_dir / 'n5m_oracle_consensus_by_budget.csv', 'vs_c3_paired_holm': output_dir / 'n5m_vs_c3_paired_holm.csv', 'adoption_relationship': output_dir / 'n5m_adoption_score_relationship.csv', 'adoption_quartiles': output_dir / 'n5m_adoption_quartiles.csv', 'action_axis': output_dir / 'n5m_action_axis_c4_c6_differences.csv', 'dimension_overlap': output_dir / 'n5m_dimension_overlap_summary.csv', 'reallocation_by_outcome': output_dir / 'n5m_reallocation_by_outcome_group.csv', 'revision_distance_relationship': output_dir / 'n5m_revision_distance_score_relationship.csv', 'feasibility_by_budget_policy': output_dir / 'n5m_feasibility_by_budget_policy.csv', 'feasibility_unique_firms': output_dir / 'n5m_feasibility_unique_firms.csv', 'auxiliary_variance': output_dir / 'n5m_auxiliary_within_firm_variance.csv', 'operating_points': output_dir / 'n5m_score_budget_auditability_operating_points.csv', 'input_files': output_dir / 'n5m_posthoc_input_files.csv', 'manifest': output_dir / 'n5m_posthoc_manifest.json'}
    pd.DataFrame(alignment_audit).to_csv(outputs['alignment_audit'], index=False, encoding='utf-8-sig')
    win_tie_loss.to_csv(outputs['win_tie_loss'], index=False, encoding='utf-8-sig')
    reference_summary.to_csv(outputs['reference_quality'], index=False, encoding='utf-8-sig')
    reference_quartiles.to_csv(outputs['reference_quality_quartiles'], index=False, encoding='utf-8-sig')
    reference_rows.to_parquet(outputs['reference_quality_rows'], index=False)
    firm_frame.to_parquet(outputs['firm_frame'], index=False)
    cross_oracle.to_csv(outputs['cross_oracle_local_q_gain'], index=False, encoding='utf-8-sig')
    reference_axis_detail.to_parquet(outputs['reference_axis_anatomy'], index=False)
    reference_axis_summary.to_csv(outputs['reference_axis_outcome_summary'], index=False, encoding='utf-8-sig')
    oracle_consensus.to_csv(outputs['oracle_consensus_by_budget'], index=False, encoding='utf-8-sig')
    vs_c3.to_csv(outputs['vs_c3_paired_holm'], index=False, encoding='utf-8-sig')
    adoption_relationship.to_csv(outputs['adoption_relationship'], index=False, encoding='utf-8-sig')
    adoption_quartiles.to_csv(outputs['adoption_quartiles'], index=False, encoding='utf-8-sig')
    action_axis.to_csv(outputs['action_axis'], index=False, encoding='utf-8-sig')
    overlap.to_csv(outputs['dimension_overlap'], index=False, encoding='utf-8-sig')
    by_outcome.to_csv(outputs['reallocation_by_outcome'], index=False, encoding='utf-8-sig')
    distance_corr.to_csv(outputs['revision_distance_relationship'], index=False, encoding='utf-8-sig')
    feasibility_summary.to_csv(outputs['feasibility_by_budget_policy'], index=False, encoding='utf-8-sig')
    feasibility_firms.to_csv(outputs['feasibility_unique_firms'], index=False, encoding='utf-8-sig')
    auxiliary_variance.to_csv(outputs['auxiliary_variance'], index=False, encoding='utf-8-sig')
    operating_points.to_csv(outputs['operating_points'], index=False, encoding='utf-8-sig')
    input_rows = []
    for arm in arms:
        for role, source_path in (
            ('stage7_action_table', arm.action_table),
            ('stage8_feasibility_audit', arm.run_dir / 'stage8_llm_multi_oracle_eval' / 'llm_stage8_failure_audit_enriched.csv'),
            ('stage9_revision_metrics', arm.run_dir / 'stage9_llm_rl_comparison' / 'llm_stage9_revision_metrics.csv'),
        ):
            input_rows.append({'run_label': arm.run_label, 'budget_label': arm.budget_label, 'role': role, 'path': str(source_path), 'size_bytes': int(source_path.stat().st_size)})
    for role, source_path in (
        ('stage6_reference_scores', stage6_path),
        ('candidate_library', candidate_library_path),
    ):
        input_rows.append({'run_label': '', 'budget_label': '', 'role': role, 'path': str(source_path), 'size_bytes': int(source_path.stat().st_size)})
    pd.DataFrame(input_rows).to_csv(outputs['input_files'], index=False, encoding='utf-8-sig')
    manifest = {'schema_version': SCHEMA_VERSION, 'created_utc': _now(), 'status': 'PASS', 'run_role': RUN_ROLE, 'information_condition': INFORMATION_CONDITION, 'design': 'matched_c4_c6', 'expected_budgets': list(expected_budgets), 'run_count': len(arms), 'firm_count': int(revision['row_id'].nunique()), 'stage7_action_row_count': int(len(action_panel)), 'stage9_revision_row_count': int(len(revision)), 'stage8_feasibility_row_count': int(len(feasibility)), 'action_columns': action_columns, 'action_bound_widths': {column: float(width) for column, width in zip(action_columns, action_widths)}, 'candidate_library_path': str(candidate_library_path), 'candidate_library_quantile': int(candidate_library_quantile), 'stage6_reference_lookup': lookup_metadata, 'firm_frame_contract': {'row_count': int(len(firm_frame)), 'unique_firm_count': int(firm_frame['row_id'].nunique()), 'key': ['budget_label', 'l1_budget', 'row_id'], 'post_c4_gate_feature_columns': ['c4_final_l1', 'c4_projection_distance', 'c4_active_dimensions'], 'post_c4_gate_feature_status': 'AVAILABLE' if {'c4_final_l1', 'c4_projection_distance', 'c4_active_dimensions'}.issubset(firm_frame.columns) else 'MISSING'}, 'cross_oracle_shared_term_sensitivity_contract': {'row_count': int(len(cross_oracle)), 'expected_row_count': 24, 'holm_family_size': 24, 'status': 'EVALUATOR_ONLY_POSTHOC'}, 'reference_axis_anatomy_contract': {'detail_row_count': int(len(reference_axis_detail)), 'summary_row_count': int(len(reference_axis_summary)), 'interpretation': 'descriptive action-mass decomposition; not axis-level causal attribution'}, 'oracle_consensus_contract': {'row_count': int(len(oracle_consensus)), 'expected_row_count': 4}, 'sustainability_critical_row_count': int(feasibility['sustainability'].astype(str).str.lower().eq('critical').sum()), 'sustainability_critical_unique_firm_count': int(len(feasibility_firms)), 'vs_c3_contrast_contract': {'canonical_reference_policy': CANONICAL_C3_POLICY, 'row_count': int(len(vs_c3)), 'expected_row_count': 24, 'holm_family': 'policy_x_oracle_across_four_budgets'}, 'adoption_diagnostics_contract': {'relationship_row_count': int(len(adoption_relationship)), 'quartile_row_count': int(len(adoption_quartiles)), 'complete_case_policy': 'metrics_defined_and_finite_adoption_and_revision_gain'}, 'main_harness_backend_decomposition_contract': {'included': False, 'excluded_run_families': ['N5', 'N5F', 'N5M'], 'reason': 'Budget is not crossed with backend in N5M; pooling it into the common-panel harness-vs-model decomposition would mechanically inflate harness R2 and confound run time.'}, 'auxiliary_variance_contract': {'status': 'SUPPLEMENTARY_ONLY', 'panel': 'N5M GPT-5.4-mini x IC-b x free_form x C4/C6 x four budgets x seed1', 'interpretation': 'descriptive within-firm R2; no backend comparison or causal attribution'}, 'interpretation_boundaries': ['Single live seed, GPT-5.4-mini, IC-b, free_form_10d only.', 'Crowd-out, reference-axis, consensus, and adoption-quartile statistics are evaluator-only mechanism diagnostics, not causal mediation estimates.', 'Cross-Oracle local-Q sensitivity reduces but does not eliminate shared-input/shared-target dependence among evaluators.', 'C4/C6 versus C3 tests use the exact Stage6 C3_candidate_iql rows; diagnostic C3 variants are forbidden.', 'Operating-point dominance is based on point estimates and is not an optimal-budget claim.', 'N5M is excluded from the main crossed harness-vs-backend variance decomposition.'], 'outputs': {key: str(path) for key, path in outputs.items() if key != 'manifest'}, 'completed_utc': _now()}
    _write_json(outputs['manifest'], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--run-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-budgets', nargs='+', default=['0.75', '1.27', '2.00', 'unbounded'])
    return parser

def _parse_budgets(values: Iterable[str]) -> tuple[float | None, ...]:
    parsed: list[float | None] = []
    for value in values:
        text = str(value).strip().lower()
        parsed.append(None if text in {'none', 'null', 'unbounded', 'inf', 'infinity', '∞'} else float(text))
    return tuple(parsed)

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_analysis(project_root=Path(args.project_root), run_dirs=[Path(value) for value in args.run_dirs], output_dir=Path(args.output_dir), expected_budgets=_parse_budgets(args.expected_budgets))
    return 0 if result.get('status') == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
