from __future__ import annotations
'Paper-facing table and figure asset builder for thesis reproduction outputs.\n\nThis module is intentionally read-only with respect to frozen Stage artifacts.  It\nconsumes outputs already generated under ``data/analysis/paper_repro/<label>``\nand writes compact, paper-facing CSV/Markdown tables plus optional PNG figures\nunder that same analysis root:\n\n    paper_tables/\n    paper_figures/\n\nIt does not rescore actions, call LLM APIs, retrain RL, or mutate canonical\nStage7/8/9 directories. Missing optional analyses are recorded in the manifest;\nin ``--strict`` mode, analyses that are relevant to the current thesis contract\n(e.g. sign-flip after the task is requested) can be required by the caller.\n'
import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import pandas as pd
from credit_recourse.analysis.paper_output_layout import build_layout, ensure_layout
from credit_recourse.contracts.paper_reproduction import load_profile
IC_ORDER = ['IC-a', 'IC-b', 'IC-c']
VARIANT_ORDER = ['sign_flip_mean_vector', 'l1_rescale_same_policy_candidate_mean', 'global_mean_vector_null_same_policy_candidate_mean', 'row_shuffle_vector_null_same_policy_candidate_mean', 'nearest_candidate_projection_native', 'global_mean_vector_null_native', 'row_shuffle_vector_null_native']
C3_ALPHA_DEFAULT = 0.630313

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _safe_name(text: str) -> str:
    s = re.sub('[^A-Za-z0-9._-]+', '_', str(text)).strip('_')
    return s or 'unnamed'

def _normalise_ic(text: str | Path) -> str | None:
    s = str(text).replace('\\', '/')
    low = s.lower().replace('_', '-')
    if re.search('(^|[^a-z0-9])ic-?a([^a-z0-9]|$)', low) or 'ica' in low:
        return 'IC-a'
    if re.search('(^|[^a-z0-9])ic-?b([^a-z0-9]|$)', low) or 'icb' in low:
        return 'IC-b'
    if re.search('(^|[^a-z0-9])ic-?c([^a-z0-9]|$)', low) or 'icc' in low:
        return 'IC-c'
    return None

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)

def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _fmt_num(x: Any, nd: int=3, signed: bool=True) -> str:
    try:
        val = float(x)
    except Exception:
        return ''
    if not math.isfinite(val):
        return ''
    prefix = '+' if signed and val >= 0 else ''
    return f'{prefix}{val:.{nd}f}'

def _fmt_pct(x: Any, nd: int=1) -> str:
    try:
        val = float(x)
    except Exception:
        return ''
    if not math.isfinite(val):
        return ''
    if abs(val) <= 1.0:
        val *= 100.0
    return f'{val:.{nd}f}%'

def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No rows._\n'
    cols = list(df.columns)
    rows = [[str(x) for x in r] for r in df.astype(object).where(pd.notna(df), '').itertuples(index=False, name=None)]
    widths = [len(str(c)) for c in cols]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]

    def fmt_row(row: Iterable[Any]) -> str:
        return '| ' + ' | '.join((str(cell).ljust(w) for cell, w in zip(row, widths))) + ' |'
    out = [fmt_row(cols), '| ' + ' | '.join(('-' * w for w in widths)) + ' |']
    out.extend((fmt_row(row) for row in rows))
    return '\n'.join(out) + '\n'

def _write_table(df: pd.DataFrame, tables_dir: Path, name: str, *, notes: str | None=None) -> dict[str, str]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f'{name}.csv'
    md_path = tables_dir / f'{name}.md'
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    md = _df_to_markdown(df)
    if notes:
        md = md + '\n' + notes.strip() + '\n'
    md_path.write_text(md, encoding='utf-8')
    return {'csv': str(csv_path), 'markdown': str(md_path), 'n_rows': str(len(df))}

def _extract_c6_alpha(summary_path: Path) -> float | None:
    df = _read_csv(summary_path)
    if 'mean_delta_R_score_alpha' not in df.columns:
        return None
    m = (df.get('policy', '').astype(str) == 'C6') & (df.get('mode', '').astype(str) == 'free_form_10d')
    sub = df.loc[m]
    if sub.empty:
        return None
    return float(pd.to_numeric(sub['mean_delta_R_score_alpha'], errors='coerce').iloc[0])

def _extract_paired_alpha(path: Path, value_col: str='mean_gap_vs_reference') -> tuple[float | None, str | None]:
    if not path.exists():
        return (None, None)
    df = _read_csv(path)
    if value_col not in df.columns or 'oracle_backend' not in df.columns:
        return (None, None)
    sub = df.loc[df['oracle_backend'].astype(str).str.lower() == 'alpha']
    if sub.empty:
        return (None, None)
    sig = None
    if 'sig_holm' in sub.columns:
        sig = str(sub['sig_holm'].iloc[0])
    return (float(pd.to_numeric(sub[value_col], errors='coerce').iloc[0]), sig)

def collect_signflip(signflip_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not signflip_dir.exists():
        return pd.DataFrame()
    for summary in sorted(signflip_dir.glob('IC-*/*/ablation_policy_summary.csv')):
        ic = _normalise_ic(summary)
        run_label = summary.parent.name
        alpha = _extract_c6_alpha(summary)
        gap_c3, sig_c3 = _extract_paired_alpha(summary.parent / 'ablation_paired_vs_reference.csv')
        gap_c6, sig_c6 = _extract_paired_alpha(summary.parent / 'ablation_paired_vs_original_target.csv')
        pos_frac = None
        df = _read_csv(summary)
        required = {'policy', 'mode'}
        missing = sorted(required - set(df.columns))
        if missing:
            raise RuntimeError(f'sign-flip summary missing required columns {missing}: {summary}')
        sub = df[(df['policy'].astype(str) == 'C6') & (df['mode'].astype(str) == 'free_form_10d')]
        if not sub.empty and 'positive_fraction_alpha' in sub.columns:
            value = pd.to_numeric(sub['positive_fraction_alpha'], errors='raise').iloc[0]
            pos_frac = float(value)
        rows.append({'information_condition': ic, 'run': run_label, 'variant': 'sign_flip_mean_vector', 'alpha_delta_noop': alpha, 'positive_fraction_alpha': pos_frac, 'gap_vs_C3_alpha': gap_c3, 'sig_vs_C3_holm': sig_c3, 'gap_vs_original_C6_alpha': gap_c6, 'sig_vs_original_C6_holm': sig_c6, 'source_dir': str(summary.parent)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['_ic_order'] = out['information_condition'].map({ic: i for i, ic in enumerate(IC_ORDER)}).fillna(99)
        out = out.sort_values(['_ic_order', 'run']).drop(columns=['_ic_order'])
    return out

def collect_postfreeze_ablation(postfreeze_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not postfreeze_dir.exists():
        return pd.DataFrame()
    for summary in sorted(postfreeze_dir.glob('IC-*/*/*/ablation_policy_summary.csv')):
        ic = _normalise_ic(summary)
        run_label = summary.parent.parent.name
        variant_cell = summary.parent.name
        alpha = _extract_c6_alpha(summary)
        gap_c3, sig_c3 = _extract_paired_alpha(summary.parent / 'ablation_paired_vs_reference.csv')
        gap_c6, sig_c6 = _extract_paired_alpha(summary.parent / 'ablation_paired_vs_original_target.csv')
        rows.append({'information_condition': ic, 'run': run_label, 'variant_cell': variant_cell, 'alpha_delta_noop': alpha, 'gap_vs_C3_alpha': gap_c3, 'sig_vs_C3_holm': sig_c3, 'gap_vs_original_C6_alpha': gap_c6, 'sig_vs_original_C6_holm': sig_c6, 'source_dir': str(summary.parent)})
    out = pd.DataFrame(rows)
    if not out.empty:
        out['_ic_order'] = out['information_condition'].map({ic: i for i, ic in enumerate(IC_ORDER)}).fillna(99)
        out['_variant_order'] = out['variant_cell'].map({v: i for i, v in enumerate(VARIANT_ORDER)}).fillna(99)
        out = out.sort_values(['_ic_order', '_variant_order', 'variant_cell']).drop(columns=['_ic_order', '_variant_order'])
    return out

def collect_reference_quality(reference_dir: Path) -> pd.DataFrame:
    p = reference_dir / 'reference_quality_acceptance_primary_c6.csv'
    return _read_csv(p) if p.exists() else pd.DataFrame()

def collect_n5_table(n5_dir: Path) -> pd.DataFrame:
    p = n5_dir / 'n5_7_10c_table_patch.csv'
    return _read_csv(p) if p.exists() else pd.DataFrame()

def collect_n5_budget_frontier(frontier_dir: Path) -> pd.DataFrame:
    p = frontier_dir / 'n5_budget_frontier_table_patch.csv'
    return _read_csv(p) if p.exists() else pd.DataFrame()

def collect_n5m_posthoc(posthoc_dir: Path) -> dict[str, pd.DataFrame]:
    files = {'n5m_win_tie_loss_by_budget_oracle': 'n5m_win_tie_loss_by_budget_oracle.csv', 'n5m_reference_quality_by_budget': 'n5m_reference_quality_by_budget.csv', 'n5m_reference_quality_quartiles': 'n5m_reference_quality_quartiles.csv', 'n5m_cross_oracle_local_q_gain': 'n5m_cross_oracle_local_q_gain.csv', 'n5m_reference_axis_outcome_summary': 'n5m_reference_axis_outcome_summary.csv', 'n5m_oracle_consensus_by_budget': 'n5m_oracle_consensus_by_budget.csv', 'n5m_vs_c3_paired_holm': 'n5m_vs_c3_paired_holm.csv', 'n5m_adoption_score_relationship': 'n5m_adoption_score_relationship.csv', 'n5m_adoption_quartiles': 'n5m_adoption_quartiles.csv', 'n5m_action_axis_c4_c6_differences': 'n5m_action_axis_c4_c6_differences.csv', 'n5m_dimension_overlap_summary': 'n5m_dimension_overlap_summary.csv', 'n5m_reallocation_by_outcome_group': 'n5m_reallocation_by_outcome_group.csv', 'n5m_revision_distance_score_relationship': 'n5m_revision_distance_score_relationship.csv', 'n5m_feasibility_by_budget_policy': 'n5m_feasibility_by_budget_policy.csv', 'n5m_feasibility_unique_firms': 'n5m_feasibility_unique_firms.csv', 'n5m_auxiliary_within_firm_variance': 'n5m_auxiliary_within_firm_variance.csv', 'n5m_score_budget_auditability_operating_points': 'n5m_score_budget_auditability_operating_points.csv'}
    return {name: _read_csv(posthoc_dir / filename) for name, filename in files.items()}

def collect_n5m_adaptive_selection(selection_dir: Path) -> dict[str, pd.DataFrame]:
    files = {'n5m_adaptive_budget_summary': 'n5m_adaptive_budget_summary.csv', 'n5m_postc4_gate_summary': 'n5m_postc4_gate_summary.csv'}
    return {name: _read_csv(selection_dir / filename) for name, filename in files.items()}

def collect_main_harness_backend_decomposition(source_dir: Path) -> dict[str, pd.DataFrame]:
    files = {'main_harness_backend_decomposition': 'main_harness_backend_decomposition.csv', 'main_harness_backend_cell_means': 'main_harness_backend_cell_means.csv', 'main_harness_backend_swing_summary': 'main_harness_backend_swing_summary.csv', 'main_harness_backend_alignment_audit': 'main_harness_backend_alignment_audit.csv'}
    return {name: _read_csv(source_dir / filename) for name, filename in files.items()}

def collect_winrate(winrate_dir: Path) -> pd.DataFrame:
    p = winrate_dir / 'win_rates_vs_C3.csv'
    if not p.exists():
        return pd.DataFrame()
    df = _read_csv(p)
    m = (df.get('policy', '').astype(str) == 'C6') & (df.get('mode', '').astype(str) == 'free_form_10d') & (df.get('oracle_backend', '').astype(str).str.lower() == 'alpha')
    cols = ['information_condition', 'n_pairs', 'n_ties', 'n_non_tie_pairs', 'win_rate_excluding_ties', 'win_rate_wilson_lo', 'win_rate_wilson_hi', 'positive_fraction_including_ties', 'zero_fraction', 'mean_gap_vs_c3', 'median_gap_vs_c3']
    return df.loc[m, [c for c in cols if c in df.columns]].copy()

def collect_heterogeneity(winrate_dir: Path) -> pd.DataFrame:
    p = winrate_dir / 'residual_heterogeneity_exploratory.csv'
    return _read_csv(p) if p.exists() else pd.DataFrame()

def collect_frontier(frontier_dir: Path) -> pd.DataFrame:
    p = frontier_dir / 'frontier_grid_raw.csv'
    if not p.exists():
        return pd.DataFrame()
    df = _read_csv(p)
    m = (df.get('policy', '').astype(str) == 'C6') & (df.get('mode', '').astype(str) == 'free_form_10d')
    return df.loc[m].copy()

def collect_icc_probe(icc_probe_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    channel_path = icc_probe_dir / 'icc_probe_channel_summary.csv'
    familiarity_path = icc_probe_dir / 'icc_probe_familiarity_distribution.csv'
    channel = _read_csv(channel_path) if channel_path.exists() else pd.DataFrame()
    familiarity = _read_csv(familiarity_path) if familiarity_path.exists() else pd.DataFrame()
    return (channel, familiarity)

def collect_b2_gap(b2_gap_dir: Path) -> pd.DataFrame:
    report = b2_gap_dir / 'b2_gap_decomposition_report.json'
    if not report.exists():
        return pd.DataFrame()
    data = json.loads(report.read_text(encoding='utf-8'))
    rows: list[dict[str, Any]] = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                walk(f'{prefix}.{key}' if prefix else str(key), item)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            rows.append({'metric': prefix, 'value': value})
    walk('', data)
    return pd.DataFrame(rows)

def collect_test3(test3_dir: Path) -> pd.DataFrame:
    path = test3_dir / 'test3_property_summary.csv'
    return _read_csv(path) if path.exists() else pd.DataFrame()

def collect_structural_slice(structural_dir: Path) -> pd.DataFrame:
    path = structural_dir / 'b2_structural_event_slice_summary.csv'
    return _read_csv(path) if path.exists() else pd.DataFrame()

def _save_structural_slice_plot(df: pd.DataFrame, path: Path, *, strict: bool=False) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if plt is None:
        return err
    if df.empty:
        return 'empty_data'
    rate_col = next((c for c in ('agreement', 'agreement_rate', 'direction_agreement') if c in df.columns), None)
    label_col = next((c for c in ('slice', 'group', 'event_slice') if c in df.columns), None)
    if rate_col is None or label_col is None:
        return f'missing columns; available={list(df.columns)}'
    plot = df[[label_col, rate_col]].copy()
    plot[rate_col] = pd.to_numeric(plot[rate_col], errors='coerce')
    plot = plot.dropna()
    if plot.empty:
        return 'empty_numeric_data'
    values = plot[rate_col] * 100.0 if plot[rate_col].abs().max() <= 1.0 else plot[rate_col]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(plot[label_col].astype(str), values)
    ax.set_ylabel('Directional agreement (%)')
    ax.set_ylim(0, 100)
    ax.set_title('Test 3′ by structural/material-event slice')
    ax.tick_params(axis='x', rotation=18)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def make_paper_ladder_table(ablation: pd.DataFrame, signflip: pd.DataFrame, n5: pd.DataFrame, winrate: pd.DataFrame, c3_alpha: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not signflip.empty:
        for _, r in signflip.iterrows():
            rows.append({'information_condition': r.get('information_condition'), 'family': 'sign-flip mean null', 'alpha_delta_noop': r.get('alpha_delta_noop'), 'gap_vs_C3_alpha': r.get('gap_vs_C3_alpha'), 'sig_vs_C3': r.get('sig_vs_C3_holm'), 'positive_fraction': r.get('positive_fraction_alpha'), 'source': 'signflip_mean_null'})
    if not ablation.empty:
        keep = {'global_mean_vector_null_native': 'native global mean null', 'row_shuffle_vector_null_native': 'native row-shuffle null', 'l1_rescale_same_policy_candidate_mean': 'L1-rescale to candidate budget', 'global_mean_vector_null_same_policy_candidate_mean': 'budgeted global mean null', 'row_shuffle_vector_null_same_policy_candidate_mean': 'budgeted row-shuffle null', 'nearest_candidate_projection_native': 'nearest P50 projection'}
        for _, r in ablation.iterrows():
            cell = str(r.get('variant_cell'))
            if cell not in keep:
                continue
            rows.append({'information_condition': r.get('information_condition'), 'family': keep[cell], 'alpha_delta_noop': r.get('alpha_delta_noop'), 'gap_vs_C3_alpha': r.get('gap_vs_C3_alpha'), 'sig_vs_C3': r.get('sig_vs_C3_holm'), 'positive_fraction': None, 'source': 'postfreeze_ablation'})
    if not n5.empty:
        for _, r in n5.iterrows():
            rows.append({'information_condition': r.get('information_condition'), 'family': 'N5 generation-time L1 budget', 'alpha_delta_noop': r.get('mean_C6', r.get('paper_C6_alpha_delta')), 'gap_vs_C3_alpha': r.get('C6_minus_C3_alpha_gap'), 'sig_vs_C3': r.get('C6_minus_C3_sig'), 'positive_fraction': None, 'source': 'n5_7_10c_holm'})
    if not winrate.empty:
        for ic in sorted(set(winrate['information_condition'].dropna()), key=lambda x: IC_ORDER.index(x) if x in IC_ORDER else 99):
            rows.append({'information_condition': ic, 'family': 'C3 reference', 'alpha_delta_noop': c3_alpha, 'gap_vs_C3_alpha': 0.0, 'sig_vs_C3': 'ref', 'positive_fraction': None, 'source': 'stage6_reference'})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {'sign-flip mean null': 0, 'L1-rescale to candidate budget': 10, 'budgeted global mean null': 11, 'budgeted row-shuffle null': 12, 'nearest P50 projection': 13, 'C3 reference': 20, 'N5 generation-time L1 budget': 30, 'native global mean null': 40, 'native row-shuffle null': 41}
    out['_ic_order'] = out['information_condition'].map({ic: i for i, ic in enumerate(IC_ORDER)}).fillna(99)
    out['_family_order'] = out['family'].map(order).fillna(99)
    out = out.sort_values(['_ic_order', '_family_order', 'family']).drop(columns=['_ic_order', '_family_order'])
    return out

def _try_import_matplotlib(strict: bool):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        return (plt, None)
    except Exception as e:
        if strict:
            raise
        return (None, repr(e))

def _save_barh_alpha(df: pd.DataFrame, path: Path, *, title: str, x_col: str='alpha_delta_noop', y_col: str='label', c3_alpha: float | None=None, strict: bool=False) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if plt is None:
        return err
    plot_df = df.copy()
    plot_df = plot_df[pd.to_numeric(plot_df[x_col], errors='coerce').notna()]
    if plot_df.empty:
        return 'empty_data'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig_h = max(4.0, 0.35 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    ax.barh(plot_df[y_col].astype(str), pd.to_numeric(plot_df[x_col], errors='coerce'))
    if c3_alpha is not None:
        ax.axvline(float(c3_alpha), linestyle='--', linewidth=1)
    ax.set_xlabel('Oracle-alpha Δnoop')
    ax.set_title(title)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _save_frontier_plot(frontier: pd.DataFrame, path: Path, *, strict: bool=False) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if plt is None:
        return err
    required = {'frontier_budget', 'frontier_variant', 'mean_delta_R_score_alpha'}
    if frontier.empty or not required.issubset(frontier.columns):
        return 'missing_frontier_columns'
    path.parent.mkdir(parents=True, exist_ok=True)
    df = frontier.copy()
    df['frontier_budget'] = pd.to_numeric(df['frontier_budget'], errors='coerce')
    df['mean_delta_R_score_alpha'] = pd.to_numeric(df['mean_delta_R_score_alpha'], errors='coerce')
    agg = df.groupby(['frontier_budget', 'frontier_variant'], as_index=False)['mean_delta_R_score_alpha'].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    for variant, sub in agg.groupby('frontier_variant'):
        sub = sub.sort_values('frontier_budget')
        ax.plot(sub['frontier_budget'], sub['mean_delta_R_score_alpha'], marker='o', label=str(variant))
    ax.axhline(C3_ALPHA_DEFAULT, linestyle='--', linewidth=1, label='C3 reference')
    ax.set_xlabel('L1 budget')
    ax.set_ylabel('Mean Oracle-alpha Δnoop')
    ax.set_title('Evaluator-only N1 budget response curve')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _save_winrate_plot(winrate: pd.DataFrame, path: Path, *, strict: bool=False) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if plt is None:
        return err
    if winrate.empty or 'win_rate_excluding_ties' not in winrate.columns:
        return 'missing_winrate_columns'
    path.parent.mkdir(parents=True, exist_ok=True)
    df = winrate.copy()
    df['win_rate_pct'] = pd.to_numeric(df['win_rate_excluding_ties'], errors='coerce') * 100.0
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(df['information_condition'].astype(str), df['win_rate_pct'])
    ax.set_ylabel('Win rate excluding ties (%)')
    ax.set_ylim(0, 100)
    ax.set_title('N3 C6 win rate vs C3')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _save_heterogeneity_plot(hetero: pd.DataFrame, path: Path, *, strict: bool=False) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if plt is None:
        return err
    if hetero.empty or not {'information_condition', 'slice', 'mean_gap'}.issubset(hetero.columns):
        return 'missing_heterogeneity_columns'
    path.parent.mkdir(parents=True, exist_ok=True)
    df = hetero.copy()
    df['slice_id'] = df.groupby('information_condition').cumcount() + 1
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ic, sub in df.groupby('information_condition'):
        sub = sub.sort_values('slice_id')
        ax.plot(sub['slice_id'], pd.to_numeric(sub['mean_gap'], errors='coerce'), marker='o', label=str(ic))
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel('log_assets quartile')
    ax.set_ylabel('C6−C3 alpha gap')
    ax.set_title('N6 exploratory size heterogeneity')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _save_icc_probe_plot(channel: pd.DataFrame, path: Path, *, strict: bool=False) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if plt is None:
        return err
    required = {'channel', 'rate'}
    if channel.empty or not required.issubset(channel.columns):
        return 'missing_icc_probe_columns'
    labels = {'firm_recognition': 'Firm recognized', 'numeric_debt_ratio_recall': 'Numeric debt-ratio recall', 'numeric_contamination_flag': 'Within-tolerance recall', 'parse_failure': 'Parse failure'}
    df = channel.copy()
    df['label'] = df['channel'].map(labels).fillna(df['channel'].astype(str))
    df['rate_pct'] = pd.to_numeric(df['rate'], errors='coerce') * 100.0
    df = df[df['rate_pct'].notna()]
    if df.empty:
        return 'empty_icc_probe_data'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(df['label'], df['rate_pct'])
    ax.set_ylabel('Share of 575 firms (%)')
    ax.set_ylim(0, 100)
    ax.set_title('IC-c prior-knowledge probe')
    ax.tick_params(axis='x', rotation=18)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _save_n5m_matched_frontier_plot(df: pd.DataFrame, path: Path, *, strict: bool) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if err:
        return err
    required = {'budget_label', 'mean_C4', 'mean_C6', 'oracle_backend'}
    if df.empty or not required.issubset(df.columns):
        return 'missing_n5m_frontier_columns'
    plot = df[df['oracle_backend'].astype(str).str.lower().eq('alpha')].copy()
    order = ['0p75', '1p27', '2p00', 'unbounded']
    plot['_order'] = plot['budget_label'].map({v: i for i, v in enumerate(order)})
    plot = plot.sort_values('_order')
    if len(plot) != 4:
        return 'n5m_frontier_alpha_not_four_arms'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(plot['budget_label'], plot['mean_C4'], marker='o', label='C4 direct generation')
    ax.plot(plot['budget_label'], plot['mean_C6'], marker='o', label='C6 reference-revision')
    ax.set_ylabel('Mean Oracle-alpha delta vs no-action')
    ax.set_xlabel('Generation-time L1 budget')
    ax.set_title('N5M matched budget frontier')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _save_n5m_winrate_plot(df: pd.DataFrame, path: Path, *, strict: bool) -> str | None:
    plt, err = _try_import_matplotlib(strict)
    if err:
        return err
    required = {'budget_label', 'oracle_backend', 'c6_win_rate_non_tie'}
    if df.empty or not required.issubset(df.columns):
        return 'missing_n5m_winrate_columns'
    plot = df[df['oracle_backend'].astype(str).str.lower().eq('alpha')].copy()
    order = ['0p75', '1p27', '2p00', 'unbounded']
    plot['_order'] = plot['budget_label'].map({v: i for i, v in enumerate(order)})
    plot = plot.sort_values('_order')
    if len(plot) != 4:
        return 'n5m_winrate_alpha_not_four_arms'
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(plot['budget_label'], pd.to_numeric(plot['c6_win_rate_non_tie'], errors='coerce') * 100.0)
    ax.axhline(50.0, linestyle='--', linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_ylabel('C6 win rate among non-ties (%)')
    ax.set_xlabel('Generation-time L1 budget')
    ax.set_title('N5M firm-level C6 vs C4 outcomes (Oracle-alpha)')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return None

def _frontier_state(layout, *, matched: bool, required: bool, matched_paper_use: str='historical') -> dict[str, Any]:
    source_dir = layout.n5_matched_budget_frontier_holm if matched else layout.n5_budget_frontier_holm
    manifest_path = source_dir / 'n5_budget_frontier_holm_manifest.json'
    files = sorted((path for path in source_dir.glob('*') if path.is_file())) if source_dir.is_dir() else []
    if not files:
        if required:
            raise FileNotFoundError(f'Required N5M matched frontier is absent: {source_dir}')
        return {'status': 'NOT_AVAILABLE_OPTIONAL', 'source_dir': str(source_dir), 'reason': 'The supplementary legacy N5F frontier has not been run.'}
    if not manifest_path.is_file():
        raise FileNotFoundError(f'Frontier directory is populated but manifest is missing: {source_dir}')
    metadata = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    expected_design = 'matched_c4_c6' if matched else 'legacy_c6_only'
    expected_role = 'paper_n5_matched_budget_frontier_icb' if matched else 'paper_n5_budget_frontier_icb'
    if metadata.get('status') != 'PASS':
        raise RuntimeError(f'Frontier manifest is not PASS: {manifest_path}')
    schema_version = metadata.get('schema_version')
    if matched:
        if schema_version != 'n5_generation_budget_frontier_holm_v2':
            raise RuntimeError(f'Required matched frontier must use v2 schema: {manifest_path}')
        if metadata.get('design') != expected_design or metadata.get('run_role') != expected_role:
            raise RuntimeError(f"Frontier identity mismatch: design={metadata.get('design')!r}, role={metadata.get('run_role')!r}")
        audit_paths = [source_dir / 'n5_budget_frontier_control_alignment_audit.csv', source_dir / 'n5_budget_frontier_budget_contract_audit.csv']
    else:
        if schema_version not in {'n5_generation_budget_frontier_holm_v1', 'n5_generation_budget_frontier_holm_v2'}:
            raise RuntimeError(f'Unsupported supplementary N5F frontier schema: {manifest_path}')
        if metadata.get('run_role') != expected_role:
            raise RuntimeError(f'Supplementary N5F role mismatch: {manifest_path}')
        if schema_version == 'n5_generation_budget_frontier_holm_v2':
            if metadata.get('design') != expected_design:
                raise RuntimeError(f'Supplementary N5F v2 design mismatch: {manifest_path}')
            audit_paths = [source_dir / 'n5_budget_frontier_control_alignment_audit.csv', source_dir / 'n5_budget_frontier_budget_contract_audit.csv']
        else:
            audit_paths = [source_dir / 'n5_budget_frontier_c4_stability_audit.csv']
    required_paths = [manifest_path, source_dir / 'n5_budget_frontier_table_patch.csv', *audit_paths]
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f'PASS frontier is missing v2 outputs: {missing}')
    return {'status': 'AVAILABLE_PASS', 'source_dir': str(source_dir), 'manifest': str(manifest_path), 'run_role': metadata.get('run_role'), 'information_condition': metadata.get('information_condition'), 'design': metadata.get('design') or 'legacy_c6_only', 'schema_version': schema_version, 'paper_use': matched_paper_use if matched else 'supplementary'}

def _generation_frontier_state(layout) -> dict[str, Any]:
    return _frontier_state(layout, matched=False, required=False)

def _matched_frontier_state(layout, *, paper_use: str) -> dict[str, Any]:
    return _frontier_state(layout, matched=True, required=False, matched_paper_use=paper_use)

def _resolve_analysis_profile_contract(analysis_dir: Path, *, project_root: Path | None, analysis_profile: str | None, n5m_paper_use: str | None) -> tuple[str, str, dict[str, Any]]:
    layout = build_layout(Path(analysis_dir).resolve())
    analysis_manifest_path = layout.manifest / 'paper_repro_analysis_manifest.json'
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding='utf-8-sig')) if analysis_manifest_path.is_file() else {}
    declared_profile = analysis_manifest.get('analysis_profile_id') or analysis_manifest.get('analysis_profile')
    if analysis_profile is None:
        if declared_profile == 'historical_20260715':
            raise ValueError('Historical paper assets require an explicit --analysis-profile contract')
        analysis_profile = str(declared_profile or 'current_comprehensive')
    if declared_profile is not None and str(declared_profile) != analysis_profile:
        raise ValueError(f'Paper-assets profile disagrees with analysis manifest: expected={analysis_profile!r}, manifest={declared_profile!r}')
    root = Path(project_root).resolve() if project_root is not None else Path(__file__).resolve().parents[3]
    profile = load_profile(root, analysis_profile)
    resolved_n5m_paper_use = str(profile['llm']['n5_matched_budget_frontier']['paper_use'])
    if n5m_paper_use is not None and n5m_paper_use != resolved_n5m_paper_use:
        raise ValueError(f'Explicit N5M paper use disagrees with named profile: profile={resolved_n5m_paper_use!r}, explicit={n5m_paper_use!r}')
    return (analysis_profile, resolved_n5m_paper_use, analysis_manifest)

def run_assets(analysis_dir: Path, out_dir: Path | None=None, *, strict: bool=False, c3_alpha: float=C3_ALPHA_DEFAULT, project_root: Path | None=None, analysis_profile: str | None=None, n5m_paper_use: str | None=None) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir).resolve()
    if out_dir is not None and Path(out_dir).resolve() != analysis_dir:
        raise ValueError('Paper assets must remain inside the canonical analysis directory; --out-dir is not supported.')
    analysis_profile, resolved_n5m_paper_use, analysis_manifest = _resolve_analysis_profile_contract(analysis_dir, project_root=project_root, analysis_profile=analysis_profile, n5m_paper_use=n5m_paper_use)
    n5m_paper_use = resolved_n5m_paper_use
    layout = build_layout(analysis_dir)
    ensure_layout(layout)
    tables_dir = layout.tables
    figures_dir = layout.figures
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = layout.paper_assets / 'paper_assets_manifest.json'
    manifest: dict[str, Any] = {'schema_version': 'paper_repro_assets_v6', 'created_utc': _now(), 'status': 'RUNNING', 'analysis_profile_id': analysis_profile, 'resolved_n5m_paper_use': resolved_n5m_paper_use, 'eligibility_contract_id': analysis_manifest.get('eligibility_contract_id'), 'eligible_catalog_role_relative_path': analysis_manifest.get('eligible_catalog_role_relative_path'), 'eligible_catalog_row_count': analysis_manifest.get('eligible_catalog_row_count'), 'eligible_catalog_unique_count': analysis_manifest.get('eligible_catalog_unique_count'), 'analysis_dir': str(analysis_dir), 'paper_tables_dir': str(tables_dir), 'paper_figures_dir': str(figures_dir), 'tables': {}, 'figures': {}, 'required_inputs': {}, 'optional_inputs': {}, 'notes': ['Presentation assets only; authoritative statistics remain in task-specific source outputs.', 'No LLM calls, RL training, Oracle refitting, or mutation of data/final_freeze occurs here.', 'The main harness-vs-backend decomposition uses only the crossed IC-b primary/supplementary panel.', 'N5/N5F/N5M are excluded from the main crossed harness-vs-backend variance decomposition; N5M paper use follows the named analysis profile.']}
    required_dirs = {'b2_gap': layout.b2_gap, 'test3': layout.test3, 'structural_slice': layout.structural_slice, 'holm': layout.holm, 'reference_quality_acceptance': layout.reference_quality_acceptance, 'ablation': layout.ablation, 'signflip': layout.signflip, 'n5_holm': layout.n5_holm, 'main_harness_backend_decomposition': layout.main_harness_backend_decomposition, 'frontier': layout.frontier, 'winrate': layout.winrate, 'icc_probe': layout.icc_probe}
    missing = [name for name, path in required_dirs.items() if not path.is_dir()]
    manifest['required_inputs'] = {name: str(path) for name, path in required_dirs.items()}
    if missing:
        raise FileNotFoundError(f'Required canonical analysis directories are missing: {missing}')
    try:
        generation_frontier_state = _generation_frontier_state(layout)
        matched_frontier_state = _matched_frontier_state(layout, paper_use=n5m_paper_use)
        manifest['optional_inputs']['n5_generation_budget_frontier'] = generation_frontier_state
        manifest['optional_inputs']['n5_matched_budget_frontier_historical'] = matched_frontier_state
        b2_gap = collect_b2_gap(layout.b2_gap)
        test3 = collect_test3(layout.test3)
        structural = collect_structural_slice(layout.structural_slice)
        reference_quality = collect_reference_quality(layout.reference_quality_acceptance)
        signflip = collect_signflip(layout.signflip)
        ablation = collect_postfreeze_ablation(layout.ablation)
        n5 = collect_n5_table(layout.n5_holm)
        n5_budget_frontier = collect_n5_budget_frontier(layout.n5_budget_frontier_holm) if generation_frontier_state['status'] == 'AVAILABLE_PASS' else None
        n5m_matched_frontier = None
        n5m_posthoc: dict[str, pd.DataFrame] = {}
        n5m_adaptive_selection: dict[str, pd.DataFrame] = {}
        if matched_frontier_state['status'] == 'AVAILABLE_PASS':
            n5m_matched_frontier = collect_n5_budget_frontier(layout.n5_matched_budget_frontier_holm)
            if layout.n5m_posthoc.is_dir():
                n5m_posthoc = collect_n5m_posthoc(layout.n5m_posthoc)
            if layout.n5m_adaptive_selection.is_dir():
                n5m_adaptive_selection = collect_n5m_adaptive_selection(layout.n5m_adaptive_selection)
            if n5m_paper_use == 'primary':
                manifest['notes'].append('E1/N5M assets were emitted as primary evidence for the named historical analysis profile.')
            else:
                manifest['notes'].append('E1/N5M assets were emitted as SUPERSEDED_FIRST_OBSERVATION historical appendix evidence only.')
        else:
            manifest['notes'].append('E1/N5M historical assets were not supplied and are not required for canonical E3 paper assets.')
        main_harness_backend = collect_main_harness_backend_decomposition(layout.main_harness_backend_decomposition)
        winrate = collect_winrate(layout.winrate)
        hetero = collect_heterogeneity(layout.winrate)
        frontier = collect_frontier(layout.frontier)
        icc_probe, icc_familiarity = collect_icc_probe(layout.icc_probe)
        datasets = {'b2_gap_key_metrics': b2_gap, 'test3_property_summary': test3, 'structural_event_slice': structural, 'reference_quality_acceptance_primary_c6': reference_quality, 'signflip_mean_null_summary': signflip, 'postfreeze_ablation_compact': ablation, 'n5_table_7_10c_reprint': n5, 'n5_winrate_vs_c3': winrate, 'n6_log_assets_heterogeneity': hetero, 'icc_probe_channel_summary': icc_probe, 'icc_probe_familiarity_distribution': icc_familiarity, **main_harness_backend}
        if n5m_matched_frontier is not None:
            n5m_prefix = '' if n5m_paper_use == 'primary' else 'historical_'
            datasets[f'{n5m_prefix}n5m_matched_budget_frontier_alpha'] = n5m_matched_frontier
            datasets.update({f'{n5m_prefix}{name}': frame for name, frame in n5m_posthoc.items()})
            datasets.update({f'{n5m_prefix}{name}': frame for name, frame in n5m_adaptive_selection.items()})
        if n5_budget_frontier is not None:
            datasets['n5f_legacy_budget_frontier_alpha'] = n5_budget_frontier
        else:
            for suffix in ('csv', 'md'):
                stale = tables_dir / f'n5f_legacy_budget_frontier_alpha.{suffix}'
                if stale.exists():
                    stale.unlink()
            manifest['notes'].append('The supplementary legacy N5F frontier was unavailable; no N5F paper table was emitted.')
        empty_required = [name for name, frame in datasets.items() if frame.empty]
        if empty_required:
            raise RuntimeError(f'Required paper asset inputs produced empty tables: {empty_required}')
        for name, frame in datasets.items():
            manifest['tables'][name] = _write_table(frame, tables_dir, name)
        frontier_agg = frontier.copy()
        if frontier_agg.empty:
            raise RuntimeError('Required frontier_grid_raw.csv is empty.')
        frontier_agg['frontier_budget'] = pd.to_numeric(frontier_agg['frontier_budget'], errors='coerce')
        frontier_agg['mean_delta_R_score_alpha'] = pd.to_numeric(frontier_agg['mean_delta_R_score_alpha'], errors='coerce')
        frontier_agg = frontier_agg.groupby(['frontier_budget', 'frontier_variant'], as_index=False)['mean_delta_R_score_alpha'].mean().sort_values(['frontier_budget', 'frontier_variant'])
        manifest['tables']['n1_frontier_alpha_mean_by_budget'] = _write_table(frontier_agg, tables_dir, 'n1_frontier_alpha_mean_by_budget')
        ladder = make_paper_ladder_table(ablation, signflip, n5, winrate, c3_alpha)
        if ladder.empty:
            raise RuntimeError('Free-form contrast ladder could not be constructed.')
        manifest['tables']['freeform_contrast_ladder_alpha'] = _write_table(ladder, tables_dir, 'freeform_contrast_ladder_alpha', notes='C3 is an interpretive anchor; use task-specific paired-inference outputs for significance.')
        plot = ladder[ladder['information_condition'].astype(str) == 'IC-b'].copy()
        plot['label'] = plot['information_condition'].astype(str) + ' · ' + plot['family'].astype(str)
        figures = [('freeform_contrast_ladder_alpha', figures_dir / 'freeform_contrast_ladder_alpha.png', _save_barh_alpha(plot, figures_dir / 'freeform_contrast_ladder_alpha.png', title='Free-form contrast ladder (Oracle-alpha)', c3_alpha=c3_alpha, strict=strict)), ('n1_frontier_alpha_curve', figures_dir / 'n1_frontier_alpha_curve.png', _save_frontier_plot(frontier, figures_dir / 'n1_frontier_alpha_curve.png', strict=strict)), ('n3_winrate_vs_c3', figures_dir / 'n3_winrate_vs_c3.png', _save_winrate_plot(winrate, figures_dir / 'n3_winrate_vs_c3.png', strict=strict)), ('n6_log_assets_heterogeneity', figures_dir / 'n6_log_assets_heterogeneity.png', _save_heterogeneity_plot(hetero, figures_dir / 'n6_log_assets_heterogeneity.png', strict=strict)), ('icc_probe_channel_rates', figures_dir / 'icc_probe_channel_rates.png', _save_icc_probe_plot(icc_probe, figures_dir / 'icc_probe_channel_rates.png', strict=strict)), ('structural_event_slice', figures_dir / 'structural_event_slice.png', _save_structural_slice_plot(structural, figures_dir / 'structural_event_slice.png', strict=strict))]
        if n5m_matched_frontier is not None:
            n5m_prefix = '' if n5m_paper_use == 'primary' else 'historical_'
            figures.append((f'{n5m_prefix}n5m_matched_budget_frontier_alpha', figures_dir / f'{n5m_prefix}n5m_matched_budget_frontier_alpha.png', _save_n5m_matched_frontier_plot(n5m_matched_frontier, figures_dir / f'{n5m_prefix}n5m_matched_budget_frontier_alpha.png', strict=strict)))
            win_table = n5m_posthoc.get('n5m_win_tie_loss_by_budget_oracle')
            if win_table is not None:
                figures.append((f'{n5m_prefix}n5m_win_tie_loss_alpha', figures_dir / f'{n5m_prefix}n5m_win_tie_loss_alpha.png', _save_n5m_winrate_plot(win_table, figures_dir / f'{n5m_prefix}n5m_win_tie_loss_alpha.png', strict=strict)))
        signplot = signflip.copy()
        signplot['label'] = signplot['information_condition'].astype(str)
        figures.append(('signflip_mean_null_alpha', figures_dir / 'signflip_mean_null_alpha.png', _save_barh_alpha(signplot, figures_dir / 'signflip_mean_null_alpha.png', title='Sign-flip mean null collapse', x_col='alpha_delta_noop', y_col='label', c3_alpha=0.0, strict=strict)))
        for name, path, err in figures:
            if err is not None:
                raise RuntimeError(f'Figure generation failed for {name}: {err}')
            manifest['figures'][name] = {'path': str(path), 'status': 'PASS'}
        readme = layout.paper_assets / 'README.md'
        readme.write_text('# Generated paper assets\n\nAll files in this directory are generated from the canonical post-freeze analysis tree. Tables and figures are drafting/defense aids; task-specific subdirectories retain authoritative statistical outputs.\n', encoding='utf-8')
        from credit_recourse.analysis.thesis_visual_builder import run_all_visuals
        visual_manifest = run_all_visuals(None, analysis_dir, figures_dir=figures_dir, tables_dir=tables_dir, strict=False)
        manifest['human_visual_catalog'] = {'status': visual_manifest.get('status'), 'registered_asset_count': visual_manifest.get('registered_asset_count'), 'attempted_asset_count': visual_manifest.get('attempted_asset_count'), 'status_counts': visual_manifest.get('status_counts'), 'plot_data_dir': visual_manifest.get('plot_data_dir'), 'registry_dir': visual_manifest.get('registry_dir')}
        manifest['status'] = 'PASS'
    except Exception as exc:
        manifest['status'] = 'FAIL'
        manifest['error'] = repr(exc)
        raise
    finally:
        _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Generate paper-facing tables/figures from unified analysis outputs.')
    ap.add_argument('--analysis-dir', required=True, help='Canonical data/analysis/paper_repro directory')
    ap.add_argument('--out-dir', default=None, help='Optional output root. Defaults to --analysis-dir')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--c3-alpha', type=float, default=C3_ALPHA_DEFAULT)
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--analysis-profile', default=None)
    ap.add_argument('--n5m-paper-use', choices=['primary', 'historical'], default=None)
    return ap

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_assets(Path(args.analysis_dir), Path(args.out_dir) if args.out_dir else None, strict=args.strict, c3_alpha=args.c3_alpha, project_root=Path(args.project_root), analysis_profile=args.analysis_profile, n5m_paper_use=args.n5m_paper_use)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
