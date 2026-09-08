from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from credit_recourse.contracts.paper_reproduction import discover_archived_runs, load_profile, select_exact_role
from .claim_evidence_common import repo_rel, write_csv, write_json
METRICS = ('projection_distance', 'l1', 'active_dimensions', 'format_failures', 'action_contract_consistency', 'reviewability_proxy')
BINARY_METRICS = {'format_failures', 'action_contract_consistency'}

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
    return tuple(map(float, np.quantile(means, [0.025, 0.975])))

def _audit_path(run_dir: Path) -> Path:
    candidates = [run_dir / 'stage7_llm_action_generation/llm_stage7_failure_audit.csv', run_dir / 'stage7_action_generation/llm_stage7_failure_audit.csv']
    matches = [path for path in candidates if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f'exactly one Stage7 failure audit required under {run_dir}; found {matches}')
    return matches[0]

def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {'true': True, '1': True, 'yes': True, 'y': True, 'false': False, '0': False, 'no': False, 'n': False, 'nan': False, 'none': False, '': False}
    unknown = sorted(set(normalized) - set(mapping))
    if unknown:
        raise RuntimeError(f'cannot parse boolean values: {unknown[:10]}')
    return normalized.map(mapping).astype(bool)

def _metric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'row_id', 'policy', 'mode', 'information_condition', 'projection_distance', 'out_of_library', 'routed_to_simulator', 'action_l1_norm', 'action_nonzero_dim_count', 'direction_error_auto', 'magnitude_error_auto'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'H2 Stage7 audit missing required columns: {missing}')
    out = frame.loc[frame['mode'].astype(str).eq('free_form_10d') & frame['information_condition'].astype(str).eq('IC-b') & frame['policy'].astype(str).isin(['C4', 'C5', 'C6', 'C7'])].copy()
    if out.duplicated(['row_id', 'policy']).any():
        raise RuntimeError('H2 input has duplicate row_id/policy keys')
    counts = out.groupby('policy')['row_id'].nunique().to_dict()
    if counts != {'C4': 575, 'C5': 575, 'C6': 575, 'C7': 575}:
        raise RuntimeError(f'H2 requires 575 firms for C4/C5/C6/C7, got {counts}')
    out['projection_distance'] = pd.to_numeric(out['projection_distance'], errors='coerce')
    out['l1'] = pd.to_numeric(out['action_l1_norm'], errors='raise')
    out['active_dimensions'] = pd.to_numeric(out['action_nonzero_dim_count'], errors='raise')
    routed = _as_bool(out['routed_to_simulator'])
    out_of_library = _as_bool(out['out_of_library'])
    out['format_failures'] = (~routed | out_of_library).astype(int)
    direction_ok = ~_as_bool(out['direction_error_auto'])
    magnitude_ok = ~_as_bool(out['magnitude_error_auto'])
    out['action_contract_consistency'] = (direction_ok & magnitude_ok).astype(int)
    projection_penalty = np.where(routed, np.minimum(out['projection_distance'].fillna(0.0).clip(lower=0.0), 1.0), 0.0)
    out['reviewability_proxy'] = 1.0 - (out['format_failures'] + (1 - out['action_contract_consistency']) + projection_penalty) / 3.0
    for metric in [name for name in METRICS if name != 'projection_distance']:
        values = pd.to_numeric(out[metric], errors='coerce')
        if not np.isfinite(values.to_numpy()).all():
            raise RuntimeError(f'H2 derived metric contains non-finite values: {metric}')
    projection = pd.to_numeric(out['projection_distance'], errors='coerce')
    invalid_projection = projection.isna() & routed
    if invalid_projection.any():
        raise RuntimeError('projection_distance may be missing only for rows that failed routing')
    return out

def _paired_test(a: np.ndarray, b: np.ndarray, metric: str, seed: int) -> dict[str, Any]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    total_pairs = len(a)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if len(a) < 500:
        raise RuntimeError(f'{metric}: fewer than 500 complete paired firms ({len(a)}/{total_pairs})')
    gap = a - b
    ci_low, ci_high = _bootstrap_ci(gap, seed=seed)
    if metric in BINARY_METRICS:
        discordant_pos = int(((a == 1) & (b == 0)).sum())
        discordant_neg = int(((a == 0) & (b == 1)).sum())
        discordant = discordant_pos + discordant_neg
        p_raw = 1.0 if discordant == 0 else float(binomtest(discordant_pos, discordant, p=0.5, alternative='two-sided').pvalue)
        test = 'paired_exact_mcnemar_binomial'
    else:
        p_raw = 1.0 if np.allclose(gap, 0.0, rtol=0.0, atol=0.0) else float(wilcoxon(gap, zero_method='pratt', alternative='two-sided', method='auto').pvalue)
        test = 'paired_wilcoxon_pratt_two_sided'
    return {'n_total_pairs': total_pairs, 'n_pairs': len(gap), 'n_omitted_pairs': total_pairs - len(gap), 'estimate': float(np.mean(gap)), 'median_diff': float(np.median(gap)), 'ci_low': ci_low, 'ci_high': ci_high, 'p_raw': p_raw, 'test': test}

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    profile = load_profile(root)
    records = discover_archived_runs(root, profile)
    role_map = {'gpt54mini': str(profile['llm']['primary']['run_role']), 'gpt41mini': str(profile['llm']['supplementary'][0]['run_role']), 'haiku45': str(profile['llm']['supplementary'][1]['run_role'])}
    rows: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []
    run_sources: dict[str, str] = {}
    seed_counter = 0
    for backend, role in role_map.items():
        record = select_exact_role(records, run_role=role, information_condition='IC-b')
        audit_path = _audit_path(record.run_dir)
        run_sources[backend] = repo_rel(root, audit_path)
        frame = _metric_frame(pd.read_csv(audit_path, encoding='utf-8-sig'))
        for contrast, policy_a, policy_b in (('C5-C4', 'C5', 'C4'), ('C7-C6', 'C7', 'C6')):
            left = frame.loc[frame['policy'].eq(policy_a)].set_index('row_id')
            right = frame.loc[frame['policy'].eq(policy_b)].set_index('row_id')
            if set(left.index) != set(right.index):
                raise RuntimeError(f'{backend} {contrast}: paired row_id universe mismatch')
            left = left.sort_index()
            right = right.reindex(left.index)
            for metric in METRICS:
                test = _paired_test(left[metric].to_numpy(), right[metric].to_numpy(), metric, seed=20260719 + seed_counter)
                seed_counter += 1
                row = {'backend': backend, 'run_role': role, 'run_label': record.run_label, 'information_condition': 'IC-b', 'mode': 'free_form_10d', 'contrast': contrast, 'metric': metric, **test, 'holm_family': 'H2_three_backends_two_contrasts_six_metrics', 'execution_contract': 'LEGACY_THREE_BACKEND_REASONING', 'not_e4_matched_budget_qc': True, 'reviewability_proxy_definition': '1 - (format_failure + (1-action_contract_consistency) + min(projection_distance_if_routed,1)) / 3', 'source_path': repo_rel(root, audit_path)}
                rows.append(row)
                raw_pvalues.append(float(test['p_raw']))
    adjusted = _holm(raw_pvalues)
    for row, p_holm in zip(rows, adjusted):
        row['p_holm'] = p_holm
        row['decision'] = 'SIGNIFICANT' if p_holm < 0.05 else 'NOT_SIGNIFICANT'
    if len(rows) != 36:
        raise RuntimeError(f'H2 exact contract requires 36 rows, got {len(rows)}')
    output_dir = root / 'data/analysis/paper_repro/02_llm_hypothesis_tests/h2_format_reviewability'
    output = output_dir / 'h2_format_reviewability_summary.csv'
    write_csv(output, rows)
    manifest = {'schema_version': 'h2_format_reviewability_v3', 'status': 'PASS', 'execution_contract': 'LEGACY_THREE_BACKEND_REASONING', 'backends': sorted(role_map), 'contrasts': ['C5-C4', 'C7-C6'], 'metrics': list(METRICS), 'metric_definitions': {'format_failures': '1 when routing failed or the action is outside the registered candidate library', 'action_contract_consistency': '1 when automated direction and magnitude contract checks both pass; this is not a text-explanation semantic-consistency measure', 'reviewability_proxy': '1 - mean(format_failure, action-contract inconsistency, clipped projection distance among routed rows)'}, 'row_count': len(rows), 'n_pairs_per_cell': 575, 'holm_family': 'H2_three_backends_two_contrasts_six_metrics', 'source_paths': run_sources}
    write_json(output_dir / 'h2_format_reviewability_manifest.json', manifest)
    return manifest

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
