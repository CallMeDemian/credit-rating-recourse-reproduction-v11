from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from credit_recourse.analysis.rl_seven_seed_summary import EXPECTED_SEEDS, ORACLES, _candidate_row, _resolve_archive
from .claim_evidence_common import repo_rel, write_csv, write_json
RUN_ROOT_REL = 'data/final_freeze/rl_reproducibility/seven_seed'
SUMMARY_ROOT_REL = 'data/analysis/paper_repro/03_output_contract_diagnostics/rl_seven_seed'

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, encoding='utf-8-sig')

def _frames_semantically_equal(left: pd.DataFrame, right: pd.DataFrame, *, key: list[str]) -> bool:
    if set(left.columns) != set(right.columns):
        return False
    ordered = sorted(left.columns)
    a = left.loc[:, ordered].sort_values(key).reset_index(drop=True)
    b = right.loc[:, ordered].sort_values(key).reset_index(drop=True)
    if len(a) != len(b):
        return False
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=False, atol=1e-12, rtol=0)
    except AssertionError:
        return False
    return True

def _resolve_policy_summary(archive: Path) -> Path:
    canonical = [archive / '00_POLICY_SUMMARY_CORE.csv', archive / 'outputs/stage6_multi_oracle_eval/final_policy_summary.csv']
    found = [path for path in canonical if path.is_file()]
    if not found:
        fallback = archive / 'outputs/stage6_candidate_selector_eval/final_policy_summary.csv'
        if fallback.is_file():
            return fallback
        raise RuntimeError(f'no archived Stage6 final policy summary under {archive}')
    reference = _candidate_row(pd.read_csv(found[0], encoding='utf-8-sig'))
    signature_columns = ['policy', 'n', *(f'mean_delta_R_score_{o}' for o in ORACLES)]
    for other in found[1:]:
        candidate = _candidate_row(pd.read_csv(other, encoding='utf-8-sig'))
        for column in signature_columns:
            if column not in reference.index or column not in candidate.index:
                raise RuntimeError(f'duplicate policy summaries lack comparison column {column}: {found[0]}, {other}')
            if column == 'policy':
                equal = str(reference[column]) == str(candidate[column])
            else:
                equal = np.isclose(float(reference[column]), float(candidate[column]), atol=1e-12, rtol=0)
            if not equal:
                raise RuntimeError(f'semantically inconsistent duplicate policy summaries: {found[0]}, {other}, column={column}')
    return found[0]

def _resolve_firm_eval(archive: Path, policy: str) -> Path:
    canonical = [archive / '00_MULTI_ORACLE_POLICY_EVAL.parquet', archive / 'outputs/stage6_multi_oracle_eval/multi_oracle_policy_eval.parquet']
    found = [path for path in canonical if path.is_file()]
    if not found:
        fallback = archive / 'outputs/stage6_candidate_selector_eval/multi_oracle_policy_eval.parquet'
        if fallback.is_file():
            return fallback
        raise RuntimeError(f'no archived Stage6 firm ledger under {archive}')
    reference = _candidate_firm_rows(_read_table(found[0]), policy)
    for other in found[1:]:
        candidate = _candidate_firm_rows(_read_table(other), policy)
        if not _frames_semantically_equal(reference, candidate, key=['row_id']):
            raise RuntimeError(f'semantically inconsistent duplicate Stage6 firm ledgers: {found[0]}, {other}')
    return found[0]

def _candidate_firm_rows(frame: pd.DataFrame, policy: str, expected_n: int=575) -> pd.DataFrame:
    required = {'row_id', 'policy', 'candidate_id', *(f'delta_R_score_{o}' for o in ORACLES)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'firm-level Stage6 ledger missing columns: {missing}')
    selected = frame.loc[frame['policy'].astype(str).eq(policy), list(required)].copy()
    if len(selected) != expected_n or selected['row_id'].nunique() != expected_n:
        raise RuntimeError(f"policy={policy}: expected {expected_n} unique firms, got rows={len(selected)}, firms={selected['row_id'].nunique()}")
    if selected['row_id'].duplicated().any():
        raise RuntimeError(f'policy={policy}: duplicate row_id')
    for oracle in ORACLES:
        selected[f'delta_R_score_{oracle}'] = pd.to_numeric(selected[f'delta_R_score_{oracle}'], errors='raise')
    return selected.sort_values('row_id').reset_index(drop=True)

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    run_root = root / RUN_ROOT_REL
    grid_path = run_root / 'UNIFIED_RUN_SUMMARY.csv'
    if not grid_path.is_file():
        raise FileNotFoundError(grid_path)
    grid = pd.read_csv(grid_path, encoding='utf-8-sig')
    required_grid = {'status', 'archive_root', 'stage3_seed', 'stage4_seed', 'stage5_seed', 'seed_lineage_mode'}
    missing = sorted(required_grid - set(grid.columns))
    if missing:
        raise RuntimeError(f'UNIFIED_RUN_SUMMARY missing columns: {missing}')
    completed = grid.loc[grid['status'].astype(str).str.startswith('COMPLETED')].copy()
    if len(completed) != 7:
        raise RuntimeError(f'expected exactly seven completed seed cells, got {len(completed)}')
    for column in ('stage3_seed', 'stage4_seed', 'stage5_seed'):
        completed[column] = pd.to_numeric(completed[column], errors='raise').astype(int)
    for row in completed.itertuples(index=False):
        seed_set = {int(row.stage3_seed), int(row.stage4_seed), int(row.stage5_seed)}
        if len(seed_set) != 1 or str(row.seed_lineage_mode) != 'aligned':
            raise RuntimeError(f"unaligned seed lineage in cell={getattr(row, 'cell_id', '')}: {seed_set}")
    observed = tuple(sorted(completed['stage3_seed'].tolist()))
    if observed != EXPECTED_SEEDS:
        raise RuntimeError(f'expected aligned seeds {EXPECTED_SEEDS}, got {observed}')
    seed_rows: list[dict[str, Any]] = []
    frames: dict[int, pd.DataFrame] = {}
    input_rows: list[dict[str, Any]] = [{'artifact': 'UNIFIED_RUN_SUMMARY', 'path': repo_rel(root, grid_path)}]
    for row in completed.sort_values('stage3_seed').itertuples(index=False):
        seed = int(row.stage3_seed)
        archive = _resolve_archive(run_root, str(row.archive_root))
        summary_path = _resolve_policy_summary(archive)
        summary = pd.read_csv(summary_path, encoding='utf-8-sig')
        selected_summary = _candidate_row(summary)
        policy = str(selected_summary['policy'])
        firm_path = _resolve_firm_eval(archive, policy)
        firm = _candidate_firm_rows(_read_table(firm_path), policy)
        frames[seed] = firm
        modal_action = str(firm['candidate_id'].astype(str).value_counts().idxmax())
        modal_share = float(firm['candidate_id'].astype(str).value_counts(normalize=True).iloc[0])
        seed_item: dict[str, Any] = {'record_type': 'SEED_POLICY_MEAN', 'record_key': f'seed:{seed}', 'seed': seed, 'seed_a': np.nan, 'seed_b': np.nan, 'policy': policy, 'n_firms': 575, 'modal_action': modal_action, 'modal_action_share': modal_share, 'action_identity_rate': np.nan, 'source_policy_summary': repo_rel(root, summary_path), 'source_firm_ledger': repo_rel(root, firm_path)}
        for oracle in ORACLES:
            values = firm[f'delta_R_score_{oracle}'].to_numpy(dtype=float)
            mean = float(values.mean())
            summary_col = f'mean_delta_R_score_{oracle}'
            if summary_col not in selected_summary.index or not np.isclose(float(selected_summary[summary_col]), mean, atol=1e-12, rtol=0):
                raise RuntimeError(f'seed={seed}, oracle={oracle}: firm mean and final summary disagree')
            seed_item[f'mean_delta_R_score_{oracle}'] = mean
            seed_item[f'effect_rank_spearman_{oracle}'] = np.nan
        seed_rows.append(seed_item)
        input_rows.extend([{'artifact': f'seed_{seed}_policy_summary', 'path': repo_rel(root, summary_path)}, {'artifact': f'seed_{seed}_firm_ledger', 'path': repo_rel(root, firm_path)}])
    pair_rows: list[dict[str, Any]] = []
    for seed_a, seed_b in combinations(EXPECTED_SEEDS, 2):
        a = frames[seed_a]
        b = frames[seed_b]
        merged = a.merge(b, on='row_id', suffixes=('_a', '_b'), validate='one_to_one')
        row: dict[str, Any] = {'record_type': 'SEED_PAIR_STABILITY', 'record_key': f'pair:{seed_a}:{seed_b}', 'seed': np.nan, 'seed_a': seed_a, 'seed_b': seed_b, 'policy': 'Candidate-IQL', 'n_firms': len(merged), 'modal_action': '', 'modal_action_share': np.nan, 'action_identity_rate': float(merged['candidate_id_a'].astype(str).eq(merged['candidate_id_b'].astype(str)).mean()), 'source_policy_summary': '', 'source_firm_ledger': f'{seed_a}|{seed_b}'}
        for oracle in ORACLES:
            row[f'mean_delta_R_score_{oracle}'] = np.nan
            row[f'effect_rank_spearman_{oracle}'] = float(merged[f'delta_R_score_{oracle}_a'].corr(merged[f'delta_R_score_{oracle}_b'], method='spearman'))
        pair_rows.append(row)
    all_rows = seed_rows + pair_rows
    out = root / 'data/analysis/paper_repro/07_claim_sources/rl_seven_seed_summary.csv'
    write_csv(out, all_rows)
    pair_summary = root / 'data/analysis/paper_repro/07_claim_sources/firm_level_reproducibility_summary.csv'
    pair_frame = pd.DataFrame(pair_rows)
    summary_rows = [{'metric': 'action_identity_rate', 'pair_count': len(pair_frame), 'mean': float(pair_frame['action_identity_rate'].mean()), 'min': float(pair_frame['action_identity_rate'].min()), 'max': float(pair_frame['action_identity_rate'].max())}]
    for oracle in ORACLES:
        column = f'effect_rank_spearman_{oracle}'
        summary_rows.append({'metric': column, 'pair_count': len(pair_frame), 'mean': float(pair_frame[column].mean()), 'min': float(pair_frame[column].min()), 'max': float(pair_frame[column].max())})
    write_csv(pair_summary, summary_rows)
    inputs_out = root / 'data/analysis/paper_repro/07_claim_sources/rl_seven_seed_claim_input_files.csv'
    write_csv(inputs_out, input_rows)
    manifest = {'schema_version': 'rl_seven_seed_claim_summary_v2', 'status': 'PASS', 'seed_rows': len(seed_rows), 'pair_rows': len(pair_rows), 'observed_seeds': list(observed), 'outputs': {'claim_summary': {'path': repo_rel(root, out), 'rows': len(all_rows)}, 'pair_summary': {'path': repo_rel(root, pair_summary), 'rows': len(summary_rows)}, 'inputs': {'path': repo_rel(root, inputs_out), 'rows': len(input_rows)}}, 'interpretation_boundary': 'Mean policy value, firm-level effect ranks, and action identity are separate reproducibility resolutions.'}
    write_json(out.with_suffix('.manifest.json'), manifest)
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
