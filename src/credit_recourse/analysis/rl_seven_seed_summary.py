from __future__ import annotations
'Build the thesis seven-seed Candidate-IQL reproducibility ledger.\n\nThe upstream PowerShell runner creates seven aligned Stage3/4/5/6 cells and\narchives every cell independently.  This module is intentionally downstream:\nit never retrains or edits a checkpoint.  It verifies the cell grid, resolves\nthe deployed Candidate-IQL row from each archived Stage6 summary, and writes a\nfirm-independent seed summary used by Appendix D / Table 44.\n'
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
EXPECTED_SEEDS = tuple(range(1, 8))
ORACLES = ('alpha', 'beta', 'gamma')

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding='utf-8-sig')

def _candidate_row(df: pd.DataFrame) -> pd.Series:
    if 'policy' not in df.columns:
        raise ValueError('final_policy_summary.csv lacks policy')
    work = df.copy()
    work['policy'] = work['policy'].astype(str)
    if 'headline_policy' in work.columns:
        flag = work['headline_policy'].astype(str).str.lower().isin({'true', '1'})
        hit = work.loc[flag]
        if len(hit) == 1:
            return hit.iloc[0]
        if len(hit) > 1:
            raise ValueError(f"multiple headline policies: {hit['policy'].tolist()}")
    preferred = ['C3_candidate_iql_actor', 'C3_candidate_iql', 'C3_candidate_iql_q_argmax']
    for name in preferred:
        hit = work.loc[work['policy'].eq(name)]
        if len(hit) == 1:
            return hit.iloc[0]
    hit = work.loc[work['policy'].str.contains('candidate_iql', case=False, regex=False)]
    hit = hit.loc[~hit['policy'].str.contains('rerank', case=False, regex=False)]
    if len(hit) != 1:
        raise ValueError(f"cannot uniquely resolve Candidate-IQL row: {hit['policy'].tolist()}")
    return hit.iloc[0]

def _resolve_archive(run_root: Path, archive_text: str) -> Path:
    p = Path(str(archive_text))
    if p.is_dir():
        return p.resolve()
    hits = [x for x in run_root.rglob(p.name) if x.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f'archive_root is unavailable and leaf fallback is not unique: {archive_text!r}, hits={hits}')
    return hits[0].resolve()

def build(run_root: Path, out_dir: Path, expected_firm_count: int=575) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    unified_path = run_root / 'UNIFIED_RUN_SUMMARY.csv'
    grid = _read_csv(unified_path)
    required = {'status', 'archive_root', 'stage3_seed', 'stage4_seed', 'stage5_seed', 'seed_lineage_mode'}
    missing = sorted(required - set(grid.columns))
    if missing:
        raise ValueError(f'UNIFIED_RUN_SUMMARY missing columns: {missing}')
    completed = grid.loc[grid['status'].astype(str).str.startswith('COMPLETED')].copy()
    if len(completed) != 7:
        raise ValueError(f'seven-seed run must contain exactly 7 completed cells; observed={len(completed)}')
    for col in ('stage3_seed', 'stage4_seed', 'stage5_seed'):
        completed[col] = pd.to_numeric(completed[col], errors='raise').astype(int)
    for _, r in completed.iterrows():
        seeds = {int(r['stage3_seed']), int(r['stage4_seed']), int(r['stage5_seed'])}
        if len(seeds) != 1:
            raise ValueError(f"stage seeds are not aligned in cell {r.get('cell_id')}: {seeds}")
        if str(r['seed_lineage_mode']) != 'aligned':
            raise ValueError(f"seed_lineage_mode must be aligned: {r.get('cell_id')}")
    observed_seeds = tuple(sorted(completed['stage3_seed'].tolist()))
    if observed_seeds != EXPECTED_SEEDS:
        raise ValueError(f'expected seeds {EXPECTED_SEEDS}, observed {observed_seeds}')
    per_seed: list[dict[str, Any]] = []
    input_rows: list[dict[str, Any]] = [{'artifact': 'UNIFIED_RUN_SUMMARY', 'path': str(unified_path)}]
    for _, r in completed.sort_values('stage3_seed').iterrows():
        archive = _resolve_archive(run_root, str(r['archive_root']))
        summary_path = archive / '00_POLICY_SUMMARY_CORE.csv'
        if not summary_path.is_file():
            summary_path = archive / 'outputs/stage6_multi_oracle_eval/final_policy_summary.csv'
        summary = _read_csv(summary_path)
        row = _candidate_row(summary)
        n = int(float(row.get('n', expected_firm_count)))
        if n != int(expected_firm_count):
            raise ValueError(f"seed={r['stage3_seed']} Candidate-IQL n={n}, expected {expected_firm_count}")
        item: dict[str, Any] = {'seed': int(r['stage3_seed']), 'cell_id': str(r.get('cell_id', '')), 'policy': str(row['policy']), 'n_firms': n, 'archive_root': str(archive)}
        for oracle in ORACLES:
            col = f'mean_delta_R_score_{oracle}'
            if col not in row.index:
                raise ValueError(f'{summary_path} missing {col}')
            item[f'mean_delta_R_score_{oracle}'] = float(row[col])
        per_seed.append(item)
        input_rows.append({'artifact': f"seed_{item['seed']}_policy_summary", 'path': str(summary_path)})
    per_seed_df = pd.DataFrame(per_seed).sort_values('seed')
    summary_rows: list[dict[str, Any]] = []
    for oracle in ORACLES:
        vals = per_seed_df[f'mean_delta_R_score_{oracle}'].to_numpy(dtype=float)
        summary_rows.append({'oracle_backend': oracle, 'n_seeds': int(len(vals)), 'mean_delta_noop': float(np.mean(vals)), 'sd_delta_noop': float(np.std(vals, ddof=1)), 'min_delta_noop': float(np.min(vals)), 'max_delta_noop': float(np.max(vals))})
    summary_df = pd.DataFrame(summary_rows)
    per_seed_path = out_dir / 'rl_seven_seed_policy_means.csv'
    summary_path = out_dir / 'rl_seven_seed_summary.csv'
    inputs_path = out_dir / 'rl_seven_seed_input_files.csv'
    per_seed_df.to_csv(per_seed_path, index=False, encoding='utf-8-sig')
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    pd.DataFrame(input_rows).to_csv(inputs_path, index=False, encoding='utf-8-sig')
    manifest = {'schema_version': 'rl_seven_seed_summary_v1', 'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'run_root': str(run_root), 'expected_seeds': list(EXPECTED_SEEDS), 'observed_seeds': list(observed_seeds), 'expected_firm_count': int(expected_firm_count), 'policy_resolution': 'headline_policy_then_exact_candidate_iql', 'outputs': {'per_seed': {'path': per_seed_path.name, 'rows': len(per_seed_df)}, 'summary': {'path': summary_path.name, 'rows': len(summary_df)}, 'inputs': {'path': inputs_path.name, 'rows': len(input_rows)}}, 'interpretation_boundary': 'Fresh aligned seven-seed reproducibility of the final RL configuration. Historical development milestones and discarded sweep cells are provenance, not part of this fresh mean/SD estimate.'}
    manifest_path = out_dir / 'rl_seven_seed_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--expected-firm-count', type=int, default=575)
    args = ap.parse_args(argv)
    manifest = build(Path(args.run_root), Path(args.out), args.expected_firm_count)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
