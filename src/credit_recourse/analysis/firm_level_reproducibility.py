from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .claim_evidence_common import repo_rel, write_json

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    source = root / 'data/analysis/paper_repro/07_claim_sources/rl_seven_seed_summary.csv'
    if not source.is_file():
        raise FileNotFoundError(f'Seven-seed claim summary missing: {source}. Run rl_seven_seed_claim_summary first.')
    frame = pd.read_csv(source, encoding='utf-8-sig')
    required = {'record_type', 'seed', 'seed_a', 'seed_b', 'n_firms', 'action_identity_rate', 'effect_rank_spearman_alpha', 'effect_rank_spearman_beta', 'effect_rank_spearman_gamma'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'firm-level reproducibility source missing columns: {missing}')
    seed_rows = frame.loc[frame['record_type'].astype(str).eq('SEED_POLICY_MEAN')]
    pair_rows = frame.loc[frame['record_type'].astype(str).eq('SEED_PAIR_STABILITY')]
    seeds = sorted(pd.to_numeric(seed_rows['seed'], errors='raise').astype(int).tolist())
    if seeds != list(range(1, 8)) or len(seed_rows) != 7:
        raise RuntimeError(f'expected one seed row for seeds 1..7, got {seeds}')
    if len(pair_rows) != 21 or pair_rows.duplicated(['seed_a', 'seed_b']).any():
        raise RuntimeError('expected 21 unique unordered seed pairs')
    if not pd.to_numeric(pair_rows['n_firms'], errors='raise').eq(575).all():
        raise RuntimeError('all seed-pair comparisons must contain 575 firms')
    numeric_columns = ['action_identity_rate', *(f'effect_rank_spearman_{o}' for o in ('alpha', 'beta', 'gamma'))]
    numeric = pair_rows[numeric_columns].apply(pd.to_numeric, errors='coerce')
    if not np.isfinite(numeric.to_numpy()).all():
        raise RuntimeError('firm-level reproducibility metrics contain non-finite values')
    if not numeric['action_identity_rate'].between(0.0, 1.0).all():
        raise RuntimeError('action identity rate outside [0,1]')
    for column in numeric_columns[1:]:
        if not numeric[column].between(-1.0, 1.0).all():
            raise RuntimeError(f'rank correlation outside [-1,1]: {column}')
    result = {'schema_version': 'firm_level_reproducibility_contract_v2', 'status': 'PASS', 'source': {'path': repo_rel(root, source)}, 'seed_count': len(seed_rows), 'pair_count': len(pair_rows), 'mean_action_identity_rate': float(numeric['action_identity_rate'].mean()), 'mean_effect_rank_spearman': {oracle: float(numeric[f'effect_rank_spearman_{oracle}'].mean()) for oracle in ('alpha', 'beta', 'gamma')}, 'interpretation_boundary': 'Aggregate policy-value stability does not imply firm-level action or rank stability.'}
    out = root / 'data/analysis/paper_repro/07_claim_sources/firm_level_reproducibility_manifest.json'
    write_json(out, result)
    return result

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
