from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .claim_evidence_common import repo_rel, write_csv, write_json
ORACLES = ('alpha', 'beta', 'gamma')
CANONICAL_INPUTS = ('data/final_freeze/stage6_multi_oracle_eval/multi_oracle_policy_eval.parquet', 'data/final_freeze/stage6_candidate_selector_eval/multi_oracle_policy_eval.parquet')

def _resolve_input(root: Path) -> Path:
    found = [root / rel for rel in CANONICAL_INPUTS if (root / rel).is_file()]
    if not found:
        raise FileNotFoundError(f'Static evaluator-resolution requires a canonical Stage6 multi_oracle_policy_eval.parquet; checked={list(CANONICAL_INPUTS)}')
    return found[0]

def _resolve_policy_column(frame: pd.DataFrame) -> str:
    for name in ('candidate_id', 'policy', 'action_id'):
        if name in frame.columns:
            return name
    raise RuntimeError('Stage6 ledger lacks candidate_id/policy/action_id')

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    source = _resolve_input(root)
    frame = pd.read_parquet(source)
    required = {'row_id', 'policy', *(f'delta_R_score_{o}' for o in ORACLES)}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f'Stage6 static-resolution input missing columns: {missing}')
    if frame.duplicated(['row_id', 'policy']).any():
        sample = frame.loc[frame.duplicated(['row_id', 'policy'], keep=False), ['row_id', 'policy']].head(10)
        raise RuntimeError(f"duplicate row_id/policy in Stage6 ledger: {sample.to_dict('records')}")
    firm_count = int(frame['row_id'].nunique())
    if firm_count != 575:
        raise RuntimeError(f'expected 575 Stage6 evaluation firms, got {firm_count}')
    action_col = _resolve_policy_column(frame)
    rows: list[dict[str, Any]] = []
    best_action_by_oracle: dict[str, pd.Series] = {}
    unique_argmax_by_oracle: dict[str, pd.Series] = {}
    for oracle in ORACLES:
        score_col = f'delta_R_score_{oracle}'
        work = frame[['row_id', 'policy', action_col, score_col]].copy()
        work[score_col] = pd.to_numeric(work[score_col], errors='coerce')
        if not np.isfinite(work[score_col].to_numpy()).all():
            raise RuntimeError(f'non-finite values in {score_col}')
        pivot = work.pivot(index='row_id', columns='policy', values=score_col)
        if pivot.isna().any().any():
            raise RuntimeError(f'Stage6 policy grid is incomplete for oracle={oracle}')
        maxima = pivot.max(axis=1)
        max_tie_count = pivot.eq(maxima, axis=0).sum(axis=1)
        tie_rate = float(max_tie_count.gt(1).mean())
        per_firm_sd = pivot.std(axis=1, ddof=0)
        mean_by_policy = pivot.mean(axis=0)
        preferred_policy = str(mean_by_policy.idxmax())
        best_action = pivot.idxmax(axis=1).astype(str)
        best_action_by_oracle[oracle] = best_action
        unique_argmax_by_oracle[oracle] = max_tie_count.eq(1)
        rows.append({'oracle_backend': oracle, 'n': firm_count, 'policy_count': int(pivot.shape[1]), 'action_effect_sd': float(per_firm_sd.mean()), 'action_effect_sd_across_all_rows': float(work[score_col].std(ddof=1)), 'tie_rate': tie_rate, 'preferred_action': preferred_policy, 'best_action_agreement': np.nan, 'all_three_best_action_agreement_rate': np.nan, 'source_path': repo_rel(root, source), 'tie_rate_definition': 'firms with >=2 policies tied at the maximum Oracle delta / 575', 'action_effect_sd_definition': 'mean within-firm population SD across available Stage6 policies'})
    all_three = unique_argmax_by_oracle['alpha'] & unique_argmax_by_oracle['beta'] & unique_argmax_by_oracle['gamma'] & best_action_by_oracle['alpha'].eq(best_action_by_oracle['beta']) & best_action_by_oracle['alpha'].eq(best_action_by_oracle['gamma'])
    all_three_rate = float(all_three.mean())
    for row in rows:
        oracle = str(row['oracle_backend'])
        other = [name for name in ORACLES if name != oracle]
        agreement = [float(best_action_by_oracle[oracle].eq(best_action_by_oracle[name]).mean()) for name in other]
        row['best_action_agreement'] = float(np.mean(agreement))
        row['all_three_best_action_agreement_rate'] = all_three_rate
    out = root / 'data/analysis/paper_repro/07_claim_sources/static_evaluator_resolution.csv'
    write_csv(out, rows)
    manifest = {'schema_version': 'static_evaluator_resolution_v3', 'status': 'PASS', 'input': {'path': repo_rel(root, source)}, 'firm_count': firm_count, 'oracle_count': len(rows), 'all_three_best_action_agreement_rate': all_three_rate, 'all_three_best_action_agreement_definition': 'share of 575 firms for which Oracle alpha, beta, and gamma select the same unique argmax policy after deterministic column ordering', 'output': {'path': repo_rel(root, out), 'rows': len(rows)}, 'interpretation_boundary': 'Static resolution describes how frozen Oracle mappings distinguish the same Stage6 policy grid; it is not a causal policy-effect model.'}
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
