from __future__ import annotations
'N1 action-budget frontier grid wrapper for existing LLM ablation scorer.\n\nThis module is a thin runner around\n``credit_recourse.analysis.llm_action_budget_ablation.run_ablation``.  It adds\nonly grid orchestration, run-directory Stage7 table discovery, manifest writing,\nand an explicit dry-run plan mode.  It does not introduce a new transformation:\neach grid cell delegates to the existing evaluator-only ablation implementation.\n'
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import pandas as pd
from credit_recourse.analysis.llm_action_budget_ablation import SUPPORTED_VARIANTS, run_ablation
from credit_recourse.contracts.score_tie import score_tie_contract_metadata
from credit_recourse.rl.common.io import write_json
FRONTIER_SCHEMA_VERSION = 'action_budget_frontier_grid_v1'
DEFAULT_GRID = (0.5, 0.75, 1.0, 1.27, 1.75, 2.5, 3.25, 4.0)
DEFAULT_VARIANTS = ('l1_rescale', 'global_mean_vector_null', 'row_shuffle_vector_null')
STAGE7_ACTION_TABLE_RELATIVE_CANDIDATES = ('stage7_llm_action_generation/llm_stage7_action_table.parquet', 'stage7_llm_action_generation/llm_stage7_action_table.csv', 'stage7_action_generation/llm_stage7_action_table.parquet', 'stage7_action_generation/llm_stage7_action_table.csv')

@dataclass(frozen=True)
class FrontierCell:
    run_dir: Path
    run_label: str
    stage7_action_table: Path
    budget: float
    variant: str
    output_dir: Path

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        return [x.strip() for x in value.split(',') if x.strip()]
    return [str(x).strip() for x in value if str(x).strip()]

def _budget_token(value: float) -> str:
    return f'{float(value):g}'.replace('-', 'm').replace('.', 'p')

def _safe_token(value: str) -> str:
    keep = []
    for ch in str(value):
        if ch.isalnum() or ch in ('-', '_', '.'):
            keep.append(ch)
        else:
            keep.append('_')
    out = ''.join(keep).strip('._')
    return out or 'run'

def find_stage7_action_table(run_dir_or_file: Path) -> Path:
    p = Path(run_dir_or_file).resolve()
    if p.is_file():
        if p.name not in {'llm_stage7_action_table.parquet', 'llm_stage7_action_table.csv'}:
            raise ValueError(f'Explicit Stage7 action table has unexpected file name: {p}')
        return p
    if not p.exists():
        raise FileNotFoundError(f'Run directory does not exist: {p}')
    for rel in STAGE7_ACTION_TABLE_RELATIVE_CANDIDATES:
        cand = p / rel
        if cand.exists() and cand.is_file():
            return cand
    found = sorted(list(p.glob('**/llm_stage7_action_table.parquet')) + list(p.glob('**/llm_stage7_action_table.csv')))
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError({'message': 'Multiple Stage7 action tables found; pass the intended file explicitly.', 'run_dir': str(p), 'candidates': [str(x) for x in found]})
    raise FileNotFoundError({'message': 'No Stage7 action table found under run directory.', 'run_dir': str(p), 'expected_relative_paths': list(STAGE7_ACTION_TABLE_RELATIVE_CANDIDATES)})

def build_frontier_plan(*, runs: Iterable[Path], out_dir: Path, grid: Iterable[float], variants: Iterable[str]) -> list[FrontierCell]:
    out_dir = Path(out_dir).resolve()
    variants_list = _parse_csv(list(variants))
    bad = sorted(set(variants_list) - set(SUPPORTED_VARIANTS))
    if bad:
        raise ValueError(f'Unsupported ablation variant(s): {bad}; supported={sorted(SUPPORTED_VARIANTS)}')
    budgets = [float(x) for x in grid]
    if not budgets or any((not pd.notna(x) or x <= 0 for x in budgets)):
        raise ValueError(f'Grid budgets must be positive finite values; got {budgets}')
    cells: list[FrontierCell] = []
    for run in runs:
        run_path = Path(run).resolve()
        table = find_stage7_action_table(run_path)
        run_label = _safe_token(run_path.stem if run_path.is_file() else run_path.name)
        for budget in budgets:
            for variant in variants_list:
                cell_dir = out_dir / run_label / f'b{_budget_token(budget)}' / _safe_token(variant)
                cells.append(FrontierCell(run_dir=run_path if run_path.is_dir() else table.parent, run_label=run_label, stage7_action_table=table, budget=float(budget), variant=variant, output_dir=cell_dir))
    if not cells:
        raise ValueError('Frontier plan has zero cells; check --runs, --grid, and --variants.')
    return cells

def _cell_to_record(cell: FrontierCell) -> dict:
    return {'run_label': cell.run_label, 'run_dir': str(cell.run_dir), 'stage7_action_table': str(cell.stage7_action_table), 'budget': float(cell.budget), 'variant': cell.variant, 'target_budget': f'fixed:{float(cell.budget):g}', 'output_dir': str(cell.output_dir)}

def run_frontier_grid(*, project_root: Path, raw_root: Path, runs: Iterable[Path], out_dir: Path, grid: Iterable[float]=DEFAULT_GRID, variants: Iterable[str]=DEFAULT_VARIANTS, policies: str='C6', modes: str='free_form_10d', active_eps: float=1e-09, random_seed: int=1, reference_policy: str='C3', dry_run: bool=False) -> dict:
    project_root = Path(project_root).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = build_frontier_plan(runs=list(runs), out_dir=out_dir, grid=grid, variants=variants)
    plan_df = pd.DataFrame([_cell_to_record(c) for c in plan])
    plan_df.to_csv(out_dir / 'frontier_grid_plan.csv', index=False, encoding='utf-8-sig')
    raw_frames: list[pd.DataFrame] = []
    status_rows: list[dict] = []
    if not dry_run:
        for cell in plan:
            cell.output_dir.mkdir(parents=True, exist_ok=True)
            meta = run_ablation(project_root=project_root, raw_root=raw_root, stage7_action_table=cell.stage7_action_table, output_dir=cell.output_dir, policies=_parse_csv(policies), modes=_parse_csv(modes), variant=cell.variant, target_budget=f'fixed:{float(cell.budget):g}', active_eps=active_eps, random_seed=random_seed, reference_policy=reference_policy, write_paired_inference=True)
            summary_path = cell.output_dir / 'ablation_policy_summary.csv'
            if not summary_path.exists():
                raise FileNotFoundError(f'Expected ablation summary not written: {summary_path}')
            summary = pd.read_csv(summary_path)
            summary['frontier_run_label'] = cell.run_label
            summary['frontier_budget'] = float(cell.budget)
            summary['frontier_variant'] = cell.variant
            summary['frontier_output_dir'] = str(cell.output_dir)
            raw_frames.append(summary)
            status_rows.append({**_cell_to_record(cell), 'status': meta.get('status', 'PASS')})
        raw = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
        raw.to_csv(out_dir / 'frontier_grid_raw.csv', index=False, encoding='utf-8-sig')
    else:
        status_rows = [{**_cell_to_record(c), 'status': 'DRY_RUN_NOT_EXECUTED'} for c in plan]
    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(out_dir / 'frontier_grid_status.csv', index=False, encoding='utf-8-sig')
    meta = {'schema_version': FRONTIER_SCHEMA_VERSION, 'created_utc': _now(), 'status': 'PASS', 'dry_run': bool(dry_run), 'project_root': str(project_root), 'n_cells': int(len(plan)), 'grid': [float(x) for x in grid], 'variants': _parse_csv(list(variants)), 'policies': _parse_csv(policies), 'modes': _parse_csv(modes), 'reference_policy': reference_policy, 'random_seed': int(random_seed), 'score_tie_contract': score_tie_contract_metadata(), 'outputs': {'frontier_grid_plan': 'frontier_grid_plan.csv', 'frontier_grid_status': 'frontier_grid_status.csv', 'frontier_grid_raw': None if dry_run else 'frontier_grid_raw.csv'}, 'note': 'Each non-dry-run cell delegates to credit_recourse.analysis.llm_action_budget_ablation; no new action transformation is introduced here.'}
    write_json(out_dir / 'metadata.json', meta)
    return meta

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='N1 full-grid budget frontier wrapper over llm_action_budget_ablation')
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--raw-root', required=True)
    ap.add_argument('--runs', nargs='+', required=True, help='Run dirs or explicit llm_stage7_action_table files')
    ap.add_argument('--grid', nargs='+', type=float, default=list(DEFAULT_GRID))
    ap.add_argument('--variants', default=','.join(DEFAULT_VARIANTS), help='Comma-separated ablation variants')
    ap.add_argument('--policies', default='C6')
    ap.add_argument('--modes', default='free_form_10d')
    ap.add_argument('--active-eps', type=float, default=1e-09)
    ap.add_argument('--random-seed', type=int, default=1)
    ap.add_argument('--reference-policy', default='C3')
    ap.add_argument('--out-dir', '--out', dest='out_dir', required=True)
    ap.add_argument('--dry-run', action='store_true', help='Write the grid plan without executing ablation scoring')
    args = ap.parse_args(argv)
    meta = run_frontier_grid(project_root=Path(args.project_root), raw_root=Path(args.raw_root), runs=[Path(p) for p in args.runs], out_dir=Path(args.out_dir), grid=args.grid, variants=_parse_csv(args.variants), policies=args.policies, modes=args.modes, active_eps=args.active_eps, random_seed=args.random_seed, reference_policy=args.reference_policy, dry_run=args.dry_run)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
