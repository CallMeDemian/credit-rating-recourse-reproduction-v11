from __future__ import annotations
'Run thesis analyses that remain after the canonical post-freeze analysis.\n\nModes\n-----\n``noapi``\n    Audit/repair the 1,000-draw unconditional/industry/rating-band shuffle\n    cells, run the reference-quality response analysis, and seal candidate-\n    library split-track provenance. No LLM API calls are made.\n\n``frontier``\n    Analyze four already-archived IC-b generation-time budget-frontier arms.\n    In ``--plan-only`` mode, validate and report the expected four-arm contract\n    before those archives exist. This module does not call an API; the\n    PowerShell runner creates the arms.\n\nThe module never performs the final end-to-end thesis reproduction.\n'
import argparse
import json
import shutil
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from credit_recourse.analysis.llm_action_budget_ablation import run_ablation
from credit_recourse.analysis.n5_budget_frontier_holm_inference import run as run_frontier_inference
from credit_recourse.analysis.n5m_posthoc import run_analysis as run_n5m_posthoc
from credit_recourse.analysis.paper_output_layout import build_layout, ensure_layout
from credit_recourse.analysis.reference_quality_acceptance import run_analysis as run_reference_quality
from credit_recourse.contracts.score_tie import SCORE_TIE_CONTRACT_VERSION, score_tie_contract_status
from credit_recourse.contracts.paper_reproduction import analysis_output_dir, discover_archived_runs, load_profile, select_budget_frontier_role, select_exact_role
from credit_recourse.verification.verify_candidate_library_provenance import verify as verify_provenance
SCHEMA_VERSION = 'remaining_thesis_analyses_v1'
SHUFFLE_CELLS = (('none', 'row_shuffle_vector_null_native'), ('industry', 'row_shuffle_vector_null_native_industry'), ('rating_band', 'row_shuffle_vector_null_native_rating_band'))

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

def _primary_runs(root: Path, profile: dict[str, Any], *, catalog_snapshot: Path | None=None):
    records = discover_archived_runs(root, profile, catalog_snapshot=catalog_snapshot)
    cfg = profile['llm']['primary']
    return ([select_exact_role(records, run_role=cfg['run_role'], information_condition=ic) for ic in cfg['information_conditions']], records)

def _analysis_manifest_status(layout) -> str | None:
    path = layout.manifest / 'paper_repro_analysis_manifest.json'
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8-sig')).get('status')
    except Exception:
        return 'UNREADABLE'

def _shuffle_semantic_contract(meta: dict[str, Any], *, expected_within: str) -> tuple[bool, str]:
    if str(meta.get('shuffle_within')) != expected_within:
        return (False, f"shuffle_within={meta.get('shuffle_within')!r}")
    try:
        count = int(meta.get('stratum_count', -1))
    except Exception:
        return (False, f"stratum_count={meta.get('stratum_count')!r}")
    if expected_within == 'none':
        return (count == 1, 'PASS' if count == 1 else f'stratum_count={count}, expected=1')
    if expected_within == 'rating_band':
        if count < 2:
            return (False, f'stratum_count={count}, expected>=2')
        return (True, 'PASS')
    try:
        unique_known = int(meta.get('stratum_unique_known', -1))
        known_fraction = float(meta.get('stratum_known_fraction', -1.0))
    except Exception:
        return (False, 'industry stratum profile fields are not numeric')
    if meta.get('stratum_resolution_status') != 'PASS':
        return (False, f"stratum_resolution_status={meta.get('stratum_resolution_status')!r}")
    if count < 2 or unique_known < 2:
        return (False, f'industry strata are not informative: count={count}, unique_known={unique_known}')
    if known_fraction < 0.9:
        return (False, f'industry known_fraction={known_fraction:.6f}, expected>=0.90')
    return (True, 'PASS')

def _shuffle_cell_ok(path: Path, *, expected_within: str, expected_draws: int) -> tuple[bool, str]:
    required = [path / 'metadata.json', path / 'shuffle_per_draw_summary.csv', path / 'shuffle_permutation_ci.csv']
    missing = [x.name for x in required if not x.is_file()]
    if missing:
        checkpoint = path / 'shuffle_permutation_checkpoint.json'
        if expected_within == 'industry' and checkpoint.is_file():
            try:
                json.loads(checkpoint.read_text(encoding='utf-8-sig'))
            except Exception as exc:
                return (False, f'STALE_INDUSTRY_CONTRACT: checkpoint_unreadable={exc!r}')
            return (False, f'RESUMABLE_PARTIAL: missing={missing}')
        return (False, f'missing={missing}')
    try:
        meta = json.loads((path / 'metadata.json').read_text(encoding='utf-8-sig'))
    except Exception as exc:
        return (False, f'metadata_unreadable={exc!r}')
    if meta.get('status') != 'PASS':
        return (False, f"status={meta.get('status')!r}")
    tie_ok, tie_detail = score_tie_contract_status(meta)
    if not tie_ok:
        return (False, f'STALE_SCORE_TIE_CONTRACT: {tie_detail}')
    if int(meta.get('shuffle_draw_count', -1)) != expected_draws:
        return (False, f"shuffle_draw_count={meta.get('shuffle_draw_count')!r}")
    semantic_ok, semantic_detail = _shuffle_semantic_contract(meta, expected_within=expected_within)
    if not semantic_ok:
        prefix = 'STALE_INDUSTRY_CONTRACT: ' if expected_within == 'industry' else ''
        return (False, prefix + semantic_detail)
    try:
        ci = pd.read_csv(path / 'shuffle_permutation_ci.csv')
    except Exception as exc:
        return (False, f'shuffle_ci_unreadable={exc!r}')
    if 'shuffle_stratum_count' not in ci.columns:
        return (False, 'shuffle CI missing shuffle_stratum_count')
    observed = pd.to_numeric(ci['shuffle_stratum_count'], errors='coerce')
    if observed.isna().any():
        return (False, 'shuffle CI contains non-numeric shuffle_stratum_count')
    expected_count = int(meta.get('stratum_count', -1))
    if not observed.astype(int).eq(expected_count).all():
        return (False, f'shuffle CI stratum count disagrees with metadata: expected={expected_count}')
    return (True, 'PASS')

def _archive_invalid_shuffle_cell(*, root: Path, path: Path, information_condition: str, run_label: str, reason: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f'Invalid shuffle cell disappeared before archive: {path}')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive = root / 'data' / 'archive' / 'paper_repro_shuffle_invalidated' / stamp / information_condition / run_label / path.name
    archive.parent.mkdir(parents=True, exist_ok=False)
    shutil.move(str(path), str(archive))
    replacement_contract = SCORE_TIE_CONTRACT_VERSION if reason.startswith('STALE_SCORE_TIE_CONTRACT:') else 'informative_industry_strata_v1' if reason.startswith('STALE_INDUSTRY_CONTRACT:') else 'canonical_shuffle_cell_regeneration'
    _write_json(archive.parent / 'invalidation_record.json', {'schema_version': 'paper_repro_shuffle_invalidation_v2', 'created_utc': _now(), 'reason': reason, 'source_path': str(path), 'archive_path': str(archive), 'replacement_contract': replacement_contract})
    return archive

def _run_noapi(*, root: Path, raw_root: Path, profile: dict[str, Any], layout, repair_missing_shuffle: bool, bootstrap_draws: int, plan_only: bool, catalog_snapshot: Path | None=None) -> dict[str, Any]:
    primary, _records = _primary_runs(root, profile, catalog_snapshot=catalog_snapshot)
    seed_spec = str(profile['analysis']['shuffle_permutation_seeds'])
    if seed_spec != '1:1000':
        raise ValueError(f'Canonical remaining analysis requires shuffle seeds 1:1000, got {seed_spec!r}')
    seeds = range(1, 1001)
    shuffle_audit: list[dict[str, Any]] = []
    for record in primary:
        ic = _normalise_ic(record.information_condition)
        action_table = record.run_dir / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet'
        if not action_table.is_file():
            raise FileNotFoundError(f'Stage7 action table missing: {action_table}')
        for shuffle_within, cell_name in SHUFFLE_CELLS:
            out = layout.ablation / ic / record.run_label / cell_name
            ok, detail = _shuffle_cell_ok(out, expected_within=shuffle_within, expected_draws=1000)
            row = {'run_label': record.run_label, 'information_condition': ic, 'shuffle_within': shuffle_within, 'output_dir': str(out), 'initial_status': 'PASS' if ok else 'MISSING_OR_INVALID', 'initial_detail': detail, 'action': 'NONE', 'final_status': None}
            if not ok:
                if not repair_missing_shuffle:
                    raise RuntimeError(f'Required shuffle cell is missing/invalid and repair is disabled: {row}')
                row['action'] = 'PLANNED_REPAIR' if plan_only else 'REPAIRED'
                if not plan_only:
                    if out.exists() and (not detail.startswith('RESUMABLE_PARTIAL:')):
                        archived = _archive_invalid_shuffle_cell(root=root, path=out, information_condition=ic, run_label=record.run_label, reason=detail)
                        row['invalidated_archive'] = str(archived)
                    run_ablation(project_root=root, raw_root=raw_root, stage7_action_table=action_table, output_dir=out, policies=['C6'], modes=['free_form_10d'], variant='row_shuffle_vector_null', target_budget='native', random_seed=1, shuffle_seeds=seeds, shuffle_within=shuffle_within, reference_policy='C3', write_paired_inference=True)
            if plan_only:
                row['final_status'] = 'PLANNED' if not ok else 'PASS'
            else:
                final_ok, final_detail = _shuffle_cell_ok(out, expected_within=shuffle_within, expected_draws=1000)
                row['final_status'] = 'PASS' if final_ok else 'FAIL'
                row['final_detail'] = final_detail
                if not final_ok:
                    raise RuntimeError(f'Shuffle repair failed: {row}')
            shuffle_audit.append(row)
    reference_out = layout.reference_quality_acceptance
    provenance_dir = layout.visual_registry.parent / 'candidate_library_provenance'
    provenance_out = provenance_dir / 'candidate_library_provenance_report.json'
    if not plan_only:
        run_reference_quality(project_root=root, run_dirs=[x.run_dir for x in primary], output_dir=reference_out, bootstrap_draws=int(bootstrap_draws), bootstrap_seed=20260711)
        provenance = verify_provenance(root, output_dir=provenance_dir)
        if provenance.get('status') != 'PASS':
            raise RuntimeError(f"Candidate-library provenance failed: {provenance.get('errors')}")
    else:
        provenance = {'status': 'PLANNED', 'output': str(provenance_out)}
    return {'status': 'PLANNED' if plan_only else 'PASS', 'analysis_manifest_status_before': _analysis_manifest_status(layout), 'shuffle_audit': shuffle_audit, 'reference_quality_output': str(reference_out), 'candidate_library_provenance': provenance}

def _run_frontier(*, root: Path, profile: dict[str, Any], layout, plan_only: bool, frontier_design: str, catalog_snapshot: Path | None=None) -> dict[str, Any]:
    records = discover_archived_runs(root, profile, catalog_snapshot=catalog_snapshot)
    profile_key = 'n5_matched_budget_frontier' if frontier_design == 'matched_c4_c6' else 'n5_budget_frontier'
    cfg = profile['llm'][profile_key]
    arms = select_budget_frontier_role(records, run_role=cfg['run_role'], information_condition=cfg['information_condition'], expected_budgets=cfg['l1_budgets'], allow_absent=plan_only)
    out = layout.n5_matched_budget_frontier_holm if frontier_design == 'matched_c4_c6' else layout.n5_budget_frontier_holm
    if plan_only:
        expected_budgets = ['unbounded' if value is None else float(value) for value in cfg['l1_budgets']]
        return {'status': 'PLANNED', 'archive_state': 'COMPLETE_ALREADY_ARCHIVED' if arms else 'ABSENT_NOT_YET_RUN', 'frontier_design': frontier_design, 'profile_key': profile_key, 'run_role': str(cfg['run_role']), 'information_condition': str(cfg['information_condition']), 'expected_budgets': expected_budgets, 'expected_conditions': list(cfg['conditions']), 'expected_modes': list(cfg['modes']), 'discovered_archive_count': len(arms), 'run_dirs': [str(x.run_dir) for x in arms], 'output_dir': str(out), 'inference_requires_complete_archives': True}
    result = run_frontier_inference(Namespace(run_dirs=[str(x.run_dir) for x in arms], out_dir=str(out), run_role=str(cfg['run_role']), information_condition=str(cfg['information_condition']), expected_budgets=['unbounded' if x is None else str(x) for x in cfg['l1_budgets']], design=frontier_design, c4_exact_tolerance=1e-09))
    if result.get('status') != 'PASS':
        raise RuntimeError(f'Frontier inference failed: {result}')
    if frontier_design == 'matched_c4_c6':
        posthoc = run_n5m_posthoc(project_root=root, run_dirs=[x.run_dir for x in arms], output_dir=layout.n5m_posthoc, expected_budgets=tuple(cfg['l1_budgets']))
        if posthoc.get('status') != 'PASS':
            raise RuntimeError(f'N5M post-hoc analysis failed: {posthoc}')
        result = dict(result)
        result['n5m_posthoc'] = posthoc
    return result

def run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    if not (root / 'src' / 'credit_recourse').is_dir():
        raise FileNotFoundError(f'Project root lacks src/credit_recourse: {root}')
    analysis_profile_name = str(getattr(args, 'analysis_profile', 'current_comprehensive'))
    profile = load_profile(root, analysis_profile_name)
    catalog_arg = getattr(args, 'eligible_run_catalog', None)
    catalog_snapshot = Path(catalog_arg).resolve() if catalog_arg else None
    analysis_dir = analysis_output_dir(root, profile)
    layout = build_layout(analysis_dir)
    ensure_layout(layout)
    manifest_path = layout.verification / 'remaining_thesis_analyses_manifest.json'
    existing_status = _analysis_manifest_status(layout)
    allow_running_analysis = bool(getattr(args, 'allow_analysis_running', False))
    if allow_running_analysis and args.mode != 'noapi':
        raise ValueError('--allow-analysis-running is restricted to the noapi resume backfill path')
    if args.mode in {'noapi', 'all'} and existing_status == 'RUNNING' and (not args.plan_only) and (not allow_running_analysis):
        raise RuntimeError('Canonical -Task Analysis is still RUNNING. Let that process finish before running RemainingNoApi/RemainingAll.')
    manifest: dict[str, Any] = {'schema_version': SCHEMA_VERSION, 'created_utc': _now(), 'status': 'RUNNING', 'mode': args.mode, 'project_root': str(root), 'analysis_dir': str(analysis_dir), 'analysis_profile': analysis_profile_name, 'eligible_run_catalog': str(catalog_snapshot) if catalog_snapshot is not None else None, 'plan_only': bool(args.plan_only), 'no_llm_api_calls_in_python_orchestrator': True, 'allow_analysis_running': allow_running_analysis, 'results': {}}
    _write_json(manifest_path, manifest)
    try:
        if args.mode in {'noapi', 'all'}:
            manifest['results']['noapi'] = _run_noapi(root=root, raw_root=Path(args.raw_root).resolve(), profile=profile, layout=layout, repair_missing_shuffle=not args.no_repair_missing_shuffle, bootstrap_draws=int(args.bootstrap_draws), plan_only=bool(args.plan_only), catalog_snapshot=catalog_snapshot)
        if args.mode in {'frontier', 'all'}:
            manifest['results']['frontier'] = _run_frontier(root=root, profile=profile, layout=layout, plan_only=bool(args.plan_only), frontier_design=str(args.frontier_design), catalog_snapshot=catalog_snapshot)
        manifest['status'] = 'PLANNED' if args.plan_only else 'PASS'
    except Exception as exc:
        manifest['status'] = 'FAIL'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['completed_utc'] = _now()
        _write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--raw-root', required=True)
    parser.add_argument('--mode', choices=('noapi', 'frontier', 'all'), required=True)
    parser.add_argument('--analysis-profile', choices=('current_comprehensive', 'historical_20260715'), default='current_comprehensive')
    parser.add_argument('--eligible-run-catalog', default=None)
    parser.add_argument('--frontier-design', choices=('legacy_c6_only', 'matched_c4_c6'), default='legacy_c6_only', help='Select the legacy C6-only budget frontier or matched C4/C6 factorial frontier.')
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--no-repair-missing-shuffle', action='store_true')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--allow-analysis-running', action='store_true', help='Internal fail-safe used only by AnalysisResume to backfill no-API additions before final assets.')
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    run(args)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
