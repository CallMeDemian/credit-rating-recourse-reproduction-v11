from __future__ import annotations
'Canonical post-freeze paper analysis for the thesis reproduction workspace.\n\nThis module is the *single* analysis orchestrator after Oracle/RL/LLM runs have\nbeen frozen. It is intentionally read-only with respect to ``data/final_freeze``\nand writes every paper-facing post-freeze result below one directory:\n\n    data/analysis/paper_repro/\n\nIt never calls an LLM API, retrains a model, or mutates source stage outputs.\nThe active runner performs a lightweight required-input existence check and then\nexecutes the analyses. Artifact-version and historical compatibility gates are\nnot part of the active analysis path.\n'
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
import pandas as pd
from credit_recourse.analysis.n5_7_10c_holm_inference import find_stage6_summary, load_stage6_canonical_means
from credit_recourse.analysis.reference_reproduction_comparison import resolve_mode as resolve_rl_validation_mode
from credit_recourse.analysis.paper_output_layout import PaperOutputLayout, build_layout, ensure_layout
from credit_recourse.contracts.score_tie import SCORE_TIE_CONTRACT_VERSION, score_tie_contract_status
from credit_recourse.contracts.paper_reproduction import ArchivedRunMetadata, ProfileError, discover_archived_runs, eligible_catalog_contract, load_profile, resolve_workspace_path, select_budget_frontier_role, select_exact_role
from credit_recourse.contracts.provenance import portable_path_identity
from credit_recourse.oracle.stage1.stage00_04_variable_selection.kw_contract import KW_CONSTANT_CONTRACT_VERSION
SCHEMA_VERSION = 'paper_repro_analysis_v2'

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _eligibility_manifest_contract(project_root: Path, catalog_snapshot: Path | None) -> dict[str, Any]:
    if catalog_snapshot is None:
        return {'eligibility_contract_id': 'discover_current_llm_runs_v1', 'eligible_catalog_path': None, 'eligible_catalog_role_relative_path': None, 'eligible_catalog_row_count': None, 'eligible_catalog_unique_count': None}
    roots: dict[str, Path] = {'data_ref': project_root / 'data' / 'ref'}
    reference_root = os.environ.get('CREDIT_RECOURSE_REFERENCE_ROOT')
    if reference_root:
        roots['reference_snapshot'] = Path(reference_root)
    identity = portable_path_identity(catalog_snapshot, roots=roots)
    role_relative_path = f"{identity['root_role']}:{identity['relative_path']}"
    contract = eligible_catalog_contract(catalog_snapshot, role_relative_path=role_relative_path)
    contract.pop('eligible_run_labels', None)
    return contract

def _command_text(cmd: Sequence[str]) -> str:
    return ' '.join((str(x) for x in cmd))

def _tail_text(value: str, *, limit: int=12000) -> str:
    text = str(value or '')
    return text if len(text) <= limit else text[-limit:]

def _child_environment(project_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    contract_keys = ('CREDIT_RECOURSE_PYTHON_SOURCE_ROOT', 'CREDIT_RECOURSE_PYTHON_SITE_PACKAGES_ROOT', 'CREDIT_RECOURSE_PYTHON_ENVIRONMENT_CONFIG', 'CREDIT_RECOURSE_PYTHON_ENVIRONMENT_CONTRACT_ID', 'CREDIT_RECOURSE_EXECUTION_RUN_ROOT', 'CREDIT_RECOURSE_TEMPORARY_ROOT')
    supplied = {key: str(env.get(key, '')).strip() for key in contract_keys}
    if any(supplied.values()):
        missing = [key for key, value in supplied.items() if not value]
        if missing:
            raise RuntimeError('incomplete Frozen Python child environment: ' + ', '.join(missing))
        source_root = Path(supplied['CREDIT_RECOURSE_PYTHON_SOURCE_ROOT']).resolve()
        site_packages_root = Path(supplied['CREDIT_RECOURSE_PYTHON_SITE_PACKAGES_ROOT']).resolve()
        environment_config = Path(supplied['CREDIT_RECOURSE_PYTHON_ENVIRONMENT_CONFIG']).resolve()
        execution_run_root = Path(supplied['CREDIT_RECOURSE_EXECUTION_RUN_ROOT']).resolve()
        temporary_root = Path(supplied['CREDIT_RECOURSE_TEMPORARY_ROOT']).resolve()
        for label, path, kind in (('Python source root', source_root, 'directory'), ('Python site-packages root', site_packages_root, 'directory'), ('Python environment config', environment_config, 'file'), ('execution RunRoot', execution_run_root, 'directory'), ('temporary root', temporary_root, 'directory')):
            exists = path.is_file() if kind == 'file' else path.is_dir()
            if not exists:
                raise RuntimeError(f'Frozen child {label} is missing: {path}')
        try:
            temporary_root.relative_to(execution_run_root)
        except ValueError as exc:
            raise RuntimeError(f'Frozen child temporary root escapes RunRoot: {temporary_root}') from exc
        env['PYTHONPATH'] = os.pathsep.join((str(source_root), str(site_packages_root)))
        env['TEMP'] = str(temporary_root)
        env['TMP'] = str(temporary_root)
        env['MPLCONFIGDIR'] = str(temporary_root / 'matplotlib')
        env['NUMBA_CACHE_DIR'] = str(temporary_root / 'numba')
        env['XDG_CACHE_HOME'] = str(temporary_root / 'xdg')
        env['TORCH_HOME'] = str(temporary_root / 'torch')
        env['HF_HOME'] = str(temporary_root / 'huggingface')
        env['JOBLIB_TEMP_FOLDER'] = str(temporary_root / 'joblib')
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        env['PYTHONUTF8'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        return env
    source_root = (project_root / 'src').resolve()
    ambient = [item for item in str(env.get('PYTHONPATH', '')).split(os.pathsep) if item and Path(item).resolve() != source_root]
    env['PYTHONPATH'] = os.pathsep.join((str(source_root), *ambient))
    env.setdefault('PYTHONUTF8', '1')
    env.setdefault('PYTHONIOENCODING', 'utf-8')
    env.setdefault('PYTHONDONTWRITEBYTECODE', '1')
    return env

def _run(cmd: list[str], *, project_root: Path, label: str, steps: list[dict[str, Any]], plan_only: bool, capture_child_output: bool=False) -> None:
    record: dict[str, Any] = {'label': label, 'command': cmd, 'command_text': _command_text(cmd), 'status': 'PLANNED' if plan_only else 'RUNNING'}
    steps.append(record)
    print(f'\n==== {label} ====')
    print('CMD>', record['command_text'])
    if plan_only:
        return
    env = _child_environment(project_root)
    if capture_child_output:
        proc = subprocess.run(cmd, cwd=project_root, env=env, check=False, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if proc.stdout:
            print(proc.stdout, end='' if proc.stdout.endswith('\n') else '\n')
        if proc.stderr:
            print(proc.stderr, file=sys.stderr, end='' if proc.stderr.endswith('\n') else '\n')
        record['stdout_tail'] = _tail_text(proc.stdout)
        record['stderr_tail'] = _tail_text(proc.stderr)
    else:
        proc = subprocess.run(cmd, cwd=project_root, env=env, check=False)
    record['exit_code'] = int(proc.returncode)
    record['status'] = 'PASS' if proc.returncode == 0 else 'FAIL'
    if proc.returncode:
        detail = ''
        if capture_child_output and record.get('stderr_tail'):
            last_lines = record['stderr_tail'].strip().splitlines()[-12:]
            detail = '\nChild stderr tail:\n' + '\n'.join(last_lines)
        raise RuntimeError(f'FAILED: {label} exit={proc.returncode}{detail}')

def _record_optional_skip(*, label: str, steps: list[dict[str, Any]], reason: str) -> None:
    record = {'label': label, 'command': [], 'command_text': '', 'status': 'SKIPPED_OPTIONAL_NOT_AVAILABLE', 'reason': reason}
    steps.append(record)
    print(f'\n==== {label} ====')
    print(f'SKIP> {reason}')

def _find_stage8(run_dir: Path) -> Path:
    candidates = (run_dir / 'stage8_llm_multi_oracle_eval', run_dir / 'stage8_multi_oracle_eval')
    hits = [p for p in candidates if (p / 'metadata.json').exists()]
    if len(hits) != 1:
        raise FileNotFoundError(f'Expected exactly one Stage8 directory in {run_dir}; found {[str(p) for p in hits]}')
    return hits[0]

def _normalise_ic(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.lower().replace('_', '-')
    if text in {'ic-a', 'ica'}:
        return 'IC-a'
    if text in {'ic-b', 'icb'}:
        return 'IC-b'
    if text in {'ic-c', 'icc'}:
        return 'IC-c'
    return value

def _resolve_probe_selection(*, project_root: Path, profile: dict[str, Any], records: list[ArchivedRunMetadata], primary: list[ArchivedRunMetadata]) -> dict[str, Any]:
    """Select one probe source using directory/file presence only.

    The active analysis path selects a source from its registered role and required
    file presence. The probe analysis module remains responsible for validating the
    actual tabular content when it imports the selected source.
    """
    required_names = ('icc_probe_summary.json', 'icc_probe_firm_level.csv', 'llm_stage7_icc_probe_checkpoint.jsonl')
    expected_role = str(profile['llm']['icc_probe']['run_role'])
    canonical: list[tuple[Path, ArchivedRunMetadata | None, str]] = []
    for record in records:
        if record.run_role != expected_role:
            continue
        probe_dir = (record.run_dir / 'stage7_icc_probe').resolve()
        if all(((probe_dir / name).is_file() for name in required_names)):
            canonical.append((probe_dir, record, 'canonical_llm_archive'))
    candidates = canonical
    if not candidates:
        legacy_roots: list[tuple[str, Path]] = [('legacy_final_freeze_probe_root', project_root / 'data' / 'final_freeze' / 'icc_probe_runs')]
        archive_root = project_root / 'data' / 'archive'
        if archive_root.is_dir():
            legacy_roots.extend((('legacy_archived_probe_root', path) for path in sorted(archive_root.glob('icc_probe_runs_legacy_*')) if path.is_dir()))
        seen: set[Path] = set()
        for source_kind, root in legacy_roots:
            if not root.is_dir():
                continue
            for summary_path in root.rglob('icc_probe_summary.json'):
                probe_dir = summary_path.parent.resolve()
                if probe_dir in seen:
                    continue
                if all(((probe_dir / name).is_file() for name in required_names)):
                    candidates.append((probe_dir, None, source_kind))
                    seen.add(probe_dir)
    if not candidates:
        raise FileNotFoundError('Required IC-c probe files were not found under the canonical LLM run root or supported legacy probe roots.')
    if len(candidates) != 1:
        raise RuntimeError(f'Expected exactly one IC-c probe directory containing the three required files; found {[str(item[0]) for item in candidates]}')
    source_dir, archive_record, source_kind = candidates[0]
    summary_path = source_dir / 'icc_probe_summary.json'
    try:
        summary = json.loads(summary_path.read_text(encoding='utf-8-sig'))
        if not isinstance(summary, dict):
            summary = {}
    except Exception:
        summary = {}
    expected_ic = _normalise_ic(str(profile['llm']['icc_probe']['information_condition']))
    base_run = next((r for r in primary if _normalise_ic(r.information_condition) == expected_ic), None)
    return {'source_dir': source_dir, 'source_kind': source_kind, 'source_aliases': [], 'archive_record': archive_record, 'run_label': archive_record.run_label if archive_record is not None else str(summary.get('run_label') or source_dir.name), 'run_role': expected_role, 'information_condition': expected_ic, 'base_run_label': str(summary.get('base_run_label') or (base_run.run_label if base_run else '')), 'probe_schema_version': summary.get('probe_schema_version'), 'probe_target_feature': summary.get('probe_target_feature'), 'probe_tolerance_relative': summary.get('probe_tolerance_relative'), 'probe_row_count': summary.get('probe_row_count'), 'created_utc': summary.get('created_utc')}

def _resolve_runs(project_root: Path, profile: dict[str, Any], *, catalog_snapshot: Path | None=None) -> dict[str, Any]:
    records = discover_archived_runs(project_root, profile, catalog_snapshot=catalog_snapshot)
    llm = profile['llm']
    primary: list[ArchivedRunMetadata] = []
    for ic in llm['primary']['information_conditions']:
        rec = select_exact_role(records, run_role=llm['primary']['run_role'], information_condition=ic)
        primary.append(rec)
    supplementary: list[ArchivedRunMetadata] = []
    for item in llm['supplementary']:
        for ic in item['information_conditions']:
            rec = select_exact_role(records, run_role=item['run_role'], information_condition=ic)
            supplementary.append(rec)
    n5: list[ArchivedRunMetadata] = []
    for ic in llm['n5']['information_conditions']:
        rec = select_exact_role(records, run_role=llm['n5']['run_role'], information_condition=ic)
        n5.append(rec)
    frontier_cfg = llm['n5_budget_frontier']
    n5_budget_frontier = select_budget_frontier_role(records, run_role=frontier_cfg['run_role'], information_condition=frontier_cfg['information_condition'], expected_budgets=frontier_cfg['l1_budgets'], allow_absent=not bool(frontier_cfg['required_for_canonical_analysis']))
    matched_cfg = llm['n5_matched_budget_frontier']
    n5_matched_budget_frontier = select_budget_frontier_role(records, run_role=matched_cfg['run_role'], information_condition=matched_cfg['information_condition'], expected_budgets=matched_cfg['l1_budgets'], allow_absent=not bool(matched_cfg['required_for_canonical_analysis']))
    probe_selection = _resolve_probe_selection(project_root=project_root, profile=profile, records=records, primary=primary)
    primary_icb = next((r for r in primary if _normalise_ic(r.information_condition) == 'IC-b'))
    holm = [primary_icb, *supplementary]
    return {'all_records': records, 'llm_runs_root': resolve_workspace_path(project_root, profile['workspace']['llm_runs_root']), 'primary': primary, 'supplementary': supplementary, 'n5': n5, 'n5_budget_frontier': n5_budget_frontier, 'n5_matched_budget_frontier': n5_matched_budget_frontier, 'holm': holm, 'probe_selection': probe_selection, 'probe_source': probe_selection['source_dir']}

def _find_required_run_input(run_dir: Path, candidates: Sequence[str], label: str) -> Path:
    hits = [run_dir / rel for rel in candidates if (run_dir / rel).is_file()]
    if not hits:
        recursive_names = {Path(rel).name for rel in candidates}
        hits = sorted((p for p in run_dir.rglob('*') if p.is_file() and p.name in recursive_names))
    if not hits:
        raise FileNotFoundError(f'Missing required {label} under {run_dir}; expected one of {list(candidates)}')
    return hits[0].resolve()

def _check_required_inputs(*, project_root: Path, raw_nonfinancial_root: Path, resolved: dict[str, Any], stage6_summary: Path) -> list[dict[str, str]]:
    """Check only files/directories directly required by the active analyses."""
    checked: list[dict[str, str]] = []

    def require(path: Path, label: str, *, directory: bool=False) -> Path:
        ok = path.is_dir() if directory else path.is_file()
        if not ok:
            kind = 'directory' if directory else 'file'
            raise FileNotFoundError(f'Missing required {label} {kind}: {path}')
        checked.append({'label': label, 'path': str(path.resolve())})
        return path.resolve()
    require(stage6_summary, 'Stage6 summary')
    require(project_root / 'data' / 'final_freeze' / 'stage6_candidate_selector_eval' / 'multi_oracle_policy_eval.parquet', 'Stage6 row-by-candidate multi-oracle table')
    b2_dir = require(project_root / 'data' / 'final_freeze' / 'stage2_substrate_loopA_loopB2', 'LoopA/B2 input', directory=True)
    require(b2_dir / 'substrate_loopA_loopB2_report.json', 'LoopA/B2 report')
    require(raw_nonfinancial_root, 'raw nonfinancial input', directory=True)
    run_historical_n5m = os.environ.get('CREDIT_RECOURSE_RUN_HISTORICAL_N5M') == '1'
    if run_historical_n5m and resolved.get('n5_matched_budget_frontier'):
        require(project_root / 'data' / 'final_freeze' / 'stage2_candidate_projection' / 'phase_eval_candidate.parquet', 'Stage2 evaluation state panel for historical N5M adaptive selection')
        require(project_root / 'data' / 'final_freeze' / 'configs' / 'n5m_adaptive_selection_contract.json', 'Frozen historical N5M adaptive-selection contract')
    selected: dict[str, ArchivedRunMetadata] = {}
    for group in ('primary', 'supplementary', 'n5', 'n5_budget_frontier', 'n5_matched_budget_frontier'):
        for record in resolved[group]:
            selected[record.run_label] = record
    stage7_candidates = ('stage7_llm_action_generation/llm_stage7_action_table.parquet', 'stage7_llm_action_generation/llm_stage7_action_table.csv', 'stage7_action_generation/llm_stage7_action_table.parquet', 'stage7_action_generation/llm_stage7_action_table.csv')
    stage9_candidates = ('stage9_policy_comparison/llm_stage9_llm_rl_comparison.parquet', 'stage9_policy_comparison/llm_stage9_llm_rl_comparison.csv', 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.parquet', 'stage9_llm_rl_comparison/llm_stage9_llm_rl_comparison.csv', 'stage9_policy_comparison/llm_stage9_paired_vs_C0_noop.parquet', 'stage9_policy_comparison/llm_stage9_paired_vs_C0_noop.csv', 'stage9_llm_rl_comparison/llm_stage9_paired_vs_C0_noop.parquet', 'stage9_llm_rl_comparison/llm_stage9_paired_vs_C0_noop.csv')
    for record in selected.values():
        stage7 = _find_required_run_input(record.run_dir, stage7_candidates, 'Stage7 action table')
        stage9 = _find_required_run_input(record.run_dir, stage9_candidates, 'Stage9 comparison table')
        checked.append({'label': f'Stage7 action table: {record.run_label}', 'path': str(stage7)})
        checked.append({'label': f'Stage9 comparison table: {record.run_label}', 'path': str(stage9)})
    revision_candidates = ('stage9_policy_comparison/llm_stage9_revision_metrics.csv', 'stage9_llm_rl_comparison/llm_stage9_revision_metrics.csv')
    for record in resolved['primary']:
        revision = _find_required_run_input(record.run_dir, revision_candidates, 'Stage9 revision metrics')
        checked.append({'label': f'Stage9 revision metrics: {record.run_label}', 'path': str(revision)})
    probe_dir = Path(resolved['probe_source']).resolve()
    for name in ('icc_probe_summary.json', 'icc_probe_firm_level.csv', 'llm_stage7_icc_probe_checkpoint.jsonl'):
        require(probe_dir / name, f'IC-c probe {name}')
    return checked

def _record_dict(record: ArchivedRunMetadata, *, paper_use: str | None=None) -> dict[str, Any]:
    value = asdict(record)
    value['run_dir'] = str(record.run_dir)
    if paper_use is not None:
        if paper_use not in {'primary', 'supplementary', 'historical', 'not_selected'}:
            raise ValueError(f'Unsupported paper_use value: {paper_use!r}')
        value['paper_use'] = paper_use
    return value

def _probe_selection_dict(selection: dict[str, Any]) -> dict[str, Any]:
    archive_record = selection.get('archive_record')
    return {'run_label': selection['run_label'], 'run_role': selection['run_role'], 'information_condition': selection['information_condition'], 'source_kind': selection['source_kind'], 'source_dir': str(selection['source_dir']), 'source_aliases': selection.get('source_aliases', []), 'base_run_label': selection['base_run_label'], 'probe_schema_version': selection.get('probe_schema_version'), 'probe_target_feature': selection['probe_target_feature'], 'probe_tolerance_relative': selection['probe_tolerance_relative'], 'probe_row_count': selection['probe_row_count'], 'created_utc': selection.get('created_utc'), 'archive_record': _record_dict(archive_record) if archive_record is not None else None}

def _run_catalog(resolved: dict[str, Any], layout: PaperOutputLayout, profile: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    selected_labels: set[str] = set()
    paper_use_by_group = {'primary': 'primary', 'supplementary': 'supplementary', 'n5': 'primary', 'n5_budget_frontier': 'supplementary', 'n5_matched_budget_frontier': str(profile['llm']['n5_matched_budget_frontier']['paper_use']), 'holm': 'primary'}
    selected_use: dict[str, str] = {}
    for group in ('primary', 'supplementary', 'n5', 'n5_budget_frontier', 'n5_matched_budget_frontier', 'holm'):
        for record in resolved[group]:
            selected_labels.add(record.run_label)
            current = selected_use.get(record.run_label)
            proposed = paper_use_by_group[group]
            if current != 'primary' or proposed == 'primary':
                selected_use[record.run_label] = proposed
    probe_selection = resolved['probe_selection']
    probe_record = probe_selection.get('archive_record')
    if probe_record is not None:
        selected_labels.add(probe_record.run_label)
    for record in resolved['all_records']:
        row = _record_dict(record)
        row['selected_for_paper'] = record.run_label in selected_labels
        row['paper_use'] = selected_use.get(record.run_label, 'not_selected')
        row['source_kind'] = 'canonical_llm_archive'
        row['portable_run_provenance'] = portable_path_identity(record.run_dir, roots={'llm_runs': resolved['llm_runs_root']})
        rows.append(row)
    if probe_record is None:
        rows.append({'run_dir': str(probe_selection['source_dir']), 'run_label': probe_selection['run_label'], 'run_role': probe_selection['run_role'], 'information_condition': probe_selection['information_condition'], 'backend_id': None, 'backend_model': None, 'backend_provider': None, 'conditions': (), 'modes': (), 'seed': None, 'seed_source': None, 'reference_draw_seed': None, 'reference_draw_seed_source': None, 'candidate_library_quantile': None, 'candidate_library_quantile_source': None, 'row_count': probe_selection['probe_row_count'], 'request_count': probe_selection['probe_row_count'], 'freeform_l1_budget': None, 'freeform_l1_budget_source': None, 'has_stage7': False, 'has_stage8': False, 'has_stage9': False, 'has_probe': True, 'selected_for_paper': True, 'paper_use': 'primary', 'source_kind': probe_selection['source_kind'], 'base_run_label': probe_selection['base_run_label']})
    _write_json(layout.manifest / 'archived_llm_run_catalog.json', rows)
    pd.DataFrame(rows).to_csv(layout.manifest / 'archived_llm_run_catalog.csv', index=False, encoding='utf-8-sig')

def _find_frontier_panel(frontier_dir: Path, n5_ic_a_label: str) -> Path:
    """Return the IC-a N5 budget=1.27 l1-rescale panel by exact run identity.

    The frontier grid is built from the three N5 IC-a/b/c runs.  Selecting a
    panel by the primary (unbounded) IC-a run label is therefore incorrect and
    used to make the fallback glob see all three information conditions.
    Resolve the exact N5 IC-a run directory instead; never choose an arbitrary
    panel from a multi-hit glob.
    """
    preferred = frontier_dir / n5_ic_a_label / 'b1p27' / 'l1_rescale' / 'simulated_oracle_input_frame.parquet'
    if preferred.exists():
        return preferred.resolve()
    hits = sorted((p for p in frontier_dir.glob('*/b1p27/l1_rescale/simulated_oracle_input_frame.parquet') if p.parents[2].name == n5_ic_a_label))
    if len(hits) != 1:
        all_hits = sorted(frontier_dir.glob('*/b1p27/l1_rescale/simulated_oracle_input_frame.parquet'))
        raise FileNotFoundError(f'Expected exactly one N5 IC-a budget=1.27 l1_rescale panel under the current frontier output; run_label={n5_ic_a_label}, preferred={preferred}, matching={[str(p) for p in hits]}, all_found={[str(p) for p in all_hits]}')
    return hits[0].resolve()

def _validate_frontier_resume_state(frontier_dir: Path, n5_ic_a_label: str) -> Path:
    """Fail fast unless the failed run completed the full frontier grid."""
    metadata_path = frontier_dir / 'metadata.json'
    status_path = frontier_dir / 'frontier_grid_status.csv'
    if not metadata_path.is_file():
        raise FileNotFoundError(f'Cannot resume after frontier: metadata is missing: {metadata_path}')
    if not status_path.is_file():
        raise FileNotFoundError(f'Cannot resume after frontier: status table is missing: {status_path}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    if metadata.get('status') != 'PASS' or bool(metadata.get('dry_run')):
        raise RuntimeError(f"Cannot resume after frontier because the frontier metadata is not a completed PASS: status={metadata.get('status')!r}, dry_run={metadata.get('dry_run')!r}")
    status = pd.read_csv(status_path)
    required = {'run_label', 'budget', 'variant', 'status', 'output_dir'}
    missing = sorted(required - set(status.columns))
    if missing:
        raise ValueError(f'Cannot resume after frontier: status table is missing columns {missing}: {status_path}')
    bad = status.loc[status['status'].astype(str) != 'PASS']
    if not bad.empty:
        raise RuntimeError(f"Cannot resume after frontier: one or more frontier cells are not PASS; bad_rows={bad[['run_label', 'budget', 'variant', 'status']].to_dict(orient='records')[:20]}")
    expected_cells = int(metadata.get('n_cells', -1))
    if expected_cells <= 0 or len(status) != expected_cells:
        raise RuntimeError(f'Cannot resume after frontier: status row count does not match metadata; metadata_n_cells={expected_cells}, status_rows={len(status)}')
    return _find_frontier_panel(frontier_dir, n5_ic_a_label)
POSTFREEZE_ABLATION_CELLS: tuple[dict[str, object], ...] = ({'variant': 'global_mean_vector_null', 'target_budget': 'native', 'cell_name': 'global_mean_vector_null_native'}, {'variant': 'row_shuffle_vector_null', 'target_budget': 'native', 'cell_name': 'row_shuffle_vector_null_native', 'shuffle_within': 'none', 'shuffle_seeds': '__PROFILE__'}, {'variant': 'row_shuffle_vector_null', 'target_budget': 'native', 'cell_name': 'row_shuffle_vector_null_native_industry', 'shuffle_within': 'industry', 'shuffle_seeds': '__PROFILE__'}, {'variant': 'row_shuffle_vector_null', 'target_budget': 'native', 'cell_name': 'row_shuffle_vector_null_native_rating_band', 'shuffle_within': 'rating_band', 'shuffle_seeds': '__PROFILE__'}, {'variant': 'l1_rescale', 'target_budget': 'same_policy_candidate_mean', 'cell_name': 'l1_rescale_same_policy_candidate_mean'}, {'variant': 'global_mean_vector_null', 'target_budget': 'same_policy_candidate_mean', 'cell_name': 'global_mean_vector_null_same_policy_candidate_mean'}, {'variant': 'row_shuffle_vector_null', 'target_budget': 'same_policy_candidate_mean', 'cell_name': 'row_shuffle_vector_null_same_policy_candidate_mean'}, {'variant': 'nearest_candidate_projection', 'target_budget': 'native', 'cell_name': 'nearest_candidate_projection_native'})

def _run_ablation_cells(*, project_root: Path, raw_root: Path, primary: Iterable[ArchivedRunMetadata], layout: PaperOutputLayout, steps: list[dict[str, Any]], plan_only: bool, shuffle_seed_spec: str, selected_cells: set[tuple[str, str]] | None=None) -> None:
    cells = POSTFREEZE_ABLATION_CELLS
    for record in primary:
        ic = _normalise_ic(record.information_condition)
        if ic is None:
            raise RuntimeError(f'Missing information condition for {record.run_label}')
        for cell in cells:
            cell_name = str(cell['cell_name'])
            if selected_cells is not None and (record.run_label, cell_name) not in selected_cells:
                continue
            target = layout.ablation / ic / record.run_label / cell_name
            command = [sys.executable, '-m', 'credit_recourse.analysis.llm_action_budget_ablation', '--project-root', str(project_root), '--raw-root', str(raw_root), '--stage7-action-table', str(record.run_dir / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet'), '--output-dir', str(target), '--policies', 'C6', '--modes', 'free_form_10d', '--variant', str(cell['variant']), '--target-budget', str(cell['target_budget']), '--random-seed', '1', '--reference-policy', 'C3']
            if cell.get('shuffle_seeds'):
                seed_value = shuffle_seed_spec if cell['shuffle_seeds'] == '__PROFILE__' else str(cell['shuffle_seeds'])
                command.extend(['--shuffle-seeds', str(seed_value)])
            if cell.get('shuffle_within'):
                command.extend(['--shuffle-within', str(cell['shuffle_within'])])
            _run(command, project_root=project_root, label=f"post-freeze ablation {ic} {cell['cell_name']}", steps=steps, plan_only=plan_only)

def _run_signflip(*, project_root: Path, raw_root: Path, primary: Iterable[ArchivedRunMetadata], layout: PaperOutputLayout, steps: list[dict[str, Any]], plan_only: bool, selected_run_labels: set[str] | None=None) -> None:
    for record in primary:
        if selected_run_labels is not None and record.run_label not in selected_run_labels:
            continue
        ic = _normalise_ic(record.information_condition)
        target = layout.signflip / str(ic) / record.run_label
        _run([sys.executable, '-m', 'credit_recourse.analysis.llm_action_budget_ablation', '--project-root', str(project_root), '--raw-root', str(raw_root), '--stage7-action-table', str(record.run_dir / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet'), '--output-dir', str(target), '--policies', 'C6', '--modes', 'free_form_10d', '--variant', 'sign_flip_mean_vector', '--target-budget', 'native', '--random-seed', '1', '--reference-policy', 'C3'], project_root=project_root, label=f'sign-flip mean null {ic}', steps=steps, plan_only=plan_only)

def _run_main_harness_backend_decomposition(*, project_root: Path, runs: list[ArchivedRunMetadata], layout: PaperOutputLayout, steps: list[dict[str, Any]], plan_only: bool, label_suffix: str='') -> None:
    if len(runs) != 3:
        raise RuntimeError(f'Canonical main harness-backend decomposition requires exactly three IC-b backend runs; found {len(runs)}')
    _run([sys.executable, '-m', 'credit_recourse.analysis.main_harness_backend_decomposition', '--project-root', str(project_root), '--run-dirs', *[str(record.run_dir) for record in runs], '--output-dir', str(layout.main_harness_backend_decomposition)], project_root=project_root, label=f'Main crossed harness-vs-backend variance decomposition{label_suffix}', steps=steps, plan_only=plan_only, capture_child_output=True)

def _run_candidate_library_provenance(*, project_root: Path, layout: PaperOutputLayout, steps: list[dict[str, Any]], plan_only: bool) -> None:
    """Generate the run-local candidate-library provenance thesis assets."""
    _run([sys.executable, '-m', 'credit_recourse.verification.verify_candidate_library_provenance', '--project-root', str(project_root), '--output-dir', str(layout.registry / 'candidate_library_provenance')], project_root=project_root, label='candidate-library provenance thesis assets', steps=steps, plan_only=plan_only)

def _run_n5m_matched_outputs(*, project_root: Path, profile: dict[str, Any], runs: list[ArchivedRunMetadata], layout: PaperOutputLayout, steps: list[dict[str, Any]], plan_only: bool, label_suffix: str='') -> None:
    cfg = profile['llm']['n5_matched_budget_frontier']
    if str(cfg.get('paper_use')) == 'historical' and os.environ.get('CREDIT_RECOURSE_RUN_HISTORICAL_N5M') != '1':
        _record_optional_skip(label=f'Historical E1/N5M matched-budget analyses{label_suffix}', steps=steps, reason='D01 marks E1/N5M as SUPERSEDED_FIRST_OBSERVATION. Set CREDIT_RECOURSE_RUN_HISTORICAL_N5M=1 only to regenerate the historical appendix lineage.')
        return
    if not runs:
        _record_optional_skip(label=f'Historical E1/N5M matched-budget analyses{label_suffix}', steps=steps, reason='No historical N5M runs were supplied.')
        return
    if len(runs) != 4:
        raise RuntimeError(f'Historical N5M matched frontier requires exactly four runs when enabled; found {len(runs)}')
    _run([sys.executable, '-m', 'credit_recourse.analysis.n5_budget_frontier_holm_inference', '--run-dirs', *[str(x.run_dir) for x in runs], '--out-dir', str(layout.n5_matched_budget_frontier_holm), '--run-role', str(cfg['run_role']), '--information-condition', str(cfg['information_condition']), '--expected-budgets', *['unbounded' if x is None else str(x) for x in cfg['l1_budgets']], '--design', 'matched_c4_c6'], project_root=project_root, label=f'N5M matched generation-time budget frontier Holm{label_suffix}', steps=steps, plan_only=plan_only)
    _run([sys.executable, '-m', 'credit_recourse.analysis.n5m_posthoc', '--project-root', str(project_root), '--run-dirs', *[str(x.run_dir) for x in runs], '--output-dir', str(layout.n5m_posthoc), '--expected-budgets', *['unbounded' if x is None else str(x) for x in cfg['l1_budgets']]], project_root=project_root, label=f'N5M matched-budget post-hoc diagnostics{label_suffix}', steps=steps, plan_only=plan_only)
    _run([sys.executable, '-m', 'credit_recourse.analysis.n5m_adaptive_selection', '--project-root', str(project_root), '--firm-frame', str(layout.n5m_posthoc / 'n5m_firm_frame.parquet'), '--state-panel', str(project_root / 'data' / 'final_freeze' / 'stage2_candidate_projection' / 'phase_eval_candidate.parquet'), '--contract', str(project_root / 'data' / 'final_freeze' / 'configs' / 'n5m_adaptive_selection_contract.json'), '--output-dir', str(layout.n5m_adaptive_selection)], project_root=project_root, label=f'N5M Section 9.8 adaptive-budget selector and post-C4 gate{label_suffix}', steps=steps, plan_only=plan_only)

def _load_score_tie_metadata_status(path: Path) -> tuple[bool, str]:
    if not path.is_file():
        return (False, 'metadata.json is missing')
    try:
        metadata = json.loads(path.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        return (False, f'metadata.json is unreadable: {exc!r}')
    return score_tie_contract_status(metadata)

def _archive_score_tie_output(*, archive_root: Path, analysis_root: Path, output_dir: Path, reason: str) -> dict[str, str]:
    if not output_dir.exists():
        return {'source': str(output_dir), 'archive': '', 'reason': reason, 'action': 'MISSING_NO_ARCHIVE'}
    relative = output_dir.resolve().relative_to(analysis_root.resolve())
    destination = archive_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f'Score-tie invalidation archive already exists: {destination}')
    shutil.move(str(output_dir), str(destination))
    return {'source': str(output_dir), 'archive': str(destination), 'reason': reason, 'action': 'ARCHIVED'}

def _repair_score_tie_sensitive_outputs(*, project_root: Path, raw_root: Path, profile: dict[str, Any], primary: list[ArchivedRunMetadata], n5: list[ArchivedRunMetadata], layout: PaperOutputLayout, steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Regenerate only outputs whose persisted zero contract is stale.

    A source-code contract change is not satisfied by adding metadata to old
    results. Stale outputs are archived outside the paper tree and regenerated
    by their canonical producers.
    """
    stale_ablation: set[tuple[str, str]] = set()
    stale_signflip: set[str] = set()
    findings: list[dict[str, str]] = []
    for record in primary:
        ic = _normalise_ic(record.information_condition)
        if ic is None:
            raise RuntimeError(f'Missing information condition for {record.run_label}')
        for cell in POSTFREEZE_ABLATION_CELLS:
            cell_name = str(cell['cell_name'])
            output_dir = layout.ablation / ic / record.run_label / cell_name
            ok, detail = _load_score_tie_metadata_status(output_dir / 'metadata.json')
            if not ok:
                stale_ablation.add((record.run_label, cell_name))
                findings.append({'kind': 'ablation', 'run_label': record.run_label, 'cell': cell_name, 'path': str(output_dir), 'reason': detail})
        signflip_dir = layout.signflip / ic / record.run_label
        ok, detail = _load_score_tie_metadata_status(signflip_dir / 'metadata.json')
        if not ok:
            stale_signflip.add(record.run_label)
            findings.append({'kind': 'signflip', 'run_label': record.run_label, 'cell': 'sign_flip_mean_vector', 'path': str(signflip_dir), 'reason': detail})
    frontier_meta_ok, frontier_detail = _load_score_tie_metadata_status(layout.frontier / 'metadata.json')
    stale_frontier_cell_count = 0
    if frontier_meta_ok:
        for metadata_path in layout.frontier.rglob('metadata.json'):
            ok, detail = _load_score_tie_metadata_status(metadata_path)
            if not ok:
                stale_frontier_cell_count += 1
                frontier_detail = f'{metadata_path}: {detail}' if stale_frontier_cell_count == 1 else frontier_detail
    frontier_stale = not frontier_meta_ok or stale_frontier_cell_count > 0
    if frontier_stale:
        findings.append({'kind': 'frontier', 'run_label': '*', 'cell': '*', 'path': str(layout.frontier), 'reason': frontier_detail})
    audit: dict[str, Any] = {'schema_version': 'paper_repro_score_tie_resume_repair_v2', 'created_utc': _now(), 'status': 'NO_ACTION', 'contract': SCORE_TIE_CONTRACT_VERSION, 'findings': findings, 'archives': [], 'regenerated': {'ablation_cells': len(stale_ablation), 'signflip_cells': len(stale_signflip), 'frontier': bool(frontier_stale)}}
    audit_path = layout.verification / 'score_tie_resume_repair.json'
    if not findings:
        _write_json(audit_path, audit)
        return audit
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive_root = project_root / 'data' / 'archive' / 'paper_repro_score_tie_invalidated' / stamp
    archive_root.mkdir(parents=True, exist_ok=False)
    for finding in findings:
        if finding['kind'] == 'frontier':
            continue
        record = _archive_score_tie_output(archive_root=archive_root, analysis_root=layout.root, output_dir=Path(finding['path']), reason=finding['reason'])
        audit['archives'].append(record)
    if frontier_stale:
        audit['archives'].append(_archive_score_tie_output(archive_root=archive_root, analysis_root=layout.root, output_dir=layout.frontier, reason=frontier_detail))
    analysis_cfg = profile['analysis']
    if stale_ablation:
        _run_ablation_cells(project_root=project_root, raw_root=raw_root, primary=primary, layout=layout, steps=steps, plan_only=False, shuffle_seed_spec=str(analysis_cfg['shuffle_permutation_seeds']), selected_cells=stale_ablation)
    if stale_signflip:
        _run_signflip(project_root=project_root, raw_root=raw_root, primary=primary, layout=layout, steps=steps, plan_only=False, selected_run_labels=stale_signflip)
    if frontier_stale:
        _run([sys.executable, '-m', 'credit_recourse.analysis.action_budget_frontier_grid', '--project-root', str(project_root), '--raw-root', str(raw_root), '--runs', *[str(x.run_dir) for x in n5], '--out-dir', str(layout.frontier), '--grid', *[str(x) for x in analysis_cfg['frontier_grid']], '--variants', ','.join(analysis_cfg['frontier_variants']), '--policies', 'C6', '--modes', 'free_form_10d', '--reference-policy', 'C3', '--random-seed', '1'], project_root=project_root, label='Regenerate stale score-tie budget frontier [resume]', steps=steps, plan_only=False)
    remaining_errors: list[str] = []
    for run_label, cell_name in sorted(stale_ablation):
        record = next((x for x in primary if x.run_label == run_label))
        ic = _normalise_ic(record.information_condition)
        path = layout.ablation / str(ic) / run_label / cell_name / 'metadata.json'
        ok, detail = _load_score_tie_metadata_status(path)
        if not ok:
            remaining_errors.append(f'{path}: {detail}')
    for run_label in sorted(stale_signflip):
        record = next((x for x in primary if x.run_label == run_label))
        ic = _normalise_ic(record.information_condition)
        path = layout.signflip / str(ic) / run_label / 'metadata.json'
        ok, detail = _load_score_tie_metadata_status(path)
        if not ok:
            remaining_errors.append(f'{path}: {detail}')
    if frontier_stale:
        for metadata_path in layout.frontier.rglob('metadata.json'):
            ok, detail = _load_score_tie_metadata_status(metadata_path)
            if not ok:
                remaining_errors.append(f'{metadata_path}: {detail}')
        root_ok, root_detail = _load_score_tie_metadata_status(layout.frontier / 'metadata.json')
        if not root_ok:
            remaining_errors.append(f"{layout.frontier / 'metadata.json'}: {root_detail}")
    audit['status'] = 'PASS' if not remaining_errors else 'FAIL'
    audit['archive_root'] = str(archive_root)
    audit['errors'] = remaining_errors
    _write_json(audit_path, audit)
    if remaining_errors:
        raise RuntimeError('Score-tie resume regeneration did not satisfy the current metadata contract: ' + '; '.join(remaining_errors[:20]))
    return audit

def _resume_post_frontier_tail(*, project_root: Path, raw_root: Path, profile: dict[str, Any], catalog_snapshot: Path | None, primary: list[ArchivedRunMetadata], n5: list[ArchivedRunMetadata], n5_matched_budget_frontier: list[ArchivedRunMetadata], resolved: dict[str, Any], layout: PaperOutputLayout, c3_alpha: float, manifest: dict[str, Any], manifest_path: Path, steps: list[dict[str, Any]]) -> None:
    """Resume only the canonical steps that follow a completed frontier grid."""
    score_tie_repair = _repair_score_tie_sensitive_outputs(project_root=project_root, raw_root=raw_root, profile=profile, primary=primary, n5=n5, layout=layout, steps=steps)
    n5_ic_a = next((x for x in n5 if _normalise_ic(x.information_condition) == 'IC-a'))
    panel = _validate_frontier_resume_state(layout.frontier, n5_ic_a.run_label)
    manifest['n6_panel'] = str(panel)
    manifest.setdefault('resume_history', []).append({'created_utc': _now(), 'mode': 'after_frontier', 'frontier_dir': str(layout.frontier), 'frontier_run_label': n5_ic_a.run_label, 'panel': str(panel), 'score_tie_repair': score_tie_repair})
    _write_json(manifest_path, manifest)
    _run([sys.executable, '-m', 'credit_recourse.analysis.remaining_thesis_analyses', '--project-root', str(project_root), '--raw-root', str(raw_root), '--mode', 'noapi', '--analysis-profile', str(profile['_analysis_profile_name']), *(['--eligible-run-catalog', str(catalog_snapshot)] if catalog_snapshot is not None else []), '--allow-analysis-running'], project_root=project_root, label='remaining no-API analysis backfill [resume]', steps=steps, plan_only=False)
    _run_n5m_matched_outputs(project_root=project_root, profile=profile, runs=n5_matched_budget_frontier, layout=layout, steps=steps, plan_only=False, label_suffix=' [resume]')
    _run_main_harness_backend_decomposition(project_root=project_root, runs=resolved['holm'], layout=layout, steps=steps, plan_only=False, label_suffix=' [resume]')
    _run([sys.executable, '-m', 'credit_recourse.analysis.winrate_heterogeneity', '--runs', *[str(x.run_dir) for x in n5], '--panel', str(panel), '--out-dir', str(layout.winrate), '--oracle', 'alpha'], project_root=project_root, label='N3 win rate and N6 heterogeneity [resume]', steps=steps, plan_only=False)
    _run([sys.executable, '-m', 'credit_recourse.analysis.icc_probe_analysis', '--project-root', str(project_root), '--out-dir', str(layout.icc_probe), '--mode', 'import', '--source-dir', str(resolved['probe_source'])], project_root=project_root, label='IC-c probe import and channel summary [resume]', steps=steps, plan_only=False)
    manifest['status'] = 'PASS'
    manifest['error'] = None
    manifest['completed_utc'] = _now()
    _write_json(manifest_path, manifest)

def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    if not (project_root / 'src' / 'credit_recourse').is_dir():
        raise FileNotFoundError(f'Project root lacks src/credit_recourse: {project_root}')
    raw_root_arg = getattr(args, 'raw_root', None) or os.environ.get('CREDIT_RECOURSE_RAW_ROOT')
    if not raw_root_arg:
        raise ValueError('RawRoot is required; implicit ProjectRoot/data/raw fallback is disabled.')
    raw_root = Path(raw_root_arg).resolve()
    raw_nonfinancial_root = raw_root / 'raw_nonfinancial'
    analysis_profile_name = str(getattr(args, 'analysis_profile', 'current_comprehensive'))
    profile = load_profile(project_root, analysis_profile_name)
    catalog_snapshot_arg = getattr(args, 'eligible_run_catalog', None)
    catalog_snapshot = Path(catalog_snapshot_arg).resolve() if catalog_snapshot_arg else None
    eligibility_contract = _eligibility_manifest_contract(project_root, catalog_snapshot)
    requested_rl_mode = str(getattr(args, 'rl_validation_mode', 'auto'))
    resolved_rl_mode, rl_mode_source = resolve_rl_validation_mode(project_root, requested_rl_mode)
    resolved = _resolve_runs(project_root, profile, catalog_snapshot=catalog_snapshot)
    primary: list[ArchivedRunMetadata] = resolved['primary']
    n5: list[ArchivedRunMetadata] = resolved['n5']
    n5_budget_frontier: list[ArchivedRunMetadata] = resolved['n5_budget_frontier']
    n5_matched_budget_frontier: list[ArchivedRunMetadata] = resolved['n5_matched_budget_frontier']
    holm: list[ArchivedRunMetadata] = resolved['holm']
    manifest_task_order = list(profile['_manifest_task_order'])
    n5m_paper_use = str(profile['llm']['n5_matched_budget_frontier']['paper_use'])
    stage6_summary = find_stage6_summary(project_root, args.stage6_summary)
    if stage6_summary is None or not stage6_summary.exists():
        raise FileNotFoundError('Required Stage6 final_policy_summary.csv was not found.')
    required_inputs = _check_required_inputs(project_root=project_root, raw_nonfinancial_root=raw_nonfinancial_root, resolved=resolved, stage6_summary=stage6_summary)
    if getattr(args, 'check_inputs_only', False):
        result = {'status': 'PASS', 'project_root': str(project_root), 'required_input_count': len(required_inputs), 'required_inputs': required_inputs, 'rl_validation_mode': resolved_rl_mode, 'rl_validation_mode_source': rl_mode_source, 'selected_run_labels': {'primary': [x.run_label for x in primary], 'supplementary': [x.run_label for x in resolved['supplementary']], 'n5': [x.run_label for x in n5], 'n5_budget_frontier': [x.run_label for x in n5_budget_frontier], 'n5_matched_budget_frontier': [x.run_label for x in n5_matched_budget_frontier], 'icc_probe': resolved['probe_selection']['run_label']}}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    c3_means = load_stage6_canonical_means(stage6_summary)
    if 'stage6_canonical_c3_mean_alpha' not in c3_means:
        raise RuntimeError(f'Stage6 summary does not contain canonical alpha C3 mean: {stage6_summary}')
    c3_alpha = float(c3_means['stage6_canonical_c3_mean_alpha'])
    resume_after_frontier = bool(getattr(args, 'resume_after_frontier', False))
    analysis_dir = Path(args.analysis_dir)
    if not analysis_dir.is_absolute():
        analysis_dir = project_root / analysis_dir
    analysis_dir = analysis_dir.resolve()
    layout = build_layout(analysis_dir)
    ensure_layout(layout)
    _run_catalog(resolved, layout, profile)
    manifest_path = layout.manifest / 'paper_repro_analysis_manifest.json'
    if resume_after_frontier:
        if not manifest_path.is_file():
            raise FileNotFoundError(f'Cannot resume after frontier because the prior analysis manifest is missing: {manifest_path}')
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        steps = list(manifest.get('steps', []))
        manifest['steps'] = steps
        manifest['status'] = 'RUNNING'
        manifest['error'] = None
        manifest['project_root'] = str(project_root)
        manifest['analysis_dir'] = str(analysis_dir)
        manifest['profile_name'] = profile['profile_name']
        manifest['tasks'] = manifest_task_order
        manifest['analysis_profile'] = analysis_profile_name
        manifest['analysis_profile_id'] = analysis_profile_name
        manifest['eligible_run_catalog'] = str(catalog_snapshot) if catalog_snapshot is not None else None
        manifest.update(eligibility_contract)
        manifest['resolved_n5m_paper_use'] = n5m_paper_use
        manifest['score_tie_contract_id'] = SCORE_TIE_CONTRACT_VERSION
        manifest['kw_contract_id'] = KW_CONSTANT_CONTRACT_VERSION
        manifest['strict_parity_status'] = 'NOT_RUN'
        manifest['scientific_parity_status'] = 'NOT_RUN'
        manifest['stage6_summary'] = str(stage6_summary)
        manifest['stage6_canonical_c3_means'] = c3_means
        manifest['rl_validation_mode'] = resolved_rl_mode
        manifest['rl_validation_mode_source'] = rl_mode_source
        manifest['required_inputs'] = required_inputs
        manifest['selected_runs'] = {'primary': [_record_dict(x, paper_use='primary') for x in primary], 'supplementary': [_record_dict(x, paper_use='supplementary') for x in resolved['supplementary']], 'n5': [_record_dict(x, paper_use='primary') for x in n5], 'n5_budget_frontier': [_record_dict(x, paper_use='supplementary') for x in n5_budget_frontier], 'n5_matched_budget_frontier': [_record_dict(x, paper_use=n5m_paper_use) for x in n5_matched_budget_frontier], 'holm': [_record_dict(x, paper_use='primary') for x in holm], 'icc_probe': _probe_selection_dict(resolved['probe_selection'])}
    else:
        steps: list[dict[str, Any]] = []
        manifest = {'schema_version': SCHEMA_VERSION, 'created_utc': _now(), 'status': 'RUNNING', 'project_root': str(project_root), 'analysis_dir': str(analysis_dir), 'profile_name': profile['profile_name'], 'tasks': manifest_task_order, 'analysis_profile': analysis_profile_name, 'analysis_profile_id': analysis_profile_name, 'eligible_run_catalog': str(catalog_snapshot) if catalog_snapshot is not None else None, **eligibility_contract, 'resolved_n5m_paper_use': n5m_paper_use, 'score_tie_contract_id': SCORE_TIE_CONTRACT_VERSION, 'kw_contract_id': KW_CONSTANT_CONTRACT_VERSION, 'strict_parity_status': 'NOT_RUN', 'scientific_parity_status': 'NOT_RUN', 'stage6_summary': str(stage6_summary), 'stage6_canonical_c3_means': c3_means, 'rl_validation_mode': resolved_rl_mode, 'rl_validation_mode_source': rl_mode_source, 'required_inputs': required_inputs, 'selected_runs': {'primary': [_record_dict(x, paper_use='primary') for x in primary], 'supplementary': [_record_dict(x, paper_use='supplementary') for x in resolved['supplementary']], 'n5': [_record_dict(x, paper_use='primary') for x in n5], 'n5_budget_frontier': [_record_dict(x, paper_use='supplementary') for x in n5_budget_frontier], 'n5_matched_budget_frontier': [_record_dict(x, paper_use=n5m_paper_use) for x in n5_matched_budget_frontier], 'holm': [_record_dict(x, paper_use='primary') for x in holm], 'icc_probe': _probe_selection_dict(resolved['probe_selection'])}, 'steps': steps, 'error': None}
    _write_json(manifest_path, manifest)
    if resume_after_frontier:
        try:
            _resume_post_frontier_tail(project_root=project_root, raw_root=raw_root, profile=profile, catalog_snapshot=catalog_snapshot, primary=primary, n5=n5, n5_matched_budget_frontier=n5_matched_budget_frontier, resolved=resolved, layout=layout, c3_alpha=c3_alpha, manifest=manifest, manifest_path=manifest_path, steps=steps)
        except Exception as exc:
            manifest['status'] = 'FAIL'
            manifest['error'] = repr(exc)
            raise
        finally:
            manifest['completed_utc'] = _now()
            _write_json(manifest_path, manifest)
        print('\nPAPER POST-FREEZE ANALYSIS RESUME FINISHED')
        print('Output:', analysis_dir)
        print('Manifest:', manifest_path)
        return manifest
    try:
        b2_dir = project_root / 'data' / 'final_freeze' / 'stage2_substrate_loopA_loopB2'
        b2_report = b2_dir / 'substrate_loopA_loopB2_report.json'
        raw_nonfinancial = raw_nonfinancial_root
        analysis_cfg = profile['analysis']
        _run([sys.executable, '-m', 'credit_recourse.oracle.verification.diagnose_b2_gap_decomposition', '--b2-dir', str(b2_dir), '--out-dir', str(layout.b2_gap), '--report', str(b2_report)], project_root=project_root, label='B2 gap decomposition', steps=steps, plan_only=args.plan_only)
        _run([sys.executable, '-m', 'credit_recourse.oracle.verification.verify_stage2_test3_counterfactual_fidelity', '--b2-dir', str(b2_dir), '--out-dir', str(layout.test3), '--report', str(b2_report), '--diagnostic-report', str(layout.b2_gap / 'b2_gap_decomposition_report.json')], project_root=project_root, label='Test3 counterfactual-device properties', steps=steps, plan_only=args.plan_only)
        _run([sys.executable, '-m', 'credit_recourse.analysis.b2_structural_event_slice', '--b2-dir', str(b2_dir), '--raw-nonfinancial-dir', str(raw_nonfinancial), '--test3-rows', str(layout.test3 / 'test3_rows.csv'), '--output-dir', str(layout.structural_slice)], project_root=project_root, label='B2 structural/material-event slice', steps=steps, plan_only=args.plan_only)
        _run_ablation_cells(project_root=project_root, raw_root=raw_root, primary=primary, layout=layout, steps=steps, plan_only=args.plan_only, shuffle_seed_spec=str(analysis_cfg['shuffle_permutation_seeds']))
        _run([sys.executable, '-m', 'credit_recourse.analysis.llm_holm_inference', '--runs', *[str(x.run_dir) for x in holm], '--backend-labels', *[x.run_label for x in holm], '--out-dir', str(layout.holm), '--holm-family', 'hypothesis_mode'], project_root=project_root, label='H1/H3 Holm inference', steps=steps, plan_only=args.plan_only)
        _run([sys.executable, '-m', 'credit_recourse.analysis.reference_quality_acceptance', '--project-root', str(project_root), '--run-dirs', *[str(x.run_dir) for x in primary], '--output-dir', str(layout.reference_quality_acceptance), '--bootstrap-draws', '1000', '--bootstrap-seed', '20260711'], project_root=project_root, label='reference-quality acceptance response', steps=steps, plan_only=args.plan_only)
        _run_candidate_library_provenance(project_root=project_root, layout=layout, steps=steps, plan_only=args.plan_only)
        _run([sys.executable, '-m', 'credit_recourse.analysis.n5_7_10c_holm_inference', '--project-root', str(project_root), '--run-dirs', *[str(x.run_dir) for x in n5], '--out-dir', str(layout.n5_holm), '--stage6-summary', str(stage6_summary), '--rl-validation-mode', resolved_rl_mode, '--holm-family', 'oracle'], project_root=project_root, label='N5 Table 7-10c Holm and C3 audit', steps=steps, plan_only=args.plan_only)
        frontier_cfg = profile['llm']['n5_budget_frontier']
        if n5_budget_frontier:
            _run([sys.executable, '-m', 'credit_recourse.analysis.n5_budget_frontier_holm_inference', '--run-dirs', *[str(x.run_dir) for x in n5_budget_frontier], '--out-dir', str(layout.n5_budget_frontier_holm), '--run-role', str(frontier_cfg['run_role']), '--information-condition', str(frontier_cfg['information_condition']), '--expected-budgets', *['unbounded' if x is None else str(x) for x in frontier_cfg['l1_budgets']]], project_root=project_root, label='N5 generation-time budget frontier Holm', steps=steps, plan_only=args.plan_only)
        else:
            _record_optional_skip(label='N5 generation-time budget frontier Holm', steps=steps, reason='No archived paper_n5_budget_frontier_icb runs were found. The live 4-arm frontier is an optional post-analysis task and is not required to complete the frozen no-API paper analysis.')
        _run_n5m_matched_outputs(project_root=project_root, profile=profile, runs=n5_matched_budget_frontier, layout=layout, steps=steps, plan_only=args.plan_only)
        _run_main_harness_backend_decomposition(project_root=project_root, runs=holm, layout=layout, steps=steps, plan_only=args.plan_only)
        _run_signflip(project_root=project_root, raw_root=raw_root, primary=primary, layout=layout, steps=steps, plan_only=args.plan_only)
        _run([sys.executable, '-m', 'credit_recourse.analysis.action_budget_frontier_grid', '--project-root', str(project_root), '--raw-root', str(raw_root), '--runs', *[str(x.run_dir) for x in n5], '--out-dir', str(layout.frontier), '--grid', *[str(x) for x in analysis_cfg['frontier_grid']], '--variants', ','.join(analysis_cfg['frontier_variants']), '--policies', 'C6', '--modes', 'free_form_10d', '--reference-policy', 'C3', '--random-seed', '1'], project_root=project_root, label='N1 full budget frontier', steps=steps, plan_only=args.plan_only)
        n5_ic_a = next((x for x in n5 if _normalise_ic(x.information_condition) == 'IC-a'))
        panel = Path('PLANNED_PANEL.parquet') if args.plan_only else _find_frontier_panel(layout.frontier, n5_ic_a.run_label)
        manifest['n6_panel'] = str(panel)
        _run([sys.executable, '-m', 'credit_recourse.analysis.winrate_heterogeneity', '--runs', *[str(x.run_dir) for x in n5], '--panel', str(panel), '--out-dir', str(layout.winrate), '--oracle', 'alpha'], project_root=project_root, label='N3 win rate and N6 heterogeneity', steps=steps, plan_only=args.plan_only)
        _run([sys.executable, '-m', 'credit_recourse.analysis.icc_probe_analysis', '--project-root', str(project_root), '--out-dir', str(layout.icc_probe), '--mode', 'import', '--source-dir', str(resolved['probe_source'])], project_root=project_root, label='IC-c probe import and channel summary', steps=steps, plan_only=args.plan_only)
        if args.plan_only:
            manifest['status'] = 'PLANNED'
        else:
            manifest['status'] = 'PASS'
            manifest['completed_utc'] = _now()
            _write_json(manifest_path, manifest)
    except Exception as exc:
        manifest['status'] = 'FAIL'
        manifest['error'] = repr(exc)
        raise
    finally:
        manifest['completed_utc'] = _now()
        _write_json(manifest_path, manifest)
    print('\nALL PAPER POST-FREEZE ANALYSES FINISHED')
    print('Output:', analysis_dir)
    print('Manifest:', manifest_path)
    return manifest

def _write_failure_report(path: Path, args: argparse.Namespace, exc: BaseException) -> None:
    report = {'schema_version': 'paper_repro_failure_report_v2', 'created_utc': _now(), 'status': 'FAIL', 'project_root': str(Path(args.project_root).resolve()), 'analysis_dir': str(Path(args.analysis_dir)), 'plan_only': bool(getattr(args, 'plan_only', False)), 'check_inputs_only': bool(getattr(args, 'check_inputs_only', False)), 'exception_type': type(exc).__name__, 'exception_message': str(exc), 'exception_repr': repr(exc), 'traceback': traceback.format_exc()}
    analysis_dir = Path(args.analysis_dir)
    manifest_path = analysis_dir / '00_manifest' / 'paper_repro_analysis_manifest.json'
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
            failed_steps = [step for step in manifest.get('steps', []) if step.get('status') == 'FAIL']
            if failed_steps:
                report['last_failed_step'] = failed_steps[-1]
        except Exception as manifest_exc:
            report['failure_context_read_error'] = repr(manifest_exc)
    _write_json(Path(path), report)

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--raw-root', default=None, help='Explicit read-only raw input root containing raw_nonfinancial. There is no ProjectRoot/data/raw fallback.')
    parser.add_argument('--analysis-dir', default=str(Path('data') / 'analysis' / 'paper_repro'))
    parser.add_argument('--stage6-summary', default=None)
    parser.add_argument('--analysis-profile', default='current_comprehensive', choices=['current_comprehensive', 'historical_20260715'], help='Named current or historical analysis-selection contract.')
    parser.add_argument('--eligible-run-catalog', default=None, help='Explicit frozen CSV catalog whose run_label values define eligibility.')
    parser.add_argument('--rl-validation-mode', choices=['auto', 'frozen_replay', 'fresh_retrain'], default='auto', help='RL/archived-LLM comparison contract. auto reuses the canonical RL marker; without a marker it defaults conservatively to frozen_replay.')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--resume-after-frontier', action='store_true', help='Resume a failed canonical analysis directory after a completed action-budget frontier. Pre-frontier diagnostics are preserved unless their persisted numerical tie contract is stale; stale ablation/sign-flip/frontier outputs are archived and regenerated before the remaining no-API tail and final contract.')
    parser.add_argument('--check-inputs-only', action='store_true', help='Check only the files/directories required by the active analysis and exit.')
    parser.add_argument('--failure-report', default=None, help='Write a structured JSON failure report before returning a non-zero exit code.')
    parser.add_argument('--concise-errors', action='store_true', help='Print only exception type/message to stderr; full traceback remains in --failure-report.')
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        run_analysis(args)
    except Exception as exc:
        if args.failure_report:
            _write_failure_report(Path(args.failure_report), args, exc)
        if args.concise_errors:
            print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
            return 1
        raise
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
