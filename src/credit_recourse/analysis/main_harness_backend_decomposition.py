from __future__ import annotations
'Canonical crossed-panel harness-vs-backend variance decomposition.\n\nThis analysis uses only the common IC-b post-patch grid where both axes vary:\n\n* harness cell: C4/C5/C6/C6X/C7/C8 x candidate_selection/free_form_10d;\n* backend: the canonical GPT-5.4-mini, GPT-4.1-mini, and Haiku 4.5 runs.\n\nN5, N5F, and N5M are explicitly forbidden because budget is not crossed with\nbackend in those runs. Pooling them would mechanically enlarge the harness axis\nand confound run time with the backend comparison.\n\nThe module is read-only with respect to ``data/final_freeze`` and writes a\nfully auditable panel, alignment audit, cell means, decomposition table, swing\nsummary, and manifest under the canonical paper analysis tree.\n'
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from credit_recourse.contracts.paper_reproduction import inspect_archived_run, load_profile
SCHEMA_VERSION = 'main_harness_backend_decomposition_v4'
PANEL_CONTRACT = 'common_crossed_icb_harness_backend_panel_v3'
SOLVER_CONTRACT = 'exact_group_projection_plus_small_additive_lstsq_v1'
ROLE_SOURCE_ARCHIVE = 'archive_manifest.run_role'
ROLE_SOURCE_STAGE7 = 'stage7.metadata.run_role'
ROLE_SOURCE_LEGACY_BRIDGE = 'paper_reproduction.inspect_archived_run.legacy_role_bridge'
ALLOWED_ROLE_SOURCES = (ROLE_SOURCE_ARCHIVE, ROLE_SOURCE_STAGE7, ROLE_SOURCE_LEGACY_BRIDGE)
ARCHIVE_MANIFEST_PRESENT = 'PRESENT'
ARCHIVE_MANIFEST_ABSENT_LEGACY_ALLOWED = 'ABSENT_LEGACY_ALLOWED'
ORACLES = ('alpha', 'beta', 'gamma')
FORBIDDEN_RUN_FAMILIES = ('N5', 'N5F', 'N5M')

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'Required JSON is missing: {path}')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return payload

def _read_optional_json(path: Path) -> tuple[dict[str, Any], bool]:
    """Read an optional legacy metadata file without weakening malformed-file checks.

    The immutable July 2026 main-backend archive set is heterogeneous: one
    canonical GPT-4.1-mini directory predates ``archive_manifest.json``.  A truly
    absent file is therefore represented as an empty object plus ``False`` and
    may only be accepted later through the canonical legacy role bridge.  If the
    path exists, it remains a strict JSON-object contract; malformed or non-file
    paths hard-fail exactly as before.
    """
    if not path.exists():
        return ({}, False)
    if not path.is_file():
        raise FileNotFoundError(f'Optional JSON path exists but is not a file: {path}')
    return (_read_json(path), True)

def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Required table is missing: {path}')
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table extension: {path}')

def _normalise_ic(value: Any) -> str:
    text = str(value or '').strip().lower().replace('_', '-')
    if text in {'ica', 'ic-a'}:
        return 'IC-a'
    if text in {'icb', 'ic-b'}:
        return 'IC-b'
    if text in {'icc', 'ic-c'}:
        return 'IC-c'
    return str(value or '').strip()

def _fit_sse(y: np.ndarray, categorical_columns: Iterable[pd.Series]) -> float:
    """Return additive-model SSE for the small crossed design only.

    This helper must never be used for the 575-level firm effect.  The firm and
    one-way categorical projections have exact group-mean solutions implemented
    in :func:`_one_way_group_sse`; using a dense 20k x 575 dummy matrix is both
    unnecessary and materially unstable/slow on the Windows reference runtime.
    """
    arrays: list[np.ndarray] = [np.ones((len(y), 1), dtype=float)]
    for series in categorical_columns:
        dummies = pd.get_dummies(series.astype(str), drop_first=True, dtype=float)
        if not dummies.empty:
            arrays.append(dummies.to_numpy(dtype=float))
    design = np.concatenate(arrays, axis=1)
    if design.shape[1] > 64:
        raise ValueError(f'Dense additive least-squares design unexpectedly exceeds 64 columns; shape={design.shape}. Firm effects must use exact group projection.')
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ beta
    return float(np.square(residual).sum())

def _one_way_group_sse(y: np.ndarray, groups: pd.Series) -> float:
    """Exact SSE after projection onto a one-way categorical group effect."""
    values = np.asarray(y, dtype=float)
    if values.ndim != 1 or len(values) != len(groups):
        raise ValueError(f'One-way projection length mismatch: y_shape={values.shape}, groups={len(groups)}')
    if not np.isfinite(values).all():
        raise ValueError('One-way projection received non-finite values')
    group_values = groups.astype(str).reset_index(drop=True)
    if group_values.isna().any():
        raise ValueError('One-way projection received missing group labels')
    frame = pd.DataFrame({'value': values, 'group': group_values})
    fitted = frame.groupby('group', sort=False)['value'].transform('mean').to_numpy(dtype=float)
    residual = values - fitted
    return float(np.square(residual).sum())

def _r2_from_sse(sse: float, sst: float) -> float:
    if not np.isfinite(sse) or not np.isfinite(sst) or (not sst > 0):
        raise ValueError(f'Invalid variance decomposition denominator: sse={sse}, sst={sst}')
    value = 1.0 - sse / sst
    if value < -1e-10 or value > 1.0 + 1e-10:
        raise ValueError(f'R2 outside numerical tolerance: {value}')
    return float(min(1.0, max(0.0, value)))

def _validate_role_config(profile: dict[str, Any]) -> dict[str, Any]:
    config = profile.get('analysis', {}).get('main_harness_backend_decomposition')
    if not isinstance(config, dict):
        raise ValueError('Profile is missing analysis.main_harness_backend_decomposition')
    required = {'information_condition', 'run_role_backend_labels', 'conditions', 'modes', 'expected_firm_count', 'minimum_backend_coverage', 'minimum_cell_coverage', 'excluded_run_families'}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f'Main decomposition profile config missing fields: {missing}')
    roles = config['run_role_backend_labels']
    if not isinstance(roles, dict) or len(roles) != 3:
        raise ValueError('Main decomposition must define exactly three crossed run roles')
    if sorted(config['excluded_run_families']) != sorted(FORBIDDEN_RUN_FAMILIES):
        raise ValueError('Main decomposition excluded_run_families must be exactly N5/N5F/N5M')
    return config

def _run_identity(run_dir: Path) -> tuple[str, str, str, str, bool, bool, dict[str, Any], dict[str, Any]]:
    """Resolve a canonical run identity without mutating frozen metadata.

    Fresh archives must persist ``run_role`` explicitly.  The July 2026 frozen
    main-backend archives predate that field, and one canonical GPT-4.1-mini run
    also predates ``archive_manifest.json`` itself.  This module therefore reuses
    the single canonical compatibility bridge in ``inspect_archived_run`` only
    after preserving strict checks for every metadata file that actually exists.
    Unknown, conflicting, or non-canonical identities remain hard failures.
    """
    archive_path = run_dir / 'archive_manifest.json'
    stage7_meta_path = run_dir / 'stage7_llm_action_generation' / 'metadata.json'
    archive, archive_manifest_present = _read_optional_json(archive_path)
    stage7_meta = _read_json(stage7_meta_path)
    archive_role = str(archive.get('run_role') or '').strip()
    stage7_role = str(stage7_meta.get('run_role') or '').strip()
    if archive_role and stage7_role and (archive_role != stage7_role):
        raise ValueError(f'Conflicting explicit run roles: archive={archive_role}, stage7={stage7_role}, run={run_dir}')
    inspected = inspect_archived_run(run_dir)
    if archive_role:
        run_role = archive_role
        run_role_source = ROLE_SOURCE_ARCHIVE
        run_role_explicit = True
    elif stage7_role:
        run_role = stage7_role
        run_role_source = ROLE_SOURCE_STAGE7
        run_role_explicit = True
    else:
        run_role = str(inspected.run_role or '').strip()
        run_role_source = ROLE_SOURCE_LEGACY_BRIDGE
        run_role_explicit = False
    if not run_role:
        raise ValueError(f'Run role is missing and the canonical frozen-run compatibility bridge could not resolve it: {run_dir}')
    if not archive_manifest_present and run_role_source != ROLE_SOURCE_LEGACY_BRIDGE:
        raise ValueError(f'archive_manifest.json may be absent only for a canonical frozen legacy main-backend run resolved through the compatibility bridge: {run_dir}')
    run_label = str(archive.get('run_label') or inspected.run_label or run_dir.name)
    backend_id = str(stage7_meta.get('backend_id') or inspected.backend_id or '').strip()
    if not backend_id:
        raise ValueError(f'Stage7 backend_id is missing: {run_dir}')
    return (run_label, run_role, backend_id, run_role_source, run_role_explicit, archive_manifest_present, archive, stage7_meta)

def _build_panel(*, run_dirs: list[Path], profile: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    config = _validate_role_config(profile)
    role_labels = {str(key): str(value) for key, value in config['run_role_backend_labels'].items()}
    expected_roles = set(role_labels)
    expected_ic = _normalise_ic(config['information_condition'])
    expected_conditions = [str(value) for value in config['conditions']]
    expected_modes = [str(value) for value in config['modes']]
    expected_cells = [f'{policy}__{mode}' for policy in expected_conditions for mode in expected_modes]
    expected_firms = int(config['expected_firm_count'])
    min_backend_coverage = float(config['minimum_backend_coverage'])
    min_cell_coverage = float(config['minimum_cell_coverage'])
    if len(run_dirs) != len(expected_roles):
        raise ValueError(f'Expected exactly {len(expected_roles)} crossed-panel runs; found {len(run_dirs)}')
    panels: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    input_files: list[dict[str, Any]] = []
    observed_roles: set[str] = set()
    common_firm_ids: set[int] | None = None
    for run_dir_raw in run_dirs:
        run_dir = Path(run_dir_raw).resolve()
        run_label, run_role, backend_id, run_role_source, run_role_explicit, archive_manifest_present, archive, stage7_meta = _run_identity(run_dir)
        if run_role not in expected_roles:
            raise ValueError(f'Run role is not part of the crossed-panel contract: role={run_role}, run={run_label}')
        if run_role in observed_roles:
            raise ValueError(f'Duplicate crossed-panel run role: {run_role}')
        observed_roles.add(run_role)
        upper_label = run_label.upper()
        if any((upper_label.startswith(prefix) for prefix in FORBIDDEN_RUN_FAMILIES)):
            raise ValueError(f'Forbidden N5/N5F/N5M run entered main decomposition: {run_label}')
        information_condition = _normalise_ic(stage7_meta.get('information_condition'))
        if information_condition != expected_ic:
            raise ValueError(f'Main decomposition requires {expected_ic}; found {information_condition} in {run_label}')
        stage7_path = run_dir / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet'
        stage8_path = run_dir / 'stage8_llm_multi_oracle_eval' / 'llm_stage8_multi_oracle_scores.parquet'
        stage7 = _read_table(stage7_path)
        stage8 = _read_table(stage8_path)
        required_key = {'row_id', 'policy', 'mode'}
        missing_stage7 = sorted(required_key - set(stage7.columns))
        missing_stage8 = sorted(required_key - set(stage8.columns))
        if missing_stage7:
            raise ValueError(f'{run_label}: Stage7 missing key columns: {missing_stage7}')
        if missing_stage8:
            raise ValueError(f'{run_label}: Stage8 missing key columns: {missing_stage8}')
        score_columns = [f'delta_R_score_{oracle}' for oracle in ORACLES]
        missing_scores = sorted(set(score_columns) - set(stage8.columns))
        if missing_scores:
            raise ValueError(f'{run_label}: Stage8 missing score columns: {missing_scores}')
        s7 = stage7.loc[stage7['policy'].astype(str).isin(expected_conditions) & stage7['mode'].astype(str).isin(expected_modes)].copy()
        s8 = stage8.loc[stage8['policy'].astype(str).isin(expected_conditions) & stage8['mode'].astype(str).isin(expected_modes)].copy()
        for frame, label in ((s7, 'Stage7'), (s8, 'Stage8')):
            frame['row_id'] = pd.to_numeric(frame['row_id'], errors='raise').astype(int)
            if frame.duplicated(['row_id', 'policy', 'mode']).any():
                sample = frame.loc[frame.duplicated(['row_id', 'policy', 'mode'], keep=False), ['row_id', 'policy', 'mode']].head(10)
                raise ValueError(f"{run_label}: duplicate {label} keys: {sample.to_dict('records')}")
        actual_conditions = sorted(s7['policy'].astype(str).unique())
        actual_modes = sorted(s7['mode'].astype(str).unique())
        if actual_conditions != sorted(expected_conditions) or actual_modes != sorted(expected_modes):
            raise ValueError(f'{run_label}: incomplete harness grid; conditions={actual_conditions}, modes={actual_modes}')
        s7_keys = set(map(tuple, s7[['row_id', 'policy', 'mode']].itertuples(index=False, name=None)))
        s8_keys = set(map(tuple, s8[['row_id', 'policy', 'mode']].itertuples(index=False, name=None)))
        if s7_keys != s8_keys:
            only_s7 = sorted(s7_keys - s8_keys)[:10]
            only_s8 = sorted(s8_keys - s7_keys)[:10]
            raise ValueError(f'{run_label}: Stage7/Stage8 key mismatch; only_stage7={only_s7}, only_stage8={only_s8}')
        firm_ids = set(s7['row_id'].tolist())
        if len(firm_ids) != expected_firms:
            raise ValueError(f'{run_label}: unique firm count={len(firm_ids)}, expected={expected_firms}')
        if common_firm_ids is None:
            common_firm_ids = firm_ids
        elif firm_ids != common_firm_ids:
            raise ValueError(f'{run_label}: firm row_id universe differs across backends')
        expected_observations = expected_firms * len(expected_cells)
        observed_observations = len(s8)
        coverage = observed_observations / expected_observations
        if coverage < min_backend_coverage:
            raise ValueError(f'{run_label}: backend observation coverage={coverage:.6f}, required>={min_backend_coverage:.6f}')
        s8['harness_cell'] = s8['policy'].astype(str) + '__' + s8['mode'].astype(str)
        for cell in expected_cells:
            count = int(s8['harness_cell'].eq(cell).sum())
            cell_coverage = count / expected_firms
            if cell_coverage < min_cell_coverage:
                raise ValueError(f'{run_label}: harness cell {cell} coverage={cell_coverage:.6f}, required>={min_cell_coverage:.6f}')
            missing_rows.append({'run_label': run_label, 'run_role': run_role, 'backend_label': role_labels[run_role], 'run_role_source': run_role_source, 'run_role_explicit': run_role_explicit, 'archive_manifest_present': archive_manifest_present, 'harness_cell': cell, 'expected_rows': expected_firms, 'observed_rows': count, 'missing_rows': expected_firms - count, 'coverage': cell_coverage})
        if s8[score_columns].apply(pd.to_numeric, errors='coerce').isna().any().any():
            raise ValueError(f'{run_label}: Stage8 contains missing/non-numeric Oracle scores')
        panel = s8[['row_id', 'policy', 'mode', 'harness_cell', *score_columns]].copy()
        panel.insert(0, 'backend_id', backend_id)
        panel.insert(0, 'backend_label', role_labels[run_role])
        panel.insert(0, 'run_role', run_role)
        panel.insert(0, 'run_label', run_label)
        panels.append(panel)
        audit_rows.append({'run_label': run_label, 'run_role': run_role, 'backend_label': role_labels[run_role], 'backend_id': backend_id, 'run_role_source': run_role_source, 'run_role_explicit': run_role_explicit, 'archive_manifest_present': archive_manifest_present, 'information_condition': information_condition, 'firm_count': len(firm_ids), 'expected_harness_cell_count': len(expected_cells), 'observed_harness_cell_count': int(s8['harness_cell'].nunique()), 'expected_observation_count': expected_observations, 'observed_observation_count': observed_observations, 'missing_observation_count': expected_observations - observed_observations, 'observation_coverage': coverage, 'stage7_stage8_key_alignment': 'PASS'})
        archive_manifest_path = run_dir / 'archive_manifest.json'
        archive_manifest_status = ARCHIVE_MANIFEST_PRESENT if archive_manifest_present else ARCHIVE_MANIFEST_ABSENT_LEGACY_ALLOWED
        input_files.extend([{'run_label': run_label, 'run_role': run_role, 'run_role_source': run_role_source, 'artifact': 'stage7_action_table', 'path': str(stage7_path), 'artifact_present': True, 'absence_policy': '', 'row_count': int(len(s7))}, {'run_label': run_label, 'run_role': run_role, 'run_role_source': run_role_source, 'artifact': 'stage8_multi_oracle_scores', 'path': str(stage8_path), 'artifact_present': True, 'absence_policy': '', 'row_count': int(len(s8))}, {'run_label': run_label, 'run_role': run_role, 'run_role_source': run_role_source, 'artifact': 'archive_manifest', 'path': str(archive_manifest_path), 'artifact_present': archive_manifest_present, 'absence_policy': archive_manifest_status, 'row_count': np.nan}])
    if observed_roles != expected_roles:
        raise ValueError(f'Crossed-panel role set mismatch: observed={sorted(observed_roles)}, expected={sorted(expected_roles)}')
    panel = pd.concat(panels, ignore_index=True, sort=False)
    if panel.duplicated(['backend_label', 'row_id', 'policy', 'mode']).any():
        raise ValueError('Combined crossed panel contains duplicate backend/firm/harness keys')
    return (panel, pd.DataFrame(audit_rows), pd.DataFrame(missing_rows), input_files)

def _decompose(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cell_mean_rows: list[dict[str, Any]] = []
    swing_rows: list[dict[str, Any]] = []
    for oracle in ORACLES:
        score_col = f'delta_R_score_{oracle}'
        work = panel[['row_id', 'backend_label', 'harness_cell', score_col]].copy()
        work['score'] = pd.to_numeric(work.pop(score_col), errors='raise')
        y = work['score'].to_numpy(dtype=float)
        total_sst = float(np.square(y - y.mean()).sum())
        firm_sse = _one_way_group_sse(y, work['row_id'])
        firm_fe_r2_total = _r2_from_sse(firm_sse, total_sst)
        work['within_score'] = work['score'] - work.groupby('row_id')['score'].transform('mean')
        y_within = work['within_score'].to_numpy(dtype=float)
        within_sst = float(np.square(y_within).sum())
        if not within_sst > 0:
            raise ValueError(f'Within-firm variance is zero for oracle={oracle}')
        sse_harness = _one_way_group_sse(y_within, work['harness_cell'])
        sse_backend = _one_way_group_sse(y_within, work['backend_label'])
        sse_additive = _fit_sse(y_within, [work['harness_cell'], work['backend_label']])
        interaction_cell = work['backend_label'].astype(str) + '||' + work['harness_cell'].astype(str)
        sse_interaction = _one_way_group_sse(y_within, interaction_cell)
        r2_harness = _r2_from_sse(sse_harness, within_sst)
        r2_backend = _r2_from_sse(sse_backend, within_sst)
        r2_additive = _r2_from_sse(sse_additive, within_sst)
        r2_interaction = _r2_from_sse(sse_interaction, within_sst)
        rows.append({'oracle_backend': oracle, 'n_firms': int(work['row_id'].nunique()), 'n_backends': int(work['backend_label'].nunique()), 'n_harness_cells': int(work['harness_cell'].nunique()), 'n_observations': int(len(work)), 'firm_fe_r2_total_variance': firm_fe_r2_total, 'harness_main_effect_r2_within_firm': r2_harness, 'backend_main_effect_r2_within_firm': r2_backend, 'additive_harness_backend_r2_within_firm': r2_additive, 'full_backend_by_harness_cell_r2_within_firm': r2_interaction, 'partial_harness_given_backend_r2_within_firm': float((sse_backend - sse_additive) / within_sst), 'partial_backend_main_given_harness_r2_within_firm': float((sse_harness - sse_additive) / within_sst), 'partial_backend_plus_interaction_given_harness_r2_within_firm': float((sse_harness - sse_interaction) / within_sst), 'solver_contract': SOLVER_CONTRACT, 'firm_fe_solver': 'exact_group_mean_projection', 'one_way_solver': 'exact_group_mean_projection', 'additive_solver': 'numpy_lstsq_small_design', 'interpretation_boundary': 'Descriptive variance explanation on the common crossed IC-b panel; not causal attribution. N5/N5F/N5M excluded.'})
        means = work.groupby(['backend_label', 'harness_cell'], as_index=False).agg(n_firms=('row_id', 'nunique'), mean_score=('score', 'mean'), median_score=('score', 'median'))
        means.insert(0, 'oracle_backend', oracle)
        cell_mean_rows.extend(means.to_dict(orient='records'))
        harness_swings = means.groupby('backend_label')['mean_score'].agg(lambda x: float(x.max() - x.min()))
        backend_swings = means.groupby('harness_cell')['mean_score'].agg(lambda x: float(x.max() - x.min()))
        swing_rows.append({'oracle_backend': oracle, 'min_harness_swing_backend_fixed': float(harness_swings.min()), 'max_harness_swing_backend_fixed': float(harness_swings.max()), 'mean_harness_swing_backend_fixed': float(harness_swings.mean()), 'mean_backend_swing_harness_fixed': float(backend_swings.mean()), 'max_backend_swing_harness_fixed': float(backend_swings.max()), 'n_backend_swing_cells': int(len(backend_swings))})
    return (pd.DataFrame(rows), pd.DataFrame(cell_mean_rows), pd.DataFrame(swing_rows))

def run_analysis(*, project_root: Path, run_dirs: Iterable[Path], output_dir: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(project_root)
    run_dir_list = [Path(path).resolve() for path in run_dirs]
    started = time.perf_counter()
    panel_started = time.perf_counter()
    panel, alignment, missing_cells, input_files = _build_panel(run_dirs=run_dir_list, profile=profile)
    panel_seconds = time.perf_counter() - panel_started
    decomposition_started = time.perf_counter()
    decomposition, cell_means, swings = _decompose(panel)
    decomposition_seconds = time.perf_counter() - decomposition_started
    outputs = {'panel': output_dir / 'main_harness_backend_panel.parquet', 'decomposition': output_dir / 'main_harness_backend_decomposition.csv', 'cell_means': output_dir / 'main_harness_backend_cell_means.csv', 'swing_summary': output_dir / 'main_harness_backend_swing_summary.csv', 'alignment_audit': output_dir / 'main_harness_backend_alignment_audit.csv', 'missing_cell_audit': output_dir / 'main_harness_backend_missing_cell_audit.csv', 'input_files': output_dir / 'main_harness_backend_input_files.csv', 'manifest': output_dir / 'main_harness_backend_decomposition_manifest.json'}
    panel.to_parquet(outputs['panel'], index=False)
    decomposition.to_csv(outputs['decomposition'], index=False, encoding='utf-8-sig')
    cell_means.to_csv(outputs['cell_means'], index=False, encoding='utf-8-sig')
    swings.to_csv(outputs['swing_summary'], index=False, encoding='utf-8-sig')
    alignment.to_csv(outputs['alignment_audit'], index=False, encoding='utf-8-sig')
    missing_cells.to_csv(outputs['missing_cell_audit'], index=False, encoding='utf-8-sig')
    pd.DataFrame(input_files).to_csv(outputs['input_files'], index=False, encoding='utf-8-sig')
    config = _validate_role_config(profile)
    manifest = {'schema_version': SCHEMA_VERSION, 'created_utc': _now(), 'completed_utc': _now(), 'status': 'PASS', 'panel_contract': PANEL_CONTRACT, 'solver_contract': SOLVER_CONTRACT, 'solver_diagnostics': {'dense_firm_dummy_lstsq_used': False, 'firm_fe_solver': 'exact_group_mean_projection', 'one_way_solver': 'exact_group_mean_projection', 'additive_solver': 'numpy_lstsq_small_design', 'maximum_dense_additive_columns': 64}, 'timing_seconds': {'panel_build': float(panel_seconds), 'decomposition': float(decomposition_seconds), 'total_before_manifest_write': float(time.perf_counter() - started)}, 'information_condition': _normalise_ic(config['information_condition']), 'expected_conditions': list(config['conditions']), 'expected_modes': list(config['modes']), 'run_role_backend_labels': dict(config['run_role_backend_labels']), 'run_role_resolution_contract': {'allowed_sources': list(ALLOWED_ROLE_SOURCES), 'resolver': 'credit_recourse.contracts.paper_reproduction.inspect_archived_run', 'fresh_runs_must_write_explicit_run_role': True, 'legacy_bridge_scope': 'frozen canonical main-backend archives selected by the paper profile', 'archive_manifest_optional_only_for_legacy_bridge': True, 'explicit_role_count': int(alignment['run_role_explicit'].astype(bool).sum()), 'legacy_bridge_count': int((~alignment['run_role_explicit'].astype(bool)).sum()), 'runs': alignment[['run_label', 'run_role', 'backend_label', 'run_role_source', 'run_role_explicit', 'archive_manifest_present']].to_dict(orient='records')}, 'archive_manifest_presence_contract': {'required_for_fresh_or_explicit_role_runs': True, 'allowed_missing_role_source': ROLE_SOURCE_LEGACY_BRIDGE, 'present_count': int(alignment['archive_manifest_present'].astype(bool).sum()), 'missing_legacy_count': int((~alignment['archive_manifest_present'].astype(bool)).sum()), 'missing_runs': alignment.loc[~alignment['archive_manifest_present'].astype(bool), ['run_label', 'run_role', 'run_role_source']].to_dict(orient='records')}, 'firm_count': int(panel['row_id'].nunique()), 'backend_count': int(panel['backend_label'].nunique()), 'harness_cell_count': int(panel['harness_cell'].nunique()), 'observation_count': int(len(panel)), 'expected_balanced_observation_count': int(int(config['expected_firm_count']) * len(config['conditions']) * len(config['modes']) * len(config['run_role_backend_labels'])), 'missing_observation_count': int(missing_cells['missing_rows'].sum()), 'crossed_axes': ['backend', 'policy_x_mode_harness_cell'], 'forbidden_run_families': list(FORBIDDEN_RUN_FAMILIES), 'n5_n5f_n5m_included': False, 'interpretation_boundaries': ['Technical variance explanation, not causal variance attribution.', 'Single live seed, IC-b, and three canonical post-patch backends.', 'N5, N5F, and N5M are excluded because budget is not crossed with backend.', 'Sparse failed generation cells remain missing; no imputation or silent fallback is used.'], 'outputs': {key: str(path) for key, path in outputs.items() if key != 'manifest'}}
    _write_json(outputs['manifest'], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--run-dirs', nargs='+', required=True)
    parser.add_argument('--output-dir', required=True)
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_analysis(project_root=Path(args.project_root), run_dirs=[Path(value) for value in args.run_dirs], output_dir=Path(args.output_dir))
    return 0 if result.get('status') == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
