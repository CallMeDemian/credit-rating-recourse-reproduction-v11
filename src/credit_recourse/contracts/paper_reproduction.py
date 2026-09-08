from __future__ import annotations
'Shared contract helpers for the fresh-workspace thesis reproduction profile.\n\nThis module centralizes the paper reproduction profile so that the PowerShell\nrunners, Python analysis orchestrator, and verifiers all resolve the same roles,\npaths, and output layout.  It contains no scorer, model, or fallback behavior.\n'
from dataclasses import dataclass
import copy
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable
PROFILE_RELATIVE_PATH = Path('credit_recourse') / 'configs' / 'paper_reproduction_profile.json'
ANALYSIS_PROFILES_RELATIVE_PATH = Path('credit_recourse') / 'configs' / 'paper_analysis_profiles.json'
ELIGIBILITY_CONTRACT_ID = 'eligible_llm_catalog_snapshot_v1'

class ProfileError(RuntimeError):
    """Raised when the paper reproduction profile is missing or inconsistent."""

def source_profile_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / 'src' / PROFILE_RELATIVE_PATH

def load_profile(project_root: Path, analysis_profile_name: str='current_comprehensive') -> dict[str, Any]:
    path = source_profile_path(project_root)
    if not path.exists():
        raise FileNotFoundError(f'Paper reproduction profile not found: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    validate_profile(data)
    profiles_path = Path(project_root).resolve() / 'src' / ANALYSIS_PROFILES_RELATIVE_PATH
    payload = json.loads(profiles_path.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 'paper_analysis_profiles_v1':
        raise ProfileError('Unsupported paper analysis profile schema')
    profiles = payload.get('profiles')
    if not isinstance(profiles, dict) or analysis_profile_name not in profiles:
        raise ProfileError(f'Unknown paper analysis profile: {analysis_profile_name!r}')
    selected = profiles[analysis_profile_name]
    if not isinstance(selected, dict):
        raise ProfileError('Selected paper analysis profile is not an object')
    required = {'catalog_mode', 'n5m_required', 'n5m_paper_use', 'manifest_task_order'}
    missing = sorted(required - set(selected))
    if missing:
        raise ProfileError(f'Analysis profile fields missing: {missing}')
    if selected['catalog_mode'] not in {'discover_current', 'explicit_snapshot'}:
        raise ProfileError('Unsupported catalog mode')
    if selected['n5m_paper_use'] not in {'primary', 'historical'}:
        raise ProfileError('Unsupported N5M paper use')
    if not isinstance(selected['manifest_task_order'], list):
        raise ProfileError('manifest_task_order must be a list')
    result = copy.deepcopy(data)
    n5m = result['llm']['n5_matched_budget_frontier']
    n5m['required_for_canonical_analysis'] = bool(selected['n5m_required'])
    n5m['paper_use'] = str(selected['n5m_paper_use'])
    result['_analysis_profile_name'] = analysis_profile_name
    result['_catalog_mode'] = str(selected['catalog_mode'])
    result['_manifest_task_order'] = list(selected['manifest_task_order'])
    return result

def validate_profile(profile: dict[str, Any]) -> None:
    if profile.get('schema_version') != 'paper_reproduction_profile_v1':
        raise ProfileError(f"Unsupported paper reproduction profile schema: {profile.get('schema_version')!r}")
    for section in ('workspace', 'raw_contract', 'oracle', 'rl', 'llm', 'analysis', 'reproduction', 'rl_reproducibility', 'extensions'):
        if not isinstance(profile.get(section), dict):
            raise ProfileError(f'Profile section must be an object: {section}')
    reproduction = profile['reproduction']
    if reproduction.get('schema_version') != 'thesis_reproduction_graph_v1':
        raise ProfileError('Unsupported reproduction graph schema')
    if reproduction.get('llm_input_mode') != 'precomputed_stage7_stage8_stage9_outputs':
        raise ProfileError('Canonical reproduction must consume precomputed Stage7/8/9 LLM outputs')
    if reproduction.get('llm_generation_required') is not False:
        raise ProfileError('Canonical reproduction must not require live LLM generation')
    if reproduction.get('llm_replay_required') is not False:
        raise ProfileError('Canonical reproduction must not require LLM replay')
    task_order = reproduction.get('task_order')
    expected_task_order = ['VerifyInputs', 'FreezeBaseline', 'Oracle', 'RL', 'Analysis', 'RLSevenSeed', 'AnalysisExtensions', 'MissingEvidenceAnalyses', 'ClaimEvidence', 'PaperAssets', 'ReviewPackage', 'VerifyAll']
    if task_order != expected_task_order:
        raise ProfileError(f'Reproduction task_order must be {expected_task_order!r}')
    rl_repro = profile['rl_reproducibility']
    if rl_repro.get('seven_seed_grid') != [1, 2, 3, 4, 5, 6, 7]:
        raise ProfileError('RL seven-seed grid must be exactly 1..7')
    if int(rl_repro.get('final_seed', -1)) != 2:
        raise ProfileError('RL final seed must remain 2')
    if int(rl_repro.get('expected_firm_count', 0)) != 575:
        raise ProfileError('RL reproducibility expected_firm_count must be 575')
    if rl_repro.get('encoder_sweep_mode') != 'REQUIRED_FRESH_RUN':
        raise ProfileError('Appendix D encoder sweep must be a required fresh run')
    encoder_cells = rl_repro.get('encoder_sweep_cells')
    expected_encoder_cells = [('b256_ep30_m030_s0', 256, 30, 0.3, 0), ('b256_ep49_m030_s42', 256, 49, 0.3, 42), ('b512_ep15_m015_s1', 512, 15, 0.15, 1), ('b512_ep30_m015_s1', 512, 30, 0.15, 1), ('b512_ep50_m030_s42', 512, 50, 0.3, 42)]
    observed_encoder_cells = [(str(x.get('cell_id')), int(x.get('batch_size', -1)), int(x.get('epochs', -1)), round(float(x.get('masking_ratio', -1)), 2), int(x.get('seed', -1))) for x in encoder_cells or []]
    if observed_encoder_cells != expected_encoder_cells:
        raise ProfileError(f'Appendix D encoder sweep grid mismatch: {observed_encoder_cells}')
    extensions = profile['extensions']
    e2 = extensions.get('e2') or {}
    if e2.get('run_role') != 'paper_c4r_matched_icb' or e2.get('budgets') != [0.75, None]:
        raise ProfileError('E2 contract must use paper_c4r_matched_icb at 0.75/unbounded')
    e3 = extensions.get('e3') or {}
    if e3.get('budgets') != [0.75, 1.27, 2.0, None]:
        raise ProfileError('E3 contract must use 0.75/1.27/2.00/unbounded')
    if int(e3.get('expected_firm_rows', 0)) != 4600 or int(e3.get('expected_contrasts', 0)) != 72 or int(e3.get('expected_interactions', 0)) != 54:
        raise ProfileError('E3 설정값은 4600/72/54여야 함')
    e4 = extensions.get('e4') or {}
    if int(e4.get('common_complete_case_n', 0)) != 571:
        raise ProfileError('E4 common complete-case contract must be 571')
    llm = profile['llm']
    primary = llm.get('primary')
    if not isinstance(primary, dict):
        raise ProfileError('llm.primary must be an object')
    for key in ('run_role', 'backend', 'information_conditions', 'conditions', 'modes'):
        if not primary.get(key):
            raise ProfileError(f'llm.primary missing required field: {key}')
    if primary.get('information_conditions') != ['IC-a', 'IC-b', 'IC-c']:
        raise ProfileError('Primary paper grid must contain IC-a/IC-b/IC-c in canonical order')
    supplementary = llm.get('supplementary')
    if not isinstance(supplementary, list) or len(supplementary) < 2:
        raise ProfileError('llm.supplementary must contain the two paper-support backends')
    roles = [primary['run_role']]
    for cell in supplementary:
        if not isinstance(cell, dict) or not cell.get('run_role'):
            raise ProfileError('Each supplementary cell must define run_role')
        roles.append(cell['run_role'])
    for key in ('n5', 'n5_budget_frontier', 'n5_matched_budget_frontier', 'icc_probe'):
        cell = llm.get(key)
        if not isinstance(cell, dict) or not cell.get('run_role'):
            raise ProfileError(f'llm.{key} must define run_role')
        roles.append(cell['run_role'])
    if len(roles) != len(set(roles)):
        raise ProfileError(f'LLM run roles must be unique: {roles}')
    frontier_specs = {'n5_budget_frontier': ['C6'], 'n5_matched_budget_frontier': ['C4', 'C6']}
    for frontier_key, expected_budgeted_conditions in frontier_specs.items():
        frontier = llm[frontier_key]
        if frontier.get('information_condition') != 'IC-b':
            raise ProfileError(f'llm.{frontier_key} must use IC-b')
        if frontier.get('conditions') != ['C4', 'C6'] or frontier.get('modes') != ['free_form_10d']:
            raise ProfileError(f'llm.{frontier_key} must contain C4/C6 free_form_10d only')
        if frontier.get('budgeted_conditions') != expected_budgeted_conditions:
            raise ProfileError(f'llm.{frontier_key} budgeted_conditions must be {expected_budgeted_conditions!r}')
        required_for_analysis = frontier.get('required_for_canonical_analysis')
        if not isinstance(required_for_analysis, bool):
            raise ProfileError(f'llm.{frontier_key}.required_for_canonical_analysis must be boolean')
        paper_use = frontier.get('paper_use')
        expected_paper_use = 'historical' if frontier_key == 'n5_matched_budget_frontier' else 'supplementary'
        if paper_use != expected_paper_use:
            raise ProfileError(f'llm.{frontier_key}.paper_use must be {expected_paper_use!r}; found {paper_use!r}')
        if frontier_key == 'n5_matched_budget_frontier' and required_for_analysis:
            raise ProfileError('D01 requires matched N5M to be historical and not required for canonical analysis')
        if frontier_key == 'n5_matched_budget_frontier' and frontier.get('evidence_class') != 'SUPERSEDED_FIRST_OBSERVATION':
            raise ProfileError('D01 requires the matched N5M evidence_class SUPERSEDED_FIRST_OBSERVATION')
        budgets = frontier.get('l1_budgets')
        if not isinstance(budgets, list) or len(budgets) != 4:
            raise ProfileError(f'llm.{frontier_key} must define four budget arms')
        finite = sorted((float(x) for x in budgets if x is not None))
        if finite != [0.75, 1.27, 2.0] or sum((x is None for x in budgets)) != 1:
            raise ProfileError(f'llm.{frontier_key} budgets must be 0.75/1.27/2.00/unbounded')
    analysis = profile['analysis']
    if not isinstance(analysis.get('shuffle_permutation_seeds'), str):
        raise ProfileError('analysis.shuffle_permutation_seeds must be a seed specification string')
    if analysis.get('shuffle_within') != ['none', 'industry', 'rating_band']:
        raise ProfileError('analysis.shuffle_within must contain none/industry/rating_band')
    tasks = analysis.get('tasks')
    if not isinstance(tasks, list) or not tasks:
        raise ProfileError('analysis.tasks must be a non-empty list')
    main_decomposition = analysis.get('main_harness_backend_decomposition')
    if not isinstance(main_decomposition, dict):
        raise ProfileError('analysis.main_harness_backend_decomposition must be an object')
    role_labels = main_decomposition.get('run_role_backend_labels')
    expected_roles = {primary['run_role'], *(cell['run_role'] for cell in supplementary)}
    if not isinstance(role_labels, dict) or set(role_labels) != expected_roles:
        raise ProfileError('main harness-backend decomposition roles must exactly match the primary IC-b and two supplementary roles')
    if main_decomposition.get('information_condition') != 'IC-b':
        raise ProfileError('main harness-backend decomposition must use IC-b')
    if main_decomposition.get('conditions') != primary.get('conditions'):
        raise ProfileError('main harness-backend decomposition conditions must match the primary full grid')
    if main_decomposition.get('modes') != primary.get('modes'):
        raise ProfileError('main harness-backend decomposition modes must match the primary full grid')
    if int(main_decomposition.get('expected_firm_count', 0)) != 575:
        raise ProfileError('main harness-backend decomposition expected_firm_count must be 575')
    if sorted(main_decomposition.get('excluded_run_families') or []) != ['N5', 'N5F', 'N5M']:
        raise ProfileError('main harness-backend decomposition must exclude N5/N5F/N5M')
    required_tasks = {'verify', 'b2-gap', 'test3', 'structural-slice', 'postfreeze-ablation', 'postfreeze-holm', 'main-harness-backend-decomposition', 'reference-quality', 'n5-holm', 'n5-frontier-holm', 'signflip', 'frontier', 'winrate', 'icc-probe', 'final-contract', 'analysis-extensions', 'thesis-registry', 'master-verification'}
    missing = sorted(required_tasks - set(tasks))
    if missing:
        raise ProfileError(f'analysis.tasks missing canonical paper tasks: {missing}')
    forbidden_historical = {'n5-matched-frontier-holm', 'n5m-posthoc', 'n5m-adaptive-selection'} & set(tasks)
    if forbidden_historical:
        raise ProfileError(f'D01 historical N5M tasks must not be canonical analysis tasks: {sorted(forbidden_historical)}')
    historical = analysis.get('historical_optional_tasks') or []
    if set(historical) != {'n5-matched-frontier-holm', 'n5m-posthoc', 'n5m-adaptive-selection', 'legacy-paper-assets'}:
        raise ProfileError('historical_optional_tasks must contain the three E1/N5M lineage tasks and legacy-paper-assets exactly')

def resolve_workspace_path(project_root: Path, relative_path: str) -> Path:
    return Path(project_root).resolve() / Path(relative_path)

def analysis_output_dir(project_root: Path, profile: dict[str, Any] | None=None) -> Path:
    profile = profile or load_profile(project_root)
    return resolve_workspace_path(project_root, profile['workspace']['analysis_root'])

def layout_paths(output_dir: Path, profile: dict[str, Any]) -> dict[str, Path]:
    layout = profile['analysis']['output_layout']
    return {key: Path(output_dir) / value for key, value in layout.items()}

@dataclass(frozen=True)
class ArchivedRunMetadata:
    run_dir: Path
    run_label: str
    run_role: str | None
    information_condition: str | None
    backend_id: str | None
    backend_model: str | None
    backend_provider: str | None
    conditions: tuple[str, ...]
    modes: tuple[str, ...]
    seed: int | None
    seed_source: str | None
    reference_draw_seed: int | None
    reference_draw_seed_source: str | None
    candidate_library_quantile: int | None
    candidate_library_quantile_source: str | None
    row_count: int | None
    request_count: int | None
    freeform_l1_budget: float | None
    freeform_l1_budget_source: str | None
    has_stage7: bool
    has_stage8: bool
    has_stage9: bool
    has_probe: bool

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding='utf-8-sig'))
    except Exception as exc:
        return {'_read_error': repr(exc)}
    return obj if isinstance(obj, dict) else {}

def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None

def _first_int(candidates: Iterable[tuple[str, Any]]) -> tuple[int | None, str | None]:
    for source, value in candidates:
        parsed = _int_or_none(value)
        if parsed is not None:
            return (parsed, source)
    return (None, None)

def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except Exception:
        return None

def _first_float(candidates: Iterable[tuple[str, Any]]) -> tuple[float | None, str | None]:
    for source, value in candidates:
        parsed = _float_or_none(value)
        if parsed is not None:
            return (parsed, source)
    return (None, None)

def _legacy_seed_from_label(run_label: str) -> tuple[int | None, str | None]:
    """Resolve the historical run seed only from an explicit ``seedN`` token.

    July 2026 frozen Stage7 metadata recorded ``reference_draw_seed`` but did
    not persist the runner-level ``seed`` field.  Their immutable run labels
    contain an explicit ``seed1`` token.  This is a narrow compatibility bridge,
    not a generic filename heuristic: no token, multiple conflicting tokens, or
    malformed values produce ``None`` and therefore still fail the paper
    contract.
    """
    matches = [int(x) for x in re.findall('(?:^|[_-])seed(\\d+)(?=$|[_-])', run_label, flags=re.IGNORECASE)]
    if not matches:
        return (None, None)
    unique = sorted(set(matches))
    if len(unique) != 1:
        return (None, None)
    return (unique[0], 'legacy_run_label.explicit_seed_token')

def inspect_archived_run(run_dir: Path) -> ArchivedRunMetadata:
    run_dir = Path(run_dir).resolve()
    stage7 = run_dir / 'stage7_llm_action_generation'
    stage8 = run_dir / 'stage8_llm_multi_oracle_eval'
    stage9 = run_dir / 'stage9_llm_rl_comparison'
    probe = run_dir / 'stage7_icc_probe'
    meta = _read_json(stage7 / 'metadata.json')
    prompt = _read_json(stage7 / 'llm_stage7_prompt_manifest.json')
    archive = _read_json(run_dir / 'archive_manifest.json')
    probe_summary = _read_json(probe / 'icc_probe_summary.json')
    backend = prompt.get('backend') if isinstance(prompt.get('backend'), dict) else {}
    probe_backend = probe_summary.get('backend') if isinstance(probe_summary.get('backend'), dict) else {}
    provider_options = backend.get('provider_options') if isinstance(backend.get('provider_options'), dict) else {}
    backend_id = meta.get('backend_id') or probe_backend.get('backend_id')
    backend_model = backend.get('model') or probe_backend.get('model')
    run_role = archive.get('run_role') or meta.get('run_role') or probe_summary.get('run_role')
    archive_extra = archive.get('extra') if isinstance(archive.get('extra'), dict) else {}
    reproduction_contract = archive_extra.get('reproduction_contract') if isinstance(archive_extra.get('reproduction_contract'), dict) else {}
    seed, seed_source = _first_int((('archive_manifest.extra.reproduction_contract.seed', reproduction_contract.get('seed')), ('stage7.metadata.seed', meta.get('seed')), ('stage7.prompt_manifest.seed', prompt.get('seed'))))
    if seed is None:
        seed, seed_source = _legacy_seed_from_label(run_dir.name)
    reference_draw_seed, reference_draw_seed_source = _first_int((('archive_manifest.extra.reproduction_contract.reference_draw_seed', reproduction_contract.get('reference_draw_seed')), ('stage7.metadata.reference_draw_seed', meta.get('reference_draw_seed')), ('stage7.prompt_manifest.reference_draw_seed', prompt.get('reference_draw_seed'))))
    candidate_library_quantile, candidate_library_quantile_source = _first_int((('archive_manifest.extra.reproduction_contract.candidate_library_quantile', reproduction_contract.get('candidate_library_quantile')), ('stage7.metadata.candidate_library_quantile', meta.get('candidate_library_quantile')), ('stage7.prompt_manifest.candidate_library_quantile', prompt.get('candidate_library_quantile'))))
    budget_contract = meta.get('action_budget_contract')
    if not isinstance(budget_contract, dict):
        budget_contract = {}
    prompt_budget_contract = prompt.get('action_budget_contract')
    if not isinstance(prompt_budget_contract, dict):
        prompt_budget_contract = {}
    budget, budget_source = _first_float((('archive_manifest.extra.reproduction_contract.freeform_l1_budget', reproduction_contract.get('freeform_l1_budget')), ('stage7.metadata.action_budget_contract.l1_budget', budget_contract.get('l1_budget')), ('stage7.prompt_manifest.action_budget_contract.l1_budget', prompt_budget_contract.get('l1_budget')), ('stage7.metadata.freeform_l1_budget', meta.get('freeform_l1_budget')), ('stage7.prompt_manifest.freeform_l1_budget', prompt.get('freeform_l1_budget'))))
    information_condition = meta.get('information_condition') or probe_summary.get('information_condition')
    conditions = tuple((str(x) for x in meta.get('conditions') or []))
    modes = tuple((str(x) for x in meta.get('modes') or []))
    if not run_role:
        name = run_dir.name.lower()
        model_low = str(backend_model or backend_id or '').lower()
        if probe_summary:
            run_role = 'paper_icc_probe'
        elif budget is not None and abs(budget - 1.27) <= 1e-09 and (set(conditions) == {'C4', 'C6'}):
            run_role = 'paper_n5_l1_1p27'
        elif 'gpt-5.4-mini' in model_low or 'gpt54' in name:
            if 'n5' not in name:
                run_role = 'paper_primary_gpt54'
        elif 'gpt-4.1-mini' in model_low or 'gpt41' in name:
            run_role = 'paper_supplementary_gpt41'
        elif 'haiku' in model_low or 'haiku' in name:
            run_role = 'paper_supplementary_haiku45'
    return ArchivedRunMetadata(run_dir=run_dir, run_label=run_dir.name, run_role=str(run_role) if run_role else None, information_condition=information_condition, backend_id=str(backend_id) if backend_id else None, backend_model=str(backend_model) if backend_model else None, backend_provider=str(provider_options.get('provider')) if provider_options.get('provider') else None, conditions=conditions, modes=modes, seed=seed, seed_source=seed_source, reference_draw_seed=reference_draw_seed, reference_draw_seed_source=reference_draw_seed_source, candidate_library_quantile=candidate_library_quantile, candidate_library_quantile_source=candidate_library_quantile_source, row_count=_int_or_none(meta.get('row_count')), request_count=_int_or_none(meta.get('request_count')), freeform_l1_budget=budget, freeform_l1_budget_source=budget_source, has_stage7=(stage7 / 'llm_stage7_action_table.parquet').exists(), has_stage8=(stage8 / 'metadata.json').exists(), has_stage9=(stage9 / 'metadata.json').exists(), has_probe=(probe / 'icc_probe_summary.json').exists())

def eligible_run_labels_from_catalog(catalog_path: Path) -> tuple[str, ...]:
    catalog_path = Path(catalog_path).resolve()
    if not catalog_path.is_file():
        raise FileNotFoundError(f'Eligible LLM catalog not found: {catalog_path}')
    with catalog_path.open('r', encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or 'run_label' not in reader.fieldnames:
            raise ProfileError('Eligible LLM catalog lacks run_label')
        labels = [str(row.get('run_label') or '').strip() for row in reader]
    if not labels or any((not label for label in labels)):
        raise ProfileError('Eligible LLM catalog contains an empty run label')
    if len(labels) != len(set(labels)):
        raise ProfileError('Eligible LLM catalog contains duplicate run labels')
    for label in labels:
        if Path(label).name != label or '/' in label or '\\' in label:
            raise ProfileError(f'Unsafe eligible LLM run label: {label!r}')
    return tuple(labels)

def eligible_catalog_contract(catalog_path: Path, *, role_relative_path: str) -> dict[str, Any]:
    """Return the immutable identity of an explicit eligibility snapshot."""
    path = Path(catalog_path).resolve()
    labels = eligible_run_labels_from_catalog(path)
    return {'eligibility_contract_id': ELIGIBILITY_CONTRACT_ID, 'eligible_catalog_path': str(path), 'eligible_catalog_role_relative_path': str(role_relative_path), 'eligible_catalog_row_count': len(labels), 'eligible_catalog_unique_count': len(set(labels)), 'eligible_run_labels': list(labels)}

def discover_archived_runs(project_root: Path, profile: dict[str, Any] | None=None, *, catalog_snapshot: Path | None=None) -> list[ArchivedRunMetadata]:
    profile = profile or load_profile(project_root)
    root = resolve_workspace_path(project_root, profile['workspace']['llm_runs_root'])
    if not root.exists():
        return []
    if catalog_snapshot is None:
        if profile.get('_catalog_mode') == 'explicit_snapshot':
            raise ProfileError('Historical analysis profile requires an eligible-run catalog')
        candidates = [p for p in sorted(root.iterdir()) if p.is_dir()]
    else:
        labels = eligible_run_labels_from_catalog(catalog_snapshot)
        candidates = []
        for label in labels:
            candidate = (root / label).resolve()
            if candidate.parent != root.resolve():
                raise ProfileError(f'Eligible run escapes llm_runs root: {label!r}')
            if not candidate.is_dir():
                raise FileNotFoundError(f'Eligible LLM run directory is missing: {candidate}')
            candidates.append(candidate)
    return [inspect_archived_run(path) for path in candidates]

def select_exact_role(records: Iterable[ArchivedRunMetadata], *, run_role: str, information_condition: str | None=None, require_probe: bool=False) -> ArchivedRunMetadata:
    matches = [r for r in records if r.run_role == run_role and (information_condition is None or r.information_condition == information_condition) and (not require_probe or r.has_probe)]
    if len(matches) != 1:
        raise ProfileError(f'Expected exactly one archived run for role={run_role!r}, IC={information_condition!r}, require_probe={require_probe}; found {len(matches)}: {[m.run_label for m in matches]}')
    return matches[0]

def select_budget_frontier_role(records: Iterable[ArchivedRunMetadata], *, run_role: str, information_condition: str, expected_budgets: Iterable[float | None], expected_conditions: Iterable[str]=('C4', 'C6'), tolerance: float=1e-09, allow_absent: bool=False) -> list[ArchivedRunMetadata]:
    """Select an exact generation-time budget frontier with within-arm controls.

    The selector requires one and only one archived run per expected budget arm.
    ``None`` is the explicit unbounded arm; it is never imputed from a finite
    default. Conditions and modes are checked here so the analysis cannot ingest
    a superficially similar but semantically different run. When ``allow_absent``
    is true, zero matching runs is an explicit optional-not-yet-run state; any
    partial or malformed frontier still hard-fails.
    """
    expected = list(expected_budgets)
    expected_condition_set = {str(value) for value in expected_conditions}
    if not expected_condition_set:
        raise ProfileError('expected_conditions must not be empty')
    matches = [record for record in records if record.run_role == run_role and record.information_condition == information_condition]
    if not matches and allow_absent:
        return []
    if len(matches) != len(expected):
        raise ProfileError(f'Expected {len(expected)} archived frontier runs for role={run_role!r}, IC={information_condition!r}; found {len(matches)}: {[x.run_label for x in matches]}')

    def equal(left: float | None, right: float | None) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return abs(float(left) - float(right)) <= tolerance
    selected: list[ArchivedRunMetadata] = []
    for budget in expected:
        arm = [record for record in matches if equal(record.freeform_l1_budget, budget)]
        if len(arm) != 1:
            raise ProfileError(f'Expected exactly one frontier arm for budget={budget!r}; found {[(x.run_label, x.freeform_l1_budget) for x in arm]}')
        record = arm[0]
        if set(record.conditions) != expected_condition_set or set(record.modes) != {'free_form_10d'}:
            raise ProfileError(f'Frontier 실행 설정이 다름: {record.run_label}: expected_conditions={sorted(expected_condition_set)}, conditions={record.conditions}, modes={record.modes}')
        if not (record.has_stage7 and record.has_stage8 and record.has_stage9):
            raise ProfileError(f'Frontier run is incomplete: {record.run_label}')
        selected.append(record)
    return selected
