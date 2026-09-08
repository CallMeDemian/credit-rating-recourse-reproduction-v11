from __future__ import annotations
'LLM free-form action-budget ablations and evaluator-only null baselines.\n\nThis module is deliberately post-hoc and evaluator-only. It does not call any\nLLM API and does not retrain RL. It reads a frozen Stage 7 LLM action table,\nconstructs counterfactual action-table variants, then scores those variants\nthrough the exact Stage 6/8 simulator + Oracle substrate.\n\nPrimary use cases for the thesis:\n  1. L1 budget rescale: test whether free-form superiority is only action size.\n  2. Nearest-candidate projection: test whether the benefit disappears when\n     free-form outputs are forced back into the 11-candidate library.\n  3. Free-form null baselines: test whether high free-form scores are explained\n     by a generic all-improve vector rather than firm-specific matching.\n  4. Paired inference against C3 or the original free-form C6 rows.\n\nAll outputs are written outside canonical stage8/stage9 directories by default\nso this cannot accidentally overwrite a frozen final run.\n'
import argparse
import json
import math
import os
import re
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import yaml
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
from credit_recourse.contracts.score_tie import score_nonzero_count, score_positive_fraction, score_tie_contract_metadata, score_wilcoxon_values
from credit_recourse.analysis.v10_stage_paths import final_root, stage_dir
from credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline import load_registry, resolve_backend_artifact, prepare_stage6_history_lookup, score_alpha, score_beta_ordered_logit_params, score_gamma_model, simulate_policy_states
from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.pipeline import _load_stage6_noop_scores, _load_stage6_simulator_identity
from credit_recourse.rl.common.actions import load_action_space, resolve_candidate_library_path
from credit_recourse.rl.common.io import read_parquet_required, write_json
SUPPORTED_VARIANTS = {'l1_rescale', 'nearest_candidate_projection', 'global_mean_vector_null', 'row_shuffle_vector_null', 'sign_preserving_random_vector', 'sign_flip_mean_vector'}

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json_dict(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
    except Exception as e:
        raise ValueError(f'Could not parse JSON metadata: {path}: {e}') from e
    if not isinstance(obj, dict):
        raise ValueError(f'JSON metadata must be an object: {path}')
    return obj

def _resolve_recorded_path(project_root: Path, raw_path: str | Path, *, field_name: str, raw_root: Path | None=None) -> Path:
    """Resolve a path recorded in Stage7 metadata against the current repo root.

    Archived metadata may contain absolute Windows paths from the machine where the
    run was produced.  This resolver first maps the suffix from ``data/...`` under
    the supplied project root, then treats the recorded absolute path as a final
    compatibility fallback.  Failure to resolve the artifact is a hard error.
    """
    if raw_path is None or str(raw_path).strip() == '':
        raise ValueError(f'Missing required Stage7 metadata field: {field_name}')
    project_root = Path(project_root).resolve()
    s_raw = str(raw_path).strip()
    candidates: list[Path] = []
    p = Path(s_raw)
    normalized = s_raw.replace('\\', '/')
    if raw_root is not None and 'data/raw/' in normalized:
        suffix = normalized.split('data/raw/', 1)[1]
        candidates.append(Path(raw_root).resolve() / Path(*suffix.split('/')))
    for marker in ('data/final_freeze/', 'data/'):
        if marker in normalized:
            suffix = normalized[normalized.index(marker):]
            candidates.append(project_root / Path(*suffix.split('/')))
            break
    # Historical metadata records the producer machine's absolute path.  In a
    # reviewer-facing copy, the same logical artifact under this project must
    # be preferred; the recorded path is only a final compatibility fallback.
    candidates.append(p if p.is_absolute() else project_root / p)
    seen: set[str] = set()
    for cand in candidates:
        try:
            resolved = cand.resolve()
        except Exception:
            resolved = cand
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if resolved.exists():
            return resolved
    raise FileNotFoundError({'message': f'Could not resolve recorded path for {field_name}', 'raw_path': s_raw, 'attempted': [str(c) for c in candidates]})

def _stage7_metadata_paths(stage7_action_table: Path) -> list[Path]:
    parent = Path(stage7_action_table).resolve().parent
    return [parent / 'metadata.json', parent / 'llm_stage7_prompt_manifest.json']

def _load_stage7_candidate_library_provenance(project_root: Path, stage7_action_table: Path) -> dict:
    """Load and validate candidate-library provenance from archived Stage7 metadata.

    The ablation table is a post-hoc transformation of a specific Stage7 run.
    Nearest-candidate projection must therefore use the same candidate library as
    that run, not the mutable active base library.  This function hard-fails when
    Stage7 provenance is absent or inconsistent.
    """
    project_root = Path(project_root).resolve()
    stage7_action_table = Path(stage7_action_table).resolve()
    metadata_payloads: list[tuple[Path, dict]] = []
    for path in _stage7_metadata_paths(stage7_action_table):
        if path.exists():
            payload = _read_json_dict(path)
            metadata_payloads.append((path, payload))
    if not metadata_payloads:
        raise FileNotFoundError({'message': 'Missing Stage7 metadata next to action table; cannot determine candidate-library provenance.', 'stage7_action_table': str(stage7_action_table), 'searched': [str(p) for p in _stage7_metadata_paths(stage7_action_table)]})
    merged: dict = {}
    source_paths: list[str] = []
    for path, payload in metadata_payloads:
        source_paths.append(str(path))
        for key, value in payload.items():
            if key not in merged or merged.get(key) in (None, '', []):
                merged[key] = value
    candidate_path_raw = merged.get('candidate_library_path')
    quantile_raw = merged.get('candidate_library_quantile')
    if candidate_path_raw:
        candidate_path = _resolve_recorded_path(project_root, candidate_path_raw, field_name='candidate_library_path')
    elif quantile_raw not in (None, ''):
        candidate_path = resolve_candidate_library_path(project_root, magnitude_quantile=int(quantile_raw))
    else:
        raise ValueError({'message': 'Stage7 metadata lacks candidate_library_path and candidate_library_quantile.', 'stage7_action_table': str(stage7_action_table), 'metadata_sources': source_paths, 'available_keys': sorted(merged.keys())})
    base_path_raw = merged.get('base_candidate_library_path')
    base_candidate_path = _resolve_recorded_path(project_root, base_path_raw, field_name='base_candidate_library_path') if base_path_raw else project_root / 'data/final_freeze/configs/final_candidate_library.yaml'
    action_contract_raw = merged.get('final_action_contract_path')
    action_contract_path = _resolve_recorded_path(project_root, action_contract_raw, field_name='final_action_contract_path') if action_contract_raw else project_root / 'data/final_freeze/configs/final_action_contract.yaml'
    quantile = None
    if quantile_raw not in (None, ''):
        quantile = int(quantile_raw)
    return {'candidate_library_path': str(candidate_path), 'candidate_library_quantile': quantile, 'candidate_action_values_source': merged.get('candidate_action_values_source'), 'base_candidate_library_path': str(base_candidate_path.resolve()), 'final_action_contract_path': str(action_contract_path.resolve()), 'candidate_library_provenance_metadata_paths': source_paths, 'candidate_library_provenance_validation': 'PASS'}

def _load_action_space_for_stage7(project_root: Path, stage7_action_table: Path) -> tuple[object, dict]:
    provenance = _load_stage7_candidate_library_provenance(project_root, stage7_action_table)
    space = load_action_space(project_root, candidate_library_path=provenance['candidate_library_path'])
    return (space, provenance)

def _parse_csv_arg(x: str | Iterable[str]) -> list[str]:
    if isinstance(x, str):
        return [p.strip() for p in x.split(',') if p.strip()]
    return [str(p).strip() for p in x if str(p).strip()]

def _parse_seed_spec(spec: str | Iterable[int] | None) -> list[int]:
    """Parse comma-separated seeds and inclusive ``start:end`` ranges."""
    if spec is None:
        return []
    if not isinstance(spec, str):
        seeds = [int(x) for x in spec]
    else:
        seeds: list[int] = []
        for token in [x.strip() for x in spec.split(',') if x.strip()]:
            if ':' in token:
                parts = token.split(':')
                if len(parts) not in (2, 3):
                    raise ValueError(f'Invalid shuffle seed range: {token!r}')
                start, end = (int(parts[0]), int(parts[1]))
                step = int(parts[2]) if len(parts) == 3 else 1 if end >= start else -1
                if step == 0 or (end - start) * step < 0:
                    raise ValueError(f'Invalid shuffle seed range step: {token!r}')
                stop = end + (1 if step > 0 else -1)
                seeds.extend(range(start, stop, step))
            else:
                seeds.append(int(token))
    if not seeds:
        raise ValueError('shuffle seed specification resolved to an empty list')
    if len(seeds) != len(set(seeds)):
        raise ValueError('shuffle seed specification contains duplicate seeds')
    return seeds
SHUFFLE_WITHIN_CHOICES = ('none', 'industry', 'rating_band')
SHUFFLE_STRATUM_COLUMNS = {'industry': ('sector_7', 'industry_class', 'sector', 'industry', '산업명', 'industry_name', '산업코드', '표준산업코드', 'industry_code'), 'rating_band': ('grade_base_10', 'rating_num_10', 'grade_base_7', 'rating_num_7', 'grade_base', 'rating_num')}
SHUFFLE_UNKNOWN_MARKERS = {'', 'UNKNOWN', 'UNKNOWN_MARKET', 'UNK', 'NONE', 'NULL', 'NAN', 'NA', 'N/A', '<NA>', '미상', 'KOSPI', 'KOSDAQ', 'KONEX'}
INDUSTRY_UNKNOWN_LABEL = '__UNKNOWN_INDUSTRY__'
MIN_INDUSTRY_STRATUM_KNOWN_FRACTION = 0.9

def _normalise_row_ids(frame: pd.DataFrame, *, label: str) -> pd.Series:
    if 'row_id' not in frame.columns:
        return pd.Series(np.arange(len(frame), dtype=int), index=frame.index, name='row_id')
    values = pd.to_numeric(frame['row_id'], errors='coerce')
    if values.isna().any() or not np.allclose(values.to_numpy(dtype=float), np.round(values.to_numpy(dtype=float))):
        raise ValueError(f'{label} row_id must be finite integers')
    return values.astype(int)

def _normalise_firm_id(value: object) -> str:
    if pd.isna(value):
        return ''
    text = str(value).strip()
    if text.endswith('.0') and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text

def _clean_stratum_text(values: pd.Series) -> pd.Series:
    return values.astype('string').str.strip()

def _valid_stratum_mask(values: pd.Series) -> pd.Series:
    clean = _clean_stratum_text(values)
    return (clean.notna() & ~clean.str.upper().isin(SHUFFLE_UNKNOWN_MARKERS)).fillna(False)

@lru_cache(maxsize=1)
def _sector7_mapping_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / 'configs' / 'oracle_components' / 'stage00_03' / 'stage_config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'Missing canonical sector mapping config: {config_path}')
    payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    mapping = payload.get('sector_7_mapping')
    if not isinstance(mapping, dict) or not isinstance(mapping.get('priority_order'), list):
        raise ValueError(f'Invalid canonical sector_7_mapping contract: {config_path}')
    for pos, rule in enumerate(mapping['priority_order']):
        if not isinstance(rule, dict) or not isinstance(rule.get('sector_name'), str) or (not isinstance(rule.get('keywords'), list)):
            raise ValueError(f'Invalid sector_7 mapping rule at position={pos}: {rule!r}')
    return mapping

def _sector7_from_industry_name(value: object) -> str:
    if pd.isna(value):
        return INDUSTRY_UNKNOWN_LABEL
    text = str(value).strip()
    if not text or text.upper() in SHUFFLE_UNKNOWN_MARKERS:
        return INDUSTRY_UNKNOWN_LABEL
    mapping = _sector7_mapping_config()
    for rule in mapping['priority_order']:
        if any((str(keyword) in text for keyword in rule['keywords'])):
            return str(rule['sector_name'])
    default = str(mapping.get('default_unknown', '미분류')).strip()
    return INDUSTRY_UNKNOWN_LABEL if not default or default.upper() in SHUFFLE_UNKNOWN_MARKERS else default

def _industry_code_group(value: object) -> str:
    if pd.isna(value):
        return INDUSTRY_UNKNOWN_LABEL
    text = re.sub('\\D', '', str(value))
    if not text:
        return INDUSTRY_UNKNOWN_LABEL
    return f'industry_code2:{text[:2]}'

def _transform_stratum_values(values: pd.Series, *, shuffle_within: str, column: str) -> tuple[pd.Series, str]:
    if shuffle_within == 'industry' and column in {'산업명', 'industry_name'}:
        return (values.map(_sector7_from_industry_name).astype('string'), 'sector_7_from_industry_name')
    if shuffle_within == 'industry' and column in {'산업코드', '표준산업코드', 'industry_code'}:
        return (values.map(_industry_code_group).astype('string'), 'industry_code_two_digit_group')
    clean = _clean_stratum_text(values)
    if shuffle_within == 'industry':
        valid = _valid_stratum_mask(clean)
        clean = clean.where(valid, INDUSTRY_UNKNOWN_LABEL)
    else:
        clean = clean.where(clean.notna() & clean.ne(''), '__UNKNOWN_RATING_BAND__')
    return (clean.astype('string'), 'direct_column')

def _stratum_profile(values: pd.Series, *, shuffle_within: str) -> dict:
    clean = _clean_stratum_text(values)
    if shuffle_within == 'industry':
        known = clean.ne(INDUSTRY_UNKNOWN_LABEL) & _valid_stratum_mask(clean)
    else:
        known = clean.ne('__UNKNOWN_RATING_BAND__') & clean.notna() & clean.ne('')
    known_count = int(known.sum())
    unique_known = int(clean.loc[known].nunique(dropna=False))
    counts = clean.fillna(INDUSTRY_UNKNOWN_LABEL if shuffle_within == 'industry' else '__UNKNOWN_RATING_BAND__').value_counts(dropna=False)
    return {'rows': int(len(clean)), 'known_count': known_count, 'unknown_count': int(len(clean) - known_count), 'known_fraction': float(known_count / max(len(clean), 1)), 'unique_known': unique_known, 'stratum_count': int(clean.nunique(dropna=False)), 'value_counts_top20': {str(k): int(v) for k, v in counts.head(20).items()}}

def _profile_aligned_strata(action_table: pd.DataFrame, values: pd.Series, *, shuffle_within: str) -> dict:
    if len(values) != len(action_table):
        raise ValueError(f'Stratum profile alignment length mismatch: values={len(values)}, action_rows={len(action_table)}')
    row_ids = _normalise_row_ids(action_table, label='Stage7 action table')
    aligned = pd.DataFrame({'row_id': row_ids.to_numpy(), 'stratum': values.astype('string').to_numpy()})
    consistency = aligned.groupby('row_id', dropna=False)['stratum'].nunique(dropna=False)
    bad = consistency[consistency != 1]
    if not bad.empty:
        raise ValueError(f'Resolved shuffle strata vary within row_id={bad.index.tolist()[:20]}')
    firm_level = aligned.drop_duplicates('row_id', keep='first')
    profile = _stratum_profile(firm_level['stratum'], shuffle_within=shuffle_within)
    profile['profile_unit'] = 'unique_row_id'
    profile['action_row_count'] = int(len(action_table))
    return profile

def _align_source_values_by_row_id(action_table: pd.DataFrame, source: pd.DataFrame, *, source_label: str, column: str) -> pd.Series:
    if 'row_id' not in source.columns:
        raise ValueError(f'Conditional shuffle source lacks row_id and cannot be aligned: {source_label}')
    action_ids = _normalise_row_ids(action_table, label='Stage7 action table')
    source_ids = _normalise_row_ids(source, label=source_label)
    lookup = pd.DataFrame({'row_id': source_ids.to_numpy(), 'stratum': source[column].to_numpy()})
    if lookup['row_id'].duplicated().any():
        consistency = lookup.groupby('row_id', dropna=False)['stratum'].nunique(dropna=False)
        bad = consistency[consistency != 1]
        if not bad.empty:
            raise ValueError(f'Conditional shuffle source has inconsistent strata for row_id={bad.index.tolist()[:20]}')
        lookup = lookup.drop_duplicates('row_id', keep='first')
    mapped = action_ids.map(lookup.set_index('row_id')['stratum'])
    if mapped.isna().any():
        missing = sorted(set(action_ids.loc[mapped.isna()].astype(int).tolist()))[:20]
        raise ValueError(f'Conditional shuffle could not align Stage7 row_id values to {source_label}: {missing}')
    return pd.Series(mapped.to_numpy(), index=action_table.index)

def _materialize_stage7_row_id_contract(frame: pd.DataFrame, *, source_label: str, allow_stable_row_order_index: bool) -> tuple[pd.DataFrame, str]:
    """Materialize the exact Stage7 row-id contract for a serving-panel source.

    Stage7 assigns ``row_id = reset_index(drop=True).index`` when the frozen
    Stage2 serving panel does not already contain a row_id column.  Conditional
    analyses must mirror that rule exactly; requiring a physical row_id column
    rejects valid final-freeze panels and breaks row_id→firm_id recovery.
    """
    out = frame.reset_index(drop=True).copy()
    if 'row_id' in out.columns:
        out['row_id'] = _normalise_row_ids(out, label=source_label).to_numpy()
        return (out, 'physical_row_id_column')
    if not allow_stable_row_order_index:
        raise ValueError(f'Identity/stratum source lacks row_id and is not an approved Stage7 serving-panel source: {source_label}')
    out['row_id'] = np.arange(len(out), dtype=int)
    return (out, 'stage7_stable_row_order_index')

def _candidate_stratum_sources(project_root: Path, action_table: pd.DataFrame, *, artifact_root: Path | str | None=None) -> list[tuple[str, pd.DataFrame, Path | None]]:
    """Return ordered stratum sources under an explicitly resolved artifact root.

    ``artifact_root`` is optional for normal production calls because the active
    run context exports ``CREDIT_RECOURSE_ARTIFACT_ROOT``. Synthetic verifiers
    pass it explicitly so they never inherit or write through a frozen reference
    compatibility view.
    """
    sources: list[tuple[str, pd.DataFrame, Path | None]] = [('stage7_action_table', action_table, None)]
    candidate_paths = [(stage_dir(project_root, 'stage2', artifact_root=artifact_root) / 'phase_eval_candidate.parquet', True), (stage_dir(project_root, 'stage2', artifact_root=artifact_root) / 'input_splits' / 'phase_eval.parquet', True), (stage_dir(project_root, 'stage1_inputs', artifact_root=artifact_root) / 'alpha_vanilla_input_candidate.parquet', False)]
    seen: set[str] = set()
    for path, allow_index_contract in candidate_paths:
        key = str(path.resolve())
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        frame = read_parquet_required(path)
        try:
            frame, row_id_derivation = _materialize_stage7_row_id_contract(frame, source_label=str(path), allow_stable_row_order_index=allow_index_contract)
        except ValueError:
            continue
        sources.append((f'{path}#row_id={row_id_derivation}', frame, path))
    return sources

def _read_excel_with_fallback(path: Path) -> pd.DataFrame:
    errors: list[str] = []
    for engine in ('calamine', 'openpyxl', None):
        try:
            if engine is None:
                return pd.read_excel(path)
            return pd.read_excel(path, engine=engine)
        except Exception as exc:
            errors.append(f"{engine or 'default'}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"Failed to read raw general-information workbook {path}: {' | '.join(errors)}")
RAW_GENERAL_MARKET_ORDER = ('kospi', 'kosdaq', 'konex')
RAW_GENERAL_REQUIRED_MARKETS = {'kospi', 'kosdaq'}
RAW_GENERAL_CANONICAL_FILENAMES = {'kospi': '코스피_전업종_폐지사 포함_일반사항.xlsx', 'kosdaq': '코스닥_전업종_폐지사 포함_일반사항.xlsx', 'konex': '코넥스_전업종_일반사항.xlsx'}

def _raw_general_paths_config_candidates(project_root: Path) -> list[Path]:
    package_root = Path(__file__).resolve().parents[1]
    return [project_root / 'data' / 'final_freeze' / 'configs' / 'oracle_components' / 'stage00_03' / 'paths.final_freeze.generated.yaml', project_root / 'data' / 'final_freeze' / 'configs' / 'oracle_components' / 'stage00_03' / 'paths.yaml', package_root / 'final_freeze_configs' / 'configs' / 'oracle_components' / 'stage00_03' / 'paths.final_freeze.generated.yaml', package_root / 'final_freeze_configs' / 'configs' / 'oracle_components' / 'stage00_03' / 'paths.yaml']

def _resolve_raw_general_information_sources(project_root: Path, raw_root: Path) -> tuple[list[tuple[str, Path]], dict]:
    """Resolve the exact Stage00-03 general-information input contract.

    Stage00-03 does not recursively merge every workbook whose name contains
    ``일반사항``.  It reads one configured workbook per market in deterministic
    KOSPI -> KOSDAQ -> KONEX order.  The post-freeze recovery path must preserve
    that source contract so stale snapshots or duplicate raw roots cannot create
    artificial firm-sector conflicts.
    """
    project_root = Path(project_root).resolve()
    config_diagnostics: list[dict] = []
    for config_path in _raw_general_paths_config_candidates(project_root):
        if not config_path.is_file():
            config_diagnostics.append({'path': str(config_path), 'status': 'MISSING'})
            continue
        try:
            payload = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
            general_info = payload.get('inputs', {}).get('raw_nonfinancial', {}).get('general_info', {})
            if not isinstance(general_info, dict):
                raise ValueError('inputs.raw_nonfinancial.general_info must be a mapping')
            resolved: list[tuple[str, Path]] = []
            missing_required: list[str] = []
            for market in RAW_GENERAL_MARKET_ORDER:
                recorded = general_info.get(market)
                if recorded in (None, ''):
                    if market in RAW_GENERAL_REQUIRED_MARKETS:
                        missing_required.append(market)
                    continue
                try:
                    path = _resolve_recorded_path(project_root, recorded, field_name=f'stage00_03.general_info.{market}', raw_root=raw_root)
                except FileNotFoundError:
                    if market in RAW_GENERAL_REQUIRED_MARKETS:
                        missing_required.append(market)
                    continue
                resolved.append((market, path.resolve()))
            if missing_required:
                raise FileNotFoundError({'message': 'Configured Stage00-03 general-information inputs are incomplete', 'config_path': str(config_path), 'missing_required_markets': sorted(set(missing_required))})
            if resolved:
                return (resolved, {'source_contract': 'stage00_03_configured_general_info_v1', 'source_config': str(config_path), 'market_priority': list(RAW_GENERAL_MARKET_ORDER), 'config_diagnostics': config_diagnostics + [{'path': str(config_path), 'status': 'SELECTED'}]})
        except Exception as exc:
            config_diagnostics.append({'path': str(config_path), 'status': 'REJECTED', 'error': repr(exc)})
            if config_path.is_relative_to(project_root):
                raise
    fallback_roots = [('canonical_raw_nonfinancial', raw_root / 'raw_nonfinancial' / 'kospi_kosdaq', raw_root / 'raw_nonfinancial' / 'konex_optional')]
    fallback_diagnostics: list[dict] = []
    for source_contract, main_root, konex_root in fallback_roots:
        resolved: list[tuple[str, Path]] = []
        missing_required: list[str] = []
        for market in RAW_GENERAL_MARKET_ORDER:
            root = konex_root if market == 'konex' else main_root
            path = root / RAW_GENERAL_CANONICAL_FILENAMES[market]
            if path.is_file():
                resolved.append((market, path.resolve()))
            elif market in RAW_GENERAL_REQUIRED_MARKETS:
                missing_required.append(market)
        fallback_diagnostics.append({'source_contract': source_contract, 'main_root': str(main_root), 'missing_required_markets': missing_required})
        if not missing_required:
            return (resolved, {'source_contract': f'{source_contract}_exact_filename_v1', 'source_config': None, 'market_priority': list(RAW_GENERAL_MARKET_ORDER), 'config_diagnostics': config_diagnostics, 'fallback_diagnostics': fallback_diagnostics})
    raise FileNotFoundError({'message': 'Could not resolve the canonical Stage00-03 general-information source set', 'required_markets': sorted(RAW_GENERAL_REQUIRED_MARKETS), 'config_diagnostics': config_diagnostics, 'fallback_diagnostics': fallback_diagnostics})

@lru_cache(maxsize=8)
def _raw_general_industry_lookup_cached(project_root_text: str, raw_root_text: str) -> tuple[pd.DataFrame, dict]:
    project_root = Path(project_root_text).resolve()
    raw_root = Path(raw_root_text).resolve()
    sources, source_meta = _resolve_raw_general_information_sources(project_root, raw_root)
    frames: list[pd.DataFrame] = []
    source_records: list[dict] = []
    for priority, (market, path) in enumerate(sources):
        frame = _read_excel_with_fallback(path)
        code_col = next((c for c in ['거래소코드', '종목코드', '회사코드', 'firm_id'] if c in frame.columns), None)
        name_col = next((c for c in ['산업명', '업종명', '산업분류', '업종', 'industry_name'] if c in frame.columns), None)
        industry_code_col = next((c for c in ['산업코드', '표준산업코드', 'industry_code'] if c in frame.columns), None)
        accounting_year_col = next((c for c in ['회계년도', '회계년월', 'fiscal_year', 'year'] if c in frame.columns), None)
        company_name_col = next((c for c in ['회사명', '기업명', 'firm_name', 'company_name'] if c in frame.columns), None)
        if code_col is None or name_col is None:
            raise ValueError(f'Canonical raw general-information workbook lacks firm identifier or industry-name column: {path}')
        out = pd.DataFrame({'firm_id': frame[code_col].map(_normalise_firm_id)})
        out['industry_name'] = frame[name_col]
        out['industry_code'] = frame[industry_code_col] if industry_code_col is not None else pd.NA
        out['accounting_year_sort'] = frame[accounting_year_col].astype(str).str.strip() if accounting_year_col is not None else ''
        out['company_name'] = frame[company_name_col] if company_name_col is not None else pd.NA
        out['sector_7_recovered'] = out['industry_name'].map(_sector7_from_industry_name)
        out['source_market'] = market.upper()
        out['source_priority'] = int(priority)
        out['source_row_order'] = np.arange(len(out), dtype=int)
        out['raw_source'] = str(path)
        out = out[out['firm_id'].astype(str).str.strip().ne('')].copy()
        frames.append(out)
        source_records.append({'market': market, 'priority': int(priority), 'path': str(path), 'rows_loaded': int(len(frame)), 'rows_with_firm_id': int(len(out)), 'firm_id_column': code_col, 'industry_name_column': name_col, 'industry_code_column': industry_code_col, 'accounting_year_column': accounting_year_col, 'company_name_column': company_name_col})
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        raise ValueError('Canonical raw general-information source set produced no usable firm rows')
    combined = combined.sort_values(['firm_id', 'accounting_year_sort', 'source_priority', 'source_row_order'], ascending=[True, False, True, True], kind='mergesort').reset_index(drop=True)
    top_period = combined.groupby('firm_id', sort=False)['accounting_year_sort'].transform('first')
    latest = combined[combined['accounting_year_sort'].eq(top_period)].copy()
    top_priority = latest.groupby('firm_id', sort=False)['source_priority'].transform('min')
    same_precedence = latest[latest['source_priority'].eq(top_priority)].copy()
    same_precedence_conflicts = same_precedence.groupby('firm_id', dropna=False)['sector_7_recovered'].nunique(dropna=False)
    bad_same_precedence = same_precedence_conflicts[same_precedence_conflicts > 1]
    if not bad_same_precedence.empty:
        sample_ids = bad_same_precedence.index.astype(str).tolist()[:20]
        sample = same_precedence[same_precedence['firm_id'].isin(sample_ids)][['firm_id', 'accounting_year_sort', 'source_market', 'company_name', 'industry_name', 'sector_7_recovered', 'raw_source']]
        raise ValueError({'message': 'Canonical raw general-information input has unresolved same-period same-market sector conflicts', 'firm_ids': sample_ids, 'sample_rows': sample.astype(object).where(pd.notna(sample), None).to_dict('records'), 'resolution_policy': 'latest_accounting_period_then_configured_market_priority'})
    sector_counts = combined.groupby('firm_id', dropna=False)['sector_7_recovered'].nunique(dropna=False)
    conflicting_ids = sector_counts[sector_counts > 1].index.astype(str).tolist()
    selected = combined.drop_duplicates('firm_id', keep='first').copy()
    selected_index = selected.set_index('firm_id')
    conflict_sample: list[dict] = []
    for firm_id in conflicting_ids[:20]:
        rows = combined[combined['firm_id'].eq(firm_id)]
        chosen = selected_index.loc[firm_id]
        conflict_sample.append({'firm_id': firm_id, 'selected_accounting_year': str(chosen['accounting_year_sort']), 'selected_market': str(chosen['source_market']), 'selected_sector': str(chosen['sector_7_recovered']), 'candidate_sectors': sorted({str(x) for x in rows['sector_7_recovered'].tolist()}), 'candidate_markets': sorted({str(x) for x in rows['source_market'].tolist()}), 'candidate_accounting_years': sorted({str(x) for x in rows['accounting_year_sort'].tolist()}, reverse=True)})
    lookup = selected[['firm_id', 'industry_name', 'industry_code', 'sector_7_recovered', 'accounting_year_sort', 'company_name', 'source_market', 'raw_source']].reset_index(drop=True)
    meta = {**source_meta, 'raw_workbooks': [str(path) for _, path in sources], 'source_records': source_records, 'resolution_policy': 'stage00_03_latest_accounting_period_then_configured_market_priority_v1', 'pre_resolution_rows': int(len(combined)), 'pre_resolution_unique_firms': int(combined['firm_id'].nunique()), 'duplicate_firm_count': int((combined.groupby('firm_id').size() > 1).sum()), 'conflicting_sector_firm_count': int(len(conflicting_ids)), 'conflict_sample': conflict_sample, 'same_precedence_conflict_count': int(len(bad_same_precedence)), 'lookup_rows': int(len(lookup)), 'lookup_sector_count': int(lookup['sector_7_recovered'].nunique(dropna=False)), 'lookup_value_counts_top20': {str(k): int(v) for k, v in lookup['sector_7_recovered'].value_counts(dropna=False).head(20).items()}, 'selected_market_counts': {str(k): int(v) for k, v in lookup['source_market'].value_counts(dropna=False).items()}, 'selected_accounting_year_counts_top20': {str(k): int(v) for k, v in lookup['accounting_year_sort'].value_counts(dropna=False).head(20).items()}}
    return (lookup, meta)

def _identity_source_candidates(project_root: Path, *, artifact_root: Path | str | None=None) -> list[tuple[Path, bool]]:
    """Ordered, auditable sources for Stage7 row_id→firm_id identity.

    The first two entries are the exact Stage2 serving-panel artifacts consumed
    by Stage7. They may omit a physical row_id column because Stage7 derives it
    from stable row order. Stage6's simulated frame is a strict frozen fallback
    because it records the same row_id and firm_id after Stage6 base-panel merge.

    ``artifact_root`` lets synthetic and run-local callers bind the lookup to a
    specific writable or read-only artifact tree without mutating process-wide
    environment state.
    """
    return [(stage_dir(project_root, 'stage2', artifact_root=artifact_root) / 'phase_eval_candidate.parquet', True), (stage_dir(project_root, 'stage2', artifact_root=artifact_root) / 'input_splits' / 'phase_eval.parquet', True), (stage_dir(project_root, 'stage6', artifact_root=artifact_root) / 'simulated_oracle_input_frame.parquet', False)]

def _resolve_action_identity_source(project_root: Path, action_table: pd.DataFrame, *, artifact_root: Path | str | None=None) -> tuple[pd.DataFrame, str, Path, dict]:
    action_ids = _normalise_row_ids(action_table, label='Stage7 action table')
    diagnostics: list[dict] = []
    firm_aliases = ['firm_id', '거래소코드', '종목코드', 'stock_code', '회사코드']
    for path, allow_index_contract in _identity_source_candidates(project_root, artifact_root=artifact_root):
        if not path.is_file():
            diagnostics.append({'path': str(path), 'status': 'MISSING'})
            continue
        source = read_parquet_required(path)
        firm_col = next((c for c in firm_aliases if c in source.columns), None)
        if firm_col is None:
            diagnostics.append({'path': str(path), 'status': 'REJECTED_NO_FIRM_KEY', 'available_columns': [str(c) for c in source.columns[:80]]})
            continue
        try:
            source, row_id_derivation = _materialize_stage7_row_id_contract(source, source_label=str(path), allow_stable_row_order_index=allow_index_contract)
        except Exception as exc:
            diagnostics.append({'path': str(path), 'status': 'REJECTED_ROW_ID_CONTRACT', 'error': repr(exc)})
            continue
        identity = pd.DataFrame({'row_id': _normalise_row_ids(source, label=str(path)).to_numpy(), 'firm_id': source[firm_col].map(_normalise_firm_id).to_numpy()})
        blank = identity['firm_id'].astype(str).str.strip().eq('')
        if blank.any():
            diagnostics.append({'path': str(path), 'status': 'REJECTED_BLANK_FIRM_ID', 'blank_rows': int(blank.sum()), 'firm_key_column': firm_col, 'row_id_derivation': row_id_derivation})
            continue
        consistency = identity.groupby('row_id', dropna=False)['firm_id'].nunique(dropna=False)
        bad = consistency[consistency != 1]
        if not bad.empty:
            raise ValueError({'message': 'Identity source has inconsistent firm_id values for row_id', 'path': str(path), 'firm_key_column': firm_col, 'row_id_derivation': row_id_derivation, 'bad_row_ids': bad.index.tolist()[:20]})
        identity = identity.drop_duplicates('row_id', keep='first')
        mapped = action_ids.map(identity.set_index('row_id')['firm_id'])
        missing_mask = mapped.isna() | mapped.astype(str).str.strip().eq('')
        if missing_mask.any():
            diagnostics.append({'path': str(path), 'status': 'REJECTED_INCOMPLETE_ROW_ID_COVERAGE', 'missing_count': int(missing_mask.sum()), 'missing_row_ids': sorted(set(action_ids.loc[missing_mask].astype(int).tolist()))[:20], 'firm_key_column': firm_col, 'row_id_derivation': row_id_derivation})
            continue
        meta = {'identity_resolution_status': 'PASS', 'identity_source': str(path), 'identity_firm_key_column': firm_col, 'identity_row_id_derivation': row_id_derivation, 'identity_source_rows': int(len(source)), 'identity_source_unique_row_ids': int(identity['row_id'].nunique()), 'identity_action_rows': int(len(action_table)), 'identity_action_unique_row_ids': int(action_ids.nunique()), 'identity_resolution_candidates': diagnostics + [{'path': str(path), 'status': 'SELECTED', 'firm_key_column': firm_col, 'row_id_derivation': row_id_derivation}]}
        return (pd.DataFrame({'row_id': action_ids.to_numpy(), 'firm_id': mapped.to_numpy()}, index=action_table.index), str(path), path, meta)
    raise FileNotFoundError({'message': 'Could not resolve a frozen Stage7 row_id→firm_id identity source', 'policy': 'Use the exact Stage2 serving-panel row-order contract when row_id is absent; otherwise use the canonical Stage6 simulated frame. No feature-based or positional cross-artifact guessing is allowed.', 'candidates': diagnostics})

def _resolve_raw_industry_strata(project_root: Path, action_table: pd.DataFrame, *, artifact_root: Path | str | None=None, raw_root: Path | str | None=None) -> tuple[pd.Series, dict]:
    identities, identity_label, identity_path, identity_meta = _resolve_action_identity_source(project_root, action_table, artifact_root=artifact_root)
    if raw_root is None:
        raise ValueError('RawRoot is required for raw industry-stratum recovery')
    raw_lookup, raw_meta = _raw_general_industry_lookup_cached(str(Path(project_root).resolve()), str(Path(raw_root).resolve()))
    mapped = identities['firm_id'].map(raw_lookup.set_index('firm_id')['sector_7_recovered'])
    mapped = mapped.fillna(INDUSTRY_UNKNOWN_LABEL).astype('string')
    profile = _profile_aligned_strata(action_table, mapped, shuffle_within='industry')
    if profile['known_fraction'] < MIN_INDUSTRY_STRATUM_KNOWN_FRACTION or profile['unique_known'] < 2:
        raise ValueError({'message': 'Raw industry-stratum recovery is not sufficiently informative', 'minimum_known_fraction': MIN_INDUSTRY_STRATUM_KNOWN_FRACTION, 'profile': profile, 'identity_source': identity_label, 'raw_meta': raw_meta})
    return (pd.Series(mapped.to_numpy(), index=action_table.index, dtype='object'), {'shuffle_within': 'industry', 'stratum_source': 'raw_general_information_recovery', 'stratum_column': 'sector_7_recovered', 'stratum_derivation': 'canonical_stage00_03_sector_name_mapping', 'stratum_identity_source': identity_label, 'stratum_identity_contract': identity_meta, 'stratum_count': int(profile['stratum_count']), 'stratum_known_count': int(profile['known_count']), 'stratum_unknown_count': int(profile['unknown_count']), 'stratum_known_fraction': float(profile['known_fraction']), 'stratum_unique_known': int(profile['unique_known']), 'stratum_value_counts_top20': profile['value_counts_top20'], 'raw_general_information': raw_meta, 'stratum_resolution_status': 'PASS'})

def _resolve_shuffle_strata(project_root: Path, action_table: pd.DataFrame, *, shuffle_within: str, artifact_root: Path | str | None=None, raw_root: Path | str | None=None) -> tuple[pd.Series, dict]:
    if shuffle_within not in SHUFFLE_WITHIN_CHOICES:
        raise ValueError(f'shuffle_within must be one of {SHUFFLE_WITHIN_CHOICES}; got {shuffle_within!r}')
    if shuffle_within == 'none':
        values = pd.Series('__ALL__', index=action_table.index, dtype='object')
        meta = {'shuffle_within': 'none', 'stratum_source': 'constant', 'stratum_column': None, 'stratum_derivation': 'constant', 'stratum_count': 1, 'stratum_known_count': int(len(values)), 'stratum_unknown_count': 0, 'stratum_known_fraction': 1.0, 'stratum_unique_known': 1, 'stratum_value_counts_top20': {'__ALL__': int(len(values))}, 'stratum_resolution_status': 'PASS'}
        return (values, meta)
    candidates = SHUFFLE_STRATUM_COLUMNS[shuffle_within]
    diagnostics: list[dict] = []
    for source_label, source, source_path in _candidate_stratum_sources(project_root, action_table, artifact_root=artifact_root):
        for column in candidates:
            if column not in source.columns:
                continue
            try:
                mapped_raw = _align_source_values_by_row_id(action_table, source, source_label=source_label, column=column)
                mapped, derivation = _transform_stratum_values(mapped_raw, shuffle_within=shuffle_within, column=column)
                profile = _profile_aligned_strata(action_table, mapped, shuffle_within=shuffle_within)
                diagnostics.append({'source': source_label, 'column': column, 'derivation': derivation, **profile})
            except Exception as exc:
                diagnostics.append({'source': source_label, 'column': column, 'error': repr(exc)})
                continue
            minimum_fraction = MIN_INDUSTRY_STRATUM_KNOWN_FRACTION if shuffle_within == 'industry' else 1.0
            if profile['known_fraction'] < minimum_fraction or profile['unique_known'] < 2:
                continue
            meta = {'shuffle_within': shuffle_within, 'stratum_source': source_label, 'stratum_column': column, 'stratum_derivation': derivation, 'stratum_count': int(profile['stratum_count']), 'stratum_known_count': int(profile['known_count']), 'stratum_unknown_count': int(profile['unknown_count']), 'stratum_known_fraction': float(profile['known_fraction']), 'stratum_unique_known': int(profile['unique_known']), 'stratum_value_counts_top20': profile['value_counts_top20'], 'stratum_resolution_candidates': diagnostics, 'stratum_resolution_status': 'PASS'}
            return (pd.Series(mapped.to_numpy(), index=action_table.index, dtype='object'), meta)
    if shuffle_within == 'industry':
        mapped, meta = _resolve_raw_industry_strata(project_root, action_table, artifact_root=artifact_root, raw_root=raw_root)
        meta['stratum_resolution_candidates'] = diagnostics
        return (mapped, meta)
    raise ValueError({'message': f'Could not resolve an informative {shuffle_within} shuffle stratum', 'tried_columns': list(candidates), 'candidate_diagnostics': diagnostics, 'minimum_unique_known': 2})

def _action_cols(space) -> list[str]:
    return list(space.columns)

def _raw_l1(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].apply(pd.to_numeric, errors='coerce').abs().sum(axis=1)

def _active_dims(df: pd.DataFrame, cols: list[str], eps: float) -> pd.Series:
    return (df[cols].apply(pd.to_numeric, errors='coerce').abs() > float(eps)).sum(axis=1)

def _clip_action_frame(df: pd.DataFrame, space) -> pd.DataFrame:
    out = df.copy()
    for col in _action_cols(space):
        lo, hi = space.bounds[col]
        out[col] = pd.to_numeric(out[col], errors='coerce').fillna(0.0).clip(lo, hi)
    return out

def _nearest_candidate_labels(X: np.ndarray, space) -> tuple[list[str], np.ndarray]:
    cols = _action_cols(space)
    cand_names = list(space.train_labels)
    C = np.stack([space.candidate_vector(name) for name in cand_names], axis=0).astype(float)
    widths = np.array([space.bound_width(c) for c in cols], dtype=float)
    d = np.abs((X[:, None, :] - C[None, :, :]) / widths[None, None, :]).mean(axis=2)
    ix = d.argmin(axis=1)
    return ([cand_names[i] for i in ix], d[np.arange(len(X)), ix])

def _resolve_target_budgets(df: pd.DataFrame, *, policies: list[str], cols: list[str], target_budget: str) -> tuple[dict[str, float | None], float | None]:
    """Resolve per-policy L1 budgets.

    target_budget="native" means do not rescale the transformed vector. This is
    especially important for null baselines where native L1 is the empirical
    action intensity of the null vector itself.
    """
    if target_budget == 'native':
        return ({pol: None for pol in policies}, None)
    budget_rows = df[df['mode'].astype(str) == 'candidate_selection'].copy()
    if target_budget == 'same_policy_candidate_mean':
        budgets = {pol: float(_raw_l1(budget_rows[budget_rows['policy'].astype(str) == pol], cols).mean()) for pol in policies}
        fallback = float(_raw_l1(budget_rows, cols).mean())
    elif target_budget == 'all_candidate_mean':
        fallback = float(_raw_l1(budget_rows, cols).mean())
        budgets = {pol: fallback for pol in policies}
    elif target_budget.startswith('fixed:'):
        fallback = float(target_budget.split(':', 1)[1])
        budgets = {pol: fallback for pol in policies}
    else:
        raise ValueError('target_budget must be native, same_policy_candidate_mean, all_candidate_mean, or fixed:<float>')
    budgets = {k: v if np.isfinite(v) and v > 0 else fallback for k, v in budgets.items()}
    return (budgets, fallback)

def _rescale_rows_to_budget(frame: pd.DataFrame, *, reference_l1: pd.Series, cols: list[str], policies: list[str], budgets: dict[str, float | None], fallback: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    rows = []
    for i, row in frame.iterrows():
        old = float(reference_l1.loc[i]) if i in reference_l1.index and pd.notna(reference_l1.loc[i]) else 0.0
        target = budgets.get(str(row['policy']), fallback)
        if target is None:
            factor = 1.0
            target_l1 = old
        else:
            target_l1 = float(target)
            factor = 0.0 if old <= 0 else target_l1 / old
            for col in cols:
                out.at[i, col] = float(row[col]) * factor
        rows.append((i, factor, target_l1, old))
    detail = pd.DataFrame(rows, columns=['_index', 'rescale_factor', 'target_l1', 'pre_rescale_l1']).set_index('_index')
    return (out, detail)

def _variant_rows(action_table: pd.DataFrame, *, space, policies: list[str], modes: list[str], variant: str, target_budget: str, active_eps: float, random_seed: int=1, shuffle_within: str='none', shuffle_strata: pd.Series | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return transformed action table and row-level diagnostics."""
    if variant == 'sign_preserving_random_vector':
        variant = 'row_shuffle_vector_null'
    if variant not in SUPPORTED_VARIANTS:
        raise ValueError(f'Unsupported variant={variant}. Supported: {sorted(SUPPORTED_VARIANTS)}')
    cols = _action_cols(space)
    df = action_table.copy()
    target_mask = df['policy'].astype(str).isin(policies) & df['mode'].astype(str).isin(modes)
    if not target_mask.any():
        raise ValueError(f'No rows match policies={policies}, modes={modes}')
    before = df.loc[target_mask].copy()
    if shuffle_within not in SHUFFLE_WITHIN_CHOICES:
        raise ValueError(f'shuffle_within must be one of {SHUFFLE_WITHIN_CHOICES}; got {shuffle_within!r}')
    if shuffle_strata is None:
        if shuffle_within != 'none':
            raise ValueError('Conditional row shuffle requires an explicitly resolved shuffle_strata series')
        before['_shuffle_stratum'] = '__ALL__'
    else:
        aligned_strata = shuffle_strata.reindex(before.index)
        if aligned_strata.isna().any():
            raise ValueError('shuffle_strata is missing one or more target action-table rows')
        before['_shuffle_stratum'] = aligned_strata.astype(str).to_numpy()
    budgets, fallback = _resolve_target_budgets(df, policies=policies, cols=cols, target_budget=target_budget)
    after = before.copy()
    variant_detail = pd.DataFrame(index=before.index)
    if variant == 'l1_rescale':
        base_l1 = _raw_l1(before, cols).replace(0, np.nan)
        after, variant_detail = _rescale_rows_to_budget(before, reference_l1=base_l1, cols=cols, policies=policies, budgets=budgets, fallback=fallback)
        variant_detail = variant_detail.rename(columns={'pre_rescale_l1': 'original_l1_detail'})
    elif variant == 'nearest_candidate_projection':
        X = before[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
        labels, d = _nearest_candidate_labels(X, space)
        rows = []
        for idx, label, dist in zip(before.index, labels, d):
            vec = space.candidate_vector(label)
            for col, val in zip(cols, vec):
                after.at[idx, col] = float(val)
            after.at[idx, 'candidate_id'] = label
            after.at[idx, 'projection_distance'] = float(dist)
            after.at[idx, 'projection_method'] = 'nearest_candidate_projection_ablation'
            after.at[idx, 'out_of_library'] = False
            rows.append((idx, label, float(dist)))
        variant_detail = pd.DataFrame(rows, columns=['_index', 'nearest_candidate_id', 'nearest_candidate_distance']).set_index('_index')
    elif variant in {'global_mean_vector_null', 'sign_flip_mean_vector'}:
        rows = []
        sign_multiplier = -1.0 if variant == 'sign_flip_mean_vector' else 1.0
        method = 'sign_flip_mean_vector' if variant == 'sign_flip_mean_vector' else 'global_mean_vector_null'
        label_prefix = 'NULL_SIGN_FLIP_MEAN' if variant == 'sign_flip_mean_vector' else 'NULL_GLOBAL_MEAN'
        for (pol, mode), idx in before.groupby(['policy', 'mode'], dropna=False).groups.items():
            sub = before.loc[idx]
            mean_vec = sub[cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).mean(axis=0) * sign_multiplier
            for ridx in idx:
                for col in cols:
                    after.at[ridx, col] = float(mean_vec[col])
                after.at[ridx, 'candidate_id'] = f'{label_prefix}_{pol}'
                after.at[ridx, 'projection_method'] = method
                after.at[ridx, 'out_of_library'] = True
                rows.append((ridx, str(pol), str(mode), float(sign_multiplier)))
        pre_l1 = _raw_l1(after, cols).replace(0, np.nan)
        after, scale_detail = _rescale_rows_to_budget(after, reference_l1=pre_l1, cols=cols, policies=policies, budgets=budgets, fallback=fallback)
        variant_detail = pd.DataFrame(rows, columns=['_index', 'null_policy', 'null_mode', 'sign_multiplier']).set_index('_index')
        variant_detail = variant_detail.join(scale_detail, how='left')
    elif variant == 'row_shuffle_vector_null':
        rng = np.random.default_rng(int(random_seed))
        rows = []
        group_cols = ['policy', 'mode', '_shuffle_stratum']
        for (pol, mode, stratum), idx_obj in before.groupby(group_cols, dropna=False).groups.items():
            idx = np.array(list(idx_obj), dtype=object)
            shuffled = idx.copy()
            if len(shuffled) > 1:
                perm = np.arange(len(shuffled))
                for pos in range(len(perm) - 1, 0, -1):
                    swap_pos = int(rng.integers(0, pos))
                    perm[pos], perm[swap_pos] = (perm[swap_pos], perm[pos])
                shuffled = idx[perm]
                if np.any(shuffled == idx):
                    raise RuntimeError('Internal derangement failure in row_shuffle_vector_null')
            for target_idx, source_idx in zip(idx, shuffled):
                for col in cols:
                    after.at[target_idx, col] = float(before.at[source_idx, col])
                after.at[target_idx, 'candidate_id'] = f'NULL_ROW_SHUFFLE_{pol}'
                after.at[target_idx, 'projection_method'] = 'row_shuffle_vector_null'
                after.at[target_idx, 'out_of_library'] = True
                source_row_id = before.at[source_idx, 'row_id'] if 'row_id' in before.columns else source_idx
                rows.append((target_idx, source_idx, source_row_id, str(pol), str(mode), str(stratum), bool(target_idx == source_idx), int(len(idx))))
        pre_l1 = _raw_l1(after, cols).replace(0, np.nan)
        after, scale_detail = _rescale_rows_to_budget(after, reference_l1=pre_l1, cols=cols, policies=policies, budgets=budgets, fallback=fallback)
        variant_detail = pd.DataFrame(rows, columns=['_index', 'null_source_index', 'null_source_row_id', 'null_policy', 'null_mode', 'shuffle_stratum', 'shuffle_kept_same_row', 'shuffle_group_size']).set_index('_index')
        variant_detail = variant_detail.join(scale_detail, how='left')
    after = _clip_action_frame(after, space)
    df.loc[target_mask, cols] = after[cols]
    for c in ['candidate_id', 'projection_distance', 'projection_method', 'out_of_library']:
        if c in df.columns and c in after.columns:
            df.loc[target_mask, c] = after[c]
    diagnostics = before[['row_id', 'policy', 'mode', 'candidate_id']].copy()
    diagnostics['variant'] = variant
    diagnostics['target_budget'] = target_budget
    diagnostics['random_seed'] = int(random_seed)
    diagnostics['shuffle_within'] = shuffle_within
    diagnostics['original_l1'] = _raw_l1(before, cols).to_numpy()
    diagnostics['variant_l1'] = _raw_l1(after, cols).to_numpy()
    diagnostics['original_active_dims'] = _active_dims(before, cols, active_eps).to_numpy()
    diagnostics['variant_active_dims'] = _active_dims(after, cols, active_eps).to_numpy()
    diagnostics['original_projection_distance'] = before.get('projection_distance', pd.Series(np.nan, index=before.index)).to_numpy()
    diagnostics['variant_projection_distance'] = after.get('projection_distance', pd.Series(np.nan, index=after.index)).to_numpy()
    diagnostics = diagnostics.join(variant_detail, how='left')
    return (df, diagnostics)

@lru_cache(maxsize=8)
def _load_stage6_base_context_cached(project_root_text: str) -> tuple[pd.DataFrame, dict]:
    project_root = Path(project_root_text).resolve()
    base_path = stage_dir(project_root, 'stage2') / 'phase_eval_candidate.parquet'
    if not base_path.exists():
        raise FileNotFoundError(f'Missing Stage 2 phase_eval_candidate.parquet: {base_path}')
    base = read_parquet_required(base_path)
    return (base, prepare_stage6_history_lookup(base))

@lru_cache(maxsize=8)
def _load_oracle_registry_cached(project_root_text: str) -> dict:
    project_root = Path(project_root_text).resolve()
    reg_path = final_root(project_root) / 'configs' / 'oracle_backend_registry.yaml'
    return load_registry(reg_path)

@lru_cache(maxsize=8)
def _load_simulator_identity_cached(project_root_text: str) -> dict:
    return _load_stage6_simulator_identity(Path(project_root_text).resolve())

@lru_cache(maxsize=8)
def _load_noop_scores_cached(project_root_text: str) -> dict[str, pd.DataFrame]:
    return _load_stage6_noop_scores(Path(project_root_text).resolve())

def _score_action_table(*, project_root: Path, action_table: pd.DataFrame, out: Path, space, output_prefix: str='ablation', write_outputs: bool=True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Stage8-equivalent scoring to a noncanonical output directory."""
    project_root = Path(project_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    final = final_root(project_root)
    base, history_lookup = _load_stage6_base_context_cached(str(project_root))
    reg = _load_oracle_registry_cached(str(project_root))
    backends = reg.get('backends', {})
    if reg.get('final_result_allowed') is not True or reg.get('status') != 'generated_by_stage1_oracle_development_verified':
        raise ValueError('Oracle backend registry is not final/generated_by_stage1_oracle_development_verified.')
    simulator_identity = _load_simulator_identity_cached(str(project_root))
    sim_state, audit = simulate_policy_states(base, action_table, space, out, predicted_fiscal_year=int(simulator_identity['predicted_fiscal_year']), preserve_current_non_current_residual=bool(simulator_identity['preserve_current_non_current_residual']), sim_business_plan_mode=str(simulator_identity['sim_business_plan_mode']), write_outputs=bool(write_outputs), collect_audit=bool(write_outputs), history_lookup=history_lookup)
    score_key_cols = ['row_id', 'policy', 'mode', 'candidate_id']
    score_key_cols = [c for c in score_key_cols if c in action_table.columns]
    all_scores = []
    for backend in ['alpha', 'beta', 'gamma']:
        b = backends[backend]
        params = resolve_backend_artifact(project_root, final, b.get('params', ''))
        if backend == 'alpha':
            score = score_alpha(sim_state, params)
            col = 'R_score_alpha'
        elif backend == 'beta':
            score = score_beta_ordered_logit_params(sim_state, params)
            col = 'R_score_beta'
        else:
            model = resolve_backend_artifact(project_root, final, b.get('model', ''))
            score = score_gamma_model(sim_state, params, model)
            col = 'R_score_gamma'
        scored = action_table[score_key_cols + list(space.columns)].copy()
        scored[col] = score.to_numpy()
        if write_outputs:
            scored.to_parquet(out / f'{output_prefix}_oracle_scores_{backend}.parquet', index=False)
        all_scores.append(scored[score_key_cols + [col]])
    merged = all_scores[0]
    for s in all_scores[1:]:
        merged = merged.merge(s, on=score_key_cols, how='outer')
    noop_by_backend = _load_noop_scores_cached(str(project_root))
    for backend in ['alpha', 'beta', 'gamma']:
        merged = merged.merge(noop_by_backend[backend], on='row_id', how='left')
        merged[f'delta_R_score_{backend}'] = pd.to_numeric(merged[f'R_score_{backend}'], errors='coerce') - pd.to_numeric(merged[f'noop_R_score_{backend}'], errors='coerce')
    summary = _per_policy_mode_summary(merged)
    if write_outputs:
        merged_path = out / f'{output_prefix}_multi_oracle_scores.parquet'
        merged.to_parquet(merged_path, index=False)
        summary.to_csv(out / f'{output_prefix}_policy_summary.csv', index=False, encoding='utf-8-sig')
    return (merged, summary, audit, simulator_identity)

def _per_policy_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [c for c in ['policy', 'mode'] if c in df.columns]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: str(val) for col, val in zip(group_cols, keys)}
        rec['n_rows'] = int(len(g))
        for bk in ['alpha', 'beta', 'gamma']:
            col = f'delta_R_score_{bk}'
            s = pd.to_numeric(g[col], errors='coerce') if col in g.columns else pd.Series(dtype=float)
            rec[f'mean_delta_R_score_{bk}'] = float(s.mean()) if s.notna().any() else float('nan')
            rec[f'median_delta_R_score_{bk}'] = float(s.median()) if s.notna().any() else float('nan')
            rec[f'positive_fraction_{bk}'] = score_positive_fraction(s.to_numpy())
            rec[f'valid_fraction_{bk}'] = float(s.notna().mean()) if len(s) else float('nan')
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)

def _signed_wilcoxon_p(x: np.ndarray) -> float:
    x = score_wilcoxon_values(x)
    if len(x) == 0:
        return math.nan
    if wilcoxon is None:
        raise RuntimeError('scipy is required for Wilcoxon p-values. Install scipy in the thesis venv.')
    return float(wilcoxon(x, alternative='two-sided', zero_method='wilcox', correction=False).pvalue)

def _holm_adjust(pvals: Iterable[float]) -> list[float]:
    p = np.asarray(list(pvals), dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    ok = np.where(np.isfinite(p))[0]
    if len(ok) == 0:
        return out.tolist()
    order = ok[np.argsort(p[ok])]
    prev = 0.0
    for rank, i in enumerate(order, start=1):
        adj = (len(order) - rank + 1) * p[i]
        prev = max(prev, adj)
        out[i] = min(prev, 1.0)
    return out.tolist()

def _p_to_stars(p: float) -> str:
    if not np.isfinite(p):
        return ''
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    return 'n.s.'

def _resolve_reference_policy(stage6: pd.DataFrame, requested: str, space) -> str:
    policies = set(stage6['policy'].astype(str))
    if requested in policies:
        return requested
    if requested == 'C3':
        candidates = [str(space.final_rl_label), f'{space.final_rl_label}_q_argmax', f'{space.final_rl_label}_q_rerank_at_7', f'{space.final_rl_label}_q_rerank_at_9']
        candidates += sorted([p for p in policies if p.startswith('C3_candidate_iql')])
        for cand in candidates:
            if cand in policies:
                return cand
    c3_like = sorted([p for p in policies if p.startswith('C3')])
    raise ValueError({'message': 'Could not resolve reference policy in Stage6 multi_oracle_policy_eval.', 'requested': requested, 'c3_like_available': c3_like, 'available_sample': sorted(list(policies))[:50]})

def _paired_against_reference(target_scores: pd.DataFrame, reference_scores: pd.DataFrame, *, reference_label: str, comparison_label: str) -> pd.DataFrame:
    rows = []
    group_cols = [c for c in ['policy', 'mode'] if c in target_scores.columns]
    for keys, g in target_scores.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base_rec = {col: str(val) for col, val in zip(group_cols, keys)}
        for bk in ['alpha', 'beta', 'gamma']:
            col = f'delta_R_score_{bk}'
            left = g[['row_id', col]].rename(columns={col: 'target_score'})
            right = reference_scores[['row_id', col]].rename(columns={col: 'reference_score'})
            m = left.merge(right, on='row_id', how='inner')
            gap = pd.to_numeric(m['target_score'], errors='coerce') - pd.to_numeric(m['reference_score'], errors='coerce')
            gap = gap[np.isfinite(gap)]
            p_raw = _signed_wilcoxon_p(gap.to_numpy()) if len(gap) else math.nan
            rec = dict(base_rec)
            rec.update({'comparison': comparison_label, 'oracle_backend': bk, 'reference_label': reference_label, 'n_pairs': int(len(gap)), 'n_nonzero_pairs': score_nonzero_count(gap.to_numpy()), 'mean_gap_vs_reference': float(gap.mean()) if len(gap) else math.nan, 'median_gap_vs_reference': float(gap.median()) if len(gap) else math.nan, 'wilcoxon_p_raw': p_raw})
            rows.append(rec)
    out = pd.DataFrame(rows)
    if not out.empty:
        out['wilcoxon_p_holm'] = _holm_adjust(out['wilcoxon_p_raw'].to_numpy())
        out['sig_holm'] = [_p_to_stars(x) for x in out['wilcoxon_p_holm']]
    return out

def _target_filter(df: pd.DataFrame, policies: list[str], modes: list[str]) -> pd.DataFrame:
    m = df['policy'].astype(str).isin(policies)
    if 'mode' in df.columns:
        m &= df['mode'].astype(str).isin(modes)
    return df.loc[m].copy()

def _load_stage6_reference_scores(project_root: Path, requested_policy: str, space=None) -> tuple[pd.DataFrame, str]:
    if space is None:
        space = load_action_space(project_root)
    p = stage_dir(project_root, 'stage6') / 'multi_oracle_policy_eval.parquet'
    if not p.exists():
        raise FileNotFoundError(f'Missing Stage6 reference eval: {p}')
    stage6 = read_parquet_required(p)
    needed = {'row_id', 'policy', 'delta_R_score_alpha', 'delta_R_score_beta', 'delta_R_score_gamma'}
    missing = needed - set(stage6.columns)
    if missing:
        raise ValueError(f'Stage6 reference eval missing columns: {missing}')
    resolved = _resolve_reference_policy(stage6, requested_policy, space)
    ref = stage6[stage6['policy'].astype(str).eq(resolved)].copy()
    if ref.empty:
        raise ValueError(f'No Stage6 rows for resolved reference policy {resolved}')
    return (ref, resolved)

def _shuffle_draw_summary(*, shuffled_scores: pd.DataFrame, original_scores: pd.DataFrame, reference_scores: pd.DataFrame, reference_label: str, policies: list[str], modes: list[str], seed: int, shuffle_within: str, row_diagnostics: pd.DataFrame) -> pd.DataFrame:
    target = _target_filter(shuffled_scores, policies, modes)
    original = _target_filter(original_scores, policies, modes)
    rows: list[dict] = []
    group_cols = [c for c in ['policy', 'mode'] if c in target.columns]
    fixed_points = int(pd.to_numeric(row_diagnostics.get('shuffle_kept_same_row', 0), errors='coerce').fillna(0).astype(bool).sum())
    singleton_rows = int((pd.to_numeric(row_diagnostics.get('shuffle_group_size', 0), errors='coerce') == 1).sum())
    stratum_count = int(row_diagnostics.get('shuffle_stratum', pd.Series(['__ALL__'])).astype(str).nunique())
    for keys, group in target.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: str(value) for col, value in zip(group_cols, keys)}
        group_original = original
        for col, value in base.items():
            if col in group_original.columns:
                group_original = group_original[group_original[col].astype(str).eq(value)]
        for backend in ['alpha', 'beta', 'gamma']:
            score_col = f'delta_R_score_{backend}'
            shuffled_part = group[['row_id', score_col]].rename(columns={score_col: 'shuffled_score'})
            original_part = group_original[['row_id', score_col]].rename(columns={score_col: 'original_score'})
            reference_part = reference_scores[['row_id', score_col]].rename(columns={score_col: 'reference_score'})
            paired = shuffled_part.merge(original_part, on='row_id', how='inner').merge(reference_part, on='row_id', how='inner')
            for col in ['shuffled_score', 'original_score', 'reference_score']:
                paired[col] = pd.to_numeric(paired[col], errors='coerce')
            paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
            if paired.empty:
                raise ValueError(f'Shuffle permutation draw has no complete pairs for backend={backend}, policy/mode={base}')
            matching_gap = paired['original_score'] - paired['shuffled_score']
            c3_gap = paired['shuffled_score'] - paired['reference_score']
            rows.append({**base, 'oracle_backend': backend, 'reference_label': reference_label, 'shuffle_seed': int(seed), 'shuffle_within': shuffle_within, 'n_pairs': int(len(paired)), 'mean_original_score': float(paired['original_score'].mean()), 'mean_shuffled_score': float(paired['shuffled_score'].mean()), 'mean_matching_gain_original_minus_shuffle': float(matching_gap.mean()), 'median_matching_gain_original_minus_shuffle': float(matching_gap.median()), 'mean_gap_shuffle_vs_reference': float(c3_gap.mean()), 'median_gap_shuffle_vs_reference': float(c3_gap.median()), 'shuffle_fixed_point_count': fixed_points, 'shuffle_singleton_row_count': singleton_rows, 'shuffle_stratum_count': stratum_count})
    return pd.DataFrame(rows)

def _aggregate_shuffle_draws(per_draw: pd.DataFrame) -> pd.DataFrame:
    if per_draw.empty:
        raise ValueError('Cannot aggregate an empty shuffle permutation table')
    group_cols = [c for c in ['policy', 'mode', 'oracle_backend', 'reference_label', 'shuffle_within'] if c in per_draw.columns]
    rows: list[dict] = []
    for keys, group in per_draw.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        rec = {col: str(value) for col, value in zip(group_cols, keys)}
        original_values = pd.to_numeric(group['mean_original_score'], errors='coerce').dropna().to_numpy(dtype=float)
        null_values = pd.to_numeric(group['mean_shuffled_score'], errors='coerce').dropna().to_numpy(dtype=float)
        matching_values = pd.to_numeric(group['mean_matching_gain_original_minus_shuffle'], errors='coerce').dropna().to_numpy(dtype=float)
        reference_values = pd.to_numeric(group['mean_gap_shuffle_vs_reference'], errors='coerce').dropna().to_numpy(dtype=float)
        n = len(null_values)
        if n == 0 or len(original_values) != n or len(matching_values) != n:
            raise ValueError(f'Incomplete shuffle draw aggregation for group={rec}')
        original_mean = float(np.mean(original_values))
        p_upper = float((1 + np.sum(null_values >= original_mean)) / (n + 1))
        p_lower = float((1 + np.sum(null_values <= original_mean)) / (n + 1))
        rec.update({'n_shuffle_draws': int(n), 'original_mean_score': original_mean, 'shuffle_null_mean_score': float(np.mean(null_values)), 'shuffle_null_score_ci_lower_2p5': float(np.quantile(null_values, 0.025)), 'shuffle_null_score_ci_median': float(np.quantile(null_values, 0.5)), 'shuffle_null_score_ci_upper_97p5': float(np.quantile(null_values, 0.975)), 'matching_gain_mean': float(np.mean(matching_values)), 'matching_gain_ci_lower_2p5': float(np.quantile(matching_values, 0.025)), 'matching_gain_ci_median': float(np.quantile(matching_values, 0.5)), 'matching_gain_ci_upper_97p5': float(np.quantile(matching_values, 0.975)), 'shuffle_vs_reference_gap_mean': float(np.mean(reference_values)), 'shuffle_vs_reference_gap_ci_lower_2p5': float(np.quantile(reference_values, 0.025)), 'shuffle_vs_reference_gap_ci_upper_97p5': float(np.quantile(reference_values, 0.975)), 'empirical_p_original_not_better_one_sided': p_upper, 'empirical_p_two_sided': min(1.0, 2.0 * min(p_upper, p_lower)), 'all_non_singleton_groups_deranged': bool((pd.to_numeric(group['shuffle_fixed_point_count'], errors='coerce') == pd.to_numeric(group['shuffle_singleton_row_count'], errors='coerce')).all()), 'max_shuffle_fixed_point_count': int(pd.to_numeric(group['shuffle_fixed_point_count'], errors='coerce').max()), 'max_shuffle_singleton_row_count': int(pd.to_numeric(group['shuffle_singleton_row_count'], errors='coerce').max()), 'shuffle_stratum_count': int(pd.to_numeric(group['shuffle_stratum_count'], errors='coerce').max())})
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)
_SHUFFLE_WORKER_STATE: dict | None = None

def _checkpoint_signature(*, stage7_action_table: Path, variant: str, target_budget: str, policies: list[str], modes: list[str], shuffle_within: str, seeds: list[int], reference_policy: str) -> dict:
    payload = {'schema_version': 'shuffle_permutation_checkpoint_v4_exact_zero', 'stage7_action_table': str(Path(stage7_action_table).resolve()), 'variant': str(variant), 'target_budget': str(target_budget), 'policies': [str(x) for x in policies], 'modes': [str(x) for x in modes], 'shuffle_within': str(shuffle_within), 'shuffle_seeds': [int(x) for x in seeds], 'reference_policy': str(reference_policy), 'engine': 'target_rows_parallel_v2', 'score_tie_contract': score_tie_contract_metadata()}
    if str(shuffle_within) == 'industry':
        payload['industry_stratum_contract_version'] = 'informative_industry_strata_v1'
    return payload

def _load_shuffle_checkpoint(*, partial_path: Path, checkpoint_path: Path, expected_signature: dict) -> pd.DataFrame:
    if not partial_path.exists() and (not checkpoint_path.exists()):
        return pd.DataFrame()
    if not checkpoint_path.exists():
        raise RuntimeError(f'Incomplete shuffle checkpoint: partial CSV exists without checkpoint JSON; partial={partial_path}')
    recorded = _read_json_dict(checkpoint_path)
    source_path = partial_path
    if not partial_path.exists():
        final_path = partial_path.parent / 'shuffle_per_draw_summary.csv'
        if recorded.get('status') == 'PASS' and final_path.is_file():
            source_path = final_path
        else:
            raise RuntimeError(f"Incomplete shuffle checkpoint: RUNNING checkpoint lacks partial CSV or PASS checkpoint lacks final CSV; checkpoint_status={recorded.get('status')!r}, partial={partial_path}, final={final_path}")
    frame = pd.read_csv(source_path)
    if frame.empty:
        return frame
    if 'shuffle_seed' not in frame.columns:
        raise ValueError(f'Shuffle checkpoint missing shuffle_seed: {partial_path}')
    seeds = pd.to_numeric(frame['shuffle_seed'], errors='coerce')
    if seeds.isna().any() or not np.allclose(seeds.to_numpy(dtype=float), np.round(seeds.to_numpy(dtype=float))):
        raise ValueError(f'Shuffle checkpoint contains non-integer seeds: {partial_path}')
    frame['shuffle_seed'] = seeds.astype(int)
    key_cols = [c for c in ['shuffle_seed', 'policy', 'mode', 'oracle_backend', 'reference_label', 'shuffle_within'] if c in frame.columns]
    if frame.duplicated(key_cols).any():
        raise ValueError(f'Shuffle checkpoint contains duplicate draw rows on keys={key_cols}: {partial_path}')
    requested = set((int(x) for x in expected_signature['shuffle_seeds']))
    unexpected = sorted(set(frame['shuffle_seed'].tolist()) - requested)
    if unexpected:
        raise ValueError(f'Shuffle checkpoint contains seeds outside the requested contract: {unexpected[:20]}')
    return frame

def _write_shuffle_checkpoint(*, frames: list[pd.DataFrame], partial_path: Path, checkpoint_path: Path, signature: dict) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        sort_cols = [c for c in ['shuffle_seed', 'policy', 'mode', 'oracle_backend'] if c in combined.columns]
        combined = combined.sort_values(sort_cols).reset_index(drop=True)
    partial_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(partial_path, index=False, encoding='utf-8-sig')
    payload = dict(signature)
    payload.update({'status': 'RUNNING', 'updated_utc': _now(), 'completed_seed_count': int(combined['shuffle_seed'].nunique()) if not combined.empty else 0, 'completed_seeds': sorted(set((int(x) for x in combined['shuffle_seed'].tolist()))) if not combined.empty else [], 'partial_csv': partial_path.name})
    write_json(checkpoint_path, payload)
    return combined

def _init_shuffle_worker(payload: dict) -> None:
    global _SHUFFLE_WORKER_STATE
    project_root = Path(payload['project_root']).resolve()
    stage7_action_table = Path(payload['stage7_action_table']).resolve()
    space, _provenance = _load_action_space_for_stage7(project_root, stage7_action_table)
    action_table = read_parquet_required(stage7_action_table)
    strata_values = payload.get('shuffle_strata_values')
    if not isinstance(strata_values, list) or len(strata_values) != len(action_table):
        raise ValueError('Shuffle worker payload must contain one pre-resolved stratum value per Stage7 action row')
    shuffle_strata = pd.Series(strata_values, index=action_table.index, dtype='object')
    original_scores = read_parquet_required(Path(payload['original_scores_path']))
    reference_scores = read_parquet_required(Path(payload['reference_scores_path']))
    _SHUFFLE_WORKER_STATE = {**payload, 'project_root_path': project_root, 'space': space, 'action_table': action_table, 'shuffle_strata': shuffle_strata, 'original_scores': original_scores, 'reference_scores': reference_scores}

def _score_shuffle_seed_worker(seed: int) -> list[dict]:
    state = _SHUFFLE_WORKER_STATE
    if state is None:
        raise RuntimeError('Shuffle worker was not initialized')
    draw_table, draw_diag = _variant_rows(state['action_table'], space=state['space'], policies=list(state['policies']), modes=list(state['modes']), variant=str(state['variant']), target_budget=str(state['target_budget']), active_eps=float(state['active_eps']), random_seed=int(seed), shuffle_within=str(state['shuffle_within']), shuffle_strata=state['shuffle_strata'])
    draw_target = _target_filter(draw_table, list(state['policies']), list(state['modes']))
    with tempfile.TemporaryDirectory(prefix=f'shuffle_seed_{seed}_') as tmp:
        draw_scores, _draw_summary, _draw_audit, draw_identity = _score_action_table(project_root=state['project_root_path'], action_table=draw_target, out=Path(tmp), space=state['space'], output_prefix=f'shuffle_seed_{seed}', write_outputs=False)
    if draw_identity != state['simulator_identity']:
        raise ValueError(f"Simulator identity changed across shuffle draws: seed={seed}, first={state['simulator_identity']}, draw={draw_identity}")
    summary = _shuffle_draw_summary(shuffled_scores=draw_scores, original_scores=state['original_scores'], reference_scores=state['reference_scores'], reference_label=str(state['resolved_reference']), policies=list(state['policies']), modes=list(state['modes']), seed=int(seed), shuffle_within=str(state['shuffle_within']), row_diagnostics=draw_diag)
    return summary.to_dict(orient='records')

def run_ablation(*, project_root: Path, raw_root: Path | None, stage7_action_table: Path | None, output_dir: Path, policies: list[str], modes: list[str], variant: str, target_budget: str, active_eps: float=1e-09, random_seed: int=1, shuffle_seeds: Iterable[int] | None=None, shuffle_within: str='none', reference_policy: str='C3', write_paired_inference: bool=True, shuffle_workers: int | None=None, shuffle_checkpoint_every: int=10, shuffle_progress_every: int=10, shuffle_resume: bool=True) -> dict:
    project_root = Path(project_root).resolve()
    if stage7_action_table is None:
        stage7_action_table = stage_dir(project_root, 'stage7') / 'llm_stage7_action_table.parquet'
    stage7_action_table = Path(stage7_action_table).resolve()
    if not stage7_action_table.exists():
        raise FileNotFoundError(stage7_action_table)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    space, candidate_provenance = _load_action_space_for_stage7(project_root, stage7_action_table)
    action_table = read_parquet_required(stage7_action_table)
    effective_variant = 'row_shuffle_vector_null' if variant == 'sign_preserving_random_vector' else variant
    explicit_shuffle_permutation = shuffle_seeds is not None
    resolved_shuffle_seeds = [int(x) for x in shuffle_seeds] if shuffle_seeds is not None else [int(random_seed)]
    if not resolved_shuffle_seeds:
        raise ValueError('shuffle_seeds cannot be empty')
    if len(resolved_shuffle_seeds) != len(set(resolved_shuffle_seeds)):
        raise ValueError('shuffle_seeds contains duplicate values')
    if explicit_shuffle_permutation and effective_variant != 'row_shuffle_vector_null':
        raise ValueError('--shuffle-seeds is only valid for row_shuffle_vector_null')
    if shuffle_within != 'none' and effective_variant != 'row_shuffle_vector_null':
        raise ValueError('--shuffle-within is only valid for row_shuffle_vector_null')
    if effective_variant == 'row_shuffle_vector_null':
        shuffle_strata, shuffle_strata_meta = _resolve_shuffle_strata(project_root, action_table, shuffle_within=shuffle_within, raw_root=raw_root)
    else:
        shuffle_strata = pd.Series('__ALL__', index=action_table.index, dtype='object')
        shuffle_strata_meta = {'shuffle_within': 'none', 'stratum_source': 'not_applicable', 'stratum_column': None, 'stratum_count': 1}
    first_seed = int(resolved_shuffle_seeds[0])
    variant_table, rowdiag = _variant_rows(action_table, space=space, policies=policies, modes=modes, variant=variant, target_budget=target_budget, active_eps=active_eps, random_seed=first_seed, shuffle_within=shuffle_within, shuffle_strata=shuffle_strata)
    variant_table_path = output_dir / 'variant_llm_stage7_action_table.parquet'
    variant_table.to_parquet(variant_table_path, index=False)
    rowdiag.to_csv(output_dir / 'variant_row_budget_diagnostics.csv', index=False, encoding='utf-8-sig')
    cols = _action_cols(space)
    budget_rows = []
    for label, table in [('original', action_table), ('variant', variant_table)]:
        mask = table['policy'].astype(str).isin(policies) & table['mode'].astype(str).isin(modes)
        group = table.loc[mask].copy()
        for (policy, mode), sub in group.groupby(['policy', 'mode'], dropna=False):
            budget_rows.append({'table': label, 'policy': str(policy), 'mode': str(mode), 'variant': variant, 'target_budget': target_budget, 'n_rows': int(len(sub)), 'mean_l1': float(_raw_l1(sub, cols).mean()), 'median_l1': float(_raw_l1(sub, cols).median()), 'mean_active_dims': float(_active_dims(sub, cols, active_eps).mean()), 'mean_projection_distance': float(pd.to_numeric(sub.get('projection_distance', np.nan), errors='coerce').mean())})
    budget_summary = pd.DataFrame(budget_rows)
    budget_summary.to_csv(output_dir / 'action_budget_summary.csv', index=False, encoding='utf-8-sig')
    original_target_scores = None
    if write_paired_inference or explicit_shuffle_permutation:
        original_target_table = action_table[action_table['policy'].astype(str).isin(policies) & action_table['mode'].astype(str).isin(modes)].copy()
        if original_target_table.empty:
            raise ValueError('No original target rows are available for paired ablation/permutation inference')
        original_target_scores, _orig_summary, orig_audit, _ = _score_action_table(project_root=project_root, action_table=original_target_table, out=output_dir, space=space, output_prefix='original_target')
        orig_audit.to_parquet(output_dir / 'original_target_action_effect_audit.parquet', index=False)
    merged, summary, audit, simulator_identity = _score_action_table(project_root=project_root, action_table=variant_table, out=output_dir, space=space, output_prefix='ablation')
    audit.to_parquet(output_dir / 'ablation_action_effect_audit.parquet', index=False)
    if effective_variant in {'global_mean_vector_null', 'row_shuffle_vector_null', 'sign_flip_mean_vector'}:
        summary.to_csv(output_dir / 'freeform_null_baseline_summary.csv', index=False, encoding='utf-8-sig')
        budget_summary.to_csv(output_dir / 'freeform_null_budget_summary.csv', index=False, encoding='utf-8-sig')
    paired_outputs: dict[str, str] = {}
    reference_scores = None
    resolved_reference = None
    if write_paired_inference or explicit_shuffle_permutation:
        try:
            reference_scores, resolved_reference = _load_stage6_reference_scores(project_root, reference_policy, space=space)
        except Exception as exc:
            if explicit_shuffle_permutation:
                raise
            write_json(output_dir / 'ablation_paired_vs_reference_error.json', {'status': 'FAIL', 'error': repr(exc)})
    if write_paired_inference:
        target_variant = _target_filter(merged, policies, modes)
        if reference_scores is not None and resolved_reference is not None:
            vs_ref = _paired_against_reference(target_variant, reference_scores, reference_label=resolved_reference, comparison_label=f'variant_vs_{reference_policy}')
            vs_ref.to_csv(output_dir / 'ablation_paired_vs_reference.csv', index=False, encoding='utf-8-sig')
            if reference_policy == 'C3':
                vs_ref.to_csv(output_dir / 'freeform_null_vs_c3_paired.csv', index=False, encoding='utf-8-sig')
            paired_outputs['ablation_paired_vs_reference'] = 'ablation_paired_vs_reference.csv'
        if original_target_scores is not None:
            vs_orig = _paired_against_reference(target_variant, original_target_scores, reference_label='original_target_rows', comparison_label='variant_vs_original_target')
            vs_orig.to_csv(output_dir / 'ablation_paired_vs_original_target.csv', index=False, encoding='utf-8-sig')
            if effective_variant in {'global_mean_vector_null', 'row_shuffle_vector_null', 'sign_flip_mean_vector'}:
                vs_orig.to_csv(output_dir / 'freeform_null_vs_c6_paired.csv', index=False, encoding='utf-8-sig')
            paired_outputs['ablation_paired_vs_original_target'] = 'ablation_paired_vs_original_target.csv'
    permutation_outputs: dict[str, str] = {}
    permutation_runtime: dict[str, object] = {}
    if explicit_shuffle_permutation:
        if original_target_scores is None or reference_scores is None or resolved_reference is None:
            raise RuntimeError('Shuffle permutation inference requires original target and Stage6 reference scores')
        if int(shuffle_checkpoint_every) <= 0 or int(shuffle_progress_every) <= 0:
            raise ValueError('shuffle checkpoint/progress intervals must be positive integers')
        first_draw = _shuffle_draw_summary(shuffled_scores=merged, original_scores=original_target_scores, reference_scores=reference_scores, reference_label=resolved_reference, policies=policies, modes=modes, seed=first_seed, shuffle_within=shuffle_within, row_diagnostics=rowdiag)
        partial_path = output_dir / 'shuffle_per_draw_summary.partial.csv'
        checkpoint_path = output_dir / 'shuffle_permutation_checkpoint.json'
        signature = _checkpoint_signature(stage7_action_table=stage7_action_table, variant=variant, target_budget=target_budget, policies=policies, modes=modes, shuffle_within=shuffle_within, seeds=resolved_shuffle_seeds, reference_policy=reference_policy)
        existing = _load_shuffle_checkpoint(partial_path=partial_path, checkpoint_path=checkpoint_path, expected_signature=signature) if shuffle_resume else pd.DataFrame()
        if not shuffle_resume and (partial_path.exists() or checkpoint_path.exists()):
            raise RuntimeError('Shuffle checkpoint exists but resume is disabled. Remove/archive the checkpoint explicitly rather than silently overwriting it.')
        draw_frames: list[pd.DataFrame] = []
        if not existing.empty:
            draw_frames.append(existing)
        completed_seeds = set(existing['shuffle_seed'].astype(int).tolist()) if not existing.empty else set()
        if first_seed not in completed_seeds:
            draw_frames.append(first_draw)
            completed_seeds.add(first_seed)
        combined = _write_shuffle_checkpoint(frames=draw_frames, partial_path=partial_path, checkpoint_path=checkpoint_path, signature=signature)
        remaining_seeds = [int(x) for x in resolved_shuffle_seeds if int(x) not in completed_seeds]
        requested_workers = int(shuffle_workers) if shuffle_workers is not None else 0
        worker_source = 'argument' if requested_workers else 'auto'
        env_workers_text = os.getenv('CREDIT_RECOURSE_SHUFFLE_WORKERS', '').strip()
        if requested_workers == 0 and env_workers_text:
            try:
                requested_workers = int(env_workers_text)
            except ValueError as exc:
                raise ValueError('CREDIT_RECOURSE_SHUFFLE_WORKERS must be a non-negative integer') from exc
            worker_source = 'environment'
        if requested_workers < 0:
            raise ValueError('shuffle_workers must be >= 0; use 0 for automatic selection')
        auto_workers = max(1, min(8, (os.cpu_count() or 2) - 1))
        resolved_workers = requested_workers or auto_workers
        resolved_workers = min(resolved_workers, max(1, len(remaining_seeds)))
        original_cache = output_dir / '.shuffle_original_target_scores.parquet'
        reference_cache = output_dir / '.shuffle_reference_scores.parquet'
        original_target_scores.to_parquet(original_cache, index=False)
        reference_scores.to_parquet(reference_cache, index=False)
        worker_payload = {'project_root': str(project_root), 'stage7_action_table': str(stage7_action_table), 'original_scores_path': str(original_cache), 'reference_scores_path': str(reference_cache), 'policies': list(policies), 'modes': list(modes), 'variant': variant, 'target_budget': target_budget, 'active_eps': float(active_eps), 'shuffle_within': shuffle_within, 'shuffle_strata_values': [str(x) for x in shuffle_strata.astype('string').tolist()], 'resolved_reference': resolved_reference, 'simulator_identity': simulator_identity}
        started = datetime.now(timezone.utc)
        completed_since_write = 0
        if remaining_seeds:
            print(f'[shuffle] start draws={len(resolved_shuffle_seeds)} completed={len(completed_seeds)} remaining={len(remaining_seeds)} workers={resolved_workers} target_rows_only=True', flush=True)
            if resolved_workers == 1:
                _init_shuffle_worker(worker_payload)
                seed_results = ((seed, _score_shuffle_seed_worker(seed)) for seed in remaining_seeds)
                for seed, records in seed_results:
                    draw_frames.append(pd.DataFrame.from_records(records))
                    completed_seeds.add(int(seed))
                    completed_since_write += 1
                    if completed_since_write >= int(shuffle_checkpoint_every):
                        combined = _write_shuffle_checkpoint(frames=draw_frames, partial_path=partial_path, checkpoint_path=checkpoint_path, signature=signature)
                        draw_frames = [combined]
                        completed_since_write = 0
                    if len(completed_seeds) % int(shuffle_progress_every) == 0:
                        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                        print(f'[shuffle] completed={len(completed_seeds)}/{len(resolved_shuffle_seeds)} elapsed_sec={elapsed:.1f}', flush=True)
            else:
                with ProcessPoolExecutor(max_workers=resolved_workers, initializer=_init_shuffle_worker, initargs=(worker_payload,)) as executor:
                    futures = {executor.submit(_score_shuffle_seed_worker, seed): seed for seed in remaining_seeds}
                    for future in as_completed(futures):
                        seed = futures[future]
                        records = future.result()
                        draw_frames.append(pd.DataFrame.from_records(records))
                        completed_seeds.add(int(seed))
                        completed_since_write += 1
                        if completed_since_write >= int(shuffle_checkpoint_every):
                            combined = _write_shuffle_checkpoint(frames=draw_frames, partial_path=partial_path, checkpoint_path=checkpoint_path, signature=signature)
                            draw_frames = [combined]
                            completed_since_write = 0
                        if len(completed_seeds) % int(shuffle_progress_every) == 0:
                            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
                            print(f'[shuffle] completed={len(completed_seeds)}/{len(resolved_shuffle_seeds)} elapsed_sec={elapsed:.1f}', flush=True)
        per_draw = _write_shuffle_checkpoint(frames=draw_frames, partial_path=partial_path, checkpoint_path=checkpoint_path, signature=signature)
        observed_seeds = set(per_draw['shuffle_seed'].astype(int).tolist())
        expected_seeds = set((int(x) for x in resolved_shuffle_seeds))
        if observed_seeds != expected_seeds:
            raise RuntimeError({'message': 'Shuffle permutation completed with an incomplete seed set', 'missing': sorted(expected_seeds - observed_seeds)[:50], 'unexpected': sorted(observed_seeds - expected_seeds)[:50]})
        aggregate = _aggregate_shuffle_draws(per_draw)
        per_draw.to_csv(output_dir / 'shuffle_per_draw_summary.csv', index=False, encoding='utf-8-sig')
        aggregate.to_csv(output_dir / 'shuffle_permutation_ci.csv', index=False, encoding='utf-8-sig')
        final_checkpoint = dict(signature)
        final_checkpoint.update({'status': 'PASS', 'completed_utc': _now(), 'completed_seed_count': len(expected_seeds), 'completed_seeds': sorted(expected_seeds), 'workers': int(resolved_workers), 'target_rows_only': True, 'target_row_count': int(len(_target_filter(action_table, policies, modes))), 'full_stage7_row_count': int(len(action_table))})
        write_json(checkpoint_path, final_checkpoint)
        partial_path.unlink(missing_ok=True)
        original_cache.unlink(missing_ok=True)
        reference_cache.unlink(missing_ok=True)
        permutation_outputs = {'shuffle_per_draw_summary': 'shuffle_per_draw_summary.csv', 'shuffle_permutation_ci': 'shuffle_permutation_ci.csv', 'shuffle_checkpoint': 'shuffle_permutation_checkpoint.json'}
        permutation_runtime = {'shuffle_engine': 'target_rows_parallel_v2', 'shuffle_workers': int(resolved_workers), 'shuffle_worker_source': worker_source, 'shuffle_checkpoint_every': int(shuffle_checkpoint_every), 'shuffle_progress_every': int(shuffle_progress_every), 'shuffle_resume_enabled': bool(shuffle_resume), 'target_rows_only': True, 'target_row_count': int(len(_target_filter(action_table, policies, modes))), 'full_stage7_row_count': int(len(action_table))}
    meta = {'stage': 'analysis.llm_action_budget_ablation', 'status': 'PASS', 'created_utc': _now(), 'project_root': str(project_root), 'source_stage7_action_table': str(stage7_action_table), 'variant': variant, 'target_budget': target_budget, 'policies': policies, 'modes': modes, 'active_eps': float(active_eps), 'score_tie_contract': score_tie_contract_metadata(), 'random_seed': first_seed, 'shuffle_permutation_requested': bool(explicit_shuffle_permutation), 'shuffle_seeds': resolved_shuffle_seeds if explicit_shuffle_permutation else [first_seed], 'shuffle_draw_count': len(resolved_shuffle_seeds) if explicit_shuffle_permutation else 1, **permutation_runtime, **shuffle_strata_meta, 'reference_policy_requested': reference_policy, 'reference_policy_resolved': resolved_reference, 'write_paired_inference': bool(write_paired_inference), 'candidate_library_path': candidate_provenance['candidate_library_path'], 'candidate_library_quantile': candidate_provenance.get('candidate_library_quantile'), 'candidate_action_values_source': candidate_provenance.get('candidate_action_values_source'), 'base_candidate_library_path': candidate_provenance.get('base_candidate_library_path'), 'final_action_contract_path': candidate_provenance['final_action_contract_path'], 'candidate_library_provenance_metadata_paths': candidate_provenance['candidate_library_provenance_metadata_paths'], 'candidate_library_provenance_validation': candidate_provenance['candidate_library_provenance_validation'], 'stage6_simulator_identity': simulator_identity, 'note': 'Evaluator-only post-hoc ablation/null baseline. No LLM API calls; no RL retraining. Not a canonical Stage8/9 overwrite.', 'outputs': {'variant_action_table': 'variant_llm_stage7_action_table.parquet', 'row_budget_diagnostics': 'variant_row_budget_diagnostics.csv', 'action_budget_summary': 'action_budget_summary.csv', 'ablation_scores': 'ablation_multi_oracle_scores.parquet', 'ablation_policy_summary': 'ablation_policy_summary.csv', 'action_effect_audit': 'ablation_action_effect_audit.parquet', **paired_outputs, **permutation_outputs}}
    write_json(output_dir / 'metadata.json', meta)
    return meta

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='LLM free-form action-budget ablation/null-baseline scorer')
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--raw-root', required=True, help='Explicit read-only raw input root')
    ap.add_argument('--stage7-action-table', default=None, help='Optional explicit llm_stage7_action_table.parquet')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--policies', default='C6', help='Comma-separated policies to transform; default C6')
    ap.add_argument('--modes', default='free_form_10d', help='Comma-separated modes to transform; default free_form_10d')
    ap.add_argument('--variant', required=True, choices=sorted(SUPPORTED_VARIANTS))
    ap.add_argument('--target-budget', default='same_policy_candidate_mean', help='native, same_policy_candidate_mean, all_candidate_mean, or fixed:<float>')
    ap.add_argument('--active-eps', type=float, default=1e-09)
    ap.add_argument('--random-seed', type=int, default=1, help='Single-draw seed; also used when --shuffle-seeds is omitted')
    ap.add_argument('--shuffle-seeds', default=None, help='Repeated row-shuffle seeds, e.g. 1:1000 or 1,7,11. Only valid for row_shuffle_vector_null.')
    ap.add_argument('--shuffle-within', choices=SHUFFLE_WITHIN_CHOICES, default='none', help='Condition row shuffling within frozen industry or rating-band strata')
    ap.add_argument('--reference-policy', default='C3', help='Stage6 reference policy for paired inference. C3 resolves to the active final RL label.')
    ap.add_argument('--shuffle-workers', type=int, default=0, help='Parallel permutation workers; 0=auto (up to 8). Environment override: CREDIT_RECOURSE_SHUFFLE_WORKERS.')
    ap.add_argument('--shuffle-checkpoint-every', type=int, default=10, help='Persist permutation progress after this many completed draws.')
    ap.add_argument('--shuffle-progress-every', type=int, default=10, help='Print permutation progress after this many total completed seeds.')
    ap.add_argument('--no-shuffle-resume', action='store_true', help='Disable checkpoint resume; fails if an existing checkpoint is present.')
    ap.add_argument('--no-paired-inference', action='store_true', help='Skip paired inference output files.')
    args = ap.parse_args(argv)
    meta = run_ablation(project_root=Path(args.project_root), raw_root=Path(args.raw_root), stage7_action_table=Path(args.stage7_action_table) if args.stage7_action_table else None, output_dir=Path(args.output_dir), policies=_parse_csv_arg(args.policies), modes=_parse_csv_arg(args.modes), variant=args.variant, target_budget=args.target_budget, active_eps=args.active_eps, random_seed=args.random_seed, shuffle_seeds=_parse_seed_spec(args.shuffle_seeds) if args.shuffle_seeds is not None else None, shuffle_within=args.shuffle_within, reference_policy=args.reference_policy, write_paired_inference=not args.no_paired_inference, shuffle_workers=args.shuffle_workers, shuffle_checkpoint_every=args.shuffle_checkpoint_every, shuffle_progress_every=args.shuffle_progress_every, shuffle_resume=not args.no_shuffle_resume)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
