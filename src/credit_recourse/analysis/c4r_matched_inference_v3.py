from __future__ import annotations
'Multi-arm, multi-cohort matched C4/C4R/C6 inference for journal extension.\n\nThis module extends, but does not modify, the frozen thesis producer\n``c4r_matched_inference.py`` (v2).  It accepts one or more backend/replication\ncohorts, any preregistered set of finite L1 arms plus an unbounded reference\narm, and preserves the exact decomposition\n\n    C6 - C4 = (C4R - C4) + (C6 - C4R)\n\nfor every firm and Oracle before aggregate inference is produced.\n\nPrimary interaction multiplicity control follows the preregistered journal\ncontract: for each cohort, finite arm, and decomposition component, Holm\ncorrection is applied across the three Oracles.  A separately labelled\nsensitivity column applies Holm across every finite-arm x Oracle interaction\nwithin a component.\n\nCLI arm syntax is repeatable and explicit::\n\n    --arm "gpt54mini|0p75=C:\\...\\archive"\n    --arm "gpt54mini|unbounded=C:\\...\\archive"\n\nThe v2 producer remains the canonical thesis artifact and is intentionally\nleft untouched.\n'
import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from credit_recourse.analysis.n5_7_10c_holm_inference import ORACLES, holm_adjust, p_to_stars, wilcoxon_paired_p
SCHEMA_VERSION = 'c4r_matched_inference_v3'
CONDITIONS = ('C4', 'C4R', 'C6')
MODE = 'free_form_10d'
IDENTITY_TOLERANCE = 1e-12
REVISION_SCORE_TOLERANCE = 1e-09
DEFAULT_BUDGET_SPECS: tuple[tuple[str, float | None], ...] = (('0p75', 0.75), ('1p27', 1.27), ('2p00', 2.0), ('unbounded', None))
CONTRAST_COLUMNS: dict[str, str] = {'self_revision': 'self_revision', 'reference_content_conditional': 'reference_content_conditional', 'reference_plus_revision_package': 'package'}

@dataclass(frozen=True)
class ArmSpec:
    cohort_id: str
    budget_label: str
    path: Path

@dataclass(frozen=True)
class BudgetSpec:
    label: str
    value: float | None

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8-sig'))

def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    if path.is_file():
        try:
            return pd.read_parquet(path)
        except ImportError:
            return pd.read_csv(path)
    csv_path = path.with_suffix('.csv')
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(path)

def _inner_archive_dir(path: Path) -> Path:
    root = Path(path).resolve()
    if (root / 'stage7_llm_action_generation').is_dir():
        return root
    if not root.is_dir():
        raise FileNotFoundError(root)
    matches = [p for p in root.iterdir() if p.is_dir() and (p / 'stage7_llm_action_generation').is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Could not resolve one Stage7-9 archive under {root}: found={len(matches)}')
    return matches[0]

def _safe_token(value: str, *, field: str) -> str:
    token = str(value).strip()
    if not token or not re.fullmatch('[A-Za-z0-9_.-]+', token):
        raise ValueError(f'{field} must be a non-empty directory-safe token; got {value!r}')
    return token

def parse_arm_spec(text: str) -> ArmSpec:
    left, sep, raw_path = str(text).partition('=')
    if sep != '=' or not raw_path.strip():
        raise ValueError(f'--arm must use cohort|budget=path syntax; got {text!r}')
    cohort, bar, budget = left.partition('|')
    if bar != '|':
        raise ValueError(f'--arm must use cohort|budget=path syntax; got {text!r}')
    return ArmSpec(cohort_id=_safe_token(cohort, field='cohort_id'), budget_label=_safe_token(budget, field='budget_label'), path=Path(raw_path.strip()))

def parse_budget_spec(text: str) -> BudgetSpec:
    label, sep, raw_value = str(text).partition('=')
    if sep != '=':
        raise ValueError(f'--budget-spec must use label=value syntax; got {text!r}')
    label = _safe_token(label, field='budget_label')
    raw_value = raw_value.strip().lower()
    value = None if raw_value in {'none', 'null', 'unbounded'} else float(raw_value)
    if value is not None and (not np.isfinite(value) or value <= 0):
        raise ValueError(f'finite budget must be positive and finite; got {value}')
    return BudgetSpec(label=label, value=value)

def parse_expected_role(text: str) -> tuple[str, str]:
    cohort, sep, role = str(text).partition('=')
    if sep != '=' or not role.strip():
        raise ValueError(f'--expected-run-role must use cohort=role syntax; got {text!r}')
    return (_safe_token(cohort, field='cohort_id'), role.strip())

def _budget_map(specs: Sequence[BudgetSpec]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for spec in specs:
        if spec.label in out:
            raise ValueError(f'duplicate budget label: {spec.label}')
        out[spec.label] = spec.value
    unbounded = [label for label, value in out.items() if value is None]
    if len(unbounded) != 1:
        raise ValueError(f'budget contract requires exactly one unbounded label; found={unbounded}')
    finite_values = [float(value) for value in out.values() if value is not None]
    if len(set(finite_values)) != len(finite_values):
        raise ValueError('finite budget values must be unique')
    return out

def _validate_budget_meta(meta: Mapping[str, Any], *, label: str, expected: float | None) -> float | None:
    contract = meta.get('action_budget_contract') or {}
    enabled = bool(contract.get('enabled'))
    if expected is None:
        if enabled:
            raise ValueError(f'unbounded arm {label!r} must not enable action-budget contract')
        return None
    if not enabled:
        raise ValueError(f'finite arm {label!r} must enable action-budget contract')
    observed = float(contract.get('l1_budget'))
    if not np.isclose(observed, expected, atol=1e-12, rtol=0):
        raise ValueError(f'budget mismatch for {label}: expected={expected}, observed={observed}')
    budgeted = set(map(str, contract.get('budgeted_conditions') or []))
    if budgeted != set(CONDITIONS):
        raise ValueError(f'finite arm {label} must budget C4/C4R/C6 equally; got {sorted(budgeted)}')
    if set(map(str, contract.get('budgeted_modes') or [])) != {MODE}:
        raise ValueError(f'finite arm {label} budgeted_modes must be {MODE} only')
    return observed

def _load_arm(spec: ArmSpec, *, budget_specs: Mapping[str, float | None], expected_role: str | None, information_condition: str, require_live: bool) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if spec.budget_label not in budget_specs:
        raise ValueError(f'arm uses undeclared budget label {spec.budget_label!r}')
    run_dir = _inner_archive_dir(spec.path)
    archive_path = run_dir / 'archive_manifest.json'
    stage7_meta_path = run_dir / 'stage7_llm_action_generation' / 'metadata.json'
    stage8_path = run_dir / 'stage8_llm_multi_oracle_eval' / 'llm_stage8_multi_oracle_scores.parquet'
    stage9_path = run_dir / 'stage9_llm_rl_comparison' / 'llm_stage9_revision_metrics.csv'
    for required in (archive_path, stage7_meta_path, stage9_path):
        if not required.is_file():
            raise FileNotFoundError(f'C4R v3 archive missing required artifact: {required}')
    archive = _read_json(archive_path)
    meta = _read_json(stage7_meta_path)
    run_role = str(archive.get('run_role') or '')
    if expected_role is not None and run_role != expected_role:
        raise ValueError(f'cohort={spec.cohort_id} run_role mismatch: expected={expected_role!r}, observed={run_role!r}')
    backend_is_live = bool(meta.get('backend_is_live'))
    allowed_statuses = {'PASS'} if require_live else {'PASS', 'PASS_REPRODUCIBILITY_BACKEND'}
    if str(meta.get('status')) not in allowed_statuses or (require_live and (not backend_is_live)):
        raise ValueError(f"cohort={spec.cohort_id} archive status/live mismatch: status={meta.get('status')!r}, live={meta.get('backend_is_live')!r}, require_live={require_live}")
    if meta.get('information_condition') != information_condition:
        raise ValueError(f"cohort={spec.cohort_id} information condition mismatch: expected={information_condition}, observed={meta.get('information_condition')!r}")
    if tuple(meta.get('conditions') or []) != CONDITIONS:
        raise ValueError(f"conditions must be exactly {CONDITIONS}: {meta.get('conditions')!r}")
    if tuple(meta.get('modes') or []) != (MODE,):
        raise ValueError(f"mode must be exactly {(MODE,)}: {meta.get('modes')!r}")
    prompt_contract = meta.get('prompt_payload_archive_contract') or {}
    if prompt_contract and prompt_contract.get('status') != 'FULL_PAYLOAD_ARCHIVED':
        raise ValueError('prompt payload archive contract is present but not FULL_PAYLOAD_ARCHIVED')
    observed_budget = _validate_budget_meta(meta, label=spec.budget_label, expected=budget_specs[spec.budget_label])
    score_storage_path = stage8_path if stage8_path.is_file() else stage8_path.with_suffix('.csv')
    scores = _read_parquet_or_csv(stage8_path)
    revision = pd.read_csv(stage9_path)
    arm = {'cohort_id': spec.cohort_id, 'run_dir': str(run_dir), 'run_label': archive.get('run_label') or run_dir.name, 'run_role': run_role, 'budget_label': spec.budget_label, 'l1_budget': observed_budget, 'backend_id': meta.get('backend_id'), 'backend_is_live': backend_is_live}
    return (arm, scores, revision)

def _arm_firm_frame(arm: Mapping[str, Any], scores: pd.DataFrame, revision: pd.DataFrame, *, expected_firm_count: int) -> pd.DataFrame:
    required_scores = {'row_id', 'policy', 'mode', *[f'delta_R_score_{o}' for o in ORACLES]}
    missing = sorted(required_scores - set(scores.columns))
    if missing:
        raise ValueError(f'C4R v3 Stage8 scores missing columns: {missing}')
    selected = scores.loc[scores['policy'].astype(str).isin(CONDITIONS) & scores['mode'].astype(str).eq(MODE)].copy()
    selected['row_id'] = pd.to_numeric(selected['row_id'], errors='raise').astype(int)
    expected_score_rows = expected_firm_count * len(CONDITIONS)
    if len(selected) != expected_score_rows or selected.duplicated(['row_id', 'policy', 'mode']).any():
        raise ValueError(f'Stage8 expected {expected_score_rows} unique rows, found {len(selected)}')
    universes = {condition: set(selected.loc[selected['policy'].astype(str).eq(condition), 'row_id']) for condition in CONDITIONS}
    if any((len(values) != expected_firm_count for values in universes.values())) or len({frozenset(v) for v in universes.values()}) != 1:
        raise ValueError('Stage8 C4/C4R/C6 row alignment failed')
    wide_parts: list[pd.DataFrame] = []
    for condition in CONDITIONS:
        part = selected.loc[selected['policy'].astype(str).eq(condition), ['row_id', *[f'delta_R_score_{o}' for o in ORACLES]]].copy()
        part = part.rename(columns={f'delta_R_score_{o}': f'{condition}_{o}' for o in ORACLES})
        wide_parts.append(part)
    frame = wide_parts[0]
    for part in wide_parts[1:]:
        frame = frame.merge(part, on='row_id', how='inner', validate='one_to_one')
    required_revision = {'row_id', 'base_condition', 'revision_condition', 'mode', 'reference_source', 'metrics_defined', 'undefined_reason', *[f'revision_delta_R_score_{o}' for o in ORACLES]}
    missing_revision = sorted(required_revision - set(revision.columns))
    if missing_revision:
        raise ValueError(f'C4R v3 Stage9 revision metrics missing columns: {missing_revision}')
    rev = revision.loc[revision['revision_condition'].astype(str).isin(['C4R', 'C6']) & revision['mode'].astype(str).eq(MODE)].copy()
    rev['row_id'] = pd.to_numeric(rev['row_id'], errors='raise').astype(int)
    expected_grid = {(rid, condition) for rid in universes['C4'] for condition in ('C4R', 'C6')}
    observed_grid = set(map(tuple, rev[['row_id', 'revision_condition']].astype({'row_id': int, 'revision_condition': str}).itertuples(index=False, name=None)))
    if observed_grid != expected_grid or rev.duplicated(['row_id', 'revision_condition', 'mode']).any():
        raise ValueError(f'Stage9 revision grid is not exact {expected_firm_count} x {{C4R,C6}}')
    c4r = rev.loc[rev['revision_condition'].astype(str).eq('C4R')]
    if not c4r['base_condition'].astype(str).eq('C4').all() or not c4r['reference_source'].astype(str).eq('none').all():
        raise ValueError('C4R Stage9 rows violate reference-free C4 pairing')
    if c4r['metrics_defined'].fillna(False).astype(bool).any() or not c4r['undefined_reason'].astype(str).eq('no_rl_reference').all():
        raise ValueError('C4R RL-adoption geometry must be undefined with no_rl_reference')
    for revision_condition in ('C4R', 'C6'):
        sub = rev.loc[rev['revision_condition'].astype(str).eq(revision_condition)].set_index('row_id')
        indexed = frame.set_index('row_id')
        for oracle in ORACLES:
            observed = pd.to_numeric(sub[f'revision_delta_R_score_{oracle}'], errors='raise').sort_index()
            expected = (indexed[f'{revision_condition}_{oracle}'] - indexed[f'C4_{oracle}']).sort_index()
            max_gap = float((observed - expected).abs().max())
            if max_gap > REVISION_SCORE_TOLERANCE:
                raise AssertionError(f'Stage8/9 revision score mismatch condition={revision_condition} oracle={oracle} max_abs_gap={max_gap:.3e}')
    frame.insert(0, 'cohort_id', arm['cohort_id'])
    frame.insert(1, 'backend_id', arm['backend_id'])
    frame.insert(2, 'run_role', arm['run_role'])
    frame.insert(3, 'run_label', arm['run_label'])
    frame.insert(4, 'budget_label', arm['budget_label'])
    frame.insert(5, 'l1_budget', np.nan if arm['l1_budget'] is None else float(arm['l1_budget']))
    for oracle in ORACLES:
        frame[f'self_revision_{oracle}'] = frame[f'C4R_{oracle}'] - frame[f'C4_{oracle}']
        frame[f'reference_content_conditional_{oracle}'] = frame[f'C6_{oracle}'] - frame[f'C4R_{oracle}']
        frame[f'package_{oracle}'] = frame[f'C6_{oracle}'] - frame[f'C4_{oracle}']
        gap = (frame[f'self_revision_{oracle}'] + frame[f'reference_content_conditional_{oracle}'] - frame[f'package_{oracle}']).abs()
        if float(gap.max()) > IDENTITY_TOLERANCE:
            raise AssertionError(f'C4R additive identity failed for {oracle}: max_abs_gap={gap.max():.3e}')
    return frame.sort_values('row_id').reset_index(drop=True)

def _contrast_table(firm_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = ['cohort_id', 'backend_id', 'run_role', 'budget_label', 'l1_budget']
    for keys, group in firm_frame.groupby(group_columns, dropna=False, sort=False):
        cohort_id, backend_id, run_role, budget_label, l1_budget = keys
        for oracle in ORACLES:
            for contrast, prefix in CONTRAST_COLUMNS.items():
                values = pd.to_numeric(group[f'{prefix}_{oracle}'], errors='raise')
                rows.append({'cohort_id': cohort_id, 'backend_id': backend_id, 'run_role': run_role, 'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'contrast': contrast, 'n_pairs': int(len(values)), 'mean_increment': float(values.mean()), 'median_increment': float(values.median()), 'win_count': int((values > 0).sum()), 'tie_count': int((values == 0).sum()), 'loss_count': int((values < 0).sum()), 'wilcoxon_p_raw': wilcoxon_paired_p(values)})
    out = pd.DataFrame(rows)
    out['wilcoxon_p_holm_across_budgets'] = np.nan
    out['holm_family_across_budgets'] = 'cohort_oracle_contrast_across_budget_arms'
    for _, idx in out.groupby(['cohort_id', 'oracle_backend', 'contrast'], sort=False).groups.items():
        out.loc[idx, 'wilcoxon_p_holm_across_budgets'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].fillna(1.0).tolist())
    out['sig_holm_across_budgets'] = out['wilcoxon_p_holm_across_budgets'].map(p_to_stars)
    return out.sort_values(['cohort_id', 'contrast', 'oracle_backend', 'l1_budget'], na_position='last').reset_index(drop=True)

def _interaction_table(firm_frame: pd.DataFrame, *, budget_specs: Mapping[str, float | None], unbounded_label: str, expected_firm_count: int) -> pd.DataFrame:
    required = {'cohort_id', 'row_id', 'budget_label'}
    for prefix in CONTRAST_COLUMNS.values():
        required.update((f'{prefix}_{oracle}' for oracle in ORACLES))
    missing = sorted(required - set(firm_frame.columns))
    if missing:
        raise ValueError(f'firm frame missing interaction columns: {missing}')
    if firm_frame.duplicated(['cohort_id', 'row_id', 'budget_label']).any():
        raise ValueError('interaction frame has duplicate cohort_id/row_id/budget_label keys')
    finite_labels = [label for label, value in budget_specs.items() if value is not None]
    rows: list[dict[str, Any]] = []
    for cohort_id, cohort in firm_frame.groupby('cohort_id', sort=False):
        unbounded = cohort.loc[cohort['budget_label'].astype(str).eq(unbounded_label)].copy()
        if len(unbounded) != expected_firm_count:
            raise ValueError(f'cohort={cohort_id} unbounded row count must be {expected_firm_count}; got {len(unbounded)}')
        for finite_label in finite_labels:
            finite = cohort.loc[cohort['budget_label'].astype(str).eq(finite_label)].copy()
            if len(finite) != expected_firm_count:
                raise ValueError(f'cohort={cohort_id} finite arm={finite_label} row count must be {expected_firm_count}; got {len(finite)}')
            if set(finite['row_id'].astype(int)) != set(unbounded['row_id'].astype(int)):
                raise ValueError(f'cohort={cohort_id} finite={finite_label}/unbounded row universes differ')
            paired = finite.merge(unbounded, on='row_id', how='inner', validate='one_to_one', suffixes=('__finite', '__unbounded')).sort_values('row_id')
            for contrast, prefix in CONTRAST_COLUMNS.items():
                for oracle in ORACLES:
                    finite_values = pd.to_numeric(paired[f'{prefix}_{oracle}__finite'], errors='raise')
                    unbounded_values = pd.to_numeric(paired[f'{prefix}_{oracle}__unbounded'], errors='raise')
                    did = finite_values - unbounded_values
                    rows.append({'cohort_id': cohort_id, 'backend_id': paired['backend_id__finite'].iloc[0], 'run_role': paired['run_role__finite'].iloc[0], 'oracle_backend': oracle, 'contrast': contrast, 'finite_budget_label': finite_label, 'finite_l1_budget': budget_specs[finite_label], 'unbounded_budget_label': unbounded_label, 'n_pairs': int(len(did)), 'finite_mean_increment': float(finite_values.mean()), 'unbounded_mean_increment': float(unbounded_values.mean()), 'finite_minus_unbounded_interaction': float(did.mean()), 'median_interaction': float(did.median()), 'win_count': int((did > 0).sum()), 'tie_count': int((did == 0).sum()), 'loss_count': int((did < 0).sum()), 'wilcoxon_p_raw': wilcoxon_paired_p(did), 'holm_family_primary': 'cohort_finite_arm_contrast_across_three_oracles', 'holm_family_sensitivity': 'cohort_contrast_across_all_finite_arms_and_oracles'})
    out = pd.DataFrame(rows)
    out['wilcoxon_p_holm_three_oracles'] = np.nan
    for _, idx in out.groupby(['cohort_id', 'finite_budget_label', 'contrast'], sort=False).groups.items():
        out.loc[idx, 'wilcoxon_p_holm_three_oracles'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].fillna(1.0).tolist())
    out['sig_holm_three_oracles'] = out['wilcoxon_p_holm_three_oracles'].map(p_to_stars)
    out['wilcoxon_p_holm_component_all_finite_oracles'] = np.nan
    for _, idx in out.groupby(['cohort_id', 'contrast'], sort=False).groups.items():
        out.loc[idx, 'wilcoxon_p_holm_component_all_finite_oracles'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].fillna(1.0).tolist())
    out['sig_holm_component_all_finite_oracles'] = out['wilcoxon_p_holm_component_all_finite_oracles'].map(p_to_stars)
    return out.sort_values(['cohort_id', 'contrast', 'finite_l1_budget', 'oracle_backend']).reset_index(drop=True)

def _v2_compat_projection(firm_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return exact v2-shaped artifacts for a single 0p75/unbounded cohort.

    This is a regression aid only.  It does not replace the v3 combined tables.
    """
    cohort_ids = firm_frame['cohort_id'].astype(str).unique().tolist()
    labels = set(firm_frame['budget_label'].astype(str))
    if len(cohort_ids) != 1 or labels != {'0p75', 'unbounded'}:
        raise ValueError('v2 compatibility projection requires one cohort with exactly 0p75 and unbounded')
    extra = {'cohort_id', 'backend_id', 'run_role'}
    v2_frame = firm_frame.drop(columns=[column for column in extra if column in firm_frame.columns]).copy()
    ordered_front = ['run_label', 'budget_label', 'l1_budget', 'row_id']
    v2_frame = v2_frame[ordered_front + [c for c in v2_frame.columns if c not in ordered_front]]
    rows: list[dict[str, Any]] = []
    for (budget_label, l1_budget), group in v2_frame.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for oracle in ORACLES:
            for contrast, prefix in CONTRAST_COLUMNS.items():
                values = pd.to_numeric(group[f'{prefix}_{oracle}'], errors='raise')
                rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'contrast': contrast, 'n_pairs': int(len(values)), 'mean_increment': float(values.mean()), 'median_increment': float(values.median()), 'win_count': int((values > 0).sum()), 'tie_count': int((values == 0).sum()), 'loss_count': int((values < 0).sum()), 'wilcoxon_p_raw': wilcoxon_paired_p(values)})
    contrasts = pd.DataFrame(rows)
    contrasts['wilcoxon_p_holm_two_budgets'] = np.nan
    for _, idx in contrasts.groupby(['oracle_backend', 'contrast'], sort=False).groups.items():
        contrasts.loc[idx, 'wilcoxon_p_holm_two_budgets'] = holm_adjust(contrasts.loc[idx, 'wilcoxon_p_raw'].fillna(1.0).tolist())
    contrasts['sig_holm'] = contrasts['wilcoxon_p_holm_two_budgets'].map(p_to_stars)
    contrasts = contrasts.sort_values(['contrast', 'oracle_backend', 'l1_budget'], na_position='last').reset_index(drop=True)
    interactions = _interaction_table(firm_frame, budget_specs={'0p75': 0.75, 'unbounded': None}, unbounded_label='unbounded', expected_firm_count=int(v2_frame['row_id'].nunique()))
    interactions = interactions.rename(columns={'holm_family_primary': 'holm_family', 'sig_holm_three_oracles': 'sig_holm'})[['oracle_backend', 'contrast', 'finite_budget_label', 'unbounded_budget_label', 'n_pairs', 'finite_mean_increment', 'unbounded_mean_increment', 'finite_minus_unbounded_interaction', 'median_interaction', 'win_count', 'tie_count', 'loss_count', 'wilcoxon_p_raw', 'holm_family', 'wilcoxon_p_holm_three_oracles', 'sig_holm']].copy()
    interactions['holm_family'] = 'contrast_across_three_oracles'
    interactions = interactions.sort_values(['contrast', 'oracle_backend']).reset_index(drop=True)
    return (v2_frame, contrasts, interactions)

def _write_frame(frame: pd.DataFrame, path: Path) -> tuple[Path, str]:
    try:
        frame.to_parquet(path, index=False)
        return (path, 'parquet')
    except ImportError:
        csv_path = path.with_suffix('.csv')
        frame.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return (csv_path, 'csv_fallback_missing_parquet_engine')

def run_analysis(*, arm_specs: Sequence[ArmSpec], out_dir: Path, budget_specs: Sequence[BudgetSpec] | None=None, expected_run_roles: Mapping[str, str] | None=None, information_condition: str='IC-b', expected_firm_count: int=575, require_live: bool=True, prereg_path: Path | None=None, write_v2_compat_projection: bool=True) -> dict[str, Any]:
    if expected_firm_count <= 0:
        raise ValueError('expected_firm_count must be positive')
    specs = list(budget_specs or [BudgetSpec(label, value) for label, value in DEFAULT_BUDGET_SPECS])
    budget_map = _budget_map(specs)
    unbounded_label = next((label for label, value in budget_map.items() if value is None))
    if not arm_specs:
        raise ValueError('at least one --arm is required')
    keys = [(spec.cohort_id, spec.budget_label) for spec in arm_specs]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate cohort_id/budget_label arm specification')
    roles = dict(expected_run_roles or {})
    cohorts = sorted({spec.cohort_id for spec in arm_specs})
    unknown_role_cohorts = sorted(set(roles) - set(cohorts))
    if unknown_role_cohorts:
        raise ValueError(f'expected run roles declared for absent cohorts: {unknown_role_cohorts}')
    for cohort in cohorts:
        labels = {spec.budget_label for spec in arm_specs if spec.cohort_id == cohort}
        if labels != set(budget_map):
            raise ValueError(f'cohort={cohort} budget grid mismatch: expected={sorted(budget_map)}, observed={sorted(labels)}')
    loaded = [_load_arm(spec, budget_specs=budget_map, expected_role=roles.get(spec.cohort_id), information_condition=information_condition, require_live=require_live) for spec in arm_specs]
    frames = [_arm_firm_frame(*item, expected_firm_count=expected_firm_count) for item in loaded]
    global_universes = {frozenset(frame['row_id'].astype(int)) for frame in frames}
    if len(global_universes) != 1 or len(next(iter(global_universes))) != expected_firm_count:
        raise ValueError('all arms/cohorts must share the same firm row universe')
    for cohort in cohorts:
        cohort_arms = [item[0] for item in loaded if item[0]['cohort_id'] == cohort]
        backend_ids = {str(arm['backend_id']) for arm in cohort_arms}
        run_roles = {str(arm['run_role']) for arm in cohort_arms}
        if len(backend_ids) != 1:
            raise ValueError(f'cohort={cohort} backend mismatch: {sorted(backend_ids)}')
        if len(run_roles) != 1:
            raise ValueError(f'cohort={cohort} run_role mismatch across arms: {sorted(run_roles)}')
    firm_frame = pd.concat(frames, ignore_index=True)
    contrasts = _contrast_table(firm_frame)
    interactions = _interaction_table(firm_frame, budget_specs=budget_map, unbounded_label=unbounded_label, expected_firm_count=expected_firm_count)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    firm_path, firm_storage = _write_frame(firm_frame, out / 'c4r_matched_v3_firm_frame.parquet')
    contrasts_path = out / 'c4r_matched_v3_contrasts.csv'
    interactions_path = out / 'c4r_matched_v3_interactions.csv'
    contrasts.to_csv(contrasts_path, index=False, encoding='utf-8-sig')
    interactions.to_csv(interactions_path, index=False, encoding='utf-8-sig')
    cohort_rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        cohort_frame = firm_frame.loc[firm_frame['cohort_id'].astype(str).eq(cohort)]
        cohort_rows.append({'cohort_id': cohort, 'backend_id': cohort_frame['backend_id'].iloc[0], 'run_role': cohort_frame['run_role'].iloc[0], 'arm_count': int(cohort_frame['budget_label'].nunique()), 'firm_count_per_arm': int(cohort_frame['row_id'].nunique()), 'firm_frame_row_count': int(len(cohort_frame)), 'contrast_row_count': int((contrasts['cohort_id'].astype(str) == cohort).sum()), 'interaction_row_count': int((interactions['cohort_id'].astype(str) == cohort).sum())})
    cohort_summary = pd.DataFrame(cohort_rows)
    cohort_summary_path = out / 'c4r_matched_v3_cohort_summary.csv'
    cohort_summary.to_csv(cohort_summary_path, index=False, encoding='utf-8-sig')
    compat_outputs: dict[str, Any] = {}
    if write_v2_compat_projection and set(budget_map) == {'0p75', 'unbounded'}:
        for cohort in cohorts:
            cohort_frame = firm_frame.loc[firm_frame['cohort_id'].astype(str).eq(cohort)].copy()
            v2_frame, v2_contrasts, v2_interactions = _v2_compat_projection(cohort_frame)
            compat_dir = out / 'v2_compat' / _safe_token(cohort, field='cohort_id')
            compat_dir.mkdir(parents=True, exist_ok=True)
            compat_firm_path, compat_storage = _write_frame(v2_frame, compat_dir / 'c4r_matched_firm_frame.parquet')
            compat_contrast_path = compat_dir / 'c4r_matched_contrasts.csv'
            compat_interaction_path = compat_dir / 'c4r_matched_interactions.csv'
            v2_contrasts.to_csv(compat_contrast_path, index=False, encoding='utf-8-sig')
            v2_interactions.to_csv(compat_interaction_path, index=False, encoding='utf-8-sig')
            compat_outputs[cohort] = {'firm_frame': {'path': str(compat_firm_path.relative_to(out)), 'storage': compat_storage}, 'contrasts': {'path': str(compat_contrast_path.relative_to(out))}, 'interactions': {'path': str(compat_interaction_path.relative_to(out))}}
    prereg_meta: dict[str, Any] | None = None
    if prereg_path is not None:
        resolved_prereg = Path(prereg_path).resolve()
        if not resolved_prereg.is_file():
            raise FileNotFoundError(f'preregistration document missing: {resolved_prereg}')
        prereg_meta = {'path': str(resolved_prereg)}
    finite_count = len([value for value in budget_map.values() if value is not None])
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'created_utc': _now(), 'information_condition': information_condition, 'conditions': list(CONDITIONS), 'mode': MODE, 'require_live': require_live, 'cohort_count': len(cohorts), 'cohorts': cohort_rows, 'budget_specs': [{'label': label, 'l1_budget': value} for label, value in budget_map.items()], 'unbounded_budget_label': unbounded_label, 'firm_count_per_arm': expected_firm_count, 'firm_frame_row_count': int(len(firm_frame)), 'contrast_row_count': int(len(contrasts)), 'interaction_row_count': int(len(interactions)), 'expected_rows_per_full_grid_cohort': {'firm_frame': expected_firm_count * len(budget_map), 'contrasts': len(budget_map) * len(ORACLES) * len(CONTRAST_COLUMNS), 'interactions': finite_count * len(ORACLES) * len(CONTRAST_COLUMNS)}, 'interaction_inference_contract': {'unit': 'firm_paired_finite_minus_unbounded_DID', 'primary_holm_family': 'cohort_finite_arm_contrast_across_three_oracles', 'primary_holm_family_size': len(ORACLES), 'sensitivity_holm_family': 'cohort_contrast_across_all_finite_arms_and_oracles', 'sensitivity_holm_family_size': finite_count * len(ORACLES), 'test': 'two_sided_wilcoxon_signed_rank_zero_discard'}, 'additive_identity_tolerance': IDENTITY_TOLERANCE, 'stage8_stage9_revision_score_tolerance': REVISION_SCORE_TOLERANCE, 'preregistration': prereg_meta, 'evidence_tier': 'PREREGISTERED_C4R_JOURNAL_EXTENSION', 'arms': [item[0] for item in loaded], 'outputs': {'firm_frame': {'path': firm_path.name, 'storage': firm_storage, 'row_count': int(len(firm_frame))}, 'contrasts': {'path': contrasts_path.name, 'row_count': int(len(contrasts))}, 'interactions': {'path': interactions_path.name, 'row_count': int(len(interactions))}, 'cohort_summary': {'path': cohort_summary_path.name, 'row_count': int(len(cohort_summary))}, 'v2_compat': compat_outputs}, 'interpretation_boundary': 'Each cohort is analyzed separately; no backend pooling is performed. C4R-C4 identifies reference-free second-pass re-review within an arm, and C6-C4R identifies the conditional increment of Candidate-IQL reference content. Claims remain specific to the frozen backend snapshot, IC-b cohort, provider protocol, and raw-coordinate L1 contract.'}
    manifest_path = out / 'c4r_matched_inference_v3_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', action='append', required=True, help='Repeat cohort|budget_label=archive_path')
    parser.add_argument('--budget-spec', action='append', default=None, help="Repeat label=value; use value 'none' for unbounded. Default: 0p75/1p27/2p00/unbounded.")
    parser.add_argument('--expected-run-role', action='append', default=[], help='Repeat cohort=run_role')
    parser.add_argument('--information-condition', default='IC-b', choices=['IC-a', 'IC-b', 'IC-c'])
    parser.add_argument('--expected-firm-count', type=int, default=575)
    parser.add_argument('--allow-nonlive', action='store_true', help='Synthetic/scripted rehearsal only; recorded in manifest')
    parser.add_argument('--prereg-path', default=None)
    parser.add_argument('--no-v2-compat-projection', action='store_true')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    arm_specs = [parse_arm_spec(value) for value in args.arm]
    budget_specs = [parse_budget_spec(value) for value in args.budget_spec] if args.budget_spec else [BudgetSpec(label, value) for label, value in DEFAULT_BUDGET_SPECS]
    roles = dict((parse_expected_role(value) for value in args.expected_run_role))
    result = run_analysis(arm_specs=arm_specs, out_dir=Path(args.out), budget_specs=budget_specs, expected_run_roles=roles, information_condition=args.information_condition, expected_firm_count=args.expected_firm_count, require_live=not args.allow_nonlive, prereg_path=Path(args.prereg_path) if args.prereg_path else None, write_v2_compat_projection=not args.no_v2_compat_projection)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
