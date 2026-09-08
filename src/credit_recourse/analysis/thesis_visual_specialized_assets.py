from __future__ import annotations
'Dedicated semantic builders for thesis visual assets.\n\nThese handlers are intentionally separate from the generic token-matching path\nin :mod:`credit_recourse.analysis.thesis_visual_builder`.  Every handler binds\nto an explicit filename/schema contract, records the exact input path through the\nshared ``BuildContext``, and hard-fails on malformed or ambiguous evidence.\nMissing evidence is represented only by :class:`SpecializedEvidenceMissing`, so\nthe parent catalogue can retain its explicit optional-asset skip policy.\n'
import json
import math
import re
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from credit_recourse.analysis.paper_output_layout import build_layout
SPECIALIZED_CONTRACT_SCHEMA = 'thesis_visual_specialized_contract_v1'
SPECIALIZED_HANDLERS = frozenset({'test3_panel', 'capture_definitions', 'performance_verifiability_frontier', 'budget_compliance_ladder', 'raw_applied_l1_distribution', 'did_equivalence_forest', 'win_tie_loss_stack', 'axis_shapley_diverging', 'tost_complete_ledger', 'harness_lever_common_scale', 'dynamic_resolution_panel', 'e2_e3_rank_correlation', 'repeat_stability'})

class SpecializedEvidenceMissing(RuntimeError):
    """Required evidence for one optional specialised asset is absent."""

def _norm(value: Any) -> str:
    return re.sub('[^a-z0-9]+', '', str(value).lower())

def _contract(ctx: Any, run: Any) -> dict[str, Any]:
    path = ctx.config_dir / 'thesis_visual_specialized_contract.yaml'
    if not path.is_file():
        raise RuntimeError(f'Specialised visual contract is missing: {path}')
    payload = ctx.read(run, path, 'specialised visual contract')
    if not isinstance(payload, dict) or payload.get('schema_version') != SPECIALIZED_CONTRACT_SCHEMA:
        raise RuntimeError(f"Unexpected specialised visual contract schema: {(payload.get('schema_version') if isinstance(payload, dict) else type(payload).__name__)}")
    return payload

def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f'{context} missing columns: {missing}')

def _finite_numeric(series: pd.Series, *, context: str, allow_missing: bool=False) -> pd.Series:
    values = pd.to_numeric(series, errors='coerce')
    if not allow_missing and values.isna().any():
        raise ValueError(f'{context} contains missing/non-numeric values')
    finite = values.dropna()
    if not np.isfinite(finite.to_numpy(dtype=float)).all():
        raise ValueError(f'{context} contains non-finite values')
    return values

def _resolve_exact(ctx: Any, *, basenames: Sequence[str], contains_all: Sequence[str]=(), contains_any: Sequence[str]=(), excludes: Sequence[str]=(), role: str) -> Path:
    names = {name.lower() for name in basenames}
    candidates = [path for path in ctx.index.files if path.name.lower() in names]
    if contains_all:
        tokens = [_norm(token) for token in contains_all]
        candidates = [path for path in candidates if all((token in _norm(str(path)) for token in tokens))]
    if contains_any:
        tokens = [_norm(token) for token in contains_any]
        candidates = [path for path in candidates if any((token in _norm(str(path)) for token in tokens))]
    if excludes:
        tokens = [_norm(token) for token in excludes]
        candidates = [path for path in candidates if not any((token in _norm(str(path)) for token in tokens))]
    if not candidates:
        raise SpecializedEvidenceMissing(f'Missing {role}: basenames={list(basenames)}, contains_all={list(contains_all)}, contains_any={list(contains_any)}')

    def rank(path: Path) -> tuple[int, int, str]:
        suffix_rank = {'.parquet': 0, '.pq': 0, '.csv': 1, '.json': 2, '.yaml': 3, '.yml': 3}.get(path.suffix.lower(), 9)
        return (suffix_rank, len(str(path)), str(path).lower())
    candidates = sorted(set(candidates), key=rank)
    if len(candidates) > 1:
        parents = {str(path.parent.resolve()).lower() for path in candidates}
        stems = {_norm(path.stem) for path in candidates}
        if len(parents) == 1 and len(stems) == 1:
            return candidates[0]
        raise RuntimeError(f'Ambiguous {role}; exact candidates={[str(path) for path in candidates]}')
    return candidates[0]

def _resolve_all_exact(ctx: Any, *, basenames: Sequence[str], contains_any: Sequence[str]=(), excludes: Sequence[str]=()) -> list[Path]:
    names = {name.lower() for name in basenames}
    candidates = [path for path in ctx.index.files if path.name.lower() in names]
    if contains_any:
        tokens = [_norm(token) for token in contains_any]
        candidates = [path for path in candidates if any((token in _norm(str(path)) for token in tokens))]
    if excludes:
        tokens = [_norm(token) for token in excludes]
        candidates = [path for path in candidates if not any((token in _norm(str(path)) for token in tokens))]
    grouped: dict[tuple[str, str], list[Path]] = {}
    for path in candidates:
        grouped.setdefault((str(path.parent.resolve()).lower(), _norm(path.stem)), []).append(path)
    out: list[Path] = []
    for paths in grouped.values():
        paths = sorted(paths, key=lambda p: ({'.parquet': 0, '.pq': 0, '.csv': 1}.get(p.suffix.lower(), 9), str(p)))
        out.append(paths[0])
    return sorted(out, key=lambda p: str(p).lower())

def _canonical_existing_file(*, candidates: Sequence[Path], role: str, missing_is_optional: bool=True) -> Path:
    existing = [Path(path).resolve() for path in candidates if Path(path).is_file()]
    if not existing:
        message = f'Missing canonical {role}: {[str(path) for path in candidates]}'
        if missing_is_optional:
            raise SpecializedEvidenceMissing(message)
        raise FileNotFoundError(message)
    parents = {str(path.parent).lower() for path in existing}
    stems = {_norm(path.stem) for path in existing}
    if len(parents) != 1 or len(stems) != 1:
        raise RuntimeError(f'Ambiguous canonical {role}: {[str(path) for path in existing]}')
    return sorted(existing, key=lambda path: ({'.parquet': 0, '.pq': 0, '.csv': 1, '.json': 2}.get(path.suffix.lower(), 9), str(path).lower()))[0]

def _catalog_selected_run(ctx: Any, run: Any, *, run_role: str, information_condition: str) -> dict[str, Any]:
    layout = build_layout(ctx.analysis_dir)
    catalog_path = layout.manifest / 'archived_llm_run_catalog.csv'
    catalog = _read_frame(ctx, run, catalog_path, 'archived LLM run catalogue')
    required = {'run_label', 'run_role', 'information_condition', 'selected_for_paper'}
    _require_columns(catalog, required, context='archived LLM run catalogue')
    selected_flag = catalog['selected_for_paper'].astype(str).str.strip().str.lower().map({'true': True, '1': True, 'false': False, '0': False})
    target = catalog.loc[catalog['run_role'].astype(str).eq(str(run_role)) & catalog['information_condition'].astype(str).eq(str(information_condition)) & selected_flag.fillna(False)].copy()
    if len(target) != 1:
        rows = target[['run_label', 'run_role', 'information_condition', 'selected_for_paper']].to_dict(orient='records')
        raise ValueError(f'Canonical run selection must resolve exactly one archived run: run_role={run_role!r}, information_condition={information_condition!r}, matches={rows}')
    return target.iloc[0].to_dict()

def _canonical_e2_evidence(ctx: Any, run: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    layout = build_layout(ctx.analysis_dir)
    frame_path = _canonical_existing_file(candidates=(layout.e2 / 'c4r_matched_firm_frame.parquet', layout.e2 / 'c4r_matched_firm_frame.csv'), role='E2 C4R matched firm frame')
    manifest_path = layout.e2 / 'c4r_matched_inference_manifest.json'
    if not manifest_path.is_file():
        raise SpecializedEvidenceMissing(f'Missing canonical E2 manifest: {manifest_path}')
    frame = _read_frame(ctx, run, frame_path, 'E2 C4R matched firm frame')
    manifest = _read_json(ctx, run, manifest_path, 'E2 C4R matched manifest')
    if manifest.get('status') != 'PASS':
        raise ValueError(f"E2 C4R manifest status is not PASS: {manifest.get('status')!r}")
    backend_id = str(manifest.get('backend_id') or '')
    if not backend_id:
        raise ValueError('E2 C4R manifest is missing backend_id')
    if str(manifest.get('information_condition') or '') != 'IC-b':
        raise ValueError(f"E2 C4R manifest must be IC-b for E2↔E3 comparison: {manifest.get('information_condition')!r}")
    return (frame, manifest)

def _canonical_e3_firm_frame(ctx: Any, run: Any) -> pd.DataFrame:
    layout = build_layout(ctx.analysis_dir)
    path = _canonical_existing_file(candidates=(layout.e3 / 'v3_matched' / 'c4r_matched_v3_firm_frame.parquet', layout.e3 / 'v3_matched' / 'c4r_matched_v3_firm_frame.csv'), role='E3 C4R v3 firm frame')
    return _read_frame(ctx, run, path, 'E3 C4R v3 firm frame')

def _read_frame(ctx: Any, run: Any, path: Path, role: str) -> pd.DataFrame:
    obj = ctx.read(run, path, role)
    if isinstance(obj, pd.DataFrame):
        frame = obj.copy()
    elif isinstance(obj, list) and all((isinstance(row, dict) for row in obj)):
        frame = pd.DataFrame(obj)
    else:
        raise ValueError(f'{role} must be a table: {path}')
    if frame.empty:
        raise ValueError(f'{role} is empty: {path}')
    return frame

def _read_json(ctx: Any, run: Any, path: Path, role: str) -> dict[str, Any]:
    obj = ctx.read(run, path, role)
    if not isinstance(obj, dict):
        raise ValueError(f'{role} must be a JSON/YAML mapping: {path}')
    return obj

def _normal_ci(values: pd.Series, level: float=0.95) -> tuple[float, float, float, int]:
    numeric = _finite_numeric(values, context='normal-CI values', allow_missing=True).dropna().astype(float)
    n = int(len(numeric))
    if n < 2:
        raise ValueError(f'At least two values are required for a CI; got n={n}')
    mean = float(numeric.mean())
    se = float(numeric.std(ddof=1) / math.sqrt(n))
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    return (mean, mean - z * se, mean + z * se, n)

def _budget_sort_key(value: Any) -> tuple[int, float, str]:
    text = str(value)
    if text.lower() in {'unbounded', 'none', 'no_budget', 'no-budget', 'inf', 'infinity'}:
        return (1, math.inf, text)
    match = re.search('(\\d+)(?:[p._](\\d+))?', text)
    if match:
        whole = float(match.group(1))
        frac = match.group(2)
        val = whole + (float(f'0.{frac}') if frac else 0.0)
        return (0, val, text)
    return (0, math.inf - 1, text)

def _budget_label_from_target(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 'unbounded'
    numeric = float(value)
    return f'{numeric:.2f}'.replace('.', 'p')

def _parse_scalar_or_range(value: Any, *, context: str) -> tuple[float | None, float | None, float | None]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return (None, None, None)
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f'{context} is non-finite')
        return (number, number, number)
    text = str(value).strip()
    if not text:
        return (None, None, None)
    if '..' in text:
        parts = text.split('..')
        if len(parts) != 2:
            raise ValueError(f'{context} malformed range: {text!r}')
        lo, hi = map(float, parts)
        if not (math.isfinite(lo) and math.isfinite(hi) and (lo <= hi)):
            raise ValueError(f'{context} invalid range: {text!r}')
        return ((lo + hi) / 2.0, lo, hi)
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f'{context} is non-finite')
    return (number, number, number)

def _save_two_panel(ctx: Any, run: Any, *, left: Mapping[str, Any], right: Mapping[str, Any], data: pd.DataFrame) -> None:
    """Shared two-panel serializer used only by dedicated handlers."""
    plt = ctx.matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4))
    left['draw'](axes[0])
    right['draw'](axes[1])
    fig.suptitle(run.spec.title, fontsize=13, fontweight='bold')
    ctx.save_figure(run, fig, data)

def _build_test3_panel(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    rows_path = _resolve_exact(ctx, basenames=('test3_rows.csv',), contains_all=('test3_counterfactual_fidelity',), role='Test3 firm rows')
    summary_path = _resolve_exact(ctx, basenames=('test3_property_summary.csv',), contains_all=('test3_counterfactual_fidelity',), role='Test3 property summary')
    yearly_candidates = _resolve_all_exact(ctx, basenames=('test3_wedge_by_year.csv',), contains_any=('test3_counterfactual_fidelity',))
    rows = _read_frame(ctx, run, rows_path, 'Test3 firm rows')
    summary = _read_frame(ctx, run, summary_path, 'Test3 property summary')
    _require_columns(rows, {'target_year', 'split', 'wedge'}, context='Test3 firm rows')
    _require_columns(summary, {'property', 'metric', 'split', 'value', 'n'}, context='Test3 property summary')
    rows['target_year'] = _finite_numeric(rows['target_year'], context='Test3 target_year').astype(int)
    rows['wedge'] = _finite_numeric(rows['wedge'], context='Test3 wedge', allow_missing=True)
    computed = rows.groupby(['target_year', 'split'], as_index=False, dropna=False)['wedge'].agg(n='count', median='median', q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)).sort_values('target_year')
    if yearly_candidates:
        yearly = _read_frame(ctx, run, yearly_candidates[0], 'Test3 yearly wedge summary')
        _require_columns(yearly, {'target_year', 'split', 'n', 'median', 'q25', 'q75'}, context='Test3 yearly wedge summary')
        merged = computed.merge(yearly, on=['target_year', 'split'], suffixes=('__computed', '__reported'), validate='one_to_one')
        if len(merged) != len(computed) or len(merged) != len(yearly):
            raise ValueError('Test3 computed and reported yearly row universes differ')
        for column in ('n', 'median', 'q25', 'q75'):
            a = pd.to_numeric(merged[f'{column}__computed'], errors='raise')
            b = pd.to_numeric(merged[f'{column}__reported'], errors='raise')
            if not np.allclose(a.to_numpy(float), b.to_numpy(float), atol=1e-12, rtol=0, equal_nan=True):
                raise ValueError(f'Test3 yearly summary mismatch for {column}')
    yearly = computed
    dev = rows.loc[rows['split'].astype(str).str.lower().eq('dev'), 'wedge'].dropna()
    if dev.empty:
        raise ValueError('Test3 DEV rows are required to derive the frozen adjustment line')
    dev_median = float(dev.median())
    cfg = contract.get('n6_test3_panel') or {}
    required_splits = [str(x) for x in cfg.get('required_splits') or ('dev', 'oot', 'post')]
    direction = summary.loc[summary['property'].astype(str).eq(str(cfg.get('direction_property') or 'P3_direction')) & summary['metric'].astype(str).eq(str(cfg.get('direction_metric') or 'agreement')) & summary['split'].astype(str).isin(required_splits)].copy()
    if set(direction['split'].astype(str)) != set(required_splits):
        raise ValueError(f"Test3 direction summary must contain splits={required_splits}; observed={sorted(direction['split'].astype(str).unique())}")
    _require_columns(direction, {'ci95_lo', 'ci95_hi'}, context='Test3 P3 direction rows')
    for column in ('value', 'ci95_lo', 'ci95_hi'):
        direction[column] = _finite_numeric(direction[column], context=f'Test3 direction {column}')
    direction['split'] = pd.Categorical(direction['split'], categories=required_splits, ordered=True)
    direction = direction.sort_values('split')
    yearly_plot = yearly.assign(panel='P1_wedge_by_year', adjustment_value=dev_median)
    direction_plot = direction.assign(panel='P3_direction_wilson')
    plot_data = pd.concat([yearly_plot, direction_plot], ignore_index=True, sort=False)

    def draw_left(ax: Any) -> None:
        x = yearly['target_year'].to_numpy(dtype=float)
        median = pd.to_numeric(yearly['median'], errors='raise').to_numpy(dtype=float)
        q25 = pd.to_numeric(yearly['q25'], errors='raise').to_numpy(dtype=float)
        q75 = pd.to_numeric(yearly['q75'], errors='raise').to_numpy(dtype=float)
        ax.plot(x, median, marker='o', linewidth=2, label='연도별 중앙값')
        ax.fill_between(x, q25, q75, alpha=0.18, label='IQR')
        ax.axhline(dev_median, linestyle='--', linewidth=1.4, label=f'DEV 고정선 c={dev_median:.3f}')
        ax.set_xlabel('target year')
        ax.set_ylabel('wedge')
        ax.set_title('P1. 연도별 수준편향')
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)

    def draw_right(ax: Any) -> None:
        x = np.arange(len(direction))
        value = direction['value'].to_numpy(dtype=float)
        lo = direction['ci95_lo'].to_numpy(dtype=float)
        hi = direction['ci95_hi'].to_numpy(dtype=float)
        ax.bar(x, value, alpha=0.78)
        ax.errorbar(x, value, yerr=[value - lo, hi - value], fmt='none', color='#333333', capsize=4)
        ax.set_xticks(x, [str(x) for x in direction['split']])
        ax.set_ylim(0, 1)
        ax.set_ylabel('direction agreement')
        ax.set_title('P3. 방향 재현율과 Wilson 95% CI')
        ax.grid(axis='y', alpha=0.22)
    _save_two_panel(ctx, run, left={'draw': draw_left}, right={'draw': draw_right}, data=plot_data)
    run.transformations.extend(['Recomputed Test3 yearly wedge median/IQR from firm rows and cross-checked the canonical yearly CSV.', 'Derived the fixed DEV adjustment line from the DEV wedge median without tuning.', 'Rendered DEV/OOT/post P3 direction agreement with reported Wilson 95% intervals.'])

def _build_capture_definitions(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    policy_path = _resolve_exact(ctx, basenames=('policy_actions.parquet', 'policy_actions.csv'), contains_all=('stage6_candidate_selector_eval',), role='Stage6 policy actions')
    score_path = _resolve_exact(ctx, basenames=('multi_oracle_policy_eval.parquet', 'multi_oracle_policy_eval.csv'), contains_any=('stage6_candidate_selector_eval', 'stage6_multi_oracle_eval'), role='Stage6 multi-Oracle policy ledger')
    library_path = ctx.config_dir / 'final_candidate_library.yaml'
    if not library_path.is_file():
        raise RuntimeError(f'Candidate library contract missing: {library_path}')
    actions = _read_frame(ctx, run, policy_path, 'Stage6 policy actions')
    scores = _read_frame(ctx, run, score_path, 'Stage6 multi-Oracle policy ledger')
    library = _read_json(ctx, run, library_path, 'candidate library contract')
    _require_columns(actions, {'row_id', 'policy', 'candidate_id'}, context='Stage6 policy actions')
    _require_columns(scores, {'row_id', 'policy', 'candidate_id'}, context='Stage6 score ledger')
    labels = [str(x) for x in library.get('main_train_labels') or []]
    final_rl_label = str(library.get('final_rl_label') or '').strip()
    if not labels or not final_rl_label:
        raise ValueError('Candidate library must define main_train_labels and final_rl_label')
    fixed_policy = {label: 'C0_noop' if label == 'A0_noop' else label for label in labels}
    selector = actions.loc[actions['policy'].astype(str).eq(final_rl_label), ['row_id', 'candidate_id']].copy()
    if selector.empty or selector['row_id'].duplicated().any():
        raise ValueError('Stage6 selector must contain one final-RL candidate per row_id')
    if not set(selector['candidate_id'].astype(str)).issubset(set(labels)):
        bad = sorted(set(selector['candidate_id'].astype(str)) - set(labels))
        raise ValueError(f'Stage6 selector contains candidates outside main_train_labels: {bad}')
    fixed_rows = []
    for candidate, policy in fixed_policy.items():
        part = scores.loc[scores['policy'].astype(str).eq(policy)].copy()
        if part.empty:
            raise ValueError(f'Stage6 score ledger is missing fixed candidate policy={policy}')
        if part['row_id'].duplicated().any():
            raise ValueError(f'Duplicate Stage6 fixed scores for policy={policy}')
        part['fixed_candidate_id'] = candidate
        fixed_rows.append(part)
    fixed = pd.concat(fixed_rows, ignore_index=True, sort=False)
    if fixed.duplicated(['row_id', 'fixed_candidate_id']).any():
        raise ValueError('Stage6 fixed candidate grid has duplicate firm/candidate keys')
    expected = int(selector['row_id'].nunique()) * len(labels)
    if len(fixed) != expected:
        raise ValueError(f'Incomplete Stage6 fixed candidate grid: rows={len(fixed)}, expected={expected}')
    tolerance = float((contract.get('c9_capture') or {}).get('numeric_tolerance', 1e-12))
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[pd.DataFrame] = []
    for oracle in ('alpha', 'beta', 'gamma'):
        score_col = f'delta_R_score_{oracle}'
        if score_col not in fixed.columns:
            raise ValueError(f'Stage6 score ledger missing {score_col}')
        work = fixed[['row_id', 'fixed_candidate_id', score_col]].copy()
        work[score_col] = _finite_numeric(work[score_col], context=f'C9 {score_col}')
        pivot = work.pivot(index='row_id', columns='fixed_candidate_id', values=score_col)
        if set(pivot.columns.astype(str)) != set(labels) or pivot.isna().any().any():
            raise ValueError(f'C9 candidate score matrix is incomplete for oracle={oracle}')
        selected = selector.set_index('row_id')['candidate_id'].astype(str).reindex(pivot.index)
        if selected.isna().any():
            raise ValueError(f'C9 selector/score row universe mismatch for oracle={oracle}')
        maxima = pivot.max(axis=1)
        is_max = pivot.sub(maxima, axis=0).abs().le(tolerance)
        max_count = is_max.sum(axis=1)
        selected_score = pd.Series([float(pivot.loc[row_id, candidate]) for row_id, candidate in selected.items()], index=pivot.index)
        max_hit = selected_score.sub(maxima).abs().le(tolerance)
        unique_eligible = max_count.eq(1)
        unique_hit = max_hit & unique_eligible
        dense_rank = pivot.rank(axis=1, method='min', ascending=False)
        top3_hit = pd.Series([float(dense_rank.loc[row_id, candidate]) <= 3.0 for row_id, candidate in selected.items()], index=pivot.index)
        c3 = scores.loc[scores['policy'].astype(str).eq(final_rl_label), ['row_id', score_col]].copy()
        if not c3.empty:
            if c3['row_id'].duplicated().any():
                raise ValueError(f'Duplicate final-RL Stage6 score rows for oracle={oracle}')
            c3 = c3.set_index('row_id')[score_col].reindex(pivot.index)
            c3 = _finite_numeric(c3, context=f'C9 final-RL {score_col}')
            if not np.allclose(c3.to_numpy(float), selected_score.to_numpy(float), atol=tolerance, rtol=0):
                raise ValueError(f'C9 selected candidate scores do not reproduce final-RL scores for oracle={oracle}')
        n = int(len(pivot))
        definitions = [('max_with_ties', max_hit, pd.Series(True, index=pivot.index), 'selected candidate score equals the within-firm maximum; tied maxima count as captured'), ('unique_argmax', unique_hit, unique_eligible, 'selected candidate equals the unique within-firm argmax'), ('top3_tie_inclusive', top3_hit, pd.Series(True, index=pivot.index), 'selected candidate has descending minimum rank <=3; ties at the rank-3 boundary are included')]
        for definition_id, hit, eligible, definition in definitions:
            eligible_n = int(eligible.sum())
            hit_n = int((hit & eligible).sum())
            output_rows.append({'oracle_backend': oracle, 'definition_id': definition_id, 'definition': definition, 'all_firm_n': n, 'eligible_n': eligible_n, 'hit_n': hit_n, 'capture_rate': float(hit_n / eligible_n) if eligible_n else np.nan, 'all_firm_capture_rate': float(hit_n / n), 'mean_max_tie_count': float(max_count.mean()), 'numeric_tolerance': tolerance})
        audit_rows.append(pd.DataFrame({'row_id': pivot.index, 'oracle_backend': oracle, 'selected_candidate': selected.to_numpy(), 'selected_score': selected_score.to_numpy(), 'maximum_score': maxima.to_numpy(), 'maximum_tie_count': max_count.to_numpy(), 'max_with_ties_hit': max_hit.to_numpy(), 'unique_argmax_eligible': unique_eligible.to_numpy(), 'unique_argmax_hit': unique_hit.to_numpy(), 'top3_tie_inclusive_hit': top3_hit.to_numpy()}))
    out = pd.DataFrame(output_rows)
    if len(out) != 9:
        raise RuntimeError(f'C9 must emit 3 definitions × 3 Oracles = 9 rows; got {len(out)}')
    ctx.write_table(run, out, note='포착률은 정의별 분모가 다르다. unique_argmax의 capture_rate는 유일 최댓값이 존재하는 기업만을 분모로 하며, all_firm_capture_rate를 함께 보고한다.')
    ctx.snapshot(run, pd.concat(audit_rows, ignore_index=True), suffix='firm_definition_audit')
    run.transformations.extend(['Reconstructed the fixed-candidate Stage6 score matrix from the frozen candidate library.', 'Verified that selected-candidate scores exactly reproduce the final-RL policy score within the registered tolerance.', 'Computed max-with-ties, unique-argmax, and tie-inclusive top-3 capture under explicit denominators.'])

def _read_stage7_via_decomposition_inputs(ctx: Any, run: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs_path = _resolve_exact(ctx, basenames=('main_harness_backend_input_files.csv',), contains_all=('main_harness_backend_decomposition',), role='main harness input-file ledger')
    alignment_path = _resolve_exact(ctx, basenames=('main_harness_backend_alignment_audit.csv',), contains_all=('main_harness_backend_decomposition',), role='main harness alignment audit')
    inputs = _read_frame(ctx, run, inputs_path, 'main harness input-file ledger')
    alignment = _read_frame(ctx, run, alignment_path, 'main harness alignment audit')
    _require_columns(inputs, {'run_label', 'artifact', 'path'}, context='main harness input-file ledger')
    _require_columns(alignment, {'run_label', 'backend_label'}, context='main harness alignment audit')
    mapping = alignment[['run_label', 'backend_label']].drop_duplicates()
    if mapping['run_label'].duplicated().any():
        raise ValueError('Main harness alignment audit maps one run_label to multiple backends')
    stage7_rows = inputs.loc[inputs['artifact'].astype(str).eq('stage7_action_table')].copy()
    if stage7_rows.empty or stage7_rows['run_label'].duplicated().any():
        raise ValueError('Main harness input ledger must contain one Stage7 action table per run_label')
    stage7_rows = stage7_rows.merge(mapping, on='run_label', how='left', validate='one_to_one')
    if stage7_rows['backend_label'].isna().any():
        raise ValueError('Main harness input ledger contains unmapped run labels')
    frames: list[pd.DataFrame] = []
    for row in stage7_rows.itertuples(index=False):
        path = Path(str(row.path)).resolve()
        if not path.is_file():
            raise SpecializedEvidenceMissing(f'Main harness Stage7 action artifact is unavailable: {path}')
        frame = _read_frame(ctx, run, path, f'Stage7 action table for {row.run_label}')
        _require_columns(frame, {'policy', 'mode', 'projection_distance'}, context=f'Stage7 action table {row.run_label}')
        frame = frame.copy()
        frame['run_label'] = str(row.run_label)
        frame['backend_label'] = str(row.backend_label)
        frames.append(frame)
    return (pd.concat(frames, ignore_index=True, sort=False), stage7_rows)

def _action_l1_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if str(column).startswith('action__')]

def _applied_l1(frame: pd.DataFrame) -> pd.Series:
    if 'budget_l1_clipped' in frame.columns:
        values = pd.to_numeric(frame['budget_l1_clipped'], errors='coerce')
        if values.notna().any():
            return values
    action_columns = _action_l1_columns(frame)
    if not action_columns:
        raise ValueError('Stage7 action table lacks budget_l1_clipped and action__ dimensions')
    return frame[action_columns].apply(pd.to_numeric, errors='raise').abs().sum(axis=1)

def _build_performance_verifiability_frontier(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    means_path = _resolve_exact(ctx, basenames=('main_harness_backend_cell_means.csv',), contains_all=('main_harness_backend_decomposition',), role='main harness cell means')
    means = _read_frame(ctx, run, means_path, 'main harness cell means')
    _require_columns(means, {'backend_label', 'harness_cell', 'oracle_backend', 'mean_score'}, context='main harness cell means')
    stage7, _ = _read_stage7_via_decomposition_inputs(ctx, run)
    stage7['projection_distance'] = _finite_numeric(stage7['projection_distance'], context='Stage7 projection_distance', allow_missing=True)
    stage7['applied_l1'] = _applied_l1(stage7)
    stage7['raw_l1'] = pd.to_numeric(stage7['budget_l1_raw'], errors='coerce') if 'budget_l1_raw' in stage7.columns else np.nan
    stage7['harness_cell'] = stage7['policy'].astype(str) + '__' + stage7['mode'].astype(str)
    geometry = stage7.groupby(['backend_label', 'harness_cell'], as_index=False).agg(n_rows=('harness_cell', 'size'), mean_projection_distance=('projection_distance', 'mean'), mean_applied_l1=('applied_l1', 'mean'), mean_raw_l1=('raw_l1', 'mean'))
    primary = str((contract.get('n1_common_scale') or {}).get('primary_oracle') or 'alpha')
    score = means.loc[means['oracle_backend'].astype(str).eq(primary)].copy()
    if score.empty:
        raise ValueError(f'Main harness cell means have no primary Oracle={primary}')
    data = score.merge(geometry, on=['backend_label', 'harness_cell'], how='inner', validate='one_to_one')
    if len(data) != len(score):
        missing = score.merge(geometry, on=['backend_label', 'harness_cell'], how='left', indicator=True)
        missing = missing.loc[missing['_merge'].ne('both'), ['backend_label', 'harness_cell']]
        raise ValueError(f"Missing Stage7 geometry for harness cells: {missing.to_dict('records')}")
    for column in ('mean_score', 'mean_projection_distance', 'mean_applied_l1'):
        data[column] = _finite_numeric(data[column], context=f'D10 {column}')
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(9.4, 6.2))
    backends = list(dict.fromkeys(data['backend_label'].astype(str)))
    palette = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
    sizes = 45 + 260 * data['mean_applied_l1'].clip(lower=0) / max(float(data['mean_applied_l1'].max()), 1e-12)
    for index, backend in enumerate(backends):
        mask = data['backend_label'].astype(str).eq(backend)
        ax.scatter(data.loc[mask, 'mean_projection_distance'], data.loc[mask, 'mean_score'], s=sizes.loc[mask], alpha=0.72, label=backend, color=palette[index % len(palette)], edgecolor='white', linewidth=0.6)
    ax.axhline(0, color='#444444', linewidth=0.8)
    ax.set_xlabel('mean projection distance')
    ax.set_ylabel(f'mean Δnoop ({primary})')
    ax.set_title(run.spec.title + '\n점 크기: 평균 applied L1')
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    ctx.save_figure(run, fig, data)
    run.transformations.extend(['Joined crossed-panel Oracle cell means to the exact Stage7 action tables listed in the decomposition input ledger.', 'Used mean projection distance as the verifiability axis and mean applied L1 as marker size.', 'Restricted the performance axis to the registered primary Oracle without mixing Oracle scales.'])

def _build_budget_compliance_ladder(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    ladder_path = _resolve_exact(ctx, basenames=('e4_budget_contract_ladder.csv',), contains_any=('e4_budget', 'e4_haiku', 'budget_contract_ladder'), role='E4 budget-contract ladder')
    inputs_path = _resolve_exact(ctx, basenames=('e4_budget_contract_input_files.csv',), contains_any=('e4_budget', 'e4_haiku', 'budget_contract_ladder'), role='E4 budget-contract input ledger')
    ladder = _read_frame(ctx, run, ladder_path, 'E4 budget-contract ladder')
    inputs = _read_frame(ctx, run, inputs_path, 'E4 budget-contract input ledger')
    compliance_columns = ['aggregate_raw_compliance', 'C4_raw_compliance', 'C4R_raw_compliance', 'C6_raw_compliance']
    _require_columns(ladder, {'configuration', 'status', *compliance_columns}, context='E4 budget-contract ladder')
    _require_columns(inputs, {'configuration', 'path'}, context='E4 budget-contract input ledger')
    long_rows: list[dict[str, Any]] = []
    for row in ladder.itertuples(index=False):
        for column in compliance_columns:
            midpoint, lo, hi = _parse_scalar_or_range(getattr(row, column), context=f'{row.configuration}/{column}')
            if midpoint is None:
                continue
            if midpoint < 0 or midpoint > 1 or lo is None or (hi is None) or (lo < 0) or (hi > 1):
                raise ValueError(f'Compliance rate outside [0,1]: {row.configuration}/{column}')
            long_rows.append({'configuration': str(row.configuration), 'status': str(row.status), 'series': column.replace('_raw_compliance', '').replace('aggregate', '전체'), 'compliance_midpoint': midpoint, 'compliance_low': lo, 'compliance_high': hi})
    long = pd.DataFrame(long_rows)
    if long.empty:
        raise ValueError('E4 compliance ladder contains no numeric rates')
    callout: dict[str, float] | None = None
    full = inputs.loc[inputs['configuration'].astype(str).str.contains('Haiku thinking-2048 full', case=False, regex=False)]
    if len(full) != 1:
        raise ValueError('E6 requires exactly one Haiku thinking-2048 full Stage7 source')
    action_path = Path(str(full.iloc[0]['path'])).resolve()
    if not action_path.is_file():
        raise SpecializedEvidenceMissing(f'Haiku full Stage7 action table unavailable: {action_path}')
    action = _read_frame(ctx, run, action_path, 'Haiku thinking full Stage7 action table')
    _require_columns(action, {'budget_l1_raw', 'budget_l1_clipped'}, context='Haiku full Stage7 action table')
    raw = _finite_numeric(action['budget_l1_raw'], context='Haiku raw L1', allow_missing=True).dropna()
    applied = _finite_numeric(action['budget_l1_clipped'], context='Haiku applied L1', allow_missing=True).dropna()
    if raw.empty or applied.empty:
        raise ValueError('Haiku full Stage7 action table lacks complete raw/applied L1 values')
    callout = {'mean_applied_l1': float(applied.mean()), 'mean_raw_l1': float(raw.mean())}
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    configs = list(dict.fromkeys(long['configuration'].astype(str)))
    series = list(dict.fromkeys(long['series'].astype(str)))
    width = 0.82 / max(len(series), 1)
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00']
    for idx, name in enumerate(series):
        part = long.loc[long['series'].astype(str).eq(name)]
        lookup = {str(row.configuration): row for row in part.itertuples(index=False)}
        x = np.arange(len(configs)) - 0.41 + width / 2 + idx * width
        values = np.array([lookup[c].compliance_midpoint if c in lookup else np.nan for c in configs], dtype=float)
        low = np.array([lookup[c].compliance_low if c in lookup else np.nan for c in configs], dtype=float)
        high = np.array([lookup[c].compliance_high if c in lookup else np.nan for c in configs], dtype=float)
        ax.bar(x, values, width=width, label=name, color=colors[idx % len(colors)], alpha=0.78)
        finite = np.isfinite(values) & np.isfinite(low) & np.isfinite(high) & (high - low > 1e-12)
        if finite.any():
            ax.errorbar(x[finite], values[finite], yerr=[values[finite] - low[finite], high[finite] - values[finite]], fmt='none', color='#333333', capsize=3)
    ax.set_xticks(np.arange(len(configs)), configs, rotation=28, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('raw-budget compliance rate')
    ax.set_title(run.spec.title)
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    ax.text(0.99, 0.03, f"Haiku full 자기검산: mean applied L1={callout['mean_applied_l1']:.3f}, raw L1={callout['mean_raw_l1']:.3f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=9, bbox={'boxstyle': 'round,pad=0.35', 'facecolor': 'white', 'edgecolor': '#777777', 'alpha': 0.9})
    plot_data = pd.concat([long.assign(record_type='compliance'), pd.DataFrame([{**callout, 'record_type': 'haiku_full_l1_callout'}])], ignore_index=True, sort=False)
    ctx.save_figure(run, fig, plot_data)
    run.transformations.extend(['Parsed scalar and range-valued compliance entries without converting missing/failed configurations into zeros.', 'Recomputed the Haiku full raw/applied L1 callout from the exact Stage7 source registered by E4.'])

def _read_e4_stage7_sources(ctx: Any, run: Any) -> pd.DataFrame:
    inputs_path = _resolve_exact(ctx, basenames=('e4_budget_contract_input_files.csv',), contains_any=('e4_budget', 'e4_haiku', 'budget_contract_ladder'), role='E4 budget-contract input ledger')
    inputs = _read_frame(ctx, run, inputs_path, 'E4 budget-contract input ledger')
    _require_columns(inputs, {'configuration', 'path'}, context='E4 budget-contract input ledger')
    frames: list[pd.DataFrame] = []
    for row in inputs.drop_duplicates('path').itertuples(index=False):
        path = Path(str(row.path)).resolve()
        if not path.is_file():
            raise SpecializedEvidenceMissing(f'E4 Stage7 action artifact unavailable: {path}')
        frame = _read_frame(ctx, run, path, f'Stage7 action table for {row.configuration}')
        _require_columns(frame, {'policy', 'mode', 'budget_l1_raw', 'budget_l1_clipped'}, context=f'Stage7 action table {row.configuration}')
        frame = frame.loc[frame['policy'].astype(str).isin(['C4', 'C4R', 'C6']) & frame['mode'].astype(str).eq('free_form_10d')].copy()
        if frame.empty:
            continue
        frame['configuration'] = str(row.configuration)
        if 'budget_contract_label' in frame.columns:
            label = frame['budget_contract_label'].astype('string')
        else:
            label = pd.Series(pd.NA, index=frame.index, dtype='string')
        if 'budget_l1_target' in frame.columns:
            derived = frame['budget_l1_target'].map(_budget_label_from_target).astype('string')
            label = label.fillna(derived)
        frame['budget_label'] = label.fillna('unbounded').astype(str)
        frames.append(frame)
    if not frames:
        raise ValueError('E4 input ledger yielded no C4/C4R/C6 free-form rows')
    return pd.concat(frames, ignore_index=True, sort=False)

def _ecdf(values: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    array = np.sort(pd.to_numeric(values, errors='coerce').dropna().to_numpy(dtype=float))
    if not len(array):
        return (np.array([]), np.array([]))
    return (array, np.arange(1, len(array) + 1, dtype=float) / len(array))

def _build_raw_applied_l1_distribution(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    frame = _read_e4_stage7_sources(ctx, run)
    raw = _finite_numeric(frame['budget_l1_raw'], context='E7 raw L1', allow_missing=True)
    applied = _finite_numeric(frame['budget_l1_clipped'], context='E7 applied L1', allow_missing=True)
    long = pd.concat([frame[['configuration', 'budget_label', 'policy']].assign(l1_type='raw', l1_value=raw), frame[['configuration', 'budget_label', 'policy']].assign(l1_type='applied', l1_value=applied)], ignore_index=True).dropna(subset=['l1_value'])
    if long.empty:
        raise ValueError('E7 has no finite raw/applied L1 rows')
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(10.4, 6.2))
    budget_labels = sorted(long['budget_label'].astype(str).unique(), key=_budget_sort_key)
    colors = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9']
    for idx, budget in enumerate(budget_labels):
        for l1_type, style in (('raw', '-'), ('applied', '--')):
            values = long.loc[long['budget_label'].astype(str).eq(budget) & long['l1_type'].eq(l1_type), 'l1_value']
            x, y = _ecdf(values)
            if len(x):
                ax.step(x, y, where='post', linestyle=style, color=colors[idx % len(colors)], label=f'{budget} · {l1_type}')
    targets = sorted({float(value) for value in pd.to_numeric(frame.get('budget_l1_target', pd.Series(dtype=float)), errors='coerce').dropna() if math.isfinite(float(value))})
    for target in targets:
        ax.axvline(target, color='#555555', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('raw/applied total L1')
    ax.set_ylabel('ECDF')
    ax.set_ylim(0, 1.01)
    ax.set_title(run.spec.title)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    long['finite_budget_targets'] = ';'.join((f'{value:.6g}' for value in targets))
    ctx.save_figure(run, fig, long)
    run.transformations.extend(['Loaded only Stage7 tables listed in the frozen E4 source ledger.', 'Rendered raw and post-clipping applied L1 ECDFs separately; no budget-missing rows were imputed.', 'Added vertical lines for every observed finite generation-time L1 target.'])

def _load_tost_contract(ctx: Any, run: Any, contract: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    cfg = contract.get('f5_equivalence_forest') or {}
    filename = str(cfg.get('preregistration_filename') or 'c4r_journal_extension_prereg_v3.json')
    path = ctx.config_dir / filename
    if not path.is_file():
        raise RuntimeError(f'TOST preregistration missing: {path}')
    payload = _read_json(ctx, run, path, 'C4R TOST preregistration')
    equivalence = payload.get('equivalence')
    if not isinstance(equivalence, dict):
        raise ValueError('C4R preregistration lacks equivalence contract')
    margin = float(equivalence.get('raw_score_margin'))
    expected = float(cfg.get('expected_margin', margin))
    if not math.isfinite(margin) or margin <= 0 or (not math.isclose(margin, expected, abs_tol=1e-12, rel_tol=0)):
        raise ValueError(f'TOST margin mismatch: observed={margin}, expected={expected}')
    if str(equivalence.get('primary_oracle')).lower() != 'alpha':
        raise ValueError('F5 equivalence forest is restricted to preregistered Oracle alpha')
    return (margin, payload)

def _build_did_equivalence_forest(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    path = _resolve_exact(ctx, basenames=('c4r_tost_interactions_alpha.csv',), contains_any=('tost', 'e3_c4r_journal'), role='C4R TOST interaction ledger')
    frame = _read_frame(ctx, run, path, 'C4R TOST interaction ledger')
    required = {'cohort_id', 'budget_label', 'component', 'oracle', 'mean_gap', 'ci_low', 'ci_high', 'equivalent', 'decision'}
    _require_columns(frame, required, context='C4R TOST interaction ledger')
    margin, _ = _load_tost_contract(ctx, run, contract)
    expected_rows = int((contract.get('f5_equivalence_forest') or {}).get('expected_interaction_rows', 18))
    if len(frame) != expected_rows:
        raise ValueError(f'F5 requires {expected_rows} TOST interaction rows; got {len(frame)}')
    if not frame['oracle'].astype(str).str.lower().eq('alpha').all():
        raise ValueError('F5 TOST interaction ledger contains a non-alpha Oracle')
    for column in ('mean_gap', 'ci_low', 'ci_high'):
        frame[column] = _finite_numeric(frame[column], context=f'F5 {column}')
    if (frame['ci_low'] > frame['mean_gap']).any() or (frame['ci_high'] < frame['mean_gap']).any():
        raise ValueError('F5 confidence intervals do not contain their estimates')
    if frame.duplicated(['cohort_id', 'budget_label', 'component']).any():
        raise ValueError('F5 TOST interaction ledger contains duplicate cohort/budget/component rows')
    data = frame.sort_values(['cohort_id', 'component', 'budget_label'], key=lambda s: s.map(_budget_sort_key) if s.name == 'budget_label' else s).reset_index(drop=True)
    labels = data['cohort_id'].astype(str) + ' · ' + data['budget_label'].astype(str) + ' · ' + data['component'].astype(str)
    y = np.arange(len(data))
    estimate = data['mean_gap'].to_numpy(dtype=float)
    lo = data['ci_low'].to_numpy(dtype=float)
    hi = data['ci_high'].to_numpy(dtype=float)
    equivalent = data['equivalent'].astype(bool).to_numpy()
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(10.2, max(6.2, 0.36 * len(data))))
    ax.axvspan(-margin, margin, color='#009E73', alpha=0.12, label=f'동등역 ±{margin:.3f}')
    for state, color, label in ((True, '#009E73', '실질적 동등'), (False, '#D55E00', '비동등/미결정')):
        mask = equivalent == state
        if mask.any():
            ax.errorbar(estimate[mask], y[mask], xerr=[estimate[mask] - lo[mask], hi[mask] - estimate[mask]], fmt='o', color=color, ecolor=color, capsize=3, label=label)
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel('finite − unbounded component effect')
    ax.set_title(run.spec.title)
    ax.grid(axis='x', alpha=0.22)
    ax.legend(frameon=False, loc='lower right')
    data = data.assign(equivalence_margin=margin)
    ctx.save_figure(run, fig, data)
    run.transformations.extend(['Bound F5 to the preregistered Oracle-alpha TOST interaction ledger rather than a generic DID token match.', 'Read the equivalence margin from the frozen preregistration and rendered it as a shaded decision region.', 'Colored estimates by the explicit equivalent decision while retaining confidence intervals.'])

def _build_win_tie_loss_stack(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    path = _resolve_exact(ctx, basenames=('c4r_matched_v3_contrasts.csv',), contains_any=('e3_c4r_journal', 'c4r_matched_v3'), role='C4R v3 contrast ledger')
    frame = _read_frame(ctx, run, path, 'C4R v3 contrast ledger')
    required = {'cohort_id', 'budget_label', 'oracle_backend', 'contrast', 'n_pairs', 'win_count', 'tie_count', 'loss_count'}
    _require_columns(frame, required, context='C4R v3 contrast ledger')
    package = frame.loc[frame['oracle_backend'].astype(str).eq('alpha') & frame['contrast'].astype(str).eq('reference_plus_revision_package')].copy()
    if len(package) != 8:
        raise ValueError(f'F7 requires exactly 8 alpha package cells; got {len(package)}')
    if package.duplicated(['cohort_id', 'budget_label']).any():
        raise ValueError('F7 package ledger contains duplicate cohort/budget cells')
    for column in ('n_pairs', 'win_count', 'tie_count', 'loss_count'):
        package[column] = _finite_numeric(package[column], context=f'F7 {column}').astype(int)
    if not (package[['win_count', 'tie_count', 'loss_count']].sum(axis=1) == package['n_pairs']).all():
        raise ValueError('F7 win/tie/loss counts do not sum to n_pairs')
    package['cell'] = package['cohort_id'].astype(str) + ' · ' + package['budget_label'].astype(str)
    long = package.melt(id_vars=['cell', 'cohort_id', 'budget_label', 'n_pairs'], value_vars=['win_count', 'tie_count', 'loss_count'], var_name='outcome', value_name='count')
    long['share'] = long['count'] / long['n_pairs']
    order = package.assign(_budget_sort=package['budget_label'].map(_budget_sort_key)).sort_values(['cohort_id', '_budget_sort'])['cell'].tolist()
    pivot = long.pivot(index='cell', columns='outcome', values='share').reindex(order).fillna(0)
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    bottom = np.zeros(len(pivot))
    mapping = [('win_count', '승', '#009E73'), ('tie_count', '동', '#999999'), ('loss_count', '패', '#D55E00')]
    for column, label, color in mapping:
        values = pivot[column].to_numpy(dtype=float)
        ax.bar(np.arange(len(pivot)), values, bottom=bottom, label=label, color=color)
        bottom += values
    ax.set_xticks(np.arange(len(pivot)), pivot.index, rotation=28, ha='right')
    ax.set_ylabel('기업 비중')
    ax.set_ylim(0, 1)
    ax.set_title(run.spec.title)
    ax.grid(axis='y', alpha=0.22)
    ax.legend(frameon=False, ncol=3)
    ctx.save_figure(run, fig, long)
    run.transformations.append('Restricted the 8-cell stack to the Oracle-alpha C6−C4 package contrast and verified exact count closure.')

def _build_axis_shapley_diverging(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    path = _resolve_exact(ctx, basenames=('c4r_axis_shapley_summary.csv',), contains_any=('axis_shapley', 'c4r_journal'), role='C4R axis Shapley summary')
    frame = _read_frame(ctx, run, path, 'C4R axis Shapley summary')
    required = {'cohort_id', 'policy_pair', 'budget_label', 'axis', 'oracle', 'mean'}
    _require_columns(frame, required, context='C4R axis Shapley summary')
    data = frame.loc[frame['oracle'].astype(str).eq('alpha')].copy()
    if data.empty:
        raise ValueError('F8 Shapley ledger has no Oracle-alpha rows')
    data['mean'] = _finite_numeric(data['mean'], context='F8 Shapley mean')
    if data.duplicated(['cohort_id', 'policy_pair', 'budget_label', 'axis']).any():
        raise ValueError('F8 Shapley ledger contains duplicate cell/axis rows')
    facets = list(dict.fromkeys(zip(data['cohort_id'].astype(str), data['policy_pair'].astype(str))))
    ncols = 2
    nrows = int(math.ceil(len(facets) / ncols))
    plt = ctx.matplotlib()
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, max(5.4, 4.4 * nrows)), squeeze=False)
    colors = {'0p75': '#D55E00', '1p27': '#E69F00', '2p00': '#0072B2', 'unbounded': '#009E73'}
    for ax, facet in zip(axes.ravel(), facets):
        cohort, pair = facet
        part = data.loc[data['cohort_id'].astype(str).eq(cohort) & data['policy_pair'].astype(str).eq(pair)].copy()
        axes_order = part.groupby('axis', as_index=False)['mean'].apply(lambda s: float(s.abs().mean())).sort_values('mean', ascending=True)['axis'].astype(str).tolist()
        budgets = sorted(part['budget_label'].astype(str).unique(), key=_budget_sort_key)
        width = 0.78 / max(len(budgets), 1)
        ybase = np.arange(len(axes_order))
        for index, budget in enumerate(budgets):
            sub = part.loc[part['budget_label'].astype(str).eq(budget)]
            lookup = dict(zip(sub['axis'].astype(str), sub['mean'].astype(float)))
            values = [lookup.get(axis, np.nan) for axis in axes_order]
            ypos = ybase - 0.39 + width / 2 + index * width
            alpha = 1.0 if 'gemini' in cohort.lower() and budget == '0p75' else 0.72
            edge = '#000000' if 'gemini' in cohort.lower() and budget == '0p75' else 'none'
            ax.barh(ypos, values, height=width, label=budget, color=colors.get(budget, '#777777'), alpha=alpha, edgecolor=edge)
        ax.axvline(0, color='#333333', linewidth=0.8)
        ax.set_yticks(ybase, axes_order)
        ax.set_title(f'{cohort} · {pair}')
        ax.grid(axis='x', alpha=0.2)
        if 'gemini' in cohort.lower() and '0p75' in set(part['budget_label'].astype(str)):
            ax.text(0.99, 0.02, 'Gemini 0.75 상쇄 셀', transform=ax.transAxes, ha='right', va='bottom', fontsize=8.5)
    for ax in axes.ravel()[len(facets):]:
        ax.axis('off')
    handles, labels = axes.ravel()[0].get_legend_handles_labels() if facets else ([], [])
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=min(4, len(labels)), loc='lower center')
    fig.suptitle(run.spec.title, fontsize=13, fontweight='bold')
    ctx.save_figure(run, fig, data)
    run.transformations.extend(['Rendered Oracle-alpha axis Shapley means by cohort, policy pair, and budget on a common diverging zero axis.', 'Marked the Gemini 0.75 cell explicitly to expose offsetting positive and negative axis contributions.'])

def _build_tost_complete_ledger(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    cfg = contract.get('g4_tost_coverage') or {}
    within_path = _resolve_exact(ctx, basenames=(str(cfg.get('within_arm_filename') or 'c4r_tost_within_arm_alpha.csv'),), contains_any=('tost', 'e3_c4r_journal'), role='TOST within-arm ledger')
    interaction_path = _resolve_exact(ctx, basenames=(str(cfg.get('interaction_filename') or 'c4r_tost_interactions_alpha.csv'),), contains_any=('tost', 'e3_c4r_journal'), role='TOST interaction ledger')
    metadata_path = _resolve_exact(ctx, basenames=(str(cfg.get('metadata_filename') or 'metadata.json'),), contains_all=('tost',), role='TOST metadata')
    within = _read_frame(ctx, run, within_path, 'TOST within-arm ledger')
    interaction = _read_frame(ctx, run, interaction_path, 'TOST interaction ledger')
    metadata = _read_json(ctx, run, metadata_path, 'TOST metadata')
    expected_within = int(cfg.get('expected_within_arm_rows', 24))
    expected_interaction = int(cfg.get('expected_interaction_rows', 18))
    expected_total = int(cfg.get('expected_total_rows', expected_within + expected_interaction))
    if len(within) != expected_within or len(interaction) != expected_interaction:
        raise ValueError(f'G4 TOST coverage mismatch: within={len(within)}/{expected_within}, interaction={len(interaction)}/{expected_interaction}')
    required = {'test_family', 'cohort_id', 'budget_label', 'component', 'oracle', 'mean_gap', 'equivalent', 'decision'}
    _require_columns(within, required, context='TOST within-arm ledger')
    _require_columns(interaction, required, context='TOST interaction ledger')
    if not within['test_family'].astype(str).eq('within_arm_component_vs_zero').all():
        raise ValueError('G4 within-arm ledger contains an unexpected test_family')
    if not interaction['test_family'].astype(str).eq('finite_minus_unbounded_interaction_vs_zero').all():
        raise ValueError('G4 interaction ledger contains an unexpected test_family')
    if not pd.concat([within['oracle'], interaction['oracle']]).astype(str).str.lower().eq(str(cfg.get('primary_oracle') or 'alpha')).all():
        raise ValueError('G4 TOST ledgers contain a non-primary Oracle')
    within_keys = ['cohort_id', 'budget_label', 'component']
    interaction_keys = ['cohort_id', 'budget_label', 'component']
    if within.duplicated(within_keys).any() or interaction.duplicated(interaction_keys).any():
        raise ValueError('G4 TOST ledger contains duplicate contract keys')
    if int(metadata.get('within_arm_row_count', -1)) != expected_within or int(metadata.get('interaction_row_count', -1)) != expected_interaction:
        raise ValueError('G4 TOST metadata row counts disagree with the ledgers')
    margin = float(metadata.get('equivalence_margin'))
    expected_margin = float((contract.get('f5_equivalence_forest') or {}).get('expected_margin', 0.109))
    if not math.isclose(margin, expected_margin, abs_tol=1e-12, rel_tol=0):
        raise ValueError(f'G4 TOST margin mismatch: {margin} vs {expected_margin}')
    combined = pd.concat([within.assign(ledger_scope='within_arm'), interaction.assign(ledger_scope='finite_minus_unbounded')], ignore_index=True, sort=False)
    if len(combined) != expected_total:
        raise RuntimeError(f'G4 combined ledger must contain {expected_total} rows; got {len(combined)}')
    combined['coverage_contract'] = f'{expected_within} within-arm + {expected_interaction} interactions'
    combined['equivalence_margin_registered'] = margin
    ctx.write_table(run, combined, note=f'전수 계약: within-arm {expected_within}행 + finite-minus-unbounded interaction {expected_interaction}행 = {expected_total}행.')
    run.transformations.append('Validated and concatenated the complete TOST ledgers under exact row-count, family, key, Oracle, and margin contracts.')

def _format_contract_rows(cell_means: pd.DataFrame, *, primary_oracle: str) -> pd.DataFrame:
    required = {'backend_label', 'harness_cell', 'oracle_backend', 'mean_score'}
    _require_columns(cell_means, required, context='main harness cell means')
    work = cell_means.loc[cell_means['oracle_backend'].astype(str).eq(primary_oracle)].copy()
    if work.empty:
        raise ValueError(f'No cell means for Oracle={primary_oracle}')
    split = work['harness_cell'].astype(str).str.rsplit('__', n=1, expand=True)
    if split.shape[1] != 2:
        raise ValueError('harness_cell must be policy__mode')
    work['policy'] = split[0]
    work['mode'] = split[1]
    pivot = work.pivot(index=['backend_label', 'policy'], columns='mode', values='mean_score')
    required_modes = {'candidate_selection', 'free_form_10d'}
    if not required_modes.issubset(set(pivot.columns)):
        raise ValueError(f'Format contrast requires modes={sorted(required_modes)}; observed={list(pivot.columns)}')
    differences = pd.to_numeric(pivot['candidate_selection'], errors='raise') - pd.to_numeric(pivot['free_form_10d'], errors='raise')
    estimate, lo, hi, n = _normal_ci(differences)
    return pd.DataFrame([{'lever_family': '출력형식 계약', 'lever': 'candidate_selection − free_form_10d', 'estimate': estimate, 'ci_low': lo, 'ci_high': hi, 'n_units': n, 'ci_method': 'normal_95_over_backend_policy_cell_pairs', 'source_contract': 'main_harness_backend_cell_means'}])

def _matching_rows(shuffle: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    required = {'policy', 'mode', 'oracle_backend', 'shuffle_within', 'matching_gain_mean', 'matching_gain_ci_lower_2p5', 'matching_gain_ci_upper_97p5', 'n_shuffle_draws'}
    _require_columns(shuffle, required, context='shuffle permutation CI')
    target = shuffle.loc[shuffle['policy'].astype(str).eq(str(cfg.get('matching_policy') or 'C6')) & shuffle['mode'].astype(str).eq(str(cfg.get('matching_mode') or 'free_form_10d')) & shuffle['oracle_backend'].astype(str).eq(str(cfg.get('primary_oracle') or 'alpha')) & shuffle['shuffle_within'].astype(str).eq(str(cfg.get('matching_shuffle_within') or 'industry'))].copy()
    if len(target) != 1:
        raise ValueError(f'N1 matching lever requires one canonical shuffle-CI row; got {len(target)}')
    row = target.iloc[0]
    return pd.DataFrame([{'lever_family': '문맥 정렬 계약', 'lever': f"{row['policy']} industry-matching gain", 'estimate': float(row['matching_gain_mean']), 'ci_low': float(row['matching_gain_ci_lower_2p5']), 'ci_high': float(row['matching_gain_ci_upper_97p5']), 'n_units': int(row['n_shuffle_draws']), 'ci_method': 'empirical_shuffle_2p5_97p5', 'source_contract': 'shuffle_permutation_ci'}])

def _reference_rows(contrasts: pd.DataFrame, firm: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.DataFrame:
    required_contrast = {'cohort_id', 'budget_label', 'oracle_backend', 'contrast', 'mean_increment', 'n_pairs'}
    _require_columns(contrasts, required_contrast, context='C4R v3 contrasts')
    target = contrasts.loc[contrasts['oracle_backend'].astype(str).eq(str(cfg.get('primary_oracle') or 'alpha')) & contrasts['contrast'].astype(str).eq(str(cfg.get('reference_contrast') or 'reference_content_conditional'))].copy()
    if target.empty:
        raise ValueError('N1 contrasts contain no canonical alpha reference-content rows')
    _require_columns(firm, {'cohort_id', 'budget_label', 'reference_content_conditional_alpha'}, context='C4R v3 firm frame')
    rows: list[dict[str, Any]] = []
    for item in target.itertuples(index=False):
        values = firm.loc[firm['cohort_id'].astype(str).eq(str(item.cohort_id)) & firm['budget_label'].astype(str).eq(str(item.budget_label)), 'reference_content_conditional_alpha']
        estimate, lo, hi, n = _normal_ci(values)
        if not math.isclose(estimate, float(item.mean_increment), abs_tol=1e-12, rel_tol=0):
            raise ValueError(f'N1 firm-frame mean does not reproduce contrast mean for {item.cohort_id}/{item.budget_label}')
        rows.append({'lever_family': '외부참조 계약', 'lever': f'{item.cohort_id} · {item.budget_label}', 'estimate': estimate, 'ci_low': lo, 'ci_high': hi, 'n_units': n, 'ci_method': 'normal_95_over_firm_paired_effects', 'source_contract': 'c4r_matched_v3_contrasts+firm_frame'})
    return pd.DataFrame(rows)

def _build_harness_lever_common_scale(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    cfg = contract.get('n1_common_scale') or {}
    layout = build_layout(ctx.analysis_dir)
    cell_path = layout.main_harness_backend_decomposition / 'main_harness_backend_cell_means.csv'
    if not cell_path.is_file():
        raise SpecializedEvidenceMissing(f'Missing canonical main-harness cell means: {cell_path}')
    matching_run_role = str(cfg.get('matching_run_role') or 'paper_primary_gpt54')
    matching_ic = str(cfg.get('matching_information_condition') or 'IC-b')
    matching_cell = str(cfg.get('matching_cell') or 'row_shuffle_vector_null_native_industry')
    selected = _catalog_selected_run(ctx, run, run_role=matching_run_role, information_condition=matching_ic)
    selected_run_label = str(selected['run_label'])
    shuffle_path = layout.ablation / matching_ic / selected_run_label / matching_cell / 'shuffle_permutation_ci.csv'
    if not shuffle_path.is_file():
        raise SpecializedEvidenceMissing(f'Missing canonical IC-b industry-shuffle CI for the selected paper run: {shuffle_path}')
    e3_root = layout.e3 / 'v3_matched'
    contrasts_path = e3_root / 'c4r_matched_v3_contrasts.csv'
    if not contrasts_path.is_file():
        raise SpecializedEvidenceMissing(f'Missing canonical E3 contrasts: {contrasts_path}')
    firm_path = _canonical_existing_file(candidates=(e3_root / 'c4r_matched_v3_firm_frame.parquet', e3_root / 'c4r_matched_v3_firm_frame.csv'), role='E3 C4R v3 firm frame')
    cell = _read_frame(ctx, run, cell_path, 'main harness cell means')
    shuffle = _read_frame(ctx, run, shuffle_path, 'shuffle permutation CI')
    contrasts = _read_frame(ctx, run, contrasts_path, 'C4R v3 contrasts')
    firm = _read_frame(ctx, run, firm_path, 'C4R v3 firm frame')
    format_rows = _format_contract_rows(cell, primary_oracle=str(cfg.get('primary_oracle') or 'alpha'))
    format_rows['information_condition'] = 'IC-b'
    format_rows['run_role'] = 'main_harness_backend_decomposition'
    format_rows['run_label'] = 'cross_backend_icb_panel'
    matching_rows = _matching_rows(shuffle, cfg)
    matching_rows['information_condition'] = matching_ic
    matching_rows['run_role'] = matching_run_role
    matching_rows['run_label'] = selected_run_label
    reference_rows = _reference_rows(contrasts, firm, cfg)
    reference_rows['information_condition'] = 'IC-b'
    reference_rows['run_role'] = 'c4r_journal_extension'
    reference_rows['run_label'] = reference_rows['lever']
    data = pd.concat([format_rows, matching_rows, reference_rows], ignore_index=True)
    for column in ('estimate', 'ci_low', 'ci_high'):
        data[column] = _finite_numeric(data[column], context=f'G5 {column}')
    data = data.sort_values(['lever_family', 'lever']).reset_index(drop=True)
    labels = data['lever_family'].astype(str) + ' · ' + data['lever'].astype(str)
    y = np.arange(len(data))
    colors = {'출력형식 계약': '#0072B2', '문맥 정렬 계약': '#E69F00', '외부참조 계약': '#009E73'}
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(10.4, max(5.6, 0.42 * len(data))))
    for family in data['lever_family'].astype(str).unique():
        mask = data['lever_family'].astype(str).eq(family).to_numpy()
        estimate = data.loc[mask, 'estimate'].to_numpy(dtype=float)
        low = data.loc[mask, 'ci_low'].to_numpy(dtype=float)
        high = data.loc[mask, 'ci_high'].to_numpy(dtype=float)
        if np.any(low > estimate) or np.any(high < estimate):
            raise ValueError(f'G5 confidence interval does not contain estimate for family={family}')
        ax.errorbar(estimate, y[mask], xerr=[estimate - low, high - estimate], fmt='o', capsize=4, color=colors.get(family, '#777777'), ecolor=colors.get(family, '#777777'), label=family)
    ax.axvline(0, color='#333333', linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel('Oracle-alpha effect on a common raw-score scale')
    ax.set_title(run.spec.title)
    ax.grid(axis='x', alpha=0.22)
    ax.legend(frameon=False)
    ctx.save_figure(run, fig, data)
    run.transformations.extend(['Resolved every G5 source from the canonical paper-output layout rather than global basename search.', f'Bound the context-alignment lever to selected run_role={matching_run_role}, information_condition={matching_ic}, run_label={selected_run_label}.', 'Placed three distinct harness lever families on the common Oracle-alpha raw-score scale.', "Preserved each source's registered uncertainty unit and labeled the CI method instead of pooling incompatible units.", 'Recomputed reference-content confidence intervals from firm-paired E3 effects and checked their means against the v3 contrast ledger.'])

def _build_dynamic_resolution_panel(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    path = _resolve_exact(ctx, basenames=('dynamic_resolution.csv',), contains_all=('dynamic_resolution',), role='dynamic evaluator-resolution summary')
    frame = _read_frame(ctx, run, path, 'dynamic evaluator-resolution summary')
    required = {'cohort_id', 'budget_label', 'oracle_backend', 'n', 'zero_n', 'positive_n', 'negative_n', 'paired_gap_q10', 'paired_gap_q50', 'paired_gap_q90'}
    _require_columns(frame, required, context='dynamic evaluator-resolution summary')
    data = frame.loc[frame['oracle_backend'].astype(str).eq('alpha')].copy()
    if len(data) != 8:
        raise ValueError(f'G6 requires exactly 8 Oracle-alpha cohort/budget cells; got {len(data)}')
    if data.duplicated(['cohort_id', 'budget_label']).any():
        raise ValueError('G6 contains duplicate cohort/budget cells')
    for column in ('n', 'zero_n', 'positive_n', 'negative_n'):
        data[column] = _finite_numeric(data[column], context=f'G6 {column}').astype(int)
    if not (data[['zero_n', 'positive_n', 'negative_n']].sum(axis=1) == data['n']).all():
        raise ValueError('G6 zero/positive/negative counts do not sum to n')
    for column in ('paired_gap_q10', 'paired_gap_q50', 'paired_gap_q90'):
        data[column] = _finite_numeric(data[column], context=f'G6 {column}')
    data['cell'] = data['cohort_id'].astype(str) + ' · ' + data['budget_label'].astype(str)
    data = data.assign(_budget_sort=data['budget_label'].map(_budget_sort_key)).sort_values(['cohort_id', '_budget_sort']).drop(columns='_budget_sort')

    def draw_left(ax: Any) -> None:
        x = np.arange(len(data))
        bottom = np.zeros(len(data))
        for column, label, color in (('zero_n', '정확한 0', '#999999'), ('positive_n', '양', '#009E73'), ('negative_n', '음', '#D55E00')):
            share = data[column].to_numpy(dtype=float) / data['n'].to_numpy(dtype=float)
            ax.bar(x, share, bottom=bottom, label=label, color=color)
            bottom += share
        ax.set_xticks(x, data['cell'], rotation=28, ha='right')
        ax.set_ylim(0, 1)
        ax.set_ylabel('기업 비중')
        ax.set_title('효과=0/양/음 관측해상도')
        ax.grid(axis='y', alpha=0.22)
        ax.legend(frameon=False)

    def draw_right(ax: Any) -> None:
        y = np.arange(len(data))
        median = data['paired_gap_q50'].to_numpy(dtype=float)
        lo = data['paired_gap_q10'].to_numpy(dtype=float)
        hi = data['paired_gap_q90'].to_numpy(dtype=float)
        ax.errorbar(median, y, xerr=[median - lo, hi - median], fmt='o', capsize=3)
        ax.axvline(0, color='#333333', linewidth=0.8)
        ax.set_yticks(y, data['cell'])
        ax.set_xlabel('paired gap q10–q90')
        ax.set_title('비영 효과분포')
        ax.grid(axis='x', alpha=0.22)
    plot = data.assign(zero_share=data['zero_n'] / data['n'], positive_share=data['positive_n'] / data['n'], negative_share=data['negative_n'] / data['n'])
    _save_two_panel(ctx, run, left={'draw': draw_left}, right={'draw': draw_right}, data=plot)
    run.transformations.append('Verified eight Oracle-alpha cells and rendered exact-zero resolution separately from the nonzero paired-gap distribution.')

def _build_e2_e3_rank_correlation(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    e2, e2_manifest = _canonical_e2_evidence(ctx, run)
    e3 = _canonical_e3_firm_frame(ctx, run)
    cfg = contract.get('n2_e2_e3_rank_correlation') or {}
    components = [str(x) for x in cfg.get('components') or ('self_revision_alpha', 'reference_content_conditional_alpha')]
    budgets = [str(x) for x in cfg.get('preferred_budgets') or ('0p75', 'unbounded')]
    _require_columns(e2, {'row_id', 'budget_label', *components}, context='E2 firm frame')
    _require_columns(e3, {'cohort_id', 'backend_id', 'row_id', 'budget_label', *components}, context='E3 firm frame')
    e2_backend_id = str(e2_manifest['backend_id'])
    e3_backend_rows = e3.loc[e3['backend_id'].astype(str).eq(e2_backend_id)].copy()
    matched_cohorts = sorted(e3_backend_rows['cohort_id'].astype(str).unique())
    if len(matched_cohorts) != 1:
        raise ValueError(f'E2/E3 lineage must resolve one E3 cohort by backend_id, not by row-key overlap: e2_backend_id={e2_backend_id!r}, matched_cohorts={matched_cohorts}')
    cohort = matched_cohorts[0]
    e2 = e2.loc[e2['budget_label'].astype(str).isin(budgets)].copy()
    if set(e2['budget_label'].astype(str)) != set(budgets):
        raise ValueError(f'E2 firm frame does not contain preferred budgets={budgets}')
    if e2.duplicated(['row_id', 'budget_label']).any():
        raise ValueError('E2 firm frame contains duplicate row_id/budget keys')
    e3_match = e3_backend_rows.loc[e3_backend_rows['cohort_id'].astype(str).eq(cohort) & e3_backend_rows['budget_label'].astype(str).isin(budgets)].copy()
    if set(e3_match['budget_label'].astype(str)) != set(budgets):
        raise ValueError(f'Matched E3 cohort={cohort} does not contain preferred budgets={budgets}')
    if e3_match.duplicated(['row_id', 'budget_label']).any():
        raise ValueError('Matched E3 cohort contains duplicate row_id/budget keys')
    e2_keys = set(map(tuple, e2[['row_id', 'budget_label']].astype({'budget_label': str}).itertuples(index=False, name=None)))
    e3_keys = set(map(tuple, e3_match[['row_id', 'budget_label']].astype({'budget_label': str}).itertuples(index=False, name=None)))
    if e2_keys != e3_keys:
        missing = sorted(e2_keys - e3_keys)[:10]
        extra = sorted(e3_keys - e2_keys)[:10]
        raise ValueError(f'E2/E3 row_id×budget universes differ after lineage matching: missing_in_e3={missing}, extra_in_e3={extra}, e2_n={len(e2_keys)}, e3_n={len(e3_keys)}')
    merged = e2[['row_id', 'budget_label', *components]].merge(e3_match[['row_id', 'budget_label', *components]], on=['row_id', 'budget_label'], how='inner', validate='one_to_one', suffixes=('__E2', '__E3'))
    if len(merged) != len(e2):
        raise ValueError('E2/E3 merge lost firm/budget rows')
    panels: list[dict[str, Any]] = []
    plot_rows: list[pd.DataFrame] = []
    for budget in budgets:
        for component in components:
            part = merged.loc[merged['budget_label'].astype(str).eq(budget), ['row_id', f'{component}__E2', f'{component}__E3']].copy()
            part.columns = ['row_id', 'e2_effect', 'e3_effect']
            part['e2_effect'] = _finite_numeric(part['e2_effect'], context=f'I5 E2 {budget}/{component}')
            part['e3_effect'] = _finite_numeric(part['e3_effect'], context=f'I5 E3 {budget}/{component}')
            if len(part) < 3:
                raise ValueError(f'I5 requires at least three paired firms per panel; got {len(part)}')
            rho = float(part['e2_effect'].rank(method='average').corr(part['e3_effect'].rank(method='average')))
            if not math.isfinite(rho):
                raise ValueError(f'I5 rank correlation is undefined because one component is constant after exact lineage matching: {budget}/{component}')
            part['budget_label'] = budget
            part['component'] = component
            part['spearman_rho'] = rho
            part['e2_backend_id'] = e2_backend_id
            part['matched_e3_backend_id'] = e2_backend_id
            part['matched_e3_cohort'] = cohort
            part['cohort_match_contract'] = 'exact_backend_id_then_exact_row_id_x_budget'
            plot_rows.append(part)
            panels.append({'budget': budget, 'component': component, 'rho': rho, 'data': part})
    if len(panels) != 4:
        raise RuntimeError(f'I5 must render 2 budgets × 2 components = 4 panels; got {len(panels)}')
    plt = ctx.matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.2), squeeze=False)
    for ax, panel in zip(axes.ravel(), panels):
        part = panel['data']
        ax.scatter(part['e2_effect'], part['e3_effect'], s=18, alpha=0.45)
        lo = float(min(part['e2_effect'].min(), part['e3_effect'].min()))
        hi = float(max(part['e2_effect'].max(), part['e3_effect'].max()))
        ax.plot([lo, hi], [lo, hi], linestyle='--', color='#777777', linewidth=0.9)
        ax.axhline(0, color='#BBBBBB', linewidth=0.6)
        ax.axvline(0, color='#BBBBBB', linewidth=0.6)
        ax.set_xlabel('E2 firm component effect')
        ax.set_ylabel('E3 firm component effect')
        ax.set_title(f"{panel['budget']} · {panel['component']}\nSpearman ρ={panel['rho']:.3f}")
        ax.grid(alpha=0.18)
    fig.suptitle(run.spec.title + f' · backend={e2_backend_id} · matched E3 cohort={cohort}', fontsize=13, fontweight='bold')
    ctx.save_figure(run, fig, pd.concat(plot_rows, ignore_index=True))
    run.transformations.extend(['Resolved E2 and E3 from canonical layout paths instead of global basename search.', 'Matched the independent E2 and E3 executions by exact backend_id lineage before checking row_id×budget equality.', 'Did not infer cohort identity from row-key overlap because the E3 producer intentionally enforces a common firm universe across cohorts.', 'Computed four firm-level Spearman correlations for two components across 0.75 and unbounded arms.'])

def _build_repeat_stability(ctx: Any, run: Any, contract: Mapping[str, Any]) -> None:
    cfg = contract.get('n10_repeat_stability') or {}
    filename = str(cfg.get('filename') or 'llm_repeat_stability.csv')
    candidates = [path for path in ctx.index.files if path.name.lower() == filename.lower()]
    if not candidates:
        raise SpecializedEvidenceMissing(f'I6 requires the conditional live repeat-cohort artifact {filename}; RL seven-seed artifacts are not substituted')
    if len(candidates) != 1:
        raise RuntimeError(f'Ambiguous I6 repeat-stability artifacts: {[str(path) for path in candidates]}')
    frame = _read_frame(ctx, run, candidates[0], 'LLM live repeat-stability ledger')
    required = {str(x) for x in cfg.get('required_columns') or []}
    _require_columns(frame, required, context='LLM live repeat-stability ledger')
    if frame.duplicated(['backend_id', 'information_condition', 'repeat_a', 'repeat_b']).any():
        raise ValueError('I6 repeat-stability ledger contains duplicate repeat pairs')
    for column, bounds in (('action_identity_rate', (0.0, 1.0)), ('effect_rank_spearman_alpha', (-1.0, 1.0))):
        values = _finite_numeric(frame[column], context=f'I6 {column}')
        if not values.between(*bounds).all():
            raise ValueError(f'I6 {column} outside bounds={bounds}')
        frame[column] = values
    summary = frame.groupby(['backend_id', 'information_condition'], as_index=False).agg(repeat_pair_count=('repeat_a', 'size'), firm_count_min=('n_firms', 'min'), mean_action_identity_rate=('action_identity_rate', 'mean'), mean_effect_rank_spearman_alpha=('effect_rank_spearman_alpha', 'mean'))
    ctx.write_table(run, summary, note=str(cfg.get('interpretation_boundary') or 'Aggregate repeat stability does not imply firm-level identity.'))
    run.transformations.append('Aggregated only the explicit live LLM repeat-cohort ledger; no RL seed artifact was used as a fallback.')

def build_specialized_asset(ctx: Any, run: Any) -> bool:
    """Build a specialised asset and return ``True`` when dispatched here."""
    handler = str(run.spec.handler)
    if handler not in SPECIALIZED_HANDLERS:
        return False
    contract = _contract(ctx, run)
    dispatch = {'test3_panel': _build_test3_panel, 'capture_definitions': _build_capture_definitions, 'performance_verifiability_frontier': _build_performance_verifiability_frontier, 'budget_compliance_ladder': _build_budget_compliance_ladder, 'raw_applied_l1_distribution': _build_raw_applied_l1_distribution, 'did_equivalence_forest': _build_did_equivalence_forest, 'win_tie_loss_stack': _build_win_tie_loss_stack, 'axis_shapley_diverging': _build_axis_shapley_diverging, 'tost_complete_ledger': _build_tost_complete_ledger, 'harness_lever_common_scale': _build_harness_lever_common_scale, 'dynamic_resolution_panel': _build_dynamic_resolution_panel, 'e2_e3_rank_correlation': _build_e2_e3_rank_correlation, 'repeat_stability': _build_repeat_stability}
    dispatch[handler](ctx, run, contract)
    return True
