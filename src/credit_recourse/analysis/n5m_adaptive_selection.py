from __future__ import annotations
'Leakage-controlled exploratory selection analyses for thesis Section 9.8.\n\nTwo distinct out-of-fold analyses are produced from the canonical N5M matched\nbudget panel:\n\n1. a pre-action firm-state Ridge selector that chooses one of the four action\n   budgets separately for C4 and C6; and\n2. a post-C4, pre-C6 Ridge gate that decides whether the C6 revision call should\n   be accepted using only features observable after C4 has been generated.\n\nThe module is deliberately downstream of :mod:`credit_recourse.analysis.n5m_posthoc`.\nIt never calls an LLM, refits an Oracle, mutates frozen stage artifacts, or uses\nC6/revision outcomes as predictors.  Every firm is held out as a group across\nall four budgets, preprocessing is fitted inside each training fold, and all\nfirm-level predictions and fold assignments are persisted for independent\nrecalculation of the reported summaries.\n'
import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from credit_recourse.contracts.stage_paths import stage_dir
SCHEMA_VERSION = 'n5m_adaptive_selection_v1'
EXPECTED_BUDGETS = ('0p75', '1p27', '2p00', 'unbounded')
EXPECTED_POLICIES = ('C4', 'C6')
EXPECTED_FIRMS = 575
EXPECTED_FIRM_BUDGET_ROWS = EXPECTED_FIRMS * len(EXPECTED_BUDGETS)

@dataclass(frozen=True)
class FoldSpec:
    analysis: str
    repeat: int
    fold: int
    train_groups: np.ndarray
    test_groups: np.ndarray

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f'Required table is missing: {path}')
    suffix = path.suffix.lower()
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if suffix in {'.csv', '.txt'}:
        return pd.read_csv(path)
    raise ValueError(f'Unsupported table format: {path}')

def _load_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'Section 9.8 selection contract is missing: {path}')
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if payload.get('schema_version') != 'n5m_adaptive_selection_contract_v1':
        raise ValueError(f"Unexpected Section 9.8 contract schema: {payload.get('schema_version')!r}")
    if payload.get('status') != 'ACTIVE':
        raise ValueError(f"Section 9.8 selection contract is not ACTIVE: {payload.get('status')!r}")
    if tuple(payload.get('budget_order', [])) != EXPECTED_BUDGETS:
        raise ValueError('Section 9.8 contract budget order is not the canonical four-arm N5M order')
    return payload

def _validate_firm_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {'budget_label', 'row_id', 'initial_delta_R_score_alpha', 'revised_delta_R_score_alpha', 'revision_delta_R_score_alpha', 'c4_final_l1', 'c4_projection_distance', 'c4_active_dimensions'}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f'Canonical N5M firm frame is missing Section 9.8 columns. Re-run n5m_posthoc_v4 from the frozen four-arm inputs. Missing={missing}')
    work = frame.copy()
    work['budget_label'] = work['budget_label'].astype(str)
    work['row_id'] = pd.to_numeric(work['row_id'], errors='raise').astype(int)
    if len(work) != EXPECTED_FIRM_BUDGET_ROWS:
        raise ValueError(f'N5M firm frame rows={len(work)}, expected={EXPECTED_FIRM_BUDGET_ROWS}')
    if work.duplicated(['row_id', 'budget_label']).any():
        raise ValueError('N5M firm frame contains duplicate row_id x budget_label keys')
    if work['row_id'].nunique() != EXPECTED_FIRMS:
        raise ValueError(f"N5M firm count={work['row_id'].nunique()}, expected={EXPECTED_FIRMS}")
    observed_budgets = tuple((budget for budget in EXPECTED_BUDGETS if budget in set(work['budget_label'])))
    if observed_budgets != EXPECTED_BUDGETS or set(work['budget_label']) != set(EXPECTED_BUDGETS):
        raise ValueError(f"N5M budget labels are not canonical: {sorted(set(work['budget_label']))}")
    per_firm = work.groupby('row_id')['budget_label'].nunique()
    if not per_firm.eq(len(EXPECTED_BUDGETS)).all():
        raise ValueError('Every N5M firm must have exactly four budget rows')
    numeric = ['initial_delta_R_score_alpha', 'revised_delta_R_score_alpha', 'revision_delta_R_score_alpha', 'c4_final_l1', 'c4_projection_distance', 'c4_active_dimensions']
    for column in numeric:
        work[column] = pd.to_numeric(work[column], errors='coerce')
        if work[column].isna().any() or not np.isfinite(work[column].to_numpy(float)).all():
            raise ValueError(f'N5M Section 9.8 column contains non-finite values: {column}')
    identity_error = (work['initial_delta_R_score_alpha'] + work['revision_delta_R_score_alpha'] - work['revised_delta_R_score_alpha']).abs().max()
    if float(identity_error) > 1e-10:
        raise ValueError(f'N5M score additivity failed before Section 9.8 analysis: max_error={identity_error}')
    return work.sort_values(['row_id', 'budget_label']).reset_index(drop=True)

def _forbidden_state_columns(columns: Iterable[str], contract: dict[str, Any]) -> list[str]:
    state_cfg = contract['adaptive_budget']
    forbidden_exact = {str(value) for value in state_cfg.get('forbidden_exact', [])}
    forbidden_prefixes = tuple((str(value) for value in state_cfg.get('forbidden_prefixes', [])))
    forbidden_substrings = tuple((str(value).lower() for value in state_cfg.get('forbidden_substrings', [])))
    out: list[str] = []
    for raw in columns:
        column = str(raw)
        lowered = column.lower()
        if column in forbidden_exact:
            out.append(column)
        elif forbidden_prefixes and column.startswith(forbidden_prefixes):
            out.append(column)
        elif any((token and token in lowered for token in forbidden_substrings)):
            out.append(column)
    return sorted(set(out))

def _select_state_features(state_panel: pd.DataFrame, contract: dict[str, Any]) -> tuple[list[str], list[str]]:
    cfg = contract['adaptive_budget']
    numeric_exact = [str(value) for value in cfg.get('numeric_exact_candidates', [])]
    numeric_prefixes = tuple((str(value) for value in cfg.get('numeric_prefixes', [])))
    categorical_candidates = [str(value) for value in cfg.get('categorical_candidates', [])]
    numeric: list[str] = []
    for column in state_panel.columns:
        name = str(column)
        if name in numeric_exact or (numeric_prefixes and name.startswith(numeric_prefixes)):
            converted = pd.to_numeric(state_panel[name], errors='coerce')
            if converted.notna().any():
                numeric.append(name)
    categorical = [column for column in categorical_candidates if column in state_panel.columns and state_panel[column].notna().any()]
    numeric = list(dict.fromkeys(numeric))
    categorical = [column for column in dict.fromkeys(categorical) if column not in numeric]
    selected = [*numeric, *categorical]
    forbidden_selected = _forbidden_state_columns(selected, contract)
    if forbidden_selected:
        raise ValueError(f'The Section 9.8 allow-list selected leakage-protected state fields: {forbidden_selected}')
    if len(selected) < int(cfg.get('minimum_feature_count', 1)):
        raise ValueError(f'Too few pre-action state features satisfy the frozen allow-list: numeric={numeric}, categorical={categorical}')
    return (numeric, categorical)

def _materialize_stage7_compatible_row_id(state_panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Materialize the same serving-row key used by Stage 7.

    ``phase_eval_candidate.parquet`` is intentionally state-only and the frozen
    Stage2 producer does not guarantee a persisted ``row_id`` column.  Stage 7
    has always resolved that case by resetting the table to a zero-based
    ``RangeIndex`` and assigning that index as ``row_id``.  Section 9.8 must use
    exactly the same rule because its historical N5M firm frame is keyed by the
    Stage7 row identifiers.

    This is not a permissive positional merge: the complete derived row-id
    universe is checked against the N5M firm frame immediately afterwards.
    """
    state = state_panel.copy()
    input_had_row_id = 'row_id' in state.columns
    if input_had_row_id:
        source = 'explicit_column'
    else:
        state = state.reset_index(drop=True)
        state.insert(0, 'row_id', np.arange(len(state), dtype=np.int64))
        source = 'stage7_range_index_fallback'
    numeric_row_id = pd.to_numeric(state['row_id'], errors='raise')
    if numeric_row_id.isna().any():
        raise ValueError('phase_eval_candidate contains null row_id values')
    row_id_values = numeric_row_id.to_numpy(dtype=float)
    if not np.isfinite(row_id_values).all() or not np.equal(row_id_values, np.floor(row_id_values)).all():
        raise ValueError('phase_eval_candidate row_id values must be finite integers')
    state['row_id'] = numeric_row_id.astype(np.int64)
    if state.duplicated('row_id').any():
        raise ValueError('phase_eval_candidate contains duplicate evaluation row_id values')
    ordered_ids = state['row_id'].astype(int).tolist()
    contract = {'source': source, 'input_had_row_id': bool(input_had_row_id), 'materialization_rule': 'preserve explicit row_id' if input_had_row_id else 'reset_index(drop=True); row_id = zero_based_range_index', 'stage7_compatibility_contract': 'final_stage7_llm_action_generation._apply_stage7_row_selection', 'row_count': int(len(state)), 'row_id_min': int(min(ordered_ids)) if ordered_ids else None, 'row_id_max': int(max(ordered_ids)) if ordered_ids else None}
    return (state, contract)

def _prepare_state_panel(state_panel: pd.DataFrame, firm_ids: set[int], contract: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str], dict[str, Any]]:
    state, row_id_contract = _materialize_stage7_compatible_row_id(state_panel)
    observed_ids = set(map(int, state['row_id'].tolist()))
    if observed_ids != firm_ids:
        missing = sorted(firm_ids - observed_ids)
        extra = sorted(observed_ids - firm_ids)
        raise ValueError(f'State-panel/N5M row universe mismatch: missing={missing[:10]}, extra={extra[:10]}')
    numeric, categorical = _select_state_features(state, contract)
    keep = ['row_id', *numeric, *categorical]
    prepared = state[keep].sort_values('row_id').reset_index(drop=True)
    row_id_contract = {**row_id_contract, 'universe_match_status': 'PASS'}
    return (prepared, numeric, categorical, row_id_contract)

def _iter_group_folds(groups: Iterable[int], *, analysis: str, n_splits: int, repeats: int, seed: int) -> Iterator[FoldSpec]:
    unique = np.array(sorted({int(value) for value in groups}), dtype=np.int64)
    if n_splits < 2 or n_splits > len(unique):
        raise ValueError(f'Invalid group-fold count: n_splits={n_splits}, groups={len(unique)}')
    for repeat in range(repeats):
        shuffled = unique.copy()
        np.random.default_rng(int(seed) + repeat).shuffle(shuffled)
        chunks = np.array_split(shuffled, n_splits)
        for fold, test_groups in enumerate(chunks):
            test_set = set((int(value) for value in test_groups.tolist()))
            train_groups = np.array([value for value in unique if int(value) not in test_set], dtype=np.int64)
            if set(map(int, train_groups)).intersection(test_set):
                raise RuntimeError('Group leakage detected while constructing Section 9.8 folds')
            yield FoldSpec(analysis=analysis, repeat=repeat, fold=fold, train_groups=train_groups, test_groups=np.asarray(test_groups, dtype=np.int64))

def _build_ridge_pipeline(*, numeric_features: list[str], categorical_features: list[str], alpha: float) -> Pipeline:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())])
        transformers.append(('numeric', numeric_pipe, numeric_features))
    if categorical_features:
        categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        transformers.append(('categorical', categorical_pipe, categorical_features))
    if not transformers:
        raise ValueError('Ridge pipeline requires at least one feature')
    preprocess = ColumnTransformer(transformers=transformers, remainder='drop')
    return Pipeline([('preprocess', preprocess), ('ridge', Ridge(alpha=float(alpha), fit_intercept=True))])

def _coefficient_rows(model: Pipeline, *, analysis: str, repeat: int, fold: int, policy: str, budget_label: str | None) -> list[dict[str, Any]]:
    preprocess: ColumnTransformer = model.named_steps['preprocess']
    ridge: Ridge = model.named_steps['ridge']
    feature_names = [str(value) for value in preprocess.get_feature_names_out()]
    coefficients = np.asarray(ridge.coef_, dtype=float).reshape(-1)
    if len(feature_names) != len(coefficients):
        raise RuntimeError('Ridge feature-name/coef length mismatch')
    rows = [{'analysis': analysis, 'repeat': repeat, 'fold': fold, 'policy': policy, 'budget_label': budget_label, 'feature': feature, 'coefficient': float(value), 'intercept': float(ridge.intercept_)} for feature, value in zip(feature_names, coefficients)]
    return rows

def _adaptive_budget_analysis(*, firm_frame: pd.DataFrame, state_panel: pd.DataFrame, numeric_features: list[str], categorical_features: list[str], contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = contract['adaptive_budget']
    score_columns = {'C4': 'initial_delta_R_score_alpha', 'C6': 'revised_delta_R_score_alpha'}
    outcomes: dict[str, pd.DataFrame] = {}
    for policy, score_column in score_columns.items():
        pivot = firm_frame.pivot(index='row_id', columns='budget_label', values=score_column)
        pivot = pivot.reindex(columns=list(EXPECTED_BUDGETS)).sort_index()
        if pivot.isna().any().any() or len(pivot) != EXPECTED_FIRMS:
            raise ValueError(f'Adaptive-budget outcome panel is incomplete for {policy}')
        outcomes[policy] = pivot
    state = state_panel.set_index('row_id').sort_index()
    if set(state.index) != set(outcomes['C4'].index):
        raise ValueError('Adaptive-budget state and outcome row universes differ')
    X_all = state[[*numeric_features, *categorical_features]]
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[dict[str, Any]] = []
    folds = list(_iter_group_folds(X_all.index, analysis='adaptive_budget', n_splits=int(cfg['outer_folds']), repeats=int(cfg['repeats']), seed=int(cfg['seed'])))
    for spec in folds:
        train_ids = list(map(int, spec.train_groups))
        test_ids = list(map(int, spec.test_groups))
        for row_id in test_ids:
            fold_assignment_rows.append({'analysis': spec.analysis, 'repeat': spec.repeat, 'fold': spec.fold, 'row_id': row_id})
        X_train = X_all.loc[train_ids]
        X_test = X_all.loc[test_ids]
        for policy in EXPECTED_POLICIES:
            observed = outcomes[policy]
            predicted_by_budget: dict[str, np.ndarray] = {}
            for budget_label in EXPECTED_BUDGETS:
                model = _build_ridge_pipeline(numeric_features=numeric_features, categorical_features=categorical_features, alpha=float(cfg['ridge_alpha']))
                model.fit(X_train, observed.loc[train_ids, budget_label].to_numpy(float))
                predicted_by_budget[budget_label] = np.asarray(model.predict(X_test), dtype=float)
                coefficient_rows.extend(_coefficient_rows(model, analysis=spec.analysis, repeat=spec.repeat, fold=spec.fold, policy=policy, budget_label=budget_label))
            predicted_matrix = np.column_stack([predicted_by_budget[budget] for budget in EXPECTED_BUDGETS])
            selected_indices = np.argmax(predicted_matrix, axis=1)
            selected_labels = [EXPECTED_BUDGETS[int(index)] for index in selected_indices]
            test_actual = observed.loc[test_ids, list(EXPECTED_BUDGETS)]
            selected_scores = np.array([float(test_actual.loc[row_id, budget_label]) for row_id, budget_label in zip(test_ids, selected_labels)])
            frozen_baseline = str(cfg['fixed_budget_baseline_label'])
            training_means = observed.loc[train_ids, list(EXPECTED_BUDGETS)].mean(axis=0)
            train_best_label = str(training_means.idxmax())
            fixed_scores = test_actual[frozen_baseline].to_numpy(float)
            train_best_scores = test_actual[train_best_label].to_numpy(float)
            hindsight_scores = test_actual.max(axis=1).to_numpy(float)
            fold_metric_rows.append({'analysis': spec.analysis, 'repeat': spec.repeat, 'fold': spec.fold, 'policy': policy, 'n_train_firms': len(train_ids), 'n_test_firms': len(test_ids), 'selected_mean_score': float(np.mean(selected_scores)), 'frozen_fixed_budget_label': frozen_baseline, 'frozen_fixed_mean_score': float(np.mean(fixed_scores)), 'training_best_fixed_budget_label': train_best_label, 'training_best_fixed_mean_score_on_test': float(np.mean(train_best_scores)), 'hindsight_budget_oracle_mean_score': float(np.mean(hindsight_scores)), 'gap_vs_frozen_fixed': float(np.mean(selected_scores - fixed_scores))})
            for test_position, row_id in enumerate(test_ids):
                selected_label = selected_labels[test_position]
                for budget_position, budget_label in enumerate(EXPECTED_BUDGETS):
                    prediction_rows.append({'analysis': spec.analysis, 'repeat': spec.repeat, 'fold': spec.fold, 'policy': policy, 'row_id': row_id, 'budget_label': budget_label, 'predicted_score': float(predicted_matrix[test_position, budget_position]), 'observed_score': float(test_actual.loc[row_id, budget_label]), 'selected_budget_label': selected_label, 'selected_flag': bool(budget_label == selected_label), 'selected_observed_score': float(selected_scores[test_position]), 'frozen_fixed_observed_score': float(fixed_scores[test_position]), 'hindsight_budget_oracle_score': float(hindsight_scores[test_position])})
    predictions = pd.DataFrame(prediction_rows)
    assignments = pd.DataFrame(fold_assignment_rows).drop_duplicates()
    coefficients = pd.DataFrame(coefficient_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    repeat_rows: list[dict[str, Any]] = []
    for (repeat, policy), group in predictions.groupby(['repeat', 'policy'], sort=True):
        selected = group.loc[group['selected_flag']].copy()
        if len(selected) != EXPECTED_FIRMS:
            raise RuntimeError(f'Adaptive-budget OOF coverage failed: repeat={repeat}, policy={policy}')
        selected_mean = float(selected['selected_observed_score'].mean())
        fixed_mean = float(selected['frozen_fixed_observed_score'].mean())
        hindsight_mean = float(selected['hindsight_budget_oracle_score'].mean())
        denominator = hindsight_mean - fixed_mean
        repeat_rows.append({'analysis': 'adaptive_budget', 'repeat': int(repeat), 'policy': policy, 'n_firms': int(len(selected)), 'selector_mean_score': selected_mean, 'frozen_fixed_budget_label': str(cfg['fixed_budget_baseline_label']), 'frozen_fixed_mean_score': fixed_mean, 'gap_vs_frozen_fixed': selected_mean - fixed_mean, 'hindsight_budget_oracle_mean_score': hindsight_mean, 'hindsight_headroom': denominator, 'headroom_capture_fraction': (selected_mean - fixed_mean) / denominator if denominator > 0 else float('nan')})
    repeat_summary = pd.DataFrame(repeat_rows)
    summary = repeat_summary.groupby(['analysis', 'policy', 'frozen_fixed_budget_label'], as_index=False).agg(repeats=('repeat', 'nunique'), n_firms=('n_firms', 'max'), selector_mean_score=('selector_mean_score', 'mean'), selector_repeat_sd=('selector_mean_score', 'std'), frozen_fixed_mean_score=('frozen_fixed_mean_score', 'mean'), gap_vs_frozen_fixed=('gap_vs_frozen_fixed', 'mean'), hindsight_budget_oracle_mean_score=('hindsight_budget_oracle_mean_score', 'mean'), hindsight_headroom=('hindsight_headroom', 'mean'), headroom_capture_fraction=('headroom_capture_fraction', 'mean'))
    return (predictions, assignments, coefficients, fold_metrics, pd.concat([summary.assign(summary_level='aggregate'), repeat_summary.assign(summary_level='repeat')], ignore_index=True, sort=False))

def _post_c4_gate_analysis(*, firm_frame: pd.DataFrame, contract: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = contract['post_c4_gate']
    feature_columns = [str(value) for value in cfg['feature_columns']]
    missing = sorted(set(feature_columns) - set(firm_frame.columns))
    if missing:
        raise ValueError(f'Post-C4 gate features are missing from N5M firm frame: {missing}')
    target_column = str(cfg['target_column'])
    if target_column not in firm_frame.columns:
        raise ValueError(f'Post-C4 gate target is missing: {target_column}')
    work = firm_frame.copy().sort_values(['row_id', 'budget_label']).reset_index(drop=True)
    for column in [*feature_columns, target_column]:
        work[column] = pd.to_numeric(work[column], errors='coerce')
        if work[column].isna().any() or not np.isfinite(work[column].to_numpy(float)).all():
            raise ValueError(f'Post-C4 gate column contains non-finite values: {column}')
    prediction_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    fold_metric_rows: list[dict[str, Any]] = []
    fold_assignment_rows: list[dict[str, Any]] = []
    folds = list(_iter_group_folds(work['row_id'], analysis='post_c4_gate', n_splits=int(cfg['outer_folds']), repeats=int(cfg['repeats']), seed=int(cfg['seed'])))
    threshold = float(cfg['decision_threshold'])
    for spec in folds:
        train_mask = work['row_id'].isin(spec.train_groups)
        test_mask = work['row_id'].isin(spec.test_groups)
        train = work.loc[train_mask].copy()
        test = work.loc[test_mask].copy()
        if set(train['row_id']).intersection(set(test['row_id'])):
            raise RuntimeError('Post-C4 gate group leakage detected')
        for row_id in sorted(set(map(int, spec.test_groups))):
            fold_assignment_rows.append({'analysis': spec.analysis, 'repeat': spec.repeat, 'fold': spec.fold, 'row_id': row_id})
        model = _build_ridge_pipeline(numeric_features=feature_columns, categorical_features=[], alpha=float(cfg['ridge_alpha']))
        model.fit(train[feature_columns], train[target_column].to_numpy(float))
        predicted_gain = np.asarray(model.predict(test[feature_columns]), dtype=float)
        choose_c6 = predicted_gain > threshold
        c4_score = test['initial_delta_R_score_alpha'].to_numpy(float)
        c6_score = test['revised_delta_R_score_alpha'].to_numpy(float)
        selected_score = np.where(choose_c6, c6_score, c4_score)
        oracle_score = np.maximum(c4_score, c6_score)
        coefficient_rows.extend(_coefficient_rows(model, analysis=spec.analysis, repeat=spec.repeat, fold=spec.fold, policy='C4_vs_C6', budget_label=None))
        fold_metric_rows.append({'analysis': spec.analysis, 'repeat': spec.repeat, 'fold': spec.fold, 'n_train_rows': int(len(train)), 'n_test_rows': int(len(test)), 'n_train_firms': int(train['row_id'].nunique()), 'n_test_firms': int(test['row_id'].nunique()), 'c4_mean_score': float(c4_score.mean()), 'c6_mean_score': float(c6_score.mean()), 'gate_mean_score': float(selected_score.mean()), 'hindsight_oracle_mean_score': float(oracle_score.mean()), 'c6_selection_rate': float(choose_c6.mean())})
        for position, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append({'analysis': spec.analysis, 'repeat': spec.repeat, 'fold': spec.fold, 'row_id': int(row['row_id']), 'budget_label': str(row['budget_label']), **{column: float(row[column]) for column in feature_columns}, 'observed_revision_gain': float(row[target_column]), 'predicted_revision_gain': float(predicted_gain[position]), 'decision_threshold': threshold, 'select_c6': bool(choose_c6[position]), 'c4_score': float(c4_score[position]), 'c6_score': float(c6_score[position]), 'selected_score': float(selected_score[position]), 'hindsight_oracle_score': float(oracle_score[position])})
    predictions = pd.DataFrame(prediction_rows)
    assignments = pd.DataFrame(fold_assignment_rows).drop_duplicates()
    coefficients = pd.DataFrame(coefficient_rows)
    fold_metrics = pd.DataFrame(fold_metric_rows)
    repeat_rows: list[dict[str, Any]] = []
    for repeat, group in predictions.groupby('repeat', sort=True):
        if len(group) != EXPECTED_FIRM_BUDGET_ROWS:
            raise RuntimeError(f'Post-C4 gate OOF coverage failed for repeat={repeat}: rows={len(group)}')
        c4_mean = float(group['c4_score'].mean())
        c6_mean = float(group['c6_score'].mean())
        gate_mean = float(group['selected_score'].mean())
        oracle_mean = float(group['hindsight_oracle_score'].mean())
        best_unconditional = max(c4_mean, c6_mean)
        headroom = oracle_mean - best_unconditional
        repeat_rows.append({'analysis': 'post_c4_gate', 'repeat': int(repeat), 'n_firm_budget_rows': int(len(group)), 'n_firms': int(group['row_id'].nunique()), 'c4_mean_score': c4_mean, 'c6_mean_score': c6_mean, 'best_unconditional_policy': 'C6' if c6_mean >= c4_mean else 'C4', 'best_unconditional_mean_score': best_unconditional, 'gate_mean_score': gate_mean, 'gap_vs_best_unconditional': gate_mean - best_unconditional, 'hindsight_oracle_mean_score': oracle_mean, 'hindsight_headroom': headroom, 'headroom_capture_fraction': (gate_mean - best_unconditional) / headroom if headroom > 0 else float('nan'), 'c6_selection_rate': float(group['select_c6'].mean())})
    repeat_summary = pd.DataFrame(repeat_rows)
    summary = pd.DataFrame([{'analysis': 'post_c4_gate', 'summary_level': 'aggregate', 'repeats': int(repeat_summary['repeat'].nunique()), 'n_firm_budget_rows': EXPECTED_FIRM_BUDGET_ROWS, 'n_firms': EXPECTED_FIRMS, 'c4_mean_score': float(repeat_summary['c4_mean_score'].mean()), 'c6_mean_score': float(repeat_summary['c6_mean_score'].mean()), 'best_unconditional_policy': str(repeat_summary['best_unconditional_policy'].mode().iloc[0]), 'best_unconditional_mean_score': float(repeat_summary['best_unconditional_mean_score'].mean()), 'gate_mean_score': float(repeat_summary['gate_mean_score'].mean()), 'gate_repeat_sd': float(repeat_summary['gate_mean_score'].std(ddof=1)) if len(repeat_summary) > 1 else 0.0, 'gap_vs_best_unconditional': float(repeat_summary['gap_vs_best_unconditional'].mean()), 'hindsight_oracle_mean_score': float(repeat_summary['hindsight_oracle_mean_score'].mean()), 'hindsight_headroom': float(repeat_summary['hindsight_headroom'].mean()), 'headroom_capture_fraction': float(repeat_summary['headroom_capture_fraction'].mean()), 'c6_selection_rate': float(repeat_summary['c6_selection_rate'].mean())}])
    repeat_summary = repeat_summary.assign(summary_level='repeat')
    return (predictions, assignments, coefficients, fold_metrics, pd.concat([summary, repeat_summary], ignore_index=True, sort=False))

def _assert_prediction_contracts(*, adaptive_predictions: pd.DataFrame, adaptive_assignments: pd.DataFrame, gate_predictions: pd.DataFrame, gate_assignments: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    adaptive_repeats = int(contract['adaptive_budget']['repeats'])
    gate_repeats = int(contract['post_c4_gate']['repeats'])
    expected_adaptive_rows = adaptive_repeats * len(EXPECTED_POLICIES) * EXPECTED_FIRMS * len(EXPECTED_BUDGETS)
    expected_gate_rows = gate_repeats * EXPECTED_FIRM_BUDGET_ROWS
    errors: list[str] = []
    if len(adaptive_predictions) != expected_adaptive_rows:
        errors.append(f'adaptive prediction rows={len(adaptive_predictions)}, expected={expected_adaptive_rows}')
    if len(gate_predictions) != expected_gate_rows:
        errors.append(f'gate prediction rows={len(gate_predictions)}, expected={expected_gate_rows}')
    if adaptive_predictions.duplicated(['repeat', 'policy', 'row_id', 'budget_label']).any():
        errors.append('adaptive predictions contain duplicate OOF keys')
    if gate_predictions.duplicated(['repeat', 'row_id', 'budget_label']).any():
        errors.append('gate predictions contain duplicate OOF keys')
    selected_counts = adaptive_predictions.groupby(['repeat', 'policy'])['selected_flag'].sum()
    if not selected_counts.eq(EXPECTED_FIRMS).all():
        errors.append('adaptive selector does not select exactly one budget per firm')
    for name, assignments, repeats in (('adaptive', adaptive_assignments, adaptive_repeats), ('gate', gate_assignments, gate_repeats)):
        expected = repeats * EXPECTED_FIRMS
        if len(assignments) != expected:
            errors.append(f'{name} fold assignments rows={len(assignments)}, expected={expected}')
        if assignments.duplicated(['analysis', 'repeat', 'row_id']).any():
            errors.append(f'{name} fold assignments contain duplicate group keys')
    if errors:
        raise RuntimeError('; '.join(errors))
    return {'status': 'PASS', 'adaptive_prediction_rows': int(len(adaptive_predictions)), 'gate_prediction_rows': int(len(gate_predictions)), 'adaptive_fold_assignment_rows': int(len(adaptive_assignments)), 'gate_fold_assignment_rows': int(len(gate_assignments)), 'group_key': 'row_id', 'group_leakage_detected': False}

def run_analysis(*, project_root: Path, firm_frame_path: Path, state_panel_path: Path, contract_path: Path, output_dir: Path) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    firm_frame_path = Path(firm_frame_path).resolve()
    state_panel_path = Path(state_panel_path).resolve()
    contract_path = Path(contract_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contract = _load_contract(contract_path)
    firm_frame = _validate_firm_frame(_read_table(firm_frame_path))
    state_panel, numeric_features, categorical_features, state_row_id_contract = _prepare_state_panel(_read_table(state_panel_path), set(map(int, firm_frame['row_id'].unique())), contract)
    adaptive = _adaptive_budget_analysis(firm_frame=firm_frame, state_panel=state_panel, numeric_features=numeric_features, categorical_features=categorical_features, contract=contract)
    gate = _post_c4_gate_analysis(firm_frame=firm_frame, contract=contract)
    adaptive_predictions, adaptive_assignments, adaptive_coefficients, adaptive_fold_metrics, adaptive_summary = adaptive
    gate_predictions, gate_assignments, gate_coefficients, gate_fold_metrics, gate_summary = gate
    verification = _assert_prediction_contracts(adaptive_predictions=adaptive_predictions, adaptive_assignments=adaptive_assignments, gate_predictions=gate_predictions, gate_assignments=gate_assignments, contract=contract)
    feature_contract = {'schema_version': 'n5m_selection_feature_contract_v1', 'created_utc': _now(), 'status': 'PASS', 'state_panel': str(state_panel_path), 'state_key': 'row_id', 'state_row_id_contract': state_row_id_contract, 'state_numeric_features': numeric_features, 'state_categorical_features': categorical_features, 'state_feature_visibility_contract': contract['adaptive_budget'].get('feature_visibility_contract'), 'state_feature_visibility_source': contract['adaptive_budget'].get('feature_visibility_source'), 'state_numeric_allowlist': list(contract['adaptive_budget'].get('numeric_exact_candidates', [])), 'state_categorical_allowlist': list(contract['adaptive_budget'].get('categorical_candidates', [])), 'post_c4_gate_features': list(contract['post_c4_gate']['feature_columns']), 'post_c4_gate_feature_timing': 'observable_after_C4_before_C6_call', 'forbidden_state_columns': contract['adaptive_budget'].get('forbidden_exact', []), 'forbidden_state_prefixes': contract['adaptive_budget'].get('forbidden_prefixes', []), 'preprocessing': {'numeric': 'training-fold median imputation with missing indicator, then training-fold standardization', 'categorical': 'training-fold most-frequent imputation, then one-hot with unknown-category ignore'}, 'leakage_boundary': 'No next__, action__, reward, rating/Oracle score, candidate label, C6 output, or revision outcome is admitted to the pre-action selector. Post-C4 gate features are limited to the frozen contract list.'}
    outputs = {'manifest': output_dir / 'n5m_adaptive_selection_manifest.json', 'input_files': output_dir / 'n5m_adaptive_selection_input_files.csv', 'feature_contract': output_dir / 'n5m_selection_feature_contract.json', 'fold_assignments': output_dir / 'n5m_selection_fold_assignments.csv', 'adaptive_predictions': output_dir / 'n5m_adaptive_budget_oof_predictions.parquet', 'gate_predictions': output_dir / 'n5m_postc4_gate_oof_predictions.parquet', 'adaptive_summary': output_dir / 'n5m_adaptive_budget_summary.csv', 'gate_summary': output_dir / 'n5m_postc4_gate_summary.csv', 'fold_metrics': output_dir / 'n5m_selection_fold_metrics.csv', 'coefficients': output_dir / 'n5m_selection_coefficients.csv'}
    adaptive_predictions.to_parquet(outputs['adaptive_predictions'], index=False)
    gate_predictions.to_parquet(outputs['gate_predictions'], index=False)
    adaptive_summary.to_csv(outputs['adaptive_summary'], index=False, encoding='utf-8-sig')
    gate_summary.to_csv(outputs['gate_summary'], index=False, encoding='utf-8-sig')
    pd.concat([adaptive_fold_metrics, gate_fold_metrics], ignore_index=True, sort=False).to_csv(outputs['fold_metrics'], index=False, encoding='utf-8-sig')
    pd.concat([adaptive_coefficients, gate_coefficients], ignore_index=True, sort=False).to_csv(outputs['coefficients'], index=False, encoding='utf-8-sig')
    pd.concat([adaptive_assignments, gate_assignments], ignore_index=True, sort=False).to_csv(outputs['fold_assignments'], index=False, encoding='utf-8-sig')
    _write_json(outputs['feature_contract'], feature_contract)
    pd.DataFrame([
        {'role': 'firm_frame', 'path': str(firm_frame_path), 'size_bytes': int(firm_frame_path.stat().st_size)},
        {'role': 'state_panel', 'path': str(state_panel_path), 'size_bytes': int(state_panel_path.stat().st_size)},
        {'role': 'contract', 'path': str(contract_path), 'size_bytes': int(contract_path.stat().st_size)},
    ]).to_csv(outputs['input_files'], index=False, encoding='utf-8-sig')
    manifest = {'schema_version': SCHEMA_VERSION, 'created_utc': _now(), 'completed_utc': _now(), 'status': 'PASS', 'project_root': str(project_root), 'firm_frame_path': str(firm_frame_path), 'state_panel_path': str(state_panel_path), 'state_row_id_contract': state_row_id_contract, 'contract_path': str(contract_path), 'firm_count': EXPECTED_FIRMS, 'firm_budget_row_count': EXPECTED_FIRM_BUDGET_ROWS, 'budget_order': list(EXPECTED_BUDGETS), 'oracle_backend': 'alpha', 'adaptive_budget_contract': contract['adaptive_budget'], 'post_c4_gate_contract': contract['post_c4_gate'], 'feature_contract': feature_contract, 'verification': verification, 'evidence_tier': 'EVALUATOR_ONLY_EXPLORATORY_OOF', 'interpretation_boundaries': ['The selector and gate are exploratory analyses on the same frozen N5M panel.', "Firm-level group holdout prevents the same firm's four budgets from crossing train/test folds.", 'Out-of-fold prediction does not constitute independent-cohort validation.', 'The pre-action selector and post-C4 gate have different information sets and must not be conflated.', 'Exact thesis values must be updated if this frozen implementation does not reproduce an earlier unarchived analysis.'], 'outputs': {key: {'path': str(path), 'size_bytes': int(path.stat().st_size)} for key, path in outputs.items() if key != 'manifest'}}
    _write_json(outputs['manifest'], manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def _default_contract_path(project_root: Path) -> Path:
    return Path(project_root).resolve() / 'data' / 'final_freeze' / 'configs' / 'n5m_adaptive_selection_contract.json'

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--firm-frame', default=None)
    parser.add_argument('--state-panel', default=None)
    parser.add_argument('--contract', default=None)
    parser.add_argument('--output-dir', required=True)
    return parser

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    firm_frame = Path(args.firm_frame).resolve() if args.firm_frame else project_root / 'data' / 'analysis' / 'paper_repro' / '03_output_contract_diagnostics' / 'n5m_posthoc' / 'n5m_firm_frame.parquet'
    state_panel = Path(args.state_panel).resolve() if args.state_panel else stage_dir(project_root, 'stage2') / 'phase_eval_candidate.parquet'
    contract = Path(args.contract).resolve() if args.contract else _default_contract_path(project_root)
    result = run_analysis(project_root=project_root, firm_frame_path=firm_frame, state_panel_path=state_panel, contract_path=contract, output_dir=Path(args.output_dir))
    return 0 if result.get('status') == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
