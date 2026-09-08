from __future__ import annotations
import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import yaml
from credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline import _logistic_cdf, _transform_ordered_logit_thresholds, resolve_backend_artifact
from credit_recourse.oracle.backends.alpha.modules.oracle_alpha_scorer import build_alpha_scorer
from .claim_evidence_common import repo_rel, write_csv, write_json

def _load_registry(root: Path) -> tuple[dict[str, Any], Path]:
    candidates = [root / 'data/final_freeze/configs/oracle_backend_registry.yaml', root / 'src/credit_recourse/configs/oracle_backend_registry.yaml']
    matches = [path for path in candidates if path.is_file()]
    if not matches:
        raise FileNotFoundError('oracle_backend_registry.yaml not found')
    path = matches[0]
    payload = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict) or not isinstance(payload.get('backends'), dict):
        raise RuntimeError('invalid Oracle backend registry')
    return (payload, path)

def _alpha_grades(frame: pd.DataFrame, params_path: Path) -> np.ndarray:
    params = json.loads(params_path.read_text(encoding='utf-8'))
    variables = params.get('selected_variables') or [item.get('variable_id') for item in params.get('variables', []) if isinstance(item, dict)]
    variables = [value for value in variables if value]
    missing = sorted(set(variables) - set(frame.columns))
    if missing:
        raise RuntimeError(f'Alpha grade mapping missing features: {missing[:20]}')
    scorer = build_alpha_scorer(params)
    grades = []
    for _, row in frame.iterrows():
        result = scorer({variable: row.get(variable) for variable in variables})
        if 'R_grade' not in result:
            raise RuntimeError('Alpha scorer did not return R_grade')
        grades.append(int(result['R_grade']))
    return np.asarray(grades, dtype=int)

def _beta_grades(frame: pd.DataFrame, params_path: Path) -> np.ndarray:
    params = json.loads(params_path.read_text(encoding='utf-8'))
    variables = params.get('selected_variables') or []
    standardization = params.get('standardization_params') or {}
    records = params.get('coefficients') or []
    coefficients = {str(record.get('variable')): float(record.get('coefficient')) for record in records if str(record.get('variable')) in variables and record.get('coefficient') is not None}
    missing = [variable for variable in variables if variable not in frame.columns or variable not in standardization or variable not in coefficients]
    if missing:
        raise RuntimeError(f'Beta grade mapping missing inputs: {missing[:20]}')
    grade_numbers = [int(value) for value in params.get('modeled_grade_nums') or params.get('probability_output_grade_nums') or []]
    if len(grade_numbers) < 2:
        raise RuntimeError('Beta params missing modeled grade numbers')
    cutpoints = params.get('ordered_logit_finite_cutpoints') or params.get('finite_cutpoints')
    if cutpoints is None:
        raw = params.get('ordered_logit_threshold_raw_params') or []
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            raw = [record.get('coefficient') for record in raw]
        if not raw:
            raw = [record.get('coefficient') for record in records if str(record.get('variable')) not in set(variables) and record.get('coefficient') is not None]
        cutpoints = _transform_ordered_logit_thresholds([float(value) for value in raw]).tolist()
    cutpoints = np.asarray(cutpoints, dtype=float)
    if len(cutpoints) != len(grade_numbers) - 1:
        raise RuntimeError('Beta cutpoint/class mismatch')
    xb = np.zeros(len(frame), dtype=float)
    for variable in variables:
        mean = float(standardization[variable].get('mean', 0.0))
        scale = float(standardization[variable].get('std', 1.0)) or 1.0
        values = pd.to_numeric(frame[variable], errors='coerce').fillna(mean)
        xb += ((values - mean) / scale).to_numpy(dtype=float) * coefficients[variable]
    thresholds = np.concatenate([[-np.inf], cutpoints, [np.inf]])
    upper = _logistic_cdf(thresholds[1:][None, :] - xb[:, None])
    lower = _logistic_cdf(thresholds[:-1][None, :] - xb[:, None])
    upper[:, -1] = 1.0
    lower[:, 0] = 0.0
    probabilities = np.clip(upper - lower, 0.0, 1.0)
    denominators = probabilities.sum(axis=1, keepdims=True)
    probabilities = np.divide(probabilities, denominators, out=np.full_like(probabilities, 1.0 / probabilities.shape[1]), where=denominators > 1e-12)
    return np.asarray(grade_numbers, dtype=int)[probabilities.argmax(axis=1)]

def _gamma_grades(frame: pd.DataFrame, params_path: Path, model_path: Path) -> np.ndarray:
    params = json.loads(params_path.read_text(encoding='utf-8'))
    variables = params.get('selected_variables') or []
    missing = sorted(set(variables) - set(frame.columns))
    if missing:
        raise RuntimeError(f'Gamma grade mapping missing inputs: {missing[:20]}')
    model = joblib.load(model_path)
    prediction = np.asarray(model.predict(frame[variables].apply(pd.to_numeric, errors='coerce')), dtype=float)
    return np.rint(prediction).clip(1, 10).astype(int)

def _grade_map(root: Path, frame: pd.DataFrame, registry: dict[str, Any]) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    final_root = root / 'data/final_freeze'
    sources: list[dict[str, str]] = []
    result: dict[str, np.ndarray] = {}
    for oracle in ('alpha', 'beta', 'gamma'):
        config = registry['backends'][oracle]
        params = resolve_backend_artifact(root, final_root, config.get('params'))
        if not params.is_file():
            raise FileNotFoundError(params)
        sources.append({'oracle_backend': oracle, 'artifact': repo_rel(root, params)})
        if oracle == 'alpha':
            result[oracle] = _alpha_grades(frame, params)
        elif oracle == 'beta':
            result[oracle] = _beta_grades(frame, params)
        else:
            model = resolve_backend_artifact(root, final_root, config.get('model'))
            if not model.is_file():
                raise FileNotFoundError(model)
            sources.append({'oracle_backend': oracle, 'artifact': repo_rel(root, model)})
            result[oracle] = _gamma_grades(frame, params, model)
    return (result, sources)

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    firm_path = root / 'data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/v3_matched/c4r_matched_v3_firm_frame.parquet'
    if not firm_path.is_file():
        raise FileNotFoundError(firm_path)
    firm = pd.read_parquet(firm_path)
    required = {'run_label', 'cohort_id', 'budget_label', 'row_id'}
    required |= {f'{policy}_{oracle}' for policy in ('C4', 'C6') for oracle in ('alpha', 'beta', 'gamma')}
    if not required.issubset(firm.columns):
        raise RuntimeError(f'dynamic-resolution firm frame missing {sorted(required - set(firm.columns))}')
    registry, registry_path = _load_registry(root)
    run_dirs = {path.name: path for path in (root / 'data/final_freeze/llm_runs').iterdir() if path.is_dir()}
    firm_ledgers: list[pd.DataFrame] = []
    grade_sources: list[dict[str, str]] = [{'oracle_backend': 'registry', 'artifact': repo_rel(root, registry_path)}]
    input_sources: list[dict[str, str]] = [{'artifact': repo_rel(root, firm_path)}]
    for run_label, group in firm.groupby('run_label', sort=False):
        if run_label not in run_dirs:
            raise RuntimeError(f'dynamic-resolution run directory missing: {run_label}')
        simulated_path = run_dirs[run_label] / 'stage8_llm_multi_oracle_eval/simulated_oracle_input_frame.parquet'
        if not simulated_path.is_file():
            raise FileNotFoundError(simulated_path)
        simulated = pd.read_parquet(simulated_path)
        if not {'row_id', 'policy'}.issubset(simulated.columns):
            raise RuntimeError(f'{run_label}: simulated Oracle input missing row_id/policy')
        selected = simulated.loc[simulated['policy'].astype(str).isin(['C4', 'C6'])].copy()
        if len(selected) != 1150 or selected.duplicated(['row_id', 'policy']).any():
            raise RuntimeError(f'{run_label}: C4/C6 simulated input key contract failed')
        grades, sources = _grade_map(root, selected, registry)
        for oracle, values in grades.items():
            selected[f'grade_{oracle}'] = values
        grade_sources.extend(sources)
        wide_parts = []
        for policy in ('C4', 'C6'):
            columns = ['row_id', *[f'grade_{oracle}' for oracle in ('alpha', 'beta', 'gamma')]]
            part = selected.loc[selected['policy'].astype(str).eq(policy), columns].copy()
            part = part.rename(columns={column: f'{policy}_{column}' for column in columns if column != 'row_id'})
            wide_parts.append(part)
        grade_wide = wide_parts[0].merge(wide_parts[1], on='row_id', validate='one_to_one')
        score = group.copy()
        if len(score) != 575 or score['row_id'].duplicated().any():
            raise RuntimeError(f'{run_label}: firm score key contract failed')
        merged = score.merge(grade_wide, on='row_id', validate='one_to_one')
        firm_ledgers.append(merged)
        input_sources.append({'artifact': repo_rel(root, simulated_path)})
    ledger = pd.concat(firm_ledgers, ignore_index=True)
    if len(ledger) != 4600 or ledger.duplicated(['run_label', 'row_id']).any():
        raise RuntimeError('dynamic-resolution ledger cardinality/key contract failed')
    summary_rows: list[dict[str, Any]] = []
    firm_rows: list[dict[str, Any]] = []
    for (cohort, budget), group in ledger.groupby(['cohort_id', 'budget_label'], dropna=False):
        for oracle in ('alpha', 'beta', 'gamma'):
            gap = pd.to_numeric(group[f'C6_{oracle}'], errors='raise') - pd.to_numeric(group[f'C4_{oracle}'], errors='raise')
            crossing = group[f'C6_grade_{oracle}'].astype(int) != group[f'C4_grade_{oracle}'].astype(int)
            changed = gap.abs() > 1e-12
            absolute_total = float(gap.abs().sum())
            share = float(gap.loc[crossing].abs().sum() / absolute_total) if absolute_total > 1e-12 else 0.0
            summary_rows.append({'cohort_id': cohort, 'budget_label': budget, 'oracle_backend': oracle, 'n': len(group), 'zero_n': int((~changed).sum()), 'positive_n': int((gap > 1e-12).sum()), 'negative_n': int((gap < -1e-12).sum()), 'score_change_without_boundary_crossing_n': int((changed & ~crossing).sum()), 'grade_boundary_crossing_n': int(crossing.sum()), 'crossing_firm_absolute_contribution_share': share, 'paired_gap_mean': float(gap.mean()), 'paired_gap_sd': float(gap.std(ddof=1)), 'paired_gap_q10': float(gap.quantile(0.1)), 'paired_gap_q50': float(gap.quantile(0.5)), 'paired_gap_q90': float(gap.quantile(0.9))})
            for row_id, score_gap, crosses, grade_before, grade_after in zip(group['row_id'], gap, crossing, group[f'C4_grade_{oracle}'], group[f'C6_grade_{oracle}']):
                firm_rows.append({'cohort_id': cohort, 'budget_label': budget, 'row_id': row_id, 'oracle_backend': oracle, 'score_gap_C6_minus_C4': score_gap, 'C4_grade': int(grade_before), 'C6_grade': int(grade_after), 'grade_boundary_crossing': bool(crosses)})
    output_dir = root / 'data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/dynamic_resolution'
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'dynamic_resolution.csv'
    firm_out = output_dir / 'dynamic_resolution_firm_ledger.parquet'
    write_csv(summary_path, summary_rows)
    pd.DataFrame(firm_rows).to_parquet(firm_out, index=False)
    unique_grade_sources = list('0')
    manifest = {'schema_version': 'c4r_dynamic_resolution_v2', 'status': 'PASS', 'evidence_class': 'EXECUTION_FROZEN_EXTENSION', 'grade_mapping_contract': {'alpha': 'Oracle-alpha scorer R_grade from frozen params/cutpoints', 'beta': 'ordered-logit posterior MAP modeled grade', 'gamma': 'frozen tree-model predicted grade rounded and clipped to 1..10', 'forbidden_method': 'floor(score)'}, 'grade_mapping_sources': unique_grade_sources, 'input_sources': input_sources, 'summary_path': repo_rel(root, summary_path), 'firm_ledger_path': repo_rel(root, firm_out), 'summary_rows': len(summary_rows), 'firm_rows': len(firm_rows), 'crossing_contribution_definition': 'absolute score effect among crossing firms / total absolute score effect'}
    write_json(output_dir / 'dynamic_resolution_manifest.json', manifest)
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
