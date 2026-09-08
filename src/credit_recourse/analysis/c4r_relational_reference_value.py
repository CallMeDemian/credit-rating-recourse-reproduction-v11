from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from credit_recourse.contracts.paper_reproduction import discover_archived_runs, load_profile, select_budget_frontier_role
from .claim_evidence_common import repo_rel, write_csv, write_json
WARNINGS = ['EVALUATOR_ONLY_POST_HOC', 'SHARED_TERM_WARNING', 'MEAN_REVERSION_WARNING', 'NO_AUTOMATIC_ROUTING']

def _candidate_library(root: Path) -> tuple[dict[str, np.ndarray], list[str], list[dict[str, str]]]:
    files = sorted(set((root / 'data/final_freeze/stage2_candidate_projection').rglob('final_candidate_library__P50.yaml')) | set((root / 'data/final_freeze/configs').rglob('final_candidate_library__P50.yaml')))
    if not files:
        raise RuntimeError('P50 candidate YAML not found')
    payload = yaml.safe_load(files[0].read_text(encoding='utf-8-sig'))
    vectors: dict[str, np.ndarray] = {}
    axis_order: list[str] | None = None
    if isinstance(payload, dict) and isinstance(payload.get('fixed_candidates'), dict):
        fixed = payload['fixed_candidates']
        labels = payload.get('main_train_labels')
        if not isinstance(labels, list) or not labels:
            raise RuntimeError('P50 YAML fixed_candidates requires non-empty main_train_labels')
        if len(labels) != len(set(map(str, labels))):
            raise RuntimeError('P50 YAML main_train_labels contains duplicates')
        missing_labels = [str(label) for label in labels if str(label) not in fixed]
        if missing_labels:
            raise RuntimeError(f'P50 YAML main labels missing from fixed_candidates: {missing_labels}')
        for raw_label in labels:
            candidate_id = str(raw_label)
            record = fixed[candidate_id]
            if not isinstance(record, dict):
                raise RuntimeError(f'candidate {candidate_id} must be a mapping')
            axes = [str(key) for key in record if str(key).startswith('action__')]
            if len(axes) != 10:
                raise RuntimeError(f'candidate {candidate_id} has {len(axes)} action axes, expected 10')
            if axis_order is None:
                axis_order = axes
            elif axes != axis_order:
                raise RuntimeError(f'candidate {candidate_id} axis order differs from canonical order')
            vector = np.asarray([record[axis] for axis in axes], dtype=float)
            if vector.shape != (10,) or not np.isfinite(vector).all():
                raise RuntimeError(f'invalid candidate vector for {candidate_id}: shape={vector.shape}')
            vectors[candidate_id] = vector
    else:
        items = payload.get('candidates') if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            raise RuntimeError('P50 YAML must contain fixed_candidates/main_train_labels or a candidate list')
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError('legacy candidate item must be a mapping')
            candidate_id = str(item.get('id') or item.get('action_id') or item.get('candidate_id'))
            raw = item.get('vector') or item.get('action') or item.get('values')
            if isinstance(raw, dict):
                axes = [str(axis) for axis in raw.keys()]
                values = [raw[axis] for axis in raw]
            else:
                axes = [f'action__axis_{index}' for index in range(len(raw or []))]
                values = raw
            if axis_order is None:
                axis_order = axes
            elif axes != axis_order:
                raise RuntimeError('candidate YAML axis order is not consistent')
            vector = np.asarray(values, dtype=float)
            if vector.shape != (10,) or not np.isfinite(vector).all():
                raise RuntimeError(f'invalid candidate vector for {candidate_id}: shape={vector.shape}')
            if candidate_id in vectors:
                raise RuntimeError(f'duplicate candidate ID {candidate_id}')
            vectors[candidate_id] = vector
    if axis_order is None or len(axis_order) != 10:
        raise RuntimeError('failed to derive the canonical 10-axis candidate order')
    lineage = [{'path': repo_rel(root, path)} for path in files]
    return (vectors, axis_order, lineage)

def _assert_semantic_mirror(canonical: pd.DataFrame, mirror: pd.DataFrame, *, key_columns: list[str], required_columns: list[str], label: str) -> None:
    if len(canonical) != len(mirror):
        raise RuntimeError(f'{label} CSV mirror row count differs from canonical parquet')
    missing = sorted(set(required_columns) - set(canonical.columns))
    mirror_missing = sorted(set(required_columns) - set(mirror.columns))
    if missing or mirror_missing:
        raise RuntimeError(f'{label} mirror required columns missing: parquet={missing}, csv={mirror_missing}')
    left = canonical[required_columns].sort_values(key_columns).reset_index(drop=True)
    right = mirror[required_columns].sort_values(key_columns).reset_index(drop=True)
    for column in required_columns:
        if pd.api.types.is_numeric_dtype(left[column]) or pd.api.types.is_numeric_dtype(right[column]):
            a = pd.to_numeric(left[column], errors='coerce').to_numpy(float)
            b = pd.to_numeric(right[column], errors='coerce').to_numpy(float)
            if not np.allclose(a, b, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise RuntimeError(f'{label} CSV mirror numeric mismatch in {column}')
        elif not left[column].fillna('').astype(str).eq(right[column].fillna('').astype(str)).all():
            raise RuntimeError(f'{label} CSV mirror text mismatch in {column}')

def _load_stage_table(run_dir: Path, *, stage_dir: str, stem: str, key_columns: list[str], required_columns: list[str]) -> tuple[pd.DataFrame, list[Path]]:
    directory = run_dir / stage_dir
    parquet = directory / f'{stem}.parquet'
    csv = directory / f'{stem}.csv'
    sources = [path for path in (parquet, csv) if path.is_file()]
    if not sources:
        raise RuntimeError(f'missing {stem} under {run_dir}')
    if parquet.is_file():
        canonical = pd.read_parquet(parquet)
        if csv.is_file():
            mirror = pd.read_csv(csv, encoding='utf-8-sig')
            _assert_semantic_mirror(canonical, mirror, key_columns=key_columns, required_columns=required_columns, label=stem)
    else:
        canonical = pd.read_csv(csv, encoding='utf-8-sig')
    missing = sorted(set(required_columns) - set(canonical.columns))
    if missing:
        raise RuntimeError(f'{stem} required columns missing: {missing}')
    return (canonical, sources)

def _residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), controls])
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coefficients

def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    frame = pd.DataFrame({'x': x, 'y': y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(frame) < 3 or frame['x'].nunique() < 2 or frame['y'].nunique() < 2:
        return (None, None, len(frame))
    result = spearmanr(frame['x'], frame['y'])
    return (float(result.statistic), float(result.pvalue), len(frame))

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    profile = load_profile(root)
    records = discover_archived_runs(root, profile)
    e3 = profile['extensions']['e3']
    selected = []
    for cohort, role in e3['cohorts'].items():
        arms = select_budget_frontier_role(records, run_role=str(role), information_condition='IC-b', expected_budgets=e3['budgets'], expected_conditions=('C4', 'C4R', 'C6'))
        selected.extend(((cohort, arm) for arm in arms))
    if len(selected) != 8:
        raise RuntimeError(f'E3 relational-reference analysis requires eight run arms, got {len(selected)}')
    firm_path = root / 'data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/v3_matched/c4r_matched_v3_firm_frame.parquet'
    if not firm_path.is_file():
        raise FileNotFoundError(firm_path)
    firm = pd.read_parquet(firm_path)
    required_firm = {'run_label', 'row_id', 'C4_alpha', 'C4R_alpha', 'C6_alpha'}
    if not required_firm.issubset(firm.columns):
        raise RuntimeError(f'E3 firm frame missing {sorted(required_firm - set(firm.columns))}')
    vectors, yaml_axes, candidate_lineage = _candidate_library(root)
    action_columns = [axis if str(axis).startswith('action__') else f'action__{axis}' for axis in yaml_axes]
    all_rows: list[pd.DataFrame] = []
    input_lineage: list[dict[str, Any]] = [{'path': repo_rel(root, firm_path)}]
    for cohort, record in selected:
        score = firm.loc[firm['run_label'].astype(str).eq(record.run_label)].copy()
        if len(score) != 575:
            raise RuntimeError(f'{record.run_label}: E3 firm frame rows={len(score)}, expected=575')
        required_stage7 = ['row_id', 'policy', 'mode', 'rl_reference_candidate', *action_columns]
        stage7, stage7_sources = _load_stage_table(record.run_dir, stage_dir='stage7_llm_action_generation', stem='llm_stage7_action_table', key_columns=['row_id', 'policy', 'mode'], required_columns=required_stage7)
        required_stage9 = ['row_id', 'policy', 'mode', 'delta_R_score_alpha']
        stage9, stage9_sources = _load_stage_table(record.run_dir, stage_dir='stage9_llm_rl_comparison', stem='llm_stage9_llm_rl_comparison', key_columns=['row_id', 'policy', 'mode'], required_columns=required_stage9)
        actions = stage7.loc[stage7['mode'].astype(str).eq('free_form_10d') & stage7['policy'].astype(str).isin(['C4', 'C4R', 'C6'])].copy()
        if actions.duplicated(['row_id', 'policy']).any() or len(actions) != 1725:
            raise RuntimeError(f'{record.run_label}: Stage7 C4/C4R/C6 key contract failed')
        pieces = []
        for policy in ('C4', 'C4R', 'C6'):
            part = actions.loc[actions['policy'].eq(policy), ['row_id', 'rl_reference_candidate', *action_columns]].copy()
            part = part.rename(columns={column: f'{policy}_{column}' for column in action_columns})
            if policy != 'C6':
                part = part.drop(columns=['rl_reference_candidate'])
            pieces.append(part)
        merged_actions = pieces[0].merge(pieces[1], on='row_id', validate='one_to_one').merge(pieces[2], on='row_id', validate='one_to_one')
        c3 = stage9.loc[stage9['policy'].astype(str).eq('C3_candidate_iql') & stage9['mode'].astype(str).eq('rl_native'), ['row_id', 'delta_R_score_alpha']].rename(columns={'delta_R_score_alpha': 'C3_alpha'})
        if len(c3) != 575 or c3['row_id'].duplicated().any():
            raise RuntimeError(f'{record.run_label}: Stage9 C3 score contract failed')
        ledger = score.merge(c3, on='row_id', validate='one_to_one').merge(merged_actions, on='row_id', validate='one_to_one')
        c4_matrix = ledger[[f'C4_{column}' for column in action_columns]].to_numpy(dtype=float)
        c4r_matrix = ledger[[f'C4R_{column}' for column in action_columns]].to_numpy(dtype=float)
        c6_matrix = ledger[[f'C6_{column}' for column in action_columns]].to_numpy(dtype=float)
        reference_matrix = np.vstack([vectors.get(str(candidate_id), np.full(10, np.nan)) for candidate_id in ledger['rl_reference_candidate']])
        if not np.isfinite(reference_matrix).all():
            missing = sorted(set(ledger.loc[~np.isfinite(reference_matrix).all(axis=1), 'rl_reference_candidate'].astype(str)))
            raise RuntimeError(f'{record.run_label}: unknown reference candidate IDs {missing}')
        distance_before = np.abs(c4_matrix - reference_matrix).sum(axis=1)
        distance_after = np.abs(c6_matrix - reference_matrix).sum(axis=1)
        abs_c4 = np.abs(c4_matrix)
        abs_c6 = np.abs(c6_matrix)
        removal_mass = np.maximum(abs_c4 - abs_c6, 0.0).sum(axis=1)
        addition_mass = np.maximum(abs_c6 - abs_c4, 0.0).sum(axis=1)
        ledger['local_reference_advantage'] = ledger['C3_alpha'] - ledger['C4_alpha']
        ledger['reference_gain'] = ledger['C6_alpha'] - ledger['C4R_alpha']
        ledger['shared_term_alternative_advantage'] = ledger['C3_alpha'] - ledger['C4R_alpha']
        ledger['adoption_distance_reduction'] = distance_before - distance_after
        adoption_rate = np.zeros_like(distance_before, dtype=float)
        np.divide(distance_before - distance_after, distance_before, out=adoption_rate, where=distance_before > 1e-12)
        ledger['adoption_rate'] = adoption_rate
        ledger['removal_mass'] = removal_mass
        ledger['addition_mass'] = addition_mass
        ledger['revision_type'] = np.where(addition_mass >= removal_mass, 'COMPLEMENT', 'SUBSTITUTE')
        ledger['cohort_id'] = cohort
        if record.freeform_l1_budget is None:
            budget_label = 'unbounded'
        else:
            budget_label = {0.75: '0p75', 1.27: '1p27', 2.0: '2p00'}.get(round(float(record.freeform_l1_budget), 2), str(record.freeform_l1_budget).replace('.', 'p'))
        ledger['budget_label'] = budget_label
        ledger['run_role'] = record.run_role
        all_rows.append(ledger)
        input_lineage.extend(({'path': repo_rel(root, source)} for source in [*stage7_sources, *stage9_sources]))
    ledger = pd.concat(all_rows, ignore_index=True)
    if len(ledger) != 4600 or ledger.duplicated(['run_label', 'row_id']).any():
        raise RuntimeError('E3 relational-reference firm ledger cardinality/key contract failed')
    ledger['advantage_quartile'] = ledger.groupby(['cohort_id', 'budget_label'], dropna=False)['local_reference_advantage'].transform(lambda series: pd.qcut(series.rank(method='first'), 4, labels=['Q1', 'Q2', 'Q3', 'Q4']))
    output_dir = root / 'data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/relational_reference'
    output_dir.mkdir(parents=True, exist_ok=True)
    firm_columns = ['cohort_id', 'run_role', 'run_label', 'budget_label', 'row_id', 'C4_alpha', 'C4R_alpha', 'C6_alpha', 'C3_alpha', 'local_reference_advantage', 'shared_term_alternative_advantage', 'reference_gain', 'adoption_distance_reduction', 'adoption_rate', 'removal_mass', 'addition_mass', 'revision_type', 'advantage_quartile']
    firm_out = output_dir / 'relational_reference_firm_ledger.parquet'
    ledger[firm_columns].to_parquet(firm_out, index=False)
    quartiles = ledger.groupby(['cohort_id', 'budget_label', 'advantage_quartile'], observed=True, dropna=False).agg(n=('row_id', 'size'), mean_local_reference_advantage=('local_reference_advantage', 'mean'), mean_reference_gain=('reference_gain', 'mean'), mean_adoption_rate=('adoption_rate', 'mean'), mean_removal_mass=('removal_mass', 'mean'), mean_addition_mass=('addition_mass', 'mean')).reset_index()
    quartile_out = output_dir / 'relational_reference_quartile_summary.csv'
    write_csv(quartile_out, quartiles.to_dict('records'))
    categories = ledger.groupby(['cohort_id', 'budget_label', 'revision_type'], dropna=False).agg(n=('row_id', 'size'), mean_reference_gain=('reference_gain', 'mean'), mean_adoption_rate=('adoption_rate', 'mean')).reset_index()
    category_out = output_dir / 'relational_reference_complement_substitute.csv'
    write_csv(category_out, categories.to_dict('records'))
    analysis_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for keys, group in ledger.groupby(['cohort_id', 'budget_label'], dropna=False):
        rho, pvalue, n = _safe_spearman(group['adoption_rate'], group['reference_gain'])
        analysis_rows.append({'cohort_id': keys[0], 'budget_label': keys[1], 'n': n, 'rho_adoption_vs_reference_gain': rho, 'p_raw': pvalue})
        valid = group[['local_reference_advantage', 'shared_term_alternative_advantage', 'reference_gain', 'C4_alpha', 'C4R_alpha']].dropna()
        residual_y = _residualize(valid['reference_gain'].to_numpy(dtype=float), valid[['C4R_alpha']].to_numpy(dtype=float))
        residual_adv = _residualize(valid['local_reference_advantage'].to_numpy(dtype=float), valid[['C4_alpha']].to_numpy(dtype=float))
        rho_resid, p_resid, _ = _safe_spearman(pd.Series(residual_adv), pd.Series(residual_y))
        rho_alt, p_alt, _ = _safe_spearman(valid['shared_term_alternative_advantage'], valid['reference_gain'])
        sensitivity_rows.append({'cohort_id': keys[0], 'budget_label': keys[1], 'n': len(valid), 'rho_mean_reversion_controlled': rho_resid, 'p_mean_reversion_controlled': p_resid, 'rho_shared_term_alternative': rho_alt, 'p_shared_term_alternative': p_alt})
    adoption_out = output_dir / 'relational_reference_adoption_gain.csv'
    write_csv(adoption_out, analysis_rows)
    sensitivity_out = output_dir / 'relational_reference_sensitivity.csv'
    write_csv(sensitivity_out, sensitivity_rows)
    manifest = {'schema_version': 'c4r_relational_reference_value_v2', 'status': 'PASS', 'evidence_class': 'EVALUATOR_ONLY_POST_HOC', 'local_reference_advantage_definition': 'C3_alpha - C4_alpha', 'reference_gain_definition': 'C6_alpha - C4R_alpha', 'adoption_definition': 'L1 distance(C4, Candidate-IQL reference) - L1 distance(C6, Candidate-IQL reference)', 'mean_reversion_sensitivity_definition': 'partial Spearman after residualizing C3-C4 on C4 and C6-C4R on C4R', 'shared_term_sensitivity_definition': 'raw Spearman of C3-C4R with C6-C4R; interpreted only as a shared-term sensitivity', 'warnings': WARNINGS, 'firm_rows': len(ledger), 'outputs': {'firm_ledger': repo_rel(root, firm_out), 'quartile_summary': repo_rel(root, quartile_out), 'complement_substitute_summary': repo_rel(root, category_out), 'adoption_gain_analysis': repo_rel(root, adoption_out), 'shared_term_sensitivity': repo_rel(root, sensitivity_out), 'mean_reversion_sensitivity': repo_rel(root, sensitivity_out)}, 'candidate_library_sources': candidate_lineage, 'input_sources': input_lineage}
    write_json(output_dir / 'relational_reference_manifest.json', manifest)
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
