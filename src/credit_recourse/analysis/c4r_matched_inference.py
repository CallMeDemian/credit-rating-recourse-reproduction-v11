from __future__ import annotations
'Matched C4/C4R/C6 inference for the reference-free re-review experiment.\n\nThe module consumes exactly two immutable Stage7-9 archives generated in one\nC4R batch: L1<=0.75 and unbounded.  It separates three paired increments:\n\n* C4R - C4: reference-free second-pass re-review;\n* C6 - C4R: conditional external-reference content increment;\n* C6 - C4: the original reference-plus-revision package increment.\n\nThe identity ``(C4R-C4) + (C6-C4R) == C6-C4`` is verified per firm and Oracle\nbefore any aggregate output is written.  This is a new planned contrast and is\nkept separate from the frozen canonical N5M analysis.\n'
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from credit_recourse.analysis.n5_7_10c_holm_inference import ORACLES, holm_adjust, p_to_stars, wilcoxon_paired_p
SCHEMA_VERSION = 'c4r_matched_inference_v2'
RUN_ROLE = 'paper_c4r_matched_icb'
CONDITIONS = ('C4', 'C4R', 'C6')
MODE = 'free_form_10d'
EXPECTED_BUDGETS = {'0p75': 0.75, 'unbounded': None}
IDENTITY_TOLERANCE = 1e-12

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
    matches = [p for p in root.iterdir() if p.is_dir() and (p / 'stage7_llm_action_generation').is_dir()]
    if len(matches) != 1:
        raise FileNotFoundError(f'Could not resolve one Stage7-9 archive under {root}: found={len(matches)}')
    return matches[0]

def _budget_from_meta(meta: dict[str, Any]) -> tuple[str, float | None]:
    contract = meta.get('action_budget_contract') or {}
    enabled = bool(contract.get('enabled'))
    if not enabled:
        return ('unbounded', None)
    value = float(contract.get('l1_budget'))
    if not np.isclose(value, 0.75, atol=1e-12, rtol=0):
        raise ValueError(f'C4R matched experiment supports only finite L1=0.75; got {value}')
    budgeted = set(map(str, contract.get('budgeted_conditions') or []))
    if budgeted != set(CONDITIONS):
        raise ValueError(f'Finite C4R arm must budget C4/C4R/C6 equally; got {sorted(budgeted)}')
    if set(map(str, contract.get('budgeted_modes') or [])) != {MODE}:
        raise ValueError('Finite C4R arm budgeted_modes must be free_form_10d only')
    return ('0p75', value)

def _load_arm(path: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    run_dir = _inner_archive_dir(path)
    archive_path = run_dir / 'archive_manifest.json'
    stage7_meta_path = run_dir / 'stage7_llm_action_generation' / 'metadata.json'
    stage8_path = run_dir / 'stage8_llm_multi_oracle_eval' / 'llm_stage8_multi_oracle_scores.parquet'
    stage9_path = run_dir / 'stage9_llm_rl_comparison' / 'llm_stage9_revision_metrics.csv'
    for required in (archive_path, stage7_meta_path, stage9_path):
        if not required.is_file():
            raise FileNotFoundError(f'C4R archive missing required artifact: {required}')
    archive = _read_json(archive_path)
    meta = _read_json(stage7_meta_path)
    if archive.get('run_role') != RUN_ROLE:
        raise ValueError(f"C4R archive run_role mismatch: {archive.get('run_role')!r}")
    if meta.get('status') != 'PASS' or meta.get('backend_is_live') is not True:
        raise ValueError(f"C4R archive must be a successful live run: status={meta.get('status')!r}, live={meta.get('backend_is_live')!r}")
    if meta.get('information_condition') != 'IC-b':
        raise ValueError(f"C4R archive information condition must be IC-b: {meta.get('information_condition')!r}")
    if tuple(meta.get('conditions') or []) != CONDITIONS:
        raise ValueError(f"C4R archive conditions must be exactly {CONDITIONS}: {meta.get('conditions')!r}")
    if tuple(meta.get('modes') or []) != (MODE,):
        raise ValueError(f"C4R archive mode must be {MODE}: {meta.get('modes')!r}")
    label, value = _budget_from_meta(meta)
    scores = _read_parquet_or_csv(stage8_path)
    revision = pd.read_csv(stage9_path)
    arm = {'run_dir': str(run_dir), 'run_label': archive.get('run_label') or run_dir.name, 'run_role': archive.get('run_role'), 'budget_label': label, 'l1_budget': value, 'backend_id': meta.get('backend_id')}
    return (arm, scores, revision)

def _arm_firm_frame(arm: dict[str, Any], scores: pd.DataFrame, revision: pd.DataFrame) -> pd.DataFrame:
    required_scores = {'row_id', 'policy', 'mode', *[f'delta_R_score_{o}' for o in ORACLES]}
    missing = sorted(required_scores - set(scores.columns))
    if missing:
        raise ValueError(f'C4R Stage8 scores missing columns: {missing}')
    selected = scores.loc[scores['policy'].astype(str).isin(CONDITIONS) & scores['mode'].astype(str).eq(MODE)].copy()
    selected['row_id'] = pd.to_numeric(selected['row_id'], errors='raise').astype(int)
    if len(selected) != 575 * len(CONDITIONS) or selected.duplicated(['row_id', 'policy', 'mode']).any():
        raise ValueError(f'C4R Stage8 expected {575 * len(CONDITIONS)} unique rows, found {len(selected)}')
    universes = {condition: set(selected.loc[selected['policy'].astype(str).eq(condition), 'row_id']) for condition in CONDITIONS}
    if any((len(values) != 575 for values in universes.values())) or len({frozenset(v) for v in universes.values()}) != 1:
        raise ValueError('C4R Stage8 C4/C4R/C6 row alignment failed')
    wide_parts = []
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
        raise ValueError(f'C4R Stage9 revision metrics missing columns: {missing_revision}')
    rev = revision.loc[revision['revision_condition'].astype(str).isin(['C4R', 'C6']) & revision['mode'].astype(str).eq(MODE)].copy()
    rev['row_id'] = pd.to_numeric(rev['row_id'], errors='raise').astype(int)
    expected_grid = {(rid, condition) for rid in universes['C4'] for condition in ('C4R', 'C6')}
    observed_grid = set(map(tuple, rev[['row_id', 'revision_condition']].astype({'row_id': int, 'revision_condition': str}).itertuples(index=False, name=None)))
    if observed_grid != expected_grid or rev.duplicated(['row_id', 'revision_condition', 'mode']).any():
        raise ValueError('C4R Stage9 revision grid is not exact 575 x {C4R,C6}')
    c4r = rev.loc[rev['revision_condition'].astype(str).eq('C4R')]
    if not c4r['base_condition'].astype(str).eq('C4').all() or not c4r['reference_source'].astype(str).eq('none').all():
        raise ValueError('C4R Stage9 rows violate reference-free C4 pairing')
    if c4r['metrics_defined'].fillna(False).astype(bool).any() or not c4r['undefined_reason'].astype(str).eq('no_rl_reference').all():
        raise ValueError('C4R RL-adoption geometry must be undefined with no_rl_reference')
    frame.insert(0, 'run_label', arm['run_label'])
    frame.insert(1, 'budget_label', arm['budget_label'])
    frame.insert(2, 'l1_budget', np.nan if arm['l1_budget'] is None else float(arm['l1_budget']))
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
    contrast_columns = {'self_revision': 'self_revision', 'reference_content_conditional': 'reference_content_conditional', 'reference_plus_revision_package': 'package'}
    for (budget_label, l1_budget), group in firm_frame.groupby(['budget_label', 'l1_budget'], dropna=False, sort=False):
        for oracle in ORACLES:
            for contrast, prefix in contrast_columns.items():
                values = pd.to_numeric(group[f'{prefix}_{oracle}'], errors='raise')
                rows.append({'budget_label': budget_label, 'l1_budget': l1_budget, 'oracle_backend': oracle, 'contrast': contrast, 'n_pairs': int(len(values)), 'mean_increment': float(values.mean()), 'median_increment': float(values.median()), 'win_count': int((values > 0).sum()), 'tie_count': int((values == 0).sum()), 'loss_count': int((values < 0).sum()), 'wilcoxon_p_raw': wilcoxon_paired_p(values)})
    out = pd.DataFrame(rows)
    out['wilcoxon_p_holm_two_budgets'] = np.nan
    for _, idx in out.groupby(['oracle_backend', 'contrast'], sort=False).groups.items():
        out.loc[idx, 'wilcoxon_p_holm_two_budgets'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].fillna(1.0).tolist())
    out['sig_holm'] = out['wilcoxon_p_holm_two_budgets'].map(p_to_stars)
    return out.sort_values(['contrast', 'oracle_backend', 'l1_budget'], na_position='last').reset_index(drop=True)

def _interaction_table(firm_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute firm-paired finite-minus-unbounded DID inference.

    The unit of inference is the same firm observed in both immutable budget
    arms.  For each decomposition component and Oracle, the row-level DID is
    ``increment_0p75 - increment_unbounded``.  Holm correction is applied over
    the three Oracles *within each component*, matching the pre-specified
    inferential family in the thesis rather than pooling all nine tests.
    """
    contrast_columns = {'self_revision': 'self_revision', 'reference_content_conditional': 'reference_content_conditional', 'reference_plus_revision_package': 'package'}
    required = {'row_id', 'budget_label'}
    for prefix in contrast_columns.values():
        required.update((f'{prefix}_{oracle}' for oracle in ORACLES))
    missing = sorted(required - set(firm_frame.columns))
    if missing:
        raise ValueError(f'C4R firm frame missing interaction columns: {missing}')
    if firm_frame.duplicated(['row_id', 'budget_label']).any():
        raise ValueError('C4R interaction frame has duplicate row_id/budget_label keys')
    finite = firm_frame.loc[firm_frame['budget_label'].astype(str).eq('0p75')].copy()
    unbounded = firm_frame.loc[firm_frame['budget_label'].astype(str).eq('unbounded')].copy()
    if len(finite) != 575 or len(unbounded) != 575:
        raise ValueError(f'C4R interaction requires exactly 575 finite and 575 unbounded firm rows: finite={len(finite)}, unbounded={len(unbounded)}')
    if set(finite['row_id'].astype(int)) != set(unbounded['row_id'].astype(int)):
        raise ValueError('C4R interaction finite/unbounded row_id universes differ')
    paired = finite.merge(unbounded, on='row_id', how='inner', validate='one_to_one', suffixes=('__finite', '__unbounded')).sort_values('row_id')
    if len(paired) != 575:
        raise ValueError(f'C4R interaction paired row count must be 575; got {len(paired)}')
    rows: list[dict[str, Any]] = []
    for contrast, prefix in contrast_columns.items():
        for oracle in ORACLES:
            finite_values = pd.to_numeric(paired[f'{prefix}_{oracle}__finite'], errors='raise')
            unbounded_values = pd.to_numeric(paired[f'{prefix}_{oracle}__unbounded'], errors='raise')
            did = finite_values - unbounded_values
            rows.append({'oracle_backend': oracle, 'contrast': contrast, 'finite_budget_label': '0p75', 'unbounded_budget_label': 'unbounded', 'n_pairs': int(len(did)), 'finite_mean_increment': float(finite_values.mean()), 'unbounded_mean_increment': float(unbounded_values.mean()), 'finite_minus_unbounded_interaction': float(did.mean()), 'median_interaction': float(did.median()), 'win_count': int((did > 0).sum()), 'tie_count': int((did == 0).sum()), 'loss_count': int((did < 0).sum()), 'wilcoxon_p_raw': wilcoxon_paired_p(did), 'holm_family': 'contrast_across_three_oracles'})
    out = pd.DataFrame(rows)
    out['wilcoxon_p_holm_three_oracles'] = np.nan
    for _, idx in out.groupby('contrast', sort=False).groups.items():
        out.loc[idx, 'wilcoxon_p_holm_three_oracles'] = holm_adjust(out.loc[idx, 'wilcoxon_p_raw'].fillna(1.0).tolist())
    out['sig_holm'] = out['wilcoxon_p_holm_three_oracles'].map(p_to_stars)
    return out.sort_values(['contrast', 'oracle_backend']).reset_index(drop=True)

def run_analysis(*, arm_dirs: list[Path], out_dir: Path) -> dict[str, Any]:
    if len(arm_dirs) != 2:
        raise ValueError('C4R matched inference requires exactly two arm archives: 0p75 and unbounded')
    loaded = [_load_arm(path) for path in arm_dirs]
    labels = [item[0]['budget_label'] for item in loaded]
    if set(labels) != set(EXPECTED_BUDGETS):
        raise ValueError(f'C4R matched inference requires budgets {sorted(EXPECTED_BUDGETS)}, got {sorted(labels)}')
    backend_ids = {str(item[0]['backend_id']) for item in loaded}
    if len(backend_ids) != 1:
        raise ValueError(f'C4R arm backend mismatch: {sorted(backend_ids)}')
    frames = [_arm_firm_frame(*item) for item in loaded]
    universe = [set(frame['row_id'].astype(int)) for frame in frames]
    if universe[0] != universe[1] or len(universe[0]) != 575:
        raise ValueError('C4R finite/unbounded arms must share the same 575-firm row universe')
    firm_frame = pd.concat(frames, ignore_index=True)
    contrasts = _contrast_table(firm_frame)
    interactions = _interaction_table(firm_frame)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    firm_path = out / 'c4r_matched_firm_frame.parquet'
    try:
        firm_frame.to_parquet(firm_path, index=False)
        firm_storage = 'parquet'
    except ImportError:
        firm_path = out / 'c4r_matched_firm_frame.csv'
        firm_frame.to_csv(firm_path, index=False, encoding='utf-8-sig')
        firm_storage = 'csv_fallback_missing_parquet_engine'
    contrasts_path = out / 'c4r_matched_contrasts.csv'
    interactions_path = out / 'c4r_matched_interactions.csv'
    contrasts.to_csv(contrasts_path, index=False, encoding='utf-8-sig')
    interactions.to_csv(interactions_path, index=False, encoding='utf-8-sig')
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'created_utc': _now(), 'run_role': RUN_ROLE, 'information_condition': 'IC-b', 'conditions': list(CONDITIONS), 'mode': MODE, 'backend_id': next(iter(backend_ids)), 'arm_count': 2, 'firm_count_per_arm': 575, 'firm_frame_row_count': int(len(firm_frame)), 'contrast_row_count': int(len(contrasts)), 'interaction_row_count': int(len(interactions)), 'interaction_inference_contract': {'unit': 'firm_paired_finite_minus_unbounded_DID', 'finite_budget_label': '0p75', 'unbounded_budget_label': 'unbounded', 'test': 'two_sided_wilcoxon_signed_rank_zero_discard', 'holm_family': 'contrast_across_three_oracles', 'holm_family_size': 3, 'component_count': 3, 'oracle_count': 3, 'expected_row_count': 9}, 'additive_identity_tolerance': IDENTITY_TOLERANCE, 'evidence_tier': 'PLANNED_C4R_MATCHED_EXPERIMENT', 'arms': [item[0] for item in loaded], 'outputs': {'firm_frame': {'path': firm_path.name, 'storage': firm_storage, 'row_count': int(len(firm_frame))}, 'contrasts': {'path': contrasts_path.name, 'row_count': int(len(contrasts))}, 'interactions': {'path': interactions_path.name, 'row_count': int(len(interactions))}}, 'interpretation_boundary': 'C4R-C4 identifies reference-free second-pass re-review within this live batch; C6-C4R identifies the conditional content increment of the shown Candidate-IQL reference. Claims remain specific to the backend, snapshot, seed, IC-b cohort, and raw-coordinate L1 contract.'}
    manifest_path = out / 'c4r_matched_inference_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm-dir', action='append', required=True, help='Repeat exactly twice: finite 0p75 and unbounded archive directories')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    result = run_analysis(arm_dirs=[Path(value) for value in args.arm_dir], out_dir=Path(args.out))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
