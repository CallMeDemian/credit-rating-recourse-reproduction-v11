"""Generation-time N5 budget-frontier inference for IC-b C4/C6 runs.

Two designs are supported without changing the legacy archive contract.

``legacy_c6_only``
    Finite arms apply the L1 budget to C6 only. C4 is an unbudgeted within-arm
    live-API control. The DID is a package-level diagnostic and must not be
    interpreted as a pure reference-by-budget interaction.

``matched_c4_c6``
    Finite arms apply the identical L1 budget to both C4 and C6. This identifies
    the C4 budget main effect, the C6-vs-C4 reference/revision effect at a fixed
    budget, and their difference-in-differences interaction relative to the
    unbounded arm.
"""
from __future__ import annotations
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import numpy as np
import pandas as pd
from credit_recourse.analysis.n5_7_10c_holm_inference import ORACLES, _policy_score_frame, _read_table, detect_score_column, find_comparison_file, holm_adjust, p_to_stars, standardize_comparison, wilcoxon_paired_p
from credit_recourse.contracts.paper_reproduction import inspect_archived_run
SCHEMA_VERSION = 'n5_generation_budget_frontier_holm_v2'
EXPECTED_CONDITIONS = {'C4', 'C6'}
EXPECTED_MODES = {'free_form_10d'}
DESIGN_SPECS: dict[str, dict[str, Any]] = {'legacy_c6_only': {'finite_budgeted_conditions': {'C6'}, 'within_arm_semantics': 'budgeted_C6_minus_unbudgeted_C4_package_gap', 'did_family': 'legacy_package_DID_vs_unbounded'}, 'matched_c4_c6': {'finite_budgeted_conditions': {'C4', 'C6'}, 'within_arm_semantics': 'reference_revision_effect_at_fixed_budget', 'did_family': 'matched_budget_reference_interaction_DID'}}

@dataclass(frozen=True)
class FrontierArm:
    run_dir: Path
    run_label: str
    budget: float | None
    comparison_file: Path
    action_table: Path
    stage7_metadata: dict[str, Any]

    @property
    def budget_label(self) -> str:
        return 'unbounded' if self.budget is None else _budget_token(self.budget)

def _budget_token(value: float) -> str:
    return f'{float(value):.2f}'.replace('.', 'p')

def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return obj

def _read_action_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:
        return pd.read_csv(path)

def _budget_equal(left: float | None, right: float | None, tol: float=1e-09) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tol

def _parse_expected_budgets(values: Iterable[str]) -> tuple[float | None, ...]:
    out: list[float | None] = []
    for raw in values:
        text = str(raw).strip().lower()
        value = None if text in {'none', 'null', 'inf', 'infinity', 'unbounded', '∞'} else float(text)
        if any((_budget_equal(value, existing) for existing in out)):
            raise ValueError(f'Duplicate expected budget: {raw}')
        out.append(value)
    if not any((value is None for value in out)):
        raise ValueError('Expected budgets must include one unbounded arm.')
    return tuple(out)

def _validate_stage7_budget_contract(*, metadata: dict[str, Any], run_label: str, budget: float | None, design: str) -> None:
    spec = DESIGN_SPECS[design]
    contract = metadata.get('action_budget_contract') or {'enabled': False}
    enabled = bool(contract.get('enabled'))
    if budget is None:
        if enabled:
            raise ValueError(f'Unbounded arm unexpectedly has an enabled budget contract: {run_label}')
        return
    if not enabled:
        raise ValueError(f'Finite arm is missing an enabled budget contract: {run_label}')
    observed_target = float(contract.get('l1_budget'))
    if not _budget_equal(observed_target, budget):
        raise ValueError(f'Budget target mismatch for {run_label}: metadata={observed_target}, archive={budget}')
    observed_conditions = set(map(str, contract.get('budgeted_conditions') or []))
    expected_conditions = set(spec['finite_budgeted_conditions'])
    if observed_conditions != expected_conditions:
        raise ValueError(f'Budgeted-condition mismatch for design={design}, run={run_label}: observed={sorted(observed_conditions)}, expected={sorted(expected_conditions)}')
    modes = set(map(str, contract.get('budgeted_modes') or []))
    if modes != EXPECTED_MODES:
        raise ValueError(f'Budgeted modes mismatch for {run_label}: {sorted(modes)}')

def validate_frontier_runs(run_dirs: Iterable[Path], *, run_role: str, information_condition: str, expected_budgets: tuple[float | None, ...], design: str) -> list[FrontierArm]:
    if design not in DESIGN_SPECS:
        raise ValueError(f'Unsupported frontier design: {design!r}')
    records = [inspect_archived_run(Path(path)) for path in run_dirs]
    if len(records) != len(expected_budgets):
        raise ValueError(f'Expected {len(expected_budgets)} frontier runs, found {len(records)}: {[record.run_label for record in records]}')
    arms: list[FrontierArm] = []
    for record in records:
        if record.run_role != run_role:
            raise ValueError(f'Frontier role mismatch for {record.run_label}: {record.run_role!r} != {run_role!r}')
        if record.information_condition != information_condition:
            raise ValueError(f'Frontier information condition mismatch for {record.run_label}: {record.information_condition!r} != {information_condition!r}')
        if set(record.conditions) != EXPECTED_CONDITIONS:
            raise ValueError(f'Frontier conditions mismatch for {record.run_label}: {record.conditions}')
        if set(record.modes) != EXPECTED_MODES:
            raise ValueError(f'Frontier modes mismatch for {record.run_label}: {record.modes}')
        if not (record.has_stage7 and record.has_stage8 and record.has_stage9):
            raise ValueError(f'Frontier run is incomplete: {record.run_label}')
        stage7 = record.run_dir / 'stage7_llm_action_generation'
        metadata_path = stage7 / 'metadata.json'
        action_table = stage7 / 'llm_stage7_action_table.parquet'
        if not metadata_path.is_file() or not action_table.is_file():
            raise FileNotFoundError(f'Frontier Stage7 contract artifacts missing for {record.run_label}: metadata={metadata_path.is_file()}, action_table={action_table.is_file()}')
        metadata = _read_json(metadata_path)
        _validate_stage7_budget_contract(metadata=metadata, run_label=record.run_label, budget=record.freeform_l1_budget, design=design)
        arms.append(FrontierArm(run_dir=record.run_dir, run_label=record.run_label, budget=record.freeform_l1_budget, comparison_file=find_comparison_file(record.run_dir), action_table=action_table, stage7_metadata=metadata))
    for expected in expected_budgets:
        matches = [arm for arm in arms if _budget_equal(arm.budget, expected)]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one frontier arm for budget={expected!r}; found {[arm.run_label for arm in matches]}')
    extra = [arm for arm in arms if not any((_budget_equal(arm.budget, expected) for expected in expected_budgets))]
    if extra:
        raise ValueError(f'Unexpected frontier budget arms: {[(arm.run_label, arm.budget) for arm in extra]}')
    return sorted(arms, key=lambda arm: math.inf if arm.budget is None else float(arm.budget))

def _paired_policy_scores(frame: pd.DataFrame, oracle: str) -> pd.DataFrame:
    score_col = detect_score_column(frame, oracle)
    c4 = _policy_score_frame(frame, 'C4', score_col)
    c6 = _policy_score_frame(frame, 'C6', score_col)
    paired = c4.merge(c6, on='row_id', how='inner', validate='one_to_one')
    if len(paired) != len(c4) or len(paired) != len(c6):
        raise ValueError(f'C4/C6 row alignment failure for oracle={oracle}: C4={len(c4)}, C6={len(c6)}, paired={len(paired)}')
    return paired.sort_values('row_id').reset_index(drop=True)

def _audit_control_alignment(score_frames: dict[str, pd.DataFrame], *, design: str, exact_tolerance: float) -> pd.DataFrame:
    reference_label = 'unbounded'
    if reference_label not in score_frames:
        raise ValueError('Unbounded C4 arm is missing.')
    reference = score_frames[reference_label][['row_id', 'score_C4']].rename(columns={'score_C4': 'reference_C4'})
    rows: list[dict[str, Any]] = []
    for label, frame in score_frames.items():
        merged = reference.merge(frame[['row_id', 'score_C4']].rename(columns={'score_C4': 'arm_C4'}), on='row_id', how='outer', validate='one_to_one', indicator=True)
        if not merged['_merge'].eq('both').all():
            raise ValueError(f'C4 row-id set differs across frontier arms: reference={reference_label}, arm={label}')
        signed = merged['arm_C4'] - merged['reference_C4']
        absolute = signed.abs()
        max_abs = float(absolute.max()) if len(absolute) else 0.0
        exact = bool(max_abs <= exact_tolerance)
        intentional = design == 'matched_c4_c6' and label != 'unbounded'
        rows.append({'design': design, 'reference_arm': reference_label, 'arm': label, 'n_rows': int(len(merged)), 'mean_C4_difference_vs_unbounded': float(signed.mean()), 'median_C4_difference_vs_unbounded': float(signed.median()), 'mean_abs_C4_difference_vs_unbounded': float(absolute.mean()), 'max_abs_C4_difference_vs_unbounded': max_abs, 'wilcoxon_p_raw': wilcoxon_paired_p(signed), 'exact_tolerance': float(exact_tolerance), 'bitwise_score_equivalent_within_tolerance': exact, 'difference_interpretation': 'INTENTIONAL_BUDGET_EFFECT_PLUS_API_VARIATION' if intentional else 'API_RERUN_VARIATION_CONTROL', 'status': 'PASS_ROW_ALIGNMENT_INTENTIONAL_DIFFERENCE' if intentional else 'PASS_EXACT' if exact else 'PASS_DRIFT_REPORTED'})
    return pd.DataFrame(rows)

def _budget_contract_audit(arms: list[FrontierArm], *, design: str) -> pd.DataFrame:
    expected_finite = set(DESIGN_SPECS[design]['finite_budgeted_conditions'])
    rows: list[dict[str, Any]] = []
    for arm in arms:
        frame = _read_action_table(arm.action_table)
        required = {'row_id', 'policy', 'mode', 'budgeted_condition_flag', 'budget_l1_target', 'budget_l1_raw', 'budget_l1_clipped', 'budget_compliant_raw', 'budget_compliant_clipped'}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f'Action table missing budget audit columns for {arm.run_label}: {missing}')
        frame = frame[frame['policy'].astype(str).isin(['C4', 'C6']) & frame['mode'].astype(str).eq('free_form_10d')].copy()
        if frame.empty:
            raise ValueError(f'No C4/C6 free-form rows in {arm.action_table}')
        for policy in ('C4', 'C6'):
            part = frame[frame['policy'].astype(str).eq(policy)].copy()
            if part.empty:
                raise ValueError(f'Missing {policy} rows in {arm.action_table}')
            expected_flag = arm.budget is not None and policy in expected_finite
            actual = part['budgeted_condition_flag'].fillna(False).astype(bool)
            if not actual.eq(expected_flag).all():
                raise ValueError(f'budgeted_condition_flag mismatch: run={arm.run_label}, policy={policy}, expected={expected_flag}, observed_true={int(actual.sum())}/{len(actual)}')
            target = pd.to_numeric(part['budget_l1_target'], errors='coerce')
            if expected_flag:
                if target.isna().any() or not np.allclose(target.to_numpy(float), float(arm.budget), atol=1e-09, rtol=0):
                    raise ValueError(f'Budget target mismatch in action table: {arm.run_label}, {policy}')
            elif target.notna().any():
                raise ValueError(f'Unbudgeted rows unexpectedly contain budget target: {arm.run_label}, {policy}')
            raw = pd.to_numeric(part['budget_l1_raw'], errors='coerce')
            clipped = pd.to_numeric(part['budget_l1_clipped'], errors='coerce')
            rows.append({'design': design, 'run_label': arm.run_label, 'budget_label': arm.budget_label, 'l1_budget': arm.budget, 'policy': policy, 'n_rows': int(len(part)), 'budget_expected': bool(expected_flag), 'mean_l1_raw': float(raw.mean()) if raw.notna().any() else np.nan, 'mean_l1_clipped': float(clipped.mean()) if clipped.notna().any() else np.nan, 'raw_compliance_rate': float(part['budget_compliant_raw'].fillna(False).astype(bool).mean()) if expected_flag else np.nan, 'clipped_compliance_rate': float(part['budget_compliant_clipped'].fillna(False).astype(bool).mean()) if expected_flag else np.nan})
    return pd.DataFrame(rows)

def compute_frontier(arms: list[FrontierArm], *, design: str, c4_exact_tolerance: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    means: list[dict[str, Any]] = []
    infer: list[dict[str, Any]] = []
    audits: list[pd.DataFrame] = []
    for oracle in ORACLES:
        per_arm: dict[str, pd.DataFrame] = {}
        arm_lookup: dict[str, FrontierArm] = {}
        for arm in arms:
            frame = standardize_comparison(_read_table(arm.comparison_file))
            paired = _paired_policy_scores(frame, oracle)
            per_arm[arm.budget_label] = paired
            arm_lookup[arm.budget_label] = arm
            within = paired['score_C6'] - paired['score_C4']
            means.append({'design': design, 'oracle_backend': oracle, 'budget_label': arm.budget_label, 'l1_budget': arm.budget, 'run_label': arm.run_label, 'n_pairs': int(len(paired)), 'mean_C4': float(paired['score_C4'].mean()), 'mean_C6': float(paired['score_C6'].mean()), 'mean_C6_minus_C4': float(within.mean())})
            infer.append({'design': design, 'oracle_backend': oracle, 'contrast_family': 'within_arm_C6_minus_C4', 'contrast_semantics': DESIGN_SPECS[design]['within_arm_semantics'], 'contrast': f'C6_minus_C4__{arm.budget_label}', 'budget_label': arm.budget_label, 'l1_budget': arm.budget, 'reference_budget_label': 'same_arm_C4', 'n_pairs': int(len(within)), 'mean_gap': float(within.mean()), 'median_gap': float(within.median()), 'wilcoxon_p_raw': wilcoxon_paired_p(within)})
        audit = _audit_control_alignment(per_arm, design=design, exact_tolerance=c4_exact_tolerance)
        audit.insert(1, 'oracle_backend', oracle)
        audits.append(audit)
        unbounded = per_arm.get('unbounded')
        if unbounded is None:
            raise ValueError('Unbounded frontier arm is missing after validation.')
        for label, frame in per_arm.items():
            if label == 'unbounded':
                continue
            merged = frame.merge(unbounded, on='row_id', how='inner', suffixes=('_finite', '_unbounded'), validate='one_to_one')
            if len(merged) != len(frame) or len(merged) != len(unbounded):
                raise ValueError(f'C4/C6 row alignment failure for finite={label} vs unbounded: finite={len(frame)}, unbounded={len(unbounded)}, paired={len(merged)}')
            for policy in ('C6', 'C4'):
                if policy == 'C4' and design != 'matched_c4_c6':
                    continue
                gap = merged[f'score_{policy}_finite'] - merged[f'score_{policy}_unbounded']
                infer.append({'design': design, 'oracle_backend': oracle, 'contrast_family': f'budget_main_effect_{policy}', 'contrast_semantics': f'{policy} finite-budget minus {policy} unbounded', 'contrast': f'{policy}_{label}_minus_{policy}_unbounded', 'budget_label': label, 'l1_budget': arm_lookup[label].budget, 'reference_budget_label': 'unbounded', 'n_pairs': int(len(gap)), 'mean_gap': float(gap.mean()), 'median_gap': float(gap.median()), 'wilcoxon_p_raw': wilcoxon_paired_p(gap)})
            did = merged['score_C6_finite'] - merged['score_C4_finite'] - (merged['score_C6_unbounded'] - merged['score_C4_unbounded'])
            infer.append({'design': design, 'oracle_backend': oracle, 'contrast_family': DESIGN_SPECS[design]['did_family'], 'contrast_semantics': 'pure reference/revision-by-budget interaction' if design == 'matched_c4_c6' else 'package-level DID; C4 is not budget matched', 'contrast': f'C6_minus_C4_{label}_minus_unbounded', 'budget_label': label, 'l1_budget': arm_lookup[label].budget, 'reference_budget_label': 'unbounded', 'n_pairs': int(len(did)), 'mean_gap': float(did.mean()), 'median_gap': float(pd.Series(did).median()), 'wilcoxon_p_raw': wilcoxon_paired_p(pd.Series(did))})
    inference = pd.DataFrame(infer)
    inference['holm_family'] = 'N5F2_' + inference['design'].astype(str) + '_' + inference['oracle_backend'].astype(str) + '_' + inference['contrast_family'].astype(str)
    inference['wilcoxon_p_holm'] = np.nan
    for _, idx in inference.groupby('holm_family', sort=False).groups.items():
        inference.loc[idx, 'wilcoxon_p_holm'] = holm_adjust(inference.loc[idx, 'wilcoxon_p_raw'].tolist())
    inference['sig_holm'] = inference['wilcoxon_p_holm'].apply(p_to_stars)
    budget_audit = _budget_contract_audit(arms, design=design)
    return (pd.DataFrame(means).sort_values(['oracle_backend', 'l1_budget'], na_position='last').reset_index(drop=True), inference.sort_values(['oracle_backend', 'contrast_family', 'l1_budget'], na_position='last').reset_index(drop=True), pd.concat(audits, ignore_index=True), budget_audit.sort_values(['l1_budget', 'policy'], na_position='last').reset_index(drop=True))

def build_table_patch(means: pd.DataFrame, inference: pd.DataFrame, *, design: str) -> pd.DataFrame:
    alpha_means = means[means['oracle_backend'].eq('alpha')].copy()
    within = inference[inference['oracle_backend'].eq('alpha') & inference['contrast_family'].eq('within_arm_C6_minus_C4')][['budget_label', 'wilcoxon_p_holm', 'sig_holm']].rename(columns={'wilcoxon_p_holm': 'C6_minus_C4_p_holm', 'sig_holm': 'C6_minus_C4_sig'})
    c6 = inference[inference['oracle_backend'].eq('alpha') & inference['contrast_family'].eq('budget_main_effect_C6')][['budget_label', 'mean_gap', 'wilcoxon_p_holm', 'sig_holm']].rename(columns={'mean_gap': 'C6_minus_unbounded_gap', 'wilcoxon_p_holm': 'C6_minus_unbounded_p_holm', 'sig_holm': 'C6_minus_unbounded_sig'})
    did_family = DESIGN_SPECS[design]['did_family']
    did = inference[inference['oracle_backend'].eq('alpha') & inference['contrast_family'].eq(did_family)][['budget_label', 'mean_gap', 'wilcoxon_p_holm', 'sig_holm']].rename(columns={'mean_gap': 'interaction_or_package_DID_gap', 'wilcoxon_p_holm': 'interaction_or_package_DID_p_holm', 'sig_holm': 'interaction_or_package_DID_sig'})
    out = alpha_means.merge(within, on='budget_label', how='left', validate='one_to_one')
    out = out.merge(c6, on='budget_label', how='left', validate='one_to_one')
    if design == 'matched_c4_c6':
        c4 = inference[inference['oracle_backend'].eq('alpha') & inference['contrast_family'].eq('budget_main_effect_C4')][['budget_label', 'mean_gap', 'wilcoxon_p_holm', 'sig_holm']].rename(columns={'mean_gap': 'C4_minus_unbounded_gap', 'wilcoxon_p_holm': 'C4_minus_unbounded_p_holm', 'sig_holm': 'C4_minus_unbounded_sig'})
        out = out.merge(c4, on='budget_label', how='left', validate='one_to_one')
    out = out.merge(did, on='budget_label', how='left', validate='one_to_one')
    return out.sort_values('l1_budget', na_position='last').reset_index(drop=True)

def _write_markdown(table: pd.DataFrame, path: Path, *, design: str) -> None:
    columns = [c for c in table.columns if c not in {'run_label', 'n_pairs', 'design', 'oracle_backend'}]
    printable = table[['budget_label'] + [c for c in columns if c != 'budget_label']].copy()
    lines = ['| ' + ' | '.join(printable.columns) + ' |', '|' + '|'.join(['---'] * len(printable.columns)) + '|']
    for _, row in printable.iterrows():
        values: list[str] = []
        for value in row:
            if pd.isna(value):
                values.append('')
            elif isinstance(value, (float, np.floating)):
                values.append(f'{float(value):.6g}')
            else:
                values.append(str(value))
        lines.append('| ' + ' | '.join(values) + ' |')
    lines.append('')
    if design == 'matched_c4_c6':
        lines.extend(['주: 유한 예산팔에서는 C4와 C6가 동일한 생성시 L1 계약을 받는다.', 'C6-C4는 동일 예산에서의 참조·수정 효과이며, DID는 무제약 대비 참조·수정×예산 상호작용이다.'])
    else:
        lines.extend(['주: C4는 유한 예산팔에서도 예산 문구가 없는 live-API 재실행 대조군이다.', 'DID는 package-level 진단이며 순수한 참조·수정×예산 상호작용으로 해석하지 않는다.'])
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

def run(args: argparse.Namespace) -> dict[str, Any]:
    design = str(getattr(args, 'design', 'legacy_c6_only'))
    run_dirs = [Path(value).resolve() for value in args.run_dirs]
    expected = _parse_expected_budgets(args.expected_budgets)
    arms = validate_frontier_runs(run_dirs, run_role=args.run_role, information_condition=args.information_condition, expected_budgets=expected, design=design)
    means, inference, control_audit, budget_audit = compute_frontier(arms, design=design, c4_exact_tolerance=float(args.c4_exact_tolerance))
    patch = build_table_patch(means, inference, design=design)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {'policy_means': out_dir / 'n5_budget_frontier_policy_means.csv', 'pairwise_holm': out_dir / 'n5_budget_frontier_pairwise_holm.csv', 'control_alignment_audit': out_dir / 'n5_budget_frontier_control_alignment_audit.csv', 'budget_contract_audit': out_dir / 'n5_budget_frontier_budget_contract_audit.csv', 'table_patch_csv': out_dir / 'n5_budget_frontier_table_patch.csv', 'table_patch_md': out_dir / 'n5_budget_frontier_table_patch.md', 'input_files': out_dir / 'n5_budget_frontier_input_files.csv', 'manifest': out_dir / 'n5_budget_frontier_holm_manifest.json'}
    means.to_csv(paths['policy_means'], index=False, encoding='utf-8-sig')
    inference.to_csv(paths['pairwise_holm'], index=False, encoding='utf-8-sig')
    control_audit.to_csv(paths['control_alignment_audit'], index=False, encoding='utf-8-sig')
    budget_audit.to_csv(paths['budget_contract_audit'], index=False, encoding='utf-8-sig')
    patch.to_csv(paths['table_patch_csv'], index=False, encoding='utf-8-sig')
    _write_markdown(patch, paths['table_patch_md'], design=design)
    '0'
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'design': design, 'run_role': args.run_role, 'information_condition': args.information_condition, 'expected_budgets': list(expected), 'finite_budgeted_conditions': sorted(DESIGN_SPECS[design]['finite_budgeted_conditions']), 'within_arm_semantics': DESIGN_SPECS[design]['within_arm_semantics'], 'did_semantics': 'PURE_REFERENCE_REVISION_BY_BUDGET_INTERACTION' if design == 'matched_c4_c6' else 'PACKAGE_LEVEL_DIAGNOSTIC_NOT_PURE_INTERACTION', 'c4_exact_tolerance': float(args.c4_exact_tolerance), 'outputs': {key: str(value) for key, value in paths.items() if key != 'manifest'}}
    paths['manifest'].write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dirs', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--run-role', default='paper_n5_budget_frontier_icb')
    parser.add_argument('--information-condition', default='IC-b')
    parser.add_argument('--expected-budgets', nargs='+', default=['0.75', '1.27', '2.00', 'unbounded'])
    parser.add_argument('--design', choices=sorted(DESIGN_SPECS), default='legacy_c6_only')
    parser.add_argument('--c4-exact-tolerance', type=float, default=1e-09)
    return parser

def main(argv: list[str] | None=None) -> int:
    run(build_arg_parser().parse_args(argv))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
