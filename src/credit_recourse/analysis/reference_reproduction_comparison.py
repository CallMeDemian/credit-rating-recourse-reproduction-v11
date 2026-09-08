from __future__ import annotations
'Compare Stage6 output with the immutable thesis reference under an explicit mode.\n\nTwo contracts are intentionally separated:\n\n``frozen_replay``\n    Use when Stage3--5 frozen thesis artifacts/checkpoints were restored.  Both\n    the C0 no-op anchors and the deployed Candidate-IQL point values are hard\n    regression gates.\n\n``fresh_retrain``\n    Use when Stage3--5 were trained again from the packaged Stage2 inputs.  The\n    C0 no-op anchors remain hard gates because they validate the cohort and\n    Oracle/evaluator substrate.  Candidate-IQL differences from the historical\n    thesis checkpoint are recorded as advisory lineage drift and do not by\n    themselves fail the fresh-training run.\n\nThe historical reference remains immutable and may never be used as a tuning\nobjective.  This module only evaluates and records; it never changes a model,\ncheckpoint, feature set, action library, or result.\n'
import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import pandas as pd
REFERENCE_REL = Path('src/credit_recourse/configs/reproduction_reference_values.json')
CANONICAL_COMPARISON_REL = Path('data/reproduction/diagnostics/oracle_rl_reference/reference_reproduction_comparison.json')
VALID_MODES = ('auto', 'frozen_replay', 'fresh_retrain')
_MODE_ALIASES = {'auto': 'auto', 'frozen': 'frozen_replay', 'frozen_replay': 'frozen_replay', 'frozenreplay': 'frozen_replay', 'fresh': 'fresh_retrain', 'fresh_retrain': 'fresh_retrain', 'freshretrain': 'fresh_retrain'}

def _normalise_mode(value: str) -> str:
    key = str(value).strip().lower().replace('-', '_')
    try:
        return _MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f'Unsupported comparison mode {value!r}; expected one of {VALID_MODES}') from exc

def _read_mode_from_manifest(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding='utf-8-sig'))
    except Exception:
        return None
    value = payload.get('comparison_mode')
    if value is None:
        return None
    try:
        mode = _normalise_mode(str(value))
    except ValueError:
        return None
    return None if mode == 'auto' else mode

def resolve_mode(project_root: Path, requested: str) -> tuple[str, str]:
    """Resolve ``auto`` without guessing fresh training from numeric outcomes.

    Resolution order is explicit CLI, environment override, prior canonical
    comparison manifest, then the conservative frozen-replay fallback.  The
    fresh RL runner passes ``fresh_retrain`` explicitly and therefore writes a
    canonical manifest that later Analysis/PaperAssets invocations can reuse.
    """
    root = Path(project_root).resolve()
    mode = _normalise_mode(requested)
    if mode != 'auto':
        return (mode, 'explicit_cli')
    env_value = os.environ.get('CREDIT_RECOURSE_RL_VALIDATION_MODE', '').strip()
    if env_value:
        env_mode = _normalise_mode(env_value)
        if env_mode == 'auto':
            raise ValueError('CREDIT_RECOURSE_RL_VALIDATION_MODE may not resolve to auto')
        return (env_mode, 'environment:CREDIT_RECOURSE_RL_VALIDATION_MODE')
    manifest_path = root / CANONICAL_COMPARISON_REL
    prior_mode = _read_mode_from_manifest(manifest_path)
    if prior_mode is not None:
        return (prior_mode, f'canonical_manifest:{manifest_path}')
    return ('frozen_replay', 'conservative_default_no_canonical_manifest')

def inspect_rl_lineage(project_root: Path, reference: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Inspect the registered Stage3--5 checkpoint paths.

    Fresh retraining and frozen replay both require the three checkpoint files
    declared in the thesis lineage registry to exist.
    """
    root = Path(project_root).resolve()
    resolved_mode = _normalise_mode(mode)
    contract = reference.get('frozen_rl_lineage')
    if not isinstance(contract, Mapping):
        if resolved_mode == 'frozen_replay':
            return {'status': 'FAIL', 'source': None, 'artifacts': {}, 'hard_failures': ['missing_frozen_rl_lineage_registry']}
        return {'status': 'PASS', 'source': None, 'artifacts': {}, 'hard_failures': []}
    checkpoints = contract.get('checkpoints')
    if not isinstance(checkpoints, Mapping) or not checkpoints:
        raise ValueError('frozen_rl_lineage.checkpoints must be a non-empty object')
    artifacts: dict[str, Any] = {}
    hard_failures: list[str] = []
    for label, spec_raw in checkpoints.items():
        if not isinstance(spec_raw, Mapping):
            raise ValueError(f'Invalid frozen checkpoint spec for {label}')
        rel = Path(str(spec_raw['path']))
        path = root / rel
        exists = path.is_file()
        if not exists:
            hard_failures.append(f'missing_checkpoint:{label}')
        artifacts[str(label)] = {'path': str(path), 'exists': exists}
    return {'status': 'FAIL' if hard_failures else 'PASS', 'source': contract.get('source'), 'artifacts': artifacts, 'hard_failures': hard_failures}

def _resolve_stage6_summary(root: Path) -> Path:
    candidates = [root / 'data/final_freeze/stage6_multi_oracle_eval/final_policy_summary.csv', root / 'data/final_freeze/stage6_candidate_selector_eval/final_policy_summary.csv']
    hits = [p for p in candidates if p.is_file()]
    if not hits:
        raise FileNotFoundError('No canonical Stage6 final_policy_summary.csv was found')
    return hits[0]

def _policy_row(df: pd.DataFrame, policy_name: str) -> pd.Series:
    if 'policy' not in df.columns:
        raise ValueError('Stage6 final_policy_summary.csv lacks policy')
    work = df.copy()
    work['policy'] = work['policy'].astype(str)
    hit = work.loc[work['policy'].eq(policy_name)]
    if len(hit) != 1:
        raise ValueError(f"Expected exactly one Stage6 row for {policy_name!r}; found {hit['policy'].tolist()}")
    return hit.iloc[0]

def _candidate_row(df: pd.DataFrame) -> pd.Series:
    if 'policy' not in df.columns:
        raise ValueError('Stage6 final_policy_summary.csv lacks policy')
    work = df.copy()
    work['policy'] = work['policy'].astype(str)
    if 'headline_policy' in work.columns:
        flags = work['headline_policy'].astype(str).str.lower().isin({'true', '1'})
        hit = work.loc[flags]
        if len(hit) == 1:
            return hit.iloc[0]
        if len(hit) > 1:
            raise ValueError(f"Multiple Stage6 headline policies: {hit['policy'].tolist()}")
    for name in ('C3_candidate_iql_actor', 'C3_candidate_iql', 'C3_candidate_iql_q_argmax'):
        hit = work.loc[work['policy'].eq(name)]
        if len(hit) == 1:
            return hit.iloc[0]
    hit = work.loc[work['policy'].str.contains('candidate_iql', case=False, regex=False)]
    hit = hit.loc[~hit['policy'].str.contains('rerank', case=False, regex=False)]
    if len(hit) != 1:
        raise ValueError(f"Cannot uniquely resolve Candidate-IQL row: {hit['policy'].tolist()}")
    return hit.iloc[0]

def _mode_contract(reference: Mapping[str, Any], mode: str) -> dict[str, Any]:
    contracts = reference.get('comparison_modes')
    if not isinstance(contracts, Mapping) or mode not in contracts:
        if mode == 'frozen_replay':
            return {'description': 'Legacy exact reference regression contract.', 'c0_noop_enforcement': 'hard', 'candidate_iql_enforcement': 'hard'}
        raise ValueError('fresh_retrain mode requires reproduction_reference_values_v4 with comparison_modes')
    contract = dict(contracts[mode])
    for key in ('c0_noop_enforcement', 'candidate_iql_enforcement'):
        value = str(contract.get(key, '')).strip().lower()
        if value not in {'hard', 'advisory'}:
            raise ValueError(f'Invalid {mode}.{key}={value!r}')
        contract[key] = value
    if contract['c0_noop_enforcement'] != 'hard':
        raise ValueError('C0 no-op anchors must remain hard in every mode')
    return contract

def evaluate_frame(frame: pd.DataFrame, reference: Mapping[str, Any], *, mode: str) -> dict[str, Any]:
    """Evaluate one Stage6 summary under the selected contract.

    This pure function is intentionally exposed for the synthetic regression
    verifier.  It performs no filesystem writes.
    """
    resolved_mode = _normalise_mode(mode)
    if resolved_mode == 'auto':
        raise ValueError('evaluate_frame requires a resolved non-auto mode')
    candidate_row = _candidate_row(frame)
    policy = dict(reference['comparison_policy'])
    mode_contract = _mode_contract(reference, resolved_mode)
    advisory = float(policy['advisory_abs_tolerance'])
    hard = float(policy['hard_abs_tolerance'])
    if not (math.isfinite(advisory) and math.isfinite(hard) and (0 <= advisory <= hard)):
        raise ValueError('Invalid advisory/hard tolerance ordering')
    comparisons: list[dict[str, Any]] = []
    comparison_specs = [('c0_noop', 'C0_noop', _policy_row(frame, 'C0_noop'), reference.get('stage6_c0_noop'), [f'mean_R_score_{backend}' for backend in ('alpha', 'beta', 'gamma')], mode_contract['c0_noop_enforcement']), ('candidate_iql', str(candidate_row['policy']), candidate_row, reference['stage6_final_candidate_iql'], [f'mean_delta_R_score_{backend}' for backend in ('alpha', 'beta', 'gamma')], mode_contract['candidate_iql_enforcement'])]
    for metric_group, policy_name, row, expected, metric_columns, enforcement in comparison_specs:
        if expected is None:
            continue
        n = int(float(row.get('n', expected['expected_firm_count'])))
        if n != int(expected['expected_firm_count']):
            raise ValueError(f"{policy_name} firm count={n}, expected={expected['expected_firm_count']}")
        for col in metric_columns:
            if col not in row.index:
                raise ValueError(f'Stage6 {policy_name} row lacks {col}')
            observed = float(row[col])
            target = float(expected[col])
            if not (math.isfinite(observed) and math.isfinite(target)):
                raise ValueError(f'Non-finite Stage6 comparison value for {policy_name}.{col}')
            delta = observed - target
            within_advisory = abs(delta) <= advisory
            within_hard = abs(delta) <= hard
            pass_for_mode = within_hard if enforcement == 'hard' else True
            if not pass_for_mode:
                mode_verdict = 'FAIL'
            elif enforcement == 'advisory' and (not within_hard):
                mode_verdict = 'ADVISORY_HISTORICAL_DRIFT'
            elif not within_advisory:
                mode_verdict = 'PASS_WITHIN_HARD_TOLERANCE'
            else:
                mode_verdict = 'PASS'
            comparisons.append({'metric_group': metric_group, 'policy': policy_name, 'metric': col, 'reference_value': target, 'observed_value': observed, 'difference': delta, 'absolute_difference': abs(delta), 'advisory_abs_tolerance': advisory, 'hard_abs_tolerance': hard, 'within_advisory_tolerance': within_advisory, 'within_hard_tolerance': within_hard, 'enforcement': enforcement, 'pass_for_mode': pass_for_mode, 'mode_verdict': mode_verdict})
    hard_drift = [x['metric'] for x in comparisons if not x['pass_for_mode']]
    historical_point_drift = [x['metric'] for x in comparisons if not x['within_hard_tolerance']]
    advisory_drift = [x['metric'] for x in comparisons if x['enforcement'] == 'advisory' and (not x['within_hard_tolerance'])]
    tolerance_warnings = [x['metric'] for x in comparisons if x['within_hard_tolerance'] and (not x['within_advisory_tolerance'])]
    should_hard_fail = bool(policy.get('hard_fail_on_reference_drift', True))
    status = 'FAIL' if hard_drift and should_hard_fail else 'PASS'
    candidate_drift = any((x['metric_group'] == 'candidate_iql' and (not x['within_hard_tolerance']) for x in comparisons))
    if status == 'FAIL':
        run_classification = 'FROZEN_REPLAY_REFERENCE_DRIFT' if resolved_mode == 'frozen_replay' else 'FRESH_RETRAIN_SUBSTRATE_OR_ENFORCED_CONTRACT_FAILURE'
    elif resolved_mode == 'frozen_replay':
        run_classification = 'FROZEN_REPLAY_EXACT_REFERENCE_PASS'
    elif candidate_drift:
        run_classification = 'FRESH_RETRAIN_COMPLETED_WITH_HISTORICAL_POINT_DRIFT'
    else:
        run_classification = 'FRESH_RETRAIN_HISTORICAL_POINT_MATCH'
    return {'status': status, 'comparison_mode': resolved_mode, 'mode_contract': mode_contract, 'policy': str(candidate_row['policy']), 'n_firms': int(float(candidate_row['n'])), 'comparisons': comparisons, 'hard_drift_metrics': hard_drift, 'historical_point_drift_metrics': historical_point_drift, 'advisory_drift_metrics': advisory_drift, 'tolerance_warning_metrics': tolerance_warnings, 'run_classification': run_classification, 'pipeline_may_continue': status == 'PASS'}

def build(project_root: Path, out_dir: Path, *, mode: str='auto') -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ref_path = root / REFERENCE_REL
    if not ref_path.is_file():
        raise FileNotFoundError(ref_path)
    reference = json.loads(ref_path.read_text(encoding='utf-8-sig'))
    if reference.get('schema_version') not in {'reproduction_reference_values_v1', 'reproduction_reference_values_v2', 'reproduction_reference_values_v3', 'reproduction_reference_values_v4'}:
        raise ValueError('Unsupported reproduction reference schema')
    resolved_mode, mode_source = resolve_mode(root, mode)
    summary_path = _resolve_stage6_summary(root)
    frame = pd.read_csv(summary_path, encoding='utf-8-sig')
    evaluated = evaluate_frame(frame, reference, mode=resolved_mode)
    lineage = inspect_rl_lineage(root, reference, mode=resolved_mode)
    if lineage['hard_failures']:
        evaluated['status'] = 'FAIL'
        evaluated['pipeline_may_continue'] = False
        evaluated['hard_drift_metrics'] = [*evaluated['hard_drift_metrics'], *lineage['hard_failures']]
        evaluated['run_classification'] = 'FROZEN_REPLAY_CHECKPOINT_LINEAGE_FAILURE' if resolved_mode == 'frozen_replay' else 'FRESH_RETRAIN_REQUIRED_CHECKPOINT_MISSING'
    comparisons = evaluated.pop('comparisons')
    csv_path = out / 'stage6_reference_comparison.csv'
    pd.DataFrame(comparisons).to_csv(csv_path, index=False, encoding='utf-8-sig')
    result = {'schema_version': 'reference_reproduction_comparison_v2', 'created_utc': datetime.now(timezone.utc).isoformat(), **evaluated, 'comparison_mode_source': mode_source, 'rl_checkpoint_lineage': lineage, 'reference_path': str(ref_path), 'stage6_summary_path': str(summary_path), 'comparison_output': {'path': csv_path.name, 'rows': len(comparisons)}, 'interpretation_boundary': reference['comparison_modes'][resolved_mode]['interpretation_boundary'] if isinstance(reference.get('comparison_modes'), Mapping) else reference['purpose'], 'historical_reference_purpose': reference['purpose'], 'prohibited_use': reference['comparison_policy']['prohibited_use']}
    manifest_path = out / 'reference_reproduction_comparison.json'
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return result

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--mode', default='auto', choices=VALID_MODES, help='frozen_replay hard-gates C0 and Candidate-IQL historical values; fresh_retrain hard-gates C0 while recording Candidate-IQL point drift as advisory; auto reuses the canonical prior manifest and otherwise defaults to frozen_replay')
    args = ap.parse_args(argv)
    result = build(Path(args.project_root), Path(args.out), mode=args.mode)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
