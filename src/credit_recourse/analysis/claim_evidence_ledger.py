from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .claim_evidence_common import load_yaml, write_csv, write_json
COMPLETED_VERDICTS = {'SUPPORTED', 'SUPPORTED_WITH_BOUNDARY', 'PARTIALLY_SUPPORTED', 'NOT_SUPPORTED'}

def _find_col(df: pd.DataFrame, *names: str) -> str | None:
    return next((name for name in names if name in df.columns), None)

def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise RuntimeError(f'JSON mapping required: {path}')
    return payload

def _source_path(root: Path, source_map: dict[str, dict[str, Any]], source_id: str) -> Path:
    return root / str(source_map[source_id]['canonical_evidence_path'])

def _source_sidecar(root: Path, source_map: dict[str, dict[str, Any]], source_id: str) -> dict[str, Any]:
    return _json(root / str(source_map[source_id]['sidecar_path']))

def _special(root: Path, claim: dict[str, Any], source_map: dict[str, dict[str, Any]]) -> tuple[str, str] | None:
    claim_id = str(claim['claim_id'])

    def path(source_id: str) -> Path:
        return _source_path(root, source_map, source_id)
    if claim_id == 'C02':
        frame = pd.read_csv(path('S03'), encoding='utf-8-sig')
        required = {'oracle_backend', 'split', 'direction_agreement', 'ci_low', 'ci_high', 'lead_rho'}
        if not required.issubset(frame.columns) or len(frame) != 1:
            return ('UNRESOLVED', f'S03 exact Oracle OOT row contract failed: columns={list(frame.columns)}, rows={len(frame)}')
        row = frame.iloc[0]
        if str(row['oracle_backend']).lower() != 'alpha' or str(row['split']).upper() != 'OOT':
            return ('UNRESOLVED', 'S03 is not the registered Oracle-alpha OOT row')
        estimate, lo, hi = map(float, (row['direction_agreement'], row['ci_low'], row['ci_high']))
        if not 0 <= lo <= estimate <= hi <= 1:
            return ('NOT_SUPPORTED', 'Oracle OOT direction/Wilson interval is invalid')
        return ('SUPPORTED_WITH_BOUNDARY', f'Oracle-alpha OOT direction agreement={estimate:.6f}; Wilson=[{lo:.6f},{hi:.6f}]')
    if claim_id == 'C04':
        frame = pd.read_csv(path('S06'), encoding='utf-8-sig')
        required = {'oracle_backend', 'n', 'action_effect_sd', 'tie_rate', 'preferred_action', 'best_action_agreement', 'all_three_best_action_agreement_rate'}
        if not required.issubset(frame.columns) or set(frame['oracle_backend'].astype(str).str.lower()) != {'alpha', 'beta', 'gamma'}:
            return ('UNRESOLVED', 'S06 static evaluator-resolution contract incomplete')
        return ('SUPPORTED_WITH_BOUNDARY', f"three-Oracle static action-resolution ledger materialized; all-three agreement={float(frame['all_three_best_action_agreement_rate'].iloc[0]):.6f}")
    if claim_id == 'C05':
        payload = _json(path('S25'))
        return ('SUPPORTED_WITH_BOUNDARY', 'P50 YAML, all Stage4/5 checkpoints and all rendered Stage7 matrices verified') if payload.get('status') == 'PASS' else ('UNRESOLVED', 'S25 candidate-vector verifier status is not PASS')
    if claim_id == 'C06':
        summary = pd.read_csv(path('S10'), encoding='utf-8-sig')
        required = {'record_type', 'seed', 'n_firms', 'mean_delta_R_score_alpha', 'mean_delta_R_score_beta', 'mean_delta_R_score_gamma'}
        if not required.issubset(summary.columns):
            return ('UNRESOLVED', f'S10 seven-seed columns missing: {sorted(required - set(summary.columns))}')
        seed_rows = summary.loc[summary['record_type'].astype(str).eq('SEED_POLICY_MEAN')].copy()
        seeds = sorted(pd.to_numeric(seed_rows['seed'], errors='coerce').dropna().astype(int).tolist())
        if seeds != list(range(1, 8)) or len(seed_rows) != 7:
            return ('UNRESOLVED', f'S10 requires exactly one aligned row for seeds 1..7, got {seeds}')
        if not pd.to_numeric(seed_rows['n_firms'], errors='coerce').eq(575).all():
            return ('UNRESOLVED', 'S10 seven-seed rows must each contain 575 firms')
        values = seed_rows[[f'mean_delta_R_score_{o}' for o in ('alpha', 'beta', 'gamma')]].apply(pd.to_numeric, errors='coerce')
        if not np.isfinite(values.to_numpy()).all():
            return ('UNRESOLVED', 'S10 seven-seed Oracle means contain non-finite values')
        return ('SUPPORTED_WITH_BOUNDARY', 'seven-seed Candidate-IQL ledger and primary reference ladder materialized')
    if claim_id == 'C09':
        frame = pd.read_csv(path('S12'), encoding='utf-8-sig')
        backend = _find_col(frame, 'backend', 'backend_id', 'backend_label')
        estimate = _find_col(frame, 'mean_diff', 'mean_gap', 'estimate')
        p_holm = _find_col(frame, 'p_holm', 'wilcoxon_p_holm', 'holm_p', 'p_adjusted')
        required_backends = {'gpt54mini', 'gpt41mini', 'haiku45'}
        if not backend or not estimate or (not p_holm):
            return ('UNRESOLVED', 'S12 missing backend/estimate/Holm columns')
        aliases = {'GPT-5.4-mini': 'gpt54mini', 'GPT-4.1-mini': 'gpt41mini', 'Haiku-4.5': 'haiku45'}
        normalized = frame[backend].astype(str).replace(aliases)
        selected = frame.loc[normalized.isin(required_backends)].copy()
        selected['_backend'] = normalized.loc[selected.index]
        if set(selected['_backend']) != required_backends or selected['_backend'].duplicated().any():
            return ('UNRESOLVED', 'S12 must contain exactly one row for each registered H1 backend')
        ok = (pd.to_numeric(selected[estimate], errors='coerce') < 0).all() and (pd.to_numeric(selected[p_holm], errors='coerce') < 0.05).all()
        return ('SUPPORTED', 'three-backend negative H1 effects, Holm-significant') if ok else ('NOT_SUPPORTED', 'registered three-backend H1 rule not met')
    if claim_id == 'C11':
        seed_frame = pd.read_csv(path('S10'), encoding='utf-8-sig')
        required = {'record_type', 'seed', 'seed_a', 'seed_b', 'n_firms', 'action_identity_rate', 'effect_rank_spearman_alpha', 'effect_rank_spearman_beta', 'effect_rank_spearman_gamma'}
        if not required.issubset(seed_frame.columns):
            return ('UNRESOLVED', f'S10 multi-resolution columns missing: {sorted(required - set(seed_frame.columns))}')
        seed_rows = seed_frame.loc[seed_frame['record_type'].astype(str).eq('SEED_POLICY_MEAN')]
        pair_rows = seed_frame.loc[seed_frame['record_type'].astype(str).eq('SEED_PAIR_STABILITY')]
        seeds = sorted(pd.to_numeric(seed_rows['seed'], errors='coerce').dropna().astype(int).tolist())
        if seeds != list(range(1, 8)) or len(pair_rows) != 21:
            return ('UNRESOLVED', f'S10 requires 7 seed rows and 21 seed-pair rows; got seeds={seeds}, pairs={len(pair_rows)}')
        if pair_rows.duplicated(['seed_a', 'seed_b']).any() or not pd.to_numeric(pair_rows['n_firms'], errors='coerce').eq(575).all():
            return ('UNRESOLVED', 'S10 seed-pair key or firm-count contract failed')
        metrics = pair_rows[['action_identity_rate', 'effect_rank_spearman_alpha', 'effect_rank_spearman_beta', 'effect_rank_spearman_gamma']].apply(pd.to_numeric, errors='coerce')
        if not np.isfinite(metrics.to_numpy()).all():
            return ('UNRESOLVED', 'S10 firm-level reproducibility metrics are non-finite')
        lineage = pd.read_csv(path('S19'), encoding='utf-8-sig')
        lineage_required = {'action_vector_match'}
        if not lineage_required.issubset(lineage.columns) or lineage.empty:
            return ('UNRESOLVED', 'S19 E2-E3 live-repeat lineage contract incomplete')
        return ('SUPPORTED_WITH_BOUNDARY', 'mean value, firm-level rank, action identity, and independent live-repeat lineage materialized as separate resolutions')
    if claim_id == 'C12':
        frame = pd.read_csv(path('S14'), encoding='utf-8-sig')
        ok = 'execution_contract' in frame.columns and set(frame['execution_contract'].dropna().astype(str)) == {'LEGACY_H3_FREEFORM_REFERENCE_SOURCE'} and ('not_e4_matched_budget_qc' in frame.columns) and frame['not_e4_matched_budget_qc'].astype(bool).all()
        return ('SUPPORTED_WITH_BOUNDARY', 'H3 legacy free-form contract explicitly separated from E4') if ok else ('UNRESOLVED', 'S14 execution-contract labels incomplete')
    if claim_id == 'C13':
        frame = pd.read_csv(path('S15'), encoding='utf-8-sig')
        required = {'backend', 'oracle_backend', 'information_condition', 'mode', 'contrast', 'n_pairs', 'mean_diff', 'ci_low', 'ci_high', 'p_raw', 'p_holm', 'decision', 'holm_family'}
        if not required.issubset(frame.columns) or len(frame) != 3:
            return ('UNRESOLVED', 'S15 requires exactly three Oracle rows for the C8 paired-inference family')
        if set(frame['oracle_backend'].astype(str).str.lower()) != {'alpha', 'beta', 'gamma'}:
            return ('UNRESOLVED', 'S15 must contain alpha/beta/gamma exactly once')
        if frame['oracle_backend'].astype(str).str.lower().duplicated().any():
            return ('UNRESOLVED', 'S15 contains duplicate Oracle rows')
        scope_ok = frame['backend'].astype(str).eq('gpt54mini').all() and frame['information_condition'].astype(str).eq('IC-b').all() and frame['mode'].astype(str).eq('free_form_10d').all() and frame['contrast'].astype(str).eq('C8-C6').all() and (frame['holm_family'].astype(str).nunique() == 1)
        numeric = frame[['n_pairs', 'mean_diff', 'ci_low', 'ci_high', 'p_raw', 'p_holm']].apply(pd.to_numeric, errors='coerce')
        if not scope_ok or not np.isfinite(numeric.to_numpy()).all():
            return ('UNRESOLVED', 'S15 scope or finite inference contract failed')
        if not ((numeric['ci_low'] <= numeric['mean_diff']) & (numeric['mean_diff'] <= numeric['ci_high'])).all():
            return ('UNRESOLVED', 'S15 confidence interval ordering failed')
        decisions = ', '.join((f'{row.oracle_backend}={row.decision}' for row in frame.sort_values('oracle_backend').itertuples()))
        return ('SUPPORTED_WITH_BOUNDARY', f'C8 timing contrast evaluated across three Oracles: {decisions}')
    if claim_id == 'C15':
        frame = pd.read_csv(path('S16'), encoding='utf-8-sig')
        component = _find_col(frame, 'component', 'contrast')
        budget = _find_col(frame, 'finite_budget_label', 'budget_label', 'budget')
        value = _find_col(frame, 'finite_minus_unbounded_interaction', 'interaction', 'mean_interaction')
        backend = _find_col(frame, 'cohort_id', 'backend', 'backend_id')
        oracle = _find_col(frame, 'oracle_backend', 'oracle')
        if not component or not budget or (not value):
            return ('UNRESOLVED', 'S16 missing component/budget/interaction')
        subset = frame.copy()
        if backend:
            subset = subset.loc[subset[backend].astype(str).str.contains('gpt54mini', case=False, regex=False)]
        if oracle:
            subset = subset.loc[subset[oracle].astype(str).str.lower().eq('alpha')]
        aliases = {'package': ['reference_plus_revision_package', 'package'], 'self': ['self_revision', 'self-review'], 'ref': ['reference_content_conditional', 'reference']}
        ok = True
        details: list[str] = []
        for label in ('0p75', '1p27', '2p00'):
            values: dict[str, float] = {}
            for key, allowed in aliases.items():
                matched = subset.loc[subset[budget].astype(str).eq(label) & subset[component].astype(str).isin(allowed)]
                if len(matched) != 1:
                    ok = False
                    continue
                values[key] = float(pd.to_numeric(matched[value], errors='raise').iloc[0])
            if len(values) == 3:
                identity = abs(values['package'] - (values['self'] + values['ref'])) <= 1e-06
                dominance = abs(values['self']) > abs(values['ref'])
                ok &= identity and dominance
                details.append(f'{label}: identity={identity}, dominance={dominance}')
        return ('SUPPORTED_WITH_BOUNDARY', '; '.join(details)) if ok else ('NOT_SUPPORTED', 'C15 component identity/dominance rule failed')
    if claim_id == 'C19':
        frame = pd.read_csv(path('S22'), encoding='utf-8-sig')
        exact = 'valid firms with at least one hard-bound-clipped axis / n'
        ok = 'denominator_definition' in frame.columns and frame['denominator_definition'].astype(str).eq(exact).all() and ('axis_bound_clip_firm_rate' in frame.columns)
        return ('SUPPORTED_WITH_BOUNDARY', 'firm-level >=1 clipped axis denominator verified') if ok else ('UNRESOLVED', 'C19 denominator contract missing')
    if claim_id == 'C20':
        frame = pd.read_csv(path('S23'), encoding='utf-8-sig')
        if 'decision' not in frame.columns:
            return ('UNRESOLVED', 'S23 decision missing')
        decisions = set(frame['decision'].dropna().astype(str))
        if decisions == {'REVISION_DILUTION_CONFIRMED'}:
            return ('SUPPORTED', 'all registered cells confirm revision dilution')
        if decisions & {'REVISION_DILUTION_CONFIRMED', 'CONSISTENT_WITH_REVISION_DILUTION'}:
            return ('PARTIALLY_SUPPORTED', 'some cells are consistent with revision dilution')
        return ('NOT_SUPPORTED', 'five-criterion revision-dilution rule not met')
    if claim_id == 'C25':
        payload = _json(path('S28'))
        required_warnings = {'EVALUATOR_ONLY_POST_HOC', 'SHARED_TERM_WARNING', 'MEAN_REVERSION_WARNING', 'NO_AUTOMATIC_ROUTING'}
        required_outputs = {'firm_ledger', 'quartile_summary', 'complement_substitute_summary', 'adoption_gain_analysis', 'shared_term_sensitivity', 'mean_reversion_sensitivity'}
        ok = payload.get('status') == 'PASS' and required_warnings <= set(payload.get('warnings', [])) and (required_outputs <= set((payload.get('outputs') or {}).keys()))
        return ('SUPPORTED_WITH_BOUNDARY', 'E3 relational-reference analysis and both sensitivity checks PASS') if ok else ('UNRESOLVED', 'S28 semantic relational-reference contract incomplete')
    if claim_id == 'C26':
        frame = pd.read_csv(path('S29'), encoding='utf-8-sig')
        required_backends = {'gpt54mini', 'gpt41mini', 'haiku45'}
        required_contrasts = {'C5-C4', 'C7-C6'}
        required_metrics = {'projection_distance', 'l1', 'active_dimensions', 'format_failures', 'action_contract_consistency', 'reviewability_proxy'}
        required_columns = {'backend', 'contrast', 'metric', 'n_pairs', 'estimate', 'ci_low', 'ci_high', 'p_raw', 'p_holm', 'decision', 'execution_contract'}
        if not required_columns.issubset(frame.columns):
            return ('UNRESOLVED', f'S29 columns incomplete: {sorted(required_columns - set(frame.columns))}')
        complete_cells = set(zip(frame['backend'].astype(str), frame['contrast'].astype(str), frame['metric'].astype(str)))
        expected_cells = {(backend, contrast, metric) for backend in required_backends for contrast in required_contrasts for metric in required_metrics}
        numeric = frame[['n_pairs', 'estimate', 'ci_low', 'ci_high', 'p_raw', 'p_holm']].apply(pd.to_numeric, errors='coerce')
        ok = expected_cells <= complete_cells and len(frame) == len(expected_cells) and np.isfinite(numeric.to_numpy()).all() and (set(frame['execution_contract'].astype(str)) == {'LEGACY_THREE_BACKEND_REASONING'})
        return ('SUPPORTED_WITH_BOUNDARY', 'H2 paired three-backend inference ledger complete') if ok else ('UNRESOLVED', 'S29 paired H2 inference contract incomplete')
    if claim_id == 'C27':
        payload = _json(path('S30'))
        required = {'grade_mapping_contract', 'summary_path', 'grade_mapping_sources'}
        return ('SUPPORTED_WITH_BOUNDARY', 'E3 dynamic resolution uses explicit Oracle grade mappings') if payload.get('status') == 'PASS' and required <= set(payload) else ('UNRESOLVED', 'S30 semantic dynamic-resolution contract incomplete')
    return None

def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / 'src/credit_recourse/configs'
    claims = load_yaml(config_root / 'claim_evidence_registry.yaml')['claims']
    sources = load_yaml(config_root / 'source_registry.yaml')['sources']
    source_map = {str(source['source_id']): source for source in sources}
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for claim in claims:
        claim_id = str(claim['claim_id'])
        source_ids = [str(source_id) for source_id in claim.get('source_ids', [])]
        missing: list[str] = []
        invalid_sidecars: list[str] = []
        for source_id in source_ids:
            source = source_map[source_id]
            canonical = root / str(source['canonical_evidence_path'])
            sidecar = root / str(source['sidecar_path'])
            if not canonical.is_file() or not sidecar.is_file():
                missing.append(source_id)
                continue
            try:
                side_payload = _json(sidecar)
                if side_payload.get('status') != 'MATERIALIZED':
                    invalid_sidecars.append(source_id)
            except Exception:
                invalid_sidecars.append(source_id)
        if missing:
            verdict, reason = ('UNRESOLVED', f'missing canonical sources: {missing}')
            evidence_status = 'MISSING'
        elif invalid_sidecars:
            verdict, reason = ('UNRESOLVED', f'source sidecars are not MATERIALIZED: {invalid_sidecars}')
            evidence_status = 'INVALID_SIDECAR'
        else:
            try:
                special = _special(root, claim, source_map)
                verdict, reason = special if special else ('SUPPORTED_WITH_BOUNDARY', 'registered source materialization and interpretation boundary passed')
                evidence_status = 'MATERIALIZED_AND_EVALUATED'
            except Exception as exc:
                verdict, reason = ('UNRESOLVED', f'runtime decision evaluation failed: {type(exc).__name__}: {exc}')
                evidence_status = 'DECISION_ERROR'
        row = {'claim_id': claim_id, 'priority': claim['priority'], 'claim_text_ko': claim['claim_text_ko'], 'evidence_class': claim['evidence_class'], 'source_ids': ';'.join(source_ids), 'decision_ids': ';'.join(claim.get('decision_ids', [])), 'final_slots': ';'.join(claim.get('final_slots', [])), 'planned_writing_status': claim.get('writing_status'), 'planned_freeze_status': claim.get('freeze_status'), 'runtime_evidence_status': evidence_status, 'runtime_verdict': verdict, 'verdict': verdict, 'reason': reason, 'allowed_interpretation': claim.get('allowed_interpretation'), 'forbidden_interpretation': claim.get('forbidden_interpretation'), 'paper_locations': ';'.join(claim.get('paper_locations', []))}
        rows.append(row)
        if verdict == 'UNRESOLVED':
            unresolved.append(row)
    output_root = root / 'data/reproduction/claim_evidence'
    fieldnames = list(rows[0]) if rows else []
    write_csv(output_root / 'claim_evidence_ledger.csv', rows, fieldnames=fieldnames)
    write_csv(output_root / 'unresolved_claims.csv', unresolved, fieldnames=fieldnames)
    write_csv(output_root / 'claim_evidence_values.csv', [{'claim_id': claim['claim_id'], 'key_evidence_text': claim['key_evidence_text'], 'runtime_verdict': next((row['runtime_verdict'] for row in rows if row['claim_id'] == claim['claim_id']))} for claim in claims])
    p0_unresolved = [row for row in unresolved if str(row['priority']).startswith('P0')]
    completed = [row for row in rows if row['runtime_verdict'] in COMPLETED_VERDICTS]
    result = {'schema_version': 'claim_evidence_ledger_v4_1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if not unresolved else 'PARTIAL', 'claim_count': len(rows), 'completed_decision_count': len(completed), 'unresolved_count': len(unresolved), 'p0_unresolved_count': len(p0_unresolved)}
    write_json(output_root / 'claim_evidence_ledger_report.json', result)
    return result

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
