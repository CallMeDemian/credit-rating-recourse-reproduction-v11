from __future__ import annotations
'Recode existing Stage8 LLM failure audits with feasibility rule v2.\n\nThis utility is for frozen/postpatch LLM runs where Stage8 already produced\n``llm_stage8_failure_audit_enriched.csv`` but feasibility was coded by the v1\nrule that treated simulator residual-presentation repair as a hard failure.  It\nrequires no LLM API call and no Oracle re-scoring; it only rewrites the\nfeasibility-related audit columns and, when requested, regenerates the Stage9\naggregated failure-audit CSV from the recoded row-level taxonomy.\n'
import argparse
import json
from pathlib import Path
from typing import Any
import pandas as pd
from credit_recourse.rl.common.io import write_json
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.failure_coder import FEASIBILITY_RULE_VERSION, FAILURE_CODER_VERSION, ORACLE_SCORES_USED_FOR_FAILURE_CODING, code_feasibility_violation
from credit_recourse.eval.final_stage8_llm_multi_oracle_eval.failure_enrichment import ENRICHED_FAILURE_AUDIT_SCHEMA_VERSION, FEASIBILITY_THRESHOLDS
from credit_recourse.verification.verify_stage8_feasibility_rule import verify_stage8_feasibility_rule
FEASIBILITY_COLUMNS = ['feasibility_violation_auto', 'feasibility_review_needed', 'feasibility_error_reason', 'plug_to_assets', 'plug_denominator_source', 'accounting_check_failed', 'negative_balance_flag', 'residual_presentation_repair_flag', 'plug_to_assets_review_exceeded', 'plug_to_assets_hard_exceeded', 'feasibility_core_violation_flag', 'feasibility_rule_version', 'failure_coder_version', 'oracle_scores_used_for_failure_coding']
STAGE9_FAILURE_CATEGORIES = ['translational_failure', 'structural_out_of_scope', 'direction_error', 'magnitude_error', 'feasibility_violation', 'liquidity_destructive_recourse', 'anchoring_or_confirmation_failure', 'ungrounded_judgment']

def _taxonomy_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and pd.isna(value):
        return set()
    return {str(t).strip() for t in str(value).split(',') if str(t).strip() and str(t).strip().lower() != 'nan'}

def _bool_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(df[col].fillna(False).astype(bool).sum())

def recode_enriched_failure_audit(stage8_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    stage8_dir = Path(stage8_dir)
    path = stage8_dir / 'llm_stage8_failure_audit_enriched.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing Stage8 enriched failure audit: {path}')
    df = pd.read_csv(path)
    if 'failure_categories' not in df.columns:
        raise KeyError('Stage8 enriched failure audit missing failure_categories')
    records: list[dict[str, Any]] = []
    new_categories: list[str] = []
    base_category_col = 'failure_categories_stage7' if 'failure_categories_stage7' in df.columns else 'failure_categories'
    for row in df.to_dict('records'):
        coded = code_feasibility_violation(row, plug_to_assets_hard=FEASIBILITY_THRESHOLDS['plug_to_assets_hard'], plug_to_assets_review=FEASIBILITY_THRESHOLDS['plug_to_assets_review'])
        records.append(coded)
        cats = _taxonomy_set(row.get(base_category_col))
        cats.discard('feasibility_violation')
        if coded['feasibility_violation_auto']:
            cats.add('feasibility_violation')
        new_categories.append(','.join(sorted(cats)))
    out = df.copy()
    for col in FEASIBILITY_COLUMNS:
        out[col] = [r[col] for r in records]
    if 'failure_categories_stage7' not in out.columns:
        out['failure_categories_stage7'] = df['failure_categories'].fillna('').astype(str)
    out['failure_categories'] = new_categories
    out['failure_count'] = out['failure_categories'].fillna('').astype(str).apply(lambda s: len([t for t in s.split(',') if t]))
    meta = {'schema_version': ENRICHED_FAILURE_AUDIT_SCHEMA_VERSION, 'failure_coder_version': FAILURE_CODER_VERSION, 'feasibility_rule_version': FEASIBILITY_RULE_VERSION, 'oracle_scores_used_for_failure_coding': ORACLE_SCORES_USED_FOR_FAILURE_CODING, 'feasibility_thresholds': FEASIBILITY_THRESHOLDS, 'row_count': int(len(out)), 'auto_feasibility_violation_count': _bool_count(out, 'feasibility_violation_auto'), 'core_feasibility_violation_count': _bool_count(out, 'feasibility_core_violation_flag'), 'review_needed_count': _bool_count(out, 'feasibility_review_needed'), 'residual_presentation_repair_count': _bool_count(out, 'residual_presentation_repair_flag'), 'plug_to_assets_review_exceeded_count': _bool_count(out, 'plug_to_assets_review_exceeded'), 'plug_to_assets_hard_exceeded_count': _bool_count(out, 'plug_to_assets_hard_exceeded'), 'output': path.name, 'recode_source': 'existing_stage8_enriched_failure_audit'}
    return (out, meta)

def aggregate_stage9_failure_audit(enriched: pd.DataFrame) -> pd.DataFrame:
    required = {'policy', 'mode', 'information_condition', 'failure_categories'}
    missing = sorted(required - set(enriched.columns))
    if missing:
        raise KeyError(f'Cannot aggregate Stage9 failure audit; missing columns: {missing}')
    df = enriched.copy()
    df['failure_categories_list'] = df['failure_categories'].fillna('').astype(str).apply(lambda s: [t for t in s.split(',') if t])
    df['has_any_failure'] = df['failure_categories_list'].apply(lambda x: len(x) > 0)
    for cat in STAGE9_FAILURE_CATEGORIES:
        df[f'has_{cat}'] = df['failure_categories_list'].apply(lambda x, c=cat: c in x)
    if 'routed_to_simulator' not in df.columns:
        df['routed_to_simulator'] = False
    rows = []
    for (policy, mode, ic), g in df.groupby(['policy', 'mode', 'information_condition'], dropna=False):
        rec = {'policy': str(policy), 'mode': str(mode), 'information_condition': str(ic), 'n_rows': int(len(g)), 'n_routed_to_simulator': int(g['routed_to_simulator'].fillna(False).astype(bool).sum()), 'n_with_any_failure': int(g['has_any_failure'].sum())}
        for cat in STAGE9_FAILURE_CATEGORIES:
            rec[f'fail_{cat}'] = int(g[f'has_{cat}'].sum())
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(['policy', 'mode', 'information_condition']).reset_index(drop=True)

def main() -> None:
    ap = argparse.ArgumentParser(description='Recode existing Stage8 feasibility audit using v2 rule')
    ap.add_argument('--stage8-dir', required=True, type=Path)
    ap.add_argument('--stage9-dir', type=Path, default=None, help='Optional Stage9 directory whose llm_stage9_failure_audit.csv should be regenerated')
    ap.add_argument('--in-place', action='store_true', help='Overwrite existing Stage8/Stage9 audit artifacts; otherwise write *.v2 files')
    ap.add_argument('--out-summary', type=Path, default=None)
    args = ap.parse_args()
    enriched, meta = recode_enriched_failure_audit(args.stage8_dir)
    stage8_dir = Path(args.stage8_dir)
    out_name = 'llm_stage8_failure_audit_enriched.csv' if args.in_place else 'llm_stage8_failure_audit_enriched.v2.csv'
    manifest_name = 'failure_coder_manifest.json' if args.in_place else 'failure_coder_manifest.v2.json'
    enriched.to_csv(stage8_dir / out_name, index=False, encoding='utf-8-sig')
    write_json(stage8_dir / manifest_name, meta)
    verifier_summary = None
    if args.in_place:
        verifier_summary = verify_stage8_feasibility_rule(stage8_dir)
    stage9_path = None
    if args.stage9_dir is not None:
        stage9_dir = Path(args.stage9_dir)
        stage9_dir.mkdir(parents=True, exist_ok=True)
        stage9 = aggregate_stage9_failure_audit(enriched)
        stage9_name = 'llm_stage9_failure_audit.csv' if args.in_place else 'llm_stage9_failure_audit.v2.csv'
        stage9_path = stage9_dir / stage9_name
        stage9.to_csv(stage9_path, index=False, encoding='utf-8-sig')
    summary = {'status': 'PASS', 'stage8_dir': str(stage8_dir), 'stage8_output': out_name, 'manifest_output': manifest_name, 'stage9_output': str(stage9_path) if stage9_path else None, 'meta': meta, 'verifier_summary': verifier_summary}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.out_summary:
        args.out_summary.parent.mkdir(parents=True, exist_ok=True)
        args.out_summary.write_text(text + '\n', encoding='utf-8')
if __name__ == '__main__':
    main()
