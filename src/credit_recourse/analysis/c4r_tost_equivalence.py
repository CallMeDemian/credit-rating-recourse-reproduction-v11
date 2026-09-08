"""Oracle-alpha TOST for the preregistered C4R journal extension.

Consumes the row-level firm frame emitted by ``c4r_matched_inference_v3`` and
writes two preregistered equivalence ledgers:

1. within-arm components: C4R-C4, C6-C4R, and C6-C4;
2. finite-minus-unbounded capacity interactions for the same components.

The equivalence margin and primary Oracle are read from the supplied frozen
preregistration JSON.  No LLM API, simulation, or model fitting is performed.
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from credit_recourse.analysis.llm_tost_equivalence import paired_tost
from credit_recourse.analysis.n5_7_10c_holm_inference import wilcoxon_paired_p
SCHEMA_VERSION = 'c4r_tost_equivalence_v1'
COMPONENTS = {'self_revision': 'self_revision_alpha', 'reference_content_conditional': 'reference_content_conditional_alpha', 'package': 'package_alpha'}
REQUIRED_BUDGETS = ('0p75', '1p27', '2p00', 'unbounded')
FINITE_BUDGETS = ('0p75', '1p27', '2p00')

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    raise ValueError(f'Unsupported firm-frame extension: {path.suffix}')

def _load_contract(path: Path) -> tuple[str, float, dict]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    eq = payload.get('equivalence')
    if not isinstance(eq, dict):
        raise ValueError('Preregistration lacks an equivalence object.')
    oracle = str(eq.get('primary_oracle', '')).strip().lower()
    margin = float(eq.get('raw_score_margin'))
    method = str(eq.get('method', '')).strip().upper()
    if oracle != 'alpha':
        raise ValueError(f'Only preregistered Oracle-alpha equivalence is allowed; got {oracle!r}')
    if method != 'TOST':
        raise ValueError(f'Expected preregistered TOST method; got {method!r}')
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError(f'Invalid preregistered equivalence margin: {margin}')
    return (oracle, margin, payload)

def _validate_frame(df: pd.DataFrame) -> list[str]:
    required = {'cohort_id', 'budget_label', 'row_id', *COMPONENTS.values()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Firm frame missing columns: {missing}')
    cohorts = sorted(df['cohort_id'].astype(str).unique().tolist())
    if not cohorts:
        raise ValueError('Firm frame has no cohorts.')
    for cohort in cohorts:
        g = df[df['cohort_id'].astype(str).eq(cohort)].copy()
        labels = set(g['budget_label'].astype(str))
        missing_budgets = sorted(set(REQUIRED_BUDGETS) - labels)
        if missing_budgets:
            raise ValueError(f'{cohort}: missing budgets {missing_budgets}')
        for budget in REQUIRED_BUDGETS:
            arm = g[g['budget_label'].astype(str).eq(budget)]
            if arm.empty:
                raise ValueError(f'{cohort}/{budget}: empty arm')
            if arm['row_id'].duplicated().any():
                raise ValueError(f'{cohort}/{budget}: duplicate row_id')
    return cohorts

def _describe_test(x: np.ndarray, *, margin: float, alpha: float) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    stats = paired_tost(x, margin=margin, alpha=alpha)
    p_w = wilcoxon_paired_p(pd.Series(x))
    stats.update({'wilcoxon_p_two_sided': float(p_w) if np.isfinite(p_w) else np.nan, 'mean_outside_margin': bool(abs(float(stats['mean_gap'])) >= margin), 'decision': 'PRACTICALLY_EQUIVALENT' if stats['equivalent'] else 'NOT_EQUIVALENT_OR_INCONCLUSIVE'})
    return stats

def run(*, firm_frame: Path, prereg_path: Path, out: Path, alpha: float=0.05) -> dict:
    firm_frame = firm_frame.resolve()
    prereg_path = prereg_path.resolve()
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    oracle, margin, prereg = _load_contract(prereg_path)
    df = _read_frame(firm_frame)
    cohorts = _validate_frame(df)
    within_rows: list[dict] = []
    interaction_rows: list[dict] = []
    for cohort in cohorts:
        cdf = df[df['cohort_id'].astype(str).eq(cohort)].copy()
        for budget in REQUIRED_BUDGETS:
            arm = cdf[cdf['budget_label'].astype(str).eq(budget)].copy()
            for component, column in COMPONENTS.items():
                s = pd.to_numeric(arm[column], errors='coerce').to_numpy(dtype=float)
                rec = _describe_test(s, margin=margin, alpha=alpha)
                rec.update({'test_family': 'within_arm_component_vs_zero', 'cohort_id': cohort, 'budget_label': budget, 'component': component, 'oracle': oracle, 'score_column': column})
                within_rows.append(rec)
        unb = cdf[cdf['budget_label'].astype(str).eq('unbounded')].copy()
        for budget in FINITE_BUDGETS:
            finite = cdf[cdf['budget_label'].astype(str).eq(budget)].copy()
            for component, column in COMPONENTS.items():
                m = finite[['row_id', column]].rename(columns={column: 'finite'}).merge(unb[['row_id', column]].rename(columns={column: 'unbounded'}), on='row_id', how='inner', validate='one_to_one')
                if len(m) != len(finite) or len(m) != len(unb):
                    raise ValueError(f'{cohort}/{budget}/{component}: finite-unbounded row universe mismatch')
                gap = (pd.to_numeric(m['finite'], errors='coerce') - pd.to_numeric(m['unbounded'], errors='coerce')).to_numpy(dtype=float)
                rec = _describe_test(gap, margin=margin, alpha=alpha)
                rec.update({'test_family': 'finite_minus_unbounded_interaction_vs_zero', 'cohort_id': cohort, 'budget_label': budget, 'reference_budget_label': 'unbounded', 'component': component, 'oracle': oracle, 'score_column': column})
                interaction_rows.append(rec)
    within = pd.DataFrame(within_rows)
    interactions = pd.DataFrame(interaction_rows)
    within_path = out / 'c4r_tost_within_arm_alpha.csv'
    interactions_path = out / 'c4r_tost_interactions_alpha.csv'
    within.to_csv(within_path, index=False, encoding='utf-8-sig')
    interactions.to_csv(interactions_path, index=False, encoding='utf-8-sig')
    summary = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'created_utc': _now(), 'firm_frame': str(firm_frame), 'preregistration': {'path': str(prereg_path), 'design_status': prereg.get('design_status')}, 'primary_oracle': oracle, 'equivalence_method': 'TOST', 'equivalence_margin': margin, 'alpha': alpha, 'ci_level': 1.0 - 2.0 * alpha, 'cohorts': cohorts, 'budgets': list(REQUIRED_BUDGETS), 'components': list(COMPONENTS), 'within_arm_row_count': int(len(within)), 'interaction_row_count': int(len(interactions)), 'within_arm_equivalent_count': int(within['equivalent'].sum()), 'interaction_equivalent_count': int(interactions['equivalent'].sum()), 'outputs': {'within_arm': {'path': within_path.name}, 'interactions': {'path': interactions_path.name}}, 'interpretation_boundary': f'Equivalence claims are restricted to Oracle alpha and the preregistered raw-score margin ±{margin}. A failed TOST is not evidence of meaningful difference; it may be inconclusive.'}
    (out / 'metadata.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Preregistered Oracle-alpha TOST for C4R v3 firm frame')
    ap.add_argument('--firm-frame', required=True, type=Path)
    ap.add_argument('--prereg-path', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--alpha', type=float, default=0.05)
    args = ap.parse_args(argv)
    result = run(firm_frame=args.firm_frame, prereg_path=args.prereg_path, out=args.out, alpha=float(args.alpha))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
