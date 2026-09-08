from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from .missing_evidence_common import load_first, col, action_cols
from .claim_evidence_common import repo_rel, write_csv

def run(root: Path):
    p, d = load_first(root, ['data/analysis/paper_repro/05_extension_e3_e4/**/c4r_matched_firm_frame*.parquet', 'data/analysis/paper_repro/05_extension_e3_e4/**/c4r_matched_firm_frame*.csv', 'data/analysis/paper_repro/05_extension_e3_e4/**/*.parquet', 'data/analysis/paper_repro/05_extension_e3_e4/**/*.csv'], required_cols=['C4_alpha', 'C4R_alpha', 'c4__action__a0', 'c4r__action__a0'])
    a4 = action_cols(d, 'c4__action__')
    ar = action_cols(d, 'c4r__action__')
    if len(a4) != len(ar):
        raise RuntimeError('C4/C4R action dimensionality mismatch')
    score4 = col(d, 'C4_alpha', 'c4_oracle_alpha', 'c4__oracle__alpha', 'c4_alpha')
    scorer = col(d, 'C4R_alpha', 'c4r_oracle_alpha', 'c4r__oracle__alpha', 'c4r_alpha')
    group = [c for c in ('backend_id', 'backend', 'budget_label', 'l1_budget') if c in d.columns]
    rows = []
    for keys, g in d.groupby(group, dropna=False):
        x = g[a4].to_numpy(float)
        y = g[ar].to_numpy(float)
        l1x = np.abs(x).sum(1)
        l1y = np.abs(y).sum(1)
        ax = (np.abs(x) > 1e-12).sum(1)
        ay = (np.abs(y) > 1e-12).sum(1)
        cx = np.max(np.abs(x), axis=1) / np.maximum(l1x, 1e-12)
        cy = np.max(np.abs(y), axis=1) / np.maximum(l1y, 1e-12)
        removal = np.minimum(np.abs(x), np.maximum(np.abs(x) - np.abs(y), 0)).sum(1)
        crit = [float((l1y <= l1x + 1e-12).mean()) >= 0.5, float((ay > ax).mean()) >= 0.5, float((cy < cx).mean()) >= 0.5, float((removal > 0).mean()) >= 0.5, float((pd.to_numeric(g[scorer]) - pd.to_numeric(g[score4]) < 0).mean()) >= 0.5]
        verdict = 'REVISION_DILUTION_CONFIRMED' if all(crit) else 'CONSISTENT_WITH_REVISION_DILUTION' if sum(crit) >= 3 else 'NOT_SUPPORTED'
        rec = dict(zip(group, keys if isinstance(keys, tuple) else (keys,)))
        rec.update({'n': len(g), 'criterion_l1_nonincrease': crit[0], 'criterion_active_dimensions_increase': crit[1], 'criterion_concentration_decrease': crit[2], 'criterion_removal_increase': crit[3], 'criterion_score_decrease': crit[4], 'criteria_met': sum(crit), 'decision': verdict, 'source_path': repo_rel(root, p)})
        rows.append(rec)
    out = root / 'data/analysis/paper_repro/05_extension_e3_e4/e3_c4r_journal/revision_dilution/revision_dilution_summary.csv'
    write_csv(out, rows)
    return {'status': 'PASS', 'rows': len(rows)}

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    a = ap.parse_args(argv)
    r = run(Path(a.project_root))
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
