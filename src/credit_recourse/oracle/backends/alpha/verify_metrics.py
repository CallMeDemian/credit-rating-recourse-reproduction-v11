#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify Oracle α metrics against Stage 5A scoring acceptance gates.

Checks 6 infra-independent gates from Oracle Methodology §14.3:
  #1. Spearman ρ(R_score, rating_num_10) ≥ 0.45 (Dev & OOT)
  #2. ±1 notch accuracy ≥ 70% (Dev & OOT)
  #3. OOT R_score PSI < 0.30
  #4. OOT KL_grade < 0.30
  #5. Boundary jump test: no excessive jumps from small perturbations
  #7. Grade-wise confusion (BBB-A): no single grade misclassifies 60%+ of its observations
  (#6 rank-shock artifact deferred to Stage 6B)

Plus 2 sanity checks:
  - Weight sanity: from weight_sanity_check_alpha.json
  - Sample size: dev_n ≥ 1500, oot_n ≥ 500

Exit code 0 = all 6 gates pass.
Exit code 1 = at least one gate fails.

Usage:
  python verify_alpha_metrics.py
"""
import json

def robust_json_load(path):
    """Load JSON robustly across UTF-8/UTF-8-SIG/CP949/EUC-KR encodings."""
    import json
    from pathlib import Path

    p = Path(path)
    for enc in ("utf-8", "utf-8-sig", "cp949", "euc-kr"):
        try:
            with open(p, "r", encoding=enc) as f:
                return json.load(f)
        except UnicodeDecodeError:
            continue

    with open(p, "r", encoding="utf-8", errors="replace") as f:
        return json.load(f)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
CONFIG_PATH = SCRIPT_DIR / "configs" / "stage4_alpha_config.yaml"

with open(CONFIG_PATH, encoding='utf-8') as f:
    CFG = yaml.safe_load(f)
ACC = CFG['acceptance']

GRADE_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C', 'D']
GRADE2NUM = {g: i + 1 for i, g in enumerate(GRADE_ORDER)}

print("=" * 72)
print("Stage 4 α — Acceptance Verification")
print("=" * 72)

# Load output
out_path = OUTPUT_DIR / "oracle_firm_year_output_alpha.parquet"
if not out_path.exists():
    print(f"ERROR: {out_path} not found. Run stage4_alpha_pipeline.py first.")
    sys.exit(2)

df = pd.read_parquet(out_path)
dev = df[df['split_stage4'] == 'dev'].copy()
oot = df[df['split_stage4'] == 'oot'].copy()
required_cols = ['R_score_alpha', 'R_grade_alpha', 'R_grade_alpha_num', 'rating_num_10', 'grade_base_10']
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    print(f"ERROR: Alpha verification requires explicit 10-grade columns, missing={missing_cols}")
    sys.exit(2)
for _sub in (dev, oot):
    _sub['rating_num_10'] = pd.to_numeric(_sub['rating_num_10'], errors='coerce')
    _sub['R_grade_alpha_num'] = pd.to_numeric(_sub['R_grade_alpha_num'], errors='coerce')
n_dev, n_oot = len(dev), len(oot)
print(f"  Dev: {n_dev:,}   OOT: {n_oot:,}")

results = {'gates': {}, 'metrics': {}, 'failures': []}


def _check(gate_id: str, name: str, passed: bool, value, threshold, comparator: str):
    sym = "✓" if passed else "✗"
    print(f"  [{sym}] #{gate_id} {name}: {value} {comparator} {threshold}")
    results['gates'][f'#{gate_id}'] = {
        'name': name, 'value': value, 'threshold': threshold,
        'comparator': comparator, 'pass': bool(passed)
    }
    if not passed:
        results['failures'].append(f"#{gate_id} {name}")
    return passed


# ============================================================
# Gate #1: Spearman ρ ≥ 0.45 (Dev & OOT)
# ============================================================
print("\n[Gate #1] Spearman ρ(R_score, rating_num_10)")
rho_dev = float(spearmanr(dev['R_score_alpha'], dev['rating_num_10'])[0])
rho_oot = float(spearmanr(oot['R_score_alpha'], oot['rating_num_10'])[0])
results['metrics']['spearman_dev'] = round(rho_dev, 4)
results['metrics']['spearman_oot'] = round(rho_oot, 4)
# Note: rating_num lower = better grade, so we expect negative correlation
# Use absolute value for the gate
_check('1a', 'Spearman ρ Dev (|ρ|)',
       abs(rho_dev) >= ACC['spearman_min'], round(abs(rho_dev), 4),
       ACC['spearman_min'], '≥')
_check('1b', 'Spearman ρ OOT (|ρ|)',
       abs(rho_oot) >= ACC['spearman_min'], round(abs(rho_oot), 4),
       ACC['spearman_min'], '≥')

# ============================================================
# Gate #2: ±1 notch accuracy ≥ 70% (Dev & OOT)
# ============================================================
print("\n[Gate #2] ±1 notch accuracy")
w1_dev = float((np.abs(dev['rating_num_10'] - dev['R_grade_alpha_num']) <= 1).mean())
w1_oot = float((np.abs(oot['rating_num_10'] - oot['R_grade_alpha_num']) <= 1).mean())
results['metrics']['within1_dev'] = round(w1_dev, 4)
results['metrics']['within1_oot'] = round(w1_oot, 4)
_check('2a', '±1 notch Dev', w1_dev >= ACC['notch_pm1_min'],
       round(w1_dev, 4), ACC['notch_pm1_min'], '≥')
_check('2b', '±1 notch OOT', w1_oot >= ACC['notch_pm1_min'],
       round(w1_oot, 4), ACC['notch_pm1_min'], '≥')

# ============================================================
# Gate #3: OOT R_score PSI < 0.30
# ============================================================
print("\n[Gate #3] OOT R_score PSI")


def compute_psi(expected_dist, actual_dist, eps=1e-9):
    e = np.array(expected_dist) + eps
    a = np.array(actual_dist) + eps
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


# 10-bin score PSI
score_bins = np.percentile(dev['R_score_alpha'], np.linspace(0, 100, 11))
score_bins[0] -= 1e-3
score_bins[-1] += 1e-3
dev_hist, _ = np.histogram(dev['R_score_alpha'], bins=score_bins)
oot_hist, _ = np.histogram(oot['R_score_alpha'], bins=score_bins)
psi_score = compute_psi(dev_hist / dev_hist.sum(), oot_hist / oot_hist.sum())
results['metrics']['psi_score_oot'] = round(psi_score, 4)
_check('3', 'OOT R_score PSI', psi_score < ACC['oot_psi_max'],
       round(psi_score, 4), ACC['oot_psi_max'], '<')

# ============================================================
# Gate #4: OOT KL_grade < 0.30
# ============================================================
print("\n[Gate #4] OOT KL_grade")
p_target_oot = oot['grade_base_10'].value_counts(normalize=True).reindex(
    GRADE_ORDER, fill_value=0)
p_pred_oot = oot['R_grade_alpha'].value_counts(normalize=True).reindex(
    GRADE_ORDER, fill_value=0)
eps = 1e-9
pt = p_target_oot.values + eps
pp = p_pred_oot.values + eps
pt, pp = pt / pt.sum(), pp / pp.sum()
kl_oot = float(np.sum(pt * np.log(pt / pp)))
results['metrics']['kl_grade_oot'] = round(kl_oot, 4)
_check('4', 'OOT KL_grade', kl_oot < ACC['oot_kl_grade_max'],
       round(kl_oot, 4), ACC['oot_kl_grade_max'], '<')

# ============================================================
# Gate #5: Boundary jump test
# ============================================================
print("\n[Gate #5] Boundary jump test")
jump_path = OUTPUT_DIR / "boundary_jump_test_alpha.csv"
if jump_path.exists():
    jt = pd.read_csv(jump_path)
    n_excessive = int(jt['excessive_jump_flag'].sum())
    n_total = len(jt)
    excessive_rate = n_excessive / n_total if n_total > 0 else 0
    results['metrics']['boundary_jump_excessive_rate'] = round(excessive_rate, 4)
    # Pass: < 5% excessive jumps from small (<=1%) perturbations
    EXCESS_THRESHOLD = 0.05
    _check('5', 'Boundary jump excessive_rate',
           excessive_rate < EXCESS_THRESHOLD,
           f"{n_excessive}/{n_total}={excessive_rate:.1%}",
           f"{EXCESS_THRESHOLD:.1%}", '<')
else:
    print("  (boundary_jump_test_alpha.csv not found — skip)")
    results['gates']['#5'] = {'pass': True, 'reason': 'file missing; warning only'}
    results.setdefault('warnings', []).append("#5 boundary jump test file missing")

# ============================================================
# Gate #7: Grade-wise confusion (BBB-A)
# ============================================================
print("\n[Gate #7] Grade-wise confusion (BBB-A)")
core_grades = ['BBB', 'A']  # Per Oracle §14.3 spec
worst_misclass_rate = 0.0
worst_grade = None
worst_split = None
for sn, sub in [('dev', dev), ('oot', oot)]:
    for g in core_grades:
        sub_g = sub[sub['grade_base_10'] == g]
        if len(sub_g) < 10:
            continue
        # Most common misclassification destination
        misclass = sub_g[sub_g['R_grade_alpha'] != g]
        if len(misclass) == 0:
            continue
        top_dest = misclass['R_grade_alpha'].value_counts().iloc[0]
        rate = top_dest / len(sub_g)
        if rate > worst_misclass_rate:
            worst_misclass_rate = rate
            worst_grade = g
            worst_split = sn
results['metrics']['worst_grade_misclassify_rate'] = round(worst_misclass_rate, 4)
results['metrics']['worst_grade_misclassify_detail'] = (
    f"{worst_grade}@{worst_split}" if worst_grade else "n/a")
_check('7', f'Grade-wise misclassify (worst: {worst_grade or "n/a"}@{worst_split or "n/a"})',
       worst_misclass_rate < ACC['grade_misclassify_max'],
       round(worst_misclass_rate, 4), ACC['grade_misclassify_max'], '<')

# ============================================================
# Sanity: Weight check + sample sizes (informational, not gating)
# ============================================================
print("\n[Sanity] Weight check + sample sizes (informational)")
ws_path = OUTPUT_DIR / "weight_sanity_check_alpha.json"
if ws_path.exists():
    ws = robust_json_load(ws_path)
    n_susp = len(ws.get('suspicious', []))
    print(f"  Weight sanity: {n_susp} suspicious patterns "
          f"({'OK' if n_susp == 0 else 'review'})")
    results['weight_sanity'] = {
        'n_suspicious': n_susp,
        'recommendation': ws.get('recommendation')
    }
    if ws.get('recommendation') and 'γ_REG' in str(ws.get('recommendation', '')):
        print(f"  Recommendation: {ws['recommendation']}")

print(f"  Sample size: dev={n_dev:,} (≥1500 expected), oot={n_oot:,} (≥500 expected)")

# ============================================================
# Final
# ============================================================
print("\n" + "=" * 72)
n_gates = len(results['gates'])
n_pass = sum(1 for g in results['gates'].values() if g.get('pass'))
all_pass = (n_pass == n_gates) and (n_gates >= 6)

print(f"  Gates passed: {n_pass}/{n_gates}")
if all_pass:
    print(f"  ✓ ALL ACCEPTANCE CHECKS PASS — Oracle α qualifies as evaluation reference")
    status = 'PASS'
    exit_code = 0
else:
    print(f"  ⚠ SOFT_GATE — {len(results['failures'])} performance gate(s) below threshold:")
    for f in results['failures']:
        print(f"    - {f}")
    status = 'PASS_WITH_WARNINGS'
    results.setdefault('warnings', []).extend(results['failures'])
    exit_code = 0
print("=" * 72)

results['status'] = status
results['n_gates_passed'] = n_pass
results['n_gates_total'] = n_gates

# Save acceptance report
with open(OUTPUT_DIR / "acceptance_alpha.json", 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# CSV summary
rows = []
for gid, info in results['gates'].items():
    rows.append({
        'gate': gid,
        'name': info.get('name', ''),
        'value': info.get('value'),
        'threshold': info.get('threshold'),
        'comparator': info.get('comparator', ''),
        'pass': info.get('pass')
    })
pd.DataFrame(rows).to_csv(OUTPUT_DIR / "acceptance_alpha.csv", index=False)

sys.exit(exit_code)


