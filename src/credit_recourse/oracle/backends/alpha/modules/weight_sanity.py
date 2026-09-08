#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Weight sanity check for Oracle α.

After joint KL optimization, verify that the weight changes from prior → optimized
follow expected credit-rating logic:
  - Strong-signal items (수익성/안정성/부채상환능력) should NOT lose much weight
  - Weak-signal items (성장성=R185, 신뢰도=financial_data_completeness, 산업위험=
    industry_bad_grade_share which has weak ρ) should NOT spike unexpectedly
  - Direction encoding consistency must be preserved
"""
import json


# Spearman ρ thresholds (absolute value) for "strong" vs "weak" signal
STRONG_RHO_THRESHOLD = 0.30
WEAK_RHO_THRESHOLD = 0.20

# Suspicious weight increase threshold for weak items (relative)
WEAK_SPIKE_THRESHOLD_PCT = 50.0  # +50% over prior is suspicious for weak items

# Suspicious weight collapse threshold for strong items (relative)
STRONG_COLLAPSE_THRESHOLD_PCT = -40.0  # -40% from prior is suspicious for strong items


def categorize_signal(spearman_rho: float) -> str:
    abs_rho = abs(spearman_rho)
    if abs_rho >= STRONG_RHO_THRESHOLD:
        return 'strong'
    elif abs_rho < WEAK_RHO_THRESHOLD:
        return 'weak'
    else:
        return 'medium'


def run_sanity_check(prior_weights: dict, optimized_weights: dict,
                     fin_ids: list, nonfin_ids: list,
                     selected_vars: list, direction_encoding: dict) -> dict:
    """
    Returns dict with:
      - direction_consistent: bool — direction encoding still valid
      - per_variable: list of dict with prior/optimized/delta/category/flag
      - suspicious: list of human-readable warnings
      - block_sums: financial/nonfinancial sum sanity
      - recommendation: optional, e.g. "increase γ_REG to 0.5"
    """
    results = {
        'direction_consistent': True,
        'per_variable': [],
        'suspicious': [],
        'block_sums': {},
        'recommendation': None,
    }

    rho_map = {v['variable_id']: v['spearman_rho'] for v in selected_vars}

    for vid in fin_ids + nonfin_ids:
        prior = prior_weights[vid]
        opt = optimized_weights[vid]
        delta = opt - prior
        delta_pct = (delta / prior * 100) if prior > 0 else 0.0
        rho = rho_map.get(vid, 0.0)
        category = categorize_signal(rho)
        block = 'financial' if vid in fin_ids else 'nonfinancial'

        flags = []
        # Weak items spiking up?
        if category == 'weak' and delta_pct > WEAK_SPIKE_THRESHOLD_PCT:
            flags.append('weak_item_spike')
            results['suspicious'].append(
                f"{vid} (weak |ρ|={abs(rho):.3f}) gained {delta_pct:+.1f}% weight "
                f"({prior:.4f} → {opt:.4f}). Possible Dev fitting artifact."
            )
        # Strong items collapsing?
        if category == 'strong' and delta_pct < STRONG_COLLAPSE_THRESHOLD_PCT:
            flags.append('strong_item_collapse')
            results['suspicious'].append(
                f"{vid} (strong |ρ|={abs(rho):.3f}) lost {delta_pct:+.1f}% weight "
                f"({prior:.4f} → {opt:.4f}). Suspicious decay of strong signal."
            )

        results['per_variable'].append({
            'variable_id': vid,
            'block': block,
            'spearman_rho': round(rho, 4),
            'signal_category': category,
            'weight_prior': round(prior, 6),
            'weight_optimized': round(opt, 6),
            'delta': round(delta, 6),
            'delta_pct': round(delta_pct, 2),
            'flags': flags
        })

    # Block sums
    fin_sum = sum(optimized_weights[v] for v in fin_ids)
    nf_sum = sum(optimized_weights[v] for v in nonfin_ids)
    results['block_sums'] = {
        'financial_sum': round(fin_sum, 6),
        'nonfinancial_sum': round(nf_sum, 6)
    }
    if abs(fin_sum - 1.0) > 0.01:
        results['suspicious'].append(
            f"Financial weight sum {fin_sum:.4f} ≠ 1.0 (should be normalized within block)"
        )
    if abs(nf_sum - 1.0) > 0.01:
        results['suspicious'].append(
            f"Nonfinancial weight sum {nf_sum:.4f} ≠ 1.0"
        )

    # Direction consistency: weight optimization shouldn't flip directions
    # (direction encoding is applied before weights, so this is informational)
    for vid, info in direction_encoding.items():
        if vid not in optimized_weights:
            continue
        if 'direction' not in info:
            results['direction_consistent'] = False

    # Recommendation
    n_weak_spike = sum(1 for w in results['per_variable']
                       if 'weak_item_spike' in w['flags'])
    n_strong_collapse = sum(1 for w in results['per_variable']
                            if 'strong_item_collapse' in w['flags'])
    if n_weak_spike >= 2 or n_strong_collapse >= 2:
        results['recommendation'] = (
            'Multiple suspicious weight changes detected. '
            'Recommend rerunning joint optimization with γ_REG ≥ 0.5 '
            '(stronger prior penalty) to anchor weights closer to Stage 3 prior.'
        )
    elif results['suspicious']:
        results['recommendation'] = (
            'Some suspicious patterns detected. Review per_variable detail and '
            'consider γ_REG sensitivity check.'
        )
    else:
        results['recommendation'] = (
            'Weight changes appear sensible. No γ_REG adjustment needed.'
        )

    return results


if __name__ == '__main__':
    # Standalone test/example usage
    import sys
    if len(sys.argv) < 4:
        print("Usage: python weight_sanity.py "
              "<prior_weights.json> <optimized_weights.json> "
              "<selected_variables_v2.json> <direction_encoding_v2.json>")
        sys.exit(1)
    with open(sys.argv[1]) as f:
        prior = json.load(f)
    with open(sys.argv[2]) as f:
        opt = json.load(f)
    with open(sys.argv[3], encoding='utf-8') as f:
        sel = json.load(f)
    with open(sys.argv[4], encoding='utf-8') as f:
        dirs = json.load(f)

    fin = [v['variable_id'] for v in sel if v['source'] == 'financial']
    nf = [v['variable_id'] for v in sel if v['source'] == 'nonfinancial']

    out = run_sanity_check(prior, opt, fin, nf, sel, dirs)
    print(json.dumps(out, indent=2, ensure_ascii=False))
