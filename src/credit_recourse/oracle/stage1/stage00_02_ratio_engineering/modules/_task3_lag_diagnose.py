import os
"""
Task 3: ratio_item_availability_report.csv
============================================
Dictionary draft에서 ratio별 availability 정보 추출하여 별도 report 생성.
"""
import pandas as pd
from pathlib import Path

OUT = Path(os.environ['STAGE2_OUT'])

dict_df = pd.read_csv(OUT / 'financial_ratio_formula_dictionary_draft.csv')
inv = pd.read_csv(OUT / 'statement_item_inventory.csv')

# inventory: item_code → non_missing_rate lookup
inv_rate_lookup = {}
for _, r in inv.iterrows():
    if pd.notna(r['item_code']) and r['item_code']:
        inv_rate_lookup[r['item_code']] = r['non_missing_rate']

def expand_components(item_code_str):
    """'+' 구분 다중 코드를 list로 분할"""
    if not item_code_str or pd.isna(item_code_str):
        return []
    if item_code_str == 'DERIVED':
        return []
    return [c.strip() for c in str(item_code_str).split('+') if c.strip()]

rows = []
for _, r in dict_df.iterrows():
    num_codes = expand_components(r['numerator_item_code'])
    den_codes = expand_components(r['denominator_item_code'])
    all_required = list(set(num_codes + den_codes))

    # availability
    found = [c for c in all_required if c in inv_rate_lookup]
    missing = [c for c in all_required if c not in inv_rate_lookup]
    all_found = (len(missing) == 0) and len(all_required) > 0

    # 가용 firm-year count = min of all required items' non_missing_count (대략)
    if all_found and len(all_required) > 0:
        # 분자/분모 모두 non-missing인 row 수의 보수적 추정 = min(non_missing_rate) * 4746
        min_rate = min(inv_rate_lookup[c] for c in all_required)
        est_available = int(min_rate * 4746)
        est_rate = round(min_rate, 4)
    else:
        est_available = 0
        est_rate = 0.0

    # eligible/fallback/decision
    avail_status = r['availability_status']
    if avail_status == 'available_direct':
        eligible = True; fallback = False
        decision = 'use_direct'
    elif avail_status == 'available_with_lag':
        eligible = True; fallback = False
        decision = 'use_direct'  # lag 처리 후 direct 계산
    elif avail_status == 'available_derived':
        eligible = True; fallback = False
        decision = 'use_derived_component'
    elif avail_status == 'ambiguous':
        eligible = False; fallback = True
        decision = 'use_precomputed_ratio_for_validation_only'
    elif avail_status == 'compound_expression':
        eligible = True; fallback = False
        decision = 'use_derived_component'  # compound → 별도 계산
    elif avail_status == 'partial_data':
        eligible = False; fallback = True
        decision = 'needs_manual_mapping'
    else:  # unknown / missing_term
        eligible = False; fallback = False
        decision = 'exclude_candidate'

    # ambiguous_items 식별
    ambiguous_items = []
    if r['numerator_status'] == 'ambiguous':
        ambiguous_items.append(f"num={r['numerator_term_orig']}")
    if r['denominator_status'] == 'ambiguous':
        ambiguous_items.append(f"den={r['denominator_term_orig']}")

    # fallback source: NICE 재무비율 panel에 동일/유사 ratio 있는지
    # 여기선 단순히 non-direct ratio들에 대해 'NICE_재무비율_panel_for_audit'
    fallback_source = 'NICE_재무비율_panel_for_audit' if fallback else ''

    rows.append({
        'ratio_id': r['ratio_id'],
        'ratio_name_ko': r['ratio_name_ko'],
        'category': r['category'],
        'required_items': '|'.join(all_required) if all_required else '',
        'all_items_found': all_found,
        'missing_items': '|'.join(missing) if missing else '',
        'ambiguous_items': '|'.join(ambiguous_items) if ambiguous_items else '',
        'available_firm_year_count': est_available,
        'available_firm_year_rate': est_rate,
        'eligible_for_direct_calculation': eligible,
        'fallback_possible': fallback,
        'fallback_source': fallback_source,
        'decision': decision,
        'availability_status': avail_status,
        'requires_lag': r['requires_lag'],
        'requires_average_denominator': r['requires_average_denominator'],
        'requires_growth_base': r['requires_growth_base'],
        'priority': r['priority'],
    })

avail_df = pd.DataFrame(rows)
avail_df.to_csv(OUT / 'ratio_item_availability_report.csv', index=False, encoding='utf-8-sig')

print(f"Availability report: {len(avail_df)}개 ratio\n")
print("Decision 분포:")
print(avail_df['decision'].value_counts().to_string())
print(f"\nDirect calculation eligible: {avail_df['eligible_for_direct_calculation'].sum()}")
print(f"Fallback possible: {avail_df['fallback_possible'].sum()}")
print(f"Excluded: {(avail_df['decision']=='exclude_candidate').sum()}")
print(f"\nCategory별 eligible:")
elig_cat = avail_df[avail_df['eligible_for_direct_calculation']].groupby('category').size()
total_cat = avail_df.groupby('category').size()
print(pd.DataFrame({'eligible': elig_cat, 'total': total_cat,
                     'pct': (elig_cat/total_cat*100).round(1)}).to_string())
