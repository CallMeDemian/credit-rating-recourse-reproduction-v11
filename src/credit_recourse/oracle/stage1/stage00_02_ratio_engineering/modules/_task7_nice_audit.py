import os
"""
Task 7 v2: ratio_name 기반 NICE audit
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

OUT = Path(os.environ['STAGE2_OUT'])

calc = pd.read_parquet(OUT / 'engineered_financial_ratios.parquet')
nice = pd.read_parquet(os.environ['STAGE2_S1B_NICE'])
final_dict = pd.read_csv(OUT / 'financial_ratio_formula_dictionary_final.csv')

META = {'회사명','거래소코드','회계년도','year','_source_file','consolidation_basis_unknown'}
nice_cols = [c for c in nice.columns if c not in META]
for c in nice_cols:
    nice[c] = pd.to_numeric(nice[c], errors='coerce')

merged = calc.merge(nice[['거래소코드','year'] + nice_cols],
                     on=['거래소코드','year'], how='left',
                     suffixes=('','_nice'))

# ============================================================
# ratio_name → NICE 컬럼 매핑 (이름 기반)
# ============================================================
NAME_TO_NICE = {
    '매출총이익률': '매출액총이익률(IFRS)',
    '영업이익률': '매출액정상영업이익률(IFRS)',
    '정상영업이익률': '매출액정상영업이익률(IFRS)',
    '순이익률': '매출액순이익률(IFRS)',
    '매출원가/매출액': '매출원가 대 매출액비율(IFRS)',
    '영업이익/총자산': '총자본정상영업이익률(IFRS)',
    '당기순이익/총자산(ROA)': '총자본순이익률(IFRS)',
    '당기순이익/자기자본(ROE)': '자기자본순이익률(IFRS)',
    '부채비율': '부채비율(IFRS)',
    '자기자본비율': '자기자본구성비율(IFRS)',
    '차입금의존도': '차입금의존도(IFRS)',
    '비유동비율': '비유동비율(IFRS)',
    '비유동부채비율': '비유동부채비율(IFRS)',
    '유보율': '유보율(IFRS)',
    '금융비용부담률': '금융비용부담률(IFRS)',
    '이자비용/매출액': '금융비용부담률(IFRS)',
    '유동비율': '유동비율(IFRS)',
    '당좌비율': '당좌비율(IFRS)',
    '현금비율': '현금비율(IFRS)',
    '재고자산회전율': '재고자산회전률(IFRS)',
    '매출채권회전율': '매출채권회전률(IFRS)',
    '매입채무회전율': '매입채무회전률(IFRS)',
    '총자산회전율': '총자본회전률(IFRS)',
    '자기자본회전율': '자기자본회전률(IFRS)',
    '비유동자산회전율': '비유동자산회전율(IFRS)',  # NICE에 있는지 미확인
    '매출액증가율': '매출액증가율(IFRS)',
    '정상영업이익증가율': '정상영업이익증가율(IFRS)',
    '영업이익증가율': '정상영업이익증가율(IFRS)',
    '순이익증가율': '순이익증가율(IFRS)',
    '총포괄이익증가율': '총포괄이익증가율(IFRS)',
    '자기자본증가율': '자기자본증가율(IFRS)',
    '총자본증가율': '총자본증가율(IFRS)',
    '총자산증가율': '총자본증가율(IFRS)',  # 별칭 (총자본 ≈ 총자산)
    '비유동자산증가율': '비유동자산증가율(IFRS)',
    '유동자산증가율': '유동자산증가율(IFRS)',
    '재고자산증가율': '재고자산증가율(IFRS)',
    '유형자산증가율': '유형자산증가율(IFRS)',
}

NICE_AVAILABLE = set(nice_cols)

# ============================================================
# Audit
# ============================================================
audit_rows = []
for rid in calc.columns:
    if not str(rid).startswith('R'):
        continue
    meta = final_dict[final_dict['ratio_id']==rid]
    if len(meta) == 0:
        continue
    name = meta.iloc[0]['ratio_name_ko']
    cat = meta.iloc[0]['category']
    unit = meta.iloc[0]['unit_output']

    nice_col = NAME_TO_NICE.get(name)
    if not nice_col or nice_col not in NICE_AVAILABLE:
        audit_rows.append({
            'ratio_id': rid, 'ratio_name': name, 'category': cat,
            'nice_column': nice_col or '', 'matched_n': 0,
            'pearson_corr': np.nan, 'spearman_corr': np.nan,
            'median_abs_diff': np.nan, 'p95_abs_diff': np.nan,
            'scale_factor_suspected': '',
            'formula_mismatch_suspected': False,
            'audit_decision': 'no_precomputed_match',
        })
        continue

    a = merged[rid]; b = merged[nice_col]
    valid = a.notna() & b.notna()
    n_match = int(valid.sum())
    if n_match < 50:
        audit_rows.append({
            'ratio_id': rid, 'ratio_name': name, 'category': cat,
            'nice_column': nice_col, 'matched_n': n_match,
            'pearson_corr': np.nan, 'spearman_corr': np.nan,
            'median_abs_diff': np.nan, 'p95_abs_diff': np.nan,
            'scale_factor_suspected': '', 'formula_mismatch_suspected': False,
            'audit_decision': 'insufficient_overlap',
        })
        continue

    a_v = a[valid].values; b_v = b[valid].values
    # unit % 인 경우 우리 × 100 = NICE 추정
    a_scaled = a_v * 100 if unit == '%' else a_v

    # outlier 제거 후 corr (양극단 1% trim)
    diff = np.abs(a_scaled - b_v)
    p99_diff = np.percentile(diff, 99)
    mask_in = diff < p99_diff
    a_in = a_scaled[mask_in]; b_in = b_v[mask_in]

    pearson = float(np.corrcoef(a_in, b_in)[0,1])
    spearman = float(spearmanr(a_in, b_in).statistic)
    med_diff = float(np.median(diff))
    p95_diff_v = float(np.percentile(diff, 95))

    ratio_a_b = b_v / np.where(np.abs(a_scaled) > 1e-6, a_scaled, np.nan)
    median_scale = float(np.nanmedian(ratio_a_b)) if not np.isnan(ratio_a_b).all() else 1.0

    if 0.95 < median_scale < 1.05:
        scale_susp = 'scale_OK'
    else:
        scale_susp = f'scale_factor={median_scale:.2f}'

    if spearman > 0.95 and 'scale_OK' in scale_susp:
        decision = 'pass'
    elif spearman > 0.95:
        decision = 'pass_with_scale_adjustment'
    elif spearman > 0.80:
        decision = 'investigate_formula'
    else:
        decision = 'investigate_formula'

    audit_rows.append({
        'ratio_id': rid, 'ratio_name': name, 'category': cat,
        'nice_column': nice_col, 'matched_n': n_match,
        'pearson_corr': round(pearson, 4),
        'spearman_corr': round(spearman, 4),
        'median_abs_diff': round(med_diff, 4),
        'p95_abs_diff': round(p95_diff_v, 4),
        'scale_factor_suspected': scale_susp,
        'formula_mismatch_suspected': bool(spearman < 0.85),
        'audit_decision': decision,
    })

audit_df = pd.DataFrame(audit_rows)
audit_df.to_csv(OUT / 'ratio_audit_against_precomputed_panel.csv',
                 index=False, encoding='utf-8-sig')

print(f"\nAudit results:")
print(audit_df['audit_decision'].value_counts().to_string())

print(f"\n매칭된 ratio (NICE에 대응 있음):")
matched = audit_df[audit_df['audit_decision'] != 'no_precomputed_match']
print(matched[['ratio_id','ratio_name','nice_column','matched_n',
                'spearman_corr','median_abs_diff','scale_factor_suspected',
                'audit_decision']].to_string())
