import os
"""
Task 8: ratio_quality_report.csv + candidate_ratio_pool_by_item.csv
====================================================================
Oracle §4.0.4 quality filter:
- 결측률 < 30%
- 분모 0 발생률 < 5%
- inf 발생률 < 1%
- P99 outlier 정상범위 100배 이상이면 flag
- 동일값 비율 ≥ 95% → 정보량 부족
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

OUT = Path(os.environ['STAGE2_OUT'])

calc = pd.read_parquet(OUT / 'engineered_financial_ratios.parquet')
final_dict = pd.read_csv(OUT / 'financial_ratio_formula_dictionary_final.csv')

# Stage 2 may calculate duplicate/alias ratios for auditability, but downstream
# variable selection must not treat them as independent candidates. The alias
# metadata is created in _task2_candidates.py and propagated here.
for _col, _default in {
    'duplicate_alias_of': '',
    'duplicate_alias_type': '',
    'duplicate_alias_reason': '',
    'stage3_exclude': False,
}.items():
    if _col not in final_dict.columns:
        final_dict[_col] = _default

final_dict['stage3_exclude'] = (
    final_dict['stage3_exclude'].astype(str).str.lower().isin(['true', '1', 'yes', 'y'])
)

calc_log = pd.read_csv(OUT / 'ratio_calculation_log.csv')

n_total = len(calc)
print(f"Total firm-year: {n_total:,}\n")

# ============================================================
# 1. quality 메트릭 계산
# ============================================================
quality_rows = []
exclusion_log = []

for _, meta in final_dict.iterrows():
    rid = meta['ratio_id']
    if rid not in calc.columns:
        continue
    s = calc[rid]

    n_missing = int(s.isna().sum())
    n_inf = int(np.isinf(s).sum())
    n_valid = n_total - n_missing
    miss_rate = n_missing / n_total
    inf_rate = n_inf / n_total

    # log에서 zero/neg denom count
    log_match = calc_log[calc_log['ratio_id']==rid]
    n_zero_denom = int(log_match.iloc[0]['zero_denom_count']) if len(log_match) > 0 else 0
    n_neg_denom = int(log_match.iloc[0]['neg_denom_count']) if len(log_match) > 0 else 0
    zero_denom_rate = n_zero_denom / n_total

    # 통계
    if n_valid > 0:
        v = s.dropna()
        v_finite = v[np.isfinite(v)]
        if len(v_finite) > 0:
            p1 = float(np.percentile(v_finite, 1))
            p50 = float(np.percentile(v_finite, 50))
            p99 = float(np.percentile(v_finite, 99))
            v_min = float(v_finite.min())
            v_max = float(v_finite.max())
            v_std = float(v_finite.std())
            # 동일값 비율
            mode_count = int(v_finite.value_counts().iloc[0]) if len(v_finite) > 0 else 0
            same_value_rate = mode_count / len(v_finite)
        else:
            p1 = p50 = p99 = v_min = v_max = v_std = np.nan
            same_value_rate = 0
    else:
        p1 = p50 = p99 = v_min = v_max = v_std = np.nan
        same_value_rate = 0

    # outlier flag (P99 / |P50| 비율로 판단)
    outlier_flag = False
    if pd.notna(p99) and pd.notna(p50) and abs(p50) > 1e-6:
        if abs(p99 / p50) > 100:
            outlier_flag = True

    # NaN breakdown
    nan_reasons = []
    if n_zero_denom > 0:
        nan_reasons.append(f'zero_denom={n_zero_denom}')
    if n_neg_denom > 0:
        nan_reasons.append(f'neg_denom={n_neg_denom}')
    other_nan = n_missing - n_zero_denom - n_neg_denom
    if other_nan > 0:
        nan_reasons.append(f'data_missing={other_nan}')

    # Quality decision
    quality_pass = True
    fail_reasons = []
    if miss_rate >= 0.30:
        quality_pass = False
        fail_reasons.append(f'high_missing={miss_rate:.1%}')
    if zero_denom_rate >= 0.05:
        quality_pass = False
        fail_reasons.append(f'zero_denom_high={zero_denom_rate:.1%}')
    if inf_rate >= 0.01:
        quality_pass = False
        fail_reasons.append(f'inf_high={inf_rate:.1%}')
    if same_value_rate >= 0.95:
        quality_pass = False
        fail_reasons.append(f'low_variance={same_value_rate:.1%}')

    quality_rows.append({
        'ratio_id': rid,
        'ratio_name': meta['ratio_name_ko'],
        'category': meta['category'],
        'priority': meta.get('priority', ''),
        'n_total': n_total,
        'n_valid': n_valid,
        'n_missing': n_missing,
        'missing_rate': round(miss_rate, 4),
        'n_zero_denom': n_zero_denom,
        'zero_denom_rate': round(zero_denom_rate, 4),
        'n_neg_denom': n_neg_denom,
        'n_inf': n_inf,
        'inf_rate': round(inf_rate, 4),
        'p1': round(p1, 4) if pd.notna(p1) else np.nan,
        'median': round(p50, 4) if pd.notna(p50) else np.nan,
        'p99': round(p99, 4) if pd.notna(p99) else np.nan,
        'min': round(v_min, 4) if pd.notna(v_min) else np.nan,
        'max': round(v_max, 4) if pd.notna(v_max) else np.nan,
        'std': round(v_std, 4) if pd.notna(v_std) else np.nan,
        'same_value_rate': round(same_value_rate, 4),
        'outlier_flag': outlier_flag,
        'nan_breakdown': '|'.join(nan_reasons) if nan_reasons else '',
        'quality_pass': quality_pass,
        'fail_reasons': '|'.join(fail_reasons) if fail_reasons else '',
        'duplicate_alias_of': meta.get('duplicate_alias_of', ''),
        'duplicate_alias_type': meta.get('duplicate_alias_type', ''),
        'duplicate_alias_reason': meta.get('duplicate_alias_reason', ''),
        'stage3_exclude': bool(meta.get('stage3_exclude', False)),
        # selected_eligible is an explicit downstream contract: quality-pass but
        # duplicate aliases are excluded from Stage 3 selection.
        'selected_eligible': bool(quality_pass and not bool(meta.get('stage3_exclude', False))),
    })

    if not quality_pass:
        exclusion_log.append({
            'ratio_id': rid, 'ratio_name': meta['ratio_name_ko'],
            'category': meta['category'],
            'reason': '|'.join(fail_reasons),
            'missing_rate': round(miss_rate, 4),
            'zero_denom_rate': round(zero_denom_rate, 4),
        })

quality_df = pd.DataFrame(quality_rows)
quality_df.to_csv(OUT / 'ratio_quality_report.csv', index=False, encoding='utf-8-sig')

alias_df = quality_df[quality_df.get('stage3_exclude', False).astype(bool)].copy()
if len(alias_df) > 0:
    alias_df[[
        'ratio_id', 'ratio_name', 'category', 'quality_pass', 'selected_eligible',
        'duplicate_alias_of', 'duplicate_alias_type', 'duplicate_alias_reason'
    ]].to_csv(OUT / 'duplicate_ratio_alias_quality_log.csv', index=False, encoding='utf-8-sig')

# ------------------------------------------------------------------
# Final Stage 2 R-code master metadata
# ------------------------------------------------------------------
# This file is the downstream contract for Stage 3 and the simulator: it
# combines formula/account-code definition, quality metrics, and alias policy.
_quality_join_cols = [
    'ratio_id', 'n_valid', 'missing_rate', 'zero_denom_rate', 'inf_rate',
    'p1', 'median', 'p99', 'same_value_rate', 'outlier_flag', 'quality_pass',
    'fail_reasons', 'selected_eligible'
]
_quality_join_cols = [c for c in _quality_join_cols if c in quality_df.columns]
ratio_code_master_final = final_dict.merge(
    quality_df[_quality_join_cols], on='ratio_id', how='left', suffixes=('', '_quality')
)

if 'canonical_ratio_id' not in ratio_code_master_final.columns:
    ratio_code_master_final['canonical_ratio_id'] = ratio_code_master_final.apply(
        lambda r: r['duplicate_alias_of'] if str(r.get('duplicate_alias_of', '')).strip() else r['ratio_id'],
        axis=1,
    )
if 'is_canonical_ratio' not in ratio_code_master_final.columns:
    ratio_code_master_final['is_canonical_ratio'] = ratio_code_master_final['duplicate_alias_of'].astype(str).str.len().eq(0)

ratio_code_master_final.to_csv(
    OUT / 'ratio_code_master_stage2_final.csv', index=False, encoding='utf-8-sig'
)
ratio_code_master_final.to_json(
    OUT / 'ratio_code_master_stage2_final.json', orient='records', force_ascii=False, indent=2
)

_alias_final = ratio_code_master_final[
    ratio_code_master_final['duplicate_alias_of'].astype(str).str.strip().ne('')
].copy()
_alias_cols = [
    'ratio_id', 'ratio_name_ko', 'category', 'canonical_ratio_id',
    'duplicate_alias_type', 'duplicate_alias_reason', 'stage3_exclude',
    'quality_pass', 'selected_eligible'
]
_alias_cols = [c for c in _alias_cols if c in _alias_final.columns]
_alias_final[_alias_cols].rename(columns={'ratio_id': 'alias_ratio_id'}).to_csv(
    OUT / 'duplicate_ratio_alias_master_final.csv', index=False, encoding='utf-8-sig'
)
with open(OUT / 'simulator_ratio_alias_map.json', 'w', encoding='utf-8') as f:
    json.dump(
        {
            str(r['ratio_id']): str(r['canonical_ratio_id'])
            for _, r in _alias_final.iterrows()
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

excl_df = pd.DataFrame(exclusion_log)
excl_df.to_csv(OUT / 'pre_exclusion_log.csv', index=False, encoding='utf-8-sig')

print(f"Quality report: {len(quality_df)} ratio")
print(f"PASS: {quality_df['quality_pass'].sum()}")
print(f"FAIL: {(~quality_df['quality_pass']).sum()}")
print(f"Stage 3 duplicate aliases excluded: {int(quality_df['stage3_exclude'].sum())}")

print(f"\n[Quality fail 사유별]")
fail_only = quality_df[~quality_df['quality_pass']]
for _, r in fail_only.iterrows():
    print(f"  {r['ratio_id']} [{r['category']}] {r['ratio_name']}: {r['fail_reasons']}")

# ============================================================
# 2. candidate_ratio_pool_by_item.csv
# ============================================================
print(f"\n[평가항목별 quality pass]")
passed = quality_df[quality_df['quality_pass']]
pool_rows = []
for cat, grp in passed.groupby('category'):
    sorted_grp = grp.sort_values(['priority','missing_rate'])
    print(f"  {cat:<10s}: {len(grp):>3d}개")
    for _, r in sorted_grp.iterrows():
        pool_rows.append({
            'category': cat,
            'ratio_id': r['ratio_id'],
            'ratio_name': r['ratio_name'],
            'priority': r['priority'],
            'missing_rate': r['missing_rate'],
            'median': r['median'],
            'p1_p99_range': f"{r['p1']}, {r['p99']}",
            'outlier_flag': r['outlier_flag'],
            'duplicate_alias_of': r.get('duplicate_alias_of', ''),
            'duplicate_alias_type': r.get('duplicate_alias_type', ''),
            'duplicate_alias_reason': r.get('duplicate_alias_reason', ''),
            'stage3_exclude': bool(r.get('stage3_exclude', False)),
            'selected_eligible': bool(r.get('selected_eligible', True)),
            'rank_in_category': 0,  # 채움
        })

# rank_in_category 부여
pool_df = pd.DataFrame(pool_rows)
pool_df['rank_in_category'] = pool_df.groupby('category').cumcount() + 1
pool_df.to_csv(OUT / 'candidate_ratio_pool_by_item.csv', index=False, encoding='utf-8-sig')

# ============================================================
# 3. stage2_summary.md
# ============================================================
n_pass = int(quality_df['quality_pass'].sum())
n_fail = int((~quality_df['quality_pass']).sum())

cat_pass = quality_df[quality_df['quality_pass']].groupby('category').size().to_dict()
cat_total = quality_df.groupby('category').size().to_dict()

# Stage 3 진입 요건: 평가항목별 ≥ 3개
target_cats = ['수익성','안정성','부채상환능력','유동성','활동성','성장성']
all_cats_meet = all(cat_pass.get(c, 0) >= 3 for c in target_cats)

verdict = "PASS" if all_cats_meet else "CONDITIONAL PASS" if sum(cat_pass.get(c, 0) >= 3 for c in target_cats) >= 5 else "FAIL"

summary = f"""# Stage 2 Summary

**Date**: 2026-04-30
**Stage**: 2A 사전검증 + 2B ratio 산출 + 2C quality filter
**Verdict**: {verdict}

## 1. 산출 ratio 현황

| Step | 단계 | 항목 | 결과 |
|---|---|---|---|
| Task 1 | Inventory | unique items | 6,423 (결측률 <30% = 1,997) |
| Task 2 | Dictionary draft | candidates 221개 + Oracle | 221 ratio |
| Task 3 | Availability | direct/derived 가능 | 174 ratio |
| Task 4 | Lag coverage | t-1 raw 추출 후 | 99.7% (4,734/4,746) |
| Task 5 | Growth policy | prior > 0 정책 | 정책 확정 |
| Task 6 | Ratio 계산 | 실제 산출 | **{len(final_dict)} ratio 산출** |
| Task 7 | NICE Audit | pass+scale_adj | 22 / 32 매칭 가능 |
| Task 8 | Quality filter | quality pass | **{n_pass} ratio** |
| Metadata | R-code master | canonical + alias map | ratio_code_master_stage2_final.csv / simulator_ratio_alias_map.json |

## 2. 평가항목별 후보 (Quality Pass)

| 평가항목 | 산출 | 통과 | 통과율 | 최소 3개 충족 |
|---|---|---|---|---|
"""
for cat in ['수익성','안정성','부채상환능력','유동성','활동성','성장성','기타/복합']:
    total = cat_total.get(cat, 0)
    p = cat_pass.get(cat, 0)
    pct = f"{p/total*100:.0f}%" if total > 0 else "-"
    meets = "✓" if p >= 3 else "❌"
    summary += f"| {cat} | {total} | **{p}** | {pct} | {meets} |\n"

summary += f"""
## 3. Stage 3 진입 가능 여부

- **평가항목별 최소 3개 후보 충족**: {'YES' if all_cats_meet else 'NO'}
- **6 평가항목 중 충족**: {sum(cat_pass.get(c, 0) >= 3 for c in target_cats)} / 6

## 4. 주요 이슈

### Quality fail로 제외된 ratio ({n_fail}개)
"""
for _, r in fail_only.iterrows():
    summary += f"- `{r['ratio_id']}` [{r['category']}] {r['ratio_name']}: {r['fail_reasons']}\n"

summary += f"""

### Manual mapping 필요 / 추후 보완
- 3년 lag (`_t-3`) 7개 ratio: CAGR 3년 등은 데이터 부재로 미산출. Stage 1B 시 t-3 raw 같이 추출하거나 Stage 3에서 제외.
- NICE audit `investigate_formula` 10개: 평균잔액 사용 여부·적자처리 정책 차이 가능성. Stage 3 변수 선택 시 NICE 정의로 cross-validate 권장.

### 직접 계산 불가능하여 제외된 후보
- `availability_status = ambiguous`: 32개 (배당성향, ROA 3개년, 영업이익률 3개년 등)
- `availability_status = unknown`: 15개
- `compound_expression`: 3개 (DSO+DIO-DPO 등)

## 5. Stage 3 권고사항

- **Main pool**: `candidate_ratio_pool_by_item.csv`의 {n_pass}개 ratio
- **Robustness 검증**: NICE 재무비율 panel 22개 (scale-adjusted) 사용 가능
- **Variable selection 시 주의**:
  - 평균분모 ratio는 lag coverage 99.7%이므로 12 firm-year 결측 발생 — 처리 정책 필요
  - 성장률 ratio는 prior ≤ 0 케이스에서 NaN — turnaround dummy 별도 고려
  - 대량의 안정성/부채상환능력 ratio들이 동일 분모(자기자본·총자산) 사용 → 다중공선성 클러스터링 필수

---
"""

with open(OUT / 'stage2_summary.md', 'w', encoding='utf-8') as f:
    f.write(summary)

print(f"\n[Verdict: {verdict}]")
print(f"6 평가항목 중 ≥3개 충족: {sum(cat_pass.get(c, 0) >= 3 for c in target_cats)} / 6")
print(f"\n→ ratio_quality_report.csv ({len(quality_df)} ratio)")
print(f"→ candidate_ratio_pool_by_item.csv ({len(pool_df)} ratio)")
print(f"→ pre_exclusion_log.csv ({len(excl_df)} ratio)")
print(f"→ stage2_summary.md")
