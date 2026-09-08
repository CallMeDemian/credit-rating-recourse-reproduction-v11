import os
"""
Task 4: lag support coverage 진단
==================================
Step 1: Stage 1B panel 내에서의 lag coverage (등급이 t-1에도 있나)
Step 2: raw 재무제표에서 t-1 row 추출 가능성
Step 3: lag_support_financial_items.parquet 생성 (필요 시)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time, gc

OUT = Path(os.environ['STAGE2_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(os.environ['STAGE2_RAW'])

import re

def _norm_code(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^0-9]", "", s)
    return s.zfill(6) if s else ""

def _read_excel_with_fallback(fp, **kwargs):
    errors=[]
    for engine in ['calamine', 'openpyxl', None]:
        try:
            if engine is None:
                return pd.read_excel(fp, **kwargs)
            return pd.read_excel(fp, engine=engine, **kwargs)
        except Exception as exc:
            errors.append(f"{engine or 'default'}:{type(exc).__name__}:{exc}")
    raise RuntimeError(f"Failed to read Excel file {fp}: {' | '.join(errors)}")


def _norm_year(x):
    if pd.isna(x):
        return -1
    try:
        return int(float(x))
    except Exception:
        m = re.search(r"(19|20)\d{2}", str(x))
        return int(m.group(0)) if m else -1

# ============================================================
# Step 1: Stage 1B panel 내 lag coverage
# ============================================================
panel = pd.read_parquet(os.environ['STAGE2_S1B_PANEL'])
panel = panel.copy()
panel['_key_code'] = panel['거래소코드'].map(_norm_code)
panel['_key_year'] = panel['year'].map(_norm_year)
elig = panel[panel['eligible_for_stage2']].copy()
print(f"[1] Stage 2 eligible: {len(elig):,} firm-year")

# 각 (firm, year)에 대해 (firm, year-1)이 panel_v1에 있는지
elig_keys = set(zip(elig['_key_code'], elig['_key_year']))
panel_keys = set(zip(panel['_key_code'], panel['_key_year']))

elig['has_prior_in_panel'] = elig.apply(
    lambda r: (r['_key_code'], r['_key_year'] - 1) in panel_keys, axis=1)
elig['has_prior_in_eligible'] = elig.apply(
    lambda r: (r['_key_code'], r['_key_year'] - 1) in elig_keys, axis=1)

n_with_prior_panel = elig['has_prior_in_panel'].sum()
n_with_prior_elig = elig['has_prior_in_eligible'].sum()
print(f"    t-1 in panel_v1 (rated):    {n_with_prior_panel:,} ({n_with_prior_panel/len(elig):.1%})")
print(f"    t-1 in eligible (Stage 2):  {n_with_prior_elig:,} ({n_with_prior_elig/len(elig):.1%})")

# ============================================================
# Step 2: raw에서 t-1 추출 가능성 (key만 빠르게)
# ============================================================
# raw 재무제표 파일에서 (거래소코드, year)만 빠르게 추출 (calamine + usecols)
PATTERNS = ['재무상태표', '손익계산서', '현금흐름표', '자본변동표',
            '이익잉여금처분계산서', '재무비율']
files = []
for f in sorted(PROJECT.glob('*.xlsx')):
    if any(p in f.name for p in PATTERNS) and ('코스피' in f.name or '코스닥' in f.name or '코넥스' in f.name):
        files.append(f)
print(f"\n[2] Raw 파일 {len(files)}개에서 (firm, year) 키 추출 중...")

raw_keys_by_stmt = {p: set() for p in PATTERNS}

for fp in files:
    t0 = time.time()
    stmt = next((p for p in PATTERNS if p in fp.name), 'unknown')
    try:
        df = _read_excel_with_fallback(fp, usecols=['거래소코드', '회계년도'])
    except Exception as e:
        print(f"    ERROR {fp.name}: {e}")
        continue
    df['_key_year'] = df['회계년도'].map(_norm_year)
    df['_key_code'] = df['거래소코드'].map(_norm_code)
    df = df[(df['_key_year'] > 0) & (df['_key_code'] != "")]
    keys = set(zip(df['_key_code'].values, df['_key_year'].values))
    raw_keys_by_stmt[stmt] |= keys
    print(f"    {fp.stem[:50]:<55s} keys={len(keys):>6,}  ({(time.time()-t0):.1f}s)")
    del df; gc.collect()

# 통합 (모든 statement에 동일 firm-year가 있을 거지만 안전하게)
raw_all_keys = set()
for k in raw_keys_by_stmt.values():
    raw_all_keys |= k
print(f"\n    Raw 파일 전체 unique (firm,year): {len(raw_all_keys):,}")
if len(files) > 0 and len(raw_all_keys) == 0:
    raise RuntimeError("Task4 lag extract found raw Excel files but extracted zero firm-year keys; refusing silent pass")

# ============================================================
# Step 3: 각 eligible firm-year의 t-1이 raw에 있는지
# ============================================================
elig['has_prior_in_raw'] = elig.apply(
    lambda r: (r['_key_code'], r['_key_year'] - 1) in raw_all_keys, axis=1)
n_with_prior_raw = elig['has_prior_in_raw'].sum()
print(f"\n[3] t-1 in raw (재무제표 raw 파일): "
      f"{n_with_prior_raw:,} ({n_with_prior_raw/len(elig):.1%})")

# ============================================================
# Step 4: 평균/성장 계산 항목별 lag coverage report
# ============================================================
LAG_ITEMS = [
    '평균총자산', '평균자기자본', '평균부채총계', '평균재고자산', '평균매출채권',
    '평균유동자산', '평균비유동자산', '평균유형자산', '평균매입채무', '평균순영업운전자본',
    '평균자본금', '평균현금성자산',
    '매출액증가율', '자기자본증가율', '정상영업이익증가율', '순이익증가율',
    '총포괄이익증가율', '총자산증가율', '총차입금증가율',
    'EBITDA증가율', 'OCF증가율', 'FCF증가율', '순차입금증가율',
    '영업이익증가율', '비유동자산증가율', '유형자산증가율',
]

# 모든 lag-required item에 대해 동일한 coverage (same firm-year × t-1 존재 여부)
n_current = len(elig)
prior_in_stage1b = int(n_with_prior_panel)
prior_in_raw = int(n_with_prior_raw)
needs_extraction = prior_in_raw > prior_in_stage1b
rate_stage1b = round(prior_in_stage1b / n_current, 4)
rate_raw = round(prior_in_raw / n_current, 4)

# decision rule
if rate_stage1b >= 0.85:
    decision = 'use_stage1b_only'
elif rate_raw >= 0.85:
    decision = 'extract_from_raw'
elif rate_raw >= 0.70:
    decision = 'extract_from_raw_with_caveat'
else:
    decision = 'lag_coverage_insufficient'

rows = []
for item in LAG_ITEMS:
    rows.append({
        'base_item': item,
        'current_year_available_n': n_current,
        'prior_year_available_in_stage1b_n': prior_in_stage1b,
        'prior_year_available_in_raw_n': prior_in_raw,
        'lag_coverage_stage1b_rate': rate_stage1b,
        'lag_coverage_raw_rate': rate_raw,
        'needs_prior_year_support_extraction': needs_extraction,
        'decision': decision,
    })

cov_df = pd.DataFrame(rows)
cov_df.to_csv(OUT / 'lag_support_coverage_report.csv', index=False, encoding='utf-8-sig')

print(f"\n[Coverage 결과]")
print(f"  Stage 1B 내부 lag coverage: {rate_stage1b:.1%}")
print(f"  Raw 파일 기준 lag coverage: {rate_raw:.1%}")
print(f"  Raw에서 추가 확보 가능: {prior_in_raw - prior_in_stage1b:,} firm-year "
      f"({(prior_in_raw - prior_in_stage1b)/n_current:.1%})")
print(f"  결정: {decision}")

# Save raw_all_keys for later Step 5 (extraction)
import pickle
Path(str(Path(os.environ['STAGE2_OUT']).parent / 'tmp')).mkdir(parents=True, exist_ok=True)
with open(str(Path(os.environ['STAGE2_OUT']).parent / 'tmp' / 'raw_all_keys.pkl'), 'wb') as f:
    pickle.dump(raw_all_keys, f)
print(f"\n→ lag_support_coverage_report.csv ({len(cov_df)}행)")
print(f"→ raw_all_keys.pkl 저장 (Task 4 Step 5에서 사용)")
