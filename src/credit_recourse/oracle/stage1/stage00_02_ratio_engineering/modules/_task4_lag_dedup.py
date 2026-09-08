import os

K_CODE_COL = "\uac70\ub798\uc18c\ucf54\ub4dc"
K_YEAR_COL = "\ud68c\uacc4\ub144\ub3c4"
K_MARKET_COL = "\uc2dc\uc7a5"
"""
Task 4 Step 5: lag support 추출
=================================
4,746 eligible firm-year의 t-1 (= 4,746 firm, year-1)을 raw에서 추출하여
재무비율 계산 보조 데이터로 사용. 신용등급 학습 표본은 4,746 그대로 유지.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time, gc

OUT = Path(os.environ['STAGE2_OUT'])
OUT.mkdir(parents=True, exist_ok=True)
PROJECT = Path(os.environ['STAGE2_RAW'])

import re

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


def _norm_market(x):
    if pd.isna(x):
        return 'UNKNOWN'
    s=str(x).strip(); su=s.upper()
    if not s or s.lower() in {'nan','none','<na>','unknown_market','unknown'}:
        return 'UNKNOWN'
    if 'KOSPI' in su or '코스피' in s or '유가' in s:
        return 'KOSPI'
    if 'KOSDAQ' in su or '코스닥' in s:
        return 'KOSDAQ'
    if 'KONEX' in su or '코넥스' in s:
        return 'KONEX'
    return 'UNKNOWN'


def _market_from_filename(name):
    return _norm_market(name)


def _canonical_item_code_from_text(text):
    if pd.isna(text):
        return ''
    m=re.search(r"(U\d{2}[A-Z]\d{9,12})", str(text))
    return m.group(1) if m else ''


def _norm_code(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = re.sub(r"[^0-9]", "", s)
    return s.zfill(6) if s else ""

def _norm_year(x):
    if pd.isna(x):
        return -1
    try:
        return int(float(x))
    except Exception:
        m = re.search(r"(19|20)\d{2}", str(x))
        return int(m.group(0)) if m else -1

panel = pd.read_parquet(os.environ['STAGE2_S1B_PANEL'])
panel = panel.copy()
panel['_key_code'] = panel['거래소코드'].map(_norm_code)
panel['_key_year'] = panel['year'].map(_norm_year)
if K_MARKET_COL not in panel.columns:
    panel[K_MARKET_COL] = 'UNKNOWN'
panel[K_MARKET_COL] = panel[K_MARKET_COL].map(_norm_market)
elig = panel[panel['eligible_for_stage2']].copy()

# t-1 키 set (eligible 4,746 firm-year의 직전년도)
prior_keys = set(zip(elig['_key_code'], elig['_key_year'] - 1))
print(f"Eligible: {len(elig):,}, prior keys (t-1): {len(prior_keys):,}")
# 일부 firm-year는 t-1이 다른 eligible firm-year와 겹칠 수 있어 unique 더 적음
# 또한 같은 prior_key가 여러 elig로부터 참조될 수 있음 (연속 등급)

PATTERNS = ['재무상태표', '손익계산서', '현금흐름표', '자본변동표',
            '이익잉여금처분계산서', '재무비율']
files = sorted([f for f in PROJECT.glob('*.xlsx')
                if any(p in f.name for p in PATTERNS) and ('코스피' in f.name or '코스닥' in f.name or '코넥스' in f.name)])
print(f"처리 대상 파일: {len(files)}\n")

# statement별 lag support 행 수집 (firm-split 통합 위해)
import collections
collected = collections.defaultdict(list)  # statement → list of dataframes

for i, fp in enumerate(files, 1):
    t0 = time.time()
    stmt = next((p for p in PATTERNS if p in fp.name), 'unknown')
    try:
        df = _read_excel_with_fallback(fp)
    except Exception as e:
        print(f"  ERROR {fp.name}: {e}")
        continue

    # year extraction + key matching; use unicode-safe column constants.
    df['year'] = df[K_YEAR_COL].map(_norm_year)
    df[K_CODE_COL] = df[K_CODE_COL].map(_norm_code)
    df = df[(df['year'] > 0) & (df[K_CODE_COL] != "")].copy()

    fy = list(zip(df[K_CODE_COL].values, df['year'].values))
    mask = pd.Series([t in prior_keys for t in fy])
    matched = df[mask.values].copy()
    if len(matched) > 0:
        matched['_source_file'] = fp.stem
        matched['_market_source'] = _market_from_filename(fp.name)
        collected[stmt].append(matched)

    print(f"  [{i:>2}/{len(files)}] {fp.stem[:55]:<60s} matched={len(matched):>4,} ({(time.time()-t0):.1f}s)")
    del df; gc.collect()

# 통합 후 중복 처리 (Stage 1B와 동일 정책 적용)
print("\n[통합 + 중복 처리]")
LAG_DIR = OUT / 'lag_support'
LAG_DIR.mkdir(exist_ok=True)

# 신용등급 시장 lookup (t-1 시점이 아닌 t 시점 시장 기준 - 안 맞을 수 있지만 일관성 위해)
panel_market = dict(zip(zip(panel['_key_code'], panel['_key_year']), panel[K_MARKET_COL].map(_norm_market)))

def resolve_dup(df):
    if not df.duplicated(['거래소코드','year']).any():
        return df, []
    keep_idx = []
    excl_log = []
    for (code, year), grp in df.groupby(['거래소코드','year']):
        if len(grp) == 1:
            keep_idx.append(grp.index[0])
            continue
        sources = grp['_source_file'].unique()
        periods = grp['회계년도'].unique() if '회계년도' in grp.columns else []
        has_kospi = any('코스피' in s for s in sources)
        has_kosdaq = any('코스닥' in s for s in sources)
        is_market_transfer = has_kospi and has_kosdaq
        is_fiscal_change = len(periods) > 1

        if is_fiscal_change and not is_market_transfer:
            scored = grp.assign(
                _pri=grp['회계년도'].apply(lambda x: 0 if str(x).endswith('/12') else 1)
            ).sort_values('_pri')
            keep_idx.append(scored.index[0])
            for i in scored.index[1:]:
                excl_log.append({'code': code, 'year': year, 'reason': 'fiscal_year_change'})
        elif is_market_transfer:
            # eligible firm-year의 t 시점 시장 기준
            mkt = panel_market.get((code, year + 1), None)  # year+1 = t시점
            target = '코스피' if mkt == 'KOSPI' else '코스닥' if mkt == 'KOSDAQ' else '코넥스' if mkt == 'KONEX' else '코스피'
            matched = grp[grp['_source_file'].str.contains(target)]
            keep_i = matched.index[0] if len(matched) > 0 else grp.index[0]
            keep_idx.append(keep_i)
            for i in grp.index:
                if i != keep_i:
                    excl_log.append({'code': code, 'year': year, 'reason': 'market_transfer'})
        else:
            keep_idx.append(grp.index[0])
            for i in grp.index[1:]:
                excl_log.append({'code': code, 'year': year, 'reason': 'unknown_dup'})
    return df.loc[keep_idx], excl_log

total_lag_keys = set()
all_excl_log = []
for stmt, dfs in collected.items():
    if not dfs:
        continue
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    cleaned, excl = resolve_dup(combined)
    total_lag_keys |= set(zip(cleaned['거래소코드'], cleaned['year']))
    all_excl_log.extend(excl)
    # parquet 저장 (object → str)
    for c in cleaned.select_dtypes(include='object').columns:
        cleaned[c] = cleaned[c].astype(str)
    cleaned.to_parquet(LAG_DIR / f'{stmt}_lag_support.parquet',
                        compression='snappy', index=False)
    print(f"  {stmt:<20s}: {len(combined):,} → {len(cleaned):,} 중복제거 후 unique firm-year")

# Long format consolidated 저장 (전체 lag support 항목들 long format)
print("\n[Long format consolidated]")
long_dfs = []
META = {'회사명','거래소코드','회계년도','year','_source_file','_market_source'}
for stmt in PATTERNS:
    fp = LAG_DIR / f'{stmt}_lag_support.parquet'
    if not fp.exists():
        continue
    df = pd.read_parquet(fp)
    item_cols = [c for c in df.columns if c not in META]
    for c in item_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    melted = df.melt(id_vars=['거래소코드','year','회계년도','_source_file'],
                      value_vars=item_cols, var_name='item', value_name='value')
    melted = melted[melted['value'].notna()]
    melted['statement'] = stmt
    long_dfs.append(melted)
    print(f"  {stmt:<20s}: {len(melted):,} 셀")
    del df, melted; gc.collect()

if long_dfs:
    long_lag = pd.concat(long_dfs, ignore_index=True)
else:
    long_lag = pd.DataFrame(columns=['거래소코드', 'year', '회계년도', '_source_file', 'item', 'value', 'statement'])
for c in long_lag.select_dtypes(include='object').columns:
    long_lag[c] = long_lag[c].astype(str)
long_lag.to_parquet(OUT / 'lag_support_financial_items.parquet',
                     compression='snappy', index=False)
print(f"\n  → lag_support_financial_items.parquet ({len(long_lag):,}행 long format)")

# Coverage 재계산
elig['has_lag_support'] = elig.apply(
    lambda r: (r['_key_code'], r['_key_year'] - 1) in total_lag_keys, axis=1)
final_coverage = elig['has_lag_support'].sum() / len(elig)
print(f"\n[Final lag coverage 재계산]")
print(f"  eligible: {len(elig):,}")
print(f"  with lag support (raw 추출 후): {elig['has_lag_support'].sum():,} ({final_coverage:.1%})")

# market carryover lookup for downstream Task6
market_lookup = panel[[K_CODE_COL, 'year', K_MARKET_COL]].copy()
market_lookup[K_CODE_COL] = market_lookup[K_CODE_COL].map(_norm_code)
market_lookup['year'] = market_lookup['year'].map(_norm_year).astype('int64')
market_lookup[K_MARKET_COL] = market_lookup[K_MARKET_COL].map(_norm_market)
raw_market_rows = []
for stmt, dfs in collected.items():
    for _df in dfs:
        if K_CODE_COL in _df.columns and 'year' in _df.columns and '_market_source' in _df.columns:
            raw_market_rows.append(_df[[K_CODE_COL, 'year', '_market_source']].rename(columns={'_market_source': K_MARKET_COL}))
if raw_market_rows:
    raw_lookup = pd.concat(raw_market_rows, ignore_index=True).drop_duplicates([K_CODE_COL, 'year'], keep='last')
    market_lookup = pd.concat([market_lookup, raw_lookup], ignore_index=True, sort=False)
market_lookup[K_MARKET_COL] = market_lookup[K_MARKET_COL].map(_norm_market)
market_lookup = market_lookup.drop_duplicates([K_CODE_COL, 'year'], keep='last')
market_lookup.to_csv(OUT / 'stage00_02_market_carryover_lookup.csv', index=False, encoding='utf-8-sig')
pd.DataFrame([{
    'panel_rows': int(len(panel)),
    'lookup_rows': int(len(market_lookup)),
    'known_market_rows': int(market_lookup[K_MARKET_COL].astype(str).ne('UNKNOWN').sum()),
    'processed_raw_files': int(len(files)),
    'lag_support_long_rows': int(len(long_lag)),
}]).to_csv(OUT / 'stage00_02_market_carryover_summary.csv', index=False, encoding='utf-8-sig')
if len(files) > 0 and len(long_lag) == 0:
    raise RuntimeError('Task4 lag dedup found raw statement files but produced zero raw long item rows; refusing silent pass')
# log 저장
if all_excl_log:
    pd.DataFrame(all_excl_log).to_csv(OUT / 'lag_support_dup_excl_log.csv',
                                       index=False, encoding='utf-8-sig')
print("\n완료.")
