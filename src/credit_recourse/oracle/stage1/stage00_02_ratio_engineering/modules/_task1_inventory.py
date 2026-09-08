"""
Task 1 v10: Inventory from wide-format cleaned panels with canonical U-code extraction.
"""
from __future__ import annotations
import os
import re
import gc
from pathlib import Path
import pandas as pd
import numpy as np

UCODE_PATTERN = re.compile(r"(U\d{2}[A-Z]\d{9,12})")

def canonical_item_code_from_text(text) -> str:
    if pd.isna(text):
        return ""
    m = UCODE_PATTERN.search(str(text))
    return m.group(1) if m else ""

def _norm_code(x):
    if pd.isna(x): return ""
    s=str(x).strip()
    if s.endswith('.0'): s=s[:-2]
    s=re.sub(r'[^0-9]','',s)
    return s.zfill(6) if s else ""

def _norm_year(x):
    if pd.isna(x): return -1
    try: return int(float(x))
    except Exception:
        m=re.search(r'(19|20)\d{2}',str(x)); return int(m.group(0)) if m else -1

PATTERN = re.compile(r'^\[(U\d{2}[A-Z]\d{9,12})\]\s*(.*?)\((IFRS)\)\((.+?)\)\s*$')

def parse_col(col):
    col_s = str(col).strip()
    code = canonical_item_code_from_text(col_s)
    m = PATTERN.match(col_s)
    if m:
        _, name, _, unit = m.groups()
    else:
        # Remove all leading U-code signatures robustly, including duplicated bracket signatures.
        name = UCODE_PATTERN.sub('', col_s)
        name = name.replace('[',' ').replace(']',' ').strip()
        unit = 'unknown'
        um = re.search(r'\(([^()]*)\)\s*$', col_s)
        if um: unit = um.group(1)
    leading = len(str(name)) - len(str(name).lstrip())
    is_marker = '(*)' in str(name)
    clean_name = str(name).strip().replace('(*)','').strip()
    return code, clean_name, leading, is_marker, unit

ROLE_HINTS = {
    '총자산':[r'^자산$',r'^총자산$'], '유동자산':[r'^유동자산$'], '비유동자산':[r'^비유동자산$'],
    '재고자산':[r'^재고자산$'], '매출채권':[r'^매출채권$', r'^매출채권 및 기타'], '당좌자산':[r'^당좌자산$'],
    '현금및현금성자산':[r'^현금및현금성자산$', r'^현금성자산$'], '유동부채':[r'^유동부채$'],
    '비유동부채':[r'^비유동부채$'], '부채총계':[r'^부채$', r'^부채총계$'], '자본총계':[r'^자본$', r'^자본총계$', r'^자기자본$'],
    '자본금':[r'^자본금$'], '이익잉여금':[r'^이익잉여금'], '단기차입금':[r'^단기차입금$'], '장기차입금':[r'^장기차입금$'],
    '사채':[r'^사채$', r'^유동사채$'], '매입채무':[r'^매입채무$', r'^매입채무 및 기타'],
    '매출액':[r'^매출액$', r'^매출액\(수익\)', r'^수익\(매출액\)'],
    '매출원가':[r'^매출원가$'], '매출총이익':[r'^매출총이익'], '영업이익':[r'^영업이익', r'영업이익\(손실\)'],
    '정상영업이익':[r'^정상영업이익'], '법인세차감전순이익':[r'^법인세차감전', r'^법인세비용차감전'],
    '당기순이익':[r'^당기순이익', r'당기순이익\(손실\)'], '계속사업이익':[r'^계속영업', r'^계속사업'],
    '총포괄이익':[r'^총포괄', r'^포괄손익'], '금융비용':[r'^금융비용', r'^이자비용'],
    '감가상각비':[r'^감가상각비', r'^감가상각'], '무형자산상각비':[r'^무형자산상각'],
    '영업활동현금흐름':[r'^영업활동.*현금흐름', r'^영업활동으로'], '투자활동현금흐름':[r'^투자활동.*현금흐름', r'^투자활동으로'],
    '재무활동현금흐름':[r'^재무활동.*현금흐름', r'^재무활동으로'], 'CAPEX_proxy':[r'유형자산.*취득', r'유형자산.*증가'],
    '무형자산':[r'^무형자산$'], '영업권_proxy':[r'^영업권$'], '현금배당_proxy':[r'^현금배당$', r'^배당금$'],
    '리스부채_proxy':[r'^리스부채$', r'^유동리스부채$'], '사용권자산_proxy':[r'^사용권자산$'], 'RD비용_proxy':[r'^연구개발비$', r'^연구비$'],
}

def assign_hint(name):
    for role, patterns in ROLE_HINTS.items():
        for pat in patterns:
            if re.search(pat, str(name)):
                return role
    return ''

META_COLS = {'회사명','거래소코드','회계년도','year','_source_file','consolidation_basis_unknown'}

def main() -> None:
    OUT=Path(os.environ['STAGE2_OUT']); OUT.mkdir(parents=True,exist_ok=True)
    SRC=Path(os.environ['STAGE2_S1B_CLEAN'])
    panel=pd.read_parquet(os.environ['STAGE2_S1B_PANEL']).copy()
    panel['_key_code']=panel['거래소코드'].map(_norm_code); panel['_key_year']=panel['year'].map(_norm_year)
    eligible_keys=set(zip(panel.loc[panel['eligible_for_stage2'],'_key_code'], panel.loc[panel['eligible_for_stage2'],'_key_year']))
    print(f"Stage 2 eligible firm-year: {len(eligible_keys):,}\n")
    inventory=[]
    for fp in sorted(SRC.glob('*_clean.parquet')):
        stmt=fp.stem.replace('_clean','')
        print(f"[{stmt}] 처리 중...")
        df=pd.read_parquet(fp)
        df['_key_code']=df['거래소코드'].map(_norm_code); df['_key_year']=df['year'].map(_norm_year)
        df=df[df.apply(lambda r:(r['_key_code'],r['_key_year']) in eligible_keys, axis=1)].copy()
        df=df.drop(columns=['_key_code','_key_year'], errors='ignore')
        print(f"  eligible 행: {len(df):,}")
        item_cols=[c for c in df.columns if c not in META_COLS]
        n_total=len(df)
        for col in item_cols:
            s=pd.to_numeric(df[col], errors='coerce')
            n_nonmiss=int(s.notna().sum())
            if n_nonmiss==0: continue
            vals=s.dropna().to_numpy()
            code, clean_name, leading, is_marker, unit=parse_col(col)
            inventory.append({
                'statement_type':stmt,'item_code':code,'item_name':clean_name,'original_column_name':str(col),
                'is_summary_marker':bool(is_marker),'indent_level':int(leading//3),'non_missing_count':n_nonmiss,
                'non_missing_rate':round(n_nonmiss/n_total,4) if n_total else 0.0,
                'sample_min':float(np.min(vals)),'sample_p1':float(np.percentile(vals,1)),'sample_median':float(np.median(vals)),
                'sample_p99':float(np.percentile(vals,99)),'sample_max':float(np.max(vals)),'unit_inferred':unit or 'unknown',
                'candidate_role_hint':assign_hint(clean_name)})
        del df; gc.collect()
    inv_df=pd.DataFrame(inventory)
    if inv_df.empty:
        raise RuntimeError('Task1 inventory is empty. Check cleaned_statement_panels item columns.')
    inv_df['item_code']=inv_df.apply(lambda r: canonical_item_code_from_text(r.get('item_code','')) or canonical_item_code_from_text(r.get('original_column_name','')), axis=1)
    broken=inv_df[inv_df['item_code'].astype(str).str.startswith('[')]
    invalid=inv_df[~inv_df['item_code'].astype(str).str.match(r'^U\d{2}[A-Z]\d{9,12}$')]
    # RAW_RATIO_UCODE_GUARD_EXEMPTION_2026_05_24
    # NICE precomputed raw-ratio columns are ratio-name based columns produced by
    # Stage00_01, not canonical KIS U-code statement items.  Keep the U-code
    # fail-fast guard for all non-RAW_RATIO rows, but allow:
    #   statement_type == "재무비율" and original_column_name startswith "[RAW_RATIO]".
    raw_ratio_exempt = (
        invalid.get('statement_type', pd.Series('', index=invalid.index)).astype(str).eq('재무비율')
        & invalid.get('original_column_name', pd.Series('', index=invalid.index)).astype(str).str.startswith('[RAW_RATIO]')
    )
    invalid = invalid.loc[~raw_ratio_exempt].copy()
    if len(broken) or len(invalid):
        sample=pd.concat([broken, invalid]).head(20)[['statement_type','item_code','original_column_name']].to_dict(orient='records')
        raise RuntimeError(f'Task1 canonical U-code guard failed; sample={sample}')
    inv_df=inv_df.sort_values(['statement_type','indent_level','item_code']).reset_index(drop=True)
    inv_df.to_csv(OUT/'statement_item_inventory.csv', index=False, encoding='utf-8-sig')
    print(f"\n총 항목: {len(inv_df):,}")
    print(f"non_missing_rate ≥ 0.7: {(inv_df['non_missing_rate']>=0.7).sum():,}")
    print(f"non_missing_rate ≥ 0.9: {(inv_df['non_missing_rate']>=0.9).sum():,}")
    print("\nRole hint 할당:")
    print(inv_df[inv_df['candidate_role_hint']!=''].groupby('candidate_role_hint').size().to_string())

if __name__ == '__main__':
    main()
