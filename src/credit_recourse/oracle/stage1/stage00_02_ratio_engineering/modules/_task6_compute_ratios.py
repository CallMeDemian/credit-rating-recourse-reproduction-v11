"""
Task 6 v10: engineered_financial_ratios generation with canonical U-code lookup,
primary+alias row-level coalesce, market carryover, and positional ratio assignment.
"""
from __future__ import annotations
import os, re, gc
from pathlib import Path
import pandas as pd
import numpy as np

UCODE_PATTERN = re.compile(r"(U\d{2}[A-Z]\d{9,12})")

def canonical_item_code_from_text(text) -> str:
    if pd.isna(text): return ""
    m = UCODE_PATTERN.search(str(text))
    return m.group(1) if m else ""

def find_column_for_code(columns_or_df, code):
    columns = list(columns_or_df.columns) if hasattr(columns_or_df, 'columns') else list(columns_or_df)
    code = canonical_item_code_from_text(code) or str(code).strip()
    for c in columns:
        if canonical_item_code_from_text(c) == code:
            return c
    return None

def _norm_code(x):
    if pd.isna(x): return ""
    s=str(x).strip()
    if s.endswith('.0'): s=s[:-2]
    s=re.sub(r'[^0-9]','',s)
    return s.zfill(6) if s else ""

def _norm_market(x):
    if pd.isna(x): return 'UNKNOWN'
    s=str(x).strip(); su=s.upper()
    if not s or s.lower() in {'nan','none','<na>','unknown_market','unknown'}: return 'UNKNOWN'
    if 'KOSPI' in su or '코스피' in s or '유가' in s: return 'KOSPI'
    if 'KOSDAQ' in su or '코스닥' in s: return 'KOSDAQ'
    if 'KONEX' in su or '코넥스' in s: return 'KONEX'
    return 'UNKNOWN'

ITEM_CODE_ALIASES = {
    'U01B100000000': ['U01B100000000','U02B100000000','U03B100000000','U04B100000000'],
    'U01B800000000': ['U01B800000000','U02B800000000','U03B800000000','U04B800000000'],
    'U01B350014300': ['U01B350014300','U02B350012400','U03B350012400','U04B350012400','U01B350012400'],
}

def _ensure_positional_series(values, index, label):
    arr = values.to_numpy() if isinstance(values, pd.Series) else np.asarray(values)
    if arr.ndim == 0: arr = np.repeat(arr, len(index))
    if len(arr) != len(index):
        raise RuntimeError(f'{label} length mismatch: got {len(arr)} expected {len(index)}')
    return pd.Series(arr, index=index)

def _coalesce_code_columns(df: pd.DataFrame, code: str, lag: bool=False) -> pd.Series:
    suffix = '__lag1' if lag else ''
    primary = f'{code}{suffix}'
    out = df[primary].copy() if primary in df.columns else pd.Series(np.nan, index=df.index, dtype='float64')
    for alias in ITEM_CODE_ALIASES.get(code, [code]):
        col = f'{alias}{suffix}'
        if col in df.columns:
            out = out.where(out.notna(), pd.to_numeric(df[col], errors='coerce'))
    return out

ROLE_TO_CODE = {
    '매출액':'U01B100000000','매출원가':'U01B200000000','매출총이익':'U01B201014400','영업이익':'U01B430000000',
    '정상영업이익':'U01B430000000','법인세차감전순이익':'U01B700000000','계속사업이익':'U01B800000000',
    '당기순이익':'U01B840000000','총포괄이익':'U01B900000000','금융비용':'U01B550000000','감가상각비':'U01B350014100',
    '무형자산상각비':'U01B350014300','총자산':'U01A100000000','유동자산':'U01A111038600','비유동자산':'U01A110000000',
    '재고자산':'U01A111038700','매출채권':'U01A111045400','현금및현금성자산':'U01A111050000','유동부채':'U01A811026000',
    '비유동부채':'U01A810000000','부채총계':'U01A800000000','자본총계':'U01A600000000','자본금':'U01A611000000',
    '이익잉여금':'U01A615000000','단기차입금':'U01A811026700','장기차입금':'U01A811012800','사채':'U01A811000000',
    '매입채무':'U01A811030700','영업활동현금흐름':'U01D100000000','투자활동현금흐름':'U01D200000000',
    '재무활동현금흐름':'U01D300000000','CAPEX_proxy':'U01D206012400','유동성장기부채':'U01A811027400','단기금융상품':'U01A111043200'}
DERIVED_NAMES=['EBITDA','EBIT','CAPEX','FCF','NOPAT','총차입금','순차입금','순운전자본','순영업운전자본','영업운전자본','유이자부채','금융부채','단기금융부채','장기금융부채','즉시가용유동성','당좌자산']

def main() -> None:
    OUT=Path(os.environ['STAGE2_OUT']); OUT.mkdir(parents=True, exist_ok=True)
    panel=pd.read_parquet(os.environ['STAGE2_S1B_PANEL']).copy()
    panel['거래소코드']=panel['거래소코드'].map(_norm_code)
    panel['year']=pd.to_numeric(panel['year'], errors='coerce').astype('Int64')
    if '시장' not in panel.columns: panel['시장']='UNKNOWN'
    panel['시장']=panel['시장'].map(_norm_market)
    if 'sector_7' not in panel.columns: panel['sector_7']='UNKNOWN'
    carry=OUT/'stage00_02_market_carryover_lookup.csv'
    if carry.exists():
        lk=pd.read_csv(carry, encoding='utf-8-sig')
        if {'거래소코드','year','시장'}.issubset(lk.columns):
            lk['거래소코드']=lk['거래소코드'].map(_norm_code); lk['year']=pd.to_numeric(lk['year'],errors='coerce').astype('Int64'); lk['시장']=lk['시장'].map(_norm_market)
            panel=panel.merge(lk[['거래소코드','year','시장']].rename(columns={'시장':'_carry_market'}), on=['거래소코드','year'], how='left')
            panel['시장']=panel['시장'].where(panel['시장'].ne('UNKNOWN'), panel['_carry_market'].map(_norm_market))
            panel=panel.drop(columns=['_carry_market']); panel['시장']=panel['시장'].map(_norm_market)
    elig=panel[panel['eligible_for_stage2']].copy(); print(f"Eligible: {len(elig):,}\n")
    dict_df=pd.read_csv(OUT/'financial_ratio_formula_dictionary_draft.csv')
    calculable=dict_df[dict_df['availability_status'].isin(['available_direct','available_with_lag','available_derived'])].copy()
    all_codes_needed=set(ROLE_TO_CODE.values())
    for aliases in ITEM_CODE_ALIASES.values(): all_codes_needed |= set(aliases)
    for _, r in calculable.iterrows():
        for field in ['numerator_item_code','denominator_item_code']:
            for code in str(r.get(field,'')).split('+'):
                code=canonical_item_code_from_text(code) or code.strip()
                if code and code not in {'DERIVED','nan','None'}: all_codes_needed.add(code)
    CLEAN_DIR=Path(os.environ['STAGE2_S1B_CLEAN']); LAG_DIR=OUT/'lag_support'; STMT=['재무상태표','손익계산서','현금흐름표','자본변동표','이익잉여금처분계산서','재무비율']
    mat_t=elig[['거래소코드','year']].copy(); elig_keys=set(zip(elig['거래소코드'], elig['year']))
    for stmt in STMT:
        fp=CLEAN_DIR/f'{stmt}_clean.parquet'
        if not fp.exists(): continue
        df=pd.read_parquet(fp); df['거래소코드']=df['거래소코드'].map(_norm_code); df['year']=pd.to_numeric(df['year'],errors='coerce').astype('Int64')
        df=df[df.apply(lambda r:(r['거래소코드'],r['year']) in elig_keys, axis=1)]
        keep=['거래소코드','year']; rename_map={}
        for code in all_codes_needed:
            col=find_column_for_code(df, code)
            if col is not None: keep.append(col); rename_map[col]=code
        if len(keep)>2:
            sub=df[keep].rename(columns=rename_map).T.groupby(level=0).first().T
            for c in [c for c in sub.columns if c not in {'거래소코드','year'}]: sub[c]=pd.to_numeric(sub[c],errors='coerce')
            mat_t=mat_t.merge(sub,on=['거래소코드','year'],how='left')
        del df; gc.collect()
    mat_lag=elig[['거래소코드','year']].copy(); mat_lag['_lag_year']=mat_lag['year']-1
    for stmt in STMT:
        fp=LAG_DIR/f'{stmt}_lag_support.parquet'
        if not fp.exists(): continue
        df=pd.read_parquet(fp); df['거래소코드']=df['거래소코드'].map(_norm_code); df['year']=pd.to_numeric(df['year'],errors='coerce').astype('Int64')
        keep=['거래소코드','year']; rename_map={}
        for code in all_codes_needed:
            col=find_column_for_code(df, code)
            if col is not None: keep.append(col); rename_map[col]=code+'__lag1'
        if len(keep)>2:
            sub=df[keep].rename(columns=rename_map).T.groupby(level=0).first().T
            for c in [c for c in sub.columns if c not in {'거래소코드','year'}]: sub[c]=pd.to_numeric(sub[c],errors='coerce')
            mat_lag=mat_lag.merge(sub,left_on=['거래소코드','_lag_year'],right_on=['거래소코드','year'],how='left',suffixes=('','_drop'))
            if 'year_drop' in mat_lag.columns: mat_lag=mat_lag.drop(columns=['year_drop'])
        del df; gc.collect()
    mat=mat_t.merge(mat_lag.drop(columns=['_lag_year']), on=['거래소코드','year'], how='left')
    def safe_get(df, code, lag=False): return _coalesce_code_columns(df, code, lag)
    def compute_derived(df,name,lag=False):
        if name=='EBITDA': return safe_get(df,'U01B430000000',lag).fillna(np.nan)+safe_get(df,'U01B350014100',lag).fillna(0)+safe_get(df,'U01B350014300',lag).fillna(0)
        if name=='EBIT': return safe_get(df,'U01B430000000',lag)
        if name=='CAPEX': return safe_get(df,'U01D206012400',lag)
        if name=='FCF': return safe_get(df,'U01D100000000',lag)-safe_get(df,'U01D206012400',lag).fillna(0)
        if name=='NOPAT':
            oi=safe_get(df,'U01B430000000',lag); ni=safe_get(df,'U01B840000000',lag); ebt=safe_get(df,'U01B700000000',lag); tr=np.clip(np.where(ebt>0,1-ni/ebt,0.25),0,0.50); return oi*(1-tr)
        if name=='총차입금': return safe_get(df,'U01A811026700',lag).fillna(0)+safe_get(df,'U01A811012800',lag).fillna(0)+safe_get(df,'U01A811000000',lag).fillna(0)+safe_get(df,'U01A811027400',lag).fillna(0)
        if name=='순차입금': return compute_derived(df,'총차입금',lag)-safe_get(df,'U01A111050000',lag).fillna(0)
        if name=='순운전자본': return safe_get(df,'U01A111038600',lag)-safe_get(df,'U01A811026000',lag)
        if name in ('순영업운전자본','영업운전자본'): return safe_get(df,'U01A111045400',lag).fillna(0)+safe_get(df,'U01A111038700',lag).fillna(0)-safe_get(df,'U01A811030700',lag).fillna(0)
        if name in ('유이자부채','금융부채'): return safe_get(df,'U01A811026700',lag).fillna(0)+safe_get(df,'U01A811012800',lag).fillna(0)+safe_get(df,'U01A811000000',lag).fillna(0)
        if name=='단기금융부채': return safe_get(df,'U01A811026700',lag).fillna(0)+safe_get(df,'U01A811027400',lag).fillna(0)
        if name=='장기금융부채': return safe_get(df,'U01A811012800',lag).fillna(0)+safe_get(df,'U01A811000000',lag).fillna(0)
        if name=='즉시가용유동성': return safe_get(df,'U01A111050000',lag).fillna(0)+safe_get(df,'U01A111043200',lag).fillna(0)
        if name=='당좌자산': return safe_get(df,'U01A111038600',lag)-safe_get(df,'U01A111038700',lag).fillna(0)
        return None
    for name in DERIVED_NAMES:
        v=compute_derived(mat,name,False); vl=compute_derived(mat,name,True)
        if v is not None: mat[f'_DERIVED_{name}']=v
        if vl is not None: mat[f'_DERIVED_{name}__lag1']=vl
    ALIAS={'자기자본':'자본총계','현금성자산':'현금및현금성자산','OCF':'영업활동현금흐름','이자비용':'금융비용','세전이익':'법인세차감전순이익','순이익':'당기순이익','CAPEX':'CAPEX_proxy'}
    def resolve_value(term, mat):
        if pd.isna(term): return None
        t=str(term).strip(); lag=0; is_growth=False
        if t.endswith('_t-1'): lag=1; t=t[:-4]
        elif t.endswith('_t-3'): lag=3; t=t[:-4]
        elif t.endswith('_t'): t=t[:-2]
        if '증가율' in t: is_growth=True; t=t.replace('증가율','')
        is_avg=t.startswith('평균')
        if is_avg:
            t=t[2:]
            if t.endswith(' × 365'): t=t[:-6]
        t=ALIAS.get(t,t); code=ROLE_TO_CODE.get(t)
        if code:
            if is_avg: return (safe_get(mat,code,False)+safe_get(mat,code,True))/2
            if lag==1: return safe_get(mat,code,True)
            if lag==3: return None
            if is_growth:
                curr=safe_get(mat,code,False); prev=safe_get(mat,code,True); return np.where(prev>0,(curr-prev)/prev,np.nan)
            return safe_get(mat,code,False)
        if t in DERIVED_NAMES:
            if is_avg: return (mat.get(f'_DERIVED_{t}',pd.Series(np.nan,index=mat.index))+mat.get(f'_DERIVED_{t}__lag1',pd.Series(np.nan,index=mat.index)))/2
            if lag==1: return mat.get(f'_DERIVED_{t}__lag1',pd.Series(np.nan,index=mat.index))
            if lag==3: return None
            if is_growth:
                curr=mat.get(f'_DERIVED_{t}',pd.Series(np.nan,index=mat.index)); prev=mat.get(f'_DERIVED_{t}__lag1',pd.Series(np.nan,index=mat.index)); return np.where(prev>0,(curr-prev)/prev,np.nan)
            return mat.get(f'_DERIVED_{t}',pd.Series(np.nan,index=mat.index))
        return None
    rating_cols=[c for c in ['rating_num','rating_num_10','grade_base_10','rating_num_7','grade_base_7','rating_num_notch','grade_base_notch'] if c in elig.columns]
    front=['거래소코드']+(['회사명'] if '회사명' in elig.columns else [])+['year','시장']+(['sector_7'] if 'sector_7' in elig.columns else [])+rating_cols+['split','eligible_for_stage2']
    result=elig[front].copy(); result=result.merge(mat[['거래소코드','year']],on=['거래소코드','year'],how='left')
    calc_log=[]; print("\n[Ratio 계산]")
    for _,row in calculable.iterrows():
        rid=row['ratio_id']; num_val=resolve_value(row['numerator_term_orig'],mat); den_val=resolve_value(row['denominator_term_orig'],mat)
        if num_val is None or den_val is None: calc_log.append({'ratio_id':rid,'status':'unresolved'}); continue
        num_val=_ensure_positional_series(num_val, mat.index, f'{rid}.numerator'); den_val=_ensure_positional_series(den_val, mat.index, f'{rid}.denominator')
        zero_mask=(den_val==0); neg_mask=(den_val<0); den_safe=den_val.copy(); den_safe[zero_mask]=np.nan
        if row.get('category') in ('성장성',) or row.get('negative_denominator_policy')=='NaN': den_safe[neg_mask]=np.nan
        is_growth=bool(row.get('requires_growth_base'))
        ratio_val=np.where(den_val>0,(num_val-den_val)/den_val,np.nan) if ('증가율' in str(row.get('ratio_name_ko')) or is_growth) else num_val.to_numpy()/den_safe.to_numpy()
        ratio_val=_ensure_positional_series(ratio_val,result.index,f'{rid}.ratio'); result[rid]=ratio_val.to_numpy()
        calc_log.append({'ratio_id':rid,'ratio_name':row.get('ratio_name_ko'),'category':row.get('category'),'status':'calculated','nan_count':int(ratio_val.isna().sum()),'zero_denom_count':int(zero_mask.sum()),'neg_denom_count':int(neg_mask.sum()),'min':float(np.nanmin(ratio_val)) if ratio_val.notna().any() else np.nan,'p1':float(np.nanpercentile(ratio_val,1)) if ratio_val.notna().any() else np.nan,'median':float(np.nanmedian(ratio_val)) if ratio_val.notna().any() else np.nan,'p99':float(np.nanpercentile(ratio_val,99)) if ratio_val.notna().any() else np.nan,'max':float(np.nanmax(ratio_val)) if ratio_val.notna().any() else np.nan})
    result.to_parquet(OUT/'engineered_financial_ratios.parquet',compression='snappy',index=False); result.to_csv(OUT/'engineered_financial_ratios.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(calc_log).to_csv(OUT/'ratio_calculation_log.csv',index=False,encoding='utf-8-sig')
    calc_rids={r['ratio_id'] for r in calc_log if r['status']=='calculated'}; dict_df[dict_df['ratio_id'].isin(calc_rids)].to_csv(OUT/'financial_ratio_formula_dictionary_final.csv',index=False,encoding='utf-8-sig')
    print(f"→ engineered_financial_ratios.parquet ({result.shape[0]}행 × {result.shape[1]}컬럼)")

if __name__ == '__main__': main()
