from __future__ import annotations
'S2 action-space coverage audit for rating movers.\n\nThe audit joins rating-mover firm-years to external event indicators that are\noutside the current 10D financial-action space (capital transactions and,\noptionally, structural/material-event tags).  It reports co-occurrence shares as\ncomposition diagnostics only; it does not make causal claims.\n'
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
from credit_recourse.rl.common.io import write_json
COVERAGE_SCHEMA_VERSION = 'action_space_coverage_audit_v1'
CAPITAL_EVENT_REGEX = re.compile('증자|감자|출자|전환|CB|BW|전환사채|신주|유상|무상|주식배당', re.IGNORECASE)

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Input table not found: {path}')
    suf = path.suffix.lower()
    if suf == '.parquet':
        return pd.read_parquet(path)
    if suf == '.csv':
        return pd.read_csv(path)
    if suf in {'.xlsx', '.xlsm', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported input extension for {path}')

def _norm_corp_key(value: object) -> str:
    if pd.isna(value):
        return ''
    s = str(value).strip()
    if s.endswith('.0') and s[:-2].isdigit():
        s = s[:-2]
    digits = re.sub('\\D', '', s)
    if digits:
        return digits.zfill(6)
    return s.upper()

def _norm_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin({'1', 'true', 't', 'yes', 'y', '예', 'y', 'positive'})

def _sniff_column(df: pd.DataFrame, candidates: Iterable[str], *, label: str, required: bool=True) -> str | None:
    cols = list(df.columns)
    lowered = {str(c).lower(): c for c in cols}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    for col in cols:
        scol = str(col).lower()
        if any((str(c).lower() in scol for c in candidates)):
            return col
    if required:
        raise ValueError({'message': f'Could not detect required column for {label}.', 'candidates': list(candidates), 'available_columns': [str(c) for c in cols]})
    return None

def normalize_movers(df: pd.DataFrame) -> pd.DataFrame:
    key_col = _sniff_column(df, ['corp_key', 'firm_id', '거래소코드', '종목코드', '회사코드', '법인코드', 'corp'], label='movers firm key')
    year_col = _sniff_column(df, ['fiscal_year', 'year', '연도', '회계연도', '평가연도'], label='movers fiscal year')
    direction_col = _sniff_column(df, ['direction', 'rating_direction', '등급방향', 'move_direction', 'rating_move'], label='movers direction', required=False)
    out = df.copy().rename(columns={key_col: 'corp_key', year_col: 'fiscal_year'})
    if direction_col and direction_col != 'direction':
        out = out.rename(columns={direction_col: 'direction'})
    if 'direction' not in out.columns:
        out['direction'] = 'ALL'
    out['corp_key'] = out['corp_key'].map(_norm_corp_key)
    out['fiscal_year'] = pd.to_numeric(out['fiscal_year'], errors='coerce').astype('Int64')
    bad = out['corp_key'].eq('') | out['fiscal_year'].isna()
    if bad.any():
        raise ValueError({'message': 'Movers contain rows with missing normalized corp_key or fiscal_year.', 'n_bad_rows': int(bad.sum()), 'bad_sample': out.loc[bad].head(10).to_dict(orient='records')})
    dup = out[['corp_key', 'fiscal_year', 'direction']].duplicated(keep=False)
    if dup.any():
        raise ValueError({'message': 'Movers contain duplicate corp_key/fiscal_year/direction rows.', 'n_duplicate_rows': int(dup.sum()), 'duplicate_sample': out.loc[dup, ['corp_key', 'fiscal_year', 'direction']].head(20).to_dict(orient='records')})
    return out

def _detect_capital_columns(df: pd.DataFrame, *, source: Path) -> tuple[str, str, list[str]]:
    key = _sniff_column(df, ['corp_key', 'firm_id', '거래소코드', '종목코드', '회사코드', '회사', '법인', '종목'], label=f'capital firm key ({source.name})')
    date = _sniff_column(df, ['변동일', '일자', '날짜', 'date', '기준일', '공시일'], label=f'capital event date ({source.name})')
    preferred_tokens = ['변동내역', '변동내용', '내용', '사유', '유형', '구분']
    type_cols: list[str] = []
    for token in preferred_tokens:
        for col in df.columns:
            if token in str(col) and col not in type_cols:
                type_cols.append(col)
    if not type_cols:
        typ = _sniff_column(df, preferred_tokens, label=f'capital event type ({source.name})')
        type_cols = [typ] if typ is not None else []
    if not type_cols:
        raise ValueError({'message': f'Could not detect capital event type/detail columns for {source.name}.', 'available_columns': [str(c) for c in df.columns]})
    return (key, date, type_cols)

def load_capital_events(paths: Iterable[Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames: list[pd.DataFrame] = []
    manifests: list[dict] = []
    for raw_path in paths:
        path = Path(raw_path).resolve()
        df = _read_table(path)
        key_col, date_col, type_cols = _detect_capital_columns(df, source=path)
        keep_cols = [key_col, date_col] + [c for c in type_cols if c not in {key_col, date_col}]
        work = df[keep_cols].copy().rename(columns={key_col: 'corp_key', date_col: 'event_date'})
        event_text = pd.Series('', index=work.index, dtype='object')
        for c in type_cols:
            if c in work.columns:
                event_text = event_text.str.cat(work[c].astype(str), sep=' | ')
        work['event_type'] = event_text.str.strip(' |')
        work['corp_key'] = work['corp_key'].map(_norm_corp_key)
        dt = pd.to_datetime(work['event_date'], errors='coerce')
        unresolved = dt.isna() & work['event_date'].notna()
        if unresolved.any():
            dt2 = pd.to_datetime(work.loc[unresolved, 'event_date'].astype(str).str.replace('\\D', '', regex=True), format='%Y%m%d', errors='coerce')
            dt.loc[unresolved] = dt2
        work['fiscal_year'] = dt.dt.year.astype('Int64')
        work['is_equity_event'] = work['event_type'].astype(str).str.contains(CAPITAL_EVENT_REGEX, na=False)
        bad_key = work['corp_key'].eq('')
        work = work.loc[~bad_key & work['fiscal_year'].notna()].copy()
        manifests.append({'path': str(path), 'rows_loaded': int(len(df)), 'rows_after_key_year_filter': int(len(work)), 'firm_key_column': str(key_col), 'date_column': str(date_col), 'type_columns': [str(c) for c in type_cols], 'equity_event_rows': int(work['is_equity_event'].sum())})
        frames.append(work[['corp_key', 'fiscal_year', 'is_equity_event']])
    if not frames:
        raise ValueError('No capital event files supplied.')
    all_events = pd.concat(frames, ignore_index=True)
    grouped = all_events.groupby(['corp_key', 'fiscal_year'], as_index=False)['is_equity_event'].max()
    return (grouped, manifests)

def load_structural_tags(path: Path) -> tuple[pd.DataFrame, dict]:
    p = Path(path).resolve()
    df = _read_table(p)
    key_col = _sniff_column(df, ['corp_key', 'firm_id', '거래소코드', '종목코드', '회사코드', 'corp'], label='structural tag firm key')
    year_col = _sniff_column(df, ['fiscal_year', 'year', '연도', '회계연도', 'target_year'], label='structural tag fiscal year')
    flag_col = _sniff_column(df, ['has_structural_event', 'structural_event', 'material_event', 'has_material_event', 'event_flag', '구조'], label='structural event flag')
    out = df[[key_col, year_col, flag_col]].copy().rename(columns={key_col: 'corp_key', year_col: 'fiscal_year', flag_col: 'has_structural_event'})
    out['corp_key'] = out['corp_key'].map(_norm_corp_key)
    out['fiscal_year'] = pd.to_numeric(out['fiscal_year'], errors='coerce').astype('Int64')
    out['has_structural_event'] = _norm_bool(out['has_structural_event'])
    out = out.loc[~out['corp_key'].eq('') & out['fiscal_year'].notna()].copy()
    out = out.groupby(['corp_key', 'fiscal_year'], as_index=False)['has_structural_event'].max()
    meta = {'path': str(p), 'rows_loaded': int(len(df)), 'rows_after_key_year_filter': int(len(out)), 'firm_key_column': str(key_col), 'year_column': str(year_col), 'flag_column': str(flag_col)}
    return (out, meta)

def _rate_frame(g: pd.DataFrame) -> dict:
    n = int(len(g))
    if n == 0:
        return {'n': 0, 'equity_event_share': math.nan, 'structural_event_share': math.nan, 'outside_action_space_share': math.nan}
    return {'n': n, 'equity_event_share': float(g['is_equity_event'].mean()), 'structural_event_share': float(g['has_structural_event'].mean()), 'outside_action_space_share': float(g['outside_action_space'].mean())}

def run_action_space_coverage_audit(*, movers: Path, capital_files: Iterable[Path], out_dir: Path, struct_tags: Path | None=None) -> dict:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    movers_df = normalize_movers(_read_table(Path(movers)))
    cap, cap_manifest = load_capital_events([Path(p) for p in capital_files])
    joined = movers_df.merge(cap, on=['corp_key', 'fiscal_year'], how='left')
    joined['is_equity_event'] = joined['is_equity_event'].apply(lambda x: bool(x) if pd.notna(x) else False).astype(bool)
    struct_meta = None
    if struct_tags is not None:
        st, struct_meta = load_structural_tags(Path(struct_tags))
        joined = joined.merge(st, on=['corp_key', 'fiscal_year'], how='left')
        joined['has_structural_event'] = joined['has_structural_event'].apply(lambda x: bool(x) if pd.notna(x) else False).astype(bool)
    else:
        joined['has_structural_event'] = False
    joined['outside_action_space'] = joined['is_equity_event'] | joined['has_structural_event']
    joined.to_csv(out_dir / 'action_space_coverage_joined.csv', index=False, encoding='utf-8-sig')
    rows: list[dict] = []
    total = _rate_frame(joined)
    rows.append({'slice': 'ALL', 'direction': 'ALL', **total})
    for direction, g in joined.groupby('direction', dropna=False):
        rows.append({'slice': 'direction', 'direction': str(direction), **_rate_frame(g)})
    summary = pd.DataFrame(rows)
    for col in ['equity_event_share', 'structural_event_share', 'outside_action_space_share']:
        summary[col.replace('_share', '_pct')] = summary[col] * 100.0
    summary.to_csv(out_dir / 'action_space_coverage.csv', index=False, encoding='utf-8-sig')
    meta = {'schema_version': COVERAGE_SCHEMA_VERSION, 'created_utc': _now(), 'status': 'PASS', 'movers': str(Path(movers).resolve()), 'capital_files': cap_manifest, 'struct_tags': struct_meta, 'n_movers': int(len(movers_df)), 'outputs': {'action_space_coverage': 'action_space_coverage.csv', 'joined': 'action_space_coverage_joined.csv'}, 'interpretation_note': 'Co-occurrence composition diagnostic only; no causal or ordering claim is implied.', 'event_regex': CAPITAL_EVENT_REGEX.pattern}
    write_json(out_dir / 'metadata.json', meta)
    return meta

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='S2 action-space coverage audit for rating movers')
    ap.add_argument('--movers', required=True)
    ap.add_argument('--capital-xlsx', nargs='+', required=True, help='TS2000 capital-change Excel/CSV/Parquet files')
    ap.add_argument('--struct-tags', default=None)
    ap.add_argument('--out-dir', '--out', dest='out_dir', required=True)
    args = ap.parse_args(argv)
    meta = run_action_space_coverage_audit(movers=Path(args.movers), capital_files=[Path(p) for p in args.capital_xlsx], struct_tags=Path(args.struct_tags) if args.struct_tags else None, out_dir=Path(args.out_dir))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
