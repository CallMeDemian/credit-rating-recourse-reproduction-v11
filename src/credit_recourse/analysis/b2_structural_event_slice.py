from __future__ import annotations
'Structural-event slice analysis for Loop B2 / Test 3 context.\n\nThis is a measurement-scope diagnostic, not simulator tuning. It tags firm-year\ntransitions with independently observed structural events from TS2000-style\nnonfinancial Excel files (merger/business transfer and important operating\nfacts), then reports B2 direction agreement separately for event and no-event\nsubdomains.\n\nThe script intentionally preserves the original B2 headline number. Conditional\nnumbers are reported as scope diagnostics only; they must not replace the full\nsample result unless declared ex ante.\n'
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
EVENT_LOAD_ERRORS: list[dict] = []

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_csv_any(path: Path) -> pd.DataFrame:
    for enc in ['utf-8-sig', 'utf-8', 'cp949']:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return pd.read_csv(path)

def _safe_year(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    m = re.search('(19|20)\\d{2}', s)
    if m:
        return float(m.group(0))
    try:
        return float(x)
    except Exception:
        return np.nan

def _norm_code(x) -> str:
    if pd.isna(x):
        return ''
    s = str(x).strip()
    if re.fullmatch('\\d+\\.0', s):
        s = s.split('.')[0]
    digits = re.sub('\\D', '', s)
    if digits:
        return digits.zfill(6)[-6:]
    return s

def _find_col(cols: Iterable[str], candidates: list[str]) -> str | None:
    norm = {str(c).strip(): c for c in cols}
    for cand in candidates:
        if cand in norm:
            return norm[cand]
    for c in cols:
        sc = str(c)
        if any((cand in sc for cand in candidates)):
            return c
    return None

def _event_type_from_path(path: Path) -> str:
    name = path.name
    if '합병' in name or '#Ud569#Ubcd1' in name:
        return 'ma_business_transfer'
    if '경영활동' in name or '#Uacbd#Uc601' in name:
        return 'material_operating_fact'
    return 'structural_event'

def _read_excel_robust(path: Path, *, header) -> pd.DataFrame:
    """Read a TS2000 Excel file with available engines.

    Some TS2000 .xlsx files contain malformed XML tokens that openpyxl cannot
    parse. If python-calamine is installed, calamine can sometimes read those
    files. We try it first when available, then fall back to pandas default /
    openpyxl.
    """
    errors: list[str] = []
    for engine in ['calamine', None, 'openpyxl']:
        try:
            kwargs = {} if engine is None else {'engine': engine}
            return pd.read_excel(path, sheet_name=0, header=header, **kwargs)
        except ImportError as e:
            errors.append(f"{engine or 'default'}:ImportError:{e}")
        except Exception as e:
            errors.append(f"{engine or 'default'}:{type(e).__name__}:{e}")
    raise RuntimeError('; '.join(errors))

def _load_event_file(path: Path) -> pd.DataFrame:
    try:
        raw = _read_excel_robust(path, header=0)
    except Exception as e0:
        try:
            tmp = _read_excel_robust(path, header=None)
            if tmp.empty:
                return pd.DataFrame()
            tmp.columns = tmp.iloc[0].astype(str).tolist()
            raw = tmp.iloc[1:].copy()
        except Exception as e1:
            raise RuntimeError(f'header0_failed=[{e0}] headerNone_failed=[{e1}]') from e1
    if raw.empty:
        return pd.DataFrame()
    code_col = _find_col(raw.columns, ['거래소코드', 'stock_code', 'ticker', '종목코드', 'code'])
    date_col = _find_col(raw.columns, ['일자', 'date', '발생일', '기준일'])
    fy_col = _find_col(raw.columns, ['회계년도', 'fiscal_year', 'year'])
    text_col = _find_col(raw.columns, ['내역', '내용', 'description', 'fact'])
    if code_col is None:
        return pd.DataFrame()
    out = pd.DataFrame()
    out['stock_code_norm'] = raw[code_col].map(_norm_code)
    out['event_year'] = raw[date_col].map(_safe_year) if date_col is not None else np.nan
    out['fiscal_year_hint'] = raw[fy_col].map(_safe_year) if fy_col is not None else np.nan
    out['event_type'] = _event_type_from_path(path)
    out['event_file'] = str(path)
    out['event_text'] = raw[text_col].astype(str) if text_col is not None else ''
    out = out[(out['stock_code_norm'] != '') & out['event_year'].notna()].copy()
    return out

def load_events(raw_nonfinancial_dir: Path, *, include_material_facts: bool=True) -> pd.DataFrame:
    raw_nonfinancial_dir = Path(raw_nonfinancial_dir)
    patterns = ['**/*합병*.xlsx', '**/*#Ud569#Ubcd1*.xlsx']
    if include_material_facts:
        patterns += ['**/*경영활동*.xlsx', '**/*#Uacbd#Uc601*.xlsx']
    paths: list[Path] = []
    for pat in patterns:
        paths += list(raw_nonfinancial_dir.glob(pat))
    frames = []
    global EVENT_LOAD_ERRORS
    EVENT_LOAD_ERRORS = []
    for path in sorted(set(paths)):
        try:
            f = _load_event_file(path)
        except Exception as e:
            EVENT_LOAD_ERRORS.append({'event_file': str(path), 'event_type_from_path': _event_type_from_path(path), 'error_type': type(e).__name__, 'error': str(e)[:2000]})
            continue
        if not f.empty:
            frames.append(f)
    if not frames:
        return pd.DataFrame(columns=['stock_code_norm', 'event_year', 'event_type', 'event_text', 'event_file'])
    ev = pd.concat(frames, ignore_index=True)
    ev = ev.drop_duplicates(subset=['stock_code_norm', 'event_year', 'event_type', 'event_text']).reset_index(drop=True)
    return ev

def _b2_score_delta(df: pd.DataFrame) -> pd.Series:
    for c in ['pred_alpha_delta', 'delta_pred_alpha', 'sim_alpha_delta', 'score_delta']:
        if c in df.columns:
            return pd.to_numeric(df[c], errors='coerce')
    pred_cols = ['pred_alpha_score_tplus1', 'alpha_score_tplus1', 'R_score_alpha_tplus1']
    base_cols = ['alpha_score_t', 'R_score_alpha_t', 'base_alpha_score', 'alpha_score_current']
    pred = next((c for c in pred_cols if c in df.columns), None)
    base = next((c for c in base_cols if c in df.columns), None)
    if pred is not None and base is not None:
        return pd.to_numeric(df[pred], errors='coerce') - pd.to_numeric(df[base], errors='coerce')
    raise ValueError('Could not find B2 score-change column. Expected one of pred_alpha_delta/delta_pred_alpha/sim_alpha_delta/score_delta or score_t and score_tplus1 columns.')

def _direction_agreement(score_delta: pd.Series, rating_delta: pd.Series) -> pd.Series:
    sd = pd.to_numeric(score_delta, errors='coerce')
    rd = pd.to_numeric(rating_delta, errors='coerce')
    moved = rd.ne(0) & rd.notna() & sd.notna()
    out = pd.Series(np.nan, index=sd.index)
    out.loc[moved] = (np.sign(sd.loc[moved]) == np.sign(rd.loc[moved])).astype(float)
    out.loc[moved & sd.eq(0)] = 0.0
    return out

def _wilson_ci(k: int, n: int, z: float=1.96) -> tuple[float, float]:
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (float(center - half), float(center + half))

def _summarize(g: pd.DataFrame) -> dict:
    valid = g['b2_direction_agree'].notna()
    n = int(valid.sum())
    k = int(g.loc[valid, 'b2_direction_agree'].sum()) if n else 0
    lo, hi = _wilson_ci(k, n)
    return {'n_movers': n, 'n_agree': k, 'agreement': float(k / n) if n else float('nan'), 'wilson95_lo': lo, 'wilson95_hi': hi, 'share_of_mover_sample': float(n / max(1, int(g.attrs.get('total_movers', n)))) if n else 0.0}

def run_slice(*, b2_dir: Path, raw_nonfinancial_dir: Path, output_dir: Path | None=None, test3_rows_path: Path | None=None, event_window_years: int=0, include_material_facts: bool=True) -> dict:
    b2_dir = Path(b2_dir)
    default_test3_path = b2_dir / 'test3_redesign' / 'test3_rows.csv'
    test3_path = Path(test3_rows_path) if test3_rows_path is not None else default_test3_path
    legacy_path = b2_dir / 'loopB2_alpha_predicted_score_vs_real_rating_change.csv'
    if test3_rows_path is not None and (not test3_path.exists()):
        raise FileNotFoundError(f'Explicit Test3 rows file does not exist: {test3_path}')
    if test3_path.exists():
        b2_path = test3_path
        b2 = _read_csv_any(b2_path)
        required = {'firm_id', 'fiscal_year', 'target_year', 'direction_match'}
        missing = required - set(b2.columns)
        if missing:
            raise ValueError(f'test3_rows.csv lacks required columns: {sorted(missing)}')
        year_col = 'fiscal_year'
        b2 = b2.copy()
        b2['stock_code_norm'] = b2['firm_id'].map(_norm_code)
        b2['fiscal_year_num'] = b2[year_col].map(_safe_year)
        b2['b2_direction_agree'] = pd.to_numeric(b2['direction_match'], errors='coerce')
    else:
        b2_path = legacy_path
        if not b2_path.exists():
            raise FileNotFoundError(f'Neither {test3_path} nor {legacy_path} exists')
        b2 = _read_csv_any(b2_path)
        if 'real_rating_delta_t_to_tplus1' not in b2.columns:
            raise ValueError('Legacy B2 csv lacks real_rating_delta_t_to_tplus1')
        if 'firm_id' not in b2.columns:
            raise ValueError('Legacy B2 csv lacks firm_id; cannot join to event files.')
        year_col = 'fiscal_year' if 'fiscal_year' in b2.columns else None
        if year_col is None:
            raise ValueError('Legacy B2 csv lacks fiscal_year; cannot align event timing.')
        b2 = b2.copy()
        b2['stock_code_norm'] = b2['firm_id'].map(_norm_code)
        b2['fiscal_year_num'] = b2[year_col].map(_safe_year)
        b2['b2_score_delta'] = _b2_score_delta(b2)
        b2['b2_direction_agree'] = _direction_agreement(b2['b2_score_delta'], b2['real_rating_delta_t_to_tplus1'])
    ev = load_events(raw_nonfinancial_dir, include_material_facts=include_material_facts)
    load_errors = list(EVENT_LOAD_ERRORS)
    if ev.empty:
        msg = f'No event rows loaded from {raw_nonfinancial_dir}'
        if load_errors:
            first = load_errors[0]
            msg += f"; first_load_error_file={first.get('event_file')} error={first.get('error')}"
        raise ValueError(msg)
    ev_small = ev[['stock_code_norm', 'event_year', 'event_type', 'event_text', 'event_file']].copy()
    joined = b2[['stock_code_norm', 'fiscal_year_num']].reset_index().merge(ev_small, on='stock_code_norm', how='left')
    joined['year_distance'] = (joined['event_year'] - joined['fiscal_year_num']).abs()
    tagged = joined[joined['year_distance'].le(float(event_window_years))].copy()
    event_idx = set(tagged['index'].dropna().astype(int).tolist())
    b2['has_structural_event'] = [i in event_idx for i in range(len(b2))]
    type_map = tagged.groupby('index')['event_type'].apply(lambda s: ';'.join(sorted(set(map(str, s))))).to_dict()
    text_map = tagged.groupby('index')['event_text'].apply(lambda s: ' || '.join(list(map(str, s.head(3))))).to_dict()
    b2['structural_event_types'] = [type_map.get(i, '') for i in range(len(b2))]
    b2['structural_event_examples'] = [text_map.get(i, '') for i in range(len(b2))]
    total_movers = int(b2['b2_direction_agree'].notna().sum())
    b2.attrs['total_movers'] = total_movers
    summaries = []
    for name, g in [('full_sample', b2), ('no_structural_event', b2[~b2['has_structural_event']]), ('structural_event', b2[b2['has_structural_event']])]:
        g = g.copy()
        g.attrs['total_movers'] = total_movers
        rec = {'slice': name}
        rec.update(_summarize(g))
        summaries.append(rec)
    summary_df = pd.DataFrame(summaries)
    if output_dir is None:
        output_dir = b2_dir / 'b2_structural_event_slice'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    load_error_df = pd.DataFrame(load_errors)
    b2.to_csv(output_dir / 'b2_rows_with_structural_event_tags.csv', index=False, encoding='utf-8-sig')
    ev.to_csv(output_dir / 'loaded_structural_events.csv', index=False, encoding='utf-8-sig')
    summary_df.to_csv(output_dir / 'b2_structural_event_slice_summary.csv', index=False, encoding='utf-8-sig')
    load_error_df.to_csv(output_dir / 'event_file_load_errors.csv', index=False, encoding='utf-8-sig')
    meta = {'stage': 'analysis.b2_structural_event_slice', 'status': 'PASS', 'created_utc': _now(), 'b2_dir': str(b2_dir), 'b2_input': str(b2_path), 'test3_rows_source': 'explicit_cli' if test3_rows_path is not None else 'default_or_legacy_resolution', 'raw_nonfinancial_dir': str(raw_nonfinancial_dir), 'event_window_years': int(event_window_years), 'include_material_facts': bool(include_material_facts), 'status_detail': 'PASS_WITH_EVENT_FILE_LOAD_WARNINGS' if load_errors else 'PASS', 'n_event_rows_loaded': int(len(ev)), 'n_event_files_failed_to_load': int(len(load_errors)), 'failed_event_file_examples': load_errors[:5], 'interpretation_guard': 'Do not tune simulator or replace the full-sample B2 number. Report conditional B2 only as an independently defined scope diagnostic. If event_file_load_errors.csv is non-empty, report this as incomplete event coverage or repair/re-export those Excel files before treating the slice as final.', 'outputs': {'summary': 'b2_structural_event_slice_summary.csv', 'tagged_rows': 'b2_rows_with_structural_event_tags.csv', 'loaded_events': 'loaded_structural_events.csv', 'load_errors': 'event_file_load_errors.csv'}}
    (output_dir / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    return meta

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Loop B2 structural-event conditional slice diagnostic')
    ap.add_argument('--b2-dir', required=True)
    ap.add_argument('--raw-nonfinancial-dir', required=True)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--test3-rows', default=None, help='Explicit test3_rows.csv path. Canonical paper analysis passes the file created by the immediately preceding Test3 step.')
    ap.add_argument('--event-window-years', type=int, default=0)
    ap.add_argument('--exclude-material-facts', action='store_true')
    args = ap.parse_args(argv)
    meta = run_slice(b2_dir=Path(args.b2_dir), raw_nonfinancial_dir=Path(args.raw_nonfinancial_dir), output_dir=Path(args.output_dir) if args.output_dir else None, test3_rows_path=Path(args.test3_rows) if args.test3_rows else None, event_window_years=int(args.event_window_years), include_material_facts=not bool(args.exclude_material_facts))
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
