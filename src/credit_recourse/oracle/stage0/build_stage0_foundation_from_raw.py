from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.oracle.stage0.rating_contract_repair import build_rating_sample, validate_stage0_contract
from credit_recourse.utils.io_contract import configure_utf8_stdio, write_json, assert_no_mojibake_columns

K_CODE = "거래소코드"
K_YEAR = "회계년도"
K_NAME = "회사명"
K_MARKET = "시장"

STATEMENT_ALIASES = {
    "재무상태표": ["재무상태표", "balance", "bs"],
    "손익계산서": ["손익계산서", "포괄손익", "income", "profit", "pnl"],
    "현금흐름표": ["현금흐름표", "cash", "cf"],
    "자본변동표": ["자본변동표", "equity", "capital_change"],
    "이익잉여금처분계산서": ["이익잉여금", "retained"],
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_col(cols: list[Any], candidates: list[str]) -> str | None:
    cols = list(cols)
    for cand in candidates:
        for c in cols:
            if str(c).strip() == cand:
                return c
    lowered = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        hit = lowered.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    for cand in candidates:
        needle = str(cand).strip().lower()
        for c in cols:
            if needle and needle in str(c).strip().lower():
                return c
    return None


def norm_code(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"[^0-9]", "", s)
    return digits.zfill(6) if digits else s


def norm_year(x: Any) -> Any:
    if pd.isna(x):
        return pd.NA
    try:
        y = int(float(x))
        if 1900 <= y <= 2100:
            return y
    except Exception:
        pass
    m = re.search(r"(19|20)\d{2}", str(x))
    return int(m.group(0)) if m else pd.NA


def read_excel_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="calamine")
    except Exception:
        return pd.read_excel(path)


def infer_statement_type(path: Path) -> str | None:
    name = path.name.lower()
    for canon, aliases in STATEMENT_ALIASES.items():
        if any(str(a).lower() in name for a in aliases):
            return canon
    return None


def raw_statement_files(raw_all_dir: Path) -> list[Path]:
    files = []
    for p in raw_all_dir.rglob("*.xlsx"):
        if p.name.startswith("~$"):
            continue
        if "재무비율" in p.name:
            continue
        if infer_statement_type(p) is not None:
            files.append(p)
    return sorted(files)


def build_statement_items(raw_all_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = raw_statement_files(raw_all_dir)
    if not files:
        raise FileNotFoundError(f"No statement Excel files found under {raw_all_dir}")
    frames: list[pd.DataFrame] = []
    file_rows: list[dict[str, Any]] = []
    for fp in files:
        stype = infer_statement_type(fp)
        print(f"[stage0] reading statement file: {fp.name} ({stype})", flush=True)
        try:
            df = read_excel_any(fp)
        except Exception as exc:
            file_rows.append({"file": fp.name, "statement_type": stype, "status": "read_error", "error": repr(exc)})
            continue
        if df.empty:
            file_rows.append({"file": fp.name, "statement_type": stype, "rows_raw": 0, "rows_long": 0, "status": "empty"})
            continue
        assert_no_mojibake_columns(df, f"raw statement {fp.name}")
        code_col = find_col(list(df.columns), [K_CODE, "종목코드", "회사코드", "stock_code", "code"])
        year_col = find_col(list(df.columns), [K_YEAR, "사업연도", "결산년도", "fiscal_year", "year"])
        name_col = find_col(list(df.columns), [K_NAME, "기업명", "종목명", "firm_name"])
        market_col = find_col(list(df.columns), [K_MARKET, "market"])
        if code_col is None or year_col is None:
            file_rows.append({"file": fp.name, "statement_type": stype, "rows_raw": int(len(df)), "rows_long": 0, "status": "missing_code_or_year"})
            continue
        id_cols = {code_col, year_col}
        if name_col is not None: id_cols.add(name_col)
        if market_col is not None: id_cols.add(market_col)
        work_ids = pd.DataFrame({
            "stock_code": df[code_col].map(norm_code),
            "fiscal_year": df[year_col].map(norm_year),
        })
        if name_col is not None:
            work_ids["firm_name_raw"] = df[name_col].astype("string")
        if market_col is not None:
            work_ids["market"] = df[market_col].astype("string")
        valid = (work_ids["stock_code"] != "") & work_ids["fiscal_year"].notna()
        work_ids = work_ids.loc[valid].copy()
        value_cols = []
        for c in df.columns:
            if c in id_cols:
                continue
            cs = str(c).strip()
            if not cs or cs.lower().startswith("unnamed"):
                continue
            s = pd.to_numeric(df.loc[valid, c], errors="coerce")
            if int(s.notna().sum()) == 0:
                continue
            value_cols.append(c)
        long_frames = []
        for c in value_cols:
            s = pd.to_numeric(df.loc[valid, c], errors="coerce")
            tmp = work_ids.copy()
            tmp["statement_type"] = stype
            tmp["item_code"] = str(c).strip()
            tmp["item_name_raw"] = str(c).strip()
            tmp["value_numeric"] = s.to_numpy()
            tmp = tmp[tmp["value_numeric"].notna()]
            if not tmp.empty:
                long_frames.append(tmp)
        if long_frames:
            long = pd.concat(long_frames, ignore_index=True, sort=False)
            frames.append(long)
            rows_long = int(len(long))
        else:
            rows_long = 0
        file_rows.append({"file": fp.name, "statement_type": stype, "rows_raw": int(len(df)), "numeric_item_columns": int(len(value_cols)), "rows_long": rows_long, "status": "ok"})
    if not frames:
        raise ValueError(f"Statement workbooks were found but no numeric statement items could be parsed under {raw_all_dir}")
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = out.drop_duplicates(["stock_code", "fiscal_year", "statement_type", "item_code"], keep="last")
    return out, {"files": file_rows, "rows": int(len(out)), "statement_counts": out["statement_type"].value_counts(dropna=False).to_dict()}


def write_stage0_canonical(sample: pd.DataFrame, statement_items: pd.DataFrame, stage0_dir: Path, counts: dict[str, Any], stmt_meta: dict[str, Any]) -> dict[str, Any]:
    can = stage0_dir / "canonical_panel"
    can.mkdir(parents=True, exist_ok=True)
    panel = sample.copy()
    panel[K_CODE] = panel[K_CODE].map(norm_code)
    panel["year"] = panel["year"].map(norm_year).astype("Int64")
    panel[K_YEAR] = panel["year"].astype("Int64").astype("string")
    rename = {
        "평가일": "rating__평가일",
        "평가사구분": "rating__평가사구분",
        "평가사명": "rating__평가사명",
        "증권구분": "rating__증권구분",
        "증권명": "rating__증권명",
        "agency_pri": "rating__agency_pri",
        "_source_file": "rating__source_file",
        "grade_base": "rating__rating",
        "grade_base_raw": "rating__grade_base_raw",
        "grade_base_notch": "rating__grade_base_notch",
        "rating_num_notch": "rating__rating_num_notch",
        "grade_base_10": "rating__grade_base_10",
        "rating_num_10": "rating__rating_num_10",
        "grade_base_7": "rating__grade_base_7",
        "rating_num_7": "rating__rating_num_7",
    }
    for src, dst in rename.items():
        if src in panel.columns and dst not in panel.columns:
            panel[dst] = panel[src]
    # Explicit aliases required by Stage boundary contract.
    panel["grade_base_raw"] = panel.get("rating__grade_base_raw", panel.get("rating__rating"))
    panel["grade_base_notch"] = panel.get("rating__grade_base_notch", panel.get("rating__rating"))
    panel["rating_num_notch"] = pd.to_numeric(panel.get("rating__rating_num_notch"), errors="coerce").astype("Int64")
    panel["grade_base_10"] = panel.get("rating__grade_base_10")
    panel["rating_num_10"] = pd.to_numeric(panel.get("rating__rating_num_10"), errors="coerce").astype("Int64")
    panel["grade_base_7"] = panel.get("rating__grade_base_7")
    panel["rating_num_7"] = pd.to_numeric(panel.get("rating__rating_num_7"), errors="coerce").astype("Int64")
    panel["rating"] = panel["grade_base_notch"]
    panel["rating_num"] = panel["rating_num_10"]
    panel["has_rating"] = panel["rating_num_10"].notna()
    if panel.duplicated([K_CODE, "year"]).any():
        raise ValueError("Stage0 builder produced duplicate firm-year rows after rating sample dedup")
    panel_path = can / "stage0_canonical_panel.parquet"
    stmt_path = can / "statement_items_panel.parquet"
    tmp_panel = can / "stage0_canonical_panel.tmp.parquet"
    tmp_stmt = can / "statement_items_panel.tmp.parquet"
    panel.to_parquet(tmp_panel, index=False)
    statement_items.to_parquet(tmp_stmt, index=False)
    tmp_panel.replace(panel_path)
    tmp_stmt.replace(stmt_path)
    meta = {
        "stage_name": "stage0_canonical_foundation",
        "contract_version": "stage0_rating_sampling_v32",
        "status": "PASS",
        "created_utc": now(),
        "output_paths": {"canonical_panel": str(panel_path), "statement_items_panel": str(stmt_path)},
        "row_counts": {"canonical_panel": int(len(panel)), "statement_items_panel": int(len(statement_items))},
        "key_columns": [K_CODE, "year"],
        "duplicate_key_count": int(panel.duplicated([K_CODE, "year"]).sum()),
        "rating_sample_counts": counts,
        "statement_items": stmt_meta,
        "schema_version": "stage0_canonical_foundation_v1",
    }
    write_json(can / "stage0_manifest.json", meta)
    write_json(stage0_dir / "stage0_manifest.json", meta)
    return meta


def build_stage0_foundation(project_root: Path, raw_all_dir: Path, raw_rating_dir: Path, stage0_dir: Path, clean: bool = False) -> dict[str, Any]:
    if clean and stage0_dir.exists():
        import shutil
        shutil.rmtree(stage0_dir)
    print(f"[stage0] raw_all_dir={raw_all_dir}", flush=True)
    print(f"[stage0] raw_rating_dir={raw_rating_dir}", flush=True)
    sample, counts = build_rating_sample(raw_rating_dir)
    print(f"[stage0] rating firm-year rows={len(sample):,}", flush=True)
    stmt, stmt_meta = build_statement_items(raw_all_dir)
    print(f"[stage0] statement item rows={len(stmt):,}", flush=True)
    meta = write_stage0_canonical(sample, stmt, stage0_dir, counts, stmt_meta)
    ok, errors, val = validate_stage0_contract(stage0_dir)
    validation = {"stage_name": "stage0_validation", "status": "PASS" if ok else "FAIL", "errors": errors, "metadata": val, "created_utc": now()}
    write_json(stage0_dir / "stage0_validation.json", validation)
    write_json(stage0_dir / "canonical_panel" / "stage0_validation.json", validation)
    if not ok:
        raise ValueError("Fresh Stage0 validation failed: " + "; ".join(errors))
    meta["validation"] = val
    return meta


def main(argv: list[str] | None = None) -> int:
    configure_utf8_stdio()
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--raw-all-dir", default=None)
    ap.add_argument("--raw-rating-dir", default=None)
    ap.add_argument("--stage0-dir", default=None)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.project_root).resolve()
    raw_all = Path(args.raw_all_dir).resolve() if args.raw_all_dir else root / "data" / "raw" / "raw_all"
    raw_rating = Path(args.raw_rating_dir).resolve() if args.raw_rating_dir else root / "data" / "raw" / "rating_sample"
    stage0 = Path(args.stage0_dir).resolve() if args.stage0_dir else root / "data" / "final_freeze" / "stage0_oracle_foundation"
    meta = build_stage0_foundation(root, raw_all, raw_rating, stage0, clean=args.clean)
    print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
