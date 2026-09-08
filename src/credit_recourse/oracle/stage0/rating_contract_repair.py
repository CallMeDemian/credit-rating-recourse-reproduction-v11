from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.oracle.contracts.rating_scale import (
    AGENCY_NAME,
    AGENCY_PRIORITY,
    ALLOWED_MAIN_AGENCY_CODES,
    ICR_ALLOWED_SECURITY_CODE,
    add_rating_scale_columns,
)

KEY_KR = "거래소코드"
YEAR_KR = "회계년도"
RATING_KR = "신용등급"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_col(cols: list[str], candidates: list[str]) -> str | None:
    cols = list(cols)
    lowered = {str(c).strip().lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        hit = lowered.get(str(cand).strip().lower())
        if hit is not None:
            return hit
    for cand in candidates:
        c_low = str(cand).strip().lower()
        for col in cols:
            if c_low and c_low in str(col).strip().lower():
                return col
    return None


def _norm_code(x: Any) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"[^0-9]", "", s)
    return digits.zfill(6) if digits else s


def _norm_year(x: Any) -> Any:
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


def _read_excel_any(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path, engine="calamine")
    except Exception:
        return pd.read_excel(path)


def _raw_rating_files(raw_dir: Path) -> list[Path]:
    """Find raw rating workbooks recursively.

    Final data may keep KOSPI/KOSDAQ and KONEX workbooks either directly under
    ``data/raw/rating_sample`` or in subfolders such as ``kospi_kosdaq`` and
    ``konex_optional``.  Use recursive discovery so Stage0 rebuild does not
    silently miss KONEX or market-specific folders.
    """
    patterns = ["*신용평가에 관한 사항*.xlsx", "*신용평가*.xlsx", "*rating*.xlsx"]
    files: list[Path] = []
    for pat in patterns:
        files.extend(raw_dir.glob(pat))
        files.extend(raw_dir.rglob(pat))
    out = sorted({f.resolve(): f for f in files if f.is_file() and not f.name.startswith("~$")}.values())
    return out


def build_rating_sample(raw_rating_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    files = _raw_rating_files(raw_rating_dir)
    if not files:
        raise FileNotFoundError(f"No raw rating Excel files found under {raw_rating_dir}")

    frames = []
    file_rows = []
    required_candidates = {
        "code": [KEY_KR, "종목코드", "회사코드", "stock_code", "code"],
        "year": [YEAR_KR, "사업연도", "결산년도", "fiscal_year", "year"],
        "rating": [RATING_KR, "등급", "평가등급", "rating", "credit_rating"],
        "security_code": ["증권구분", "security_type_code"],
        "agency_code": ["평가사구분", "agency_code"],
        "eval_date": ["평가일", "평정일", "evaluation_date", "rating_date"],
    }

    for fp in files:
        raw = _read_excel_any(fp)
        if raw.empty:
            file_rows.append({"file": fp.name, "rows_raw": 0, "rows_used": 0, "status": "empty"})
            continue
        cols = list(raw.columns)
        found = {k: _find_col(cols, v) for k, v in required_candidates.items()}
        missing = [k for k, v in found.items() if v is None]
        if missing:
            file_rows.append({"file": fp.name, "rows_raw": len(raw), "rows_used": 0, "status": f"missing:{missing}"})
            continue
        name_col = _find_col(cols, ["회사명", "기업명", "종목명", "firm_name", "company"])
        market_col = _find_col(cols, ["시장", "market"])
        sec_name_col = _find_col(cols, ["증권명", "security_name"])
        agency_name_col = _find_col(cols, ["평가사명", "agency", "agency_name"])

        df = pd.DataFrame({
            "거래소코드": raw[found["code"]].map(_norm_code),
            "year": raw[found["year"]].map(_norm_year),
            "신용등급": raw[found["rating"]],
            "증권구분": pd.to_numeric(raw[found["security_code"]], errors="coerce"),
            "평가사구분": pd.to_numeric(raw[found["agency_code"]], errors="coerce"),
            "평가일": pd.to_datetime(raw[found["eval_date"]], errors="coerce"),
            "_source_file": fp.name,
        })
        if name_col:
            df["회사명"] = raw[name_col].astype("string")
        if market_col:
            df["시장"] = raw[market_col].astype("string")
        if sec_name_col:
            df["증권명"] = raw[sec_name_col].astype("string")
        if agency_name_col:
            df["평가사명_raw"] = raw[agency_name_col].astype("string")
        frames.append(df)
        file_rows.append({"file": fp.name, "rows_raw": len(raw), "rows_used": len(df), "status": "read"})

    if not frames:
        raise ValueError(f"Raw rating files were found but none had the required columns: {raw_rating_dir}")

    raw_all = pd.concat(frames, ignore_index=True, sort=False)
    counts: dict[str, Any] = {"raw_all": int(len(raw_all)), "files": file_rows}

    df = raw_all[raw_all["증권구분"].eq(ICR_ALLOWED_SECURITY_CODE)].copy()
    counts["icr_security_code_40"] = int(len(df))
    df = df[df["평가사구분"].isin(ALLOWED_MAIN_AGENCY_CODES)].copy()
    counts["allowed_agency_codes_10_20_30_60_90"] = int(len(df))
    df = df[df["신용등급"].notna()].copy()
    counts["rating_notna"] = int(len(df))

    df = add_rating_scale_columns(df.rename(columns={"신용등급": "grade_base"}), source_col="grade_base")
    df = df[df["rating_num_10"].notna()].copy()
    counts["valid_long_term_grade"] = int(len(df))

    df = df[(df["거래소코드"] != "") & df["year"].notna()].copy()
    df["year"] = df["year"].astype(int)
    df["평가사구분"] = pd.to_numeric(df["평가사구분"], errors="coerce").astype("Int64")
    df["증권구분"] = pd.to_numeric(df["증권구분"], errors="coerce").astype("Int64")
    df["평가사명"] = df["평가사구분"].map(AGENCY_NAME).astype("string")
    df["agency_pri"] = df["평가사구분"].map(AGENCY_PRIORITY).fillna(99).astype(int)

    before_dedup = len(df)
    sample = (
        df.sort_values(["거래소코드", "year", "agency_pri", "평가일"], ascending=[True, True, True, False])
        .drop_duplicates(["거래소코드", "year"], keep="first")
        .copy()
    )
    counts["before_dedup"] = int(before_dedup)
    counts["dedup_firm_year"] = int(len(sample))
    counts["grade_distribution_10"] = sample["grade_base_10"].value_counts(dropna=False).to_dict()
    counts["agency_distribution"] = sample["평가사구분"].astype("string").value_counts(dropna=False).to_dict()

    keep = [
        "거래소코드", "year", "회사명", "시장", "평가일", "평가사구분", "평가사명",
        "증권명", "증권구분", "grade_base", "grade_base_raw", "grade_base_notch",
        "rating_num_notch", "grade_base_10", "rating_num_10", "grade_base_7", "rating_num_7",
        "agency_pri", "_source_file",
    ]
    keep = [c for c in keep if c in sample.columns]
    return sample[keep].copy(), counts


def _canonical_columns(panel_path: Path) -> dict[str, str | None]:
    cols = list(pd.read_parquet(panel_path).columns)
    return {
        "code": _find_col(cols, ["stock_code", "financial__stock_code", "rating__stock_code", "code", "거래소코드", "corp_code", "firm_id"]),
        "year": _find_col(cols, ["fiscal_year", "year", "회계년도"]),
        "rating": _find_col(cols, ["rating__rating", "rating", "grade_base", "신용등급"]),
        "security_code": _find_col(cols, ["rating__증권구분", "증권구분", "security_type_code"]),
        "agency_code": _find_col(cols, ["rating__평가사구분", "평가사구분", "agency_code"]),
        "eval_date": _find_col(cols, ["rating__평가일", "평가일", "evaluation_date"]),
    }


def validate_stage0_contract(stage0_dir: Path) -> tuple[bool, list[str], dict[str, Any]]:
    panel_path = stage0_dir / "canonical_panel" / "stage0_canonical_panel.parquet"
    if not panel_path.exists():
        return False, [f"missing canonical panel: {panel_path}"], {}
    found = _canonical_columns(panel_path)
    errors = []
    for k in ["code", "year", "rating", "security_code", "agency_code", "eval_date"]:
        if found.get(k) is None:
            errors.append(f"stage0 canonical panel missing required rating metadata column: {k}")
    if errors:
        return False, errors, {"inferred_columns": found}

    cols = [v for v in found.values() if v]
    df = pd.read_parquet(panel_path, columns=cols)
    work = pd.DataFrame({
        "거래소코드": df[found["code"]].map(_norm_code),
        "year": df[found["year"]].map(_norm_year),
        "grade_base": df[found["rating"]],
        "증권구분": pd.to_numeric(df[found["security_code"]], errors="coerce"),
        "평가사구분": pd.to_numeric(df[found["agency_code"]], errors="coerce"),
        "평가일": pd.to_datetime(df[found["eval_date"]], errors="coerce"),
    })
    bad_sec = int((work["증권구분"].notna() & ~work["증권구분"].eq(ICR_ALLOWED_SECURITY_CODE)).sum())
    bad_ag = int((work["평가사구분"].notna() & ~work["평가사구분"].isin(ALLOWED_MAIN_AGENCY_CODES)).sum())
    scaled = add_rating_scale_columns(work, source_col="grade_base")
    bad_grade = int(scaled["rating_num_10"].isna().sum())
    dup = int(work.duplicated(["거래소코드", "year"]).sum())
    if bad_sec:
        errors.append(f"non-ICR 증권구분 rows in Stage0 canonical: {bad_sec}")
    if bad_ag:
        errors.append(f"disallowed 평가사구분 rows in Stage0 canonical: {bad_ag}")
    if bad_grade:
        errors.append(f"invalid/non-long-term rating rows in Stage0 canonical: {bad_grade}")
    if dup:
        errors.append(f"duplicate firm-year rows in Stage0 canonical: {dup}")
    meta = {
        "inferred_columns": found,
        "rows": int(len(work)),
        "bad_security_rows": bad_sec,
        "bad_agency_rows": bad_ag,
        "bad_grade_rows": bad_grade,
        "duplicate_firm_year_rows": dup,
    }
    return len(errors) == 0, errors, meta


def repair_stage0_canonical(stage0_dir: Path, raw_rating_dir: Path) -> dict[str, Any]:
    panel_path = stage0_dir / "canonical_panel" / "stage0_canonical_panel.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing Stage0 canonical panel: {panel_path}")
    sample, counts = build_rating_sample(raw_rating_dir)
    panel = pd.read_parquet(panel_path)
    cols = _canonical_columns(panel_path)
    code_col, year_col = cols.get("code"), cols.get("year")
    if code_col is None or year_col is None:
        raise KeyError(f"Cannot infer canonical code/year columns for repair: {cols}")

    panel["__repair_code"] = panel[code_col].map(_norm_code)
    panel["__repair_year"] = panel[year_col].map(_norm_year)

    # A stale Stage0 built before the final rating-sampling contract can already
    # contain duplicate firm-year rows.  Repair must collapse the base panel before
    # merging the repaired rating sample; otherwise a correctly de-duplicated
    # rating sample is multiplied back into duplicate canonical rows.
    panel["__nonnull_count"] = panel.notna().sum(axis=1)
    sort_cols = ["__repair_code", "__repair_year", "__nonnull_count"]
    panel = (
        panel.sort_values(sort_cols, ascending=[True, True, False])
        .drop_duplicates(["__repair_code", "__repair_year"], keep="first")
        .drop(columns=["__nonnull_count"], errors="ignore")
        .copy()
    )
    sample_key = sample.rename(columns={"거래소코드": "__repair_code", "year": "__repair_year"})

    # Drop legacy rating columns that violate the final contract, then replace from repaired sample.
    drop_prefixes = ("rating__",)
    drop_exact = {"rating", "grade_base", "rating_num", "증권구분", "평가사구분", "평가일", "증권명", "평가사명"}
    drop_cols = [c for c in panel.columns if str(c).startswith(drop_prefixes) or str(c) in drop_exact]
    panel = panel.drop(columns=drop_cols, errors="ignore")

    rename_map = {
        "회사명": "rating__firm_name_raw",
        "시장": "rating__market",
        "평가일": "rating__평가일",
        "평가사구분": "rating__평가사구분",
        "평가사명": "rating__평가사명",
        "증권명": "rating__증권명",
        "증권구분": "rating__증권구분",
        "grade_base_notch": "rating__rating",
        "grade_base_raw": "rating__grade_base_raw",
        "grade_base_10": "rating__grade_base_10",
        "rating_num_10": "rating__rating_num_10",
        "grade_base_7": "rating__grade_base_7",
        "rating_num_7": "rating__rating_num_7",
        "rating_num_notch": "rating__rating_num_notch",
        "agency_pri": "rating__agency_pri",
        "_source_file": "rating__source_file",
    }
    sample_cols = ["__repair_code", "__repair_year"] + [c for c in rename_map if c in sample_key.columns]
    sample_key = sample_key[sample_cols].rename(columns=rename_map)
    merged = panel.merge(sample_key, on=["__repair_code", "__repair_year"], how="inner")
    # Enforce the canonical one-row-per-firm-year invariant after merge as well.
    # This protects against accidental duplicate raw keys or legacy canonical aliases.
    merged = (
        merged.sort_values(["__repair_code", "__repair_year", "rating__agency_pri", "rating__평가일"], ascending=[True, True, True, False])
        .drop_duplicates(["__repair_code", "__repair_year"], keep="first")
        .copy()
    )
    merged = merged.drop(columns=["__repair_code", "__repair_year"], errors="ignore")

    # Preserve common unprefixed aliases for old Stage00-01 readers while keeping explicit scale columns.
    if "rating__rating" in merged.columns:
        merged["grade_base"] = merged["rating__grade_base_10"]
        merged["rating"] = merged["rating__rating"]
    if "rating__rating_num_10" in merged.columns:
        merged["rating_num_10"] = pd.to_numeric(merged["rating__rating_num_10"], errors="coerce").astype("Int64")
        merged["rating_num"] = merged["rating_num_10"]
    for c in ["grade_base_10", "grade_base_7", "rating_num_7", "rating_num_notch", "grade_base_notch"]:
        rc = f"rating__{c}"
        if rc in merged.columns:
            merged[c] = merged[rc]

    if merged.empty:
        raise ValueError("Stage0 rating repair produced empty canonical panel; check raw rating keys vs canonical keys")

    backup = panel_path.with_suffix(panel_path.suffix + f".bak_rating_contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    panel_path.replace(backup)
    merged.to_parquet(panel_path, index=False)

    ok, errors, val_meta = validate_stage0_contract(stage0_dir)
    if not ok:
        raise ValueError("Stage0 repair validation failed: " + "; ".join(errors))

    meta = {
        "stage": "stage0_rating_contract_repair",
        "status": "PASS",
        "created_utc": _now(),
        "raw_rating_dir": str(raw_rating_dir),
        "stage0_dir": str(stage0_dir),
        "backup": str(backup),
        "rows_after_repair": int(len(merged)),
        "rating_sample_counts": counts,
        "validation": val_meta,
        "contract": "ICR security code 40 + agency codes 10/20/30/60/90 + long-term grades + agency priority/latest evaluation-date dedup + explicit rating_num_10/7/notch scales",
    }
    out_meta = stage0_dir / "canonical_panel" / "stage0_rating_contract_repair_metadata.json"
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--stage0-dir", default=None)
    p.add_argument("--raw-rating-dir", default=None)
    p.add_argument("--force", action="store_true", help="Repair even if Stage0 currently validates.")
    args = p.parse_args(argv)

    root = Path(args.project_root).resolve()
    stage0_dir = Path(args.stage0_dir).resolve() if args.stage0_dir else root / "data" / "final_freeze" / "stage0_oracle_foundation"
    raw_rating_dir = Path(args.raw_rating_dir).resolve() if args.raw_rating_dir else root / "data" / "raw" / "rating_sample"

    ok, errors, meta = validate_stage0_contract(stage0_dir)
    if ok and not args.force:
        result = {"stage": "stage0_rating_contract_repair", "status": "SKIP_ALREADY_VALID", "created_utc": _now(), "validation": meta}
    else:
        result = repair_stage0_canonical(stage0_dir, raw_rating_dir)
        result["pre_repair_errors"] = errors
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
