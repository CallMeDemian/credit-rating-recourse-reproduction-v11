"""
Task 9: Growth Candidate Audit + 3-Year Simple Growth Expansion
================================================================
Stage 2 pipeline의 optional post-quality-filter task.
Task 8(_task8_quality_filter.py) 완료 후 실행.

설계 원칙:
- STAGE2_OUT의 파일을 read-only로 읽고, growth_audit/ 하위 폴더에만 쓴다.
- engineered_financial_ratios.parquet / candidate_ratio_pool_by_item.csv /
  ratio_quality_report.csv 는 절대 덮어쓰지 않는다.
- main_eligible은 실제 데이터 품질(missing_rate, n_dev, unique_count, inf_rate)로
  계산한다. pool 정의에 hardcode하지 않는다.
- sector-relative 변수는 sector_7이 Stage 2 범위 밖(Stage 1C 산출물)이므로
  Task 9에서 생성하지 않는다.

입력 환경변수 (stage2_pipeline.py가 설정):
  STAGE2_OUT        : Stage 2 results_dir
  STAGE2_S1B_PANEL  : firm_year_panel_v1.parquet
  STAGE2_S1B_CLEAN  : cleaned statement panels 디렉토리

보호 파일 (절대 덮어쓰기 금지):
  engineered_financial_ratios.parquet
  candidate_ratio_pool_by_item.csv
  ratio_quality_report.csv

산출물 (STAGE2_OUT/growth_audit/ 하위):
  growth_audit_existing.csv        기존 성장성 ratio audit
  growth_audit_new.csv             신규 GX ratio audit
  engineered_financial_ratios_growth_expanded.parquet  기존 + GX 합본
  candidate_ratio_pool_growth_expanded.csv  full pool (성장성 교체)
  growth_audit_report.md           요약 리포트
"""
import os
from typing import Optional
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 환경변수
# ─────────────────────────────────────────────────────────────
_out = os.environ.get("STAGE2_OUT")
_panel = os.environ.get("STAGE2_S1B_PANEL")
_clean = os.environ.get("STAGE2_S1B_CLEAN")

if not _out:
    raise EnvironmentError(
        "[Task 9] STAGE2_OUT 환경변수 없음.\n"
        "stage2_pipeline.py 를 통해 실행하세요."
    )

OUT   = Path(_out)
AUDIT = OUT / "growth_audit"
AUDIT.mkdir(parents=True, exist_ok=True)

PROTECTED = {
    "engineered_financial_ratios.parquet",
    "candidate_ratio_pool_by_item.csv",
    "ratio_quality_report.csv",
}

# ─────────────────────────────────────────────────────────────
# 품질 기준 (stage_config에서 읽을 수 없으면 default 사용)
# ─────────────────────────────────────────────────────────────
MISS_MAX   = 0.30
INF_MAX    = 0.01
ZERO_MAX   = 0.05
MIN_N_DEV  = 100
MIN_UNIQUE = 2     # strictly greater than

print("\n" + "=" * 72)
print("  Task 9: Growth Candidate Audit + 3-Year Simple Growth Expansion")
print("=" * 72)

# STAGE00_02_TASK9_MERGE_KEY_NORMALIZATION_2026_05_24

# ─────────────────────────────────────────────────────────────
# Merge-key normalization helpers
# ─────────────────────────────────────────────────────────────
def _normalize_merge_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize firm-year merge keys before every merge.

    Stage00_02 can receive `거래소코드` as pandas Int64 from panel artifacts and
    as string/object from ratio artifacts or clean statement panels.  Pandas
    refuses to merge mixed string/Int64 keys, so Task9 must canonicalize both
    sides before any merge.  This is intentionally narrow: it only touches the
    two merge keys and preserves all analytical columns as-is.
    """
    out = df.copy()
    if "거래소코드" in out.columns:
        code = (
            out["거래소코드"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.replace(r"\D", "", regex=True)
        )
        out["거래소코드"] = code.where(code.str.len() > 0, pd.NA).str.zfill(6)
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    elif "회계년도" in out.columns:
        out["year"] = pd.to_numeric(out["회계년도"].astype(str).str[:4], errors="coerce").astype("Int64")
    return out


# ─────────────────────────────────────────────────────────────
# STEP 0  입력 로드
# ─────────────────────────────────────────────────────────────
print("\n[0] 입력 로드 ...")

for fname in ["engineered_financial_ratios.parquet",
              "candidate_ratio_pool_by_item.csv",
              "ratio_quality_report.csv"]:
    if not (OUT / fname).exists():
        raise FileNotFoundError(
            f"[Task 9] {fname} 없음 ({OUT / fname})\n"
            "Task 8이 정상 완료됐는지 확인하세요."
        )

ratios = pd.read_parquet(OUT / "engineered_financial_ratios.parquet")
ratios = _normalize_merge_keys(ratios)
pool   = pd.read_csv(OUT / "candidate_ratio_pool_by_item.csv")
qr     = pd.read_csv(OUT / "ratio_quality_report.csv")
print(f"  ratios: {ratios.shape}")
print(f"  pool  : {pool.shape}")
print(f"  qr    : {qr.shape}")

# panel (split / rating_num)
panel = None
if _panel and Path(_panel).exists():
    panel = pd.read_parquet(_panel)
    if "year" not in panel.columns:
        panel["year"] = panel["회계년도"].astype(str).str[:4]
    panel = _normalize_merge_keys(panel)
    print(f"  panel : {panel.shape}")
else:
    print("  panel : 없음 (dev split 계산 불가, 전체 데이터로 대체)")

# ─────────────────────────────────────────────────────────────
# split / rating_num merge (중복 방지)
# ─────────────────────────────────────────────────────────────
KEY = ["거래소코드", "year"]

if panel is not None:
    add_cols = ["split"]
    if "rating_num" not in ratios.columns:
        add_cols.append("rating_num")

    add_df = panel[KEY + [c for c in add_cols if c in panel.columns]].drop_duplicates()
    add_df = _normalize_merge_keys(add_df)
    ratios = _normalize_merge_keys(ratios)
    ratios = ratios.merge(add_df, on=KEY, how="left", validate="many_to_one")

    # 혹시 _x/_y suffix 생겼으면 정리
    for col in ["rating_num", "split"]:
        for suf in ["_x", "_y"]:
            if col + suf in ratios.columns:
                if col not in ratios.columns:
                    ratios.rename(columns={col + suf: col}, inplace=True)
                else:
                    ratios.drop(columns=[col + suf], inplace=True)

if "split" not in ratios.columns:
    # firm_year_panel에 split이 없으면 year 기준으로 직접 생성
    ratios["split"] = ratios["year"].apply(
        lambda y: "Dev" if y <= 2019 else "OOT"
    )

dev_mask = ratios["split"].str.lower().isin(["dev", "train"])
dev_r    = ratios[dev_mask].copy()
print(f"  Dev: {dev_mask.sum()}  OOT: {(~dev_mask).sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 1  기존 성장성 ratio audit
# ─────────────────────────────────────────────────────────────
print("\n[1] 기존 성장성 ratio audit ...")

id_col = "ratio_id" if "ratio_id" in pool.columns else "variable_id"
growth_pool = pool[pool["category"] == "성장성"].copy()
growth_ids  = growth_pool[id_col].tolist()
print(f"  기존 성장성 후보: {len(growth_ids)}개")

def audit_col(rid: str, df_all: pd.DataFrame, df_dev: pd.DataFrame) -> dict:
    if rid not in df_all.columns:
        return dict(ratio_id=rid, exists=False,
                    missing_rate_all=np.nan, missing_rate_dev=np.nan,
                    n_dev=0, unique_count=0,
                    inf_rate=np.nan, zero_denom_rate=np.nan,
                    denom_instability=False,
                    spearman_with_rating_num=np.nan,
                    main_eligible=False,
                    exclusion_reason="column not found")

    s_all = pd.to_numeric(df_all[rid], errors="coerce")
    s_dev = pd.to_numeric(df_dev[rid], errors="coerce") if len(df_dev) else s_all

    n_all = len(df_all); n_dev = s_dev.notna().sum()
    miss_all = float(s_all.isna().mean())
    miss_dev = float(1 - n_dev / max(len(df_dev), 1)) if len(df_dev) else miss_all
    inf_r    = float(np.isinf(s_all.fillna(0)).mean())
    s_clean  = s_all.dropna()
    unique_c = int(s_clean.nunique())

    p01 = float(s_clean.quantile(0.01)) if len(s_clean) else np.nan
    p99 = float(s_clean.quantile(0.99)) if len(s_clean) else np.nan
    denom_flag = (inf_r > INF_MAX
                  or (np.isfinite(p01) and abs(p01) > 100)
                  or (np.isfinite(p99) and p99 > 100))

    # Spearman with rating_num (Dev only)
    sp = np.nan
    if "rating_num" in df_dev.columns and n_dev >= 10:
        try:
            from scipy.stats import spearmanr
            cl = df_dev[[rid, "rating_num"]].copy()
            cl[rid] = pd.to_numeric(cl[rid], errors="coerce")
            cl = cl.dropna()
            if len(cl) >= 10:
                rho, _ = spearmanr(cl[rid], cl["rating_num"])
                sp = round(float(rho), 4)
        except Exception:
            pass

    ok = (miss_dev < MISS_MAX and not denom_flag
          and n_dev >= MIN_N_DEV and unique_c > MIN_UNIQUE)
    reasons = []
    if miss_dev >= MISS_MAX:   reasons.append(f"miss_dev={miss_dev:.1%}>={MISS_MAX:.0%}")
    if denom_flag:             reasons.append("denom_instability")
    if n_dev < MIN_N_DEV:     reasons.append(f"n_dev={n_dev}<{MIN_N_DEV}")
    if unique_c <= MIN_UNIQUE: reasons.append(f"unique={unique_c}<={MIN_UNIQUE}")

    return dict(
        ratio_id=rid, exists=True,
        missing_rate_all=round(miss_all, 4),
        missing_rate_dev=round(miss_dev, 4),
        n_dev=int(n_dev), unique_count=unique_c,
        inf_rate=round(inf_r, 4), denom_instability=denom_flag,
        p01=round(p01, 4) if np.isfinite(p01) else np.nan,
        p99=round(p99, 4) if np.isfinite(p99) else np.nan,
        spearman_with_rating_num=sp,
        main_eligible=ok,
        exclusion_reason="; ".join(reasons),
    )

existing_audit = pd.DataFrame([audit_col(rid, ratios, dev_r) for rid in growth_ids])
existing_audit.to_csv(AUDIT / "growth_audit_existing.csv",
                      index=False, encoding="utf-8-sig")
print(f"  existing audit → {len(existing_audit)} rows, "
      f"main_eligible={existing_audit['main_eligible'].sum()}")

# ─────────────────────────────────────────────────────────────
# STEP 2  신규 3-year simple growth 계산
# ─────────────────────────────────────────────────────────────
print("\n[2] 신규 3-year simple growth 계산 ...")

# cleaned panel 폴더에서 손익계산서 / 재무상태표 탐색
new_vars: dict[str, pd.Series] = {}
new_meta: list[dict] = []

CLEAN_DIR = Path(_clean) if _clean else None

def find_col_by_code(df: pd.DataFrame, code_fragment: str) -> Optional[str]:
    """KIS 코드 일부로 컬럼 탐색"""
    for c in df.columns:
        if code_fragment in str(c):
            return c
    return None

def load_clean(clean_dir: Path, name_fragment: str) -> Optional[pd.DataFrame]:
    """name_fragment를 파일명에 포함하는 clean parquet 로드"""
    if not clean_dir or not clean_dir.exists():
        return None
    matches = list(clean_dir.glob("*.parquet"))
    for p in matches:
        try:
            df = pd.read_parquet(p)
            # 컬럼으로 식별 (filename이 URL-encoded일 수 있음)
            if name_fragment in str(p) or any(name_fragment in c for c in df.columns):
                return df
        except Exception:
            continue
    return None

def three_year_growth(df: pd.DataFrame,
                      val_col: str,
                      gx_id: str,
                      gx_name: str) -> tuple[pd.Series, dict]:
    """
    3년 단순 성장률 = val_t / val_{t-3} - 1
    조건: val_{t-3} > 0 (분모 양수만)
    """
    key = ["거래소코드", "year"]
    sub = df[key + [val_col]].copy()
    sub = _normalize_merge_keys(sub)
    sub[val_col] = pd.to_numeric(sub[val_col], errors="coerce")
    sub = sub.dropna(subset=[val_col])
    sub = sub.sort_values(key)

    sub["val_lag3"] = sub.groupby("거래소코드")[val_col].shift(3)
    sub["gr"] = np.where(
        sub["val_lag3"] > 0,
        sub[val_col] / sub["val_lag3"] - 1,
        np.nan
    )
    # 극단값 cap: |growth| > 100 → NaN
    sub["gr"] = sub["gr"].where(sub["gr"].abs() <= 100, np.nan)

    merged = ratios[["거래소코드", "year"]].merge(
        sub[["거래소코드", "year", "gr"]], on=key, how="left"
    )
    series = merged["gr"]
    series.index = ratios.index

    info = dict(
        ratio_id=gx_id, ratio_name=gx_name, category="성장성",
        formula=f"({val_col}_t / {val_col}_t-3) - 1, denom>0, |gr|<=100",
        source_col=val_col,
    )
    return series, info


# ── 손익계산서에서 매출액 ────────────────────────────────────
inc_df = None
if CLEAN_DIR:
    for p in CLEAN_DIR.glob("*.parquet"):
        try:
            df_tmp = _normalize_merge_keys(pd.read_parquet(p))
            if find_col_by_code(df_tmp, "B100000000"):
                inc_df = df_tmp
                break
        except Exception:
            continue

if inc_df is not None:
    sales_col = find_col_by_code(inc_df, "B100000000")
    if sales_col:
        s, info = three_year_growth(inc_df, sales_col, "GX001", "매출액3년단순증가율")
        new_vars["GX001"] = s
        new_meta.append(info)
        print(f"  GX001 매출액3년단순증가율: n_valid={s.notna().sum()}")
    else:
        print("  GX001: 매출액 컬럼 탐색 실패")
else:
    print("  GX001: 손익계산서 clean 파일 없음 → 건너뜀")

# ── 재무상태표에서 총자산 / 자기자본 ─────────────────────────
bs_df = None
if CLEAN_DIR:
    for p in CLEAN_DIR.glob("*.parquet"):
        try:
            df_tmp = _normalize_merge_keys(pd.read_parquet(p))
            if find_col_by_code(df_tmp, "A100000000"):
                bs_df = df_tmp
                break
        except Exception:
            continue

if bs_df is not None:
    # 총자산
    assets_col = find_col_by_code(bs_df, "A100000000")
    if assets_col:
        s, info = three_year_growth(bs_df, assets_col, "GX002", "총자산3년단순증가율")
        new_vars["GX002"] = s
        new_meta.append(info)
        print(f"  GX002 총자산3년단순증가율: n_valid={s.notna().sum()}")

    # 자기자본
    equity_col = find_col_by_code(bs_df, "A600000000")
    if equity_col:
        s, info = three_year_growth(bs_df, equity_col, "GX003", "자기자본3년단순증가율")
        new_vars["GX003"] = s
        new_meta.append(info)
        print(f"  GX003 자기자본3년단순증가율: n_valid={s.notna().sum()}")
else:
    print("  GX002/GX003: 재무상태표 clean 파일 없음 → 건너뜀")

print(f"  신규 변수 계산 완료: {list(new_vars.keys())}")

# ─────────────────────────────────────────────────────────────
# STEP 3  신규 변수 품질 audit
# ─────────────────────────────────────────────────────────────
print("\n[3] 신규 변수 품질 audit ...")

# ratios_expanded: 기존 + 신규 컬럼 (index 정렬 필요)
ratios_exp = ratios.copy()
for gx_id, series in new_vars.items():
    ratios_exp[gx_id] = series.values

dev_exp = ratios_exp[dev_mask].copy()

new_audit_rows = []
for info in new_meta:
    gx_id = info["ratio_id"]
    row = audit_col(gx_id, ratios_exp, dev_exp)
    row["formula"] = info.get("formula", "")
    new_audit_rows.append(row)

new_audit = pd.DataFrame(new_audit_rows) if new_audit_rows else pd.DataFrame()
if len(new_audit):
    new_audit.to_csv(AUDIT / "growth_audit_new.csv",
                     index=False, encoding="utf-8-sig")
    print(f"  new audit → {len(new_audit)} rows, "
          f"main_eligible={new_audit['main_eligible'].sum()}")
else:
    print("  신규 변수 없음 (STAGE2_S1B_CLEAN 미설정 또는 컬럼 탐색 실패)")

# ─────────────────────────────────────────────────────────────
# STEP 4  engineered_financial_ratios_growth_expanded.parquet 저장
# ─────────────────────────────────────────────────────────────
print("\n[4] engineered_financial_ratios_growth_expanded.parquet 저장 ...")

# 보호 파일 덮어쓰기 방지
assert "engineered_financial_ratios.parquet" not in \
       str(AUDIT / "engineered_financial_ratios_growth_expanded.parquet")

ratios_exp.to_parquet(
    AUDIT / "engineered_financial_ratios_growth_expanded.parquet",
    index=False
)
print(f"  → {ratios_exp.shape}  "
      f"(기존 {ratios.shape[1]}열 + 신규 {len(new_vars)}열)")

# ─────────────────────────────────────────────────────────────
# STEP 5  candidate_ratio_pool_growth_expanded.csv
#         - 기존 pool 전체 유지
#         - 성장성 category만 v3.2 결과로 교체
# ─────────────────────────────────────────────────────────────
print("\n[5] candidate_ratio_pool_growth_expanded.csv 생성 ...")

# 기존 pool에서 성장성 제거 후 v3.2 후보 추가
pool_non_growth = pool[pool["category"] != "성장성"].copy()

# 기존 성장성 중 main_eligible=True
existing_eligible = existing_audit[existing_audit["main_eligible"]].copy()
existing_eligible_ids = existing_eligible["ratio_id"].tolist()

# 신규 GX main_eligible=True
new_eligible = new_audit[new_audit["main_eligible"]].copy() if len(new_audit) else pd.DataFrame()
new_eligible_ids = new_eligible["ratio_id"].tolist() if len(new_eligible) else []

growth_v32_rows = []

# 기존 eligible 유지 (pool 원본 정보 + eligible 상태 업데이트)
for _, r in growth_pool.iterrows():
    rid = r.get(id_col, "")
    is_elig = rid in existing_eligible_ids
    audit_row = existing_audit[existing_audit["ratio_id"] == rid]
    miss_dev = float(audit_row["missing_rate_dev"].iloc[0]) if len(audit_row) else np.nan
    excl = str(audit_row["exclusion_reason"].iloc[0]) if len(audit_row) else ""
    growth_v32_rows.append({
        "category": "성장성",
        id_col: rid,
        "ratio_name": r.get("ratio_name", rid),
        "priority": r.get("priority", ""),
        "missing_rate": miss_dev,
        "median": r.get("median", np.nan),
        "main_eligible_v3_2": is_elig,
        "is_new_gx": False,
        "exclusion_reason_v3_2": excl,
    })

# 신규 GX 추가
for info in new_meta:
    gx_id = info["ratio_id"]
    audit_row = new_audit[new_audit["ratio_id"] == gx_id] if len(new_audit) else pd.DataFrame()
    is_elig = gx_id in new_eligible_ids
    miss_dev = float(audit_row["missing_rate_dev"].iloc[0]) if len(audit_row) else np.nan
    excl = str(audit_row["exclusion_reason"].iloc[0]) if len(audit_row) else ""
    growth_v32_rows.append({
        "category": "성장성",
        id_col: gx_id,
        "ratio_name": info.get("ratio_name", gx_id),
        "priority": "v3.2_new",
        "missing_rate": miss_dev,
        "median": np.nan,
        "main_eligible_v3_2": is_elig,
        "is_new_gx": True,
        "exclusion_reason_v3_2": excl,
    })

growth_v32_df = pd.DataFrame(growth_v32_rows)
pool_expanded = pd.concat([pool_non_growth, growth_v32_df],
                           ignore_index=True, sort=False)
pool_expanded.to_csv(AUDIT / "candidate_ratio_pool_growth_expanded.csv",
                     index=False, encoding="utf-8-sig")

n_growth_elig = growth_v32_df["main_eligible_v3_2"].sum()
print(f"  → pool_expanded: {len(pool_expanded)} rows  "
      f"성장성 main_eligible v3.2: {n_growth_elig}")

# ─────────────────────────────────────────────────────────────
# STEP 6  growth_audit_report.md
# ─────────────────────────────────────────────────────────────
print("\n[6] growth_audit_report.md 작성 ...")

from datetime import datetime, timezone, timedelta
from typing import Optional
KST = timezone(timedelta(hours=9))
now = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")

# 기존 eligible 목록
exist_elig_list = "\n".join(
    f"- `{r['ratio_id']}` {r.get('ratio_name','')}: "
    f"miss_dev={r['missing_rate_dev']:.1%}, ρ={r['spearman_with_rating_num']}"
    for _, r in existing_audit[existing_audit["main_eligible"]].iterrows()
) or "  (없음)"

exist_notelig_list = "\n".join(
    f"- `{r['ratio_id']}` {r.get('ratio_name','')}: {r['exclusion_reason']}"
    for _, r in existing_audit[~existing_audit["main_eligible"]].iterrows()
) or "  (없음)"

new_elig_list = "\n".join(
    f"- `{r['ratio_id']}` {r.get('formula','')}: miss_dev={r['missing_rate_dev']:.1%}"
    for _, r in (new_eligible.iterrows() if len(new_eligible) else pd.DataFrame().iterrows())
) or "  (없음)"

report = f"""# Stage 2 v3.2 Growth Candidate Audit Report

_Generated: {now}_

## 1. 작업 요약

| 항목 | 내용 |
|---|---|
| 기존 성장성 후보 | {len(growth_ids)}개 |
| 기존 main_eligible | {existing_audit["main_eligible"].sum()}개 |
| 신규 GX 계산 | {len(new_vars)}개 |
| 신규 main_eligible | {len(new_eligible) if len(new_audit) else 0}개 |
| 덮어쓴 main 산출물 | 없음 (보호 파일 유지) |

## 2. 기존 성장성 — main_eligible=True

{exist_elig_list}

## 3. 기존 성장성 — 제외 사유

{exist_notelig_list}

## 4. 신규 GX 변수 — main_eligible=True

{new_elig_list}

## 5. 핵심 설계 원칙

- **분모 처리**: base_value ≤ 0 → NaN (stage_config growth_ratio.positive_base_only 정책 동일)
- **3-year simple growth**: CAGR 대신 단순 비율 (음수 base 문제 최소화)
- **Extreme cap**: |growth| > 100 → NaN
- **main_eligible 기준**: miss_dev < {MISS_MAX}, n_dev ≥ {MIN_N_DEV}, unique > {MIN_UNIQUE}, inf_rate ≤ {INF_MAX}
- **sector-relative**: sector_7은 Stage 1C 산출물이므로 Stage 2 Task 9에서 생성하지 않음

## 6. 산출물 위치

```
STAGE2_OUT/growth_audit/
  growth_audit_existing.csv                       기존 성장성 audit
  growth_audit_new.csv                            신규 GX audit
  engineered_financial_ratios_growth_expanded.parquet
  candidate_ratio_pool_growth_expanded.csv
  growth_audit_report.md
```

## 7. Stage 3 사용 방법

Stage 3 v3.2에서 성장성 후보 pool을 확장하려면:
- `growth_audit/candidate_ratio_pool_growth_expanded.csv`에서 `main_eligible_v3_2=True` 행 사용
- 재무비율 데이터는 `growth_audit/engineered_financial_ratios_growth_expanded.parquet` 사용
- 기존 main output(`candidate_ratio_pool_by_item.csv`)은 Stage 3 v2 호환용으로 유지
"""

(AUDIT / "growth_audit_report.md").write_text(report, encoding="utf-8")
print(f"  → growth_audit_report.md")

print(f"\n{'=' * 72}")
print(f"  Task 9 완료 → {AUDIT}")
print(f"  기존 main_eligible: {existing_audit['main_eligible'].sum()}개")
print(f"  신규 GX eligible  : {len(new_eligible) if len(new_audit) else 0}개")
print(f"  main 산출물 덮어쓰기: 없음 ✓")
print(f"{'=' * 72}")
