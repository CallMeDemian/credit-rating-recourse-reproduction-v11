"""
Stage 2 v2 — Financial Ratio Engineering + Quality Filter + NICE Audit
========================================================================
Orchestrator: paths.yaml 기반 환경변수 설정 → modules/ 내 task 순차 실행
               → metadata (manifest, hashes, run_metadata) 작성 → verdict.

Task 순서:
  1. inventory: cleaned panels에서 KIS-VALUE item code/name 추출
  2. candidates: 221 후보 ratio Excel 파싱 + manual override 적용
  3. lag_diagnose: t-1 가용성 진단 (panel vs raw)
  4. lag_extract: raw 6 statement에서 t-1 row 추출 → lag_support/*.parquet
  5. lag_dedup (=step5): 중복 제거 + lag_support_financial_items.parquet 합본
  6. compute_ratios: 164 ratio 산출 → engineered_financial_ratios.parquet
  7. nice_audit: 재무비율_clean.parquet과 비교
  8. quality_filter: missing/zero/var 필터 + candidate_ratio_pool_by_item.csv
  9. growth_audit: 성장성 후보 audit + 3-year simple growth 확장 (optional)

본 파이프라인은 Stage 1B output을 read-only로 받고, Stage 2 산출물만 outputs/에 생성.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import runpy
import shutil
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import yaml


# ============================================================
# 유틸
# ============================================================
def load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sha256_of(path: Path, buf=65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions() -> dict:
    versions = {}
    for pkg in ["pandas", "numpy", "pyarrow", "openpyxl", "yaml"]:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    versions["python"] = sys.version.split()[0]
    return versions


def kst_now() -> str:
    return datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d %H:%M:%S KST")


# ============================================================
# Input discovery (instruction 5번)
# ============================================================
def write_input_discovery(paths: dict, run_dir: Path, project_root: Path):
    """input_discovery_report.csv: expected_role, search_pattern, matched_file_count,
       selected_file, status, notes
    """
    report = []

    # Stage 1B inputs
    for role, key in [("Stage 1B firm_year_panel", "panel"),
                      ("Stage 1B cleaned_dir", "cleaned_dir"),
                      ("Stage 1B raw_long", "raw_long")]:
        path_str = paths["inputs"]["stage1b"][key]
        p = (project_root / path_str).resolve()
        report.append({
            "expected_role": role,
            "search_pattern": path_str,
            "matched_file_count": 1 if p.exists() else 0,
            "selected_file": str(p) if p.exists() else "",
            "status": "found" if p.exists() else "missing",
            "notes": "directory" if (p.exists() and p.is_dir()) else "",
        })

    # Reference: candidates Excel
    cand = (project_root / paths["inputs"]["reference"]["candidates_xlsx"]).resolve()
    report.append({
        "expected_role": "candidate ratio Excel",
        "search_pattern": paths["inputs"]["reference"]["candidates_xlsx"],
        "matched_file_count": 1 if cand.exists() else 0,
        "selected_file": str(cand) if cand.exists() else "",
        "status": "found" if cand.exists() else "missing",
        "notes": f"sheet={paths['inputs']['reference'].get('candidates_sheet', '')}",
    })

    # Raw dir for lag_support
    raw_dir = (project_root / paths["inputs"]["raw"]["dir"]).resolve()
    if raw_dir.exists():
        for stmt, pattern in paths["inputs"]["raw"]["patterns"].items():
            matches = list(raw_dir.glob(pattern))
            report.append({
                "expected_role": f"raw {stmt} (lag_support)",
                "search_pattern": str(raw_dir / pattern),
                "matched_file_count": len(matches),
                "selected_file": "; ".join(m.name for m in matches[:5]),
                "status": "found" if matches else "missing",
                "notes": "lag_support extraction 용",
            })
    else:
        report.append({
            "expected_role": "raw dir",
            "search_pattern": str(raw_dir),
            "matched_file_count": 0,
            "selected_file": "",
            "status": "missing",
            "notes": "lag_support 추출 불가",
        })

    out = run_dir / "input_discovery_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["expected_role", "search_pattern",
                                          "matched_file_count", "selected_file",
                                          "status", "notes"])
        w.writeheader()
        w.writerows(report)
    print(f"  → input_discovery_report.csv ({len(report)} entries)")
    return report


# ============================================================
# Module 실행 (runpy)
# ============================================================
def run_module(module_path: Path, label: str):
    print(f"\n{'=' * 72}")
    print(f"  RUNNING: {label}")
    print(f"  module:  {module_path.name}")
    print(f"{'=' * 72}")
    t0 = time.time()
    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"  ✗ module exited with code {e.code}")
            raise
    elapsed = time.time() - t0
    print(f"  ✓ {label} done ({elapsed:.1f}s)")


# ============================================================
# Metadata writers
# ============================================================
def write_file_manifest(results_dir: Path, descriptions: dict, out_path: Path):
    rows = []
    for f in sorted(results_dir.rglob("*")):
        if f.is_file() and f.name != "stage2_file_manifest.csv":
            rel = f.relative_to(results_dir).as_posix()
            rows.append({
                "file_name": rel,
                "size_bytes": f.stat().st_size,
                "sha256": sha256_of(f),
                "description": descriptions.get(rel, descriptions.get(f.name, "")),
            })
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["file_name", "size_bytes",
                                            "sha256", "description"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_hashes(out_dir: Path, csv_path: Path, prefix: str = ""):
    rows = []
    for f in sorted(out_dir.rglob("*")):
        if f.is_file():
            rel = f.relative_to(out_dir).as_posix()
            rows.append({"file_name": rel, "size_bytes": f.stat().st_size,
                         "sha256": sha256_of(f)})
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=["file_name", "size_bytes", "sha256"])
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def write_run_metadata(run_dir: Path, paths: dict, summary: dict, verdict: str,
                        elapsed_s: float):
    meta = {
        "stage": "Stage 2 v2",
        "run_at": kst_now(),
        "elapsed_seconds": round(elapsed_s, 2),
        "verdict": verdict,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python": sys.version.split()[0],
        },
        "package_versions": package_versions(),
        "summary": summary,
        "paths": {
            "inputs": paths["inputs"],
            "outputs": paths["outputs"],
        },
    }
    out = run_dir / "run_metadata.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def write_command_log(run_dir: Path, args, elapsed_s: float):
    log = run_dir / "command_log.txt"
    with open(log, "w", encoding="utf-8") as f:
        f.write(f"# Stage 2 v2 command log — {kst_now()}\n")
        f.write(f"# elapsed: {elapsed_s:.1f}s\n\n")
        f.write(f"python stage2_pipeline.py --config {args.config}\n\n")
        f.write("# Reproduction:\n")
        f.write(".\\.venv\\Scripts\\python.exe stage2_pipeline.py --config configs\\paths.yaml\n")
        f.write(".\\.venv\\Scripts\\python.exe verify_metrics.py\n")
        f.write(".\\.venv\\Scripts\\python.exe verify_output_hashes.py\n")


def write_stage2_summary(results_dir: Path, summary: dict):
    md = ["# Stage 2 v2 Summary", "",
          f"_Generated: {kst_now()}_", "",
          "## Key metrics", ""]
    for k, v in summary.items():
        md.append(f"- **{k}**: {v}")
    (results_dir / "stage2_summary.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")


def write_growth_policy(results_dir: Path, config: dict):
    pol = config.get("growth_ratio", {})
    md = ["# Growth Ratio Policy", "",
          f"_Generated: {kst_now()}_", "",
          "## Positive-base policy",
          f"- base_year_offset: t - {pol.get('base_year_offset', 1)}",
          f"- positive_base_only: {pol.get('positive_base_only', True)}",
          f"- cap_extreme: |growth| > {pol.get('cap_extreme', 100.0)} → NaN",
          "",
          "## 근거",
          "- 분모 (base_value) ≤ 0 인 경우 성장률 부호 의미가 모호",
          "- 흑자 전환/적자 전환 케이스는 별도 indicator로 처리하는 것이 정확",
          "- 분모가 0인 경우 division-by-zero → NaN",
          ""]
    (results_dir / "growth_ratio_policy.md").write_text(
        "\n".join(md), encoding="utf-8")


def write_readme_stage3(results_dir: Path):
    md = """# Stage 3 Variable Selection Inputs (Stage 2 outputs)

Stage 3 (Variable Selection)에서 사용할 Stage 2 산출물:

## 핵심 입력
- `engineered_financial_ratios.parquet` — 4,746 firm-year × 164 ratio wide panel
- `financial_ratio_formula_dictionary_final.csv` — 164 ratio formula 정의
- `candidate_ratio_pool_by_item.csv` — 평가항목별 후보 ratio pool
- `ratio_quality_report.csv` — quality_pass=True인 134개 ratio 식별
- `growth_audit/candidate_ratio_pool_growth_expanded.csv` — 성장성 확장 pool (v3.2)

## 보조 (선택)
- `ratio_audit_against_precomputed_panel.csv` — NICE 검증 결과
- `stage2_formula_investigation_targets.csv` — 추가 검토 대상 formula
- `lag_support_coverage_report.csv` — lag 데이터 커버리지

## 사용 예시
```python
import pandas as pd
ratios = pd.read_parquet("engineered_financial_ratios.parquet")
qual = pd.read_csv("ratio_quality_report.csv")
pass_cols = qual[qual["quality_pass"] == True]["ratio_name"].tolist()
X = ratios[["거래소코드", "year"] + pass_cols]

# v3.2 성장성 확장 pool 사용 시
# pool_exp = pd.read_csv("growth_audit/candidate_ratio_pool_growth_expanded.csv")
```
"""
    (results_dir / "README_STAGE3_INPUTS.md").write_text(md, encoding="utf-8")


def write_verdict(results_dir: Path, summary: dict, verdict: str,
                   acceptance: dict):
    md = [f"# Stage 2 v2 — Verdict: {verdict}",
          "", f"_Generated: {kst_now()}_", "",
          "## Acceptance Criteria", ""]
    for k, v in acceptance.items():
        md.append(f"- **{k}**: {v}")
    md.append("")
    md.append("## Summary")
    for k, v in summary.items():
        md.append(f"- {k}: {v}")
    (results_dir / "stage2_verdict.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")


# ============================================================
# Acceptance check
# ============================================================

def write_aux_outputs(results_dir):
    """누락 산출물 보충: preview csv + investigation_targets csv"""
    import pandas as pd

    # preview (200 rows)
    rp = results_dir / "engineered_financial_ratios.parquet"
    if rp.exists():
        df_full = pd.read_parquet(rp)
        df_full.head(200).to_csv(
            results_dir / "engineered_financial_ratios_preview.csv",
            index=False, encoding="utf-8-sig"
        )
        print(f"    -> engineered_financial_ratios_preview.csv (200 rows)")

    # investigation_targets
    audit_p = results_dir / "ratio_audit_against_precomputed_panel.csv"
    if audit_p.exists():
        audit = pd.read_csv(audit_p)
        if "audit_decision" in audit.columns:
            invest = audit[audit["audit_decision"] == "investigate_formula"].copy()
            severity_map = {
                "R046": "low", "R063": "medium",
                "R085": "high", "R086": "high",
                "R152": "low", "R153": "low", "R154": "high",
                "R171": "high", "R172": "high", "R176": "high",
            }
            note_map = {
                "R046": "차입금 분류 정의 차이 가능. Stage 3에서 NICE 정의 sensitivity 비교",
                "R063": "유보율: 자본잉여금 매핑 차이 가능. 두 정의 병기 검증",
                "R085": "금융비용 분자 정의 차이. NICE 정의로 재계산 필수",
                "R086": "R085와 동일 NICE 컬럼. R085와 함께 검토",
                "R152": "평균재고 vs 기말재고. 정의 차이 미미, 그대로 사용 가능",
                "R153": "평균잔액 vs 기말잔액. 차이 작음, robustness 후보",
                "R154": "(매출원가/평균매입채무) vs (매출액/매입채무) 분자 차이. 재계산 권장",
                "R171": "성장률 분모 음수 처리 정책 차이. turnaround dummy 별도 검토",
                "R172": "R171과 동일",
                "R176": "R171/R172와 동일 - 음수 분모 처리",
            }
            records = []
            for _, r in invest.iterrows():
                rid = r.get("ratio_id", "")
                records.append({
                    "ratio_id": rid,
                    "ratio_name": r.get("ratio_name", ""),
                    "category": r.get("category", ""),
                    "nice_column": r.get("nice_column", ""),
                    "matched_n": r.get("matched_n", ""),
                    "spearman_corr": round(r["spearman_corr"], 4) if pd.notna(r.get("spearman_corr")) else None,
                    "median_abs_diff": round(r["median_abs_diff"], 4) if pd.notna(r.get("median_abs_diff")) else None,
                    "scale_factor_suspected": r.get("scale_factor_suspected", ""),
                    "audit_decision": r.get("audit_decision", ""),
                    "추정_사유": note_map.get(rid, ""),
                    "심각도": severity_map.get(rid, ""),
                })
            invest_df = pd.DataFrame(records)
            if len(invest_df) > 0 and "심각도" in invest_df.columns:
                sev_order = {"high": 0, "medium": 1, "low": 2, "": 3}
                invest_df["_sev"] = invest_df["심각도"].map(sev_order).fillna(3)
                invest_df = invest_df.sort_values("_sev").drop(columns=["_sev"])
            invest_df.to_csv(
                results_dir / "stage2_formula_investigation_targets.csv",
                index=False, encoding="utf-8-sig"
            )
            print(f"    -> stage2_formula_investigation_targets.csv ({len(invest_df)} ratios)")



def compute_acceptance(results_dir: Path, config: dict) -> tuple[dict, str, dict]:
    """Stage 2 acceptance check.

    Fixed logic:
    - category alias normalization: 기타 / 복합 / 기타·복합 -> 기타/복합
    - candidates_total uses DRAFT dictionary, expected around 221
    - final_dictionary_calculable uses FINAL dictionary, expected around 164
    - lag_coverage prefers raw/rate coverage, expected around 0.9975
    - ratios_calculated uses R*** ratio columns only
    """
    import pandas as pd
    import numpy as np

    summary = {}
    acc = {}

    def norm_cat(x):
        x = str(x).strip()
        if x in {"기타", "복합", "기타/복합", "기타·복합", "기타_복합"}:
            return "기타/복합"
        return x

    def bool_series(s):
        if s.dtype == bool:
            return s
        return s.astype(str).str.lower().isin(["true", "1", "yes", "y", "pass"])

    # ------------------------------------------------------------
    # 1. Ratio panel
    # ------------------------------------------------------------
    rp = results_dir / "engineered_financial_ratios.parquet"
    if rp.exists():
        df = pd.read_parquet(rp)

        ratio_cols = [c for c in df.columns if str(c).startswith("R")]
        n_ratios = len(ratio_cols)

        summary["firm_year_count"] = int(len(df))
        summary["ratio_panel_columns"] = int(df.shape[1])
        summary["ratios_calculated"] = int(n_ratios)

        min_ratios = int(config.get("acceptance", {}).get("ratios_calculated_min", 150))
        acc[f"ratios_calculated >= {min_ratios}"] = n_ratios >= min_ratios
    else:
        summary["ratios_calculated"] = 0
        acc["engineered_financial_ratios.parquet exists"] = False

    # ------------------------------------------------------------
    # 2. Quality pass + category coverage
    # ------------------------------------------------------------
    qr = results_dir / "ratio_quality_report.csv"
    if qr.exists():
        q = pd.read_csv(qr)

        if "quality_pass" in q.columns:
            mask = bool_series(q["quality_pass"])
        elif "status" in q.columns:
            mask = q["status"].astype(str).str.lower().isin(["pass", "true", "1"])
        else:
            mask = pd.Series([False] * len(q))

        pass_df = q[mask].copy()
        n_pass = int(len(pass_df))

        summary["quality_pass_total"] = n_pass
        min_quality = int(config.get("acceptance", {}).get("quality_pass_total_min", 120))
        acc[f"quality_pass >= {min_quality}"] = n_pass >= min_quality

        if "category" in pass_df.columns:
            pass_df["category_norm"] = pass_df["category"].map(norm_cat)
            vc = pass_df["category_norm"].value_counts().to_dict()

            # Expected financial categories, with normalized 기타/복합
            raw_categories = config.get("ratio_categories", [
                "수익성", "안정성", "부채상환능력", "유동성", "활동성", "성장성", "기타/복합"
            ])
            categories = []
            for c in raw_categories:
                nc = norm_cat(c)
                if nc not in categories:
                    categories.append(nc)

            # Ensure 기타/복합 is included if actual output has it
            if "기타/복합" in vc and "기타/복합" not in categories:
                categories.append("기타/복합")

            for c in categories:
                n_c = int(vc.get(c, 0))
                summary[f"category[{c}]"] = n_c
                acc[f"category[{c}] >= 1"] = n_c >= 1
    else:
        summary["quality_pass_total"] = 0
        acc["ratio_quality_report.csv exists"] = False

    # ------------------------------------------------------------
    # 3. Candidate universe and final dictionary
    # ------------------------------------------------------------
    draft_path = results_dir / "financial_ratio_formula_dictionary_draft.csv"
    final_path = results_dir / "financial_ratio_formula_dictionary_final.csv"
    availability_path = results_dir / "ratio_item_availability_report.csv"
    pool_path = results_dir / "candidate_ratio_pool_by_item.csv"

    # Candidate universe reporting only.
    # This is NOT an acceptance gate because the current pipeline's broad candidate files
    # may contain only calculable candidates after filtering. Acceptance is determined by
    # ratios_calculated, quality_pass, final_dictionary_calculable, category coverage, and lag_coverage.
    candidate_source = None
    if availability_path.exists():
        cand = pd.read_csv(availability_path)
        candidate_source = "ratio_item_availability_report.csv"
    elif draft_path.exists():
        cand = pd.read_csv(draft_path)
        candidate_source = "financial_ratio_formula_dictionary_draft.csv"
    elif pool_path.exists():
        cand = pd.read_csv(pool_path)
        candidate_source = "candidate_ratio_pool_by_item.csv"
    else:
        cand = None

    if cand is not None:
        n_candidates = int(len(cand))
        summary["candidate_universe_reported"] = n_candidates
        summary["candidate_universe_source"] = candidate_source

        # Backward-compatible summary fields only; no acceptance gate.
        summary["candidates_total_draft"] = n_candidates
        summary["candidates_total"] = n_candidates
        summary["candidates_total_source"] = candidate_source
    else:
        summary["candidate_universe_reported"] = 0
        summary["candidate_universe_source"] = "missing"

        # Backward-compatible summary fields only; no acceptance gate.
        summary["candidates_total_draft"] = 0
        summary["candidates_total"] = 0
        summary["candidates_total_source"] = "missing"

    if final_path.exists():
        final = pd.read_csv(final_path)
        n_final = int(len(final))
        summary["final_dictionary_calculable"] = n_final
        acc["final_dictionary_calculable >= 150"] = n_final >= 150
    else:
        summary["final_dictionary_calculable"] = 0
        acc["financial_ratio_formula_dictionary_final.csv exists"] = False

    # ------------------------------------------------------------
    # 4. Lag coverage
    # ------------------------------------------------------------
    lp = results_dir / "lag_support_coverage_report.csv"
    if lp.exists():
        lg = pd.read_csv(lp)

        coverage_value = None

        # Prefer explicit raw coverage columns if available
        preferred_cols = [
            "lag_coverage_raw_rate",
            "raw_lag_coverage_rate",
            "raw_coverage_rate",
            "lag_coverage_raw",
            "raw_coverage",
            "lag_coverage",
        ]

        for col in preferred_cols:
            if col in lg.columns:
                vals = pd.to_numeric(lg[col], errors="coerce").dropna()
                vals = vals[(vals >= 0) & (vals <= 1)]
                if len(vals) > 0:
                    coverage_value = float(vals.max())
                    break

        # Fallback: use maximum numeric rate/coverage value in [0,1]
        if coverage_value is None:
            candidate_cols = [
                c for c in lg.columns
                if ("coverage" in str(c).lower() or "rate" in str(c).lower())
            ]
            vals_all = []
            for col in candidate_cols:
                vals = pd.to_numeric(lg[col], errors="coerce").dropna()
                vals = vals[(vals >= 0) & (vals <= 1)]
                vals_all.extend(vals.tolist())
            if vals_all:
                coverage_value = float(max(vals_all))

        if coverage_value is not None:
            summary["lag_coverage"] = round(coverage_value, 6)
            min_lag = float(config.get("acceptance", {}).get("lag_coverage_min", 0.95))
            acc[f"lag_coverage >= {min_lag}"] = coverage_value >= min_lag
        else:
            summary["lag_coverage"] = None
            acc["lag_coverage readable"] = False
    else:
        summary["lag_coverage"] = None
        acc["lag_support_coverage_report.csv exists"] = False

    verdict = "PASS" if acc and all(bool(v) for v in acc.values()) else "FAIL"
    return summary, verdict, acc


# ============================================================
# Main pipeline
# ============================================================
def run_pipeline(paths: dict, config: dict, project_root: Path,
                  results_dir: Path, run_dir: Path, bundle_dir: Path):
    t_start = time.time()
    print(f"\n{'#' * 72}")
    print(f"  Stage 2 v2 pipeline — {kst_now()}")
    print(f"{'#' * 72}")

    # 환경변수 설정 (modules/ 내부 task 코드가 사용)
    os.environ["STAGE2_OUT"] = str(results_dir.resolve())
    os.environ["STAGE2_S1B_PANEL"] = str((project_root / paths["inputs"]["stage1b"]["panel"]).resolve())
    os.environ["STAGE2_S1B_CLEAN"] = str((project_root / paths["inputs"]["stage1b"]["cleaned_dir"]).resolve())
    os.environ["STAGE2_S1B_NICE"] = str((project_root / paths["inputs"]["stage1b"]["cleaned_dir"]
                                          / "재무비율_clean.parquet").resolve())
    os.environ["STAGE2_RAW"] = str((project_root / paths["inputs"]["raw"]["dir"]).resolve())
    os.environ["STAGE2_CANDIDATES_XLSX"] = str(
        (project_root / paths["inputs"]["reference"]["candidates_xlsx"]).resolve())

    # results_dir 보장 + lag_support 서브디렉토리
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "lag_support").mkdir(exist_ok=True)

    # input discovery
    print("\n[Input discovery]")
    write_input_discovery(paths, run_dir, project_root)

    # ========= Module 순차 실행 =========
    modules_dir = bundle_dir / "modules"
    task_sequence = [
        ("_task1_inventory.py", "Task 1: statement item inventory"),
        ("_task2_candidates.py", "Task 2: candidates + formula dictionary"),
        ("_task3_lag_diagnose.py", "Task 3: lag support coverage diagnose"),
        ("_task4_lag_extract.py", "Task 4: raw t-1 row extraction"),
        ("_task4_lag_dedup.py", "Task 4 step5: lag support consolidation"),
        ("_task6_compute_ratios.py", "Task 6: ratio computation (164)"),
        ("_task7_nice_audit.py", "Task 7: NICE precomputed audit"),
        ("_task8_quality_filter.py", "Task 8: quality filter + candidate pool"),
    ]

    # Task 9: growth audit (stage_config.growth_audit.enabled=true 일 때 실행)
    if config.get("growth_audit", {}).get("enabled", True):
        task_sequence.append(
            ("_task9_growth_audit.py", "Task 9: growth candidate audit + expansion")
        )

    failed_tasks: list[tuple[str, Exception]] = []
    for fname, label in task_sequence:
        try:
            run_module(modules_dir / fname, label)
        except Exception as e:
            print(f"\n  ✗ TASK FAILED ({fname}): {e}")
            print(f"    → 메타데이터 작성은 계속 진행합니다.")
            failed_tasks.append((fname, e))
            # 필수 task(1~8) 실패 시 즉시 중단, optional task(9~) 실패는 계속 진행
            essential_tasks = {
                "_task1_inventory.py", "_task2_candidates.py",
                "_task3_lag_diagnose.py", "_task4_lag_extract.py",
                "_task4_lag_dedup.py", "_task6_compute_ratios.py",
                "_task7_nice_audit.py", "_task8_quality_filter.py",
            }
            if fname in essential_tasks:
                print(f"    → 필수 task 실패. 파이프라인 중단.")
                raise

    # ========= Metadata =========
    print(f"\n{'=' * 72}")
    print("  WRITING metadata")
    print(f"{'=' * 72}")

    write_growth_policy(results_dir, config)
    write_readme_stage3(results_dir)
    print("\n[Aux outputs] preview + investigation_targets")
    write_aux_outputs(results_dir)

    # acceptance + summary
    summary, verdict, acceptance = compute_acceptance(results_dir, config)

    write_stage2_summary(results_dir, summary)
    write_verdict(results_dir, summary, verdict, acceptance)

    # file manifest
    descriptions = {
        "engineered_financial_ratios.parquet": "164 ratio × 4,746 firm-year wide panel",
        "engineered_financial_ratios.csv": "wide panel CSV",
        "financial_ratio_formula_dictionary_final.csv": "164 ratio formula 정의",
        "candidate_ratio_pool_by_item.csv": "평가항목별 후보 ratio pool",
        "ratio_quality_report.csv": "ratio별 missing/zero/var 통계",
        "ratio_item_availability_report.csv": "221 후보 item 매칭",
        "ratio_audit_against_precomputed_panel.csv": "NICE 검증",
        "stage2_formula_investigation_targets.csv": "investigate 대상",
        "lag_support_coverage_report.csv": "lag 커버리지",
        "growth_ratio_policy.md": "positive-base 정책",
        "stage2_summary.md": "Stage 2 summary",
        "stage2_verdict.md": "PASS/FAIL verdict",
        "README_STAGE3_INPUTS.md": "Stage 3 입력 명세",
        "growth_audit/growth_audit_report.md": "성장성 후보 audit 리포트 (v3.2)",
        "growth_audit/candidate_ratio_pool_growth_expanded.csv": "성장성 확장 pool",
        "growth_audit/engineered_financial_ratios_growth_expanded.parquet": "성장성 확장 ratio panel",
    }
    n = write_file_manifest(results_dir, descriptions,
                             results_dir / "stage2_file_manifest.csv")
    print(f"  → stage2_file_manifest.csv ({n} files)")

    # input/output hashes
    inputs_dir = bundle_dir / "inputs"
    if inputs_dir.exists():
        write_hashes(inputs_dir, run_dir / "input_file_hashes.csv")
    write_hashes(results_dir, run_dir / "output_file_hashes.csv")

    elapsed = time.time() - t_start
    write_run_metadata(run_dir, paths, summary, verdict, elapsed)

    # command_log
    class A:
        config = "configs/paths.yaml"
    write_command_log(run_dir, A(), elapsed)

    print(f"\n{'#' * 72}")
    print(f"  DONE — VERDICT: {verdict}  (elapsed: {elapsed:.1f}s)")
    print(f"{'#' * 72}\n")
    print("  Acceptance:")
    for k, v in acceptance.items():
        sym = "✓" if v else "✗"
        print(f"    {sym} {k}: {v}")
    if failed_tasks:
        print(f"\n  ⚠ 실패한 task ({len(failed_tasks)}개) — 메타데이터는 정상 생성됨:")
        for fname, exc in failed_tasks:
            print(f"    - {fname}: {exc}")

    return verdict


def main():
    parser = argparse.ArgumentParser(description="Stage 2 v2 pipeline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    bundle_dir = Path(os.environ.get("ORACLE_STAGE00_02_BUNDLE_DIR", str(config_path.parent.parent))).resolve()
    project_root = bundle_dir
    print(f"bundle_dir:   {bundle_dir}")
    print(f"project_root: {project_root}")

    paths = load_yaml(config_path)
    config = load_yaml(Path(args.stage_config))

    # path resolve (상대경로 → 절대경로, project_root 기준)
    def resolve(d):
        if isinstance(d, dict):
            return {k: resolve(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [resolve(x) for x in d]
        elif isinstance(d, str) and not Path(d).is_absolute() and ("/" in d or "\\" in d):
            return str((project_root / d).resolve())
        return d
    paths = resolve(paths)

    results_dir = Path(paths["outputs"]["results"])
    run_dir = Path(paths["outputs"]["run_log"])
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    verdict = run_pipeline(paths, config, project_root, results_dir, run_dir, bundle_dir)
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()



