"""검사 3 재설계 (Test 3') — 반사실 장치 속성 검증. artifact-only, 재시뮬레이션 없음.

배경 (근거: b2_gap_decomposition v2, 2026-07-05)
------------------------------------------------
등급 기준 방향 일치(구 검사 3)는 시뮬레이터 게이트로 무효:
  (i)  채점 재구성 경로가 동일 실측 상태에서 25.2pp를 소실 (B1 72.0% → Cell Y 46.9%;
       공통 mover 짝지은 비교 74.3% vs 62.9%, McNemar p=0.050)
  (ii) 측정값이 arm별 부호편향 × mover 구성비 산술에 근사 (sim: 상향예측기, emp: 하향예측기)
  (iii) 실현 재현을 암묵 요구 — 그러나 시뮬레이터의 목표는 반사실(ceteris paribus)이며,
       실측과의 격차는 오차가 아니라 반사실–실현 wedge. 정책 비교 지표 Δnoop은
       시뮬 상태 간 차분이라 공통 wedge가 상쇄됨.

재설계: 반사실 비교 도구가 갖추어야 할 속성 3개를 동일 재계산 경로 위에서 직접 측정.
  P1. wedge 안정성   : wedge = recomputed(sim t) − recomputed(emp t) 분포와 연도 안정성.
                       |wedge|→0 은 게이트 대상이 아님 (예측 모형 요구로의 범주 오류).
  P2. 순위 보존      : Spearman(recomputed(sim t), recomputed(emp t)).
  P3. 방향 재현      : DEV에서 추정한 상수 c(중앙값) 보정 후,
                       sign[(sim_t − c) − emp_{t−1}] vs sign[emp_t − emp_{t−1}].

시간 분할 (사전등록 규율)
  DEV  = transition_target_year ≤ 2019   ← 보정치 c 추정은 여기서만
  OOT  = loopB2_oot_2020_2023 플래그      ← 게이트/헤드라인
  POST = transition_target_year ≥ 2024   ← 보고만 (추정·게이트 불사용)
  주의: 단순 ~OOT 마스크는 POST(t=2023, target 2024)를 포함하므로 추정에 쓰면 안 됨.

게이트 모드
  --gate-config 미지정  → REPORT_ONLY (본 논문 모드: 지표 정의 + 기준값 문서화)
  --gate-config 지정    → 4개 임계 전부 필수(부분 제공 시 하드페일; 체리피킹 방지), PASS/FAIL 산출
      {"direction_reproduction_oot_min": .., "rank_preservation_oot_min": ..,
       "wedge_yearly_median_max_abs_dev_from_dev": .., "wedge_dev_to_oot_median_shift_max_abs": ..}

실행
    python -m credit_recourse.oracle.verification.verify_stage2_test3_counterfactual_fidelity \
        --b2-dir data/final_freeze/stage2_substrate_loopA_loopB2
산출: <b2-dir>/test3_redesign/
    test3_counterfactual_fidelity_report.json, test3_property_summary.csv,
    test3_wedge_by_year.csv, test3_rows.csv

계약: 동결 산출물 무수정(읽기 전용 + 신규 하위폴더). 파일/컬럼/정합 위반 즉시 하드페일.
mover/CI 의미론은 verify_stage1_substrate_validation.compute_direction_agreement 재사용
(참조=실측 점수변화 d_emp, 예측=보정 시뮬 점수변화; d_emp=0 제외, 예측 0은 비적중 처리).
구 검사 3(등급 기준)은 b2_gap_decomposition 리포트가 있으면 맥락 진단으로 첨부.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.oracle.verification.verify_stage1_substrate_validation import (
    compute_direction_agreement,
)
from credit_recourse.simulator.action import ACTION_BOUNDS

TEST_ID = "test3_redesign_counterfactual_device_properties_v1"

RETENTION_CSV = "loopB2_simulator_score_retention.csv"
REPLAY_CSV = "loopB2_replay_alpha_predicted_score_vs_real_rating_change.csv"
REPORT_JSON = "substrate_loopA_loopB2_report.json"
DIAG_REPORT_REL = Path("b2_gap_decomposition") / "b2_gap_decomposition_report.json"

RETENTION_PRED_COL = "pred_alpha_score_t_from_observed_action_sim_tminus1_to_t"
RETENTION_PREV_SCORE_COL = "score_tminus1_real"
RETENTION_TARGET_YEAR_COL = "transition_target_year"
RETENTION_OOT_COL = "loopB2_oot_2020_2023"
REPLAY_PRED_COL = "pred_alpha_score_tplus1"
REPLAY_PANEL_SCORE_COL = "score_t_real"
REV_AUDIT_COL = "sim_input__action__revenue_growth"

DEV_TARGET_YEAR_MAX = 2019
POST_TARGET_YEAR_MIN = 2024
YEARLY_MIN_N_FOR_STABILITY = 30

GATE_KEYS = (
    "direction_reproduction_oot_min",
    "rank_preservation_oot_min",
    "wedge_yearly_median_max_abs_dev_from_dev",
    "wedge_dev_to_oot_median_shift_max_abs",
)


def _fail(msg: str) -> "None":
    raise SystemExit(f"[TEST3-REDESIGN HARD FAIL] {msg}")


def _require_file(path: Path, role: str) -> Path:
    if not path.exists():
        _fail(f"{role} 파일이 없습니다: {path}")
    return path


def _require_cols(df: pd.DataFrame, cols: list[str], src: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"{src} 에 필요한 컬럼이 없습니다: {missing}")


def _normalise_firm_key_value(value: Any) -> str:
    """verify_stage2_substrate_loopA_loopB2._normalise_loopb2_firm_key_value 동일 로직 사본."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if len(s) > 1 and s[0].upper() == "A" and s[1:].isdigit():
        s = s[1:]
    if s.isdigit():
        return s.zfill(6)
    return s


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _agreement_block(pred_delta: pd.Series, ref_delta: pd.Series) -> dict[str, Any]:
    stats = compute_direction_agreement(pred_delta, ref_delta)
    if "direction_agreement" in stats:
        stats.setdefault("agreement", stats["direction_agreement"])
    if "direction_agreement_ci95" in stats:
        stats.setdefault("ci95", stats["direction_agreement_ci95"])
    return stats


# ---------------------------------------------------------------- frame build
def build_property_frame(retention: pd.DataFrame, replay: pd.DataFrame) -> pd.DataFrame:
    """동결 retention/replay CSV → 속성 계산용 정규화 프레임.

    반환 컬럼: firm_id, fiscal_year, target_year, oot, split, sim_t, emp_t, emp_tm1,
              panel_tm1, wedge, rev_at_bound(옵션)
    """
    _require_cols(retention, ["firm_id", "fiscal_year", RETENTION_PRED_COL, RETENTION_PREV_SCORE_COL,
                              RETENTION_TARGET_YEAR_COL, RETENTION_OOT_COL], RETENTION_CSV)
    _require_cols(replay, ["firm_id", "fiscal_year", REPLAY_PRED_COL, REPLAY_PANEL_SCORE_COL], REPLAY_CSV)

    rep = replay.copy()
    rep["__k"] = rep["firm_id"].map(_normalise_firm_key_value).astype(str)
    rep["__y"] = pd.to_numeric(rep["fiscal_year"], errors="coerce").astype("Int64")
    dup = int(rep.duplicated(subset=["__k", "__y"]).sum())
    if dup:
        _fail(f"{REPLAY_CSV} (firm, fiscal_year) 키 중복 {dup}행 — 시프트 조인 불가")
    lk = rep.set_index(["__k", "__y"])[[REPLAY_PRED_COL, REPLAY_PANEL_SCORE_COL]]

    fr = retention.copy()
    fr["__k"] = fr["firm_id"].map(_normalise_firm_key_value).astype(str)
    fr["__y"] = pd.to_numeric(fr["fiscal_year"], errors="coerce").astype("Int64")
    fr["__ym1"] = fr["__y"] - 1
    fr["__ym2"] = fr["__y"] - 2
    fr = fr.join(lk.rename(columns={REPLAY_PRED_COL: "emp_t", REPLAY_PANEL_SCORE_COL: "panel_tm1_replay"}),
                 on=["__k", "__ym1"], how="left")
    fr = fr.join(lk[[REPLAY_PRED_COL]].rename(columns={REPLAY_PRED_COL: "emp_tm1"}),
                 on=["__k", "__ym2"], how="left")

    if int(fr["emp_t"].notna().sum()) == 0:
        _fail("replay 시프트 조인 결과 emp_t 매칭 0행 — firm 키/연도 정합 확인 필요")

    # 내적 정합 가드: 두 경로의 panel(t−1) 동일성
    pj = pd.to_numeric(fr["panel_tm1_replay"], errors="coerce")
    pr = pd.to_numeric(fr[RETENTION_PREV_SCORE_COL], errors="coerce")
    both = pj.notna() & pr.notna()
    if both.any():
        max_diff = float((pj[both] - pr[both]).abs().max())
        if max_diff > 1e-6:
            _fail(f"panel(t−1) 두 경로 불일치 max|diff|={max_diff} — 스코어 패널 정합 위반")

    out = pd.DataFrame({
        "firm_id": fr["firm_id"],
        "fiscal_year": fr["__y"].astype("Int64"),
        "target_year": pd.to_numeric(fr[RETENTION_TARGET_YEAR_COL], errors="coerce").astype("Int64"),
        "oot": fr[RETENTION_OOT_COL].astype(bool),
        "sim_t": pd.to_numeric(fr[RETENTION_PRED_COL], errors="coerce"),
        "emp_t": pd.to_numeric(fr["emp_t"], errors="coerce"),
        "emp_tm1": pd.to_numeric(fr["emp_tm1"], errors="coerce"),
        "panel_tm1": pr,
    })
    out["wedge"] = out["sim_t"] - out["emp_t"]
    ty = out["target_year"]
    split = np.where(out["oot"], "oot",
             np.where(ty.notna() & (ty <= DEV_TARGET_YEAR_MAX), "dev",
              np.where(ty.notna() & (ty >= POST_TARGET_YEAR_MIN), "post", "other")))
    out["split"] = split
    n_other = int((out["split"] == "other").sum())
    if n_other:
        _fail(f"시간 분할 불능 행 {n_other}개 — target_year 값 확인 필요 (dev≤{DEV_TARGET_YEAR_MAX}, oot=플래그, post≥{POST_TARGET_YEAR_MIN})")

    if REV_AUDIT_COL in retention.columns:
        lo, hi = ACTION_BOUNDS["revenue_growth"]
        raw = pd.to_numeric(retention[REV_AUDIT_COL], errors="coerce")
        out["rev_at_bound"] = pd.Series(
            np.isclose(raw.abs().to_numpy(), max(abs(lo), abs(hi)), rtol=1e-9, atol=1e-12), index=out.index)
    return out


# ---------------------------------------------------------------- properties
def _wedge_stats(w: pd.Series) -> dict[str, Any]:
    if not w.notna().any():
        return {"n": 0}
    return {"n": int(w.notna().sum()), "median": float(w.median()), "mean": float(w.mean()),
            "share_positive": float((w > 0).mean()),
            "iqr": [float(w.quantile(0.25)), float(w.quantile(0.75))],
            "p05_p95": [float(w.quantile(0.05)), float(w.quantile(0.95))]}


def compute_test3_properties(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """속성 3종 계산. 반환: (결과 dict, 연도별 wedge 표, 행 단위 표)."""
    f = frame
    dev = f["split"].eq("dev")
    oot = f["split"].eq("oot")
    post = f["split"].eq("post")

    # ---- P1: wedge 분포·안정성
    by_year_rows = []
    for ty, g in f.groupby("target_year"):
        w = g["wedge"]
        by_year_rows.append({"target_year": int(ty), "split": g["split"].iloc[0], "n": int(w.notna().sum()),
                             "median": float(w.median()) if w.notna().any() else None,
                             "q25": float(w.quantile(0.25)) if w.notna().any() else None,
                             "q75": float(w.quantile(0.75)) if w.notna().any() else None})
    wedge_by_year = pd.DataFrame(by_year_rows).sort_values("target_year")
    dev_median = float(f.loc[dev, "wedge"].median()) if dev.any() else None
    oot_median = float(f.loc[oot, "wedge"].median()) if oot.any() else None
    elig = wedge_by_year[wedge_by_year["n"] >= YEARLY_MIN_N_FOR_STABILITY]
    max_abs_dev = (float((elig["median"] - dev_median).abs().max())
                   if dev_median is not None and len(elig) else None)
    p1 = {
        "definition": "wedge = recomputed(sim t) − recomputed(emp t); 크기 자체는 게이트 비대상(반사실–실현 격차)",
        "overall": _wedge_stats(f["wedge"]),
        "dev": _wedge_stats(f.loc[dev, "wedge"]),
        "oot": _wedge_stats(f.loc[oot, "wedge"]),
        "post": _wedge_stats(f.loc[post, "wedge"]),
        "stability": {
            "yearly_min_n": YEARLY_MIN_N_FOR_STABILITY,
            "yearly_median_max_abs_dev_from_dev_median": max_abs_dev,
            "dev_to_oot_median_shift": (oot_median - dev_median) if (dev_median is not None and oot_median is not None) else None,
        },
    }

    # ---- P2: 순위 보존
    def _spear(mask: pd.Series) -> dict[str, Any]:
        sub = f[mask & f["sim_t"].notna() & f["emp_t"].notna()]
        if len(sub) < 10:
            return {"n": int(len(sub)), "spearman": None}
        return {"n": int(len(sub)), "spearman": float(sub["sim_t"].rank().corr(sub["emp_t"].rank()))}
    p2 = {"definition": "Spearman(recomputed(sim t), recomputed(emp t)) — 기업 간 순위 보존",
          "overall": _spear(pd.Series(True, index=f.index)), "dev": _spear(dev), "oot": _spear(oot), "post": _spear(post)}

    # ---- P3: 방향 재현 (DEV 상수 보정)
    if not dev.any():
        _fail("DEV 분할이 비어 있어 보정치 추정 불가")
    c = float(f.loc[dev, "wedge"].median())
    d_emp = f["emp_t"] - f["emp_tm1"]
    d_sim_adj = (f["sim_t"] - c) - f["emp_tm1"]
    chain = d_emp.notna() & d_sim_adj.notna()

    def _dir(mask: pd.Series) -> dict[str, Any]:
        m = mask & chain
        blk = _agreement_block(d_sim_adj[m], d_emp[m])
        blk["rows_in_split"] = int(mask.sum())
        blk["chain_matched_rows"] = int(m.sum())
        return blk

    p3 = {
        "definition": "sign[(sim_t − c) − emp_{t−1}] vs sign[emp_t − emp_{t−1}]; 참조 0행 제외, 예측 0은 비적중",
        "adjustment_rule": f"c = DEV(target_year ≤ {DEV_TARGET_YEAR_MAX}) wedge 중앙값 — 규칙 고정, 튜닝 불가",
        "adjustment_value": c,
        "overall": _dir(pd.Series(True, index=f.index)),
        "dev": _dir(dev), "oot": _dir(oot), "post": _dir(post),
        "note": "본 지표는 '수준 보정된 시뮬레이터'라는 측정 구성물의 방향 재현율이며 운영 시뮬레이터의 예측력이 아님",
    }
    robustness = {}
    if "rev_at_bound" in f.columns:
        robustness["direction_oot_revenue_not_at_bound"] = _dir(oot & ~f["rev_at_bound"].fillna(False))
        robustness["direction_oot_revenue_at_bound"] = _dir(oot & f["rev_at_bound"].fillna(False))

    rows = f[["firm_id", "fiscal_year", "target_year", "split", "sim_t", "emp_t", "emp_tm1", "wedge"]].copy()
    rows["d_emp"] = d_emp
    rows["d_sim_adjusted"] = d_sim_adj
    rows["direction_match"] = np.where(chain & d_emp.ne(0), np.sign(d_sim_adj) == np.sign(d_emp), np.nan)

    result = {"P1_wedge": p1, "P2_rank_preservation": p2, "P3_direction_reproduction": p3,
              "robustness_saturation": robustness}
    return result, wedge_by_year, rows


# ---------------------------------------------------------------- gate
def _load_gate_config(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    cfg = json.loads(_require_file(path, "게이트 설정").read_text(encoding="utf-8"))
    missing = [k for k in GATE_KEYS if k not in cfg]
    extra = [k for k in cfg if k not in GATE_KEYS]
    if missing or extra:
        _fail(f"게이트 설정은 4개 키 전부·정확히 필요 (사전등록 규율). missing={missing}, extra={extra}")
    return {k: float(cfg[k]) for k in GATE_KEYS}


def _evaluate_gate(props: dict[str, Any], cfg: dict[str, float]) -> dict[str, Any]:
    obs = {
        "direction_reproduction_oot": ((props["P3_direction_reproduction"]["oot"] or {}).get("agreement")),
        "rank_preservation_oot": ((props["P2_rank_preservation"]["oot"] or {}).get("spearman")),
        "wedge_yearly_median_max_abs_dev_from_dev": props["P1_wedge"]["stability"]["yearly_median_max_abs_dev_from_dev_median"],
        "wedge_dev_to_oot_median_shift_abs": (abs(props["P1_wedge"]["stability"]["dev_to_oot_median_shift"])
                                              if props["P1_wedge"]["stability"]["dev_to_oot_median_shift"] is not None else None),
    }
    checks = {
        "direction_reproduction_oot_min": (obs["direction_reproduction_oot"] is not None
                                           and obs["direction_reproduction_oot"] >= cfg["direction_reproduction_oot_min"]),
        "rank_preservation_oot_min": (obs["rank_preservation_oot"] is not None
                                      and obs["rank_preservation_oot"] >= cfg["rank_preservation_oot_min"]),
        "wedge_yearly_median_max_abs_dev_from_dev": (obs["wedge_yearly_median_max_abs_dev_from_dev"] is not None
                                                     and obs["wedge_yearly_median_max_abs_dev_from_dev"] <= cfg["wedge_yearly_median_max_abs_dev_from_dev"]),
        "wedge_dev_to_oot_median_shift_max_abs": (obs["wedge_dev_to_oot_median_shift_abs"] is not None
                                                  and obs["wedge_dev_to_oot_median_shift_abs"] <= cfg["wedge_dev_to_oot_median_shift_max_abs"]),
    }
    return {"mode": "GATED", "thresholds": cfg, "observed": obs, "checks": checks,
            "status": "PASS" if all(checks.values()) else "FAIL"}


# ---------------------------------------------------------------- run
def _property_summary_table(props: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("overall", "dev", "oot", "post"):
        w = props["P1_wedge"][split]
        if w.get("n"):
            rows.append({"property": "P1_wedge", "metric": "median", "split": split, "value": w["median"], "n": w["n"]})
            rows.append({"property": "P1_wedge", "metric": "share_positive", "split": split, "value": w["share_positive"], "n": w["n"]})
        s = props["P2_rank_preservation"][split]
        rows.append({"property": "P2_rank", "metric": "spearman", "split": split, "value": s["spearman"], "n": s["n"]})
        d = props["P3_direction_reproduction"][split]
        rows.append({"property": "P3_direction", "metric": "agreement", "split": split,
                     "value": d.get("agreement"), "n": d.get("n_movers"),
                     "ci95_lo": (d.get("ci95") or [None, None])[0], "ci95_hi": (d.get("ci95") or [None, None])[1]})
    st = props["P1_wedge"]["stability"]
    rows.append({"property": "P1_wedge", "metric": "yearly_median_max_abs_dev_from_dev", "split": "all_years",
                 "value": st["yearly_median_max_abs_dev_from_dev_median"], "n": None})
    rows.append({"property": "P1_wedge", "metric": "dev_to_oot_median_shift", "split": "dev→oot",
                 "value": st["dev_to_oot_median_shift"], "n": None})
    for k, d in (props.get("robustness_saturation") or {}).items():
        rows.append({"property": "P3_direction", "metric": k, "split": "oot",
                     "value": d.get("agreement"), "n": d.get("n_movers")})
    return pd.DataFrame(rows)


def run(b2_dir: Path, out_dir: Path | None, report_path: Path | None,
        gate_config_path: Path | None, diagnostic_report_path: Path | None) -> dict[str, Any]:
    b2_dir = b2_dir.resolve()
    out_dir = out_dir or (b2_dir / "test3_redesign")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_path or (b2_dir / REPORT_JSON)
    frozen = json.loads(_require_file(report_path, "동결 B2 리포트").read_text(encoding="utf-8"))

    retention = pd.read_csv(_require_file(b2_dir / RETENTION_CSV, "retention rows"))
    replay = pd.read_csv(_require_file(b2_dir / REPLAY_CSV, "replay rows"))
    frame = build_property_frame(retention, replay)
    props, wedge_by_year, rows = compute_test3_properties(frame)

    cfg = _load_gate_config(gate_config_path)
    gate = _evaluate_gate(props, cfg) if cfg else {
        "mode": "REPORT_ONLY", "status": "REPORT_ONLY_BASELINE_DOCUMENTED",
        "note": "게이트 임계값은 본검증 사전등록 항목 — 본 실행은 지표 정의와 기준값 문서화 목적",
    }

    diag_path = diagnostic_report_path or (b2_dir / DIAG_REPORT_REL)
    context: dict[str, Any]
    if diag_path.exists():
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        context = {"source": str(diag_path), "legacy_rating_anchored_ledger": diag.get("ledger"),
                   "mcnemar_B2_vs_Y": diag.get("mcnemar_B2_vs_Y"),
                   "status": "attached_as_demoted_context"}
    else:
        context = {"status": "DIAGNOSTIC_REPORT_NOT_FOUND", "expected_path": str(diag_path),
                   "note": "구 검사3 맥락은 diagnose_b2_gap_decomposition 실행으로 생성"}

    result = {
        "test_id": TEST_ID,
        "b2_dir": str(b2_dir),
        "frozen_report": str(report_path),
        "sim_business_plan_mode_frozen": frozen.get("sim_business_plan_mode"),
        "splits": {"dev_rule": f"target_year ≤ {DEV_TARGET_YEAR_MAX}", "oot_rule": f"{RETENTION_OOT_COL} 플래그",
                   "post_rule": f"target_year ≥ {POST_TARGET_YEAR_MIN}",
                   "counts": frame["split"].value_counts().to_dict()},
        **props,
        "gate": gate,
        "context_legacy_test3": context,
    }

    wedge_by_year.to_csv(out_dir / "test3_wedge_by_year.csv", index=False, encoding="utf-8-sig")
    _property_summary_table(props).to_csv(out_dir / "test3_property_summary.csv", index=False, encoding="utf-8-sig")
    rows.to_csv(out_dir / "test3_rows.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "test3_counterfactual_fidelity_report.json", "w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, ensure_ascii=False, indent=2)

    p1, p2, p3 = result["P1_wedge"], result["P2_rank_preservation"], result["P3_direction_reproduction"]
    print(f"=== 검사 3' — 반사실 장치 속성 검증 ({gate['mode']}) ===")
    print(f"P1 wedge median  dev/oot/post : {p1['dev'].get('median')} / {p1['oot'].get('median')} / {p1['post'].get('median')}")
    print(f"P1 안정성  연도최대이탈 / dev→oot : {p1['stability']['yearly_median_max_abs_dev_from_dev_median']} / {p1['stability']['dev_to_oot_median_shift']}")
    print(f"P2 Spearman      dev/oot      : {p2['dev']['spearman']} / {p2['oot']['spearman']}")
    print(f"P3 보정치 c(DEV)               : {p3['adjustment_value']}")
    print(f"P3 방향재현 OOT               : {p3['oot'].get('agreement')} (n={p3['oot'].get('n_movers')}, CI {p3['oot'].get('ci95')})")
    print(f"게이트: {gate['status']}  → 출력 {out_dir}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="검사 3 재설계 — 반사실 장치 속성 검증 (artifact-only)")
    ap.add_argument("--b2-dir", type=Path, default=Path("data/final_freeze/stage2_substrate_loopA_loopB2"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--gate-config", type=Path, default=None)
    ap.add_argument("--diagnostic-report", type=Path, default=None)
    args = ap.parse_args(argv)
    run(args.b2_dir, args.out_dir, args.report, args.gate_config, args.diagnostic_report)


if __name__ == "__main__":
    main()
