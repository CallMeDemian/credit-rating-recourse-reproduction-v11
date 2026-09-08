"""Loop B2 gap decomposition diagnostic v2 — artifact-only (재시뮬레이션·재채점 없음).

v1 대비 추가 (2026-07-05 v2):
  - Cell Z        : 재계산 경로 '내부' 실측 델타 vs 등급.
                    dZ = replay_pred(firm, t−1행) − replay_pred(firm, t−2행)
                       = recomputed(emp t) − recomputed(emp t−1).
                    Z ≈ B1 → 재계산 경로가 within-firm 델타 신호를 보존(오프셋이 기업-안정적).
                    Z ≈ 50% → formula 재구성 자체가 stage00 의미론과 괴리 — 정렬 선행 필요.
  - 충실도 방향 재현 : sign[sim_t − emp_{t−1}] vs sign[emp_t − emp_{t−1}] (등급 무관, 순수 시뮬 fidelity).
                    낙관편향 중앙값 보정 변형(sim_t − median_opt) 병행 보고.
  - 낙관편향       : 동일 경로 sim_t − emp_t 분포(중앙값/양수비율/사분위별 B2 적중률).
  - 거울상 편향표   : arm별 delta<0 비율, 상·하향 조건부 일치율, bias-only 기대 일치율.
  - McNemar       : Y vs B2 짝지은 정확 이항검정 (동일 160 OOT movers).
  - at-bound      : 원천 action 패널이 선(先)절단되어 있어 '초과'는 항상 0 —
                    대신 경계 정확 포화 비율을 보고(전체/mover), 매출 포화 조건부 일치율 분할.

v1과 동일: Cell Y, Cell X, 변수 귀속, overlay 카운트, 하드페일 계약, 동결 산출물 무수정.

실행
    python -m credit_recourse.oracle.verification.diagnose_b2_gap_decomposition \
        --b2-dir data/final_freeze/stage2_substrate_loopA_loopB2
산출: <b2-dir>/b2_gap_decomposition/
    b2_gap_decomposition_report.json, cellY_rows.csv, cellZ_fidelity_rows.csv,
    sign_bias_table.csv, clipping_observability_crosstab.csv, variable_attribution.csv
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from credit_recourse.oracle.verification.verify_stage1_substrate_validation import (
    compute_direction_agreement,
)
from credit_recourse.simulator.action import ACTION_BOUNDS

RETENTION_CSV = "loopB2_simulator_score_retention.csv"
REPLAY_CSV = "loopB2_replay_alpha_predicted_score_vs_real_rating_change.csv"
RETENTION_INPUT_FRAME_CSV = "loopB2_simulator_score_retention_scoring_input_frame.csv"
REPLAY_INPUT_FRAME_CSV = "loopB2_replay_alpha_scoring_input_frame.csv"
REPLAY_META_JSON = "loopB2_replay_alpha_scoring_frame_meta_pre_guard.json"
REPORT_JSON = "substrate_loopA_loopB2_report.json"

RETENTION_PRED_COL = "pred_alpha_score_t_from_observed_action_sim_tminus1_to_t"
RETENTION_PREV_SCORE_COL = "score_tminus1_real"
RETENTION_DELTA_COL = "delta_sim_alpha_score_tminus1_to_t"
RETENTION_RATING_DELTA_COL = "real_rating_delta_t_to_tplus1"
RETENTION_OOT_COL = "loopB2_oot_2020_2023"
RETENTION_MATCH_COL = "loopB2_direction_match"
REPLAY_PRED_COL = "pred_alpha_score_tplus1"
REPLAY_PANEL_SCORE_COL = "score_t_real"

ACTION_AUDIT_PREFIX = "sim_input__action__"
ACTION_FLAG_AUDIT_PREFIX = "sim_input__action_observed__"
DEBT_CHANNEL_DIMS = ("short_debt_pct", "long_debt_pct", "bond_pct")
EXCEED_EPS = 1e-12

FINANCIAL_RCODE_CANDIDATES = ("R006", "R064", "R085", "R116", "R133", "R136", "R148", "R157", "R174", "R182", "R185")


def _fail(msg: str) -> "None":
    raise SystemExit(f"[B2-GAP-DECOMP HARD FAIL] {msg}")


def _require_file(path: Path, role: str) -> Path:
    if not path.exists():
        _fail(f"{role} 파일이 없습니다: {path}")
    return path


def _require_cols(df: pd.DataFrame, cols: list[str], src: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        _fail(f"{src} 에 필요한 컬럼이 없습니다: {missing} (보유 컬럼 수={len(df.columns)})")


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


def _firm_key(series: pd.Series) -> pd.Series:
    return series.map(_normalise_firm_key_value).astype(str)


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


def _agreement_block(score_delta: pd.Series, rating_delta: pd.Series, rows: int) -> dict[str, Any]:
    stats = compute_direction_agreement(score_delta, rating_delta)
    stats["rows"] = int(rows)
    if "direction_agreement" in stats:
        stats.setdefault("agreement", stats["direction_agreement"])
    if "direction_agreement_ci95" in stats:
        stats.setdefault("ci95", stats["direction_agreement_ci95"])
    return stats


def _load_report(report_path: Path) -> dict[str, Any]:
    with open(report_path, encoding="utf-8") as fh:
        return json.load(fh)


def _prep_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__firm_key"] = _firm_key(out["firm_id"])
    out["__year"] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    return out


def _cell_y(retention: pd.DataFrame, replay: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rep = _prep_keys(replay)
    dup = int(rep.duplicated(subset=["__firm_key", "__year"]).sum())
    if dup:
        _fail(f"{REPLAY_CSV} 의 (firm, fiscal_year) 키가 유일하지 않습니다: 중복 {dup}행 — 시프트 조인 불가")
    lookup = rep.set_index(["__firm_key", "__year"])[[REPLAY_PRED_COL, REPLAY_PANEL_SCORE_COL]]

    ret = _prep_keys(retention)
    ret["__prev_year"] = ret["__year"] - 1
    ret["__prev2_year"] = ret["__year"] - 2

    joined = ret.join(
        lookup.rename(columns={
            REPLAY_PRED_COL: "recomputed_empirical_alpha_score_t",
            REPLAY_PANEL_SCORE_COL: "panel_alpha_score_tminus1_from_replay",
        }),
        on=["__firm_key", "__prev_year"], how="left",
    )
    joined = joined.join(
        lookup[[REPLAY_PRED_COL]].rename(columns={REPLAY_PRED_COL: "recomputed_empirical_alpha_score_tminus1"}),
        on=["__firm_key", "__prev2_year"], how="left",
    )
    joined["delta_cellY_empirical_recomputed_minus_panel_tminus1"] = (
        pd.to_numeric(joined["recomputed_empirical_alpha_score_t"], errors="coerce")
        - pd.to_numeric(joined[RETENTION_PREV_SCORE_COL], errors="coerce")
    )
    # 조인 내적 검증: 두 경로의 panel(t-1)이 동일해야 함
    pj = pd.to_numeric(joined["panel_alpha_score_tminus1_from_replay"], errors="coerce")
    pr = pd.to_numeric(joined[RETENTION_PREV_SCORE_COL], errors="coerce")
    both = pj.notna() & pr.notna()
    max_panel_diff = float((pj[both] - pr[both]).abs().max()) if both.any() else None
    if max_panel_diff is not None and max_panel_diff > 1e-6:
        _fail(f"panel(t-1) 두 경로 불일치 max|diff|={max_panel_diff} — 조인 키/스코어 패널 정합 확인 필요")

    matched = int(joined["recomputed_empirical_alpha_score_t"].notna().sum())
    valid = joined[
        joined["delta_cellY_empirical_recomputed_minus_panel_tminus1"].notna()
        & joined[RETENTION_RATING_DELTA_COL].notna()
    ]
    oot = valid[valid[RETENTION_OOT_COL].astype(bool)]
    meta = {
        "definition": "delta_Y = recomputed(emp t) − panel(t−1); mover/OOT는 retention 정의 재사용",
        "retention_rows": int(len(retention)),
        "replay_rows": int(len(replay)),
        "matched_rows": matched,
        "matched_share": (matched / len(retention)) if len(retention) else None,
        "panel_tminus1_two_route_max_abs_diff": max_panel_diff,
        "all": _agreement_block(valid["delta_cellY_empirical_recomputed_minus_panel_tminus1"], valid[RETENTION_RATING_DELTA_COL], len(valid)),
        "oot": _agreement_block(oot["delta_cellY_empirical_recomputed_minus_panel_tminus1"], oot[RETENTION_RATING_DELTA_COL], len(oot)),
    }
    return joined, meta


def _cell_x(replay: pd.DataFrame) -> dict[str, Any]:
    rep = _prep_keys(replay)
    lookup = rep.set_index(["__firm_key", "__year"])[REPLAY_PRED_COL]
    rep["__prev_year"] = rep["__year"] - 1
    rep = rep.join(lookup.rename("recomputed_empirical_alpha_score_t"), on=["__firm_key", "__prev_year"], how="left")
    pair = rep[rep["recomputed_empirical_alpha_score_t"].notna() & pd.to_numeric(rep[REPLAY_PANEL_SCORE_COL], errors="coerce").notna()].copy()
    if pair.empty:
        return {"status": "NO_PAIRS", "pairs": 0}
    a = pd.to_numeric(pair["recomputed_empirical_alpha_score_t"], errors="coerce")
    b = pd.to_numeric(pair[REPLAY_PANEL_SCORE_COL], errors="coerce")
    diff = a - b
    return {
        "status": "ok",
        "pairs": int(len(pair)),
        "spearman_recomputed_vs_panel_same_year": float(a.rank().corr(b.rank())),
        "pearson": float(a.corr(b)),
        "diff_recomputed_minus_panel": {
            "mean": float(diff.mean()), "median": float(diff.median()),
            "p05": float(diff.quantile(0.05)), "p95": float(diff.quantile(0.95)),
            "share_abs_gt_1sd_of_panel": float((diff.abs() > b.std()).mean()),
        },
        "note": "동일 (firm, t) 재계산 경로 점수 vs Stage1 패널 점수 — featurization 정합 직접 측정",
    }


def _cell_z_and_fidelity(cellY_rows: pd.DataFrame) -> dict[str, Any]:
    """Cell Z(경로 내부 실측 델타 vs 등급) + 충실도 방향 재현(sim vs emp 같은 경로 변화)."""
    emp_t = pd.to_numeric(cellY_rows["recomputed_empirical_alpha_score_t"], errors="coerce")
    emp_tm1 = pd.to_numeric(cellY_rows["recomputed_empirical_alpha_score_tminus1"], errors="coerce")
    sim_t = pd.to_numeric(cellY_rows[RETENTION_PRED_COL], errors="coerce")
    rd = pd.to_numeric(cellY_rows[RETENTION_RATING_DELTA_COL], errors="coerce")
    oot = cellY_rows[RETENTION_OOT_COL].astype(bool)

    dZ = emp_t - emp_tm1
    z_valid = dZ.notna() & rd.notna()
    z_oot = z_valid & oot
    cell_z = {
        "definition": "dZ = recomputed(emp t) − recomputed(emp t−1) — 재계산 경로 내부 실측 델타 vs Δrating(t→t+1)",
        "matched_rows": int(dZ.notna().sum()),
        "matched_share_of_retention": float(dZ.notna().mean()),
        "coverage_note": "t−2→t−1 관측 전이가 추가로 필요하므로 커버리지는 Cell Y보다 낮을 수 있음",
        "all": _agreement_block(dZ[z_valid], rd[z_valid], int(z_valid.sum())),
        "oot": _agreement_block(dZ[z_oot], rd[z_oot], int(z_oot.sum())),
    }

    d_sim = sim_t - emp_tm1
    opt = sim_t - emp_t
    opt_median_all = float(opt.median()) if opt.notna().any() else None
    fid_valid = d_sim.notna() & dZ.notna()
    emp_zero = fid_valid & dZ.eq(0)
    fid_eval = fid_valid & dZ.ne(0)

    def _fid(mask: pd.Series, delta_sim: pd.Series) -> dict[str, Any]:
        n = int(mask.sum())
        if n == 0:
            return {"n": 0, "agreement": None}
        agree = (np.sign(delta_sim[mask]) == np.sign(dZ[mask]))
        return {"n": n, "agreement": float(agree.mean()),
                "sim_delta_positive_share": float((delta_sim[mask] > 0).mean()),
                "emp_delta_positive_share": float((dZ[mask] > 0).mean())}

    d_sim_adj = d_sim - (opt_median_all if opt_median_all is not None else 0.0)
    fidelity = {
        "definition": "sign[sim_t − emp_{t−1}] vs sign[emp_t − emp_{t−1}] — 등급 무관, 동일 재계산 경로의 순수 시뮬 방향 충실도",
        "emp_zero_change_rows_excluded": int(emp_zero.sum()),
        "raw": {"all": _fid(fid_eval, d_sim), "oot": _fid(fid_eval & oot, d_sim)},
        "optimism_median_adjusted": {
            "adjustment": opt_median_all,
            "all": _fid(fid_eval, d_sim_adj),
            "oot": _fid(fid_eval & oot, d_sim_adj),
        },
    }

    opt_stats = {
        "definition": "opt = recomputed(sim t) − recomputed(emp t) — 동일 경로 시뮬레이터 점수 오차(낙관편향)",
        "n": int(opt.notna().sum()),
        "median": opt_median_all,
        "mean": float(opt.mean()) if opt.notna().any() else None,
        "share_positive": float((opt > 0).mean()) if opt.notna().any() else None,
        "iqr": [float(opt.quantile(0.25)), float(opt.quantile(0.75))] if opt.notna().any() else None,
        "p05_p95": [float(opt.quantile(0.05)), float(opt.quantile(0.95))] if opt.notna().any() else None,
        "spearman_sim_vs_emp_levels": float(sim_t.rank().corr(emp_t.rank())),
    }
    return {"cellZ": cell_z, "fidelity_direction": fidelity, "optimism": opt_stats}


def _sign_bias_and_mcnemar(cellY_rows: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    rd = pd.to_numeric(cellY_rows[RETENTION_RATING_DELTA_COL], errors="coerce")
    dB2 = pd.to_numeric(cellY_rows[RETENTION_DELTA_COL], errors="coerce")
    dY = pd.to_numeric(cellY_rows["delta_cellY_empirical_recomputed_minus_panel_tminus1"], errors="coerce")
    oot = cellY_rows[RETENTION_OOT_COL].astype(bool)
    m = oot & rd.ne(0) & rd.notna() & dB2.notna() & dY.notna()
    n = int(m.sum())
    down = rd[m] < 0
    rows = []
    for arm, d in (("trueB2_sim", dB2[m]), ("cellY_emp", dY[m])):
        neg = d < 0
        rows.append({
            "arm": arm, "n_oot_movers": n,
            "delta_negative_share": float(neg.mean()),
            "downgrade_mover_share": float(down.mean()),
            "agreement_overall": float((np.sign(d) == np.sign(rd[m])).mean()),
            "agreement_on_downgrades": float((np.sign(d[down]) == np.sign(rd[m][down])).mean()) if down.any() else None,
            "agreement_on_upgrades": float((np.sign(d[~down]) == np.sign(rd[m][~down])).mean()) if (~down).any() else None,
            "bias_only_expected_agreement": float(neg.mean() * down.mean() + (1 - neg.mean()) * (1 - down.mean())),
        })
    table = pd.DataFrame(rows)
    mb2 = np.sign(dB2[m]) == np.sign(rd[m])
    my = np.sign(dY[m]) == np.sign(rd[m])
    b = int((mb2 & ~my).sum()); c = int((~mb2 & my).sum())
    mcnemar = {
        "paired_oot_movers": n,
        "b_B2_only_match": b, "c_Y_only_match": c,
        "both_match": int((mb2 & my).sum()), "both_miss": int((~mb2 & ~my).sum()),
        "exact_binom_p_two_sided": float(binomtest(min(b, c), b + c, 0.5).pvalue) if (b + c) > 0 else None,
        "note": "B2 vs Cell Y 방향적중의 짝지은 비교 — 유의하지 않으면 −8pp류 차이는 확립 불가",
    }
    return table, mcnemar


def _clipping_observability(retention: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    dims = list(ACTION_BOUNDS.keys())
    audit_cols = {d: f"{ACTION_AUDIT_PREFIX}{d}" for d in dims}
    flag_cols = {d: f"{ACTION_FLAG_AUDIT_PREFIX}{d}" for d in dims}
    _require_cols(retention, list(audit_cols.values()), RETENTION_CSV)

    df = retention.copy()
    rd = pd.to_numeric(df[RETENTION_RATING_DELTA_COL], errors="coerce")
    oot = df[RETENTION_OOT_COL].astype(bool)
    mover = oot & rd.ne(0) & rd.notna()

    per_dim: dict[str, Any] = {}
    exceed_any = pd.Series(False, index=df.index)
    at_any = pd.Series(False, index=df.index)
    for d in dims:
        lo, hi = ACTION_BOUNDS[d]
        raw = pd.to_numeric(df[audit_cols[d]], errors="coerce")
        exceed = (raw < lo - EXCEED_EPS) | (raw > hi + EXCEED_EPS)
        at_bound = pd.Series(np.isclose(raw.abs().to_numpy(), max(abs(lo), abs(hi)), rtol=1e-9, atol=1e-12), index=df.index)
        df[f"at_bound__{d}"] = at_bound
        exceed_any = exceed_any | exceed.fillna(False)
        at_any = at_any | at_bound.fillna(False)
        per_dim[d] = {
            "bound": [lo, hi],
            "exceed_share_guard": float(exceed.fillna(False).mean()),
            "at_bound_share_all": float(at_bound.mean()),
            "at_bound_share_oot_movers": float(at_bound[mover].mean()) if mover.any() else None,
        }
    df["at_bound_any_dim"] = at_any
    rev_at = df["at_bound__revenue_growth"].fillna(False)

    have_flags = all(c in df.columns for c in flag_cols.values())
    if have_flags:
        debt_unobs = pd.Series(False, index=df.index)
        for d in DEBT_CHANNEL_DIMS:
            debt_unobs = debt_unobs | (~df[flag_cols[d]].astype(bool))
        df["debt_channel_any_unobserved"] = debt_unobs

    delta = pd.to_numeric(df[RETENTION_DELTA_COL], errors="coerce")

    def _cond(mask: pd.Series, label: str) -> dict[str, Any]:
        sub = mask & oot
        blk = _agreement_block(delta[sub], rd[sub], int(sub.sum()))
        blk["subset"] = label
        return blk

    crosstab_rows = [
        _cond(pd.Series(True, index=df.index), "oot_all"),
        _cond(~rev_at, "oot_revenue_not_at_bound"),
        _cond(rev_at, "oot_revenue_at_bound_pm15pct"),
        _cond(~at_any, "oot_no_dim_at_bound"),
        _cond(at_any, "oot_any_dim_at_bound"),
    ]
    if have_flags:
        crosstab_rows.append(_cond(~df["debt_channel_any_unobserved"].astype(bool), "oot_debt_channels_all_observed"))
        crosstab_rows.append(_cond(df["debt_channel_any_unobserved"].astype(bool), "oot_debt_channel_any_unobserved"))
    crosstab = pd.DataFrame(crosstab_rows)
    meta = {
        "raw_value_semantics": (
            "sim_input__action__* 는 원천 Stage2A 패널의 사본이며, 원천 패널 자체가 경계값으로 선절단되어 저장됨"
            " (모든 차원 exceed=0 & p95==bound 로 확인). '초과'는 측정 불가하고 '경계 정확 포화'가 절단 유병률의 하한."
        ),
        "per_dimension": per_dim,
        "at_bound_any_dim_share_all": float(at_any.mean()),
        "at_bound_any_dim_share_oot_movers": float(at_any[mover].mean()) if mover.any() else None,
        "exceed_any_share_guard": float(exceed_any.mean()),
        "have_observed_flags": bool(have_flags),
    }
    return crosstab, meta


def _variable_attribution(
    b2dir: Path, retention: pd.DataFrame, replay: pd.DataFrame, cellY_rows: pd.DataFrame, selected_variables: list[str],
) -> tuple[pd.DataFrame | None, dict[str, Any]]:
    ret_frame_p = b2dir / RETENTION_INPUT_FRAME_CSV
    rep_frame_p = b2dir / REPLAY_INPUT_FRAME_CSV
    if not ret_frame_p.exists() or not rep_frame_p.exists():
        return None, {"status": "SKIPPED_MISSING_INPUT_FRAMES", "paths": [str(ret_frame_p), str(rep_frame_p)]}
    ret_frame = pd.read_csv(ret_frame_p)
    rep_frame = pd.read_csv(rep_frame_p)
    if len(ret_frame) != len(retention):
        _fail(f"{RETENTION_INPUT_FRAME_CSV} 행수({len(ret_frame)}) ≠ {RETENTION_CSV} 행수({len(retention)}) — 위치 정렬 계약 위반")
    if len(rep_frame) != len(replay):
        _fail(f"{REPLAY_INPUT_FRAME_CSV} 행수({len(rep_frame)}) ≠ {REPLAY_CSV} 행수({len(replay)}) — 위치 정렬 계약 위반")

    fin_vars = [v for v in selected_variables if v in FINANCIAL_RCODE_CANDIDATES]
    fin_vars = [v for v in fin_vars if v in ret_frame.columns and v in rep_frame.columns]
    if not fin_vars:
        return None, {"status": "SKIPPED_NO_COMMON_FINANCIAL_VARS", "selected": selected_variables}

    rep_keys = _prep_keys(replay)[["__firm_key", "__year"]].reset_index(drop=True)
    emp = pd.concat([rep_keys, rep_frame[fin_vars].reset_index(drop=True)], axis=1).set_index(["__firm_key", "__year"])

    base = cellY_rows[["__firm_key", "__prev_year", RETENTION_MATCH_COL, RETENTION_OOT_COL, RETENTION_RATING_DELTA_COL]].copy()
    base = pd.concat([base.reset_index(drop=True), ret_frame[fin_vars].add_prefix("sim__").reset_index(drop=True)], axis=1)
    base = base.join(emp.add_prefix("emp__"), on=["__firm_key", "__prev_year"], how="left")

    movers = base[
        base[RETENTION_OOT_COL].astype(bool)
        & pd.to_numeric(base[RETENTION_RATING_DELTA_COL], errors="coerce").fillna(0).ne(0)
    ].copy()
    rows = []
    for v in fin_vars:
        d = pd.to_numeric(movers[f"sim__{v}"], errors="coerce") - pd.to_numeric(movers[f"emp__{v}"], errors="coerce")
        for match_flag, label in ((False, "direction_mismatch"), (True, "direction_match")):
            grp = d[movers[RETENTION_MATCH_COL].astype(bool) == match_flag]
            rows.append({
                "variable": v, "group": label, "n": int(grp.notna().sum()),
                "median_signed_delta_sim_minus_emp": float(grp.median()) if grp.notna().any() else None,
                "median_abs_delta": float(grp.abs().median()) if grp.notna().any() else None,
                "p90_abs_delta": float(grp.abs().quantile(0.90)) if grp.notna().any() else None,
            })
    out = pd.DataFrame(rows).sort_values(["group", "median_abs_delta"], ascending=[True, False])
    return out, {"status": "ok", "financial_variables": fin_vars, "oot_mover_rows": int(len(movers))}


def _overlay_usage(b2dir: Path) -> dict[str, Any]:
    p = b2dir / REPLAY_META_JSON
    if not p.exists():
        return {"status": "META_NOT_FOUND", "path": str(p)}
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {"status": "META_UNREADABLE", "error": str(exc)}
    counts = meta.get("selected_formula_value_source_counts") or {}
    overlay = {k: (v or {}).get("observed_stage2_next_ratio") for k, v in counts.items()}
    return {"status": "ok", "replay_observed_next_ratio_overlay_counts": overlay,
            "note": "replay는 R157/R182 overlay 허용, True B2는 금지 — overlay>0이면 경로가 해당 행에서 갈림"}


def run(b2_dir: Path, out_dir: Path | None, report_path: Path | None) -> dict[str, Any]:
    b2_dir = b2_dir.resolve()
    report_path = report_path or (b2_dir / REPORT_JSON)
    out_dir = out_dir or (b2_dir / "b2_gap_decomposition")
    out_dir.mkdir(parents=True, exist_ok=True)

    report = _load_report(_require_file(report_path, "동결 B2 리포트"))
    b1 = ((report.get("loopB2") or {}).get("b1_ref") or {})
    b2 = report.get("loopB2") or {}
    replay_rep = report.get("loopB2_replay_diagnostic") or {}
    selected_variables = list(((b2.get("alpha_scoring_frame") or {}).get("selected_variables")) or [])
    if not selected_variables:
        _fail("리포트에서 alpha selected_variables 를 찾지 못했습니다")

    retention = pd.read_csv(_require_file(b2_dir / RETENTION_CSV, "True B2 retention rows"))
    replay = pd.read_csv(_require_file(b2_dir / REPLAY_CSV, "replay rows"))
    _require_cols(retention, ["firm_id", "fiscal_year", RETENTION_PRED_COL, RETENTION_PREV_SCORE_COL,
                              RETENTION_DELTA_COL, RETENTION_RATING_DELTA_COL, RETENTION_OOT_COL, RETENTION_MATCH_COL], RETENTION_CSV)
    _require_cols(replay, ["firm_id", "fiscal_year", REPLAY_PRED_COL, REPLAY_PANEL_SCORE_COL], REPLAY_CSV)

    cellY_rows, cellY = _cell_y(retention, replay)
    cellX = _cell_x(replay)
    zfo = _cell_z_and_fidelity(cellY_rows)
    sign_bias_table, mcnemar = _sign_bias_and_mcnemar(cellY_rows)
    crosstab, clip_meta = _clipping_observability(retention)
    attribution, attr_meta = _variable_attribution(b2_dir, retention, replay, cellY_rows, selected_variables)
    overlay = _overlay_usage(b2_dir)

    b1_agree = b1.get("agreement")
    y_oot = (cellY.get("oot") or {}).get("agreement")
    z_oot = (zfo["cellZ"].get("oot") or {}).get("agreement")
    b2_agree = b2.get("agreement")
    ledger = {
        "B1_oot_agreement": b1_agree,
        "cellZ_oot_agreement_recomputed_internal_delta": z_oot,
        "cellY_oot_agreement": y_oot,
        "trueB2_oot_agreement": b2_agree,
        "share_scoring_path_plus_population_pp": (None if (b1_agree is None or y_oot is None) else (b1_agree - y_oot) * 100.0),
        "share_simulator_plus_extraction_pp_NOT_ESTABLISHED_IF_MCNEMAR_NS": (
            None if (y_oot is None or b2_agree is None) else (y_oot - b2_agree) * 100.0),
        "mcnemar_p_B2_vs_Y": mcnemar.get("exact_binom_p_two_sided"),
        "simulator_optimism_median_same_path": zfo["optimism"].get("median"),
        "fidelity_direction_agreement_raw_oot": ((zfo["fidelity_direction"]["raw"].get("oot") or {}).get("agreement")),
        "fidelity_direction_agreement_bias_adjusted_oot": ((zfo["fidelity_direction"]["optimism_median_adjusted"].get("oot") or {}).get("agreement")),
        "replay_contemporaneous_oot_agreement": replay_rep.get("agreement"),
        "reading_guide": (
            "Z≈B1 → 재계산 경로가 델타 신호 보존, 유효 게이트는 same-path 재설계(B2′) / Z≈50% → formula 재구성 자체를 stage00 의미론에 먼저 정렬. "
            "방향일치는 arm별 부호편향×mover 구성비 산술에 지배되므로, 시뮬 충실도는 낙관편향(레벨)과 fidelity_direction(방향)으로 따로 읽을 것."
        ),
    }

    result = {
        "b2_dir": str(b2_dir), "frozen_report": str(report_path),
        "sim_business_plan_mode_frozen": report.get("sim_business_plan_mode"),
        "ledger": ledger,
        "cellY": cellY, "cellZ": zfo["cellZ"],
        "fidelity_direction": zfo["fidelity_direction"],
        "simulator_optimism_same_path": zfo["optimism"],
        "mcnemar_B2_vs_Y": mcnemar,
        "cellX_featurization_consistency": cellX,
        "clipping_observability": clip_meta,
        "variable_attribution_meta": attr_meta,
        "replay_overlay_usage": overlay,
        "selected_variables": selected_variables,
    }

    keep_drop = ["__firm_key", "__year", "__prev_year", "__prev2_year"]
    cellY_rows.drop(columns=keep_drop, errors="ignore").to_csv(out_dir / "cellY_rows.csv", index=False, encoding="utf-8-sig")
    zcols = ["firm_id", "fiscal_year", RETENTION_PRED_COL, "recomputed_empirical_alpha_score_t",
             "recomputed_empirical_alpha_score_tminus1", RETENTION_RATING_DELTA_COL, RETENTION_OOT_COL]
    cellY_rows[[c for c in zcols if c in cellY_rows.columns]].to_csv(out_dir / "cellZ_fidelity_rows.csv", index=False, encoding="utf-8-sig")
    sign_bias_table.to_csv(out_dir / "sign_bias_table.csv", index=False, encoding="utf-8-sig")
    crosstab.to_csv(out_dir / "clipping_observability_crosstab.csv", index=False, encoding="utf-8-sig")
    if attribution is not None:
        attribution.to_csv(out_dir / "variable_attribution.csv", index=False, encoding="utf-8-sig")
    with open(out_dir / "b2_gap_decomposition_report.json", "w", encoding="utf-8") as fh:
        json.dump(_json_safe(result), fh, ensure_ascii=False, indent=2)

    print("=== B2 gap decomposition v2 (artifact-only) ===")
    print(f"B1 OOT                     : {b1_agree}")
    print(f"Cell Z OOT (경로내 실측델타) : {z_oot}  (matched {zfo['cellZ']['matched_rows']}/{cellY['retention_rows']})")
    print(f"Cell Y OOT                 : {y_oot}")
    print(f"True B2 OOT                : {b2_agree}   McNemar(B2 vs Y) p={mcnemar.get('exact_binom_p_two_sided')}")
    print(f"시뮬 낙관편향 median (동일경로): {zfo['optimism'].get('median')}")
    print(f"충실도 방향재현 raw/보정 (OOT): {ledger['fidelity_direction_agreement_raw_oot']} / {ledger['fidelity_direction_agreement_bias_adjusted_oot']}")
    print(f"출력 → {out_dir}")
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Loop B2 gap decomposition diagnostic v2 (artifact-only)")
    ap.add_argument("--b2-dir", type=Path, default=Path("data/final_freeze/stage2_substrate_loopA_loopB2"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args(argv)
    run(args.b2_dir, args.out_dir, args.report)


if __name__ == "__main__":
    main()
