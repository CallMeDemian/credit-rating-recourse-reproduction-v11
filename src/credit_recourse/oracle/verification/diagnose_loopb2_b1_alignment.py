"""Loop B2 vs Loop B1 sign/scale alignment diagnostic.

This diagnostic does not change the B2 gate verdict.  It answers one question:
when Loop B2 OOT direction agreement is far below chance, are the B2 score/rating
orientation conventions aligned with the exact Stage1 Loop B1 panel and lead-pair
construction?

Outputs:
  data/final_freeze/stage2_substrate_loopA_loopB2/
    loopB2_b1_b2_sign_alignment_report.json
    loopB2_b1_b2_sign_alignment_rows.csv
    loopB2_b1_b2_sign_alignment_oot_mover_mismatches.csv
    loopB2_b1_b2_sign_alignment_summary.csv
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
    KEY as STAGE1_KEY,
    compute_direction_agreement,
    load_stage1_backend_score_panel,
)
from credit_recourse.rl.common.io import write_json

B2_BACKEND = "alpha"
B2_OOT_YEAR_MIN = 2020
B2_OOT_YEAR_MAX = 2023
B2_SCORE_CSV_NAME = "loopB2_alpha_predicted_score_vs_real_rating_change.csv"
B2_REPORT_NAME = "substrate_loopA_loopB2_report.json"
REPORT_NAME = "loopB2_b1_b2_sign_alignment_report.json"
ROWS_NAME = "loopB2_b1_b2_sign_alignment_rows.csv"
OOT_MISMATCH_NAME = "loopB2_b1_b2_sign_alignment_oot_mover_mismatches.csv"
SUMMARY_CSV_NAME = "loopB2_b1_b2_sign_alignment_summary.csv"


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not math.isfinite(val) else val
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    if pd.isna(obj) if not isinstance(obj, (str, bytes, bool, dict, list, tuple)) else False:
        return None
    return obj


def _normalise_firm_key_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if not s:
        return ""
    # pandas often stringifies numeric firm ids as 123456.0.
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if len(s) > 1 and s[0].upper() == "A" and s[1:].isdigit():
        s = s[1:]
    if s.isdigit():
        return s.zfill(6)
    return s


def _normalise_firm_key(series: pd.Series) -> pd.Series:
    return series.map(_normalise_firm_key_value).astype(str)


def _first_present_numeric(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), vals)
    return out


def _transition_target_years(df: pd.DataFrame) -> pd.Series:
    next_year = _first_present_numeric(
        df,
        (
            "fiscal_year__next",
            "year__next",
            "next__fiscal_year",
            "next__year",
            "fiscal_year_tplus1",
            "year_tplus1",
            "transition_target_year",
        ),
    )
    base_year = _first_present_numeric(df, ("fiscal_year", "year"))
    return next_year.where(next_year.notna(), base_year + 1.0)


def _bool_mask(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df), index=df.index, dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    text = s.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "none", "null"}
    mapped = text.map(lambda x: True if x in true_values else (False if x in false_values else default))
    return mapped.astype(bool)


def _sign_agreement_stats(df: pd.DataFrame, score_col: str, rating_col: str) -> dict[str, Any]:
    if score_col not in df.columns or rating_col not in df.columns:
        return {
            "status": "missing_columns",
            "score_col": score_col,
            "rating_col": rating_col,
            "missing": [c for c in (score_col, rating_col) if c not in df.columns],
        }
    raw = compute_direction_agreement(df[score_col], df[rating_col])
    s = pd.to_numeric(df[score_col], errors="coerce")
    r = pd.to_numeric(df[rating_col], errors="coerce")
    valid = s.notna() & r.notna() & r.ne(0)
    out = {
        "status": raw.get("status"),
        "rows": int(len(df)),
        "valid_mover_rows": int(valid.sum()),
        "n_movers": int(raw.get("n_movers", 0) or 0),
        "n_matches": int(raw.get("n_matches", 0) or 0) if raw.get("n_matches") is not None else None,
        "agreement": raw.get("direction_agreement"),
        "agreement_pct": None if raw.get("direction_agreement") is None else float(raw["direction_agreement"]) * 100.0,
        "ci95": raw.get("direction_agreement_ci95"),
        "ci_method": raw.get("ci_method"),
        "n_score_delta_zero_movers": int(raw.get("n_score_delta_zero_movers", 0) or 0),
        "delta_zero_rule": raw.get("delta_zero_rule"),
        "score_col": score_col,
        "rating_col": rating_col,
    }
    if int(valid.sum()) > 0:
        sign_score = np.sign(s[valid].to_numpy(dtype=float))
        sign_rating = np.sign(r[valid].to_numpy(dtype=float))
        out["score_sign_counts"] = {
            "positive": int((sign_score > 0).sum()),
            "zero": int((sign_score == 0).sum()),
            "negative": int((sign_score < 0).sum()),
        }
        out["rating_sign_counts"] = {
            "positive_improvement": int((sign_rating > 0).sum()),
            "negative_deterioration": int((sign_rating < 0).sum()),
            "zero": int((sign_rating == 0).sum()),
        }
    return out


def _build_stage1_transition_panel(score_panel: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Reconstruct the exact Loop B1 lead-pair keys plus extra actual-forward deltas.

    B1's gate uses score change (t-1 -> t) against rating change (t -> t+1).
    This diagnostic also records the contemporaneous actual score change
    (t -> t+1) when score_{t+1} is present, because it is the closest direct
    analogue to B2's pred_score_{t+1} - score_t comparison.
    """
    required = {STAGE1_KEY, "year", score_col, "rating_num_10"}
    missing = sorted(required - set(score_panel.columns))
    if missing:
        raise ValueError(f"Stage1 panel missing required columns for B1/B2 alignment diagnostic: {missing}")
    df = score_panel.copy()
    df["__firm_key"] = _normalise_firm_key(df[STAGE1_KEY])
    df["__year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df["rating_num_10"] = pd.to_numeric(df["rating_num_10"], errors="coerce")
    df = df[df["__firm_key"].ne("") & df["__year"].notna()].copy()

    conflicts = (
        df[df[score_col].notna()]
        .groupby(["__firm_key", "__year"])[score_col]
        .nunique(dropna=True)
    )
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        sample = [
            {"firm_key": str(k[0]), "year": int(k[1]), "n_distinct_scores": int(v)}
            for k, v in conflicts.head(10).items()
        ]
        raise ValueError(f"Stage1 panel has conflicting duplicate firm-year scores: {sample}")

    # Keep the first row per key.  Conflicting non-null values already failed.
    df = df.drop_duplicates(subset=["__firm_key", "__year"], keep="first")
    recs: list[dict[str, Any]] = []
    split_col = "split_stage4" if "split_stage4" in df.columns else None
    for firm_key, g in df.sort_values(["__firm_key", "__year"]).groupby("__firm_key", sort=False):
        by_year = {int(row["__year"]): row for _, row in g.iterrows() if not pd.isna(row["__year"])}
        years = sorted(by_year)
        for y in years:
            if (y - 1) not in by_year or (y + 1) not in by_year:
                continue
            row_tm1 = by_year[y - 1]
            row_t = by_year[y]
            row_tp1 = by_year[y + 1]
            score_tm1 = pd.to_numeric(pd.Series([row_tm1.get(score_col)]), errors="coerce").iloc[0]
            score_t = pd.to_numeric(pd.Series([row_t.get(score_col)]), errors="coerce").iloc[0]
            score_tp1 = pd.to_numeric(pd.Series([row_tp1.get(score_col)]), errors="coerce").iloc[0]
            rating_t = pd.to_numeric(pd.Series([row_t.get("rating_num_10")]), errors="coerce").iloc[0]
            rating_tp1 = pd.to_numeric(pd.Series([row_tp1.get("rating_num_10")]), errors="coerce").iloc[0]
            if pd.isna(score_tm1) or pd.isna(score_t) or pd.isna(rating_t) or pd.isna(rating_tp1):
                continue
            recs.append({
                "__firm_key": str(firm_key),
                "firm_id_stage1": row_t.get(STAGE1_KEY),
                "base_year_t": int(y),
                "target_year_tplus1": int(y + 1),
                "b1_score_tminus1": float(score_tm1),
                "b1_score_t": float(score_t),
                "b1_score_tplus1_actual": None if pd.isna(score_tp1) else float(score_tp1),
                "b1_delta_score_lead_tminus1_to_t": float(score_t - score_tm1),
                "b1_delta_score_actual_forward_t_to_tplus1": None if pd.isna(score_tp1) else float(score_tp1 - score_t),
                "b1_rating_t": float(rating_t),
                "b1_rating_tplus1": float(rating_tp1),
                "b1_rating_improvement_t_to_tplus1": float(rating_t - rating_tp1),
                "b1_d_rating_raw_tplus1_minus_t": float(rating_tp1 - rating_t),
                "b1_split_at_tplus1": row_tp1.get(split_col) if split_col else "",
            })
    return pd.DataFrame(recs)


def _load_b2_scored_output(project_root: Path, b2_csv: Path | None = None) -> pd.DataFrame:
    final = project_root / "data" / "final_freeze"
    path = b2_csv or final / "stage2_substrate_loopA_loopB2" / B2_SCORE_CSV_NAME
    if not path.exists():
        raise FileNotFoundError(f"Missing Loop B2 scored CSV: {path}")
    b2 = pd.read_csv(path, encoding="utf-8-sig")
    required = {"firm_id", "fiscal_year", "score_t_real", "pred_alpha_score_tplus1", "real_rating_delta_t_to_tplus1"}
    missing = sorted(required - set(b2.columns))
    if missing:
        raise ValueError(f"Loop B2 scored CSV missing required columns: {missing}; path={path}")
    b2 = b2.copy()
    b2["__firm_key"] = _normalise_firm_key(b2["firm_id"])
    b2["base_year_t"] = pd.to_numeric(b2["fiscal_year"], errors="coerce").astype("Int64")
    b2["target_year_tplus1"] = pd.to_numeric(_transition_target_years(b2), errors="coerce").astype("Int64")
    if "is_observed_transition" in b2.columns:
        b2["is_observed_transition"] = _bool_mask(b2, "is_observed_transition", default=True)
    else:
        b2["is_observed_transition"] = True
    if "loopB2_oot_2020_2023" in b2.columns:
        b2["loopB2_oot_2020_2023"] = _bool_mask(b2, "loopB2_oot_2020_2023", default=False)
    else:
        b2["loopB2_oot_2020_2023"] = pd.to_numeric(b2["target_year_tplus1"], errors="coerce").between(
            B2_OOT_YEAR_MIN, B2_OOT_YEAR_MAX, inclusive="both"
        )
    b2["score_t_real"] = pd.to_numeric(b2["score_t_real"], errors="coerce")
    b2["pred_alpha_score_tplus1"] = pd.to_numeric(b2["pred_alpha_score_tplus1"], errors="coerce")
    if "delta_sim_alpha_score_t_to_pred_tplus1" not in b2.columns:
        b2["delta_sim_alpha_score_t_to_pred_tplus1"] = b2["pred_alpha_score_tplus1"] - b2["score_t_real"]
    else:
        b2["delta_sim_alpha_score_t_to_pred_tplus1"] = pd.to_numeric(
            b2["delta_sim_alpha_score_t_to_pred_tplus1"], errors="coerce"
        )
    b2["delta_sim_alpha_score_inverted_pred_to_t"] = -b2["delta_sim_alpha_score_t_to_pred_tplus1"]
    b2["real_rating_delta_t_to_tplus1"] = pd.to_numeric(b2["real_rating_delta_t_to_tplus1"], errors="coerce")
    b2["real_rating_downgrade_t_to_tplus1"] = -b2["real_rating_delta_t_to_tplus1"]
    return b2


def _load_loopb2_report(final: Path) -> dict[str, Any]:
    p = final / "stage2_substrate_loopA_loopB2" / B2_REPORT_NAME
    if not p.exists():
        return {"status": "MISSING", "path": str(p)}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "PARSE_ERROR", "path": str(p), "error": str(exc)}


def _score_t_consistency(joined: pd.DataFrame) -> dict[str, Any]:
    diff = pd.to_numeric(joined.get("score_t_real"), errors="coerce") - pd.to_numeric(joined.get("b1_score_t"), errors="coerce")
    finite = diff.dropna()
    abs_diff = finite.abs()
    return {
        "rows": int(len(joined)),
        "finite_rows": int(len(finite)),
        "max_abs_diff": None if abs_diff.empty else float(abs_diff.max()),
        "mean_abs_diff": None if abs_diff.empty else float(abs_diff.mean()),
        "mismatch_rows_abs_gt_1e_9": int((abs_diff > 1e-9).sum()) if not abs_diff.empty else None,
        "status": "PASS" if not abs_diff.empty and int((abs_diff > 1e-9).sum()) == 0 else "FAIL_OR_EMPTY",
    }


def _rating_delta_consistency(joined: pd.DataFrame) -> dict[str, Any]:
    diff = pd.to_numeric(joined.get("real_rating_delta_t_to_tplus1"), errors="coerce") - pd.to_numeric(
        joined.get("b1_rating_improvement_t_to_tplus1"), errors="coerce"
    )
    finite = diff.dropna()
    abs_diff = finite.abs()
    return {
        "rows": int(len(joined)),
        "finite_rows": int(len(finite)),
        "max_abs_diff": None if abs_diff.empty else float(abs_diff.max()),
        "mismatch_rows_abs_gt_1e_9": int((abs_diff > 1e-9).sum()) if not abs_diff.empty else None,
        "status": "PASS" if not abs_diff.empty and int((abs_diff > 1e-9).sum()) == 0 else "FAIL_OR_EMPTY",
    }


def _summarise_block(name: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    metrics = [
        ("b2_current_delta_vs_rating_improvement", "delta_sim_alpha_score_t_to_pred_tplus1", "real_rating_delta_t_to_tplus1"),
        ("b2_inverted_delta_vs_rating_improvement", "delta_sim_alpha_score_inverted_pred_to_t", "real_rating_delta_t_to_tplus1"),
        ("b2_current_delta_vs_rating_downgrade", "delta_sim_alpha_score_t_to_pred_tplus1", "real_rating_downgrade_t_to_tplus1"),
        ("b1_lead_delta_vs_rating_improvement", "b1_delta_score_lead_tminus1_to_t", "b1_rating_improvement_t_to_tplus1"),
        ("stage1_actual_forward_delta_vs_rating_improvement", "b1_delta_score_actual_forward_t_to_tplus1", "b1_rating_improvement_t_to_tplus1"),
        ("b2_current_delta_vs_b1_lead_delta", "delta_sim_alpha_score_t_to_pred_tplus1", "b1_delta_score_lead_tminus1_to_t"),
        ("b2_current_delta_vs_stage1_actual_forward_delta", "delta_sim_alpha_score_t_to_pred_tplus1", "b1_delta_score_actual_forward_t_to_tplus1"),
    ]
    rows: list[dict[str, Any]] = []
    for metric, score_col, rating_col in metrics:
        s = _sign_agreement_stats(df, score_col, rating_col)
        rows.append({"sample": name, "metric": metric, **s})
    return rows


def _classify_alignment(summary_rows: list[dict[str, Any]], report_meta: dict[str, Any]) -> dict[str, Any]:
    by_key = {(r.get("sample"), r.get("metric")): r for r in summary_rows}
    cur = by_key.get(("oot", "b2_current_delta_vs_rating_improvement"), {})
    inv = by_key.get(("oot", "b2_inverted_delta_vs_rating_improvement"), {})
    b1 = by_key.get(("oot", "b1_lead_delta_vs_rating_improvement"), {})
    score_t = report_meta.get("score_t_consistency", {})
    rating_delta = report_meta.get("rating_delta_consistency", {})
    cur_agree = cur.get("agreement")
    inv_agree = inv.get("agreement")
    cur_ci = cur.get("ci95") or [None, None]
    verdict = "UNCLASSIFIED"
    reasons: list[str] = []
    if score_t.get("status") != "PASS":
        reasons.append("score_t_real does not match Stage1 B1 score_t on the joined transitions")
    if rating_delta.get("status") != "PASS":
        reasons.append("real_rating_delta_t_to_tplus1 does not match B1 rating improvement convention")
    if cur_agree is not None and float(cur_agree) < 0.5:
        reasons.append("B2 current direction agreement is below 50% on OOT movers")
    if cur_ci and cur_ci[1] is not None and float(cur_ci[1]) < 0.5:
        reasons.append("B2 current OOT Wilson upper CI is below 50%, indicating below-chance alignment")
    if inv_agree is not None and float(inv_agree) > 0.5:
        reasons.append("Inverting the B2 score delta improves agreement above 50%, a sign-orientation suspect")
    if score_t.get("status") == "PASS" and rating_delta.get("status") == "PASS" and cur_agree is not None:
        if float(cur_agree) < 0.5 and inv_agree is not None and float(inv_agree) > 0.5:
            verdict = "SYSTEMATIC_B2_SCORE_DELTA_INVERSION_SUSPECT"
        elif float(cur_agree) < 0.5:
            verdict = "B2_BELOW_CHANCE_ALIGNMENT_REQUIRES_SCORE_PATH_AUDIT"
        else:
            verdict = "NO_BELOW_CHANCE_B2_SIGN_INVERSION_EVIDENCE"
    elif reasons:
        verdict = "CONTRACT_MISMATCH_BEFORE_B2_PERFORMANCE_INTERPRETATION"
    return {
        "diagnosis_status": verdict,
        "reasons": reasons,
        "interpretation": (
            "Do not treat a below-chance B2 agreement as substantive partial_pass until score_t, rating_delta, "
            "and B2 score-delta orientation have been audited on identical transitions."
        ),
        "oot_b2_current_agreement": cur_agree,
        "oot_b2_current_ci95": cur.get("ci95"),
        "oot_b2_inverted_agreement": inv_agree,
        "oot_b1_lead_overlap_agreement": b1.get("agreement"),
    }


def run(project_root: Path, *, b2_csv: Path | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    final = root / "data" / "final_freeze"
    out_dir = output_dir or final / "stage2_substrate_loopA_loopB2"
    out_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    score_panel, score_col, score_path, score_status = load_stage1_backend_score_panel(root, B2_BACKEND, errors)
    if score_status != "ok" or errors:
        raise RuntimeError(f"Could not load Stage1 {B2_BACKEND} score panel via shared B1 loader: status={score_status} errors={errors}")
    b1_panel = _build_stage1_transition_panel(score_panel, score_col)
    b2 = _load_b2_scored_output(root, b2_csv)
    b2_observed = b2[b2["is_observed_transition"]].copy()

    if b2_observed.duplicated(subset=["__firm_key", "base_year_t", "target_year_tplus1"]).any():
        dup = b2_observed[b2_observed.duplicated(subset=["__firm_key", "base_year_t", "target_year_tplus1"], keep=False)]
        sample = dup[["firm_id", "fiscal_year", "target_year_tplus1"]].head(10).to_dict("records")
        raise ValueError(f"B2 observed scored CSV has duplicate firm/base-year/target-year rows: {sample}")
    if b1_panel.duplicated(subset=["__firm_key", "base_year_t", "target_year_tplus1"]).any():
        dup = b1_panel[b1_panel.duplicated(subset=["__firm_key", "base_year_t", "target_year_tplus1"], keep=False)]
        sample = dup[["firm_id_stage1", "base_year_t", "target_year_tplus1"]].head(10).to_dict("records")
        raise ValueError(f"B1 reconstructed transition panel has duplicate keys: {sample}")

    joined = b2_observed.merge(
        b1_panel,
        on=["__firm_key", "base_year_t", "target_year_tplus1"],
        how="left",
        validate="one_to_one",
        suffixes=("_b2", "_b1"),
    )
    joined["b1_joined"] = joined["b1_score_t"].notna()
    joined["score_t_real_minus_b1_score_t"] = pd.to_numeric(joined["score_t_real"], errors="coerce") - pd.to_numeric(joined["b1_score_t"], errors="coerce")
    joined["b2_rating_delta_minus_b1_improvement"] = pd.to_numeric(joined["real_rating_delta_t_to_tplus1"], errors="coerce") - pd.to_numeric(joined["b1_rating_improvement_t_to_tplus1"], errors="coerce")
    joined["b2_current_sign_match_rating"] = (
        np.sign(pd.to_numeric(joined["delta_sim_alpha_score_t_to_pred_tplus1"], errors="coerce"))
        == np.sign(pd.to_numeric(joined["real_rating_delta_t_to_tplus1"], errors="coerce"))
    ) & pd.to_numeric(joined["real_rating_delta_t_to_tplus1"], errors="coerce").ne(0)
    joined["b2_inverted_sign_match_rating"] = (
        np.sign(pd.to_numeric(joined["delta_sim_alpha_score_inverted_pred_to_t"], errors="coerce"))
        == np.sign(pd.to_numeric(joined["real_rating_delta_t_to_tplus1"], errors="coerce"))
    ) & pd.to_numeric(joined["real_rating_delta_t_to_tplus1"], errors="coerce").ne(0)
    joined["b1_lead_sign_match_rating"] = (
        np.sign(pd.to_numeric(joined["b1_delta_score_lead_tminus1_to_t"], errors="coerce"))
        == np.sign(pd.to_numeric(joined["b1_rating_improvement_t_to_tplus1"], errors="coerce"))
    ) & pd.to_numeric(joined["b1_rating_improvement_t_to_tplus1"], errors="coerce").ne(0)

    joined_path = out_dir / ROWS_NAME
    joined.to_csv(joined_path, index=False, encoding="utf-8-sig")

    oot = joined[joined["loopB2_oot_2020_2023"]].copy()
    oot_movers = oot[pd.to_numeric(oot["real_rating_delta_t_to_tplus1"], errors="coerce").ne(0)].copy()
    oot_mismatches = oot_movers[~oot_movers["b2_current_sign_match_rating"].fillna(False)].copy()
    oot_mismatch_cols = [
        c for c in [
            "firm_id", "firm_id_stage1", "base_year_t", "target_year_tplus1", "candidate_id",
            "score_t_real", "pred_alpha_score_tplus1", "delta_sim_alpha_score_t_to_pred_tplus1",
            "delta_sim_alpha_score_inverted_pred_to_t", "real_rating_delta_t_to_tplus1",
            "b1_score_tminus1", "b1_score_t", "b1_score_tplus1_actual",
            "b1_delta_score_lead_tminus1_to_t", "b1_delta_score_actual_forward_t_to_tplus1",
            "b1_rating_improvement_t_to_tplus1", "b2_current_sign_match_rating",
            "b2_inverted_sign_match_rating", "b1_lead_sign_match_rating",
        ] if c in oot_mismatches.columns
    ]
    oot_mismatches[oot_mismatch_cols].to_csv(out_dir / OOT_MISMATCH_NAME, index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    summary_rows.extend(_summarise_block("all_observed", joined))
    summary_rows.extend(_summarise_block("oot", oot))
    summary_rows.extend(_summarise_block("oot_joined_to_b1", oot[oot["b1_joined"]].copy()))
    pd.DataFrame(summary_rows).to_csv(out_dir / SUMMARY_CSV_NAME, index=False, encoding="utf-8-sig")

    score_t_consistency = _score_t_consistency(joined[joined["b1_joined"]].copy())
    rating_delta_consistency = _rating_delta_consistency(joined[joined["b1_joined"]].copy())
    report_context = {
        "score_t_consistency": score_t_consistency,
        "rating_delta_consistency": rating_delta_consistency,
    }
    diagnosis = _classify_alignment(summary_rows, report_context)
    loopb2_report = _load_loopb2_report(final)
    report = {
        "stage": "diagnose_loopb2_b1_alignment",
        "status": "PASS",
        "purpose": "Side-by-side audit of B1 score/rating conventions versus B2 pred-score/rating conventions on the same observed transitions.",
        "does_not_change_gate_verdict": True,
        "project_root": str(root),
        "inputs": {
            "stage1_score_panel": str(score_path),
            "stage1_score_loader": "verify_stage1_substrate_validation.load_stage1_backend_score_panel",
            "stage1_score_col": score_col,
            "b2_scored_csv": str(b2_csv or final / "stage2_substrate_loopA_loopB2" / B2_SCORE_CSV_NAME),
            "loopb2_report": str(final / "stage2_substrate_loopA_loopB2" / B2_REPORT_NAME),
        },
        "conventions": {
            "stage1_B1_gate": "score change t-1->t versus rating improvement t->t+1",
            "stage2_B2_gate": "pred_alpha_score_tplus1 - score_t_real versus rating_num_10(t)-rating_num_10(t+1)",
            "rating_num_10": "lower_is_better; positive rating improvement means rating_num_10(t) - rating_num_10(t+1) > 0",
            "score": "higher_is_better; positive score delta means model score improvement",
        },
        "row_counts": {
            "b1_transition_rows_reconstructed": int(len(b1_panel)),
            "b2_rows_total": int(len(b2)),
            "b2_observed_rows": int(len(b2_observed)),
            "joined_rows": int(len(joined)),
            "joined_to_b1_rows": int(joined["b1_joined"].sum()),
            "oot_rows": int(len(oot)),
            "oot_movers": int(len(oot_movers)),
            "oot_current_mismatches": int(len(oot_mismatches)),
        },
        "contract_checks": {
            "score_t_consistency": score_t_consistency,
            "rating_delta_consistency": rating_delta_consistency,
        },
        "summary": summary_rows,
        "diagnosis": diagnosis,
        "loopb2_report_contract_snapshot": {
            "status": loopb2_report.get("status"),
            "substrate_tier": loopb2_report.get("substrate_tier"),
            "loopB2_status": (loopb2_report.get("loopB2") or {}).get("status") if isinstance(loopb2_report.get("loopB2"), dict) else None,
            "loopB2_rule_status": (loopb2_report.get("loopB2") or {}).get("rule_status") if isinstance(loopb2_report.get("loopB2"), dict) else None,
            "loopB2_agreement": (loopb2_report.get("loopB2") or {}).get("agreement") if isinstance(loopb2_report.get("loopB2"), dict) else None,
            "loopB2_ci95": (loopb2_report.get("loopB2") or {}).get("ci95") if isinstance(loopb2_report.get("loopB2"), dict) else None,
            "loopB2_gap_pp": (loopb2_report.get("loopB2") or {}).get("gap_pp") if isinstance(loopb2_report.get("loopB2"), dict) else None,
            "b1_ref_agreement": (((loopb2_report.get("loopB2") or {}).get("b1_ref") or {}).get("agreement") if isinstance(loopb2_report.get("loopB2"), dict) else None),
        },
        "outputs": {
            "joined_rows_csv": str(joined_path),
            "oot_mover_mismatches_csv": str(out_dir / OOT_MISMATCH_NAME),
            "summary_csv": str(out_dir / SUMMARY_CSV_NAME),
            "report_json": str(out_dir / REPORT_NAME),
        },
    }
    write_json(out_dir / REPORT_NAME, _json_safe(report))
    return _json_safe(report)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Diagnose LoopB2 vs Stage1 B1 sign/scale alignment on identical transitions")
    p.add_argument("--project-root", required=True)
    p.add_argument("--b2-csv", default=None, help="Optional explicit path to loopB2_alpha_predicted_score_vs_real_rating_change.csv")
    p.add_argument("--output-dir", default=None, help="Optional output directory; defaults to the LoopB2 output dir")
    p.add_argument(
        "--fail-on-inversion-suspect",
        action="store_true",
        help="Exit 2 if the diagnostic classifies the result as a systematic B2 score-delta inversion suspect.",
    )
    args = p.parse_args(argv)
    report = run(
        Path(args.project_root),
        b2_csv=Path(args.b2_csv).resolve() if args.b2_csv else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    status = ((report.get("diagnosis") or {}).get("diagnosis_status") or "")
    if args.fail_on_inversion_suspect and "INVERSION" in status:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
