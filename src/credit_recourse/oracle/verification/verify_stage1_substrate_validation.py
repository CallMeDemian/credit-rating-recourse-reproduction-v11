"""Stage 1 Loop B1 substrate-validation gate.

Tests whether the Oracle score *change* tracks the real 10-grade rating *change*
(RESEARCH_FINAL_METHODOLOGY_AND_DESIGN.md §4.2, RQ0). Simulator-independent: runs
entirely on the Stage 1 backend firm-year outputs.

The contemporaneous oriented alignment (level Spearman) serves as the premise
gate — reported alongside its Wilson CI — while the pass/fail VERDICT is
computed on the LEAD relationship: the Oracle score change over (t-1 -> t)
versus the real rating change over (t -> t+1). (Docstring corrected 2026-07-04,
ORA-DOC-001: an earlier draft wrongly described the contemporaneous
relationship as null.) Loops A and B2 (simulator-mediated) are NOT part of this gate; they
run as a Stage 2 extension.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from credit_recourse.rl.common.io import write_json

KEY = "거래소코드"

# Pre-registered thresholds (RESEARCH §4.2; fixed before the final run).
# rating_num_10 is encoded lower-is-better (AAA=1 ... D=10), while R_score_* is
# higher-is-better. Therefore the raw level Spearman should be negative when the
# backend is valid. Verdicts use the oriented value: -raw_spearman.
LEVEL_SPEARMAN_MIN = 0.55
LOOPB1_AGREEMENT_TARGET = 0.65
LOOPB1_SPEARMAN_TARGET = 0.35
RATING_NUM_10_ORIENTATION = "lower_is_better"
SCORE_ORIENTATION = "higher_is_better"

BACKENDS = {
    "alpha": ("alpha/oracle_firm_year_output_alpha.parquet", "R_score_alpha"),
    "beta": ("beta/benchmark_firm_year_output_beta.parquet", "R_score_beta"),
    "gamma": ("gamma/benchmark_firm_year_output_gamma.parquet", "R_score_gamma"),
}
DIRECTION_AGREEMENT_MIN_MOVERS = 5
DIRECTION_AGREEMENT_CI_METHOD = "wilson"


def _wilson_ci(k: int, n: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    The helper is deliberately local to the Stage1 verifier so downstream Loop B2
    can reuse the exact same CI and sign-agreement convention as Loop B1.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    z = float(stats.norm.ppf(1.0 - (1.0 - confidence_level) / 2.0))
    phat = float(k) / float(n)
    denom = 1.0 + z * z / float(n)
    centre = phat + z * z / (2.0 * float(n))
    margin = z * np.sqrt((phat * (1.0 - phat) + z * z / (4.0 * float(n))) / float(n))
    low = (centre - margin) / denom
    high = (centre + margin) / denom
    return (float(max(0.0, low)), float(min(1.0, high)))


def compute_direction_agreement(
    score_delta: pd.Series | np.ndarray | list[Any],
    rating_improvement: pd.Series | np.ndarray | list[Any],
    *,
    min_movers: int = DIRECTION_AGREEMENT_MIN_MOVERS,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Compute mover-only score/rating direction agreement with Wilson CI.

    Convention shared by Loop B1 and Loop B2:
    * rows with real rating delta/improvement equal to zero are not movers and
      are excluded from the denominator;
    * score-delta zero among real movers is retained and counted as a
      non-match, because ``sign(0)`` cannot agree with an upgrade/downgrade;
    * rating_num_10 is lower-is-better, so positive ``rating_improvement`` means
      a real credit-rating improvement.
    """
    s = pd.to_numeric(pd.Series(score_delta), errors="coerce")
    r = pd.to_numeric(pd.Series(rating_improvement), errors="coerce")
    valid = s.notna() & r.notna() & r.ne(0)
    movers_s = s[valid]
    movers_r = r[valid]
    out: dict[str, Any] = {
        "n_movers": int(valid.sum()),
        "n_valid_direction_rows": int(valid.sum()),
        "n_score_delta_zero_movers": int((movers_s == 0).sum()),
        "delta_zero_rule": "rating_delta_zero_excluded; score_delta_zero_among_movers_counts_as_nonmatch",
        "ci_method": DIRECTION_AGREEMENT_CI_METHOD,
        "confidence_level": float(confidence_level),
    }
    if int(valid.sum()) < int(min_movers):
        out["status"] = "insufficient_movers"
        out["min_movers"] = int(min_movers)
        return out
    sign_match = np.sign(movers_s.to_numpy(dtype=float)) == np.sign(movers_r.to_numpy(dtype=float))
    n = int(len(movers_s))
    k = int(sign_match.sum())
    ci_low, ci_high = _wilson_ci(k, n, confidence_level=confidence_level)
    out.update({
        "status": "ok",
        "n_matches": k,
        "direction_agreement": float(k / n),
        "direction_agreement_ci95": [float(ci_low), float(ci_high)],
    })
    return out


def _auc_upgrade_vs_downgrade(score: np.ndarray, improvement: np.ndarray) -> float | None:
    """Probability a randomly chosen upgrade has a higher score change than a downgrade."""
    up = score[improvement > 0]
    dn = score[improvement < 0]
    if len(up) == 0 or len(dn) == 0:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in up for b in dn)
    return float(wins / (len(up) * len(dn)))


def _lead_pairs(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Build (rating change t->t+1, score change t-1->t) records per firm.

    Requires three consecutive fiscal years (t-1, t, t+1) for the lead score change.
    """
    recs: list[dict[str, Any]] = []
    for _, g in df.sort_values([KEY, "year"]).groupby(KEY):
        yrs = g["year"].to_numpy()
        rat = g["rating_num_10"].to_numpy(dtype=float)
        sc = g[score_col].to_numpy(dtype=float)
        sp = g["split_stage4"].to_numpy() if "split_stage4" in g.columns else np.array([""] * len(g))
        for i in range(1, len(g) - 1):
            if yrs[i + 1] - yrs[i] != 1 or yrs[i] - yrs[i - 1] != 1:
                continue
            if np.isnan(rat[i]) or np.isnan(rat[i + 1]) or np.isnan(sc[i]) or np.isnan(sc[i - 1]):
                continue
            recs.append({
                "d_rating": rat[i + 1] - rat[i],          # >0 = downgrade (higher num = worse)
                "improvement": -(rat[i + 1] - rat[i]),     # >0 = upgrade
                "d_score_lead": sc[i] - sc[i - 1],         # score change t-1 -> t
                "split": sp[i + 1],
            })
    return pd.DataFrame(recs, columns=["d_rating", "improvement", "d_score_lead", "split"])


def _evaluate_split(pairs: pd.DataFrame, level_df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    movers = pairs[pairs["d_rating"] != 0]
    out: dict[str, Any] = {"n_pairs": int(len(pairs)), "n_movers": int(len(movers))}
    # level validity (premise): score vs rating level
    lv = level_df.dropna(subset=[score_col, "rating_num_10"])
    raw_level_rho = (
        float(stats.spearmanr(lv[score_col], lv["rating_num_10"]).statistic) if len(lv) > 2 else None
    )
    # R_score_* is higher-is-better, but rating_num_10 is lower-is-better.
    # A valid backend should therefore show a negative raw rho. The pre-registered
    # level-validity threshold is applied to the oriented value.
    oriented_level_rho = -raw_level_rho if raw_level_rho is not None else None
    out["level_validity_spearman_raw"] = raw_level_rho
    out["level_validity_spearman_oriented"] = oriented_level_rho
    # Backward-compatible alias, but do not use this field for verdicts.
    out["level_validity_spearman"] = raw_level_rho
    out["rating_num_10_orientation"] = RATING_NUM_10_ORIENTATION
    out["score_orientation"] = SCORE_ORIENTATION
    agreement = compute_direction_agreement(movers["d_score_lead"], movers["improvement"])
    out["direction_agreement_ci_method"] = agreement.get("ci_method")
    out["delta_zero_rule"] = agreement.get("delta_zero_rule")
    out["n_score_delta_zero_movers"] = agreement.get("n_score_delta_zero_movers")
    if agreement.get("status") == "insufficient_movers":
        out["status"] = "insufficient_movers"
        out["min_movers"] = agreement.get("min_movers")
        return out
    out["lead_direction_agreement"] = agreement["direction_agreement"]
    out["lead_direction_agreement_ci95"] = agreement["direction_agreement_ci95"]
    out["lead_direction_agreement_matches"] = agreement["n_matches"]
    rho = stats.spearmanr(movers["d_score_lead"], movers["improvement"])
    out["lead_spearman"] = float(rho.statistic)
    out["lead_spearman_p"] = float(rho.pvalue)
    out["upgrade_vs_downgrade_auc"] = _auc_upgrade_vs_downgrade(
        movers["d_score_lead"].to_numpy(), movers["improvement"].to_numpy()
    )
    return out


def _verdict(oot: dict[str, Any]) -> str:
    """Three-tier verdict from the out-of-sample (held-out) mover statistics."""
    if oot.get("status") == "insufficient_movers" or "lead_direction_agreement" not in oot:
        return "fail"
    ci_low = oot["lead_direction_agreement_ci95"][0]
    agree = oot["lead_direction_agreement"]
    lv = oot.get("level_validity_spearman_oriented")
    if lv is not None and lv < LEVEL_SPEARMAN_MIN:
        return "fail"               # premise (oriented level validity) not met
    if ci_low <= 0.50:
        return "fail"               # not significantly above chance
    if agree >= LOOPB1_AGREEMENT_TARGET:
        return "strong_pass"
    return "partial_pass"


def _load_authoritative_rating_map(backends_dir: Path, errors: list[str]) -> pd.DataFrame:
    """Load the 10-grade rating map used by the Stage1 B1 verifier."""
    alpha_path = backends_dir / BACKENDS["alpha"][0]
    rating_map = pd.DataFrame(columns=[KEY, "year", "rating_num_10"])
    if alpha_path.exists():
        a = pd.read_parquet(alpha_path)
        if {KEY, "year", "rating_num_10"}.issubset(a.columns):
            rating_map = a[[KEY, "year", "rating_num_10"]].drop_duplicates()
        else:
            errors.append("alpha output missing one of 거래소코드/year/rating_num_10 for rating map")
    else:
        errors.append(f"missing alpha output for authoritative rating map: {alpha_path}")
    return rating_map


def _load_backend_score_panel(
    name: str,
    path: Path,
    score_col: str,
    rating_map: pd.DataFrame,
    errors: list[str],
) -> tuple[pd.DataFrame | None, str, str]:
    """Load the firm-year score/rating panel exactly as Loop B1 evaluates it."""
    if not path.exists():
        errors.append(f"missing backend output: {path}")
        return None, score_col, "missing_output"
    df = pd.read_parquet(path)
    if score_col not in df.columns:
        # fall back: resolve any R_score_* column
        cand = [c for c in df.columns if str(c).startswith("R_score")]
        if not cand:
            errors.append(f"{name}: no score column found in {path.name}")
            return None, score_col, "no_score_column"
        score_col = cand[0]
    for c in (KEY, "year"):
        if c not in df.columns:
            errors.append(f"{name}: output missing key column {c}")
            return None, score_col, "missing_key_column"
    # authoritative 10-grade rating: use the file's own column if present, else join from alpha
    if "rating_num_10" not in df.columns:
        df = df.merge(rating_map, on=[KEY, "year"], how="left")
    keep = [KEY, "year", score_col, "rating_num_10"] + (["split_stage4"] if "split_stage4" in df.columns else [])
    df = df[keep].copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["rating_num_10"] = pd.to_numeric(df["rating_num_10"], errors="coerce")
    return df, score_col, "ok"


def load_stage1_backend_score_panel(
    project_root: Path,
    backend: str = "alpha",
    errors: list[str] | None = None,
) -> tuple[pd.DataFrame, str, Path, str]:
    """Public Loop-B1 score-panel loader reused by Stage2 Loop B2.

    Returns ``(panel, resolved_score_col, backend_path, status)``.  The returned
    panel is the same firm-year score/rating table that ``verify_backend()`` uses
    for the Stage1 B1 agreement calculation, so later B1-vs-B2 comparisons do
    not drift by path, score definition, rating source, or split handling.
    """
    err_list = errors if errors is not None else []
    if backend not in BACKENDS:
        raise KeyError(f"Unknown Stage1 backend for substrate validation: {backend}")
    root = Path(project_root).resolve()
    backends_dir = root / "data" / "final_freeze" / "stage1_oracle_backends"
    rating_map = _load_authoritative_rating_map(backends_dir, err_list)
    rel, score_col = BACKENDS[backend]
    backend_path = backends_dir / rel
    panel, resolved_score_col, status = _load_backend_score_panel(backend, backend_path, score_col, rating_map, err_list)
    if panel is None:
        panel = pd.DataFrame(columns=[KEY, "year", resolved_score_col, "rating_num_10"])
    return panel, resolved_score_col, backend_path, status


def evaluate_stage1_backend_panel(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    """Evaluate a loaded Stage1 backend score panel with the Loop B1 logic."""
    pairs = _lead_pairs(df, score_col)
    res: dict[str, Any] = {"all": _evaluate_split(pairs, df, score_col)}
    if "split" in pairs.columns and (pairs["split"] == "oot").any():
        res["oot"] = _evaluate_split(pairs[pairs["split"] == "oot"], df, score_col)
        res["dev"] = _evaluate_split(pairs[pairs["split"] == "dev"], df, score_col)
        res["verdict"] = _verdict(res["oot"])
        res["verdict_basis"] = "oot"
    else:
        res["verdict"] = _verdict(res["all"])
        res["verdict_basis"] = "all"
    return res


def verify_backend(name: str, path: Path, score_col: str, rating_map: pd.DataFrame,
                   errors: list[str]) -> dict[str, Any]:
    res: dict[str, Any] = {"backend": name}
    df, score_col, load_status = _load_backend_score_panel(name, path, score_col, rating_map, errors)
    res["score_column"] = score_col
    if df is None:
        res["status"] = load_status
        return res
    res.update(evaluate_stage1_backend_panel(df, score_col))
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Stage 1 Loop B1 substrate-validation gate")
    p.add_argument("--project-root", required=True)
    args = p.parse_args(argv)
    root = Path(args.project_root).resolve()
    final = root / "data" / "final_freeze"
    backends_dir = final / "stage1_oracle_backends"
    errors: list[str] = []

    # authoritative 10-grade rating map from the Alpha output
    rating_map = _load_authoritative_rating_map(backends_dir, errors)

    per_backend = {
        name: verify_backend(name, backends_dir / rel, col, rating_map, errors)
        for name, (rel, col) in BACKENDS.items()
    }

    verdicts = {n: r.get("verdict", "fail") for n, r in per_backend.items()}
    # the gate verdict is the Alpha (main backend) verdict; Beta/Gamma reported alongside
    gate_verdict = verdicts.get("alpha", "fail")

    result = {
        "stage": "verify_stage1_substrate_validation",
        "loop": "B1",
        "specification": "lead (Oracle score change t-1->t vs real 10-grade rating change t->t+1)",
        "scale": "rating_num_10",
        "thresholds": {
            "level_validity_spearman_oriented_min": LEVEL_SPEARMAN_MIN,
            "level_validity_orientation_rule": "oriented = -spearman(R_score, rating_num_10), because R_score is higher-is-better and rating_num_10 is lower-is-better",
            "loopB1_lead_direction_agreement_target": LOOPB1_AGREEMENT_TARGET,
            "loopB1_lead_direction_agreement_min_rule": "95pct_CI_lower_bound_gt_0.50",
            "loopB1_direction_agreement_ci_method": DIRECTION_AGREEMENT_CI_METHOD,
            "loopB1_delta_zero_rule": "rating_delta_zero_excluded; score_delta_zero_among_movers_counts_as_nonmatch",
            "loopB1_lead_spearman_target": LOOPB1_SPEARMAN_TARGET,
        },
        "per_backend": per_backend,
        "backend_verdicts": verdicts,
        "gate_verdict": gate_verdict,
        "gate_verdict_basis": "alpha_main_backend",
        "note": "Loops A and B2 (simulator-mediated) are a Stage 2 extension and are not part of this gate.",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
    }
    out = final / "ledgers" / "stage1_substrate_validation_loopB1.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # exit non-zero only on infrastructure errors; a 'fail' verdict is a finding, not a crash
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
