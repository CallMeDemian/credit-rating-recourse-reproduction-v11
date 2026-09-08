"""Stage 9 statistical inference: hypothesis tests and MDE pre-registration.

This module implements the per-hypothesis significance tests required by the
research design (§4.4) and contract v4 (MDE pre-registration).  It is
additive — it does not modify any Stage 7/8/9 pipeline outputs.

Hypotheses covered
------------------
H1 (scoring effect of reasoning):
    Paired Wilcoxon + Holm across (C5-C4), (C7-C6), and each reasoning
    condition minus C1_random_uniform.  Family = all H1 comparisons.

H2 (form effect of reasoning; thesis proposal §1.2 H2, §3.2-4):
    Reasoning prompts should make the recommendation more reviewable.
    Operationalized on the proposal's two auditability layers: paired
    Wilcoxon on free-form projection distance (C5-C4, C7-C6) and per-category
    exact McNemar on the eight-type failure taxonomy; rationale length is
    descriptive.  Holm across the whole H2 family.

H3 (selective adoption — reference identity contrast):
    Two-sided test of RL-adoption-ratio difference: C6 vs C6X.
    Sources: ``llm_stage9_identity_contrast.csv``.
    H3 also covers the categorical adoption rate (revised_candidate_id ==
    rl_reference_candidate), derived from the revision-metrics table.

H4-a (substrate robustness):
    For each primary result (C3>C0, C3>C2 directional, each LLM condition
    vs C0), verify the qualitative verdict is preserved across all three
    Oracle backends (α/β/γ).  Reports backend-specific Holm-adjusted p.

H4-b (action-space trade-off; proposal H4-b):
    Score side — paired Wilcoxon on delta_R(free_form_10d) minus
    delta_R(candidate_selection) per LLM condition × backend, Holm family.
    Auditability side — free-form projection distance / out-of-library /
    bound-clip descriptives per condition (candidate mode is 0-distance by
    construction).

H5 (information scope):
    Pairwise comparisons of mean delta_R_score across IC-a, IC-b, IC-c
    for each condition×mode cell (requires cross-run aggregation; gracefully
    degrades to skip when cross-run data is absent).

MDE pre-registration
--------------------
``--mde-freeze`` produces ``llm_stage9_mde_table.json`` before any LLM
API calls have been made for the remaining cells.  Once frozen, the MDE
table is hashed and the hash is recorded in Stage 9 metadata so that
post-hoc revision of the power analysis is detectable.

Run modes
---------
  python -m credit_recourse.eval.final_stage9_statistical_inference
         --project-root $Root
         [--mde-freeze]          # write MDE table without requiring Stage8
         [--llm-runs-dir DIR]    # H5: directory of per-IC run snapshots
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.contracts.stage_paths import final_root, stage_dir
from credit_recourse.eval.final_stage6_statistical_inference import (
    _holm_adjust,
    _wilcoxon,
    _cluster_bootstrap,
    _BOOTSTRAP_SEED_SCHEME,
)
from credit_recourse.rl.common.io import write_json

_SCHEMA_VERSION = "stage9_statistical_inference_v1"

# Hypothesis family definitions — used for Holm correction grouping.
# Each family is corrected independently; cross-family inflation is not corrected.
_H1_PAIRS = [
    ("C5", "C4"),   # reasoning vs direct, no reference
    ("C7", "C6"),   # reasoning vs direct, RL reference
]
_H1_RANDOM_REF = "C1_random_uniform"  # if available in comparison frame


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mde_content_for_compare(obj: Any) -> Any:
    """Return the MDE table content with volatile timestamps removed.

    ``--mde-freeze`` is a pre-registration artifact, so repeated invocations
    with identical statistical content must not churn the file hash merely
    because ``created_utc`` changed.  The on-disk file keeps its original
    timestamp when the content is unchanged.
    """
    if isinstance(obj, dict):
        return {str(k): _mde_content_for_compare(v) for k, v in obj.items() if str(k) != "created_utc"}
    if isinstance(obj, list):
        return [_mde_content_for_compare(v) for v in obj]
    return obj


def _write_mde_if_changed(path: Path, obj: dict) -> tuple[bool, str]:
    """Write the MDE JSON only when non-volatile content changed.

    Returns ``(written, sha256_of_on_disk_file)``.  Existing files with the same
    content ignoring ``created_utc`` are preserved byte-for-byte to keep the
    preregistration hash stable across reruns.
    """
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _mde_content_for_compare(existing) == _mde_content_for_compare(obj):
                return False, _sha256_bytes(path.read_bytes())
        except Exception:
            pass
    write_json(path, obj)
    return True, _sha256_bytes(path.read_bytes())


def _bootstrap_seed_h(tag: str) -> int:
    return zlib.crc32(tag.encode()) & 0xFFFFFFFF


# ---------------------------------------------------------------------------
# Helpers: load Stage 9 outputs
# ---------------------------------------------------------------------------

def _load_comparison(project_root: Path) -> pd.DataFrame | None:
    p = stage_dir(project_root, "stage9") / "llm_stage9_llm_rl_comparison.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _load_revision_metrics(project_root: Path) -> pd.DataFrame | None:
    p = stage_dir(project_root, "stage9") / "llm_stage9_revision_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_identity_contrast(project_root: Path) -> pd.DataFrame | None:
    p = stage_dir(project_root, "stage9") / "llm_stage9_identity_contrast.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


def _load_stage6_eval(project_root: Path) -> pd.DataFrame | None:
    p = stage_dir(project_root, "stage6") / "multi_oracle_policy_eval.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Paired test: one comparison in a family
# ---------------------------------------------------------------------------

def _paired_test(
    df: pd.DataFrame,
    pol_a: str,
    pol_b: str,
    backend: str,
    *,
    mode_filter: str | None = None,
    cluster_col: str = "row_id",
) -> dict[str, Any]:
    """Return paired statistics for delta_R_score_{backend}(pol_a) - (pol_b)."""
    col = f"delta_R_score_{backend}"
    sub = df.copy()
    if mode_filter is not None and "mode" in sub.columns:
        sub = sub[sub["mode"].astype(str) == mode_filter]
    piv = sub.pivot_table(index="row_id", columns="policy", values=col, aggfunc="first")
    if pol_a not in piv.columns or pol_b not in piv.columns:
        return {"status": "SKIP", "reason": f"{pol_a} or {pol_b} not in comparison frame"}
    diff = (piv[pol_a] - piv[pol_b]).dropna()
    if len(diff) == 0:
        return {"status": "SKIP", "reason": "no paired rows"}
    clusters = pd.Series(diff.index.astype(str), index=diff.index)
    seed = _bootstrap_seed_h(f"h_pair|{backend}|{pol_a}|{pol_b}|{mode_filter or 'all'}")
    se, lo, hi = _cluster_bootstrap(diff, clusters, seed=seed)
    W, p = _wilcoxon(diff)
    return {
        "status": "OK",
        "policy_a": pol_a,
        "policy_b": pol_b,
        "backend": backend,
        "mode_filter": mode_filter,
        "n_pairs": int(len(diff)),
        "mean_diff": float(diff.mean()),
        "median_diff": float(diff.median()),
        "cluster_se": se,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "wilcoxon_W": W,
        "wilcoxon_p": p,
        # holm_adjusted_p is filled by caller after family collection
    }


# ---------------------------------------------------------------------------
# H1
# ---------------------------------------------------------------------------

def run_h1(comparison: pd.DataFrame, backends: list[str] = ("alpha", "beta", "gamma")) -> dict:
    """H1: reasoning conditions score higher than direct conditions and C1."""
    rows: list[dict] = []
    for backend in backends:
        col = f"delta_R_score_{backend}"
        if col not in comparison.columns:
            continue
        for mode in sorted(comparison["mode"].astype(str).unique()) if "mode" in comparison.columns else [None]:
            for pol_a, pol_b in _H1_PAIRS:
                r = _paired_test(comparison, pol_a, pol_b, backend, mode_filter=mode)
                r["hypothesis"] = "H1"
                r["comparison_label"] = f"{pol_a}_minus_{pol_b}"
                rows.append(r)
            # vs C1: the proposal's H1 pre-registers only the REASONING
            # conditions against the random baseline ("이들 추론 조건은
            # 무작위 기준선(C1)보다") — C4/C6 vs C1 belong to the RQ2 ladder,
            # not this family (EVAL-P3-002).
            for pol_a in ["C5", "C7"]:
                if _H1_RANDOM_REF in set(comparison["policy"].astype(str)):
                    r = _paired_test(comparison, pol_a, _H1_RANDOM_REF, backend, mode_filter=mode)
                    r["hypothesis"] = "H1"
                    r["comparison_label"] = f"{pol_a}_minus_C1"
                    rows.append(r)
    # Holm within H1 family, per backend
    for backend in backends:
        fam = [i for i, r in enumerate(rows) if r.get("backend") == backend and r.get("status") == "OK"]
        pvals = [rows[i]["wilcoxon_p"] for i in fam]
        adj = _holm_adjust(pvals)
        for i, a in zip(fam, adj):
            rows[i]["holm_adjusted_p"] = a
    return {"hypothesis": "H1", "rows": rows}


# ---------------------------------------------------------------------------
# H2: form effect of reasoning (proposal §1.2 H2; auditability layers §3.2-4)
# ---------------------------------------------------------------------------

_H2_PAIRS = [("C5", "C4"), ("C7", "C6")]
_FAILURE_TAXONOMY = [
    "translational_failure", "structural_out_of_scope", "direction_error",
    "magnitude_error", "feasibility_violation", "liquidity_destructive_recourse",
    "anchoring_or_confirmation_failure", "ungrounded_judgment",
]


def _mcnemar_exact_p(n_only_a: int, n_only_b: int) -> float:
    """Exact McNemar p via a two-sided binomial test on discordant pairs."""
    n = int(n_only_a) + int(n_only_b)
    if n == 0:
        return 1.0
    from scipy.stats import binomtest
    return float(binomtest(int(n_only_a), n, 0.5).pvalue)


def run_h2(project_root: Path) -> dict:
    """H2: reasoning prompts change the FORM of the recommendation.

    Reads the Stage 7 failure audit (per-request, both modes).  Layer 1 —
    paired Wilcoxon on free-form projection distance (C5-C4, C7-C6).  Layer 2
    — per-category exact McNemar on the closed eight-type failure taxonomy,
    per mode.  Rationale length is descriptive only.  Holm is applied across
    every H2 test (projection + all category tests) as one pre-registered
    family.
    """
    audit_path = stage_dir(project_root, "stage8") / "llm_stage8_failure_audit_enriched.csv"
    if not audit_path.exists():
        audit_path = stage_dir(project_root, "stage7") / "llm_stage7_failure_audit.csv"
    if not audit_path.exists():
        return {"hypothesis": "H2", "status": "SKIP",
                "reason": f"failure audit not found: {audit_path}"}
    audit = pd.read_csv(audit_path)
    required = {"row_id", "policy", "mode", "failure_categories",
                "projection_distance", "rationale_length"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise KeyError(f"H2 requires stage7 failure-audit columns {missing} (file: {audit_path})")
    cat_sets = audit["failure_categories"].fillna("").astype(str).apply(
        lambda s: {t for t in s.split(",") if t}
    )
    for c in _FAILURE_TAXONOMY:
        audit[f"has_{c}"] = cat_sets.apply(lambda st, _c=c: _c in st)

    rows: list[dict] = []

    # Layer 1: projection distance (free-form rows only; candidate mode is
    # 0-distance by construction and would only dilute the paired test).
    ff = audit[audit["mode"].astype(str) == "free_form_10d"]
    if not ff.empty:
        piv = ff.pivot_table(index="row_id", columns="policy",
                             values="projection_distance", aggfunc="first")
        for pol_a, pol_b in _H2_PAIRS:
            if pol_a not in piv.columns or pol_b not in piv.columns:
                continue
            diff = (pd.to_numeric(piv[pol_a], errors="coerce")
                    - pd.to_numeric(piv[pol_b], errors="coerce")).dropna()
            if len(diff) == 0:
                continue
            W, p = _wilcoxon(diff)
            rows.append({
                "hypothesis": "H2", "measure": "projection_distance",
                "mode": "free_form_10d",
                "comparison_label": f"{pol_a}_minus_{pol_b}",
                "n_pairs": int(len(diff)),
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "wilcoxon_W": W, "p_value": p,
                "test": "wilcoxon_signed_rank",
            })

    # Layer 2: closed eight-type failure taxonomy, per mode.
    for mode in sorted(audit["mode"].astype(str).unique()):
        sub = audit[audit["mode"].astype(str) == mode]
        for pol_a, pol_b in _H2_PAIRS:
            for c in _FAILURE_TAXONOMY:
                piv = sub.pivot_table(index="row_id", columns="policy",
                                      values=f"has_{c}", aggfunc="first")
                if pol_a not in piv.columns or pol_b not in piv.columns:
                    continue
                both = piv[[pol_a, pol_b]].dropna()
                if both.empty:
                    continue
                a = both[pol_a].astype(bool)
                b = both[pol_b].astype(bool)
                n_a_only = int((a & ~b).sum())
                n_b_only = int((~a & b).sum())
                rows.append({
                    "hypothesis": "H2", "measure": f"failure_rate::{c}",
                    "mode": mode,
                    "comparison_label": f"{pol_a}_minus_{pol_b}",
                    "n_pairs": int(len(both)),
                    "rate_a": float(a.mean()), "rate_b": float(b.mean()),
                    "discordant_a_only": n_a_only, "discordant_b_only": n_b_only,
                    "p_value": _mcnemar_exact_p(n_a_only, n_b_only),
                    "test": "mcnemar_exact_binomial",
                })

    # Descriptive: rationale length (reported, never Holm-corrected).
    for mode in sorted(audit["mode"].astype(str).unique()):
        sub = audit[audit["mode"].astype(str) == mode]
        piv = sub.pivot_table(index="row_id", columns="policy",
                              values="rationale_length", aggfunc="first")
        for pol_a, pol_b in _H2_PAIRS:
            if pol_a not in piv.columns or pol_b not in piv.columns:
                continue
            d = (piv[pol_a] - piv[pol_b]).dropna()
            if len(d) == 0:
                continue
            rows.append({
                "hypothesis": "H2", "measure": "rationale_length_descriptive",
                "mode": mode, "comparison_label": f"{pol_a}_minus_{pol_b}",
                "n_pairs": int(len(d)), "mean_diff": float(d.mean()),
                "p_value": None, "test": "descriptive_only",
            })

    pvals = [r["p_value"] for r in rows if r.get("p_value") is not None]
    adj = _holm_adjust(pvals)
    it = iter(adj)
    for r in rows:
        if r.get("p_value") is not None:
            r["holm_adjusted_p"] = next(it)
    return {"hypothesis": "H2", "status": "OK" if rows else "SKIP", "rows": rows}


# ---------------------------------------------------------------------------
# H3: selective adoption (C6 vs C6X)
# ---------------------------------------------------------------------------

def run_h3(
    revision_metrics: pd.DataFrame | None,
    identity_contrast: pd.DataFrame | None,
) -> dict:
    """H3: adoption differs between C6 (RL reference) and C6X (random reference).

    PRIMARY (paired; EVAL-P3-001 update 2026-07-04): C6 and C6X are elicited
    on the SAME firm-years — the proposal's design is a within-firm source
    contrast, so the primary tests pair by (row_id, mode):
      * continuous — Wilcoxon signed-rank on per-row
        rl_adoption_ratio(C6) − rl_adoption_ratio(C6X), per mode, Holm across
        modes;
      * categorical — exact McNemar (binomial on discordant pairs) on
        per-row "revised_candidate_id == shown reference", per mode.
    SECONDARY (unpaired robustness): the original Mann-Whitney U on pooled
    adoption ratios is retained, demoted, for comparability with earlier
    drafts.  Group-level adoption rates stay as descriptives.
    """
    result: dict[str, Any] = {
        "hypothesis": "H3",
        "paired_primary": {"continuous": [], "categorical": []},
        "unpaired_secondary": None,
        "descriptives": None,
    }
    if revision_metrics is not None and "revision_condition" in revision_metrics.columns:
        rm = revision_metrics.copy()
        rm["revision_condition"] = rm["revision_condition"].astype(str)
        c6  = rm[rm["revision_condition"] == "C6"]
        c6x = rm[rm["revision_condition"] == "C6X"]
        modes = sorted(rm["mode"].astype(str).unique()) if "mode" in rm.columns else [None]

        # --- paired primary: continuous adoption ratio ---
        cont_rows: list[dict] = []
        if {"row_id", "rl_adoption_ratio"}.issubset(rm.columns):
            for mode in modes:
                sub = rm if mode is None else rm[rm["mode"].astype(str) == mode]
                piv = sub.pivot_table(index="row_id", columns="revision_condition",
                                      values="rl_adoption_ratio", aggfunc="first")
                if "C6" not in piv.columns or "C6X" not in piv.columns:
                    continue
                diff = (pd.to_numeric(piv["C6"], errors="coerce")
                        - pd.to_numeric(piv["C6X"], errors="coerce")).dropna()
                if len(diff) == 0:
                    continue
                W, p = _wilcoxon(diff)
                cont_rows.append({
                    "measure": "rl_adoption_ratio", "mode": mode,
                    "comparison_label": "C6_minus_C6X_paired",
                    "n_pairs": int(len(diff)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "wilcoxon_W": W, "p_value": p,
                    "test": "wilcoxon_signed_rank_paired",
                })
            adj = _holm_adjust([r["p_value"] for r in cont_rows])
            for r, a in zip(cont_rows, adj):
                r["holm_adjusted_p"] = a
        result["paired_primary"]["continuous"] = cont_rows

        # --- paired primary: categorical adoption (McNemar) ---
        cat_rows: list[dict] = []
        if {"row_id", "revised_candidate_id", "rl_reference_candidate"}.issubset(rm.columns):
            rm2 = rm[rm["rl_reference_candidate"].notna()].copy()
            rm2["_adopted"] = (
                rm2["revised_candidate_id"].astype(str)
                == rm2["rl_reference_candidate"].astype(str)
            )
            for mode in modes:
                sub = rm2 if mode is None else rm2[rm2["mode"].astype(str) == mode]
                piv = sub.pivot_table(index="row_id", columns="revision_condition",
                                      values="_adopted", aggfunc="first")
                if "C6" not in piv.columns or "C6X" not in piv.columns:
                    continue
                both = piv[["C6", "C6X"]].dropna()
                if both.empty:
                    continue
                a = both["C6"].astype(bool)
                b = both["C6X"].astype(bool)
                n_a_only = int((a & ~b).sum())
                n_b_only = int((~a & b).sum())
                cat_rows.append({
                    "measure": "categorical_adoption", "mode": mode,
                    "comparison_label": "C6_vs_C6X_paired",
                    "n_pairs": int(len(both)),
                    "rate_c6": float(a.mean()), "rate_c6x": float(b.mean()),
                    "discordant_c6_only": n_a_only, "discordant_c6x_only": n_b_only,
                    "p_value": _mcnemar_exact_p(n_a_only, n_b_only),
                    "test": "mcnemar_exact_binomial",
                })
        result["paired_primary"]["categorical"] = cat_rows

        # --- secondary: original unpaired Mann-Whitney (retained, demoted) ---
        if "rl_adoption_ratio" in rm.columns and len(c6) > 0 and len(c6x) > 0:
            from scipy.stats import mannwhitneyu
            try:
                stat, p = mannwhitneyu(
                    c6["rl_adoption_ratio"].dropna().to_numpy(),
                    c6x["rl_adoption_ratio"].dropna().to_numpy(),
                    alternative="two-sided",
                )
                result["unpaired_secondary"] = {
                    "test": "Mann-Whitney U (two-sided; unpaired robustness only)",
                    "n_c6": int(len(c6["rl_adoption_ratio"].dropna())),
                    "n_c6x": int(len(c6x["rl_adoption_ratio"].dropna())),
                    "mean_c6": float(c6["rl_adoption_ratio"].mean()),
                    "mean_c6x": float(c6x["rl_adoption_ratio"].mean()),
                    "U_stat": float(stat),
                    "p_value": float(p),
                }
            except Exception as exc:
                result["unpaired_secondary"] = {"status": "FAIL", "error": str(exc)}

        # --- descriptives: pooled group rates ---
        if {"revised_candidate_id", "rl_reference_candidate"}.issubset(rm.columns):
            def _adoption_rate(sub: pd.DataFrame) -> float | None:
                s = sub[sub["rl_reference_candidate"].notna()]
                if len(s) == 0:
                    return None
                return float((s["revised_candidate_id"].astype(str)
                              == s["rl_reference_candidate"].astype(str)).mean())
            rate_c6 = _adoption_rate(c6)
            rate_c6x = _adoption_rate(c6x)
            result["descriptives"] = {
                "categorical_adoption_rate_c6": rate_c6,
                "categorical_adoption_rate_c6x": rate_c6x,
                "gap_c6_minus_c6x": (float(rate_c6 - rate_c6x)
                                     if rate_c6 is not None and rate_c6x is not None else None),
                "note": "pooled fraction where revised_candidate_id matches the shown reference",
            }
    if identity_contrast is not None and not identity_contrast.empty:
        result["identity_contrast_summary"] = {
            k: float(identity_contrast[k].iloc[0]) if k in identity_contrast.columns else None
            for k in ["identity_adoption_gap", "identity_retention_gap", "identity_drift_gap"]
        }
    return result


# ---------------------------------------------------------------------------
# H4-a: substrate robustness
# ---------------------------------------------------------------------------

def run_h4a(comparison: pd.DataFrame, stage6_eval: pd.DataFrame | None) -> dict:
    """H4-a: primary verdict ordering is preserved across α/β/γ.

    Checks: C3 > C0, C3 > C2 direction, each LLM condition vs C0.
    Reports per-backend Holm-adjusted p and qualitative verdict.
    """
    rows: list[dict] = []
    key_pairs = [
        ("C3_candidate_iql", "C0_noop"),
        ("C3_candidate_iql", "C2_weakest_component_rule"),
    ]
    llm_conditions = ["C4", "C5", "C6", "C6X", "C7", "C8"]
    pols_present = set(comparison["policy"].astype(str)) if "policy" in comparison.columns else set()
    for cond in llm_conditions:
        if cond in pols_present:
            key_pairs.append((cond, "C0_noop"))

    all_data = comparison.copy()
    if stage6_eval is not None:
        # Ensure stage6 RL rows are in the comparison frame
        extra = stage6_eval[stage6_eval["policy"].astype(str).str.startswith("C3")]
        if not extra.empty and "mode" not in extra.columns:
            extra = extra.copy()
            extra["mode"] = "rl_native"
        all_data = pd.concat([all_data, extra], ignore_index=True, sort=False)

    for backend in ["alpha", "beta", "gamma"]:
        col = f"delta_R_score_{backend}"
        if col not in all_data.columns:
            continue
        backend_rows = []
        for pol_a, pol_b in key_pairs:
            r = _paired_test(all_data, pol_a, pol_b, backend)
            r["hypothesis"] = "H4-a"
            r["comparison_label"] = f"{pol_a}_minus_{pol_b}"
            backend_rows.append(r)
        pvals = [r["wilcoxon_p"] for r in backend_rows if r.get("status") == "OK"]
        adj = _holm_adjust(pvals)
        adj_iter = iter(adj)
        for r in backend_rows:
            if r.get("status") == "OK":
                r["holm_adjusted_p"] = next(adj_iter)
        rows.extend(backend_rows)

    # Robustness summary: for each RL pair, count backends with same sign
    summary: dict[str, Any] = {}
    for pol_a, pol_b in [("C3_candidate_iql", "C0_noop"), ("C3_candidate_iql", "C2_weakest_component_rule")]:
        label = f"{pol_a}_minus_{pol_b}"
        signs = {}
        for r in rows:
            if r.get("comparison_label") == label and r.get("status") == "OK":
                signs[r["backend"]] = "positive" if r["mean_diff"] > 0 else "non_positive"
        summary[label] = signs
    return {"hypothesis": "H4-a", "rows": rows, "robustness_summary": summary}


# ---------------------------------------------------------------------------
# H4-b: action-space trade-off (free_form vs candidate_selection)
# ---------------------------------------------------------------------------


def run_h4b(comparison: pd.DataFrame | None, project_root: Path) -> dict:
    """H4-b: expressiveness/score vs auditability trade-off between modes.

    Score side: paired Wilcoxon + Holm on delta_R(free_form_10d) minus
    delta_R(candidate_selection) per LLM condition × backend, same firm rows.
    Auditability side (descriptive; candidate mode is 0-distance by
    construction): free-form projection distance, out-of-library rate, and
    bound-clip dimension counts per condition from the Stage 7 failure audit.
    """
    result: dict = {"hypothesis": "H4-b", "rows": [], "auditability_descriptives": []}
    if comparison is None or "mode" not in comparison.columns:
        result["status"] = "SKIP"
        result["reason"] = "stage9 comparison with mode axis unavailable"
    else:
        rows: list[dict] = []
        for backend in ["alpha", "beta", "gamma"]:
            col = f"delta_R_score_{backend}"
            if col not in comparison.columns:
                continue
            for pol in ["C4", "C5", "C6", "C6X", "C7", "C8"]:
                sub = comparison[comparison["policy"].astype(str) == pol]
                if sub.empty:
                    continue
                piv = sub.pivot_table(index="row_id", columns="mode", values=col, aggfunc="first")
                if "free_form_10d" not in piv.columns or "candidate_selection" not in piv.columns:
                    continue
                diff = (piv["free_form_10d"] - piv["candidate_selection"]).dropna()
                if len(diff) == 0:
                    continue
                clusters = pd.Series(diff.index.astype(str), index=diff.index)
                se, lo, hi = _cluster_bootstrap(
                    diff, clusters, seed=_bootstrap_seed_h(f"h4b|{backend}|{pol}")
                )
                W, p = _wilcoxon(diff)
                rows.append({
                    "hypothesis": "H4-b", "backend": backend, "policy": pol,
                    "comparison_label": "free_form_minus_candidate",
                    "n_pairs": int(len(diff)),
                    "mean_diff": float(diff.mean()),
                    "median_diff": float(diff.median()),
                    "cluster_se": se, "ci95_lo": lo, "ci95_hi": hi,
                    "wilcoxon_W": W, "wilcoxon_p": p,
                })
        pvals = [r["wilcoxon_p"] for r in rows]
        adj = _holm_adjust(pvals)
        for r, a in zip(rows, adj):
            r["holm_adjusted_p"] = a
        result["rows"] = rows
        result["status"] = "OK" if rows else "SKIP"

    audit_path = stage_dir(project_root, "stage8") / "llm_stage8_failure_audit_enriched.csv"
    if not audit_path.exists():
        audit_path = stage_dir(project_root, "stage7") / "llm_stage7_failure_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        ff = audit[audit["mode"].astype(str) == "free_form_10d"]
        if not ff.empty and "projection_distance" in ff.columns:
            g = (ff.groupby("policy")
                   .agg(n=("row_id", "count"),
                        mean_projection_distance=("projection_distance", "mean"),
                        median_projection_distance=("projection_distance", "median"),
                        out_of_library_rate=("out_of_library", "mean"),
                        mean_bound_clip_dims=("bound_clip_dim_count", "mean"))
                   .reset_index())
            result["auditability_descriptives"] = g.to_dict(orient="records")
    return result


# ---------------------------------------------------------------------------
# H5: information scope (requires cross-run data)
# ---------------------------------------------------------------------------

def _read_snapshot_metadata(run_dir: Path) -> dict[str, Any]:
    """Read per-run metadata from an archived run snapshot."""
    candidates = [
        run_dir / "stage7_llm_action_generation" / "metadata.json",
        run_dir / "stage8_llm_multi_oracle_eval" / "metadata.json",
        run_dir / "stage9_llm_rl_comparison" / "metadata.json",
    ]
    out: dict[str, Any] = {"run_label": run_dir.name, "run_dir": str(run_dir)}
    for p in candidates:
        if p.exists():
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            stage = p.parent.name.split("_")[0]
            out[f"{p.parent.name}_metadata_path"] = str(p)
            for k in ["information_condition", "backend_id", "backend_is_live", "final_paper_run_allowed", "response_parser_grounding_matcher_version"]:
                if k in obj and k not in out:
                    out[k] = obj.get(k)
    if "information_condition" not in out:
        for ic in ["IC-a", "IC-b", "IC-c", "ICa", "ICb", "ICc"]:
            if ic in run_dir.name:
                out["information_condition"] = {"ICa":"IC-a","ICb":"IC-b","ICc":"IC-c"}.get(ic, ic)
                break
    return out


def _load_snapshot_failure_audit(run_dir: Path) -> pd.DataFrame | None:
    for p in [
        run_dir / "stage8_llm_multi_oracle_eval" / "llm_stage8_failure_audit_enriched.csv",
        run_dir / "stage7_llm_action_generation" / "llm_stage7_failure_audit.csv",
    ]:
        if p.exists():
            df = pd.read_csv(p)
            return df
    return None


def run_h5(llm_runs_dir: Path | None) -> dict:
    """H5: compare IC-a / IC-b / IC-c across conditions.

    Uses metadata-driven grouping from archived ``run_llm_stages --run-label``
    snapshots.  Produces score tests plus action, failure and rationale
    diagnostics required by the proposal's information-scope hypothesis.
    """
    if llm_runs_dir is None or not llm_runs_dir.exists():
        return {
            "hypothesis": "H5",
            "status": "SKIP",
            "reason": "llm_runs_dir not provided or does not exist; run with --llm-runs-dir after all IC runs complete",
        }
    run_records: list[dict[str, Any]] = []
    score_frames: list[pd.DataFrame] = []
    failure_frames: list[pd.DataFrame] = []
    for subdir in sorted(llm_runs_dir.iterdir()):
        if not subdir.is_dir():
            continue
        meta = _read_snapshot_metadata(subdir)
        ic = meta.get("information_condition")
        p = subdir / "stage9_llm_rl_comparison" / "llm_stage9_llm_rl_comparison.parquet"
        if not p.exists():
            p = subdir / "llm_stage9_llm_rl_comparison.parquet"
        if not p.exists() or not ic:
            continue
        df = pd.read_parquet(p).copy()
        df["run_label"] = subdir.name
        df["information_condition"] = str(ic)
        score_frames.append(df)
        fa = _load_snapshot_failure_audit(subdir)
        if fa is not None:
            fa = fa.copy()
            fa["run_label"] = subdir.name
            fa["information_condition"] = str(ic)
            failure_frames.append(fa)
        run_records.append(meta)
    if not score_frames:
        return {"hypothesis": "H5", "status": "SKIP", "reason": "no valid run snapshots found in llm_runs_dir"}
    combined = pd.concat(score_frames, ignore_index=True, sort=False)
    ics = sorted(set(combined["information_condition"].astype(str)))

    score_rows: list[dict] = []
    action_rows: list[dict] = []
    switching_rows: list[dict] = []
    rationale_rows: list[dict] = []
    failure_rows: list[dict] = []
    anchoring_rows: list[dict] = []

    for backend in ["alpha", "beta", "gamma"]:
        col = f"delta_R_score_{backend}"
        if col not in combined.columns:
            continue
        for pol in ["C4", "C5", "C6", "C6X", "C7", "C8"]:
            if pol not in set(combined["policy"].astype(str)):
                continue
            for mode in sorted(combined["mode"].astype(str).unique()) if "mode" in combined.columns else [None]:
                sub = combined[combined["policy"].astype(str) == pol]
                if mode is not None and "mode" in sub.columns:
                    sub = sub[sub["mode"].astype(str) == mode]
                for i in range(len(ics)):
                    for j in range(i + 1, len(ics)):
                        ic_a, ic_b = ics[i], ics[j]
                        da = sub[sub["information_condition"] == ic_a].set_index("row_id")[col]
                        db = sub[sub["information_condition"] == ic_b].set_index("row_id")[col]
                        common = da.index.intersection(db.index)
                        if len(common) == 0:
                            continue
                        diff = (da.loc[common] - db.loc[common]).dropna()
                        if len(diff) == 0:
                            continue
                        W, pval = _wilcoxon(diff)
                        score_rows.append({
                            "hypothesis": "H5", "measure": "score", "backend": backend,
                            "policy": pol, "mode": mode, "ic_a": ic_a, "ic_b": ic_b,
                            "n_pairs": int(len(diff)), "mean_diff": float(diff.mean()),
                            "median_diff": float(diff.median()), "wilcoxon_W": W, "wilcoxon_p": pval,
                        })
    pvals = [r["wilcoxon_p"] for r in score_rows]
    adj = _holm_adjust(pvals)
    for r, a in zip(score_rows, adj):
        r["holm_adjusted_p"] = a

    # Action distribution by IC/condition/mode.
    if "candidate_id" in combined.columns:
        g = (combined.groupby(["information_condition", "policy", "mode", "candidate_id"], dropna=False)
             .agg(n=("row_id", "count")).reset_index())
        totals = g.groupby(["information_condition", "policy", "mode"], dropna=False)["n"].transform("sum")
        g["share"] = g["n"] / totals.replace(0, np.nan)
        action_rows = g.to_dict(orient="records")
        # Paired action switching for IC pairs.
        for pol in ["C4", "C5", "C6", "C6X", "C7", "C8"]:
            for mode in sorted(combined["mode"].astype(str).unique()) if "mode" in combined.columns else [None]:
                sub = combined[combined["policy"].astype(str) == pol]
                if mode is not None and "mode" in sub.columns:
                    sub = sub[sub["mode"].astype(str) == mode]
                for i in range(len(ics)):
                    for j in range(i + 1, len(ics)):
                        ic_a, ic_b = ics[i], ics[j]
                        a = sub[sub["information_condition"] == ic_a].set_index("row_id")["candidate_id"]
                        b = sub[sub["information_condition"] == ic_b].set_index("row_id")["candidate_id"]
                        common = a.index.intersection(b.index)
                        if len(common) == 0:
                            continue
                        switch_rate = float((a.loc[common].astype(str) != b.loc[common].astype(str)).mean())
                        switching_rows.append({"policy": pol, "mode": mode, "ic_a": ic_a, "ic_b": ic_b, "n_pairs": int(len(common)), "action_switch_rate": switch_rate})

    if failure_frames:
        fa = pd.concat(failure_frames, ignore_index=True, sort=False)
        if "failure_categories" in fa.columns:
            cats = [
                "translational_failure", "structural_out_of_scope", "direction_error", "magnitude_error",
                "feasibility_violation", "liquidity_destructive_recourse",
                "anchoring_or_confirmation_failure", "ungrounded_judgment",
            ]
            cat_sets = fa["failure_categories"].fillna("").astype(str).apply(lambda s: {t for t in s.split(",") if t})
            for c in cats:
                fa[f"has_{c}"] = cat_sets.apply(lambda st, _c=c: _c in st)
            group_cols = [c for c in ["information_condition", "policy", "mode"] if c in fa.columns]
            for keys, g in fa.groupby(group_cols, dropna=False):
                if not isinstance(keys, tuple): keys = (keys,)
                rec = dict(zip(group_cols, keys))
                rec["n_rows"] = int(len(g))
                rec["any_failure_rate"] = float(cat_sets.loc[g.index].apply(bool).mean())
                for c in cats:
                    rec[f"rate_{c}"] = float(g[f"has_{c}"].mean())
                failure_rows.append(rec)
            if "rationale_length" in fa.columns:
                rat = fa.groupby(group_cols, dropna=False)["rationale_length"].agg(["count", "mean", "median"]).reset_index()
                rationale_rows = rat.rename(columns={"count":"n_rows", "mean":"mean_rationale_length", "median":"median_rationale_length"}).to_dict(orient="records")
            icc = fa[fa["information_condition"].astype(str) == "IC-c"] if "information_condition" in fa.columns else pd.DataFrame()
            if not icc.empty:
                anchoring_rows.append({
                    "information_condition": "IC-c",
                    "n_rows": int(len(icc)),
                    "anchoring_or_confirmation_rate": float(icc.get("has_anchoring_or_confirmation_failure", pd.Series(False, index=icc.index)).mean()) if "has_anchoring_or_confirmation_failure" in icc.columns else None,
                    "ungrounded_rate": float(icc.get("has_ungrounded_judgment", pd.Series(False, index=icc.index)).mean()) if "has_ungrounded_judgment" in icc.columns else None,
                })

    common_manifest = {
        "hypothesis": "H5",
        "llm_runs_dir": str(llm_runs_dir),
        "runs": run_records,
        "information_conditions_found": ics,
        "common_row_counts": {},
    }
    for pol in sorted(set(combined["policy"].astype(str))):
        sub = combined[combined["policy"].astype(str) == pol]
        ids_by_ic = {ic: set(sub[sub["information_condition"] == ic]["row_id"].astype(str)) for ic in ics}
        if ids_by_ic:
            common = set.intersection(*ids_by_ic.values()) if all(ids_by_ic.values()) else set()
            common_manifest["common_row_counts"][pol] = int(len(common))

    return {
        "hypothesis": "H5", "status": "OK", "rows": score_rows,
        "score_rows": score_rows,
        "action_distribution_rows": action_rows,
        "action_switching_rows": switching_rows,
        "failure_rows": failure_rows,
        "rationale_rows": rationale_rows,
        "anchoring_rows": anchoring_rows,
        "common_manifest": common_manifest,
        "ic_labels_found": ics,
    }


# ---------------------------------------------------------------------------
# MDE pre-registration
# ---------------------------------------------------------------------------

def compute_mde_table(
    comparison: pd.DataFrame | None,
    stage6_eval: pd.DataFrame | None,
    n: int = 575,
    power: float = 0.80,
    alpha: float = 0.05,
) -> dict:
    """Pre-register minimum detectable effects (MDE) before LLM API calls.

    Uses cluster-bootstrap SE estimates from Stage 6 data as the noise reference.
    MDE = (z_{alpha/2} + z_{power}) * SE, where SE is estimated from the
    C3-vs-C0 paired distribution (same n, same Oracle substrate, same firms).

    Returned dict is hashed and the hash is stored in Stage 9 metadata
    so that post-hoc revision of the power analysis is detectable.
    """
    z_alpha = 1.959964  # two-sided alpha=0.05
    z_power = {0.80: 0.8416, 0.90: 1.2816}.get(power, 0.8416)

    rows: list[dict] = []
    se_ref: dict[str, float] = {}

    # Estimate SE from Stage 6 C3 vs C0 differences (our best noise reference)
    if stage6_eval is not None:
        for backend in ["alpha", "beta", "gamma"]:
            col = f"R_score_{backend}"
            if col not in stage6_eval.columns:
                # try delta column
                col = f"delta_R_score_{backend}"
            if col not in stage6_eval.columns:
                continue
            piv = stage6_eval.pivot_table(index="row_id", columns="policy", values=col, aggfunc="first")
            if "C3_candidate_iql" not in piv.columns or "C0_noop" not in piv.columns:
                # Try any C3* prefix
                c3_cols = [c for c in piv.columns if str(c).startswith("C3")]
                if not c3_cols:
                    continue
                piv["C3_candidate_iql"] = piv[c3_cols[0]]
            if "C0_noop" not in piv.columns:
                continue
            diff = (piv["C3_candidate_iql"] - piv["C0_noop"]).dropna()
            if len(diff) == 0:
                continue
            clusters = pd.Series(diff.index.astype(str), index=diff.index)
            seed = _bootstrap_seed_h(f"mde_se_ref|{backend}")
            se, _, _ = _cluster_bootstrap(diff, clusters, seed=seed)
            if se is not None and np.isfinite(se) and se > 0:
                se_ref[backend] = float(se)

    if not se_ref and comparison is not None:
        # Fallback: estimate from C4-C0 in comparison frame if available
        for backend in ["alpha", "beta", "gamma"]:
            col = f"delta_R_score_{backend}"
            if col not in comparison.columns:
                continue
            piv = comparison.pivot_table(index="row_id", columns="policy", values=col, aggfunc="first")
            if "C4" not in piv.columns or "C0_noop" not in piv.columns:
                continue
            diff = (piv["C4"] - piv["C0_noop"]).dropna()
            if len(diff) == 0:
                continue
            clusters = pd.Series(diff.index.astype(str), index=diff.index)
            seed = _bootstrap_seed_h(f"mde_se_fallback|{backend}")
            se, _, _ = _cluster_bootstrap(diff, clusters, seed=seed)
            if se is not None and np.isfinite(se) and se > 0:
                se_ref[backend] = float(se)

    hypotheses = [
        ("H1_C5_minus_C4",       "H1", "C5-C4",       "candidate_selection"),
        ("H1_C5_minus_C4_ff",    "H1", "C5-C4",       "free_form_10d"),
        ("H1_C7_minus_C6",       "H1", "C7-C6",       "candidate_selection"),
        ("H1_C5_minus_C1",       "H1", "C5-C1",       "candidate_selection"),
        ("H1_C7_minus_C1",       "H1", "C7-C1",       "candidate_selection"),
        ("H4b_C4_ff_minus_cs",   "H4b","C4 ff-cs",    "mode_contrast"),
        ("H4b_C7_ff_minus_cs",   "H4b","C7 ff-cs",    "mode_contrast"),
        ("H3_C6_vs_C6X",         "H3", "C6-C6X adopt","all"),
        ("H4a_C3_minus_C0",      "H4a","C3-C0",       "rl_native"),
        ("H4a_C3_minus_C2",      "H4a","C3-C2",       "rl_native"),
    ]
    for label, hyp, comparison_desc, mode in hypotheses:
        for backend, se in se_ref.items():
            mde = (z_alpha + z_power) * se
            rows.append({
                "label": label,
                "hypothesis": hyp,
                "comparison": comparison_desc,
                "mode": mode,
                "backend": backend,
                "n": n,
                "power": power,
                "alpha": alpha,
                "se_estimate": se,
                "mde": mde,
                "se_source": "stage6_C3_minus_C0" if backend in se_ref else "fallback_C4_minus_C0",
            })
    if not rows:
        for label, hyp, comparison_desc, mode in hypotheses:
            rows.append({
                "label": label, "hypothesis": hyp, "comparison": comparison_desc,
                "mode": mode, "backend": "alpha", "n": n, "power": power, "alpha": alpha,
                "se_estimate": None, "mde": None,
                "se_source": "unavailable_no_stage6_data",
            })

    return {
        "schema_version": _SCHEMA_VERSION,
        "created_utc": _now(),
        "n_firms": n,
        "power": power,
        "alpha_two_sided": alpha,
        "z_alpha_over_2": z_alpha,
        "z_power": z_power,
        "se_reference_source": "stage6_C3_minus_C0_cluster_bootstrap",
        "bootstrap_seed_scheme": _BOOTSTRAP_SEED_SCHEME,
        "rows": rows,
        "note": (
            "MDE pre-registered before remaining LLM API calls. "
            "Hash of this document is recorded in Stage 9 metadata. "
            "Post-hoc revision detectable via hash mismatch."
        ),
    }


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def run(
    project_root: Path,
    *,
    mde_freeze: bool = False,
    llm_runs_dir: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    project_root = Path(project_root).resolve()
    out = Path(out_dir).resolve() if out_dir is not None else stage_dir(project_root, "stage9")
    out.mkdir(parents=True, exist_ok=True)

    comparison     = _load_comparison(project_root)
    revision       = _load_revision_metrics(project_root)
    identity       = _load_identity_contrast(project_root)
    stage6_eval    = _load_stage6_eval(project_root)

    results: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "created_utc": _now(),
        "project_root": str(project_root),
        "output_dir": str(out),
    }

    # --- MDE pre-registration (can run before Stage 8 if stage6_eval exists) ---
    mde_path = out / "llm_stage9_mde_table.json"
    if mde_freeze or not mde_path.exists():
        mde = compute_mde_table(comparison, stage6_eval)
        mde_written, mde_hash = _write_mde_if_changed(mde_path, mde)
        results["mde_analysis_artifact_path"] = str(mde_path)
        results["mde_analysis_artifact_hash"] = mde_hash
        results["mde_analysis_artifact_rewritten"] = bool(mde_written)
        if mde_freeze:
            action = "frozen" if mde_written else "already_frozen_content_unchanged"
            print(f"MDE table {action}: {mde_path}  hash={mde_hash}")
    else:
        results["mde_analysis_artifact_path"] = str(mde_path)
        results["mde_analysis_artifact_hash"] = _sha256_bytes(mde_path.read_bytes())
        results["mde_analysis_artifact_rewritten"] = False

    # --- Hypothesis tests (require Stage 9 outputs) ---
    if comparison is not None:
        h1 = run_h1(comparison)
        h4a = run_h4a(comparison, stage6_eval)
        pd.DataFrame(h1["rows"]).to_csv(out / "llm_stage9_h1_tests.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(h4a["rows"]).to_csv(out / "llm_stage9_h4a_robustness.csv", index=False, encoding="utf-8-sig")
        results["h1"] = {"n_tests": len(h1["rows"])}
        results["h4a"] = {"robustness_summary": h4a["robustness_summary"]}
    else:
        results["h1"] = {"status": "SKIP", "reason": "Stage 9 comparison not available"}
        results["h4a"] = {"status": "SKIP", "reason": "Stage 9 comparison not available"}

    h2 = run_h2(project_root)
    if h2.get("rows"):
        pd.DataFrame(h2["rows"]).to_csv(out / "llm_stage9_h2_form_effect.csv", index=False, encoding="utf-8-sig")
    results["h2"] = {"status": h2.get("status"), "n_tests": len(h2.get("rows") or []), "reason": h2.get("reason")}

    h4b = run_h4b(comparison, project_root)
    if h4b.get("rows"):
        pd.DataFrame(h4b["rows"]).to_csv(out / "llm_stage9_h4b_action_space.csv", index=False, encoding="utf-8-sig")
    if h4b.get("auditability_descriptives"):
        pd.DataFrame(h4b["auditability_descriptives"]).to_csv(out / "llm_stage9_h4b_auditability.csv", index=False, encoding="utf-8-sig")
    results["h4b"] = {"status": h4b.get("status"), "n_tests": len(h4b.get("rows") or [])}

    h3 = run_h3(revision, identity)
    h3_rows = (h3.get("paired_primary", {}).get("continuous") or []) + \
              (h3.get("paired_primary", {}).get("categorical") or [])
    if h3_rows:
        pd.DataFrame(h3_rows).to_csv(out / "llm_stage9_h3_paired_adoption.csv", index=False, encoding="utf-8-sig")
    results["h3"] = {k: v for k, v in h3.items() if k != "hypothesis"}

    h5 = run_h5(llm_runs_dir)
    if h5.get("rows"):
        pd.DataFrame(h5["rows"]).to_csv(out / "llm_stage9_h5_ic_comparison.csv", index=False, encoding="utf-8-sig")
    if h5.get("action_distribution_rows"):
        pd.DataFrame(h5["action_distribution_rows"]).to_csv(out / "h5_action_distribution_by_ic.csv", index=False, encoding="utf-8-sig")
    if h5.get("action_switching_rows"):
        pd.DataFrame(h5["action_switching_rows"]).to_csv(out / "h5_pairwise_action_switching.csv", index=False, encoding="utf-8-sig")
    if h5.get("failure_rows"):
        pd.DataFrame(h5["failure_rows"]).to_csv(out / "h5_failure_rate_by_ic.csv", index=False, encoding="utf-8-sig")
    if h5.get("rationale_rows"):
        pd.DataFrame(h5["rationale_rows"]).to_csv(out / "h5_rationale_length_by_ic.csv", index=False, encoding="utf-8-sig")
    if h5.get("anchoring_rows"):
        pd.DataFrame(h5["anchoring_rows"]).to_csv(out / "h5_ic_c_anchoring_diagnostic.csv", index=False, encoding="utf-8-sig")
    if h5.get("common_manifest"):
        write_json(out / "h5_common_row_manifest.json", h5["common_manifest"])
    results["h5"] = {"status": h5.get("status", "OK"), "reason": h5.get("reason"), "ic_labels_found": h5.get("ic_labels_found")}

    results["status"] = "PASS"
    write_json(out / "llm_stage9_inference_metadata.json", results)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Stage 9 statistical inference and MDE pre-registration")
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--mde-freeze", action="store_true",
                    help="Write/overwrite the MDE table and print its hash. Use before remaining LLM API calls.")
    ap.add_argument("--llm-runs-dir", default=None,
                    help="Directory of per-IC run snapshots for H5 comparison (optional).")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory for inference tables. Default: canonical Stage9 comparison dir.")
    args = ap.parse_args(argv)
    res = run(
        Path(args.project_root),
        mde_freeze=args.mde_freeze,
        llm_runs_dir=Path(args.llm_runs_dir) if args.llm_runs_dir else None,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows",)},
                     ensure_ascii=False, indent=2, default=str))
    return 0 if res.get("status") == "PASS" else 1


if __name__ == "__main__":
    import json
    raise SystemExit(main())
