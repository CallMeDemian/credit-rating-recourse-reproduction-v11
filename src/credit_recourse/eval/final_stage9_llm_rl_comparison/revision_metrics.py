from __future__ import annotations

"""Revision-behavior metrics (RESEARCH §11).

For revision conditions C4R, C6, C6X, and C7, the LLM produces an initial action ``a_0``
and then, after seeing the RL reference, a revised action ``a_1``.  The
research design measures revision in bound-normalized 10D action space:

::

    u = a_RL - a_0
    v = a_1  - a_0

    RL Adoption Ratio    = (v · u) / ||u||^2
    Self-Retention Ratio = 1 - RL Adoption Ratio
    Orthogonal Drift     = || v - (RL Adoption Ratio) * u || / ||u||

Interpretation:

* Adoption near 0 → no movement toward RL reference.
* 0 < adoption < 1 → partial movement.
* Adoption ≈ 1 → full movement to the RL action.
* Adoption > 1 → over-correction beyond it.
* Adoption < 0 → movement opposite to it.
* High orthogonal drift → an unstable third-direction revision rather than
  movement along the RL direction.

C8 receives the RL reference upfront and has no ``a_0``; the within-case
metrics are therefore undefined and Stage 9 reports them as NA, never zero
(per the LLM contract §8 fail-fast rule).

In addition to the geometric metrics, two structural fields are reported:

* ``revision_changed_candidate_flag`` — whether the projected (or selected)
  v32 candidate id changed between ``a_0`` and ``a_1``.
* ``revision_changed_active_dimensions`` — count of action dimensions where
  the bound-normalized magnitude crossed the activity threshold
  (``|x| / bound_width > 1e-3``).
"""

from typing import Optional

import numpy as np
import pandas as pd

from credit_recourse.rl.common.actions import ActionSpace


REVISION_METRIC_COLS = [
    "row_id",
    "base_condition",
    "revision_condition",
    "mode",
    "rl_reference_candidate",
    "reference_source",
    "reference_draw_seed",
    "initial_candidate_id",
    "revised_candidate_id",
    "revision_changed_candidate_flag",
    "revision_changed_active_dimensions",
    "revision_l1_distance",
    "revision_l2_distance",
    "revision_cosine_distance",
    "rl_adoption_ratio",
    "self_retention_ratio",
    "orthogonal_drift",
    "u_norm_squared",
    "initial_delta_R_score_alpha",
    "revised_delta_R_score_alpha",
    "revision_delta_R_score_alpha",
    "initial_delta_R_score_beta",
    "revised_delta_R_score_beta",
    "revision_delta_R_score_beta",
    "initial_delta_R_score_gamma",
    "revised_delta_R_score_gamma",
    "revision_delta_R_score_gamma",
    "metrics_defined",
    "undefined_reason",
]


def _action_vec(row: pd.Series, action_cols: list[str]) -> np.ndarray:
    return np.array(
        [float(pd.to_numeric(pd.Series([row.get(c, 0.0)]), errors="coerce").fillna(0.0).iloc[0]) for c in action_cols],
        dtype=float,
    )


def _bound_normalize(vec: np.ndarray, widths: np.ndarray) -> np.ndarray:
    return vec / np.maximum(widths, 1e-12)


def compute_pairwise_revision(
    *,
    a_initial: pd.Series,
    a_revised: pd.Series,
    rl_reference_candidate: str | None,
    space: ActionSpace,
    initial_deltas: dict[str, float] | None = None,
    revised_deltas: dict[str, float] | None = None,
) -> dict:
    """Compute one revision-metric record for one (row_id, base, revision) pair.

    ``a_initial`` and ``a_revised`` are rows from the LLM action table for
    the base condition (C4/C5) and the revision condition (C4R/C6/C6X/C7)
    respectively.  ``rl_reference_candidate`` is the candidate the RL policy
    selected and that was shown to the LLM as the revision reference.

    Score deltas are optional; pass them to record per-backend before/after
    delta_R_score in the same row.
    """
    cols = list(space.columns)
    widths = np.array([space.bound_width(c) for c in cols], dtype=float)

    a0 = _action_vec(a_initial, cols)
    a1 = _action_vec(a_revised, cols)
    a0n = _bound_normalize(a0, widths)
    a1n = _bound_normalize(a1, widths)

    rec: dict = {
        "initial_candidate_id": str(a_initial.get("candidate_id", "")),
        "revised_candidate_id": str(a_revised.get("candidate_id", "")),
        "rl_reference_candidate": rl_reference_candidate,
        "reference_source": str(a_revised.get("reference_source", "")),
        "reference_draw_seed": a_revised.get("reference_draw_seed", None),
        "metrics_defined": False,
        "undefined_reason": None,
        "revision_changed_candidate_flag": bool(
            str(a_initial.get("candidate_id", "")) != str(a_revised.get("candidate_id", ""))
        ),
        "revision_l1_distance": float(np.sum(np.abs(a1n - a0n))),
        "revision_l2_distance": float(np.linalg.norm(a1n - a0n)),
    }
    # Cosine distance over bound-normalized vectors (treat zero vector as
    # max-distance/nan to surface ungrounded comparisons).
    norm0 = float(np.linalg.norm(a0n))
    norm1 = float(np.linalg.norm(a1n))
    if norm0 < 1e-12 or norm1 < 1e-12:
        rec["revision_cosine_distance"] = float("nan")
    else:
        cos = float(np.dot(a0n, a1n) / (norm0 * norm1))
        rec["revision_cosine_distance"] = 1.0 - cos

    # changed_active_dimensions: count of dims that crossed activity threshold
    eps = 1e-3
    active0 = np.abs(a0n) > eps
    active1 = np.abs(a1n) > eps
    rec["revision_changed_active_dimensions"] = int(np.sum(active0 != active1))

    # Geometric metrics need the RL reference vector
    if rl_reference_candidate is None:
        rec["undefined_reason"] = "no_rl_reference"
        rec["rl_adoption_ratio"] = float("nan")
        rec["self_retention_ratio"] = float("nan")
        rec["orthogonal_drift"] = float("nan")
        rec["u_norm_squared"] = float("nan")
    elif rl_reference_candidate not in space.fixed_candidates:
        rec["undefined_reason"] = "rl_reference_not_in_v32"
        rec["rl_adoption_ratio"] = float("nan")
        rec["self_retention_ratio"] = float("nan")
        rec["orthogonal_drift"] = float("nan")
        rec["u_norm_squared"] = float("nan")
    else:
        # Use the canonical ActionSpace helper so the dict-key convention in
        # space.fixed_candidates (action__-prefixed keys) is honored.
        rl_vec = np.asarray(space.candidate_vector(rl_reference_candidate), dtype=float)
        rl_n = _bound_normalize(rl_vec, widths)
        u = rl_n - a0n
        v = a1n - a0n
        u_sq = float(np.dot(u, u))
        rec["u_norm_squared"] = u_sq
        if u_sq < 1e-12:
            rec["undefined_reason"] = "rl_reference_equals_a0_in_normalized_space"
            rec["rl_adoption_ratio"] = float("nan")
            rec["self_retention_ratio"] = float("nan")
            rec["orthogonal_drift"] = float("nan")
        else:
            adoption = float(np.dot(v, u) / u_sq)
            ortho = v - adoption * u
            rec["rl_adoption_ratio"] = adoption
            rec["self_retention_ratio"] = 1.0 - adoption
            rec["orthogonal_drift"] = float(np.linalg.norm(ortho) / np.sqrt(u_sq))
            rec["metrics_defined"] = True

    initial_deltas = initial_deltas or {}
    revised_deltas = revised_deltas or {}
    for bk in ["alpha", "beta", "gamma"]:
        idel = initial_deltas.get(bk)
        rdel = revised_deltas.get(bk)
        rec[f"initial_delta_R_score_{bk}"] = (
            float(idel) if idel is not None and not pd.isna(idel) else float("nan")
        )
        rec[f"revised_delta_R_score_{bk}"] = (
            float(rdel) if rdel is not None and not pd.isna(rdel) else float("nan")
        )
        rec[f"revision_delta_R_score_{bk}"] = (
            rec[f"revised_delta_R_score_{bk}"] - rec[f"initial_delta_R_score_{bk}"]
            if not (pd.isna(rec[f"revised_delta_R_score_{bk}"]) or pd.isna(rec[f"initial_delta_R_score_{bk}"]))
            else float("nan")
        )

    return rec


def build_revision_table(
    *,
    action_table: pd.DataFrame,
    stage8_scores: pd.DataFrame,
    space: ActionSpace,
) -> pd.DataFrame:
    """Build the full revision-metrics frame for C4R, C6, C6X, and C7 (paired with
    C4 and C5 respectively, by row_id and mode).

    C8 is included with metrics_defined=False and undefined_reason
    ``c8_has_no_pre_reference_a0`` per the LLM contract §8.
    """
    cols = list(space.columns)

    score_lookup: dict[tuple[int, str, str], dict[str, float]] = {}
    for _, r in stage8_scores.iterrows():
        try:
            rid = int(r["row_id"])
            pol = str(r["policy"])
            mode = str(r.get("mode", ""))
        except (KeyError, ValueError):
            continue
        deltas = {
            bk: float(r[f"delta_R_score_{bk}"])
            for bk in ["alpha", "beta", "gamma"]
            if f"delta_R_score_{bk}" in r and pd.notna(r[f"delta_R_score_{bk}"])
        }
        score_lookup[(rid, pol, mode)] = deltas

    rows = []
    # Build a lookup keyed on (row_id, base condition, mode).
    by_key: dict[tuple[int, str, str], pd.Series] = {}
    for _, r in action_table.iterrows():
        try:
            rid = int(r["row_id"])
        except Exception:
            continue
        pol = str(r["policy"])
        mode = str(r.get("mode", ""))
        by_key[(rid, pol, mode)] = r

    pairs = [("C4", "C4R"), ("C4", "C6"), ("C4", "C6X"), ("C5", "C7")]
    for base, rev in pairs:
        for (rid, pol, mode), r_init in list(by_key.items()):
            if pol != base:
                continue
            r_rev = by_key.get((rid, rev, mode))
            if r_rev is None:
                continue
            rl_ref = r_rev.get("rl_reference_candidate")
            if isinstance(rl_ref, float) and pd.isna(rl_ref):
                rl_ref = None
            elif rl_ref is None or rl_ref == "":
                rl_ref = None
            rec = compute_pairwise_revision(
                a_initial=r_init,
                a_revised=r_rev,
                rl_reference_candidate=(str(rl_ref) if rl_ref is not None else None),
                space=space,
                initial_deltas=score_lookup.get((rid, base, mode)),
                revised_deltas=score_lookup.get((rid, rev, mode)),
            )
            rec["row_id"] = rid
            rec["base_condition"] = base
            rec["revision_condition"] = rev
            rec["mode"] = mode
            rows.append(rec)

    # C8 rows: report as NA (LLM contract §8 fail-fast).
    for (rid, pol, mode), r in list(by_key.items()):
        if pol != "C8":
            continue
        rl_ref = r.get("rl_reference_candidate")
        if isinstance(rl_ref, float) and pd.isna(rl_ref):
            rl_ref = None
        c8_deltas = score_lookup.get((rid, "C8", mode))
        rec = {
            "row_id": rid,
            "base_condition": "C8",
            "revision_condition": "C8",
            "mode": mode,
            "rl_reference_candidate": (str(rl_ref) if rl_ref is not None else None),
            "reference_source": str(r.get("reference_source", "")),
            "reference_draw_seed": r.get("reference_draw_seed", None),
            "initial_candidate_id": "",
            "revised_candidate_id": str(r.get("candidate_id", "")),
            "revision_changed_candidate_flag": False,
            "revision_changed_active_dimensions": 0,
            "revision_l1_distance": float("nan"),
            "revision_l2_distance": float("nan"),
            "revision_cosine_distance": float("nan"),
            "rl_adoption_ratio": float("nan"),
            "self_retention_ratio": float("nan"),
            "orthogonal_drift": float("nan"),
            "u_norm_squared": float("nan"),
            "metrics_defined": False,
            "undefined_reason": "c8_has_no_pre_reference_a0",
        }
        for bk in ["alpha", "beta", "gamma"]:
            rec[f"initial_delta_R_score_{bk}"] = float("nan")
            rec[f"revised_delta_R_score_{bk}"] = (
                float(c8_deltas[bk]) if c8_deltas and bk in c8_deltas else float("nan")
            )
            rec[f"revision_delta_R_score_{bk}"] = float("nan")
        rows.append(rec)

    if not rows:
        return pd.DataFrame(columns=REVISION_METRIC_COLS)
    df = pd.DataFrame(rows)
    return df[REVISION_METRIC_COLS]


def build_identity_contrast_table(revision_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the C6-vs-C6X source-identity contrast by row and mode.

    C6 and C6X share the same base condition (C4) and differ only by the shown
    reference source (RL vs seeded random).  The resulting gaps quantify whether
    adoption/retention/drift is specific to the RL reference or generic anchoring.
    """
    cols = [
        "row_id", "mode",
        "rl_reference_candidate_c6", "random_reference_candidate_c6x",
        "identity_adoption_gap", "identity_retention_gap", "identity_drift_gap",
        "c6_metrics_defined", "c6x_metrics_defined",
    ]
    if revision_df.empty:
        return pd.DataFrame(columns=cols)
    c6 = revision_df[revision_df["revision_condition"].astype(str) == "C6"].copy()
    c6x = revision_df[revision_df["revision_condition"].astype(str) == "C6X"].copy()
    if c6.empty or c6x.empty:
        return pd.DataFrame(columns=cols)
    keep = ["row_id", "mode", "rl_reference_candidate", "rl_adoption_ratio", "self_retention_ratio", "orthogonal_drift", "metrics_defined"]
    left = c6[keep].rename(columns={
        "rl_reference_candidate": "rl_reference_candidate_c6",
        "rl_adoption_ratio": "rl_adoption_ratio_c6",
        "self_retention_ratio": "self_retention_ratio_c6",
        "orthogonal_drift": "orthogonal_drift_c6",
        "metrics_defined": "c6_metrics_defined",
    })
    right = c6x[keep].rename(columns={
        "rl_reference_candidate": "random_reference_candidate_c6x",
        "rl_adoption_ratio": "rl_adoption_ratio_c6x",
        "self_retention_ratio": "self_retention_ratio_c6x",
        "orthogonal_drift": "orthogonal_drift_c6x",
        "metrics_defined": "c6x_metrics_defined",
    })
    out = left.merge(right, on=["row_id", "mode"], how="inner")
    if out.empty:
        return pd.DataFrame(columns=cols)
    out["identity_adoption_gap"] = pd.to_numeric(out["rl_adoption_ratio_c6"], errors="coerce") - pd.to_numeric(out["rl_adoption_ratio_c6x"], errors="coerce")
    out["identity_retention_gap"] = pd.to_numeric(out["self_retention_ratio_c6"], errors="coerce") - pd.to_numeric(out["self_retention_ratio_c6x"], errors="coerce")
    out["identity_drift_gap"] = pd.to_numeric(out["orthogonal_drift_c6"], errors="coerce") - pd.to_numeric(out["orthogonal_drift_c6x"], errors="coerce")
    return out[cols]
