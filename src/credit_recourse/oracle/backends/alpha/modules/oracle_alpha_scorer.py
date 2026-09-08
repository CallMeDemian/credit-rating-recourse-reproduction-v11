#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oracle α scorer — callable interface for Stage 6 rank-shock interventions.

This scorer accepts both the in-memory structure used inside the Alpha pipeline
and the exported ``oracle_alpha_params.json`` structure used by later stages.
It can optionally apply linear interpolation for continuous bin-score tables, but defaults to exact step-bin replication of the fitted Alpha output.
"""
from __future__ import annotations

from typing import Callable, Mapping, Any
import numpy as np
import pandas as pd

from credit_recourse.oracle.contracts.rating_scale import assign_grade_10, ensure_10_grade_contract, GRADE2NUM_10


def _as_float_dict(d: Mapping[Any, Any] | None) -> dict[int, float]:
    out: dict[int, float] = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def _derive_ids(params: Mapping[str, Any]) -> tuple[list[str], list[str], list[str]]:
    sel = list(params.get("selected_variables") or [])
    records = params.get("selected_variable_records") or params.get("variables") or []
    if not sel and records:
        sel = [str(r.get("variable_id")) for r in records if isinstance(r, Mapping) and r.get("variable_id")]
    fin = list(params.get("fin_ids") or [])
    nf = list(params.get("nonfin_ids") or [])
    if (not fin or not nf) and records:
        fin = [str(r.get("variable_id")) for r in records if isinstance(r, Mapping) and r.get("source") == "financial" and r.get("variable_id")]
        nf = [str(r.get("variable_id")) for r in records if isinstance(r, Mapping) and r.get("source") == "nonfinancial" and r.get("variable_id")]
    if not fin:
        fin = [v for v in sel if str(v).startswith("R")]
    if not nf:
        nf = [v for v in sel if v not in fin]
    return sel, fin, nf


def _normalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    sel, fin, nf = _derive_ids(params)
    dirs = params.get("directions") or params.get("direction_encoding") or {}
    winsor = params.get("winsor") or params.get("winsorization_params") or []
    bin_edges = params.get("bin_edges") or {}
    iso_tables_raw = params.get("iso_tables") or params.get("bin_score_table_isotonic") or {}
    iso_tables = {str(vid): _as_float_dict(tbl) for vid, tbl in iso_tables_raw.items()}
    weights = params.get("item_weights") or params.get("optimized_weights") or {}
    block_norm = params.get("block_norm") or params.get("block_normalization") or {}
    boundaries = params.get("boundaries") or {}
    sparse_tail_collapse_map = params.get("sparse_tail_collapse_map") or {}
    imp = params.get("imputation_map") or {}
    pd_map = params.get("pd_map") or params.get("master_scale_pd") or params.get("master_scale_pd_mapping") or {}
    grade2num = params.get("grade2num") or GRADE2NUM_10
    combined = params.get("combined_weights") or {"financial": 0.70, "nonfinancial": 0.30}
    mode = str(params.get("item_score_interpolation", "step")).lower()
    if not sel:
        raise KeyError("Alpha params missing selected_variables/selected_variable_records")
    missing = [k for k in ["financial", "nonfinancial"] if k not in block_norm]
    if missing:
        raise KeyError(f"Alpha params missing block normalization keys: {missing}")
    return {
        "selected_variables": sel,
        "fin_ids": fin,
        "nonfin_ids": nf,
        "directions": dirs,
        "winsor": winsor,
        "bin_edges": bin_edges,
        "iso_tables": iso_tables,
        "item_weights": {str(k): float(v) for k, v in weights.items()},
        "block_norm": block_norm,
        "boundaries": boundaries,
        "sparse_tail_collapse_map": {str(k): str(v) for k, v in sparse_tail_collapse_map.items()},
        "imputation_map": {str(k): float(v) for k, v in imp.items()},
        "pd_map": pd_map,
        "grade2num": grade2num,
        "combined_weights": combined,
        "item_score_interpolation": mode,
    }


def build_alpha_scorer(params: Mapping[str, Any]) -> Callable:
    """Factory returning a stateless scorer function from Oracle α parameters."""
    p = _normalize_params(params)
    SEL_IDS = list(p["selected_variables"])
    FIN_IDS = list(p["fin_ids"])
    NONFIN_IDS = list(p["nonfin_ids"])
    WINSOR = {str(w["variable_id"]): w for w in p["winsor"] if isinstance(w, Mapping) and w.get("variable_id")}
    BIN_EDGES = p["bin_edges"]
    ISO_TABLES = p["iso_tables"]
    WEIGHTS = p["item_weights"]
    BLOCK_NORM = p["block_norm"]
    BOUNDARIES = ensure_10_grade_contract(p["boundaries"])
    SPARSE_TAIL_COLLAPSE_MAP = p.get("sparse_tail_collapse_map", {}) or {}
    IMP = p["imputation_map"]
    PD_MAP = p["pd_map"]
    GRADE2NUM = p["grade2num"]
    COMBINED = p["combined_weights"]
    INTERP = p["item_score_interpolation"]

    def _edges_array(vid: str) -> np.ndarray:
        binfo = BIN_EDGES[vid]
        return np.asarray(binfo.get("edges", []), dtype=float)

    def _value_to_bin(vid: str, raw: float) -> int:
        binfo = BIN_EDGES[vid]
        edges = _edges_array(vid)
        if len(edges) == 0:
            return 0
        if binfo.get("is_binary") or binfo.get("is_low_unique"):
            return int(min(range(len(edges)), key=lambda i: abs(float(raw) - float(edges[i]))))
        return max(0, min(int(np.searchsorted(edges, raw, side="right")) - 1, len(edges) - 2))

    def _continuous_score(vid: str, raw: float) -> float | None:
        """Linearly interpolate neighboring bin scores for continuous variables."""
        binfo = BIN_EDGES[vid]
        if binfo.get("is_binary") or binfo.get("is_low_unique") or INTERP in {"none", "step", "bin"}:
            return None
        edges = _edges_array(vid)
        table = ISO_TABLES.get(vid, {})
        if len(edges) < 2 or not table:
            return None
        centers = []
        scores = []
        for i in range(len(edges) - 1):
            if i in table:
                centers.append((float(edges[i]) + float(edges[i + 1])) / 2.0)
                scores.append(float(table[i]))
        if len(centers) < 2:
            return None
        return float(np.interp(float(raw), np.asarray(centers), np.asarray(scores), left=scores[0], right=scores[-1]))

    def _value_to_score(vid: str, raw_value: Any) -> tuple[float, bool]:
        """Returns (score, was_imputed)."""
        if raw_value is None or pd.isna(raw_value):
            return float(IMP.get(vid, 50.0)), True
        raw = float(raw_value)
        wp = WINSOR.get(vid, {})
        if wp.get("winsor_applied"):
            raw = float(np.clip(raw, wp["p01_dev"], wp["p99_dev"]))
        score = _continuous_score(vid, raw)
        if score is not None:
            return score, False
        b = _value_to_bin(vid, raw)
        score = ISO_TABLES.get(vid, {}).get(b)
        if score is None:
            return float(IMP.get(vid, 50.0)), True
        return float(score), False

    def _scale_block(raw: float, block: str) -> float:
        p01 = float(BLOCK_NORM[block]["p01"])
        p99 = float(BLOCK_NORM[block]["p99"])
        denom = max(p99 - p01, 1e-12)
        return float(np.clip((raw - p01) / denom * 100.0, 0.0, 100.0))

    def _assign_grade(R: float) -> str:
        raw_grade = str(assign_grade_10([R], BOUNDARIES)[0])
        return str(SPARSE_TAIL_COLLAPSE_MAP.get(raw_grade, raw_grade))

    def _assign_raw_grade(R: float) -> str:
        return str(assign_grade_10([R], BOUNDARIES)[0])

    def scorer(raw_values: Mapping[str, Any]) -> dict[str, Any]:
        item_scores: dict[str, float] = {}
        imputed: dict[str, bool] = {}
        for vid in SEL_IDS:
            score, was_imp = _value_to_score(vid, raw_values.get(vid))
            item_scores[vid] = score
            imputed[vid] = was_imp

        fin_raw = sum(item_scores[v] * WEIGHTS[v] for v in FIN_IDS)
        nf_raw = sum(item_scores[v] * WEIGHTS[v] for v in NONFIN_IDS)
        fin_score = _scale_block(fin_raw, "financial")
        nf_score = _scale_block(nf_raw, "nonfinancial")
        wf = float(COMBINED.get("financial", 0.70))
        wnf = float(COMBINED.get("nonfinancial", 0.30))
        denom = max(wf + wnf, 1e-12)
        R_score = (wf * fin_score + wnf * nf_score) / denom
        R_grade_raw = _assign_raw_grade(R_score)
        R_grade = SPARSE_TAIL_COLLAPSE_MAP.get(R_grade_raw, R_grade_raw)
        R_grade_num = GRADE2NUM.get(R_grade, GRADE2NUM_10.get(R_grade, 10))
        R_PD = PD_MAP.get(R_grade, 1.0)

        return {
            "item_scores": item_scores,
            "imputed": imputed,
            "fin_score": fin_score,
            "nonfin_score": nf_score,
            "R_score": float(R_score),
            "R_grade_raw": R_grade_raw,
            "R_grade": R_grade,
            "R_grade_num": int(R_grade_num),
            "R_PD": float(R_PD),
        }

    return scorer
