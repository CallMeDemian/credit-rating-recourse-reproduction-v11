from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from credit_recourse.rl.common.io import final_root, read_parquet_required, write_json
from credit_recourse.rl.common.actions import load_action_space
from credit_recourse.simulator.financial_simulator import FinancialSimulator
from credit_recourse.simulator.business_plan import calibrate_business_plan, BusinessPlan
from credit_recourse.simulator.action import Action, clip_action
from credit_recourse.simulator.firm_state import FirmState, load_firm_state_from_columns
from credit_recourse.oracle.verification.verify_stage2_substrate_loopA_loopB2 import (
    MERTON_FIDELITY_FIELDS,
    LOOPA_ACTUAL_NEXT_ALIASES,
    LOOPA_PREDICTED_ALIASES,
    _build_loopa_phase,
    _compute_error_rows,
    _summarize_errors,
    _normalise_loopa_keys,
    _actual_next_match_count,
    _load_observed_next_handoff,
)

_STATE_VALUE_FIELDS = [
    f.name
    for f in fields(FirmState)
    if f.name not in {"firm_id", "year", "sector", "rating_num", "rating_grade"}
]


def _finite(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _row_to_state(row: pd.Series) -> FirmState:
    d = row.to_dict()
    firm_id = str(row.get("firm_id", row.get("corp_code", "UNKNOWN")))
    year = int(float(row.get("fiscal_year", row.get("year", 0)) or 0))
    sector = str(row.get("sector_7", row.get("industry_class", "Unknown")))
    fs = load_firm_state_from_columns(d, firm_id=firm_id, year=year, sector=sector)
    fs.rating_grade = row.get("rating_grade", None)
    fs.rating_num = row.get("rating_num", row.get("rating_num_10", None))
    return fs


def _action_from_row(row: pd.Series, space) -> Action:
    return clip_action(Action(**{c.replace("action__", ""): float(row.get(c, 0.0) or 0.0) for c in space.columns}))


def _zero_action(space) -> Action:
    return clip_action(Action(**{c.replace("action__", ""): 0.0 for c in space.columns}))


def _select_business_plan(mode: str, history: list[FirmState], *, rating_grade=None) -> BusinessPlan:
    if mode == "default":
        return BusinessPlan()
    if mode == "calibrated":
        return calibrate_business_plan(history, grade=rating_grade) if history else BusinessPlan()
    raise ValueError(f"Unsupported sim_business_plan_mode={mode}")


def _build_history(phase: pd.DataFrame, fs: FirmState) -> list[FirmState]:
    if "firm_id" not in phase.columns:
        return []
    hdf = phase[
        (phase.get("firm_id", "").astype(str) == str(fs.firm_id))
        & (pd.to_numeric(phase.get("fiscal_year", 0), errors="coerce") <= float(fs.year))
    ].sort_values("fiscal_year").tail(3)
    out: list[FirmState] = []
    for _, hr in hdf.iterrows():
        try:
            out.append(_row_to_state(hr))
        except Exception:
            pass
    return out


def _simulate_rows(phase: pd.DataFrame, *, project_root: Path, action_mode: str, sim_business_plan_mode: str, preserve_current_non_current_residual: bool) -> list[dict[str, Any]]:
    space = load_action_space(project_root)
    sim = FinancialSimulator(preserve_current_non_current_residual=preserve_current_non_current_residual)
    rows: list[dict[str, Any]] = []
    for _, r in phase.iterrows():
        fs = _row_to_state(r)
        hist = _build_history(phase, fs)
        bp = _select_business_plan(sim_business_plan_mode, hist, rating_grade=fs.rating_grade)
        act = _zero_action(space) if action_mode == "zero_action" else _action_from_row(r, space)
        res = sim.simulate(fs, bp, act)
        row = {"firm_id": fs.firm_id, "fiscal_year": fs.year}
        d = res.state_t1.to_dict()
        for k in _STATE_VALUE_FIELDS:
            row[k] = _finite(d.get(k))
        row["simulator_sustainability"] = res.sustainability
        row["simulator_plug_used"] = res.plug_used
        row["simulator_plug_amount"] = float(res.plug_amount)
        row["action_mode"] = action_mode
        rows.append(row)
    return rows


def _stored_next_sim_rows(phase: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, r in phase.iterrows():
        row = {"firm_id": r.get("firm_id", r.get("corp_code", "UNKNOWN")), "fiscal_year": r.get("fiscal_year", r.get("year", 0))}
        d = r.to_dict()
        any_found = False
        for field in _STATE_VALUE_FIELDS:
            aliases = LOOPA_PREDICTED_ALIASES.get(field, (field, f"next__sim__{field}", f"sim__{field}"))
            # For stored mode, prefer explicit next__sim__ aliases; do not use current sim__ fallback.
            stored_aliases = tuple(a for a in aliases if str(a).startswith("next__sim__")) + (f"next__sim__{field}",)
            val = None
            for a in stored_aliases:
                if a in d and pd.notna(d.get(a)):
                    val = d.get(a)
                    break
            if val is not None:
                row[field] = _finite(val)
                any_found = True
        if any_found:
            row["action_mode"] = "stored_next_sim"
            rows.append(row)
    return rows


def _merton_summary(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    sub = summary[summary["dimension"].astype(str).isin(sorted(MERTON_FIDELITY_FIELDS))].copy()
    sub = sub.sort_values("median_abs_err_over_assets", ascending=False)
    cols = [
        "dimension", "count", "coverage_share", "median_signed_error", "median_abs_error",
        "median_abs_err_over_assets", "p95_abs_err_over_assets", "spearman",
        "predicted_nonzero_rate", "actual_nonzero_rate",
    ]
    return sub[[c for c in cols if c in sub.columns]].to_dict(orient="records")


def _key_profile(df: pd.DataFrame, name: str) -> dict[str, Any]:
    out = {"name": name, "rows": int(len(df)), "columns": int(len(df.columns))}
    if "firm_id" in df.columns and "fiscal_year" in df.columns:
        y = pd.to_numeric(df["fiscal_year"], errors="coerce")
        out.update({
            "unique_keys": int(df[["firm_id", "fiscal_year"]].drop_duplicates().shape[0]),
            "firm_id_nunique": int(df["firm_id"].astype(str).nunique()),
            "year_min": int(y.min()) if y.notna().any() else None,
            "year_max": int(y.max()) if y.notna().any() else None,
        })
    out["has_required_raw_next_merton_rows"] = int(_actual_next_match_count(df))
    return out


def _action_profile(phase: pd.DataFrame, project_root: Path) -> dict[str, Any]:
    space = load_action_space(project_root)
    cols = [c for c in space.columns if c in phase.columns]
    out = {"action_columns_present": cols, "n_action_columns_present": int(len(cols))}
    if not cols:
        return out
    mat = phase[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    l1 = mat.abs().sum(axis=1)
    out.update({
        "l1_mean": float(l1.mean()),
        "l1_median": float(l1.median()),
        "l1_p95": float(l1.quantile(0.95)),
        "zero_action_share": float((l1 <= 1e-12).mean()),
    })
    if "candidate_id" in phase.columns:
        vc = phase["candidate_id"].astype(str).value_counts().head(20)
        out["candidate_id_top20"] = {str(k): int(v) for k, v in vc.items()}
    return out


def run(project_root: Path, *, sim_business_plan_mode: str = "calibrated", preserve_current_non_current_residual: bool = False, max_rows: int | None = None, output_dir: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    final = final_root(root)
    out = output_dir or (final / "stage2_substrate_loopA_loopB2" / "diagnostics")
    out.mkdir(parents=True, exist_ok=True)

    phase, phase_meta = _build_loopa_phase(final)
    if max_rows is not None and max_rows > 0:
        phase = phase.head(int(max_rows)).copy()

    phase_eval_path = final / "stage2_candidate_projection" / "phase_eval_candidate.parquet"
    phase_eval = _normalise_loopa_keys(read_parquet_required(phase_eval_path))
    observed_next, observed_source = _load_observed_next_handoff(final)
    observed_next = _normalise_loopa_keys(observed_next)
    eval_keys = set(map(tuple, phase_eval[["firm_id", "fiscal_year"]].dropna().astype(str).to_numpy()))
    obs_keys = set(map(tuple, observed_next[["firm_id", "fiscal_year"]].dropna().astype(str).to_numpy()))

    mode_results: dict[str, Any] = {}
    all_summary_rows = []
    for mode in ["candidate_action", "zero_action"]:
        sim_rows = _simulate_rows(
            phase,
            project_root=root,
            action_mode=mode,
            sim_business_plan_mode=sim_business_plan_mode,
            preserve_current_non_current_residual=preserve_current_non_current_residual,
        )
        errs = _compute_error_rows(phase, sim_rows)
        summary = _summarize_errors(errs)
        errs.to_csv(out / f"loopA_errors__{mode}.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / f"loopA_summary__{mode}.csv", index=False, encoding="utf-8-sig")
        for _, rr in summary.iterrows():
            d = rr.to_dict(); d["mode"] = mode; all_summary_rows.append(d)
        worst = summary[summary["dimension"].astype(str).isin(sorted(MERTON_FIDELITY_FIELDS))].copy()
        worst_val = float(pd.to_numeric(worst.get("median_abs_err_over_assets"), errors="coerce").max()) if not worst.empty else float("nan")
        mode_results[mode] = {
            "rows": int(len(sim_rows)),
            "error_rows": int(len(errs)),
            "merton_worst_median_abs_err_over_assets": worst_val,
            "merton_by_dimension": _merton_summary(summary),
        }

    stored_rows = _stored_next_sim_rows(phase)
    if stored_rows:
        errs = _compute_error_rows(phase, stored_rows)
        summary = _summarize_errors(errs)
        errs.to_csv(out / "loopA_errors__stored_next_sim.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(out / "loopA_summary__stored_next_sim.csv", index=False, encoding="utf-8-sig")
        for _, rr in summary.iterrows():
            d = rr.to_dict(); d["mode"] = "stored_next_sim"; all_summary_rows.append(d)
        worst = summary[summary["dimension"].astype(str).isin(sorted(MERTON_FIDELITY_FIELDS))].copy()
        worst_val = float(pd.to_numeric(worst.get("median_abs_err_over_assets"), errors="coerce").max()) if not worst.empty else float("nan")
        mode_results["stored_next_sim"] = {
            "rows": int(len(stored_rows)),
            "error_rows": int(len(errs)),
            "merton_worst_median_abs_err_over_assets": worst_val,
            "merton_by_dimension": _merton_summary(summary),
        }
    else:
        mode_results["stored_next_sim"] = {"status": "SKIP_NO_NEXT_SIM_COLUMNS"}

    if all_summary_rows:
        pd.DataFrame(all_summary_rows).to_csv(out / "loopA_summary_by_mode.csv", index=False, encoding="utf-8-sig")

    diagnosis = {
        "status": "PASS_DIAGNOSTICS_WRITTEN",
        "sim_business_plan_mode": sim_business_plan_mode,
        "preserve_current_non_current_residual": bool(preserve_current_non_current_residual),
        "loopA_driver_meta": phase_meta,
        "phase_eval_profile": _key_profile(phase_eval, "phase_eval_candidate"),
        "observed_next_profile": _key_profile(observed_next, "observed_next_handoff"),
        "observed_next_source": observed_source,
        "phase_eval_observed_key_overlap_same_year": int(len(eval_keys & obs_keys)),
        "phase_eval_observed_key_overlap_share_of_phase_eval": float(len(eval_keys & obs_keys) / max(len(eval_keys), 1)),
        "selected_loopA_phase_profile": _key_profile(_normalise_loopa_keys(phase), "selected_loopA_phase"),
        "action_profile": _action_profile(phase, root),
        "mode_results": mode_results,
        "interpretation_rules": {
            "candidate_action_high_zero_action_low": "Action projection/candidate action magnitudes are the main fidelity driver.",
            "candidate_action_high_stored_next_sim_low": "Verifier re-simulation settings differ from the Stage2 stored transition generation settings.",
            "all_modes_high": "BusinessPlan/account mapping/structural simulator assumptions are the main fidelity driver.",
            "phase_eval_overlap_zero": "LoopA is validating on phase3 training handoff rather than phase_eval because eval rows have no aligned observed next raw values.",
        },
        "outputs": [
            "loopA_errors__candidate_action.csv",
            "loopA_summary__candidate_action.csv",
            "loopA_errors__zero_action.csv",
            "loopA_summary__zero_action.csv",
            "loopA_summary_by_mode.csv",
            "loopA_fidelity_diagnosis.json",
        ],
        "output_dir": str(out),
    }
    if stored_rows:
        diagnosis["outputs"].extend(["loopA_errors__stored_next_sim.csv", "loopA_summary__stored_next_sim.csv"])
    write_json(out / "loopA_fidelity_diagnosis.json", diagnosis)
    print(json.dumps(diagnosis, ensure_ascii=False, indent=2))
    return diagnosis


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--sim-business-plan-mode", choices=["default", "calibrated"], default="calibrated")
    ap.add_argument("--preserve-current-non-current-residual", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args(argv)
    run(
        Path(args.project_root),
        sim_business_plan_mode=args.sim_business_plan_mode,
        preserve_current_non_current_residual=args.preserve_current_non_current_residual,
        max_rows=args.max_rows,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
