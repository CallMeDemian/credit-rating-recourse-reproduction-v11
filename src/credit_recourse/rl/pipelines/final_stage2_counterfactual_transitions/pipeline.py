from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.rl.common.io import final_root, read_parquet_required, write_json
from credit_recourse.rl.common.actions import load_action_space, assert_training_labels_allowed
from credit_recourse.rl.common.temporal import load_temporal_contract
from credit_recourse.rl.pipelines.final_stage2_candidate_projection.pipeline import (
    compute_phi_with_frozen_cdf,
    materialize_phi_aliases,
    sector_col,
    _vals,
    _compute_aux_reward_raw_deltas,
    apply_merton_fcff_aux_reward,
    PHI_COMPONENTS,
    REQ_REWARD,
)
from credit_recourse.simulator.action import Action, clip_action
from credit_recourse.simulator.business_plan import BusinessPlan, calibrate_business_plan
from credit_recourse.simulator.financial_simulator import FinancialSimulator
from credit_recourse.simulator.firm_state import load_firm_state_from_registry
from credit_recourse.contracts.account_registry import ACCOUNT_REGISTRY


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expand_stage2_state_aliases(d: dict, *, next_state: bool = False) -> dict:
    """Expose Stage2 canonical sim__/raw__/reward_only__ columns as FirmState fields.

    load_firm_state_from_columns intentionally understands raw U-code headers and
    bare FirmState field names. Stage2 transition rows, however, mainly carry
    canonical columns such as sim__revenue and sim__net_income.  Without this
    bridge the counterfactual simulator can receive an almost empty FirmState.
    """
    out = dict(d)
    prefixes = ["next__"] if next_state else [""]
    for concept in ACCOUNT_REGISTRY.values():
        field = concept.field
        if field in out:
            continue
        candidates: list[str] = []
        for prefix in prefixes:
            for alias in concept.aliases:
                candidates.append(prefix + alias)
            for code in concept.codes:
                candidates.append(prefix + code)
                candidates.append(prefix + f"raw__avs__[{code}]")
                candidates.append(prefix + f"raw__avs__{code}")
                candidates.append(prefix + f"income_statement__[{code}]")
                candidates.append(prefix + f"balance_sheet__[{code}]")
                candidates.append(prefix + f"cash_flow__[{code}]")
        # reward_only aliases are current-state only and used for CF reward plumbing.
        if not next_state:
            candidates.extend([f"reward_only__sim__{field}", f"reward_only__{field}"])
        for c in candidates:
            if c in out and out[c] is not None:
                out[field] = out[c]
                break
    return out


def _row_to_state(row: pd.Series):
    d = _expand_stage2_state_aliases(row.to_dict(), next_state=False)
    firm_id = str(row.get("firm_id", row.get("corp_code", "UNKNOWN"))).replace(".0", "").zfill(6)
    year = int(float(row.get("fiscal_year", row.get("year", 0)) or 0))
    sector = str(row.get("sector_7", row.get("industry_class", row.get("sector", "Unknown"))))
    fs = load_firm_state_from_registry(d, firm_id=firm_id, year=year, sector=sector)
    fs.rating_grade = row.get("rating_grade", None)
    fs.rating_num = row.get("rating_num", row.get("rating_num_10", None))
    return fs


def _candidate_action(row: pd.Series, columns: list[str]) -> Action:
    return clip_action(Action(**{c.replace("action__", ""): float(row.get(c, 0.0) or 0.0) for c in columns}))


def _derived_from_state_dict(d: dict) -> dict[str, float]:
    def val(k):
        x = d.get(k, 0.0)
        try:
            x = float(x)
        except Exception:
            return 0.0
        return x if np.isfinite(x) else 0.0

    revenue = val("revenue")
    assets = val("total_assets")
    return {
        "derived__roa_proxy": val("net_income") / assets if assets else 0.0,
        "derived__operating_margin": val("operating_income") / revenue if revenue else 0.0,
        "derived__cogs_to_revenue": val("cogs") / revenue if revenue else 0.0,
        "derived__sga_to_revenue": val("sga") / revenue if revenue else 0.0,
        "derived__financial_cost_to_revenue": val("financial_cost") / revenue if revenue else 0.0,
        "derived__debt_to_assets": val("total_liabilities") / assets if assets else 0.0,
    }


def _build_history(df: pd.DataFrame, firm_id: str, year: int) -> list:
    hist = df[(df["firm_id"].astype(str) == str(firm_id)) & (pd.to_numeric(df["fiscal_year"], errors="coerce") <= float(year))]
    hist = hist.sort_values("fiscal_year").tail(3)
    out = []
    for _, hr in hist.iterrows():
        try:
            out.append(_row_to_state(hr))
        except Exception:
            pass
    return out


def _load_reward_stats(stage2_dir: Path) -> dict:
    path = stage2_dir / "reward_standardization_stats.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Stage2 reward standardization stats required for counterfactual transitions: {path}. "
            "Run final_stage2_candidate_projection first; identity fallback is forbidden for final runs."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ["lambda_phi", "reward_mean_train", "reward_std_train", "aux_reward_stats"]
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Malformed reward_standardization_stats.json; missing keys: {missing}")
    return payload

def _load_frozen_phi_cdf_from_stage2(inner_train_df: pd.DataFrame, stage2_dir: Path) -> dict:
    """Reconstruct the Stage2 frozen CDF in memory without writing a new CDF artifact.

    The current Stage2 projection persists breakpoints/metadata but not the full
    sorted reference arrays needed by compute_phi_with_frozen_cdf.  To preserve
    DNT-6 semantics without creating a second CDF artifact, this helper rebuilds
    the in-memory reference from the exact Stage2 inner-train projection rows and
    asserts the original frozen-CDF metadata exists.
    """
    meta_path = stage2_dir / "sector_phi_cdf_metadata.json"
    breakpoints_path = stage2_dir / "sector_phi_breakpoints.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing Stage2 frozen sector-phi metadata: {meta_path}")
    if not breakpoints_path.exists():
        raise FileNotFoundError(f"Missing Stage2 frozen sector-phi breakpoints: {breakpoints_path}")
    train_df = materialize_phi_aliases(inner_train_df, next_state=False)
    missing = [c for c in PHI_COMPONENTS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Cannot reconstruct frozen sector-phi CDF; missing components: {missing}")
    sec = sector_col(train_df)
    ref = {}
    for comp in PHI_COMPONENTS:
        ref[comp] = {}
        global_vals = np.sort(_vals(train_df[comp]))
        if len(global_vals) == 0:
            raise ValueError(f"No non-null values for sector-phi component {comp}")
        ref[comp]["__GLOBAL__"] = global_vals.tolist()
        group_iter = train_df.groupby(sec, dropna=False) if sec else [("__GLOBAL__", train_df)]
        for key, g in group_iter:
            vals = np.sort(_vals(g[comp]))
            if len(vals) < 20:
                vals = global_vals
            ref[comp][str(key)] = vals.tolist()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "counterfactual_cdf_policy": "in_memory_reconstruction_from_existing_stage2_projection_inner_train_rows",
        "counterfactual_new_cdf_artifact_written": False,
        "stage2_sector_phi_breakpoints_sha256": sha256_file(breakpoints_path),
    })
    return {"reference": ref, "sector_column": sec, "metadata": meta}


def _select_business_plan(mode: str, hist: list, rating_grade=None) -> BusinessPlan:
    if mode == "default":
        return BusinessPlan()
    if mode == "calibrated":
        return calibrate_business_plan(hist, grade=rating_grade) if hist else BusinessPlan()
    raise ValueError(f"Unsupported sim_business_plan_mode: {mode}")


def _find_loopa_summary(stage2_dir: Path) -> Path | None:
    candidates = [
        stage2_dir / "verification" / "loopA_dimension_bias_summary.csv",
        stage2_dir / "loopA_dimension_bias_summary.csv",
        stage2_dir.parent / "stage2_substrate_loopA_loopB2" / "loopA_dimension_bias_summary.csv",
        stage2_dir.parent / "stage2_substrate_validation" / "loopA_dimension_bias_summary.csv",
        stage2_dir.parent / "verification" / "loopA_dimension_bias_summary.csv",
        stage2_dir.parent / "oracle_verification" / "loopA_dimension_bias_summary.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _find_loopa_report(stage2_dir: Path) -> Path | None:
    for p in [
        stage2_dir / "verification" / "substrate_loopA_loopB2_report.json",
        stage2_dir.parent / "stage2_substrate_loopA_loopB2" / "substrate_loopA_loopB2_report.json",
        stage2_dir.parent / "stage2_substrate_validation" / "substrate_loopA_loopB2_report.json",
        stage2_dir.parent / "verification" / "substrate_loopA_loopB2_report.json",
    ]:
        if p.exists():
            return p
    return None


def _coerce_metric_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    cols = [str(c) for c in df.columns]
    for pat in patterns:
        for c in cols:
            if pat.lower() in c.lower():
                return c
    return None


def _component_rows(df: pd.DataFrame, *, token: str, dim_col: str, component_col: str | None) -> pd.DataFrame:
    if component_col is not None:
        sub = df[df[component_col].astype(str).str.lower().eq(token)].copy()
        if not sub.empty:
            return sub
    dim = df[dim_col].astype(str).str.lower()
    if token == "merton":
        names = {"total_assets", "short_term_debt", "long_term_debt", "bonds"}
        return df[dim.isin(names)].copy()
    if token == "fcff":
        names = {"operating_cf"}
        return df[dim.isin(names) | dim.str.contains("operating|cash|flow|ocf", regex=True, na=False)].copy()
    return df[dim.str.contains(token, na=False)].copy()


def _fidelity_gate_policy(reward_mode: str, fidelity_gate: str, max_rel_err_assets: float, stage2_dir: Path) -> dict:
    if fidelity_gate not in {"strict", "warn", "off"}:
        raise ValueError(f"Unsupported fidelity_gate={fidelity_gate}")
    meta = {
        "fidelity_gate": fidelity_gate,
        "max_rel_err_assets": float(max_rel_err_assets),
        "gate_checked_components": ["phi", "merton"] if reward_mode == "phi_merton" else ["phi", "merton", "fcff"],
        "strict_gate_enforced": fidelity_gate == "strict",
        "numeric_loopA_summary_required_for_strict": True,
        "gate_policy": "component_group_worst_dimension_median_abs_err_over_assets",
    }
    if fidelity_gate == "off":
        meta["status"] = "SKIPPED_BY_FLAG"
        return meta

    summary_path = _find_loopa_summary(stage2_dir)
    report_path = _find_loopa_report(stage2_dir)
    if report_path is not None:
        try:
            meta["loopA_report_path"] = str(report_path)
            meta["loopA_report"] = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            meta["loopA_report_read_error"] = str(exc)

    if summary_path is None:
        msg = (
            "Missing loopA_dimension_bias_summary.csv for counterfactual fidelity gate. "
            "Run credit_recourse.oracle.verification.verify_stage2_substrate_loopA_loopB2 "
            "with the same --sim-business-plan-mode before using strict counterfactual training."
        )
        meta["status"] = "MISSING_LOOPA_SUMMARY"
        meta["message"] = msg
        if fidelity_gate == "strict":
            raise FileNotFoundError(msg)
        return meta

    df = pd.read_csv(summary_path)
    meta["loopA_summary_path"] = str(summary_path)
    meta["loopA_summary_rows"] = int(len(df))
    dim_col = _coerce_metric_column(df, ["dimension", "metric", "variable", "target"])
    component_col = _coerce_metric_column(df, ["component_group", "component"])
    rel_col = _coerce_metric_column(df, ["median_abs_err_over_assets", "median_abs_error_over_assets", "median_rel_err_assets", "median_abs_rel_err", "median_abs_pct_assets"])
    spear_col = _coerce_metric_column(df, ["spearman", "rank_corr"])
    coverage_col = _coerce_metric_column(df, ["coverage_share", "coverage"])

    if dim_col is None or rel_col is None or len(df) <= 0:
        msg = f"LoopA fidelity summary lacks required non-empty dimension/median-relative-error columns: {summary_path} columns={list(df.columns)} rows={len(df)}"
        meta["status"] = "MALFORMED_LOOPA_SUMMARY"
        meta["message"] = msg
        if fidelity_gate == "strict":
            raise ValueError(msg)
        return meta

    needed = ["merton"]
    if reward_mode in {"phi_merton_fcff", "phi_merton_fcff_liquidity"}:
        needed.append("fcff")
    checks = []
    violations = []
    for token in needed:
        sub = _component_rows(df, token=token, dim_col=dim_col, component_col=component_col)
        if sub.empty:
            violations.append({"component": token, "reason": "missing_component_rows"})
            continue
        rel = pd.to_numeric(sub[rel_col], errors="coerce").dropna()
        # Conservative gate: all required component dimensions must be below
        # threshold.  Older code used min(), which could pass if only one
        # dimension was good while other reward-critical dimensions were bad.
        val = float(rel.max()) if len(rel) else float("nan")
        rec = {
            "component": token,
            "metric_column": rel_col,
            "aggregation": "max_across_component_dimensions",
            "worst_median_rel_err_assets": val,
            "dimensions": sorted(sub[dim_col].astype(str).unique().tolist()),
        }
        if coverage_col is not None:
            cov = pd.to_numeric(sub[coverage_col], errors="coerce").dropna()
            if len(cov):
                rec["min_coverage_share"] = float(cov.min())
        if spear_col is not None:
            sp = pd.to_numeric(sub[spear_col], errors="coerce").dropna()
            if len(sp):
                rec["min_spearman"] = float(sp.min())
        checks.append(rec)
        if (not np.isfinite(val)) or val > float(max_rel_err_assets):
            violations.append({"component": token, "worst_median_rel_err_assets": val, "threshold": float(max_rel_err_assets), "dimensions": rec["dimensions"]})
    meta["checks"] = checks
    meta["violations"] = violations
    if violations:
        meta["status"] = "FAIL"
        if fidelity_gate == "strict":
            raise ValueError(f"Counterfactual fidelity gate failed: {violations}")
    else:
        meta["status"] = "PASS"
    return meta


def _standardize(out: pd.DataFrame, stats: dict, reward_mode: str) -> pd.DataFrame:
    out = out.copy()
    obs = out.get("is_observed_transition")
    if obs is None:
        obs = pd.Series(False, index=out.index)
    obs = obs.fillna(False).astype(bool)

    raw_notch = pd.to_numeric(
        out["reward_raw_notch"] if "reward_raw_notch" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)
    raw = pd.to_numeric(
        out["reward_raw"] if "reward_raw" in out.columns else pd.Series(0.0, index=out.index),
        errors="coerce",
    ).fillna(0.0)
    # Option B: keep the real rating-notch reward only for the matched observed cell;
    # non-observed candidate rows have no realized rating outcome and therefore get 0.
    out["reward_raw_notch"] = raw_notch.where(obs, 0.0)
    out["reward_raw"] = raw.where(obs, 0.0)
    out["delta_phi_clipped"] = pd.to_numeric(out["delta_phi"], errors="coerce").clip(-1.0, 1.0)
    out["lambda_phi"] = float(stats.get("lambda_phi", 1.0) or 1.0)
    out["reward_aux_phi"] = out["lambda_phi"] * out["delta_phi_clipped"].fillna(0.0)

    aux_stats = dict(stats.get("aux_reward_stats") or {})
    if reward_mode not in {"phi_merton_fcff", "phi_merton_fcff_liquidity"}:
        aux_stats["lambda_fcff"] = 0.0
    if reward_mode not in {"phi_merton_liquidity", "phi_merton_fcff_liquidity"}:
        aux_stats["lambda_liquidity"] = 0.0
    out, _aux_meta = apply_merton_fcff_aux_reward(out, aux_stats, phase_name="counterfactual_phase3_iql")
    out["reward_total_raw"] = (
        out["reward_raw"]
        + out["reward_aux_phi"].fillna(0.0)
        + pd.to_numeric(out.get("reward_aux_merton", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("reward_aux_fcff", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(out.get("reward_aux_liquidity", 0.0), errors="coerce").fillna(0.0)
    )
    mean = float(stats.get("reward_mean_train", 0.0) or 0.0)
    std = float(stats.get("reward_std_train", 1.0) or 1.0)
    if not np.isfinite(std) or abs(std) <= 1e-12:
        std = 1.0
    out["reward_mean_train"] = mean
    out["reward_std_train"] = std
    out["reward_train"] = (pd.to_numeric(out["reward_total_raw"], errors="coerce").fillna(mean) - mean) / std
    out["reward_original"] = out["reward_raw"]
    out["reward"] = out["reward_train"]
    for c in REQ_REWARD:
        if c not in out.columns:
            out[c] = 0.0
    return out

def _validate_counterfactual(df: pd.DataFrame, base_rows: int, n_candidates: int, labels: list[str]) -> None:
    if len(df) != int(base_rows) * int(n_candidates):
        raise ValueError(f"counterfactual row count mismatch: got={len(df)} expected={base_rows*n_candidates}")
    # Guard against reference-oracle leakage without rejecting non-oracle diagnostics.
    # Columns such as transition__positive_transition_prior_score are behavior/transition
    # priors, not alpha/beta/gamma/reference-oracle scores.  A broad endswith("score")
    # or substring token like "r_score" incorrectly blocks them.
    forbidden_score_tokens = (
        "alpha_score",
        "beta_score",
        "gamma_score",
        "oracle_score",
        "reference_oracle_score",
        "oracle_alpha",
        "oracle_beta",
        "oracle_gamma",
    )
    allowed_non_oracle_score_tokens = (
        "transition_prior_score",
        "positive_transition_prior_score",
        "negative_transition_prior_score",
    )
    bad = []
    for c in df.columns:
        lc = str(c).lower()
        if any(tok in lc for tok in allowed_non_oracle_score_tokens):
            continue
        if any(tok in lc for tok in forbidden_score_tokens):
            bad.append(c)
    if bad:
        raise ValueError(f"Forbidden oracle score columns in counterfactual transitions: {bad[:20]}")
    miss = sorted(set(labels) - set(df["candidate_id"].astype(str).unique()))
    if miss:
        raise ValueError(f"Counterfactual output missing candidates: {miss}")
    if "is_observed_transition" in df.columns:
        obs = df["is_observed_transition"].fillna(False).astype(bool)
        n_obs = int(obs.sum())
        if n_obs != int(base_rows):
            raise ValueError(f"expected exactly one observed transition per firm-year: got {n_obs}, base_rows={base_rows}")
        key_cols = [c for c in ["firm_id", "fiscal_year"] if c in df.columns]
        if key_cols:
            per_key = df.loc[obs].groupby(key_cols, dropna=False).size()
            if len(per_key) != int(base_rows) or int(per_key.max()) != 1 or int(per_key.min()) != 1:
                raise ValueError("observed transition marker must appear exactly once per firm-year key")
    finite_cols = ["reward_train", "reward_total_raw", "phi_t", "phi_tplusH", "delta_phi"]
    for c in finite_cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if x.isna().any():
            raise ValueError(f"Non-finite counterfactual reward column {c}: n={int(x.isna().sum())}")


def run(project_root: Path, *, magnitude_quantile: int = 50, reward_mode: str = "phi_merton", done_mode: str = "terminal", fidelity_gate: str = "strict", max_rel_err_assets: float = 0.05, sim_business_plan_mode: str = "default", preserve_current_non_current_residual: bool = False, max_rows: int | None = None) -> dict:
    root = project_root.resolve()
    temporal = load_temporal_contract(root)
    final = final_root(root)
    stage2 = final / "stage2_candidate_projection"
    out_dir = stage2
    phase_path = stage2 / f"phase3_iql_candidate__P{magnitude_quantile}.parquet"
    if not phase_path.exists():
        phase_path = stage2 / "phase3_iql_candidate.parquet"
    phase = read_parquet_required(phase_path)
    if max_rows is not None and int(max_rows) > 0:
        phase = phase.head(int(max_rows)).copy()
    if phase.empty:
        raise ValueError(f"Empty phase3 input for counterfactual transitions: {phase_path}")
    space = load_action_space(root)
    assert_training_labels_allowed(phase, space, "candidate_id")
    cand = space.frame(include_stage6_extras=False)
    cand = cand[cand["candidate_id"].astype(str).isin(space.train_labels)].copy()
    if len(cand) != len(space.train_labels):
        raise ValueError("Counterfactual candidate frame does not cover all train labels")

    # Rebuild the same train-only frozen CDF reference from current phase3 rows.
    y = pd.to_numeric(phase.get("fiscal_year", phase.get("year")), errors="coerce")
    inner = phase.loc[y <= int(temporal.inner_train_year_max)].copy()
    if inner.empty:
        raise ValueError("Cannot build counterfactual frozen CDF: empty inner-train partition")
    frozen = _load_frozen_phi_cdf_from_stage2(inner, stage2)
    reward_stats = _load_reward_stats(stage2)
    gate_meta = _fidelity_gate_policy(reward_mode, fidelity_gate, max_rel_err_assets, stage2)

    sim = FinancialSimulator(preserve_current_non_current_residual=preserve_current_non_current_residual)
    rows = []
    for _, r in phase.iterrows():
        fs = _row_to_state(r)
        hist = _build_history(phase, str(fs.firm_id), int(fs.year))
        bp = _select_business_plan(sim_business_plan_mode, hist, rating_grade=fs.rating_grade)
        current_state = fs.to_dict()
        current_derived = _derived_from_state_dict(current_state)
        observed_cid = str(r.get("candidate_id"))
        for _, cr in cand.iterrows():
            cid = str(cr["candidate_id"])
            row = r.to_dict()
            row["candidate_id"] = cid
            for k, v in current_derived.items():
                row[k] = float(v)

            if cid == observed_cid:
                # Option B matched cell: use the observed real transition already carried
                # by phase3_iql_candidate. Do not overwrite next__sim__/next__derived
                # with simulator output and do not erase the real rating reward.
                row["done"] = 1.0 if done_mode == "terminal" else 0.0
                row["counterfactual_transition"] = False
                row["is_observed_transition"] = True
                row["counterfactual_done_mode"] = done_mode
                row["simulator_sustainability"] = float("nan")
                row["simulator_plug_used"] = "observed_real_transition"
                row["simulator_plug_amount"] = 0.0
                rows.append(row)
                continue

            # Non-observed cell: simulate exactly the candidate-specific alternate future.
            act = _candidate_action(cr, space.columns)
            res = sim.simulate(fs, bp, act)
            next_state = res.state_t1.to_dict()
            for col in space.columns:
                row[col] = float(cr.get(col, 0.0) or 0.0)
            for k, v in next_state.items():
                if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
                    row[f"next__sim__{k}"] = 0.0 if v is None else float(v)
            next_derived = _derived_from_state_dict(next_state)
            for k, v in next_derived.items():
                row[f"next__{k}"] = float(v)
            row["done"] = 1.0 if done_mode == "terminal" else 0.0
            row["counterfactual_transition"] = True
            row["is_observed_transition"] = False
            row["counterfactual_done_mode"] = done_mode
            row["simulator_sustainability"] = res.sustainability
            row["simulator_plug_used"] = res.plug_used
            row["simulator_plug_amount"] = float(res.plug_amount)
            rows.append(row)
    out = pd.DataFrame(rows)

    # Terminal mode ignores V(s'), but Stage5 still requires next__ feature columns.
    # Preserve schema reachability by supplying next__<feature>=current feature for
    # non-sim/non-derived columns that the one-year simulator cannot materialize.
    feature_manifest = stage2 / "feature_manifest.json"
    if feature_manifest.exists():
        manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
        for f in manifest.get("features", []):
            name = f.get("name") if isinstance(f, dict) else None
            if name and name in out.columns and f"next__{name}" not in out.columns:
                out[f"next__{name}"] = out[name]

    out["phi_t"] = compute_phi_with_frozen_cdf(out, frozen)
    nx = out.copy()
    for c in PHI_COMPONENTS:
        nx[c] = out[f"next__{c}"]
    out["phi_tplusH"] = compute_phi_with_frozen_cdf(nx, frozen)
    out["delta_phi"] = out["phi_tplusH"] - out["phi_t"]
    out = _standardize(out, reward_stats, reward_mode)
    _validate_counterfactual(out, len(phase), len(cand), space.train_labels)

    stem = f"phase3_iql_counterfactual_candidate__P{magnitude_quantile}.parquet"
    out_path = out_dir / stem
    out.to_parquet(out_path, index=False)
    meta = {
        "stage": "final_stage2_counterfactual_transitions",
        "created_utc": now(),
        "status": "PASS",
        "input_phase3": str(phase_path.relative_to(root)),
        "input_phase3_sha256": sha256_file(phase_path),
        "output": str(out_path.relative_to(root)),
        "output_sha256": sha256_file(out_path),
        "base_rows": int(len(phase)),
        "candidate_count": int(len(cand)),
        "rows": int(len(out)),
        "observed_real_rows": int(out.get("is_observed_transition", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum()),
        "simulated_rows": int(len(out) - out.get("is_observed_transition", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum()),
        "matched_cell_reward_source": "real_rating_plus_real_nextstate_phi_merton_optional_fcff_liquidity",
        "nonobserved_cell_reward_source": "simulated_phi_merton_optional_fcff_liquidity_no_rating",
        "magnitude_quantile": int(magnitude_quantile),
        "reward_mode": reward_mode,
        "done_mode": done_mode,
        "fidelity_gate": fidelity_gate,
        "fidelity_gate_metadata": gate_meta,
        "sim_business_plan_mode": sim_business_plan_mode,
        "preserve_current_non_current_residual": bool(preserve_current_non_current_residual),
        "candidate_library_hash": space.candidate_library_hash,
        "final_action_contract_hash": space.final_action_contract_hash,
        "reference_oracle_scores_used": False,
        "frozen_cdf_source": "existing_stage2_projection_inner_train_cdf_in_memory_reconstruction_no_new_artifact",
        "new_frozen_cdf_artifact_written": False,
        "reward_stats_source": "stage2_candidate_projection/reward_standardization_stats.json (required; no identity fallback)",
        "final_paper_run_allowed": bool(max_rows is None),
    }
    write_json(out_dir / f"counterfactual_transitions_metadata__P{magnitude_quantile}.json", meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--magnitude-quantile", type=int, default=50, choices=[50, 65, 75, 85])
    ap.add_argument("--reward-mode", choices=["phi_merton", "phi_merton_fcff", "phi_merton_liquidity", "phi_merton_fcff_liquidity"], default="phi_merton")
    ap.add_argument("--done-mode", choices=["terminal", "bootstrap"], default="terminal")
    ap.add_argument("--fidelity-gate", choices=["strict", "warn", "off"], default="strict")
    ap.add_argument("--max-rel-err-assets", type=float, default=0.05)
    ap.add_argument("--sim-business-plan-mode", choices=["default", "calibrated"], default="default")
    ap.add_argument("--preserve-current-non-current-residual", action="store_true")
    ap.add_argument("--max-rows", type=int, default=None, help="Debug smoke limit only; omitted for full output")
    args = ap.parse_args(argv)
    run(Path(args.project_root), magnitude_quantile=args.magnitude_quantile, reward_mode=args.reward_mode, done_mode=args.done_mode, fidelity_gate=args.fidelity_gate, max_rel_err_assets=args.max_rel_err_assets, sim_business_plan_mode=args.sim_business_plan_mode, preserve_current_non_current_residual=args.preserve_current_non_current_residual, max_rows=args.max_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
