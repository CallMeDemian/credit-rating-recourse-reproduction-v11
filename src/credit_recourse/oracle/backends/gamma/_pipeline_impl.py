#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Stage 4 γ — Gamma-ML Nonlinear Rating Benchmark
==========================================================================
Role:
  Gamma-ML is a nonlinear rating-prediction robustness backend. It is not
  the simulator-mediated intervention engine; Final Stage6 remains that engine.
  Gamma-ML is a scoring backend that can be plugged into Final Stage6 later.

Design:
  - Same 11 variables as Alpha/Beta
  - Same Dev/OOT split
  - Tree-boosting regressor-to-notch target: folded rating_num
  - R_score_gamma_ml is 0-100, higher is better

Why regressor-to-notch instead of multiclass:
  Credit ratings are ordinal. Regression-to-notch makes larger notch errors
  costlier than smaller notch errors and gives a direct continuous score for
  PV-style robustness evaluation.
==========================================================================
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from scipy.stats import spearmanr

from credit_recourse.oracle.contracts.rating_scale import (
    GRADE_ORDER_10, GRADE2NUM_10, NUM2GRADE_10, PD_MAP_10,
    add_rating_scale_columns, fold_to_10, assign_grade_10, ensure_10_grade_contract, assert_grade_order_10,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor

SCRIPT_DIR = Path(__file__).parent
# Project root: <project>/src/credit_recourse/oracle/pipelines/stage01_oracle_construction/<backend>
PROJECT_ROOT = SCRIPT_DIR.parents[5]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "final_freeze" / "stage1_oracle_backends" / "gamma"
INPUT_DIR = Path(os.environ.get("ORACLE_INPUT_DIR", SCRIPT_DIR / "inputs"))
OUTPUT_DIR = Path(os.environ.get("ORACLE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
CONFIG_PATH = Path(os.environ.get("ORACLE_CONFIG", SCRIPT_DIR / "configs" / "stage4_gamma_config.yaml"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ORDER = GRADE_ORDER_10
GRADE2NUM = GRADE2NUM_10
NUM2GRADE = NUM2GRADE_10
PD_MAP = PD_MAP_10
assert_grade_order_10(GRADE_ORDER)
KEY = "거래소코드"


# ============================================================
# Year-scope control
# ============================================================
def _parse_oracle_max_year() -> int | None:
    """Return optional max year for fit/apply panel, controlled by env.

    Supported usage:
      ORACLE_MAX_YEAR=2023        -> keep rows with year <= 2023
      ORACLE_YEAR_SCOPE=Exclude2024 -> same as max year 2023
      ORACLE_YEAR_SCOPE=Include2024 -> no hard max year filter
    """
    raw = str(os.environ.get("ORACLE_MAX_YEAR", "")).strip()
    if raw:
        return int(raw)
    scope = str(os.environ.get("ORACLE_YEAR_SCOPE", "")).strip().lower()
    if scope in {"exclude2024", "exclude_2024", "upto2023", "up_to_2023", "le2023", "2023"}:
        return 2023
    return None


ORACLE_MAX_YEAR = _parse_oracle_max_year()
ORACLE_YEAR_SCOPE = str(os.environ.get(
    "ORACLE_YEAR_SCOPE",
    f"year_le_{ORACLE_MAX_YEAR}" if ORACLE_MAX_YEAR is not None else "include2024"
)).strip()


def apply_oracle_year_scope(df: pd.DataFrame, *, label: str = "panel") -> pd.DataFrame:
    if ORACLE_MAX_YEAR is None:
        print(f"  [YEAR SCOPE] {label}: Include2024/full panel; no max-year filter")
        return df
    if "year" not in df.columns:
        raise KeyError(f"ORACLE_MAX_YEAR={ORACLE_MAX_YEAR} was requested, but {label} has no 'year' column")
    before = len(df)
    out = df[df["year"] <= ORACLE_MAX_YEAR].copy()
    print(f"  [YEAR SCOPE] {label}: year <= {ORACLE_MAX_YEAR}: {before:,} -> {len(out):,}")
    return out


def scoped_oot_years(default_oot_years):
    years = list(default_oot_years)
    if ORACLE_MAX_YEAR is not None:
        years[1] = min(int(years[1]), int(ORACLE_MAX_YEAR))
    return years


def _parse_year_pair_from_env(name: str) -> list[int] | None:
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    parts = raw.replace("-", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as START,END or START-END; got {raw!r}")
    return [int(parts[0]), int(parts[1])]


def split_years_from_config(config: dict) -> tuple[list[int], list[int]]:
    split_cfg = config.get("split", {}) if isinstance(config, dict) else {}
    dev = list(split_cfg.get("dev", [2002, 2019]))
    oot = list(split_cfg.get("oot", [2020, ORACLE_MAX_YEAR or 2024]))
    env_dev = _parse_year_pair_from_env("ORACLE_DEV_YEARS")
    env_oot = _parse_year_pair_from_env("ORACLE_OOT_YEARS")
    if env_dev is not None:
        dev = env_dev
    if env_oot is not None:
        oot = env_oot
    oot = scoped_oot_years(oot)
    if int(dev[1]) >= int(oot[0]):
        raise ValueError(f"Dev/OOT split must be non-overlapping: dev={dev}, oot={oot}")
    if int(oot[1]) < int(oot[0]):
        raise ValueError(f"OOT split is empty after score-end-year scoping: oot={oot}, ORACLE_MAX_YEAR={ORACLE_MAX_YEAR}")
    return [int(dev[0]), int(dev[1])], [int(oot[0]), int(oot[1])]



def _read_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


CONFIG = _read_config()
MODELED_GRADES = CONFIG.get("model", {}).get(
    "modeled_grades", GRADE_ORDER_10
)
MODELED_GRADES = [g for g in GRADE_ORDER_10 if g in set(MODELED_GRADES)]
DEV_YEARS, OOT_YEARS = split_years_from_config(CONFIG)
CROSS_BAND_LOW = float(CONFIG.get("cross_oracle", {}).get("spearman_band_low", 0.55))
CROSS_BAND_HIGH = float(CONFIG.get("cross_oracle", {}).get("spearman_band_high", 0.90))
MODEL_CFG = CONFIG.get("tree_boosting", {})
RANDOM_STATE = int(MODEL_CFG.get("random_state", 42))

MODELED_NUMS = [GRADE2NUM[g] for g in MODELED_GRADES]
MIN_MODELED_NUM = min(MODELED_NUMS)
MAX_MODELED_NUM = max(MODELED_NUMS)


def fold_grade(g: object) -> object:
    folded = fold_to_10(g)
    return np.nan if pd.isna(folded) else folded


def score_from_predicted_notch(pred_rating_num: np.ndarray) -> np.ndarray:
    pred = np.clip(pred_rating_num, 1, 10)
    return 100.0 * (10.0 - pred) / 9.0


def nearest_modeled_grade_num(x: np.ndarray) -> np.ndarray:
    arr = np.rint(np.clip(x, MIN_MODELED_NUM, MAX_MODELED_NUM)).astype(int)
    valid = np.array(MODELED_NUMS)
    out = []
    for v in arr:
        out.append(int(valid[np.argmin(np.abs(valid - v))]))
    return np.array(out, dtype=int)


def calc_metrics(sub: pd.DataFrame) -> dict:
    valid = sub[sub["rating_num"].notna() & sub["R_grade_gamma_num"].notna()].copy()
    if valid.empty:
        return {"n": 0, "exact": np.nan, "within1": np.nan, "mae": np.nan, "rmse": np.nan, "rho": np.nan, "bias": np.nan}
    obs = pd.to_numeric(valid["rating_num_10"], errors="coerce") if "rating_num_10" in valid.columns else valid["rating_num"].astype(float)
    pred_cont = valid["predicted_rating_num_gamma"].astype(float)
    pred_grade_num = valid["R_grade_gamma_num"].astype(float)
    actual_grade = valid["grade_base_10"].map(fold_grade) if "grade_base_10" in valid.columns else valid["grade_base"].map(fold_grade)
    exact = float((actual_grade == valid["R_grade_gamma"]).mean())
    within1 = float((np.abs(obs - pred_grade_num) <= 1).mean())
    mae = float(mean_absolute_error(obs, pred_cont))
    rmse = float(mean_squared_error(obs, pred_cont) ** 0.5)
    rho = float(spearmanr(valid["R_score_gamma"], obs)[0])
    bias = float((pred_cont - obs).mean())
    return {
        "n": int(len(valid)),
        "exact": round(exact, 4),
        "within1": round(within1, 4),
        "mae_continuous": round(mae, 3),
        "rmse_continuous": round(rmse, 3),
        "rho": round(rho, 4),
        "rho_abs": round(abs(rho), 4),
        "bias_pred_minus_actual_notch": round(bias, 4),
    }


print("=" * 72)
print("Stage 4 γ — Gamma-ML Tree-Boosting Regressor-to-Notch Benchmark")
print("=" * 72)
print(f"  Modeled grades: {MODELED_GRADES}")
print(f"  Dev years: {DEV_YEARS[0]}-{DEV_YEARS[1]} | OOT years: {OOT_YEARS[0]}-{OOT_YEARS[1]}")

# ------------------------------------------------------------------
# Phase 1. Load inputs
# ------------------------------------------------------------------
print("\n[Phase 1] Loading inputs")
with open(INPUT_DIR / "stage3_v2" / "selected_variables_v2.json", encoding="utf-8") as f:
    sel_vars = json.load(f)
SEL_IDS = [v["variable_id"] for v in sel_vars]
FIN_IDS = [v["variable_id"] for v in sel_vars if v.get("source") == "financial"]
NONFIN_IDS = [v["variable_id"] for v in sel_vars if v.get("source") == "nonfinancial"]
assert len(SEL_IDS) == 11, f"Expected 11 selected variables, got {len(SEL_IDS)}"
assert "kospi_dummy" not in SEL_IDS, "Stage 1C v3.2 violation: kospi_dummy must not be selected"

base = pd.read_parquet(INPUT_DIR / "stage1b" / "firm_year_panel_v1.parquet")
fin = pd.read_parquet(INPUT_DIR / "stage2" / "engineered_financial_ratios.parquet")
nonfin = pd.read_parquet(INPUT_DIR / "stage1c_v3" / "nonfinancial_metadata_panel.parquet")

df = base.merge(fin, on=[KEY, "year"], how="left", suffixes=("", "_f"))
df = df.merge(nonfin, on=[KEY, "year"], how="left", suffixes=("", "_n"))
df = apply_oracle_year_scope(df, label='merged panel')
if "grade_base" not in df.columns and "rating" in df.columns:
    df["grade_base"] = df["rating"]
if "grade_base_10" not in df.columns or "rating_num_10" not in df.columns:
    df = add_rating_scale_columns(df, source_col="grade_base")
df["grade_base"] = df["grade_base_10"]
df["rating_num"] = pd.to_numeric(df["rating_num_10"], errors="coerce")
print(f"  Merged panel: {len(df):,} rows")
print(f"  Variables: {len(SEL_IDS)} (Fin:{len(FIN_IDS)}, NF:{len(NONFIN_IDS)})")

# ------------------------------------------------------------------
# Phase 2. Sample filtering + split
# ------------------------------------------------------------------
print("\n[Phase 2] Sample filtering + split")
init_n = len(df)
if "eligible_for_stage2" in df.columns:
    df = df[df["eligible_for_stage2"] == True].copy()
df = df[~df[FIN_IDS].isna().all(axis=1)].copy()
df["split_stage4"] = df["year"].apply(
    lambda y: "dev" if DEV_YEARS[0] <= y <= DEV_YEARS[1]
    else "oot" if OOT_YEARS[0] <= y <= OOT_YEARS[1]
    else "out_of_scope"
)
dev_mask = df["split_stage4"] == "dev"
oot_mask = df["split_stage4"] == "oot"
print(f"  Before: {init_n:,} -> After: {len(df):,}")
print(f"  Dev: {int(dev_mask.sum()):,} | OOT: {int(oot_mask.sum()):,}")

# ------------------------------------------------------------------
# Phase 3. Train LightGBM regressor-to-notch
# ------------------------------------------------------------------
print("\n[Phase 3] Fit tree-boosting regressor-to-notch on Dev")
df["grade_folded"] = df["grade_base"].map(fold_grade)
df["grade_folded_num"] = df["grade_folded"].map(GRADE2NUM)
train_mask = dev_mask & df["grade_folded_num"].notna()
X_train = df.loc[train_mask, SEL_IDS]
y_train = df.loc[train_mask, "grade_folded_num"].astype(float)
print(f"  Fit sample: {len(X_train):,}")

tree_params = {
    "loss": "huber",
    "n_estimators": int(MODEL_CFG.get("n_estimators", 120)),
    "learning_rate": float(MODEL_CFG.get("learning_rate", 0.04)),
    "max_depth": int(MODEL_CFG.get("max_depth", 3)),
    "min_samples_leaf": int(MODEL_CFG.get("min_samples_leaf", 30)),
    "subsample": float(MODEL_CFG.get("subsample", 0.85)),
    "random_state": RANDOM_STATE,
}

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    # Tree models do not require scaling. StandardScaler is intentionally omitted.
    ("tree_boosting", GradientBoostingRegressor(**tree_params)),
])
model.fit(X_train, y_train)

# ------------------------------------------------------------------
# Phase 4. Predict and score
# ------------------------------------------------------------------
print("\n[Phase 4] Predict full panel")
pred = model.predict(df[SEL_IDS])
pred = np.clip(pred, 1, 10)

df["predicted_rating_num_gamma"] = pred
df["R_score_gamma"] = score_from_predicted_notch(pred)
# Gamma is a direct rating-number regressor. Do NOT re-map its score via
# median score boundaries; use the rounded/clipped predicted rating number.
# The boundary remap artificially inflated AAA/D tails and understated Gamma.
gamma_pred_num = np.rint(pred).clip(1, 10).astype(int)
df["R_grade_gamma_num"] = pd.Series(gamma_pred_num, index=df.index).astype("Int64").values
df["R_grade_gamma"] = pd.Series(gamma_pred_num, index=df.index).map(NUM2GRADE)
df["R_PD_gamma"] = df["R_grade_gamma"].map(PD_MAP)
boundaries_gamma = {}
grade_assignment_rule_gamma = "rounded_clipped_predicted_rating_num"
for vid in SEL_IDS:
    df[f"{vid}_imputed"] = df[vid].isna()

# Feature importance.
# GradientBoostingRegressor exposes impurity-based feature importances.
importance_df = pd.DataFrame({
    "variable": SEL_IDS,
    "importance_gain": getattr(model.named_steps["tree_boosting"], "feature_importances_", np.zeros(len(SEL_IDS))),
    "note": "sklearn GradientBoostingRegressor impurity-based feature importance",
}).sort_values("importance_gain", ascending=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance_gamma_ml.csv", index=False, encoding="utf-8-sig")

# ------------------------------------------------------------------
# Phase 5. Diagnostics
# ------------------------------------------------------------------
print("\n[Phase 5] Diagnostics")
metrics_rows = []
for split in ["dev", "oot"]:
    m = calc_metrics(df[df["split_stage4"] == split])
    m.update({"specification": "gamma_ml_tree_boosting_regressor_to_notch", "split": split})
    metrics_rows.append(m)
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_DIR / "preliminary_dev_oot_metrics_gamma.csv", index=False)
print(metrics_df.to_string(index=False))

for split in ["dev", "oot"]:
    valid = df[(df["split_stage4"] == split) & df["grade_base"].notna() & df["R_grade_gamma"].notna()].copy()
    if valid.empty:
        continue
    valid["grade_obs_folded"] = valid["grade_base_10"].map(fold_grade) if "grade_base_10" in valid.columns else valid["grade_base"].map(fold_grade)
    labels = [g for g in MODELED_GRADES if g in set(valid["grade_obs_folded"]) | set(valid["R_grade_gamma"])]
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for o, p in zip(valid["grade_obs_folded"], valid["R_grade_gamma"]):
        if o in cm.index and p in cm.columns:
            cm.loc[o, p] += 1
    cm.to_csv(OUTPUT_DIR / f"confusion_matrix_gamma_{split}.csv", encoding="utf-8-sig")

# Cross-oracle Alpha/Gamma comparison.
cross_results = []
alpha_out_path = INPUT_DIR / "stage4_alpha" / "oracle_firm_year_output_alpha.parquet"
if alpha_out_path.exists():
    print("\n  Cross-oracle α/γ-ML comparison")
    alpha = pd.read_parquet(alpha_out_path)
    needed = [KEY, "year", "R_score_alpha", "R_grade_alpha_num"]
    merged = df[[KEY, "year", "split_stage4", "R_score_gamma", "R_grade_gamma_num"]].merge(
        alpha[[c for c in needed if c in alpha.columns]], on=[KEY, "year"], how="inner"
    )
    for split in ["dev", "oot"]:
        sub = merged[merged["split_stage4"] == split].copy()
        if sub.empty:
            continue
        rho = float(spearmanr(sub["R_score_alpha"], sub["R_score_gamma"])[0])
        grade_exact = float((sub["R_grade_alpha_num"] == sub["R_grade_gamma_num"]).mean())
        within1 = float((np.abs(sub["R_grade_alpha_num"] - sub["R_grade_gamma_num"]) <= 1).mean())
        in_band = CROSS_BAND_LOW <= abs(rho) <= CROSS_BAND_HIGH
        cross_results.append({
            "split": split,
            "n": int(len(sub)),
            "spearman_alpha_gamma_ml": round(rho, 4),
            "in_operating_band": bool(in_band),
            "grade_exact_agree": round(grade_exact, 4),
            "within1_agree": round(within1, 4),
        })
        print(f"  [{split}] alpha/gamma-ML rho={rho:.4f}; grade agree={grade_exact:.1%}; within1={within1:.1%}")
    pd.DataFrame(cross_results).to_csv(OUTPUT_DIR / "cross_oracle_alpha_gamma_comparison.csv", index=False)
else:
    print("  Alpha output not found; cross-oracle comparison skipped")

# Acceptance-style diagnostics, not hard gates.
def psi_from_dev_oot(score_col: str) -> float:
    dev_scores = df.loc[dev_mask, score_col].dropna()
    oot_scores = df.loc[oot_mask, score_col].dropna()
    if len(dev_scores) < 10 or len(oot_scores) < 10:
        return np.nan
    bins = np.percentile(dev_scores, np.linspace(0, 100, 11))
    bins = np.unique(bins)
    if len(bins) < 3:
        return np.nan
    bins[0] -= 1e-6
    bins[-1] += 1e-6
    dev_h, _ = np.histogram(dev_scores, bins=bins)
    oot_h, _ = np.histogram(oot_scores, bins=bins)
    eps = 1e-9
    e = dev_h / max(dev_h.sum(), 1) + eps
    a = oot_h / max(oot_h.sum(), 1) + eps
    return float(np.sum((a - e) * np.log(a / e)))

gates = {}
for split, mask in [("dev", dev_mask), ("oot", oot_mask)]:
    sub = df[mask]
    rho = float(spearmanr(sub["R_score_gamma"], sub["rating_num"])[0])
    obs_gate = pd.to_numeric(sub["rating_num_10"], errors="coerce") if "rating_num_10" in sub.columns else sub["rating_num"]
    within1 = float((np.abs(obs_gate - sub["R_grade_gamma_num"]) <= 1).mean())
    gates[f"spearman_{split}"] = {"value": round(abs(rho), 4), "pass": bool(abs(rho) >= 0.45)}
    gates[f"within1_{split}"] = {"value": round(within1, 4), "pass": bool(within1 >= 0.70)}
gates["oot_psi"] = {"value": round(psi_from_dev_oot("R_score_gamma"), 4), "pass": bool(psi_from_dev_oot("R_score_gamma") < 0.30)}
pd.DataFrame([{"gate": k, **v} for k, v in gates.items()]).to_csv(OUTPUT_DIR / "acceptance_gamma.csv", index=False)

# ------------------------------------------------------------------
# Phase 6. Save outputs
# ------------------------------------------------------------------
base_cols = [KEY]
for c in ["회사명", "시장", "sector_7"]:
    if c in df.columns:
        base_cols.append(c)
base_cols += ["year", "split_stage4", "grade_base", "rating_num", "grade_base_10", "rating_num_10", "grade_base_7", "rating_num_7", "grade_base_notch", "rating_num_notch", "grade_folded", "grade_folded_num"]
out_cols = base_cols + SEL_IDS + [f"{v}_imputed" for v in SEL_IDS] + [
    "predicted_rating_num_gamma",
    "R_score_gamma",
    "R_grade_gamma",
    "R_grade_gamma_num",
    "R_PD_gamma",
]
out_cols = [c for c in out_cols if c in df.columns]
out = df[out_cols].copy()
out["specification"] = "gamma_ml_tree_boosting_regressor_to_notch"
out.to_parquet(OUTPUT_DIR / "benchmark_firm_year_output_gamma.parquet", index=False)
out.to_csv(OUTPUT_DIR / "benchmark_firm_year_output_gamma.csv", index=False, encoding="utf-8-sig")

# Persist fitted model for Final Stage6 robustness scoring.
joblib.dump(model, OUTPUT_DIR / "benchmark_gamma_model.joblib")

params = {
    "model_name": "Gamma-ML Tree-Boosting Regressor-to-Notch Rating Benchmark",
    "version": "stage4_gamma_ml_tree_boosting_regressor_to_notch_v1.0",
    "role": "nonlinear ML rating-prediction robustness backend",
    "evaluation_note": "Final Stage6 is the simulator-mediated multi-oracle evaluation engine; Gamma-ML is only a scoring backend.",
    "split_policy": {"dev": f"{DEV_YEARS[0]}-{DEV_YEARS[1]}", "oot": f"{OOT_YEARS[0]}-{OOT_YEARS[1]}"},
    "sample_size": {"total": int(len(df)), "dev": int(dev_mask.sum()), "oot": int(oot_mask.sum())},
    "selected_variables": SEL_IDS,
    "grade_order": GRADE_ORDER,
    "modeled_grades": MODELED_GRADES,
    "modeled_grade_nums": MODELED_NUMS,
    "final_boundaries": boundaries_gamma,
    "fallback_grade": "D",
    "score_definition": "R_score_gamma = 100 * (10 - predicted_rating_num_gamma) / 9; R_grade_gamma is assigned by rounded/clipped predicted rating number, not by score-boundary remapping",
    "grade_assignment_rule": grade_assignment_rule_gamma,
    "tree_boosting_params": tree_params,
    "model_artifact": "benchmark_gamma_model.joblib",
    "feature_importance": importance_df.to_dict(orient="records"),
    "metrics": metrics_rows,
    "cross_oracle_alpha_gamma_ml": cross_results,
    "acceptance_gates": gates,
}
with open(OUTPUT_DIR / "benchmark_gamma_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame([{"filename": p.name, "size": p.stat().st_size} for p in sorted(OUTPUT_DIR.glob("*")) if p.is_file()]).to_csv(
    OUTPUT_DIR / "stage4_gamma_file_manifest.csv", index=False
)

print("\n" + "=" * 72)
print("Stage 4 γ Gamma-ML Complete")
print("=" * 72)
print(f"  Outputs: {OUTPUT_DIR}")
