#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Stage 4 β — Ordered Logistic Rating Benchmark
==========================================================================
Role:
  Beta is an empirical ordinal rating benchmark. It is intentionally not a
  second scorecard. It learns external credit ratings directly with an
  Ordered Logistic model using the same 11 variables as Alpha.

Interpretation:
  Alpha = frozen scorecard-style reference evaluator.
  Beta  = linear/statistical ordinal rating predictor robustness backend.

Outputs:
  - R_score_beta: 0-100 credit-quality score, higher is better
  - R_grade_beta / R_grade_beta_num / R_PD_beta
  - expected_rating_num_beta
  - ordered-logit probabilities by modeled grade
  - diagnostics and alpha/beta cross-oracle comparison
==========================================================================
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from credit_recourse.oracle.contracts.rating_scale import (
    GRADE_ORDER_10, GRADE2NUM_10, NUM2GRADE_10, PD_MAP_10,
    add_rating_scale_columns, fold_to_10, assign_grade_10, ensure_10_grade_contract, assert_grade_order_10,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
except ImportError as exc:  # pragma: no cover
    raise ImportError("statsmodels>=0.14 is required. Run: pip install statsmodels") from exc

SCRIPT_DIR = Path(__file__).parent
# Project root: <project>/src/credit_recourse/oracle/pipelines/stage01_oracle_construction/<backend>
PROJECT_ROOT = SCRIPT_DIR.parents[5]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "final_freeze" / "stage1_oracle_backends" / "beta"
INPUT_DIR = Path(os.environ.get("ORACLE_INPUT_DIR", SCRIPT_DIR / "inputs"))
OUTPUT_DIR = Path(os.environ.get("ORACLE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
CONFIG_PATH = Path(os.environ.get("ORACLE_CONFIG", SCRIPT_DIR / "configs" / "stage4_beta_config.yaml"))
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
CROSS_BAND_LOW = float(CONFIG.get("cross_oracle", {}).get("spearman_band_low", 0.55))
CROSS_BAND_HIGH = float(CONFIG.get("cross_oracle", {}).get("spearman_band_high", 0.90))
DEV_YEARS, OOT_YEARS = split_years_from_config(CONFIG)

MODELED_NUMS = [GRADE2NUM[g] for g in MODELED_GRADES]
MIN_MODELED_NUM = min(MODELED_NUMS)
MAX_MODELED_NUM = max(MODELED_NUMS)


def fold_grade(g: object) -> object:
    folded = fold_to_10(g)
    return np.nan if pd.isna(folded) else folded


def score_from_expected_rating(expected_rating_num: np.ndarray) -> np.ndarray:
    clipped = np.clip(expected_rating_num, 1, 10)
    return 100.0 * (10.0 - clipped) / 9.0


def nearest_modeled_grade_num(x: np.ndarray) -> np.ndarray:
    arr = np.rint(np.clip(x, MIN_MODELED_NUM, MAX_MODELED_NUM)).astype(int)
    valid = np.array(MODELED_NUMS)
    out = []
    for v in arr:
        out.append(int(valid[np.argmin(np.abs(valid - v))]))
    return np.array(out, dtype=int)


def calc_metrics(sub: pd.DataFrame, score_col: str, pred_num_col: str, pred_grade_col: str) -> dict:
    valid = sub[sub["rating_num"].notna() & sub[pred_num_col].notna()].copy()
    if valid.empty:
        return {"n": 0, "exact": np.nan, "within1": np.nan, "mae": np.nan, "rho": np.nan, "bias": np.nan}
    obs = pd.to_numeric(valid["rating_num_10"], errors="coerce") if "rating_num_10" in valid.columns else valid["rating_num"].astype(float)
    pred = valid[pred_num_col].astype(float)
    actual_grade = valid["grade_base_10"].map(fold_grade) if "grade_base_10" in valid.columns else valid["grade_base"].map(fold_grade)
    exact = float((actual_grade == valid[pred_grade_col]).mean())
    within1 = float((np.abs(obs - pred) <= 1).mean())
    mae = float(np.abs(obs - pred).mean())
    rho = float(spearmanr(valid[score_col], obs)[0])
    bias = float((pred - obs).mean())
    return {
        "n": int(len(valid)),
        "exact": round(exact, 4),
        "within1": round(within1, 4),
        "mae": round(mae, 3),
        "rho": round(rho, 4),
        "rho_abs": round(abs(rho), 4),
        "bias_pred_minus_actual_notch": round(bias, 4),
    }


print("=" * 72)
print("Stage 4 β — Ordered Logistic Rating Benchmark")
print("=" * 72)
print(f"  Modeled grades: {MODELED_GRADES}")
print(f"  Dev years: {DEV_YEARS[0]}-{DEV_YEARS[1]} | OOT years: {OOT_YEARS[0]}-{OOT_YEARS[1]}")

# ------------------------------------------------------------------
# Phase 1. Load inputs
# ------------------------------------------------------------------
print("\n[Phase 1] Loading inputs")
with open(INPUT_DIR / "stage3_v2" / "selected_variables_v2.json", encoding="utf-8") as f:
    sel_vars = json.load(f)
with open(INPUT_DIR / "stage3_v2" / "direction_encoding_v2.json", encoding="utf-8") as f:
    _dirs = json.load(f)

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
# Phase 2. Sample filter + split
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
# Phase 3. Standardization
# ------------------------------------------------------------------
print("\n[Phase 3] Mean imputation + z-score standardization")
train_mask = dev_mask & df["grade_base"].notna()
dev_means = df.loc[dev_mask, SEL_IDS].mean(numeric_only=True)
X_imp = df[SEL_IDS].fillna(dev_means)
scaler = StandardScaler()
scaler.fit(X_imp.loc[dev_mask])
X_z_all = pd.DataFrame(scaler.transform(X_imp), index=df.index, columns=SEL_IDS)
for vid in SEL_IDS:
    df[f"{vid}_imputed"] = df[vid].isna()
standardization_params = {
    vid: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
    for i, vid in enumerate(SEL_IDS)
}

# ------------------------------------------------------------------
# Phase 4. Ordered Logistic fit
# ------------------------------------------------------------------
print("\n[Phase 4] Fit Ordered Logistic on Dev")
df["grade_folded"] = df["grade_base"].map(fold_grade)
df["grade_folded_num"] = df["grade_folded"].map(GRADE2NUM)
train_idx = df.index[train_mask & df["grade_folded_num"].notna()]
X_train = X_z_all.loc[train_idx]
y_train_num = df.loc[train_idx, "grade_folded_num"].astype(int).values
ordered_cats = sorted(np.unique(y_train_num).tolist())
# Actual posterior classes fitted by statsmodels. These are the grades that
# result.predict(...) will emit probability columns for. Do not confuse this
# with CONFIG["model"]["modeled_grades"], which is treated as a requested/
# configured scale hint only. Sparse grades can disappear, while C/D can still
# be present if observed in Dev.
FITTED_GRADE_NUMS = [int(c) for c in ordered_cats]
FITTED_GRADES = [NUM2GRADE[int(c)] for c in FITTED_GRADE_NUMS]
y_train = pd.Series(pd.Categorical(y_train_num, categories=ordered_cats, ordered=True))
print(f"  Fit sample: {len(train_idx):,}; categories={ordered_cats} ({FITTED_GRADES})")

model = OrderedModel(y_train, X_train.reset_index(drop=True), distr="logit")
result = model.fit(method="bfgs", disp=False, maxiter=500)
print(f"  Converged: {result.mle_retvals.get('converged', 'n/a')}")
print(f"  Log-likelihood: {result.llf:.2f} | AIC: {result.aic:.2f}")

param_names = [str(x) for x in getattr(result.params, "index", result.model.exog_names)]
coef_df = pd.DataFrame({
    "variable": param_names,
    "coefficient": np.asarray(result.params.values, dtype=float),
    "std_err": np.asarray(result.bse.values, dtype=float),
    "z_stat": np.asarray(result.tvalues.values, dtype=float),
    "p_value": np.asarray(result.pvalues.values, dtype=float),
})
threshold_names = [n for n in param_names if n not in set(SEL_IDS)]
threshold_raw_params = [float(result.params[n]) for n in threshold_names]
try:
    threshold_all = result.model.transform_threshold_params(np.asarray(result.params.values, dtype=float))
    finite_cutpoints = [float(x) for x in threshold_all[1:-1]]
except Exception:
    raw = np.asarray(threshold_raw_params, dtype=float)
    finite_cutpoints = [float(x) for x in np.cumsum(np.concatenate([raw[:1], np.exp(raw[1:])]))]
coef_df.to_csv(OUTPUT_DIR / "ordered_logit_coefficients_beta.csv", index=False)
with open(OUTPUT_DIR / "ordered_logit_summary_beta.txt", "w", encoding="utf-8") as f:
    f.write(str(result.summary()))

# ------------------------------------------------------------------
# Phase 5. Predict
# ------------------------------------------------------------------
print("\n[Phase 5] Predict full panel")
probs = result.predict(X_z_all.reset_index(drop=True))
PROB_GRADE_COLUMNS = [f"prob_beta_grade_{NUM2GRADE[int(c)]}" for c in ordered_cats]
probs = pd.DataFrame(probs.values, index=df.index, columns=PROB_GRADE_COLUMNS)
prob_values = probs.values
cat_values = np.array(ordered_cats, dtype=float)
expected_rating = (prob_values * cat_values.reshape(1, -1)).sum(axis=1)
pred_class = np.argmax(prob_values, axis=1)
pred_num = np.array([int(ordered_cats[i]) for i in pred_class], dtype=int)

# Main continuous score is expected-notch based, 0-100, higher better.
# IMPORTANT: Beta is already an ordinal classifier. Do NOT re-map its
# expected-rating score through scorecard-style median score boundaries.
# That double-calibration creates artificial AAA/D tails and was the
# source of the poor Beta within-one/bias diagnostics.
df["expected_rating_num_beta"] = expected_rating
df["R_score_beta"] = score_from_expected_rating(expected_rating)
df["R_grade_beta_num"] = pd.Series(pred_num, index=df.index).astype("Int64").values
df["R_grade_beta"] = pd.Series(pred_num, index=df.index).map(NUM2GRADE)
df["R_PD_beta"] = df["R_grade_beta"].map(PD_MAP)
boundaries_beta = {}
grade_assignment_rule_beta = "posterior_MAP_class_from_ordered_logit_probabilities"
df = pd.concat([df, probs], axis=1)

# ------------------------------------------------------------------
# Phase 6. Diagnostics
# ------------------------------------------------------------------
print("\n[Phase 6] Diagnostics")
metrics_rows = []
for split in ["dev", "oot"]:
    m = calc_metrics(df[df["split_stage4"] == split], "R_score_beta", "R_grade_beta_num", "R_grade_beta")
    m.update({"specification": "beta_ordered_logit", "split": split})
    metrics_rows.append(m)
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUTPUT_DIR / "preliminary_dev_oot_metrics_beta.csv", index=False)
print(metrics_df.to_string(index=False))

# Confusion matrices.
for split in ["dev", "oot"]:
    valid = df[(df["split_stage4"] == split) & df["grade_base"].notna() & df["R_grade_beta"].notna()].copy()
    if valid.empty:
        continue
    valid["grade_obs_folded"] = valid["grade_base_10"].map(fold_grade) if "grade_base_10" in valid.columns else valid["grade_base"].map(fold_grade)
    # Use the actual observed/predicted output grades, not the configured
    # MODELED_GRADES metadata. Otherwise sparse but real C/D predictions are
    # silently omitted from the confusion matrix.
    present_grades = set(valid["grade_obs_folded"].dropna()) | set(valid["R_grade_beta"].dropna())
    labels = [g for g in GRADE_ORDER if g in present_grades]
    cm = pd.DataFrame(0, index=labels, columns=labels)
    for o, p in zip(valid["grade_obs_folded"], valid["R_grade_beta"]):
        if o in cm.index and p in cm.columns:
            cm.loc[o, p] += 1
    cm.to_csv(OUTPUT_DIR / f"confusion_matrix_beta_{split}.csv", encoding="utf-8-sig")

# Cross-oracle Alpha/Beta comparison.
cross_results = []
alpha_out_path = INPUT_DIR / "stage4_alpha" / "oracle_firm_year_output_alpha.parquet"
if alpha_out_path.exists():
    print("\n  Cross-oracle α/β comparison")
    alpha = pd.read_parquet(alpha_out_path)
    need_cols = [KEY, "year", "R_score_alpha", "R_grade_alpha_num"]
    merged = df[[KEY, "year", "split_stage4", "R_score_beta", "R_grade_beta_num"]].merge(
        alpha[[c for c in need_cols if c in alpha.columns]], on=[KEY, "year"], how="inner"
    )
    for split in ["dev", "oot"]:
        sub = merged[merged["split_stage4"] == split].copy()
        if sub.empty:
            continue
        rho = float(spearmanr(sub["R_score_alpha"], sub["R_score_beta"])[0])
        grade_exact = float((sub["R_grade_alpha_num"] == sub["R_grade_beta_num"]).mean())
        within1 = float((np.abs(sub["R_grade_alpha_num"] - sub["R_grade_beta_num"]) <= 1).mean())
        in_band = CROSS_BAND_LOW <= abs(rho) <= CROSS_BAND_HIGH
        cross_results.append({
            "split": split,
            "n": int(len(sub)),
            "spearman_alpha_beta": round(rho, 4),
            "in_operating_band": bool(in_band),
            "grade_exact_agree": round(grade_exact, 4),
            "within1_agree": round(within1, 4),
        })
        print(f"  [{split}] alpha/beta rho={rho:.4f}; grade agree={grade_exact:.1%}; within1={within1:.1%}")
    pd.DataFrame(cross_results).to_csv(OUTPUT_DIR / "cross_oracle_alpha_beta_comparison.csv", index=False)
else:
    print("  Alpha output not found; cross-oracle comparison skipped")

# Acceptance-style diagnostics, not hard gates for thesis main pipeline.
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
    rho = float(spearmanr(sub["R_score_beta"], sub["rating_num"])[0])
    obs_gate = pd.to_numeric(sub["rating_num_10"], errors="coerce") if "rating_num_10" in sub.columns else sub["rating_num"]
    within1 = float((np.abs(obs_gate - sub["R_grade_beta_num"]) <= 1).mean())
    gates[f"spearman_{split}"] = {"value": round(abs(rho), 4), "pass": bool(abs(rho) >= 0.45)}
    gates[f"within1_{split}"] = {"value": round(within1, 4), "pass": bool(within1 >= 0.70)}
gates["oot_psi"] = {"value": round(psi_from_dev_oot("R_score_beta"), 4), "pass": bool(psi_from_dev_oot("R_score_beta") < 0.30)}
pd.DataFrame([{"gate": k, **v} for k, v in gates.items()]).to_csv(OUTPUT_DIR / "acceptance_beta.csv", index=False)

# ------------------------------------------------------------------
# Phase 7. Save outputs
# ------------------------------------------------------------------
base_cols = [KEY]
for c in ["회사명", "시장", "sector_7"]:
    if c in df.columns:
        base_cols.append(c)
base_cols += ["year", "split_stage4", "grade_base", "rating_num", "grade_base_10", "rating_num_10", "grade_base_7", "rating_num_7", "grade_base_notch", "rating_num_notch", "grade_folded", "grade_folded_num"]
out_cols = base_cols + SEL_IDS + [f"{v}_imputed" for v in SEL_IDS] + list(probs.columns) + [
    "expected_rating_num_beta",
    "R_score_beta",
    "R_grade_beta",
    "R_grade_beta_num",
    "R_PD_beta",
]
out_cols = [c for c in out_cols if c in df.columns]
out = df[out_cols].copy()
out["specification"] = "beta_ordered_logit"
out.to_parquet(OUTPUT_DIR / "benchmark_firm_year_output_beta.parquet", index=False)
out.to_csv(OUTPUT_DIR / "benchmark_firm_year_output_beta.csv", index=False, encoding="utf-8-sig")

params = {
    "model_name": "Beta Ordered Logistic Rating Benchmark",
    "version": "stage4_beta_ordered_logit_v1.0",
    "role": "linear/statistical ordinal rating-prediction robustness backend",
    "split_policy": {"dev": f"{DEV_YEARS[0]}-{DEV_YEARS[1]}", "oot": f"{OOT_YEARS[0]}-{OOT_YEARS[1]}"},
    "sample_size": {"total": int(len(df)), "dev": int(dev_mask.sum()), "oot": int(oot_mask.sum())},
    "selected_variables": SEL_IDS,
    "grade_order": GRADE_ORDER,
    # Configured grades are retained only as provenance. The authoritative
    # modeled/probability classes are the actual fitted posterior classes
    # returned by statsmodels and written as prob_beta_grade_* columns.
    "configured_modeled_grades": MODELED_GRADES,
    "configured_modeled_grade_nums": MODELED_NUMS,
    "modeled_grades": FITTED_GRADES,
    "modeled_grade_nums": FITTED_GRADE_NUMS,
    "probability_output_grades": FITTED_GRADES,
    "probability_output_grade_nums": FITTED_GRADE_NUMS,
    "probability_output_columns": PROB_GRADE_COLUMNS,
    "ordered_logit_threshold_names": threshold_names,
    "ordered_logit_threshold_raw_params": threshold_raw_params,
    "ordered_logit_finite_cutpoints": finite_cutpoints,
    "predicted_output_grades": [g for g in GRADE_ORDER if g in set(df["R_grade_beta"].dropna())],
    "metadata_consistency_note": "modeled_grades equals the fitted posterior probability classes in prob_beta_grade_* columns; configured_model_grades is retained separately as provenance.",
    "final_boundaries": boundaries_beta,
    "fallback_grade": "D",
    "score_definition": "R_score_beta = 100 * (10 - E[rating_num_10]) / 9; R_grade_beta is assigned by ordered-logit posterior MAP class, not by score-boundary remapping",
    "grade_assignment_rule": grade_assignment_rule_beta,
    "standardization_params": standardization_params,
    "fit": {
        "log_likelihood": round(float(result.llf), 4),
        "aic": round(float(result.aic), 4),
        "bic": round(float(result.bic), 4),
        "converged": bool(result.mle_retvals.get("converged")),
    },
    "coefficients": coef_df.to_dict(orient="records"),
    "metrics": metrics_rows,
    "cross_oracle_alpha_beta": cross_results,
    "acceptance_gates": gates,
}
with open(OUTPUT_DIR / "benchmark_beta_params.json", "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2, ensure_ascii=False, default=str)

pd.DataFrame([{"filename": p.name, "size": p.stat().st_size} for p in sorted(OUTPUT_DIR.glob("*")) if p.is_file()]).to_csv(
    OUTPUT_DIR / "stage4_beta_file_manifest.csv", index=False
)

print("\n" + "=" * 72)
print("Stage 4 β Ordered Logistic Complete")
print("=" * 72)
print(f"  Outputs: {OUTPUT_DIR}")
