#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
==========================================================================
Stage 4 α — Performance-Optimized Scorecard Oracle (Reference Rating Oracle)
==========================================================================
Spec: Isotonic + regularized joint weight-boundary KL optimization
Source of truth: Stage 3 v2 selected_variables_v2.json + direction_encoding_v2.json
Sample: Dev 2002-2019 / OOT 2020-2024 (Oracle Methodology canonical)

Pipeline phases:
  Phase 1.  Load inputs + verify (Stage 1B / 2 / 1C v3.2 / 3 v2)
  Phase 2.  Sample filtering (eligible_for_stage2 + non-all-fin-missing)
  Phase 3.  Split (Dev 2002-2019 / OOT 2020-2024)
  Phase 4.  Winsorization (Dev P1/P99)
  Phase 5.  Binning (50 equal-width / reduced-unique / binary)
  Phase 6.  Isotonic bin-to-score (direction-aware from Stage 3)
  Phase 7.  Item scoring + missing imputation
  Phase 8.  Stage 3 prior weights (cap/floor + donor pool)
  Phase 9.  Dev inner CV split (train 2002-2016 / val 2017-2019, anchor-stratified)
  Phase 10. Hyperparameter grid search on Dev val
  Phase 11. Joint optimization (final) with selected hyperparameters on full Dev
  Phase 12. Weight sanity check
  Phase 13. PD alignment + final outputs (alpha)
  Phase 14. Diagnostics (preliminary metrics, confusion, boundary jump)

OOT is reserved for final reporting only — never used for tuning.

Output: outputs/oracle_alpha_*.{parquet,csv,json} + diagnostic CSVs.
Acceptance check: separate `verify_alpha_metrics.py`.
==========================================================================
"""
import json
import os
import warnings
from pathlib import Path
import sys

# [HOTFIX alpha_modules_import_v1]
# pipeline.py is executed by runpy.run_path() from a wrapper, so the
# alpha backend directory is not always on sys.path. Keep local
# imports such as `from modules.weight_sanity import ...` working.
ALPHA_BACKEND_DIR = Path(__file__).resolve().parent
if str(ALPHA_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(ALPHA_BACKEND_DIR))
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, cohen_kappa_score

from credit_recourse.oracle.contracts.rating_scale import (
    GRADE_ORDER_10, GRADE2NUM_10, NUM2GRADE_10, PD_MAP_10,
    add_rating_scale_columns, assign_grade_10, ensure_10_grade_contract, assert_grade_order_10,
)

warnings.filterwarnings('ignore')

# ============================================================
# Constants
# ============================================================
SCRIPT_DIR  = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parents[5]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "final_freeze" / "stage1_oracle_backends" / "alpha"
INPUT_DIR = Path(os.environ.get("ORACLE_INPUT_DIR", SCRIPT_DIR / "inputs"))
OUTPUT_DIR = Path(os.environ.get("ORACLE_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))
CONFIG_PATH = Path(os.environ.get("ORACLE_CONFIG", SCRIPT_DIR / "configs" / "stage4_alpha_config.yaml"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRADE_ORDER = GRADE_ORDER_10
GRADE2NUM   = GRADE2NUM_10
PD_MAP = PD_MAP_10
assert_grade_order_10(GRADE_ORDER)
KEY = '거래소코드'
RNG_SEED = 42


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
    """Resolve Oracle split policy with environment overrides.

    Final thesis default is Dev 2002-2019 / OOT 2020-score_end_year.
    This function makes the policy explicit and reproducible instead of
    silently hard-coding it in multiple places.
    """
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
    if len(dev) != 2 or len(oot) != 2:
        raise ValueError(f"Invalid split config: dev={dev}, oot={oot}")
    if int(dev[1]) >= int(oot[0]):
        raise ValueError(f"Dev/OOT split must be non-overlapping: dev={dev}, oot={oot}")
    if int(oot[1]) < int(oot[0]):
        raise ValueError(f"OOT split is empty after score-end-year scoping: oot={oot}, ORACLE_MAX_YEAR={ORACLE_MAX_YEAR}")
    return [int(dev[0]), int(dev[1])], [int(oot[0]), int(oot[1])]


# ============================================================
# Load config
# ============================================================
with open(CONFIG_PATH, encoding='utf-8') as f:
    CONFIG = yaml.safe_load(f)

CFG_OPT = CONFIG['optimization']
CFG_GRID = CONFIG['hyperparameter_grid']
CFG_CV = CONFIG['inner_cv']
DEV_YEARS, OOT_YEARS = split_years_from_config(CONFIG)

# Default hyperparameters (Oracle Methodology §6, §8)
DEFAULT_ALPHA_EM = float(CFG_OPT.get('default_alpha_em', 1.5))
DEFAULT_ALPHA_W1 = float(CFG_OPT.get('default_alpha_w1', 0.5))
DEFAULT_DELTA_MAE = float(CFG_OPT.get('default_delta_mae', 0.5))
DEFAULT_BETA_KL = float(CFG_OPT.get('default_beta_kl', 0.8))
DEFAULT_GAMMA_REG = float(CFG_OPT.get('default_gamma_reg', 0.3))
W_FLOOR = float(CFG_OPT.get('w_floor', 0.05))
W_CAP = float(CFG_OPT.get('w_cap', 0.30))
N_OUTER_ITER = int(CFG_OPT.get('n_outer_iter', 5))
N_TRIALS = int(CFG_OPT.get('n_weight_trials', 500))
LOW_UNIQUE_THRESHOLD = int(CONFIG['scoring'].get('low_unique_threshold', 20))

# Alpha finalization controls. These are deliberately conservative:
# - exact-boost retunes only grade boundaries on the fixed Alpha R_score, using Dev only.
# - sparse-tail guard is opt-in; by default Alpha emits the full 10-grade scale.
CFG_FINAL = CONFIG.get('finalization', {}) if isinstance(CONFIG, dict) else {}
ALPHA_EXACT_BOOST = str(os.environ.get(
    'ORACLE_ALPHA_EXACT_BOOST', CFG_FINAL.get('exact_boost_boundaries', True)
)).strip().lower() not in {'0', 'false', 'no', 'off'}
ALPHA_SPARSE_TAIL_GUARD = str(os.environ.get(
    'ORACLE_ALPHA_SPARSE_TAIL_GUARD', CFG_FINAL.get('sparse_tail_guard', False)
)).strip().lower() not in {'0', 'false', 'no', 'off'}
ALPHA_SPARSE_TAIL_MIN_DEV_N = int(os.environ.get(
    'ORACLE_ALPHA_SPARSE_TAIL_MIN_DEV_N', CFG_FINAL.get('sparse_tail_min_dev_n', 20)
))
ALPHA_SPARSE_TAIL_FLOOR_GRADE = str(os.environ.get(
    'ORACLE_ALPHA_SPARSE_TAIL_FLOOR_GRADE', CFG_FINAL.get('sparse_tail_floor_grade', 'CCC')
)).strip().upper()

print("=" * 72)
print("Stage 4 α — Performance-Optimized Scorecard Oracle")
print("=" * 72)
print(f"  Bundle dir:  {SCRIPT_DIR}")
print(f"  Defaults: α_EM={DEFAULT_ALPHA_EM}, α_W1={DEFAULT_ALPHA_W1}, "
      f"δ_MAE={DEFAULT_DELTA_MAE}, β_KL={DEFAULT_BETA_KL}, γ_REG={DEFAULT_GAMMA_REG}")
print(f"  Weight: floor={W_FLOOR}, cap={W_CAP}")

# ============================================================
# Phase 1: Load inputs + verification
# ============================================================
print("\n[Phase 1] Loading inputs & verification")

with open(INPUT_DIR / "stage3_v2" / "selected_variables_v2.json", encoding='utf-8') as f:
    SEL_VARS = json.load(f)
with open(INPUT_DIR / "stage3_v2" / "direction_encoding_v2.json", encoding='utf-8') as f:
    DIRS = json.load(f)

SEL_IDS = [v['variable_id'] for v in SEL_VARS]
FIN_IDS = [v['variable_id'] for v in SEL_VARS if v['source'] == 'financial']
NONFIN_IDS = [v['variable_id'] for v in SEL_VARS if v['source'] == 'nonfinancial']
SCORES_RAW = {v['variable_id']: v.get('score', v.get('selection_score', 1.0))
              for v in SEL_VARS}

assert len(SEL_IDS) in (10, 11), f"Expected 10 or 11 variables, got {len(SEL_IDS)}"
assert len(FIN_IDS) == 6,  f"Expected 6 financial, got {len(FIN_IDS)}"
assert len(NONFIN_IDS) in (4, 5), f"Expected 4 or 5 nonfinancial, got {len(NONFIN_IDS)}"
# Stage 1C v3.2 guard: kospi_dummy must NOT be selected
assert 'kospi_dummy' not in SEL_IDS, \
    "Stage 1C v3.2 violation: kospi_dummy should NEVER be selected"

print(f"  Variables: {len(SEL_IDS)} (Fin:{len(FIN_IDS)}, NF:{len(NONFIN_IDS)})")
print(f"  Financial:    {FIN_IDS}")
print(f"  Nonfinancial: {NONFIN_IDS}")
for vid in SEL_IDS:
    d = DIRS[vid]['direction']
    rho = DIRS[vid].get('spearman_rho', float('nan'))
    print(f"    {vid:45s} direction={d:18s}  ρ={rho:+.3f}")

df_base = pd.read_parquet(INPUT_DIR / "stage1b" / "firm_year_panel_v1.parquet")
df_fin = pd.read_parquet(INPUT_DIR / "stage2" / "engineered_financial_ratios.parquet")
df_nonfin = pd.read_parquet(INPUT_DIR / "stage1c_v3" / "nonfinancial_metadata_panel.parquet")

df = df_base.merge(df_fin, on=[KEY, 'year'], how='left', suffixes=('', '_f'))
df = df.merge(df_nonfin, on=[KEY, 'year'], how='left', suffixes=('', '_n'))
print(f"\n  Merged panel: {len(df):,} rows")
df = apply_oracle_year_scope(df, label='merged Alpha panel')
if 'grade_base_10' not in df.columns or 'rating_num_10' not in df.columns:
    if 'grade_base' not in df.columns and 'rating' in df.columns:
        df['grade_base'] = df['rating']
    df = add_rating_scale_columns(df, source_col='grade_base')
df['grade_base'] = df['grade_base_10']
df['rating_num'] = pd.to_numeric(df['rating_num_10'], errors='coerce')

# Verify all selected variables present
missing_vars = [v for v in SEL_IDS if v not in df.columns]
assert not missing_vars, f"Variables missing from merged panel: {missing_vars}"

# ============================================================
# Phase 2: Sample filtering
# ============================================================
print("\n[Phase 2] Sample filtering")

init_n = len(df)
excl_elig = excl_fin = 0
if 'eligible_for_stage2' in df.columns:
    mask_e = df['eligible_for_stage2'] == True
    excl_elig = int((~mask_e).sum())
    df = df[mask_e].copy()

mask_fin_miss = df[FIN_IDS].isna().all(axis=1)
excl_fin = int(mask_fin_miss.sum())
df = df[~mask_fin_miss].copy()

remain = int(df[FIN_IDS].isna().all(axis=1).sum())
assert remain == 0, f"FAIL: {remain} all-fin-missing rows remain"

pd.DataFrame([{
    'total_firm_years_before_filter': init_n,
    'excluded_by_eligibility_flag': excl_elig,
    'excluded_all_financial_missing_rows': excl_fin,
    'final_modeling_rows': len(df),
    'financial_all_missing_rows_remaining': remain,
}]).to_csv(OUTPUT_DIR / "sample_filter_report.csv", index=False, encoding='utf-8-sig')

print(f"  Before: {init_n:,} → After: {len(df):,} (excl: {init_n - len(df):,})")

# ============================================================
# Phase 3: Split
# ============================================================
print(f"\n[Phase 3] Split policy (Dev {DEV_YEARS[0]}-{DEV_YEARS[1]} / OOT {OOT_YEARS[0]}-{OOT_YEARS[1]})")

df['split_stage4'] = df['year'].apply(
    lambda y: 'dev' if DEV_YEARS[0] <= y <= DEV_YEARS[1]
    else 'oot' if OOT_YEARS[0] <= y <= OOT_YEARS[1]
    else 'out_of_scope'
)

dev_mask = df['split_stage4'] == 'dev'
oot_mask = df['split_stage4'] == 'oot'
n_dev, n_oot = int(dev_mask.sum()), int(oot_mask.sum())
print(f"  Dev: {n_dev:,}   OOT: {n_oot:,}")
if n_dev == 0 or n_oot == 0:
    raise ValueError(f"Empty split after applying policy: dev={DEV_YEARS}, oot={OOT_YEARS}, n_dev={n_dev}, n_oot={n_oot}")
assert int(df.loc[dev_mask, 'year'].max()) <= DEV_YEARS[1]
assert int(df.loc[oot_mask, 'year'].min()) >= OOT_YEARS[0]

sr = df.groupby(['split_stage4', 'year']).size().reset_index(name='n')
sr.to_csv(OUTPUT_DIR / "split_reconciliation_stage4.csv", index=False, encoding='utf-8-sig')

r_min = float(df.loc[dev_mask, 'rating_num'].min())
r_max = float(df.loc[dev_mask, 'rating_num'].max())

# ============================================================
# Phase 4: Winsorization (Dev P1/P99)
# ============================================================
print("\n[Phase 4] Winsorization (Dev P1/P99)")

winsor_records = []
for vid in SEL_IDS:
    vals = df.loc[dev_mask, vid].dropna()
    n_unique = len(vals.unique())
    is_bin = n_unique <= 2

    if is_bin:
        winsor_records.append({
            'variable_id': vid, 'is_binary': True, 'n_unique_dev': n_unique,
            'p01_dev': None, 'p99_dev': None, 'winsor_applied': False,
            'n_dev_nonmissing': len(vals), 'special_handling_flag': True
        })
        df[f'{vid}_w'] = df[vid].copy()
    else:
        p01, p99 = np.percentile(vals, [1, 99])
        winsor_records.append({
            'variable_id': vid, 'is_binary': False, 'n_unique_dev': n_unique,
            'p01_dev': float(p01), 'p99_dev': float(p99), 'winsor_applied': True,
            'n_dev_nonmissing': len(vals),
            'special_handling_flag': n_unique <= LOW_UNIQUE_THRESHOLD
        })
        df[f'{vid}_w'] = df[vid].clip(p01, p99)

pd.DataFrame(winsor_records).to_csv(OUTPUT_DIR / "winsor_params_alpha.csv", index=False)
with open(OUTPUT_DIR / "winsor_params_alpha.json", 'w') as f:
    json.dump(winsor_records, f, indent=2)

# ============================================================
# Phase 5: Binning
# ============================================================
print("\n[Phase 5] Binning")

bin_edges_all = {}
bin_diag = []
for vid in SEL_IDS:
    vals = df.loc[dev_mask, f'{vid}_w'].dropna()
    n_unique = len(vals.unique())
    is_bin = n_unique <= 2

    if is_bin:
        edges = sorted(vals.unique())
        method, n_act = 'unique_value', n_unique
    elif n_unique <= LOW_UNIQUE_THRESHOLD:
        edges = sorted(vals.unique())
        method, n_act = 'reduced_unique_value', n_unique
    else:
        lo, hi = vals.min(), vals.max()
        edges = np.linspace(lo, hi, 51)
        method, n_act = 'equal_width_50', 50

    bin_edges_all[vid] = {
        'edges': [float(e) for e in edges], 'method': method,
        'n_bins': n_act, 'is_binary': is_bin,
        'is_low_unique': n_unique <= LOW_UNIQUE_THRESHOLD and not is_bin
    }

    if method in ('unique_value', 'reduced_unique_value'):
        vc = vals.value_counts()
        n_empty = 0
        min_bc, med_bc, max_bc = int(vc.min()), int(vc.median()), int(vc.max())
    else:
        hist, _ = np.histogram(vals, bins=edges)
        n_empty = int((hist == 0).sum())
        min_bc, med_bc, max_bc = int(hist.min()), int(np.median(hist)), int(hist.max())

    bin_diag.append({
        'variable_id': vid,
        'n_bins_requested': 50 if method == 'equal_width_50' else n_unique,
        'n_bins_actual': n_act, 'n_empty_bins_dev': n_empty,
        'min_bin_count_dev': min_bc, 'median_bin_count_dev': med_bc,
        'max_bin_count_dev': max_bc, 'n_unique_dev': n_unique,
        'special_handling_flag': is_bin or n_unique <= LOW_UNIQUE_THRESHOLD,
        'binning_method': method
    })

pd.DataFrame(bin_diag).to_csv(OUTPUT_DIR / "bin_assignment_diagnostics.csv", index=False)
with open(OUTPUT_DIR / "bin_edges_alpha.json", 'w') as f:
    json.dump(bin_edges_all, f, indent=2)

# Assign bin IDs
for vid in SEL_IDS:
    binfo = bin_edges_all[vid]
    if binfo['is_binary'] or binfo.get('is_low_unique'):
        uvals = binfo['edges']
        df[f'{vid}_bin'] = df[f'{vid}_w'].apply(
            lambda x, uv=uvals: min(range(len(uv)), key=lambda i: abs(x - uv[i]))
            if pd.notna(x) else np.nan
        )
    else:
        edges = np.array(binfo['edges'])
        df[f'{vid}_bin'] = df[f'{vid}_w'].apply(
            lambda x, e=edges: max(0, min(int(np.searchsorted(e, x, side='right')) - 1, len(e) - 2))
            if pd.notna(x) else np.nan
        )

# ============================================================
# Phase 6: Isotonic bin-to-score (direction-aware)
# ============================================================
print("\n[Phase 6] Direction-aware isotonic scoring")

iso_tables = {}
dev_df = df[dev_mask].copy()

for vid in SEL_IDS:
    direction = DIRS[vid]['direction']
    n_bins = bin_edges_all[vid]['n_bins']

    bins_data = []
    for b in range(n_bins):
        sub = dev_df[dev_df[f'{vid}_bin'] == b]
        if len(sub) > 0:
            emp = 100.0 * (r_max - sub['rating_num'].mean()) / (r_max - r_min)
            bins_data.append({'raw_bin': b, 'emp': emp, 'count': len(sub)})

    if not bins_data:
        iso_tables[vid] = {}
        continue

    bdf = pd.DataFrame(bins_data)
    bdf['gd'] = ((n_bins - 1) - bdf['raw_bin']) if direction == 'value_down_good' else bdf['raw_bin']
    bdf = bdf.sort_values('gd')

    X = bdf['gd'].values.astype(float)
    y = bdf['emp'].values
    w = bdf['count'].values.astype(float)

    if len(X) >= 2:
        iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
        fitted = iso.fit_transform(X, y, sample_weight=w)
    else:
        fitted = y.copy()

    fmin, fmax = fitted.min(), fitted.max()
    if fmax > fmin:
        fitted = (fitted - fmin) / (fmax - fmin) * 100.0
    else:
        fitted = np.full_like(fitted, 50.0)

    score_map = {}
    for i, row in bdf.iterrows():
        idx_in = list(bdf.index).index(i)
        score_map[int(row['raw_bin'])] = float(np.clip(fitted[idx_in], 0, 100))
    iso_tables[vid] = score_map

with open(OUTPUT_DIR / "bin_score_table_isotonic_alpha.json", 'w') as f:
    json.dump({k: {str(b): v for b, v in m.items()} for k, m in iso_tables.items()},
              f, indent=2)

# ============================================================
# Phase 7: Item scoring + missing imputation
# ============================================================
print("\n[Phase 7] Item scoring + missing imputation")

# Pre-compute item scores using isotonic
for vid in SEL_IDS:
    score_map = iso_tables[vid]
    df[f'{vid}_iso_score'] = df[f'{vid}_bin'].apply(
        lambda b, sm=score_map: sm.get(int(b), np.nan) if pd.notna(b) else np.nan
    )

# Median imputation per variable from Dev (only for nonfinancial; financial all-missing already filtered)
imputation_map = {}
for vid in SEL_IDS:
    dev_med = df.loc[dev_mask, f'{vid}_iso_score'].median()
    df[f'{vid}_iso_score'] = df[f'{vid}_iso_score'].fillna(dev_med)
    df[f'{vid}_imputed'] = df[vid].isna()
    imputation_map[vid] = float(dev_med)

with open(OUTPUT_DIR / "imputation_map.json", 'w') as f:
    json.dump(imputation_map, f, indent=2)

# Build pre-computed score matrices for optimization
dev_idx_array = np.where(dev_mask)[0]
dev_fin_scores = df.loc[dev_mask, [f'{v}_iso_score' for v in FIN_IDS]].values
dev_nf_scores = df.loc[dev_mask, [f'{v}_iso_score' for v in NONFIN_IDS]].values
dev_actual = df.loc[dev_mask, 'grade_base'].values
dev_rating_num = df.loc[dev_mask, 'rating_num'].values
dev_year = df.loc[dev_mask, 'year'].values

# Anchor stratification flag (derived from grade_full presence in anchor agencies)
# We derive anchor membership from rating provider field if available; otherwise treat all equal
ANCHOR_COL = next((c for c in ['anchor_agency', 'is_anchor', 'agency_anchor']
                   if c in df.columns), None)
if ANCHOR_COL is not None:
    dev_anchor = df.loc[dev_mask, ANCHOR_COL].astype(bool).values
else:
    dev_anchor = np.zeros(n_dev, dtype=bool)

print(f"  Item scores pre-computed: {len(dev_fin_scores)} dev rows")

# ============================================================
# Phase 8: Stage 3 prior weights
# ============================================================
print("\n[Phase 8] Stage 3 prior weights")


def compute_prior_weights(var_ids, scores_dict, floor=W_FLOOR, cap=W_CAP, max_iter=20):
    raw = {v: scores_dict[v] for v in var_ids}
    total = sum(raw.values())
    w = {v: raw[v] / total for v in var_ids}
    for _ in range(max_iter):
        # Clip
        w = {v: min(max(x, floor), cap) for v, x in w.items()}
        s = sum(w.values())
        # Donor pool redistribute to sum=1
        if abs(s - 1.0) < 1e-6:
            break
        # If sum != 1, scale unbinding ones
        binding = {v for v, x in w.items() if abs(x - floor) < 1e-9 or abs(x - cap) < 1e-9}
        free = {v: x for v, x in w.items() if v not in binding}
        if not free:
            w = {v: x / s for v, x in w.items()}
            break
        target = 1.0 - sum(w[v] for v in binding)
        free_sum = sum(free.values())
        if free_sum <= 0:
            w = {v: x / s for v, x in w.items()}
            break
        w = {v: (target * x / free_sum if v in free else w[v]) for v, x in w.items()}
    return w


prior_fin_w = compute_prior_weights(FIN_IDS, SCORES_RAW)
prior_nf_w = compute_prior_weights(NONFIN_IDS, SCORES_RAW)
prior_all = {**prior_fin_w, **prior_nf_w}

print("  Prior weights (Stage 3 selection scores):")
for vid in SEL_IDS:
    block = '재무' if vid in FIN_IDS else '비재무'
    print(f"    [{block}] {vid:45s} prior={prior_all[vid]:.4f}")

with open(OUTPUT_DIR / "prior_weights_alpha.json", 'w') as f:
    json.dump({k: round(v, 6) for k, v in prior_all.items()}, f, indent=2)

# ============================================================
# Phase 9: Dev inner CV split (anchor-stratified)
# ============================================================
print("\n[Phase 9] Dev inner CV split")

INNER_TRAIN_END = int(CFG_CV.get('inner_train_end', 2016))
INNER_VAL_START = int(CFG_CV.get('inner_val_start', 2017))
INNER_VAL_END = int(CFG_CV.get('inner_val_end', 2019))

inner_train_mask = (dev_year <= INNER_TRAIN_END)
inner_val_mask = (dev_year >= INNER_VAL_START) & (dev_year <= INNER_VAL_END)
n_train, n_val = int(inner_train_mask.sum()), int(inner_val_mask.sum())
print(f"  Inner train ({DEV_YEARS[0]}-{INNER_TRAIN_END}): {n_train:,}")
print(f"  Inner val   ({INNER_VAL_START}-{INNER_VAL_END}): {n_val:,}")
assert n_train > 0 and n_val > 0, "Inner CV split failed"

# Anchor-stratification check (informational only — time split takes precedence)
if ANCHOR_COL is not None:
    train_anchor_pct = dev_anchor[inner_train_mask].mean() * 100
    val_anchor_pct = dev_anchor[inner_val_mask].mean() * 100
    overall_anchor = dev_anchor.mean() * 100
    print(f"  Anchor pct — overall: {overall_anchor:.1f}%, "
          f"train: {train_anchor_pct:.1f}%, val: {val_anchor_pct:.1f}%")
else:
    print(f"  (No anchor column found; stratification skipped)")

# Target distribution (full Dev for boundary; train portion for grid search inner objective)
P_target_full_dev = pd.Series(dev_actual).value_counts(normalize=True).reindex(
    GRADE_ORDER, fill_value=0)
P_target_inner_train = pd.Series(dev_actual[inner_train_mask]).value_counts(
    normalize=True).reindex(GRADE_ORDER, fill_value=0)


# ============================================================
# Joint optimization machinery
# ============================================================
def assign_grade_v53(R, b):
    return assign_grade_10(R, b)


def r_score_from_weights(fin_scores, nf_scores, w_fin, w_nf, dev_idx=None):
    fin_raw = fin_scores @ w_fin
    nf_raw = nf_scores @ w_nf
    if dev_idx is not None:
        fp01, fp99 = np.percentile(fin_raw[dev_idx], [1, 99])
        np01, np99 = np.percentile(nf_raw[dev_idx], [1, 99])
    else:
        fp01, fp99 = np.percentile(fin_raw, [1, 99])
        np01, np99 = np.percentile(nf_raw, [1, 99])
    if fp99 <= fp01:
        fp99 = fp01 + 1
    if np99 <= np01:
        np99 = np01 + 1
    fin_norm = np.clip((fin_raw - fp01) / (fp99 - fp01) * 100, 0, 100)
    nf_norm = np.clip((nf_raw - np01) / (np99 - np01) * 100, 0, 100)
    R = 0.70 * fin_norm + 0.30 * nf_norm
    bn = {'financial': {'p01': float(fp01), 'p99': float(fp99)},
          'nonfinancial': {'p01': float(np01), 'p99': float(np99)}}
    return R, bn


def compute_metrics(R, actual, rating_num, boundaries):
    """Returns EM, W1, MAE, and predicted grade distribution; ArrowExtensionArray-safe."""
    pred = np.asarray(assign_grade_v53(R, boundaries), dtype=object)
    actual_arr = np.asarray(pd.Series(actual).astype("object"), dtype=object)
    rating_num_arr = pd.to_numeric(pd.Series(rating_num), errors="coerce").to_numpy(dtype=float)

    em = float(np.mean(pred == actual_arr)) if len(pred) else 0.0

    pred_num = pd.to_numeric(pd.Series(pred, dtype="object").map(GRADE2NUM), errors="coerce").to_numpy(dtype=float)
    diff = np.abs(pred_num - rating_num_arr)
    valid = np.isfinite(diff)
    if valid.any():
        w1 = float(np.mean(diff[valid] <= 1))
        mae = float(np.mean(diff[valid]))
    else:
        w1 = 0.0
        mae = float("nan")

    p_pred = (
        pd.Series(pred, dtype="object")
        .value_counts(normalize=True)
        .reindex(GRADE_ORDER, fill_value=0.0)
        .astype(float)
    )
    return em, w1, mae, p_pred

def kl_divergence(p_target, p_pred, eps=1e-9):
    """ArrowExtensionArray-safe KL divergence over GRADE_ORDER."""
    pt = (
        pd.to_numeric(pd.Series(p_target).reindex(GRADE_ORDER, fill_value=0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + eps
    )
    pp = (
        pd.to_numeric(pd.Series(p_pred).reindex(GRADE_ORDER, fill_value=0.0), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        + eps
    )
    pt_sum = float(np.sum(pt))
    pp_sum = float(np.sum(pp))
    if pt_sum <= 0.0 or pp_sum <= 0.0:
        return 0.0
    pt = pt / pt_sum
    pp = pp / pp_sum
    return float(np.sum(pt * np.log(pt / pp)))

def sparse_tail_penalty(p_pred, sparse_grades=('CCC', 'CC', 'C', 'D'), threshold=0.02):
    """Penalty if predicted distribution puts substantial mass in sparse tail grades."""
    return float(sum(max(p_pred.get(g, 0) - threshold, 0) for g in sparse_grades))


def evaluate_objective(w_fin_arr, w_nf_arr, boundaries,
                       fin_scores, nf_scores, actual, rating_num,
                       p_target, prior_fin_arr, prior_nf_arr,
                       hp):
    R, _ = r_score_from_weights(fin_scores, nf_scores, w_fin_arr, w_nf_arr)
    em, w1, mae, p_pred = compute_metrics(R, actual, rating_num, boundaries)
    kl = kl_divergence(p_target, p_pred)
    reg = float(np.sum((w_fin_arr - prior_fin_arr) ** 2)
                + np.sum((w_nf_arr - prior_nf_arr) ** 2))
    stp = sparse_tail_penalty(p_pred)
    obj = (hp['alpha_em'] * em + hp['alpha_w1'] * w1
           - hp['delta_mae'] * mae - hp['beta_kl'] * kl
           - hp['gamma_reg'] * reg - stp)
    return {'obj': obj, 'em': em, 'w1': w1, 'mae': mae, 'kl': kl, 'reg': reg, 'stp': stp}


def project_weights(w_arr, floor=W_FLOOR, cap=W_CAP):
    w = np.clip(w_arr, floor, cap)
    return w / w.sum()


def optimize_boundaries_for_R(R_dev, actual, rating_num, p_target, hp, max_iter=20):
    """v5.3-style boundary optimization given fixed R_score, with within-1 + MAE in obj."""
    R_max = float(np.percentile(R_dev, 99))
    R_sorted = np.sort(R_dev)[::-1]
    grades_opt = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
    b = {}
    cum = 0.0
    for g in grades_opt:
        cum += p_target.get(g, 0)
        if cum < 1.0:
            idx = int(cum * len(R_dev))
            b[g] = float(R_sorted[idx]) if idx < len(R_sorted) else 0.0
    for i in range(1, len(grades_opt)):
        if grades_opt[i] in b and grades_opt[i - 1] in b:
            if b[grades_opt[i]] >= b[grades_opt[i - 1]]:
                b[grades_opt[i]] = b[grades_opt[i - 1]] - 1.0
    if 'AAA' in b:
        b['AAA'] = min(b['AAA'], R_max)

    def eval_b(b_):
        em, w1, mae, p_pred = compute_metrics(R_dev, actual, rating_num, b_)
        kl = kl_divergence(p_target, p_pred)
        stp = sparse_tail_penalty(p_pred)
        return (hp['alpha_em'] * em + hp['alpha_w1'] * w1
                - hp['delta_mae'] * mae - hp['beta_kl'] * kl - stp)

    best_obj = eval_b(b)
    for _ in range(max_iter):
        improved = False
        for g in grades_opt:
            if g not in b:
                continue
            cur = b[g]
            best_v = cur
            gi = grades_opt.index(g)
            for delta in np.arange(-15, 15.5, 0.5):
                nv = cur + delta
                # Maintain min spacing 0.5 between adjacent boundaries
                if gi > 0 and grades_opt[gi - 1] in b and nv >= b[grades_opt[gi - 1]] - 0.5:
                    continue
                if gi < len(grades_opt) - 1 and grades_opt[gi + 1] in b \
                        and nv <= b[grades_opt[gi + 1]] + 0.5:
                    continue
                if not (0 <= nv <= 100):
                    continue
                if g == 'AAA' and nv > R_max:
                    continue
                trial = b.copy()
                trial[g] = nv
                tobj = eval_b(trial)
                if tobj > best_obj:
                    best_obj = tobj
                    best_v = nv
                    improved = True
            b[g] = best_v
        if not improved:
            break
    return b




def optimize_exact_boost_boundaries_for_R(R_dev, actual, rating_num, p_target,
                                          start_boundaries, max_iter=30):
    """Conservative final boundary retune on fixed R_score using Dev only.

    This intentionally does not change item scores or weights. It is a final
    calibration layer for grade assignment, with a heavier exact/MAE emphasis
    than the joint optimization objective. The score itself remains unchanged
    for downstream reward/rank-shock use.
    """
    grades_opt = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
    R_dev = np.asarray(R_dev, dtype=float)
    b = {g: float(start_boundaries[g]) for g in grades_opt if g in start_boundaries}
    if 'AAA' in b:
        b['AAA'] = min(b['AAA'], float(np.percentile(R_dev, 99)))

    def _objective(b_):
        em, w1, mae, p_pred = compute_metrics(R_dev, actual, rating_num, b_)
        kl = kl_divergence(p_target, p_pred)
        stp = sparse_tail_penalty(p_pred, threshold=0.01)
        # Exact/MAE-forward, but still ordinal and distribution-aware.
        return 4.0 * em + 1.25 * w1 - 1.25 * mae - 0.25 * kl - 1.50 * stp

    best_obj = _objective(b)
    step_grid = list(np.arange(-12.0, 12.25, 0.25))
    for _ in range(max_iter):
        improved = False
        for g in grades_opt:
            if g not in b:
                continue
            cur = b[g]
            best_v = cur
            gi = grades_opt.index(g)
            for delta in step_grid:
                nv = cur + float(delta)
                if gi > 0 and grades_opt[gi - 1] in b and nv >= b[grades_opt[gi - 1]] - 0.5:
                    continue
                if gi < len(grades_opt) - 1 and grades_opt[gi + 1] in b and nv <= b[grades_opt[gi + 1]] + 0.5:
                    continue
                if not (0.0 <= nv <= 100.0):
                    continue
                if g == 'AAA' and nv > float(np.percentile(R_dev, 99)):
                    continue
                trial = b.copy()
                trial[g] = nv
                obj = _objective(trial)
                if obj > best_obj + 1e-12:
                    best_obj = obj
                    best_v = nv
                    improved = True
            b[g] = best_v
        if not improved:
            break
    return b


def make_sparse_tail_collapse_map(actual_dev, min_dev_n=20, floor_grade='CCC'):
    """Return a prediction-only map for unstable lower-tail grades.

    The master scale and PD table remain intact. The guard only suppresses
    separate CC/C/D emissions when there is not enough Dev support to estimate
    those grades reliably.
    """
    floor_grade = str(floor_grade).upper()
    if floor_grade not in GRADE2NUM:
        floor_grade = 'CCC'
    counts = pd.Series(actual_dev).value_counts().to_dict()
    floor_num = int(GRADE2NUM[floor_grade])
    collapse = {}
    for g in GRADE_ORDER:
        if int(GRADE2NUM[g]) > floor_num and int(counts.get(g, 0)) < int(min_dev_n):
            collapse[g] = floor_grade
    return collapse


def apply_grade_collapse(grades, collapse_map):
    if not collapse_map:
        return np.asarray(grades, dtype=object)
    return np.asarray([collapse_map.get(str(g), str(g)) for g in grades], dtype=object)

def optimize_weights(boundaries, fin_scores, nf_scores, actual, rating_num,
                     p_target, w_fin_init, w_nf_init,
                     prior_fin_arr, prior_nf_arr, hp,
                     n_trials=N_TRIALS, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    best_w_fin = w_fin_init.copy()
    best_w_nf = w_nf_init.copy()
    best_eval = evaluate_objective(best_w_fin, best_w_nf, boundaries,
                                   fin_scores, nf_scores, actual, rating_num,
                                   p_target, prior_fin_arr, prior_nf_arr, hp)
    for _ in range(n_trials):
        noise_fin = rng.normal(0, 0.03, len(w_fin_init))
        noise_nf = rng.normal(0, 0.03, len(w_nf_init))
        trial_fin = project_weights(best_w_fin + noise_fin)
        trial_nf = project_weights(best_w_nf + noise_nf)
        ev = evaluate_objective(trial_fin, trial_nf, boundaries,
                                fin_scores, nf_scores, actual, rating_num,
                                p_target, prior_fin_arr, prior_nf_arr, hp)
        if ev['obj'] > best_eval['obj']:
            best_eval = ev
            best_w_fin = trial_fin
            best_w_nf = trial_nf
    return best_w_fin, best_w_nf, best_eval


def run_joint_loop(fin_scores, nf_scores, actual, rating_num, p_target,
                   prior_fin_arr, prior_nf_arr, hp, n_outer=N_OUTER_ITER, rng=None):
    if rng is None:
        rng = np.random.default_rng(RNG_SEED)
    w_fin_arr = prior_fin_arr.copy()
    w_nf_arr = prior_nf_arr.copy()

    R_init, _ = r_score_from_weights(fin_scores, nf_scores, w_fin_arr, w_nf_arr)
    boundaries = optimize_boundaries_for_R(R_init, actual, rating_num, p_target, hp)
    init_eval = evaluate_objective(w_fin_arr, w_nf_arr, boundaries,
                                   fin_scores, nf_scores, actual, rating_num,
                                   p_target, prior_fin_arr, prior_nf_arr, hp)

    trace = [{'iter': 0, 'phase': 'init', **{k: round(v, 4) for k, v in init_eval.items()}}]
    prev_obj = init_eval['obj']

    for outer in range(1, n_outer + 1):
        # Step A: optimize weights given boundaries
        w_fin_arr, w_nf_arr, w_eval = optimize_weights(
            boundaries, fin_scores, nf_scores, actual, rating_num, p_target,
            w_fin_arr, w_nf_arr, prior_fin_arr, prior_nf_arr, hp, rng=rng
        )
        trace.append({'iter': outer, 'phase': 'weight',
                      **{k: round(v, 4) for k, v in w_eval.items()}})

        # Step B: optimize boundaries given weights
        R_new, _ = r_score_from_weights(fin_scores, nf_scores, w_fin_arr, w_nf_arr)
        boundaries = optimize_boundaries_for_R(R_new, actual, rating_num, p_target, hp)
        b_eval = evaluate_objective(w_fin_arr, w_nf_arr, boundaries,
                                    fin_scores, nf_scores, actual, rating_num,
                                    p_target, prior_fin_arr, prior_nf_arr, hp)
        trace.append({'iter': outer, 'phase': 'boundary',
                      **{k: round(v, 4) for k, v in b_eval.items()}})

        if abs(b_eval['obj'] - prev_obj) < 1e-4:
            break
        prev_obj = b_eval['obj']

    return w_fin_arr, w_nf_arr, boundaries, b_eval, trace


# ============================================================
# Phase 10: Hyperparameter grid search on inner val
# ============================================================
print("\n[Phase 10] Hyperparameter grid search on Dev inner validation")

prior_fin_arr = np.array([prior_fin_w[v] for v in FIN_IDS])
prior_nf_arr = np.array([prior_nf_w[v] for v in NONFIN_IDS])

train_fin = dev_fin_scores[inner_train_mask]
train_nf = dev_nf_scores[inner_train_mask]
train_actual = dev_actual[inner_train_mask]
train_rating_num = dev_rating_num[inner_train_mask]

val_fin = dev_fin_scores[inner_val_mask]
val_nf = dev_nf_scores[inner_val_mask]
val_actual = dev_actual[inner_val_mask]
val_rating_num = dev_rating_num[inner_val_mask]

grid_alpha_em = CFG_GRID['alpha_em']
grid_beta_kl = CFG_GRID['beta_kl']
grid_gamma_reg = CFG_GRID['gamma_reg']

print(f"  Grid: α_EM∈{grid_alpha_em} × β_KL∈{grid_beta_kl} × γ_REG∈{grid_gamma_reg}")
print(f"        ({len(grid_alpha_em) * len(grid_beta_kl) * len(grid_gamma_reg)} combinations)")

grid_results = []
combo_idx = 0
total_combos = len(grid_alpha_em) * len(grid_beta_kl) * len(grid_gamma_reg)
for a_em in grid_alpha_em:
    for b_kl in grid_beta_kl:
        for g_reg in grid_gamma_reg:
            combo_idx += 1
            hp = {'alpha_em': float(a_em), 'alpha_w1': DEFAULT_ALPHA_W1,
                  'delta_mae': DEFAULT_DELTA_MAE, 'beta_kl': float(b_kl),
                  'gamma_reg': float(g_reg)}
            # Train on inner-train, evaluate on inner-val
            w_fin, w_nf, b_, _, _ = run_joint_loop(
                train_fin, train_nf, train_actual, train_rating_num,
                P_target_inner_train, prior_fin_arr, prior_nf_arr, hp,
                n_outer=3, rng=np.random.default_rng(RNG_SEED + combo_idx)
            )
            # Score on val
            R_val, _ = r_score_from_weights(val_fin, val_nf, w_fin, w_nf)
            v_em, v_w1, v_mae, _ = compute_metrics(R_val, val_actual, val_rating_num, b_)
            sel_metric = v_em + 0.5 * v_w1 - 0.5 * v_mae
            grid_results.append({
                'alpha_em': a_em, 'beta_kl': b_kl, 'gamma_reg': g_reg,
                'val_em': round(v_em, 4), 'val_w1': round(v_w1, 4),
                'val_mae': round(v_mae, 4), 'selection_metric': round(sel_metric, 4)
            })
            print(f"  [{combo_idx:3d}/{total_combos}] α={a_em} β={b_kl} γ={g_reg} → "
                  f"val EM={v_em:.3f} W1={v_w1:.3f} MAE={v_mae:.3f} "
                  f"sel={sel_metric:.3f}")

grid_df = pd.DataFrame(grid_results).sort_values('selection_metric', ascending=False)
grid_df.to_csv(OUTPUT_DIR / "hyperparameter_grid_search.csv", index=False)
best = grid_df.iloc[0].to_dict()
print(f"\n  Best hyperparameters (Dev val):")
print(f"    α_EM = {best['alpha_em']}")
print(f"    β_KL = {best['beta_kl']}")
print(f"    γ_REG = {best['gamma_reg']}")
print(f"    val EM={best['val_em']:.3f}, W1={best['val_w1']:.3f}, MAE={best['val_mae']:.3f}")

# ============================================================
# Phase 11: Final joint optimization on full Dev
# ============================================================
print("\n[Phase 11] Final joint optimization on full Dev")

selected_hp = {
    'alpha_em': float(best['alpha_em']),
    'alpha_w1': DEFAULT_ALPHA_W1,
    'delta_mae': DEFAULT_DELTA_MAE,
    'beta_kl': float(best['beta_kl']),
    'gamma_reg': float(best['gamma_reg'])
}

w_fin_final, w_nf_final, boundaries, final_eval, opt_trace = run_joint_loop(
    dev_fin_scores, dev_nf_scores, dev_actual, dev_rating_num,
    P_target_full_dev, prior_fin_arr, prior_nf_arr, selected_hp,
    n_outer=N_OUTER_ITER, rng=np.random.default_rng(RNG_SEED)
)

print(f"\n  Final Dev metrics: EM={final_eval['em']:.3f}, W1={final_eval['w1']:.3f}, "
      f"MAE={final_eval['mae']:.3f}, KL={final_eval['kl']:.3f}")
print(f"  Boundaries:")
for g in ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']:
    if g in boundaries:
        print(f"    {g} >= {boundaries[g]:.2f}")

pd.DataFrame(opt_trace).to_csv(OUTPUT_DIR / "joint_optimization_trace.csv", index=False)

fin_w = {v: float(w_fin_final[i]) for i, v in enumerate(FIN_IDS)}
nonfin_w = {v: float(w_nf_final[i]) for i, v in enumerate(NONFIN_IDS)}
all_weights = {**fin_w, **nonfin_w}

# ============================================================
# Phase 12: Weight sanity check
# ============================================================
print("\n[Phase 12] Weight sanity check")

from modules.weight_sanity import run_sanity_check
sanity_result = run_sanity_check(prior_all, all_weights, FIN_IDS, NONFIN_IDS,
                                 SEL_VARS, DIRS)

print(f"  Direction consistency: {'✓ OK' if sanity_result['direction_consistent'] else '✗ FAIL'}")
print(f"  Suspicious patterns: {len(sanity_result['suspicious'])}")
if sanity_result['suspicious']:
    for s in sanity_result['suspicious']:
        print(f"    ⚠ {s}")

with open(OUTPUT_DIR / "weight_sanity_check_alpha.json", 'w') as f:
    json.dump(sanity_result, f, indent=2, ensure_ascii=False)

pd.DataFrame([{
    'variable_id': v,
    'block': 'financial' if v in FIN_IDS else 'nonfinancial',
    'weight_prior': round(prior_all[v], 6),
    'weight_optimized': round(all_weights[v], 6),
    'delta': round(all_weights[v] - prior_all[v], 6),
    'delta_pct': round((all_weights[v] - prior_all[v]) / prior_all[v] * 100, 2),
} for v in SEL_IDS]).to_csv(OUTPUT_DIR / "item_weights_alpha.csv", index=False)

with open(OUTPUT_DIR / "item_weights_alpha.json", 'w') as f:
    json.dump({k: round(v, 6) for k, v in all_weights.items()}, f, indent=2)

print(f"\n  {'Variable':<45s} {'Prior':>9s} {'Optimized':>11s} {'Δ':>9s}")
for vid in SEL_IDS:
    print(f"  {vid:<45s} {prior_all[vid]:9.4f} {all_weights[vid]:11.4f} "
          f"{all_weights[vid] - prior_all[vid]:+9.4f}")

# ============================================================
# Phase 13: Apply final weights to full sample + outputs
# ============================================================
print("\n[Phase 13] Applying final weights to full sample + outputs")

fin_s_full = df[[f'{v}_iso_score' for v in FIN_IDS]].values
nf_s_full = df[[f'{v}_iso_score' for v in NONFIN_IDS]].values
w_f = np.array([fin_w[v] for v in FIN_IDS])
w_n = np.array([nonfin_w[v] for v in NONFIN_IDS])

df['fin_raw_alpha'] = fin_s_full @ w_f
df['nonfin_raw_alpha'] = nf_s_full @ w_n

fp01 = df.loc[dev_mask, 'fin_raw_alpha'].quantile(0.01)
fp99 = df.loc[dev_mask, 'fin_raw_alpha'].quantile(0.99)
np01 = df.loc[dev_mask, 'nonfin_raw_alpha'].quantile(0.01)
np99 = df.loc[dev_mask, 'nonfin_raw_alpha'].quantile(0.99)

df['fin_score_alpha'] = ((df['fin_raw_alpha'] - fp01) / (fp99 - fp01) * 100).clip(0, 100)
df['nonfin_score_alpha'] = ((df['nonfin_raw_alpha'] - np01) / (np99 - np01) * 100).clip(0, 100)
df['R_score_alpha'] = 0.70 * df['fin_score_alpha'] + 0.30 * df['nonfin_score_alpha']

# Optional final boundary exact-boost on fixed R_score. Uses Dev only and does
# not alter R_score, item scores, weights, or block normalization.
boundary_finalization_report = {
    'exact_boost_enabled': bool(ALPHA_EXACT_BOOST),
    'sparse_tail_guard_enabled': bool(ALPHA_SPARSE_TAIL_GUARD),
    'sparse_tail_min_dev_n': int(ALPHA_SPARSE_TAIL_MIN_DEV_N),
    'sparse_tail_floor_grade': ALPHA_SPARSE_TAIL_FLOOR_GRADE,
}
base_boundaries = ensure_10_grade_contract(boundaries)
if ALPHA_EXACT_BOOST:
    before_metrics = compute_metrics(
        df.loc[dev_mask, 'R_score_alpha'].values,
        df.loc[dev_mask, 'grade_base_10'].values,
        df.loc[dev_mask, 'rating_num_10'].values,
        base_boundaries,
    )[:3]
    boosted = optimize_exact_boost_boundaries_for_R(
        df.loc[dev_mask, 'R_score_alpha'].values,
        df.loc[dev_mask, 'grade_base_10'].values,
        df.loc[dev_mask, 'rating_num_10'].values,
        P_target_full_dev,
        base_boundaries,
    )
    boosted = ensure_10_grade_contract(boosted)
    after_metrics = compute_metrics(
        df.loc[dev_mask, 'R_score_alpha'].values,
        df.loc[dev_mask, 'grade_base_10'].values,
        df.loc[dev_mask, 'rating_num_10'].values,
        boosted,
    )[:3]
    # Accept only if Dev exact improves or ties with lower MAE; never accept a
    # materially worse within-1 rate.
    accept = ((after_metrics[0] > before_metrics[0] + 1e-12)
              or (abs(after_metrics[0] - before_metrics[0]) <= 1e-12 and after_metrics[2] < before_metrics[2])) \
             and (after_metrics[1] >= before_metrics[1] - 0.005)
    boundary_finalization_report.update({
        'dev_metrics_before_exact_boost': {'exact': before_metrics[0], 'within1': before_metrics[1], 'mae': before_metrics[2]},
        'dev_metrics_after_exact_boost_candidate': {'exact': after_metrics[0], 'within1': after_metrics[1], 'mae': after_metrics[2]},
        'exact_boost_accepted': bool(accept),
    })
    boundaries = boosted if accept else base_boundaries
else:
    boundaries = base_boundaries

sparse_tail_collapse_map = {}
if ALPHA_SPARSE_TAIL_GUARD:
    sparse_tail_collapse_map = make_sparse_tail_collapse_map(
        df.loc[dev_mask, 'grade_base_10'].values,
        min_dev_n=ALPHA_SPARSE_TAIL_MIN_DEV_N,
        floor_grade=ALPHA_SPARSE_TAIL_FLOOR_GRADE,
    )
raw_alpha_grade = assign_grade_v53(df['R_score_alpha'].values, boundaries)
df['R_grade_alpha_raw'] = raw_alpha_grade
df['R_grade_alpha'] = apply_grade_collapse(raw_alpha_grade, sparse_tail_collapse_map)
df['R_grade_alpha_num'] = df['R_grade_alpha'].map(GRADE2NUM)
df['R_PD_alpha'] = df['R_grade_alpha'].map(PD_MAP)
boundary_finalization_report['sparse_tail_collapse_map'] = sparse_tail_collapse_map
with open(OUTPUT_DIR / "alpha_boundary_finalization_report.json", 'w', encoding='utf-8') as f:
    json.dump(boundary_finalization_report, f, indent=2, ensure_ascii=False, default=str)

block_norm = {'financial': {'p01': float(fp01), 'p99': float(fp99)},
              'nonfinancial': {'p01': float(np01), 'p99': float(np99)}}
with open(OUTPUT_DIR / "block_normalization_alpha.json", 'w') as f:
    json.dump(block_norm, f, indent=2)

# Boundary records
boundary_records = []
grades_opt = GRADE_ORDER
grade_counts_dev = pd.Series(dev_actual).value_counts()
for i in range(len(grades_opt) - 1):
    g = grades_opt[i]
    if g in boundaries:
        sparse = grade_counts_dev.get(g, 0) < 20
        boundary_records.append({
            'boundary_id': i, 'upper_grade': g, 'lower_grade': grades_opt[i + 1],
            'threshold_score': round(boundaries[g], 2),
            'estimable': g not in ['CC', 'C'], 'fixed': sparse or g in ['CC', 'C'], 'sparse_tail_flag': sparse or g in ['CC', 'C'],
            'reason': 'deterministic_tail_extrapolation' if g in ['CC', 'C'] else ('sparse_tail_fixed' if sparse else 'joint_optimized')
        })

pd.DataFrame(boundary_records).to_csv(OUTPUT_DIR / "boundary_table_alpha.csv", index=False)
with open(OUTPUT_DIR / "boundary_table_alpha.json", 'w') as f:
    json.dump(boundary_records, f, indent=2, default=str)

# Master scale PD reference
with open(OUTPUT_DIR / "master_scale_pd_mapping.json", 'w') as f:
    json.dump(PD_MAP, f, indent=2)

# Firm-year output
base_cols = [KEY]
for c in ['회사명', '시장', 'sector_7']:
    if c in df.columns:
        base_cols.append(c)
base_cols += ['year', 'split_stage4', 'grade_base', 'rating_num', 'grade_base_10', 'rating_num_10', 'grade_base_7', 'rating_num_7', 'grade_base_notch', 'rating_num_notch']

out_cols = base_cols + SEL_IDS + \
    [f'{v}_iso_score' for v in SEL_IDS] + \
    [f'{v}_imputed' for v in SEL_IDS] + \
    ['fin_raw_alpha', 'nonfin_raw_alpha', 'fin_score_alpha', 'nonfin_score_alpha',
     'R_score_alpha', 'R_grade_alpha_raw', 'R_grade_alpha', 'R_grade_alpha_num', 'R_PD_alpha']
out_cols = [c for c in out_cols if c in df.columns]
out_df = df[out_cols].copy()
out_df['specification'] = 'alpha'
out_df.to_parquet(OUTPUT_DIR / "oracle_firm_year_output_alpha.parquet")
out_df.to_csv(OUTPUT_DIR / "oracle_firm_year_output_alpha.csv",
              index=False, encoding='utf-8-sig')
print(f"  oracle_firm_year_output_alpha: {len(out_df):,} rows")

# Modeling panel
mp_cols = [c for c in base_cols + SEL_IDS + ['eligible_for_stage2'] if c in df.columns]
df[mp_cols].to_parquet(OUTPUT_DIR / "oracle_modeling_panel_alpha.parquet")

# ============================================================
# Phase 14: Diagnostics (preliminary metrics, confusion, boundary jump)
# ============================================================
print("\n[Phase 14] Diagnostics")


def calc_metrics(sub):
    valid = sub[sub['grade_base'].notna() & sub['R_grade_alpha'].notna()]
    if len(valid) == 0:
        return {}
    obs = pd.to_numeric(valid['rating_num_10'], errors='coerce') if 'rating_num_10' in valid.columns else valid['rating_num']
    pred = valid['R_grade_alpha_num']
    exact = float(((valid['grade_base_10'] if 'grade_base_10' in valid.columns else valid['grade_base']) == valid['R_grade_alpha']).mean())
    w1 = float((np.abs(obs - pred) <= 1).mean())
    mae = float(np.abs(obs - pred).mean())
    rho = float(spearmanr(valid['R_score_alpha'], obs)[0])
    tgt = (obs <= 4).astype(int)
    try:
        ar = float((2 * roc_auc_score(tgt, valid['R_score_alpha']) - 1) * 100)
    except Exception:
        ar = np.nan
    try:
        qk = float(cohen_kappa_score(obs.astype(int), pred.astype(int), weights='quadratic'))
    except Exception:
        qk = np.nan
    return {'n': len(valid), 'exact': round(exact, 4), 'within1': round(w1, 4),
            'mae': round(mae, 3), 'rho': round(rho, 4),
            'AR': round(ar, 1) if not np.isnan(ar) else None,
            'QK': round(qk, 3) if not np.isnan(qk) else None}


metrics_rows = []
for sn in ['dev', 'oot']:
    m = calc_metrics(df[df['split_stage4'] == sn])
    m.update({'specification': 'alpha', 'split': sn})
    metrics_rows.append(m)
pd.DataFrame(metrics_rows).to_csv(OUTPUT_DIR / "preliminary_dev_oot_metrics_alpha.csv",
                                  index=False)

dev_m = [m for m in metrics_rows if m['split'] == 'dev'][0]
oot_m = [m for m in metrics_rows if m['split'] == 'oot'][0]
print(f"\n  === DEV (alpha) ===")
print(f"  n={dev_m['n']}  Exact: {dev_m['exact']:.1%}  Within-1: {dev_m['within1']:.1%}  "
      f"MAE: {dev_m['mae']:.3f}  ρ: {dev_m['rho']:.3f}")
print(f"\n  === OOT (alpha) ===")
print(f"  n={oot_m['n']}  Exact: {oot_m['exact']:.1%}  Within-1: {oot_m['within1']:.1%}  "
      f"MAE: {oot_m['mae']:.3f}  ρ: {oot_m['rho']:.3f}")

# Confusion matrices
for sn in ['dev', 'oot']:
    valid = df[(df['split_stage4'] == sn) & df['grade_base'].notna()
               & df['R_grade_alpha'].notna()]
    if len(valid) > 0:
        labels = sorted(set(valid['grade_base']) | set(valid['R_grade_alpha']),
                        key=lambda g: GRADE2NUM.get(g, 99))
        cm = pd.DataFrame(0, index=labels, columns=labels)
        for o, p in zip(valid['grade_base'], valid['R_grade_alpha']):
            if o in cm.index and p in cm.columns:
                cm.loc[o, p] += 1
        cm.to_csv(OUTPUT_DIR / f"confusion_matrix_alpha_{sn}.csv", encoding='utf-8-sig')

# Grade distribution comparison
gd_comp = []
for sn in ['dev', 'oot']:
    sub = df[df['split_stage4'] == sn]
    for g in GRADE_ORDER:
        gd_comp.append({
            'split': sn, 'grade': g,
            'observed': int((sub['grade_base'] == g).sum()),
            'predicted_alpha': int((sub['R_grade_alpha'] == g).sum())
        })
pd.DataFrame(gd_comp).to_csv(OUTPUT_DIR / "grade_distribution_alpha.csv", index=False)

# Boundary jump test
print("\n  Running boundary jump test (50 sample firms × 11 vars × 6 perturbations)...")
from modules.oracle_alpha_scorer import build_alpha_scorer
scorer = build_alpha_scorer({
    'selected_variables': SEL_IDS,
    'fin_ids': FIN_IDS, 'nonfin_ids': NONFIN_IDS,
    'directions': DIRS, 'winsor': winsor_records,
    'bin_edges': bin_edges_all, 'iso_tables': iso_tables,
    'item_weights': all_weights, 'block_norm': block_norm,
    'boundaries': boundaries, 'imputation_map': imputation_map,
    'pd_map': PD_MAP, 'grade2num': GRADE2NUM
})

dev_df_full = df[dev_mask].copy()
sample_dev = dev_df_full.sample(n=min(50, len(dev_df_full)),
                                random_state=RNG_SEED).reset_index(drop=True)
perturbations = [-0.05, -0.01, -0.005, 0.005, 0.01, 0.05]
jump_rows = []
for _, row in sample_dev.iterrows():
    orig_vals = {vid: row[vid] for vid in SEL_IDS}
    orig_out = scorer(orig_vals)
    for vid in SEL_IDS:
        ov = row[vid]
        if pd.isna(ov) or ov == 0:
            continue
        for pert in perturbations:
            nv = ov * (1 + pert)
            new_vals = dict(orig_vals)
            new_vals[vid] = nv
            new_out = scorer(new_vals)
            jump_rows.append({
                'firm_id': row[KEY], 'year': int(row['year']), 'variable_id': vid,
                'perturbation': pert,
                'original_value': round(ov, 6), 'perturbed_value': round(nv, 6),
                'item_score_before': round(orig_out['item_scores'][vid], 2),
                'item_score_after': round(new_out['item_scores'][vid], 2),
                'item_score_delta': round(
                    new_out['item_scores'][vid] - orig_out['item_scores'][vid], 2),
                'R_score_before': round(orig_out['R_score'], 4),
                'R_score_after': round(new_out['R_score'], 4),
                'R_score_delta': round(new_out['R_score'] - orig_out['R_score'], 4),
                'R_grade_before': orig_out['R_grade'],
                'R_grade_after': new_out['R_grade'],
                'boundary_crossed': orig_out['R_grade'] != new_out['R_grade'],
                'excessive_jump_flag': abs(new_out['item_scores'][vid]
                                           - orig_out['item_scores'][vid]) > 20
                                       and abs(pert) <= 0.01
            })
pd.DataFrame(jump_rows).to_csv(OUTPUT_DIR / "boundary_jump_test_alpha.csv", index=False)
print(f"  {len(jump_rows)} tests, "
      f"{sum(1 for r in jump_rows if r['boundary_crossed'])} grade changes, "
      f"{sum(1 for r in jump_rows if r['excessive_jump_flag'])} excessive jumps")

# ============================================================
# Final: oracle_alpha_params.json + manifest
# ============================================================
print("\n[Final] Saving oracle_alpha_params.json")

params = {
    'model_name': 'Reference Credit Rating Oracle α',
    'version': 'stage4_alpha_v1.0',
    'spec': 'Isotonic + regularized joint weight-boundary KL optimization',
    'sample_period': f'{int(df["year"].min())}-{int(df["year"].max())}',
    'year_scope': ORACLE_YEAR_SCOPE,
    'score_end_year': ORACLE_MAX_YEAR,
    'split_policy': {'dev': f'{DEV_YEARS[0]}-{DEV_YEARS[1]}', 'oot': f'{OOT_YEARS[0]}-{OOT_YEARS[1]}'},
    'inner_cv_split': {
        'inner_train': f'{DEV_YEARS[0]}-{INNER_TRAIN_END}',
        'inner_val': f'{INNER_VAL_START}-{INNER_VAL_END}'
    },
    'sample_size': {'total': len(df), 'dev': n_dev, 'oot': n_oot},
    'grade_order': GRADE_ORDER,
    'fallback_grade': 'D',
    'selected_variables': SEL_IDS,
    'selected_variable_records': SEL_VARS,
    'direction_encoding': DIRS,
    'winsorization_params': winsor_records,
    'bin_edges': bin_edges_all,
    'bin_score_table_isotonic': {vid: {str(k): round(v, 4) for k, v in m.items()}
                                 for vid, m in iso_tables.items()},
    'imputation_map': imputation_map,
    'prior_weights': {k: round(v, 6) for k, v in prior_all.items()},
    'optimized_weights': {k: round(v, 6) for k, v in all_weights.items()},
    'block_normalization': block_norm,
    'combined_weights': {'financial': 0.70, 'nonfinancial': 0.30},
    'boundaries': {g: round(float(boundaries[g]), 2) for g in boundaries},
    'sparse_tail_collapse_map': sparse_tail_collapse_map,
    'boundary_finalization': boundary_finalization_report,
    'boundary_table': boundary_records,
    'master_scale_pd': PD_MAP,
    'objective_function': (
        'α_EM·EM + α_W1·Within-1 - δ_MAE·MAE - β_KL·KL(P_target||P_pred) '
        '- γ_REG·||w-w_prior||² - sparse_tail_penalty'
    ),
    'selected_hyperparameters': selected_hp,
    'hyperparameter_selection': 'Dev inner validation (2002-2016 train / 2017-2019 val)',
    'optimization_trace': opt_trace,
    'final_dev_metrics': {k: round(v, 4) for k, v in final_eval.items()},
    'preliminary_metrics': metrics_rows,
    'weight_sanity_check': sanity_result,
    'caveats': [
        'OOT was reserved for final reporting only; never used for tuning',
        'Boundary optimization is Dev in-sample tuning',
        'PD alignment is not calibration',
        'Nonfinancial component is proxy-based',
        'Rank-shock deferred to Stage 6',
        'Stage 5A formal acceptance check via verify_alpha_metrics.py'
    ]
}

with open(OUTPUT_DIR / "oracle_alpha_params.json", 'w', encoding='utf-8') as f:
    json.dump(params, f, indent=2, ensure_ascii=False, default=str)

# Rank-shock base panel
rs_cols = [KEY, 'year', 'split_stage4', 'grade_base', 'rating_num', 'grade_base_10', 'rating_num_10',
           'R_score_alpha', 'R_grade_alpha'] + SEL_IDS
if 'sector_7' in df.columns:
    rs_cols.insert(1, 'sector_7')
if '회사명' in df.columns:
    rs_cols.insert(1, '회사명')
rs_cols = [c for c in rs_cols if c in df.columns]
df[rs_cols].to_parquet(OUTPUT_DIR / "rank_shock_base_panel_alpha.parquet")

# File manifest
files = sorted(OUTPUT_DIR.glob('*'))
manifest = [{'filename': f.name, 'size': f.stat().st_size}
            for f in files if f.is_file()]
pd.DataFrame(manifest).to_csv(OUTPUT_DIR / "stage4_alpha_file_manifest.csv", index=False)

print(f"\n{'=' * 72}")
print(f"Stage 4 α Pipeline Complete")
print(f"{'=' * 72}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Files:  {len(manifest)}")
print(f"  Sample: {len(df):,} firm-years")
print(f"  Dev exact: {dev_m['exact']:.1%}   OOT exact: {oot_m['exact']:.1%}")
print(f"  Dev W1:    {dev_m['within1']:.1%}   OOT W1:    {oot_m['within1']:.1%}")
print(f"  Next: run verify_alpha_metrics.py to check 6 acceptance gates")
print(f"{'=' * 72}")

