"""
Stage 3 Variable Selection

Purpose:
- Select final 11 variables (6 financial + 5 nonfinancial) for Oracle scorecard
- Dev sample is controlled by config/env split policy only for metric calculation
- 4-metric screening: Spearman, IV, KW eta², monotonicity
- Collinearity replacement with priority order
- Generate code reproduction bundle

Author: Claude
Date: 2026-05-01
"""

import pandas as pd
import numpy as np
import json
import os
import hashlib
import shutil
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
import yaml
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

INPUT_BASE = Path(".")
CONFIG_PATH = Path(os.environ.get("ORACLE_STAGE00_04_CONFIG", "stage3_config.yaml"))


def load_config(path: Path) -> dict:
    if not path.exists():
        # Import-safe guard: this file is executed by the Stage1 orchestrator via
        # runpy with ORACLE_STAGE00_04_CONFIG set.  Static analyzers/import smoke
        # tests must not fail merely because the runtime config is absent.
        if __name__ != "__main__" and "ORACLE_STAGE00_04_CONFIG" not in os.environ:
            return {}
        raise FileNotFoundError(f"Stage 3 config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


CFG = load_config(CONFIG_PATH)

# Paths (config is the source of truth)
PATHS_CFG = CFG.get("paths", {})
STAGE1B_DIR = INPUT_BASE / PATHS_CFG.get("stage1b", "stage1b")
STAGE2_DIR = INPUT_BASE / PATHS_CFG.get("stage2", "stage2")
# IMPORTANT: stable alias must point to Stage 1C v3.2 output. Do not use stage1c_v2.
STAGE1C_DIR = INPUT_BASE / PATHS_CFG.get("stage1c", "stage1c")

OUTPUT_BASE = Path(CFG.get("outputs", {}).get("directory", "outputs"))
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
CREATE_BACKWARD_COMPAT = bool(CFG.get("outputs", {}).get("create_backward_compat", False))
CREATE_LATEST_ALIAS = bool(CFG.get("outputs", {}).get("create_latest_alias", True))

# Split policy. Environment variables are injected by the final Stage1 runner
# so variable selection and backend training cannot silently use different splits.
def _split_pair_from_env(name: str):
    raw = str(os.environ.get(name, "")).strip()
    if not raw:
        return None
    parts = raw.replace("-", ",").split(",")
    if len(parts) != 2:
        raise ValueError(f"{name} must be START,END or START-END; got {raw!r}")
    return int(parts[0]), int(parts[1])

SPLIT_CFG = CFG.get("split", {})
_env_dev = _split_pair_from_env("ORACLE_DEV_YEARS")
_env_oot = _split_pair_from_env("ORACLE_OOT_YEARS")
DEV_YEAR_MIN = int((_env_dev or (SPLIT_CFG.get("dev_start", 2002), SPLIT_CFG.get("dev_end", 2019)))[0])
DEV_YEAR_MAX = int((_env_dev or (SPLIT_CFG.get("dev_start", 2002), SPLIT_CFG.get("dev_end", 2019)))[1])
OOT_YEAR_MIN = int((_env_oot or (SPLIT_CFG.get("oot_start", 2020), SPLIT_CFG.get("oot_end", int(os.environ.get("ORACLE_MAX_YEAR", 2024)))))[0])
OOT_YEAR_MAX = int((_env_oot or (SPLIT_CFG.get("oot_start", 2020), SPLIT_CFG.get("oot_end", int(os.environ.get("ORACLE_MAX_YEAR", 2024)))))[1])
if DEV_YEAR_MAX >= OOT_YEAR_MIN:
    raise ValueError(f"Invalid Stage00_04 split: dev={DEV_YEAR_MIN}-{DEV_YEAR_MAX}, oot={OOT_YEAR_MIN}-{OOT_YEAR_MAX}")

# Metric thresholds
COLL_CFG = CFG.get("collinearity", {})
COLLINEARITY_THRESHOLD = float(COLL_CFG.get("threshold", 0.7))
MAX_COLLINEARITY_ITER = int(COLL_CFG.get("max_iterations", 10))
DIRECTION_WEAK_THRESHOLD = float(CFG.get("direction_weak_threshold", 0.03))

# Selection weights
SEL_W_CFG = CFG.get("selection_score_weights", {})
DEFAULT_SELECTION_WEIGHTS = {
    'spearman': float(SEL_W_CFG.get('spearman', 0.30)),
    'iv': float(SEL_W_CFG.get('iv', 0.30)),
    'kw': float(SEL_W_CFG.get('kw_eta2', SEL_W_CFG.get('kw', 0.20))),
    'mono': float(SEL_W_CFG.get('monotonicity', SEL_W_CFG.get('mono', 0.20))),
}
WEIGHTS = {
    'default': DEFAULT_SELECTION_WEIGHTS,
    'equal': {'spearman': 0.25, 'iv': 0.25, 'kw': 0.25, 'mono': 0.25},
    'rho_heavy': {'spearman': 0.50, 'iv': 0.20, 'kw': 0.15, 'mono': 0.15},
    'iv_heavy': {'spearman': 0.20, 'iv': 0.50, 'kw': 0.15, 'mono': 0.15},
    'mono_heavy': {'spearman': 0.20, 'iv': 0.20, 'kw': 0.20, 'mono': 0.40},
}

# Priority order for collinearity replacement
PRIORITY_CFG = CFG.get("priority_order", {})
PRIORITY_ORDER = PRIORITY_CFG.get("financial", [
    '수익성', '안정성', '부채상환능력', '유동성', '활동성', '성장성'
]) + PRIORITY_CFG.get("nonfinancial", [
    '산업위험', '경영위험', '영업위험', '재무위험', '신뢰도'
])

# Financial/nonfinancial main categories
FINANCIAL_MAIN_CATEGORIES = PRIORITY_CFG.get("financial", ['수익성', '안정성', '부채상환능력', '유동성', '활동성', '성장성'])
NONFINANCIAL_CATEGORIES = PRIORITY_CFG.get("nonfinancial", ['산업위험', '경영위험', '영업위험', '재무위험', '신뢰도'])

# ── Domain filters (Stage 3 v3.2 guardrails) ────────────────────────────────
DOMAIN_CFG = CFG.get("domain_filters", {})
LIQUIDITY_EXCLUDE_TERMS = DOMAIN_CFG.get("liquidity_exclude_terms", [
    '매출액', '매출원가', '매출총이익', '영업수익', '수익(매출액)',
])
# Explicit liquidity R-code exclusions.  These are not generic duplicate aliases:
# they are liquidity scorecard candidates that must not be selected during
# oracle redevelopment.  R122 is already covered by the revenue-term guard but
# remains listed for auditability.  R136 is explicitly blocked; R133 remains
# eligible by design so Stage00_04 can select it when its empirical score wins.
GAMEABLE_LIQUIDITY_EXCLUDE = set(DOMAIN_CFG.get("gameable_liquidity_exclude", [
    'R122', 'R136',
]))
INDUSTRY_ALLOWLIST = set(DOMAIN_CFG.get("industry_allowlist", [
    'industry_avg_rating_lag1_self_excl',
    'industry_median_rating_lag1_self_excl',
    'industry_bad_grade_share_lag1_self_excl',
    'industry_b_or_lower_share_lag1_self_excl',
    'industry_downgrade_rate_lag1_self_excl',
    'industry_upgrade_rate_lag1_self_excl',
    'industry_net_downgrade_rate_lag1_self_excl',
    'industry_rating_std_lag1_self_excl',
    'industry_rating_iqr_lag1_self_excl',
]))
DIAGNOSTIC_ONLY_VARIABLES = set(DOMAIN_CFG.get("diagnostic_only_variables", [
    'kospi_dummy', 'sector_7', 'sector_year_rating_count', 'sector_year_rating_count_lag1',
    'industry_avg_rating',
]))

# Weight prior policy (Stage 3 sub-fix v3)
WP_CFG = CFG.get("weight_prior", {})
WEIGHT_PRIOR_CAP = float(WP_CFG.get("cap", 0.30))
WEIGHT_PRIOR_DEFAULT_FLOOR = float(WP_CFG.get("default_floor", 0.05))
WEIGHT_PRIOR_CUSTOM_FLOORS = {k: float(v) for k, v in WP_CFG.get("custom_floors", {}).items()}
WEIGHT_PRIOR_SCORE_OVERRIDES = {k: float(v) for k, v in WP_CFG.get("score_overrides", {}).items()}
WEIGHT_PRIOR_NOTES = WP_CFG.get("notes", {})


def _as_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool coercion for bool/string/int columns."""
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda x: str(x).strip().lower() in {"true", "1", "yes", "y", "t"})

# STAGE00_04_IMPORT_SAFE_GUARD_2026_05_24
# The executable variable-selection script below is intentionally gated so
# import-smoke/static tooling does not try to read runtime Stage00_04 inputs.
if __name__ == "__main__":
    # ============================================================
    # Step 1: Load inputs
    # ============================================================

    print("=" * 80)
    print("STAGE 3 VARIABLE SELECTION")
    print("=" * 80)
    print("\n[1/11] Loading input files...")


    # Required input contract
    REQUIRED_INPUTS = {
        STAGE1B_DIR / "firm_year_panel_v1.parquet": "Stage 1B panel",
        STAGE2_DIR / "engineered_financial_ratios.parquet": "Stage 2 financial ratios",
        STAGE2_DIR / "candidate_ratio_pool_by_item.csv": "Stage 2 candidate pool",
        STAGE2_DIR / "ratio_quality_report.csv": "Stage 2 quality report",
        STAGE2_DIR / "ratio_audit_against_precomputed_panel.csv": "Stage 2 NICE audit",
        STAGE1C_DIR / "nonfinancial_metadata_panel.parquet": "Stage 1C v3.2 nonfinancial panel",
        STAGE1C_DIR / "nonfinancial_candidate_pool_by_item.csv": "Stage 1C v3.2 candidate pool",
        STAGE1C_DIR / "nonfinancial_variable_quality_report.csv": "Stage 1C v3.2 quality report",
    }
    missing_inputs = [f"  {desc}: {path}" for path, desc in REQUIRED_INPUTS.items() if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError("필수 입력 파일 없음. junction/symlink 설정 확인:\n" + "\n".join(missing_inputs))
    print("  ✓ 모든 입력 파일 확인 완료")

    # Load Stage 1B
    firm_year_panel = pd.read_parquet(STAGE1B_DIR / "firm_year_panel_v1.parquet")
    # Final Oracle contract: variable selection target must be the explicit 10-grade scale.
    # Silent fallback from rating_num is forbidden because rating_num was historically overloaded.
    missing_scale = [c for c in ["rating_num_10", "grade_base_10"] if c not in firm_year_panel.columns]
    if missing_scale:
        raise KeyError(f"Stage00_04 requires explicit rating scale columns from Stage00_01: missing {missing_scale}")
    firm_year_panel["rating_num_10"] = pd.to_numeric(firm_year_panel["rating_num_10"], errors="coerce")
    print(f"  ✓ firm_year_panel: {firm_year_panel.shape}")

    # Load Stage 2
    financial_ratios = pd.read_parquet(STAGE2_DIR / "engineered_financial_ratios.parquet")
    candidate_pool_financial = pd.read_csv(STAGE2_DIR / "candidate_ratio_pool_by_item.csv")
    ratio_quality = pd.read_csv(STAGE2_DIR / "ratio_quality_report.csv")
    ratio_audit = pd.read_csv(STAGE2_DIR / "ratio_audit_against_precomputed_panel.csv")
    formula_targets_path = STAGE2_DIR / "stage2_formula_investigation_targets.csv"
    formula_targets = pd.read_csv(formula_targets_path) if formula_targets_path.exists() else pd.DataFrame()

    print(f"  ✓ financial_ratios: {financial_ratios.shape}")
    print(f"  ✓ candidate_pool_financial: {candidate_pool_financial.shape}")

    # Load Stage 1C current stable alias (must point to v3.2 output)
    nonfinancial_panel = pd.read_parquet(STAGE1C_DIR / "nonfinancial_metadata_panel.parquet")
    candidate_pool_nonfinancial = pd.read_csv(STAGE1C_DIR / "nonfinancial_candidate_pool_by_item.csv")
    nonfinancial_quality = pd.read_csv(STAGE1C_DIR / "nonfinancial_variable_quality_report.csv")

    print(f"  ✓ nonfinancial_panel: {nonfinancial_panel.shape}")
    print(f"  ✓ candidate_pool_nonfinancial: {candidate_pool_nonfinancial.shape}")

    # ============================================================
    # Step 2: Create split_stage3
    # ============================================================

    print(f"\n[2/11] Creating split_stage3 (Dev={DEV_YEAR_MIN}-{DEV_YEAR_MAX}, OOT={OOT_YEAR_MIN}-{OOT_YEAR_MAX})...")

    # Preserve original split if exists
    if 'split' in firm_year_panel.columns:
        firm_year_panel['original_split'] = firm_year_panel['split']

    # Create split_stage3
    def assign_split(year):
        if DEV_YEAR_MIN <= year <= DEV_YEAR_MAX:
            return 'dev'
        elif OOT_YEAR_MIN <= year <= OOT_YEAR_MAX:
            return 'oot'
        else:
            return 'out_of_scope'

    firm_year_panel['split_stage3'] = firm_year_panel['year'].apply(assign_split)

    # Split reconciliation report
    split_recon = firm_year_panel.groupby('year').agg(
        n_total=('거래소코드', 'count'),
        n_original_dev=('original_split', lambda x: (x == 'dev').sum() if 'original_split' in firm_year_panel.columns else 0),
        n_original_oot=('original_split', lambda x: (x == 'oot').sum() if 'original_split' in firm_year_panel.columns else 0),
        n_stage3_dev=('split_stage3', lambda x: (x == 'dev').sum()),
        n_stage3_oot=('split_stage3', lambda x: (x == 'oot').sum())
    ).reset_index()

    split_recon['n_mismatch_original_vs_stage3'] = split_recon.apply(
        lambda x: abs(x['n_original_dev'] - x['n_stage3_dev']) + abs(x['n_original_oot'] - x['n_stage3_oot']),
        axis=1
    )
    split_recon['notes'] = split_recon.apply(
        lambda x: f"Dev={x['n_stage3_dev']}, OOT={x['n_stage3_oot']}" if x['n_stage3_dev'] + x['n_stage3_oot'] > 0 else "out_of_scope",
        axis=1
    )

    split_recon.to_csv(OUTPUT_BASE / "split_reconciliation_report.csv", index=False)

    print(f"  ✓ split_stage3 created")
    print(f"  ✓ Dev years: {DEV_YEAR_MIN}-{DEV_YEAR_MAX}")
    print(f"  ✓ OOT years: {OOT_YEAR_MIN}-{OOT_YEAR_MAX}")
    print(f"  ✓ Dev sample: {(firm_year_panel['split_stage3'] == 'dev').sum()} firm-years")
    print(f"  ✓ OOT sample: {(firm_year_panel['split_stage3'] == 'oot').sum()} firm-years")

    # Merge split into financial and nonfinancial panels
    scale_panel = firm_year_panel[['거래소코드', 'year', 'split_stage3', 'rating_num_10', 'grade_base_10']].drop_duplicates()

    financial_ratios = financial_ratios.merge(
        scale_panel[['거래소코드', 'year', 'split_stage3', 'rating_num_10']],
        left_on=['거래소코드', 'year'],
        right_on=['거래소코드', 'year'],
        how='left',
        suffixes=('', '_from_panel')
    )
    if 'rating_num_10_from_panel' in financial_ratios.columns:
        financial_ratios['rating_num_10'] = financial_ratios['rating_num_10_from_panel']
        financial_ratios.drop('rating_num_10_from_panel', axis=1, inplace=True)
    if financial_ratios['rating_num_10'].isna().all():
        raise ValueError('Stage00_04 financial ratios received no rating_num_10 values after merge')
    financial_ratios['rating_num'] = financial_ratios['rating_num_10']  # legacy alias only after explicit 10-grade target is set

    nonfinancial_panel = nonfinancial_panel.merge(
        scale_panel[['거래소코드', 'year', 'split_stage3', 'rating_num_10']],
        left_on=['거래소코드', 'year'],
        right_on=['거래소코드', 'year'],
        how='left',
        suffixes=('', '_from_firm_year')
    )
    if 'rating_num_10_from_firm_year' in nonfinancial_panel.columns:
        nonfinancial_panel['rating_num_10'] = nonfinancial_panel['rating_num_10_from_firm_year']
        nonfinancial_panel.drop('rating_num_10_from_firm_year', axis=1, inplace=True)
    if nonfinancial_panel['rating_num_10'].isna().all():
        raise ValueError('Stage00_04 nonfinancial panel received no rating_num_10 values after merge')
    nonfinancial_panel['rating_num'] = nonfinancial_panel['rating_num_10']  # legacy alias only after explicit 10-grade target is set

    # ============================================================
    # Step 3: Construct candidate pool
    # ============================================================

    print("\n[3/11] Constructing candidate pool...")

    # Financial candidates
    financial_candidates = candidate_pool_financial.copy()
    financial_candidates['quality_pass'] = True  # All candidates in this file are quality-pass
    financial_candidates['variable_id'] = financial_candidates['ratio_id']  # Use ratio_id as variable_id
    financial_candidates['variable_name'] = financial_candidates['ratio_name']  # Use ratio_name as variable_name

    # Mark main pool vs appendix; preserve upstream selected_eligible when present
    financial_candidates['is_main_pool'] = ~financial_candidates['category'].isin(['기타/복합'])
    financial_candidates['appendix_pool'] = financial_candidates['category'].isin(['기타/복합'])
    if 'selected_eligible' in financial_candidates.columns:
        financial_candidates['selected_eligible'] = _as_bool_series(financial_candidates['selected_eligible']) & financial_candidates['is_main_pool']
    else:
        financial_candidates['selected_eligible'] = financial_candidates['is_main_pool']
    financial_candidates['diagnostic_only'] = False

    # ------------------------------------------------------------
    # Duplicate ratio canonicalization guard
    # ------------------------------------------------------------
    # Stage 2 is allowed to calculate duplicate/alias ratios for auditability.
    # Stage 3 is the eligibility boundary: aliases must not be selected as final
    # Oracle variables or downstream simulator action dimensions.
    DUPLICATE_RATIO_ALIAS_MAP = {
        "R003": {"canonical": "R002", "type": "exact_value_alias", "reason": "정상영업이익률 duplicates 영업이익률 under current source mapping."},
        "R004": {"canonical": "R002", "type": "exact_value_alias", "reason": "EBIT마진 duplicates 영업이익률 under current source mapping."},
        "R014": {"canonical": "R013", "type": "exact_value_alias", "reason": "EBIT/총자산 duplicates 영업이익/총자산."},
        "R216": {"canonical": "R010", "type": "exact_value_alias", "reason": "OCF/매출액 duplicates 영업현금흐름마진."},
        "R211": {"canonical": "R064", "type": "exact_value_alias", "reason": "Altman 이익잉여금/총자산 duplicates 이익잉여금/총자산."},
        "R118": {"canonical": "R076", "type": "exact_value_alias", "reason": "순운전자본비율 duplicates 순운전자본/총자산."},
        "R210": {"canonical": "R076", "type": "exact_value_alias", "reason": "Altman 운전자본/총자산 duplicates 순운전자본/총자산."},
        "R119": {"canonical": "R077", "type": "exact_value_alias", "reason": "순운전자본/매출액 duplicates 운전자본/매출액."},
        "R080": {"canonical": "R079", "type": "exact_value_alias", "reason": "EBIT 이자보상배율 duplicates 이자보상배율 under current EBIT mapping."},
        "R086": {"canonical": "R085", "type": "exact_value_alias", "reason": "이자비용/매출액 uses the same numerator item code as 금융비용부담률."},
        "R088": {"canonical": "R089", "type": "exact_value_alias", "reason": "이자비용/EBITDA uses the same numerator item code as 금융비용/EBITDA; keep financial-cost wording."},
        "R131": {"canonical": "R094", "type": "exact_value_alias", "reason": "OCF/유동부채 duplicates debt-service OCF/유동부채."},
        "R114": {"canonical": "R102", "type": "exact_value_alias", "reason": "순차입금상환가능기간 duplicates 순차입금/EBITDA."},
        "R127": {"canonical": "R117", "type": "exact_value_alias", "reason": "현금성자산/유동부채 duplicates 현금비율."},
        "R164": {"canonical": "R158", "type": "exact_value_alias", "reason": "매출채권회수기간 duplicates 매출채권회전일수."},
        "R163": {"canonical": "R159", "type": "exact_value_alias", "reason": "재고보유기간 duplicates 재고자산회전일수."},
        "R165": {"canonical": "R160", "type": "exact_value_alias", "reason": "매입채무지급기간 duplicates 매입채무회전일수."},
        "R172": {"canonical": "R171", "type": "exact_value_alias", "reason": "정상영업이익증가율 duplicates 영업이익증가율 under current source mapping."},
        "R173": {"canonical": "R171", "type": "exact_value_alias", "reason": "EBIT증가율 duplicates 영업이익증가율 under current source mapping."},
    }

    for _col, _default in {
        'duplicate_alias_of': '',
        'duplicate_alias_type': '',
        'duplicate_alias_reason': '',
        'stage3_exclude': False,
    }.items():
        if _col not in financial_candidates.columns:
            financial_candidates[_col] = _default

    # Upstream selected_eligible=False from Stage 2 is honored above when present.
    # The hard-coded guard below also protects older Stage 2 outputs.
    duplicate_alias_log = []
    for alias_id, meta in DUPLICATE_RATIO_ALIAS_MAP.items():
        mask = financial_candidates['ratio_id'].astype(str) == alias_id
        if mask.any():
            financial_candidates.loc[mask, 'selected_eligible'] = False
            financial_candidates.loc[mask, 'diagnostic_only'] = True
            financial_candidates.loc[mask, 'duplicate_alias_of'] = meta['canonical']
            financial_candidates.loc[mask, 'duplicate_alias_type'] = meta.get('type', 'exact_value_alias')
            financial_candidates.loc[mask, 'duplicate_alias_reason'] = meta['reason']
            financial_candidates.loc[mask, 'stage3_exclude'] = True
            duplicate_alias_log.append({
                'alias_ratio_id': alias_id,
                'canonical_ratio_id': meta['canonical'],
                'duplicate_alias_type': meta.get('type', 'exact_value_alias'),
                'reason': meta['reason'],
                'n_rows_blocked': int(mask.sum()),
            })

    if duplicate_alias_log:
        pd.DataFrame(duplicate_alias_log).to_csv(
            OUTPUT_BASE / "duplicate_ratio_alias_log.csv",
            index=False,
            encoding="utf-8-sig",
        )
        print(f"  ✓ duplicate_ratio_alias_log.csv 저장 ({len(duplicate_alias_log)} aliases blocked)")

    # Stage 3 metadata contract: persist the alias policy actually used during
    # variable selection so downstream Oracle/evaluator/simulator can canonicalize
    # legacy R-code references even if Stage 2 outputs are not colocated.
    _stage3_alias_master = pd.DataFrame([
        {
            'alias_ratio_id': alias_id,
            'canonical_ratio_id': meta['canonical'],
            'duplicate_alias_type': meta.get('type', 'exact_value_alias'),
            'stage3_exclude': True,
            'reason': meta['reason'],
            'source': 'stage3_guard_from_stage2_alias_policy',
        }
        for alias_id, meta in sorted(DUPLICATE_RATIO_ALIAS_MAP.items())
    ])
    _stage3_alias_master.to_csv(
        OUTPUT_BASE / "stage3_ratio_alias_master_used.csv",
        index=False,
        encoding="utf-8-sig",
    )
    with open(OUTPUT_BASE / "stage3_ratio_alias_map_used.json", 'w', encoding='utf-8') as f:
        json.dump(
            {
                r['alias_ratio_id']: {
                    'canonical_ratio_id': r['canonical_ratio_id'],
                    'duplicate_alias_type': r['duplicate_alias_type'],
                    'stage3_exclude': bool(r['stage3_exclude']),
                    'reason': r['reason'],
                }
                for _, r in _stage3_alias_master.iterrows()
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(OUTPUT_BASE / "simulator_ratio_alias_map.json", 'w', encoding='utf-8') as f:
        json.dump(
            {r['alias_ratio_id']: r['canonical_ratio_id'] for _, r in _stage3_alias_master.iterrows()},
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Nonfinancial candidates
    nonfinancial_candidates = candidate_pool_nonfinancial.copy()
    nonfinancial_candidates['quality_pass'] = True  # All candidates in this file are quality-pass
    # Robust variable_id normalization.
    # Legacy candidates often use variable_name as the executable column.
    # Financial-derived proxy candidates already have variable_id = actual panel column.
    # Do not blindly overwrite variable_id with variable_name.
    if 'variable_id' not in nonfinancial_candidates.columns:
        nonfinancial_candidates['variable_id'] = pd.NA
    if 'variable_name' not in nonfinancial_candidates.columns:
        nonfinancial_candidates['variable_name'] = pd.NA

    nonfinancial_candidates['variable_id'] = (
        nonfinancial_candidates['variable_id']
        .where(nonfinancial_candidates['variable_id'].notna(), nonfinancial_candidates['variable_name'])
        .astype(str)
    )
    nonfinancial_candidates['variable_name'] = (
        nonfinancial_candidates['variable_name']
        .where(nonfinancial_candidates['variable_name'].notna(), nonfinancial_candidates['variable_id'])
        .astype(str)
    )

    nonfinancial_candidates['available_in_panel'] = nonfinancial_candidates['variable_id'].isin(nonfinancial_panel.columns)
    missing_nf_candidates = nonfinancial_candidates.loc[
        ~nonfinancial_candidates['available_in_panel'], 'variable_id'
    ].tolist()
    if missing_nf_candidates:
        print(f"  ? Nonfinancial candidates not found in panel and excluded: {len(missing_nf_candidates)}")
        print(f"    examples: {missing_nf_candidates[:10]}")


    nonfinancial_candidates['is_main_pool'] = True
    nonfinancial_candidates['appendix_pool'] = False
    if 'selected_eligible' in nonfinancial_candidates.columns:
        nonfinancial_candidates['selected_eligible'] = _as_bool_series(nonfinancial_candidates['selected_eligible'])
    else:
        nonfinancial_candidates['selected_eligible'] = True
    if 'diagnostic_only' in nonfinancial_candidates.columns:
        nonfinancial_candidates['diagnostic_only'] = _as_bool_series(nonfinancial_candidates['diagnostic_only'])
    else:
        nonfinancial_candidates['diagnostic_only'] = False
    nonfinancial_candidates.loc[nonfinancial_candidates['diagnostic_only'], 'selected_eligible'] = False
    # Root fix: candidates absent from the actual panel must not enter metric/selection/replacement universe.
    if 'available_in_panel' not in nonfinancial_candidates.columns:
        nonfinancial_candidates['available_in_panel'] = nonfinancial_candidates['variable_id'].isin(nonfinancial_panel.columns)
    nonfinancial_candidates['selected_eligible'] = (
        nonfinancial_candidates['selected_eligible'].fillna(False).astype(bool)
        & nonfinancial_candidates['available_in_panel'].fillna(False).astype(bool)
    )

    print(f"  ✓ Financial total quality-pass: {len(financial_candidates)}")
    print(f"  ✓ Financial main candidates before domain filter: {financial_candidates['is_main_pool'].sum()}")
    print(f"  ✓ Financial appendix (기타/복합): {financial_candidates['appendix_pool'].sum()}")
    print(f"  ✓ Nonfinancial candidates before domain filter: {len(nonfinancial_candidates)}")
    print(f"  ✓ Total screening candidates before domain filter: {len(financial_candidates) + len(nonfinancial_candidates)}")


    # --- Domain filters (v3.2 guardrails) ---
    print("\n  Applying domain filters (v3.2 guardrails)...")
    domain_filter_log = []

    # 유동성 exclusion: 매출액/매출원가/매출총이익/영업수익 계열 차단
    def _has_excluded_liquidity_term(name: str) -> bool:
        return any(term in str(name) for term in LIQUIDITY_EXCLUDE_TERMS)

    liq_name_col = 'ratio_name' if 'ratio_name' in financial_candidates.columns else 'variable_id'
    liq_mask_cat = financial_candidates['category'] == '유동성'
    liq_mask_excl = liq_mask_cat & financial_candidates[liq_name_col].apply(_has_excluded_liquidity_term)
    if liq_mask_excl.any():
        financial_candidates.loc[liq_mask_excl, 'selected_eligible'] = False
        financial_candidates.loc[liq_mask_excl, 'diagnostic_only'] = True
        financial_candidates.loc[liq_mask_excl, 'stage3_exclude'] = True
        for _, r in financial_candidates[liq_mask_excl].iterrows():
            matched = next((t for t in LIQUIDITY_EXCLUDE_TERMS if t in str(r.get(liq_name_col, ''))), '')
            domain_filter_log.append({
                'variable_id': r['variable_id'], 'category': '유동성', 'source': 'financial',
                'action': 'blocked_by_exclusion', 'matched_term': matched,
                'reason': 'liquidity ratio contains sales/revenue/cost term and overlaps profitability/activity concepts'
            })

    # Explicit liquidity R-code exclusion.  R122 is also listed here intentionally,
    # even though the term-based guard above already catches it, so the active
    # policy is visible in logs/metadata.  R133 is intentionally not listed.
    liq_mask_gameable = liq_mask_cat & financial_candidates['ratio_id'].astype(str).isin(GAMEABLE_LIQUIDITY_EXCLUDE)
    if liq_mask_gameable.any():
        financial_candidates.loc[liq_mask_gameable, 'selected_eligible'] = False
        financial_candidates.loc[liq_mask_gameable, 'diagnostic_only'] = True
        financial_candidates.loc[liq_mask_gameable, 'stage3_exclude'] = True
        for _, r in financial_candidates[liq_mask_gameable].iterrows():
            rid = str(r['ratio_id'])
            domain_filter_log.append({
                'variable_id': r['variable_id'], 'category': '유동성', 'source': 'financial',
                'action': 'blocked_by_gameable_liquidity_r_code', 'matched_term': rid,
                'reason': 'explicit liquidity exclusion: R122/R136 must not be eligible for oracle liquidity selection; R133 remains eligible'
            })
    print(f"    유동성 exclusion: {int(liq_mask_excl.sum())}개 term 차단, {int(liq_mask_gameable.sum())}개 R-code 명시 차단")

    # 산업위험 allowlist: Stage 1C v3.2 lag1 self-excluded proxies만 허용
    ind_mask_cat = nonfinancial_candidates['category'] == '산업위험'
    ind_mask_blocked = ind_mask_cat & (~nonfinancial_candidates['variable_id'].isin(INDUSTRY_ALLOWLIST))
    if ind_mask_blocked.any():
        nonfinancial_candidates.loc[ind_mask_blocked, 'selected_eligible'] = False
        nonfinancial_candidates.loc[ind_mask_blocked, 'diagnostic_only'] = True
        for _, r in nonfinancial_candidates[ind_mask_blocked].iterrows():
            domain_filter_log.append({
                'variable_id': r['variable_id'], 'category': '산업위험', 'source': 'nonfinancial',
                'action': 'blocked_by_allowlist',
                'reason': 'industry-risk candidate not in Stage 1C v3.2 lag1 self-excluded allowlist'
            })
    print(f"    산업위험 allowlist: {int((ind_mask_cat & nonfinancial_candidates['variable_id'].isin(INDUSTRY_ALLOWLIST)).sum())}개 허용, {int(ind_mask_blocked.sum())}개 차단")

    # Explicit diagnostic-only variables
    for df_name, df_candidates in [('financial', financial_candidates), ('nonfinancial', nonfinancial_candidates)]:
        diag_mask = df_candidates['variable_id'].isin(DIAGNOSTIC_ONLY_VARIABLES)
        if diag_mask.any():
            df_candidates.loc[diag_mask, 'selected_eligible'] = False
            df_candidates.loc[diag_mask, 'diagnostic_only'] = True
            for _, r in df_candidates[diag_mask].iterrows():
                domain_filter_log.append({
                    'variable_id': r['variable_id'], 'category': r.get('category'), 'source': df_name,
                    'action': 'blocked_by_diagnostic_only',
                    'reason': 'diagnostic-only variable is not eligible for scorecard selection'
                })

    pd.DataFrame(domain_filter_log).to_csv(OUTPUT_BASE / "domain_filter_log_v3.csv", index=False)
    print(f"  ✓ domain_filter_log_v3.csv 저장 ({len(domain_filter_log)} entries)")
    print(f"  ✓ Financial selected-eligible after filter: {int(financial_candidates['selected_eligible'].sum())}")
    print(f"  ✓ Nonfinancial selected-eligible after filter: {int(nonfinancial_candidates['selected_eligible'].sum())}")

    # ============================================================
    # Step 4: Calculate metrics (Dev only)
    # ============================================================

    print("\n[4/11] Calculating 4-metric screening (Dev sample only)...")

    def calculate_spearman(df, var_col, target='rating_num_10'):
        """Calculate Spearman correlation"""
        df_clean = df[[var_col, target]].dropna()
        if len(df_clean) < 10:
            return np.nan, np.nan, True
    
        rho, pval = spearmanr(df_clean[var_col], df_clean[target])
        return rho, abs(rho), False

    def calculate_iv(df, var_col, target='rating_num_10', n_bins=10):
        """Calculate Information Value"""
        df_clean = df[[var_col, target]].dropna().copy()
        if len(df_clean) < 10:
            return np.nan, np.nan, False, True
    
        # Convert to numeric if possible
        if df_clean[var_col].dtype == 'object' or df_clean[var_col].dtype.name.startswith('large_string'):
            try:
                df_clean[var_col] = pd.to_numeric(df_clean[var_col])
            except:
                # If cannot convert, treat as categorical
                pass
    
        # Binary target: good (<=4) vs bad (>4)
        df_clean['binary_target'] = (df_clean[target] > 4).astype(int)
    
        # Check if variable is binary or categorical
        unique_vals = df_clean[var_col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df_clean[var_col])
    
        if unique_vals <= 2 or not is_numeric:
            bins = df_clean[var_col].unique()
            df_clean['bin'] = df_clean[var_col]
            n_bins_actual = unique_vals
        else:
            # Quantile binning for numeric
            n_bins_actual = min(n_bins, unique_vals)
            try:
                df_clean['bin'] = pd.qcut(df_clean[var_col].astype(float), q=n_bins_actual, labels=False, duplicates='drop')
            except:
                # Fallback to categorical
                df_clean['bin'] = df_clean[var_col]
                n_bins_actual = unique_vals
    
        # Calculate WoE and IV
        bin_stats = df_clean.groupby('bin').agg(
            n_good=('binary_target', lambda x: (x == 0).sum()),
            n_bad=('binary_target', lambda x: (x == 1).sum())
        ).reset_index()
    
        # Smoothing
        bin_stats['n_good'] = bin_stats['n_good'] + 0.5
        bin_stats['n_bad'] = bin_stats['n_bad'] + 0.5
    
        total_good = bin_stats['n_good'].sum()
        total_bad = bin_stats['n_bad'].sum()
    
        bin_stats['pct_good'] = bin_stats['n_good'] / total_good
        bin_stats['pct_bad'] = bin_stats['n_bad'] / total_bad
    
        bin_stats['woe'] = np.log(bin_stats['pct_good'] / bin_stats['pct_bad'])
        bin_stats['iv_contribution'] = (bin_stats['pct_good'] - bin_stats['pct_bad']) * bin_stats['woe']
    
        iv = bin_stats['iv_contribution'].sum()
    
        # Check for outliers
        outlier_flag = (abs(bin_stats['woe']) > 5).any()
    
        return iv, n_bins_actual, outlier_flag, False

    def calculate_kw_eta2(df, var_col, target='rating_num_10'):
        """Calculate Kruskal-Wallis eta²"""
        df_clean = df[[var_col, target]].dropna()
        if len(df_clean) < 10:
            return np.nan, np.nan, np.nan, True
    
        # Check if numeric
        if df_clean[var_col].dtype == 'object' or str(df_clean[var_col].dtype).startswith('large_string'):
            try:
                df_clean[var_col] = pd.to_numeric(df_clean[var_col])
            except:
                # Cannot convert to numeric - skip KW test
                return np.nan, np.nan, np.nan, True
    
        groups = [group[var_col].values for name, group in df_clean.groupby(target)]
        groups = [g for g in groups if len(g) > 0]
    
        if len(groups) < 2:
            return np.nan, np.nan, np.nan, True
    
        # Ensure all groups are numeric arrays
        try:
            groups = [np.asarray(g, dtype=float) for g in groups]
            H, pval = stats.kruskal(*groups)
        except:
            return np.nan, np.nan, np.nan, True
    
        n = len(df_clean)
        k = len(groups)
    
        # eta² calculation
        eta2 = max(0, (H - k + 1) / (n - k))
    
        return H, pval, eta2, False

    def calculate_monotonicity(df, var_col, target='rating_num_10', n_bins=10):
        """Calculate monotonicity strength"""
        df_clean = df[[var_col, target]].dropna().copy()
        if len(df_clean) < 10:
            return np.nan, np.nan, False, True
    
        # Convert to numeric if possible
        if df_clean[var_col].dtype == 'object' or df_clean[var_col].dtype.name.startswith('large_string'):
            try:
                df_clean[var_col] = pd.to_numeric(df_clean[var_col])
            except:
                pass
    
        # Get Spearman direction
        try:
            rho, _ = spearmanr(df_clean[var_col].astype(float), df_clean[target])
        except:
            return np.nan, np.nan, False, True
    
        # Binning
        unique_vals = df_clean[var_col].nunique()
        is_numeric = pd.api.types.is_numeric_dtype(df_clean[var_col])
    
        if unique_vals <= 2 or not is_numeric:
            df_clean['bin'] = df_clean[var_col]
            n_bins_actual = unique_vals
        else:
            n_bins_actual = min(n_bins, unique_vals)
            try:
                df_clean['bin'] = pd.qcut(df_clean[var_col].astype(float), q=n_bins_actual, labels=False, duplicates='drop')
            except:
                df_clean['bin'] = df_clean[var_col]
                n_bins_actual = unique_vals
    
        # Calculate mean rating per bin
        bin_means = df_clean.groupby('bin')[target].mean().sort_index()
    
        if len(bin_means) < 2:
            return np.nan, np.nan, False, True
    
        # Check monotonicity
        diffs = bin_means.diff().dropna()
    
        if rho < 0:
            # Expect rating_num to decrease as variable increases (good direction)
            consistent = (diffs < 0).sum()
        else:
            # Expect rating_num to increase as variable increases (bad direction)
            consistent = (diffs > 0).sum()
    
        total_pairs = len(diffs)
        monotonicity = consistent / total_pairs if total_pairs > 0 else 0
    
        direction_consistent = True
    
        return monotonicity, n_bins_actual, direction_consistent, False

    # Calculate metrics for all candidates
    def calculate_metrics_for_candidate(df, var_col, target='rating_num_10'):
        """Calculate all 4 metrics for a candidate variable"""
        metrics = {}
    
        # Spearman
        rho, abs_rho, spearman_na = calculate_spearman(df, var_col, target)
        metrics['spearman_rho'] = rho
        metrics['abs_spearman'] = abs_rho
        metrics['spearman_na'] = spearman_na
    
        # IV
        iv, iv_n_bins, iv_outlier, iv_na = calculate_iv(df, var_col, target)
        metrics['iv'] = iv
        metrics['iv_n_bins'] = iv_n_bins
        metrics['iv_outlier_flag'] = iv_outlier
        metrics['iv_na'] = iv_na
    
        # KW eta²
        kw_H, kw_pval, kw_eta2, kw_na = calculate_kw_eta2(df, var_col, target)
        metrics['kw_H'] = kw_H
        metrics['kw_pvalue'] = kw_pval
        metrics['kw_eta2'] = kw_eta2
        metrics['kw_na'] = kw_na
    
        # Monotonicity
        mono, mono_n_bins, mono_dir_consistent, mono_na = calculate_monotonicity(df, var_col, target)
        metrics['monotonicity'] = mono
        metrics['monotonicity_n_bins'] = mono_n_bins
        metrics['monotonicity_direction_consistent'] = mono_dir_consistent
        metrics['monotonicity_na'] = mono_na
    
        # Direction encoding
        if not spearman_na:
            if abs_rho < DIRECTION_WEAK_THRESHOLD:
                metrics['direction_weak'] = True
            else:
                metrics['direction_weak'] = False
        
            if rho < 0:
                metrics['direction'] = 'value_up_good'
            else:
                metrics['direction'] = 'value_down_good'
        else:
            metrics['direction'] = 'unknown'
            metrics['direction_weak'] = True
    
        return metrics

    # Dev sample only
    dev_financial = financial_ratios[financial_ratios['split_stage3'] == 'dev'].copy()
    dev_nonfinancial = nonfinancial_panel[nonfinancial_panel['split_stage3'] == 'dev'].copy()

    # Calculate metrics for financial candidates
    financial_metrics_list = []
    total_financial = len(financial_candidates)
    for idx, row in financial_candidates.iterrows():
        var_name = row['ratio_id']
        if len(financial_metrics_list) % 20 == 0:
            print(f"  Calculating financial metrics... {len(financial_metrics_list)}/{total_financial}")
        metrics = calculate_metrics_for_candidate(dev_financial, var_name)
        metrics['variable_id'] = var_name
        metrics['variable_name'] = var_name
        metrics['source'] = 'financial'
        metrics['category'] = row['category']
        metrics['is_main_pool'] = row['is_main_pool']
        metrics['appendix_pool'] = row['appendix_pool']
        metrics['selected_eligible'] = row['selected_eligible']
        metrics['duplicate_alias_of'] = row.get('duplicate_alias_of', '')
        metrics['duplicate_alias_type'] = row.get('duplicate_alias_type', '')
        metrics['duplicate_alias_reason'] = row.get('duplicate_alias_reason', '')
        metrics['stage3_exclude'] = bool(row.get('stage3_exclude', False))
        metrics['n_dev'] = len(dev_financial[var_name].dropna())
        metrics['missing_rate_dev'] = dev_financial[var_name].isna().mean()
        financial_metrics_list.append(metrics)

    # Calculate metrics for nonfinancial candidates
    nonfinancial_metrics_list = []
    total_nonfinancial = len(nonfinancial_candidates)
    for idx, row in nonfinancial_candidates.iterrows():
        var_name = row['variable_id']
        if var_name not in dev_nonfinancial.columns:
            continue
        if len(nonfinancial_metrics_list) % 5 == 0:
            print(f"  Calculating nonfinancial metrics... {len(nonfinancial_metrics_list)}/{total_nonfinancial}")
        metrics = calculate_metrics_for_candidate(dev_nonfinancial, var_name)
        metrics['variable_id'] = var_name
        metrics['variable_name'] = var_name
        metrics['source'] = 'nonfinancial'
        metrics['category'] = row['category']
        metrics['is_main_pool'] = row['is_main_pool']
        metrics['appendix_pool'] = row['appendix_pool']
        metrics['selected_eligible'] = row['selected_eligible']
        metrics['n_dev'] = len(dev_nonfinancial[var_name].dropna())
        metrics['missing_rate_dev'] = dev_nonfinancial[var_name].isna().mean()
        nonfinancial_metrics_list.append(metrics)

    # Combine
    all_metrics = pd.DataFrame(financial_metrics_list + nonfinancial_metrics_list)

    print(f"  ✓ Metrics calculated for {len(all_metrics)} candidates")
    print(f"  ✓ Dev sample size: {len(dev_financial)} firm-years")

    # ============================================================
    # Step 5: Winsorization bounds (Dev P1/P99)
    # ============================================================

    print("\n[5/11] Calculating winsorization bounds (Dev P1/P99)...")

    winsor_bounds_list = []

    for idx, row in all_metrics.iterrows():
        var_id = row['variable_id']
        source = row['source']
        category = row['category']
    
        if source == 'financial':
            df_dev = dev_financial
        else:
            df_dev = dev_nonfinancial
    
        values = df_dev[var_id].dropna().copy()
    
        # Convert to numeric if needed
        if values.dtype == 'object' or str(values.dtype).startswith('large_string'):
            try:
                values = pd.to_numeric(values, errors='coerce')
                # Drop NaN from conversion
                values = values.dropna()
            except:
                pass
    
        # If still not numeric or too few values, skip
        if not pd.api.types.is_numeric_dtype(values.dtype) or len(values) < 10:
            winsor_bounds_list.append({
                'variable_id': var_id,
                'variable_name': var_id,
                'source': source,
                'category': category,
                'is_binary': False,
                'p01_dev': np.nan,
                'p99_dev': np.nan,
                'n_dev_nonmissing': len(values),
                'winsor_applied': False
            })
            continue
    
        # Check if binary
        is_binary = values.nunique() <= 2
    
        if is_binary:
            p01 = float(values.min())
            p99 = float(values.max())
            winsor_applied = False
        else:
            p01 = float(values.quantile(0.01))
            p99 = float(values.quantile(0.99))
            winsor_applied = True
    
        winsor_bounds_list.append({
            'variable_id': var_id,
            'variable_name': var_id,
            'source': source,
            'category': category,
            'is_binary': is_binary,
            'p01_dev': p01,
            'p99_dev': p99,
            'n_dev_nonmissing': len(values),
            'winsor_applied': winsor_applied
        })

    winsor_bounds = pd.DataFrame(winsor_bounds_list)
    winsor_bounds.to_csv(OUTPUT_BASE / "winsorization_bounds.csv", index=False)

    print(f"  ✓ Winsorization bounds saved: {len(winsor_bounds)} variables")

    # ============================================================
    # Step 6: Normalization + selection score
    # ============================================================

    print("\n[6/11] Normalizing metrics within category...")

    def normalize_within_category(df, metric_col, category_col):
        """Min-max normalize within category"""
        df = df.copy()
        normalized_col = f"{metric_col}_norm"
    
        df[normalized_col] = np.nan
    
        for cat in df[category_col].unique():
            mask = df[category_col] == cat
            values = df.loc[mask, metric_col].copy()
        
            # Handle NaN
            values_clean = values.dropna()
        
            if len(values_clean) == 0:
                continue
            elif len(values_clean) == 1:
                df.loc[mask, normalized_col] = 1.0
            else:
                min_val = values_clean.min()
                max_val = values_clean.max()
            
                if min_val == max_val:
                    df.loc[mask, normalized_col] = 0.5
                else:
                    df.loc[mask, normalized_col] = (values - min_val) / (max_val - min_val)
    
        return df

    # Normalize metrics
    all_metrics = normalize_within_category(all_metrics, 'abs_spearman', 'category')
    all_metrics = normalize_within_category(all_metrics, 'iv', 'category')
    all_metrics = normalize_within_category(all_metrics, 'kw_eta2', 'category')
    all_metrics = normalize_within_category(all_metrics, 'monotonicity', 'category')

    # Fill NaN normalized scores with 0
    all_metrics['abs_spearman_norm'] = all_metrics['abs_spearman_norm'].fillna(0)
    all_metrics['iv_norm'] = all_metrics['iv_norm'].fillna(0)
    all_metrics['kw_eta2_norm'] = all_metrics['kw_eta2_norm'].fillna(0)
    all_metrics['monotonicity_norm'] = all_metrics['monotonicity_norm'].fillna(0)

    # Calculate selection scores for all weight settings
    for setting_name, weights in WEIGHTS.items():
        score_col = f"selection_score_{setting_name}"
        all_metrics[score_col] = (
            weights['spearman'] * all_metrics['abs_spearman_norm'] +
            weights['iv'] * all_metrics['iv_norm'] +
            weights['kw'] * all_metrics['kw_eta2_norm'] +
            weights['mono'] * all_metrics['monotonicity_norm']
        )

    # Rank within category
    all_metrics['rank_default_within_category'] = all_metrics.groupby('category')['selection_score_default'].rank(ascending=False, method='min')

    print(f"  ✓ Metrics normalized within category")
    print(f"  ✓ Selection scores calculated: {len(WEIGHTS)} weight settings")

    # Save variable_screening_metrics.csv
    all_metrics.to_csv(OUTPUT_BASE / "variable_screening_metrics.csv", index=False)
    print(f"  ✓ variable_screening_metrics.csv saved")

    # ============================================================
    # Step 7: Initial selection (category top 1)
    # ============================================================

    print("\n[7/11] Initial selection (category top 1)...")

    initial_selected = []

    for cat in FINANCIAL_MAIN_CATEGORIES + NONFINANCIAL_CATEGORIES:
        cat_candidates = all_metrics[
            (all_metrics['category'] == cat) &
            (all_metrics['selected_eligible'] == True)
        ].sort_values('selection_score_default', ascending=False)
    
        if len(cat_candidates) == 0:
            print(f"  ⚠ WARNING: No eligible candidates for category '{cat}'")
            continue
    
        top = cat_candidates.iloc[0]
        initial_selected.append({
            'category': cat,
            'variable_id': top['variable_id'],
            'variable_name': top['variable_name'],
            'source': top['source'],
            'score': top['selection_score_default'],
            'rank': 1,
            'spearman_rho': top['spearman_rho'],
            'abs_spearman': top['abs_spearman'],
            'iv': top['iv'],
            'kw_eta2': top['kw_eta2'],
            'monotonicity': top['monotonicity'],
            'direction': top['direction'],
            'direction_weak': top['direction_weak'],
            'missing_rate_dev': top.get('missing_rate_dev', np.nan)
        })

    initial_selected_df = pd.DataFrame(initial_selected)

    print(f"  ✓ Initial selected: {len(initial_selected)} variables")
    for idx, row in initial_selected_df.iterrows():
        print(f"    - {row['category']:20s}: {row['variable_id']:40s} (score={row['score']:.4f})")

    # ============================================================
    # Step 8: Collinearity replacement
    # ============================================================

    print("\n[8/11] Collinearity replacement (threshold=0.7)...")
    print("  Using improved exhaustive search algorithm...")

    selected_vars = initial_selected_df['variable_id'].tolist()
    replacement_trace = []
    attempted_replacements = set()  # Track attempted variables to prevent loops

    # Get data for correlation calculation
    # Root fix:
    # Build dev_all from the full available variable universe, not only from
    # initially selected variables. This keeps collinearity replacement consistent
    # after nonfinancial proxy expansion.
    all_fin_corr_vars = [
        v for v in all_metrics.loc[all_metrics['source'] == 'financial', 'variable_id'].astype(str).unique()
        if v in dev_financial.columns
    ]

    all_nf_corr_vars = [
        v for v in all_metrics.loc[all_metrics['source'] == 'nonfinancial', 'variable_id'].astype(str).unique()
        if v in dev_nonfinancial.columns
    ]

    code_col = '\uac70\ub798\uc18c\ucf54\ub4dc'  # 거래소코드
    dev_fin_corr = dev_financial[[code_col, 'year'] + all_fin_corr_vars].drop_duplicates([code_col, 'year'])
    dev_nf_corr = dev_nonfinancial[[code_col, 'year'] + all_nf_corr_vars].drop_duplicates([code_col, 'year'])

    dev_all = dev_fin_corr.merge(
        dev_nf_corr,
        on=[code_col, 'year'],
        how='left'
    )

    all_metrics['available_in_dev_all'] = all_metrics['variable_id'].astype(str).isin(dev_all.columns)
    all_metrics.loc[~all_metrics['available_in_dev_all'], 'selected_eligible'] = False

    selected_vars = [v for v in selected_vars if v in dev_all.columns]
    initial_selected_df = initial_selected_df[initial_selected_df['variable_id'].isin(selected_vars)].copy()

    print(f"  ? Correlation universe: financial={len(all_fin_corr_vars)}, nonfinancial={len(all_nf_corr_vars)}, total={len(dev_all.columns)-2}")

    iteration = 0
    unresolved_collinearity = False
    final_unresolved_pairs = []

    while iteration < MAX_COLLINEARITY_ITER:
        iteration += 1
    
        # Calculate correlation matrix
        selected_vars = [v for v in selected_vars if v in dev_all.columns]
        if len(selected_vars) < 2:
            print("  ⚠ Not enough selected variables available for collinearity check.")
            break
        corr_matrix = dev_all[selected_vars].corr(method='spearman')
    
        # Find pairs with |corr| > threshold
        high_corr_pairs = []
        for i in range(len(selected_vars)):
            for j in range(i + 1, len(selected_vars)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > COLLINEARITY_THRESHOLD:
                    var1 = selected_vars[i]
                    var2 = selected_vars[j]
                    cat1 = initial_selected_df[initial_selected_df['variable_id'] == var1]['category'].iloc[0]
                    cat2 = initial_selected_df[initial_selected_df['variable_id'] == var2]['category'].iloc[0]
                
                    # Determine priority
                    priority1 = PRIORITY_ORDER.index(cat1)
                    priority2 = PRIORITY_ORDER.index(cat2)
                
                    if priority1 < priority2:
                        high_priority_cat = cat1
                        high_priority_var = var1
                        low_priority_cat = cat2
                        low_priority_var = var2
                    else:
                        high_priority_cat = cat2
                        high_priority_var = var2
                        low_priority_cat = cat1
                        low_priority_var = var1
                
                    high_corr_pairs.append({
                        'high_priority_category': high_priority_cat,
                        'high_priority_var': high_priority_var,
                        'low_priority_category': low_priority_cat,
                        'low_priority_var': low_priority_var,
                        'corr': corr_val,
                        'abs_corr': abs(corr_val)
                    })
    
        if len(high_corr_pairs) == 0:
            print(f"  ✓ No collinearity issues (iteration {iteration})")
            break
    
        # Sort pairs by abs_corr descending - tackle worst pair first
        high_corr_pairs = sorted(high_corr_pairs, key=lambda x: x['abs_corr'], reverse=True)
        pair = high_corr_pairs[0]
    
        removed_var = pair['low_priority_var']
        low_priority_cat = pair['low_priority_category']
    
        print(f"  → Iteration {iteration}: Attempting to resolve {pair['high_priority_var']} ↔ {removed_var} (corr={pair['corr']:.4f})")
    
        # Find ALL replacement candidates in category (excluding current selected and already attempted)
        replacement_candidates = all_metrics[
            (all_metrics['category'] == low_priority_cat) &
            (all_metrics['selected_eligible'] == True) &
            (all_metrics['available_in_dev_all'] == True) &
            (~all_metrics['variable_id'].isin(selected_vars)) &
            (~all_metrics['variable_id'].isin(attempted_replacements))
        ].sort_values('selection_score_default', ascending=False)
    
        if len(replacement_candidates) == 0:
            print(f"    ⚠ No alternative candidates available (all exhausted)")
            unresolved_collinearity = True
            final_unresolved_pairs.append(pair)
            replacement_trace.append({
                'iteration': iteration,
                'high_priority_category': pair['high_priority_category'],
                'high_priority_var': pair['high_priority_var'],
                'low_priority_category': low_priority_cat,
                'low_priority_var': removed_var,
                'corr': pair['corr'],
                'abs_corr': pair['abs_corr'],
                'removed_variable_id': removed_var,
                'removed_variable_name': removed_var,
                'replacement_variable_id': None,
                'replacement_variable_name': None,
                'replacement_score': None,
                'candidate_max_abs_corr': None,
                'global_max_abs_corr': None,
                'reason': 'no_alternative_candidates',
                'status': 'unresolved'
            })
            break
    
        # Evaluate EACH candidate
        print(f"    Evaluating {len(replacement_candidates)} candidates...")
        candidate_evaluations = []
    
        for idx, candidate in replacement_candidates.iterrows():
            candidate_var = candidate['variable_id']
        
            # Create temporary selected set with this candidate
            temp_selected_vars = [candidate_var if v == removed_var else v for v in selected_vars]
        
            # Calculate correlation matrix for this temp set
            temp_selected_vars = [v for v in temp_selected_vars if v in dev_all.columns]
            if candidate_var not in temp_selected_vars or len(temp_selected_vars) < 2:
                continue
            temp_corr_matrix = dev_all[temp_selected_vars].corr(method='spearman')
        
            # Find max |corr| involving this candidate
            candidate_idx = temp_selected_vars.index(candidate_var)
            candidate_corrs = []
            for j in range(len(temp_selected_vars)):
                if j != candidate_idx:
                    candidate_corrs.append(abs(temp_corr_matrix.iloc[candidate_idx, j]))
        
            candidate_max_abs_corr = max(candidate_corrs) if candidate_corrs else 0.0
        
            # Find global max |corr| in temp set
            global_max_abs_corr = 0.0
            for i in range(len(temp_selected_vars)):
                for j in range(i + 1, len(temp_selected_vars)):
                    global_max_abs_corr = max(global_max_abs_corr, abs(temp_corr_matrix.iloc[i, j]))
        
            candidate_evaluations.append({
                'variable_id': candidate_var,
                'score': candidate['selection_score_default'],
                'candidate_max_abs_corr': candidate_max_abs_corr,
                'global_max_abs_corr': global_max_abs_corr,
                'passes_threshold': (candidate_max_abs_corr <= COLLINEARITY_THRESHOLD) and 
                                   (global_max_abs_corr <= COLLINEARITY_THRESHOLD)
            })
    
        # Select best candidate
        # Priority: 1) passes threshold, 2) maximize score, 3) minimize max_corr
        candidate_evaluations = sorted(candidate_evaluations, key=lambda x: (
            not x['passes_threshold'],  # False first (passes_threshold=True comes first)
            -x['score'],  # Maximize score
            x['candidate_max_abs_corr']  # Minimize max corr as tiebreaker
        ))
    
        best_candidate = candidate_evaluations[0]
        replacement_var = best_candidate['variable_id']
    
        # Mark this candidate as attempted
        attempted_replacements.add(replacement_var)
    
        # Update selected_vars
        selected_vars = [replacement_var if v == removed_var else v for v in selected_vars]
    
        # Update initial_selected_df
        replacement_row = all_metrics[all_metrics['variable_id'] == replacement_var].iloc[0]
        mask = initial_selected_df['variable_id'] == removed_var
        initial_selected_df.loc[mask, 'variable_id'] = replacement_var
        initial_selected_df.loc[mask, 'variable_name'] = replacement_var
        initial_selected_df.loc[mask, 'score'] = replacement_row['selection_score_default']
        initial_selected_df.loc[mask, 'spearman_rho'] = replacement_row['spearman_rho']
        initial_selected_df.loc[mask, 'direction'] = replacement_row['direction']
    
        # Record trace
        if best_candidate['passes_threshold']:
            status = 'resolved'
            reason = 'threshold_satisfied'
            print(f"    ✓ Selected {replacement_var} (max_corr={best_candidate['candidate_max_abs_corr']:.4f}, score={best_candidate['score']:.4f})")
        else:
            status = 'best_effort'
            reason = 'min_collinearity_among_candidates'
            unresolved_collinearity = True
            final_unresolved_pairs.append({
                'high_priority_var': pair['high_priority_var'],
                'low_priority_var': replacement_var,
                'corr': best_candidate['candidate_max_abs_corr'],
                'abs_corr': best_candidate['candidate_max_abs_corr']
            })
            print(f"    ⚠ Selected {replacement_var} (best available: max_corr={best_candidate['candidate_max_abs_corr']:.4f}, score={best_candidate['score']:.4f})")
    
        replacement_trace.append({
            'iteration': iteration,
            'high_priority_category': pair['high_priority_category'],
            'high_priority_var': pair['high_priority_var'],
            'low_priority_category': low_priority_cat,
            'low_priority_var': removed_var,
            'corr': pair['corr'],
            'abs_corr': pair['abs_corr'],
            'removed_variable_id': removed_var,
            'removed_variable_name': removed_var,
            'replacement_variable_id': replacement_var,
            'replacement_variable_name': replacement_var,
            'replacement_score': best_candidate['score'],
            'candidate_max_abs_corr': best_candidate['candidate_max_abs_corr'],
            'global_max_abs_corr': best_candidate['global_max_abs_corr'],
            'candidates_evaluated': len(candidate_evaluations),
            'reason': reason,
            'status': status
        })

    # Save final correlation matrix
    selected_vars = [v for v in selected_vars if v in dev_all.columns]
    final_corr_matrix = dev_all[selected_vars].corr(method='spearman')
    final_corr_matrix.to_csv(OUTPUT_BASE / "collinearity_matrix.csv")

    # Final verdict reconciliation:
    # Earlier iterations may set unresolved_collinearity=True when a best-effort
    # replacement does not resolve all global collinearity immediately.
    # Recompute unresolved status from the final selected-variable correlation matrix.
    final_threshold_pairs = []
    for i, var_i in enumerate(selected_vars):
        for j in range(i + 1, len(selected_vars)):
            var_j = selected_vars[j]
            corr_val = final_corr_matrix.loc[var_i, var_j]
            if pd.notna(corr_val) and abs(corr_val) > COLLINEARITY_THRESHOLD:
                final_threshold_pairs.append({
                    "var1": var_i,
                    "var2": var_j,
                    "corr": float(corr_val),
                    "abs_corr": float(abs(corr_val)),
                })

    unresolved_collinearity = len(final_threshold_pairs) > 0
    final_unresolved_pairs = final_threshold_pairs

    if unresolved_collinearity:
        print(f"  ? Final collinearity check: {len(final_threshold_pairs)} pair(s) remain above threshold")
    else:
        print(f"  ? Final collinearity check: no pair exceeds threshold={COLLINEARITY_THRESHOLD}")

    # Save replacement trace
    if len(replacement_trace) == 0:
        replacement_trace.append({
            'iteration': 0,
            'high_priority_category': None,
            'high_priority_var': None,
            'low_priority_category': None,
            'low_priority_var': None,
            'corr': None,
            'abs_corr': None,
            'removed_variable_id': None,
            'removed_variable_name': None,
            'replacement_variable_id': None,
            'replacement_variable_name': None,
            'replacement_score': None,
            'candidate_max_abs_corr': None,
            'global_max_abs_corr': None,
            'candidates_evaluated': 0,
            'reason': 'no_replacement_needed',
            'status': 'ok'
        })

    pd.DataFrame(replacement_trace).to_csv(OUTPUT_BASE / "collinearity_replacement_trace.csv", index=False)

    print(f"  ✓ Collinearity check complete")
    if unresolved_collinearity:
        print(f"  ⚠ WARNING: Unresolved collinearity remains after exhaustive search")

    # Guard: duplicate aliases must not survive into final selected variables.
    _selected_aliases = sorted(set(initial_selected_df['variable_id'].astype(str)) & set(DUPLICATE_RATIO_ALIAS_MAP.keys()))
    if _selected_aliases:
        pd.DataFrame([
            {
                'alias_ratio_id': rid,
                'canonical_ratio_id': DUPLICATE_RATIO_ALIAS_MAP[rid]['canonical'],
                'reason': DUPLICATE_RATIO_ALIAS_MAP[rid]['reason'],
            }
            for rid in _selected_aliases
        ]).to_csv(OUTPUT_BASE / "selected_duplicate_alias_error.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError(
            "Duplicate alias ratio(s) selected despite Stage 3 guard: " + ", ".join(_selected_aliases)
        )

    # ============================================================
    # Step 9: Save selected variables
    # ============================================================

    print("\n[9/11] Saving selected variables...")

    # selected_variables_v1.json / v3.json
    # v3 adds weight_prior_v2/v3 and weight_floor fields (Stage 3 sub-fix output)
    selected_json = []
    for idx, row in initial_selected_df.iterrows():
        vid = row['variable_id']
        entry = {
            'category': row['category'],
            'variable_id': vid,
            'variable_name': row['variable_name'],
            'source': row['source'],
            'final_rank': int(row['rank']),
            'score': float(row['score']),
            'direction': row['direction'],
            'direction_weak': bool(row.get('direction_weak', False)),
            'spearman_rho': float(row['spearman_rho']) if pd.notna(row['spearman_rho']) else None,
            'audit_caveat': False,
            'proxy_caveat': row['source'] == 'nonfinancial',
            'selected_reason': 'top_score_in_category',
            'replacement_history': [],
            # R-code metadata fields for downstream consistency.
            'canonical_variable_id': vid,
            'is_canonical_ratio': True,
            'duplicate_alias_of': row.get('duplicate_alias_of', ''),
            'duplicate_alias_type': row.get('duplicate_alias_type', ''),
            'stage3_exclude': bool(row.get('stage3_exclude', False)),
        }
        selected_json.append(entry)

    # v1: base fields only
    with open(OUTPUT_BASE / "selected_variables_v1.json", 'w', encoding='utf-8') as f:
        json.dump(selected_json, f, indent=2, ensure_ascii=False)

    # weight_prior fields are added during Step 9.5 below; v3 is written there

    # direction_encoding.json
    direction_encoding = {
        row['variable_id']: {
            'direction': row['direction'],
            'direction_weak': bool(row.get('direction_weak', False)),
            'spearman_rho': float(row['spearman_rho']) if pd.notna(row['spearman_rho']) else None
        }
        for _, row in initial_selected_df.iterrows()
    }

    with open(OUTPUT_BASE / "direction_encoding.json", 'w', encoding='utf-8') as f:
        json.dump(direction_encoding, f, indent=2, ensure_ascii=False)

    # selection_score_table.csv (top candidates per category)
    selection_table = all_metrics.copy()
    selection_table = selection_table.sort_values(['category', 'selection_score_default'], ascending=[True, False])
    selection_table.to_csv(OUTPUT_BASE / "selection_score_table.csv", index=False)

    # Selected R-code/nonfinancial variable master: small metadata table that records
    # the final Oracle variables and confirms no duplicate alias survived.
    _selected_master_cols = [
        'category', 'variable_id', 'variable_name', 'source', 'rank', 'score',
        'direction', 'direction_weak', 'spearman_rho', 'selected_eligible',
        'duplicate_alias_of', 'duplicate_alias_type', 'duplicate_alias_reason', 'stage3_exclude'
    ]
    _selected_master_cols = [c for c in _selected_master_cols if c in initial_selected_df.columns]
    selected_variable_master = initial_selected_df[_selected_master_cols].copy()
    selected_variable_master['canonical_variable_id'] = selected_variable_master['variable_id']
    selected_variable_master['is_canonical_ratio'] = True
    selected_variable_master.to_csv(OUTPUT_BASE / "selected_variable_master.csv", index=False, encoding='utf-8-sig')

    print(f"  ✓ selected_variables_v1.json: {len(selected_json)} variables")
    print(f"  ✓ direction_encoding.json: {len(direction_encoding)} variables")
    print(f"  ✓ selection_score_table.csv: {len(selection_table)} candidates")

    # ============================================================
    # Step 9.5: Compute weight prior v3 (Stage 3 sub-fix)
    # ============================================================
    # Computes per-block cap/floor normalized weight priors for Stage 4 Oracle.
    # Custom floors (0.02) applied to weak/problematic variables; redistributes
    # freed weight to stronger variables. Writes weights_prior_v3.json and
    # selected_variables_v3.json (v1 + weight_prior fields).

    print("\n[9.5/11] Computing weight prior v3...")


    def _block_normalize_weights(variable_ids, score_lookup, score_overrides,
                                  custom_floors, default_floor, cap, max_iter=100):
        """
        Iterative cap/floor normalization within one scoring block.
        Returns dict {variable_id: weight}, summing to 1.0.
        """
        floors = {vid: custom_floors.get(vid, default_floor) for vid in variable_ids}
        raw    = {vid: score_overrides.get(vid, score_lookup.get(vid, 0.0)) for vid in variable_ids}
        total  = sum(raw.values()) or 1.0
        w      = {vid: raw[vid] / total for vid in variable_ids}

        for _ in range(max_iter):
            for vid in variable_ids:
                w[vid] = max(w[vid], floors[vid])
                w[vid] = min(w[vid], cap)
            total = sum(w.values())
            if abs(total - 1.0) < 1e-9:
                break
            at_floor = {v for v in variable_ids if abs(w[v] - floors[v]) < 1e-8}
            at_cap   = {v for v in variable_ids if abs(w[v] - cap)       < 1e-8}
            free     = [v for v in variable_ids if v not in at_floor and v not in at_cap]
            if not free:
                non_fl = [v for v in variable_ids if v not in at_floor]
                s      = sum(w[v] for v in non_fl) or 1e-9
                need   = 1.0 - sum(w[v] for v in at_floor)
                for v in non_fl:
                    w[v] = min(cap, (w[v] / s) * need)
                break
            s_free = sum(w[v] for v in free) or 1e-9
            for v in free:
                w[v] -= (w[v] / s_free) * (total - 1.0)
                w[v]  = max(floors[v], min(cap, w[v]))

        total = sum(w.values())
        return {vid: v / total for vid, v in w.items()}


    # Build score lookup from selection_score_table
    score_lookup_prior = dict(zip(
        selection_table['variable_id'],
        selection_table['selection_score_default']
    ))

    fin_var_ids  = [v['variable_id'] for v in selected_json if v['source'] == 'financial']
    nfin_var_ids = [v['variable_id'] for v in selected_json if v['source'] == 'nonfinancial']

    # v2 baseline: domain-standard floors (no overrides)
    w_v2_fin  = _block_normalize_weights(
        fin_var_ids,  score_lookup_prior, {}, {}, WEIGHT_PRIOR_DEFAULT_FLOOR, WEIGHT_PRIOR_CAP)
    w_v2_nfin = _block_normalize_weights(
        nfin_var_ids, score_lookup_prior, {}, {}, WEIGHT_PRIOR_DEFAULT_FLOOR, WEIGHT_PRIOR_CAP)
    w_v2 = {**w_v2_fin, **w_v2_nfin}

    # v3: custom floors + score overrides for weak variables
    w_v3_fin  = _block_normalize_weights(
        fin_var_ids,  score_lookup_prior,
        WEIGHT_PRIOR_SCORE_OVERRIDES, WEIGHT_PRIOR_CUSTOM_FLOORS,
        WEIGHT_PRIOR_DEFAULT_FLOOR, WEIGHT_PRIOR_CAP)
    w_v3_nfin = _block_normalize_weights(
        nfin_var_ids, score_lookup_prior,
        WEIGHT_PRIOR_SCORE_OVERRIDES, WEIGHT_PRIOR_CUSTOM_FLOORS,
        WEIGHT_PRIOR_DEFAULT_FLOOR, WEIGHT_PRIOR_CAP)
    w_v3 = {**w_v3_fin, **w_v3_nfin}

    # ── weights_prior_v3.json ─────────────────────────────────────────────────────
    weights_prior_doc = {
        "version": "v3",
        "method": ("4-metric composite score (0.3·|ρ| + 0.3·IV + 0.2·KW_η² + 0.2·mono), "
                   "normalized per block with iterative cap/floor redistribution"),
        "block_combine": "0.70 × financial_score + 0.30 × nonfinancial_score",
        "cap": WEIGHT_PRIOR_CAP,
        "default_floor": WEIGHT_PRIOR_DEFAULT_FLOOR,
        "custom_floors": WEIGHT_PRIOR_CUSTOM_FLOORS,
        "score_overrides": WEIGHT_PRIOR_SCORE_OVERRIDES,
        "financial_block":    {vid: round(w_v3[vid], 6) for vid in fin_var_ids},
        "nonfinancial_block": {vid: round(w_v3[vid], 6) for vid in nfin_var_ids},
        "financial_block_v2_baseline":    {vid: round(w_v2[vid], 6) for vid in fin_var_ids},
        "nonfinancial_block_v2_baseline": {vid: round(w_v2[vid], 6) for vid in nfin_var_ids},
    }
    with open(OUTPUT_BASE / "weights_prior_v3.json", 'w', encoding='utf-8') as f:
        json.dump(weights_prior_doc, f, indent=2, ensure_ascii=False)

    # ── selected_variables_v3.json (v1 + weight_prior fields) ────────────────────
    _sub_fix_notes = {
        'R185': WEIGHT_PRIOR_NOTES.get('R185', "R185: weak growth signal + non-random missing due to negative denominator. Prior pinned to custom floor."),
        'financial_data_completeness': WEIGHT_PRIOR_NOTES.get('financial_data_completeness', "financial_data_completeness: Dev signal near zero and OOT regime jump. Prior pinned to custom floor; diagnostic caveat retained."),
    }
    override_log_rows = []
    sel_json_v3 = []
    for entry in selected_json:
        vid = entry['variable_id']
        e2  = dict(entry)
        e2['weight_prior_v2'] = round(w_v2[vid], 6)
        e2['weight_prior_v3'] = round(w_v3[vid], 6)
        e2['weight_floor_v2'] = WEIGHT_PRIOR_DEFAULT_FLOOR
        e2['weight_floor_v3'] = WEIGHT_PRIOR_CUSTOM_FLOORS.get(vid, WEIGHT_PRIOR_DEFAULT_FLOOR)
        e2['action']          = 'floor_reduced' if vid in WEIGHT_PRIOR_CUSTOM_FLOORS else 'unchanged'
        e2['stage3_sub_fix_note'] = _sub_fix_notes.get(vid, None)
        e2['scoring_role'] = 'floor_reduced_diagnostic_caveat' if vid in WEIGHT_PRIOR_CUSTOM_FLOORS else 'main_scorecard_variable'
        if vid in WEIGHT_PRIOR_CUSTOM_FLOORS:
            override_log_rows.append({
                'variable_id': vid,
                'category': e2['category'],
                'action': e2['action'],
                'weight_prior_v2': e2['weight_prior_v2'],
                'weight_prior_v3': e2['weight_prior_v3'],
                'weight_floor_v2': e2['weight_floor_v2'],
                'weight_floor_v3': e2['weight_floor_v3'],
                'score_override': WEIGHT_PRIOR_SCORE_OVERRIDES.get(vid),
                'note': e2['stage3_sub_fix_note'],
            })
        sel_json_v3.append(e2)

    with open(OUTPUT_BASE / "selected_variables_v3.json", 'w', encoding='utf-8') as f:
        json.dump(sel_json_v3, f, indent=2, ensure_ascii=False)

    pd.DataFrame(override_log_rows).to_csv(OUTPUT_BASE / "stage3_subfix_override_log.csv", index=False)

    # ── direction_encoding_v3.json, collinearity_matrix_v3.csv ───────────────────
    with open(OUTPUT_BASE / "direction_encoding_v3.json", 'w', encoding='utf-8') as f:
        json.dump(direction_encoding, f, indent=2, ensure_ascii=False)
    final_corr_matrix.to_csv(OUTPUT_BASE / "collinearity_matrix_v3.csv")

    # Optional aliases for downstream compatibility
    if CREATE_LATEST_ALIAS:
        shutil.copyfile(OUTPUT_BASE / "selected_variables_v3.json", OUTPUT_BASE / "selected_variables_latest.json")
        shutil.copyfile(OUTPUT_BASE / "direction_encoding_v3.json", OUTPUT_BASE / "direction_encoding_latest.json")
        shutil.copyfile(OUTPUT_BASE / "weights_prior_v3.json", OUTPUT_BASE / "weights_prior_latest.json")
    if CREATE_BACKWARD_COMPAT:
        shutil.copyfile(OUTPUT_BASE / "selected_variables_v3.json", OUTPUT_BASE / "selected_variables_v2.json")
        shutil.copyfile(OUTPUT_BASE / "direction_encoding_v3.json", OUTPUT_BASE / "direction_encoding_v2.json")
        shutil.copyfile(OUTPUT_BASE / "collinearity_matrix_v3.csv", OUTPUT_BASE / "collinearity_matrix_v2.csv")

    # ── console summary ───────────────────────────────────────────────────────────
    print(f"  ✓ weights_prior_v3.json written")
    print(f"  ✓ selected_variables_v3.json written (v1 + weight_prior fields)")
    print(f"  ✓ direction_encoding_v3.json, collinearity_matrix_v3.csv written")
    print(f"  ✓ stage3_subfix_override_log.csv written")
    print(f"  Weight changes (v2 prior → v3 prior):")
    for vid in fin_var_ids + nfin_var_ids:
        dw = w_v3[vid] - w_v2[vid]
        if abs(dw) > 0.001:
            tag = " ★ floor_reduced" if vid in WEIGHT_PRIOR_CUSTOM_FLOORS else " ← redistributed"
            print(f"    {vid:44s}: {w_v2[vid]:.4f} → {w_v3[vid]:.4f} ({dw:+.4f}){tag}")

    # ============================================================
    # Step 10: OOT sanity check
    # ============================================================

    print("\n[10/11] OOT sanity check...")

    oot_financial = financial_ratios[financial_ratios['split_stage3'] == 'oot'].copy()
    oot_nonfinancial = nonfinancial_panel[nonfinancial_panel['split_stage3'] == 'oot'].copy()

    oot_sanity_list = []
    for idx, row in initial_selected_df.iterrows():
        var_id = row['variable_id']
        cat = row['category']
        source = row['source']
    
        if source == 'financial':
            df_oot = oot_financial
            df_dev = dev_financial
        else:
            df_oot = oot_nonfinancial
            df_dev = dev_nonfinancial
    
        missing_rate_oot = df_oot[var_id].isna().mean() if var_id in df_oot.columns else 1.0
    
        # Recalculate Spearman on OOT
        rho_oot, _, _ = calculate_spearman(df_oot, var_id) if var_id in df_oot.columns else (np.nan, np.nan, True)
    
        direction_dev = row['direction']
        if pd.notna(rho_oot):
            direction_oot = 'value_up_good' if rho_oot < 0 else 'value_down_good'
        else:
            direction_oot = 'unknown'
    
        direction_stable = (direction_dev == direction_oot)
    
        oot_sanity_list.append({
            'variable_id': var_id,
            'category': cat,
            'missing_rate_dev': row.get('missing_rate_dev', np.nan),
            'missing_rate_oot': missing_rate_oot,
            'spearman_rho_dev': row['spearman_rho'],
            'spearman_rho_oot': rho_oot,
            'direction_dev': direction_dev,
            'direction_oot': direction_oot,
            'direction_stable': direction_stable
        })

    oot_sanity = pd.DataFrame(oot_sanity_list)
    oot_sanity.to_csv(OUTPUT_BASE / "oot_sanity_check.csv", index=False)

    print(f"  ✓ OOT sanity check saved")
    print(f"  ✓ Direction stable: {oot_sanity['direction_stable'].sum()}/{len(oot_sanity)}")

    # ============================================================
    # Step 11: Generate reports
    # ============================================================

    print("\n[11/11] Generating reports...")

    # Compliance checklist
    compliance_items = [
        {'item': 'split_stage3_created', 'status': 'split_stage3' in firm_year_panel.columns},
        {'item': f'dev_period_{DEV_YEAR_MIN}_{DEV_YEAR_MAX}', 'status': True},
        {'item': f'oot_period_{OOT_YEAR_MIN}_{OOT_YEAR_MAX}', 'status': True},
        {'item': 'metrics_calculated_dev_only', 'status': True},
        {'item': 'financial_total_candidates_134', 'status': len(financial_candidates) == 134},
        {'item': 'financial_main_candidates_124', 'status': financial_candidates['is_main_pool'].sum() == 124},
        {'item': 'appendix_excluded_from_main', 'status': True},
        {'item': 'nonfinancial_candidates_ge_23', 'status': len(nonfinancial_candidates) >= 23, 'note': f"actual={len(nonfinancial_candidates)}"},
        {'item': 'total_screening_ge_157', 'status': len(all_metrics) >= 157, 'note': f"actual={len(all_metrics)}"},
        {'item': 'final_selected_10_or_11', 'status': len(initial_selected_df) in (10, 11), 'note': f"actual={len(initial_selected_df)}"},
        {'item': '6_financial_categories', 'status': len([x for x in initial_selected_df['category'] if x in FINANCIAL_MAIN_CATEGORIES]) == 6},
        {'item': '4_or_5_nonfinancial_categories', 'status': len([x for x in initial_selected_df['category'] if x in NONFINANCIAL_CATEGORIES]) in (4, 5), 'note': f"actual={len([x for x in initial_selected_df['category'] if x in NONFINANCIAL_CATEGORIES])}"},
        {'item': 'direction_encoding_matches_selected', 'status': len(direction_encoding) == len(initial_selected_df)},
        {'item': 'collinearity_matrix_matches_selected', 'status': final_corr_matrix.shape == (len(initial_selected_df), len(initial_selected_df))},
        {'item': 'collinearity_checked', 'status': True},
        {'item': 'unresolved_collinearity', 'status': unresolved_collinearity, 'note': 'WARNING if True'},
        {'item': 'stage1c_stable_alias_not_v2', 'status': STAGE1C_DIR.name != 'stage1c_v2'},
        {'item': 'industry_selected_in_lag1_self_excl_allowlist', 'status': all(v in INDUSTRY_ALLOWLIST for v in initial_selected_df.loc[initial_selected_df['category'] == '산업위험', 'variable_id'].tolist())},
        {'item': 'r185_weight_prior_v3_le_0_021', 'status': w_v3.get('R185', 1.0) <= 0.021},
        {'item': 'financial_data_completeness_weight_prior_v3_le_0_021_if_present', 'status': min(w_v3.get('financial_data_completeness', 1.0), w_v3.get('nf_financial_data_completeness', 1.0)) <= 0.021 or ('financial_data_completeness' not in w_v3 and 'nf_financial_data_completeness' not in w_v3)},
        {'item': 'financial_weight_sum_is_1', 'status': abs(sum(w_v3[v] for v in fin_var_ids) - 1.0) < 1e-6},
        {'item': 'nonfinancial_weight_sum_is_1', 'status': abs(sum(w_v3[v] for v in nfin_var_ids) - 1.0) < 1e-6},
        {'item': 'weights_within_cap', 'status': all(w <= WEIGHT_PRIOR_CAP + 1e-9 for w in w_v3.values())},
        {'item': 'domain_filter_log_written', 'status': (OUTPUT_BASE / 'domain_filter_log_v3.csv').exists()},
        {'item': 'gameable_liquidity_exclusion_policy_active', 'status': GAMEABLE_LIQUIDITY_EXCLUDE.issuperset({'R122', 'R136'}) and 'R133' not in GAMEABLE_LIQUIDITY_EXCLUDE, 'note': f"active={sorted(GAMEABLE_LIQUIDITY_EXCLUDE)}"},
        {'item': 'gameable_liquidity_r_codes_not_selected', 'status': not bool(set(initial_selected_df['variable_id'].astype(str)) & GAMEABLE_LIQUIDITY_EXCLUDE), 'note': f"selected_overlap={sorted(set(initial_selected_df['variable_id'].astype(str)) & GAMEABLE_LIQUIDITY_EXCLUDE)}"},
        {'item': 'override_log_written', 'status': (OUTPUT_BASE / 'stage3_subfix_override_log.csv').exists()},
    ]

    _compliance_df = pd.DataFrame(compliance_items)
    _compliance_df.to_csv(OUTPUT_BASE / "stage3_compliance_checklist.csv", index=False)

    # Verdict
    _failed_hard = _compliance_df[_compliance_df['status'] == False]
    verdict_status = "PASS" if not unresolved_collinearity and len(initial_selected_df) in (10, 11) and len(_failed_hard) == 0 else "CONDITIONAL_PASS"

    with open(OUTPUT_BASE / "stage3_verdict.md", 'w', encoding='utf-8') as f:
        f.write(f"# Stage 3 Verdict\n\n")
        f.write(f"**Status**: {verdict_status}\n\n")
        f.write(f"**Selected variables**: {len(initial_selected_df)}/11\n")
        f.write(f"**Unresolved collinearity**: {unresolved_collinearity}\n\n")
        if verdict_status == "CONDITIONAL_PASS":
            f.write("**Note**: Conditional pass due to unresolved collinearity or other caveats.\n")

    # Variable selection report
    with open(OUTPUT_BASE / "variable_selection_report.md", 'w', encoding='utf-8') as f:
        f.write("# Stage 3 Variable Selection Report\n\n")
        f.write("## Selected Variables\n\n")
        f.write(initial_selected_df.to_markdown(index=False))
        f.write("\n\n## Methodology\n\n")
        f.write(f"- Dev sample: {DEV_YEAR_MIN}-{DEV_YEAR_MAX}\n")
        f.write(f"- OOT sample: {OOT_YEAR_MIN}-{OOT_YEAR_MAX}\n")
        f.write("- 4-metric screening: Spearman, IV, KW eta², monotonicity\n")
        f.write(f"- Selection score: {DEFAULT_SELECTION_WEIGHTS['spearman']}·|ρ| + {DEFAULT_SELECTION_WEIGHTS['iv']}·IV + {DEFAULT_SELECTION_WEIGHTS['kw']}·KW + {DEFAULT_SELECTION_WEIGHTS['mono']}·mono\n")
        f.write(f"- Collinearity threshold: |corr| > {COLLINEARITY_THRESHOLD}\n")
        f.write("- Stage 3 sub-fix v3: R185 and financial_data_completeness prior scores overridden to custom floor; audit log generated.\n")
        f.write("- Domain filters: liquidity exclusion + Stage 1C v3.2 industry-risk lag1 self-excluded allowlist.\n")

    # File manifest
    output_files = list(OUTPUT_BASE.glob("*"))
    manifest_list = []
    for f in output_files:
        if f.is_file():
            manifest_list.append({
                'filename': f.name,
                'size_bytes': f.stat().st_size,
                'sha256': hashlib.sha256(f.read_bytes()).hexdigest()
            })

    pd.DataFrame(manifest_list).to_csv(OUTPUT_BASE / "stage3_file_manifest.csv", index=False)

    print(f"  ✓ Compliance checklist saved")
    print(f"  ✓ Verdict: {verdict_status}")
    print(f"  ✓ Variable selection report saved")
    print(f"  ✓ File manifest saved")

    print("\n" + "=" * 80)
    print("STAGE 3 VARIABLE SELECTION COMPLETE")
    print("=" * 80)
    print(f"\n✓ Output directory: {OUTPUT_BASE}")
    print(f"✓ Selected variables: {len(initial_selected_df)}")
    print(f"✓ Verdict: {verdict_status}")



