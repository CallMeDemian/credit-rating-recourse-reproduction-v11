from __future__ import annotations

import argparse
import json
import shutil
import yaml
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
from credit_recourse.contracts.account_registry import resolved_field_values
from credit_recourse.eval.final_stage6_multi_oracle_eval.pipeline import score_alpha, resolve_backend_artifact
from credit_recourse.oracle.verification.verify_stage1_substrate_validation import (
    KEY as STAGE1_KEY,
    RATING_NUM_10_ORIENTATION,
    SCORE_ORIENTATION,
    compute_direction_agreement,
    evaluate_stage1_backend_panel,
    load_stage1_backend_score_panel,
)
from credit_recourse.simulator.oracle_variables import compute_oracle_variables, ORACLE_FORMULA_REGISTRY


MERTON_FIDELITY_FIELDS = {"total_assets", "short_term_debt", "long_term_debt", "bonds"}
FCFF_FIDELITY_FIELDS = {"operating_cf"}
IDENTITY_FIELDS = {"total_assets", "total_liabilities", "total_equity"}
_STATE_VALUE_FIELDS = [
    f.name
    for f in fields(FirmState)
    if f.name not in {"firm_id", "year", "sector", "rating_num", "rating_grade"}
]

B2_BACKEND = "alpha"
B2_OOT_YEAR_MIN = 2020
B2_OOT_YEAR_MAX = 2023
B2_MAX_B1_GAP = 0.10
B2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT = 0.50
B2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS = 0.10
B2_PRIMARY_MISSING_SUSPECTS = ("R157", "R182")
B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE = 0.50
B2_PREV_STATE_JOIN_KEY_FIRM = "__loopb2_prev_state_firm_key"
B2_PREV_STATE_JOIN_KEY_YEAR = "__loopb2_prev_state_year"
B2_NEXT_STATE_REQUIRED_FORMULA_FIELDS = ("revenue", "non_current_assets")
B2_SCORING_STATE_FIELD_GUARD_FIELDS = (
    "prev_state.capital_stock",
    "prev_state.non_current_assets",
    "state_t1.revenue",
    "state_t1.non_current_assets",
)
B2_SELECTED_FORMULA_ID_PRESERVATION_GUARD_IDS = ("R157", "R182")
B2_SELECTED_FORMULA_MIN_NONNULL_SHARE = B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE
B2_DIRECT_NEXT_RATIO_OVERLAY_GUARD_IDS = ("R157", "R182")
B2_DIRECT_NEXT_RATIO_OVERLAY_POLICY = (
    "observed_transition_rows_only; use Stage2 mixed-transition next__R-code/R-code__next "
    "selected-ratio columns only when the deterministic FirmState formula is undefined because "
    "the handoff carries zero-filled accounting denominator sentinels"
)
B2_SCORING_POPULATION_POLICY = (
    "Loop B2 alpha scoring is restricted before scoring to is_observed_transition == True rows; "
    "the legacy 10pp B2 diagnostic is defined only on the observed transitions"
)
B2_REPLAY_DIAGNOSTIC_POLICY = (
    "Observed-next replay alignment diagnostic only: the Stage2 mixed-transition observed row "
    "preserves empirical t+1 values, so this diagnostic is not an action-conditioned "
    "simulator validation gate and must not determine substrate_tier."
)
B2_CURRENT_RESEARCH_GATE_POLICY = (
    "Current 검사3제거 research contract with configured Stage2 diagnostics: Stage2 "
    "simulator financial-fidelity uses a 10% median-absolute-error-over-assets "
    "diagnostic tolerance, and simulator score-retention is summarized as PASS when "
    "the OOT direction-agreement point estimate is at least 50%. These Stage2 "
    "diagnostic pass/fail labels do not replace the preregistered Stage1/LoopB1 "
    "Oracle lead-validation gate; they prevent stale B1-minus-B2 10pp semantics from "
    "mislabeling current outputs. Broken pipeline contracts such as row alignment, "
    "missing columns, bad C_obs action values, R157/R182 all-NaN, accounting identity "
    "violations, or missing expected artifacts still hard-fail."
)
B2_STAGE2_DIAGNOSTIC_TIER = "oracle_validated_simulator_diagnostics_reported"
B2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS = "ORACLE_VALIDATED_SIMULATOR_DIAGNOSTICS_REPORTED"
B2_SIMULATOR_SCORE_RETENTION_POLICY = (
    "Simulator score-retention diagnostic: simulate the observed historical action from "
    "year t-1 to t, score the simulated t state with the Stage1 alpha backend, compute "
    "sim_score(t)-real_score(t-1), and compare that lead score change to the real rating "
    "improvement over t to t+1 using the exact Stage1 Loop B1 time-lag convention. "
    "The current configured diagnostic PASS rule is OOT direction agreement >= 50%."
)
B2_SIMULATOR_VALIDATION_TYPE = "test3_simulator_score_retention_observed_action_lead_convention"
B2_TRUE_SIMULATOR_ACTION_SOURCE = "C_obs_realized_action_values_from_stage2_raw_action_source_panel"
# IMPORTANT: action_observed__* columns are observation masks, not action magnitudes.
# The realized C_obs action magnitudes are action__* values loaded from the Stage2A
# raw action source panel.  Candidate/projection handoffs can also contain
# action__* columns, so true Test2/Test3 must overwrite them from the direct
# raw action source panel before simulation.
B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX = "action__"
B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX = "action_observed__"
B2_TRUE_SIMULATOR_ACTION_COLUMN_PREFIX = B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX
B2_TRUE_SIMULATOR_ACTION_AUDIT_PREFIX = "sim_input__action__"
B2_TRUE_SIMULATOR_ACTION_FLAG_AUDIT_PREFIX = "sim_input__action_observed__"
B2_TRUE_SIMULATOR_MIN_ACTION_FINITE_SHARE = 1.0
B2_TRUE_SIMULATOR_MIN_ACTION_FLAG_OBSERVED_SHARE = 0.0
B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY = (
    "True Loop B2 simulator validation must use realized C_obs action magnitudes "
    "from action__* columns loaded from the Stage2A direct raw action source panel. "
    "action_observed__* columns are boolean observation masks, not magnitudes, and "
    "must never be passed to FinancialSimulator as action values. action__* columns "
    "already present in candidate/projection handoffs are diagnostic only until they "
    "are overwritten from the direct raw action source panel. Missing/non-finite "
    "action values or dead action_observed__ masks are hard failures."
)
B2_TRUE_SIMULATOR_NO_NEXT_RATIO_OVERLAY_POLICY = (
    "True Loop B2 simulator score-retention must compute Stage1-selected R-code values "
    "from the simulated financial state and its audited previous state. Observed next__R-code "
    "ratio overlays are allowed only for the observed-next replay diagnostic and are forbidden "
    "inside the true simulator validation gate."
)
B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY = (
    "True Loop B2 Test2/Test3 prev_state denominators must come from the actual Stage1 "
    "firm-year financial panel or cleaned statement panel U-code columns, not from Stage2 P-handoff "
    "zero-filled denominator sentinels. Non-null but all-zero capital_stock/non_current_assets "
    "sources are hard failures because R157/R182 would be undefined."
)
B2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE = 0.50
B2_REPLAY_DIAGNOSTIC_TYPE = "observed_next_replay_alignment_not_simulator_validation"
B2_PREV_STATE_UCODE_COLUMNS: dict[str, dict[str, Any]] = {
    "non_current_assets": {
        "ucode": "U01A110000000",
        "canonical_column": "balance_sheet__[U01A110000000]   비유동자산(*)(IFRS)(천원)",
        "aliases": ("non_current_assets", "noncurrent_assets"),
    },
    "capital_stock": {
        "ucode": "U01A611000000",
        "canonical_column": "balance_sheet__[U01A611000000]   자본금(*)(IFRS)(천원)",
        "aliases": ("capital_stock", "capital"),
    },
}


def _json_safe(obj: Any) -> Any:
    """Convert numpy/pandas scalar containers into strict JSON-serializable objects.

    Pandas groupby summaries can leak numpy scalar dtypes (for example np.int64)
    into the report metadata.  The verifier should not fail after successful
    computation merely because the final report contains numpy scalar types.
    """
    if obj is None or isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(_json_safe(k)): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return obj


# LoopA fidelity must compare simulator-predicted t+1 states against observed
# raw t+1 values.  Stage2 artifacts use several historical naming conventions;
# keep strict actual-side aliases raw-only so strict fidelity does not compare
# simulator output against simulator output.
LOOPA_PREDICTED_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("total_assets", "next__sim__total_assets", "sim__total_assets"),
    "short_term_debt": ("short_term_debt", "next__sim__short_term_debt", "sim__short_term_debt"),
    "long_term_debt": ("long_term_debt", "next__sim__long_term_debt", "sim__long_term_debt"),
    "bonds": ("bonds", "bond", "next__sim__bonds", "next__sim__bond", "sim__bonds", "sim__bond"),
    "operating_cf": ("operating_cf", "next__sim__operating_cf", "sim__operating_cf"),
}
LOOPA_ACTUAL_NEXT_ALIASES: dict[str, tuple[str, ...]] = {
    "total_assets": ("next__raw__total_assets", "next__total_assets", "total_assets__next"),
    "short_term_debt": ("next__raw__short_term_debt", "next__raw__short_debt", "next__short_term_debt", "short_term_debt__next"),
    "long_term_debt": ("next__raw__long_term_debt", "next__raw__long_debt", "next__long_term_debt", "long_term_debt__next"),
    "bonds": ("next__raw__bonds", "next__raw__bond", "next__bonds", "bonds__next"),
    "operating_cf": ("next__raw__operating_cf", "next__operating_cf", "operating_cf__next"),
}
LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS = {
    "total_assets": ("next__raw__total_assets",),
    "short_term_debt": ("next__raw__short_term_debt", "next__raw__short_debt"),
    "long_term_debt": ("next__raw__long_term_debt", "next__raw__long_debt"),
    "bonds": ("next__raw__bonds", "next__raw__bond"),
}


def _load_registry_file(path: Path) -> dict:
    """Load a registry from JSON or YAML while preserving backward compatibility."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _resolve_alpha_params(root: Path, final: Path) -> tuple[Path | None, str, str]:
    """Resolve Stage1 alpha params from the real final_freeze contract.

    Historical Loop B2 expected data/final_freeze/oracle_backend_registry.json with
    {"alpha": {"params": ...}}.  Stage1 final development now publishes the canonical
    registry as data/final_freeze/configs/oracle_backend_registry.yaml with
    {"backends": {"alpha": {"params": ...}}}.  Accept both contracts and fall back
    to the canonical alpha backend artifact path.
    """
    candidates = [
        final / "oracle_backend_registry.json",                 # legacy compat
        final / "configs" / "oracle_backend_registry.yaml",     # canonical final_freeze
        final / "configs" / "oracle_backend_registry.yml",
        root / "src" / "credit_recourse" / "configs" / "oracle_backend_registry.yaml",  # packaged default, last resort
    ]
    for reg_path in candidates:
        if not reg_path.exists():
            continue
        try:
            reg = _load_registry_file(reg_path)
        except Exception as e:
            continue
        alpha = {}
        if isinstance(reg.get("backends"), dict):
            alpha = reg.get("backends", {}).get("alpha", {}) or {}
        if not alpha and isinstance(reg.get("alpha"), dict):
            alpha = reg.get("alpha", {}) or {}
        params_ref = alpha.get("params") if isinstance(alpha, dict) else None
        if params_ref:
            params = resolve_backend_artifact(root, final, params_ref)
            if params.exists():
                return params, str(reg_path), str(params_ref)

    fallback = final / "stage1_oracle_backends" / "alpha" / "oracle_alpha_params.json"
    if fallback.exists():
        return fallback, "direct_alpha_backend_fallback", str(fallback)
    return None, "missing_alpha_params", ""


def _load_alpha_selected_variables(params_path: Path) -> list[str]:
    """Return selected variables exported by the alpha backend params."""
    params = json.loads(params_path.read_text(encoding="utf-8"))
    vars_raw = params.get("selected_variables") or [
        v.get("variable_id")
        for v in params.get("variables", [])
        if isinstance(v, dict)
    ]
    return [str(v) for v in vars_raw if v]


def _is_next_state_column_name(column: Any) -> bool:
    s = str(column)
    return s.startswith("next__") or s.endswith("__next")


def _find_base_state_input_column(df: pd.DataFrame, field: str) -> str | None:
    """Find the base-year column that resolves to a FirmState field.

    The lookup is intentionally base-state only: next__* and *__next columns are
    excluded so the t+1 handoff values cannot leak into prev_state used for
    R157/R182.  U-code substring matching mirrors account_registry semantics.

    This broad resolver is safe for coverage diagnostics after B2 has forced the
    two prev_state denominator fields.  It is *not* used to choose the handoff
    source column, because generic aliases such as ``non_current_assets`` can
    refer to a simulator/used-next column in Stage2 handoff artifacts and can
    shadow the real base-year U-code values.
    """
    spec = B2_PREV_STATE_UCODE_COLUMNS[field]
    columns = list(df.columns)
    original = {str(c): c for c in columns}
    direct_candidates = [str(spec["canonical_column"]), *map(str, spec.get("aliases", ())), str(spec["ucode"])]
    for cand in direct_candidates:
        if cand in original and not _is_next_state_column_name(cand):
            return str(original[cand])
    code = str(spec["ucode"])
    for col in columns:
        s = str(col)
        if _is_next_state_column_name(s):
            continue
        if code in s:
            return s
    return None


def _find_base_state_ucode_input_column(df: pd.DataFrame, field: str) -> str | None:
    """Find the real base-year U-code source column for a B2 prev_state field.

    Unlike `_find_base_state_input_column`, this function deliberately ignores
    generic aliases such as `capital_stock` and `non_current_assets`.  In Stage2
    P-handoff tables those bare aliases may be simulator/used-next columns or
    all-NaN alias columns.  The B2 prev_state denominator source must be the
    base-year audited balance-sheet U-code column, never a generic alias and
    never a next-state column.
    """
    spec = B2_PREV_STATE_UCODE_COLUMNS[field]
    columns = list(df.columns)
    original = {str(c): c for c in columns}
    canonical = str(spec["canonical_column"])
    if canonical in original and not _is_next_state_column_name(canonical):
        return str(original[canonical])
    code = str(spec["ucode"])
    for col in columns:
        s = str(col)
        if _is_next_state_column_name(s):
            continue
        if code in s:
            return s
    return None


def _loopb2_prev_state_ucode_source_coverage(df: pd.DataFrame) -> dict[str, Any]:
    """Coverage diagnostics using only base-year U-code source columns."""
    details: dict[str, Any] = {}
    for field, spec in B2_PREV_STATE_UCODE_COLUMNS.items():
        col = _find_base_state_ucode_input_column(df, field)
        if col is None:
            details[field] = {
                "status": "MISSING",
                "column": None,
                "ucode": spec["ucode"],
                "nonnull_share": 0.0,
                "nonnull_rows": 0,
                "lookup_policy": "base_year_ucode_only",
            }
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        finite = numeric.notna()
        nonzero = finite & (numeric.abs() > 1e-9)
        details[field] = {
            "status": "FOUND",
            "column": str(col),
            "ucode": spec["ucode"],
            "nonnull_share": float(finite.mean()) if len(numeric) else 0.0,
            "nonnull_rows": int(finite.sum()),
            "nonzero_share": float(nonzero.mean()) if len(numeric) else 0.0,
            "nonzero_rows": int(nonzero.sum()),
            "lookup_policy": "base_year_ucode_only",
        }
    return details


def _loopb2_prev_state_input_coverage(df: pd.DataFrame) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for field, spec in B2_PREV_STATE_UCODE_COLUMNS.items():
        col = _find_base_state_input_column(df, field)
        if col is None:
            details[field] = {
                "status": "MISSING",
                "column": None,
                "ucode": spec["ucode"],
                "nonnull_share": 0.0,
                "nonnull_rows": 0,
            }
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        details[field] = {
            "status": "FOUND",
            "column": str(col),
            "ucode": spec["ucode"],
            "nonnull_share": float(numeric.notna().mean()) if len(numeric) else 0.0,
            "nonnull_rows": int(numeric.notna().sum()),
        }
    return details


def _add_loopb2_prev_state_join_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Add robust firm-year join keys for Loop B2 prev_state handoff merges.

    Stage2 artifacts have appeared with both exchange-code style firm ids
    (for example ``A123456``) and bare six-digit ids.  The B2 score_t join
    already normalises this convention; prev_state enrichment must use the same
    key policy or the audited U-code columns can be present with 100% source
    coverage yet still merge into zero scoring rows.
    """
    out = _normalise_loopa_keys(df).copy()
    out[B2_PREV_STATE_JOIN_KEY_FIRM] = _normalise_loopb2_firm_key(out["firm_id"])
    out[B2_PREV_STATE_JOIN_KEY_YEAR] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    return out


def _nonnull_share(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or len(df) == 0:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").notna().mean())


def _verify_loopb2_prev_state_ucode_loader_contract() -> dict[str, Any]:
    """Unit-smoke the registry-aware FirmState loader for the two B2 denominators."""
    payload = {
        B2_PREV_STATE_UCODE_COLUMNS["non_current_assets"]["canonical_column"]: 123456.0,
        B2_PREV_STATE_UCODE_COLUMNS["capital_stock"]["canonical_column"]: 7890.0,
    }
    fs = load_firm_state_from_columns(payload, firm_id="LOOPB2_UCODE_SMOKE", year=2020, sector="Smoke")
    checks = {
        "non_current_assets": getattr(fs, "non_current_assets", None),
        "capital_stock": getattr(fs, "capital_stock", None),
    }
    ok = (checks["non_current_assets"] == 123456.0) and (checks["capital_stock"] == 7890.0)
    if not ok:
        raise AssertionError(
            "load_firm_state_from_columns did not resolve Loop B2 prev-state U-code columns: "
            + json.dumps(checks, ensure_ascii=False, default=str)
        )
    return {
        "status": "PASS",
        "loader": "load_firm_state_from_columns",
        "resolver": "resolved_field_values",
        "checked_columns": {field: spec["canonical_column"] for field, spec in B2_PREV_STATE_UCODE_COLUMNS.items()},
        "resolved_values": checks,
    }


def _merge_loopb2_prev_state_inputs_from_handoff(
    phase: pd.DataFrame,
    handoff: pd.DataFrame,
    *,
    source_label: str,
    source_role: str = "stage2_observed_next_handoff",
    source_lookup_policy: str = "required base-year U-code only; generic aliases are not accepted as handoff source",
    require_nonzero_source: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join base-year U-code denominators into the rows used for B2 prev_state.

    Stage2 P-handoff artifacts carry base-year balance-sheet U-code columns even
    when the Loop B2 phase table uses a narrower schema.  R157 and R182 require
    those t-year denominators via prev_state.  This merge only uses base-year
    columns keyed by firm_id x fiscal_year; next__balance_sheet__* columns are
    explicitly excluded to prevent future-state leakage.
    """
    phase_n = _add_loopb2_prev_state_join_keys(phase).copy()
    handoff_n = _add_loopb2_prev_state_join_keys(handoff).copy()
    before = _loopb2_prev_state_input_coverage(phase_n)
    source_coverage = _loopb2_prev_state_input_coverage(handoff_n)
    source_ucode_coverage = _loopb2_prev_state_ucode_source_coverage(handoff_n)

    # Source selection must be stricter than general FirmState resolution.  Generic
    # aliases in P-handoff tables can refer to used-next/simulator fields and can
    # shadow audited base-year values.  For B2 prev_state denominators we require
    # the audited base-year balance-sheet U-code columns found by U-code substring.
    source_cols = {field: _find_base_state_ucode_input_column(handoff_n, field) for field in B2_PREV_STATE_UCODE_COLUMNS}
    missing_source = [field for field, col in source_cols.items() if col is None]
    if missing_source:
        raise ValueError(
            "Loop B2 prev_state handoff source is missing required base-year U-code columns "
            f"{missing_source}; source={source_label}; available_leading_columns={list(map(str, handoff_n.columns[:80]))}"
        )
    low_source_coverage = []
    for field, detail in source_ucode_coverage.items():
        share = float((detail or {}).get("nonnull_share") or 0.0)
        nonzero_share = float((detail or {}).get("nonzero_share") or 0.0)
        if share <= B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE:
            low_source_coverage.append({"field": field, "nonnull_share": share, "nonzero_share": nonzero_share, "column": (detail or {}).get("column")})
        if bool(require_nonzero_source) and nonzero_share <= B2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE:
            low_source_coverage.append({
                "field": field,
                "nonnull_share": share,
                "nonzero_share": nonzero_share,
                "column": (detail or {}).get("column"),
                "reason": "non_null_but_zero_filled_denominator_source",
            })
    if low_source_coverage:
        raise ValueError(
            "Loop B2 prev_state source U-code coverage/nonzero check failed for R157/R182 denominators: "
            + json.dumps({
                "source": str(source_label),
                "source_role": str(source_role),
                "require_nonzero_source": bool(require_nonzero_source),
                "policy": B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY if require_nonzero_source else source_lookup_policy,
                "violations": low_source_coverage,
            }, ensure_ascii=False, default=str)
        )

    merge_payload = handoff_n[[B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR]].copy()
    temp_cols: dict[str, str] = {}
    for field, col in source_cols.items():
        temp_col = f"__loopb2_prevsrc__{field}"
        merge_payload[temp_col] = pd.to_numeric(handoff_n[col], errors="coerce")
        temp_cols[field] = temp_col

    conflict_samples: list[dict[str, Any]] = []
    for field, temp_col in temp_cols.items():
        nunique = merge_payload.groupby([B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR], dropna=False)[temp_col].nunique(dropna=True)
        conflicts = nunique[nunique > 1]
        if not conflicts.empty:
            for key, n_distinct in conflicts.head(10).items():
                conflict_samples.append({
                    "field": field,
                    "normalised_firm_key": str(key[0]),
                    "fiscal_year": None if pd.isna(key[1]) else int(key[1]),
                    "n_distinct_values": int(n_distinct),
                })
    if conflict_samples:
        raise ValueError(
            "Loop B2 prev_state handoff source has conflicting duplicate firm-year U-code values: "
            + json.dumps(conflict_samples, ensure_ascii=False, default=str)
        )

    # Collapse duplicate candidate rows per firm-year without discarding a later
    # non-null base-year denominator when an earlier duplicate happens to be NaN.
    merge_payload = (
        merge_payload
        .groupby([B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR], dropna=False, as_index=False)
        .agg(lambda s: s.dropna().iloc[0] if s.dropna().shape[0] else np.nan)
    )
    merged = phase_n.merge(
        merge_payload,
        on=[B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR],
        how="left",
        validate="many_to_one",
    )
    matched_rows = int(merged[list(temp_cols.values())].notna().any(axis=1).sum()) if temp_cols else 0
    matched_share = float(matched_rows / len(merged)) if len(merged) else 0.0
    if matched_share <= B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE:
        sample_cols = [c for c in ["firm_id", "fiscal_year", B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR] if c in merged.columns]
        unmatched_sample = merged.loc[~merged[list(temp_cols.values())].notna().any(axis=1), sample_cols].head(10).to_dict("records") if temp_cols else []
        raise ValueError(
            "Loop B2 prev_state U-code handoff merge matched too few rows after robust firm-key normalisation: "
            + json.dumps({
                "source": str(source_label),
                "rows": int(len(merged)),
                "matched_rows_any_required_field": matched_rows,
                "matched_share": matched_share,
                "threshold_strictly_greater_than": B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE,
                "join_key_policy": "normalised_loopb2_firm_key + fiscal_year",
                "unmatched_sample": unmatched_sample,
            }, ensure_ascii=False, default=str)
        )

    phase_conflicts: list[dict[str, Any]] = []
    filled_rows: dict[str, int] = {}
    output_columns: dict[str, str] = {}
    forced_field_columns: dict[str, str] = {}
    preexisting_alias_columns: dict[str, str | None] = {}
    preexisting_ucode_columns: dict[str, str | None] = {}
    alias_shadow_rows: dict[str, int] = {}
    zero_sentinel_override_rows: dict[str, int] = {}
    for field, spec in B2_PREV_STATE_UCODE_COLUMNS.items():
        canonical_col = str(spec["canonical_column"])
        field_col = str(field)
        existing_alias_col = _find_base_state_input_column(phase_n, field)
        existing_ucode_col = _find_base_state_ucode_input_column(phase_n, field)
        preexisting_alias_columns[field] = None if existing_alias_col is None else str(existing_alias_col)
        preexisting_ucode_columns[field] = None if existing_ucode_col is None else str(existing_ucode_col)
        source_temp_col = temp_cols[field]
        source_vals = pd.to_numeric(merged[source_temp_col], errors="coerce")

        # Only an existing U-code column is allowed to compete with the handoff
        # U-code source.  Bare aliases on the phase row are ambiguous in Stage2
        # artifacts and can otherwise shadow the audited U-code value inside
        # resolved_field_values().  Therefore the final prev_state row writes the
        # handoff value to both the canonical raw header and the bare FirmState
        # field column.  This is not imputation: it is a deterministic projection
        # of the already-present audited U-code value onto the loader's highest
        # priority alias.
        if existing_ucode_col is not None:
            existing_vals = pd.to_numeric(merged[existing_ucode_col], errors="coerce")
            if bool(require_nonzero_source):
                # In true simulator validation, an existing all-zero U-code value in
                # Stage2 handoff is a known sentinel, not an audited denominator.
                # Treat zero as missing for the two guarded denominator fields so
                # the actual Stage1 financial panel can replace it.  Non-zero
                # disagreements are still hard conflicts.
                existing_usable = existing_vals.notna() & (existing_vals.abs() > 1e-9)
                zero_override = existing_vals.notna() & (existing_vals.abs() <= 1e-9) & source_vals.notna()
                zero_sentinel_override_rows[field] = int(zero_override.sum())
            else:
                existing_usable = existing_vals.notna()
                zero_sentinel_override_rows[field] = 0
            overlap = existing_usable & source_vals.notna()
            conflict = overlap & ((existing_vals - source_vals).abs() > 1e-6)
            if bool(conflict.any()):
                sample_cols = [c for c in ["firm_id", "fiscal_year"] if c in merged.columns]
                sample = merged.loc[conflict, sample_cols].head(10).to_dict("records")
                phase_conflicts.append({
                    "field": field,
                    "existing_ucode_column": str(existing_ucode_col),
                    "source_column": str(source_cols[field]),
                    "conflict_rows": int(conflict.sum()),
                    "sample": sample,
                    "conflict_policy": "non-zero existing values may not disagree with actual financial source",
                })
            combined = existing_vals.where(existing_usable, source_vals)
            filled_rows[field] = int((~existing_usable & combined.notna()).sum())
        else:
            combined = source_vals
            zero_sentinel_override_rows[field] = 0
            filled_rows[field] = int(source_vals.notna().sum())

        if existing_alias_col is not None and str(existing_alias_col) != str(existing_ucode_col):
            alias_vals = pd.to_numeric(merged[existing_alias_col], errors="coerce")
            alias_shadow_rows[field] = int((alias_vals.notna() & source_vals.notna() & ((alias_vals - source_vals).abs() > 1e-6)).sum())
        else:
            alias_shadow_rows[field] = 0

        merged[canonical_col] = combined
        merged[field_col] = combined
        output_columns[field] = canonical_col
        forced_field_columns[field] = field_col

    if phase_conflicts:
        raise ValueError(
            "Loop B2 prev_state base-year values conflict between phase rows and handoff source: "
            + json.dumps(phase_conflicts, ensure_ascii=False, default=str)
        )

    after = _loopb2_prev_state_input_coverage(merged)
    low_after_coverage = []
    for field, detail in after.items():
        share = float((detail or {}).get("nonnull_share") or 0.0)
        if share <= B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE:
            low_after_coverage.append({"field": field, "nonnull_share": share, "column": (detail or {}).get("column")})
    if low_after_coverage:
        raise ValueError(
            "Loop B2 prev_state handoff merge produced insufficient post-merge denominator coverage: "
            + json.dumps(low_after_coverage, ensure_ascii=False, default=str)
        )
    merged = merged.drop(columns=list(temp_cols.values()) + [B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR], errors="ignore")
    meta = {
        "status": "PASS",
        "source": str(source_label),
        "source_role": str(source_role),
        "join_key": ["firm_id", "fiscal_year"],
        "robust_join_key_columns": [B2_PREV_STATE_JOIN_KEY_FIRM, B2_PREV_STATE_JOIN_KEY_YEAR],
        "join_key_policy": "normalised_loopb2_firm_key + fiscal_year; accepts A-prefixed and bare numeric firm ids",
        "matched_share_any_required_field": matched_share,
        "required_fields": list(B2_PREV_STATE_UCODE_COLUMNS.keys()),
        "source_columns": {field: str(col) for field, col in source_cols.items()},
        "source_lookup_policy": str(source_lookup_policy),
        "require_nonzero_source": bool(require_nonzero_source),
        "nonzero_source_policy": B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY if require_nonzero_source else "not required for observed-next replay diagnostic",
        "output_columns": output_columns,
        "forced_firm_state_field_columns": forced_field_columns,
        "prev_state_loader_alias_precedence_guard": "write audited U-code values to both the canonical U-code header and the bare FirmState field columns so ambiguous phase aliases cannot shadow them",
        "preexisting_alias_columns": preexisting_alias_columns,
        "preexisting_ucode_columns": preexisting_ucode_columns,
        "alias_shadow_rows_before_override": alias_shadow_rows,
        "zero_sentinel_override_rows": zero_sentinel_override_rows,
        "rows": int(len(merged)),
        "matched_rows_any_required_field": matched_rows,
        "filled_rows_by_field": filled_rows,
        "before_coverage": before,
        "source_coverage": source_coverage,
        "source_ucode_coverage": source_ucode_coverage,
        "after_coverage": after,
        "future_state_leakage_guard": "base-year U-code columns only; next__* and *__next columns are excluded",
    }
    return merged, meta



def _loopb2_parse_fiscal_year_series(values: Any, *, index: pd.Index | None = None) -> pd.Series:
    """Parse fiscal-year values from numeric years or TS2000 strings like `2017/12`."""
    ser = pd.Series(values, index=index) if index is not None else pd.Series(values)
    direct = pd.to_numeric(ser, errors="coerce")
    text = ser.astype("string")
    extracted = pd.to_numeric(text.str.extract(r"(\d{4})", expand=False), errors="coerce")
    out = direct.where(direct.notna(), extracted)
    return out.astype("Int64")


def _normalise_loopb2_actual_financial_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise firm/year keys for Stage1 actual financial statement panels."""
    out = df.copy()
    firm_candidates = [
        "firm_id", "corp_code", "거래소코드", "exchange_code", "stock_code", "company_code", "종목코드",
        "_firm", "financial__stock_code", "rating__stock_code", "code",
    ]
    year_candidates = [
        "fiscal_year", "year", "회계년도", "사업연도", "결산년도", "평가년도", "_year",
    ]
    if "firm_id" not in out.columns:
        firm_col = next((c for c in firm_candidates if c in out.columns), None)
        if firm_col is not None:
            out["firm_id"] = out[firm_col]
    if "fiscal_year" not in out.columns:
        year_col = next((c for c in year_candidates if c in out.columns), None)
        if year_col is not None:
            out["fiscal_year"] = out[year_col]
    if "firm_id" not in out.columns or "fiscal_year" not in out.columns:
        raise ValueError(
            "Loop B2 actual financial source requires firm/year keys; "
            f"available leading columns={list(map(str, out.columns[:80]))}"
        )
    out["firm_id"] = out["firm_id"].astype(str)
    out["fiscal_year"] = _loopb2_parse_fiscal_year_series(out["fiscal_year"], index=out.index)
    return out


def _loopb2_find_column_by_alias(columns: list[Any], aliases: tuple[str, ...]) -> str | None:
    """Resolve a source column by exact or case-insensitive alias without substring guessing."""
    as_str = {str(c): c for c in columns}
    for alias in aliases:
        if alias in as_str:
            return str(as_str[alias])
    lowered = {str(c).strip().lower(): c for c in columns}
    for alias in aliases:
        key = str(alias).strip().lower()
        if key in lowered:
            return str(lowered[key])
    return None


def _prepare_loopb2_actual_financial_source_frame(raw: pd.DataFrame, source_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Prepare a candidate actual financial statement source for true B2.

    The Stage00-01 adapter writes two useful actual-statement artifacts:
    a wide cleaned balance-sheet panel and a compact long raw-audit panel.  The
    former already contains U-code columns such as ``[U01A110000000] ...``; the
    latter must be pivoted by firm-year and item code.  Earlier v13 code looked
    only at firm-year/rating panels and therefore missed the real denominator
    source even when Stage00-01 had materialized it under
    ``cleaned_statement_panels/재무상태표_clean.parquet``.
    """
    path_s = str(source_path)
    columns = list(raw.columns)
    code_col = _loopb2_find_column_by_alias(columns, ("_icode", "item_code", "account_code", "item_id", "financial__item_code"))
    value_col = _loopb2_find_column_by_alias(columns, ("_value", "value_numeric", "value", "amount", "value_raw", "financial__value_numeric"))
    firm_col = _loopb2_find_column_by_alias(columns, ("_firm", "firm_id", "corp_code", "거래소코드", "exchange_code", "stock_code", "company_code", "종목코드", "financial__stock_code", "rating__stock_code", "code"))
    year_col = _loopb2_find_column_by_alias(columns, ("_year", "fiscal_year", "year", "회계년도", "사업연도", "결산년도", "평가년도"))
    stmt_col = _loopb2_find_column_by_alias(columns, ("_stmt", "statement_type", "statement", "stmt_type", "financial__statement_type"))
    name_col = _loopb2_find_column_by_alias(columns, ("_iname", "item_name_raw", "item_name", "account_name", "financial__item_name_raw"))

    required_ucodes = {spec["ucode"] for spec in B2_PREV_STATE_UCODE_COLUMNS.values()}
    if code_col is not None and value_col is not None and firm_col is not None and year_col is not None:
        work = pd.DataFrame({
            "firm_id": raw[firm_col].astype(str),
            "fiscal_year": _loopb2_parse_fiscal_year_series(raw[year_col], index=raw.index),
            "__ucode": raw[code_col].astype(str).str.strip(),
            "__value": pd.to_numeric(raw[value_col], errors="coerce"),
        })
        if stmt_col is not None:
            stmt = raw[stmt_col].astype(str).str.strip()
            bs_mask = stmt.isin({"재무상태표", "balance_sheet", "Balance Sheet", "BS"}) | stmt.str.contains("재무상태", na=False)
            work = work.loc[bs_mask.to_numpy()].copy()
        if name_col is not None:
            work["__item_name"] = raw.loc[work.index, name_col].astype(str).to_numpy() if len(work) else []
        else:
            work["__item_name"] = ""
        work = work.loc[work["__ucode"].isin(required_ucodes)].dropna(subset=["fiscal_year"]).copy()
        if work.empty:
            raise ValueError(
                f"Long actual financial source {path_s} contains no required balance-sheet U-codes "
                f"{sorted(required_ucodes)} after statement filtering"
            )
        conflict_records: list[dict[str, Any]] = []
        for ucode in sorted(required_ucodes):
            sub = work.loc[work["__ucode"].eq(ucode), ["firm_id", "fiscal_year", "__value"]].dropna(subset=["__value"])
            if sub.empty:
                continue
            nunique = sub.groupby(["firm_id", "fiscal_year"], dropna=False)["__value"].nunique(dropna=True)
            bad = nunique[nunique > 1]
            for key, n_distinct in bad.head(10).items():
                conflict_records.append({
                    "ucode": ucode,
                    "firm_id": str(key[0]),
                    "fiscal_year": None if pd.isna(key[1]) else int(key[1]),
                    "n_distinct_values": int(n_distinct),
                })
        if conflict_records:
            raise ValueError(
                "Long actual financial source has conflicting duplicate firm-year denominator values: "
                + json.dumps(conflict_records, ensure_ascii=False, default=str)
            )
        agg = (
            work.groupby(["firm_id", "fiscal_year", "__ucode"], dropna=False, as_index=False)["__value"]
            .agg(lambda s: s.dropna().iloc[-1] if s.dropna().shape[0] else np.nan)
        )
        wide = agg.pivot(index=["firm_id", "fiscal_year"], columns="__ucode", values="__value").reset_index()
        wide.columns.name = None
        for field, spec in B2_PREV_STATE_UCODE_COLUMNS.items():
            ucode = str(spec["ucode"])
            if ucode in wide.columns:
                wide[str(spec["canonical_column"])] = wide[ucode]
                wide = wide.drop(columns=[ucode])
        wide = _normalise_loopb2_actual_financial_keys(wide)
        return wide, {
            "source_format": "long_statement_item_panel_pivoted_to_wide",
            "source_path": path_s,
            "firm_column": str(firm_col),
            "year_column": str(year_col),
            "code_column": str(code_col),
            "value_column": str(value_col),
            "statement_column": None if stmt_col is None else str(stmt_col),
            "rows_before_filter": int(len(raw)),
            "rows_after_required_ucode_filter": int(len(work)),
            "wide_rows": int(len(wide)),
        }

    wide = _normalise_loopb2_actual_financial_keys(raw)
    return wide, {
        "source_format": "wide_actual_financial_panel",
        "source_path": path_s,
        "rows": int(len(wide)),
        "columns": int(len(wide.columns)),
    }


def _loopb2_actual_financial_source_candidates(final: Path) -> list[Path]:
    """Candidate actual firm-year statement panels for true simulator B2 denominators."""
    s1 = final / "stage1_oracle_inputs" / "stage00_01_rating_statement_integration"
    legacy_s1 = final / "stage00_01_rating_statement_integration"
    s0 = final / "stage0_oracle_foundation"
    return [
        # Preferred: the Stage00-01 cleaned balance-sheet panel contains the real
        # wide U-code statement columns needed by R157/R182 denominators.
        s1 / "cleaned_statement_panels" / "재무상태표_clean.parquet",
        legacy_s1 / "cleaned_statement_panels" / "재무상태표_clean.parquet",
        # Fallback source class is still deterministic and audited: compact long
        # statement item panels are pivoted only for the two required U-codes.
        s1 / "financial_statement_items_raw.parquet",
        legacy_s1 / "financial_statement_items_raw.parquet",
        s0 / "canonical_panel" / "statement_items_panel.parquet",
        # Lower priority panels are kept only for projects that materialize U-code
        # columns directly in their firm-year panel variants.
        s1 / "firm_year_panel_v1.parquet",
        legacy_s1 / "firm_year_panel_v1.parquet",
        final / "stage1_oracle_inputs" / "stage00_02_ratio_engineering" / "firm_year_panel_v1.parquet",
        final / "stage00_02_ratio_engineering" / "firm_year_panel_v1.parquet",
        s0 / "firm_year_panel_v1.parquet",
        s0 / "canonical_panel" / "stage0_canonical_panel.parquet",
    ]


def _load_loopb2_actual_financial_source(final: Path) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Load the actual firm-year financial panel used by true LoopB2.

    Stage2 P-handoff artifacts can carry base-year U-code columns that are
    non-null but all zero.  Those sentinels are useful for replay diagnostics but
    invalid for true simulator Test2/Test3 because R157/R182 denominators become
    undefined.  This loader requires a Stage1 actual financial panel with real
    non-zero coverage for the two denominator U-codes.
    """
    attempted: list[dict[str, Any]] = []
    for path in _loopb2_actual_financial_source_candidates(final):
        if not path.exists():
            attempted.append({"path": str(path), "status": "MISSING_FILE"})
            continue
        try:
            raw = read_parquet_required(path)
            df, prep_meta = _prepare_loopb2_actual_financial_source_frame(raw, path)
        except Exception as exc:
            attempted.append({"path": str(path), "status": "READ_OR_PREPARE_FAIL", "error": repr(exc)})
            continue
        cov = _loopb2_prev_state_ucode_source_coverage(df)
        missing = [field for field, detail in cov.items() if (detail or {}).get("status") != "FOUND"]
        low_nonnull = [
            {"field": field, "detail": detail}
            for field, detail in cov.items()
            if float((detail or {}).get("nonnull_share") or 0.0) <= B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE
        ]
        low_nonzero = [
            {"field": field, "detail": detail}
            for field, detail in cov.items()
            if float((detail or {}).get("nonzero_share") or 0.0) <= B2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE
        ]
        rec = {
            "path": str(path),
            "status": "FOUND",
            "rows": int(len(df)),
            "prepared_source": prep_meta,
            "coverage": cov,
            "missing_required_fields": missing,
            "low_nonnull_fields": low_nonnull,
            "low_nonzero_fields": low_nonzero,
        }
        attempted.append(rec)
        if not missing and not low_nonnull and not low_nonzero:
            meta = {
                "status": "PASS",
                "source": str(path),
                "source_role": "stage1_actual_firm_year_financial_panel",
                "policy": B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY,
                "rows": int(len(df)),
                "prepared_source": prep_meta,
                "coverage": cov,
                "attempted_sources": attempted,
            }
            return df, str(path), meta
    raise ValueError(
        "Loop B2 true simulator validation could not find a Stage1 actual financial panel "
        "with non-zero R157/R182 denominator U-code coverage. Do not use Stage2 P-handoff "
        "zero-filled sentinels for true simulator validation. Details: "
        + json.dumps(attempted, ensure_ascii=False, default=str)[:6000]
    )


def _merge_loopb2_prev_state_inputs_from_actual_financial_panel(
    phase: pd.DataFrame,
    final: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge real base-year denominators into the LoopA/Test2-Test3 driver.

    This is the semantic source for true simulator validation.  It intentionally
    bypasses Stage2 P-handoff denominator columns because local evidence showed
    they can be non-null zero sentinels.  Candidate/action columns still come from
    Stage2 handoff; only raw accounting denominators used by FirmState and
    R157/R182 are taken from the Stage1 actual financial panel.
    """
    src, src_label, src_meta = _load_loopb2_actual_financial_source(final)
    merged, merge_meta = _merge_loopb2_prev_state_inputs_from_handoff(
        phase,
        src,
        source_label=src_label,
        source_role="stage1_actual_firm_year_financial_panel",
        source_lookup_policy=B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY,
        require_nonzero_source=True,
    )
    merge_meta["actual_financial_source_contract"] = src_meta
    merge_meta["true_simulator_prev_state_source_policy"] = B2_TRUE_SIMULATOR_PREV_STATE_SOURCE_POLICY
    return merged, merge_meta


def _loopb2_is_numeric_present(value: Any) -> bool:
    """Return True only for finite numeric values usable in formula diagnostics."""
    if value is None:
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(v))


def _loopb2_avg_numeric(a: Any, b: Any) -> float | None:
    vals: list[float] = []
    for x in (a, b):
        if _loopb2_is_numeric_present(x):
            vals.append(float(x))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _loopb2_truthy_scalar(value: Any, *, default: bool = False) -> bool:
    """Parse a scalar boolean value using the same convention as `_bool_mask`."""
    if value is None:
        return bool(default)
    try:
        if pd.isna(value):
            return bool(default)
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        try:
            return bool(float(value) != 0.0)
        except Exception:
            return bool(default)
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _loopb2_row_is_observed_transition(row: pd.Series) -> bool:
    """Return whether a Stage2 row belongs to the observed-transition B2 sample."""
    if "is_observed_transition" in row.index:
        return _loopb2_truthy_scalar(row.get("is_observed_transition"), default=False)
    if "counterfactual_transition" in row.index:
        return not _loopb2_truthy_scalar(row.get("counterfactual_transition"), default=True)
    return False


def _loopb2_next_ratio_alias_candidates(variable_id: str) -> tuple[str, ...]:
    var = str(variable_id)
    return (f"next__{var}", f"{var}__next")


def _loopb2_observed_next_ratio_value(row: pd.Series, variable_id: str) -> tuple[Any, str | None]:
    """Resolve a selected next-state R-code value from a mixed-transition row.

    This is deliberately limited to observed-transition rows by the caller.  It
    is not a generic counterfactual formula fallback: for non-observed cells,
    next__R-code columns are empirical future labels and must not be used.
    """
    for col in _loopb2_next_ratio_alias_candidates(variable_id):
        if col not in row.index:
            continue
        value = row.get(col)
        if _loopb2_is_numeric_present(value):
            return float(value), col
    return None, None


def _loopb2_observed_transition_mask(df: pd.DataFrame) -> pd.Series:
    """Mask the exact population on which the B2 10pp rule is defined."""
    if "is_observed_transition" in df.columns:
        return _bool_mask(df, "is_observed_transition", default=False)
    if "counterfactual_transition" in df.columns:
        return ~_bool_mask(df, "counterfactual_transition", default=True)
    return pd.Series(False, index=df.index, dtype=bool)


def _restrict_loopb2_scoring_population_to_observed(
    b2_phase: pd.DataFrame,
    b2_pred: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Restrict Loop B2 alpha scoring to the observed-transition rule sample.

    Earlier builds scored every counterfactual candidate row and then filtered to
    observed rows during agreement calculation.  That made the scoring frame both
    slow and semantically noisy: empirical next__R-code values are contract-valid
    only on the observed mixed-transition row.  The B2 rule itself is defined on
    the 3,159 observed transitions, so this filter is applied before scoring and
    is recorded in metadata.
    """
    phase_n = b2_phase.reset_index(drop=True).copy()
    pred_n = b2_pred.reset_index(drop=True).copy()
    n = min(len(phase_n), len(pred_n))
    phase_n = phase_n.iloc[:n].reset_index(drop=True)
    pred_n = pred_n.iloc[:n].reset_index(drop=True)
    observed_mask = _loopb2_observed_transition_mask(phase_n)
    observed_rows = int(observed_mask.sum())
    if observed_rows == 0:
        raise ValueError(
            "Loop B2 observed-transition scoring population is empty; "
            "the B2 10pp rule requires is_observed_transition == True rows."
        )
    filtered_phase = phase_n.loc[observed_mask].reset_index(drop=True)
    filtered_pred = pred_n.loc[observed_mask].reset_index(drop=True)
    meta = {
        "status": "PASS",
        "policy": B2_SCORING_POPULATION_POLICY,
        "source_rows_before_filter": int(n),
        "scored_rows_after_observed_filter": observed_rows,
        "observed_share": float(observed_rows / n) if n else 0.0,
        "non_observed_rows_excluded_before_scoring": int(n - observed_rows),
        "filter_columns_present": {
            "is_observed_transition": "is_observed_transition" in phase_n.columns,
            "counterfactual_transition": "counterfactual_transition" in phase_n.columns,
        },
        "future_label_leakage_guard": (
            "Stage2 next__R-code selected ratios may be used only after this observed-only filter; "
            "non-observed counterfactual rows are excluded before alpha scoring."
        ),
    }
    return filtered_phase, filtered_pred, meta


def _loopb2_formula_component_values(prev_state: FirmState, state_t1: FirmState) -> dict[str, Any]:
    """Expose the formula components that can make R157/R182 undefined.

    Field-level coverage alone is insufficient: a denominator can be present but
    zero/non-finite, and a selected formula can be removed from the dictionary by
    global duplicate-alias canonicalization.  These component diagnostics are
    persisted before the hard R157/R182 guard so local failures can be separated
    into input-value problems versus R-code id preservation problems.
    """
    r157_den = _loopb2_avg_numeric(getattr(prev_state, "capital_stock", None), getattr(state_t1, "capital_stock", None))
    r182_den = getattr(prev_state, "non_current_assets", None)
    return {
        "R157.numerator.state_t1.revenue": getattr(state_t1, "revenue", None),
        "R157.denominator.avg_capital_stock": r157_den,
        "R157.denominator.prev_state.capital_stock": getattr(prev_state, "capital_stock", None),
        "R157.denominator.state_t1.capital_stock": getattr(state_t1, "capital_stock", None),
        "R182.numerator.state_t1.non_current_assets": getattr(state_t1, "non_current_assets", None),
        "R182.denominator.prev_state.non_current_assets": r182_den,
    }


def _compute_loopb2_oracle_variables_preserving_selected_ids(
    state_t1: FirmState,
    *,
    prev_state: FirmState,
    exogenous: dict[str, Any],
    selected_variables: list[str],
    stage2_row: pd.Series | None = None,
    allow_observed_next_ratio_overlay: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute B2 alpha variables while preserving exact Stage1-selected R-code ids.

    ``compute_oracle_variables`` returns a globally canonicalized dictionary so
    duplicate ratio aliases can be collapsed for Stage6 diagnostics.  Loop B2 is
    different: the alpha backend params and the B1 score_t panel are keyed by the
    exact Stage1 selected ids.  If a final_freeze duplicate-alias map canonicalizes
    a selected id away, the scoring frame can have valid formula inputs with
    all-NaN ``R157``/``R182`` columns.  Therefore B2 overlays raw selected formula
    ids from ``ORACLE_FORMULA_REGISTRY`` after calling the shared simulator helper.

    Local diagnostic evidence showed a second, distinct case: the Stage2 P50
    handoff can contain the required U-code denominator columns with 100%
    non-null coverage but all values equal to zero.  Those are zero-filled
    sentinels, not usable denominators.  For observed-transition rows only,
    the Stage2 mixed-transition artifact also carries audited next-state selected
    ratios under ``next__Rxxx`` / ``Rxxx__next``.  When the deterministic formula
    is undefined for guarded selected ids, B2 materializes those observed next
    selected ratios under the exact Stage1 id.  Non-observed counterfactual rows
    are excluded before scoring and are never allowed to use empirical next R-code
    labels.

    This is not imputation and it does not change formulas: it preserves the
    exact Stage1-selected id using either the deterministic formula value or,
    where the handoff supplies only zero-filled denominators, the already-present
    observed mixed-transition next selected-ratio value.
    """
    canonicalized = dict(compute_oracle_variables(state_t1, prev_state=prev_state, exogenous=exogenous))
    raw_formula_values: dict[str, Any] = {}
    raw_formula_errors: dict[str, str] = {}
    selected_value_sources: dict[str, str] = {}
    direct_next_ratio_sources: dict[str, str] = {}
    direct_next_ratio_values: dict[str, Any] = {}
    observed_row = _loopb2_row_is_observed_transition(stage2_row) if stage2_row is not None else False
    for var in selected_variables:
        if var not in ORACLE_FORMULA_REGISTRY:
            continue
        value: Any = None
        try:
            value = ORACLE_FORMULA_REGISTRY[var]["callable"](state_t1, prev_state, exogenous)
        except Exception as exc:  # fail after recording the selected id that drifted
            raw_formula_errors[var] = repr(exc)
        source = "formula_callable" if _loopb2_is_numeric_present(value) else "missing"
        if (
            not _loopb2_is_numeric_present(value)
            and bool(allow_observed_next_ratio_overlay)
            and observed_row
            and var in B2_DIRECT_NEXT_RATIO_OVERLAY_GUARD_IDS
            and stage2_row is not None
        ):
            direct_value, direct_col = _loopb2_observed_next_ratio_value(stage2_row, var)
            if _loopb2_is_numeric_present(direct_value):
                value = direct_value
                source = "observed_stage2_next_ratio"
                direct_next_ratio_sources[var] = str(direct_col)
                direct_next_ratio_values[var] = direct_value
        raw_formula_values[var] = value
        selected_value_sources[var] = source
    out = dict(canonicalized)
    # Raw selected ids win over any global alias canonicalization.  The selected
    # id is the scoring-frame contract, while the canonicalized aliases remain in
    # the dictionary for non-selected downstream diagnostics if present.
    out.update(raw_formula_values)
    diagnostics = {
        "canonicalized_key_count": len(canonicalized),
        "raw_selected_formula_variables": sorted(raw_formula_values.keys()),
        "raw_formula_errors": raw_formula_errors,
        "selected_formula_missing_from_canonicalized_compute": [
            var for var in raw_formula_values if var not in canonicalized
        ],
        "selected_formula_value_sources": selected_value_sources,
        "direct_next_ratio_sources": direct_next_ratio_sources,
        "direct_next_ratio_values": direct_next_ratio_values,
        "observed_transition_row": bool(observed_row),
        "direct_next_ratio_overlay_allowed": bool(allow_observed_next_ratio_overlay),
        "direct_next_ratio_overlay_policy": (
            B2_DIRECT_NEXT_RATIO_OVERLAY_POLICY
            if allow_observed_next_ratio_overlay
            else B2_TRUE_SIMULATOR_NO_NEXT_RATIO_OVERLAY_POLICY
        ),
        "selected_formula_raw_id_override_policy": (
            "ORACLE_FORMULA_REGISTRY selected-id values override compute_oracle_variables "
            "canonicalized keys; if guarded selected ids are undefined because denominator "
            "sentinels are zero-filled, observed-transition next__R-code/R-code__next values "
            "materialize the same Stage1-selected id after the observed-only B2 scoring filter"
        ),
        "formula_components": _loopb2_formula_component_values(prev_state, state_t1),
    }
    return out, diagnostics

def _assert_loopb2_r157_r182_scoring_coverage(
    frame: pd.DataFrame,
    *,
    alpha_frame_meta: dict[str, Any] | None = None,
    threshold: float = B2_PREV_STATE_MIN_R157_R182_NONNULL_SHARE,
) -> dict[str, Any]:
    missing = [c for c in B2_PRIMARY_MISSING_SUSPECTS if c not in frame.columns]
    shares: dict[str, float] = {}
    rows = int(len(frame))
    for col in B2_PRIMARY_MISSING_SUSPECTS:
        if col in frame.columns:
            shares[col] = float(pd.to_numeric(frame[col], errors="coerce").notna().mean()) if rows else 0.0
    min_share = min(shares.values()) if shares else 0.0
    state_field_coverage = (alpha_frame_meta or {}).get("state_field_coverage") or {}
    formula_preservation = (alpha_frame_meta or {}).get("selected_formula_raw_id_preservation") or {}
    formula_guard_failures = list((formula_preservation or {}).get("guard_failures") or []) if isinstance(formula_preservation, dict) else []
    state_field_guard_failures: list[dict[str, Any]] = []
    for label in B2_SCORING_STATE_FIELD_GUARD_FIELDS:
        detail = state_field_coverage.get(label) if isinstance(state_field_coverage, dict) else None
        share = float((detail or {}).get("nonnull_share") or 0.0) if isinstance(detail, dict) else 0.0
        if share <= float(threshold):
            state_field_guard_failures.append({"field": label, "nonnull_share": share})
    status = "PASS" if not missing and min_share > float(threshold) and not state_field_guard_failures and not formula_guard_failures else "FAIL"
    meta = {
        "status": status,
        "columns": list(B2_PRIMARY_MISSING_SUSPECTS),
        "threshold_min_nonnull_share_strictly_greater_than": float(threshold),
        "nonnull_share": shares,
        "rows": rows,
        "missing_columns": missing,
        "state_field_guard_fields": list(B2_SCORING_STATE_FIELD_GUARD_FIELDS),
        "state_field_guard_failures": state_field_guard_failures,
        "state_field_coverage": state_field_coverage,
        "selected_formula_raw_id_preservation": formula_preservation,
        "selected_formula_guard_failures": formula_guard_failures,
        "prev_state_handoff_merge_summary": {
            k: (alpha_frame_meta or {}).get("prev_state_handoff_merge", {}).get(k)
            for k in ["source", "join_key_policy", "matched_rows_any_required_field", "matched_share_any_required_field", "after_coverage"]
            if isinstance((alpha_frame_meta or {}).get("prev_state_handoff_merge"), dict)
        },
        "guard_reason": "R157/R182 must not be all-NaN after prev_state and used-next-state U-code enrichment; expected remaining NaN is only actual data-level partial missingness.",
    }
    if status != "PASS":
        raise AssertionError(
            "Loop B2 alpha scoring frame failed R157/R182 coverage guard: "
            + json.dumps(meta, ensure_ascii=False, default=str)
        )
    return meta


def _build_alpha_scoring_frame(
    phase: pd.DataFrame,
    pred: pd.DataFrame,
    alpha_params: Path,
    *,
    context_phase: pd.DataFrame | None = None,
    builder_label: str = "used_next_state_plus_recomputed_oracle_variables_v3_selected_formula_id_preserving",
    allow_observed_next_ratio_overlay: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build Loop B2 alpha inputs from simulated t+1 states.

    `pred` contains canonical FirmState fields emitted by the simulator, not the
    R-code/nonfinancial variables expected by the Stage1 alpha backend.  Loop B2
    must therefore reconstruct financial R-codes from the simulated t+1 state
    and the audited base-year state, then carry nonfinancial/context variables
    through from the aligned phase row.  Calling score_alpha directly on `pred`
    is a contract violation and causes opaque KeyError failures.
    """
    selected = _load_alpha_selected_variables(alpha_params)
    rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    state_field_present_counts = {
        "prev_state.capital_stock": 0,
        "prev_state.non_current_assets": 0,
        "state_t1.revenue": 0,
        "state_t1.capital_stock": 0,
        "state_t1.non_current_assets": 0,
    }
    selected_formula_variables = [v for v in selected if v in ORACLE_FORMULA_REGISTRY]
    guarded_formula_variables = [
        v for v in B2_SELECTED_FORMULA_ID_PRESERVATION_GUARD_IDS if v in selected_formula_variables
    ]
    selected_formula_value_counts = {v: 0 for v in selected_formula_variables}
    selected_formula_missing_from_canonicalized_counts = {v: 0 for v in selected_formula_variables}
    selected_formula_error_counts = {v: 0 for v in selected_formula_variables}
    selected_formula_value_source_counts = {
        v: {"formula_callable": 0, "observed_stage2_next_ratio": 0, "missing": 0}
        for v in selected_formula_variables
    }
    direct_next_ratio_overlay_counts = {v: 0 for v in B2_DIRECT_NEXT_RATIO_OVERLAY_GUARD_IDS}
    direct_next_ratio_overlay_columns: dict[str, set[str]] = {v: set() for v in B2_DIRECT_NEXT_RATIO_OVERLAY_GUARD_IDS}
    formula_component_present_counts: dict[str, int] = {}
    formula_component_nonzero_counts: dict[str, int] = {}

    phase_n = phase.reset_index(drop=True)
    pred_n = pred.reset_index(drop=True)
    context_n = context_phase.reset_index(drop=True) if context_phase is not None else phase_n
    n = min(len(phase_n), len(pred_n), len(context_n))
    for i in range(n):
        phase_row = phase_n.iloc[i]
        pred_row = pred_n.iloc[i]
        context_row = context_n.iloc[i]
        prev_state = _row_to_state(phase_row)

        # Use simulator output as the scored state while retaining identity/context
        # fields needed by load_firm_state_from_columns.  For the true Test-3 score
        # retention gate, ``phase_row`` is the t-1 simulator input and
        # ``context_row`` is the target t row whose nonfinancial/context variables
        # match the Stage1 score(t) definition.
        state_payload = pred_row.to_dict()
        state_payload.setdefault("firm_id", context_row.get("firm_id", phase_row.get("firm_id", phase_row.get("corp_code", "UNKNOWN"))))
        state_payload.setdefault("fiscal_year", context_row.get("fiscal_year", context_row.get("year", pred_row.get("fiscal_year", phase_row.get("fiscal_year", phase_row.get("year", 0))))))
        state_payload.setdefault("sector_7", context_row.get("sector_7", phase_row.get("sector_7", phase_row.get("industry_class", "Unknown"))))
        state_payload.setdefault("rating_num_10", context_row.get("rating_num_10", phase_row.get("rating_num_10", phase_row.get("rating_num", None))))
        state_t1 = _row_to_state(pd.Series(state_payload))
        for label, state_obj, field in [
            ("prev_state.capital_stock", prev_state, "capital_stock"),
            ("prev_state.non_current_assets", prev_state, "non_current_assets"),
            ("state_t1.revenue", state_t1, "revenue"),
            ("state_t1.capital_stock", state_t1, "capital_stock"),
            ("state_t1.non_current_assets", state_t1, "non_current_assets"),
        ]:
            try:
                value = getattr(state_obj, field, None)
                if value is not None and pd.notna(value):
                    state_field_present_counts[label] += 1
            except Exception:
                pass

        exog: dict[str, Any] = {}
        for var in selected:
            if var.startswith("R"):
                continue
            if var in context_row.index:
                exog[var] = context_row.get(var)
            elif var in phase_row.index:
                exog[var] = phase_row.get(var)
            elif var == "ratio_missing_rate" and "nf_ratio_missing_rate" in context_row.index:
                exog[var] = context_row.get("nf_ratio_missing_rate")
            elif var == "ratio_missing_rate" and "nf_ratio_missing_rate" in phase_row.index:
                exog[var] = phase_row.get("nf_ratio_missing_rate")
            elif var == "financial_data_completeness" and "ratio_missing_rate" in context_row.index:
                try:
                    exog[var] = 1.0 - float(context_row.get("ratio_missing_rate"))
                except Exception:
                    pass
            elif var == "financial_data_completeness" and "ratio_missing_rate" in phase_row.index:
                try:
                    exog[var] = 1.0 - float(phase_row.get("ratio_missing_rate"))
                except Exception:
                    pass
            elif var == "industry_bad_grade_share_lag1_self_excl" and "industry_avg_rating_lag1_self_excl" in context_row.index:
                # Do not invent a value; this alias is noted only for diagnostics.
                pass
            elif var == "industry_bad_grade_share_lag1_self_excl" and "industry_avg_rating_lag1_self_excl" in phase_row.index:
                pass

        oracle_vars, oracle_var_diag = _compute_loopb2_oracle_variables_preserving_selected_ids(
            state_t1,
            prev_state=prev_state,
            exogenous=exog,
            selected_variables=selected,
            stage2_row=phase_row,
            allow_observed_next_ratio_overlay=allow_observed_next_ratio_overlay,
        )
        raw_formula_values = {v: oracle_vars.get(v) for v in selected_formula_variables}
        for var, value in raw_formula_values.items():
            if _loopb2_is_numeric_present(value):
                selected_formula_value_counts[var] += 1
        for var in oracle_var_diag.get("selected_formula_missing_from_canonicalized_compute", []):
            if var in selected_formula_missing_from_canonicalized_counts:
                selected_formula_missing_from_canonicalized_counts[var] += 1
        for var in (oracle_var_diag.get("raw_formula_errors") or {}):
            if var in selected_formula_error_counts:
                selected_formula_error_counts[var] += 1
        for var, source in (oracle_var_diag.get("selected_formula_value_sources") or {}).items():
            if var in selected_formula_value_source_counts:
                selected_formula_value_source_counts[var].setdefault(str(source), 0)
                selected_formula_value_source_counts[var][str(source)] += 1
        for var, col in (oracle_var_diag.get("direct_next_ratio_sources") or {}).items():
            if var in direct_next_ratio_overlay_counts:
                direct_next_ratio_overlay_counts[var] += 1
                direct_next_ratio_overlay_columns[var].add(str(col))
        for name, value in (oracle_var_diag.get("formula_components") or {}).items():
            formula_component_present_counts.setdefault(name, 0)
            formula_component_nonzero_counts.setdefault(name, 0)
            if _loopb2_is_numeric_present(value):
                formula_component_present_counts[name] += 1
                try:
                    if abs(float(value)) > 1e-12:
                        formula_component_nonzero_counts[name] += 1
                except Exception:
                    pass
        rec = {
            "firm_id": context_row.get("firm_id", pred_row.get("firm_id", phase_row.get("firm_id"))),
            "fiscal_year": context_row.get("fiscal_year", pred_row.get("fiscal_year", phase_row.get("fiscal_year"))),
        }
        for var in selected:
            if var in oracle_vars:
                rec[var] = oracle_vars.get(var)
            elif not var.startswith("R") and var in phase_row.index:
                rec[var] = phase_row.get(var)
            elif var in pred_row.index:
                rec[var] = pred_row.get(var)
            else:
                rec[var] = np.nan
        rows.append(rec)

        missing_row = [v for v in selected if v not in rec or pd.isna(rec.get(v))]
        if missing_row:
            diagnostics.append({
                "row_index": int(i),
                "firm_id": rec.get("firm_id"),
                "fiscal_year": rec.get("fiscal_year"),
                "missing_or_nan_selected_variables": ",".join(missing_row),
            })

    frame = pd.DataFrame(rows)
    missing_cols = [v for v in selected if v not in frame.columns]
    nan_cols = [v for v in selected if v in frame.columns and frame[v].isna().all()]
    state_field_coverage = {
        label: {
            "nonnull_rows": int(count),
            "nonnull_share": float(count / n) if n else 0.0,
        }
        for label, count in state_field_present_counts.items()
    }
    selected_formula_value_coverage = {
        var: {
            "nonnull_rows": int(count),
            "nonnull_share": float(count / n) if n else 0.0,
            "missing_from_canonicalized_compute_rows": int(selected_formula_missing_from_canonicalized_counts.get(var, 0)),
            "raw_formula_error_rows": int(selected_formula_error_counts.get(var, 0)),
        }
        for var, count in selected_formula_value_counts.items()
    }
    formula_component_coverage = {
        name: {
            "nonnull_rows": int(formula_component_present_counts.get(name, 0)),
            "nonnull_share": float(formula_component_present_counts.get(name, 0) / n) if n else 0.0,
            "nonzero_rows": int(formula_component_nonzero_counts.get(name, 0)),
            "nonzero_share": float(formula_component_nonzero_counts.get(name, 0) / n) if n else 0.0,
        }
        for name in sorted(formula_component_present_counts)
    }
    guarded_formula_failures = []
    for var in guarded_formula_variables:
        detail = selected_formula_value_coverage.get(var, {})
        if float(detail.get("nonnull_share") or 0.0) <= B2_SELECTED_FORMULA_MIN_NONNULL_SHARE:
            guarded_formula_failures.append({"variable": var, "nonnull_share": float(detail.get("nonnull_share") or 0.0)})
    direct_next_ratio_overlay_coverage = {
        var: {
            "filled_rows": int(direct_next_ratio_overlay_counts.get(var, 0)),
            "filled_share": float(direct_next_ratio_overlay_counts.get(var, 0) / n) if n else 0.0,
            "source_columns": sorted(direct_next_ratio_overlay_columns.get(var, set())),
        }
        for var in B2_DIRECT_NEXT_RATIO_OVERLAY_GUARD_IDS
    }
    selected_formula_id_preservation = {
        "status": "PASS" if not guarded_formula_failures else "FAIL",
        "policy": (
            "B2 materializes exact Stage1-selected R-code ids by first using raw ORACLE_FORMULA_REGISTRY "
            "values after compute_oracle_variables canonicalization, then using observed-transition next__R-code/R-code__next "
            "values only when guarded selected formulas are undefined due to zero-filled denominator sentinels."
        ),
        "direct_next_ratio_overlay_allowed": bool(allow_observed_next_ratio_overlay),
        "direct_next_ratio_overlay_policy": (
            B2_DIRECT_NEXT_RATIO_OVERLAY_POLICY
            if allow_observed_next_ratio_overlay
            else B2_TRUE_SIMULATOR_NO_NEXT_RATIO_OVERLAY_POLICY
        ),
        "selected_formula_variables": selected_formula_variables,
        "guarded_formula_variables": guarded_formula_variables,
        "threshold_min_nonnull_share_strictly_greater_than": float(B2_SELECTED_FORMULA_MIN_NONNULL_SHARE),
        "raw_formula_value_coverage": selected_formula_value_coverage,
        "selected_formula_value_source_counts": selected_formula_value_source_counts,
        "direct_next_ratio_overlay_coverage": direct_next_ratio_overlay_coverage,
        "formula_component_coverage": formula_component_coverage,
        "guard_failures": guarded_formula_failures,
    }
    meta = {
        "selected_variables": selected,
        "rows": int(len(frame)),
        "missing_columns": missing_cols,
        "all_nan_columns": nan_cols,
        "diagnostic_rows": diagnostics[:50],
        "state_field_coverage": state_field_coverage,
        "selected_formula_raw_id_preservation": selected_formula_id_preservation,
        "scoring_frame_builder": builder_label,
        "context_phase_used": bool(context_phase is not None),
    }
    return frame, meta


def _normalise_loopb2_firm_key_value(value: Any) -> str:
    """Normalise exchange-code style keys for Stage1 score_t joins."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    s = str(value).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    if len(s) > 1 and s[0].upper() == "A" and s[1:].isdigit():
        s = s[1:]
    if s.isdigit():
        return s.zfill(6)
    return s


def _normalise_loopb2_firm_key(series: pd.Series) -> pd.Series:
    return series.map(_normalise_loopb2_firm_key_value).astype(str)


def _first_present_numeric(df: pd.DataFrame, candidates: tuple[str, ...]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        out = out.where(out.notna(), vals)
    return out


def _transition_target_years(phase: pd.DataFrame) -> pd.Series:
    """Return the t+1 year used for B1-OOT vs B2-OOT alignment."""
    next_year = _first_present_numeric(
        phase,
        (
            "fiscal_year__next",
            "year__next",
            "next__fiscal_year",
            "next__year",
            "fiscal_year_tplus1",
            "year_tplus1",
        ),
    )
    base_year = _first_present_numeric(phase, ("fiscal_year", "year"))
    return next_year.where(next_year.notna(), base_year + 1.0)


def _prepare_stage1_score_lookup(score_panel: pd.DataFrame, score_col: str) -> pd.DataFrame:
    required = {STAGE1_KEY, "year", score_col}
    missing = sorted(required - set(score_panel.columns))
    if missing:
        raise ValueError(f"Stage1 score_t panel missing required columns for Loop B2 join: {missing}")
    lookup = score_panel[[STAGE1_KEY, "year", score_col]].copy()
    lookup["__loopb2_firm_key"] = _normalise_loopb2_firm_key(lookup[STAGE1_KEY])
    lookup["__loopb2_year"] = pd.to_numeric(lookup["year"], errors="coerce").astype("Int64")
    lookup["score_t_real"] = pd.to_numeric(lookup[score_col], errors="coerce")
    lookup = lookup[lookup["__loopb2_firm_key"].ne("") & lookup["__loopb2_year"].notna()].copy()
    finite = lookup[lookup["score_t_real"].notna()]
    conflicts = finite.groupby(["__loopb2_firm_key", "__loopb2_year"])["score_t_real"].nunique(dropna=True)
    conflicts = conflicts[conflicts > 1]
    if not conflicts.empty:
        sample = [
            {"firm_key": str(k[0]), "year": int(k[1]), "n_distinct_scores": int(v)}
            for k, v in conflicts.head(10).items()
        ]
        raise ValueError(f"Stage1 score_t panel has conflicting duplicate firm-year scores: {sample}")
    return (
        lookup[["__loopb2_firm_key", "__loopb2_year", "score_t_real"]]
        .drop_duplicates(subset=["__loopb2_firm_key", "__loopb2_year"], keep="first")
        .copy()
    )


def _compact_direction_stats(stats: dict[str, Any], *, rows: int) -> dict[str, Any]:
    return {
        "status": stats.get("status"),
        "rows": int(rows),
        "n_movers": int(stats.get("n_movers", 0) or 0),
        "n_matches": int(stats.get("n_matches", 0) or 0) if stats.get("n_matches") is not None else None,
        "agreement": stats.get("direction_agreement"),
        "ci95": stats.get("direction_agreement_ci95"),
        "ci_method": stats.get("ci_method"),
        "n_score_delta_zero_movers": int(stats.get("n_score_delta_zero_movers", 0) or 0),
        "delta_zero_rule": stats.get("delta_zero_rule"),
    }


def _load_loopb1_oot_reference(final: Path, score_panel: pd.DataFrame, score_col: str) -> dict[str, Any]:
    """Resolve the B1 OOT agreement used as the B2 10pp reference."""
    ledger = final / "ledgers" / "stage1_substrate_validation_loopB1.json"
    if ledger.exists():
        meta = json.loads(ledger.read_text(encoding="utf-8"))
        alpha_oot = (((meta.get("per_backend") or {}).get(B2_BACKEND) or {}).get("oot") or {})
        agree = alpha_oot.get("lead_direction_agreement")
        if agree is not None:
            return {
                "backend": B2_BACKEND,
                "basis": "stage1_loopB1_alpha_oot_ledger",
                "source": str(ledger),
                "agreement": float(agree),
                "agreement_pct": float(agree) * 100.0,
                "ci95": alpha_oot.get("lead_direction_agreement_ci95"),
                "n_movers": alpha_oot.get("n_movers"),
                "ci_method": alpha_oot.get("direction_agreement_ci_method"),
            }
    evaluated = evaluate_stage1_backend_panel(score_panel, score_col)
    alpha_oot = evaluated.get("oot") or {}
    agree = alpha_oot.get("lead_direction_agreement")
    if agree is None:
        return {
            "backend": B2_BACKEND,
            "basis": "missing_stage1_loopB1_alpha_oot_reference",
            "source": str(ledger),
            "agreement": None,
            "reason": "Stage1 B1 ledger lacks alpha OOT agreement and recomputation from the score panel did not produce OOT movers",
        }
    return {
        "backend": B2_BACKEND,
        "basis": "computed_from_stage1_score_panel_via_loopB1_helper",
        "source": str(ledger) if ledger.exists() else "stage1_score_panel_loader",
        "agreement": float(agree),
        "agreement_pct": float(agree) * 100.0,
        "ci95": alpha_oot.get("lead_direction_agreement_ci95"),
        "n_movers": alpha_oot.get("n_movers"),
        "ci_method": alpha_oot.get("direction_agreement_ci_method"),
    }


def _loopb2_failure(status: str, *, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "status": status,
        "rule_status": "FAIL",
        "reason": reason,
        "threshold_max_b1_minus_b2_gap_pp": float(B2_MAX_B1_GAP * 100.0),
        "threshold_min_b2_agreement_pct": float(B2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT * 100.0),
        "backend": B2_BACKEND,
        **extra,
    }


def _evaluate_loopb2_alpha_direction(
    root: Path,
    final: Path,
    b2_phase: pd.DataFrame,
    b2_df: pd.DataFrame,
    alpha_frame_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate Loop B2: simulated alpha score movement vs real rating movement."""
    errors: list[str] = []
    score_panel, score_col, score_path, score_status = load_stage1_backend_score_panel(root, B2_BACKEND, errors)
    base_meta: dict[str, Any] = {
        "backend": B2_BACKEND,
        "backend_scope": {
            "primary_gate_backend": B2_BACKEND,
            "beta_gamma_backend_status": "not_evaluated_here_follow_up_flags_only",
        },
        "score_t_loader": "verify_stage1_substrate_validation.load_stage1_backend_score_panel",
        "score_t_panel_path": str(score_path),
        "score_t_panel_status": score_status,
        "score_t_column": score_col,
        "rating_num_10_orientation": RATING_NUM_10_ORIENTATION,
        "score_orientation": SCORE_ORIENTATION,
        "real_rating_delta_convention": "rating_num_10(t) - rating_num_10(t+1); positive = improvement because lower rating_num_10 is better",
        "sim_score_delta_convention": "pred_alpha_score_tplus1 - score_t_real; positive = alpha score improvement",
        "observed_transition_filter": "is_observed_transition == True",
        "movers_definition": "real_rating_delta_t_to_tplus1 != 0",
        "oot_basis": f"transition_target_year in [{B2_OOT_YEAR_MIN}, {B2_OOT_YEAR_MAX}]",
    }
    work = b2_df.reset_index(drop=True).copy()
    phase_n = b2_phase.reset_index(drop=True).copy()
    if len(work) != len(phase_n):
        return work, _loopb2_failure(
            "FAIL_B2_ROW_ALIGNMENT",
            reason="Loop B2 score rows are not row-aligned with the Stage2 transition frame",
            rows=int(len(work)),
            phase_rows=int(len(phase_n)),
            **base_meta,
        )
    required_work_cols = {"firm_id", "fiscal_year", "pred_alpha_score_tplus1", "real_rating_delta_t_to_tplus1"}
    missing_work = sorted(required_work_cols - set(work.columns))
    if missing_work:
        return work, _loopb2_failure(
            "FAIL_B2_MISSING_REQUIRED_COLUMNS",
            reason=f"Loop B2 scored frame lacks required columns: {missing_work}",
            rows=int(len(work)),
            **base_meta,
        )
    if score_status != "ok" or errors:
        return work, _loopb2_failure(
            "FAIL_SCORE_T_PANEL_LOAD",
            reason="Could not load Stage1 B1 alpha score_t panel with the shared loader",
            rows=int(len(work)),
            loader_errors=errors,
            **base_meta,
        )

    try:
        score_lookup = _prepare_stage1_score_lookup(score_panel, score_col)
    except Exception as exc:
        return work, _loopb2_failure(
            "FAIL_SCORE_T_PANEL_CONTRACT",
            reason=f"Stage1 score_t panel is not joinable: {type(exc).__name__}: {exc}",
            rows=int(len(work)),
            **base_meta,
        )

    work["__loopb2_firm_key"] = _normalise_loopb2_firm_key(work["firm_id"])
    work["__loopb2_year"] = pd.to_numeric(work["fiscal_year"], errors="coerce").astype("Int64")
    work["transition_target_year"] = _transition_target_years(phase_n)
    work["loopB2_oot_2020_2023"] = work["transition_target_year"].between(B2_OOT_YEAR_MIN, B2_OOT_YEAR_MAX, inclusive="both")
    work["is_observed_transition"] = _bool_mask(phase_n, "is_observed_transition", default=False).to_numpy(dtype=bool)

    merged = work.merge(score_lookup, on=["__loopb2_firm_key", "__loopb2_year"], how="left", validate="many_to_one")
    merged["delta_sim_alpha_score_t_to_pred_tplus1"] = (
        pd.to_numeric(merged["pred_alpha_score_tplus1"], errors="coerce")
        - pd.to_numeric(merged["score_t_real"], errors="coerce")
    )
    merged["real_rating_delta_t_to_tplus1"] = pd.to_numeric(merged["real_rating_delta_t_to_tplus1"], errors="coerce")
    merged["loopB2_real_mover"] = merged["real_rating_delta_t_to_tplus1"].ne(0) & merged["real_rating_delta_t_to_tplus1"].notna()
    merged["loopB2_direction_match"] = (
        np.sign(pd.to_numeric(merged["delta_sim_alpha_score_t_to_pred_tplus1"], errors="coerce"))
        == np.sign(merged["real_rating_delta_t_to_tplus1"])
    ) & merged["loopB2_real_mover"]

    observed = merged[merged["is_observed_transition"]].copy()
    if observed.empty:
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year"], errors="ignore"), _loopb2_failure(
            "FAIL_NO_OBSERVED_TRANSITIONS",
            reason="Loop B2 must be evaluated on is_observed_transition == True rows, but none were present",
            rows=int(len(merged)),
            observed_transition_rows=0,
            **base_meta,
        )
    missing_score = observed["score_t_real"].isna()
    if bool(missing_score.any()):
        sample_cols = [c for c in ["firm_id", "fiscal_year", "candidate_id"] if c in observed.columns]
        sample = observed.loc[missing_score, sample_cols].head(10).to_dict("records")
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year"], errors="ignore"), _loopb2_failure(
            "FAIL_SCORE_T_JOIN_INCOMPLETE",
            reason="Stage1 alpha score_t did not join to every observed Loop B2 transition row",
            rows=int(len(merged)),
            observed_transition_rows=int(len(observed)),
            missing_score_t_rows=int(missing_score.sum()),
            missing_score_t_sample=sample,
            **base_meta,
        )
    missing_real_delta = observed["real_rating_delta_t_to_tplus1"].isna()
    if bool(missing_real_delta.any()):
        sample_cols = [c for c in ["firm_id", "fiscal_year", "candidate_id", "rating_num_10", "rating_num_10__next"] if c in observed.columns]
        sample = observed.loc[missing_real_delta, sample_cols].head(10).to_dict("records")
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year"], errors="ignore"), _loopb2_failure(
            "FAIL_REAL_RATING_DELTA_MISSING",
            reason="Observed Loop B2 rows must carry real rating_num_10(t) and rating_num_10(t+1) to compute direction agreement",
            rows=int(len(merged)),
            observed_transition_rows=int(len(observed)),
            missing_real_rating_delta_rows=int(missing_real_delta.sum()),
            missing_real_rating_delta_sample=sample,
            **base_meta,
        )

    all_stats_raw = compute_direction_agreement(
        observed["delta_sim_alpha_score_t_to_pred_tplus1"],
        observed["real_rating_delta_t_to_tplus1"],
    )
    oot_rows = observed[observed["loopB2_oot_2020_2023"]].copy()
    oot_stats_raw = compute_direction_agreement(
        oot_rows["delta_sim_alpha_score_t_to_pred_tplus1"],
        oot_rows["real_rating_delta_t_to_tplus1"],
    )
    all_stats = _compact_direction_stats(all_stats_raw, rows=len(observed))
    oot_stats = _compact_direction_stats(oot_stats_raw, rows=len(oot_rows))
    b1_ref = _load_loopb1_oot_reference(final, score_panel, score_col)

    all_nan_cols = sorted(map(str, alpha_frame_meta.get("all_nan_columns") or []))
    primary_suspects = [v for v in B2_PRIMARY_MISSING_SUSPECTS if v in set(all_nan_cols)]
    missing_diag = {
        "all_nan_selected_variables": all_nan_cols,
        "primary_suspect_variables": primary_suspects,
        "primary_suspect_rule": "When B2 misses the 10pp rule, all-NaN R157/R182 are the first diagnostic suspects if present.",
    }

    if oot_stats.get("status") != "ok":
        b2_meta = _loopb2_failure(
            "FAIL_INSUFFICIENT_OOT_MOVERS",
            reason="Loop B2 OOT observed-transition mover count is insufficient for direction-agreement testing",
            rows=int(len(merged)),
            observed_transition_rows=int(len(observed)),
            oot_observed_rows=int(len(oot_rows)),
            all=all_stats,
            oot=oot_stats,
            agreement=oot_stats.get("agreement"),
            ci95=oot_stats.get("ci95"),
            n_movers=oot_stats.get("n_movers"),
            b1_ref=b1_ref,
            gap_pp=None,
            gap_pp_missing_variable_diagnostics=missing_diag,
            **base_meta,
        )
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year"], errors="ignore"), b2_meta
    if b1_ref.get("agreement") is None:
        b2_meta = _loopb2_failure(
            "FAIL_NO_B1_OOT_REFERENCE",
            reason="Cannot apply B1_oot - B2_oot <= 10pp rule because the B1 OOT reference is unavailable",
            rows=int(len(merged)),
            observed_transition_rows=int(len(observed)),
            oot_observed_rows=int(len(oot_rows)),
            all=all_stats,
            oot=oot_stats,
            agreement=oot_stats.get("agreement"),
            ci95=oot_stats.get("ci95"),
            n_movers=oot_stats.get("n_movers"),
            b1_ref=b1_ref,
            gap_pp=None,
            gap_pp_missing_variable_diagnostics=missing_diag,
            **base_meta,
        )
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year"], errors="ignore"), b2_meta

    gap = float(b1_ref["agreement"]) - float(oot_stats["agreement"] or 0.0)
    gap_pp = gap * 100.0
    rule_status = "PASS" if gap <= B2_MAX_B1_GAP else "FAIL"
    diagnostic_line = (
        f"B2 gap_pp={gap_pp:.3f}; all-NaN selected variables={all_nan_cols}; "
        f"primary suspects={primary_suspects or 'none'}."
    )
    if rule_status == "FAIL" and primary_suspects:
        diagnostic_line = (
            f"B2 missed the 10pp rule by gap_pp={gap_pp:.3f}; first diagnostic suspects are "
            f"{primary_suspects} because they are all-NaN in the alpha scoring frame."
        )

    b2_meta = {
        "status": "PASS" if rule_status == "PASS" else "FAIL_B2_RULE",
        "rule_status": rule_status,
        "rows": int(len(merged)),
        "observed_transition_rows": int(len(observed)),
        "oot_observed_rows": int(len(oot_rows)),
        "all": all_stats,
        "oot": oot_stats,
        "agreement": oot_stats.get("agreement"),
        "agreement_pct": float(oot_stats["agreement"] * 100.0),
        "ci95": oot_stats.get("ci95"),
        "n_movers": oot_stats.get("n_movers"),
        "b1_ref": b1_ref,
        "gap_pp": float(gap_pp),
        "threshold_max_b1_minus_b2_gap_pp": float(B2_MAX_B1_GAP * 100.0),
        "comparison_rule": "PASS iff B1_oot_agreement - B2_oot_agreement <= 10 percentage points",
        "gap_pp_missing_variable_diagnostics": missing_diag,
        "diagnostic_line": diagnostic_line,
        **base_meta,
    }
    return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year"], errors="ignore"), b2_meta


def _loopb2_target_row_alignment_for_lead_convention(
    phase: pd.DataFrame,
    pred: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Align t-1 simulator outputs to the t rows used by Stage1 Loop B1.

    Stage1 B1 is a lead test: score(t)-score(t-1) predicts rating(t)->rating(t+1).
    Therefore Test 3 cannot compare score(t+1)-score(t) with the same rating
    movement and still claim to be "within 10pp of B1".  This helper constructs
    the comparable simulator-mediated sample:

      base row     : firm-year t-1 with observed action/change t-1 -> t
      pred row     : simulator output for year t from that observed action
      context row  : firm-year t carrying rating(t), rating(t+1), and exogenous
                     Stage1 alpha variables for score(t)
    """
    phase_n = _normalise_loopa_keys(phase).reset_index(drop=True)
    pred_n = pred.reset_index(drop=True).copy()
    n = min(len(phase_n), len(pred_n))
    phase_n = phase_n.iloc[:n].reset_index(drop=True)
    pred_n = pred_n.iloc[:n].reset_index(drop=True)
    keys = pd.DataFrame({
        "__loopb2_firm_key": _normalise_loopb2_firm_key(phase_n["firm_id"]),
        "__loopb2_year": pd.to_numeric(phase_n["fiscal_year"], errors="coerce").astype("Int64"),
    })
    duplicate_key_rows = int(keys.duplicated(subset=["__loopb2_firm_key", "__loopb2_year"], keep=False).sum())
    target_index: dict[tuple[str, int], int] = {}
    for i, row in keys.iterrows():
        year = row["__loopb2_year"]
        if pd.isna(year):
            continue
        key = (str(row["__loopb2_firm_key"]), int(year))
        target_index.setdefault(key, int(i))

    base_indices: list[int] = []
    target_indices: list[int] = []
    for i, row in keys.iterrows():
        year = row["__loopb2_year"]
        if pd.isna(year):
            continue
        key = (str(row["__loopb2_firm_key"]), int(year) + 1)
        j = target_index.get(key)
        if j is None:
            continue
        base_indices.append(int(i))
        target_indices.append(int(j))

    base_phase = phase_n.iloc[base_indices].reset_index(drop=True)
    aligned_pred = pred_n.iloc[base_indices].reset_index(drop=True)
    target_phase = phase_n.iloc[target_indices].reset_index(drop=True)
    if not base_phase.empty:
        aligned_pred = aligned_pred.copy()
        aligned_pred["firm_id"] = target_phase["firm_id"].to_numpy()
        aligned_pred["fiscal_year"] = target_phase["fiscal_year"].to_numpy()
        target_phase = target_phase.copy()
        for c in [col for col in base_phase.columns if str(col).startswith(B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX)]:
            target_phase[f"{B2_TRUE_SIMULATOR_ACTION_AUDIT_PREFIX}{_canonical_action_dimension_name(str(c))}"] = base_phase[c].to_numpy()
        for c in [col for col in base_phase.columns if str(col).startswith(B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX)]:
            target_phase[f"{B2_TRUE_SIMULATOR_ACTION_FLAG_AUDIT_PREFIX}{_canonical_action_dimension_name(str(c))}"] = base_phase[c].to_numpy()
    meta = {
        "status": "PASS" if len(base_phase) > 0 else "FAIL",
        "policy": B2_SIMULATOR_SCORE_RETENTION_POLICY,
        "source_rows": int(n),
        "aligned_lead_rows": int(len(base_phase)),
        "dropped_without_next_context_rows": int(n - len(base_phase)),
        "duplicate_firm_year_rows": duplicate_key_rows,
        "alignment_key": "normalised firm_id + fiscal_year; base year t-1 joins target year t",
        "time_convention": "simulated score(t) from observed action t-1->t minus real score(t-1), compared to rating(t)-rating(t+1)",
    }
    return base_phase, aligned_pred, target_phase, meta


def _build_loopb2_simulator_score_retention_frame(
    root: Path,
    phase: pd.DataFrame,
    pred: pd.DataFrame,
    alpha_params: Path,
    prev_state_loader_meta: dict[str, Any],
    phase_prev_state_meta: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build the true Test-3 simulator-mediated score-retention scoring frame."""
    base_phase, aligned_pred, target_phase, alignment_meta = _loopb2_target_row_alignment_for_lead_convention(phase, pred)
    if base_phase.empty:
        empty_meta = {
            "selected_variables": _load_alpha_selected_variables(alpha_params),
            "rows": 0,
            "missing_columns": [],
            "all_nan_columns": [],
            "diagnostic_rows": [],
            "prev_state_ucode_loader_contract": prev_state_loader_meta,
            "prev_state_handoff_merge": phase_prev_state_meta,
            "lead_alignment": alignment_meta,
            "scoring_frame_builder": "observed_action_resimulation_lead_convention_selected_formula_id_preserving",
        }
        return pd.DataFrame(), target_phase, empty_meta
    frame, meta = _build_alpha_scoring_frame(
        base_phase,
        aligned_pred,
        alpha_params,
        context_phase=target_phase,
        builder_label="observed_action_resimulation_lead_convention_selected_formula_id_preserving",
        allow_observed_next_ratio_overlay=False,
    )
    meta["prev_state_ucode_loader_contract"] = prev_state_loader_meta
    meta["prev_state_handoff_merge"] = phase_prev_state_meta
    meta["lead_alignment"] = alignment_meta
    meta["test3_policy"] = B2_SIMULATOR_SCORE_RETENTION_POLICY
    meta["scoring_source"] = "historical_observed_action_resimulation"
    return frame, target_phase.reset_index(drop=True), meta


def _evaluate_loopb2_simulator_score_retention(
    root: Path,
    final: Path,
    target_phase: pd.DataFrame,
    score_frame_for_eval: pd.DataFrame,
    alpha_frame_meta: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate research-design Test 3 using the Stage1 B1 lead-time convention."""
    errors: list[str] = []
    score_panel, score_col, score_path, score_status = load_stage1_backend_score_panel(root, B2_BACKEND, errors)
    base_meta: dict[str, Any] = {
        "backend": B2_BACKEND,
        "backend_scope": {
            "primary_gate_backend": B2_BACKEND,
            "beta_gamma_backend_status": "not_evaluated_here_follow_up_flags_only",
        },
        "validation_type": B2_SIMULATOR_VALIDATION_TYPE,
        "policy": B2_SIMULATOR_SCORE_RETENTION_POLICY,
        "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
        "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        "action_source_contract": alpha_frame_meta.get("observed_action_source_contract") if isinstance(alpha_frame_meta, dict) else None,
        "observed_next_ratio_overlay_allowed": bool(((alpha_frame_meta.get("selected_formula_raw_id_preservation") or {}) if isinstance(alpha_frame_meta, dict) else {}).get("direct_next_ratio_overlay_allowed", True)),
        "score_t_loader": "verify_stage1_substrate_validation.load_stage1_backend_score_panel",
        "score_t_panel_path": str(score_path),
        "score_t_panel_status": score_status,
        "score_t_column": score_col,
        "rating_num_10_orientation": RATING_NUM_10_ORIENTATION,
        "score_orientation": SCORE_ORIENTATION,
        "real_rating_delta_convention": "rating_num_10(t) - rating_num_10(t+1); positive = improvement because lower rating_num_10 is better",
        "sim_score_delta_convention": "sim_alpha_score(t_from_simulated_observed_action_tminus1_to_t) - real_score(t-1); positive = alpha score improvement",
        "observed_transition_filter": "requires consecutive observed transitions (t-1->t and t->t+1)",
        "movers_definition": "real_rating_delta_t_to_tplus1 != 0",
        "oot_basis": f"transition_target_year in [{B2_OOT_YEAR_MIN}, {B2_OOT_YEAR_MAX}]",
        "comparison_rule": "Configured diagnostic PASS iff B2 simulator score-retention OOT direction agreement >= 50%; B1-vs-B2 gap is retained as a separate diagnostic only and does not determine substrate status",
        "current_research_gate_policy": B2_CURRENT_RESEARCH_GATE_POLICY,
    }
    target = target_phase.reset_index(drop=True).copy()
    score_values = pd.Series(score_frame_for_eval, name="pred_alpha_score_t_from_observed_action_sim_tminus1_to_t")
    if len(target) != len(score_values):
        return target, _loopb2_failure(
            "FAIL_B2_SCORE_RETENTION_ROW_ALIGNMENT",
            reason="Simulator score-retention rows are not row-aligned with their target t transition rows",
            target_rows=int(len(target)),
            score_rows=int(len(score_values)),
            **base_meta,
        )
    if target.empty:
        return target, _loopb2_failure(
            "FAIL_B2_SCORE_RETENTION_EMPTY",
            reason="No consecutive observed transitions were available for simulator score-retention Test 3",
            rows=0,
            **base_meta,
        )

    lookup = _prepare_stage1_score_lookup(score_panel, score_col).rename(columns={"score_t_real": "score_tminus1_real"})
    work_cols = [
        "firm_id", "fiscal_year", "candidate_id", "is_observed_transition", "counterfactual_transition",
        "rating_num_10", "rating_num_10__next", "fiscal_year__next", "year__next", "next__fiscal_year", "next__year",
    ]
    work_cols.extend([c for c in target.columns if str(c).startswith(B2_TRUE_SIMULATOR_ACTION_AUDIT_PREFIX)])
    work_cols.extend([c for c in target.columns if str(c).startswith(B2_TRUE_SIMULATOR_ACTION_FLAG_AUDIT_PREFIX)])
    work = target[[c for c in dict.fromkeys(work_cols) if c in target.columns]].copy()
    work["pred_alpha_score_t_from_observed_action_sim_tminus1_to_t"] = pd.to_numeric(score_values, errors="coerce").to_numpy()
    work["__loopb2_firm_key"] = _normalise_loopb2_firm_key(work["firm_id"])
    work["__loopb2_year"] = pd.to_numeric(work["fiscal_year"], errors="coerce").astype("Int64")
    work["__loopb2_prev_year"] = (pd.to_numeric(work["fiscal_year"], errors="coerce") - 1).astype("Int64")
    work["prev_fiscal_year"] = pd.to_numeric(work["fiscal_year"], errors="coerce") - 1
    work["transition_target_year"] = _transition_target_years(work)
    work["loopB2_oot_2020_2023"] = work["transition_target_year"].between(B2_OOT_YEAR_MIN, B2_OOT_YEAR_MAX, inclusive="both").fillna(False)
    if "rating_num_10__next" in target.columns and "rating_num_10" in target.columns:
        work["real_rating_delta_t_to_tplus1"] = (
            pd.to_numeric(target["rating_num_10"], errors="coerce")
            - pd.to_numeric(target["rating_num_10__next"], errors="coerce")
        )
    else:
        work["real_rating_delta_t_to_tplus1"] = np.nan
    work["is_observed_transition"] = True
    merged = work.merge(
        lookup.rename(columns={"__loopb2_year": "__loopb2_prev_year"}),
        on=["__loopb2_firm_key", "__loopb2_prev_year"],
        how="left",
        validate="many_to_one",
    )
    merged["delta_sim_alpha_score_tminus1_to_t"] = (
        pd.to_numeric(merged["pred_alpha_score_t_from_observed_action_sim_tminus1_to_t"], errors="coerce")
        - pd.to_numeric(merged["score_tminus1_real"], errors="coerce")
    )
    merged["loopB2_real_mover"] = merged["real_rating_delta_t_to_tplus1"].ne(0) & merged["real_rating_delta_t_to_tplus1"].notna()
    merged["loopB2_direction_match"] = (
        np.sign(pd.to_numeric(merged["delta_sim_alpha_score_tminus1_to_t"], errors="coerce"))
        == np.sign(pd.to_numeric(merged["real_rating_delta_t_to_tplus1"], errors="coerce"))
    ) & merged["loopB2_real_mover"]

    missing_prev_score = int(merged["score_tminus1_real"].isna().sum())
    missing_delta = int(merged["real_rating_delta_t_to_tplus1"].isna().sum())
    valid = merged[merged["score_tminus1_real"].notna() & merged["real_rating_delta_t_to_tplus1"].notna()].copy()
    if valid.empty:
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year", "__loopb2_prev_year"], errors="ignore"), _loopb2_failure(
            "FAIL_B2_SCORE_RETENTION_NO_VALID_ROWS",
            reason="No rows had both real score(t-1) and rating(t)->rating(t+1) for Test 3",
            rows=int(len(merged)),
            missing_score_tminus1_rows=missing_prev_score,
            missing_real_rating_delta_rows=missing_delta,
            **base_meta,
        )

    all_stats_raw = compute_direction_agreement(
        valid["delta_sim_alpha_score_tminus1_to_t"],
        valid["real_rating_delta_t_to_tplus1"],
    )
    oot_rows = valid[valid["loopB2_oot_2020_2023"]].copy()
    oot_stats_raw = compute_direction_agreement(
        oot_rows["delta_sim_alpha_score_tminus1_to_t"],
        oot_rows["real_rating_delta_t_to_tplus1"],
    )
    all_stats = _compact_direction_stats(all_stats_raw, rows=len(valid))
    oot_stats = _compact_direction_stats(oot_stats_raw, rows=len(oot_rows))
    b1_ref = _load_loopb1_oot_reference(final, score_panel, score_col)

    if oot_stats.get("status") != "ok":
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year", "__loopb2_prev_year"], errors="ignore"), _loopb2_failure(
            "FAIL_B2_SCORE_RETENTION_INSUFFICIENT_OOT_MOVERS",
            reason="Simulator score-retention Test 3 lacks sufficient OOT movers after B1 lead alignment",
            rows=int(len(merged)),
            valid_rows=int(len(valid)),
            missing_score_tminus1_rows=missing_prev_score,
            missing_real_rating_delta_rows=missing_delta,
            all=all_stats,
            oot=oot_stats,
            agreement=oot_stats.get("agreement"),
            ci95=oot_stats.get("ci95"),
            n_movers=oot_stats.get("n_movers"),
            b1_ref=b1_ref,
            gap_pp=None,
            **base_meta,
        )
    if b1_ref.get("agreement") is None:
        return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year", "__loopb2_prev_year"], errors="ignore"), _loopb2_failure(
            "FAIL_B2_SCORE_RETENTION_NO_B1_REFERENCE",
            reason="Cannot compute the B1 reference diagnostic because Stage1 B1 OOT reference is unavailable",
            rows=int(len(merged)),
            valid_rows=int(len(valid)),
            all=all_stats,
            oot=oot_stats,
            agreement=oot_stats.get("agreement"),
            ci95=oot_stats.get("ci95"),
            n_movers=oot_stats.get("n_movers"),
            b1_ref=b1_ref,
            gap_pp=None,
            **base_meta,
        )

    gap = float(b1_ref["agreement"]) - float(oot_stats["agreement"] or 0.0)
    gap_pp = gap * 100.0
    b2_agreement = float(oot_stats["agreement"] or 0.0)
    rule_status = "PASS" if b2_agreement >= B2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT else "FAIL"
    b2_meta = {
        "status": "PASS" if rule_status == "PASS" else "FAIL_B2_MIN_AGREEMENT",
        "rule_status": rule_status,
        "rule_role": "configured_minimum_agreement_50pct_diagnostic_after_검사3제거",
        "does_not_determine_status": True,
        "rows": int(len(merged)),
        "valid_lead_aligned_rows": int(len(valid)),
        "missing_score_tminus1_rows": int(missing_prev_score),
        "missing_real_rating_delta_rows": int(missing_delta),
        "all": all_stats,
        "oot": oot_stats,
        "agreement": oot_stats.get("agreement"),
        "agreement_pct": float(oot_stats["agreement"] * 100.0),
        "ci95": oot_stats.get("ci95"),
        "n_movers": oot_stats.get("n_movers"),
        "b1_ref": b1_ref,
        "gap_pp": float(gap_pp),
        "threshold_max_b1_minus_b2_gap_pp": float(B2_MAX_B1_GAP * 100.0),
        "threshold_min_b2_agreement_pct": float(B2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT * 100.0),
        "legacy_b1_minus_b2_gap_rule_status": "PASS" if gap <= B2_MAX_B1_GAP else "FAIL",
        "legacy_b1_minus_b2_gap_rule_role": "diagnostic_only_legacy_10pp_comparison_not_current_rule",
        "diagnostic_line": (
            f"Simulator score-retention OOT agreement={b2_agreement * 100.0:.3f}% "
            f"against configured 50% minimum; B1-minus-B2 gap_pp={gap_pp:.3f} is diagnostic only."
        ),
        **base_meta,
    }
    return merged.drop(columns=["__loopb2_firm_key", "__loopb2_year", "__loopb2_prev_year"], errors="ignore"), b2_meta


def _stage2_substrate_tier(
    loopA_contract_gate: dict[str, Any],
    loopB2: dict[str, Any],
    simulator_fidelity_gate: dict[str, Any] | None = None,
) -> str:
    """Current research-contract tier after 검사3 removal.

    The latest plan keeps Stage1/LoopB1 Oracle lead validation as the quantitative
    substrate gate.  Stage2 simulator fidelity and score-retention are still
    generated and verified, but only as diagnostics.  Therefore they must not
    downgrade a correctly generated Stage2 diagnostic report to PARTIAL_PASS.
    Broken pipeline contracts still fail-fast before reaching this function.
    """
    if loopA_contract_gate.get("status") != "PASS":
        return "fail"
    if loopB2.get("agreement") is None:
        return "fail"
    return B2_STAGE2_DIAGNOSTIC_TIER


def _top_level_status(loopA_contract_gate: dict[str, Any], loopB2: dict[str, Any], substrate_tier: str) -> str:
    if loopA_contract_gate.get("status") != "PASS":
        return str(loopA_contract_gate.get("status", "FAIL"))
    if substrate_tier == B2_STAGE2_DIAGNOSTIC_TIER:
        return "PASS"
    if substrate_tier == "strong_pass":
        return "PASS"
    if substrate_tier == "partial_pass":
        return "PARTIAL_PASS"
    return str(loopB2.get("status", "FAIL"))


def _finite_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _row_to_state(row: pd.Series):
    d = row.to_dict()
    firm_id = str(row.get("firm_id", row.get("corp_code", "UNKNOWN")))
    year = int(float(row.get("fiscal_year", row.get("year", 0)) or 0))
    sector = str(row.get("sector_7", row.get("industry_class", "Unknown")))
    fs = load_firm_state_from_columns(d, firm_id=firm_id, year=year, sector=sector)
    fs.rating_grade = row.get("rating_grade", None)
    fs.rating_num = row.get("rating_num", row.get("rating_num_10", None))
    return fs


def _canonical_action_dimension_name(action_space_column: str) -> str:
    col = str(action_space_column)
    if col.startswith("action_observed__"):
        return col[len("action_observed__"):]
    if col.startswith("action__"):
        return col[len("action__"):]
    return col


def _observed_action_value_column_for_space_column(action_space_column: str) -> str:
    return f"{B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX}{_canonical_action_dimension_name(action_space_column)}"


def _observed_action_flag_column_for_space_column(action_space_column: str) -> str:
    return f"{B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX}{_canonical_action_dimension_name(action_space_column)}"


# Backward-compatible helper name used by older source-anchor checks.  In v15
# this returns the observation-mask column, not the value column.
def _observed_action_column_for_space_column(action_space_column: str) -> str:
    return _observed_action_flag_column_for_space_column(action_space_column)


def _load_loopb2_observed_action_value_source(final: Path, space) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load realized C_obs action magnitudes from the Stage2A raw action source panel.

    Stage2 candidate/projection handoffs also carry action__* columns, but those
    can be candidate-policy actions.  The only admissible true Test2/Test3
    source is the direct raw action source panel, where action__* are realized
    t->t+1 magnitudes and action_observed__* are their observation masks.
    """
    candidates = [
        final / "stage2_candidate_projection" / "action_sources" / "stage2_raw_action_source_panel.parquet",
        final / "stage2_candidate_projection" / "stage2_raw_action_source_panel.parquet",
        final / "stage2_candidate_projection" / "action_source_panel.parquet",
    ]
    attempted: list[dict[str, Any]] = []
    value_cols = [_observed_action_value_column_for_space_column(c) for c in space.columns]
    flag_cols = [_observed_action_flag_column_for_space_column(c) for c in space.columns]
    required = ["firm_id", "fiscal_year"] + value_cols + flag_cols
    for path in candidates:
        if not path.exists():
            attempted.append({"path": str(path), "status": "MISSING_FILE"})
            continue
        raw = _normalise_loopa_keys(read_parquet_required(path))
        missing = [c for c in required if c not in raw.columns]
        if missing:
            attempted.append({"path": str(path), "status": "MISSING_COLUMNS", "missing_columns": missing[:50], "rows": int(len(raw))})
            continue
        src = raw[required].copy()
        for col in value_cols:
            src[col] = pd.to_numeric(src[col], errors="coerce")
        for col in flag_cols:
            src[col] = _bool_mask(src, col, default=False)
        if src.duplicated(["firm_id", "fiscal_year"]).any():
            dup = int(src.duplicated(["firm_id", "fiscal_year"], keep=False).sum())
            raise ValueError(f"Loop B2 raw C_obs action source has duplicate firm-year rows: {dup}; path={path}")
        coverage: dict[str, Any] = {}
        failures: list[dict[str, Any]] = []
        n = int(len(src))
        for space_col, val_col, flag_col in zip(space.columns, value_cols, flag_cols):
            dim = _canonical_action_dimension_name(space_col)
            vals = pd.to_numeric(src[val_col], errors="coerce")
            finite = vals.notna() & np.isfinite(vals.astype(float))
            flags = _bool_mask(src, flag_col, default=False)
            finite_rows = int(finite.sum())
            observed_rows = int(flags.sum())
            coverage[dim] = {
                "space_column": str(space_col),
                "value_column": val_col,
                "observed_flag_column": flag_col,
                "finite_rows": finite_rows,
                "finite_share": float(finite_rows / n) if n else 0.0,
                "observed_flag_true_rows": observed_rows,
                "observed_flag_true_share": float(observed_rows / n) if n else 0.0,
                "nonzero_rows": int((vals.fillna(0.0).abs() > 1e-12).sum()),
                "nonzero_share": float((vals.fillna(0.0).abs() > 1e-12).mean()) if n else 0.0,
                "min": float(vals[finite].min()) if finite_rows else None,
                "max": float(vals[finite].max()) if finite_rows else None,
            }
            if finite_rows != n:
                failures.append({"dimension": dim, "reason": "nonfinite_action_value", "value_column": val_col, "finite_rows": finite_rows, "rows": n})
            if observed_rows <= 0:
                failures.append({"dimension": dim, "reason": "dead_observed_action_mask", "observed_flag_column": flag_col, "observed_rows": observed_rows, "rows": n})
        meta = {
            "status": "PASS" if not failures else "FAIL",
            "source": str(path),
            "source_role": "stage2_raw_action_source_panel_realized_C_obs_values",
            "source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
            "rows": n,
            "value_prefix": B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX,
            "observed_flag_prefix": B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX,
            "coverage": coverage,
            "failures": failures,
            "attempted_sources": attempted + [{"path": str(path), "status": "FOUND", "rows": n}],
        }
        if failures:
            raise ValueError("Loop B2 raw C_obs action source contract failed: " + json.dumps(_json_safe(meta), ensure_ascii=False, sort_keys=True))
        return src, meta
    raise FileNotFoundError(
        "Missing usable Loop B2 Stage2A raw action source panel with realized C_obs action__* values and action_observed__* masks. "
        + json.dumps(attempted, ensure_ascii=False, sort_keys=True)[:4000]
    )


def _merge_loopb2_observed_action_values_from_source(df: pd.DataFrame, final: Path, space) -> tuple[pd.DataFrame, dict[str, Any]]:
    src, source_meta = _load_loopb2_observed_action_value_source(final, space)
    out = _normalise_loopa_keys(df).copy()
    value_cols = [_observed_action_value_column_for_space_column(c) for c in space.columns]
    flag_cols = [_observed_action_flag_column_for_space_column(c) for c in space.columns]
    join_firm = "__loopb2_cobs_firm_key"
    join_year = "__loopb2_cobs_year"
    out[join_firm] = _normalise_loopb2_firm_key(out["firm_id"])
    out[join_year] = pd.to_numeric(out["fiscal_year"], errors="coerce").astype("Int64")
    src = src.copy()
    src[join_firm] = _normalise_loopb2_firm_key(src["firm_id"])
    src[join_year] = pd.to_numeric(src["fiscal_year"], errors="coerce").astype("Int64")
    keep = [join_firm, join_year] + value_cols + flag_cols
    if src.duplicated([join_firm, join_year]).any():
        dup = int(src.duplicated([join_firm, join_year], keep=False).sum())
        raise ValueError(f"Loop B2 raw C_obs action source has duplicate normalized firm-year rows: {dup}")
    rename = {c: f"__loopb2_cobs_src__{c}" for c in value_cols + flag_cols if c in out.columns}
    src_small = src[keep].rename(columns={c: rename.get(c, c) for c in value_cols + flag_cols})
    merged = out.merge(src_small, on=[join_firm, join_year], how="left", validate="many_to_one")
    missing_rows_any = pd.Series(False, index=merged.index)
    overwritten_rows: dict[str, int] = {}
    value_conflict_rows: dict[str, int] = {}
    flag_false_rows: dict[str, int] = {}
    for val_col, flag_col in zip(value_cols, flag_cols):
        src_val_col = rename.get(val_col, val_col)
        src_flag_col = rename.get(flag_col, flag_col)
        if src_val_col not in merged.columns or src_flag_col not in merged.columns:
            raise ValueError(f"Loop B2 C_obs source merge internal error for {val_col}/{flag_col}")
        src_vals = pd.to_numeric(merged[src_val_col], errors="coerce")
        src_flags = _bool_mask(merged, src_flag_col, default=False)
        missing = src_vals.isna()
        missing_rows_any = missing_rows_any | missing
        # Existing action__* values in candidate/projection frames are not trusted;
        # count differences before replacing them with direct raw C_obs values.
        if val_col in merged.columns:
            existing = pd.to_numeric(merged[val_col], errors="coerce")
            both = existing.notna() & src_vals.notna()
            conflict = both & ((existing - src_vals).abs() > 1e-12)
            value_conflict_rows[val_col] = int(conflict.sum())
            overwritten_rows[val_col] = int(src_vals.notna().sum())
        else:
            overwritten_rows[val_col] = int(src_vals.notna().sum())
        merged[val_col] = src_vals
        merged[flag_col] = src_flags
        flag_false_rows[flag_col] = int((~src_flags).sum())
        for c in [src_val_col, src_flag_col]:
            if c in merged.columns and c not in {val_col, flag_col}:
                merged = merged.drop(columns=[c])
    matched = int((~missing_rows_any).sum())
    rows = int(len(merged))
    if matched != rows:
        sample_cols = [c for c in ["firm_id", "fiscal_year"] + value_cols if c in merged.columns]
        sample = merged.loc[missing_rows_any, sample_cols].head(10).to_dict("records")
        meta = {
            "status": "FAIL",
            "source_contract": source_meta,
            "rows": rows,
            "matched_rows_all_action_values": matched,
            "missing_rows": int(missing_rows_any.sum()),
            "missing_sample": sample,
            "value_columns": value_cols,
            "observed_flag_columns": flag_cols,
            "source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        }
        raise ValueError("Loop B2 C_obs action value merge did not cover every Test2/Test3 row: " + json.dumps(_json_safe(meta), ensure_ascii=False, sort_keys=True))
    meta = {
        "status": "PASS",
        "source": source_meta.get("source"),
        "source_role": "stage2_raw_action_source_panel_realized_C_obs_values",
        "source_contract": source_meta,
        "rows": rows,
        "matched_rows_all_action_values": matched,
        "value_columns": value_cols,
        "observed_flag_columns": flag_cols,
        "value_columns_overwritten_from_source_rows": overwritten_rows,
        "preexisting_value_conflict_rows_before_override": value_conflict_rows,
        "observed_flag_false_rows": flag_false_rows,
        "source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        "candidate_projection_action_guard": "Any pre-existing action__* columns in phase/candidate handoffs are overwritten by the direct raw C_obs source before simulator validation.",
    }
    merged = merged.drop(columns=["__loopb2_cobs_firm_key", "__loopb2_cobs_year"], errors="ignore")
    return merged, meta


def _validate_loopb2_observed_action_source(df: pd.DataFrame, space, *, label: str) -> dict[str, Any]:
    """Fail-fast contract for true simulator validation action source.

    True Test2/Test3 use realized action magnitudes from action__* columns and
    use action_observed__* only as source-observation masks.  Treating masks as
    magnitudes was the root cause of the inflated total-assets fidelity error.
    """
    rows = int(len(df))
    if rows <= 0:
        raise ValueError(f"Loop B2 observed-action source validation received empty frame for {label}")
    required_values = [_observed_action_value_column_for_space_column(c) for c in space.columns]
    required_flags = [_observed_action_flag_column_for_space_column(c) for c in space.columns]
    missing_values = [c for c in required_values if c not in df.columns]
    missing_flags = [c for c in required_flags if c not in df.columns]
    if missing_values or missing_flags:
        raise ValueError(
            "Loop B2 true simulator validation requires realized C_obs action__* value columns "
            "and action_observed__* observation masks; "
            f"missing_values={missing_values}, missing_flags={missing_flags} for {label}"
        )

    coverage: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    for space_col, val_col, flag_col in zip(space.columns, required_values, required_flags):
        dim = _canonical_action_dimension_name(space_col)
        vals = pd.to_numeric(df[val_col], errors="coerce")
        vals_float = vals.astype(float)
        finite = vals.notna() & np.isfinite(vals_float)
        flags = _bool_mask(df, flag_col, default=False)
        finite_rows = int(finite.sum())
        finite_share = float(finite_rows / rows) if rows else 0.0
        observed_rows = int(flags.sum())
        detail: dict[str, Any] = {
            "space_column": str(space_col),
            "value_column": val_col,
            "observed_flag_column": flag_col,
            "dimension": dim,
            "finite_rows": finite_rows,
            "finite_share": finite_share,
            "observed_flag_true_rows": observed_rows,
            "observed_flag_true_share": float(observed_rows / rows) if rows else 0.0,
            "nonnull_rows": int(pd.notna(df[val_col]).sum()),
            "nonzero_rows": int((vals.fillna(0.0).abs() > 1e-12).sum()),
            "nonzero_share": float((vals.fillna(0.0).abs() > 1e-12).mean()) if rows else 0.0,
            "min": float(vals_float[finite].min()) if finite_rows else None,
            "max": float(vals_float[finite].max()) if finite_rows else None,
        }
        if finite_rows != rows:
            bad_sample = df.loc[~finite, [c for c in ["firm_id", "fiscal_year", val_col] if c in df.columns]].head(10).to_dict("records")
            failures.append({
                "dimension": dim,
                "reason": "nonfinite_action_value",
                "value_column": val_col,
                "finite_rows": finite_rows,
                "rows": rows,
                "finite_share": finite_share,
                "bad_sample": bad_sample,
            })
        if observed_rows <= 0:
            failures.append({
                "dimension": dim,
                "reason": "dead_observed_action_mask",
                "observed_flag_column": flag_col,
                "observed_flag_true_rows": observed_rows,
                "rows": rows,
            })
        coverage[dim] = detail

    meta = {
        "status": "PASS" if not failures else "FAIL",
        "label": label,
        "source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
        "source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        "value_prefix": B2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX,
        "observed_flag_prefix": B2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX,
        "required_value_columns": required_values,
        "required_observed_flag_columns": required_flags,
        "rows": rows,
        "coverage": coverage,
        "failures": failures,
        "hard_fail_rule": "C_obs action__* magnitudes must be finite for every Test2/Test3 row; action_observed__* masks must exist and have nonzero dimension coverage.",
    }
    if failures:
        raise ValueError("Loop B2 true simulator observed-action source contract failed: " + json.dumps(_json_safe(meta), ensure_ascii=False, sort_keys=True))
    return meta


def _observed_action(row: pd.Series, space) -> Action:
    values: dict[str, float] = {}
    for c in space.columns:
        dim = _canonical_action_dimension_name(c)
        val_col = _observed_action_value_column_for_space_column(c)
        if val_col not in row.index:
            raise KeyError(f"Missing required realized C_obs action value column for Loop B2 true simulator validation: {val_col}")
        val = _finite_float(row.get(val_col))
        if not np.isfinite(val):
            raise ValueError(f"Non-finite realized C_obs action value for {val_col}: {row.get(val_col)!r}")
        values[dim] = val
    return clip_action(Action(**values, source="historical"))

def _select_business_plan(mode: str, history: list[FirmState], *, rating_grade=None) -> BusinessPlan:
    if mode == "default":
        return BusinessPlan()
    if mode == "calibrated":
        return calibrate_business_plan(history, grade=rating_grade) if history else BusinessPlan()
    raise ValueError(f"Unsupported sim_business_plan_mode={mode}")


def _first_present_value(row_dict: dict[Any, Any], candidates: tuple[str, ...]) -> tuple[Any, str | None]:
    for key in candidates:
        if key not in row_dict:
            continue
        val = row_dict.get(key)
        if val is None:
            continue
        try:
            if pd.isna(val):
                continue
        except Exception:
            pass
        return val, key
    return None, None


def _actual_next_values(row: pd.Series) -> dict[str, Any]:
    """Resolve observed t+1 values from Stage2 handoff columns.

    Strict LoopA fidelity must use observed next raw values as the actual side.
    Do not let next__sim__* satisfy the actual side: that would compare the
    simulator against its own simulated transition and would mask exploitation.
    """
    d = row.to_dict()
    out: dict[str, Any] = {}

    # Explicit raw-next aliases first.  These are the canonical actual side for
    # observed t+1 comparisons in phase3_iql_candidate__P50 artifacts.
    for field in _STATE_VALUE_FIELDS:
        aliases = LOOPA_ACTUAL_NEXT_ALIASES.get(field, (f"next__raw__{field}", f"next__{field}", f"{field}__next"))
        val, _ = _first_present_value(d, aliases)
        if val is not None:
            out[field] = val

    # Registry/U-code fallback can recover legacy raw next U-code columns, but
    # keep the explicit raw aliases above dominant.
    reg_out = dict(resolved_field_values(d, next_state=True))
    for field, val in reg_out.items():
        if field not in out and field in _STATE_VALUE_FIELDS:
            out[field] = val
    return out


def _predicted_next_values(pred_row: dict[str, Any]) -> dict[str, Any]:
    """Resolve simulator-predicted t+1 values to canonical dimensions."""
    out: dict[str, Any] = {}
    for field in _STATE_VALUE_FIELDS:
        aliases = LOOPA_PREDICTED_ALIASES.get(field, (field, f"next__sim__{field}", f"sim__{field}"))
        val, _ = _first_present_value(pred_row, aliases)
        if val is not None:
            out[field] = val
    return out


def _component_group(dimension: str) -> str:
    if dimension in MERTON_FIDELITY_FIELDS:
        return "merton"
    if dimension in FCFF_FIDELITY_FIELDS:
        return "fcff"
    if dimension in IDENTITY_FIELDS:
        return "accounting_identity"
    return "state"


def _numeric_state_dict(fs: FirmState) -> dict[str, float]:
    d = fs.to_dict()
    return {k: _finite_float(v) for k, v in d.items() if k in _STATE_VALUE_FIELDS}


def _compute_error_rows(phase: pd.DataFrame, sim_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (_, r), pred_row_raw in zip(phase.iterrows(), sim_rows):
        actual_next = _actual_next_values(r)
        pred_row = _predicted_next_values(pred_row_raw)
        current_assets = _finite_float(r.get("sim__total_assets", r.get("raw__total_assets", r.get("total_assets", np.nan))))
        actual_assets = _finite_float(actual_next.get("total_assets", np.nan))
        predicted_assets = _finite_float(pred_row.get("total_assets", np.nan))
        asset_scale = np.nanmax(np.abs([current_assets, actual_assets, predicted_assets]))
        if not np.isfinite(asset_scale) or asset_scale <= 1e-9:
            asset_scale = np.nan
        for field in _STATE_VALUE_FIELDS:
            if field not in pred_row or field not in actual_next:
                continue
            pred = _finite_float(pred_row.get(field))
            actual = _finite_float(actual_next.get(field))
            if not (np.isfinite(pred) and np.isfinite(actual)):
                continue
            signed = pred - actual
            abs_err = abs(signed)
            rows.append(
                {
                    "firm_id": str(r.get("firm_id", r.get("corp_code", "UNKNOWN"))),
                    "fiscal_year": int(float(r.get("fiscal_year", r.get("year", 0)) or 0)),
                    "dimension": field,
                    "component_group": _component_group(field),
                    "predicted": pred,
                    "actual": actual,
                    "signed_error": signed,
                    "abs_error": abs_err,
                    "asset_scale": asset_scale,
                    "abs_err_over_assets": abs_err / asset_scale if np.isfinite(asset_scale) and asset_scale > 0 else np.nan,
                    "signed_err_over_assets": signed / asset_scale if np.isfinite(asset_scale) and asset_scale > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _spearman(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    m = x.notna() & y.notna()
    if int(m.sum()) < 3:
        return float("nan")
    if float(x[m].nunique()) < 2 or float(y[m].nunique()) < 2:
        return float("nan")
    return float(x[m].rank().corr(y[m].rank()))


def _summarize_errors(errs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dimension",
        "component_group",
        "count",
        "coverage_share",
        "mean_signed_error",
        "median_signed_error",
        "std_signed_error",
        "mean_abs_error",
        "median_abs_error",
        "p95_abs_error",
        "mean_abs_err_over_assets",
        "median_abs_err_over_assets",
        "p95_abs_err_over_assets",
        "spearman",
        "predicted_nonzero_rate",
        "actual_nonzero_rate",
    ]
    if errs.empty:
        return pd.DataFrame(columns=cols)

    total_rows = max(int(errs[["firm_id", "fiscal_year"]].drop_duplicates().shape[0]), 1)
    records = []
    for dim, g in errs.groupby("dimension", dropna=False):
        pred = pd.to_numeric(g["predicted"], errors="coerce")
        actual = pd.to_numeric(g["actual"], errors="coerce")
        signed = pd.to_numeric(g["signed_error"], errors="coerce")
        abs_err = pd.to_numeric(g["abs_error"], errors="coerce")
        rel = pd.to_numeric(g["abs_err_over_assets"], errors="coerce")
        records.append(
            {
                "dimension": str(dim),
                "component_group": _component_group(str(dim)),
                "count": int(len(g)),
                "coverage_share": float(len(g) / total_rows),
                "mean_signed_error": float(signed.mean()) if signed.notna().any() else np.nan,
                "median_signed_error": float(signed.median()) if signed.notna().any() else np.nan,
                "std_signed_error": float(signed.std()) if signed.notna().sum() > 1 else np.nan,
                "mean_abs_error": float(abs_err.mean()) if abs_err.notna().any() else np.nan,
                "median_abs_error": float(abs_err.median()) if abs_err.notna().any() else np.nan,
                "p95_abs_error": float(abs_err.quantile(0.95)) if abs_err.notna().any() else np.nan,
                "mean_abs_err_over_assets": float(rel.mean()) if rel.notna().any() else np.nan,
                "median_abs_err_over_assets": float(rel.median()) if rel.notna().any() else np.nan,
                "p95_abs_err_over_assets": float(rel.quantile(0.95)) if rel.notna().any() else np.nan,
                "spearman": _spearman(pred, actual),
                "predicted_nonzero_rate": float((pred.fillna(0.0).abs() > 1e-12).mean()),
                "actual_nonzero_rate": float((actual.fillna(0.0).abs() > 1e-12).mean()),
            }
        )
    return pd.DataFrame(records, columns=cols)


def _group_gate(summary: pd.DataFrame, *, max_rel_err_assets: float, min_coverage_share: float) -> dict:
    required = {
        "merton": sorted(MERTON_FIDELITY_FIELDS),
    }
    out = {"max_rel_err_assets": float(max_rel_err_assets), "min_coverage_share": float(min_coverage_share), "groups": {}, "violations": []}
    for group, dims in required.items():
        sub = summary[summary["component_group"].astype(str).eq(group)].copy() if not summary.empty else pd.DataFrame()
        present = sorted(set(sub["dimension"].astype(str))) if not sub.empty else []
        missing = sorted(set(dims) - set(present))
        rel = pd.to_numeric(sub.get("median_abs_err_over_assets", pd.Series(dtype=float)), errors="coerce")
        cov = pd.to_numeric(sub.get("coverage_share", pd.Series(dtype=float)), errors="coerce")
        worst_rel = float(rel.max()) if rel.notna().any() else float("nan")
        min_cov = float(cov.min()) if cov.notna().any() else 0.0
        rec = {"required_dimensions": dims, "present_dimensions": present, "missing_dimensions": missing, "worst_median_abs_err_over_assets": worst_rel, "min_coverage_share": min_cov}
        out["groups"][group] = rec
        if missing:
            out["violations"].append({"component_group": group, "reason": "missing_required_dimensions", "missing": missing})
        if (not np.isfinite(worst_rel)) or worst_rel > float(max_rel_err_assets):
            out["violations"].append({"component_group": group, "reason": "relative_error_threshold", "worst_median_abs_err_over_assets": worst_rel, "threshold": float(max_rel_err_assets)})
        if min_cov < float(min_coverage_share):
            out["violations"].append({"component_group": group, "reason": "coverage_below_threshold", "min_coverage_share": min_cov, "threshold": float(min_coverage_share)})
    out["status"] = "PASS" if not out["violations"] else "FAIL"
    return out


def _write_compatibility_outputs(primary_out: Path, final: Path, files: list[str]) -> list[str]:
    targets = [
        final / "stage2_substrate_validation",
        final / "stage2_candidate_projection" / "verification",
        final / "verification",
    ]
    written = []
    for d in targets:
        if d == primary_out:
            continue
        d.mkdir(parents=True, exist_ok=True)
        for name in files:
            src = primary_out / name
            if src.exists():
                dst = d / name
                shutil.copy2(src, dst)
                written.append(str(dst))
    return written


def _normalise_loopa_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "firm_id" not in out.columns and "corp_code" in out.columns:
        out["firm_id"] = out["corp_code"]
    if "fiscal_year" not in out.columns and "year" in out.columns:
        out["fiscal_year"] = out["year"]
    if "firm_id" not in out.columns or "fiscal_year" not in out.columns:
        raise ValueError("LoopA fidelity requires firm_id and fiscal_year/year columns for observed-next alignment")
    out["firm_id"] = out["firm_id"].astype(str)
    out["fiscal_year"] = _loopb2_parse_fiscal_year_series(out["fiscal_year"], index=out.index)
    return out


def _has_merton_actual_next_columns(df: pd.DataFrame) -> bool:
    cols = set(map(str, df.columns))
    return all(any(alias in cols for alias in aliases) for aliases in LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS.values())


def _load_observed_next_handoff(final: Path) -> tuple[pd.DataFrame, str]:
    """Load a Stage2 handoff that carries observed next raw columns.

    phase_eval_candidate.parquet is the driver for LoopA, but some runs keep
    observed next raw columns only in phase3_iql_candidate__P50.parquet.  Use
    observed artifacts only for the actual side; never use next__sim__ fallback
    as an observed target in strict fidelity.
    """
    candidates = [
        final / "stage2_candidate_projection" / "phase_eval_candidate.parquet",
        final / "stage2_candidate_projection" / "phase3_iql_candidate__P50.parquet",
        final / "stage2_candidate_projection" / "phase3_iql_counterfactual_candidate__P50.parquet",
    ]
    available: list[tuple[Path, pd.DataFrame]] = []
    for path in candidates:
        if not path.exists():
            continue
        df = _normalise_loopa_keys(read_parquet_required(path))
        available.append((path, df))
        if _has_merton_actual_next_columns(df):
            return df, str(path)

    details = {str(path): list(map(str, df.columns[:80])) for path, df in available}
    raise ValueError(
        "No LoopA observed-next handoff with required raw next Merton columns. "
        "Expected aliases include next__raw__total_assets, next__raw__short_term_debt, "
        "next__raw__long_term_debt, next__raw__bonds. Available leading columns: "
        + json.dumps(details, ensure_ascii=False, default=str)[:4000]
    )


def _merge_observed_next_into_phase(phase: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    phase_n = _normalise_loopa_keys(phase)
    obs_n = _normalise_loopa_keys(observed)
    actual_cols: list[str] = []
    for aliases in LOOPA_ACTUAL_NEXT_ALIASES.values():
        for alias in aliases:
            if alias in obs_n.columns and alias not in actual_cols:
                actual_cols.append(alias)
    observed_action_cols = [c for c in obs_n.columns if str(c).startswith(B2_TRUE_SIMULATOR_ACTION_COLUMN_PREFIX)]
    if not actual_cols:
        raise ValueError("Observed next handoff has no recognized LoopA actual-next columns after alias resolution")
    keep = ["firm_id", "fiscal_year"] + actual_cols + observed_action_cols
    keep = list(dict.fromkeys(keep))
    obs_small = obs_n[keep].drop_duplicates(subset=["firm_id", "fiscal_year"], keep="first")

    # Preserve phase columns; add missing actual-next and realized C_obs action
    # columns from the observed handoff.  The action columns are required by the
    # true simulator validation path and must not be silently replaced by
    # generic action__* candidate columns.  If phase already has an
    # action_observed__* column but carries missing values, fill only those
    # missing cells from the observed handoff and hard-fail on conflicting
    # non-missing values.
    missing_actual = [c for c in actual_cols if c not in phase_n.columns]
    observed_actions_needing_source = [
        c for c in observed_action_cols
        if c not in phase_n.columns or bool(pd.isna(phase_n[c]).any())
    ]
    merge_missing = list(dict.fromkeys(missing_actual + observed_actions_needing_source))
    if not merge_missing and _has_merton_actual_next_columns(phase_n):
        phase_n.attrs["loopA_actual_next_source"] = "phase_eval_candidate.parquet"
        phase_n.attrs["loopB2_observed_action_source"] = "phase_eval_candidate.parquet"
        return phase_n

    obs_action_rename = {
        c: f"__loopb2_observed_action_src__{i}"
        for i, c in enumerate(observed_actions_needing_source)
        if c in phase_n.columns
    }
    obs_for_merge = obs_small[["firm_id", "fiscal_year"] + merge_missing].rename(columns=obs_action_rename)
    merged = phase_n.merge(obs_for_merge, on=["firm_id", "fiscal_year"], how="left", validate="many_to_one")
    for action_col in observed_actions_needing_source:
        src_col = obs_action_rename.get(action_col, action_col)
        if src_col not in merged.columns:
            continue
        if action_col not in merged.columns:
            merged[action_col] = merged[src_col]
        else:
            left = pd.to_numeric(merged[action_col], errors="coerce")
            right = pd.to_numeric(merged[src_col], errors="coerce")
            both = left.notna() & right.notna()
            conflict = both & ((left - right).abs() > 1e-12)
            if bool(conflict.any()):
                sample = merged.loc[conflict, [c for c in ["firm_id", "fiscal_year", action_col, src_col] if c in merged.columns]].head(10).to_dict("records")
                raise ValueError(
                    "Loop B2 observed-action merge conflict: phase and observed handoff disagree for "
                    f"{action_col}; sample={sample}"
                )
            merged[action_col] = merged[action_col].where(merged[action_col].notna(), merged[src_col])
        if src_col != action_col:
            merged = merged.drop(columns=[src_col])
    merged.attrs["loopA_actual_next_source"] = "observed_stage2_handoff_merge"
    merged.attrs["loopB2_observed_action_source"] = "observed_stage2_handoff_merge"
    matched = int(merged[missing_actual].notna().any(axis=1).sum()) if missing_actual else int(len(merged))
    if matched <= 0:
        raise ValueError("LoopA observed-next merge produced zero rows with actual next values; check firm_id/fiscal_year alignment")
    return merged


def _actual_next_match_count(df: pd.DataFrame) -> int:
    """Count rows that carry at least one required observed raw-next Merton value."""
    if df.empty:
        return 0
    cols = [alias for aliases in LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS.values() for alias in aliases if alias in df.columns]
    if not cols:
        return 0
    return int(df[cols].notna().any(axis=1).sum())


def _build_loopa_phase(final: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the LoopA driver with observed raw next-state targets.

    `phase_eval_candidate.parquet` is the preferred driver when it can be
    aligned with observed next raw columns.  Some final-freeze runs use OOT
    evaluation rows whose `firm_id + fiscal_year` keys do not overlap the
    observed transition handoff.  In that case, fidelity must fall back to the
    observed Stage2 handoff itself instead of raising before any error can be
    measured.  The fallback still compares simulator predictions against
    `next__raw__*` observed targets; it never uses `next__sim__*` as actuals.
    """
    phase_path = final / "stage2_candidate_projection" / "phase_eval_candidate.parquet"
    phase_raw = read_parquet_required(phase_path)
    if phase_raw.empty:
        raise ValueError("LoopA/B2 verifier received empty phase_eval_candidate.parquet")

    observed_next, observed_next_source = _load_observed_next_handoff(final)
    meta: dict[str, Any] = {
        "loopA_phase_eval_source": str(phase_path),
        "loopA_observed_next_source": observed_next_source,
        "loopA_driver_source": str(phase_path),
        "loopA_phase_eval_alignment_status": "NOT_CHECKED",
        "loopA_driver_selection_reason": None,
        "loopA_driver_fallback_reason": None,  # legacy metadata alias; no longer a silent fallback
    }

    try:
        phase_candidate = _merge_observed_next_into_phase(phase_raw, observed_next)
        match_rows = _actual_next_match_count(phase_candidate)
        if match_rows > 0:
            meta["loopA_driver_source"] = str(phase_path)
            meta["loopA_observed_next_match_rows"] = int(match_rows)
            meta["loopA_actual_next_source"] = phase_candidate.attrs.get("loopA_actual_next_source", observed_next_source)
            meta["loopA_phase_eval_alignment_status"] = "PASS"
            meta["loopA_driver_selection_reason"] = "phase_eval_candidate aligned with observed next raw targets"
            return phase_candidate, meta
        meta["loopA_phase_eval_alignment_status"] = "ZERO_ACTUAL_NEXT_ROWS"
        meta["loopA_driver_selection_reason"] = "phase_eval_candidate did not carry observed t+1 targets; selecting observed transition handoff for historical Test2/Test3"
        meta["loopA_driver_fallback_reason"] = "phase_eval_merge_produced_zero_actual_next_rows"
    except Exception as exc:
        meta["loopA_phase_eval_alignment_status"] = "FAILED"
        meta["loopA_driver_selection_reason"] = "phase_eval_candidate is not usable as a historical observed-transition driver; selecting observed transition handoff for Test2/Test3"
        meta["loopA_driver_fallback_reason"] = f"phase_eval_merge_failed: {type(exc).__name__}: {exc}"

    # Use the observed Stage2 transition handoff as the canonical historical
    # Test2/Test3 driver.  This is the table guaranteed to carry observed next
    # raw financials.  The legacy `fallback_reason` metadata remains only for
    # traceability.
    fallback = _normalise_loopa_keys(observed_next)
    match_rows = _actual_next_match_count(fallback)
    if match_rows <= 0:
        raise ValueError(
            "LoopA observed handoff fallback has zero rows with actual next raw values; "
            "check phase3_iql_candidate__P50 observed next columns"
        )
    meta["loopA_driver_source"] = observed_next_source
    meta["loopA_actual_next_source"] = observed_next_source
    meta["loopA_observed_next_match_rows"] = int(match_rows)
    meta["loopA_phase_eval_rows"] = int(len(phase_raw))
    meta["loopA_driver_fallback_used"] = False
    meta["loopA_driver_source_role"] = "observed_transition_handoff_for_historical_test2_test3"
    return fallback, meta



SIM_NEXT_ALIASES: dict[str, tuple[str, ...]] = {
    field: (f"next__sim__{field}", f"next__{field}")
    for field in _STATE_VALUE_FIELDS
}


def _bool_mask(df: pd.DataFrame, col: str, *, default: bool) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(default).astype(bool)
    txt = s.astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "t", "yes", "y"})


def _load_counterfactual_handoff(final: Path, magnitude_quantile: int) -> tuple[pd.DataFrame | None, str | None]:
    candidates = [
        final / "stage2_candidate_projection" / f"phase3_iql_counterfactual_candidate__P{int(magnitude_quantile)}.parquet",
        final / "stage2_candidate_projection" / "phase3_iql_counterfactual_candidate.parquet",
    ]
    for path in candidates:
        if path.exists():
            return _normalise_loopa_keys(read_parquet_required(path)), str(path)
    return None, None


def _sim_next_values(row: pd.Series) -> dict[str, Any]:
    d = row.to_dict()
    out: dict[str, Any] = {}
    for field in _STATE_VALUE_FIELDS:
        val, _ = _first_present_value(d, SIM_NEXT_ALIASES.get(field, (f"next__sim__{field}", f"next__{field}")))
        if val is not None:
            out[field] = val
    return out


def _registry_next_values(row: pd.Series) -> dict[str, Any]:
    """Resolve next-state U-code/header fields from a Stage2 mixed-transition row.

    `_sim_next_values` only handles canonical next__sim__ aliases.  P-handoff
    artifacts can instead carry used t+1 accounting values under raw U-code
    headers such as `next__balance_sheet__[U01...]` and
    `next__income_statement__[U01...]`.  Field-wise registry resolution prevents
    a partial next__sim__ row from dropping formula-critical fields such as
    revenue and non_current_assets.
    """
    out: dict[str, Any] = {}
    for field, value in resolved_field_values(row.to_dict(), next_state=True).items():
        if field not in _STATE_VALUE_FIELDS:
            continue
        if value is None:
            continue
        try:
            if pd.isna(value):
                continue
        except Exception:
            pass
        out[field] = value
    return out


def _used_next_values(row: pd.Series) -> dict[str, Any]:
    """Resolve the next state actually used by Stage2's mixed transition contract.

    Observed matched cells intentionally use the empirical observed next state.
    Non-observed counterfactual cells use simulator-generated next__sim__ fields.
    """
    is_obs = bool(row.get("is_observed_transition", False))
    registry_vals = _registry_next_values(row)
    if is_obs:
        vals = _actual_next_values(row)
        # `_actual_next_values` prefers explicit raw-next aliases and then uses the
        # registry.  Keep a field-wise registry fill here so partial explicit raw
        # aliases do not discard next-state U-code values required by R157/R182.
        for field, value in registry_vals.items():
            vals.setdefault(field, value)
        if vals:
            return vals
    vals = _sim_next_values(row)
    # Some P-handoff rows have a partial canonical next__sim__ state plus the rest
    # of the used t+1 state under next__<statement>__[U-code] columns.  Do a
    # field-wise fill rather than the previous all-or-nothing fallback.
    for field, value in registry_vals.items():
        vals.setdefault(field, value)
    if vals:
        return vals
    return _actual_next_values(row)


def _build_used_next_state_frame(stage2_rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in stage2_rows.reset_index(drop=True).iterrows():
        vals = _used_next_values(row)
        rec = {
            "firm_id": row.get("firm_id", row.get("corp_code", "UNKNOWN")),
            "fiscal_year": row.get("fiscal_year", row.get("year", 0)),
        }
        for field in _STATE_VALUE_FIELDS:
            if field in vals:
                rec[field] = vals[field]
        records.append(rec)
    return pd.DataFrame(records)


def _observed_transition_preservation_gate(cf: pd.DataFrame, *, atol: float = 1e-6) -> dict[str, Any]:
    """Hard gate for the empirical half of the Stage2 mixed-transition contract.

    For the matched observed action cell, Stage2 is supposed to preserve the real
    t+1 transition.  Therefore the correct check is preservation of next raw
    values, not re-simulation fidelity against real next values.
    """
    is_obs = _bool_mask(cf, "is_observed_transition", default=False) | (~_bool_mask(cf, "counterfactual_transition", default=True))
    obs = cf.loc[is_obs].copy()
    checks: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    if obs.empty:
        return {"status": "FAIL", "rows": 0, "violations": [{"reason": "no_observed_transition_rows"}], "checks": []}

    for field, raw_aliases in LOOPA_REQUIRED_ACTUAL_NEXT_COLUMNS.items():
        sim_aliases = SIM_NEXT_ALIASES.get(field, (f"next__sim__{field}", f"next__{field}"))
        raw_col = next((c for c in raw_aliases if c in obs.columns), None)
        sim_col = next((c for c in sim_aliases if c in obs.columns), None)
        rec: dict[str, Any] = {"dimension": field, "raw_column": raw_col, "used_column": sim_col}
        if raw_col is None:
            rec.update({"status": "FAIL", "reason": "missing_raw_next_column"})
            violations.append({"dimension": field, "reason": "missing_raw_next_column"})
            checks.append(rec)
            continue
        # If no next__sim__ alias exists on observed rows, using raw next directly is still contract-valid.
        if sim_col is None:
            rec.update({"status": "PASS_RAW_ONLY", "coverage_share": float(obs[raw_col].notna().mean())})
            checks.append(rec)
            continue
        a = pd.to_numeric(obs[raw_col], errors="coerce")
        b = pd.to_numeric(obs[sim_col], errors="coerce")
        comparable = a.notna() & b.notna()
        diff = (a - b).abs()
        max_abs = float(diff[comparable].max()) if comparable.any() else float("nan")
        equal_share = float((diff[comparable] <= float(atol)).mean()) if comparable.any() else 0.0
        rec.update({
            "status": "PASS" if comparable.any() and max_abs <= float(atol) else "FAIL",
            "comparable_rows": int(comparable.sum()),
            "coverage_share": float(comparable.mean()),
            "max_abs_diff": max_abs,
            "equal_share_atol": equal_share,
            "atol": float(atol),
        })
        if rec["status"] != "PASS":
            violations.append({"dimension": field, "reason": "observed_next_not_preserved", "max_abs_diff": max_abs, "atol": float(atol)})
        checks.append(rec)
    return {"status": "PASS" if not violations else "FAIL", "rows": int(len(obs)), "violations": violations, "checks": checks}


def _counterfactual_simulator_sanity_gate(
    cf: pd.DataFrame,
    *,
    max_identity_rel_err_assets: float = 0.02,
    min_coverage_share: float = 0.80,
) -> dict[str, Any]:
    """Hard gate for the simulated half of the Stage2 mixed-transition contract.

    Non-observed counterfactual rows cannot be validated against a real next state.
    The appropriate gate is therefore internal sanity: required simulated state
    coverage, accounting identity, non-negativity, and finite rewards.
    """
    is_sim = _bool_mask(cf, "counterfactual_transition", default=False) | (~_bool_mask(cf, "is_observed_transition", default=True))
    sim = cf.loc[is_sim].copy()
    violations: list[dict[str, Any]] = []
    if sim.empty:
        return {"status": "FAIL", "rows": 0, "violations": [{"reason": "no_simulated_counterfactual_rows"}], "checks": {}}

    checks: dict[str, Any] = {"required_next_sim_columns": {}, "nonnegative": {}, "finite_rewards": {}}
    # Coverage of Merton-critical simulated next state columns.
    for field in sorted(MERTON_FIDELITY_FIELDS):
        aliases = (f"next__sim__{field}", f"next__{field}")
        col = next((c for c in aliases if c in sim.columns), None)
        cov = float(pd.to_numeric(sim[col], errors="coerce").notna().mean()) if col else 0.0
        checks["required_next_sim_columns"][field] = {"column": col, "coverage_share": cov}
        if col is None or cov < float(min_coverage_share):
            violations.append({"dimension": field, "reason": "missing_or_low_coverage_next_sim", "column": col, "coverage_share": cov, "threshold": float(min_coverage_share)})

    def num(col: str) -> pd.Series:
        return pd.to_numeric(sim[col], errors="coerce") if col in sim.columns else pd.Series(np.nan, index=sim.index)

    ta = num("next__sim__total_assets")
    tl = num("next__sim__total_liabilities")
    te = num("next__sim__total_equity")
    identity_comparable = ta.notna() & tl.notna() & te.notna() & (ta.abs() > 1e-9)
    identity_rel = ((ta - (tl + te)).abs() / ta.abs()).where(identity_comparable)
    max_identity = float(identity_rel.max()) if identity_rel.notna().any() else float("nan")
    p95_identity = float(identity_rel.quantile(0.95)) if identity_rel.notna().any() else float("nan")
    checks["accounting_identity"] = {
        "formula": "next__sim__total_assets == next__sim__total_liabilities + next__sim__total_equity",
        "comparable_rows": int(identity_comparable.sum()),
        "coverage_share": float(identity_comparable.mean()),
        "max_rel_err_assets": max_identity,
        "p95_rel_err_assets": p95_identity,
        "threshold": float(max_identity_rel_err_assets),
    }
    if int(identity_comparable.sum()) == 0:
        violations.append({"reason": "no_accounting_identity_comparable_rows"})
    elif max_identity > float(max_identity_rel_err_assets):
        violations.append({"reason": "accounting_identity_threshold", "max_rel_err_assets": max_identity, "threshold": float(max_identity_rel_err_assets)})

    if "next__sim__current_assets" in sim.columns and "next__sim__non_current_assets" in sim.columns:
        ca = num("next__sim__current_assets")
        nca = num("next__sim__non_current_assets")
        comp = ta.notna() & ca.notna() & nca.notna() & (ta.abs() > 1e-9)
        rel = ((ta - (ca + nca)).abs() / ta.abs()).where(comp)
        checks["asset_subtotal_identity"] = {
            "formula": "next__sim__total_assets == next__sim__current_assets + next__sim__non_current_assets",
            "comparable_rows": int(comp.sum()),
            "coverage_share": float(comp.mean()),
            "max_rel_err_assets": float(rel.max()) if rel.notna().any() else float("nan"),
            "p95_rel_err_assets": float(rel.quantile(0.95)) if rel.notna().any() else float("nan"),
        }

    nonneg_fields = [
        "total_assets", "current_assets", "non_current_assets", "cash", "short_term_investments",
        "receivables", "inventory", "ppe", "total_liabilities", "current_liabilities",
        "non_current_liabilities", "short_term_debt", "current_portion_long_debt", "long_term_debt",
        "bonds", "payables",
    ]
    for field in nonneg_fields:
        col = f"next__sim__{field}"
        if col not in sim.columns:
            continue
        x = pd.to_numeric(sim[col], errors="coerce")
        finite = x.notna()
        neg = finite & (x < -1e-6)
        checks["nonnegative"][field] = {"column": col, "finite_rows": int(finite.sum()), "negative_rows": int(neg.sum()), "negative_share": float(neg.mean())}
        if int(neg.sum()) > 0:
            violations.append({"dimension": field, "reason": "negative_simulated_balance_sheet_value", "negative_rows": int(neg.sum()), "negative_share": float(neg.mean())})

    for col in ["reward_train", "reward_total_raw", "phi_t", "phi_tplusH", "delta_phi", "reward_aux_merton", "reward_aux_liquidity", "reward_aux_fcff"]:
        if col not in sim.columns:
            continue
        x = pd.to_numeric(sim[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        bad = int(x.isna().sum())
        checks["finite_rewards"][col] = {"bad_rows": bad, "bad_share": float(x.isna().mean())}
        if bad > 0 and col in {"reward_train", "reward_total_raw", "phi_t", "phi_tplusH", "delta_phi"}:
            violations.append({"column": col, "reason": "nonfinite_required_reward_column", "bad_rows": bad})

    # Diagnostic, not a hard gate: action semantics can be ambiguous after mixed actions.
    if "candidate_id" in sim.columns:
        checks["candidate_id_top20"] = {str(k): int(v) for k, v in sim["candidate_id"].astype(str).value_counts().head(20).to_dict().items()}

    return {
        "status": "PASS" if not violations else "FAIL",
        "rows": int(len(sim)),
        "violations": violations,
        "checks": checks,
        "gate_type": "counterfactual_internal_sanity_no_actual_next_comparison",
    }



def _require_configured_loopb2_fidelity_threshold(max_rel_err_assets: float) -> float:
    """Fail fast when an active runner accidentally reintroduces the stale 5% fidelity threshold.

    The current configured Stage2 diagnostic contract uses 10% median absolute error
    over assets.  Older local command snippets often passed ``--max-rel-err-assets 0.05``
    explicitly, which silently regenerated stale PARTIAL_PASS reports even after the
    v17 semantics patch.  Keep the value configurable only through the declared
    contract constant and reject mismatches here so the pipeline, verifier, metadata,
    and runner cannot drift apart.
    """
    try:
        value = float(max_rel_err_assets)
    except Exception as exc:
        raise ValueError(f"max_rel_err_assets must be numeric, got {max_rel_err_assets!r}") from exc
    expected = float(B2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS)
    if not np.isfinite(value):
        raise ValueError(f"max_rel_err_assets must be finite, got {value!r}")
    if abs(value - expected) > 1e-12:
        raise ValueError(
            "LoopB2 configured diagnostic contract requires --max-rel-err-assets "
            f"{expected:.2f}; got {value:.6g}. Remove stale command-line overrides such "
            "as -MaxRelErrAssets 0.05 or pass 0.10 explicitly."
        )
    return expected

def run(
    project_root: Path,
    *,
    sim_business_plan_mode: str = "default",
    preserve_current_non_current_residual: bool = False,
    max_rel_err_assets: float = B2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS,
    min_coverage_share: float = 0.80,
    output_dir: Path | None = None,
    magnitude_quantile: int = 50,
    max_identity_rel_err_assets: float = 0.02,
) -> dict:
    """Verify Stage2 substrate using the mixed-transition contract.

    Important contract distinction:
    * observed/matched action rows intentionally preserve the empirical observed t+1 transition;
    * non-observed counterfactual rows are generated by the simulator and must be checked by internal sanity, not by comparison to a non-existent real t+1 counterfactual;
    * historical projected-action re-simulation is still written as a stress diagnostic, but it is not a hard gate.
    """
    root = project_root.resolve()
    final = final_root(root)
    out = output_dir if output_dir is not None else final / "stage2_substrate_loopA_loopB2"
    out.mkdir(parents=True, exist_ok=True)
    max_rel_err_assets = _require_configured_loopb2_fidelity_threshold(max_rel_err_assets)

    # Diagnostic-only historical re-simulation stress test.  This preserves the
    # previous LoopA outputs for comparability, but it no longer determines the
    # Stage2 contract verdict.
    phase, loopa_driver_meta = _build_loopa_phase(final)
    prev_state_handoff, prev_state_handoff_source = _load_observed_next_handoff(final)
    prev_state_loader_meta = _verify_loopb2_prev_state_ucode_loader_contract()
    phase_before_actual_denominator_merge = phase.copy()

    # True Test2/Test3 simulator validation must start from actual t-year
    # financial statement denominators.  The Stage2 P-handoff can contain the
    # same U-code headers with all-zero sentinels; using those values makes
    # R157/R182 undefined even though coverage appears to be 100%.
    phase, phase_prev_state_meta = _merge_loopb2_prev_state_inputs_from_actual_financial_panel(
        phase,
        final,
    )

    # Keep a separate observed-next handoff merge for the replay diagnostic and
    # for traceability, but do not use this P-handoff merge as the true simulator
    # prev_state source.
    _, replay_prev_state_meta = _merge_loopb2_prev_state_inputs_from_handoff(
        phase_before_actual_denominator_merge,
        prev_state_handoff,
        source_label=prev_state_handoff_source,
        source_role="stage2_observed_next_handoff_replay_diagnostic",
        source_lookup_policy="observed-next replay diagnostic denominator trace only; not used for true simulator validation",
        require_nonzero_source=False,
    )
    loopa_driver_meta["loopB2_prev_state_handoff_source"] = prev_state_handoff_source
    loopa_driver_meta["loopB2_prev_state_loader_contract"] = prev_state_loader_meta
    loopa_driver_meta["loopB2_prev_state_actual_financial_source_merge_for_loopA_driver"] = phase_prev_state_meta
    loopa_driver_meta["loopB2_prev_state_handoff_merge_for_replay_diagnostic"] = replay_prev_state_meta

    space = load_action_space(root)
    phase, observed_action_value_source_merge_meta = _merge_loopb2_observed_action_values_from_source(
        phase,
        final,
        space,
    )
    loopa_driver_meta["loopB2_observed_action_value_source_merge"] = observed_action_value_source_merge_meta
    observed_action_source_meta = _validate_loopb2_observed_action_source(
        phase,
        space,
        label="LoopA/Test2-Test3 historical observed-action driver",
    )
    observed_action_source_meta["source_merge"] = observed_action_value_source_merge_meta
    loopa_driver_meta["loopB2_observed_action_source_contract"] = observed_action_source_meta
    sim = FinancialSimulator(preserve_current_non_current_residual=preserve_current_non_current_residual)
    sim_rows = []
    for _, r in phase.iterrows():
        fs = _row_to_state(r)
        act = _observed_action(r, space)
        hist = (
            phase[
                (phase.get("firm_id", "").astype(str) == str(fs.firm_id))
                & (pd.to_numeric(phase.get("fiscal_year", 0), errors="coerce") <= float(fs.year))
            ]
            .sort_values("fiscal_year")
            .tail(3)
            if "firm_id" in phase.columns
            else pd.DataFrame()
        )
        h = []
        for _, hr in hist.iterrows():
            try:
                h.append(_row_to_state(hr))
            except Exception:
                pass
        bp = _select_business_plan(sim_business_plan_mode, h, rating_grade=fs.rating_grade)
        res = sim.simulate(fs, bp, act)
        row = {"firm_id": fs.firm_id, "fiscal_year": fs.year}
        row.update(_numeric_state_dict(res.state_t1))
        row["simulator_sustainability"] = res.sustainability
        row["simulator_plug_used"] = res.plug_used
        row["simulator_plug_amount"] = float(res.plug_amount)
        row["preserve_current_non_current_residual"] = bool(preserve_current_non_current_residual)
        residual_audit = (res.diagnostics or {}).get("residual_audit", {}) or {}
        row["simulator_residual_audit_json"] = json.dumps(residual_audit, ensure_ascii=False, sort_keys=True)
        row["simulator_residual_negative_flag"] = bool(residual_audit.get("residual_negative_flag", False))
        row["simulator_residual_fallback_used"] = bool(residual_audit.get("residual_fallback_used", False))
        sim_rows.append(row)

    pred = pd.DataFrame(sim_rows)
    errs = _compute_error_rows(phase, sim_rows)
    summary = _summarize_errors(errs)
    stress_gate = _group_gate(summary, max_rel_err_assets=max_rel_err_assets, min_coverage_share=min_coverage_share)
    if errs.empty:
        stress_gate["status"] = "FAIL"
        stress_gate.setdefault("violations", []).append({"reason": "no_comparable_next_state_values"})
    pred.to_parquet(out / "loopA_predicted_tplus1_states.parquet", index=False)
    errs.to_csv(out / "loopA_dimension_errors.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out / "loopA_dimension_bias_summary.csv", index=False, encoding="utf-8-sig")

    cf, cf_source = _load_counterfactual_handoff(final, magnitude_quantile)
    loopA_contract_gate: dict[str, Any]
    observed_gate: dict[str, Any]
    counterfactual_gate: dict[str, Any]
    b2_phase = phase
    b2_pred = pred
    b2_source = "historical_projected_action_resimulation_stress"
    b2_prev_state_meta = phase_prev_state_meta

    if cf is not None and not cf.empty:
        observed_gate = _observed_transition_preservation_gate(cf)
        counterfactual_gate = _counterfactual_simulator_sanity_gate(
            cf,
            max_identity_rel_err_assets=max_identity_rel_err_assets,
            min_coverage_share=min_coverage_share,
        )
        loopA_contract_gate = {
            "status": "PASS" if observed_gate.get("status") == "PASS" and counterfactual_gate.get("status") == "PASS" else "FAIL",
            "gate_type": "stage2_mixed_transition_contract",
            "counterfactual_artifact": cf_source,
            "observed_transition_preservation_gate": observed_gate,
            "counterfactual_simulator_sanity_gate": counterfactual_gate,
            "note": "Historical observed-action re-simulation is reported as stress diagnostic only and does not determine this gate.",
        }
        b2_phase = cf.reset_index(drop=True)
        b2_phase, b2_prev_state_meta = _merge_loopb2_prev_state_inputs_from_handoff(
            b2_phase,
            prev_state_handoff,
            source_label=prev_state_handoff_source,
        )
        b2_pred = _build_used_next_state_frame(cf).reset_index(drop=True)
        b2_source = "stage2_mixed_transition_used_next_state"
        b2_pred.to_parquet(out / "loopB2_used_next_state_frame.parquet", index=False)
    else:
        observed_gate = {"status": "SKIP_NO_COUNTERFACTUAL_ARTIFACT"}
        counterfactual_gate = {"status": "SKIP_NO_COUNTERFACTUAL_ARTIFACT"}
        loopA_contract_gate = {
            "status": "WARN_NO_COUNTERFACTUAL_ARTIFACT",
            "gate_type": "stage2_mixed_transition_contract",
            "counterfactual_artifact": None,
            "observed_transition_preservation_gate": observed_gate,
            "counterfactual_simulator_sanity_gate": counterfactual_gate,
            "note": "No counterfactual artifact found; only historical re-simulation stress outputs were written.",
        }

    # Loop B2 true gate: research-design Test 3.  This is the actual
    # simulator validation: simulate observed historical action t-1->t, score
    # the simulated t state, and compare that lead score change to the real
    # t->t+1 rating movement using the same time convention as Stage1 Loop B1.
    alpha_params, registry_source, alpha_params_ref = _resolve_alpha_params(root, final)
    b2 = {
        "status": "SKIP_NO_ALPHA_PARAMS",
        "rule_status": "FAIL",
        "reason": "No alpha params could be resolved from canonical registry or fallback path",
        "registry_source": registry_source,
        "alpha_params_ref": alpha_params_ref,
        "validation_type": B2_SIMULATOR_VALIDATION_TYPE,
        "scoring_source": "historical_observed_action_resimulation",
        "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
        "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
    }
    replay_b2 = {
        "status": "SKIP_NOT_RUN",
        "rule_status": "FAIL",
        "reason": "Observed-next replay diagnostic was not run because alpha params or mixed-transition rows were unavailable",
        "validation_type": B2_REPLAY_DIAGNOSTIC_TYPE,
        "does_not_determine_status": True,
        "policy": B2_REPLAY_DIAGNOSTIC_POLICY,
        "scoring_source": b2_source,
    }

    if alpha_params is not None:
        # Diagnostic only: keep the previously implemented observed-next replay test
        # as a named diagnostic so existing evidence is not lost, but never allow it
        # to determine substrate_tier.  It is not action-conditioned simulator
        # validation because observed rows preserve empirical t+1 values.
        if not b2_pred.empty:
            b2_scoring_phase, b2_scoring_pred, b2_scoring_population_meta = _restrict_loopb2_scoring_population_to_observed(
                b2_phase,
                b2_pred,
            )
            replay_alpha_frame, replay_alpha_frame_meta = _build_alpha_scoring_frame(
                b2_scoring_phase,
                b2_scoring_pred,
                alpha_params,
            )
            replay_alpha_frame_meta["prev_state_ucode_loader_contract"] = prev_state_loader_meta
            replay_alpha_frame_meta["prev_state_handoff_merge"] = b2_prev_state_meta
            replay_alpha_frame_meta["scoring_population"] = b2_scoring_population_meta
            replay_alpha_frame.to_csv(out / "loopB2_replay_alpha_scoring_input_frame.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(replay_alpha_frame_meta.get("diagnostic_rows", [])).to_csv(out / "loopB2_replay_alpha_scoring_input_missingness.csv", index=False, encoding="utf-8-sig")
            write_json(out / "loopB2_replay_alpha_scoring_frame_meta_pre_guard.json", _json_safe(replay_alpha_frame_meta))
            replay_alpha_frame_meta["r157_r182_nonnull_share_guard"] = _assert_loopb2_r157_r182_scoring_coverage(
                replay_alpha_frame,
                alpha_frame_meta=replay_alpha_frame_meta,
            )
            if replay_alpha_frame_meta.get("missing_columns"):
                replay_b2 = {
                    "status": "FAIL_MISSING_ALPHA_INPUT_COLUMNS",
                    "rule_status": "FAIL",
                    "rows": int(len(replay_alpha_frame)),
                    "registry_source": registry_source,
                    "alpha_params": str(alpha_params),
                    "alpha_params_ref": alpha_params_ref,
                    "alpha_scoring_frame": replay_alpha_frame_meta,
                    "validation_type": B2_REPLAY_DIAGNOSTIC_TYPE,
                    "does_not_determine_status": True,
                    "policy": B2_REPLAY_DIAGNOSTIC_POLICY,
                    "scoring_source": b2_source,
                }
            else:
                replay_score = score_alpha(replay_alpha_frame, alpha_params)
                base_cols = ["firm_id", "fiscal_year"]
                optional_cols = [
                    "candidate_id",
                    "is_observed_transition",
                    "counterfactual_transition",
                    "rating_num_10",
                    "rating_num_10__next",
                    "fiscal_year__next",
                    "year__next",
                    "next__fiscal_year",
                    "next__year",
                ]
                base_cols.extend([c for c in optional_cols if c in b2_scoring_phase.columns and c not in base_cols])
                replay_df = b2_scoring_phase[[c for c in base_cols if c in b2_scoring_phase.columns]].copy()
                replay_df["pred_alpha_score_tplus1"] = replay_score.to_numpy()
                if "rating_num_10__next" in b2_scoring_phase.columns and "rating_num_10" in b2_scoring_phase.columns:
                    replay_df["real_rating_delta_t_to_tplus1"] = (
                        pd.to_numeric(b2_scoring_phase["rating_num_10"], errors="coerce")
                        - pd.to_numeric(b2_scoring_phase["rating_num_10__next"], errors="coerce")
                    )
                scored_replay_df, replay_eval = _evaluate_loopb2_alpha_direction(root, final, b2_scoring_phase, replay_df, replay_alpha_frame_meta)
                scored_replay_df.to_csv(out / "loopB2_replay_alpha_predicted_score_vs_real_rating_change.csv", index=False, encoding="utf-8-sig")
                replay_b2 = {
                    **replay_eval,
                    "validation_type": B2_REPLAY_DIAGNOSTIC_TYPE,
                    "does_not_determine_status": True,
                    "policy": B2_REPLAY_DIAGNOSTIC_POLICY,
                    "registry_source": registry_source,
                    "alpha_params": str(alpha_params),
                    "alpha_params_ref": alpha_params_ref,
                    "alpha_scoring_frame": replay_alpha_frame_meta,
                    "scoring_source": b2_source,
                }

        # Actual Test 3 gate: score retention through the historical observed-action
        # simulator path, aligned to Stage1 B1 lead convention.
        if not pred.empty:
            sim_alpha_frame, sim_target_phase, sim_alpha_frame_meta = _build_loopb2_simulator_score_retention_frame(
                root,
                phase,
                pred,
                alpha_params,
                prev_state_loader_meta,
                phase_prev_state_meta,
            )
            sim_alpha_frame_meta["observed_action_source_contract"] = observed_action_source_meta
            sim_alpha_frame_meta["action_source"] = B2_TRUE_SIMULATOR_ACTION_SOURCE
            sim_alpha_frame_meta["action_source_policy"] = B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY
            sim_alpha_frame.to_csv(out / "loopB2_simulator_score_retention_scoring_input_frame.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame(sim_alpha_frame_meta.get("diagnostic_rows", [])).to_csv(out / "loopB2_simulator_score_retention_scoring_input_missingness.csv", index=False, encoding="utf-8-sig")
            write_json(out / "loopB2_simulator_score_retention_scoring_frame_meta_pre_guard.json", _json_safe(sim_alpha_frame_meta))
            if sim_alpha_frame.empty:
                b2 = {
                    "status": "FAIL_B2_SCORE_RETENTION_EMPTY",
                    "rule_status": "FAIL",
                    "reason": "No consecutive observed transitions were available for Test 3 simulator score retention",
                    "registry_source": registry_source,
                    "alpha_params": str(alpha_params),
                    "alpha_params_ref": alpha_params_ref,
                    "alpha_scoring_frame": sim_alpha_frame_meta,
                    "validation_type": B2_SIMULATOR_VALIDATION_TYPE,
                    "scoring_source": "historical_observed_action_resimulation",
                    "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
                    "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
                }
            else:
                sim_alpha_frame_meta["r157_r182_nonnull_share_guard"] = _assert_loopb2_r157_r182_scoring_coverage(
                    sim_alpha_frame,
                    alpha_frame_meta=sim_alpha_frame_meta,
                )
                sim_score = score_alpha(sim_alpha_frame, alpha_params)
                scored_sim_df, sim_eval = _evaluate_loopb2_simulator_score_retention(
                    root,
                    final,
                    sim_target_phase,
                    pd.Series(sim_score.to_numpy(), name="pred_alpha_score_t_from_observed_action_sim_tminus1_to_t"),
                    sim_alpha_frame_meta,
                )
                scored_sim_df.to_csv(out / "loopB2_simulator_score_retention.csv", index=False, encoding="utf-8-sig")
                b2 = {
                    **sim_eval,
                    "registry_source": registry_source,
                    "alpha_params": str(alpha_params),
                    "alpha_params_ref": alpha_params_ref,
                    "alpha_scoring_frame": sim_alpha_frame_meta,
                    "scoring_source": "historical_observed_action_resimulation",
                    "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
                    "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
                }
        else:
            b2 = {
                "status": "SKIP_EMPTY_HISTORICAL_SIMULATION_PRED",
                "rule_status": "FAIL",
                "registry_source": registry_source,
                "alpha_params": str(alpha_params),
                "alpha_params_ref": alpha_params_ref,
                "validation_type": B2_SIMULATOR_VALIDATION_TYPE,
                "scoring_source": "historical_observed_action_resimulation",
                "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
                "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
            }

    files = [
        "loopA_predicted_tplus1_states.parquet",
        "loopA_dimension_errors.csv",
        "loopA_dimension_bias_summary.csv",
        "substrate_loopA_loopB2_report.json",
    ]
    for extra in [
        "loopB2_simulator_score_retention.csv",
        "loopB2_simulator_score_retention_scoring_input_frame.csv",
        "loopB2_simulator_score_retention_scoring_input_missingness.csv",
        "loopB2_simulator_score_retention_scoring_frame_meta_pre_guard.json",
        "loopB2_replay_alpha_predicted_score_vs_real_rating_change.csv",
        "loopB2_replay_alpha_scoring_input_frame.csv",
        "loopB2_replay_alpha_scoring_input_missingness.csv",
        "loopB2_replay_alpha_scoring_frame_meta_pre_guard.json",
        "loopB2_used_next_state_frame.parquet",
        # legacy names from the pre-v11 replay implementation, if a local run still carries them
        "loopB2_alpha_predicted_score_vs_real_rating_change.csv",
        "loopB2_alpha_scoring_input_frame.csv",
        "loopB2_alpha_scoring_input_missingness.csv",
        "loopB2_alpha_scoring_frame_meta_pre_guard.json",
    ]:
        if (out / extra).exists():
            files.append(extra)

    simulator_financial_fidelity_gate = {
        **stress_gate,
        "gate_type": "test2_simulator_financial_fidelity_diagnostic_observed_action_resimulation",
        "determines_strong_pass": False,
        "does_not_determine_status": True,
        "threshold_role": "engineering_diagnostic_threshold_not_preregistered_research_gate",
        "policy": "Diagnostic only after 검사3 removal: compares simulator-predicted t+1 financial statement dimensions against actual t+1 financial statement dimensions for observed historical actions; reported for audit but not used as top-level pass/fail criterion.",
        "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
        "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        "current_research_gate_policy": B2_CURRENT_RESEARCH_GATE_POLICY,
    }
    substrate_tier = _stage2_substrate_tier(loopA_contract_gate, b2, simulator_financial_fidelity_gate)
    status = _top_level_status(loopA_contract_gate, b2, substrate_tier)
    meta = {
        "status": status,
        "substrate_tier": substrate_tier,
        "substrate_tier_rule": "Current 검사3제거 contract: PASS iff Stage2 mixed-transition internal sanity and all artifact/semantic guards pass; simulator financial-fidelity and score-retention are diagnostics only and do not determine top-level status.",
        "research_interpretation_status": B2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS if substrate_tier == B2_STAGE2_DIAGNOSTIC_TIER else ("SIMULATOR_VALIDATED" if substrate_tier == "strong_pass" else "ORACLE_VALIDATED_SIMULATOR_NOT_VALIDATED"),
        "current_research_gate_policy": B2_CURRENT_RESEARCH_GATE_POLICY,
        "sim_business_plan_mode": sim_business_plan_mode,
        "preserve_current_non_current_residual": bool(preserve_current_non_current_residual),
        "magnitude_quantile": int(magnitude_quantile),
        "loopA_rows": int(len(pred)),
        "loopA_phase_rows": int(len(phase)),
        **loopa_driver_meta,
        "loopA_comparable_error_rows": int(len(errs)),
        "loopA_dimensions": int(summary["dimension"].nunique()) if not summary.empty else 0,
        "loopA_contract_gate": loopA_contract_gate,
        "simulator_financial_fidelity_gate": simulator_financial_fidelity_gate,
        "loopB2_observed_action_source_contract": observed_action_source_meta,
        "loopA_historical_resimulation_stress": {
            "status": stress_gate.get("status"),
            "gate_type": "test2_simulator_financial_fidelity_diagnostic_observed_action_resimulation",
            "fidelity_gate": stress_gate,
            "interpretation": "Diagnostic-only simulator re-simulation of realized C_obs observed actions vs actual next state under the current 검사3제거 contract.",
            "determines_strong_pass": False,
            "does_not_determine_status": True,
            "threshold_role": "engineering_diagnostic_threshold_not_preregistered_research_gate",
            "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
            "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        },
        # Backward-compatible key, now explicitly points to the Test2 fidelity gate.
        "loopA_fidelity_gate": {
            **stress_gate,
            "gate_type": "test2_simulator_financial_fidelity_diagnostic_observed_action_resimulation",
            "does_not_determine_status": True,
            "determines_strong_pass": False,
            "threshold_role": "engineering_diagnostic_threshold_not_preregistered_research_gate",
            "action_source": B2_TRUE_SIMULATOR_ACTION_SOURCE,
            "action_source_policy": B2_TRUE_SIMULATOR_ACTION_SOURCE_POLICY,
        },
        "loopB2": b2,
        "loopB2_replay_diagnostic": replay_b2,
        "outputs": files,
        "primary_output_dir": str(out),
        "compatibility_output_dirs": [],
    }
    meta = _json_safe(meta)
    write_json(out / "substrate_loopA_loopB2_report.json", meta)
    compat = _write_compatibility_outputs(out, final, files)
    meta["compatibility_output_dirs"] = sorted(set(str(Path(x).parent) for x in compat))
    meta = _json_safe(meta)
    write_json(out / "substrate_loopA_loopB2_report.json", meta)
    _write_compatibility_outputs(out, final, files)

    return meta

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--sim-business-plan-mode", choices=["default", "calibrated"], default="default")
    ap.add_argument("--preserve-current-non-current-residual", action="store_true")
    ap.add_argument("--max-rel-err-assets", type=float, default=B2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS)
    ap.add_argument("--min-coverage-share", type=float, default=0.80)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--magnitude-quantile", type=int, default=50, choices=[50, 65, 75, 85])
    ap.add_argument("--max-identity-rel-err-assets", type=float, default=0.02)
    args = ap.parse_args(argv)
    meta = run(
        Path(args.project_root).resolve(),
        sim_business_plan_mode=args.sim_business_plan_mode,
        preserve_current_non_current_residual=args.preserve_current_non_current_residual,
        max_rel_err_assets=args.max_rel_err_assets,
        min_coverage_share=args.min_coverage_share,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        magnitude_quantile=args.magnitude_quantile,
        max_identity_rel_err_assets=args.max_identity_rel_err_assets,
    )
    print(json.dumps(_json_safe(meta), ensure_ascii=False, indent=2))
    return 0 if meta.get("status") == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
