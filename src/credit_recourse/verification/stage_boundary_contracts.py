from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from credit_recourse.contracts.stage_paths import CANONICAL_STAGE_DIRS, DEPRECATED_STAGE_DIR_ALIASES, final_root
from credit_recourse.oracle.stage0.rating_contract_repair import validate_stage0_contract
from credit_recourse.utils.io_contract import read_json, write_json, resolve_selected_variables, sha256_file
from credit_recourse.rl.common.actions import load_action_space, resolve_candidate_library_path
from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.budget_contract import BUDGET_AUDIT_COLUMNS, ACTION_BUDGET_CONTRACT_SCHEMA_VERSION
from credit_recourse.rl.contracts.avs256_acd_v2 import SCHEMA_VERSION, CONTINUOUS_COLUMNS, ACD_TARGET_COLUMNS, CATEGORICAL_COLUMNS, EXPECTED_BLOCK_COUNTS, DIRECTION_VOCAB

V32_LABELS = [
    "A0_noop",
    "DL1_deleverage_mild",
    "DL2_deleverage_moderate",
    "RF1_short_debt_refinance",
    "CX1_capex_discipline",
    "WC1_working_capital_tightening",
    "WC2_supplier_financing",
    "OE1_cost_efficiency_mild",
    "OE2_cost_efficiency_moderate",
    "MX1_cost_and_deleverage",
    "MX2_liquidity_rescue",
]
ACTION_COLS = [
    "action__ppe_pct", "action__inv_turnover_chg", "action__ar_turnover_chg", "action__ap_turnover_chg",
    "action__short_debt_pct", "action__long_debt_pct", "action__bond_pct", "action__revenue_growth",
    "action__cogs_ratio_chg", "action__sga_ratio_chg",
]



def _final_encoder_contract() -> dict[str, int]:
    """Resolve encoder architecture constants only for Stage3/RL checks.

    The encoder module imports torch because it defines the model class. Keeping
    this dependency behind the Stage3 verifier boundary allows Stage7/8/9 and
    config-only contract checks to import in environments without torch.
    """
    try:
        from credit_recourse.rl.final_candidate.encoder import (
            FINAL_ENCODER_D_MODEL,
            FINAL_ENCODER_N_HEADS,
            FINAL_ENCODER_N_LAYERS,
            FINAL_ENCODER_FF_MULTIPLIER,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        if exc.name == "torch" or str(exc.name).startswith("torch."):
            raise RuntimeError(
                "PyTorch is required to verify Stage3 encoder architecture. "
                "Install the project RL environment before running Stage3 checks."
            ) from exc
        raise
    return {
        "d_model": int(FINAL_ENCODER_D_MODEL),
        "n_heads": int(FINAL_ENCODER_N_HEADS),
        "n_layers": int(FINAL_ENCODER_N_LAYERS),
        "ff_multiplier": int(FINAL_ENCODER_FF_MULTIPLIER),
    }


def _torch_load_checkpoint(path: Path) -> Any:
    """Load a torch checkpoint only when a checkpoint verifier is invoked.

    Keeping torch out of module import time allows config/schema-only contract
    checks to run in lightweight analysis environments. Checkpoint validation
    still hard-fails with a precise dependency error when torch is unavailable.
    """
    try:
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"PyTorch is required to verify checkpoint artifact: {path}. "
            "Install the project RL environment before running this verifier."
        ) from exc
    return torch.load(path, map_location="cpu")

ACTION_MIN_OBSERVED_RATE = {
    "ppe_pct": 0.20,
    "inv_turnover_chg": 0.20,
    "ar_turnover_chg": 0.20,
    "ap_turnover_chg": 0.20,
    "short_debt_pct": 0.10,
    "long_debt_pct": 0.10,
    "bond_pct": 0.05,
    "revenue_growth": 0.20,
    "cogs_ratio_chg": 0.20,
    "sga_ratio_chg": 0.20,
}

def action_cols(root: Path) -> list[str]:
    return list(load_action_space(root).columns)

def train_labels(root: Path) -> list[str]:
    return list(load_action_space(root).train_labels)

REWARD_COLS = [
    "reward_raw_notch", "reward_raw", "phi_t", "phi_tplusH", "delta_phi", "delta_phi_clipped",
    "lambda_phi", "reward_aux_phi", "reward_total_raw", "reward_mean_train", "reward_std_train",
    "reward_train", "reward_original", "reward",
]

MERTON_AUX_REWARD_COLS = [
    "merton_default_point_t", "merton_default_point_tplusH", "merton_badness_t", "merton_badness_tplusH",
    "delta_merton_badness", "delta_merton_badness_scaled", "lambda_merton", "reward_aux_merton",
]
FCFF_AUX_REWARD_COLS = [
    "fcff_capacity_t", "fcff_capacity_tplusH", "delta_fcff_capacity", "delta_fcff_capacity_scaled",
    "lambda_fcff", "reward_aux_fcff",
]
LIQUIDITY_AUX_REWARD_COLS = [
    "liquid_capacity_t", "liquid_capacity_tplusH", "delta_liquid_capacity",
    "delta_liquid_capacity_scaled", "lambda_liquidity", "reward_aux_liquidity",
]
AUX_REWARD_COLS = MERTON_AUX_REWARD_COLS + FCFF_AUX_REWARD_COLS + LIQUIDITY_AUX_REWARD_COLS

STAGE6_REQUIRED_FIRMSTATE_FIELDS = [
    "revenue", "cogs", "sga", "total_assets", "current_assets", "current_liabilities", "cash",
    "inventory", "receivables", "payables", "ppe", "short_term_debt", "long_term_debt", "bonds",
    "total_liabilities", "total_equity",
]

STAGE6_FIRMSTATE_FIELD_ALIASES = {
    "receivables": ["accounts_receivable"],
    "payables": ["accounts_payable"],
    "short_term_debt": ["short_debt"],
    "long_term_debt": ["long_debt"],
    "bonds": ["bond"],
}


# Claim-mode policy for sensitivity-grid verification.
#
# The stage-boundary verifier is used in two different contexts:
#   1. fixed-config/local downstream cells, where sensitivity grids may
#      explicitly register candidate configurations that were intentionally
#      not run; and
#   2. completed full-sweep claims, where every registered row must be run.
#
# Default CLI mode is (1).  Use --strict-full-sweep-claim for (2).
STRICT_FULL_SWEEP_CLAIM = False

REGISTERED_NOT_RUN_SOFT_GATE_CODE = "FIXED_CONFIG_SOFT_GATE"
REGISTERED_NOT_RUN_STRICT_GATE_CODE = "STRICT_FULL_SWEEP_CLAIM"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def nonempty(path: Path, errors: list[str], label: str | None = None) -> bool:
    if not path.exists():
        errors.append(f"missing {label or path}")
        return False
    if path.is_file() and path.stat().st_size <= 0:
        errors.append(f"empty {label or path}")
        return False
    return True



def check_sensitivity_grid(path: Path, errors: list[str], warnings: list[str], label: str, *, require_selected: bool = True) -> dict[str, Any]:
    """Validate a phase sensitivity grid with explicit claim-mode semantics.

    REGISTERED_NOT_RUN is not intrinsically a stage-boundary failure.  It means
    that the grid contains registered configurations that were not executed.
    That is acceptable for a fixed-config/local downstream cell, but it is not
    acceptable when the run is being presented as a completed full sensitivity
    sweep.  The verifier therefore treats it as:

    * warning in the default fixed-config boundary-check mode;
    * error only when --strict-full-sweep-claim is supplied.
    """
    if not nonempty(path, errors, label):
        return {"status": "MISSING"}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"cannot read {label}: {exc!r}")
        return {"status": "FAIL"}
    if "status" not in df.columns:
        errors.append(f"{label} missing status column")
        return {"status": "FAIL", "rows": int(len(df))}

    counts = df["status"].astype(str).value_counts().to_dict()
    selected_n = int(counts.get("SELECTED_RUN", 0))
    registered_not_run_n = int(counts.get("REGISTERED_NOT_RUN", 0))

    if require_selected and selected_n < 1:
        errors.append(f"{label} must contain at least one SELECTED_RUN row")
    elif (not require_selected) and selected_n < 1:
        warnings.append(f"{label} has no SELECTED_RUN row; accepted for fixed-config/local runs only")

    if registered_not_run_n > 0:
        msg = (
            f"{label} contains REGISTERED_NOT_RUN rows; "
            f"rows={registered_not_run_n}; "
            "acceptable for fixed-config/local downstream cells, "
            "not acceptable for a completed full sensitivity sweep claim"
        )
        if STRICT_FULL_SWEEP_CLAIM:
            errors.append(f"[{REGISTERED_NOT_RUN_STRICT_GATE_CODE}] {msg}")
        else:
            warnings.append(f"[{REGISTERED_NOT_RUN_SOFT_GATE_CODE}] {msg}")

    status = "PASS" if (selected_n >= 1 or not require_selected) else "FAIL"
    if registered_not_run_n > 0 and STRICT_FULL_SWEEP_CLAIM:
        status = "FAIL_FULL_SWEEP_CLAIM"

    return {
        "status": status,
        "rows": int(len(df)),
        "status_counts": counts,
        "selected_run_rows": selected_n,
        "registered_not_run_rows": registered_not_run_n,
        "registered_not_run_policy": (
            "ERROR_STRICT_FULL_SWEEP_CLAIM" if STRICT_FULL_SWEEP_CLAIM else "WARNING_FIXED_CONFIG_SOFT_GATE"
        ),
        "strict_full_sweep_claim": bool(STRICT_FULL_SWEEP_CLAIM),
    }

def read_parquet(path: Path, errors: list[str], label: str | None = None) -> pd.DataFrame:
    if not nonempty(path, errors, label):
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        errors.append(f"cannot read parquet {label or path}: {exc!r}")
        return pd.DataFrame()


def check_cols(df: pd.DataFrame, cols: list[str], errors: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"{where} missing columns: {missing}")


def _stage6_field_alias_matches(column: str, field: str) -> bool:
    c = str(column)
    aliases = [field] + list(STAGE6_FIRMSTATE_FIELD_ALIASES.get(field, []))
    for alias in aliases:
        if c == alias or c == f"sim__{alias}":
            return True
        if c == f"raw__{alias}" or c == f"raw__sim__{alias}" or c == f"avs__{alias}" or c == f"raw__avs__{alias}":
            return True
        if c.endswith(f"__{alias}"):
            return True
    return False


def check_stage6_serving_state_columns(df: pd.DataFrame, errors: list[str], where: str, *, min_coverage: float = 0.50) -> dict[str, Any]:
    rows = []
    for field in STAGE6_REQUIRED_FIRMSTATE_FIELDS:
        matches = [c for c in df.columns if _stage6_field_alias_matches(str(c), field)]
        coverage = 0.0
        if matches:
            coverage = max(float(pd.to_numeric(df[c], errors="coerce").notna().mean()) for c in matches)
        rows.append({"field": field, "matched_columns": matches[:20], "coverage": coverage, "status": "PASS" if coverage >= min_coverage else "LOW_COVERAGE"})
    bad = [r for r in rows if r["status"] != "PASS"]
    if bad:
        errors.append(f"{where} missing/low coverage Stage6 FirmState serving columns: {bad[:8]}")
    return {"status": "PASS" if not bad else "FAIL", "min_coverage": min_coverage, "rows": rows}


def check_unique(df: pd.DataFrame, keys: list[str], errors: list[str], where: str) -> int:
    if not all(k in df.columns for k in keys):
        errors.append(f"{where} cannot check duplicates; missing key columns {keys}")
        return -1
    dup = int(df.duplicated(keys).sum())
    if dup:
        errors.append(f"{where} duplicate key rows for {keys}: {dup}")
    return dup


def metadata_status(path: Path, errors: list[str], allow_prefix: tuple[str, ...] = ("PASS",)) -> dict[str, Any]:
    if not nonempty(path, errors):
        return {}
    try:
        meta = read_json(path)
    except Exception as exc:
        errors.append(f"cannot read metadata json {path}: {exc!r}")
        return {}
    status = str(meta.get("status", ""))
    if status and not status.startswith(allow_prefix):
        errors.append(f"metadata status not allowed at {path}: {status}")
    return meta


def verify_stage0(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    stage0 = final_root(root) / CANONICAL_STAGE_DIRS["stage0"]
    ok, errs, meta = validate_stage0_contract(stage0)
    if not ok:
        errors.extend(errs)
    nonempty(stage0 / "canonical_panel" / "stage0_canonical_panel.parquet", errors)
    nonempty(stage0 / "canonical_panel" / "statement_items_panel.parquet", errors)
    metadata_status(stage0 / "stage0_manifest.json", errors)
    return {"stage0_contract": meta}


def verify_stage1_bridge(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    p = final / "stage1_oracle_inputs" / "alpha_vanilla_input_candidate.parquet"
    meta_p = final / "stage1_oracle_inputs" / "alpha_vanilla_input_candidate_metadata.json"
    df = read_parquet(p, errors, "Stage1→Stage2 bridge")
    selected_contract = resolve_selected_variables(final)
    selected = selected_contract.get("selected_variables", [])
    if not df.empty:
        check_cols(df, ["firm_id", "fiscal_year", "거래소코드", "year", "rating_num_10", "rating_num", "split", "selected_variables_all_complete"], errors, "bridge")
        check_unique(df, ["firm_id", "fiscal_year"], errors, "bridge")
        missing = [v for v in selected if v not in df.columns]
        if missing:
            errors.append(f"bridge missing dynamic selected variables: {missing}")
        if "selected_variables_all_complete" in df.columns and int(df["selected_variables_all_complete"].sum()) == 0:
            errors.append("bridge has zero selected_variables_all_complete rows")
        if "rating_num_10" in df.columns and "rating_num" in df.columns:
            neq = int((pd.to_numeric(df["rating_num"], errors="coerce") != pd.to_numeric(df["rating_num_10"], errors="coerce")).fillna(False).sum())
            if neq:
                errors.append(f"bridge rating_num alias differs from rating_num_10 rows={neq}")
    bmeta = metadata_status(meta_p, errors)
    return {"rows": int(len(df)), "selected_variable_contract": selected_contract, "bridge_metadata_status": bmeta.get("status")}


def verify_stage1(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    ledger = final / "ledgers" / "stage1_oracle_backends_full_development.json"
    meta = metadata_status(ledger, errors)
    for p in [
        final / "stage1_oracle_backends" / "alpha" / "oracle_alpha_params.json",
        final / "stage1_oracle_backends" / "alpha" / "oracle_firm_year_output_alpha.parquet",
        final / "stage1_oracle_backends" / "beta" / "benchmark_beta_params.json",
        final / "stage1_oracle_backends" / "beta" / "benchmark_firm_year_output_beta.parquet",
        final / "stage1_oracle_backends" / "gamma" / "benchmark_gamma_params.json",
        final / "stage1_oracle_backends" / "gamma" / "benchmark_firm_year_output_gamma.parquet",
        final / "configs" / "oracle_backend_registry.yaml",
    ]:
        nonempty(p, errors)
    bridge = verify_stage1_bridge(root, errors, warnings)
    substrate_path = final / "ledgers" / "stage1_substrate_validation_loopB1.json"
    substrate = metadata_status(substrate_path, errors, allow_prefix=("PASS",))
    if substrate.get("status") not in {"PASS", "PASS_PARTIAL"}:
        errors.append(f"Stage1 substrate validation Loop B1 gate missing or failed: {substrate_path}")
    return {"ledger_status": meta.get("status"), "bridge": bridge, "substrate_validation_loopB1_status": substrate.get("status")}




def _root_from_final_path(path: Path) -> Path:
    parts = list(path.resolve().parts)
    for i in range(len(parts)-1):
        if parts[i] == "data" and i+1 < len(parts) and parts[i+1] == "final_freeze":
            return Path(*parts[:i])
    return path.resolve().parents[3]

def check_action_source_coverage(path: Path, errors: list[str], warnings: list[str], where: str, *, min_observed_rate: float = 0.05) -> dict[str, Any]:
    if not nonempty(path, errors, f"{where} action_source_coverage.csv"):
        return {"status": "missing"}
    try:
        cov = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        cov = pd.read_csv(path, encoding="utf-8")
    except Exception as exc:
        errors.append(f"cannot read {where} action_source_coverage.csv: {exc!r}")
        return {"status": "unreadable"}
    required = [c.replace("action__", "") for c in action_cols(_root_from_final_path(path))]
    if "action_dim" not in cov.columns:
        errors.append(f"{where} action_source_coverage.csv missing action_dim column")
        return {"status": "invalid"}
    dims = set(cov["action_dim"].astype(str))
    missing_rows = sorted(set(required) - dims)
    if missing_rows:
        errors.append(f"{where} action coverage missing dimensions: {missing_rows}")
    bad = []
    for _, r in cov.iterrows():
        dim = str(r.get("action_dim", ""))
        if dim not in required:
            continue
        src = r.get("source_column", None)
        src_missing = pd.isna(src) or str(src).strip() == ""
        rate = pd.to_numeric(pd.Series([r.get("observed_rate", None)]), errors="coerce").iloc[0]
        dim_threshold = max(float(min_observed_rate), float(ACTION_MIN_OBSERVED_RATE.get(dim, min_observed_rate)))
        if src_missing or pd.isna(rate) or float(rate) < dim_threshold:
            bad.append({
                "action_dim": dim,
                "source_column": None if src_missing else str(src),
                "observed_rate": None if pd.isna(rate) else float(rate),
                "min_required_observed_rate": dim_threshold,
            })
    if bad:
        errors.append(
            f"{where} has insufficient direct raw pseudo-action source coverage; proxy fallback is forbidden. "
            + json.dumps(bad, ensure_ascii=False)
        )
    low_warning = []
    if "observed_rate" in cov.columns:
        for _, r in cov.iterrows():
            dim = str(r.get("action_dim", ""))
            if dim not in required:
                continue
            rate = pd.to_numeric(pd.Series([r.get("observed_rate", None)]), errors="coerce").iloc[0]
            if pd.notna(rate) and float(rate) < 0.50:
                low_warning.append({"action_dim": dim, "observed_rate": float(rate), "warning_reference_rate": 0.50})
    if low_warning:
        warnings.append(f"{where} direct raw source coverage below 50% warning reference for structurally sparse dimensions: " + json.dumps(low_warning, ensure_ascii=False))
    return {
        "status": "PASS" if not bad and not missing_rows else "FAIL",
        "global_min_observed_rate": min_observed_rate,
        "action_min_observed_rate": ACTION_MIN_OBSERVED_RATE,
        "warning_reference_observed_rate": 0.50,
        "rows": int(len(cov)),
        "bad_dimensions": bad,
        "low_warning_dimensions": low_warning,
    }


def check_operating_cf_degeneracy(df: pd.DataFrame, *, require_non_degenerate: bool, errors: list[str], warnings: list[str], where: str) -> dict[str, Any]:
    if "sim__operating_cf" not in df.columns:
        msg = f"{where} missing sim__operating_cf for cash-flow degeneracy check"
        (errors if require_non_degenerate else warnings).append(msg)
        return {"status": "MISSING", "nonzero_rate": 0.0}
    x = pd.to_numeric(df["sim__operating_cf"], errors="coerce").fillna(0.0)
    nonzero_rate = float((x.abs() > 1e-12).mean()) if len(x) else 0.0
    status = "PASS" if nonzero_rate > 0.05 else "DEGENERATE"
    if status != "PASS":
        msg = f"{where} sim__operating_cf degenerate: nonzero_rate={nonzero_rate:.4f}"
        (errors if require_non_degenerate else warnings).append(msg)
    return {"status": status, "nonzero_rate": nonzero_rate, "rows": int(len(df))}


def verify_stage2_input(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    d = final / "stage2_candidate_projection" / "input_splits"
    rows = {}
    for name in ["phase1_pretrain", "phase2_bc", "phase3_iql", "phase_eval"]:
        df = read_parquet(d / f"{name}.parquet", errors, name)
        rows[name] = int(len(df))
        if not df.empty:
            if name == "phase_eval":
                check_cols(df, ["firm_id", "fiscal_year"] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS, errors, name)
                check_stage6_serving_state_columns(df, errors, name)
                forbidden = [c for c in df.columns if str(c).startswith(("action__", "action_observed__", "next__", "soft_cand_")) or c in REWARD_COLS or c in {"candidate_id", "projection_distance", "done"}]
                if forbidden:
                    errors.append(f"phase_eval must be state-only; forbidden columns present: {forbidden[:20]}")
            else:
                check_cols(df, ["firm_id", "fiscal_year"] + ACTION_COLS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS, errors, name)
                if name == "phase1_pretrain":
                    check_cols(df, [f"next__{c}" for c in ACD_TARGET_COLUMNS], errors, name)
            check_unique(df, ["firm_id", "fiscal_year"], errors, name)
            clipped = [c for c in df.columns if str(c).startswith("clipped_")]
            if clipped:
                errors.append(f"{name} contains forbidden clipped_* columns: {clipped[:20]}")
    meta = metadata_status(d / "metadata.json", errors)
    cf_join = meta.get("cash_flow_substrate_join", {}) if isinstance(meta, dict) else {}
    require_ocf = bool(cf_join.get("cash_flow_substrate_joined", False))
    ocf_checks = {}
    for _phase in ["phase2_bc", "phase3_iql"]:
        _df = read_parquet(d / f"{_phase}.parquet", errors, f"{_phase} OCF check")
        if not _df.empty:
            ocf_checks[_phase] = check_operating_cf_degeneracy(_df, require_non_degenerate=require_ocf, errors=errors, warnings=warnings, where=f"Stage2 input {_phase}")
    coverage_check = check_action_source_coverage(d / "action_source_coverage.csv", errors, warnings, "Stage2 input splits")
    if nonempty(d / "transition_gap_diagnostics.json", errors):
        try:
            _diag = read_json(d / "transition_gap_diagnostics.json")
            _avs = _diag.get("avs256_enrichment", {}) if isinstance(_diag, dict) else {}
            for _name in ("broad", "rated"):
                _meta = _avs.get(_name, {}) if isinstance(_avs, dict) else {}
                _status = _meta.get("transition_proximity_status")
                if _status != "computed":
                    errors.append(f"Stage2 input transition_proximity_status for {_name} must be computed, got {_status!r}")
        except Exception as exc:
            errors.append(f"cannot inspect transition_gap_diagnostics.json transition proximity status: {exc!r}")
    for _f in ["transition_proximity_metadata.json", "transition_proximity_prototypes.parquet"]:
        nonempty(d / _f, errors, _f)
    if (d / "transition_proximity_metadata.json").exists():
        try:
            _tm = read_json(d / "transition_proximity_metadata.json")
            if _tm.get("status") != "PASS":
                errors.append(f"Stage2 transition_proximity_metadata status must be PASS, got {_tm.get('status')!r}")
        except Exception as exc:
            errors.append(f"cannot inspect transition_proximity_metadata.json: {exc!r}")
    return {"rows": rows, "metadata_status": meta.get("status"), "action_source_coverage": coverage_check, "operating_cf_degeneracy": ocf_checks}



def check_recalibrated_candidate_library_uniqueness(path: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing recalibrated candidate library: {path.name}")
        return {"exists": False}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        fixed = data.get("fixed_candidates") or {}
        seen: dict[tuple[float, ...], str] = {}
        duplicates = []
        for cid, vec in fixed.items():
            key = tuple(round(float((vec or {}).get(c, 0.0) or 0.0), 12) for c in ACTION_COLS)
            if key in seen:
                duplicates.append((seen[key], cid))
            else:
                seen[key] = str(cid)
        if duplicates:
            errors.append(f"{path.name}: duplicate fixed candidate action vectors after recalibration: {duplicates}")
        meta = data.get("magnitude_recalibration") or {}
        if meta.get("method") != "tier_preserving_per_dimension_inner_train_abs_action_quantile":
            warnings.append(f"{path.name}: unexpected recalibration method {meta.get('method')!r}")
        return {"exists": True, "fixed_candidate_count": len(fixed), "unique_action_vector_count": len(seen), "duplicate_count": len(duplicates)}
    except Exception as exc:
        errors.append(f"cannot inspect recalibrated candidate library {path.name}: {exc!r}")
        return {"exists": True, "error": repr(exc)}

def check_stage2_aux_reward_contract(d: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    meta_path = d / "metadata.json"
    if not meta_path.exists():
        return {"status": "metadata_missing"}
    meta = read_json(meta_path)
    aux = meta.get("aux_reward_stats", {}) if isinstance(meta, dict) else {}
    m_enabled = bool(aux.get("merton_aux_enabled"))
    f_enabled = bool(aux.get("fcff_aux_enabled"))
    l_enabled = bool(aux.get("liquidity_aux_enabled"))
    enabled = bool(m_enabled or f_enabled or l_enabled)
    out = {
        "enabled": enabled,
        "merton_aux_enabled": m_enabled,
        "fcff_aux_enabled": f_enabled,
        "liquidity_aux_enabled": l_enabled,
        "lambda_merton": aux.get("lambda_merton"),
        "lambda_fcff": aux.get("lambda_fcff"),
        "lambda_liquidity": aux.get("lambda_liquidity"),
    }
    if not enabled:
        return out
    if aux.get("reference_oracle_scores_used") is not False or aux.get("reference_oracle_variables_used") is not False or aux.get("r_code_fallback_allowed") is not False:
        errors.append("Stage2 Merton/FCFF/liquidity aux metadata must state oracle_scores_used=false, oracle_variables_used=false, r_code_fallback_allowed=false")
    df = read_parquet(d / "phase3_iql_candidate.parquet", errors, "phase3_iql_candidate aux reward contract")
    if df.empty:
        return out
    required_by_component: list[tuple[str, list[str]]] = []
    if m_enabled:
        required_by_component.append(("Merton", MERTON_AUX_REWARD_COLS))
    if f_enabled:
        required_by_component.append(("FCFF", FCFF_AUX_REWARD_COLS))
    if l_enabled:
        required_by_component.append(("liquidity", LIQUIDITY_AUX_REWARD_COLS))
    for label, cols in required_by_component:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            errors.append(f"phase3_iql_candidate missing enabled {label} auxiliary reward columns: {missing}")
    for c in [x for _, cols in required_by_component for x in cols if x in df.columns]:
        if str(c).startswith("R") or str(c).startswith("next__R"):
            errors.append(f"Stage2 aux reward column path contains forbidden R-code column label: {c}")
    lambda_checks = [
        ("lambda_merton", "Merton"),
        ("lambda_fcff", "FCFF"),
        ("lambda_liquidity", "liquidity"),
    ]
    for col, label in lambda_checks:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist()
            if len(vals) != 1 or abs(float(vals[0]) - float(aux.get(col, 0.0))) > 1e-9:
                errors.append(f"phase3_iql_candidate {col} does not match metadata for {label}: values={vals}, metadata={aux.get(col)}")
        elif abs(float(aux.get(col, 0.0) or 0.0)) > 1e-12:
            errors.append(f"phase3_iql_candidate missing {col} despite nonzero metadata value {aux.get(col)}")
    return out


LOOPB2_EXPECTED_OBSERVED_TRANSITION_ROWS = 3159
LOOPB2_EXPECTED_ALL_MOVER_ROWS = 567
LOOPB2_OOT_YEAR_MIN = 2020
LOOPB2_OOT_YEAR_MAX = 2023
LOOPB2_MAX_B1_MINUS_B2_GAP_PP = 10.0
LOOPB2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT = 0.50
LOOPB2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS = 0.10
LOOPB2_MIN_R157_R182_NONNULL_SHARE = 0.50
LOOPB2_STAGE_NAME = "stage2_substrate_loopA_loopB2"
LOOPB2_REQUIRED_REPORT_TOP_KEYS = [
    "status",
    "substrate_tier",
    "research_interpretation_status",
    "loopA_contract_gate",
    "simulator_financial_fidelity_gate",
    "loopB2_observed_action_source_contract",
    "loopB2_prev_state_actual_financial_source_merge_for_loopA_driver",
    "loopB2",
    "loopB2_replay_diagnostic",
    "outputs",
    "primary_output_dir",
]
LOOPB2_REQUIRED_REPORT_FIELDS = [
    "status",
    "rule_status",
    "validation_type",
    "policy",
    "action_source",
    "action_source_policy",
    "agreement",
    "ci95",
    "n_movers",
    "b1_ref",
    "gap_pp",
    "threshold_max_b1_minus_b2_gap_pp",
    "threshold_min_b2_agreement_pct",
    "comparison_rule",
    "score_t_loader",
    "observed_transition_filter",
    "movers_definition",
    "oot_basis",
]
LOOPB2_REQUIRED_CSV_COLUMNS = [
    "firm_id",
    "fiscal_year",
    "prev_fiscal_year",
    "pred_alpha_score_t_from_observed_action_sim_tminus1_to_t",
    "score_tminus1_real",
    "delta_sim_alpha_score_tminus1_to_t",
    "real_rating_delta_t_to_tplus1",
    "is_observed_transition",
    "sim_input__action__ppe_pct",
    "sim_input__action__inv_turnover_chg",
    "sim_input__action__ar_turnover_chg",
    "sim_input__action__ap_turnover_chg",
    "sim_input__action__short_debt_pct",
    "sim_input__action__long_debt_pct",
    "sim_input__action__bond_pct",
    "sim_input__action__revenue_growth",
    "sim_input__action__cogs_ratio_chg",
    "sim_input__action__sga_ratio_chg",
    "sim_input__action_observed__ppe_pct",
    "sim_input__action_observed__inv_turnover_chg",
    "sim_input__action_observed__ar_turnover_chg",
    "sim_input__action_observed__ap_turnover_chg",
    "sim_input__action_observed__short_debt_pct",
    "sim_input__action_observed__long_debt_pct",
    "sim_input__action_observed__bond_pct",
    "sim_input__action_observed__revenue_growth",
    "sim_input__action_observed__cogs_ratio_chg",
    "sim_input__action_observed__sga_ratio_chg",
    "loopB2_real_mover",
    "loopB2_direction_match",
    "loopB2_oot_2020_2023",
]
LOOPB2_REPLAY_DIAGNOSTIC_CSV_COLUMNS = [
    "firm_id",
    "fiscal_year",
    "pred_alpha_score_tplus1",
    "score_t_real",
    "delta_sim_alpha_score_t_to_pred_tplus1",
    "real_rating_delta_t_to_tplus1",
    "is_observed_transition",
    "loopB2_real_mover",
    "loopB2_direction_match",
    "loopB2_oot_2020_2023",
]
LOOPB2_SIMULATOR_VALIDATION_TYPE = "test3_simulator_score_retention_observed_action_lead_convention"
LOOPB2_TRUE_SIMULATOR_ACTION_SOURCE = "C_obs_realized_action_values_from_stage2_raw_action_source_panel"
LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX = "action__"
LOOPB2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX = "action_observed__"
LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_AUDIT_PREFIX = "sim_input__action__"
LOOPB2_TRUE_SIMULATOR_ACTION_FLAG_AUDIT_PREFIX = "sim_input__action_observed__"
# Backward-compatible constant name: this is the observation flag prefix, not a value prefix.
LOOPB2_TRUE_SIMULATOR_ACTION_PREFIX = LOOPB2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX
LOOPB2_TRUE_SIMULATOR_ACTION_AUDIT_PREFIX = LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_AUDIT_PREFIX
LOOPB2_TRUE_SIMULATOR_PREV_STATE_SOURCE_ROLE = "stage1_actual_firm_year_financial_panel"
LOOPB2_TRUE_SIMULATOR_PREV_STATE_POLICY_ANCHOR = "Stage1 actual firm-year financial panel"
LOOPB2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE = 0.50
LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_SOURCE_LOADER_ANCHOR = "_load_loopb2_observed_action_value_source"
LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_SOURCE_MERGE_ANCHOR = "_merge_loopb2_observed_action_values_from_source"

LOOPB2_TRUE_SIMULATOR_ACTION_DIMS = [
    "ppe_pct",
    "inv_turnover_chg",
    "ar_turnover_chg",
    "ap_turnover_chg",
    "short_debt_pct",
    "long_debt_pct",
    "bond_pct",
    "revenue_growth",
    "cogs_ratio_chg",
    "sga_ratio_chg",
]
LOOPB2_REPLAY_DIAGNOSTIC_TYPE = "observed_next_replay_alignment_not_simulator_validation"
LOOPB2_STAGE2_DIAGNOSTIC_TIER = "oracle_validated_simulator_diagnostics_reported"
LOOPB2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS = "ORACLE_VALIDATED_SIMULATOR_DIAGNOSTICS_REPORTED"
LOOPB2_CURRENT_RESEARCH_GATE_POLICY_ANCHOR = "Current 검사3제거 research contract"
LOOPB2_DIAGNOSTIC_RULE_ROLE = "configured_minimum_agreement_50pct_diagnostic_after_검사3제거"
LOOPB2_PRIMARY_MISSING_SUSPECTS = ["R157", "R182"]
LOOPB2_SELECTED_FORMULA_ID_PRESERVATION_GUARD_IDS = ["R157", "R182"]
LOOPB2_SCORING_POPULATION_POLICY_ANCHOR = "observed-transition B2 sample before alpha scoring"
LOOPB2_DIRECT_NEXT_RATIO_OVERLAY_POLICY_ANCHOR = "observed-transition next__R-code/R-code__next"


def _contract_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return out


def _contract_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).ne(0)
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def _read_csv_contract(path: Path, errors: list[str], label: str) -> pd.DataFrame:
    if not nonempty(path, errors, label):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, encoding="utf-8")
        except Exception as exc:
            errors.append(f"cannot read CSV {label}: {exc!r}")
            return pd.DataFrame()
    except Exception as exc:
        errors.append(f"cannot read CSV {label}: {exc!r}")
        return pd.DataFrame()


def _require_report_fields(doc: dict[str, Any], fields: list[str], errors: list[str], where: str) -> None:
    missing = [f for f in fields if f not in doc]
    if missing:
        errors.append(f"{where} missing required fields: {missing}")


def _check_loopb2_report_contract(meta: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    _require_report_fields(meta, LOOPB2_REQUIRED_REPORT_TOP_KEYS, errors, "Loop B2 report top level")
    loopa = meta.get("loopA_contract_gate") or {}
    fidelity = meta.get("simulator_financial_fidelity_gate") or {}
    b2 = meta.get("loopB2") or {}
    replay = meta.get("loopB2_replay_diagnostic") or {}
    if not isinstance(loopa, dict):
        errors.append("Loop B2 report loopA_contract_gate must be an object")
        loopa = {}
    if not isinstance(fidelity, dict):
        errors.append("Loop B2 report simulator_financial_fidelity_gate must be an object")
        fidelity = {}
    if not isinstance(b2, dict):
        errors.append("Loop B2 report loopB2 must be an object")
        b2 = {}
    if not isinstance(replay, dict):
        errors.append("Loop B2 report loopB2_replay_diagnostic must be an object")
        replay = {}

    _require_report_fields(b2, LOOPB2_REQUIRED_REPORT_FIELDS, errors, "Loop B2 report loopB2")

    action_contract = meta.get("loopB2_observed_action_source_contract") or {}
    if not isinstance(action_contract, dict):
        errors.append("Loop B2 report loopB2_observed_action_source_contract must be an object")
        action_contract = {}
    elif action_contract.get("status") != "PASS":
        errors.append(f"Loop B2 observed action source contract must be PASS, got {action_contract.get('status')!r}")
    if action_contract.get("source") != LOOPB2_TRUE_SIMULATOR_ACTION_SOURCE:
        errors.append(f"Loop B2 observed action source contract source mismatch: {action_contract.get('source')!r}")
    if action_contract.get("value_prefix") != LOOPB2_TRUE_SIMULATOR_ACTION_VALUE_PREFIX:
        errors.append(f"Loop B2 observed action source value_prefix mismatch: {action_contract.get('value_prefix')!r}")
    if action_contract.get("observed_flag_prefix") != LOOPB2_TRUE_SIMULATOR_ACTION_FLAG_PREFIX:
        errors.append(f"Loop B2 observed action source observed_flag_prefix mismatch: {action_contract.get('observed_flag_prefix')!r}")
    coverage = action_contract.get("coverage") if isinstance(action_contract, dict) else {}
    if not isinstance(coverage, dict):
        errors.append("Loop B2 observed action source coverage must be an object")
        coverage = {}
    for dim in LOOPB2_TRUE_SIMULATOR_ACTION_DIMS:
        detail = coverage.get(dim)
        share = _contract_float((detail or {}).get("finite_share")) if isinstance(detail, dict) else None
        if share is None or abs(share - 1.0) > 1e-12:
            errors.append(f"Loop B2 observed action source finite_share for {dim} must equal 1.0, got {share}")
        flag_share = _contract_float((detail or {}).get("observed_flag_true_share")) if isinstance(detail, dict) else None
        if flag_share is None or flag_share <= 0.0:
            errors.append(f"Loop B2 observed action source observed_flag_true_share for {dim} must be positive, got {flag_share}")

    prev_source = meta.get("loopB2_prev_state_actual_financial_source_merge_for_loopA_driver") or {}
    if not isinstance(prev_source, dict):
        errors.append("Loop B2 true simulator prev_state actual financial source merge must be an object")
        prev_source = {}
    else:
        if prev_source.get("status") != "PASS":
            errors.append(f"Loop B2 true simulator prev_state actual financial source merge must be PASS, got {prev_source.get('status')!r}")
        if prev_source.get("source_role") != LOOPB2_TRUE_SIMULATOR_PREV_STATE_SOURCE_ROLE:
            errors.append(
                "Loop B2 true simulator prev_state source must be the Stage1 actual firm-year financial panel, "
                f"got source_role={prev_source.get('source_role')!r}"
            )
        if prev_source.get("require_nonzero_source") is not True:
            errors.append("Loop B2 true simulator prev_state source must require non-zero denominator coverage")
        source_policy = str(prev_source.get("source_lookup_policy") or "")
        if "Stage1" not in source_policy and "actual" not in source_policy:
            errors.append(f"Loop B2 true simulator prev_state source policy must mention actual Stage1 panel, got {source_policy!r}")
        actual_contract = prev_source.get("actual_financial_source_contract") if isinstance(prev_source, dict) else {}
        if not isinstance(actual_contract, dict) or actual_contract.get("status") != "PASS":
            errors.append("Loop B2 true simulator actual financial source contract must be present and PASS")
            actual_contract = {}
        prepared_source = actual_contract.get("prepared_source") if isinstance(actual_contract, dict) else {}
        if not isinstance(prepared_source, dict):
            errors.append("Loop B2 true simulator actual financial source must record prepared_source metadata")
            prepared_source = {}
        source_format = str(prepared_source.get("source_format") or "")
        if source_format not in {"wide_actual_financial_panel", "long_statement_item_panel_pivoted_to_wide"}:
            errors.append(
                "Loop B2 true simulator actual financial source must be a cleaned wide statement panel or "
                f"a pivoted long statement item panel, got source_format={source_format!r}"
            )
        source_path = str(actual_contract.get("source") or prepared_source.get("source_path") or "")
        if not any(token in source_path for token in ("cleaned_statement_panels", "financial_statement_items_raw", "statement_items_panel")):
            errors.append(
                "Loop B2 true simulator actual financial source should come from Stage00-01 cleaned/raw statement items, "
                f"got source={source_path!r}"
            )
        source_cov = prev_source.get("source_ucode_coverage") if isinstance(prev_source, dict) else {}
        if isinstance(source_cov, dict):
            for field in ("non_current_assets", "capital_stock"):
                detail = source_cov.get(field) or {}
                nonzero_share = _contract_float(detail.get("nonzero_share")) if isinstance(detail, dict) else None
                if nonzero_share is None or nonzero_share <= LOOPB2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE:
                    errors.append(
                        f"Loop B2 true simulator prev_state source nonzero_share for {field} must exceed "
                        f"{LOOPB2_TRUE_SIMULATOR_MIN_DENOMINATOR_NONZERO_SHARE}, got {nonzero_share}"
                    )

    top_status = str(meta.get("status") or "")
    tier = str(meta.get("substrate_tier") or "")
    interpretation = str(meta.get("research_interpretation_status") or "")
    loopa_status = str(loopa.get("status") or "")
    fidelity_status = str(fidelity.get("status") or "")
    b2_rule_status = str(b2.get("rule_status") or "")
    b2_status = str(b2.get("status") or "")

    if top_status != "PASS":
        errors.append(f"Current 검사3제거 contract expects Loop B2 stage report top-level status PASS after usable diagnostics, got {top_status!r}")
    if tier != LOOPB2_STAGE2_DIAGNOSTIC_TIER:
        errors.append(
            f"Current 검사3제거 contract expects substrate_tier={LOOPB2_STAGE2_DIAGNOSTIC_TIER!r}, got {tier!r}"
        )
    if interpretation != LOOPB2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS:
        errors.append(
            f"Current 검사3제거 contract expects research_interpretation_status="
            f"{LOOPB2_STAGE2_DIAGNOSTIC_INTERPRETATION_STATUS!r}, got {interpretation!r}"
        )
    policy_anchor = str(meta.get("current_research_gate_policy") or "")
    if LOOPB2_CURRENT_RESEARCH_GATE_POLICY_ANCHOR not in policy_anchor:
        errors.append("Loop B2 report must record the current 검사3제거 research-gate policy anchor")
    if loopa_status != "PASS":
        errors.append(f"Loop B2 verifier requires loopA_contract_gate.status=PASS, got {loopa_status!r}")
    if b2_rule_status not in {"PASS", "FAIL"}:
        errors.append(f"Loop B2 diagnostic rule_status must be PASS or FAIL, got {b2_rule_status!r}")
    if b2.get("validation_type") != LOOPB2_SIMULATOR_VALIDATION_TYPE:
        errors.append(
            "Loop B2 report loopB2 must be the true simulator score-retention Test3 gate, "
            f"got validation_type={b2.get('validation_type')!r}"
        )
    if b2.get("action_source") != LOOPB2_TRUE_SIMULATOR_ACTION_SOURCE:
        errors.append(
            "Loop B2 true simulator validation must report C_obs action values from the Stage2A raw action source panel, "
            f"got action_source={b2.get('action_source')!r}"
        )
    action_policy = str(b2.get("action_source_policy") or "")
    if "action__*" not in action_policy or "action_observed__*" not in action_policy or "masks" not in action_policy:
        errors.append(f"Loop B2 action_source_policy must state that action__* are values and action_observed__* are masks, got {action_policy!r}")
    if b2.get("observed_next_ratio_overlay_allowed") is not False:
        errors.append(
            "Loop B2 true simulator validation must not allow observed next__R-code ratio overlays; "
            f"got observed_next_ratio_overlay_allowed={b2.get('observed_next_ratio_overlay_allowed')!r}"
        )
    if replay:
        if replay.get("validation_type") != LOOPB2_REPLAY_DIAGNOSTIC_TYPE:
            errors.append(f"Loop B2 replay diagnostic validation_type mismatch: {replay.get('validation_type')!r}")
        if replay.get("does_not_determine_status") is not True:
            errors.append("Loop B2 observed-next replay diagnostic must not determine substrate_tier")
    if fidelity.get("does_not_determine_status") is not True:
        errors.append("simulator_financial_fidelity_gate must be diagnostic-only and does_not_determine_status=true")
    if fidelity.get("determines_strong_pass") is not False:
        errors.append("simulator_financial_fidelity_gate must not determine strong_pass under the current 검사3제거 contract")
    max_rel_err_assets = _contract_float(fidelity.get("max_rel_err_assets"))
    if max_rel_err_assets is None or abs(max_rel_err_assets - LOOPB2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS) > 1e-9:
        errors.append(
            f"simulator_financial_fidelity_gate.max_rel_err_assets must use the configured 10% diagnostic tolerance "
            f"({LOOPB2_SIMULATOR_FIDELITY_MAX_REL_ERR_ASSETS}), got {max_rel_err_assets!r}"
        )
    if b2.get("does_not_determine_status") is not True:
        errors.append("Loop B2 simulator score-retention comparison must be diagnostic-only and does_not_determine_status=true")
    if b2.get("rule_role") != LOOPB2_DIAGNOSTIC_RULE_ROLE:
        errors.append(
            f"Loop B2 rule_role must be {LOOPB2_DIAGNOSTIC_RULE_ROLE!r}, got {b2.get('rule_role')!r}"
        )
    if b2_status not in {"PASS", "FAIL_B2_MIN_AGREEMENT"}:
        errors.append(f"Loop B2 status must follow the configured 50% agreement diagnostic rule, got {b2_status!r}")

    threshold_pp = _contract_float(b2.get("threshold_max_b1_minus_b2_gap_pp"))
    if threshold_pp is None or abs(threshold_pp - LOOPB2_MAX_B1_MINUS_B2_GAP_PP) > 1e-9:
        errors.append(f"Loop B2 legacy gap diagnostic threshold must remain 10 percentage points, got {threshold_pp!r}")
    min_agreement_pct = _contract_float(b2.get("threshold_min_b2_agreement_pct"))
    if min_agreement_pct is None or abs(min_agreement_pct - LOOPB2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT * 100.0) > 1e-9:
        errors.append(
            f"Loop B2 configured minimum agreement threshold must be "
            f"{LOOPB2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT * 100.0:.1f}%, got {min_agreement_pct!r}"
        )
    comparison_rule = str(b2.get("comparison_rule") or "")
    if "50%" not in comparison_rule or "B1-vs-B2" not in comparison_rule or "diagnostic" not in comparison_rule.lower():
        errors.append(f"Loop B2 comparison_rule must state the configured 50% diagnostic rule and legacy gap diagnostic: {comparison_rule!r}")
    b2_policy_anchor = str(b2.get("current_research_gate_policy") or "")
    if LOOPB2_CURRENT_RESEARCH_GATE_POLICY_ANCHOR not in b2_policy_anchor:
        errors.append("Loop B2 metadata must carry the current 검사3제거 diagnostic-only policy anchor")
    if b2.get("score_t_loader") != "verify_stage1_substrate_validation.load_stage1_backend_score_panel":
        errors.append(f"Loop B2 score_t_loader must name the shared Stage1 B1 loader, got {b2.get('score_t_loader')!r}")
    obs_filter = str(b2.get("observed_transition_filter") or "")
    if "consecutive observed transitions" not in obs_filter and "is_observed_transition" not in obs_filter:
        errors.append(f"Loop B2 observed transition filter/time alignment mismatch: {obs_filter!r}")
    if b2.get("movers_definition") != "real_rating_delta_t_to_tplus1 != 0":
        errors.append(f"Loop B2 mover definition mismatch: {b2.get('movers_definition')!r}")
    oot_basis = str(b2.get("oot_basis") or "")
    if str(LOOPB2_OOT_YEAR_MIN) not in oot_basis or str(LOOPB2_OOT_YEAR_MAX) not in oot_basis:
        errors.append(f"Loop B2 OOT basis must contain {LOOPB2_OOT_YEAR_MIN}-{LOOPB2_OOT_YEAR_MAX}, got {oot_basis!r}")

    b1_ref = b2.get("b1_ref") or {}
    if not isinstance(b1_ref, dict):
        errors.append("Loop B2 b1_ref must be an object")
        b1_ref = {}
    if b1_ref.get("backend") not in {"alpha", None}:
        errors.append(f"Loop B2 b1_ref backend must be alpha, got {b1_ref.get('backend')!r}")
    b1_agree = _contract_float(b1_ref.get("agreement"))
    b2_agree = _contract_float(b2.get("agreement"))
    gap_pp = _contract_float(b2.get("gap_pp"))
    if b1_agree is None or b2_agree is None or gap_pp is None:
        errors.append("Loop B2 report must contain numeric b1_ref.agreement, loopB2.agreement, and gap_pp")
    else:
        expected_gap_pp = (b1_agree - b2_agree) * 100.0
        if abs(expected_gap_pp - gap_pp) > 1e-6:
            errors.append(f"Loop B2 gap_pp mismatch: expected={expected_gap_pp}, found={gap_pp}")
        expected_status = "PASS" if b2_agree >= LOOPB2_MIN_SIMULATOR_SCORE_RETENTION_AGREEMENT else "FAIL"
        if b2_rule_status != expected_status:
            errors.append(
                f"Loop B2 rule_status {b2_rule_status!r} contradicts configured 50% agreement rule: "
                f"agreement={b2_agree}"
            )
        legacy_status = str(b2.get("legacy_b1_minus_b2_gap_rule_status") or "")
        if legacy_status:
            expected_legacy = "PASS" if gap_pp <= LOOPB2_MAX_B1_MINUS_B2_GAP_PP + 1e-9 else "FAIL"
            if legacy_status != expected_legacy:
                errors.append(
                    f"Loop B2 legacy_b1_minus_b2_gap_rule_status {legacy_status!r} contradicts gap_pp={gap_pp}"
                )

    ci95 = b2.get("ci95")
    if not isinstance(ci95, list) or len(ci95) != 2:
        errors.append(f"Loop B2 ci95 must be a two-element list, got {ci95!r}")
    else:
        lo = _contract_float(ci95[0])
        hi = _contract_float(ci95[1])
        if lo is None or hi is None or lo > hi:
            errors.append(f"Loop B2 ci95 is invalid: {ci95!r}")
        elif b2_agree is not None and not (lo - 1e-12 <= b2_agree <= hi + 1e-12):
            warnings.append(f"Loop B2 agreement is outside reported ci95: agreement={b2_agree}, ci95={ci95}")

    alpha_meta = b2.get("alpha_scoring_frame") or {}
    if not isinstance(alpha_meta, dict):
        errors.append("Loop B2 alpha_scoring_frame must be an object")
        alpha_meta = {}
    loader_guard = alpha_meta.get("prev_state_ucode_loader_contract") or {}
    if not isinstance(loader_guard, dict) or loader_guard.get("status") != "PASS":
        errors.append("Loop B2 alpha_scoring_frame.prev_state_ucode_loader_contract must be present and PASS")
    merge_guard = alpha_meta.get("prev_state_handoff_merge") or {}
    if not isinstance(merge_guard, dict) or merge_guard.get("status") != "PASS":
        errors.append("Loop B2 alpha_scoring_frame.prev_state_handoff_merge must be present and PASS")
    else:
        lookup_policy = str(merge_guard.get("source_lookup_policy") or "")
        if "U-code" not in lookup_policy and "ucode" not in lookup_policy.lower():
            errors.append(f"Loop B2 prev_state source_lookup_policy must require base-year U-code columns, got {lookup_policy!r}")
        join_policy = str(merge_guard.get("join_key_policy") or "")
        if "normalised_loopb2_firm_key" not in join_policy:
            errors.append(f"Loop B2 prev_state join_key_policy must use normalised_loopb2_firm_key, got {join_policy!r}")
        matched_rows = _contract_float(merge_guard.get("matched_rows_any_required_field"))
        merge_rows = _contract_float(merge_guard.get("rows"))
        matched_share = _contract_float(merge_guard.get("matched_share_any_required_field"))
        if matched_share is None and matched_rows is not None and merge_rows not in (None, 0):
            matched_share = matched_rows / merge_rows
        if matched_share is None or matched_share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
            errors.append(
                f"Loop B2 prev_state handoff merge matched share must exceed "
                f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {matched_share}"
            )
        forced_cols = merge_guard.get("forced_firm_state_field_columns") or {}
        if not isinstance(forced_cols, dict):
            errors.append("Loop B2 prev_state forced_firm_state_field_columns must be an object")
            forced_cols = {}
        for field in ["non_current_assets", "capital_stock"]:
            if forced_cols.get(field) != field:
                errors.append(
                    "Loop B2 prev_state merge must force audited U-code values into the bare FirmState "
                    f"field column {field!r}; got {forced_cols.get(field)!r}"
                )
        source_ucode = merge_guard.get("source_ucode_coverage") or {}
        if not isinstance(source_ucode, dict):
            errors.append("Loop B2 prev_state source_ucode_coverage must be an object")
            source_ucode = {}
        for field in ["non_current_assets", "capital_stock"]:
            detail = source_ucode.get(field) if isinstance(source_ucode, dict) else None
            share = _contract_float((detail or {}).get("nonnull_share")) if isinstance(detail, dict) else None
            col = str((detail or {}).get("column") or "") if isinstance(detail, dict) else ""
            if share is None or share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
                errors.append(
                    f"Loop B2 prev_state U-code source coverage for {field} must exceed "
                    f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {share}"
                )
            if field == "non_current_assets" and "U01A110000000" not in col:
                errors.append(f"Loop B2 non_current_assets source must be U01A110000000, got {col!r}")
            if field == "capital_stock" and "U01A611000000" not in col:
                errors.append(f"Loop B2 capital_stock source must be U01A611000000, got {col!r}")
        after = merge_guard.get("after_coverage") or {}
        for field in ["non_current_assets", "capital_stock"]:
            detail = after.get(field) if isinstance(after, dict) else None
            share = _contract_float((detail or {}).get("nonnull_share")) if isinstance(detail, dict) else None
            if share is None or share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
                errors.append(
                    f"Loop B2 prev_state handoff merge coverage for {field} must exceed "
                    f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {share}"
                )
    lead_alignment = alpha_meta.get("lead_alignment") or {}
    scoring_population = alpha_meta.get("scoring_population") or {}
    if isinstance(lead_alignment, dict) and lead_alignment.get("status") == "PASS":
        aligned_rows = _contract_float(lead_alignment.get("aligned_lead_rows"))
        policy_text = str(lead_alignment.get("policy") or "")
        if aligned_rows is None or aligned_rows <= 0:
            errors.append(f"Loop B2 Test3 lead_alignment must contain positive aligned_lead_rows, got {aligned_rows}")
        if "Stage1 Loop B1" not in policy_text and "score-retention" not in policy_text:
            errors.append(f"Loop B2 Test3 lead_alignment policy must state B1 lead convention, got {policy_text!r}")
    elif isinstance(scoring_population, dict) and scoring_population.get("status") == "PASS":
        # Backward compatibility for replay diagnostic metadata; the true gate should normally use lead_alignment.
        scored_rows = _contract_float(scoring_population.get("scored_rows_after_observed_filter"))
        source_rows = _contract_float(scoring_population.get("source_rows_before_filter"))
        policy_text = str(scoring_population.get("policy") or "")
        leakage_guard = str(scoring_population.get("future_label_leakage_guard") or "")
        if scored_rows is None or int(scored_rows) != LOOPB2_EXPECTED_OBSERVED_TRANSITION_ROWS:
            errors.append(
                f"Loop B2 observed-only scoring population must have {LOOPB2_EXPECTED_OBSERVED_TRANSITION_ROWS} rows, "
                f"got {scored_rows}"
            )
        if source_rows is not None and scored_rows is not None and source_rows < scored_rows:
            errors.append(f"Loop B2 scoring population source rows cannot be less than scored rows: source={source_rows}, scored={scored_rows}")
        if "observed" not in policy_text.lower() or "before scoring" not in policy_text.lower():
            errors.append(f"Loop B2 scoring_population policy must state observed filtering before scoring, got {policy_text!r}")
        if "non-observed" not in leakage_guard.lower() and "counterfactual" not in leakage_guard.lower():
            errors.append(f"Loop B2 scoring_population future_label_leakage_guard must mention non-observed/counterfactual exclusion, got {leakage_guard!r}")
    else:
        errors.append("Loop B2 alpha_scoring_frame must contain either Test3 lead_alignment PASS or replay scoring_population PASS")

    state_field_coverage = alpha_meta.get("state_field_coverage") or {}
    if not isinstance(state_field_coverage, dict):
        errors.append("Loop B2 alpha_scoring_frame.state_field_coverage must be an object")
        state_field_coverage = {}
    for label in ["prev_state.capital_stock", "prev_state.non_current_assets", "state_t1.revenue", "state_t1.non_current_assets"]:
        detail = state_field_coverage.get(label) if isinstance(state_field_coverage, dict) else None
        share = _contract_float((detail or {}).get("nonnull_share")) if isinstance(detail, dict) else None
        if share is None or share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
            errors.append(
                f"Loop B2 alpha scoring state field coverage for {label} must exceed "
                f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {share}"
            )

    formula_preservation = alpha_meta.get("selected_formula_raw_id_preservation") or {}
    if not isinstance(formula_preservation, dict) or formula_preservation.get("status") != "PASS":
        errors.append("Loop B2 alpha_scoring_frame.selected_formula_raw_id_preservation must be present and PASS")
        formula_preservation = formula_preservation if isinstance(formula_preservation, dict) else {}
    policy = str(formula_preservation.get("policy") or "")
    if "Stage1-selected R-code ids" not in policy and "selected" not in policy.lower():
        errors.append(f"Loop B2 selected formula id preservation policy is not explicit: {policy!r}")
    raw_cov = formula_preservation.get("raw_formula_value_coverage") or {}
    if not isinstance(raw_cov, dict):
        errors.append("Loop B2 selected formula raw_formula_value_coverage must be an object")
        raw_cov = {}
    for col in LOOPB2_SELECTED_FORMULA_ID_PRESERVATION_GUARD_IDS:
        detail = raw_cov.get(col) if isinstance(raw_cov, dict) else None
        share = _contract_float((detail or {}).get("nonnull_share")) if isinstance(detail, dict) else None
        if share is None or share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
            errors.append(
                f"Loop B2 raw selected formula value coverage for {col} must exceed "
                f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {share}"
            )
    overlay_allowed = formula_preservation.get("direct_next_ratio_overlay_allowed")
    overlay_policy = str(formula_preservation.get("direct_next_ratio_overlay_policy") or "")
    overlay_cov = formula_preservation.get("direct_next_ratio_overlay_coverage") or {}
    if not isinstance(overlay_cov, dict):
        errors.append("Loop B2 direct_next_ratio_overlay_coverage must be an object")
        overlay_cov = {}
    if overlay_allowed is not False:
        errors.append(
            "Loop B2 true simulator alpha frame must disable observed next-ratio overlays; "
            f"direct_next_ratio_overlay_allowed={overlay_allowed!r}, policy={overlay_policy!r}"
        )
    if "forbidden" not in overlay_policy.lower() and "disabled" not in overlay_policy.lower():
        errors.append(f"Loop B2 true simulator overlay policy must state disabled/forbidden next-ratio overlay, got {overlay_policy!r}")
    for col in LOOPB2_SELECTED_FORMULA_ID_PRESERVATION_GUARD_IDS:
        detail = overlay_cov.get(col) if isinstance(overlay_cov, dict) else None
        if isinstance(detail, dict) and int(detail.get("filled_rows") or 0) != 0:
            errors.append(f"Loop B2 true simulator validation leaked observed next-ratio overlay for {col}: {detail}")

    component_cov = formula_preservation.get("formula_component_coverage") or {}
    if not isinstance(component_cov, dict):
        errors.append("Loop B2 selected formula formula_component_coverage must be an object")
        component_cov = {}
    for comp in [
        "R157.numerator.state_t1.revenue",
        "R157.denominator.avg_capital_stock",
        "R182.numerator.state_t1.non_current_assets",
        "R182.denominator.prev_state.non_current_assets",
    ]:
        detail = component_cov.get(comp) if isinstance(component_cov, dict) else None
        share = _contract_float((detail or {}).get("nonnull_share")) if isinstance(detail, dict) else None
        if share is None or share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
            errors.append(
                f"Loop B2 formula component coverage for {comp} must exceed "
                f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {share}"
            )

    scoring_guard = alpha_meta.get("r157_r182_nonnull_share_guard") or {}
    if not isinstance(scoring_guard, dict) or scoring_guard.get("status") != "PASS":
        errors.append("Loop B2 alpha_scoring_frame.r157_r182_nonnull_share_guard must be present and PASS")
    else:
        shares = scoring_guard.get("nonnull_share") or {}
        for col in LOOPB2_PRIMARY_MISSING_SUSPECTS:
            share = _contract_float(shares.get(col)) if isinstance(shares, dict) else None
            if share is None or share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
                errors.append(
                    f"Loop B2 alpha scoring guard for {col} must exceed "
                    f"{LOOPB2_MIN_R157_R182_NONNULL_SHARE}, got {share}"
                )

    outputs = meta.get("outputs") or []
    if not isinstance(outputs, list):
        errors.append("Loop B2 report outputs must be a list")
        outputs = []
    for required_output in [
        "substrate_loopA_loopB2_report.json",
        "loopB2_simulator_score_retention.csv",
        "loopB2_simulator_score_retention_scoring_input_frame.csv",
        "loopB2_simulator_score_retention_scoring_frame_meta_pre_guard.json",
    ]:
        if required_output not in outputs:
            errors.append(f"Loop B2 report outputs list missing {required_output}")

    return {
        "status": top_status,
        "substrate_tier": tier,
        "research_interpretation_status": interpretation,
        "loopA_contract_status": loopa_status,
        "simulator_financial_fidelity_status": fidelity_status,
        "simulator_financial_fidelity_role": "diagnostic_only",
        "loopB2_status": b2_status,
        "loopB2_rule_status": b2_rule_status,
        "loopB2_rule_role": b2.get("rule_role"),
        "agreement": b2_agree,
        "gap_pp": gap_pp,
        "b1_ref_agreement": b1_agree,
    }


def _check_loopb2_csv_contract(df: pd.DataFrame, report_meta: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if df.empty:
        return {"status": "MISSING_OR_EMPTY"}
    check_cols(df, LOOPB2_REQUIRED_CSV_COLUMNS, errors, "Loop B2 simulator score-retention CSV")
    missing_required = [c for c in LOOPB2_REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_required:
        return {"status": "FAIL", "rows": int(len(df)), "missing_columns": missing_required}

    action_audit_columns = [f"{LOOPB2_TRUE_SIMULATOR_ACTION_AUDIT_PREFIX}{dim}" for dim in LOOPB2_TRUE_SIMULATOR_ACTION_DIMS]
    for col in action_audit_columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.isna().any():
            errors.append(f"Loop B2 simulator score-retention CSV has non-finite C_obs simulator-input action column {col}")
    forbidden_generic_action_cols = [c for c in df.columns if str(c).startswith("action__")]
    if forbidden_generic_action_cols:
        errors.append(f"Loop B2 simulator score-retention CSV must not expose generic action__* as simulator input: {forbidden_generic_action_cols[:10]}")

    observed = _contract_bool_series(df["is_observed_transition"])
    oot = _contract_bool_series(df["loopB2_oot_2020_2023"])
    real_delta = pd.to_numeric(df["real_rating_delta_t_to_tplus1"], errors="coerce")
    pred = pd.to_numeric(df["pred_alpha_score_t_from_observed_action_sim_tminus1_to_t"], errors="coerce")
    score_prev = pd.to_numeric(df["score_tminus1_real"], errors="coerce")
    delta_sim = pd.to_numeric(df["delta_sim_alpha_score_tminus1_to_t"], errors="coerce")
    real_mover = _contract_bool_series(df["loopB2_real_mover"])
    direction_match = _contract_bool_series(df["loopB2_direction_match"])

    if not observed.all():
        errors.append("Loop B2 simulator score-retention CSV must contain only lead-aligned observed transition target rows")
    if score_prev.isna().all():
        errors.append("Loop B2 simulator score-retention CSV score_tminus1_real is entirely missing")

    expected_mover_col = real_delta.notna() & real_delta.ne(0)
    mover_n = int(expected_mover_col.sum())
    if mover_n <= 0:
        errors.append("Loop B2 simulator score-retention CSV has zero real rating movers")
    mover_mismatch = int((real_mover.ne(expected_mover_col)).sum())
    if mover_mismatch:
        errors.append(f"Loop B2 loopB2_real_mover column mismatch rows={mover_mismatch}")

    expected_delta = pred - score_prev
    comparable_delta = expected_delta.notna() & delta_sim.notna()
    if comparable_delta.any():
        max_abs_delta_err = float((expected_delta[comparable_delta] - delta_sim[comparable_delta]).abs().max())
        if max_abs_delta_err > 1e-9:
            errors.append(f"Loop B2 simulator score-retention delta column mismatch: max_abs_error={max_abs_delta_err}")
    if delta_sim.isna().all():
        errors.append("Loop B2 simulator score-retention delta_sim is entirely missing")

    sign_sim = delta_sim.gt(0).astype(int) - delta_sim.lt(0).astype(int)
    sign_real = real_delta.gt(0).astype(int) - real_delta.lt(0).astype(int)
    expected_match = expected_mover_col & sign_sim.eq(sign_real)
    match_mismatch = int((direction_match.ne(expected_match)).sum())
    if match_mismatch:
        errors.append(f"Loop B2 loopB2_direction_match column mismatch rows={match_mismatch}")

    if "transition_target_year" in df.columns:
        target_year = pd.to_numeric(df["transition_target_year"], errors="coerce")
        expected_oot = target_year.between(LOOPB2_OOT_YEAR_MIN, LOOPB2_OOT_YEAR_MAX, inclusive="both").fillna(False)
        oot_mismatch = int(oot.ne(expected_oot).sum())
        if oot_mismatch:
            errors.append(f"Loop B2 OOT flag mismatch rows={oot_mismatch}")
    else:
        warnings.append("Loop B2 scored CSV lacks transition_target_year; OOT flag cannot be independently recomputed")

    valid_rows = score_prev.notna() & real_delta.notna()
    oot_movers = valid_rows & expected_mover_col & oot
    oot_mover_n = int(oot_movers.sum())
    oot_match_n = int(direction_match[oot_movers].sum())
    csv_agreement = (float(oot_match_n) / float(oot_mover_n)) if oot_mover_n else None

    b2 = report_meta.get("loopB2") or {}
    report_all = b2.get("all") or {}
    report_oot = b2.get("oot") or {}
    report_valid = b2.get("valid_lead_aligned_rows")
    report_all_movers = report_all.get("n_movers")
    report_oot_movers = b2.get("n_movers")
    report_agreement = _contract_float(b2.get("agreement"))
    if report_valid is not None and int(report_valid) != int(valid_rows.sum()):
        errors.append(f"Loop B2 report valid_lead_aligned_rows mismatch: report={report_valid}, csv={int(valid_rows.sum())}")
    if report_all_movers is not None and int(report_all_movers) != int((valid_rows & expected_mover_col).sum()):
        errors.append(f"Loop B2 report all.n_movers mismatch: report={report_all_movers}, csv={int((valid_rows & expected_mover_col).sum())}")
    if report_oot_movers is not None and int(report_oot_movers) != oot_mover_n:
        errors.append(f"Loop B2 report n_movers/OOT mismatch: report={report_oot_movers}, csv={oot_mover_n}")
    if report_oot.get("n_movers") is not None and int(report_oot.get("n_movers")) != oot_mover_n:
        errors.append(f"Loop B2 report oot.n_movers mismatch: report={report_oot.get('n_movers')}, csv={oot_mover_n}")
    if csv_agreement is not None and report_agreement is not None and abs(csv_agreement - report_agreement) > 1e-12:
        errors.append(f"Loop B2 report agreement mismatch: report={report_agreement}, csv={csv_agreement}")

    return {
        "status": "PASS",
        "rows": int(len(df)),
        "valid_lead_aligned_rows": int(valid_rows.sum()),
        "all_lead_aligned_movers": int((valid_rows & expected_mover_col).sum()),
        "oot_movers": oot_mover_n,
        "oot_matches": oot_match_n,
        "oot_agreement_recomputed": csv_agreement,
    }

def _check_loopb2_missing_suspects(scoring_frame_path: Path, report_meta: dict[str, Any], errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if not scoring_frame_path.exists():
        errors.append(f"missing Loop B2 alpha scoring input frame: {scoring_frame_path}")
        return {"status": "MISSING"}
    df = _read_csv_contract(scoring_frame_path, errors, "Loop B2 alpha scoring input frame")
    if df.empty:
        return {"status": "EMPTY"}
    all_nan_suspects = []
    nonnull_share: dict[str, float | None] = {}
    low_coverage: list[str] = []
    missing_columns: list[str] = []
    for col in LOOPB2_PRIMARY_MISSING_SUSPECTS:
        if col not in df.columns:
            missing_columns.append(col)
            nonnull_share[col] = None
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        share = float(numeric.notna().mean()) if len(numeric) else 0.0
        nonnull_share[col] = share
        if numeric.isna().all():
            all_nan_suspects.append(col)
        if share <= LOOPB2_MIN_R157_R182_NONNULL_SHARE:
            low_coverage.append(col)
    if missing_columns:
        errors.append(f"Loop B2 alpha scoring frame missing R157/R182 guard columns: {missing_columns}")
    if low_coverage:
        errors.append(
            f"Loop B2 R157/R182 non-null coverage must exceed {LOOPB2_MIN_R157_R182_NONNULL_SHARE}; "
            f"low_coverage={low_coverage}, shares={nonnull_share}"
        )
    diag = ((report_meta.get("loopB2") or {}).get("gap_pp_missing_variable_diagnostics") or {})
    reported = set(map(str, diag.get("primary_suspect_variables") or [])) if isinstance(diag, dict) else set()
    missing_reported = [c for c in all_nan_suspects if c not in reported]
    if missing_reported:
        errors.append(f"Loop B2 R157/R182 all-NaN suspects not recorded in report diagnostics: {missing_reported}")
    return {
        "status": "PASS" if not missing_reported and not low_coverage and not missing_columns else "FAIL",
        "rows": int(len(df)),
        "all_nan_primary_suspects": all_nan_suspects,
        "reported_primary_suspects": sorted(reported),
        "nonnull_share": nonnull_share,
        "min_nonnull_share_threshold": LOOPB2_MIN_R157_R182_NONNULL_SHARE,
        "low_coverage_columns": low_coverage,
    }


def verify_stage2_substrate_loopA_loopB2(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    d = final / LOOPB2_STAGE_NAME
    report_path = d / "substrate_loopA_loopB2_report.json"
    scored_csv_path = d / "loopB2_simulator_score_retention.csv"
    scoring_frame_path = d / "loopB2_simulator_score_retention_scoring_input_frame.csv"
    replay_csv_path = d / "loopB2_replay_alpha_predicted_score_vs_real_rating_change.csv"
    nonempty(d, errors, "Loop A/B2 substrate output directory")
    meta: dict[str, Any] = {}
    if nonempty(report_path, errors, "Loop A/B2 substrate report"):
        try:
            meta = read_json(report_path)
        except Exception as exc:
            errors.append(f"cannot read Loop A/B2 substrate report: {exc!r}")
            meta = {}
    report_checks = _check_loopb2_report_contract(meta, errors, warnings) if meta else {"status": "MISSING"}
    scored_df = _read_csv_contract(scored_csv_path, errors, "Loop B2 scored CSV")
    csv_checks = _check_loopb2_csv_contract(scored_df, meta, errors, warnings) if not scored_df.empty else {"status": "MISSING_OR_EMPTY"}
    suspect_checks = _check_loopb2_missing_suspects(scoring_frame_path, meta, errors, warnings) if meta else {"status": "MISSING_REPORT"}
    return {
        "output_dir": str(d),
        "report_path": str(report_path),
        "scored_csv_path": str(scored_csv_path),
        "scoring_frame_path": str(scoring_frame_path),
        "replay_csv_path": str(replay_csv_path),
        "report_contract": report_checks,
        "csv_contract": csv_checks,
        "missing_suspect_diagnostics": suspect_checks,
        "legacy_replay_expected_observed_transition_rows": LOOPB2_EXPECTED_OBSERVED_TRANSITION_ROWS,
        "legacy_replay_expected_all_observed_movers": LOOPB2_EXPECTED_ALL_MOVER_ROWS,
        "oot_window": [LOOPB2_OOT_YEAR_MIN, LOOPB2_OOT_YEAR_MAX],
        "max_b1_minus_b2_gap_pp": LOOPB2_MAX_B1_MINUS_B2_GAP_PP,
    }


def verify_stage2(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    d = final / "stage2_candidate_projection"
    rows = {}
    for name in ["phase1_pretrain_candidate", "phase2_bc_candidate", "phase3_iql_candidate", "phase_eval_candidate"]:
        df = read_parquet(d / f"{name}.parquet", errors, name)
        rows[name] = int(len(df))
        if not df.empty:
            if name == "phase_eval_candidate":
                required = ["firm_id", "fiscal_year"] + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
                check_cols(df, required, errors, name)
                check_stage6_serving_state_columns(df, errors, name)
                forbidden = [c for c in df.columns if str(c).startswith(("action__", "action_observed__", "next__", "soft_cand_")) or c in REWARD_COLS or c in {"candidate_id", "projection_distance", "done"}]
                if forbidden:
                    errors.append(f"phase_eval_candidate must be state-only; forbidden columns present: {forbidden[:20]}")
                continue
            required = ["firm_id", "fiscal_year", "candidate_id"] + ACTION_COLS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS
            if name == "phase3_iql_candidate":
                required += REWARD_COLS
            check_cols(df, required, errors, name)
            if name == "phase2_bc_candidate" and any(c in df.columns for c in REWARD_COLS):
                warnings.append(f"{name}: broad BC phase should not require external-rating reward; reward columns are ignored if present")
            if "candidate_id" in df.columns:
                labels = set(df["candidate_id"].dropna().astype(str).unique())
                if "C2" in labels or any(x.startswith("C2") for x in labels):
                    errors.append(f"{name}: C2 appears as train/projected candidate label")
                unknown = sorted(labels - set(V32_LABELS))
                if unknown:
                    errors.append(f"{name}: labels outside v32 main training labels: {unknown}")
    for f in ["metadata.json", "feature_manifest.json", "candidate_library_metadata.json", "projection_support_by_candidate.csv", "magnitude_recalibrated_libraries_metadata.json"]:
        nonempty(d / f, errors)
    # Stage2 input-split magnitude audit artifacts are produced by the split builder,
    # while candidate projection produces the recalibrated P-library metadata.
    for f in ["candidate_magnitude_audit.csv", "magnitude_calibration_metadata.json"]:
        nonempty(d / "input_splits" / f, errors, f"input_splits/{f}")
    # Guard against reusing old Stage2 runs where 8/10 action dimensions were
    # silently zero-imputed. This verifier gate forces a rebuild from Stage2
    # input splits onward when the action source mapper changes.
    coverage_check = check_action_source_coverage(d / "input_splits" / "action_source_coverage.csv", errors, warnings, "Stage2 projection")
    recalibrated_library_checks = {}
    for q in [50, 65, 75, 85]:
        recalibrated_library_checks[f"P{q}"] = check_recalibrated_candidate_library_uniqueness(d / f"final_candidate_library__P{q}.yaml", errors, warnings)
    aux_reward_contract = check_stage2_aux_reward_contract(d, errors, warnings)
    meta = metadata_status(d / "metadata.json", errors)
    return {"rows": rows, "metadata_status": meta.get("status"), "action_source_coverage": coverage_check, "recalibrated_candidate_libraries": recalibrated_library_checks, "aux_reward_contract": aux_reward_contract}





def check_stage3_encoder_architecture(path: Path, schema: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    expected = _final_encoder_contract()
    observed: dict[str, Any] = {}
    if path.exists() and path.stat().st_size > 0:
        try:
            payload = _torch_load_checkpoint(path)
            cfg = payload.get("model_config", {}) if isinstance(payload, dict) else {}
            observed = {
                "d_model": cfg.get("d_model"),
                "n_heads": cfg.get("n_heads"),
                "n_layers": cfg.get("n_layers"),
                "ff_multiplier": cfg.get("ff_multiplier", expected["ff_multiplier"]),
            }
        except Exception as exc:
            errors.append(f"cannot inspect Stage3 ssl_encoder.pt architecture: {exc!r}")
            return {"status": "FAIL", "expected": expected, "observed": observed}
    schema_arch = schema.get("encoder_architecture") or {}
    for key, val in expected.items():
        actual = observed.get(key, schema_arch.get(key))
        try:
            actual_int = int(actual)
        except Exception:
            actual_int = None
        if actual_int != int(val):
            errors.append(f"Stage3 encoder architecture {key} must be {val}, got {actual}")
    return {"status": "PASS" if all(int(observed.get(k, schema_arch.get(k, -999))) == int(v) for k, v in expected.items() if observed.get(k, schema_arch.get(k)) is not None) else "CHECKED", "expected": expected, "observed": observed, "schema_encoder_architecture": schema_arch}

def verify_stage3_downstream_schema(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    final = final_root(root)
    s3_meta = final / "stage3_acd_ssl" / "feature_schema.json"
    if not s3_meta.exists():
        return {"status": "SKIP_NO_STAGE3_SCHEMA"}
    try:
        schema = read_json(s3_meta)
    except Exception as exc:
        errors.append(f"cannot read Stage3 feature schema: {exc!r}")
        return {"status": "FAIL"}
    features = list(schema.get("continuous_columns") or [])
    cats = list(schema.get("categorical_columns") or [])
    checks = {}
    phase_paths = {
        "phase1_pretrain": "stage2_candidate_projection/input_splits/phase1_pretrain.parquet",
        "phase2_bc": "stage2_candidate_projection/input_splits/phase2_bc.parquet",
        "phase3_iql": "stage2_candidate_projection/input_splits/phase3_iql.parquet",
        "phase_eval": "stage2_candidate_projection/input_splits/phase_eval.parquet",
        "phase1_pretrain_candidate": "stage2_candidate_projection/phase1_pretrain_candidate.parquet",
        "phase2_bc_candidate": "stage2_candidate_projection/phase2_bc_candidate.parquet",
        "phase3_iql_candidate": "stage2_candidate_projection/phase3_iql_candidate.parquet",
        "phase_eval_candidate": "stage2_candidate_projection/phase_eval_candidate.parquet",
    }
    for name, rel in phase_paths.items():
        df = read_parquet(final / rel, errors, name)
        if df.empty:
            checks[name] = {"rows": 0, "missing_features": features[:20], "missing_categorical": cats}
            continue
        missing = [c for c in features if c not in df.columns]
        missing_cat = [c for c in cats if c not in df.columns]
        missing_next = []
        if name == "phase1_pretrain":
            missing_next = [f"next__{c}" for c in ACD_TARGET_COLUMNS if f"next__{c}" not in df.columns]
        clipped = [c for c in df.columns if str(c).startswith("clipped_")]
        checks[name] = {
            "rows": int(len(df)),
            "missing_feature_count": len(missing),
            "missing_features_sample": missing[:20],
            "missing_categorical": missing_cat,
            "missing_acd_next_target_count": len(missing_next),
            "missing_acd_next_target_sample": missing_next[:20],
            "clipped_columns_sample": clipped[:20],
        }
        if missing:
            errors.append(f"{name} missing Stage3 encoder feature columns; first missing: {missing[:20]}")
        if missing_cat:
            errors.append(f"{name} missing Stage3 categorical columns: {missing_cat}")
        if missing_next:
            errors.append(f"{name} missing ACD next target columns; first missing: {missing_next[:20]}")
        if clipped:
            errors.append(f"{name} contains forbidden clipped_* columns: {clipped[:20]}")
    fail = any(v.get("missing_feature_count", 0) or v.get("missing_categorical") or v.get("missing_acd_next_target_count", 0) or v.get("clipped_columns_sample") for v in checks.values())
    return {"status": "FAIL" if fail else "PASS", "n_features": len(features), "checks": checks}


def verify_stage3(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage3_acd_ssl"
    for f in ["ssl_encoder.pt", "metadata.json", "feature_schema.json", "preprocess_stats.json", "training_log.csv", "feature_leakage_audit.json"]:
        nonempty(d / f, errors)
    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    # Final-freeze checkpoint contract: final-refit cells must expose the fulltrain
    # encoder alias, while inner-dev winner is required only for actual inner-dev
    # selection runs.  A fixed-config Phase0/final_refit cell intentionally trains
    # no inner-dev winner; treating that absence as a hard failure stops valid
    # downstream Stage4/5/6 archives even though they consume the fulltrain alias.
    fulltrain_alias = d / "stage3_encoder_avs256_final_refit_fulltrain.pt"
    innerdev_alias = d / "stage3_encoder_avs256_innerdev_winner.pt"
    nonempty(fulltrain_alias, errors)
    train_mode = str(meta.get("train_mode") or "").lower()
    if train_mode == "final_refit":
        if not innerdev_alias.exists():
            warnings.append("Stage3 innerdev winner checkpoint missing; accepted for final_refit/fixed-config run because downstream Stage4/5/6 consume stage3_encoder_avs256_final_refit_fulltrain.pt")
        elif innerdev_alias.stat().st_size <= 0:
            errors.append(f"empty {innerdev_alias}")
    else:
        nonempty(innerdev_alias, errors)
    schema = {}
    if (d / "feature_schema.json").exists():
        try:
            schema = read_json(d / "feature_schema.json")
        except Exception as exc:
            errors.append(f"cannot read Stage3 feature_schema.json: {exc!r}")
    if schema:
        if schema.get("schema_version") != SCHEMA_VERSION:
            errors.append(f"Stage3 schema_version must be {SCHEMA_VERSION}, got {schema.get('schema_version')}")
        if int(schema.get("n_continuous_features", -1)) != 129:
            errors.append(f"Stage3 n_continuous_features must equal 129, got {schema.get('n_continuous_features')}")
        if int(schema.get("n_categorical_fields", -1)) != 2:
            errors.append(f"Stage3 n_categorical_fields must equal 2, got {schema.get('n_categorical_fields')}")
        if int(schema.get("n_acd_targets", -1)) != 118:
            errors.append(f"Stage3 n_acd_targets must equal 118, got {schema.get('n_acd_targets')}")
        if schema.get("acd_head_class") != "ActionConditionalForwardHead":
            errors.append(f"Stage3 acd_head_class must be ActionConditionalForwardHead, got {schema.get('acd_head_class')}")
        if schema.get("acd_uses_interaction") is not True:
            errors.append("Stage3 acd_uses_interaction must be true")
        schema_arch = schema.get("encoder_architecture") or {}
        for _k, _v in _final_encoder_contract().items():
            _actual = schema_arch.get(_k)
            if _actual is not None and int(_actual) != int(_v):
                errors.append(f"Stage3 feature_schema encoder_architecture {_k} must be {_v}, got {_actual}")
        if list(schema.get("categorical_columns") or []) != CATEGORICAL_COLUMNS:
            errors.append(f"Stage3 categorical_columns must be {CATEGORICAL_COLUMNS}, got {schema.get('categorical_columns')}")
        if list(schema.get("continuous_columns") or []) != CONTINUOUS_COLUMNS:
            errors.append("Stage3 continuous_columns do not exactly match AVS256 binding order")
        if list(schema.get("acd_target_columns") or []) != ACD_TARGET_COLUMNS:
            errors.append("Stage3 acd_target_columns do not exactly match AVS256 ACD target order")
        if dict(schema.get("block_realized_counts") or {}) != EXPECTED_BLOCK_COUNTS:
            errors.append(f"Stage3 block counts mismatch: {schema.get('block_realized_counts')}")
        if set((schema.get("direction_vocab") or {}).keys()) != set(DIRECTION_VOCAB.keys()):
            errors.append(f"Stage3 direction_vocab keys mismatch: {schema.get('direction_vocab')}")
        for k, expected in [("mcm_weight", 1.0), ("acd_weight", 0.5), ("contrastive_weight", 0.3)]:
            if float(schema.get(k, -999)) != expected:
                errors.append(f"Stage3 {k} must be {expected}, got {schema.get(k)}")
        if any(str(c).startswith("clipped_") for c in (schema.get("continuous_columns") or [])):
            errors.append("Stage3 continuous_columns contain forbidden clipped_* column")
    for k in ["feature_schema_hash", "candidate_library_hash", "final_action_contract_hash", "optimizer", "learning_rate", "weight_decay", "lr_scheduler", "total_steps_mode", "optimizer_steps", "epochs"]:
        if k not in meta:
            errors.append(f"Stage3 metadata missing {k}")
    if meta.get("optimizer") not in (None, "AdamW"):
        errors.append(f"Stage3 optimizer must be AdamW, got {meta.get('optimizer')}")
    if "learning_rate" in meta and float(meta.get("learning_rate", 0.0)) <= 0.0:
        errors.append(f"Stage3 learning_rate must be positive, got {meta.get('learning_rate')}")
    if "weight_decay" in meta and float(meta.get("weight_decay", -1.0)) < 0.0:
        errors.append(f"Stage3 weight_decay must be non-negative, got {meta.get('weight_decay')}")
    if meta.get("lr_scheduler") not in (None, "none"):
        errors.append(f"Stage3 lr_scheduler must be 'none' until scheduler semantics are implemented, got {meta.get('lr_scheduler')}")
    if meta.get("total_steps_mode") not in (None, "epoch_based"):
        errors.append(f"Stage3 total_steps_mode must be epoch_based, got {meta.get('total_steps_mode')}")
    encoder_architecture = check_stage3_encoder_architecture(d / "ssl_encoder.pt", schema, errors)
    downstream_schema = verify_stage3_downstream_schema(root, errors, warnings)
    sweep_grid = check_sensitivity_grid(d / "stage3_sensitivity_phase_alpha.csv", errors, warnings, "Stage3 phase-alpha sensitivity grid", require_selected=False)
    return {"metadata_status": meta.get("status"), "schema_version": schema.get("schema_version"), "encoder_architecture": encoder_architecture, "downstream_schema_compatibility": downstream_schema, "sweep_grid": sweep_grid}


def _load_recalibrated_library_hash(root: Path, q: int, errors: list[str]) -> str:
    path = final_root(root) / "stage2_candidate_projection" / f"final_candidate_library__P{int(q)}.yaml"
    if not path.exists():
        errors.append(f"missing selected recalibrated candidate library for P{q}: {path}")
        return ""
    return sha256_file(path)


def _check_policy_checkpoint_recalibrated_action_payload(root: Path, ckpt_path: Path, stage: str, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    if not ckpt_path.exists():
        errors.append(f"{stage} missing checkpoint for recalibrated action payload check: {ckpt_path.name}")
        return {"exists": False}
    try:
        payload = _torch_load_checkpoint(ckpt_path)
    except Exception as exc:
        errors.append(f"{stage} cannot load checkpoint for recalibrated action payload check {ckpt_path.name}: {exc!r}")
        return {"exists": True, "load_error": repr(exc)}
    q = payload.get("selected_magnitude_quantile") or payload.get("magnitude_quantile")
    if q is None:
        errors.append(f"{stage} checkpoint missing selected_magnitude_quantile/magnitude_quantile: {ckpt_path.name}")
        return {"exists": True, "selected_magnitude_quantile": None}
    expected_hash = _load_recalibrated_library_hash(root, int(q), errors)
    found_hash = payload.get("selected_recalibrated_candidate_library_hash")
    if not found_hash:
        errors.append(f"{stage} checkpoint missing selected_recalibrated_candidate_library_hash: {ckpt_path.name}")
    elif expected_hash and str(found_hash) != str(expected_hash):
        errors.append(f"{stage} selected recalibrated candidate library hash mismatch for P{q}: checkpoint={found_hash} expected={expected_hash}")
    if payload.get("candidate_action_values_source") != "stage2_recalibrated_candidate_library":
        errors.append(f"{stage} candidate_action_values_source must be stage2_recalibrated_candidate_library, got {payload.get('candidate_action_values_source')}")
    rows = payload.get("candidate_action_values")
    labels = train_labels(root)
    cols = action_cols(root)
    if not isinstance(rows, list) or not rows:
        errors.append(f"{stage} checkpoint missing non-empty candidate_action_values: {ckpt_path.name}")
        return {"exists": True, "selected_magnitude_quantile": int(q), "candidate_action_value_count": 0}
    by_id = {str(r.get("candidate_id")): r for r in rows if isinstance(r, dict) and r.get("candidate_id") is not None}
    missing = [x for x in labels if x not in by_id]
    if missing:
        errors.append(f"{stage} candidate_action_values missing train labels: {missing[:8]}")
    duplicate_vectors = []
    seen = {}
    for cid in labels:
        r = by_id.get(cid)
        if not r:
            continue
        vec = tuple(round(float(r.get(c, 0.0) or 0.0), 12) for c in cols)
        if vec in seen:
            duplicate_vectors.append((seen[vec], cid))
        else:
            seen[vec] = cid
    if duplicate_vectors:
        errors.append(f"{stage} duplicate fixed candidate action vectors in checkpoint payload: {duplicate_vectors[:8]}")
    return {
        "exists": True,
        "selected_magnitude_quantile": int(q),
        "expected_selected_recalibrated_candidate_library_hash": expected_hash,
        "checkpoint_selected_recalibrated_candidate_library_hash": found_hash,
        "candidate_action_value_count": len(rows),
        "fixed_candidate_payload_count": len([x for x in labels if x in by_id]),
        "unique_fixed_action_vector_count": len(seen),
        "duplicate_fixed_action_vector_count": len(duplicate_vectors),
    }

def verify_stage4(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage4_candidate_bc"
    for f in ["candidate_bc_policy.pt", "metadata.json", "validation_metrics.json"]:
        nonempty(d / f, errors)
    sweep_grid = check_sensitivity_grid(d / "stage4_bc_sensitivity_grid.csv", errors, warnings, "Stage4 BC sensitivity grid")
    # Require at least one explicit final-refit checkpoint artifact.
    if not any(d.glob("stage4_bc_final_refit__P*__*.pt")):
        errors.append("Stage4 missing stage4_bc_final_refit__P{q}__{mode}.pt checkpoint")
    meta = metadata_status(d / "metadata.json", errors)
    if meta.get("hard_target_fallback_used"):
        errors.append("Stage4 hard target fallback was used; final mode forbids it")
    if meta.get("action_vocabulary") and list(meta.get("action_vocabulary")) != train_labels(root):
        errors.append("Stage4 action vocabulary differs from v32 main labels")
    if bool(meta.get("class_balanced_loss")):
        nonempty(d / str(meta.get("class_balance_audit_file") or "class_balance_audit.csv"), errors, "Stage4 class balance audit")
    if bool(meta.get("family_balanced_loss")):
        fam_file = str(meta.get("family_balance_audit_file") or "family_balance_audit.csv")
        q_file = str(meta.get("stage4_label_quality_audit_file") or "stage4_label_quality_audit.csv")
        nonempty(d / fam_file, errors, "Stage4 family balance audit")
        nonempty(d / q_file, errors, "Stage4 label quality audit")
        if not isinstance(meta.get("action_family_by_candidate"), dict) or not meta.get("action_family_by_candidate"):
            errors.append("Stage4 family-balanced run missing action_family_by_candidate metadata")
        if not isinstance(meta.get("loss_weights_by_candidate"), dict) or not meta.get("loss_weights_by_candidate"):
            errors.append("Stage4 family-balanced run missing loss_weights_by_candidate metadata")
        for k in ["family_balance_power", "family_weight_cap", "combined_weight_cap"]:
            if k not in meta:
                errors.append(f"Stage4 family-balanced run missing {k} metadata")
        labels = train_labels(root)
        fam_keys = set(str(x) for x in (meta.get("action_family_by_candidate") or {}).keys())
        if labels and fam_keys and set(labels) != fam_keys:
            errors.append("Stage4 family-balanced action_family_by_candidate keys differ from train labels")
    q = int(meta.get("magnitude_quantile") or 50)
    mode = "lr_scale_0.1" if bool(meta.get("encoder_finetune")) else "frozen"
    final_ckpt = d / f"stage4_bc_final_refit__P{q}__{mode}.pt"
    action_payload = _check_policy_checkpoint_recalibrated_action_payload(root, final_ckpt, "Stage4", errors, warnings)
    return {"metadata_status": meta.get("status"), "sweep_grid": sweep_grid, "recalibrated_action_payload": action_payload}


def verify_stage5(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage5_candidate_iql"
    for f in ["candidate_iql_policy.pt", "metadata.json", "checkpoint_selection_report.json", "validation_metrics.json"]:
        nonempty(d / f, errors)
    # Final Stage6 consumes the full-train final_refit checkpoint.  The inner-dev
    # winner is optional for fast/local final_refit-only reruns and should not block
    # the final Stage6 path when the final_refit artifact exists.
    if not (d / "stage5_candidate_iql_final_refit_fulltrain.pt").exists():
        nonempty(d / "stage5_candidate_iql_final_refit_fulltrain.pt", errors)
    if not (d / "stage5_candidate_iql_innerdev_winner.pt").exists():
        warnings.append("Stage5 innerdev winner checkpoint missing; accepted for fast/local final_refit-only run because Stage6 consumes stage5_candidate_iql_final_refit_fulltrain.pt")
    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    if meta.get("action_vocabulary") and list(meta.get("action_vocabulary")) != train_labels(root):
        errors.append("Stage5 action vocabulary differs from v32 main labels")
    sweep_grid = check_sensitivity_grid(d / "stage5_sensitivity_phase_gamma.csv", errors, warnings, "Stage5 phase-gamma sensitivity grid")
    for k in ["candidate_library_hash", "final_action_contract_hash", "stage4_checkpoint_hash"]:
        if k not in meta:
            errors.append(f"Stage5 metadata missing {k}")
    if "stage3_schema_hash" not in meta and "stage3_feature_schema_hash" not in meta:
        errors.append("Stage5 metadata missing stage3_schema_hash/stage3_feature_schema_hash")
    action_payload = _check_policy_checkpoint_recalibrated_action_payload(root, d / "stage5_candidate_iql_final_refit_fulltrain.pt", "Stage5", errors, warnings)
    return {"metadata_status": meta.get("status"), "sweep_grid": sweep_grid, "recalibrated_action_payload": action_payload}


def verify_stage6_actions(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage6_candidate_selector_eval"
    df = read_parquet(d / "policy_actions.parquet", errors, "policy_actions")
    if not df.empty:
        check_cols(df, ["row_id", "policy", "candidate_id"] + action_cols(root), errors, "policy_actions")
        policies = set(df["policy"].astype(str)) if "policy" in df.columns else set()
        if "C_obs" in policies:
            errors.append("Stage6 primary policy_actions must not contain C_obs; C_obs is secondary inner-dev only")
        if "C0_noop" not in policies:
            errors.append("Stage6 policy_actions lacks C0_noop policy rows")
        else:
            all_ids=set(pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
            noop_ids=set(pd.to_numeric(df.loc[df["policy"].astype(str)=="C0_noop", "row_id"], errors="coerce").dropna().astype(int).tolist())
            missing=sorted(all_ids-noop_ids)
            if missing:
                errors.append(f"Stage6 policy_actions no-op pairing violation; missing row_ids sample={missing[:20]} count={len(missing)}")
    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    return {"rows": int(len(df)), "metadata_status": meta.get("status")}


def verify_stage6(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    d = final_root(root) / "stage6_candidate_selector_eval"
    for f in ["policy_actions.parquet", "simulated_oracle_input_frame.parquet", "action_effect_audit.parquet", "oracle_scores_alpha.parquet", "oracle_scores_beta.parquet", "oracle_scores_gamma.parquet", "multi_oracle_policy_eval.parquet", "multi_oracle_metadata.json"]:
        nonempty(d / f, errors)
    df = read_parquet(d / "multi_oracle_policy_eval.parquet", errors, "multi_oracle_policy_eval")
    sim_df = read_parquet(d / "simulated_oracle_input_frame.parquet", errors, "simulated_oracle_input_frame")
    if not sim_df.empty:
        try:
            _tc = __import__("credit_recourse.rl.common.temporal", fromlist=["load_temporal_contract"]).load_temporal_contract(root)
            # Stage6 currently simulates the serving eval-base state and stamps
            # predicted_fiscal_year with temporal_split.eval_base_year.  Older
            # verifier code expected a removed rollout_target_year attribute,
            # which made the boundary check fail even when Stage6 metadata and
            # outputs were otherwise PASS.  If a future temporal_split.yaml
            # explicitly contains rollout_target_year, honor it from raw;
            # otherwise use the binding eval_base_year used by Stage6 pipelines.
            _expected_pred_year = int(getattr(_tc, "rollout_target_year", (_tc.raw or {}).get("rollout_target_year", _tc.eval_base_year)))
            if "predicted_fiscal_year" not in sim_df.columns:
                errors.append("simulated_oracle_input_frame missing predicted_fiscal_year")
            else:
                _years = sorted([int(x) for x in pd.to_numeric(sim_df["predicted_fiscal_year"], errors="coerce").dropna().unique().tolist()])
                if _years != [_expected_pred_year]:
                    errors.append(f"simulated_oracle_input_frame predicted_fiscal_year must be {_expected_pred_year}, got {_years}")
        except Exception as exc:
            errors.append(f"cannot verify Stage6 predicted_fiscal_year: {exc!r}")
    if not df.empty:
        delta_cols = [c for c in df.columns if str(c).startswith("delta_R_score_")]
        if not delta_cols:
            errors.append("multi_oracle_policy_eval has no explicit delta_R_score_* columns")
        if "policy" in df.columns:
            policies=set(df["policy"].astype(str))
            if "C0_noop" not in policies:
                errors.append("Stage6 evaluation lacks C0_noop policy rows")
            if "C_obs" in policies:
                errors.append("Stage6 primary evaluation must not contain C_obs; C_obs is secondary inner-dev only")
            if "C0_noop" in policies and "row_id" in df.columns:
                all_ids=set(pd.to_numeric(df["row_id"], errors="coerce").dropna().astype(int).tolist())
                noop_ids=set(pd.to_numeric(df.loc[df["policy"].astype(str)=="C0_noop", "row_id"], errors="coerce").dropna().astype(int).tolist())
                missing=sorted(all_ids-noop_ids)
                if missing:
                    errors.append(f"Stage6 evaluation no-op pairing violation; missing row_ids sample={missing[:20]} count={len(missing)}")
    meta = metadata_status(d / "multi_oracle_metadata.json", errors, allow_prefix=("PASS",))
    for extra in ["final_policy_summary.csv", "firm_state_input_audit.json", "policy_pairing_audit.json", "stage6_policy_actions_inner_dev.parquet"]:
        nonempty(d / extra, errors, extra)
    supply = d / "variable_supply_manifest.json"
    if supply.exists():
        sm = read_json(supply)
        missing = sm.get("missing_required_variables_by_backend") or {}
        bad = {k: v for k, v in missing.items() if v}
        if bad:
            errors.append(f"Stage6 missing required variables by backend: {bad}")
    else:
        warnings.append("Stage6 variable_supply_manifest.json not found")
    return {"rows": int(len(df)), "metadata_status": meta.get("status")}



def verify_stage7(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    """Verify Stage 7 LLM action generation outputs.

    Checks:

    * Required output files exist and are non-empty.
    * ``llm_stage7_action_table.parquet`` action column order matches
      ``final_action_contract.yaml``.
    * Every ``candidate_id`` is in the v32 main_train_labels.
    * Every action value is within bounds (post-clipping contract).
    * Every ``policy`` is in {C4, C4R, C5, C6, C6X, C7, C8}.
    * Metadata carries ``candidate_library_hash``, ``final_action_contract_hash``,
      ``backend_id``, ``backend_is_live``, and ``final_paper_run_allowed``.
    * The action column order in the table is identical to the active
      ``ActionSpace``.
    """
    from credit_recourse.contracts.stage_paths import stage_dir as _stage_dir

    d = _stage_dir(root, "stage7")
    for f in [
        "llm_stage7_action_table.parquet",
        "llm_stage7_prompt_manifest.json",
        "llm_stage7_failure_audit.csv",
        "llm_stage7_response_log.parquet",
        "metadata.json",
    ]:
        nonempty(d / f, errors)

    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    df = read_parquet(d / "llm_stage7_action_table.parquet", errors, "llm_stage7_action_table")

    if not df.empty:
        cols = action_cols(root)
        action_cols_in_frame = [c for c in df.columns if c.startswith("action__")]
        if action_cols_in_frame != cols:
            errors.append(
                f"Stage7 action column order differs from final_action_contract: "
                f"got {action_cols_in_frame}, expected {cols}"
            )
        check_cols(df, ["row_id", "policy", "candidate_id"] + cols, errors, "llm_stage7_action_table")
        pols = set(df["policy"].astype(str).unique()) if "policy" in df.columns else set()
        forbidden_pols = pols - {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"}
        if forbidden_pols:
            errors.append(f"Stage7 action table contains forbidden policy codes: {sorted(forbidden_pols)}")
        for required_col in ["reference_source", "reference_draw_seed"]:
            if required_col not in df.columns:
                errors.append(f"Stage7 action table missing required v4 C6X column: {required_col}")
        if "reference_source" in df.columns:
            bad_sources = set(df["reference_source"].dropna().astype(str).unique()) - {"rl", "random", "none"}
            if bad_sources:
                errors.append(f"Stage7 action table contains invalid reference_source values: {sorted(bad_sources)}")
            if "policy" in df.columns:
                c6x = df[df["policy"].astype(str) == "C6X"]
                if not c6x.empty:
                    if (c6x["reference_source"].astype(str) != "random").any():
                        errors.append("Stage7 C6X rows must have reference_source='random'")
                    if "reference_draw_seed" not in c6x.columns or c6x["reference_draw_seed"].isna().any():
                        errors.append("Stage7 C6X rows must have non-null reference_draw_seed")
                rl_ref_rows = df[df["policy"].astype(str).isin(["C6", "C7", "C8"])]
                if not rl_ref_rows.empty and (rl_ref_rows["reference_source"].astype(str) != "rl").any():
                    errors.append("Stage7 C6/C7/C8 rows must have reference_source='rl'")
                c4r = df[df["policy"].astype(str).eq("C4R")]
                if not c4r.empty:
                    if (c4r["reference_source"].astype(str) != "none").any():
                        errors.append("Stage7 C4R rows must have reference_source='none'")
                    if "reference_draw_seed" in c4r.columns and c4r["reference_draw_seed"].notna().any():
                        errors.append("Stage7 C4R rows must have null reference_draw_seed")
                    if "rl_reference_candidate" in c4r.columns and c4r["rl_reference_candidate"].notna().any():
                        errors.append("Stage7 C4R rows must not contain an external reference candidate")
        candidates = set(df["candidate_id"].astype(str).unique()) if "candidate_id" in df.columns else set()
        bad = candidates - set(train_labels(root))
        if bad:
            errors.append(f"Stage7 action table contains candidates outside v32 main_train_labels: {sorted(bad)}")
        # Bound-violation check
        try:
            space = load_action_space(root)
            for col in cols:
                raw_name = col.replace("action__", "")
                bkey = col if col in space.bounds else raw_name
                lo, hi = space.bounds[bkey]
                s = pd.to_numeric(df[col], errors="coerce")
                if (s < lo - 1e-9).any() or (s > hi + 1e-9).any():
                    n_bad = int(((s < lo - 1e-9) | (s > hi + 1e-9)).sum())
                    errors.append(f"Stage7 action {col} out of bounds [{lo},{hi}] in {n_bad} rows")
        except Exception as exc:
            errors.append(f"Stage7 bound check failed: {exc}")

    for k in [
        "candidate_library_hash", "candidate_library_path", "candidate_action_values_source",
        "candidate_library_quantile", "selected_recalibrated_candidate_library_hash",
        "final_action_contract_hash", "backend_id", "backend_is_live",
        "final_paper_run_allowed", "reference_draw_seed", "reference_source_policy",
    ]:
        if k not in meta:
            errors.append(f"Stage7 metadata missing required key: {k}")
    if meta.get("candidate_action_values_source") != "stage2_recalibrated_candidate_library":
        errors.append(
            "Stage7 must materialize LLM prompt/projection/simulator vectors from "
            "Stage2 recalibrated candidate library (P50), not active base YAML"
        )
    try:
        q = int(meta.get("candidate_library_quantile"))
        if q != 50:
            errors.append(f"Stage7 candidate_library_quantile must be 50 for final LLM/RL comparable runs, got {q}")
        p50_path = resolve_candidate_library_path(root, magnitude_quantile=q)
        p50_space = load_action_space(root, candidate_library_path=p50_path)
        if meta.get("candidate_library_hash") != p50_space.candidate_library_hash:
            errors.append(
                f"Stage7 metadata candidate_library_hash must match selected P{q} library: "
                f"meta={meta.get('candidate_library_hash')}, expected={p50_space.candidate_library_hash}"
            )
    except Exception as exc:
        errors.append(f"Stage7 selected candidate-library lineage check failed: {exc}")
    if not meta.get("backend_is_live") and meta.get("final_paper_run_allowed"):
        errors.append("Stage7 final_paper_run_allowed=True but backend_is_live=False; reproducibility backend cannot claim final-paper status")

    # Stage7 failure audit must expose executable pre-sim coders for the closed
    # taxonomy direction/magnitude categories.  Feasibility is enriched by
    # Stage8 after simulation.
    try:
        audit_cols = list(pd.read_csv(d / "llm_stage7_failure_audit.csv", nrows=0).columns)
        for col in [
            "direction_error_auto", "direction_review_needed", "direction_error_reason", "direction_rule_version",
            "magnitude_error_auto", "magnitude_review_needed", "magnitude_error_reason", "magnitude_rule_version",
            "failure_coder_version", "oracle_scores_used_for_failure_coding",
        ]:
            if col not in audit_cols:
                errors.append(f"Stage7 failure audit missing executable failure-coder column: {col}")
    except Exception as exc:
        errors.append(f"Stage7 failure-audit coder-column check failed: {exc}")


    # --- N5/free-form action-budget contract checks ---
    budget_contract = meta.get("action_budget_contract") or {"enabled": False}
    row_selection_contract = meta.get("row_selection_contract") or {"enabled": False}
    try:
        prompt_manifest = read_json(d / "llm_stage7_prompt_manifest.json") if (d / "llm_stage7_prompt_manifest.json").exists() else {}
    except Exception as exc:
        prompt_manifest = {}
        errors.append(f"Stage7 cannot inspect prompt manifest for budget contract: {exc!r}")
    if bool(budget_contract.get("enabled")):
        if budget_contract.get("schema_version") != ACTION_BUDGET_CONTRACT_SCHEMA_VERSION:
            errors.append(
                "Stage7 action_budget_contract has unexpected schema_version: "
                f"{budget_contract.get('schema_version')!r}"
            )
        if not budget_contract.get("label"):
            errors.append("Stage7 action_budget_contract enabled but missing label")
        try:
            budget_target = float(budget_contract.get("l1_budget"))
            if not budget_target > 0:
                errors.append(f"Stage7 action_budget_contract l1_budget must be positive, got {budget_target}")
        except Exception as exc:
            budget_target = None
            errors.append(f"Stage7 action_budget_contract l1_budget not numeric: {exc!r}")
        budgeted_conditions = set(map(str, budget_contract.get("budgeted_conditions") or []))
        budgeted_modes = set(map(str, budget_contract.get("budgeted_modes") or []))
        if not budgeted_conditions:
            errors.append("Stage7 action_budget_contract enabled but budgeted_conditions is empty")
        if budgeted_modes != {"free_form_10d"}:
            errors.append(f"Stage7 action_budget_contract budgeted_modes must be ['free_form_10d'], got {sorted(budgeted_modes)}")
        if prompt_manifest.get("action_budget_contract") != budget_contract:
            errors.append("Stage7 prompt manifest action_budget_contract does not match metadata")
        try:
            audit_df = pd.read_csv(d / "llm_stage7_failure_audit.csv")
        except Exception as exc:
            audit_df = pd.DataFrame()
            errors.append(f"Stage7 cannot read failure audit for budget contract: {exc!r}")
        for where, frame in [("llm_stage7_action_table", df), ("llm_stage7_failure_audit", audit_df)]:
            missing_cols = [c for c in BUDGET_AUDIT_COLUMNS if c not in frame.columns]
            if missing_cols:
                errors.append(f"Stage7 {where} missing N5 budget audit columns: {missing_cols}")
        if not df.empty and all(c in df.columns for c in BUDGET_AUDIT_COLUMNS):
            expected_flag = df["policy"].astype(str).isin(budgeted_conditions) & df["mode"].astype(str).isin(budgeted_modes)
            actual_flag = df["budgeted_condition_flag"].fillna(False).astype(bool)
            if (expected_flag != actual_flag).any():
                n_bad = int((expected_flag != actual_flag).sum())
                errors.append(f"Stage7 action table budgeted_condition_flag mismatch in {n_bad} rows")
            applicable = df[expected_flag].copy()
            if applicable.empty:
                errors.append("Stage7 action_budget_contract enabled but no action-table rows are budgeted")
            elif budget_target is not None:
                target_vals = pd.to_numeric(applicable["budget_l1_target"], errors="coerce")
                if target_vals.isna().any() or (target_vals - budget_target).abs().max() > 1e-9:
                    errors.append("Stage7 action table budget_l1_target does not equal metadata l1_budget for all budgeted rows")
                l1_calc = applicable[cols].abs().sum(axis=1)
                l1_reported = pd.to_numeric(applicable["budget_l1_clipped"], errors="coerce")
                if l1_reported.isna().any() or (l1_calc - l1_reported).abs().max() > 1e-9:
                    errors.append("Stage7 action table budget_l1_clipped does not equal post-clipping action-column L1")
                tol = float(budget_contract.get("tolerance", 1.0e-9) or 0.0)
                compliant_expected = l1_reported <= budget_target + tol
                compliant_reported = applicable["budget_compliant_clipped"].fillna(False).astype(bool)
                if (compliant_expected.astype(bool).to_numpy() != compliant_reported.to_numpy()).any():
                    errors.append("Stage7 action table budget_compliant_clipped is inconsistent with budget_l1_clipped and l1_budget")
        if not audit_df.empty and all(c in audit_df.columns for c in BUDGET_AUDIT_COLUMNS):
            expected_audit_flag = audit_df["policy"].astype(str).isin(budgeted_conditions) & audit_df["mode"].astype(str).isin(budgeted_modes)
            actual_audit_flag = audit_df["budgeted_condition_flag"].fillna(False).astype(bool)
            if (expected_audit_flag != actual_audit_flag).any():
                n_bad = int((expected_audit_flag != actual_audit_flag).sum())
                errors.append(f"Stage7 failure audit budgeted_condition_flag mismatch in {n_bad} rows")
    elif any(k in meta for k in ["freeform_l1_budget", "budget_contract_label"]):
        warnings.append("Stage7 has legacy budget keys but action_budget_contract.enabled is false")

    # --- full prompt payload archive (future-run contract) ---
    prompt_archive = meta.get("prompt_payload_archive_contract")
    if prompt_archive:
        if prompt_archive.get("schema_version") != "stage7_prompt_payload_archive_v1":
            errors.append(
                f"Stage7 prompt payload archive schema mismatch: {prompt_archive.get('schema_version')!r}"
            )
        if prompt_archive.get("status") != "FULL_PAYLOAD_ARCHIVED":
            errors.append("Stage7 prompt payload archive status must be FULL_PAYLOAD_ARCHIVED")
        archive_rel = prompt_archive.get("path")
        archive_path = Path(archive_rel) if archive_rel else None
        if archive_path is not None and not archive_path.is_absolute():
            archive_path = d / archive_path
        if archive_path is None or not archive_path.is_file():
            errors.append(f"Stage7 prompt payload archive missing: {archive_path}")
        else:
            if str(prompt_archive.get("sha256")) != sha256_file(archive_path):
                errors.append("Stage7 prompt payload archive SHA-256 mismatch")
            try:
                records = [json.loads(line) for line in archive_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                expected_count = int(prompt_archive.get("record_count", -1))
                if len(records) != expected_count or len(records) != int(meta.get("request_count", -1)):
                    errors.append(
                        f"Stage7 prompt payload count mismatch: file={len(records)}, "
                        f"metadata={expected_count}, requests={meta.get('request_count')}"
                    )
                required_prompt_keys = {
                    "schema_version", "request_fingerprint", "row_id", "condition", "mode",
                    "information_condition", "prompt_payload", "prompt_sha256",
                }
                for idx, record in enumerate(records):
                    missing_prompt_keys = sorted(required_prompt_keys - set(record))
                    if missing_prompt_keys:
                        errors.append(f"Stage7 prompt payload row {idx} missing keys: {missing_prompt_keys}")
                        break
                    payload_hash = __import__("hashlib").sha256(
                        json.dumps(
                            record["prompt_payload"], ensure_ascii=False, sort_keys=True,
                            separators=(",", ":"), default=str,
                        ).encode("utf-8")
                    ).hexdigest()
                    if payload_hash != str(record.get("prompt_sha256")):
                        errors.append(f"Stage7 prompt payload row {idx} hash mismatch")
                        break
                    if str(record.get("condition")) == "C4R":
                        payload = record.get("prompt_payload") or {}
                        if payload.get("reference_source") != "none":
                            errors.append(f"Stage7 C4R prompt payload row {idx} must have reference_source='none'")
                            break
                        if payload.get("rl_reference_candidate") not in (None, ""):
                            errors.append(f"Stage7 C4R prompt payload row {idx} leaked an external reference candidate")
                            break
                        if not isinstance(payload.get("initial_action"), dict):
                            errors.append(f"Stage7 C4R prompt payload row {idx} lacks the paired C4 initial_action")
                            break
            except Exception as exc:
                errors.append(f"Stage7 prompt payload archive cannot be validated: {exc!r}")
        prompt_manifest_archive = prompt_manifest.get("prompt_payload_archive") or {}
        if prompt_manifest_archive != {
            "schema_version": prompt_archive.get("schema_version"),
            "path": prompt_archive.get("path"),
            "sha256": prompt_archive.get("sha256"),
            "record_count": prompt_archive.get("record_count"),
            "preservation_policy": "full_structured_prompt_payload_per_request",
        }:
            errors.append("Stage7 prompt manifest and metadata prompt-payload archive contracts differ")
    else:
        warnings.append(
            "Stage7 archive is hash-only and predates stage7_prompt_payload_archive_v1; "
            "full prompt text cannot be byte-audited retrospectively"
        )

    if bool(row_selection_contract.get("enabled")):
        sel_path = row_selection_contract.get("selected_row_ids_file")
        if not sel_path:
            errors.append("Stage7 row_selection_contract enabled but selected_row_ids_file is missing")
        else:
            sel_resolved = Path(sel_path) if Path(sel_path).is_absolute() else d / str(sel_path)
            if not sel_resolved.exists():
                errors.append(f"Stage7 row_selection_contract enabled but selected_row_ids_file is missing: {sel_path}")
        try:
            selected_n = int(row_selection_contract.get("selected_row_count"))
            if selected_n <= 0:
                errors.append(f"Stage7 row_selection_contract selected_row_count must be positive, got {selected_n}")
        except Exception as exc:
            errors.append(f"Stage7 row_selection_contract selected_row_count invalid: {exc!r}")
        try:
            audit_df_for_rows = pd.read_csv(d / "llm_stage7_failure_audit.csv")
            audit_unique = int(pd.to_numeric(audit_df_for_rows["row_id"], errors="raise").nunique())
            if audit_unique != int(row_selection_contract.get("selected_row_count")):
                errors.append(
                    f"Stage7 failure audit row_id count {audit_unique} does not match row_selection selected_row_count "
                    f"{row_selection_contract.get('selected_row_count')}"
                )
        except Exception as exc:
            errors.append(f"Stage7 row_selection_contract audit row check failed: {exc!r}")

    # --- information-condition consistency (LLM789-010) ---
    meta_ic = meta.get("information_condition")
    if not df.empty and "information_condition" in df.columns:
        table_ics = sorted(set(df["information_condition"].dropna().astype(str).unique()))
        if len(table_ics) > 1:
            errors.append(
                f"Stage7 action table mixes information conditions {table_ics}; "
                f"one run = one IC (contract v4 §6.1); multi-IC studies are separate runs"
            )
        if meta_ic is None:
            warnings.append("Stage7 metadata lacks information_condition (legacy pre-2026-07-04 run)")
        elif table_ics and str(meta_ic) not in table_ics:
            errors.append(
                f"Stage7 metadata information_condition={meta_ic!r} does not match "
                f"the action table {table_ics}"
            )

    # --- 2026-07-04 addendum: IC-c firm-name exposure + icc probe (LLM789-008/009) ---
    addendum = meta.get("stage7_contract_addendum")
    if addendum:
        if str(meta_ic) == "IC-c" and not meta.get("ic_c_firm_name_exposed"):
            errors.append(
                "Stage7 IC-c run under the 2026-07-04 addendum must expose the firm "
                "name (ic_c_firm_name_exposed=True; LLM789-008)"
            )
        if meta.get("icc_probe_enabled"):
            probe_cols = ["icc_probe_response_raw", "icc_probe_value", "icc_contamination_flag"]
            for col in probe_cols:
                if col not in df.columns:
                    errors.append(f"Stage7 icc_probe_enabled run missing action-table column: {col}")
            try:
                audit_cols = list(pd.read_csv(d / "llm_stage7_failure_audit.csv", nrows=0).columns)
                for col in probe_cols:
                    if col not in audit_cols:
                        errors.append(f"Stage7 icc_probe_enabled run missing failure-audit column: {col}")
            except Exception as exc:
                errors.append(f"Stage7 failure-audit probe-column check failed: {exc}")
    elif str(meta_ic) == "IC-c":
        warnings.append(
            "Legacy IC-c run predates the 2026-07-04 firm-name addendum; results "
            "are NOT comparable to firm-name IC-c runs (LLM789-008)"
        )

    return {
        "rows": int(len(df)) if not df.empty else 0,
        "metadata_status": meta.get("status"),
        "backend_id": meta.get("backend_id"),
        "backend_is_live": meta.get("backend_is_live"),
    }


def verify_stage8(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    """Verify Stage 8 LLM multi-oracle evaluation outputs.

    Checks:

    * Required output files exist and are non-empty.
    * Stage 8 metadata hashes match the active candidate-library /
      action-contract hashes.
    * Stage 8 metadata records ``stage6_substrate_hashes`` (proof that the
      same Oracle backend artifacts Stage 6 used were used here).
    * The Stage 8 score frame's ``policy`` column is restricted to
      {C4, C4R, C5, C6, C6X, C7, C8} — no Stage 6 ladder rows may leak into Stage 8.
    * ``delta_R_score_{alpha,beta,gamma}`` columns are present.
    """
    from credit_recourse.contracts.stage_paths import stage_dir as _stage_dir

    d = _stage_dir(root, "stage8")
    for f in [
        "llm_stage8_multi_oracle_scores.parquet",
        "llm_stage8_policy_summary.csv",
        "simulated_oracle_input_frame.parquet",
        "llm_stage8_failure_audit_enriched.csv",
        "failure_coder_manifest.json",
        "metadata.json",
    ]:
        nonempty(d / f, errors)

    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    for k in [
        "candidate_library_hash", "candidate_library_path", "candidate_action_values_source",
        "candidate_library_quantile", "selected_recalibrated_candidate_library_hash",
        "base_candidate_library_hash",
        "final_action_contract_hash",
        "stage6_substrate_hashes",
        "stage6_simulator_identity",
        "sim_business_plan_mode",
        "preserve_current_non_current_residual",
        "predicted_fiscal_year",
    ]:
        if k not in meta:
            errors.append(f"Stage8 metadata missing required key: {k}")
    if meta.get("candidate_action_values_source") != "stage2_recalibrated_candidate_library":
        errors.append("Stage8 must score LLM actions using Stage2 recalibrated candidate vectors (P50), not active base YAML")
    try:
        q = int(meta.get("candidate_library_quantile"))
        if q != 50:
            errors.append(f"Stage8 candidate_library_quantile must be 50 for final LLM/RL comparable runs, got {q}")
        p50_path = resolve_candidate_library_path(root, magnitude_quantile=q)
        p50_space = load_action_space(root, candidate_library_path=p50_path)
        if meta.get("candidate_library_hash") != p50_space.candidate_library_hash:
            errors.append(
                f"Stage8 metadata candidate_library_hash must match selected P{q} library: "
                f"meta={meta.get('candidate_library_hash')}, expected={p50_space.candidate_library_hash}"
            )
    except Exception as exc:
        errors.append(f"Stage8 selected candidate-library lineage check failed: {exc}")
    if not isinstance(meta.get("stage6_substrate_hashes"), dict):
        errors.append("Stage8 metadata stage6_substrate_hashes must be a dict")
    if not isinstance(meta.get("stage6_simulator_identity"), dict):
        errors.append("Stage8 metadata stage6_simulator_identity must be a dict")
    else:
        ident = meta.get("stage6_simulator_identity") or {}
        for k in ["sim_business_plan_mode", "preserve_current_non_current_residual", "predicted_fiscal_year"]:
            if k not in ident:
                errors.append(f"Stage8 stage6_simulator_identity missing key: {k}")
            elif meta.get(k) != ident.get(k):
                errors.append(f"Stage8 top-level simulator identity mismatch for {k}: top={meta.get(k)!r}, identity={ident.get(k)!r}")
    if not meta.get("scored_via_simulator_only"):
        errors.append("Stage8 metadata must record scored_via_simulator_only=True (must not score raw s_{t+1})")
    if not isinstance(meta.get("failure_coder_manifest"), dict):
        errors.append("Stage8 metadata must record failure_coder_manifest dict")
    else:
        fcm = meta.get("failure_coder_manifest") or {}
        if fcm.get("oracle_scores_used_for_failure_coding") is not False:
            errors.append("Stage8 failure coding must not use Oracle scores")

    # Enriched failure audit must carry the executable 8-taxonomy post-sim columns.
    try:
        fa = pd.read_csv(d / "llm_stage8_failure_audit_enriched.csv")
        for col in [
            "failure_categories", "feasibility_violation_auto", "feasibility_review_needed",
            "plug_to_assets", "plug_denominator_source", "accounting_check_failed", "negative_balance_flag",
            "residual_presentation_repair_flag", "plug_to_assets_review_exceeded",
            "plug_to_assets_hard_exceeded", "feasibility_core_violation_flag",
            "failure_coder_version", "oracle_scores_used_for_failure_coding",
        ]:
            if col not in fa.columns:
                errors.append(f"Stage8 enriched failure audit missing column: {col}")
        if "oracle_scores_used_for_failure_coding" in fa.columns and fa["oracle_scores_used_for_failure_coding"].astype(str).str.lower().isin(["true", "1"]).any():
            errors.append("Stage8 enriched failure audit indicates Oracle scores were used for failure coding")
        if {"feasibility_violation_auto", "feasibility_core_violation_flag"}.issubset(fa.columns):
            auto = fa["feasibility_violation_auto"].fillna(False).astype(bool)
            core = fa["feasibility_core_violation_flag"].fillna(False).astype(bool)
            if (auto != core).any():
                errors.append("Stage8 feasibility audit contract violation: automatic feasibility failures must equal core post-sim feasibility violations")
            if len(fa) > 0 and auto.all() and not core.all():
                errors.append("Stage8 feasibility audit flagged every row without every row carrying a core post-sim feasibility violation")
    except Exception as exc:
        errors.append(f"Stage8 enriched failure audit read/check failed: {exc}")

    df = read_parquet(d / "llm_stage8_multi_oracle_scores.parquet", errors, "llm_stage8_multi_oracle_scores")
    if not df.empty:
        if "policy" in df.columns:
            pols = set(df["policy"].astype(str).unique())
            forbidden_pols = pols - {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"}
            if forbidden_pols:
                errors.append(f"Stage8 score frame contains non-LLM policy codes (Stage 6 ladder must not be re-scored here): {sorted(forbidden_pols)}")
        for bk in ["alpha", "beta", "gamma"]:
            if f"delta_R_score_{bk}" not in df.columns:
                errors.append(f"Stage8 score frame missing delta_R_score_{bk}")
        if "row_id" in df.columns and "policy" in df.columns:
            # When both LLM modes are present in one run, (row_id, policy) is
            # not the primary key — (row_id, policy, mode) is.  Honor the
            # mode column if it exists; otherwise fall back to (row_id, policy).
            dedup_keys = ["row_id", "policy"] + (["mode"] if "mode" in df.columns else [])
            n_dupes = int(df.duplicated(subset=dedup_keys).sum())
            if n_dupes > 0:
                errors.append(f"Stage8 score frame has {n_dupes} duplicate {tuple(dedup_keys)} rows")

    return {
        "rows": int(len(df)) if not df.empty else 0,
        "metadata_status": meta.get("status"),
    }


def verify_stage9(root: Path, errors: list[str], warnings: list[str]) -> dict[str, Any]:
    """Verify Stage 9 LLM–RL comparison outputs.

    Checks:

    * Required output files exist and are non-empty.
    * Stage 9 metadata hashes match the active configs.
    * ``c8_within_case_revision_metrics_undefined=True`` in metadata
      (LLM contract §8 fail-fast).
    * No C8 row in the revision metrics has ``metrics_defined=True``.
    * Comparison frame contains both Stage 6 ladder policies and Stage 8 LLM
      policies (so the comparison is paired).
    """
    from credit_recourse.contracts.stage_paths import stage_dir as _stage_dir

    d = _stage_dir(root, "stage9")
    for f in [
        "llm_stage9_llm_rl_comparison.parquet",
        "llm_stage9_llm_rl_comparison.csv",
        "llm_stage9_policy_summary.csv",
        "llm_stage9_revision_metrics.csv",
        "llm_stage9_identity_contrast.csv",
        "llm_stage9_failure_audit.csv",
        "metadata.json",
    ]:
        nonempty(d / f, errors)

    meta = metadata_status(d / "metadata.json", errors, allow_prefix=("PASS",))
    for k in [
        "candidate_library_hash", "candidate_library_path", "candidate_action_values_source",
        "candidate_library_quantile", "selected_recalibrated_candidate_library_hash",
        "base_candidate_library_hash", "final_action_contract_hash",
        "stage6_multi_oracle_eval_consumed", "stage8_multi_oracle_scores_consumed",
        "identity_contrast_row_count",
    ]:
        if k not in meta:
            errors.append(f"Stage9 metadata missing required key: {k}")
    if meta.get("candidate_action_values_source") != "stage2_recalibrated_candidate_library":
        errors.append("Stage9 revision metrics must use Stage2 recalibrated candidate vectors (P50), not active base YAML")
    try:
        q = int(meta.get("candidate_library_quantile"))
        if q != 50:
            errors.append(f"Stage9 candidate_library_quantile must be 50 for final LLM/RL comparable runs, got {q}")
        p50_path = resolve_candidate_library_path(root, magnitude_quantile=q)
        p50_space = load_action_space(root, candidate_library_path=p50_path)
        if meta.get("candidate_library_hash") != p50_space.candidate_library_hash:
            errors.append(
                f"Stage9 metadata candidate_library_hash must match selected P{q} library: "
                f"meta={meta.get('candidate_library_hash')}, expected={p50_space.candidate_library_hash}"
            )
    except Exception as exc:
        errors.append(f"Stage9 selected candidate-library lineage check failed: {exc}")
    if not meta.get("c8_within_case_revision_metrics_undefined"):
        errors.append("Stage9 metadata must record c8_within_case_revision_metrics_undefined=True (LLM contract §8)")

    cmp_df = read_parquet(d / "llm_stage9_llm_rl_comparison.parquet", errors, "llm_stage9_llm_rl_comparison")
    if not cmp_df.empty and "policy" in cmp_df.columns:
        pols = set(cmp_df["policy"].astype(str).unique())
        has_llm = any(p in {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"} for p in pols)
        has_baseline = any(p in {"C0_noop", "C_obs"} for p in pols)
        if not has_llm:
            errors.append("Stage9 comparison frame contains no LLM policies")
        if not has_baseline:
            errors.append("Stage9 comparison frame contains no baseline (C0_noop/C_obs) policies; comparison is unanchored")

    rev_path = d / "llm_stage9_revision_metrics.csv"
    if rev_path.exists() and rev_path.stat().st_size > 0:
        try:
            rev_df = pd.read_csv(rev_path)
            if "base_condition" in rev_df.columns and "metrics_defined" in rev_df.columns:
                c8 = rev_df[rev_df["base_condition"].astype(str) == "C8"]
                if (c8["metrics_defined"].fillna(False).astype(bool)).any():
                    errors.append("Stage9 revision_metrics has C8 row(s) with metrics_defined=True (LLM contract §8)")
            if "revision_condition" in rev_df.columns:
                revision_conditions = set(rev_df["revision_condition"].astype(str))
                if "C4R" in revision_conditions:
                    c4r = rev_df[rev_df["revision_condition"].astype(str).eq("C4R")]
                    if not c4r["base_condition"].astype(str).eq("C4").all():
                        errors.append("Stage9 C4R rows must be paired with base_condition='C4'")
                    if "reference_source" in c4r.columns and not c4r["reference_source"].astype(str).eq("none").all():
                        errors.append("Stage9 C4R rows must retain reference_source='none'")
                    if "metrics_defined" in c4r.columns and c4r["metrics_defined"].fillna(False).astype(bool).any():
                        errors.append("Stage9 C4R RL-adoption geometry must remain undefined without an RL reference")
                    if "undefined_reason" in c4r.columns and not c4r["undefined_reason"].astype(str).eq("no_rl_reference").all():
                        errors.append("Stage9 C4R rows must use undefined_reason='no_rl_reference'")
                if "C6X" in revision_conditions:
                    if int(meta.get("identity_contrast_row_count") or 0) <= 0:
                        errors.append("Stage9 metadata reports no identity contrast rows even though C6X revision metrics exist")
                    ident_path = d / "llm_stage9_identity_contrast.csv"
                    ident_df = pd.read_csv(ident_path) if ident_path.exists() and ident_path.stat().st_size > 0 else pd.DataFrame()
                    for col in ["identity_adoption_gap", "identity_retention_gap", "identity_drift_gap"]:
                        if col not in ident_df.columns:
                            errors.append(f"Stage9 identity contrast missing column: {col}")
        except Exception as exc:
            errors.append(f"Stage9 revision_metrics CSV read failed: {exc}")

    return {
        "rows": int(len(cmp_df)) if not cmp_df.empty else 0,
        "metadata_status": meta.get("status"),
    }



ALL_STAGE_NAMES = [
    "stage0", "stage1", "stage1_bridge", "stage2_input", "stage2",
    "stage3", "stage4", "stage5", "stage6_actions", "stage6",
    "stage7", "stage8", "stage9",
]

def verify(root: Path, stage: str) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    if stage == "stage0":
        checks = verify_stage0(root, errors, warnings)
    elif stage == "stage1":
        checks = verify_stage1(root, errors, warnings)
    elif stage == "stage1_bridge":
        checks = verify_stage1_bridge(root, errors, warnings)
    elif stage == "stage2_input":
        checks = verify_stage2_input(root, errors, warnings)
    elif stage == "stage2":
        checks = verify_stage2(root, errors, warnings)
    elif stage == "stage2_substrate_loopA_loopB2":
        checks = verify_stage2_substrate_loopA_loopB2(root, errors, warnings)
    elif stage == "stage3":
        checks = verify_stage3(root, errors, warnings)
    elif stage == "stage4":
        checks = verify_stage4(root, errors, warnings)
    elif stage == "stage5":
        checks = verify_stage5(root, errors, warnings)
    elif stage == "stage6_actions":
        checks = verify_stage6_actions(root, errors, warnings)
    elif stage == "stage6":
        checks = verify_stage6(root, errors, warnings)
    elif stage == "stage7":
        checks = verify_stage7(root, errors, warnings)
    elif stage == "stage8":
        checks = verify_stage8(root, errors, warnings)
    elif stage == "stage9":
        checks = verify_stage9(root, errors, warnings)
    else:
        raise ValueError(f"Unknown stage verifier: {stage}")
    return {
        "stage_name": f"verify_{stage}",
        "contract_version": "stage_boundary_contract_v1_runner_paths",
        "created_utc": now(),
        "status": "PASS" if not errors else "FAIL",
        "sweep_claim_mode": "strict_full_sweep_claim" if STRICT_FULL_SWEEP_CLAIM else "fixed_config_boundary_check",
        "strict_full_sweep_claim": bool(STRICT_FULL_SWEEP_CLAIM),
        "canonical_stage_dirs": CANONICAL_STAGE_DIRS,
        "deprecated_stage_dir_aliases": DEPRECATED_STAGE_DIR_ALIASES,
        "errors": errors,
        "warnings": warnings,
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--stage", default="all", help="Stage to verify, or 'all' to run the full boundary suite")
    ap.add_argument(
        "--strict-full-sweep-claim",
        action="store_true",
        help=(
            "Promote REGISTERED_NOT_RUN sensitivity-grid rows to errors. "
            "Use only when claiming a completed full sensitivity sweep."
        ),
    )
    args = ap.parse_args(argv)
    global STRICT_FULL_SWEEP_CLAIM
    STRICT_FULL_SWEEP_CLAIM = bool(args.strict_full_sweep_claim)
    root = Path(args.project_root).resolve()
    ledger_dir = final_root(root) / "ledgers"
    if str(args.stage).lower() == "all":
        results = [verify(root, stage) for stage in ALL_STAGE_NAMES]
        result = {
            "stage_name": "verify_all",
            "contract_version": "stage_boundary_contract_v1_runner_paths",
            "created_utc": now(),
            "status": "PASS" if all(r.get("status") == "PASS" for r in results) else "FAIL",
            "sweep_claim_mode": "strict_full_sweep_claim" if STRICT_FULL_SWEEP_CLAIM else "fixed_config_boundary_check",
            "strict_full_sweep_claim": bool(STRICT_FULL_SWEEP_CLAIM),
            "stages": results,
            "errors": {r.get("stage_name", "unknown"): r.get("errors", []) for r in results if r.get("errors")},
            "warnings": {r.get("stage_name", "unknown"): r.get("warnings", []) for r in results if r.get("warnings")},
        }
        out = ledger_dir / "verify_all.json"
    else:
        result = verify(root, args.stage)
        out = ledger_dir / f"verify_{args.stage}.json"
    write_json(out, result)
    # Keep CLI output ASCII-escaped so Windows PowerShell 5.x can pipe/capture
    # verifier JSON without corrupting Korean column names into invalid JSON.
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
