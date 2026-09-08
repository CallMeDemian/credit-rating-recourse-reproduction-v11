from __future__ import annotations

"""Post-simulation enrichment of the Stage 7 LLM failure audit.

Stage 7 can judge syntax, grounding, direction and raw/projection magnitude.
Feasibility must be judged after the financial simulator has run.  This module
joins Stage 7's per-request audit with Stage 8 simulator diagnostics and emits
an enriched audit that keeps the closed eight-type taxonomy executable end to
end.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.failure_coder import (
    FEASIBILITY_RULE_VERSION,
    FAILURE_CODER_VERSION,
    ORACLE_SCORES_USED_FOR_FAILURE_CODING,
    code_feasibility_violation,
)
from credit_recourse.rl.common.io import write_json

ENRICHED_FAILURE_AUDIT_SCHEMA_VERSION = "stage8_failure_audit_enriched_v1"
FEASIBILITY_THRESHOLDS = {
    "plug_to_assets_hard": 0.10,
    "plug_to_assets_review": 0.05,
}

KEY_CANDIDATES = ["row_id", "policy", "mode"]


def _taxonomy_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, float) and pd.isna(value):
        return set()
    return {t for t in str(value).split(",") if t}


def _sustainability_rank(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, float) and pd.isna(value):
        return 0
    token = str(value).strip().lower()
    return {
        "ok": 1,
        "true": 1,
        "1": 1,
        "pass": 1,
        "passed": 1,
        "fragile": 2,
        "critical": 3,
        "false": 3,
        "0": 3,
        "fail": 3,
        "failed": 3,
        "error": 3,
        "bad": 3,
    }.get(token, 0)


def _worst_sustainability(values: Any) -> Any:
    """Aggregate the simulator's ordered sustainability labels by worst status.

    Pandas' string ``max`` is lexicographic, which ranks ``ok`` above
    ``fragile`` and ``critical``.  This preserves the simulator contract: critical
    > fragile > ok.
    """
    best_value = None
    best_rank = -1
    for value in values:
        rank = _sustainability_rank(value)
        if rank > best_rank:
            best_rank = rank
            best_value = value
    return best_value


def _diagnostic_by_key(sim_state: pd.DataFrame, action_effect_audit: pd.DataFrame) -> pd.DataFrame:
    """Return one row per Stage8 request key with simulator diagnostics.

    ``action_effect_audit`` can be empty for no-op-like rows because Stage6's
    audit records only nonzero action dimensions.  ``sim_state`` contains one
    row for every routed request and is therefore the primary source; audit rows
    are aggregated only to add preflight/accounting details where present.
    """
    key = [c for c in KEY_CANDIDATES if c in sim_state.columns]
    if "mode" not in key and "mode" in action_effect_audit.columns:
        key.append("mode")
    if not {"row_id", "policy"}.issubset(key):
        raise KeyError("Stage8 feasibility enrichment requires row_id and policy in simulated state.")

    cols = [c for c in [
        "row_id", "policy", "mode", "candidate_id", "sustainability", "plug_used", "plug_amount",
        "current_assets_before", "current_assets_after", "current_liabilities_before", "current_liabilities_after",
        "residual_negative_flag", "total_assets_before", "simulator_preflight_status", "accounting_check",
    ] if c in sim_state.columns]
    base = sim_state[cols].copy()
    if "simulator_preflight_status" not in base.columns:
        base["simulator_preflight_status"] = "ok"
    if "accounting_check" not in base.columns:
        base["accounting_check"] = "{}"

    if action_effect_audit is not None and not action_effect_audit.empty:
        audit_key = [c for c in key if c in action_effect_audit.columns]
        agg_map = {}
        for c in ["simulator_preflight_status", "accounting_check"]:
            if c in action_effect_audit.columns:
                agg_map[c] = "first"
        for c in ["residual_negative_flag", "plug_used"]:
            if c in action_effect_audit.columns:
                agg_map[c] = "max"
        if "sustainability" in action_effect_audit.columns:
            agg_map["sustainability"] = _worst_sustainability
        if agg_map and audit_key:
            agg = action_effect_audit.groupby(audit_key, dropna=False).agg(agg_map).reset_index()
            base = base.merge(agg, on=audit_key, how="left", suffixes=("", "__audit"))
            for c in ["simulator_preflight_status", "accounting_check", "residual_negative_flag", "sustainability", "plug_used"]:
                ac = f"{c}__audit"
                if ac in base.columns:
                    base[c] = base[ac].combine_first(base.get(c))
                    base.drop(columns=[ac], inplace=True)
    return base


def enrich_failure_audit(
    *,
    stage7_failure_audit: pd.DataFrame,
    simulated_state: pd.DataFrame,
    action_effect_audit: pd.DataFrame,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """Join Stage7 audit with post-sim feasibility flags and write artifacts."""
    out_dir = Path(out_dir)
    required = {"row_id", "policy", "mode", "failure_categories"}
    missing = sorted(required - set(stage7_failure_audit.columns))
    if missing:
        raise KeyError(f"Stage7 failure audit missing required columns for enrichment: {missing}")
    diag = _diagnostic_by_key(simulated_state, action_effect_audit)
    join_key = [c for c in ["row_id", "policy", "mode"] if c in stage7_failure_audit.columns and c in diag.columns]
    if not {"row_id", "policy"}.issubset(join_key):
        raise KeyError(f"Cannot join failure audit and simulator diagnostics; join_key={join_key}")
    merged = stage7_failure_audit.merge(diag, on=join_key, how="left", suffixes=("", "__sim"))
    if len(merged) != len(stage7_failure_audit):
        raise ValueError(
            f"Failure-audit enrichment changed row count: before={len(stage7_failure_audit)} after={len(merged)}"
        )

    records = []
    new_categories: list[str] = []
    for _, row in merged.iterrows():
        d = code_feasibility_violation(
            row.to_dict(),
            plug_to_assets_hard=FEASIBILITY_THRESHOLDS["plug_to_assets_hard"],
            plug_to_assets_review=FEASIBILITY_THRESHOLDS["plug_to_assets_review"],
        )
        records.append(d)
        cats = _taxonomy_set(row.get("failure_categories"))
        if d["feasibility_violation_auto"]:
            cats.add("feasibility_violation")
        new_categories.append(",".join(sorted(cats)))
    enrich = pd.DataFrame(records)
    out = pd.concat([merged.reset_index(drop=True), enrich.reset_index(drop=True)], axis=1)
    out["failure_categories_stage7"] = merged["failure_categories"].fillna("").astype(str)
    out["failure_categories"] = new_categories
    out["failure_count"] = out["failure_categories"].fillna("").astype(str).apply(lambda s: len([t for t in s.split(",") if t]))
    out["failure_coder_version"] = FAILURE_CODER_VERSION
    out["feasibility_rule_version"] = FEASIBILITY_RULE_VERSION
    out["oracle_scores_used_for_failure_coding"] = ORACLE_SCORES_USED_FOR_FAILURE_CODING

    path = out_dir / "llm_stage8_failure_audit_enriched.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")

    def _bool_count(col: str) -> int:
        if col not in out.columns:
            return 0
        return int(out[col].fillna(False).astype(bool).sum())

    meta = {
        "schema_version": ENRICHED_FAILURE_AUDIT_SCHEMA_VERSION,
        "failure_coder_version": FAILURE_CODER_VERSION,
        "feasibility_rule_version": FEASIBILITY_RULE_VERSION,
        "oracle_scores_used_for_failure_coding": ORACLE_SCORES_USED_FOR_FAILURE_CODING,
        "feasibility_thresholds": FEASIBILITY_THRESHOLDS,
        "row_count": int(len(out)),
        "auto_feasibility_violation_count": _bool_count("feasibility_violation_auto"),
        "core_feasibility_violation_count": _bool_count("feasibility_core_violation_flag"),
        "review_needed_count": _bool_count("feasibility_review_needed"),
        "residual_presentation_repair_count": _bool_count("residual_presentation_repair_flag"),
        "plug_to_assets_review_exceeded_count": _bool_count("plug_to_assets_review_exceeded"),
        "plug_to_assets_hard_exceeded_count": _bool_count("plug_to_assets_hard_exceeded"),
        "output": path.name,
    }
    write_json(out_dir / "failure_coder_manifest.json", meta)
    return out, meta
