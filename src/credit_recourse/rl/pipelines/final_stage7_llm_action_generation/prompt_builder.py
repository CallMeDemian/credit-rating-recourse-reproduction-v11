from __future__ import annotations

"""Prompt builder for Stage 7 LLM action generation.

Constructs structured firm-state prompts from ``phase_eval_candidate.parquet``
rows.  Honors the research design's information conditions (RQ5):

* **IC-a** — anonymous tabular state only.  No industry, no year, no firm id.
* **IC-b** — IC-a plus industry class and fiscal year.
* **IC-c** — IC-b plus the firm identifier AND the firm name (named-firm;
  research decision 2026-07-04, LLM789-008).  Recorded with a contamination
  flag because the LLM may have prior knowledge of named firms.  When IC-c is
  requested, the Stage 7 pipeline injects ``firm_name`` from a frozen lookup
  and HARD-FAILS if any evaluation firm lacks a name.

The builder also embeds the full v32 candidate library (label, controllability
tier, mechanical effect summary) so the LLM is constrained to a known
vocabulary.  For reference conditions (C6/C6X/C7/C8) a source-blind reference candidate is
included. C6/C7/C8 use the RL reference; C6X uses a seeded random in-vocabulary reference.

Feature selection is deliberately conservative: only features defined in the
AVS256 manifest plus a small set of nonfinancial context variables are
exposed.  No Oracle outputs, item scores, boundaries, or evaluation-stage
artifacts are ever exposed to the LLM (per RESEARCH §4 backend isolation).
"""

import json
from typing import Any

import pandas as pd

from .budget_contract import budget_applies, normalize_budget_contract

# Features safe to expose to the LLM under IC-a.  These are all derived
# financial-ratio and trend features from the AVS256 contract; they include
# no rating, no Oracle score, and no evaluation-stage artifact.
IC_A_DERIVED_FEATURES = [
    "derived__debt_to_assets",
    "derived__equity_to_assets",
    "derived__current_ratio",
    "derived__cash_ratio",
    "derived__operating_margin",
    "derived__gross_margin",
    "derived__net_margin",
    "derived__roa_proxy",
    "derived__cogs_to_revenue",
    "derived__sga_to_revenue",
    "derived__financial_cost_to_revenue",
    "derived__capex_to_revenue",
    "derived__ppe_to_assets",
    "derived__short_debt_to_total_debt",
    "derived__long_debt_to_total_debt",
    "derived__bond_to_total_debt",
    "derived__inventory_to_revenue",
    "derived__receivables_to_revenue",
    "derived__payables_to_revenue",
    "delta_1y__derived__operating_margin",
    "delta_1y__derived__debt_to_assets",
    "delta_1y__derived__current_ratio",
    "delta_1y__derived__roa_proxy",
]

# IC-b adds these context fields.
IC_B_CONTEXT_FEATURES = [
    "industry_class",
    "market",
    "year",
    "fiscal_year",
    "sector_7",
    "log_assets",
    "nf_log_assets",
]

# IC-c additionally exposes the firm identifier and — per the 2026-07-04
# research decision (LLM789-008) — the firm NAME.  The contamination caution
# is recorded in metadata; fields are included as-is when present.  The
# Stage 7 pipeline is responsible for materializing ``firm_name`` onto the
# serving panel before prompt construction and fail-fasts on missing names.
IC_C_ID_FEATURES = [
    "firm_id",
    "corp_code",
    "company_id",
    "firm_name",
    "회사명",
]

# Forbidden fields — never exposed to the LLM regardless of IC.  These are
# Oracle outputs and any field whose presence would leak the evaluator into
# the policy.
FORBIDDEN_FIELDS = {
    "R_score_alpha", "R_score_beta", "R_score_gamma",
    "delta_R_score_alpha", "delta_R_score_beta", "delta_R_score_gamma",
    "rating_num", "rating_grade", "grade_base_10", "rating_num_10",
    "grade_base_notch", "rating_num_notch", "grade_base_raw",
    "phi_t", "phi_tplusH", "delta_phi", "reward_train", "reward_raw",
    "candidate_id",  # The Stage 2 projected label; would leak the answer.
    "projected_candidate_id", "nearest_candidate_id",
    "soft_cand_id_1", "soft_cand_id_2", "soft_cand_id_3",
    "soft_cand_prob_1", "soft_cand_prob_2", "soft_cand_prob_3",
}


def _safe_value(v: Any) -> Any:
    """Convert pandas/NumPy types to JSON-safe values."""
    if v is None:
        return None
    if isinstance(v, float):
        if v != v:  # NaN
            return None
        return float(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, bool):
        return bool(v)
    if hasattr(v, "item"):
        try:
            return _safe_value(v.item())
        except Exception:
            pass
    try:
        return str(v)
    except Exception:
        return None


def _extract_firm_state(row: pd.Series, information_condition: str) -> dict:
    """Extract the firm-state dict for one row per the IC level."""
    state: dict = {}

    for feat in IC_A_DERIVED_FEATURES:
        if feat in row.index and feat not in FORBIDDEN_FIELDS:
            state[feat] = _safe_value(row[feat])

    if information_condition in {"IC-b", "IC-c"}:
        for feat in IC_B_CONTEXT_FEATURES:
            if feat in row.index and feat not in FORBIDDEN_FIELDS:
                state[feat] = _safe_value(row[feat])

    if information_condition == "IC-c":
        for feat in IC_C_ID_FEATURES:
            if feat in row.index and feat not in FORBIDDEN_FIELDS:
                state[feat] = _safe_value(row[feat])

    return state


def _candidate_library_summary(space) -> dict:
    """Compact v32 library summary exposed to the LLM.

    Includes label, controllability tier (if present in the candidate
    definition), and a per-dimension mechanical effect dict.

    Uses :meth:`ActionSpace.candidate_vector` so the underlying dict-key
    convention in ``space.fixed_candidates`` (which stores
    ``action__``-prefixed keys) is honored.
    """
    summary: dict = {"main_train_labels": list(space.train_labels), "candidates": {}}
    cols_full = list(space.columns)
    cols_bare = [c.replace("action__", "") for c in cols_full]
    for name in space.train_labels:
        cand = space.fixed_candidates.get(name, {})
        vec = space.candidate_vector(name)
        active = {
            bare: float(vec[i])
            for i, bare in enumerate(cols_bare)
            if abs(float(vec[i])) > 0
        }
        summary["candidates"][name] = {
            "tier": cand.get("tier"),
            "paper_role": cand.get("paper_role"),
            "active_dimensions": active,
        }
    return summary


def build_prompt(
    row: pd.Series,
    condition: str,
    mode: str,
    information_condition: str,
    space,
    rl_reference_candidate: str | None = None,
    reference_source: str = "none",
    reference_draw_seed: int | None = None,
    initial_action: dict | None = None,
    action_budget_contract: dict | None = None,
) -> dict:
    """Build the structured prompt payload for one Stage 7 request.

    Parameters
    ----------
    row : pd.Series
        One row from ``phase_eval_candidate.parquet``.
    condition : str
        Policy condition code (``C4``/``C4R``/``C5``/``C6``/``C6X``/``C7``/``C8``).
    mode : str
        ``"candidate_selection"`` or ``"free_form_10d"``.
    information_condition : str
        ``IC-a`` / ``IC-b`` / ``IC-c``.
    space : ActionSpace
        Loaded action space (provides v32 vocabulary).
    rl_reference_candidate : str | None
        For C6/C7/C8: the v32 candidate selected by the RL policy for this row. For C6X: the seeded random in-vocabulary reference.
    initial_action : dict | None
        For C4R/C6/C6X/C7: the LLM's pre-revision action ``a_0``.
    action_budget_contract : dict | None
        Optional LLM-facing free-form L1 budget contract for N5-style pilot runs.

    Returns
    -------
    dict
        Structured prompt payload with keys
        ``firm_state``, ``candidate_library``, ``condition``, ``mode``,
        ``information_condition``, ``rl_reference_candidate``,
        ``reference_source``, ``reference_draw_seed``, ``initial_action``, ``instructions``.
    """
    if condition not in {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"}:
        raise ValueError(f"Unsupported Stage 7 condition: {condition}")
    if mode not in {"candidate_selection", "free_form_10d"}:
        raise ValueError(f"Unsupported Stage 7 mode: {mode}")
    if information_condition not in {"IC-a", "IC-b", "IC-c"}:
        raise ValueError(f"Unsupported information condition: {information_condition}")
    if reference_source not in {"rl", "random", "none"}:
        raise ValueError(f"Unsupported reference_source: {reference_source}")
    expected_source = "random" if condition == "C6X" else ("rl" if condition in {"C6", "C7", "C8"} else "none")
    if reference_source != expected_source:
        raise ValueError(
            f"Condition {condition} requires reference_source={expected_source!r}; got {reference_source!r}."
        )
    if condition == "C6X" and reference_draw_seed is None:
        raise ValueError("C6X requires non-null reference_draw_seed.")
    if condition != "C6X" and reference_draw_seed is not None:
        raise ValueError(f"Only C6X may carry reference_draw_seed; got condition={condition}.")

    normalized_budget_contract = normalize_budget_contract(action_budget_contract)
    budget_contract_applies = budget_applies(
        normalized_budget_contract, condition=condition, mode=mode
    )

    firm_state = _extract_firm_state(row, information_condition)

    # Cross-check forbidden fields are not present even if they appear in IC
    # feature lists by accident.
    leaked = [k for k in firm_state.keys() if k in FORBIDDEN_FIELDS]
    if leaked:
        raise ValueError(
            f"Prompt construction leaked forbidden fields into firm_state: {leaked}"
        )

    instructions: list[str] = []
    if mode == "candidate_selection":
        instructions.append(
            "Select exactly one candidate from main_train_labels and explain "
            "your reasoning.  Your selection must be a string equal to one of "
            "the v32 main labels."
        )
    else:
        instructions.append(
            "Emit a continuous 10D action vector covering all of: "
            "ppe_pct, inv_turnover_chg, ar_turnover_chg, ap_turnover_chg, "
            "short_debt_pct, long_debt_pct, bond_pct, revenue_growth, "
            "cogs_ratio_chg, sga_ratio_chg.  Values must respect the bounds "
            "implied by the v32 candidate library."
        )
        if budget_contract_applies:
            assert normalized_budget_contract is not None
            instructions.append(
                "N5 action-budget contract: the sum of absolute values across "
                "the 10 action_vector components must be <= "
                f"{float(normalized_budget_contract['l1_budget']):.12g}. "
                "Prefer a feasible budget-compliant action over a larger "
                "unconstrained action. This is a generation-time output "
                "contract, not a post-hoc rescale."
            )
    if condition in {"C5", "C7"}:
        instructions.append(
            "Reasoning-first: produce a structured diagnosis of the firm's "
            "weakest dimension(s) before recommending the recourse program."
        )
    if condition == "C4R":
        instructions.append(
            "You have already produced an initial action ``a_0``.  Re-examine "
            "that same draft without any external reference.  You may retain, "
            "revise, or reject parts of your own draft, but use only the provided "
            "firm state and do not infer or invent an outside recommendation."
        )
    if condition in {"C6", "C6X", "C7"}:
        instructions.append(
            "You have already produced an initial action ``a_0``; a reference "
            "candidate is now shown.  Revise your action.  You may adopt, "
            "reject, or partially incorporate the shown reference."
        )
    if condition == "C8":
        instructions.append(
            "A reference candidate is shown before you produce any "
            "action.  Produce a single action.  No initial ``a_0`` exists, so "
            "within-case revision metrics are not defined for this condition."
        )

    payload = {
        "row_id": int(row.get("row_id", -1)) if "row_id" in row.index else None,
        "condition": condition,
        "mode": mode,
        "information_condition": information_condition,
        "named_firm_contamination_caution": (information_condition == "IC-c"),
        "firm_state": firm_state,
        "candidate_library": _candidate_library_summary(space),
        "rl_reference_candidate": rl_reference_candidate,
        "reference_source": reference_source,
        "reference_draw_seed": reference_draw_seed,
        "initial_action": initial_action,
        "action_budget_contract": normalized_budget_contract if budget_contract_applies else None,
        "instructions": instructions,
    }
    return payload


def build_prompts_for_panel(
    panel: pd.DataFrame,
    conditions: list[str],
    modes: list[str],
    information_condition: str,
    space,
    rl_reference: dict[int, str] | None = None,
    c6x_reference: dict[int, str] | None = None,
    reference_draw_seed: int | None = None,
) -> list[dict]:
    """Build the full list of (row, condition, mode) prompt records.

    Parameters
    ----------
    panel : pd.DataFrame
        ``phase_eval_candidate.parquet`` rows.  Must have a ``row_id`` column
        or be reset-indexed so the row index can serve as ``row_id``.
    conditions : list[str]
        Subset of ``{C4, C4R, C5, C6, C6X, C7, C8}``.
    modes : list[str]
        Subset of ``{candidate_selection, free_form_10d}``.
    information_condition : str
    space : ActionSpace
    rl_reference : dict[int, str] | None
        For RL-reference conditions (C6/C7/C8), map ``row_id`` → RL reference
        candidate.  Required when any revision condition is requested.

    Returns
    -------
    list[dict]
        List of prompt payloads.  C4R/C6/C7 prompts have ``initial_action=None``
        at this stage; the pipeline fills in ``a_0`` after the corresponding
        C4/C5 generation pass.
    """
    if not all(c in {"C4", "C4R", "C5", "C6", "C6X", "C7", "C8"} for c in conditions):
        raise ValueError(f"Unsupported conditions: {conditions}")
    revision_conditions = {c for c in conditions if c in {"C6", "C7", "C8"}}
    if "C6X" in conditions and (c6x_reference is None or reference_draw_seed is None):
        raise ValueError("C6X prompt construction requires c6x_reference and reference_draw_seed.")
    if revision_conditions and rl_reference is None:
        raise ValueError(
            f"Conditions {revision_conditions} require an rl_reference map."
        )

    if "row_id" not in panel.columns:
        work = panel.reset_index(drop=True).copy()
        work["row_id"] = work.index
    else:
        work = panel.copy()

    prompts: list[dict] = []
    for _, row in work.iterrows():
        rid = int(row["row_id"])
        rl_ref = rl_reference.get(rid) if rl_reference is not None else None
        random_ref = c6x_reference.get(rid) if c6x_reference is not None else None
        for cond in conditions:
            if cond == "C6X":
                ref_for_cond = random_ref
                reference_source = "random"
                ref_seed_for_cond = int(reference_draw_seed) if reference_draw_seed is not None else None
            elif cond in {"C6", "C7", "C8"}:
                ref_for_cond = rl_ref
                reference_source = "rl"
                ref_seed_for_cond = None
            else:
                ref_for_cond = None
                reference_source = "none"
                ref_seed_for_cond = None
            if cond in {"C6", "C6X", "C7", "C8"} and ref_for_cond is None:
                raise ValueError(
                    f"Missing reference for row {rid} required by {cond}."
                )
            for mode in modes:
                prompt = build_prompt(
                    row=row,
                    condition=cond,
                    mode=mode,
                    information_condition=information_condition,
                    space=space,
                    rl_reference_candidate=ref_for_cond,
                    reference_source=reference_source,
                    reference_draw_seed=ref_seed_for_cond,
                    initial_action=None,
                )
                prompts.append(prompt)
    return prompts


def serialize_manifest(prompts: list[dict], backend_manifest: dict) -> str:
    """Serialize the full Stage 7 prompt manifest to JSON.

    The manifest is the auditable record of what was asked of the LLM.  It
    includes every prompt, the backend identity, and a deterministic prompt
    hash for reproducibility.
    """
    return json.dumps(
        {
            "backend": backend_manifest,
            "prompt_count": len(prompts),
            "prompts": prompts,
        },
        ensure_ascii=False,
        indent=2,
    )


# ---------------------------------------------------------------------------
# IC-c contamination probe — numeric recall v2 (LLM789-009; contract v4
# icc_probe columns; RESEARCH design "IC-c memorization probe")
# ---------------------------------------------------------------------------

ICC_PROBE_SCHEMA_VERSION = "icc_probe_numeric_recall_v2"

# The single 2024 statement item the probe asks the model to recall.  Compared
# offline against this serving-panel column under a pre-registered relative
# tolerance; never shown to the model.
ICC_PROBE_TARGET_FEATURE = "derived__debt_to_assets"


def build_icc_probe_prompt(row: pd.Series, information_condition: str) -> dict:
    """Build the IC-c contamination probe payload for one firm (v2).

    v2 supersedes the v1 self-report probe (update history: LLM789-009).  Per
    the research design's "IC-c memorization probe": the model is asked to
    state a specific fiscal-2024 statement item — the debt-to-assets ratio —
    for the NAMED firm from its own pretraining knowledge, before any
    financial state is shown.  The parsed numeric answer is compared offline
    with the serving-panel value under a pre-registered relative tolerance;
    the result is the per-firm ``icc_contamination_flag``.  The thesis
    proposal (§3.2-4, H5) treats IC-c contamination as an interpretive
    limitation and keeps it outside the failure taxonomy — this probe is the
    optional opt-in diagnostic that quantifies that limitation.

    Self-reported recognition/familiarity are retained as SECONDARY fields in
    the same call (no extra API cost); they never drive the contamination
    flag.

    Expected model output (JSON only)::

        {"recalled_debt_ratio": <number|null>, "recognized": <true|false>,
         "familiarity": 0|1|2|3, "known_facts": [<=3 short strings]}

    Downstream columns (contract v4 names + v2 extensions)::

        icc_probe_response_raw   — raw text
        icc_probe_value          — recalled debt-to-assets ratio (float|null)
        icc_contamination_flag   — relative error <= pre-registered tolerance
        icc_probe_parse_error, icc_probe_rel_err, icc_probe_panel_value
    """
    if information_condition != "IC-c":
        raise ValueError(
            f"The IC-c probe is defined only for information_condition='IC-c'; got {information_condition!r}."
        )
    identity: dict = {}
    for feat in IC_C_ID_FEATURES + IC_B_CONTEXT_FEATURES:
        if feat in row.index and feat not in FORBIDDEN_FIELDS:
            v = _safe_value(row[feat])
            if v is not None:
                identity[feat] = v
    if not (identity.get("firm_name") or identity.get("회사명")):
        raise ValueError(
            "IC-c probe requires firm_name on the serving row; the Stage 7 "
            "pipeline must materialize it before probing (LLM789-008)."
        )
    system_prompt = (
        "You are being asked ONLY about your prior knowledge of a specific "
        "Korean listed company. No financial statements are provided and you "
        "must not infer from any data in this conversation. Respond with a "
        "single JSON object and nothing else: "
        "{\"recalled_debt_ratio\": <number|null>, \"recognized\": <true|false>, "
        "\"familiarity\": <0|1|2|3>, \"known_facts\": [<up to 3 short strings>]}. "
        "recalled_debt_ratio: the firm's fiscal-year 2024 total-debt to "
        "total-assets ratio as a decimal (for example 0.43) ONLY if you "
        "actually know it from pretraining; otherwise null. Do not guess or "
        "estimate a number. familiarity: 0 = never heard of it, 1 = name "
        "only, 2 = some concrete facts, 3 = detailed knowledge. known_facts: "
        "facts you believe you know from pretraining, not inferences."
    )
    user_prompt = json.dumps(
        {
            "probe_schema_version": ICC_PROBE_SCHEMA_VERSION,
            "task": "numeric_recall_contamination_probe",
            "target_item": "fiscal-2024 debt-to-assets ratio (total debt / total assets, decimal)",
            "firm_identity": identity,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "probe_schema_version": ICC_PROBE_SCHEMA_VERSION,
        "row_id": int(row.get("row_id", -1)) if "row_id" in row.index else None,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "firm_identity": identity,
    }
