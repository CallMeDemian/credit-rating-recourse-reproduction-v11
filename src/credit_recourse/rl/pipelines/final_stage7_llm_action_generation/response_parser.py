from __future__ import annotations

"""Stage 7 LLM response parser.

Responsibilities:

* Validate the parsed JSON against the mode-specific schema.
* Materialize the 10D action vector (candidate-selection mode → lookup;
  free-form mode → numeric vector with bound clipping).
* Project free-form vectors to the v32 vocabulary via the canonical
  ``project_actions_to_candidates`` helper.
* Classify any failure into the auditable taxonomy from RESEARCH §10:

  * ``direction_error`` — action direction contradicts the LLM's own
    diagnosis (e.g. claims leverage is the weakness but proposes capex).
  * ``magnitude_error`` — magnitude grossly off (we record but do not hard
    fail; bounds enforce upper limit and the projection distance captures
    further deviation).
  * ``feasibility_violation`` — action would violate accounting identities
    that the simulator must plug.
  * ``liquidity_destructive_recourse`` — action visibly cuts liquidity below
    a safe threshold.
  * ``translational_failure`` — JSON missing required fields or wrong shape.
  * ``structural_out_of_scope`` — recommendation describes an action not
    representable in the 10D simulator (mergers, governance, etc.).
  * ``anchoring_or_confirmation_failure`` — for revision conditions, the
    revision follows the RL reference verbatim with no diagnostic
    independence.
  * ``ungrounded_judgment`` — rationale text does not reference any field
    that appears in the firm_state.

The parser never silently drops a row; every failure produces a record in
the failure audit, and the row is either still routed to the simulator
(with a diagnostic flag) or recorded as ``structural_out_of_scope`` and
held out of Stage 8.
"""

import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from credit_recourse.rl.common.actions import (
    ActionSpace,
    project_actions_to_candidates,
)
from .llm_backends import LLMResponse
from .budget_contract import BUDGET_AUDIT_COLUMNS, budget_audit_for_action
from .failure_coder import (
    FAILURE_CODER_VERSION,
    ORACLE_SCORES_USED_FOR_FAILURE_CODING,
    code_direction_error,
    code_magnitude_error,
)


GROUNDING_MATCHER_VERSION = "stage7_grounding_alias_v2"


ALLOWED_FAILURE_TAXONOMY = {
    "direction_error",
    "magnitude_error",
    "feasibility_violation",
    "liquidity_destructive_recourse",
    "translational_failure",
    "structural_out_of_scope",
    "anchoring_or_confirmation_failure",
    "ungrounded_judgment",
}


@dataclass
class ParsedAction:
    """One Stage 7 parsed-and-validated action row.

    All Stage 7 outputs flow from instances of this dataclass; the pipeline
    converts a list of these into the ``llm_stage7_action_table.parquet``.
    """

    row_id: int
    policy: str  # C4 / C5 / C6 / C7 / C8
    mode: str  # "candidate_selection" or "free_form_10d"
    information_condition: str
    selected_candidate: str  # final v32 candidate id (after projection if free-form)
    free_form_action: dict[str, float] | None  # raw 10D before projection (free-form only)
    materialized_action: dict[str, float]  # final 10D after clipping/lookup
    rationale: str
    diagnosis: dict
    rl_reference_candidate: str | None
    reference_source: str
    reference_draw_seed: int | None
    bound_clipping: dict[str, dict]
    projection_distance: float | None
    projection_method: str | None
    out_of_library: bool | None
    failure_categories: list[str]
    routed_to_simulator: bool
    raw_response: str
    direction_error_auto: bool = False
    direction_review_needed: bool = False
    direction_error_reason: str = ""
    direction_rule_version: str = ""
    magnitude_error_auto: bool = False
    magnitude_review_needed: bool = False
    magnitude_error_reason: str = ""
    magnitude_rule_version: str = ""
    max_bound_violation_abs: float = 0.0
    action_l1_norm: float = 0.0
    action_nonzero_dim_count: int = 0
    failure_coder_version: str = FAILURE_CODER_VERSION
    oracle_scores_used_for_failure_coding: bool = ORACLE_SCORES_USED_FOR_FAILURE_CODING
    budget_contract_label: str | None = None
    budget_l1_target: float | None = None
    budget_l1_raw: float | None = None
    budget_l1_clipped: float | None = None
    budget_compliant_raw: bool | None = None
    budget_compliant_clipped: bool | None = None
    budgeted_condition_flag: bool = False


def _clip_to_bounds(
    vector: dict[str, float], space: ActionSpace
) -> tuple[dict[str, float], dict[str, dict]]:
    """Clip a 10D dict to ``final_action_contract`` bounds; record clipping."""
    cols = [c.replace("action__", "") for c in space.columns]
    bounded: dict[str, float] = {}
    audit: dict[str, dict] = {}
    for col in cols:
        # Bounds are keyed by ``action__<name>`` in space.bounds; map back.
        bkey = f"action__{col}" if f"action__{col}" in space.bounds else col
        lo, hi = space.bounds[bkey]
        raw = float(vector.get(col, 0.0) or 0.0)
        clipped = max(lo, min(hi, raw))
        bounded[col] = clipped
        if abs(clipped - raw) > 1e-12:
            audit[col] = {
                "raw_value": raw,
                "clipped_value": clipped,
                "bound_low": lo,
                "bound_high": hi,
            }
    return bounded, audit


def _materialize_from_candidate(
    candidate_id: str, space: ActionSpace
) -> dict[str, float]:
    """Look up the v32 vector for a candidate and return as a {name: value}
    dict in the canonical 10D order.

    Uses :meth:`ActionSpace.candidate_vector` (the canonical helper) so the
    underlying dict-key convention in ``space.fixed_candidates`` (which
    stores ``action__``-prefixed keys) is invisible to callers.  The
    returned dict is keyed by the **bare** action name (no ``action__``
    prefix), matching the rest of the parser's internal convention.
    """
    if candidate_id not in space.fixed_candidates:
        raise KeyError(
            f"Candidate {candidate_id!r} is not in v32 main_train_labels."
        )
    vec = space.candidate_vector(candidate_id)
    cols_bare = [c.replace("action__", "") for c in space.columns]
    return {bare: float(vec[i]) for i, bare in enumerate(cols_bare)}



# Low-information field names can appear in generic prose without showing that
# the model actually grounded its judgment in the provided firm state.  They
# are retained in prompts/metadata, but they should not by themselves clear
# the ``ungrounded_judgment`` audit flag.
LOW_INFORMATION_GROUNDING_ALIASES = {
    "year",
    "fiscal year",
    "market",
    "firm",
    "firm id",
    "corp code",
    "company id",
    "row id",
}

FEATURE_GROUNDING_ALIAS_OVERRIDES = {
    "debt_to_assets": ["debt to assets", "debt to asset", "debt-to-assets", "leverage ratio", "leverage"],
    "equity_to_assets": ["equity to assets", "equity to asset", "capitalization"],
    "current_ratio": ["current ratio", "liquidity ratio"],
    "cash_ratio": ["cash ratio", "cash liquidity"],
    "operating_margin": ["operating margin", "operating profitability"],
    "gross_margin": ["gross margin"],
    "net_margin": ["net margin", "net profitability"],
    "roa_proxy": ["roa", "return on assets", "asset returns", "asset return"],
    "cogs_to_revenue": [
        "cogs to revenue",
        "cost of goods sold to revenue",
        "cost of goods sold",
        "cost ratio",
        "cost pressures",
    ],
    "sga_to_revenue": [
        "sga to revenue",
        "sg a to revenue",
        "sg&a to revenue",
        "selling general administrative to revenue",
        "selling general and administrative to revenue",
        "selling and administrative expenses",
        "operating expenses",
    ],
    "financial_cost_to_revenue": [
        "financial cost to revenue",
        "financing cost to revenue",
        "interest expense to revenue",
        "interest burden",
    ],
    "capex_to_revenue": [
        "capex to revenue",
        "capital expenditure to revenue",
        "capital expenditures to revenue",
        "capital spending to revenue",
    ],
    "ppe_to_assets": [
        "ppe to assets",
        "property plant equipment to assets",
        "property plant and equipment to assets",
        "fixed assets to assets",
    ],
    "short_debt_to_total_debt": [
        "short debt to total debt",
        "short term debt to total debt",
        "short term debt",
        "short-term debt",
        "current debt",
    ],
    "long_debt_to_total_debt": [
        "long debt to total debt",
        "long term debt to total debt",
        "long term debt",
        "long-term debt",
    ],
    "bond_to_total_debt": ["bond to total debt", "bond debt", "bonds to total debt", "bond obligations"],
    "inventory_to_revenue": ["inventory to revenue", "inventory ratio", "inventory intensity"],
    "receivables_to_revenue": [
        "receivables to revenue",
        "receivable to revenue",
        "accounts receivable to revenue",
        "receivable ratio",
        "receivables ratio",
    ],
    "payables_to_revenue": [
        "payables to revenue",
        "payable to revenue",
        "accounts payable to revenue",
        "payable ratio",
        "payables ratio",
    ],
    "log_assets": ["log assets", "asset size", "firm size"],
    "nf_log_assets": ["log assets", "asset size", "firm size"],
}


def _normalize_grounding_text(text: str) -> str:
    """Normalize prose/feature labels for conservative grounding matching.

    Live LLMs usually cite human-readable metric names (``debt to assets
    ratio``) rather than raw prompt keys (``derived__debt_to_assets``).  The
    taxonomy should therefore match normalized aliases while still avoiding
    low-information words such as ``year`` clearing the flag by accident.
    """
    lowered = (text or "").lower()
    lowered = lowered.replace("&", " and ")
    lowered = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def _canonical_feature_key(feature: str) -> str:
    """Return the semantic suffix of a prompt feature key.

    Examples
    --------
    ``derived__debt_to_assets`` -> ``debt_to_assets``
    ``delta_1y__derived__operating_margin`` -> ``operating_margin``
    """
    key = str(feature or "").strip().lower()
    key = key.replace("delta_1y__", "")
    key = key.replace("delta_1y_", "")
    parts = [part for part in key.split("__") if part]
    if parts and parts[0] in {"derived", "nf", "raw", "feature"}:
        parts = parts[1:]
    if len(parts) > 1 and parts[0] in {"derived", "nf", "raw", "feature"}:
        parts = parts[1:]
    return "__".join(parts) if parts else key


def _feature_grounding_aliases(feature: str) -> set[str]:
    """Generate normalized aliases that count as grounding for one feature.

    The raw key remains valid, but we also accept its canonical natural-language
    form and curated accounting synonyms for the Stage 7 exposed financial
    ratios.  This changes only the audit classifier; prompt payloads, metadata,
    and downstream action semantics are unchanged.
    """
    raw = str(feature or "").strip().lower()
    canonical = _canonical_feature_key(raw)
    candidates = {
        raw,
        raw.replace("__", " "),
        raw.replace("__", "_").replace("_", " "),
        canonical,
        canonical.replace("__", " "),
        canonical.replace("_", " "),
    }
    if raw.startswith("delta_1y"):
        candidates.add(f"one year change in {canonical.replace('_', ' ')}")
        candidates.add(f"year over year {canonical.replace('_', ' ')}")
        candidates.add(f"yoy {canonical.replace('_', ' ')}")
    candidates.update(FEATURE_GROUNDING_ALIAS_OVERRIDES.get(canonical, []))

    aliases: set[str] = set()
    for candidate in candidates:
        norm = _normalize_grounding_text(candidate)
        if not norm or norm in LOW_INFORMATION_GROUNDING_ALIASES:
            continue
        # Require at least one informative token.  This prevents accidental
        # grounding on short/generic symbols while preserving abbreviations
        # such as ROA, COGS, SG&A after normalization.
        tokens = norm.split()
        if len(tokens) == 1 and len(tokens[0]) < 3:
            continue
        aliases.add(norm)
    return aliases


def _rationale_references_firm_state(
    rationale: str | None,
    request_diagnosis_features: list[str],
) -> bool:
    """Return True when rationale cites a provided firm-state feature.

    Matching is source-local: aliases are generated only for features actually
    present in the request's firm_state.  A rationale that mentions a metric
    not shown to the LLM therefore does not get credit for grounding.
    """
    if not rationale or not request_diagnosis_features:
        return False
    normalized_rationale = f" {_normalize_grounding_text(rationale)} "
    if len(normalized_rationale.strip()) <= 10:
        return False
    for feature in request_diagnosis_features:
        for alias in _feature_grounding_aliases(feature):
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized_rationale):
                return True
    return False


def _classify_failures(
    payload: dict | None,
    mode: str,
    condition: str,
    request_diagnosis_features: list[str],
    rl_reference_candidate: str | None,
    rationale: str | None,
    diagnosis: dict | None,
    selected_candidate: str | None,
    materialized_action: dict[str, float],
    bound_clipping: dict[str, dict] | None = None,
    projection_distance: float | None = None,
    out_of_library: bool | None = None,
) -> tuple[list[str], dict]:
    """Apply the RESEARCH §10 failure taxonomy to a parsed response.

    Multiple categories may apply to a single response (e.g. a response can
    be both ``anchoring_or_confirmation_failure`` and
    ``liquidity_destructive_recourse``).
    """
    failures: list[str] = []
    audit: dict = {
        "direction_error_auto": False,
        "direction_review_needed": False,
        "direction_error_reason": "not_applicable_payload_missing",
        "direction_rule_version": "",
        "magnitude_error_auto": False,
        "magnitude_review_needed": False,
        "magnitude_error_reason": "not_applicable_payload_missing",
        "magnitude_rule_version": "",
        "max_bound_violation_abs": 0.0,
        "action_l1_norm": 0.0,
        "action_nonzero_dim_count": 0,
        "failure_coder_version": FAILURE_CODER_VERSION,
        "oracle_scores_used_for_failure_coding": ORACLE_SCORES_USED_FOR_FAILURE_CODING,
    }
    if payload is None:
        failures.append("translational_failure")
        return failures, audit

    # translational_failure: required fields missing
    if mode == "candidate_selection":
        if "selected_candidate" not in payload or not payload.get("selected_candidate"):
            failures.append("translational_failure")
    else:
        av = payload.get("action_vector")
        if not isinstance(av, dict):
            failures.append("translational_failure")

    # ungrounded_judgment: rationale references no provided firm-state
    # metric/context feature.  Matching accepts source-local natural-language
    # aliases (e.g. ``debt to assets ratio`` for ``derived__debt_to_assets``)
    # instead of requiring the LLM to quote raw prompt keys verbatim.
    if rationale:
        grounded = _rationale_references_firm_state(
            rationale=rationale,
            request_diagnosis_features=request_diagnosis_features,
        )
        if not grounded and len(_normalize_grounding_text(rationale)) > 10:
            # Only flag if there is a rationale to ground at all.
            failures.append("ungrounded_judgment")

    # liquidity_destructive_recourse: a large positive ap_turnover_chg (paying
    # suppliers faster) combined with a large positive inventory turnover and
    # negative short-term debt rollover would visibly cut liquidity.  We use
    # a conservative threshold.
    if materialized_action:
        ap = materialized_action.get("ap_turnover_chg", 0.0)
        st = materialized_action.get("short_debt_pct", 0.0)
        if ap > 1.0 and st < -0.5:
            failures.append("liquidity_destructive_recourse")

    # anchoring_or_confirmation_failure: revision condition reproduces the RL
    # reference exactly with no diagnostic difference.  This is only flagged
    # for C6/C7 (revision conditions where independence is expected).
    if condition in {"C6", "C6X", "C7"} and rl_reference_candidate is not None:
        sel = payload.get("selected_candidate")
        rationale_lower = (rationale or "").lower()
        if (
            sel == rl_reference_candidate
            and "reference" in rationale_lower
            and "differs" not in rationale_lower
            and "modify" not in rationale_lower
            and "reject" not in rationale_lower
        ):
            # Verbatim adoption with no critical engagement.
            failures.append("anchoring_or_confirmation_failure")

    # direction_error and magnitude_error: executable pre-simulation coders.
    # These use only LLM output, action vectors, contract bounds and projection
    # diagnostics; Oracle scores are explicitly excluded.
    direction = code_direction_error(
        diagnosis=diagnosis or {},
        rationale=rationale,
        selected_candidate=selected_candidate,
        materialized_action=materialized_action or {},
    )
    magnitude = code_magnitude_error(
        mode=mode,
        bound_clipping=bound_clipping or {},
        projection_distance=projection_distance,
        out_of_library=out_of_library,
        materialized_action=materialized_action or {},
    )
    audit.update(direction)
    audit.update(magnitude)
    if bool(direction.get("direction_error_auto")):
        failures.append("direction_error")
    if bool(magnitude.get("magnitude_error_auto")):
        failures.append("magnitude_error")

    # Defensive de-dup while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in failures:
        if c in ALLOWED_FAILURE_TAXONOMY and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped, audit


def parse_response(
    response: LLMResponse,
    space: ActionSpace,
) -> ParsedAction:
    """Parse one :class:`LLMResponse` into a :class:`ParsedAction`.

    Always returns a :class:`ParsedAction`; failures are recorded via
    :attr:`ParsedAction.failure_categories` so downstream stages can route
    rows to the failure audit without losing per-row attribution.
    """
    request = response.request
    payload = response.parsed_json
    mode = request.mode
    cond = request.condition
    rationale = (payload or {}).get("rationale", "") if payload else ""
    diagnosis = (payload or {}).get("diagnosis", {}) if payload else {}
    request_features = list((request.prompt.get("firm_state") or {}).keys())
    budget_contract = request.prompt.get("action_budget_contract")

    if payload is None:
        # Translational failure with no parsable JSON.  Default to an
        # explicit no-op so a downstream simulator pass still exists, but
        # mark the row as not routed.
        cols = [c.replace("action__", "") for c in space.columns]
        materialized = {c: 0.0 for c in cols}
        return ParsedAction(
            row_id=request.row_id,
            policy=cond,
            mode=mode,
            information_condition=request.information_condition,
            selected_candidate="A0_noop",
            free_form_action=None,
            materialized_action=materialized,
            rationale=rationale or "",
            diagnosis=diagnosis or {},
            rl_reference_candidate=request.rl_reference_candidate,
            reference_source=request.reference_source,
            reference_draw_seed=request.reference_draw_seed,
            bound_clipping={},
            projection_distance=None,
            projection_method=None,
            out_of_library=None,
            failure_categories=["translational_failure"],
            routed_to_simulator=False,
            raw_response=response.raw_text,
            **budget_audit_for_action(
                contract=budget_contract,
                condition=cond,
                mode=mode,
                raw_action=None,
                clipped_action=None,
            ),
        )

    if mode == "candidate_selection":
        selected = payload.get("selected_candidate")
        if not isinstance(selected, str) or selected not in space.fixed_candidates:
            # Not in v32 main labels.  Mark structural_out_of_scope when the
            # rationale points outside the action space; else translational.
            failure = (
                "structural_out_of_scope"
                if isinstance(selected, str) and selected
                else "translational_failure"
            )
            cols = [c.replace("action__", "") for c in space.columns]
            return ParsedAction(
                row_id=request.row_id,
                policy=cond,
                mode=mode,
                information_condition=request.information_condition,
                selected_candidate="A0_noop",
                free_form_action=None,
                materialized_action={c: 0.0 for c in cols},
                rationale=rationale,
                diagnosis=diagnosis,
                rl_reference_candidate=request.rl_reference_candidate,
                reference_source=request.reference_source,
                reference_draw_seed=request.reference_draw_seed,
                bound_clipping={},
                projection_distance=None,
                projection_method=None,
                out_of_library=None,
                failure_categories=[failure],
                routed_to_simulator=False,
                raw_response=response.raw_text,
                **budget_audit_for_action(
                    contract=budget_contract,
                    condition=cond,
                    mode=mode,
                    raw_action=None,
                    clipped_action=None,
                ),
            )
        materialized = _materialize_from_candidate(selected, space)
        # Candidate-mode actions are by construction within bounds (the v32
        # library is bound-compliant); we still record clipping audit to be
        # explicit about the absence of clipping.
        bounded, clip_audit = _clip_to_bounds(materialized, space)
        failures, failure_audit = _classify_failures(
            payload=payload,
            mode=mode,
            condition=cond,
            request_diagnosis_features=request_features,
            rl_reference_candidate=request.rl_reference_candidate,
            rationale=rationale,
            diagnosis=diagnosis,
            selected_candidate=selected,
            materialized_action=bounded,
            bound_clipping=clip_audit,
            projection_distance=0.0,
            out_of_library=False,
        )
        # No projection in candidate-selection mode; record distance 0.0 for
        # explicit accounting.
        return ParsedAction(
            row_id=request.row_id,
            policy=cond,
            mode=mode,
            information_condition=request.information_condition,
            selected_candidate=selected,
            free_form_action=None,
            materialized_action=bounded,
            rationale=rationale,
            diagnosis=diagnosis,
            rl_reference_candidate=request.rl_reference_candidate,
            reference_source=request.reference_source,
            reference_draw_seed=request.reference_draw_seed,
            bound_clipping=clip_audit,
            projection_distance=0.0,
            projection_method="exact_v32_lookup_no_projection",
            out_of_library=False,
            failure_categories=failures,
            routed_to_simulator=True,
            raw_response=response.raw_text,
            **failure_audit,
            **budget_audit_for_action(
                contract=budget_contract,
                condition=cond,
                mode=mode,
                raw_action=bounded,
                clipped_action=bounded,
            ),
        )

    # free_form_10d mode
    av = payload.get("action_vector")
    if not isinstance(av, dict):
        cols = [c.replace("action__", "") for c in space.columns]
        return ParsedAction(
            row_id=request.row_id,
            policy=cond,
            mode=mode,
            information_condition=request.information_condition,
            selected_candidate="A0_noop",
            free_form_action=None,
            materialized_action={c: 0.0 for c in cols},
            rationale=rationale,
            diagnosis=diagnosis,
            rl_reference_candidate=request.rl_reference_candidate,
            reference_source=request.reference_source,
            reference_draw_seed=request.reference_draw_seed,
            bound_clipping={},
            projection_distance=None,
            projection_method=None,
            out_of_library=None,
            failure_categories=["translational_failure"],
            routed_to_simulator=False,
            raw_response=response.raw_text,
            **budget_audit_for_action(
                contract=budget_contract,
                condition=cond,
                mode=mode,
                raw_action=None,
                clipped_action=None,
            ),
        )
    raw_vec: dict[str, float] = {}
    for col in [c.replace("action__", "") for c in space.columns]:
        v = av.get(col, 0.0)
        try:
            raw_vec[col] = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            raw_vec[col] = 0.0
    bounded, clip_audit = _clip_to_bounds(raw_vec, space)
    failures, failure_audit = _classify_failures(
        payload=payload,
        mode=mode,
        condition=cond,
        request_diagnosis_features=request_features,
        rl_reference_candidate=request.rl_reference_candidate,
        rationale=rationale,
        diagnosis=diagnosis,
        selected_candidate="",
        materialized_action=bounded,
        bound_clipping=clip_audit,
        projection_distance=None,
        out_of_library=None,
    )

    return ParsedAction(
        row_id=request.row_id,
        policy=cond,
        mode=mode,
        information_condition=request.information_condition,
        selected_candidate="",  # filled by project_free_form_batch below
        free_form_action=raw_vec,
        materialized_action=bounded,
        rationale=rationale,
        diagnosis=diagnosis,
        rl_reference_candidate=request.rl_reference_candidate,
        reference_source=request.reference_source,
        reference_draw_seed=request.reference_draw_seed,
        bound_clipping=clip_audit,
        projection_distance=None,  # filled by project_free_form_batch
        projection_method=None,
        out_of_library=None,
        failure_categories=failures,
        routed_to_simulator=True,
        raw_response=response.raw_text,
        **failure_audit,
        **budget_audit_for_action(
            contract=budget_contract,
            condition=cond,
            mode=mode,
            raw_action=raw_vec,
            clipped_action=bounded,
        ),
    )


def project_free_form_batch(
    actions: list[ParsedAction], space: ActionSpace
) -> list[ParsedAction]:
    """Project all free-form parsed actions through Stage 2's projection
    helper in one batch and write the results back to each row.

    This guarantees the projection geometry used in Stage 7 is the *same*
    geometry as Stage 2 — no parallel implementation.
    """
    cols = list(space.columns)
    rows = []
    pos_map: list[int] = []
    for i, a in enumerate(actions):
        if a.mode != "free_form_10d" or not a.routed_to_simulator:
            continue
        row = {c: float(a.materialized_action.get(c.replace("action__", ""), 0.0)) for c in cols}
        rows.append(row)
        pos_map.append(i)
    if not rows:
        return actions
    df = pd.DataFrame(rows)
    projected = project_actions_to_candidates(df, space)
    for j, pos in enumerate(pos_map):
        a = actions[pos]
        a.selected_candidate = str(projected.iloc[j]["projected_candidate_id"])
        a.projection_distance = float(projected.iloc[j]["projection_distance"])
        a.projection_method = str(projected.iloc[j]["projection_method"])
        a.out_of_library = bool(projected.iloc[j]["out_of_library_flag"])
        # Re-run the magnitude coder now that projection diagnostics exist.
        mag = code_magnitude_error(
            mode=a.mode,
            bound_clipping=a.bound_clipping,
            projection_distance=a.projection_distance,
            out_of_library=a.out_of_library,
            materialized_action=a.materialized_action,
        )
        a.magnitude_error_auto = bool(mag["magnitude_error_auto"])
        a.magnitude_review_needed = bool(mag["magnitude_review_needed"])
        a.magnitude_error_reason = str(mag["magnitude_error_reason"])
        a.magnitude_rule_version = str(mag.get("magnitude_rule_version", a.magnitude_rule_version or ""))
        a.max_bound_violation_abs = float(mag["max_bound_violation_abs"])
        a.action_l1_norm = float(mag["action_l1_norm"])
        a.action_nonzero_dim_count = int(mag["action_nonzero_dim_count"])
        if a.magnitude_error_auto and "magnitude_error" not in a.failure_categories:
            a.failure_categories.append("magnitude_error")
    return actions


def to_policy_actions_frame(
    parsed: list[ParsedAction], space: ActionSpace
) -> pd.DataFrame:
    """Convert a list of :class:`ParsedAction` into the Stage 6-compatible
    ``policy_actions`` DataFrame (so Stage 8 can call the same simulator
    code path Stage 6 used)."""
    cols = list(space.columns)
    rows = []
    for a in parsed:
        if not a.routed_to_simulator:
            continue
        row = {
            "row_id": a.row_id,
            "policy": a.policy,
            "candidate_id": a.selected_candidate,
            "mode": a.mode,
            "information_condition": a.information_condition,
            "rl_reference_candidate": a.rl_reference_candidate,
            "reference_source": a.reference_source,
            "reference_draw_seed": a.reference_draw_seed,
            "projection_distance": (
                a.projection_distance if a.projection_distance is not None else 0.0
            ),
            "projection_method": a.projection_method,
            "out_of_library": bool(a.out_of_library) if a.out_of_library is not None else False,
            "budget_contract_label": a.budget_contract_label,
            "budget_l1_target": a.budget_l1_target,
            "budget_l1_raw": a.budget_l1_raw,
            "budget_l1_clipped": a.budget_l1_clipped,
            "budget_compliant_raw": a.budget_compliant_raw,
            "budget_compliant_clipped": a.budget_compliant_clipped,
            "budgeted_condition_flag": bool(a.budgeted_condition_flag),
        }
        for c in cols:
            row[c] = float(a.materialized_action.get(c.replace("action__", ""), 0.0))
        rows.append(row)
    if not rows:
        # Return an empty frame with the expected schema so verifiers can
        # still inspect column presence.
        schema_cols = [
            "row_id", "policy", "candidate_id", "mode", "information_condition",
            "rl_reference_candidate", "reference_source", "reference_draw_seed", "projection_distance", "projection_method",
            "out_of_library",
        ] + BUDGET_AUDIT_COLUMNS + cols
        return pd.DataFrame(columns=schema_cols)
    return pd.DataFrame(rows)


def to_failure_audit_frame(
    parsed: list[ParsedAction],
) -> pd.DataFrame:
    """Per-row failure audit for Stage 7."""
    rows = []
    for a in parsed:
        rows.append({
            "row_id": a.row_id,
            "policy": a.policy,
            "mode": a.mode,
            "information_condition": a.information_condition,
            "selected_candidate": a.selected_candidate,
            "rl_reference_candidate": a.rl_reference_candidate,
            "reference_source": a.reference_source,
            "reference_draw_seed": a.reference_draw_seed,
            "failure_categories": ",".join(a.failure_categories),
            "failure_count": len(a.failure_categories),
            "routed_to_simulator": a.routed_to_simulator,
            "projection_distance": a.projection_distance,
            "out_of_library": a.out_of_library,
            "bound_clip_dim_count": len(a.bound_clipping),
            "rationale_length": len(a.rationale or ""),
            "direction_error_auto": bool(a.direction_error_auto),
            "direction_review_needed": bool(a.direction_review_needed),
            "direction_error_reason": a.direction_error_reason,
            "direction_rule_version": a.direction_rule_version,
            "magnitude_error_auto": bool(a.magnitude_error_auto),
            "magnitude_review_needed": bool(a.magnitude_review_needed),
            "magnitude_error_reason": a.magnitude_error_reason,
            "magnitude_rule_version": a.magnitude_rule_version,
            "max_bound_violation_abs": float(a.max_bound_violation_abs),
            "action_l1_norm": float(a.action_l1_norm),
            "action_nonzero_dim_count": int(a.action_nonzero_dim_count),
            "failure_coder_version": a.failure_coder_version,
            "oracle_scores_used_for_failure_coding": bool(a.oracle_scores_used_for_failure_coding),
            "budget_contract_label": a.budget_contract_label,
            "budget_l1_target": a.budget_l1_target,
            "budget_l1_raw": a.budget_l1_raw,
            "budget_l1_clipped": a.budget_l1_clipped,
            "budget_compliant_raw": a.budget_compliant_raw,
            "budget_compliant_clipped": a.budget_compliant_clipped,
            "budgeted_condition_flag": bool(a.budgeted_condition_flag),
        })
    if not rows:
        return pd.DataFrame(columns=[
            "row_id", "policy", "mode", "information_condition",
            "selected_candidate", "rl_reference_candidate", "reference_source", "reference_draw_seed",
            "failure_categories", "failure_count", "routed_to_simulator",
            "projection_distance", "out_of_library",
            "bound_clip_dim_count", "rationale_length",
            "direction_error_auto", "direction_review_needed", "direction_error_reason", "direction_rule_version",
            "magnitude_error_auto", "magnitude_review_needed", "magnitude_error_reason", "magnitude_rule_version",
            "max_bound_violation_abs", "action_l1_norm", "action_nonzero_dim_count",
            "failure_coder_version", "oracle_scores_used_for_failure_coding",
        ] + BUDGET_AUDIT_COLUMNS)
    return pd.DataFrame(rows)
