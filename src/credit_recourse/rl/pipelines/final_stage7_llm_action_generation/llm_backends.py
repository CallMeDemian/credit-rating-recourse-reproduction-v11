from __future__ import annotations

"""LLM backend interface for Stage 7.

Defines the abstract contract every Stage 7 LLM backend must satisfy and
provides two concrete implementations:

1. ``ScriptedReproducibilityBackend`` — a deterministic, seed-driven backend
   that emits *contract-correct* structured responses derived from the firm's
   own state.  This is the offline reproduction backend used when no live LLM
   credentials are available, when running smoke tests, and when verifying
   the Stage 7 contract.  It is **not** a placeholder: it produces realistic
   v32-vocabulary candidate selections and free-form 10D vectors that pass
   every downstream verifier.  Final paper runs require a live backend; this
   is enforced in Stage 7 metadata (``llm_backend_is_live=False`` blocks
   ``final_paper_run_allowed``).

2. ``LiveLLMBackend`` — a thin adapter around an HTTP LLM provider.  Concrete
   provider classes (``OpenAIBackend``, ``AnthropicBackend``,
   ``GeminiBackend``) extend it.  These
   are imported lazily so the module does not require ``openai`` /
   ``anthropic`` packages to be installed for offline runs.

The backend's only responsibility is to return a raw response dict; all
validation, clipping, and projection live in ``response_parser.py``.
"""

import abc
import hashlib
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from .budget_contract import action_l1, budget_applies


@dataclass(frozen=True)
class LLMRequest:
    """One LLM invocation request.

    Attributes
    ----------
    row_id : int
        Firm-year row identifier (matches the Stage 2 ``phase_eval_candidate``
        row order).
    condition : str
        Policy condition code (``C4``/``C4R``/``C5``/``C6``/``C6X``/``C7``/``C8``).
    mode : str
        ``"candidate_selection"`` (primary) or ``"free_form_10d"`` (diagnostic).
    information_condition : str
        ``"IC-a"`` (anonymous tabular), ``"IC-b"`` (plus industry/year), or
        ``"IC-c"`` (plus named firm; used with contamination caution).
    prompt : dict
        Structured prompt payload produced by ``prompt_builder``.
    rl_reference_candidate : str | None
        For C6/C7/C8: the v32 candidate selected by the RL policy for this
        firm-year. For C6X: the seeded random in-vocabulary reference.
        ``None`` for C4/C5.
    initial_action : dict | None
        For C4R/C6/C6X/C7: the LLM's pre-revision action ``a_0``; the LLM is asked to
        revise it after seeing the shown reference.  ``None`` for C4/C5/C8.
    """

    row_id: int
    condition: str
    mode: str
    information_condition: str
    prompt: dict
    rl_reference_candidate: str | None = None
    reference_source: str = "none"
    reference_draw_seed: int | None = None
    initial_action: dict | None = None


@dataclass
class LLMResponse:
    """Raw LLM response prior to parsing/validation.

    Attributes
    ----------
    request : LLMRequest
        The request that produced this response.
    raw_text : str
        The raw model output text.
    parsed_json : dict | None
        The JSON parsed from ``raw_text`` if extractable; ``None`` if parsing
        failed.  ``response_parser`` performs full validation; this field is
        only the first parse pass.
    parse_error : str | None
        Reason for parse failure; ``None`` on success.
    backend_metadata : dict
        Backend-specific metadata (model name, temperature, latency, etc.)
        recorded into the Stage 7 manifest.
    """

    request: LLMRequest
    raw_text: str
    parsed_json: dict | None
    parse_error: str | None
    backend_metadata: dict = field(default_factory=dict)


class LLMBackend(abc.ABC):
    """Abstract Stage 7 LLM backend.

    Subclasses must implement :meth:`generate` and :attr:`backend_id`.  Stage 7
    records ``backend_id``, ``is_live``, and ``backend_metadata`` in the prompt
    manifest so any reproducer can identify which backend produced an action.
    """

    #: Stable identifier (e.g. ``"scripted_reproducibility_v1"``,
    #: ``"openai_gpt-4o_2024-08-06"``) recorded in the Stage 7 manifest.
    backend_id: str = "abstract"

    #: ``True`` if this backend calls an external LLM API; ``False`` if it is
    #: a deterministic offline backend.  Stage 7 metadata blocks
    #: ``final_paper_run_allowed`` when ``is_live`` is ``False``.
    is_live: bool = False

    @abc.abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one response for one request.

        Implementations must not mutate ``request``.  Implementations must
        catch parser-relevant exceptions and surface them via
        :attr:`LLMResponse.parse_error` rather than raising — this preserves
        per-row error attribution in Stage 7's failure audit.
        """

    def manifest(self) -> dict:
        """Backend identity recorded in the Stage 7 prompt manifest."""
        return {
            "backend_id": self.backend_id,
            "is_live": bool(self.is_live),
            "class": type(self).__name__,
        }

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        """Free-form single call used by the IC-c prior-knowledge probe
        (LLM789-009).  Not part of the recourse-generation contract; the
        response is parsed by the Stage 7 probe layer, not response_parser.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement generate_raw(); the "
            f"IC-c probe requires a backend with a raw-call path."
        )


# ---------------------------------------------------------------------------
# Scripted reproducibility backend
# ---------------------------------------------------------------------------


class ScriptedReproducibilityBackend(LLMBackend):
    """Deterministic offline backend that emits v32-vocabulary responses.

    The backend reads the firm-state prompt produced by ``prompt_builder``,
    inspects the financial-state features available there (leverage,
    liquidity, margin, capex, etc.), and selects a v32 candidate consistent
    with the firm's weakest dimension.  This mirrors the structure a real
    LLM would produce when asked to recourse a firm but is fully
    deterministic.

    The selection rule is:

    * High debt-to-assets (``debt_to_assets`` above its sector p75) → choose a
      deleveraging candidate (``DL1`` or ``DL2`` based on severity).
    * High SG&A-to-revenue or COGS-to-revenue ratio → cost-efficiency
      (``OE1``/``OE2``).
    * Low current ratio → liquidity rescue (``MX2``) or working-capital
      tightening (``WC1``).
    * High capex without margin support → capex discipline (``CX1``).
    * Default → ``A0_noop``.

    For reasoning-mode (C5/C7), the rationale string is extended with a
    diagnosis breakdown.  For revision conditions (C6/C7/C8), the backend
    incorporates the RL reference candidate by computing an adoption
    probability based on the agreement between its own scripted selection
    and the RL reference.

    For free-form mode, the backend emits a 10D vector that lies on the
    selected candidate's direction with a deterministic perturbation; this
    exercises the projection-distance machinery.
    """

    backend_id = "scripted_reproducibility_v1"
    is_live = False

    def __init__(self, seed: int = 20260523):
        self.seed = int(seed)

    @staticmethod
    def _hash_seed(row_id: int, condition: str, mode: str, base_seed: int) -> int:
        """Deterministic per-row seed; identical row × condition × mode →
        identical response across runs."""
        h = hashlib.sha256(f"{base_seed}|{row_id}|{condition}|{mode}".encode("utf-8")).hexdigest()
        return int(h[:8], 16)

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        """Deterministic offline IC-c probe response (LLM789-009).

        Familiarity is derived from a sha256 of (seed, user_prompt) so the
        offline probe path is fully reproducible and varies per firm.
        """
        import hashlib as _hashlib
        import json as _json
        digest = _hashlib.sha256(f"icc_probe|{self.seed}|{user_prompt}".encode("utf-8")).hexdigest()
        familiarity = int(digest[:8], 16) % 4
        recognized = familiarity >= 2
        # v2 numeric-recall: emit a deterministic decimal only when the
        # scripted persona "knows concrete facts" (familiarity >= 2), else null
        # — mirrors the live-prompt instruction "null if you do not know".
        recalled = (
            round((int(digest[8:16], 16) % 1000) / 1000.0, 3)
            if familiarity >= 2 else None
        )
        facts = [f"scripted_probe_fact_{i}" for i in range(min(familiarity, 3))]
        return _json.dumps(
            {"recalled_debt_ratio": recalled, "recognized": recognized,
             "familiarity": familiarity, "known_facts": facts},
            ensure_ascii=False,
        )

    @staticmethod
    def _safe_float(prompt_state: dict, key: str, default: float = 0.0) -> float:
        v = prompt_state.get(key)
        if v is None:
            return float(default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _diagnose_weakness(self, state: dict) -> tuple[str, str, dict]:
        """Pick a candidate from v32 main labels based on firm state.

        Returns
        -------
        candidate_id : str
        primary_weakness : str
        diagnosis : dict
            Supporting facts used by the rationale builder.
        """
        debt = self._safe_float(state, "derived__debt_to_assets")
        sga = self._safe_float(state, "derived__sga_to_revenue")
        cogs = self._safe_float(state, "derived__cogs_to_revenue")
        cr = self._safe_float(state, "derived__current_ratio", default=1.5)
        capex = self._safe_float(state, "derived__capex_to_revenue")
        opm = self._safe_float(state, "derived__operating_margin")

        diagnosis = {
            "debt_to_assets": debt,
            "sga_to_revenue": sga,
            "cogs_to_revenue": cogs,
            "current_ratio": cr,
            "capex_to_revenue": capex,
            "operating_margin": opm,
        }

        if cr < 1.0:
            return "MX2_liquidity_rescue", "liquidity", diagnosis
        if debt > 0.60:
            severity = "moderate" if debt > 0.75 else "mild"
            return (
                "DL2_deleverage_moderate" if severity == "moderate" else "DL1_deleverage_mild",
                "leverage",
                diagnosis,
            )
        if opm < 0.03 and (sga > 0.15 or cogs > 0.80):
            severity = "moderate" if (sga > 0.20 or cogs > 0.85) else "mild"
            return (
                "OE2_cost_efficiency_moderate" if severity == "moderate" else "OE1_cost_efficiency_mild",
                "margin",
                diagnosis,
            )
        if capex > 0.10 and opm < 0.05:
            return "CX1_capex_discipline", "capital_allocation", diagnosis
        if cr < 1.2:
            return "WC1_working_capital_tightening", "working_capital", diagnosis
        return "A0_noop", "no_dominant_weakness", diagnosis

    @staticmethod
    def _candidate_vector_proxy(candidate_id: str) -> dict:
        """Deterministic proxy 10D vector for a candidate (used when prompt
        builder hasn't injected the full candidate library)."""
        zero = {
            "ppe_pct": 0.0, "inv_turnover_chg": 0.0, "ar_turnover_chg": 0.0,
            "ap_turnover_chg": 0.0, "short_debt_pct": 0.0, "long_debt_pct": 0.0,
            "bond_pct": 0.0, "revenue_growth": 0.0, "cogs_ratio_chg": 0.0,
            "sga_ratio_chg": 0.0,
        }
        # Direction-only proxies; magnitudes are taken from the real candidate
        # library in the parser, so these are only used for free-form mode and
        # are intentionally conservative.
        direction = dict(zero)
        if candidate_id.startswith("DL"):
            direction["short_debt_pct"] = -0.05
            direction["long_debt_pct"] = -0.05
        elif candidate_id == "RF1_short_debt_refinance":
            direction["short_debt_pct"] = -0.10
            direction["long_debt_pct"] = 0.10
        elif candidate_id == "CX1_capex_discipline":
            direction["ppe_pct"] = -0.10
        elif candidate_id.startswith("WC"):
            direction["inv_turnover_chg"] = 0.30
            direction["ar_turnover_chg"] = 0.30
            if candidate_id == "WC2_supplier_financing":
                direction["ap_turnover_chg"] = -0.30
        elif candidate_id.startswith("OE"):
            direction["sga_ratio_chg"] = -0.005
            direction["cogs_ratio_chg"] = -0.005
        elif candidate_id == "MX1_cost_and_deleverage":
            direction["sga_ratio_chg"] = -0.005
            direction["long_debt_pct"] = -0.05
        elif candidate_id == "MX2_liquidity_rescue":
            direction["short_debt_pct"] = -0.10
            direction["inv_turnover_chg"] = 0.30
        return direction

    def _build_rationale(
        self,
        candidate: str,
        weakness: str,
        diagnosis: dict,
        reasoning_mode: bool,
        rl_reference: str | None,
    ) -> str:
        diag_text = ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in diagnosis.items()
        )
        grounding_keys = ", ".join(f"derived__{k}" for k in diagnosis.keys())
        base = (
            f"Primary weakness detected: {weakness}. Selected {candidate} as a "
            f"governance-compatible recourse program addressing this dimension. "
            f"Grounding firm_state keys: {grounding_keys}."
        )
        if reasoning_mode:
            base += f" Supporting state: {diag_text}."
        if rl_reference is not None:
            agreement = "matches" if rl_reference == candidate else "differs from"
            base += (
                f" The shown reference suggested {rl_reference}; this recommendation "
                f"{agreement} the shown reference."
            )
        return base

    def _select_with_rl_revision(
        self,
        own_candidate: str,
        own_weakness: str,
        diagnosis: dict,
        rl_reference: str | None,
        deterministic_int: int,
    ) -> str:
        """Decide revised candidate given the LLM's own pick and the RL
        reference.  Deterministic adoption schedule based on row hash."""
        if rl_reference is None or rl_reference == own_candidate:
            return own_candidate
        # Adoption schedule: ~40% adopt RL outright, ~40% retain own,
        # ~20% adopt a compromise (own candidate when RL is incompatible
        # with diagnosed weakness, else RL).
        bucket = deterministic_int % 10
        if bucket < 4:
            return rl_reference
        if bucket < 8:
            return own_candidate
        # Compromise: keep RL only if it addresses a recognised weakness.
        weakness_compatible = (
            (rl_reference.startswith("DL") and own_weakness == "leverage")
            or (rl_reference.startswith("OE") and own_weakness == "margin")
            or (rl_reference.startswith("WC") and own_weakness in {"working_capital", "liquidity"})
            or (rl_reference == "MX2_liquidity_rescue" and own_weakness == "liquidity")
        )
        return rl_reference if weakness_compatible else own_candidate

    def generate(self, request: LLMRequest) -> LLMResponse:
        try:
            state = request.prompt.get("firm_state") or {}
            own_cand, own_weak, diag = self._diagnose_weakness(state)
            det_int = self._hash_seed(
                request.row_id, request.condition, request.mode, self.seed
            )

            # Resolve final candidate based on condition semantics.
            if request.condition in {"C4", "C5"}:
                final_cand = own_cand
            elif request.condition == "C4R":
                # Reference-free second pass: preserve the same state-only diagnosis
                # but allow deterministic reconsideration without an external candidate.
                final_cand = own_cand
            elif request.condition in {"C6", "C6X", "C7", "C8"}:
                final_cand = self._select_with_rl_revision(
                    own_cand, own_weak, diag, request.rl_reference_candidate, det_int
                )
            else:
                final_cand = own_cand

            reasoning_mode = request.condition in {"C5", "C7"}
            rl_ref_shown = request.rl_reference_candidate if request.condition in {"C6", "C6X", "C7", "C8"} else None
            rationale = self._build_rationale(
                final_cand, own_weak, diag, reasoning_mode, rl_ref_shown
            )

            if request.mode == "candidate_selection":
                payload = {
                    "mode": "candidate_selection",
                    "selected_candidate": final_cand,
                    "rationale": rationale,
                    "diagnosis": {
                        "primary_weakness": own_weak,
                        "supporting_facts": diag,
                    },
                    "confidence": "medium",
                    "constraints_checked": {
                        "candidate_in_v32_main_labels": True,
                        "uses_only_provided_facts": True,
                    },
                }
            elif request.mode == "free_form_10d":
                vec = self._candidate_vector_proxy(final_cand)
                # Deterministic small perturbation to exercise projection
                # distance without leaving bounds.
                pert_seed = det_int
                pert = {}
                for k, v in vec.items():
                    pert[k] = float(v) + ((pert_seed % 7) - 3) * 0.001
                    pert_seed = pert_seed // 7 + 1
                budget_contract = request.prompt.get("action_budget_contract")
                if budget_applies(budget_contract, condition=request.condition, mode=request.mode):
                    target = float(budget_contract["l1_budget"])
                    l1 = action_l1(pert) or 0.0
                    if l1 > target and l1 > 0:
                        scale = target / l1
                        pert = {k: float(v) * scale for k, v in pert.items()}
                payload = {
                    "mode": "free_form_10d",
                    "action_vector": pert,
                    "rationale": rationale,
                    "diagnosis": {
                        "primary_weakness": own_weak,
                        "supporting_facts": diag,
                    },
                    "confidence": "medium",
                }
            else:
                return LLMResponse(
                    request=request,
                    raw_text="",
                    parsed_json=None,
                    parse_error=f"Unknown mode: {request.mode}",
                    backend_metadata={"deterministic_int": det_int},
                )

            raw_text = json.dumps(payload, ensure_ascii=False)
            return LLMResponse(
                request=request,
                raw_text=raw_text,
                parsed_json=payload,
                parse_error=None,
                backend_metadata={
                    "deterministic_int": det_int,
                    "seed": self.seed,
                },
            )
        except Exception as exc:  # pragma: no cover - defensive only
            return LLMResponse(
                request=request,
                raw_text="",
                parsed_json=None,
                parse_error=f"{type(exc).__name__}: {exc}",
                backend_metadata={},
            )


# ---------------------------------------------------------------------------
# Live backend skeleton (lazy provider imports)
# ---------------------------------------------------------------------------


class LiveLLMBackend(LLMBackend):
    """Common base for live HTTP LLM backends.

    Concrete subclasses must set :attr:`backend_id`, :attr:`model`, and
    implement :meth:`_call_provider`.  Live backends:

    * Always set :attr:`is_live` ``True``.
    * Record full ``backend_metadata`` (model id, temperature, system prompt
      hash, parser version) in every response so a paper run can be
      reproduced bit-for-bit at the prompt+model level.
    * Extract JSON from arbitrary text using :func:`_extract_json_block`.
    """

    is_live = True

    def __init__(self, model: str, temperature: float | None = 0.0, **kwargs):
        self.model = model
        self.temperature = None if temperature is None else float(temperature)
        self.kwargs = kwargs
        # Stage 7 can call one backend instance from multiple worker threads.
        # Provider response metadata must therefore be request-local rather than
        # a shared mutable dict, or response IDs/usage can be attributed to the
        # wrong firm under --max-concurrency > 1.
        self._provider_metadata_local = threading.local()
        self._last_provider_metadata = {}

    @property
    def _last_provider_metadata(self) -> dict:
        return dict(getattr(self._provider_metadata_local, "value", {}))

    @_last_provider_metadata.setter
    def _last_provider_metadata(self, value: dict | None) -> None:
        self._provider_metadata_local.value = dict(value or {})

    @abc.abstractmethod
    def _call_provider(self, system_prompt: str, user_prompt: str) -> str:
        """Subclasses implement the HTTP call and return the raw text."""

    def manifest(self) -> dict:
        """Backend identity and provider configuration for reproducibility."""
        return {
            "backend_id": self.backend_id,
            "is_live": bool(self.is_live),
            "class": type(self).__name__,
            "model": self.model,
            "temperature": self.temperature,
            "provider_options": self._manifest_provider_options(),
        }

    def generate_raw(self, system_prompt: str, user_prompt: str) -> str:
        """IC-c probe raw call (LLM789-009): delegate to the provider path."""
        return self._call_provider(system_prompt, user_prompt)

    def _manifest_provider_options(self) -> dict:
        """Subclass hook for non-secret provider options."""
        return {}

    def _response_backend_metadata(self, system_prompt: str) -> dict:
        """Common per-response metadata, excluding secrets and prompt text."""
        meta = {
            "model": self.model,
            "temperature": self.temperature,
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        }
        meta.update(self._manifest_provider_options())
        if self._last_provider_metadata:
            meta["provider_response"] = dict(self._last_provider_metadata)
        return meta

    @staticmethod
    def _extract_json_block(text: str) -> tuple[dict | None, str | None]:
        """Extract the first ``{...}`` JSON object from ``text``.

        Returns ``(parsed, None)`` on success or ``(None, error_message)`` on
        failure.  Live model outputs sometimes include fenced code blocks or
        leading prose; this strips both.
        """
        if not text:
            return None, "empty response"
        stripped = text.strip()
        # Strip markdown fence if present.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.DOTALL)
        if fence:
            stripped = fence.group(1).strip()
        # Find first balanced JSON object.
        start = stripped.find("{")
        if start < 0:
            return None, "no JSON object found"
        depth = 0
        end = -1
        for i in range(start, len(stripped)):
            c = stripped[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            return None, "unbalanced JSON braces"
        chunk = stripped[start:end]
        try:
            return json.loads(chunk), None
        except json.JSONDecodeError as exc:
            return None, f"json decode error: {exc}"

    def _build_system_prompt(self, request: LLMRequest) -> str:
        """Build the system prompt for the LLM.  Subclasses may override."""
        reasoning = request.condition in {"C5", "C7"}
        mode = request.mode
        rl_ref = request.rl_reference_candidate
        parts = [
            "You are a credit-rating recourse policy.  Given a firm's financial "
            "state, you propose a one-year recourse action.  Output strictly "
            "valid JSON; do not include prose outside the JSON object.",
        ]
        if mode == "candidate_selection":
            parts.append(
                "Mode: candidate_selection.  Output schema: "
                '{"mode":"candidate_selection","selected_candidate":"<v32_label>",'
                '"rationale":"<text>","diagnosis":{"primary_weakness":"<text>",'
                '"supporting_facts":{...}},"confidence":"low|medium|high",'
                '"constraints_checked":{"candidate_in_v32_main_labels":true,'
                '"uses_only_provided_facts":true}}'
            )
        else:
            parts.append(
                "Mode: free_form_10d.  Output schema: "
                '{"mode":"free_form_10d","action_vector":{"ppe_pct":<float>,'
                '"inv_turnover_chg":<float>,"ar_turnover_chg":<float>,'
                '"ap_turnover_chg":<float>,"short_debt_pct":<float>,'
                '"long_debt_pct":<float>,"bond_pct":<float>,'
                '"revenue_growth":<float>,"cogs_ratio_chg":<float>,'
                '"sga_ratio_chg":<float>},"rationale":"<text>",'
                '"diagnosis":{...},"confidence":"low|medium|high"}'
            )
        budget_contract = request.prompt.get("action_budget_contract")
        if budget_applies(budget_contract, condition=request.condition, mode=request.mode):
            parts.append(
                "N5 generation-time budget contract: the L1 norm of the emitted "
                f"10D action_vector must be <= {float(budget_contract['l1_budget']):.12g}. "
                "This constraint applies to the raw JSON action_vector you emit; "
                "do not rely on downstream clipping or projection to satisfy it."
            )
        if reasoning:
            parts.append(
                "Begin with a structured diagnosis of the firm's weakest "
                "dimension, then choose the recourse program that addresses it."
            )
        if rl_ref is not None:
            parts.append(
                f"A reference candidate {rl_ref!r} is shown for this firm.  "
                "Consider this reference but make your own judgment; you may "
                "adopt, reject, or modify it."
            )
        return "\n\n".join(parts)

    def _build_user_prompt(self, request: LLMRequest) -> str:
        # Source-blind: C6X -> "C6" in LLM-facing payload so the prompt is
        # byte-identical to C6 except for the reference content itself.
        # The true condition code lives in metadata, not in the LLM prompt.
        prompt_condition = "C6" if request.condition == "C6X" else request.condition
        return json.dumps(
            {
                "row_id": request.row_id,
                "condition": prompt_condition,
                "information_condition": request.information_condition,
                "firm_state": request.prompt.get("firm_state", {}),
                "candidate_library": request.prompt.get("candidate_library", {}),
                "reference_candidate": request.rl_reference_candidate,
                "initial_action": request.initial_action,
                "action_budget_contract": request.prompt.get("action_budget_contract"),
            },
            ensure_ascii=False,
        )

    def generate(self, request: LLMRequest) -> LLMResponse:
        system_prompt = self._build_system_prompt(request)
        user_prompt = self._build_user_prompt(request)
        try:
            raw = self._call_provider(system_prompt, user_prompt)
        except Exception as exc:
            return LLMResponse(
                request=request,
                raw_text="",
                parsed_json=None,
                parse_error=f"provider_call_failed: {type(exc).__name__}: {exc}",
                backend_metadata=self._response_backend_metadata(system_prompt),
            )
        parsed, err = self._extract_json_block(raw)
        return LLMResponse(
            request=request,
            raw_text=raw,
            parsed_json=parsed,
            parse_error=err,
            backend_metadata=self._response_backend_metadata(system_prompt),
        )


def _require_env(*names: str) -> dict[str, str]:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise EnvironmentError(
            f"Live LLM backend requires environment variables: {missing}. "
            f"Set them or use ScriptedReproducibilityBackend for offline runs."
        )
    return {n: os.environ[n] for n in names}


class OpenAIBackend(LiveLLMBackend):
    """OpenAI backend with opt-in Responses API reasoning support.

    Default behavior preserves the legacy Chat Completions path for non-
    reasoning models.  Reasoning is activated only when ``api_mode`` is set to
    ``"responses"`` or ``reasoning_effort`` is provided by the runner.
    Requires ``openai`` Python package and ``OPENAI_API_KEY``.
    """

    _VALID_API_MODES = {"chat", "responses"}
    _VALID_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}

    def __init__(
        self,
        model: str = "gpt-4o-2024-08-06",
        temperature: float = 0.0,
        **kwargs,
    ):
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        api_mode = kwargs.pop("api_mode", None)
        max_output_tokens = kwargs.pop("max_output_tokens", None)
        if api_mode is None:
            api_mode = "responses" if reasoning_effort is not None else "chat"
        api_mode = str(api_mode).strip().lower()
        if api_mode not in self._VALID_API_MODES:
            raise ValueError(
                f"OpenAI api_mode must be one of {sorted(self._VALID_API_MODES)}; got {api_mode!r}."
            )
        if reasoning_effort is not None:
            reasoning_effort = str(reasoning_effort).strip().lower()
            if reasoning_effort not in self._VALID_REASONING_EFFORTS:
                raise ValueError(
                    "OpenAI reasoning_effort must be one of "
                    f"{sorted(self._VALID_REASONING_EFFORTS)}; got {reasoning_effort!r}."
                )
            if api_mode != "responses":
                raise ValueError(
                    "OpenAI reasoning_effort requires api_mode='responses'.  "
                    "Do not enable reasoning for legacy Chat Completions runs."
                )
        if max_output_tokens is not None:
            max_output_tokens = int(max_output_tokens)
            if max_output_tokens <= 0:
                raise ValueError(f"max_output_tokens must be positive; got {max_output_tokens}.")
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.api_mode = api_mode
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.backend_id = f"openai_{model}_{self.api_mode}"
        if self.reasoning_effort is not None:
            self.backend_id += f"_reasoning-{self.reasoning_effort}"
        if self.max_output_tokens is not None:
            self.backend_id += f"_maxout-{self.max_output_tokens}"
        _require_env("OPENAI_API_KEY")
        try:
            import openai  # noqa: F401 - presence check
        except ImportError as exc:
            raise ImportError(
                "OpenAI backend requested but the 'openai' package is not "
                "installed.  Install it or use ScriptedReproducibilityBackend."
            ) from exc

    def _manifest_provider_options(self) -> dict:
        return {
            "provider": "openai",
            "api_mode": self.api_mode,
            "reasoning_enabled": self.reasoning_effort is not None,
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
        }

    def _call_provider(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - live call
        import openai

        self._last_provider_metadata = {}
        client = openai.OpenAI()
        if self.api_mode == "responses":
            payload = {
                "model": self.model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if self.reasoning_effort is not None:
                payload["reasoning"] = {"effort": self.reasoning_effort}
            if self.max_output_tokens is not None:
                payload["max_output_tokens"] = int(self.max_output_tokens)
            payload.update(self.kwargs)
            resp = client.responses.create(**payload)
            status = getattr(resp, "status", None)
            self._last_provider_metadata = {
                "api_mode": "responses",
                "response_id": getattr(resp, "id", None),
                "status": status,
            }
            incomplete = getattr(resp, "incomplete_details", None)
            if incomplete is not None:
                reason = getattr(incomplete, "reason", None)
                self._last_provider_metadata["incomplete_reason"] = reason
            if status == "incomplete":
                reason = self._last_provider_metadata.get("incomplete_reason")
                raise RuntimeError(f"openai_responses_incomplete: {reason or 'unknown'}")
            return getattr(resp, "output_text", None) or ""

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        payload.update(self.kwargs)
        resp = client.chat.completions.create(**payload)
        self._last_provider_metadata = {
            "api_mode": "chat",
            "response_id": getattr(resp, "id", None),
        }
        return resp.choices[0].message.content or ""


class AnthropicBackend(LiveLLMBackend):
    """Anthropic Claude backend with opt-in manual extended thinking.

    Legacy calls preserve the historical non-thinking contract and backend id.
    Manual thinking is enabled only when ``thinking_budget_tokens`` is passed.
    Anthropic forbids custom temperature/top-k settings while thinking is
    enabled, so the thinking path omits ``temperature`` entirely.

    Requires ``anthropic`` Python package and ``ANTHROPIC_API_KEY``.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float | None = None,
        **kwargs,
    ):
        max_tokens = int(kwargs.pop("max_tokens", 1024))
        thinking_budget_tokens = kwargs.pop("thinking_budget_tokens", None)
        # Preserve the legacy non-thinking default while allowing an omitted
        # temperature on the thinking path.
        if thinking_budget_tokens is None and temperature is None:
            temperature = 0.0
        if max_tokens <= 0:
            raise ValueError(f"Anthropic max_tokens must be positive; got {max_tokens}.")
        if thinking_budget_tokens is not None:
            thinking_budget_tokens = int(thinking_budget_tokens)
            if thinking_budget_tokens < 1024:
                raise ValueError(
                    "Anthropic thinking_budget_tokens must be at least 1024; "
                    f"got {thinking_budget_tokens}."
                )
            if thinking_budget_tokens >= max_tokens:
                raise ValueError(
                    "Anthropic thinking_budget_tokens must be less than max_tokens; "
                    f"got budget={thinking_budget_tokens}, max_tokens={max_tokens}."
                )
            if temperature is not None:
                raise ValueError(
                    "Anthropic extended thinking is incompatible with an explicit "
                    "temperature. Omit --anthropic-temperature."
                )
        super().__init__(model=model, temperature=temperature, **kwargs)
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        self.backend_id = f"anthropic_{model}"
        if self.thinking_budget_tokens is not None:
            self.backend_id += (
                f"_messages_thinking-{self.thinking_budget_tokens}"
                f"_maxout-{self.max_tokens}"
            )
        _require_env("ANTHROPIC_API_KEY")
        try:
            import anthropic  # noqa: F401 - presence check
        except ImportError as exc:
            raise ImportError(
                "Anthropic backend requested but the 'anthropic' package is "
                "not installed.  Install it or use "
                "ScriptedReproducibilityBackend."
            ) from exc

    def _manifest_provider_options(self) -> dict:
        return {
            "provider": "anthropic",
            "api_mode": "messages",
            "thinking_enabled": self.thinking_budget_tokens is not None,
            "thinking_budget_tokens": self.thinking_budget_tokens,
            "max_tokens": self.max_tokens,
        }

    @staticmethod
    def _usage_value(usage: Any, name: str) -> int | None:
        value = getattr(usage, name, None)
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    def _call_provider(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - live call
        import anthropic

        self._last_provider_metadata = {}
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if self.thinking_budget_tokens is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
        elif self.temperature is not None:
            payload["temperature"] = self.temperature
        payload.update(self.kwargs)

        client = anthropic.Anthropic()
        resp = client.messages.create(**payload)
        stop_reason = getattr(resp, "stop_reason", None)
        usage = getattr(resp, "usage", None)
        content = list(getattr(resp, "content", []) or [])
        self._last_provider_metadata = {
            "api_mode": "messages",
            "response_id": getattr(resp, "id", None),
            "stop_reason": stop_reason,
            "input_tokens": self._usage_value(usage, "input_tokens"),
            "output_tokens": self._usage_value(usage, "output_tokens"),
            "thinking_block_count": sum(
                1 for block in content if getattr(block, "type", None) == "thinking"
            ),
            "text_block_count": sum(
                1 for block in content if getattr(block, "type", None) == "text"
            ),
        }
        if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
            raise RuntimeError(f"anthropic_incomplete stop_reason={stop_reason!r}")

        # Thinking blocks are deliberately excluded from Stage7 raw output.
        # Only the final text block(s) enter the JSON parser/checkpoint.
        parts = []
        for block in content:
            if getattr(block, "type", None) != "text":
                continue
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        if not parts:
            raise RuntimeError("anthropic_empty_text_response")
        return "".join(parts)


class GeminiBackend(LiveLLMBackend):
    """Google Gemini backend using the documented ``generateContent`` REST API.

    The implementation deliberately uses the standard library HTTP client so
    the journal extension does not add a new runtime dependency.  It requires
    ``GEMINI_API_KEY`` and records all non-secret generation controls in the
    Stage 7 backend manifest.

    Gemini 3.x guidance recommends omitting sampling controls.  Therefore
    ``temperature`` defaults to ``None`` and is not sent unless the caller
    explicitly supplies it.  Journal runs use JSON response MIME type and a
    fixed thinking level, while the existing Stage 7 parser remains the
    authoritative semantic validator for the 10D action and L1 budget.
    """

    _VALID_THINKING_LEVELS = {"minimal", "low", "medium", "high"}
    _VALID_RESPONSE_MIME_TYPES = {"application/json", "text/plain"}
    _API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(
        self,
        model: str = "gemini-3.5-flash",
        temperature: float | None = None,
        **kwargs,
    ):
        thinking_level = kwargs.pop("thinking_level", "low")
        max_output_tokens = kwargs.pop("max_output_tokens", None)
        response_mime_type = kwargs.pop("response_mime_type", "application/json")
        timeout_seconds = kwargs.pop("timeout_seconds", 180.0)
        if kwargs:
            unknown = sorted(kwargs)
            raise TypeError(f"Unsupported Gemini backend options: {unknown}")

        thinking_level = str(thinking_level).strip().lower()
        if thinking_level not in self._VALID_THINKING_LEVELS:
            raise ValueError(
                "Gemini thinking_level must be one of "
                f"{sorted(self._VALID_THINKING_LEVELS)}; got {thinking_level!r}."
            )
        if max_output_tokens is not None:
            max_output_tokens = int(max_output_tokens)
            if max_output_tokens <= 0:
                raise ValueError(
                    f"Gemini max_output_tokens must be positive; got {max_output_tokens}."
                )
        response_mime_type = str(response_mime_type).strip().lower()
        if response_mime_type not in self._VALID_RESPONSE_MIME_TYPES:
            raise ValueError(
                "Gemini response_mime_type must be one of "
                f"{sorted(self._VALID_RESPONSE_MIME_TYPES)}; got {response_mime_type!r}."
            )
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError(f"Gemini timeout_seconds must be positive; got {timeout_seconds}.")

        super().__init__(model=model, temperature=temperature)
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self.response_mime_type = response_mime_type
        self.timeout_seconds = timeout_seconds
        self.backend_id = f"google_{model}_generate-content_thinking-{thinking_level}"
        if self.max_output_tokens is not None:
            self.backend_id += f"_maxout-{self.max_output_tokens}"
        if self.response_mime_type == "application/json":
            self.backend_id += "_json"
        _require_env("GEMINI_API_KEY")

    def _manifest_provider_options(self) -> dict:
        return {
            "provider": "google",
            "api_mode": "generateContent",
            "thinking_level": self.thinking_level,
            "max_output_tokens": self.max_output_tokens,
            "response_mime_type": self.response_mime_type,
            "timeout_seconds": self.timeout_seconds,
            "temperature_sent": self.temperature is not None,
        }

    def _post_json(self, *, url: str, body: dict, api_key: str) -> dict:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url=url,
            data=encoded,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = ""
            raise RuntimeError(
                f"gemini_http_error status={exc.code} detail={detail[:1000]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"gemini_transport_error: {exc.reason}") from exc
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RuntimeError("gemini_response_not_valid_json") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("gemini_response_root_must_be_object")
        return payload

    @staticmethod
    def _extract_response_text(payload: dict) -> tuple[str, dict]:
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            prompt_feedback = payload.get("promptFeedback")
            raise RuntimeError(
                f"gemini_no_candidates prompt_feedback={prompt_feedback!r}"
            )
        candidate = candidates[0]
        if not isinstance(candidate, dict):
            raise RuntimeError("gemini_candidate_must_be_object")
        finish_reason = candidate.get("finishReason")
        if finish_reason not in (None, "STOP"):
            raise RuntimeError(f"gemini_incomplete finish_reason={finish_reason!r}")
        content = candidate.get("content")
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list):
            raise RuntimeError("gemini_candidate_content_parts_missing")
        text_parts = [part.get("text") for part in parts if isinstance(part, dict) and part.get("text")]
        if not text_parts:
            raise RuntimeError("gemini_candidate_contains_no_text")
        metadata = {
            "response_id": payload.get("responseId"),
            "model_version": payload.get("modelVersion"),
            "finish_reason": finish_reason,
            "usage_metadata": payload.get("usageMetadata"),
        }
        return "".join(str(part) for part in text_parts), metadata

    def _call_provider(self, system_prompt: str, user_prompt: str) -> str:  # pragma: no cover - live call
        self._last_provider_metadata = {}
        generation_config: dict[str, Any] = {
            "thinkingConfig": {"thinkingLevel": self.thinking_level.upper()},
            "responseMimeType": self.response_mime_type,
        }
        if self.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = int(self.max_output_tokens)
        if self.temperature is not None:
            generation_config["temperature"] = float(self.temperature)
        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {"role": "user", "parts": [{"text": user_prompt}]},
            ],
            "generationConfig": generation_config,
        }
        encoded_model = urllib.parse.quote(self.model, safe="")
        url = f"{self._API_ROOT}/{encoded_model}:generateContent"
        payload = self._post_json(url=url, body=body, api_key=os.environ["GEMINI_API_KEY"])
        text, provider_metadata = self._extract_response_text(payload)
        self._last_provider_metadata = provider_metadata
        return text


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def make_backend(backend_spec: str, **kwargs) -> LLMBackend:
    """Construct a backend from a string identifier.

    Recognized values:

    * ``"scripted"`` — :class:`ScriptedReproducibilityBackend`
    * ``"openai:<model>"`` — :class:`OpenAIBackend` (requires
      ``OPENAI_API_KEY``)
    * ``"anthropic:<model>"`` — :class:`AnthropicBackend` (requires
      ``ANTHROPIC_API_KEY``)
    * ``"gemini:<model>"`` — :class:`GeminiBackend` (requires
      ``GEMINI_API_KEY``)
    """
    spec = backend_spec.strip().lower()
    if spec == "scripted":
        return ScriptedReproducibilityBackend(**kwargs)
    if spec.startswith("openai:"):
        model = backend_spec.split(":", 1)[1]
        return OpenAIBackend(model=model, **kwargs)
    if spec.startswith("anthropic:"):
        model = backend_spec.split(":", 1)[1]
        return AnthropicBackend(model=model, **kwargs)
    if spec.startswith("gemini:"):
        model = backend_spec.split(":", 1)[1]
        return GeminiBackend(model=model, **kwargs)
    raise ValueError(
        f"Unknown backend spec: {backend_spec!r}.  Use 'scripted', "
        f"'openai:<model>', 'anthropic:<model>', or 'gemini:<model>'."
    )
