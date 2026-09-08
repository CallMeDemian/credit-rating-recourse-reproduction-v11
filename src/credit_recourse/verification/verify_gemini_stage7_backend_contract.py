from __future__ import annotations

"""Contract-faithful offline verifier for the Stage 7 Gemini backend.

No external API call is made.  The verifier injects a deterministic fake
``generateContent`` response and checks request construction, backend identity,
JSON parsing, provider metadata, fail-fast handling, CLI reachability, and the
C4R journal amendment runner/config linkage.
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.llm_backends import (
    GeminiBackend,
    LLMRequest,
    make_backend,
)

SCHEMA_VERSION = "gemini_stage7_backend_contract_v2"
EXPECTED_MODEL = "gemini-3.1-flash-lite"
EXPECTED_BACKEND_ID = (
    "google_gemini-3.1-flash-lite_generate-content_thinking-low_maxout-4096_json"
)


def _request(row_id: int = 17) -> LLMRequest:
    return LLMRequest(
        row_id=int(row_id),
        condition="C4",
        mode="free_form_10d",
        information_condition="IC-b",
        prompt={
            "firm_state": {"derived__debt_to_assets": 0.61},
            "candidate_library": {},
            "action_budget_contract": {
                "enabled": True,
                "label": "gemini_contract_smoke_0p75",
                "l1_budget": 0.75,
                "budgeted_conditions": ["C4", "C4R", "C6"],
                "budgeted_modes": ["free_form_10d"],
                "tolerance": 1.0e-9,
            },
        },
    )


def _valid_action_json() -> str:
    action = {
        "ppe_pct": -0.10,
        "inv_turnover_chg": 0.10,
        "ar_turnover_chg": 0.10,
        "ap_turnover_chg": 0.0,
        "short_debt_pct": -0.10,
        "long_debt_pct": -0.10,
        "bond_pct": 0.0,
        "revenue_growth": 0.10,
        "cogs_ratio_chg": -0.05,
        "sga_ratio_chg": -0.05,
    }
    return json.dumps(
        {
            "mode": "free_form_10d",
            "action_vector": action,
            "rationale": "contract smoke",
            "diagnosis": {"primary_weakness": "leverage"},
            "confidence": "medium",
        },
        ensure_ascii=False,
    )


def verify(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    errors: list[str] = []
    source_files = {
        "backend": root
        / "src"
        / "credit_recourse"
        / "rl"
        / "pipelines"
        / "final_stage7_llm_action_generation"
        / "llm_backends.py",
        "stage7_cli": root
        / "src"
        / "credit_recourse"
        / "rl"
        / "pipelines"
        / "final_stage7_llm_action_generation"
        / "pipeline.py",
        "runner_cli": root / "src" / "credit_recourse" / "utils" / "run_llm_stages.py",
        "journal_runner": root / "tools" / "run_c4r_journal_grid.ps1",
        "amendment_v2": root
        / "src"
        / "credit_recourse"
        / "configs"
        / "c4r_journal_extension_prereg_v2.json",
        "amendment_v3": root
        / "src"
        / "credit_recourse"
        / "configs"
        / "c4r_journal_extension_prereg_v3.json",
    }
    for label, path in source_files.items():
        if not path.is_file():
            errors.append(f"required Gemini contract file missing ({label}): {path}")

    marker_contracts = {
        "backend": (
            "class GeminiBackend",
            "GEMINI_API_KEY",
            "generateContent",
            "thinkingConfig",
            "responseMimeType",
            "gemini_incomplete",
            "gemini:<model>",
        ),
        "stage7_cli": (
            "--gemini-thinking-level",
            "--gemini-max-output-tokens",
            "--gemini-response-mime-type",
            "--gemini-timeout-seconds",
        ),
        "runner_cli": (
            "gemini_options",
            "--gemini-thinking-level",
            "--gemini-max-output-tokens",
            "--gemini-response-mime-type",
            "--gemini-timeout-seconds",
        ),
        "journal_runner": (
            "GEMINI_API_KEY",
            "ReuseGridManifest",
            "PASS_REUSED",
            "AllowIncompleteGrid",
            "verify_gemini_stage7_backend_contract",
            "$RequestedCohortIds = @(Normalize-StringSet $CohortIds)",
            "$RequestedBudgetLabels = @(Normalize-StringSet $BudgetLabels)",
            "$SelectedCohortIds = @(",
            "$SelectedBudgetLabels = @(",
        ),
    }
    for label, markers in marker_contracts.items():
        path = source_files[label]
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8-sig")
        for marker in markers:
            if marker not in text:
                errors.append(f"{label} marker missing: {marker}")

    runner_static_contract = {
        "singleton_filter_arrays_preserved": False,
    }
    journal_runner = source_files["journal_runner"]
    if journal_runner.is_file():
        runner_text = journal_runner.read_text(encoding="utf-8-sig")
        runner_static_contract["singleton_filter_arrays_preserved"] = all(
            marker in runner_text
            for marker in (
                "$RequestedCohortIds = @(Normalize-StringSet $CohortIds)",
                "$RequestedBudgetLabels = @(Normalize-StringSet $BudgetLabels)",
                "$SelectedCohortIds = @(",
                "$SelectedBudgetLabels = @(",
            )
        )

    amendments: dict[str, dict[str, Any]] = {}
    for key in ("amendment_v2", "amendment_v3"):
        path = source_files[key]
        if not path.is_file():
            continue
        try:
            amendments[key] = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as exc:
            errors.append(f"{key} JSON load failed: {type(exc).__name__}: {exc}")

    amendment_v2 = amendments.get("amendment_v2", {})
    if amendment_v2:
        if amendment_v2.get("schema_version") != "c4r_journal_extension_prereg_v2":
            errors.append("historical Gemini 3.5 amendment schema_version mismatch")
        v2_cohorts = {str(x.get("cohort_id")): x for x in amendment_v2.get("cohorts", [])}
        gemini35 = v2_cohorts.get("gemini35flash")
        if not isinstance(gemini35, dict):
            errors.append("historical Gemini amendment cohort gemini35flash missing")
        else:
            historical_expected = {
                "backend": "gemini:gemini-3.5-flash",
                "run_role": "paper_c4r_journal_ext_gemini35flash",
                "execution_policy": "generate",
                "thinking_level": "low",
                "max_output_tokens": 1200,
                "response_mime_type": "application/json",
            }
            for key, value in historical_expected.items():
                if gemini35.get(key) != value:
                    errors.append(
                        f"historical Gemini 3.5 amendment {key} mismatch: "
                        f"expected={value!r}, observed={gemini35.get(key)!r}"
                    )

    amendment_v3 = amendments.get("amendment_v3", {})
    if amendment_v3:
        if amendment_v3.get("schema_version") != "c4r_journal_extension_prereg_v3":
            errors.append("Flash-Lite amendment schema_version mismatch")
        if (amendment_v3.get("amends") or {}).get("schema_version") != "c4r_journal_extension_prereg_v2":
            errors.append("Flash-Lite amendment must explicitly amend v2")
        cohorts = {str(x.get("cohort_id")): x for x in amendment_v3.get("cohorts", [])}
        flashlite = cohorts.get("gemini31flashlite")
        gpt = cohorts.get("gpt54mini")
        if not isinstance(flashlite, dict):
            errors.append("Flash-Lite amendment cohort gemini31flashlite missing")
        else:
            expected = {
                "backend": "gemini:gemini-3.1-flash-lite",
                "run_role": "paper_c4r_journal_ext_gemini31flashlite",
                "execution_policy": "generate",
                "thinking_level": "low",
                "max_output_tokens": 4096,
                "response_mime_type": "application/json",
            }
            for key, value in expected.items():
                if flashlite.get(key) != value:
                    errors.append(
                        f"Flash-Lite amendment {key} mismatch: "
                        f"expected={value!r}, observed={flashlite.get(key)!r}"
                    )
            if flashlite.get("temperature") is not None:
                errors.append("Gemini 3.1 Flash-Lite amendment must omit temperature (JSON null)")
        if not isinstance(gpt, dict) or gpt.get("execution_policy") != "reuse_required":
            errors.append("GPT cohort must remain reuse_required in the Flash-Lite amendment")
        excluded = {
            str(x.get("cohort_id")): str(x.get("analysis_status"))
            for x in amendment_v3.get("excluded_feasibility_cohorts", [])
        }
        required_status = "PROVIDER_FEASIBILITY_FAILURE_NOT_INCLUDED_IN_MATCHED_INFERENCE"
        for failed_cohort in ("haiku45", "gemini35flash"):
            if excluded.get(failed_cohort) != required_status:
                errors.append(f"{failed_cohort} feasibility failure exclusion is not frozen in v3")
        amended = amendment_v3.get("amends") or {}
        if int(amended.get("observed_checkpoint_rows", -1)) != 534:
            errors.append("Gemini 3.5 failed checkpoint row count is not frozen at 534")
        if int(amended.get("observed_simulator_routed_rows", -1)) != 520:
            errors.append("Gemini 3.5 simulator-routed row count is not frozen at 520")
        if int(amended.get("observed_translational_failure_rows", -1)) != 14:
            errors.append("Gemini 3.5 translational failure count is not frozen at 14")
        if "MAX_TOKENS" not in str(amended.get("terminal_failure", "")):
            errors.append("Gemini 3.5 MAX_TOKENS terminal failure is not frozen in v3")

    runtime: dict[str, Any] = {}
    previous_key = os.environ.get("GEMINI_API_KEY")
    try:
        os.environ["GEMINI_API_KEY"] = "unit-test-secret-must-not-leak"
        backend = GeminiBackend(
            model=EXPECTED_MODEL,
            thinking_level="low",
            max_output_tokens=4096,
            response_mime_type="application/json",
            timeout_seconds=180.0,
        )
        if backend.backend_id != EXPECTED_BACKEND_ID:
            errors.append(
                f"Gemini backend_id mismatch: expected={EXPECTED_BACKEND_ID!r}, observed={backend.backend_id!r}"
            )
        factory = make_backend(
            "gemini:gemini-3.1-flash-lite",
            thinking_level="low",
            max_output_tokens=4096,
            response_mime_type="application/json",
            timeout_seconds=180.0,
        )
        if not isinstance(factory, GeminiBackend):
            errors.append("make_backend did not construct GeminiBackend")

        captured: dict[str, Any] = {}

        def fake_post_json(*, url: str, body: dict, api_key: str) -> dict:
            captured.update({"url": url, "body": body, "api_key": api_key})
            return {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": _valid_action_json()}]},
                    }
                ],
                "responseId": "synthetic-gemini-response",
                "modelVersion": EXPECTED_MODEL,
                "usageMetadata": {
                    "promptTokenCount": 101,
                    "candidatesTokenCount": 55,
                    "totalTokenCount": 156,
                },
            }

        backend._post_json = fake_post_json  # type: ignore[method-assign]
        response = backend.generate(_request())
        if response.parse_error is not None or not isinstance(response.parsed_json, dict):
            errors.append(f"Gemini deterministic response failed to parse: {response.parse_error}")
        else:
            vector = response.parsed_json.get("action_vector")
            if not isinstance(vector, dict) or len(vector) != 10:
                errors.append("Gemini parsed action_vector must contain exactly 10 axes")
            else:
                observed_l1 = sum(abs(float(value)) for value in vector.values())
                if observed_l1 > 0.75 + 1.0e-12:
                    errors.append(
                        f"Gemini synthetic action violates the 0.75 L1 contract: {observed_l1}"
                    )
        body = captured.get("body") or {}
        generation = body.get("generationConfig") if isinstance(body, dict) else {}
        if generation.get("thinkingConfig") != {"thinkingLevel": "LOW"}:
            errors.append(f"Gemini thinkingConfig mismatch: {generation.get('thinkingConfig')!r}")
        if generation.get("maxOutputTokens") != 4096:
            errors.append("Gemini maxOutputTokens was not propagated")
        if generation.get("responseMimeType") != "application/json":
            errors.append("Gemini JSON MIME contract was not propagated")
        if "temperature" in generation:
            errors.append("Gemini request unexpectedly sent temperature despite amendment null")
        if captured.get("api_key") != "unit-test-secret-must-not-leak":
            errors.append("Gemini API key was not passed to HTTP layer")
        if not str(captured.get("url", "")).endswith(
            "/gemini-3.1-flash-lite:generateContent"
        ):
            errors.append(f"Gemini generateContent URL mismatch: {captured.get('url')!r}")
        serialized_metadata = json.dumps(response.backend_metadata, ensure_ascii=False)
        if "unit-test-secret-must-not-leak" in serialized_metadata:
            errors.append("Gemini API key leaked into response backend metadata")
        provider_response = response.backend_metadata.get("provider_response") or {}
        if provider_response.get("response_id") != "synthetic-gemini-response":
            errors.append("Gemini response ID metadata missing")

        def fake_concurrent(*, url: str, body: dict, api_key: str) -> dict:
            user_text = str(body["contents"][0]["parts"][0]["text"])
            row_id = int(json.loads(user_text)["row_id"])
            # Force overlapping completion order so shared mutable response
            # metadata would be exposed by the assertion below.
            time.sleep(0.04 if row_id == 17 else 0.01)
            return {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": _valid_action_json()}]},
                    }
                ],
                "responseId": f"synthetic-row-{row_id}",
                "modelVersion": EXPECTED_MODEL,
            }

        backend._post_json = fake_concurrent  # type: ignore[method-assign]
        with ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_responses = list(pool.map(backend.generate, [_request(17), _request(18)]))
        concurrent_ids = {
            item.request.row_id: (item.backend_metadata.get("provider_response") or {}).get("response_id")
            for item in concurrent_responses
        }
        expected_concurrent_ids = {17: "synthetic-row-17", 18: "synthetic-row-18"}
        if concurrent_ids != expected_concurrent_ids:
            errors.append(
                "Gemini concurrent response metadata was cross-attributed: "
                f"expected={expected_concurrent_ids}, observed={concurrent_ids}"
            )

        def fake_incomplete(*, url: str, body: dict, api_key: str) -> dict:
            return {
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS",
                        "content": {"parts": [{"text": "{\"mode\":\"free_form_10d\""}]},
                    }
                ]
            }

        backend._post_json = fake_incomplete  # type: ignore[method-assign]
        incomplete = backend.generate(_request())
        if not str(incomplete.parse_error).startswith("provider_call_failed: RuntimeError: gemini_incomplete"):
            errors.append(
                f"Gemini non-STOP response did not fail fast: {incomplete.parse_error!r}"
            )
        if "provider_response" in incomplete.backend_metadata:
            errors.append("Gemini failed call retained stale provider-response metadata")
        runtime = {
            "backend_id": backend.backend_id,
            "request_url": captured.get("url"),
            "json_parse_pass": response.parse_error is None,
            "ten_axis_pass": isinstance(response.parsed_json, dict)
            and isinstance(response.parsed_json.get("action_vector"), dict)
            and len(response.parsed_json["action_vector"]) == 10,
            "synthetic_action_l1": (
                sum(abs(float(value)) for value in response.parsed_json["action_vector"].values())
                if isinstance(response.parsed_json, dict)
                and isinstance(response.parsed_json.get("action_vector"), dict)
                else None
            ),
            "temperature_omitted": "temperature" not in generation,
            "concurrent_metadata_isolated": concurrent_ids == expected_concurrent_ids,
            "incomplete_response_fail_fast": str(incomplete.parse_error).startswith(
                "provider_call_failed: RuntimeError: gemini_incomplete"
            ),
            "failed_call_metadata_cleared": "provider_response" not in incomplete.backend_metadata,
        }

        del os.environ["GEMINI_API_KEY"]
        try:
            GeminiBackend(model=EXPECTED_MODEL)
        except EnvironmentError:
            pass
        else:
            errors.append("GeminiBackend did not require GEMINI_API_KEY")
    except Exception as exc:
        errors.append(f"Gemini backend synthetic verification failed: {type(exc).__name__}: {exc}")
    finally:
        if previous_key is None:
            os.environ.pop("GEMINI_API_KEY", None)
        else:
            os.environ["GEMINI_API_KEY"] = previous_key

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "expected_model": EXPECTED_MODEL,
        "expected_backend_id": EXPECTED_BACKEND_ID,
        "source_files": {key: str(value) for key, value in source_files.items()},
        "runner_static_contract": runner_static_contract,
        "synthetic_runtime": runtime,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args(argv)
    result = verify(Path(args.project_root))
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
