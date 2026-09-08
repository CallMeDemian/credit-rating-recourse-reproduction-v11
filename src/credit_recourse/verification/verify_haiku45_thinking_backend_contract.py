from __future__ import annotations

"""Synthetic/static verifier for the Haiku 4.5 manual-thinking pilot contract."""

import argparse
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "haiku45_thinking_backend_contract_v1"
EXPECTED_MODEL = "claude-haiku-4-5-20251001"
EXPECTED_BACKEND_ID = (
    "anthropic_claude-haiku-4-5-20251001_messages_thinking-2048_maxout-4096"
)
EXPECTED_DESIGN_STATUS = (
    "LOCKED_THIRD_AMENDMENT_AFTER_HAIKU45_NONTHINKING_FAILURE_"
    "BEFORE_HAIKU45_THINKING_PILOT"
)


def _attr(**kwargs: Any) -> Any:
    return types.SimpleNamespace(**kwargs)


def verify(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    errors: list[str] = []
    src_root = project_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    backend_path = src_root / "credit_recourse/rl/pipelines/final_stage7_llm_action_generation/llm_backends.py"
    runner_path = src_root / "credit_recourse/utils/run_llm_stages.py"
    pipeline_path = src_root / "credit_recourse/rl/pipelines/final_stage7_llm_action_generation/pipeline.py"
    prereg_path = src_root / "credit_recourse/configs/c4r_journal_extension_prereg_v4_haiku_thinking_pilot.json"
    ps_path = project_root / "tools/run_haiku45_thinking_pilot.ps1"
    for path in (backend_path, runner_path, pipeline_path, prereg_path, ps_path):
        if not path.is_file():
            errors.append(f"missing source file: {path}")

    prereg: dict[str, Any] = {}
    if prereg_path.is_file():
        try:
            prereg = json.loads(prereg_path.read_text(encoding="utf-8-sig"))
            if prereg.get("design_status") != EXPECTED_DESIGN_STATUS:
                errors.append("unexpected prereg design_status")
            scope = prereg.get("pilot_scope") or {}
            provider = prereg.get("provider_contract") or {}
            gates = prereg.get("quality_gates") or {}
            checks = {
                "model": scope.get("backend") == f"anthropic:{EXPECTED_MODEL}",
                "backend_id": scope.get("expected_backend_id") == EXPECTED_BACKEND_ID,
                "sample_size": int(scope.get("sample_size", -1)) == 50,
                "request_count": int(scope.get("planned_live_request_count", -1)) == 150,
                "thinking": provider.get("manual_extended_thinking") is True,
                "thinking_budget": int(provider.get("thinking_budget_tokens", -1)) == 2048,
                "max_tokens": int(provider.get("max_tokens", -1)) == 4096,
                "temperature_omitted": provider.get("temperature") is None,
                "stage7_rows": int(gates.get("expected_stage7_action_rows", -1)) == 150,
                "stage9_rows": int(gates.get("expected_stage9_revision_rows", -1)) == 100,
            }
            for name, ok in checks.items():
                if not ok:
                    errors.append(f"prereg contract check failed: {name}")
        except Exception as exc:
            errors.append(f"prereg parse failed: {type(exc).__name__}: {exc}")

    static_contract = {
        "runner_cli_flag": False,
        "pipeline_cli_flag": False,
        "pilot_script_flag": False,
    }
    try:
        static_contract["runner_cli_flag"] = "--anthropic-thinking-budget-tokens" in runner_path.read_text(encoding="utf-8")
        static_contract["pipeline_cli_flag"] = "--anthropic-thinking-budget-tokens" in pipeline_path.read_text(encoding="utf-8")
        static_contract["pilot_script_flag"] = "--anthropic-thinking-budget-tokens" in ps_path.read_text(encoding="utf-8")
        if not all(static_contract.values()):
            errors.append(f"static CLI contract failed: {static_contract}")
    except Exception as exc:
        errors.append(f"static contract read failed: {type(exc).__name__}: {exc}")

    runtime: dict[str, Any] = {}
    old_key = os.environ.get("ANTHROPIC_API_KEY")
    old_module = sys.modules.get("anthropic")
    captured: list[dict[str, Any]] = []

    class FakeMessages:
        def __init__(self, *, incomplete: bool = False):
            self.incomplete = incomplete

        def create(self, **payload: Any) -> Any:
            captured.append(payload)
            if self.incomplete:
                return _attr(
                    id="msg_incomplete",
                    stop_reason="max_tokens",
                    usage=_attr(input_tokens=100, output_tokens=4096),
                    content=[_attr(type="thinking", thinking="summary")],
                )
            return _attr(
                id="msg_ok",
                stop_reason="end_turn",
                usage=_attr(input_tokens=100, output_tokens=777),
                content=[
                    _attr(type="thinking", thinking="summary not persisted"),
                    _attr(type="text", text='{"mode":"free_form_10d","action_vector":{}}'),
                ],
            )

    class FakeAnthropicClient:
        def __init__(self, *, incomplete: bool = False):
            self.messages = FakeMessages(incomplete=incomplete)

    fake_module = types.SimpleNamespace(Anthropic=lambda: FakeAnthropicClient())
    try:
        os.environ["ANTHROPIC_API_KEY"] = "synthetic-key"
        sys.modules["anthropic"] = fake_module
        from credit_recourse.rl.pipelines.final_stage7_llm_action_generation.llm_backends import AnthropicBackend

        backend = AnthropicBackend(
            model=EXPECTED_MODEL,
            max_tokens=4096,
            thinking_budget_tokens=2048,
        )
        raw = backend._call_provider("system", "user")
        payload = captured[-1]
        meta = backend._last_provider_metadata
        manifest = backend.manifest()
        runtime = {
            "backend_id": backend.backend_id,
            "raw_text_only": raw == '{"mode":"free_form_10d","action_vector":{}}',
            "thinking_payload": payload.get("thinking") == {"type": "enabled", "budget_tokens": 2048},
            "max_tokens": payload.get("max_tokens") == 4096,
            "temperature_omitted": "temperature" not in payload,
            "thinking_block_count": meta.get("thinking_block_count"),
            "text_block_count": meta.get("text_block_count"),
            "usage_recorded": meta.get("output_tokens") == 777,
            "manifest_thinking_enabled": (manifest.get("provider_options") or {}).get("thinking_enabled") is True,
            "explicit_temperature_rejected": False,
            "incomplete_fail_fast": False,
            "legacy_backend_id_preserved": False,
            "legacy_temperature_preserved": False,
        }
        try:
            AnthropicBackend(
                model=EXPECTED_MODEL,
                temperature=0.0,
                max_tokens=4096,
                thinking_budget_tokens=2048,
            )
        except ValueError:
            runtime["explicit_temperature_rejected"] = True

        legacy = AnthropicBackend(model=EXPECTED_MODEL)
        runtime["legacy_backend_id_preserved"] = legacy.backend_id == f"anthropic_{EXPECTED_MODEL}"
        runtime["legacy_temperature_preserved"] = legacy.temperature == 0.0

        sys.modules["anthropic"] = types.SimpleNamespace(
            Anthropic=lambda: FakeAnthropicClient(incomplete=True)
        )
        incomplete_backend = AnthropicBackend(
            model=EXPECTED_MODEL,
            temperature=None,
            max_tokens=4096,
            thinking_budget_tokens=2048,
        )
        try:
            incomplete_backend._call_provider("system", "user")
        except RuntimeError as exc:
            runtime["incomplete_fail_fast"] = "anthropic_incomplete" in str(exc)

        expected = {
            "backend_id": runtime["backend_id"] == EXPECTED_BACKEND_ID,
            "raw_text_only": runtime["raw_text_only"],
            "thinking_payload": runtime["thinking_payload"],
            "max_tokens": runtime["max_tokens"],
            "temperature_omitted": runtime["temperature_omitted"],
            "thinking_block_count": runtime["thinking_block_count"] == 1,
            "text_block_count": runtime["text_block_count"] == 1,
            "usage_recorded": runtime["usage_recorded"],
            "manifest_thinking_enabled": runtime["manifest_thinking_enabled"],
            "explicit_temperature_rejected": runtime["explicit_temperature_rejected"],
            "incomplete_fail_fast": runtime["incomplete_fail_fast"],
            "legacy_backend_id_preserved": runtime["legacy_backend_id_preserved"],
            "legacy_temperature_preserved": runtime["legacy_temperature_preserved"],
        }
        for name, ok in expected.items():
            if not ok:
                errors.append(f"synthetic runtime check failed: {name}")
    except Exception as exc:
        errors.append(f"synthetic runtime failed: {type(exc).__name__}: {exc}")
    finally:
        if old_key is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = old_key
        if old_module is None:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = old_module

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(project_root),
        "expected_model": EXPECTED_MODEL,
        "expected_backend_id": EXPECTED_BACKEND_ID,
        "static_contract": static_contract,
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
