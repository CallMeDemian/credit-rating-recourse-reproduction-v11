"""V10 analysis-only artifact path resolver.

The canonical Stage0-9 producer keeps the original thesis_repo path contract.
V10 post-freeze analysis, however, was authored against an explicit
``artifact_root`` argument.  Keeping this resolver in the analysis namespace
allows both original codebases to coexist without changing Stage0-9 behavior.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from credit_recourse.common.run_context import resolve_artifact_root


CANONICAL_STAGE_DIRS: dict[str, str] = {
    "stage0": "stage0_oracle_foundation",
    "stage1_inputs": "stage1_oracle_inputs",
    "stage1_backends": "stage1_oracle_backends",
    "stage2": "stage2_candidate_projection",
    "stage3": "stage3_acd_ssl",
    "stage4": "stage4_candidate_bc",
    "stage5": "stage5_candidate_iql",
    "stage6": "stage6_candidate_selector_eval",
    "stage6_multi": "stage6_multi_oracle_eval",
}

LLM_BUNDLE_STAGE_DIRS: dict[str, str] = {
    "stage7": "stage7_llm_action_generation",
    "stage8": "stage8_llm_multi_oracle_eval",
    "stage9": "stage9_llm_rl_comparison",
}

DEPRECATED_STAGE_DIR_ALIASES: dict[str, str] = {
    "stage2_rl_data_action_reward_projection": "stage2_candidate_projection",
    "stage3_rl_ssl_encoder": "stage3_acd_ssl",
    "stage4_rl_candidate_bc": "stage4_candidate_bc",
    "stage5_rl_candidate_iql": "stage5_candidate_iql",
    "stage6_rl_candidate_selector_eval": "stage6_candidate_selector_eval",
}


def final_root(
    project_root: Path,
    artifact_root: Path | str | None = None,
    *,
    must_exist: bool = True,
    allow_legacy: bool = False,
) -> Path:
    return resolve_artifact_root(
        project_root,
        explicit=artifact_root,
        must_exist=must_exist,
        allow_legacy=allow_legacy,
    )


def stage_dir(
    project_root: Path,
    stage_key: str,
    artifact_root: Path | str | None = None,
    *,
    must_exist_root: bool = True,
) -> Path:
    key = DEPRECATED_STAGE_DIR_ALIASES.get(stage_key, stage_key)
    if key in LLM_BUNDLE_STAGE_DIRS:
        raise ValueError(
            f"{key} is owned by an LLM run bundle. "
            f"Resolve artifacts/llm_runs/<run_id>/{LLM_BUNDLE_STAGE_DIRS[key]} explicitly."
        )
    name = CANONICAL_STAGE_DIRS.get(key, key)
    return final_root(
        project_root,
        artifact_root,
        must_exist=must_exist_root,
    ) / name


def as_manifest(
    project_root: Path,
    artifact_root: Path | str | None = None,
) -> dict[str, Any]:
    root = final_root(project_root, artifact_root)
    return {
        "contract_version": "explicit_artifact_root_v2",
        "artifact_root": str(root),
        "canonical_stage_dirs": CANONICAL_STAGE_DIRS,
        "llm_bundle_stage_dirs": LLM_BUNDLE_STAGE_DIRS,
        "deprecated_aliases": DEPRECATED_STAGE_DIR_ALIASES,
        "policy": (
            "Implicit data/final_freeze resolution is forbidden. Stage0-6 use "
            "an explicit artifact root; logical Stage7-9 live only inside "
            "llm_runs/<run_id>."
        ),
    }
