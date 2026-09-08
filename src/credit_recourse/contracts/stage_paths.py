from __future__ import annotations

from pathlib import Path
from typing import Any

CANONICAL_STAGE_DIRS: dict[str, str] = {
    "stage0": "stage0_oracle_foundation",
    "stage1_inputs": "stage1_oracle_inputs",
    "stage1_backends": "stage1_oracle_backends",
    "stage2": "stage2_candidate_projection",
    "stage3": "stage3_acd_ssl",
    "stage4": "stage4_candidate_bc",
    "stage5": "stage5_candidate_iql",
    "stage6": "stage6_candidate_selector_eval",
    "stage7": "stage7_llm_action_generation",
    "stage8": "stage8_llm_multi_oracle_eval",
    "stage9": "stage9_llm_rl_comparison",
}

# Deprecated design-document aliases.  Verifiers should report these only as aliases;
# production runners use CANONICAL_STAGE_DIRS above.
DEPRECATED_STAGE_DIR_ALIASES: dict[str, str] = {
    "stage2_rl_data_action_reward_projection": "stage2_candidate_projection",
    "stage3_rl_ssl_encoder": "stage3_acd_ssl",
    "stage4_rl_candidate_bc": "stage4_candidate_bc",
    "stage5_rl_candidate_iql": "stage5_candidate_iql",
    "stage6_rl_candidate_selector_eval": "stage6_candidate_selector_eval",
}


def final_root(project_root: Path) -> Path:
    return Path(project_root).resolve() / "data" / "final_freeze"


def stage_dir(project_root: Path, stage_key: str) -> Path:
    key = DEPRECATED_STAGE_DIR_ALIASES.get(stage_key, stage_key)
    name = CANONICAL_STAGE_DIRS.get(key, key)
    return final_root(project_root) / name


def as_manifest(project_root: Path) -> dict[str, Any]:
    root = final_root(project_root)
    return {
        "contract_version": "canonical_stage_paths_v1_runner_paths_are_source_of_truth",
        "final_root": str(root),
        "canonical_stage_dirs": CANONICAL_STAGE_DIRS,
        "deprecated_aliases": DEPRECATED_STAGE_DIR_ALIASES,
        "policy": "Production runners and stage boundary verifiers use canonical runner paths; deprecated design-document aliases are not accepted as output roots.",
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", required=True)
    args = ap.parse_args()
    print(json.dumps(as_manifest(Path(args.project_root)), ensure_ascii=False, indent=2))
