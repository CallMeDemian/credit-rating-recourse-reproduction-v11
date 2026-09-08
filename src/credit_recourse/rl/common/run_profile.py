from __future__ import annotations

"""Final paper run profile loader (RL-RUN-007).

``final_run_profile.yaml`` pins the frozen 2026-06-06 paper configuration for
Stages 2-6 inside the repository, so ``src`` alone is sufficient to
re-issue the exact paper invocation without the external PowerShell runner.

Sentinel policy
---------------
Values that are traceable to frozen artifacts (the reproducibility comparison
document, the June pre-experiment report, the run label
``sector0_m0p05_f0p45_liq0p45_s2_...``) are filled directly.  Values that are
recorded only inside the frozen run's own metadata artifacts carry the
sentinel ``__CONFIRM_FROM_FROZEN_RUN_ARTIFACT__`` plus a per-key
``confirm_hints`` entry naming exactly which frozen artifact/key to read.
The loader HARD-FAILS while any sentinel remains for a requested stage —
a profile with guessed values would silently produce a non-paper run labeled
as the paper run, which is strictly worse than an actionable failure.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from credit_recourse.rl.common.io import config_root

SENTINEL = "__CONFIRM_FROM_FROZEN_RUN_ARTIFACT__"
PROFILE_FILENAME = "final_run_profile.yaml"
PROFILE_SCHEMA_VERSION = "final_run_profile_v1"


@dataclass(frozen=True)
class FinalRunProfile:
    path: Path
    sha256: str
    raw: dict


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_final_run_profile(project_root: Path) -> FinalRunProfile:
    path = config_root(Path(project_root)) / PROFILE_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing final run profile: {path}. Run the config materializer "
            f"(credit_recourse.utils.materialize_final_freeze_configs) first."
        )
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"final_run_profile.yaml schema_version must be "
            f"{PROFILE_SCHEMA_VERSION!r}; got {raw.get('schema_version')!r}."
        )
    return FinalRunProfile(path=path, sha256=_sha256_file(path), raw=raw)


def _hint_for(profile: FinalRunProfile, stage_key: str, arg_key: str) -> str:
    hints = profile.raw.get("confirm_hints", {}) or {}
    return str(hints.get(f"{stage_key}.{arg_key}", "frozen run metadata / frozen runner ps1"))


def stage_argv(
    profile: FinalRunProfile, stage_key: str
) -> tuple[list[str], list[str]]:
    """Return (argv, unresolved) for one stage.

    ``argv`` excludes ``--project-root`` (the runner supplies it).
    ``unresolved`` lists sentinel keys with confirm hints; a non-empty list
    means the profile is not yet usable for this stage.
    """
    stages = profile.raw.get("stages", {}) or {}
    node = stages.get(stage_key)
    if node is None:
        raise KeyError(
            f"final_run_profile.yaml has no stage entry {stage_key!r}. "
            f"Available: {sorted(stages)}"
        )
    argv: list[str] = []
    unresolved: list[str] = []
    for k, v in (node.get("args", {}) or {}).items():
        if isinstance(v, str) and v == SENTINEL:
            unresolved.append(f"{stage_key}.args.{k}  (hint: {_hint_for(profile, stage_key, k)})")
            continue
        argv.extend([f"--{k}", str(v)])
    for k, v in (node.get("flags", {}) or {}).items():
        if isinstance(v, str) and v == SENTINEL:
            unresolved.append(f"{stage_key}.flags.{k}  (hint: {_hint_for(profile, stage_key, k)})")
            continue
        if bool(v):
            argv.append(f"--{k}")
    return argv, unresolved


def assert_profile_resolved(profile: FinalRunProfile, stage_keys: list[str]) -> None:
    """Fail fast with every unresolved sentinel across the requested stages."""
    pending: list[str] = []
    for sk in stage_keys:
        _, unresolved = stage_argv(profile, sk)
        pending.extend(unresolved)
    if pending:
        lines = "\n  - ".join(pending)
        raise ValueError(
            "final_run_profile.yaml still contains CONFIRM sentinels for the "
            "requested stages. Fill them from the frozen run artifacts before "
            "issuing a paper-profile run:\n  - " + lines
        )
