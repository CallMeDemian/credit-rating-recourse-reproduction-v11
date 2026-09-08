from __future__ import annotations

"""Opt-in strict GPU/torch determinism for RL training stages (RL-DET-001).

The frozen 2026-06-06 run (and the 2026-06-21 reproduction) demonstrated that
seed-fixing alone does not make GPU-trained Stage 3-5 artifacts bit-reproducible
(reproducibility_comparison_original_vs_reproduction.md §6): PyTorch CUDA
kernels are non-deterministic by default and ``torch.use_deterministic_algorithms``
was never enforced.  This module provides the opt-in control.

Design constraints:

* **Default off.**  Enabling determinism changes training numerics, so the
  frozen s2 policy is *not* recoverable by turning this on.  The flag exists so
  that *future* runs can be bit-replayed on an identical GPU/driver/library
  stack.  Stages record the setting in metadata either way.
* **Fail-fast on late activation.**  ``CUBLAS_WORKSPACE_CONFIG`` must be set
  before the CUDA context is created.  If CUDA is already initialized when this
  function runs, we raise instead of silently claiming determinism.
* **Lazy torch import.**  The module stays importable in torch-free
  environments (verifiers, doc tooling).
"""

import os
from typing import Any

DEFAULT_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def torch_determinism_disabled_metadata() -> dict[str, Any]:
    """Metadata block recorded when strict determinism is NOT requested."""
    return {
        "enabled": False,
        "policy": "default_nondeterministic_gpu_training",
        "note": (
            "seed fixes RNG streams only; CUDA kernel nondeterminism is not "
            "controlled (see reproducibility comparison §6). Policy artifacts "
            "are a run distribution, not a bit-reproducible point."
        ),
    }


def enable_torch_determinism(
    *, cublas_workspace_config: str = DEFAULT_CUBLAS_WORKSPACE_CONFIG
) -> dict[str, Any]:
    """Enable strict torch determinism for the current process.

    Must be called BEFORE any CUDA work (including ``torch.cuda.manual_seed_all``,
    which initializes the CUDA context).  Raises if CUDA is already initialized,
    because at that point the CUBLAS workspace setting can no longer be
    guaranteed to take effect and a silent partial-determinism claim would be
    worse than an honest failure.

    Returns a metadata dict for the stage ``metadata.json``.
    """
    applied: dict[str, Any] = {
        "enabled": True,
        "cublas_workspace_config": None,
        "use_deterministic_algorithms": False,
        "cudnn_deterministic": None,
        "cudnn_benchmark": None,
        "torch_version": None,
        "cuda_available": None,
        "notes": [],
    }

    prior = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if prior is not None and prior != cublas_workspace_config:
        applied["notes"].append(
            f"CUBLAS_WORKSPACE_CONFIG was already set to {prior!r}; respecting the existing value."
        )
        applied["cublas_workspace_config"] = prior
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = cublas_workspace_config
        applied["cublas_workspace_config"] = cublas_workspace_config

    import torch  # lazy: keep module importable without torch

    applied["torch_version"] = str(torch.__version__)
    applied["cuda_available"] = bool(torch.cuda.is_available())

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        raise RuntimeError(
            "enable_torch_determinism() was called after the CUDA context was "
            "initialized; CUBLAS_WORKSPACE_CONFIG cannot be guaranteed to take "
            "effect. Call it before set_seed()/any CUDA op (RL-DET-001)."
        )

    torch.use_deterministic_algorithms(True)
    applied["use_deterministic_algorithms"] = True

    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        applied["cudnn_deterministic"] = True
        applied["cudnn_benchmark"] = False

    return applied
