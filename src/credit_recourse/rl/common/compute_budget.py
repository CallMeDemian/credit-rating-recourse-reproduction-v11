from __future__ import annotations
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import hashlib
import yaml
import torch
from credit_recourse.rl.common.io import config_root

@dataclass(frozen=True)
class ComputeBudget:
    profile_name: str
    profile_sha256: str
    profile_path: Path
    stage_key: str
    seeds: list[int]
    epochs: int | None
    epochs_sweep_cap: int | None
    epochs_override: int | None
    batch_size: int
    amp_dtype: str | None
    def as_metadata(self) -> dict[str, Any]:
        return {
            'compute_budget_profile_name': self.profile_name,
            'compute_budget_profile_sha256': self.profile_sha256,
            'compute_budget_stage_key': self.stage_key,
            'compute_budget_seeds': list(self.seeds),
            'compute_budget_epochs': self.epochs,
            'compute_budget_epochs_sweep_cap': self.epochs_sweep_cap,
            'compute_budget_epochs_override': self.epochs_override,
            'compute_budget_batch_size': self.batch_size,
            'compute_budget_amp_dtype': self.amp_dtype,
        }

def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_compute_budget(project_root: Path, stage_key: str) -> ComputeBudget:
    path=config_root(project_root)/'compute_budget_profile.yaml'
    if not path.exists():
        raise FileNotFoundError(f'Missing compute budget profile: {path}')
    raw=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    profile_name=raw.get('active_profile')
    if not profile_name: raise ValueError('compute_budget_profile.yaml missing active_profile')
    profiles=raw.get('profiles') or {}
    if profile_name not in profiles: raise ValueError(f'Unknown compute_budget profile: {profile_name}')
    profile=profiles[profile_name]
    if stage_key.startswith('final_refit_stage'):
        n=stage_key.replace('final_refit_stage','')
        block=(profile.get('final_refit') or {}).get(f'stage{n}')
        if block is None: raise ValueError(f'Profile {profile_name} missing final_refit.stage{n}')
    else:
        block=profile.get(stage_key)
        if block is None: raise ValueError(f'Profile {profile_name} missing stage block: {stage_key}')
    return ComputeBudget(profile_name=profile_name, profile_sha256=_sha256_file(path), profile_path=path, stage_key=stage_key,
        seeds=[int(s) for s in (block.get('seeds') or [])],
        epochs=int(block['epochs']) if block.get('epochs') is not None else None,
        epochs_sweep_cap=int(block['epochs_sweep_cap']) if block.get('epochs_sweep_cap') is not None else None,
        epochs_override=int(block['epochs_override']) if block.get('epochs_override') is not None else None,
        batch_size=int(block.get('batch_size',256)), amp_dtype=block.get('amp_dtype'))

def effective_epochs(grid_value: int, budget: ComputeBudget) -> tuple[int, bool]:
    if budget.epochs_sweep_cap is None: return int(grid_value), False
    if int(grid_value)>int(budget.epochs_sweep_cap): return int(budget.epochs_sweep_cap), True
    return int(grid_value), False

@contextmanager
def amp_context(amp_dtype: str | None) -> Iterator[None]:
    if amp_dtype is None or not torch.cuda.is_available():
        with nullcontext(): yield
        return
    dtype={'bf16':torch.bfloat16,'fp16':torch.float16}.get(amp_dtype)
    if dtype is None: raise ValueError(f'Unsupported amp_dtype: {amp_dtype}')
    with torch.amp.autocast('cuda', dtype=dtype): yield

def setup_precision_globals() -> None:
    try: torch.set_float32_matmul_precision('high')
    except Exception: pass
    try: torch.backends.cudnn.benchmark=True
    except Exception: pass
