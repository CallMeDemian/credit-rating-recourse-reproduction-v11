from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import yaml
from credit_recourse.rl.common.io import config_root

@dataclass(frozen=True)
class SeedProtocol:
    path: Path
    sha256: str
    raw: dict

def _sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def load_seed_protocol(project_root: Path) -> SeedProtocol:
    path=config_root(project_root)/'seed_protocol.yaml'
    if not path.exists(): raise FileNotFoundError(f'Missing seed protocol: {path}')
    raw=yaml.safe_load(path.read_text(encoding='utf-8')) or {}
    return SeedProtocol(path=path, sha256=_sha256_file(path), raw=raw)

def seed_for(protocol: SeedProtocol, key: str) -> int:
    node=protocol.raw
    for part in key.split('.'):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f'seed_protocol.yaml missing key: {key} (failed at {part})')
        node=node[part]
    if not isinstance(node, int): raise ValueError(f'seed_protocol.yaml key {key} must be int')
    return int(node)
