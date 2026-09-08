from __future__ import annotations
from pathlib import Path
import hashlib


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_jsonable(obj) -> str:
    import json
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def sha256_torch_state_dict(sd: dict) -> str:
    import io, torch
    buf = io.BytesIO()
    torch.save(sd, buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()
