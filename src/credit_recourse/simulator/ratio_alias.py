"""
ratio_alias.py — final-freeze R-code alias canonicalization utility.

Final runtime must not consult deprecated intermediate/canonical roots. Alias maps
are loaded only from explicit env override, final_freeze config/stage1 artifacts,
or local package fallback JSON files. If no artifact exists, a conservative
hardcoded fallback is used; the final Stage01 verifier requires the actual
Stage1 alias artifacts for paper runs.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, Optional

FALLBACK_ALIAS_MAP: Dict[str, str] = {
    "R086": "R085",
    "R131": "R094",
    "R003": "R002",
    "R004": "R002",
    "R014": "R013",
    "R080": "R079",
    "R088": "R089",
    "R114": "R102",
    "R118": "R076",
    "R119": "R077",
    "R127": "R117",
    "R163": "R159",
    "R164": "R158",
    "R165": "R160",
    "R172": "R171",
    "R173": "R171",
    "R210": "R076",
    "R211": "R064",
    "R216": "R010",
}


def _load_map_from_file(path: str | Path) -> Optional[Dict[str, str]]:
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return None
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if raw and isinstance(next(iter(raw.values())), str):
            return dict(raw)
        if raw and isinstance(next(iter(raw.values())), dict):
            return {k: v["canonical_ratio_id"] for k, v in raw.items() if "canonical_ratio_id" in v}
    except Exception:
        pass
    return None


def _project_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ["CR_PROJECT_ROOT", "PROJECT_ROOT"]:
        env = os.environ.get(env_name, "").strip()
        if env:
            roots.append(Path(env).resolve())
    here = Path(__file__).resolve()
    roots.append(Path.cwd().resolve())
    for parent in here.parents:
        if (parent / "data" / "final_freeze").exists() or (parent / "pyproject.toml").exists():
            roots.append(parent)
    if len(here.parents) > 3:
        roots.append(here.parents[3])  # expected repo root for <root>/src/credit_recourse/simulator/ratio_alias.py
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _build_alias_map() -> Dict[str, str]:
    search_paths: list[Path] = []
    env_path = os.environ.get("CR_RATIO_ALIAS_MAP", "").strip()
    if env_path:
        search_paths.append(Path(env_path))

    rels = [
        "data/final_freeze/configs/simulator_ratio_alias_map.json",
        "data/final_freeze/configs/duplicate_ratio_alias_map.json",
        "data/final_freeze/stage1_oracle_inputs/stage00_04_variable_selection/simulator_ratio_alias_map.json",
        "data/final_freeze/stage1_oracle_inputs/stage00_04_variable_selection/duplicate_ratio_alias_map.json",
        "data/final_freeze/stage1_oracle_inputs/stage00_02_financial_ratio_engineering/simulator_ratio_alias_map.json",
        "data/final_freeze/stage1_oracle_inputs/stage00_02_financial_ratio_engineering/duplicate_ratio_alias_map.json",
        "data/final_freeze/stage1_oracle_backends/alpha/simulator_ratio_alias_map.json",
    ]
    for root in _project_roots():
        for rel in rels:
            search_paths.append(root / rel)

    here = Path(__file__).resolve().parent
    search_paths.extend([here / "simulator_ratio_alias_map.json", here / "duplicate_ratio_alias_map.json"])

    for path in search_paths:
        m = _load_map_from_file(path)
        if m:
            cleaned = {k: v for k, v in m.items() if k != v}
            if cleaned:
                return cleaned
    return dict(FALLBACK_ALIAS_MAP)


ALIAS_MAP: Dict[str, str] = _build_alias_map()


def canonicalize(var_id: str, alias_map: Optional[Dict[str, str]] = None) -> str:
    m = alias_map if alias_map is not None else ALIAS_MAP
    return m.get(var_id, var_id)


def canonicalize_dict(d: Dict[str, object], alias_map: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    m = alias_map if alias_map is not None else ALIAS_MAP
    result: Dict[str, object] = {}
    for k, v in d.items():
        canonical_k = m.get(k, k)
        if canonical_k not in result:
            result[canonical_k] = v
    return result


def canonicalize_list(vars_: list, alias_map: Optional[Dict[str, str]] = None) -> list:
    m = alias_map if alias_map is not None else ALIAS_MAP
    seen = set()
    result = []
    for v in vars_:
        c = m.get(v, v)
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result
