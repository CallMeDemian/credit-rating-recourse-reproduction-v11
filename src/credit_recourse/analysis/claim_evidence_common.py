from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable
import yaml
FORBIDDEN_GENERATED_TOKENS = ('/mnt/data', 'thesis_repo', 'C:\\Users\\', '<exact_', 'TODO', 'TBD', 'current thesis')
CLAIM_CLASSES = {'BOUNDARY_OR_DESIGN_CONTRACT', 'PREREGISTERED', 'EXECUTION_FROZEN_EXTENSION', 'EVALUATOR_ONLY_POST_HOC', 'EXPLORATORY', 'QC_OR_PROTOCOL_AUDIT', 'INTEGRATED_OPERATIONAL_CONCLUSION'}

def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    obj = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError(f'YAML mapping required: {path}')
    return obj

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def repo_rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()

def match_files(root: Path, pattern: str) -> list[Path]:
    if not pattern:
        return []
    if any((token in pattern for token in ('<exact_', '...'))):
        return []
    target = root / pattern
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted((path for path in target.rglob('*') if path.is_file()))
    return sorted((path for path in root.glob(pattern) if path.is_file()))

def find_one(root: Path, candidates: Iterable[str], label: str) -> Path:
    found: list[Path] = []
    for candidate in candidates:
        found.extend(match_files(root, candidate))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in found:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    if len(unique) != 1:
        raise RuntimeError(f'{label}: expected exactly one file, found {len(unique)}: {[str(path) for path in unique[:20]]}')
    return unique[0]

def generated_surface_scan(paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() not in {'.yaml', '.yml', '.json', '.csv', '.md', '.html', '.txt'}:
            continue
        text = path.read_text(encoding='utf-8-sig', errors='replace')
        for token in FORBIDDEN_GENERATED_TOKENS:
            if token in text:
                errors.append(f'{path}: forbidden token {token!r}')
    return errors

def parse_csv(path: Path):
    import pandas as pd
    return pd.read_csv(path, encoding='utf-8-sig')

def parse_table(path: Path):
    import pandas as pd
    suffix = path.suffix.lower()
    if suffix == '.parquet':
        return pd.read_parquet(path)
    if suffix in {'.csv', '.txt'}:
        return pd.read_csv(path, encoding='utf-8-sig')
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f'Unsupported table format: {path}')

def canonical_cell(value: Any) -> str:
    """Stable textual representation used by YAML↔workbook verification."""
    if value is None:
        return ''
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, float):
        if value != value:
            return ''
        return format(value, '.17g')
    return str(value)

def flatten_record(record: dict[str, Any]) -> dict[str, str]:
    return {str(key): canonical_cell(value) for key, value in record.items()}

def expand_claim_ids(values: Iterable[str]) -> list[str]:
    """Expand entries such as C12–C17 or C12-C17 without losing explicit IDs."""
    out: list[str] = []
    for raw in values:
        text = str(raw).strip()
        match = re.fullmatch('C(\\d{2})\\s*[–-]\\s*C(\\d{2})', text)
        if match:
            start, end = map(int, match.groups())
            if end < start:
                raise ValueError(f'descending Claim range is invalid: {text}')
            out.extend((f'C{number:02d}' for number in range(start, end + 1)))
        else:
            out.append(text)
    return list(dict.fromkeys(out))
AUXILIARY_PANEL_LABELS = {'BOUNDARY_SCOPE', 'S25 provenance', 'attribution summary', 'deployment audit axes'}

def normalize_function_id(value: Any) -> str | None:
    """Return the canonical v4 function ID, or None for panel-only labels."""
    text = re.sub('^[A-D]\\s+', '', str(value).strip())
    text = re.sub('^NEW:\\s*', '', text)
    if text.endswith(' data'):
        text = text[:-5].strip()
    if text in AUXILIARY_PANEL_LABELS:
        return None
    return text or None
