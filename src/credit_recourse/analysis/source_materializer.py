from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .claim_evidence_common import load_yaml, match_files, parse_table, repo_rel, write_csv, write_json

def _resolve_column(df: pd.DataFrame, canonical: str, aliases: dict[str, list[str]]) -> str:
    candidates = [canonical, *aliases.get(canonical, [])]
    found = [candidate for candidate in candidates if candidate in df.columns]
    if len(found) != 1:
        raise RuntimeError(f'column {canonical!r}: expected exactly one of {candidates}, found {found}')
    return found[0]

def _filter_frame(df: pd.DataFrame, filters: dict[str, Any], aliases: dict[str, list[str]]) -> pd.DataFrame:
    result = df.copy()
    for canonical, expected in filters.items():
        column = _resolve_column(result, canonical, aliases)
        if isinstance(expected, dict):
            if 'not_in' in expected:
                result = result.loc[~result[column].astype(str).isin([str(v) for v in expected['not_in']])]
            elif 'regex' in expected:
                result = result.loc[result[column].astype(str).str.fullmatch(str(expected['regex']), na=False)]
            elif 'is_null' in expected:
                result = result.loc[result[column].isna() if expected['is_null'] else result[column].notna()]
            else:
                raise RuntimeError(f'unsupported filter for {canonical}: {expected}')
        elif isinstance(expected, list):
            result = result.loc[result[column].astype(str).isin([str(value) for value in expected])]
        elif expected is None:
            result = result.loc[result[column].isna()]
        else:
            result = result.loc[result[column].astype(str).eq(str(expected))]
    return result.copy()

def _validate_table(df: pd.DataFrame, spec: dict[str, Any], source_id: str) -> dict[str, Any]:
    expected = spec.get('expected_cardinality')
    if expected is not None:
        if isinstance(expected, int) and len(df) != expected:
            raise RuntimeError(f'{source_id}: expected {expected} rows after filters, got {len(df)}')
        if isinstance(expected, dict):
            minimum = expected.get('min')
            maximum = expected.get('max')
            if minimum is not None and len(df) < int(minimum):
                raise RuntimeError(f'{source_id}: expected at least {minimum} rows, got {len(df)}')
            if maximum is not None and len(df) > int(maximum):
                raise RuntimeError(f'{source_id}: expected at most {maximum} rows, got {len(df)}')
    unique_key = list(spec.get('unique_key') or [])
    if unique_key:
        missing = [column for column in unique_key if column not in df.columns]
        if missing:
            raise RuntimeError(f'{source_id}: unique key columns missing: {missing}')
        duplicate_count = int(df.duplicated(unique_key, keep=False).sum())
        if duplicate_count:
            sample = df.loc[df.duplicated(unique_key, keep=False), unique_key].head(10).to_dict('records')
            raise RuntimeError(f'{source_id}: duplicate key rows={duplicate_count}; sample={sample}')
    for column in spec.get('finite_numeric_columns') or []:
        if column not in df.columns:
            raise RuntimeError(f'{source_id}: finite numeric column missing: {column}')
        values = pd.to_numeric(df[column], errors='coerce')
        if not np.isfinite(values).all():
            raise RuntimeError(f'{source_id}: non-finite values in {column}')
    for column in spec.get('pvalue_columns') or []:
        if column not in df.columns:
            raise RuntimeError(f'{source_id}: p-value column missing: {column}')
        values = pd.to_numeric(df[column], errors='coerce')
        if values.isna().any() or not values.between(0.0, 1.0, inclusive='both').all():
            raise RuntimeError(f'{source_id}: invalid p-values in {column}')
    for triplet in spec.get('ci_triplets') or []:
        estimate, lower, upper = triplet
        missing = [column for column in triplet if column not in df.columns]
        if missing:
            raise RuntimeError(f'{source_id}: CI columns missing: {missing}')
        e = pd.to_numeric(df[estimate], errors='coerce')
        lo = pd.to_numeric(df[lower], errors='coerce')
        hi = pd.to_numeric(df[upper], errors='coerce')
        if (e.isna() | lo.isna() | hi.isna() | (lo > e) | (e > hi)).any():
            raise RuntimeError(f'{source_id}: invalid CI order for {triplet}')
    return {'row_count': int(len(df)), 'column_count': int(len(df.columns)), 'columns': [str(column) for column in df.columns], 'unique_key': unique_key}

def _extract(root: Path, source: dict[str, Any], upstream: Path, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(source['source_id'])
    spec = source.get('extract')
    if not isinstance(spec, dict):
        raise RuntimeError(f'{source_id}: EXTRACT requires a machine-readable extract contract')
    aliases = {str(key): [str(value) for value in values] for key, values in (spec.get('column_aliases') or {}).items()}
    frame = parse_table(upstream)
    original_rows = len(frame)
    frame = _filter_frame(frame, spec.get('filters') or {}, aliases)
    columns = list(spec.get('columns') or [])
    if not columns:
        raise RuntimeError(f'{source_id}: EXTRACT requires a non-empty columns list')
    selected: dict[str, Any] = {}
    for canonical in columns:
        actual = _resolve_column(frame, str(canonical), aliases)
        selected[str(canonical)] = frame[actual]
    out_frame = pd.DataFrame(selected, index=frame.index).reset_index(drop=True)
    for key, value in (spec.get('constant_columns') or {}).items():
        out_frame[str(key)] = value
    required = [str(column) for column in spec.get('required_columns') or columns]
    missing = [column for column in required if column not in out_frame.columns]
    if missing:
        raise RuntimeError(f'{source_id}: required extracted columns missing: {missing}')
    validation = _validate_table(out_frame, spec, source_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_frame.to_csv(output, index=False, encoding='utf-8-sig')
    lineage = [{'path': repo_rel(root, upstream), 'size_bytes': int(upstream.stat().st_size), 'original_row_count': int(original_rows), 'filtered_row_count': int(len(out_frame))}]
    return (lineage, validation)

def _semantic_manifest(root: Path, source: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(source['source_id'])
    semantic_path = source.get('semantic_manifest_path')
    if semantic_path:
        files = match_files(root, str(semantic_path))
        if len(files) != 1:
            raise RuntimeError(f'{source_id}: semantic_manifest_path must resolve exactly one file; found {len(files)}')
        manifest = files[0]
        payload = json.loads(manifest.read_text(encoding='utf-8-sig'))
        if not isinstance(payload, dict) or payload.get('status') != 'PASS':
            raise RuntimeError(f'{source_id}: semantic manifest must be a JSON mapping with status=PASS')
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(manifest, output)
        return ([{'path': repo_rel(root, manifest), 'size_bytes': int(manifest.stat().st_size), 'semantic_manifest': True}], {'manifest_status': 'PASS', 'manifest_type': payload.get('schema_version')})
    files = match_files(root, str(source.get('upstream_path') or ''))
    if not files:
        raise RuntimeError(f'{source_id}: no files found for manifest inventory')
    rows = [{'path': repo_rel(root, path), 'size_bytes': int(path.stat().st_size)} for path in files]
    inventory = {'status': 'PASS', 'manifest_type': 'FILE_INVENTORY_ONLY', 'source_id': source_id, 'file_count': len(rows), 'files': rows}
    write_json(output, inventory)
    return (rows, {'manifest_status': 'PASS', 'manifest_type': 'FILE_INVENTORY_ONLY'})

def _transform_tabular_copy(root: Path, source: dict[str, Any], upstream: Path, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Copy a small table while applying an explicit canonical transform.

    COPY_SMALL remains a literal byte copy unless ``canonical_transform`` is
    present.  When present, it is processed with the same filtering, column
    selection, constants, and validation contract as EXTRACT.
    """
    transform = source.get('canonical_transform')
    if transform is None:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(upstream, output)
        return ([{'path': repo_rel(root, upstream), 'size_bytes': int(upstream.stat().st_size)}], {})
    if not isinstance(transform, dict):
        raise RuntimeError(f"{source['source_id']}: canonical_transform must be a mapping")
    shadow = dict(source)
    shadow['extract'] = transform
    return _extract(root, shadow, upstream, output)

def _materialize(root: Path, source: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_id = str(source['source_id'])
    policy = str(source.get('materialization_policy'))
    files = match_files(root, str(source.get('upstream_path') or ''))
    if policy == 'EXTRACT':
        if len(files) != 1:
            raise RuntimeError(f'{source_id}: EXTRACT expects exactly one upstream file, found {len(files)}')
        return _extract(root, source, files[0], output)
    if policy in {'COPY_SMALL', 'GENERATE'}:
        if len(files) != 1:
            raise RuntimeError(f'{source_id}: {policy} expects exactly one upstream file, found {len(files)}')
        return _transform_tabular_copy(root, source, files[0], output)
    if policy == 'REFERENCE':
        if len(files) != 1:
            raise RuntimeError(f'{source_id}: REFERENCE expects exactly one upstream file, found {len(files)}')
        upstream = files[0]
        write_json(output, {'status': 'PASS', 'referenced_path': repo_rel(root, upstream), 'size_bytes': int(upstream.stat().st_size)})
        return ([{'path': repo_rel(root, upstream), 'size_bytes': int(upstream.stat().st_size)}], {})
    if policy == 'GENERATE_MANIFEST':
        return _semantic_manifest(root, source, output)
    raise RuntimeError(f'{source_id}: unsupported materialization policy {policy!r}')

def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / 'src/credit_recourse/configs'
    registry = load_yaml(config_root / 'source_registry.yaml')
    resolved: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for source in registry['sources']:
        source_id = str(source['source_id'])
        output = root / str(source['canonical_evidence_path'])
        validation: dict[str, Any] = {}
        try:
            lineage, validation = _materialize(root, source, output)
            if not output.is_file():
                raise RuntimeError('materializer returned without creating canonical evidence')
            status = 'MATERIALIZED'
        except Exception as exc:
            status = 'UNRESOLVED'
            lineage = []
            unresolved.append({'source_id': source_id, 'upstream_path': source.get('upstream_path'), 'policy': source.get('materialization_policy'), 'error': f'{type(exc).__name__}: {exc}'})
            if output.exists():
                output.unlink()
        sidecar = {'schema_version': 'canonical_source_sidecar_v4_1', 'source_id': source_id, 'status': status, 'canonical_evidence_path': str(source['canonical_evidence_path']), 'upstream_sources': lineage, 'materialization_policy': source.get('materialization_policy'), 'extract_contract': source.get('extract'), 'semantic_manifest_path': source.get('semantic_manifest_path'), 'filter_and_key_contract': source.get('filter_and_key_contract'), 'required_columns_or_metrics': source.get('required_columns_or_metrics'), 'validation': validation, 'producer_module': source.get('producer'), 'verifier_module': source.get('verifier'), 'created_utc': datetime.now(timezone.utc).isoformat()}
        write_json(root / str(source['sidecar_path']), sidecar)
        resolved.append({'source_id': source_id, 'status': status, 'canonical_evidence_path': source['canonical_evidence_path'], 'sidecar_path': source['sidecar_path'], 'row_count': validation.get('row_count')})
    output_root = root / 'data/reproduction/claim_evidence'
    write_csv(output_root / 'source_registry_resolved.csv', resolved)
    write_csv(output_root / 'unresolved_sources.csv', unresolved)
    result = {'schema_version': 'source_materialization_v4_1', 'status': 'PASS' if not unresolved else 'PARTIAL', 'source_count': len(resolved), 'materialized_count': sum((row['status'] == 'MATERIALIZED' for row in resolved)), 'unresolved_count': len(unresolved), 'unresolved_sources': unresolved}
    write_json(output_root / 'source_materialization_report.json', result)
    return result

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
