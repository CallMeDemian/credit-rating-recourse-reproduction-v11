from __future__ import annotations
import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
from .claim_evidence_common import expand_claim_ids, load_yaml, repo_rel, write_csv, write_json
MAX_ROWS_PER_SOURCE = 250
MAX_SVG_METRICS_PER_SOURCE = 8

def _flatten_json(value: Any, prefix: str='') -> list[tuple[str, Any]]:
    rows: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f'{prefix}.{key}' if prefix else str(key)
            rows.extend(_flatten_json(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_flatten_json(child, f'{prefix}[{index}]'))
    else:
        rows.append((prefix or 'value', value))
    return rows

def _source_long_rows(source_id: str, path: Path, root: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    rows: list[dict[str, Any]] = []
    if suffix == '.csv':
        frame = pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)
        if len(frame) > MAX_ROWS_PER_SOURCE:
            raise RuntimeError(f'{source_id}: canonical evidence has {len(frame)} rows; expected a inspection extract <= {MAX_ROWS_PER_SOURCE}')
        for row_index, record in enumerate(frame.to_dict('records')):
            for field, value in record.items():
                rows.append({'source_id': source_id, 'source_path': repo_rel(root, path), 'row_index': row_index, 'field': str(field), 'value': value})
    elif suffix in {'.json', '.yaml', '.yml'}:
        if suffix == '.json':
            payload = json.loads(path.read_text(encoding='utf-8-sig'))
        else:
            payload = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        for row_index, (field, value) in enumerate(_flatten_json(payload)):
            if row_index >= MAX_ROWS_PER_SOURCE * 20:
                raise RuntimeError(f'{source_id}: canonical structured evidence is too broad for a paper slot')
            rows.append({'source_id': source_id, 'source_path': repo_rel(root, path), 'row_index': row_index, 'field': field, 'value': value})
    else:
        rows.append({'source_id': source_id, 'source_path': repo_rel(root, path), 'row_index': 0, 'field': 'artifact_size_bytes', 'value': int(path.stat().st_size)})
    if not rows:
        raise RuntimeError(f'{source_id}: canonical evidence produced no displayable rows')
    return rows

def _numeric_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    numeric: list[dict[str, Any]] = []
    for row in rows:
        try:
            value = float(row['value'])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            numeric.append({**row, 'numeric_value': value})
    return numeric

def _svg(title: str, source_rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]]) -> str:
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in _numeric_metrics(source_rows):
        by_source.setdefault(str(row['source_id']), []).append(row)
    source_ids = list(dict.fromkeys((str(row['source_id']) for row in source_rows)))
    panel_height = 235
    width = 1500
    height = 150 + panel_height * max(1, len(source_ids)) + 70
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="45" y="55" font-family="Arial, sans-serif" font-size="31" font-weight="bold">{html.escape(title)}</text>', '<text x="45" y="88" font-family="Arial, sans-serif" font-size="16">Canonical evidence values; raw Oracle scales are not compared across backends.</text>']
    for panel_index, source_id in enumerate(source_ids):
        y0 = 115 + panel_index * panel_height
        all_source = [row for row in source_rows if str(row['source_id']) == source_id]
        metrics = by_source.get(source_id, [])[:MAX_SVG_METRICS_PER_SOURCE]
        parts.append(f'<rect x="35" y="{y0}" width="1430" height="210" rx="8" fill="#f7f7f7" stroke="#222"/>')
        parts.append(f'<text x="55" y="{y0 + 31}" font-family="Arial, sans-serif" font-size="21" font-weight="bold">{html.escape(source_id)}</text>')
        source_path = str(all_source[0]['source_path'])
        parts.append(f'<text x="150" y="{y0 + 31}" font-family="Arial, sans-serif" font-size="14">{html.escape(source_path)}</text>')
        if not metrics:
            text_values = all_source[:5]
            for i, row in enumerate(text_values):
                text = f"{row['field']} = {row['value']}"
                parts.append(f'<text x="65" y="{y0 + 68 + i * 27}" font-family="Arial, sans-serif" font-size="16">{html.escape(text[:170])}</text>')
            continue
        max_abs = max((abs(float(row['numeric_value'])) for row in metrics)) or 1.0
        for i, row in enumerate(metrics):
            value = float(row['numeric_value'])
            bar_width = 430 * abs(value) / max_abs
            bar_x = 760 if value >= 0 else 760 - bar_width
            fill = '#4c78a8' if value >= 0 else '#e45756'
            y = y0 + 62 + i * 18
            label = f"{row['field']} [{row['row_index']}]"
            parts.append(f'<text x="65" y="{y + 12}" font-family="Arial, sans-serif" font-size="13">{html.escape(label[:85])}</text>')
            parts.append(f'<line x1="760" y1="{y}" x2="760" y2="{y + 15}" stroke="#555"/>')
            parts.append(f'<rect x="{bar_x:.1f}" y="{y + 2}" width="{max(1.0, bar_width):.1f}" height="11" fill="{fill}"/>')
            parts.append(f'<text x="1210" y="{y + 12}" font-family="Arial, sans-serif" font-size="13">{value:.6g}</text>')
    verdict_text = '; '.join((f"{row['claim_id']}={row['runtime_verdict']}" for row in claim_rows))
    parts.append(f'<text x="45" y="{height - 30}" font-family="Arial, sans-serif" font-size="15">Claim verdicts: {html.escape(verdict_text)}</text>')
    parts.append('</svg>')
    return ''.join(parts)

def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / 'src/credit_recourse/configs'
    display = load_yaml(config_root / 'paper_display_registry.yaml')
    source_registry = load_yaml(config_root / 'source_registry.yaml')['sources']
    source_map = {str(source['source_id']): source for source in source_registry}
    ledger_path = root / 'data/reproduction/claim_evidence/claim_evidence_ledger.csv'
    if not ledger_path.is_file():
        raise FileNotFoundError(ledger_path)
    ledger = pd.read_csv(ledger_path, encoding='utf-8-sig', keep_default_na=False)
    if len(ledger) != 28 or ledger['claim_id'].duplicated().any():
        raise RuntimeError('paper display assets require exactly 28 unique ledger rows')
    ledger_map = {str(row['claim_id']): row.to_dict() for _, row in ledger.iterrows()}
    output_root = root / 'data/reproduction/paper_assets'
    tables_root = output_root / 'tables'
    figures_root = output_root / 'figures'
    slot_root = output_root / 'slot_manifests'
    for directory in (tables_root, figures_root, slot_root):
        directory.mkdir(parents=True, exist_ok=True)
    for stale in [*tables_root.glob('*'), *figures_root.glob('*'), *slot_root.glob('*.json')]:
        if stale.is_file():
            stale.unlink()
    slot_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for slot in display['slots']:
        slot_id = str(slot['slot_id'])
        claim_ids = expand_claim_ids(slot.get('claim_ids', []))
        missing_claims = [claim_id for claim_id in claim_ids if claim_id not in ledger_map]
        claim_rows = [ledger_map[claim_id] for claim_id in claim_ids if claim_id in ledger_map]
        unresolved_claims = [str(row['claim_id']) for row in claim_rows if str(row.get('runtime_verdict')) == 'UNRESOLVED']
        source_ids: list[str] = []
        for row in claim_rows:
            source_ids.extend((value for value in str(row.get('source_ids', '')).split(';') if value))
        source_ids = list(dict.fromkeys(source_ids))
        missing_sources: list[str] = []
        source_files: list[dict[str, Any]] = []
        source_data_rows: list[dict[str, Any]] = []
        for source_id in source_ids:
            source = source_map.get(source_id)
            if source is None:
                missing_sources.append(source_id)
                continue
            path = root / str(source['canonical_evidence_path'])
            if not path.is_file():
                missing_sources.append(source_id)
                continue
            source_files.append({'source_id': source_id, 'path': repo_rel(root, path), 'size_bytes': int(path.stat().st_size)})
            source_data_rows.extend(_source_long_rows(source_id, path, root))
        status = 'PASS' if not (missing_claims or unresolved_claims or missing_sources) and source_data_rows else 'UNRESOLVED'
        if status != 'PASS':
            errors.append(f'{slot_id}: missing_claims={missing_claims}, unresolved_claims={unresolved_claims}, missing_sources={missing_sources}, data_rows={len(source_data_rows)}')
        data_path = (figures_root if str(slot['slot_type']) == '본문 그림' else tables_root) / f'{slot_id}_data.csv'
        write_csv(data_path, source_data_rows)
        if str(slot['slot_type']) == '본문 그림':
            asset_path = figures_root / f'{slot_id}.svg'
            asset_path.write_text(_svg(str(slot['title_and_role']), source_data_rows, claim_rows), encoding='utf-8')
        else:
            asset_path = tables_root / f'{slot_id}.csv'
            write_csv(asset_path, source_data_rows)
        manifest = {'schema_version': 'paper_display_slot_v4_2', 'status': status, 'slot_id': slot_id, 'slot_type': slot['slot_type'], 'title_and_role': slot['title_and_role'], 'panel_or_function_ids': slot.get('panel_or_function_ids', []), 'claim_ids': claim_ids, 'claim_verdicts': {str(row['claim_id']): str(row['runtime_verdict']) for row in claim_rows}, 'source_ids': source_ids, 'source_files': source_files, 'source_data_row_count': len(source_data_rows), 'numeric_data_row_count': len(_numeric_metrics(source_data_rows)), 'asset_path': repo_rel(root, asset_path), 'data_path': repo_rel(root, data_path), 'paper_locations': slot.get('paper_locations', []), 'legacy_65_artifact_registry_status': 'HISTORICAL_ONLY', 'created_utc': datetime.now(timezone.utc).isoformat()}
        manifest_path = slot_root / f'{slot_id}.json'
        write_json(manifest_path, manifest)
        slot_rows.append({'slot_id': slot_id, 'slot_type': slot['slot_type'], 'status': status, 'claim_count': len(claim_ids), 'source_count': len(source_ids), 'source_data_row_count': len(source_data_rows), 'asset_path': manifest['asset_path'], 'data_path': manifest['data_path'], 'manifest_path': repo_rel(root, manifest_path)})
    main_slots = [row for row in slot_rows if row['slot_id'].startswith(('FIG-', 'TAB-'))]
    figure_count = sum((row['slot_id'].startswith('FIG-') for row in main_slots))
    table_count = sum((row['slot_id'].startswith('TAB-') for row in main_slots))
    if figure_count != 7 or table_count != 9 or len(main_slots) != 16:
        errors.append(f'main slot contract requires 7 figures + 9 tables = 16; got {figure_count}+{table_count}')
    write_csv(output_root / 'paper_display_assets.csv', slot_rows)
    result = {'schema_version': 'paper_display_assets_v4_2', 'status': 'PASS' if not errors else 'FAIL', 'slot_count': len(slot_rows), 'main_slot_count': len(main_slots), 'main_figure_count': figure_count, 'main_table_count': table_count, 'appendix_slot_count': len(slot_rows) - len(main_slots), 'resolved_slot_count': sum((row['status'] == 'PASS' for row in slot_rows)), 'legacy_registry': 'src/credit_recourse/configs/thesis_artifact_registry.json', 'legacy_registry_status': 'HISTORICAL_ONLY', 'errors': errors}
    write_json(output_root / 'paper_display_assets_manifest.json', result)
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
