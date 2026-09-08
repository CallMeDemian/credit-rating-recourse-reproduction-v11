from __future__ import annotations
'Resolve the v4 16-slot paper display contract.\n\nThe previous 55-table + 10-figure registry remains in the repository only as a\nhistorical map. It is not a completion gate for the rewritten thesis.\n'
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from .claim_evidence_common import expand_claim_ids, load_yaml, repo_rel, write_json

def build(project_root: Path, analysis_dir: Path | None=None) -> dict[str, Any]:
    root = project_root.resolve()
    analysis = (analysis_dir or root / 'data/analysis/paper_repro').resolve()
    registry_path = root / 'src/credit_recourse/configs/paper_display_registry.yaml'
    registry = load_yaml(registry_path)
    paper_root = root / 'data/reproduction/paper_assets'
    output = analysis / '06_thesis_registry'
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for slot in registry['slots']:
        slot_id = str(slot['slot_id'])
        manifest_path = paper_root / 'slot_manifests' / f'{slot_id}.json'
        if not manifest_path.is_file():
            errors.append(f'{slot_id}: slot manifest missing')
            rows.append({'slot_id': slot_id, 'status': 'UNRESOLVED'})
            continue
        payload = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        asset_path = root / str(payload.get('asset_path'))
        status = 'RESOLVED' if payload.get('status') == 'PASS' and asset_path.is_file() else 'UNRESOLVED'
        if status != 'RESOLVED':
            errors.append(f"{slot_id}: status={payload.get('status')}, asset_exists={asset_path.is_file()}")
        rows.append({'slot_id': slot_id, 'slot_type': slot['slot_type'], 'title_and_role': slot['title_and_role'], 'claim_ids': ';'.join(expand_claim_ids(slot.get('claim_ids', []))), 'source_ids': ';'.join(payload.get('source_ids') or []), 'status': status, 'asset_path': payload.get('asset_path'), 'manifest_path': repo_rel(root, manifest_path), 'legacy_registry_status': 'HISTORICAL_ONLY'})
    frame = pd.DataFrame(rows)
    frame.to_csv(output / 'thesis_artifact_resolution.csv', index=False, encoding='utf-8-sig')
    main = frame[frame['slot_id'].astype(str).str.startswith(('FIG-', 'TAB-'))]
    result = {'schema_version': 'thesis_artifact_resolution_v4_1', 'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS' if not errors else 'FAIL', 'registry_path': repo_rel(root, registry_path), 'slot_count': len(frame), 'main_slot_count': len(main), 'main_figure_count': int(main['slot_id'].str.startswith('FIG-').sum()), 'main_table_count': int(main['slot_id'].str.startswith('TAB-').sum()), 'resolved_slot_count': int((frame['status'] == 'RESOLVED').sum()), 'legacy_65_artifact_registry': 'src/credit_recourse/configs/thesis_artifact_registry.json', 'legacy_65_artifact_registry_status': 'HISTORICAL_ONLY', 'errors': errors}
    write_json(output / 'artifact_resolution.json', result)
    return result

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--analysis-dir', default=None)
    args = parser.parse_args(argv)
    root = Path(args.project_root)
    analysis = Path(args.analysis_dir) if args.analysis_dir else None
    result = build(root, analysis)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'PASS' else 1
if __name__ == '__main__':
    raise SystemExit(main())
