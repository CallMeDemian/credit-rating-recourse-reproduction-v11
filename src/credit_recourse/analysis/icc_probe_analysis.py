from __future__ import annotations
'Post-freeze IC-c prior-knowledge probe analysis.\n\nThis module is the canonical home for the IC-c identity/numeric-recall probe.\nIt validates and normalizes a completed probe-only LLM artifact into the\nsingle paper-analysis tree.  Live/API probe execution belongs to the Stage7\nLLM runner (``icc_probe_runner``), not to post-freeze analysis.\n\nThe module never calls an LLM API and never mutates frozen Stage7/8/9 artifacts.\nIts canonical output is under the paper reproduction analysis directory.\n'
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from credit_recourse.utils.writable_outputs import atomic_copy_generated_file
REQUIRED_SOURCE_FILES = ('icc_probe_summary.json', 'icc_probe_firm_level.csv', 'llm_stage7_icc_probe_checkpoint.jsonl')

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Required JSON file not found: {path}')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(data, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return data

def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def _wilson_interval(k: int, n: int, z: float=1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return (None, None)
    p = k / n
    den = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / den
    half = z / den * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))

def _bool_from_value(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes'}:
        return True
    if text in {'false', '0', 'no'}:
        return False
    return None

def _parse_raw_response(raw: Any) -> tuple[bool | None, int | None, int, bool]:
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        return (None, None, 0, False)
    try:
        obj = json.loads(str(raw))
    except Exception:
        return (None, None, 0, False)
    if not isinstance(obj, dict):
        return (None, None, 0, False)
    recognized = obj.get('recognized') if isinstance(obj.get('recognized'), bool) else None
    familiarity = obj.get('familiarity')
    if isinstance(familiarity, bool) or not isinstance(familiarity, int) or (not 0 <= familiarity <= 3):
        familiarity = None
    facts = obj.get('known_facts')
    fact_count = len(facts) if isinstance(facts, list) else 0
    recalled = obj.get('recalled_debt_ratio')
    numeric_recall = isinstance(recalled, (int, float)) and (not isinstance(recalled, bool)) and math.isfinite(float(recalled))
    return (recognized, familiarity, fact_count, numeric_recall)

def _inspect_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f'Probe checkpoint not found: {path}')
    row_ids: list[int] = []
    schema_versions: set[str] = set()
    backend_ids: set[str] = set()
    with path.open('r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError as exc:
                raise ValueError(f'Corrupt probe checkpoint at {path}:{line_no}: {exc}') from exc
            if not isinstance(rec, dict):
                raise ValueError(f'Checkpoint record is not an object at {path}:{line_no}')
            if 'row_id' not in rec:
                raise ValueError(f'Checkpoint record lacks row_id at {path}:{line_no}')
            row_ids.append(int(rec['row_id']))
            if rec.get('probe_schema_version'):
                schema_versions.add(str(rec['probe_schema_version']))
            if rec.get('backend_id'):
                backend_ids.add(str(rec['backend_id']))
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f'Probe checkpoint has duplicate row_id values: {path}')
    return {'path': str(path), 'record_count': len(row_ids), 'unique_row_ids': len(set(row_ids)), 'row_ids': sorted(row_ids), 'schema_versions': sorted(schema_versions), 'backend_ids': sorted(backend_ids)}

def _enrich_and_validate(*, summary: dict[str, Any], firm_csv: Path, checkpoint: Path, out_dir: Path, expected_full_rows: int | None) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    df = pd.read_csv(firm_csv)
    required_cols = {'row_id', 'icc_probe_response_raw', 'icc_probe_value', 'icc_contamination_flag', 'icc_probe_parse_error', 'icc_probe_rel_err', 'icc_probe_panel_value'}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f'Probe firm-level CSV lacks required columns: {missing}')
    df['row_id'] = pd.to_numeric(df['row_id'], errors='raise').astype(int)
    if df['row_id'].duplicated().any():
        dup = sorted(df.loc[df['row_id'].duplicated(), 'row_id'].unique().tolist())[:20]
        raise ValueError(f'Probe firm-level CSV has duplicate row_id values: {dup}')
    parsed = df['icc_probe_response_raw'].map(_parse_raw_response)
    df['icc_probe_recognized'] = [x[0] for x in parsed]
    df['icc_probe_familiarity'] = [x[1] for x in parsed]
    df['icc_probe_known_facts_count'] = [x[2] for x in parsed]
    df['icc_probe_numeric_recall'] = [x[3] for x in parsed]
    df['icc_probe_parse_failure'] = df['icc_probe_parse_error'].notna()
    df['icc_probe_contamination_flag'] = df['icc_contamination_flag'].map(_bool_from_value)
    checkpoint_meta = _inspect_checkpoint(checkpoint)
    csv_ids = sorted(df['row_id'].tolist())
    if csv_ids != checkpoint_meta['row_ids']:
        raise ValueError('Probe CSV row_id set does not match checkpoint row_id set.')
    n = int(len(df))
    if expected_full_rows is not None and n != int(expected_full_rows):
        raise ValueError(f'Expected {expected_full_rows} probe rows, found {n}')
    if int(summary.get('probe_row_count', n)) != n:
        raise ValueError(f"Summary probe_row_count={summary.get('probe_row_count')} does not match CSV rows={n}")
    recognized_count = int(df['icc_probe_recognized'].eq(True).sum())
    numeric_recall_count = int(df['icc_probe_numeric_recall'].sum())
    contamination_count = int(df['icc_probe_contamination_flag'].eq(True).sum())
    parse_failure_count = int(df['icc_probe_parse_failure'].sum())
    for key, observed in [('probe_recognized_count', recognized_count), ('numeric_recall_count', numeric_recall_count), ('probe_contaminated_count', contamination_count), ('probe_parse_failures', parse_failure_count)]:
        recorded = summary.get(key)
        if recorded is not None and int(recorded) != observed:
            raise ValueError(f'Summary {key}={recorded} does not match recomputed value={observed}')
    channel_rows: list[dict[str, Any]] = []
    for channel, count in [('firm_recognition', recognized_count), ('numeric_debt_ratio_recall', numeric_recall_count), ('numeric_contamination_flag', contamination_count), ('parse_failure', parse_failure_count)]:
        lo, hi = _wilson_interval(count, n)
        channel_rows.append({'channel': channel, 'count': count, 'n': n, 'rate': count / n if n else None, 'wilson_95_lo': lo, 'wilson_95_hi': hi})
    channel = pd.DataFrame(channel_rows)
    familiarity = df['icc_probe_familiarity'].value_counts(dropna=False).rename_axis('familiarity').reset_index(name='count')
    familiarity['n'] = n
    familiarity['rate'] = familiarity['count'] / n if n else None
    familiarity['familiarity'] = familiarity['familiarity'].map(lambda x: 'missing' if pd.isna(x) else str(int(x)))
    familiarity = familiarity.sort_values('familiarity').reset_index(drop=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    enriched_path = out_dir / 'icc_probe_firm_level_enriched.csv'
    channel_path = out_dir / 'icc_probe_channel_summary.csv'
    familiarity_path = out_dir / 'icc_probe_familiarity_distribution.csv'
    df.to_csv(enriched_path, index=False, encoding='utf-8-sig')
    channel.to_csv(channel_path, index=False, encoding='utf-8-sig')
    familiarity.to_csv(familiarity_path, index=False, encoding='utf-8-sig')
    recomputed = {'probe_row_count': n, 'probe_recognized_count': recognized_count, 'probe_recognized_rate': recognized_count / n if n else None, 'numeric_recall_count': numeric_recall_count, 'numeric_recall_rate': numeric_recall_count / n if n else None, 'probe_contaminated_count': contamination_count, 'probe_contamination_rate': contamination_count / n if n else None, 'probe_parse_failures': parse_failure_count, 'probe_parse_failure_rate': parse_failure_count / n if n else None, 'checkpoint': {k: v for k, v in checkpoint_meta.items() if k != 'row_ids'}, 'derived_outputs': {'firm_level_enriched': str(enriched_path), 'channel_summary': str(channel_path), 'familiarity_distribution': str(familiarity_path)}}
    return (df, recomputed, checkpoint_meta)

def _copy_source_files(source_dir: Path, out_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_SOURCE_FILES:
        src = source_dir / name
        if not src.exists():
            raise FileNotFoundError(f'Completed probe source is missing {name}: {source_dir}')
        dst = out_dir / name
        result = atomic_copy_generated_file(src, dst, skip_if_identical=True)
        copied[name] = str(dst)
    return copied

def _completed_probe_candidate(*, source_dir: Path, source_kind: str, archive_run_dir: Path | None) -> dict[str, Any] | None:
    source_dir = Path(source_dir).resolve()
    required = {name: source_dir / name for name in REQUIRED_SOURCE_FILES}
    if not all((path.exists() for path in required.values())):
        return None
    summary = _read_json(required['icc_probe_summary.json'])
    if summary.get('status') != 'PASS' or int(summary.get('probe_row_count', 0)) != 575:
        return None
    content_inventory = tuple((name, int(required[name].stat().st_size)) for name in REQUIRED_SOURCE_FILES)
    return {'source_dir': source_dir, 'source_kind': source_kind, 'archive_run_dir': archive_run_dir.resolve() if archive_run_dir else None, 'summary': summary, 'content_inventory': content_inventory}

def _iter_legacy_probe_source_dirs(legacy_root: Path) -> list[Path]:
    """Return completed-probe candidate directories below one supported legacy root.

    Historical runs were moved with PowerShell ``Move-Item`` under slightly
    different directory shapes.  Search is deliberately restricted to the
    supported probe roots, but recursive within those roots so both
    ``<root>/<run_label>/...`` and ``<root>/icc_probe_runs/<run_label>/...``
    layouts are resolved.  A directory is returned only when all three
    required source files are co-located.
    """
    legacy_root = Path(legacy_root).resolve()
    if not legacy_root.is_dir():
        return []
    candidates: set[Path] = set()
    for summary_path in legacy_root.rglob('icc_probe_summary.json'):
        source_dir = summary_path.parent.resolve()
        if all(((source_dir / name).is_file() for name in REQUIRED_SOURCE_FILES)):
            candidates.add(source_dir)
    return sorted(candidates, key=str)

def resolve_completed_probe_source(project_root: Path) -> dict[str, Any] | None:
    """Resolve exactly one completed 575-row paper IC-c probe artifact.

    Fresh runs are selected from the canonical immutable LLM archive root.  The
    two historical probe-only locations are read-only compatibility sources:
    ``data/final_freeze/icc_probe_runs`` and timestamped
    ``data/archive/icc_probe_runs_legacy_*`` directories.

    Multiple byte-identical copies are treated as aliases of one artifact.
    Multiple distinct completed probes hard-fail instead of silently selecting
    the newest directory.
    """
    from credit_recourse.contracts.paper_reproduction import discover_archived_runs, load_profile
    root = Path(project_root).resolve()
    profile = load_profile(root)
    expected_role = profile['llm']['icc_probe']['run_role']
    candidates: list[dict[str, Any]] = []
    seen_dirs: set[Path] = set()
    for record in discover_archived_runs(root, profile):
        if record.run_role != expected_role or not record.has_probe:
            continue
        probe_dir = (record.run_dir / 'stage7_icc_probe').resolve()
        candidate = _completed_probe_candidate(source_dir=probe_dir, source_kind='canonical_llm_archive', archive_run_dir=record.run_dir)
        if candidate is not None and probe_dir not in seen_dirs:
            candidates.append(candidate)
            seen_dirs.add(probe_dir)
    legacy_roots: list[tuple[str, Path]] = [('legacy_final_freeze_probe_root', root / 'data' / 'final_freeze' / 'icc_probe_runs')]
    archive_root = root / 'data' / 'archive'
    if archive_root.exists():
        legacy_roots.extend((('legacy_archived_probe_root', path) for path in sorted(archive_root.glob('icc_probe_runs_legacy_*')) if path.is_dir()))
    for source_kind, legacy_root in legacy_roots:
        for resolved_dir in _iter_legacy_probe_source_dirs(legacy_root):
            if resolved_dir in seen_dirs:
                continue
            candidate = _completed_probe_candidate(source_dir=resolved_dir, source_kind=source_kind, archive_run_dir=None)
            if candidate is not None:
                candidates.append(candidate)
                seen_dirs.add(resolved_dir)
    if not candidates:
        return None
    by_content: dict[tuple[tuple[str, int], ...], list[dict[str, Any]]] = {}
    for candidate in candidates:
        by_content.setdefault(candidate['content_inventory'], []).append(candidate)
    if len(by_content) != 1:
        diagnostic = [{'source_dir': str(candidate['source_dir']), 'source_kind': candidate['source_kind']} for candidate in candidates]
        raise RuntimeError(f'Multiple distinct completed IC-c probe artifacts were found; selection must be exact: {diagnostic}')
    aliases = next(iter(by_content.values()))
    priority = {'canonical_llm_archive': 0, 'legacy_final_freeze_probe_root': 1, 'legacy_archived_probe_root': 2}
    aliases = sorted(aliases, key=lambda item: (priority.get(str(item['source_kind']), 99), str(item['source_dir'])))
    selected = dict(aliases[0])
    selected['source_aliases'] = [{'source_dir': str(alias['source_dir']), 'source_kind': alias['source_kind']} for alias in aliases]
    selected.pop('content_inventory', None)
    return selected

def discover_completed_probe_source(project_root: Path) -> Path | None:
    """Compatibility path-only wrapper around :func:`resolve_completed_probe_source`."""
    selection = resolve_completed_probe_source(project_root)
    return Path(selection['source_dir']) if selection is not None else None

def import_completed_probe(*, source_dir: Path, out_dir: Path, expected_full_rows: int | None=575) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    out_dir = out_dir.resolve()
    copied = _copy_source_files(source_dir, out_dir)
    summary_path = out_dir / 'icc_probe_summary.json'
    firm_path = out_dir / 'icc_probe_firm_level.csv'
    checkpoint_path = out_dir / 'llm_stage7_icc_probe_checkpoint.jsonl'
    summary = _read_json(summary_path)
    _, recomputed, _ = _enrich_and_validate(summary=summary, firm_csv=firm_path, checkpoint=checkpoint_path, out_dir=out_dir, expected_full_rows=expected_full_rows)
    manifest = {'schema_version': 'icc_probe_analysis_manifest_v1', 'created_utc': _now(), 'status': 'PASS', 'mode': 'import', 'source_dir': str(source_dir), 'output_dir': str(out_dir), 'source_files': {name: {'path': str(source_dir / name)} for name in REQUIRED_SOURCE_FILES}, 'copied_files': copied, 'recomputed_contract': recomputed, 'main_stage7_api_calls': 0, 'interpretation_boundary': summary.get('interpretation_boundary')}
    _write_json(out_dir / 'icc_probe_analysis_manifest.json', manifest)
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Import a completed Stage7 IC-c probe into paper analysis.')
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--mode', choices=['auto', 'import'], default='auto')
    ap.add_argument('--source-dir', default=None, help='Completed stage7_icc_probe directory for import mode.')
    return ap

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir = out_dir.resolve()
    source_dir = Path(args.source_dir).resolve() if args.source_dir else None
    if args.mode == 'auto' and source_dir is None:
        source_dir = discover_completed_probe_source(root)
    if source_dir is None:
        raise FileNotFoundError('No completed 575-row IC-c probe archive was found. Run the LLM paper profile first; post-freeze analysis never starts paid probe calls.')
    manifest = import_completed_probe(source_dir=source_dir, out_dir=out_dir, expected_full_rows=575)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
