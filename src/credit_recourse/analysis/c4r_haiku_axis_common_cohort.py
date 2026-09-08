"""Build one fixed complete-case row_id cohort for Haiku thinking/non-thinking axis analysis."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
SCHEMA_VERSION = 'c4r_haiku_axis_common_cohort_v1'
DEFAULT_POLICIES = ('C4', 'C4R', 'C6')
DEFAULT_MODE = 'free_form_10d'

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _parse_run(spec: str) -> tuple[str, Path]:
    label, sep, raw = str(spec).partition('=')
    if sep != '=' or not label.strip() or (not raw.strip()):
        raise ValueError(f'--run must use protocol=archive_dir; got {spec!r}')
    return (label.strip(), Path(raw.strip()).resolve())

def _resolve_inner(path: Path) -> Path:
    if (path / 'stage7_llm_action_generation').is_dir():
        return path
    candidates = [p for p in path.iterdir() if (p / 'stage7_llm_action_generation').is_dir()]
    if len(candidates) != 1:
        raise FileNotFoundError(f'stage7 directory not uniquely found under {path}')
    return candidates[0]

def run(*, runs: list[str], out: Path, expected_n: int, policies: tuple[str, ...], mode: str) -> dict:
    if expected_n <= 0:
        raise ValueError(f'expected_n must be positive; got {expected_n}')
    if len(policies) < 2 or len(set(policies)) != len(policies):
        raise ValueError(f'policies must contain distinct names; got {policies}')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    parsed = [_parse_run(spec) for spec in runs]
    if len({label for label, _ in parsed}) != len(parsed):
        raise ValueError('duplicate protocol labels')
    all_sets: list[set[int]] = []
    protocol_records: list[dict] = []
    universe: set[int] = set()
    for protocol, archive in parsed:
        if not archive.is_dir():
            raise FileNotFoundError(f'archive directory missing: {archive}')
        inner = _resolve_inner(archive)
        stage7 = inner / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet'
        if not stage7.is_file():
            raise FileNotFoundError(stage7)
        frame = pd.read_parquet(stage7)
        required = {'row_id', 'policy'}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f'{protocol}: Stage7 missing columns {missing}')
        if 'mode' in frame.columns:
            frame = frame[frame['mode'].astype(str).eq(str(mode))].copy()
        frame['row_id'] = pd.to_numeric(frame['row_id'], errors='raise').astype(int)
        key = ['row_id', 'policy'] + (['mode'] if 'mode' in frame.columns else [])
        if frame.duplicated(key).any():
            raise ValueError(f'{protocol}: duplicate Stage7 keys: {key}')
        by_policy: dict[str, set[int]] = {}
        for policy in policies:
            ids = set(frame.loc[frame['policy'].astype(str).eq(policy), 'row_id'].tolist())
            by_policy[policy] = ids
            all_sets.append(ids)
            universe |= ids
        protocol_records.append({'protocol': protocol, 'archive': str(inner), 'stage7_path': str(stage7), 'available_rows_by_policy': {p: len(by_policy[p]) for p in policies}, 'row_ids_by_policy': by_policy})
    common = set.intersection(*all_sets) if all_sets else set()
    common_sorted = sorted(common)
    if len(common_sorted) != expected_n:
        details = {rec['protocol']: rec['available_rows_by_policy'] for rec in protocol_records}
        raise ValueError(f'common complete-case cohort mismatch: expected={expected_n}, observed={len(common_sorted)}, available={details}')
    cohort_path = out / 'haiku_common_complete_case_row_ids.csv'
    pd.DataFrame({'row_id': common_sorted}).to_csv(cohort_path, index=False, encoding='utf-8-sig')
    records = []
    for rec in protocol_records:
        row_sets = rec.pop('row_ids_by_policy')
        rec['unavailable_row_ids_by_policy'] = {p: sorted(universe - row_sets[p]) for p in policies}
        rec['available_but_excluded_from_common_by_policy'] = {p: sorted(row_sets[p] - common) for p in policies}
        records.append(rec)
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'PASS', 'created_utc': _now(), 'mode': mode, 'policies': list(policies), 'protocol_count': len(records), 'expected_common_n': expected_n, 'common_n': len(common_sorted), 'common_row_id_min': min(common_sorted), 'common_row_id_max': max(common_sorted), 'common_row_ids_path': str(cohort_path), 'protocols': records, 'interpretation_boundary': 'The fixed cohort is the intersection of successfully routed Stage7 free-form rows across C4, C4R, and C6 for both Haiku protocols. It is used only for directly comparable evaluator-only mechanism attribution; it does not convert failed rows into successes or alter raw-budget QC.'}
    manifest_path = out / 'haiku_common_complete_case_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

def main(argv: list[str] | None=None) -> int:
    ap = argparse.ArgumentParser(description='Build common complete-case Haiku axis cohort')
    ap.add_argument('--run', action='append', required=True, help='protocol=archive_dir')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--expected-n', type=int, default=571)
    ap.add_argument('--policies', default=','.join(DEFAULT_POLICIES))
    ap.add_argument('--mode', default=DEFAULT_MODE)
    args = ap.parse_args(argv)
    policies = tuple((x.strip() for x in args.policies.split(',') if x.strip()))
    meta = run(runs=args.run, out=args.out, expected_n=args.expected_n, policies=policies, mode=args.mode)
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
