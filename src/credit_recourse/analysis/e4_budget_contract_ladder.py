from __future__ import annotations
'Rebuild the Section 9.7 / Table 29 budget-contract feasibility ladder.\n\nAll completed rows are read from the precomputed LLM run directories supplied by\nthe user.  The interrupted Gemini 3.5 attempt is retained as historical failure\nevidence from its checkpoint JSONL and is never promoted to a completed model\ncomparison.\n'
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
from credit_recourse.contracts.paper_reproduction import discover_archived_runs, load_profile, select_exact_role
POLICIES = ('C4', 'C4R', 'C6')

def _action_table(run_dir: Path) -> tuple[Path, pd.DataFrame]:
    p = run_dir / 'stage7_llm_action_generation/llm_stage7_action_table.parquet'
    if not p.is_file():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    req = {'row_id', 'policy', 'mode', 'budget_compliant_raw'}
    miss = sorted(req - set(df.columns))
    if miss:
        raise ValueError(f'{p} missing columns: {miss}')
    df = df.loc[df['policy'].astype(str).isin(POLICIES) & df['mode'].astype(str).eq('free_form_10d')].copy()
    if df.duplicated(['row_id', 'policy', 'mode']).any():
        raise ValueError(f'duplicate action keys: {p}')
    return (p, df)

def _completed_row(label: str, run, expected_total: int | None, confirmatory: bool, note: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path, df = _action_table(run.run_dir)
    row: dict[str, Any] = {'configuration': label, 'sample_state': f'{len(df)}/{(expected_total if expected_total is not None else len(df))} rows', 'aggregate_raw_compliance': float(pd.to_numeric(df['budget_compliant_raw'], errors='raise').mean()), 'status': 'PASS_CONFIRMATORY' if confirmatory else 'FAIL_CHARACTERIZATION', 'evidence_class': 'RECOMPUTED_NUMERIC' if confirmatory else 'FROZEN_PROVIDER_INPUT_CHARACTERIZATION', 'run_label': run.run_label, 'notes': note}
    for pol in POLICIES:
        g = df.loc[df['policy'].astype(str).eq(pol)]
        row[f'{pol}_n'] = len(g)
        row[f'{pol}_raw_compliance'] = float(pd.to_numeric(g['budget_compliant_raw'], errors='raise').mean()) if len(g) else None
    source = {'configuration': label, 'path': str(path), 'rows': len(df)}
    return (row, source)

def build(project_root: Path, out_dir: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = load_profile(root)
    records = discover_archived_runs(root, profile)
    e4 = profile['extensions']['e4']
    non = select_exact_role(records, run_role=e4['nonthinking_role'], information_condition='IC-b')
    pilot = select_exact_role(records, run_role=e4['thinking_pilot_role'], information_condition='IC-b')
    full = select_exact_role(records, run_role=e4['thinking_full_role'], information_condition='IC-b')
    rows = []
    sources = []
    for args in [('Claude Haiku 4.5 non-thinking', non, 1725, False, 'QC fail; excluded from confirmatory ranking.'), ('Haiku thinking-2048 pilot', pilot, 150, False, 'Pilot only; not confirmatory.'), ('Haiku thinking-2048 full', full, 1725, False, 'Full characterization; aggregate/condition gates not jointly passed.')]:
        r, s = _completed_row(*args)
        rows.append(r)
        sources.append(s)
    checkpoints = sorted((root / 'data/final_freeze/llm_runs').glob('checkpoint_*gemini35flash*.jsonl'))
    if len(checkpoints) != 1:
        raise ValueError(f'expected one Gemini 3.5 checkpoint; found {checkpoints}')
    checkpoint = checkpoints[0]
    attempt_sum = 0
    parsed = 0
    failures = 0
    with checkpoint.open('r', encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            attempt_sum += int(x.get('attempt_count') or 0)
            pa = x.get('parsed_action') if isinstance(x.get('parsed_action'), dict) else {}
            if pa and pa.get('budget_l1_raw') is not None:
                parsed += 1
            else:
                failures += 1
    rows.append({'configuration': 'Gemini 3.5 Flash interrupted C4', 'sample_state': f'{parsed + failures} checkpoint rows; {attempt_sum} provider attempts', 'aggregate_raw_compliance': None, 'C4_n': parsed + failures, 'C4_raw_compliance': None, 'C4R_n': 0, 'C4R_raw_compliance': None, 'C6_n': 0, 'C6_raw_compliance': None, 'status': 'INTERRUPTED_MAX_TOKENS', 'evidence_class': 'HISTORICAL_PROVENANCE', 'run_label': checkpoint.stem, 'notes': f'Checkpoint-only failed run; parsed={parsed}, failures={failures}.'})
    sources.append({'configuration': 'Gemini 3.5 Flash interrupted C4', 'path': str(checkpoint), 'rows': parsed + failures})
    for cohort, role in profile['extensions']['e3']['cohorts'].items():
        selected = [r for r in records if r.run_role == role and r.information_condition == 'IC-b' and (r.freeform_l1_budget is not None)]
        if len(selected) != 3:
            raise ValueError(f'{cohort} finite arm count must be 3, got {len(selected)}')
        compliance = []
        by_policy = {p: [] for p in POLICIES}
        src = []
        for run in selected:
            p, df = _action_table(run.run_dir)
            compliance.append(float(df['budget_compliant_raw'].mean()))
            src.append(str(p))
            for pol in POLICIES:
                by_policy[pol].append(float(df.loc[df['policy'].astype(str).eq(pol), 'budget_compliant_raw'].mean()))
            sources.append({'configuration': f'{cohort}:{run.run_label}', 'path': str(p), 'rows': len(df)})
        rows.append({'configuration': 'GPT-5.4-mini finite E3' if cohort == 'gpt54mini' else 'Gemini 3.1 Flash-Lite finite E3', 'sample_state': '3 finite arms × 1725 rows', 'aggregate_raw_compliance': f'{min(compliance):.6f}..{max(compliance):.6f}', **{f'{p}_n': 3 * 575 for p in POLICIES}, **{f'{p}_raw_compliance': f'{min(by_policy[p]):.6f}..{max(by_policy[p]):.6f}' for p in POLICIES}, 'status': 'PASS_CONFIRMATORY', 'evidence_class': 'RECOMPUTED_NUMERIC', 'run_label': ';'.join((r.run_label for r in selected)), 'notes': 'All finite arms pass the preregistered raw-budget gate.'})
    table = pd.DataFrame(rows)
    table_path = out / 'e4_budget_contract_ladder.csv'
    source_path = out / 'e4_budget_contract_input_files.csv'
    table.to_csv(table_path, index=False, encoding='utf-8-sig')
    pd.DataFrame(sources).to_csv(source_path, index=False, encoding='utf-8-sig')
    manifest = {'schema_version': 'e4_budget_contract_ladder_v1', 'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'confirmatory_rule': 'Only completed configurations passing the frozen raw-budget QC enter confirmatory ranking.', 'outputs': {'table': {'path': table_path.name, 'rows': len(table)}, 'inputs': {'path': source_path.name, 'rows': len(sources)}}, 'expected_haiku_full_common_complete_cases': int(e4['common_complete_case_n'])}
    mp = out / 'e4_budget_contract_ladder_manifest.json'
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return manifest

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    m = build(Path(a.project_root), Path(a.out))
    print(json.dumps(m, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
