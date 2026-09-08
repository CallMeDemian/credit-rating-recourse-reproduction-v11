from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
from .claim_evidence_common import repo_rel, write_csv, write_json

def run(root: Path) -> dict[str, Any]:
    root = root.resolve()
    ledger = root / 'data/final_freeze/ledgers/stage1_substrate_validation_loopB1.json'
    if not ledger.is_file():
        raise FileNotFoundError(f'exact Stage1 Loop-B1 ledger missing: {ledger}')
    payload = json.loads(ledger.read_text(encoding='utf-8-sig'))
    if payload.get('status') != 'PASS' or payload.get('gate_verdict_basis') != 'alpha_main_backend':
        raise RuntimeError('Stage1 Loop-B1 ledger is not a PASS alpha-main-backend contract')
    alpha = (payload.get('per_backend') or {}).get('alpha')
    if not isinstance(alpha, dict):
        raise RuntimeError('Stage1 Loop-B1 ledger lacks per_backend.alpha')
    oot = alpha.get('oot')
    if not isinstance(oot, dict):
        raise RuntimeError('Stage1 Loop-B1 ledger lacks per_backend.alpha.oot')
    ci = oot.get('lead_direction_agreement_ci95')
    required = {'lead_direction_agreement', 'lead_spearman', 'level_validity_spearman_oriented', 'n_pairs', 'n_movers'}
    missing = sorted(required - set(oot))
    if missing or not isinstance(ci, list) or len(ci) != 2:
        raise RuntimeError(f'Oracle-alpha OOT ledger contract incomplete: missing={missing}, ci={ci}')
    row = {'oracle_backend': 'alpha', 'split': 'OOT', 'direction_agreement': float(oot['lead_direction_agreement']), 'ci_low': float(ci[0]), 'ci_high': float(ci[1]), 'concurrent_rho': float(oot['level_validity_spearman_oriented']), 'lead_rho': float(oot['lead_spearman']), 'n_pairs': int(oot['n_pairs']), 'n_movers': int(oot['n_movers']), 'direction_agreement_ci_method': oot.get('direction_agreement_ci_method'), 'backend_verdict': alpha.get('verdict'), 'gate_verdict': payload.get('gate_verdict'), 'source_path': repo_rel(root, ledger)}
    if not 0.0 <= row['ci_low'] <= row['direction_agreement'] <= row['ci_high'] <= 1.0:
        raise RuntimeError('Oracle-alpha OOT Wilson interval ordering failed')
    output = root / 'data/analysis/paper_repro/07_claim_sources/oracle_oot_claim_summary.csv'
    write_csv(output, [row])
    manifest = {'schema_version': 'oracle_oot_claim_summary_v2', 'status': 'PASS', 'source_path': row['source_path'], 'row_count': 1, 'scope': {'oracle_backend': 'alpha', 'split': 'OOT'}, 'output': repo_rel(root, output)}
    write_json(output.with_suffix('.manifest.json'), manifest)
    return manifest

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
