from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
from typing import Any
import pandas as pd
from .claim_evidence_common import load_yaml, write_json
COMPLETED_VERDICTS = {'SUPPORTED', 'SUPPORTED_WITH_BOUNDARY', 'PARTIALLY_SUPPORTED', 'NOT_SUPPORTED'}

def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / 'src/credit_recourse/configs'
    claims = load_yaml(config_root / 'claim_evidence_registry.yaml')['claims']
    profile = load_yaml(config_root / 'verification_package_profile.yaml')
    review_root = root / profile['root']
    for directory in profile['directories']:
        (review_root / directory).mkdir(parents=True, exist_ok=True)
    evidence_root = root / 'data/reproduction/claim_evidence'
    ledger_path = evidence_root / 'claim_evidence_ledger.csv'
    if not ledger_path.is_file():
        raise FileNotFoundError(f'claim evidence ledger missing: {ledger_path}')
    ledger = pd.read_csv(ledger_path, encoding='utf-8-sig', keep_default_na=False)
    if len(ledger) != 28 or ledger['claim_id'].duplicated().any():
        raise RuntimeError('review package requires exactly 28 unique ledger rows')
    ledger_map = {str(row['claim_id']): row for _, row in ledger.iterrows()}
    card_source = evidence_root / 'claim_cards'
    card_source.mkdir(parents=True, exist_ok=True)
    card_target = review_root / '01_CLAIM_EVIDENCE/claim_cards'
    card_target.mkdir(parents=True, exist_ok=True)
    for stale in list(card_source.glob('C*.md')) + list(card_target.glob('C*.md')):
        stale.unlink()
    index: list[tuple[str, str, str]] = []
    for claim in claims:
        claim_id = str(claim['claim_id'])
        if claim_id not in ledger_map:
            raise RuntimeError(f'ledger row missing for {claim_id}')
        row = ledger_map[claim_id]
        verdict = str(row.get('runtime_verdict') or row.get('verdict') or 'UNRESOLVED')
        reason = str(row.get('reason') or '')
        card_path = card_source / f'{claim_id}.md'
        card_text = f"# {claim_id} — {claim['claim_type']}\n\n## Claim\n{claim['claim_text_ko']}\n\n## Scope\n{claim['scope_text']}\n\n## Extracted values\n{claim['key_evidence_text']}\n\n## Decision rule\n{claim['uncertainty_and_decision_rule']}\n\n## Runtime evidence status\n{row.get('runtime_evidence_status', '')}\n\n## Verdict\n{verdict}\n\n## Verdict basis\n{reason}\n\n## Planned writing status\n{row.get('planned_writing_status', '')}\n\n## Allowed interpretation\n{claim['allowed_interpretation']}\n\n## Forbidden interpretation\n{claim['forbidden_interpretation']}\n\n## Canonical evidence\n{claim['canonical_evidence_contract']}\n\n## 입력 자료와 계산 경로\n관련 source sidecar: {', '.join(claim.get('source_ids', []))}.\n\n## Paper location\n{', '.join(claim.get('paper_locations', []))}\n\n## Reproduction tier\n{claim['evidence_class']}\n"
        card_path.write_text(card_text, encoding='utf-8')
        shutil.copy2(card_path, card_target / card_path.name)
        index.append((claim_id, str(claim['claim_type']), verdict))
    for filename in ('claim_evidence_ledger.csv', 'unresolved_claims.csv', 'source_registry_resolved.csv'):
        source = evidence_root / filename
        if source.is_file():
            shutil.copy2(source, review_root / '01_CLAIM_EVIDENCE' / filename)
    start = review_root / '00_START_HERE'
    completed_mask = ledger['runtime_verdict'].astype(str).isin(COMPLETED_VERDICTS)
    unresolved_count = int((~completed_mask).sum())
    supported_count = int(completed_mask.sum())
    (start / 'README.md').write_text('# 논문 재현 검증 패키지\n\n확인 순서: `thesis_claim_index.md` 또는 `thesis_claim_index.csv`에서 주장 ID 검색 → 검증 카드 확인.\n', encoding='utf-8')
    (start / 'WHAT_WAS_REPRODUCED.md').write_text(f'# 재현된 항목\n\n실행 중 평가된 주장: {supported_count}/28. 정확한 판정 위치: claim evidence ledger.\n', encoding='utf-8')
    (start / 'WHAT_WAS_NOT_REPRODUCED.md').write_text(f'# 재현되지 않은 항목\n\n미해결 주장: {unresolved_count}. Frozen LLM 출력: 고정 입력, 재생성 제외.\n', encoding='utf-8')
    (start / 'reproduction_modes.md').write_text((config_root / 'reproduction_modes.yaml').read_text(encoding='utf-8'), encoding='utf-8')
    claim_index_rows = [{'주장 ID': claim_id, '유형': claim_type, '실행 판정': verdict, '카드 경로': f'../01_CLAIM_EVIDENCE/claim_cards/{claim_id}.md'} for claim_id, claim_type, verdict in index]
    pd.DataFrame(claim_index_rows).to_csv(start / 'thesis_claim_index.csv', index=False, encoding='utf-8-sig')
    claim_lines = ['# 논문 주장 검증 목록', '', '| 주장 ID | 유형 | 실행 판정 | 검증 카드 |', '|---|---|---|---|']
    claim_lines.extend((f'| {claim_id} | {claim_type} | {verdict} | [카드](../01_CLAIM_EVIDENCE/claim_cards/{claim_id}.md) |' for claim_id, claim_type, verdict in index))
    (start / 'thesis_claim_index.md').write_text('\n'.join(claim_lines) + '\n', encoding='utf-8')
    status = {'schema_version': 'verification_package_status_v4_1', 'status': 'PASS' if unresolved_count == 0 else 'PARTIAL', 'claim_count': len(claims), 'runtime_evaluated_count': supported_count, 'unresolved_count': unresolved_count, 'verdict_counts': ledger['runtime_verdict'].astype(str).value_counts().to_dict()}
    write_json(start / 'final_status.json', status)
    return status

def main(argv: list[str] | None=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True)
    args = parser.parse_args(argv)
    result = run(Path(args.project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
