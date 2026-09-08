from __future__ import annotations
'Build a broad, human-readable thesis visual catalogue from frozen artifacts.\n\nThe builder is deliberately downstream-only. It reads ``data/final_freeze`` and\ncanonical post-freeze analysis artifacts, writes human-facing tables/figures and\nexact plot data only under ``04_paper_assets``, and writes inventory, lineage,\nsidecar manifests, and claim-evidence links only under\n``06_thesis_registry/visual_assets``. Missing optional evidence skips one asset\nwithout stopping the rest.\n'
import argparse
import json
import math
import os
import re
import shutil
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
import pandas as pd
from credit_recourse.analysis.paper_output_layout import build_layout, ensure_layout
from credit_recourse.analysis.thesis_visual_specialized_assets import SpecializedEvidenceMissing, build_specialized_asset
SCHEMA = 'thesis_visual_catalog_v2'
CLAIM_MAP_SCHEMA = 'thesis_visual_claim_map_v1'
DPI = 360
PALETTE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']
CATEGORY_DIRS = {'A': '01_oracle_validation', 'B': '02_action_space', 'C': '03_rl_reference', 'D': '04_llm_core', 'E': '05_action_budget', 'F': '06_c4r_decomposition', 'G': '07_backend_robustness', 'H': '08_information_conditions', 'I': '09_revision_reference', 'J': '10_null_tests', 'K': '11_design_diagrams', 'L': '12_appendix'}
PRIORITY = {'P0': 'A1 A4 B1 B2 C1 C2 C4 D1 D2 E1 E2 F1 F2 F4 G1 G2 G5 H1 H2 I1 I5 K1 K2'.split(), 'P1': 'A3 A5 A6 A8 A9 B3 C3 C5 C6 D3 D4 D5 D8 D9 D10 E3 E4 E6 E7 F3 F5 F6 F7 F8 G3 G4 G6 H3 I2 I3 J1 J2 K3 K4 K5'.split(), 'P2': 'A2 A7 B4 C7 C8 C9 D6 D7 E5 H4 I4 I6 J3 L1 L2 L3 L4 L5 L6 L7 L8 L9 L10 L11 L12'.split()}

@dataclass(frozen=True)
class AssetSpec:
    asset_id: str
    kind: str
    title: str
    handler: str
    source_hints: tuple[str, ...] = ()
    priority: str = 'P2'

    @property
    def category(self) -> str:
        return CATEGORY_DIRS[self.asset_id[0]]

    @property
    def slug(self) -> str:
        return re.sub('[^a-z0-9]+', '_', self.handler.lower()).strip('_')

def _priority_of(asset_id: str) -> str:
    return next((p for p, ids in PRIORITY.items() if asset_id in ids))

def _spec(asset_id: str, kind: str, title: str, handler: str, *hints: str) -> AssetSpec:
    return AssetSpec(asset_id, kind, title, handler, tuple(hints), _priority_of(asset_id))
SPECS: list[AssetSpec] = [_spec('A1', 'table', 'Oracle 3종 OOT 검증 요약', 'oracle_validation', 'stage1_substrate_validation_loopB1.json'), _spec('A2', 'table', 'DEV와 OOT 구간별 Oracle 검증', 'oracle_validation', 'stage1_substrate_validation_loopB1.json'), _spec('A3', 'figure', 'Oracle-α 점수와 실제 등급 관계', 'oracle_scatter', 'oracle', 'rating_sample'), _spec('A4', 'figure', 'Oracle별 DEV·OOT 방향일치율', 'oracle_validation_chart', 'stage1_substrate_validation_loopB1.json'), _spec('A5', 'table', 'B1 직접경로와 B2 시뮬레이터 경로 격차', 'b1_b2', 'verify_stage2_substrate_loopA_loopB2.json', 'b2_gap'), _spec('A6', 'figure', 'B1 대 B2 방향일치율', 'b1_b2_chart', 'verify_stage2_substrate_loopA_loopB2.json', 'b2_gap'), _spec('A7', 'table', '시뮬레이터 구조 안정성 지표', 'test3', 'verify_stage2_substrate_loopA_loopB2.json', 'test3'), _spec('A8', 'figure', '구조적 사건 유무별 B2 방향일치율', 'structural', 'structural_event_slice'), _spec('A9', 'figure', '검사3 연도별 수준편향과 방향 재현율', 'test3_panel', 'test3_rows.csv', 'test3_property_summary.csv', 'test3_wedge_by_year.csv'), _spec('B1', 'table', '11개 후보행동 정의와 10차원 벡터', 'candidate_library', 'final_candidate_library.yaml'), _spec('B2', 'table', '10차원 행동공간 계약', 'action_contract', 'final_action_contract.yaml'), _spec('B3', 'figure', '후보행동 벡터 히트맵', 'candidate_heatmap', 'final_candidate_library.yaml'), _spec('B4', 'table', 'P50 보정 전후 후보벡터 비교', 'candidate_compare', 'final_candidate_library.yaml', 'final_candidate_library_calibrated.yaml'), _spec('C1', 'table', '정책별 Δnoop 요약', 'rl_policy', 'rl_stage6_policy_summary.csv'), _spec('C2', 'figure', '정책별 Oracle-α Δnoop', 'rl_policy_alpha_chart', 'rl_stage6_policy_summary.csv'), _spec('C3', 'figure', '정책별 3-Oracle Δnoop', 'rl_policy_oracle_chart', 'rl_stage6_policy_summary.csv'), _spec('C4', 'table', 'Candidate-IQL 7-seed 안정성', 'seven_seed', 'rl_seven_seed_summary.csv'), _spec('C5', 'figure', '7-seed Δnoop 분포', 'seven_seed_chart', 'rl_seven_seed_summary.csv'), _spec('C6', 'figure', 'C3 행동 선택 분포', 'action_distribution', 'rl_stage6_policy_summary.csv', 'stage6'), _spec('C7', 'table', '선택지형 flatness와 동점률', 'flatness', 'stage6', 'candidate_selector'), _spec('C8', 'figure', 'IQL Pareto 절충점', 'pareto', 'pareto'), _spec('C9', 'table', '최적행동 포착률 정의 3종', 'capture_definitions', 'policy_actions.parquet', 'multi_oracle_policy_eval.parquet', 'final_candidate_library.yaml'), _spec('D1', 'table', 'LLM N5 핵심 결과표', 'n5', 'n5_7_10c_table_patch.csv', 'n5'), _spec('D2', 'figure', 'RL·LLM 정책 사다리', 'policy_ladder', 'n5_7_10c_table_patch.csv', 'rl_stage6_policy_summary.csv'), _spec('D3', 'figure', '정보조건별 Δnoop', 'n5_ic_chart', 'n5_7_10c_table_patch.csv'), _spec('D4', 'table', '조건별 C3 대비 승률', 'winrate', 'winrate'), _spec('D5', 'figure', '조건별 C3 대비 승률', 'winrate_chart', 'winrate'), _spec('D6', 'table', '기업규모별 정책효과 이질성', 'heterogeneity', 'heterogeneity', 'log_assets'), _spec('D7', 'figure', '자산규모 분위별 Δnoop', 'heterogeneity_chart', 'heterogeneity', 'log_assets'), _spec('D8', 'table', 'LLM 실패유형 빈도', 'failure', 'llm_stage7_failure_audit.csv'), _spec('D9', 'figure', '정보조건별 실패유형 분포', 'failure_chart', 'llm_stage7_failure_audit.csv'), _spec('D10', 'figure', '성과–검증가능성 프런티어', 'performance_verifiability_frontier', 'main_harness_backend_cell_means.csv', 'main_harness_backend_input_files.csv'), _spec('E1', 'table', '행동크기 제약별 정책효과', 'budget', 'n5_budget_frontier_table_patch.csv', 'budget_frontier'), _spec('E2', 'figure', '행동크기 제약–성과 frontier', 'budget_curve', 'n5_budget_frontier_table_patch.csv', 'budget_frontier'), _spec('E3', 'figure', '예산×조건 상호작용', 'budget_heatmap', 'ablation_cells', 'budget'), _spec('E4', 'table', '예산별 행동 선택 분포', 'budget_action', 'llm_stage7_actions.parquet', 'budget'), _spec('E5', 'figure', '행동크기 제약별 clipping rate', 'clipping', 'clipping', 'budget'), _spec('E6', 'figure', '계약 준수 사다리와 Haiku 자기검산', 'budget_compliance_ladder', 'e4_budget_contract_ladder.csv', 'e4_budget_contract_input_files.csv'), _spec('E7', 'figure', 'Raw 대 Applied 총 L1 분포', 'raw_applied_l1_distribution', 'e4_budget_contract_input_files.csv'), _spec('F1', 'table', 'C4·C4R·C6 matched 성분 분해', 'c4r', 'c4r', 'contrasts'), _spec('F2', 'figure', 'C4→C4R→C6 성분 분해', 'c4r_chart', 'c4r', 'contrasts'), _spec('F3', 'figure', '예산별 수정 성분효과', 'c4r_interaction', 'c4r', 'interaction'), _spec('F4', 'table', 'C4R 예산 상호작용 DID', 'did', 'c4r', 'interaction'), _spec('F5', 'figure', 'C4R DID 계수와 실질적 동등역', 'did_equivalence_forest', 'c4r_tost_interactions_alpha.csv', 'metadata.json'), _spec('F6', 'table', 'C6X와 C6 참조출처 수용', 'c6x', 'c6x', 'revision'), _spec('F7', 'figure', '기업별 승·동·패 구성', 'win_tie_loss_stack', 'c4r_matched_v3_contrasts.csv'), _spec('F8', 'figure', 'Shapley 축귀속 발산막대', 'axis_shapley_diverging', 'c4r_axis_shapley_summary.csv'), _spec('G1', 'table', '하네스·백엔드 분산분해', 'decomposition', 'main_harness_backend_decomposition'), _spec('G2', 'figure', '하네스·백엔드 설명분산', 'decomposition_chart', 'main_harness_backend_decomposition'), _spec('G3', 'figure', '백엔드별 정책순위 변동', 'rank_chart', 'n5', 'backend'), _spec('G4', 'table', 'TOST 실질적 동등성 전수 원장', 'tost_complete_ledger', 'c4r_tost_within_arm_alpha.csv', 'c4r_tost_interactions_alpha.csv', 'metadata.json'), _spec('G5', 'figure', '정책 하네스 레버 공통눈금 효과', 'harness_lever_common_scale', 'main_harness_backend_cell_means.csv', 'shuffle_permutation_ci.csv', 'c4r_matched_v3_contrasts.csv'), _spec('G6', 'figure', '평가함수 관측 해상도', 'dynamic_resolution_panel', 'dynamic_resolution.csv'), _spec('H1', 'table', 'ICC 탐침 채널별 결과', 'icc', 'icc_probe_channel_summary.csv', 'icc_probe'), _spec('H2', 'figure', 'ICC 탐침 채널별 인식률', 'icc_chart', 'icc_probe_channel_summary.csv', 'icc_probe'), _spec('H3', 'figure', 'IC-a·b·c 정책효과 분포', 'ic_distribution', 'n5', 'information_condition'), _spec('H4', 'table', '정보조건별 행동선택 패턴', 'ic_actions', 'llm_stage7_actions.parquet'), _spec('I1', 'table', '수정·참조 채택 지표', 'revision', 'llm_stage9_revision_metrics.csv'), _spec('I2', 'figure', '참조 채택률과 자기유지율', 'revision_scatter', 'llm_stage9_revision_metrics.csv'), _spec('I3', 'table', '참조품질별 수용 분석', 'reference_quality', 'reference_quality'), _spec('I4', 'figure', '참조 전후 기업별 점수 변화', 'paired_revision', 'revision', 'stage8'), _spec('I5', 'figure', 'E2↔E3 기업별 성분효과 순위상관', 'e2_e3_rank_correlation', 'c4r_matched_firm_frame.csv', 'c4r_matched_v3_firm_frame.csv'), _spec('I6', 'table', 'LLM 반복 실행 안정성', 'repeat_stability', 'llm_repeat_stability.csv'), _spec('J1', 'table', 'Sign-flip null 검정', 'signflip', 'signflip'), _spec('J2', 'figure', 'Sign-flip permutation 분포', 'permutation', 'signflip', 'permutation'), _spec('J3', 'table', 'Shuffle null 비교', 'shuffle', 'ablation', 'shuffle'), _spec('K1', 'diagram', 'Oracle→RL→LLM 3단계 파이프라인', 'diagram_pipeline', 'paper_display_registry.yaml'), _spec('K2', 'diagram', '세 가지 성과귀속 오류', 'diagram_attribution', 'paper_display_registry.yaml'), _spec('K3', 'diagram', 'LLM 실험조건 계통도', 'diagram_experiment', 'paper_display_registry.yaml'), _spec('K4', 'diagram', 'C4→C4R→C6 식별 구조', 'diagram_c4r', 'paper_display_registry.yaml'), _spec('K5', 'diagram', '평가기반 검증 절차', 'diagram_validation', 'paper_display_registry.yaml'), _spec('L1', 'table', 'Revision dilution 상세', 'revision_dilution', 'revision_dilution'), _spec('L2', 'table', '관계적 참조가치 상세', 'relational', 'relational_reference'), _spec('L3', 'table', '전체 LLM run 목록', 'run_inventory', 'llm_runs'), _spec('L4', 'table', 'Stage별 산출물 계보', 'artifact_chain', 'thesis_artifact_registry.json'), _spec('L5', 'figure', 'RL 인코더 sweep 비교', 'encoder_sweep', 'rl_encoder_sweep'), _spec('L6', 'table', '선택변수와 Oracle 구성', 'variables', 'selected_variable_master.csv', 'oracle'), _spec('L7', 'figure', 'Oracle-α 가중치', 'alpha_weights', 'selected_variable_master.csv', 'alpha'), _spec('L8', 'table', '표본 주요 변수 기술통계', 'descriptive', 'stage0', 'stage2'), _spec('L9', 'figure', '주요 재무변수 분포', 'histograms', 'stage0', 'stage2'), _spec('L10', 'table', '산업별 기업 수', 'industry', 'stage0', 'industry'), _spec('L11', 'figure', '실제 신용등급 분포', 'rating_distribution', 'rating', 'stage1'), _spec('L12', 'table', 'Holm 보정 p-value 원장', 'holm', 'holm', 'inference')]

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _jsonable(v: Any) -> Any:
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (pd.Timestamp,)):
        return v.isoformat()
    if pd.isna(v) if not isinstance(v, (list, dict, tuple, set)) else False:
        return None
    if hasattr(v, 'item'):
        try:
            return v.item()
        except Exception:
            pass
    return v

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_jsonable) + '\n', encoding='utf-8')

def _markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return '_No rows._\n'

    def cell(x: Any) -> str:
        if isinstance(x, (dict, list, tuple, set)):
            return json.dumps(x, ensure_ascii=False, default=_jsonable)
        try:
            if pd.isna(x):
                return ''
        except Exception:
            pass
        return str(x)
    cols = [str(c) for c in df.columns]
    rows = [[cell(x) for x in row] for row in df.itertuples(index=False, name=None)]
    widths = [len(c) for c in cols]
    for row in rows:
        widths = [max(w, len(x)) for w, x in zip(widths, row)]
    fmt = lambda row: '| ' + ' | '.join((x.ljust(w) for x, w in zip(row, widths))) + ' |'
    return '\n'.join([fmt(cols), '| ' + ' | '.join(('-' * w for w in widths)) + ' |', *[fmt(r) for r in rows]]) + '\n'

def _norm(s: Any) -> str:
    return re.sub('[^a-z0-9]+', '', str(s).lower())

def _find_col(df: pd.DataFrame, candidates: Sequence[str], contains: Sequence[str]=()) -> str | None:
    exact = {_norm(c): c for c in df.columns}
    for c in candidates:
        if _norm(c) in exact:
            return exact[_norm(c)]
    for col in df.columns:
        n = _norm(col)
        if any((_norm(x) in n for x in contains)):
            return str(col)
    return None

def _numeric_cols(df: pd.DataFrame, minimum: int=2) -> list[str]:
    out = []
    for c in df.columns:
        x = pd.to_numeric(df[c], errors='coerce')
        if int(x.notna().sum()) >= minimum:
            out.append(str(c))
    return out

def _flatten_json(obj: Any, prefix: str='') -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f'{prefix}.{k}' if prefix else str(k)
            if isinstance(v, (dict, list)):
                rows.extend(_flatten_json(v, p))
            else:
                rows.append({'metric': p, 'value': v})
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            rows.extend(_flatten_json(v, f'{prefix}[{i}]'))
    else:
        rows.append({'metric': prefix or 'value', 'value': obj})
    return rows

class EvidenceMissing(RuntimeError):
    pass

@dataclass
class AssetRun:
    spec: AssetSpec
    started_utc: str = field(default_factory=_now)
    status: str = 'RUNNING'
    inputs: list[dict[str, Any]] = field(default_factory=list)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    transformations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

class ArtifactIndex:

    def __init__(self, roots: Sequence[Path], excluded: Sequence[Path]=()) -> None:
        self.roots = [Path(p).resolve() for p in roots if Path(p).exists()]
        self.excluded = [Path(p).resolve() for p in excluded]
        self.files: list[Path] = []
        for root in self.roots:
            for p in root.rglob('*'):
                if not p.is_file():
                    continue
                rp = p.resolve()
                if any((e == rp or e in rp.parents for e in self.excluded)):
                    continue
                if any((part in {'__pycache__', '.git'} for part in p.parts)):
                    continue
                self.files.append(rp)
        self.files = sorted(set(self.files), key=lambda x: str(x).lower())

    def find(self, *hints: str, suffixes: Sequence[str]=()) -> list[Path]:
        hs = [_norm(h) for h in hints if h]
        sf = {s.lower() for s in suffixes}
        scored: list[tuple[int, Path]] = []
        for p in self.files:
            if sf and p.suffix.lower() not in sf:
                continue
            text = _norm(str(p))
            base = _norm(p.name)
            matched = sum((1 for h in hs if h in text))
            if hs and matched == 0:
                continue
            score = matched * 100 + sum((40 for h in hs if h in base))
            if 'finalfreeze' in text:
                score += 12
            if 'paperrepro' in text:
                score += 8
            scored.append((score, p))
        return [p for _, p in sorted(scored, key=lambda t: (-t[0], len(str(t[1])), str(t[1]).lower()))]

    def first(self, *hints: str, suffixes: Sequence[str]=()) -> Path:
        found = self.find(*hints, suffixes=suffixes)
        if not found:
            raise EvidenceMissing(f'No artifact matched hints={hints}, suffixes={tuple(suffixes)}')
        return found[0]

def _load_yaml_required(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f'Required visual registry config is missing: {path}')
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError(f'PyYAML is required to load {path}: {exc!r}') from exc
    obj = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise RuntimeError(f'Registry config must be a mapping: {path}')
    return obj

def _load_claim_registry_contract(config_dir: Path, project_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    visual_path = config_dir / 'thesis_visual_claim_map.yaml'
    display_path = config_dir / 'paper_display_registry.yaml'
    claim_path = config_dir / 'claim_evidence_registry.yaml'
    specialized_path = config_dir / 'thesis_visual_specialized_contract.yaml'
    visual = _load_yaml_required(visual_path)
    display = _load_yaml_required(display_path)
    claims = _load_yaml_required(claim_path)
    specialized = _load_yaml_required(specialized_path)
    if visual.get('schema_version') != CLAIM_MAP_SCHEMA:
        raise RuntimeError(f"Unexpected visual claim-map schema: {visual.get('schema_version')!r}")
    if specialized.get('schema_version') != 'thesis_visual_specialized_contract_v1':
        raise RuntimeError(f"Unexpected specialised visual contract schema: {specialized.get('schema_version')!r}")
    forbidden_terms = [str(x) for x in (specialized.get('terminology') or {}).get('forbidden_asset_title_terms') or []]
    violations = [f'{spec.asset_id}:{term}' for spec in SPECS for term in forbidden_terms if term and term in spec.title]
    if violations:
        raise RuntimeError(f'Registered visual titles violate terminology contract: {violations}')
    rows = visual.get('assets')
    if not isinstance(rows, list):
        raise RuntimeError('thesis_visual_claim_map.yaml must contain an assets list')
    expected = {spec.asset_id for spec in SPECS}
    ids = [str(row.get('asset_id', '')) for row in rows if isinstance(row, dict)]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Duplicate asset_id in thesis_visual_claim_map.yaml')
    if set(ids) != expected:
        raise RuntimeError(f'Visual claim-map IDs do not match the registered visual catalogue: missing={sorted(expected - set(ids))}, extra={sorted(set(ids) - expected)}')
    slot_ids = {str(row.get('slot_id')) for row in display.get('slots') or [] if isinstance(row, dict) and row.get('slot_id')}
    claim_ids = {str(row.get('claim_id')) for row in claims.get('claims') or [] if isinstance(row, dict) and row.get('claim_id')}
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        aid = str(row['asset_id'])
        slot = str(row.get('slot_id', ''))
        linked_claims = [str(x) for x in row.get('claim_ids') or []]
        section = str(row.get('paper_section', ''))
        if slot not in slot_ids:
            raise RuntimeError(f'{aid}: unknown display slot {slot!r}')
        unknown_claims = sorted(set(linked_claims) - claim_ids)
        if unknown_claims:
            raise RuntimeError(f'{aid}: unknown claim IDs {unknown_claims}')
        if not section:
            raise RuntimeError(f'{aid}: paper_section is required')
        mapping[aid] = {'slot': slot, 'claims': linked_claims, 'section': section}
    sources = []
    for role, path, obj in (('visual_claim_map', visual_path, visual), ('paper_display_registry', display_path, display), ('claim_evidence_registry', claim_path, claims), ('specialized_visual_contract', specialized_path, specialized)):
        sources.append({'role': role, 'relative_path': _relative(path, project_root), 'schema_version': obj.get('schema_version')})
    validation = {'status': 'PASS', 'asset_count': len(mapping), 'display_slot_count': len(slot_ids), 'claim_count': len(claim_ids), 'rules': ['registered visual asset IDs are exact and unique', 'every slot_id exists in paper_display_registry.yaml', 'every claim_id exists in claim_evidence_registry.yaml', 'registered titles satisfy thesis_visual_specialized_contract.yaml terminology']}
    return (mapping, sources, validation)

def _remove_recognized_legacy_catalog(legacy_dir: Path, project_root: Path) -> dict[str, Any]:
    if not legacy_dir.exists():
        return {'status': 'NOT_PRESENT', 'relative_path': _relative(legacy_dir, project_root)}
    files = [p for p in legacy_dir.rglob('*') if p.is_file()]
    if not files:
        legacy_dir.rmdir()
        return {'status': 'REMOVED_EMPTY', 'relative_path': _relative(legacy_dir, project_root)}
    manifest_path = legacy_dir / 'visual_assets_manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('Legacy 04_paper_assets/visual_catalog exists without its recognized manifest; refusing to delete an unclassified directory')
    obj = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if obj.get('schema_version') != 'thesis_visual_catalog_v1':
        raise RuntimeError(f"Legacy visual_catalog has an unrecognized schema {obj.get('schema_version')!r}; manual review is required")
    evidence = {'status': 'REMOVED_REGENERABLE_V1_OUTPUT', 'relative_path': _relative(legacy_dir, project_root), 'legacy_file_count': len(files)}
    shutil.rmtree(legacy_dir)
    return evidence

class BuildContext:

    def __init__(self, project_root: Path, analysis_dir: Path, figures_dir: Path, tables_dir: Path, plot_data_dir: Path, registry_dir: Path, strict: bool) -> None:
        self.project_root = project_root.resolve()
        self.analysis_dir = analysis_dir.resolve()
        self.figures_dir = figures_dir.resolve()
        self.tables_dir = tables_dir.resolve()
        self.assets_root = self.figures_dir.parent
        self.snapshot_dir = plot_data_dir.resolve()
        self.registry_dir = registry_dir.resolve()
        self.manifest_dir = self.registry_dir / 'asset_manifests'
        self.strict = strict
        self.final_freeze = self.project_root / 'data' / 'final_freeze'
        self.config_dir = self.project_root / 'src' / 'credit_recourse' / 'configs'
        self.claim_map, self.registry_sources, self.registry_validation = _load_claim_registry_contract(self.config_dir, self.project_root)
        self.legacy_layout_cleanup = _remove_recognized_legacy_catalog(self.assets_root / 'visual_catalog', self.project_root)
        roots = [self.final_freeze, self.analysis_dir, self.config_dir]
        self.index = ArtifactIndex(roots, excluded=[self.assets_root, self.registry_dir])
        self._plt: Any = None
        self._mpl_error: str | None = None
        for d in [self.figures_dir, self.tables_dir, self.snapshot_dir, self.registry_dir, self.manifest_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def matplotlib(self) -> Any:
        if self._plt is not None:
            return self._plt
        if self._mpl_error:
            raise EvidenceMissing(self._mpl_error)
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from matplotlib import font_manager
            preferred = ['Malgun Gothic', 'NanumGothic', 'Noto Sans CJK KR', 'AppleGothic', 'Arial Unicode MS']
            names = {f.name for f in font_manager.fontManager.ttflist}
            font = next((x for x in preferred if x in names), None)
            if font:
                matplotlib.rcParams['font.family'] = font
            matplotlib.rcParams.update({'axes.unicode_minus': False, 'figure.dpi': 120, 'savefig.dpi': DPI, 'font.size': 10.5, 'axes.titlesize': 12, 'axes.labelsize': 10.5, 'legend.fontsize': 9, 'xtick.labelsize': 9, 'ytick.labelsize': 9, 'figure.constrained_layout.use': True})
            self._plt = plt
            return plt
        except Exception as exc:
            self._mpl_error = f'matplotlib unavailable: {exc!r}'
            raise EvidenceMissing(self._mpl_error)

    def record_input(self, run: AssetRun, path: Path, role: str) -> None:
        path = path.resolve()
        st = path.stat()
        entry = {'role': role, 'path': str(path), 'relative_path': _relative(path, self.project_root), 'size_bytes': st.st_size, 'mtime_utc': datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(), 'reader': _reader_name(path)}
        run.inputs.append(entry)

    def read(self, run: AssetRun, path: Path, role: str='source') -> Any:
        self.record_input(run, path, role)
        suf = path.suffix.lower()
        if suf == '.csv':
            return pd.read_csv(path)
        if suf in {'.parquet', '.pq'}:
            return pd.read_parquet(path)
        if suf == '.json':
            return json.loads(path.read_text(encoding='utf-8-sig'))
        if suf in {'.yaml', '.yml'}:
            try:
                import yaml
            except Exception as exc:
                raise EvidenceMissing(f'PyYAML unavailable for {path}: {exc!r}')
            return yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        if suf in {'.xlsx', '.xls'}:
            return pd.read_excel(path)
        if suf in {'.md', '.txt'}:
            return path.read_text(encoding='utf-8-sig')
        raise EvidenceMissing(f'Unsupported source type: {path}')

    def table_dir(self, spec: AssetSpec) -> Path:
        p = self.tables_dir / spec.category
        p.mkdir(parents=True, exist_ok=True)
        return p

    def figure_dir(self, spec: AssetSpec) -> Path:
        p = self.figures_dir / spec.category
        p.mkdir(parents=True, exist_ok=True)
        return p

    def stem(self, spec: AssetSpec) -> str:
        return f'{spec.asset_id}_{spec.slug}'

    def write_table(self, run: AssetRun, df: pd.DataFrame, note: str='') -> None:
        if df.empty:
            raise EvidenceMissing('Resolved source produced an empty table')
        df = df.copy()
        out = self.table_dir(run.spec)
        stem = self.stem(run.spec)
        csv_path, md_path = (out / f'{stem}.csv', out / f'{stem}.md')
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        md_path.write_text(f'# {run.spec.asset_id}. {run.spec.title}\n\n' + _markdown(df) + (f'\n{note}\n' if note else ''), encoding='utf-8')
        for p, fmt in [(csv_path, 'csv'), (md_path, 'markdown')]:
            self.record_output(run, p, fmt, len(df), list(df.columns))

    def snapshot(self, run: AssetRun, df: pd.DataFrame, suffix: str='plot_data') -> Path:
        p = self.snapshot_dir / f'{self.stem(run.spec)}_{suffix}.csv'
        df.to_csv(p, index=False, encoding='utf-8-sig')
        self.record_output(run, p, 'data_snapshot', len(df), list(df.columns))
        return p

    def save_figure(self, run: AssetRun, fig: Any, data: pd.DataFrame, svg: bool=False) -> None:
        out = self.figure_dir(run.spec)
        stem = self.stem(run.spec)
        self.snapshot(run, data)
        for ext in ['png', 'pdf', 'svg'] if svg else ['png', 'pdf']:
            p = out / f'{stem}.{ext}'
            fig.savefig(p, dpi=DPI if ext == 'png' else None, bbox_inches='tight', metadata={'Title': run.spec.title, 'Creator': SCHEMA})
            self.record_output(run, p, ext, None, None)
        self.matplotlib().close(fig)

    def record_output(self, run: AssetRun, path: Path, fmt: str, rows: int | None, columns: list[str] | None) -> None:
        run.outputs.append({'format': fmt, 'path': str(path.resolve()), 'relative_path': _relative(path.resolve(), self.project_root), 'size_bytes': path.stat().st_size, 'row_count': rows, 'columns': columns})

def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())

def _reader_name(path: Path) -> str:
    return {'.csv': 'pandas.read_csv', '.parquet': 'pandas.read_parquet', '.pq': 'pandas.read_parquet', '.json': 'json.loads', '.yaml': 'yaml.safe_load', '.yml': 'yaml.safe_load', '.xlsx': 'pandas.read_excel'}.get(path.suffix.lower(), 'text/binary reader')

def _as_frame(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, list):
        if all((isinstance(x, dict) for x in obj)):
            return pd.DataFrame(obj)
        return pd.DataFrame({'value': obj})
    if isinstance(obj, dict):
        candidates = [v for v in obj.values() if isinstance(v, list) and v and all((isinstance(x, dict) for x in v))]
        if candidates:
            return pd.DataFrame(max(candidates, key=len))
        return pd.DataFrame(_flatten_json(obj))
    return pd.DataFrame({'value': [obj]})

def _config_source_allowed(hints: Sequence[str]) -> bool:
    joined = ' '.join(hints).lower()
    return any((x in joined for x in ('final_candidate_library', 'final_action_contract', 'selected_variable_master', 'thesis_artifact_registry', 'paper_display_registry', 'oracle_backend_registry')))

def _evidence_paths(ctx: BuildContext, hints: Sequence[str], suffixes: Sequence[str]) -> list[Path]:
    filename_hints = [Path(h).name.lower() for h in hints if Path(h).suffix.lower() in {x.lower() for x in suffixes}]
    exact = [p for p in ctx.index.files if p.suffix.lower() in {x.lower() for x in suffixes} and p.name.lower() in filename_hints]
    found = exact if exact else ctx.index.find(*hints, suffixes=suffixes)
    if not _config_source_allowed(hints):
        found = [p for p in found if ctx.config_dir.resolve() not in p.parents]
    return found

def _read_first(ctx: BuildContext, run: AssetRun, *hints: str, suffixes: Sequence[str]=('.csv', '.json', '.parquet', '.yaml', '.yml', '.xlsx')) -> tuple[Path, Any]:
    found = _evidence_paths(ctx, hints, suffixes)
    if not found:
        raise EvidenceMissing(f'No artifact matched hints={hints}, suffixes={tuple(suffixes)}')
    p = found[0]
    return (p, ctx.read(run, p))

def _combine_frames(ctx: BuildContext, run: AssetRun, paths: Sequence[Path], role: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            obj = ctx.read(run, p, role)
        except (ImportError, ValueError, OSError) as exc:
            run.notes.append(f'Skipped unreadable optional source {p}: {exc!r}')
            continue
        f = _as_frame(obj)
        if not f.empty:
            f = f.copy()
            f['source_file'] = _relative(p, ctx.project_root)
            frames.append(f)
    if not frames:
        raise EvidenceMissing(f'No readable rows for {role}')
    return pd.concat(frames, ignore_index=True, sort=False)

def _long_numeric(df: pd.DataFrame, id_cols: Sequence[str]=()) -> pd.DataFrame:
    nums = _numeric_cols(df)
    if not nums:
        raise EvidenceMissing('No numeric columns available')
    ids = [c for c in id_cols if c in df.columns]
    return df.melt(id_vars=ids, value_vars=nums, var_name='metric', value_name='value').dropna(subset=['value'])

def _bar(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, x: str, y: str, hue: str | None=None, horizontal: bool=False, zero: bool=True) -> None:
    plt = ctx.matplotlib()
    data = df.copy()
    data[y] = pd.to_numeric(data[y], errors='coerce')
    data = data.dropna(subset=[y])
    if data.empty:
        raise EvidenceMissing('No finite data for chart')
    fig, ax = plt.subplots(figsize=(9.4, max(4.8, 0.38 * len(data[x].astype(str).unique())) if horizontal else 5.6))
    cats = list(dict.fromkeys(data[x].astype(str)))
    if hue and hue in data.columns:
        hs = list(dict.fromkeys(data[hue].astype(str)))
        width = 0.8 / max(1, len(hs))
        for i, h in enumerate(hs):
            sub = data[data[hue].astype(str).eq(h)]
            vals = {str(a): b for a, b in zip(sub[x], sub[y])}
            pos = [j - 0.4 + width / 2 + i * width for j in range(len(cats))]
            ax.barh(pos, [vals.get(c, math.nan) for c in cats], height=width, label=h, color=PALETTE[i % len(PALETTE)]) if horizontal else ax.bar(pos, [vals.get(c, math.nan) for c in cats], width=width, label=h, color=PALETTE[i % len(PALETTE)])
        ticks = list(range(len(cats)))
        ax.legend(frameon=False)
    else:
        vals = data.groupby(data[x].astype(str), sort=False)[y].mean()
        cats = list(vals.index)
        ticks = list(range(len(cats)))
        ax.barh(ticks, vals.values, color=PALETTE[0]) if horizontal else ax.bar(ticks, vals.values, color=PALETTE[0])
    if horizontal:
        ax.set_yticks(ticks, cats)
        ax.set_xlabel(y)
        ax.set_ylabel('')
    else:
        ax.set_xticks(ticks, cats, rotation=35, ha='right')
        ax.set_ylabel(y)
        ax.set_xlabel('')
    if zero:
        ax.axvline(0, color='#444444', linewidth=0.8) if horizontal else ax.axhline(0, color='#444444', linewidth=0.8)
    ax.set_title(run.spec.title)
    ax.grid(axis='x' if horizontal else 'y', alpha=0.22)
    ctx.save_figure(run, fig, data)

def _line(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, x: str, y: str, hue: str | None=None) -> None:
    plt = ctx.matplotlib()
    data = df.copy()
    data[x] = pd.to_numeric(data[x], errors='coerce')
    data[y] = pd.to_numeric(data[y], errors='coerce')
    data = data.dropna(subset=[x, y])
    if data.empty:
        raise EvidenceMissing('No finite data for line chart')
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    groups = data.groupby(hue, dropna=False) if hue and hue in data.columns else [(None, data)]
    for i, (name, sub) in enumerate(groups):
        agg = sub.groupby(x, as_index=False)[y].mean().sort_values(x)
        ax.plot(agg[x], agg[y], marker='o', linewidth=2, label=str(name) if name is not None else None, color=PALETTE[i % len(PALETTE)])
    ax.axhline(0, color='#444444', linewidth=0.8)
    ax.set_title(run.spec.title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.grid(alpha=0.22)
    if hue and hue in data.columns:
        ax.legend(frameon=False)
    ctx.save_figure(run, fig, data)

def _heatmap(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, row: str, col: str, value: str) -> None:
    plt = ctx.matplotlib()
    data = df.copy()
    data[value] = pd.to_numeric(data[value], errors='coerce')
    p = data.pivot_table(index=row, columns=col, values=value, aggfunc='mean')
    if p.empty:
        raise EvidenceMissing('No matrix data for heatmap')
    fig, ax = plt.subplots(figsize=(max(7, 0.65 * len(p.columns)), max(4.8, 0.38 * len(p.index))))
    vmax = float(abs(p.to_numpy()).max()) if p.size else 1.0
    vmax = max(vmax, 1e-09)
    im = ax.imshow(p.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(p.columns)), [str(x) for x in p.columns], rotation=40, ha='right')
    ax.set_yticks(range(len(p.index)), [str(x) for x in p.index])
    ax.set_title(run.spec.title)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ctx.save_figure(run, fig, data)

def _scatter(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, x: str, y: str, hue: str | None=None) -> None:
    plt = ctx.matplotlib()
    data = df.copy()
    data[x] = pd.to_numeric(data[x], errors='coerce')
    data[y] = pd.to_numeric(data[y], errors='coerce')
    data = data.dropna(subset=[x, y])
    if data.empty:
        raise EvidenceMissing('No paired numeric data')
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    groups = data.groupby(hue, dropna=False) if hue and hue in data.columns else [(None, data)]
    for i, (name, sub) in enumerate(groups):
        ax.scatter(sub[x], sub[y], s=24, alpha=0.58, label=str(name) if name is not None else None, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(run.spec.title)
    ax.grid(alpha=0.2)
    if hue and hue in data.columns:
        ax.legend(frameon=False)
    ctx.save_figure(run, fig, data)

def _candidate_frame(obj: dict[str, Any]) -> pd.DataFrame:
    candidates = obj.get('fixed_candidates') or obj.get('candidates') or {}
    action_cols = obj.get('action_columns') or sorted({k for v in candidates.values() if isinstance(v, dict) for k in v if str(k).startswith('action__')})
    rows = []
    for name, body in candidates.items():
        if not isinstance(body, dict):
            continue
        row = {'candidate': name, 'tier': body.get('tier'), 'paper_role': body.get('paper_role'), 'intent': body.get('intent')}
        row.update({c: body.get(c, 0.0) for c in action_cols})
        rows.append(row)
    return pd.DataFrame(rows)

def _action_contract_frame(obj: dict[str, Any]) -> pd.DataFrame:
    for key in ('actions', 'action_dimensions', 'dimensions', 'action_columns'):
        val = obj.get(key)
        if isinstance(val, list):
            if all((isinstance(x, dict) for x in val)):
                return pd.DataFrame(val)
            return pd.DataFrame({'action_dimension': val})
        if isinstance(val, dict):
            return pd.DataFrame([{'action_dimension': k, **(v if isinstance(v, dict) else {'value': v})} for k, v in val.items()])
    rows = [x for x in _flatten_json(obj) if 'action' in str(x['metric']).lower() or 'tier' in str(x['metric']).lower()]
    return pd.DataFrame(rows)

def _oracle_validation(ctx: BuildContext, run: AssetRun) -> pd.DataFrame:
    _, obj = _read_first(ctx, run, 'stage1_substrate_validation_loopB1', suffixes=('.json',))
    df = _as_frame(obj)
    if set(df.columns) == {'metric', 'value'}:
        mask = df['metric'].astype(str).str.contains('alpha|beta|gamma|oot|dev|spearman|direction|mover|wilson|ci', case=False, regex=True)
        df = df.loc[mask]
    return df

def _first_frame(ctx: BuildContext, run: AssetRun, hints: Sequence[str], suffixes: Sequence[str]=('.csv', '.json', '.parquet', '.xlsx')) -> pd.DataFrame:
    _, obj = _read_first(ctx, run, *hints, suffixes=suffixes)
    return _as_frame(obj)

def _metric_value_chart(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, preferred_x: Sequence[str], preferred_y: Sequence[str], hue: Sequence[str]=(), horizontal: bool=False) -> None:
    x = _find_col(df, preferred_x, preferred_x)
    y = _find_col(df, preferred_y, preferred_y)
    h = _find_col(df, hue, hue) if hue else None
    if not x or not y:
        long = _long_numeric(df, [c for c in df.columns if df[c].dtype == object][:2])
        x = 'metric'
        y = 'value'
        df = long
        h = None
    _bar(ctx, run, df, x, y, h, horizontal=horizontal)

def _filter_metrics(df: pd.DataFrame, tokens: str) -> pd.DataFrame:
    pat = '|'.join((re.escape(x) for x in tokens.split('|')))
    mask = pd.Series(False, index=df.index)
    for c in df.columns:
        if df[c].dtype == object:
            mask |= df[c].astype(str).str.contains(pat, case=False, regex=True, na=False)
    out = df.loc[mask]
    return out if not out.empty else df

def _tabular_generic(ctx: BuildContext, run: AssetRun, hints: Sequence[str], tokens: str='') -> None:
    paths = _evidence_paths(ctx, hints, ('.csv', '.json', '.parquet', '.xlsx', '.yaml', '.yml'))[:8]
    if not paths:
        raise EvidenceMissing(f'No tabular evidence for {hints}')
    df = _combine_frames(ctx, run, paths, 'evidence source')
    if tokens:
        df = _filter_metrics(df, tokens)
    ctx.write_table(run, df)

def _figure_generic(ctx: BuildContext, run: AssetRun, hints: Sequence[str], mode: str='bar', tokens: str='') -> None:
    paths = _evidence_paths(ctx, hints, ('.csv', '.json', '.parquet', '.xlsx'))[:5]
    if not paths:
        raise EvidenceMissing(f'No chart evidence for {hints}')
    df = _combine_frames(ctx, run, paths, 'chart evidence')
    if tokens:
        df = _filter_metrics(df, tokens)
    nums = _numeric_cols(df)
    if not nums:
        raise EvidenceMissing('No numeric chart columns')
    objects = [c for c in df.columns if c not in nums and c != 'source_file']
    x = objects[0] if objects else 'source_file'
    y = nums[0]
    if mode == 'line':
        xn = next((c for c in nums if any((k in _norm(c) for k in ('budget', 'seed', 'year', 'l1')))), nums[0])
        yy = next((c for c in nums if c != xn), None)
        if yy is None:
            raise EvidenceMissing('Line chart needs two numeric columns')
        hue = objects[0] if objects else None
        _line(ctx, run, df, xn, yy, hue)
    elif mode == 'scatter':
        if len(nums) < 2:
            raise EvidenceMissing('Scatter needs two numeric columns')
        _scatter(ctx, run, df, nums[0], nums[1], objects[0] if objects else None)
    elif mode == 'heatmap':
        if len(objects) < 2:
            raise EvidenceMissing('Heatmap needs two categorical columns')
        _heatmap(ctx, run, df, objects[0], objects[1], y)
    else:
        hue = objects[1] if len(objects) > 1 and df[objects[1]].nunique(dropna=True) <= 8 else None
        _bar(ctx, run, df, x, y, hue, horizontal=df[x].nunique(dropna=True) > 7)

def _build_diagram(ctx: BuildContext, run: AssetRun, nodes: list[tuple[str, float, float, float, float]], edges: list[tuple[int, int, str]], footer: str) -> None:
    plt = ctx.matplotlib()
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    rows = []
    for i, (label, x, y, w, h) in enumerate(nodes):
        box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.015,rounding_size=0.02', facecolor=PALETTE[i % 6] + '22', edgecolor=PALETTE[i % 6], linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=10.5, weight='semibold', wrap=True)
        rows.append({'node_id': i, 'label': label, 'x': x, 'y': y, 'width': w, 'height': h})
    for a, b, label in edges:
        _, x1, y1, w1, h1 = nodes[a]
        _, x2, y2, w2, h2 = nodes[b]
        start = (x1 + w1 / 2, y1 + h1 / 2)
        end = (x2 + w2 / 2, y2 + h2 / 2)
        arr = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14, linewidth=1.2, color='#444444', shrinkA=32, shrinkB=32)
        ax.add_patch(arr)
        if label:
            ax.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.025, label, ha='center', va='center', fontsize=8.5, backgroundcolor='white')
    ax.set_title(run.spec.title, pad=12, weight='bold')
    ax.text(0.5, 0.02, footer, ha='center', va='bottom', fontsize=8.5, color='#444444')
    df = pd.DataFrame(rows)
    registry = ctx.config_dir / 'paper_display_registry.yaml'
    if registry.exists():
        ctx.record_input(run, registry, 'design/claim registry')
    ctx.save_figure(run, fig, df, svg=True)

def _boxplot(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, category: str, value: str, violin: bool=False) -> None:
    plt = ctx.matplotlib()
    data = df.copy()
    data[value] = pd.to_numeric(data[value], errors='coerce')
    data = data.dropna(subset=[value])
    groups = [(str(k), g[value].to_numpy()) for k, g in data.groupby(category, sort=False) if len(g)]
    if not groups:
        raise EvidenceMissing('No groups for distribution plot')
    fig, ax = plt.subplots(figsize=(max(7.5, 0.85 * len(groups)), 5.5))
    vals = [v for _, v in groups]
    if violin and all((len(v) >= 2 for v in vals)):
        parts = ax.violinplot(vals, showmeans=True, showextrema=True)
        for body in parts.get('bodies', []):
            body.set_facecolor(PALETTE[0])
            body.set_alpha(0.55)
    else:
        bp = ax.boxplot(vals, patch_artist=True, showmeans=True)
        for i, b in enumerate(bp['boxes']):
            b.set_facecolor(PALETTE[i % len(PALETTE)])
            b.set_alpha(0.55)
    ax.set_xticks(range(1, len(groups) + 1), [k for k, _ in groups], rotation=30, ha='right')
    ax.axhline(0, color='#444', linewidth=0.8)
    ax.set_ylabel(value)
    ax.set_title(run.spec.title)
    ax.grid(axis='y', alpha=0.22)
    ctx.save_figure(run, fig, data)

def _stacked_bar(ctx: BuildContext, run: AssetRun, df: pd.DataFrame, x: str, stack: str, value: str) -> None:
    plt = ctx.matplotlib()
    data = df.copy()
    data[value] = pd.to_numeric(data[value], errors='coerce')
    p = data.pivot_table(index=x, columns=stack, values=value, aggfunc='sum', fill_value=0)
    if p.empty:
        raise EvidenceMissing('No stacked-bar matrix')
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = pd.Series(0.0, index=p.index)
    for i, c in enumerate(p.columns):
        ax.bar(p.index.astype(str), p[c], bottom=bottom, label=str(c), color=PALETTE[i % len(PALETTE)])
        bottom = bottom + p[c]
    ax.set_title(run.spec.title)
    ax.set_ylabel(value)
    ax.tick_params(axis='x', rotation=30)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.2)
    ctx.save_figure(run, fig, data)

def _forest(ctx: BuildContext, run: AssetRun, df: pd.DataFrame) -> None:
    est = _find_col(df, ['estimate', 'coefficient', 'interaction_estimate', 'difference', 'mean_gap'], ['estimate', 'coefficient', 'interaction', 'difference', 'gap'])
    lo = _find_col(df, ['ci_lo', 'ci_lower', 'lower'], ['cilo', 'lower'])
    hi = _find_col(df, ['ci_hi', 'ci_upper', 'upper'], ['cihi', 'upper'])
    labels = [c for c in df.columns if c not in _numeric_cols(df) and c != 'source_file']
    if not est:
        raise EvidenceMissing('Forest plot estimate column not found')
    data = df.copy()
    data[est] = pd.to_numeric(data[est], errors='coerce')
    data = data.dropna(subset=[est]).head(40)
    if data.empty:
        raise EvidenceMissing('No estimates for forest plot')
    label = data[labels].astype(str).agg(' · '.join, axis=1) if labels else data.index.astype(str)
    y = list(range(len(data)))
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(9, max(4.8, 0.34 * len(data))))
    if lo and hi:
        l = pd.to_numeric(data[lo], errors='coerce')
        u = pd.to_numeric(data[hi], errors='coerce')
        xerr = [(data[est] - l).clip(lower=0), (u - data[est]).clip(lower=0)]
        ax.errorbar(data[est], y, xerr=xerr, fmt='o', color=PALETTE[0], ecolor='#666', capsize=3)
    else:
        ax.scatter(data[est], y, color=PALETTE[0])
    ax.axvline(0, color='#444', linewidth=0.8)
    ax.set_yticks(y, label)
    ax.set_xlabel(est)
    ax.set_title(run.spec.title)
    ax.grid(axis='x', alpha=0.2)
    ctx.save_figure(run, fig, data)

def _rank_bump(ctx: BuildContext, run: AssetRun) -> None:
    paths = _evidence_paths(ctx, ('n5', 'backend'), ('.csv', '.parquet'))[:20]
    if not paths:
        raise EvidenceMissing('No backend-policy result files')
    df = _combine_frames(ctx, run, paths, 'backend policy results')
    backend = _find_col(df, ['backend', 'model', 'oracle_backend'], ['backend', 'model'])
    policy = _find_col(df, ['policy', 'condition'], ['policy', 'condition'])
    score = _find_col(df, ['mean_delta_R_score_alpha', 'delta_noop', 'mean_score', 'estimate'], ['delta', 'score', 'estimate'])
    rank = _find_col(df, ['rank', 'policy_rank'], ['rank'])
    if not backend or not policy:
        raise EvidenceMissing('Backend/policy columns not found')
    data = df.copy()
    if rank:
        data['rank_value'] = pd.to_numeric(data[rank], errors='coerce')
    elif score:
        data[score] = pd.to_numeric(data[score], errors='coerce')
        data['rank_value'] = data.groupby(backend)[score].rank(method='dense', ascending=False)
    else:
        raise EvidenceMissing('Neither score nor rank column found')
    data = data.dropna(subset=['rank_value'])
    backs = list(dict.fromkeys(data[backend].astype(str)))
    pols = list(dict.fromkeys(data[policy].astype(str)))
    if not backs or not pols:
        raise EvidenceMissing('No bump-chart rows')
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(backs)), 5.8))
    for i, pname in enumerate(pols[:15]):
        sub = data[data[policy].astype(str).eq(pname)]
        vals = {str(a): b for a, b in zip(sub[backend], sub['rank_value'])}
        ys = [vals.get(b, math.nan) for b in backs]
        ax.plot(range(len(backs)), ys, marker='o', label=pname, color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(range(len(backs)), backs)
    ax.invert_yaxis()
    ax.set_ylabel('policy rank (1=best)')
    ax.set_title(run.spec.title)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')
    ctx.save_figure(run, fig, data)

def _paired_before_after(ctx: BuildContext, run: AssetRun) -> None:
    paths = _evidence_paths(ctx, ('llm_stage9_revision_metrics.csv',), ('.csv',))
    if not paths:
        raise EvidenceMissing('No revision-metrics files')
    df = _combine_frames(ctx, run, paths, 'revision metrics')
    before = _find_col(df, ['score_before', 'pre_score', 'initial_score', 'c4_score'], ['before', 'initial'])
    after = _find_col(df, ['score_after', 'post_score', 'revised_score', 'c6_score'], ['after', 'revised'])
    if not before or not after:
        raise EvidenceMissing('Before/after score columns not found')
    data = df.copy()
    data[before] = pd.to_numeric(data[before], errors='coerce')
    data[after] = pd.to_numeric(data[after], errors='coerce')
    data = data.dropna(subset=[before, after])
    if data.empty:
        raise EvidenceMissing('No complete before/after rows')
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 5.7))
    n = min(len(data), 300)
    for _, r in data.head(n).iterrows():
        ax.plot([0, 1], [r[before], r[after]], color='#777', alpha=0.16, linewidth=0.7)
    ax.scatter([0] * n, data[before].head(n), s=12, alpha=0.45, color=PALETTE[0])
    ax.scatter([1] * n, data[after].head(n), s=12, alpha=0.45, color=PALETTE[1])
    ax.set_xticks([0, 1], ['참조 전', '참조 후'])
    ax.set_ylabel('score')
    ax.set_title(run.spec.title)
    ax.grid(axis='y', alpha=0.2)
    ctx.save_figure(run, fig, data)

def _rating_bar(ctx: BuildContext, run: AssetRun) -> None:
    paths = _evidence_paths(ctx, ('rating', 'stage1'), ('.csv', '.parquet'))[:30] + _evidence_paths(ctx, ('rating', 'stage0'), ('.csv', '.parquet'))[:20]
    for p in paths:
        try:
            df = _as_frame(ctx.read(run, p, 'rating observations'))
        except (ImportError, ValueError, OSError) as exc:
            run.notes.append(f'Skipped unreadable {p}: {exc!r}')
            continue
        c = _find_col(df, ['actual_rating', 'rating', 'rating_grade', 'grade'], ['rating', 'grade'])
        if c:
            counts = df[c].astype(str).value_counts(dropna=False).rename_axis('rating').reset_index(name='count')
            _bar(ctx, run, counts, 'rating', 'count')
            return
    raise EvidenceMissing('No actual rating column found')

def _hist_grid(ctx: BuildContext, run: AssetRun) -> None:
    paths = _evidence_paths(ctx, ('stage0', 'stage2'), ('.csv', '.parquet'))[:25]
    for p in paths:
        try:
            df = _as_frame(ctx.read(run, p, 'numeric sample panel'))
        except (ImportError, ValueError, OSError) as exc:
            run.notes.append(f'Skipped unreadable {p}: {exc!r}')
            continue
        cols = [c for c in _numeric_cols(df, 50) if not any((k in _norm(c) for k in ('id', 'year', 'code')))][:12]
        if len(cols) >= 2:
            plt = ctx.matplotlib()
            n = len(cols)
            ncols = 3
            nrows = math.ceil(n / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.2 * nrows))
            axes = list(getattr(axes, 'flat', [axes]))
            for i, c in enumerate(cols):
                axes[i].hist(pd.to_numeric(df[c], errors='coerce').dropna(), bins=30, color=PALETTE[i % len(PALETTE)], alpha=0.78)
                axes[i].set_title(c)
                axes[i].grid(axis='y', alpha=0.18)
            for ax in axes[n:]:
                ax.axis('off')
            fig.suptitle(run.spec.title)
            ctx.save_figure(run, fig, df[cols].copy())
            return
    raise EvidenceMissing('No sufficiently populated numeric panel for histograms')

def _permutation_hist(ctx: BuildContext, run: AssetRun) -> None:
    paths = _evidence_paths(ctx, ('signflip', 'permutation'), ('.csv', '.parquet'))[:20]
    if not paths:
        raise EvidenceMissing('No sign-flip permutation draws')
    df = _combine_frames(ctx, run, paths, 'sign-flip permutation draws')
    nums = _numeric_cols(df)
    val = next((c for c in nums if any((k in _norm(c) for k in ('stat', 'permutation', 'null', 'gap')))), nums[0] if nums else None)
    if not val:
        raise EvidenceMissing('No permutation statistic column')
    data = df.copy()
    values = pd.to_numeric(data[val], errors='coerce').dropna()
    if len(values) < 3:
        raise EvidenceMissing('Permutation distribution has fewer than three draws')
    obs_col = _find_col(data, ['observed', 'observed_statistic'], ['observed'])
    observed = float(pd.to_numeric(data[obs_col], errors='coerce').dropna().iloc[0]) if obs_col and pd.to_numeric(data[obs_col], errors='coerce').notna().any() else None
    plt = ctx.matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5.3))
    ax.hist(values, bins=min(40, max(10, int(math.sqrt(len(values))))), color=PALETTE[0], alpha=0.75)
    if observed is not None:
        ax.axvline(observed, color=PALETTE[3], linewidth=2, label=f'observed={observed:.3g}')
        ax.legend(frameon=False)
    ax.set_title(run.spec.title)
    ax.set_xlabel(val)
    ax.set_ylabel('count')
    ax.grid(axis='y', alpha=0.2)
    ctx.save_figure(run, fig, data)

def _handle_diagram(ctx: BuildContext, run: AssetRun) -> None:
    h = run.spec.handler
    if h == 'diagram_pipeline':
        nodes = [('1. 평가기반 검증\nSimulator + Oracle α·β·γ', 0.04, 0.58, 0.24, 0.2), ('2. 독립 참조정책\nOracle-blind Candidate-IQL', 0.38, 0.58, 0.24, 0.2), ('3. LLM 정책시스템\n출력·정보·예산·수정·참조', 0.72, 0.58, 0.24, 0.2), ('동결 입력·계약', 0.06, 0.2, 0.2, 0.14), ('기업별 반사실 평가', 0.4, 0.2, 0.2, 0.14), ('claim–evidence 감사', 0.74, 0.2, 0.2, 0.14)]
        edges = [(0, 1, '검증 후 비교'), (1, 2, '독립 참조'), (3, 0, '입력'), (4, 1, '공통 평가'), (2, 5, '산출')]
        footer = '정책효과를 주장하기 전에 평가자 정렬성과 반사실 장치의 속성을 먼저 검증한다.'
    elif h == 'diagram_attribution':
        nodes = [('평가자 귀속 오류\n점수도구가 현실과 맞는가?', 0.06, 0.62, 0.25, 0.2), ('정책 귀속 오류\n모델인가, 출력계약인가?', 0.375, 0.62, 0.25, 0.2), ('수정 귀속 오류\n재검토인가, 참조내용인가?', 0.69, 0.62, 0.25, 0.2), ('Validation-first', 0.13, 0.24, 0.2, 0.14), ('Policy-system audit', 0.4, 0.24, 0.2, 0.14), ('C4→C4R→C6 분해', 0.67, 0.24, 0.2, 0.14)]
        edges = [(0, 3, '통제'), (1, 4, '통제'), (2, 5, '통제')]
        footer = '같은 평균점수도 서로 다른 생성·수정·평가 경로에서 발생할 수 있다.'
    elif h == 'diagram_experiment':
        nodes = [('C3\nCandidate-IQL', 0.05, 0.68, 0.18, 0.15), ('C4\n직접 자유작성', 0.28, 0.68, 0.18, 0.15), ('C4R\n참조 없는 재검토', 0.52, 0.68, 0.18, 0.15), ('C6\nCandidate-IQL 참조 후 수정', 0.76, 0.68, 0.19, 0.15), ('C5\n추론 유도', 0.28, 0.34, 0.18, 0.15), ('C6X\n무작위 참조', 0.52, 0.34, 0.18, 0.15), ('C8\n참조시점 변경', 0.76, 0.34, 0.19, 0.15)]
        edges = [(1, 2, '2차 검토'), (2, 3, '참조내용'), (0, 3, '참조'), (1, 4, '추론'), (1, 5, '대조 참조'), (3, 6, '시점')]
        footer = '각 조건은 모델명이 아니라 배포 정책 시스템의 한 구성으로 취급한다.'
    elif h == 'diagram_c4r':
        nodes = [('C4\n직접 생성', 0.08, 0.55, 0.22, 0.22), ('C4R\n외부참조 없는 재검토', 0.39, 0.55, 0.22, 0.22), ('C6\nCandidate-IQL 참조 후 수정', 0.7, 0.55, 0.22, 0.22), ('자기 재검토 효과\nC4R−C4', 0.23, 0.2, 0.22, 0.14), ('참조내용 증분\nC6−C4R', 0.55, 0.2, 0.22, 0.14), ('전체 패키지\nC6−C4', 0.76, 0.2, 0.18, 0.14)]
        edges = [(0, 1, 'same draft'), (1, 2, 'reference'), (0, 3, ''), (1, 4, ''), (2, 5, '')]
        footer = '전체 효과 = 자기 재검토 성분 + 조건부 참조내용 성분.'
    else:
        nodes = [('전제 검증\n표본·분할·계약·동결', 0.05, 0.58, 0.24, 0.2), ('검사 1\nOracle 외부 정렬성', 0.38, 0.58, 0.24, 0.2), ('검사 2\nSimulator 구조·행동 속성', 0.71, 0.58, 0.24, 0.2), ('통과/경계 기록', 0.39, 0.22, 0.22, 0.14), ('정책 상대비교', 0.71, 0.22, 0.22, 0.14)]
        edges = [(0, 1, ''), (1, 2, ''), (1, 3, 'gate'), (2, 4, 'scope')]
        footer = '검증 실패를 숨기지 않고 사용 가능한 주장 범위를 함께 기록한다.'
    _build_diagram(ctx, run, nodes, edges, footer)

def _handler(ctx: BuildContext, run: AssetRun) -> None:
    s = run.spec
    h = s.handler
    run.transformations.append(f'handler={h}; all values derived from recorded inputs')
    if build_specialized_asset(ctx, run):
        return
    if h.startswith('diagram_'):
        return _handle_diagram(ctx, run)
    if h == 'oracle_validation':
        df = _oracle_validation(ctx, run)
        token = 'oot' if s.asset_id == 'A1' else 'dev|oot'
        ctx.write_table(run, _filter_metrics(df, token))
    elif h == 'oracle_validation_chart':
        df = _oracle_validation(ctx, run)
        if set(df.columns) >= {'metric', 'value'}:
            d = _filter_metrics(df, 'direction|agreement|accuracy').copy()
            d['oracle'] = d['metric'].astype(str).str.extract('(?i)(alpha|beta|gamma)', expand=False).fillna('overall')
            d['split'] = d['metric'].astype(str).str.extract('(?i)(dev|oot)', expand=False).fillna('unspecified')
            _bar(ctx, run, d, 'oracle', 'value', 'split')
        else:
            _metric_value_chart(ctx, run, df, ['oracle', 'backend'], ['directional_agreement', 'direction_accuracy', 'accuracy'], ['split', 'period'])
    elif h == 'oracle_scatter':
        paths = _evidence_paths(ctx, ('oracle',), ('.csv', '.parquet'))[:20] + _evidence_paths(ctx, ('rating',), ('.csv', '.parquet'))[:20]
        candidates = []
        for p in paths:
            try:
                obj = ctx.read(run, p, 'Oracle/rating observation')
            except (ImportError, ValueError, OSError) as exc:
                run.notes.append(f'Skipped unreadable optional source {p}: {exc!r}')
                continue
            df = _as_frame(obj)
            nums = _numeric_cols(df)
            score = _find_col(df, ['oracle_alpha', 'score_alpha', 'predicted_score'], ['alpha', 'score'])
            rating = _find_col(df, ['rating', 'actual_rating', 'rating_num'], ['rating'])
            if score and rating:
                candidates.append(df[[score, rating]].rename(columns={score: 'Oracle-α score', rating: 'actual rating'}))
        if not candidates:
            raise EvidenceMissing('Could not locate paired Oracle-α score and actual rating columns')
        _scatter(ctx, run, pd.concat(candidates, ignore_index=True), 'Oracle-α score', 'actual rating')
    elif h in {'b1_b2', 'test3'}:
        tokens = 'b1|b2|gap|direction' if h == 'b1_b2' else 'test3|identity|error|preserv|stability'
        _tabular_generic(ctx, run, s.source_hints, tokens)
    elif h == 'b1_b2_chart':
        _figure_generic(ctx, run, s.source_hints, 'bar', 'b1|b2|direction|agreement')
    elif h == 'structural':
        _figure_generic(ctx, run, s.source_hints, 'bar', 'structural|event|direction|agreement')
    elif h == 'candidate_library':
        p = ctx.index.first('final_candidate_library.yaml', suffixes=('.yaml',))
        obj = ctx.read(run, p, 'candidate contract')
        ctx.write_table(run, _candidate_frame(obj))
    elif h == 'action_contract':
        p = ctx.index.first('final_action_contract.yaml', suffixes=('.yaml',))
        obj = ctx.read(run, p, 'action contract')
        ctx.write_table(run, _action_contract_frame(obj))
    elif h == 'candidate_heatmap':
        p = ctx.index.first('final_candidate_library.yaml', suffixes=('.yaml',))
        obj = ctx.read(run, p, 'candidate contract')
        df = _candidate_frame(obj)
        actions = [c for c in df if c.startswith('action__')]
        long = df.melt(id_vars=['candidate'], value_vars=actions, var_name='action_dimension', value_name='value')
        _heatmap(ctx, run, long, 'candidate', 'action_dimension', 'value')
    elif h == 'candidate_compare':
        p1 = ctx.index.first('final_candidate_library.yaml', suffixes=('.yaml',))
        p2 = ctx.index.first('final_candidate_library_calibrated.yaml', suffixes=('.yaml',))
        a = _candidate_frame(ctx.read(run, p1, 'raw candidate contract'))
        b = _candidate_frame(ctx.read(run, p2, 'calibrated candidate contract'))
        keys = [c for c in a if c.startswith('action__')]
        aa = a[['candidate', *keys]].melt('candidate', var_name='dimension', value_name='raw')
        bb = b.reindex(columns=['candidate', *keys], fill_value=0).melt('candidate', var_name='dimension', value_name='calibrated')
        out = aa.merge(bb, on=['candidate', 'dimension'], how='outer').fillna(0)
        out['difference'] = pd.to_numeric(out['calibrated'], errors='coerce') - pd.to_numeric(out['raw'], errors='coerce')
        ctx.write_table(run, out)
    elif h == 'rl_policy':
        _tabular_generic(ctx, run, s.source_hints, 'C0|C1|C2|C3|C_obs|C_fix|delta|noop|alpha|beta|gamma')
    elif h in {'rl_policy_alpha_chart', 'rl_policy_oracle_chart'}:
        df = _first_frame(ctx, run, s.source_hints)
        x = _find_col(df, ['policy', 'condition'], ['policy'])
        y = _find_col(df, ['mean_delta_R_score_alpha', 'delta_noop_alpha'], ['delta', 'alpha'])
        hue = None
        if h == 'rl_policy_oracle_chart':
            vals = [c for c in df if 'delta' in _norm(c) and any((o in _norm(c) for o in ('alpha', 'beta', 'gamma')))]
            if x and vals:
                df = df.melt(id_vars=[x], value_vars=vals, var_name='oracle', value_name='delta_noop')
                y = 'delta_noop'
                hue = 'oracle'
        if not x or not y:
            return _figure_generic(ctx, run, s.source_hints, 'bar', 'policy|delta|noop')
        _bar(ctx, run, df, x, y, hue, horizontal=True)
    elif h == 'seven_seed':
        _tabular_generic(ctx, run, s.source_hints, 'seed|C3|delta|action|std')
    elif h == 'seven_seed_chart':
        _figure_generic(ctx, run, s.source_hints, 'bar', 'seed|delta|alpha|beta|gamma')
    elif h in {'action_distribution', 'flatness', 'capture', 'pareto'}:
        _tabular_generic(ctx, run, s.source_hints, 'action|tie|variance|capture|optimal') if s.kind == 'table' else _figure_generic(ctx, run, s.source_hints, 'scatter' if h == 'pareto' else 'bar', 'action|share|q|diversity')
    elif h in {'n5', 'winrate', 'heterogeneity', 'failure', 'budget', 'c4r', 'did', 'c6x', 'decomposition', 'tost', 'icc', 'revision', 'reference_quality', 'signflip', 'shuffle', 'revision_dilution', 'relational', 'variables', 'descriptive', 'industry', 'holm'}:
        tokens = {'n5': 'condition|policy|IC-|delta|p_', 'winrate': 'win|tie|loss|rate', 'heterogeneity': 'asset|quintile|heterogeneity|effect', 'failure': 'failure|error|audit|count', 'budget': 'budget|L1|delta|condition', 'c4r': 'C4|C4R|C6|self|reference|package', 'did': 'interaction|DID|coefficient|p_', 'c6x': 'C6X|C6|reference|adoption', 'decomposition': 'harness|backend|variance|R2', 'tost': 'TOST|equivalence|margin|p_', 'icc': 'probe|channel|recognition|recall', 'revision': 'adoption|retention|revision', 'reference_quality': 'reference|quality|accept', 'signflip': 'sign|permutation|null|p_', 'shuffle': 'shuffle|null|industry|rating', 'revision_dilution': 'dilution|active|L1|axis', 'relational': 'relational|reference|advantage|quartile', 'variables': 'variable|weight|direction|category', 'descriptive': 'mean|std|min|max|quantile', 'industry': 'industry|count|firm', 'holm': 'holm|p_value|comparison'}[h]
        _tabular_generic(ctx, run, s.source_hints, tokens)
    elif h == 'rank_chart':
        _rank_bump(ctx, run)
    elif h == 'paired_revision':
        _paired_before_after(ctx, run)
    elif h == 'rating_distribution':
        _rating_bar(ctx, run)
    elif h == 'histograms':
        _hist_grid(ctx, run)
    elif h == 'permutation':
        _permutation_hist(ctx, run)
    elif h == 'did_forest':
        paths = _evidence_paths(ctx, s.source_hints, ('.csv', '.json', '.parquet'))[:12]
        if not paths:
            raise EvidenceMissing('No DID inference source')
        _forest(ctx, run, _combine_frames(ctx, run, paths, 'DID inference'))
    elif h in {'policy_ladder', 'n5_ic_chart', 'winrate_chart', 'heterogeneity_chart', 'failure_chart', 'budget_curve', 'budget_heatmap', 'clipping', 'c4r_chart', 'c4r_interaction', 'decomposition_chart', 'icc_chart', 'ic_distribution', 'revision_scatter', 'encoder_sweep', 'alpha_weights'}:
        mode = 'bar'
        if h in {'budget_curve', 'c4r_interaction', 'rank_chart', 'encoder_sweep'}:
            mode = 'line'
        if h in {'budget_heatmap'}:
            mode = 'heatmap'
        if h in {'revision_scatter', 'paired_revision'}:
            mode = 'scatter'
        tokens = {'policy_ladder': 'policy|condition|delta|noop', 'n5_ic_chart': 'IC-|condition|delta|noop', 'winrate_chart': 'win|tie|loss|rate', 'heterogeneity_chart': 'asset|quintile|effect|delta', 'failure_chart': 'failure|error|count|IC-', 'budget_curve': 'budget|L1|delta|alpha', 'budget_heatmap': 'budget|condition|delta|effect', 'clipping': 'clip|budget|rate', 'c4r_chart': 'C4|C4R|C6|self|reference|package', 'c4r_interaction': 'budget|component|self|reference|package', 'did_forest': 'DID|interaction|coefficient|estimate', 'decomposition_chart': 'harness|backend|variance|share', 'rank_chart': 'backend|policy|rank', 'icc_chart': 'probe|channel|recognition|rate', 'ic_distribution': 'IC-|delta|noop', 'revision_scatter': 'adoption|retention', 'paired_revision': 'before|after|score|delta', 'permutation': 'permutation|null|observed|statistic', 'encoder_sweep': 'encoder|performance|score|seed', 'alpha_weights': 'weight|alpha|variable', 'histograms': 'ratio|asset|debt|profit|liquid', 'rating_distribution': 'rating|grade|count'}[h]
        _figure_generic(ctx, run, s.source_hints, mode, tokens)
    elif h in {'budget_action', 'ic_actions'}:
        paths = _evidence_paths(ctx, ('llm_stage7_actions',), ('.parquet', '.csv'))[:30]
        if not paths:
            raise EvidenceMissing('No Stage7 action artifacts')
        df = _combine_frames(ctx, run, paths, 'Stage7 actions')
        dims = [c for c in df if c.startswith('action__')]
        group = _find_col(df, ['budget', 'information_condition', 'condition'], ['budget'] if h == 'budget_action' else ['informationcondition', 'ic'])
        if dims and group:
            out = df.groupby(group)[dims].agg(['mean', lambda x: (pd.to_numeric(x, errors='coerce').abs() > 1e-12).mean()])
            out.columns = [f"{a}__{(b if isinstance(b, str) else 'active_rate')}" for a, b in out.columns]
            out = out.reset_index()
        else:
            out = df
        ctx.write_table(run, out)
    elif h == 'run_inventory':
        root = ctx.final_freeze / 'llm_runs'
        if not root.exists():
            raise EvidenceMissing(f'Missing {root}')
        rows = []
        for d in sorted([x for x in root.iterdir() if x.is_dir()]):
            files = list(d.rglob('*'))
            rows.append({'run_label': d.name, 'backend_hint': next((x for x in d.name.split('_') if any((k in x.lower() for k in ('gpt', 'gemini', 'haiku', 'claude')))), ''), 'information_condition': next((x for x in ('IC-a', 'IC-b', 'IC-c') if x.lower().replace('-', '') in d.name.lower().replace('-', '')), ''), 'file_count': sum((p.is_file() for p in files)), 'total_bytes': sum((p.stat().st_size for p in files if p.is_file())), 'path': _relative(d, ctx.project_root)})
        ctx.write_table(run, pd.DataFrame(rows))
    elif h == 'artifact_chain':
        p = ctx.index.first('thesis_artifact_registry.json', suffixes=('.json',))
        obj = ctx.read(run, p, 'thesis artifact registry')
        rows = []
        for a in obj.get('artifacts', []):
            if not isinstance(a, dict):
                continue
            row = {k: v for k, v in a.items() if not isinstance(v, (dict, list))}
            row['source_paths'] = json.dumps(a.get('source_paths') or a.get('sources') or [], ensure_ascii=False)
            rows.append(row)
        if not rows:
            rows = _flatten_json(obj)
        ctx.write_table(run, pd.DataFrame(rows))
    else:
        raise EvidenceMissing(f'No implemented handler for {h}')

def _make_asset_function(spec: AssetSpec) -> Callable[[BuildContext], AssetRun]:

    def build(ctx: BuildContext) -> AssetRun:
        run = AssetRun(spec)
        try:
            _handler(ctx, run)
            run.status = 'PASS'
        except (EvidenceMissing, SpecializedEvidenceMissing) as exc:
            run.status = 'SKIPPED_MISSING_EVIDENCE'
            run.error = str(exc)
        except Exception as exc:
            run.status = 'FAIL'
            run.error = repr(exc)
            run.notes.append(traceback.format_exc())
        return run
    build.__name__ = f'build_{spec.asset_id}'
    build.__doc__ = f'Build {spec.asset_id}: {spec.title}.'
    return build
for _s in SPECS:
    globals()[f'build_{_s.asset_id}'] = _make_asset_function(_s)

def _run_dict(run: AssetRun, ctx: BuildContext) -> dict[str, Any]:
    claim_evidence = ctx.claim_map[run.spec.asset_id]
    return {'asset_id': run.spec.asset_id, 'priority': run.spec.priority, 'category': run.spec.category, 'kind': run.spec.kind, 'title': run.spec.title, 'handler': run.spec.handler, 'status': run.status, 'started_utc': run.started_utc, 'finished_utc': _now(), 'claim_evidence': claim_evidence, 'source_hints': list(run.spec.source_hints), 'inputs_used': run.inputs, 'transformations': run.transformations, 'outputs': run.outputs, 'notes': run.notes, 'error': run.error, 'project_root': str(ctx.project_root)}

def _write_catalog(ctx: BuildContext, records: list[dict[str, Any]], stage: str) -> dict[str, Any]:
    for rec in records:
        _write_json(ctx.manifest_dir / f"{rec['asset_id']}.json", rec)
    inventory = []
    lineage = []
    mapping = {}
    source_paths = {x['role']: x['relative_path'] for x in ctx.registry_sources}
    for r in records:
        mapping[r['asset_id']] = {**r['claim_evidence'], 'title': r['title'], 'kind': r['kind'], 'priority': r['priority'], 'status': r['status'], 'manifest': _relative(ctx.manifest_dir / f"{r['asset_id']}.json", ctx.project_root), 'outputs': [o['relative_path'] for o in r['outputs']], 'mapping_source': source_paths['visual_claim_map'], 'slot_registry_source': source_paths['paper_display_registry'], 'claim_registry_source': source_paths['claim_evidence_registry'], 'canonical_registry_validation': 'PASS'}
        inventory.append({'asset_id': r['asset_id'], 'priority': r['priority'], 'category': r['category'], 'kind': r['kind'], 'title': r['title'], 'status': r['status'], 'slot': r['claim_evidence']['slot'], 'claims': ';'.join(r['claim_evidence']['claims']), 'section': r['claim_evidence']['section'], 'n_inputs': len(r['inputs_used']), 'n_outputs': len(r['outputs']), 'input_paths': ';'.join((x['relative_path'] for x in r['inputs_used'])), 'output_paths': ';'.join((x['relative_path'] for x in r['outputs'])), 'skip_or_error': r['error'] or ''})
        for x in r['inputs_used']:
            lineage.append({'asset_id': r['asset_id'], 'priority': r['priority'], 'role': x['role'], 'input_path': x['relative_path'], 'reader': x['reader'], 'size_bytes': x['size_bytes']})
    inv = pd.DataFrame(inventory)
    lin = pd.DataFrame(lineage, columns=['asset_id', 'priority', 'role', 'input_path', 'reader', 'size_bytes'])
    for name, df in [('visual_assets_inventory', inv), ('visual_input_lineage', lin)]:
        df.to_csv(ctx.registry_dir / f'{name}.csv', index=False, encoding='utf-8-sig')
        (ctx.registry_dir / f'{name}.md').write_text(_markdown(df), encoding='utf-8')
    _write_json(ctx.registry_dir / 'visual_claim_evidence_map.json', mapping)
    mapdf = pd.DataFrame([{'asset_id': k, **{x: ';'.join(v[x]) if isinstance(v.get(x), list) else v.get(x) for x in ('slot', 'claims', 'section', 'title', 'kind', 'priority', 'status', 'manifest', 'mapping_source', 'slot_registry_source', 'claim_registry_source', 'canonical_registry_validation')}, 'outputs': ';'.join(v['outputs'])} for k, v in mapping.items()])
    mapdf.to_csv(ctx.registry_dir / 'visual_claim_evidence_map.csv', index=False, encoding='utf-8-sig')
    (ctx.registry_dir / 'visual_claim_evidence_map.md').write_text(_markdown(mapdf), encoding='utf-8')
    counts = pd.Series([r['status'] for r in records]).value_counts().to_dict()
    manifest = {'schema_version': SCHEMA, 'created_utc': _now(), 'status': 'FAIL' if counts.get('FAIL', 0) else 'PASS_WITH_SKIPS' if counts.get('SKIPPED_MISSING_EVIDENCE', 0) else 'PASS', 'completion_stage': stage, 'registered_asset_count': len(SPECS), 'attempted_asset_count': len(records), 'status_counts': counts, 'project_root': str(ctx.project_root), 'analysis_dir': str(ctx.analysis_dir), 'final_freeze_dir': str(ctx.final_freeze), 'tables_dir': str(ctx.tables_dir), 'figures_dir': str(ctx.figures_dir), 'plot_data_dir': str(ctx.snapshot_dir), 'registry_dir': str(ctx.registry_dir), 'asset_manifest_dir': str(ctx.manifest_dir), 'layer_contract': {'01_execution_and_freeze': {'root': _relative(ctx.final_freeze, ctx.project_root), 'contents': 'Oracle, RL, and supplied/frozen LLM execution artifacts', 'builder_access': 'READ_ONLY'}, '02_human_paper_assets': {'root': _relative(ctx.assets_root, ctx.project_root), 'contents': ['tables', 'figures', 'plot_data'], 'claim_mapping_allowed': False}, '03_claim_evidence_registry': {'root': _relative(ctx.registry_dir, ctx.project_root), 'contents': ['inventory', 'input_lineage', 'claim_evidence_map', 'asset_manifests', 'registry_manifest'], 'paper_binary_assets_allowed': False}}, 'canonical_registry_sources': ctx.registry_sources, 'canonical_registry_validation': ctx.registry_validation, 'legacy_layout_cleanup': ctx.legacy_layout_cleanup, 'read_only_contract': {'final_freeze_mutation_allowed': False}, 'figure_contract': {'formats': ['PNG', 'PDF'], 'diagram_extra_format': 'SVG', 'dpi': DPI, 'palette': 'Okabe-Ito colorblind-safe', 'korean_font_resolution': 'Malgun Gothic → NanumGothic → Noto Sans CJK KR → system fallback'}, 'records': records}
    _write_json(ctx.registry_dir / 'visual_assets_manifest.json', manifest)
    _write_json(ctx.registry_dir / 'visual_asset_registry.json', {'schema_version': 'thesis_visual_asset_registry_v2', 'canonical_registry_sources': ctx.registry_sources, 'assets': [{'asset_id': spec.asset_id, 'priority': spec.priority, 'category': spec.category, 'kind': spec.kind, 'title': spec.title, 'handler': spec.handler, 'source_hints': list(spec.source_hints), 'claim_evidence': ctx.claim_map[spec.asset_id]} for spec in SPECS]})
    registry_readme = '# Thesis visual claim–evidence registry\n\n이 디렉터리는 표·그림 파일 자체가 아니라 등록 자산의 위치, 실제 입력 경로, 변환 기록, 논문 slot과 claim 연결을 관리한다.\n\n- 실행·동결 산출물: `data/final_freeze/`\n- 사람이 사용할 표·그림·plot data: `data/analysis/paper_repro/04_paper_assets/`\n- 이 claim–evidence/lineage registry: `data/analysis/paper_repro/06_thesis_registry/visual_assets/`\n\n`visual_assets_inventory.*`는 전체 산출물 위치를, `visual_input_lineage.*`는 실제 입력 경로·reader·파일 크기를, `visual_claim_evidence_map.*`는 canonical slot·claim 연결을 보여준다. `asset_manifests/<ID>.json`에는 개별 변환·입력·출력·누락 사유가 기록된다.\n'
    (ctx.registry_dir / 'README.md').write_text(registry_readme, encoding='utf-8')
    asset_readme = '# Human paper assets\n\n이 디렉터리에는 논문 작성자가 직접 사용할 가공 산출물만 둔다.\n\n- `tables/`: CSV와 Markdown 표\n- `figures/`: PNG/PDF 및 설계도 SVG\n- `plot_data/`: 각 그림에 실제 사용한 가공 데이터 CSV\n\n입력 경로·claim 연결·개별 manifest는 같은 paper-repro root의 `06_thesis_registry/visual_assets/`에서 별도로 관리된다.\n'
    (ctx.assets_root / 'VISUAL_ASSET_LAYOUT.md').write_text(asset_readme, encoding='utf-8')
    return manifest

def run_all_visuals(project_root: Path | None, analysis_dir: Path, figures_dir: Path | None=None, tables_dir: Path | None=None, plot_data_dir: Path | None=None, registry_dir: Path | None=None, *, strict: bool=False, only_ids: Sequence[str] | None=None, priorities: Sequence[str]=('P0', 'P1', 'P2')) -> dict[str, Any]:
    analysis_dir = Path(analysis_dir).resolve()
    if project_root is None:
        candidates = [analysis_dir, *analysis_dir.parents]
        project_root = next((p for p in candidates if (p / 'src' / 'credit_recourse').exists() or (p / 'data' / 'final_freeze').exists()), analysis_dir)
    project_root = Path(project_root).resolve()
    layout = build_layout(analysis_dir)
    ensure_layout(layout)
    figures_dir = Path(figures_dir).resolve() if figures_dir else layout.figures
    tables_dir = Path(tables_dir).resolve() if tables_dir else layout.tables
    plot_data_dir = Path(plot_data_dir).resolve() if plot_data_dir else layout.visual_plot_data
    registry_dir = Path(registry_dir).resolve() if registry_dir else layout.visual_registry
    if not all((p.parent == layout.paper_assets for p in (figures_dir, tables_dir, plot_data_dir))):
        raise ValueError('Figures, tables, and plot data must be direct children of canonical 04_paper_assets')
    if registry_dir.parent != layout.registry:
        raise ValueError('Visual lineage and claim mapping must be a direct child of canonical 06_thesis_registry')
    if registry_dir == layout.paper_assets or layout.paper_assets in registry_dir.parents:
        raise ValueError('Claim-evidence registry must not be nested inside 04_paper_assets')
    ctx = BuildContext(project_root, analysis_dir, figures_dir, tables_dir, plot_data_dir, registry_dir, strict)
    selected = {x.upper() for x in only_ids} if only_ids else {s.asset_id for s in SPECS}
    bad = selected - {s.asset_id for s in SPECS}
    if bad:
        raise ValueError(f'Unknown visual asset IDs: {sorted(bad)}')
    records = []
    for priority in priorities:
        for spec in SPECS:
            if spec.priority != priority or spec.asset_id not in selected:
                continue
            run = globals()[f'build_{spec.asset_id}'](ctx)
            rec = _run_dict(run, ctx)
            records.append(rec)
            _write_json(ctx.manifest_dir / f'{spec.asset_id}.json', rec)
        _write_catalog(ctx, records, priority)
    manifest = _write_catalog(ctx, records, 'COMPLETE')
    if strict and manifest['status_counts'].get('FAIL', 0):
        failed = [r['asset_id'] for r in records if r['status'] == 'FAIL']
        raise RuntimeError(f'Thesis visual builder internal failures: {failed}')
    return manifest

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='Generate maximum human-readable thesis tables and figures with input provenance.')
    ap.add_argument('--project-root', default='.')
    ap.add_argument('--output-dir', dest='analysis_dir', default='data/analysis/paper_repro', help='Canonical paper-repro analysis root')
    ap.add_argument('--figures-dir', default=None)
    ap.add_argument('--tables-dir', default=None)
    ap.add_argument('--plot-data-dir', default=None)
    ap.add_argument('--registry-dir', default=None)
    ap.add_argument('--priority', action='append', choices=['P0', 'P1', 'P2'], help='Run selected priority group(s)')
    ap.add_argument('--only', nargs='*', default=None, help='Optional asset IDs, e.g. --only A1 B1 K1')
    ap.add_argument('--strict', action='store_true', help='Fail only on builder defects; missing optional evidence remains a recorded skip')
    return ap

def main(argv: list[str] | None=None) -> int:
    args = build_arg_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    analysis = Path(args.analysis_dir)
    analysis = analysis if analysis.is_absolute() else root / analysis

    def resolve_optional(value: str | None) -> Path | None:
        if not value:
            return None
        path = Path(value)
        return path if path.is_absolute() else root / path
    manifest = run_all_visuals(root, analysis, figures_dir=resolve_optional(args.figures_dir), tables_dir=resolve_optional(args.tables_dir), plot_data_dir=resolve_optional(args.plot_data_dir), registry_dir=resolve_optional(args.registry_dir), strict=args.strict, only_ids=args.only, priorities=tuple(args.priority or ('P0', 'P1', 'P2')))
    print(json.dumps({k: manifest[k] for k in ('schema_version', 'status', 'registered_asset_count', 'attempted_asset_count', 'status_counts', 'plot_data_dir', 'registry_dir')}, ensure_ascii=False, indent=2))
    return 1 if manifest['status'] == 'FAIL' else 0
if __name__ == '__main__':
    raise SystemExit(main())
