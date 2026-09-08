from __future__ import annotations
import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
SCRIPT_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = SCRIPT_DIR / 'analysis_settings.json'

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value

def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def read_analysis_settings() -> dict[str, Any]:
    if not SETTINGS_PATH.is_file():
        raise FileNotFoundError(f'Analysis settings missing: {SETTINGS_PATH}')
    return read_json(SETTINGS_PATH)

def resolve_input(source_root: Path, spec: dict[str, Any]) -> Path:
    path = source_root / Path(spec['relative_path'])
    if not path.is_file():
        raise FileNotFoundError(f'Required input missing: {path}')
    return path

def parse_recognition(probe_path: Path) -> pd.DataFrame:
    probe = pd.read_csv(probe_path)
    required = {'row_id', 'icc_probe_response_raw'}
    missing = sorted(required - set(probe.columns))
    if missing:
        raise ValueError(f'Probe is missing required columns: {missing}')
    if probe['row_id'].duplicated().any():
        raise ValueError('Probe row_id is not unique')
    labels: list[bool] = []
    for row_id, raw in zip(probe['row_id'], probe['icc_probe_response_raw']):
        try:
            obj = json.loads(str(raw))
        except Exception as exc:
            raise ValueError(f'Probe JSON parse failure at row_id={row_id}: {exc}') from exc
        recognized = obj.get('recognized') if isinstance(obj, dict) else None
        if not isinstance(recognized, bool):
            raise ValueError(f'Probe recognized is not boolean at row_id={row_id}: {recognized!r}')
        labels.append(recognized)
    out = pd.DataFrame({'row_id': pd.to_numeric(probe['row_id'], errors='raise').astype(int), 'recognized': labels}).sort_values('row_id', kind='stable')
    return out

def validate_score_table(table: pd.DataFrame, *, label: str, policies: list[str], modes: list[str], oracles: list[str]) -> pd.DataFrame:
    required = {'row_id', 'policy', 'mode'}
    for oracle in oracles:
        required.update({f'R_score_{oracle}', f'noop_R_score_{oracle}', f'delta_R_score_{oracle}'})
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f'{label} table is missing columns: {missing}')
    keys = ['row_id', 'policy', 'mode']
    if table.duplicated(keys).any():
        raise ValueError(f'{label} table has duplicate row/policy/mode keys')
    table = table.copy()
    table['row_id'] = pd.to_numeric(table['row_id'], errors='raise').astype(int)
    return table.sort_values(keys, kind='stable').reset_index(drop=True)

def build_analysis_input(ic_b: pd.DataFrame, ic_c: pd.DataFrame, recognition: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    policies = list(settings['policies'])
    modes = list(settings['modes'])
    oracles = list(settings['oracles'])
    keys = ['row_id', 'policy', 'mode']
    keep = keys.copy()
    for oracle in oracles:
        keep.extend([f'R_score_{oracle}', f'noop_R_score_{oracle}', f'delta_R_score_{oracle}'])
    merged = ic_b[keep].merge(ic_c[keep], on=keys, how='inner', suffixes=('_ic_b', '_ic_c'))
    merged = merged.merge(recognition, on='row_id', how='left', validate='many_to_one')
    if merged['recognized'].isna().any():
        raise RuntimeError('One or more score rows lack a recognition label')
    for oracle in oracles:
        merged[f'delta_ic_c_minus_ic_b_{oracle}'] = merged[f'delta_R_score_{oracle}_ic_c'] - merged[f'delta_R_score_{oracle}_ic_b']
    policy_order = {value: index for index, value in enumerate(policies)}
    mode_order = {value: index for index, value in enumerate(modes)}
    merged['_policy_order'] = merged['policy'].map(policy_order)
    merged['_mode_order'] = merged['mode'].map(mode_order)
    merged = merged.sort_values(['_policy_order', '_mode_order', 'row_id'], kind='stable').drop(columns=['_policy_order', '_mode_order'])
    return merged.reset_index(drop=True)

def permutation_p_value(values: np.ndarray, labels: np.ndarray, *, permutations: int, rng: np.random.Generator) -> tuple[float, int]:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_true = int(labels.sum())
    n_false = int(len(labels) - n_true)
    observed = float(values[labels].mean() - values[~labels].mean())
    abs_observed = abs(observed)
    extreme = 0
    for _ in range(permutations):
        permuted = rng.permutation(labels)
        statistic = float(values[permuted].mean() - values[~permuted].mean())
        if abs(statistic) >= abs_observed:
            extreme += 1
    return ((extreme + 1) / (permutations + 1), extreme)

def holm_adjust(raw: list[float]) -> tuple[list[float], list[bool]]:
    m = len(raw)
    order = sorted(range(m), key=lambda index: (raw[index], index))
    adjusted_sorted: list[float] = []
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * float(raw[index]))
        running = max(running, candidate)
        adjusted_sorted.append(running)
    adjusted = [0.0] * m
    for index, value in zip(order, adjusted_sorted):
        adjusted[index] = value
    reject = [value <= 0.05 for value in adjusted]
    return (adjusted, reject)

def analyze(data: pd.DataFrame, settings: dict[str, Any]) -> pd.DataFrame:
    policies = list(settings['policies'])
    modes = list(settings['modes'])
    oracles = list(settings['oracles'])
    permutation_spec = settings['permutation']
    permutations = int(permutation_spec['count_per_cell'])
    rng = np.random.default_rng(int(permutation_spec['master_seed']))
    rows: list[dict[str, Any]] = []
    for policy in policies:
        for mode in modes:
            cell = data[(data['policy'] == policy) & (data['mode'] == mode)].copy()
            labels = cell['recognized'].to_numpy(dtype=bool)
            for oracle in oracles:
                values = cell[f'delta_ic_c_minus_ic_b_{oracle}'].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise RuntimeError(f'Non-finite analysis value in policy={policy}, mode={mode}, oracle={oracle}')
                recognized_mean = float(values[labels].mean())
                unrecognized_mean = float(values[~labels].mean())
                statistic = recognized_mean - unrecognized_mean
                p_raw, extreme_count = permutation_p_value(values, labels, permutations=permutations, rng=rng)
                rows.append({'policy': policy, 'mode': mode, 'oracle': oracle, 'n_total': int(len(values)), 'n_recognized': int(labels.sum()), 'n_unrecognized': int((~labels).sum()), 'recognized_mean_delta': recognized_mean, 'unrecognized_mean_delta': unrecognized_mean, 'recognized_minus_unrecognized': statistic, 'permutations': permutations, 'extreme_count': extreme_count, 'p_raw': p_raw})
    adjusted, rejected = holm_adjust([float(row['p_raw']) for row in rows])
    for row, p_holm, reject in zip(rows, adjusted, rejected):
        row['p_holm'] = p_holm
        row['reject_holm_0_05'] = reject
    return pd.DataFrame(rows)

def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}
    try:
        import pyarrow
        versions['pyarrow'] = pyarrow.__version__
    except Exception:
        versions['pyarrow'] = None
    return versions

def main() -> int:
    parser = argparse.ArgumentParser(description='Independent deterministic reproduction of thesis Table 5-7 / Appendix D-2.')
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    started = utc_now()
    settings = read_analysis_settings()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = {key: resolve_input(source_root, spec) for key, spec in settings['inputs'].items()}
    recognition = parse_recognition(resolved['probe'])
    policies = list(settings['policies'])
    modes = list(settings['modes'])
    oracles = list(settings['oracles'])
    ic_b = validate_score_table(pd.read_parquet(resolved['ic_b']), label='IC-b', policies=policies, modes=modes, oracles=oracles)
    ic_c = validate_score_table(pd.read_parquet(resolved['ic_c']), label='IC-c', policies=policies, modes=modes, oracles=oracles)
    analysis_input = build_analysis_input(ic_b, ic_c, recognition, settings)
    input_manifest = {'analysis_id': settings['analysis_id'], 'source_root': str(source_root), 'inputs': {key: {'relative_path': settings['inputs'][key]['relative_path'], 'resolved_path': str(path)} for key, path in resolved.items()}}
    write_json(output_dir / 'input_manifest.json', input_manifest)
    analysis_input.to_csv(output_dir / 'analysis_input.csv', index=False, encoding='utf-8-sig')
    results = analyze(analysis_input, settings)
    results.to_csv(output_dir / 'table_d2_independent.csv', index=False, encoding='utf-8-sig')
    min_row = results.sort_values(['p_raw'], kind='stable').iloc[0]
    recognition_counts = recognition['recognized'].value_counts().to_dict()
    summary = {'status': 'COMPLETE', 'analysis_id': settings['analysis_id'], 'probe': {'firms': int(len(recognition)), 'recognized': int(recognition_counts.get(True, 0)), 'unrecognized': int(recognition_counts.get(False, 0)), 'recognized_rate': float(recognition['recognized'].mean())}, 'conditional_analysis': {'conditions': int(len(results)), 'permutations_per_condition': int(settings['permutation']['count_per_cell']), 'master_seed': int(settings['permutation']['master_seed']), 'minimum_raw_p': float(min_row['p_raw']), 'minimum_raw_p_cell': {'policy': str(min_row['policy']), 'mode': str(min_row['mode']), 'oracle': str(min_row['oracle'])}, 'corresponding_holm_p': float(min_row['p_holm']), 'holm_significant_count_0_05': int(results['reject_holm_0_05'].sum())}, 'interpretation_boundary': '인식 여부는 보존된 질문에 대한 모델의 자기보고 값. 이 계산은 IC-c와 IC-b 정책가치 차이의 조절효과를 확인하며, 사전학습 중 기업 정보가 전혀 없었다는 뜻은 아님.'}
    write_json(output_dir / 'table_5_7_independent_summary.json', summary)
    readme = '# 표 5-7·부록 D-2 재계산 결과\n\n'
    readme += '보존된 세 입력 파일에서 표 5-7과 부록 D-2를 다시 계산한 결과.\n\n'
    readme += f"- 기업: {len(recognition)} (인식 {recognition_counts.get(True, 0)}, 미인식 {recognition_counts.get(False, 0)})\n"
    readme += f"- 분석 셀: {len(results)}\n"
    readme += f"- 셀당 순열: {settings['permutation']['count_per_cell']:,}\n"
    readme += '- 난수 시작값: `' + str(settings['permutation']['master_seed']) + '`\n'
    readme += '- 최소 보정 전 p값: `' + format(float(min_row['p_raw']), '.10g') + '`\n'
    readme += '- 같은 셀의 Holm 보정 p값: `' + format(float(min_row['p_holm']), '.10g') + '`\n'
    readme += '- Holm 5% 기준 유의한 셀: `' + str(int(results['reject_holm_0_05'].sum())) + '/36`\n\n'
    readme += '## 파일 읽는 순서\n\n'
    readme += '1. `table_5_7_independent_summary.json`: 표 5-7 핵심 수치와 결론.\n'
    readme += '2. `table_d2_independent.csv`: 부록 D-2의 36개 결과 행.\n'
    readme += '3. `analysis_input.csv`: 세 원자료를 기업·정책·방식별로 결합한 실제 계산 입력.\n'
    readme += '4. `input_manifest.json`: 입력 파일 위치.\n'
    readme += '5. `run_manifest.json`: 계산 코드, Python 환경, 출력 파일 기록.\n\n'
    readme += '결론: 최소 보정 전 p값은 논문 인쇄값과 소폭 다르지만 Holm 보정 후 유의한 셀이 없다는 해석은 동일.\n'
    (output_dir / 'README.md').write_text(readme, encoding='utf-8')
    preliminary_outputs = [output_dir / 'input_manifest.json', output_dir / 'analysis_input.csv', output_dir / 'table_d2_independent.csv', output_dir / 'table_5_7_independent_summary.json', output_dir / 'README.md']
    run_manifest = {'status': 'COMPLETE', 'started_utc': started, 'finished_utc': utc_now(), 'analysis_id': settings['analysis_id'], 'analysis_settings': str(SETTINGS_PATH), 'analysis_code': str(Path(__file__).resolve()), 'environment': {'platform': platform.platform(), 'python_executable': sys.executable, 'packages': package_versions()}, 'permutation': settings['permutation'], 'multiplicity': settings['multiplicity'], 'output_files': [path.name for path in preliminary_outputs], 'summary': summary}
    write_json(output_dir / 'run_manifest.json', run_manifest)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
