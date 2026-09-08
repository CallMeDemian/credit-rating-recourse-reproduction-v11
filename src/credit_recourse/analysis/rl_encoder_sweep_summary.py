from __future__ import annotations
'Summarize the five-cell Stage3 SSL encoder sweep reported in Appendix D.'
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
EXPECTED = [('b256_ep30_m030_s0', 256, 30, 0.3, 0), ('b256_ep49_m030_s42', 256, 49, 0.3, 42), ('b512_ep15_m015_s1', 512, 15, 0.15, 1), ('b512_ep30_m015_s1', 512, 30, 0.15, 1), ('b512_ep50_m030_s42', 512, 50, 0.3, 42)]

def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(payload, dict):
        raise TypeError(f'JSON root must be an object: {path}')
    return payload

def _require_equal(cell: str, source: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f'{cell} {source}: expected={expected!r}, observed={observed!r}')

def build(run_root: Path, out: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    observed_dirs = sorted((p.name for p in run_root.iterdir() if p.is_dir()))
    expected_dirs = sorted((cell for cell, *_ in EXPECTED))
    if observed_dirs != expected_dirs:
        raise ValueError(f'Encoder-sweep grid mismatch: expected={expected_dirs}, observed={observed_dirs}')
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for cell, batch, epochs, mask, seed in EXPECTED:
        directory = run_root / cell
        meta_path = directory / 'metadata.json'
        contract_path = directory / 'sweep_cell_contract.json'
        metadata = _load_json(meta_path)
        contract = _load_json(contract_path)
        _require_equal(cell, 'contract schema', contract.get('schema_version'), 'rl_encoder_sweep_cell_contract_v1')
        _require_equal(cell, 'contract cell_id', contract.get('cell_id'), cell)
        for key, expected in {'batch_size': batch, 'epochs': epochs, 'seed': seed, 'train_mode': 'selection'}.items():
            _require_equal(cell, f'contract {key}', contract.get(key), expected)
            _require_equal(cell, f'metadata {key}', metadata.get(key), expected)
        if abs(float(contract.get('masking_ratio')) - mask) > 1e-12:
            raise ValueError(f'{cell} contract masking_ratio mismatch')
        if 'masking_ratio' in metadata and abs(float(metadata['masking_ratio']) - mask) > 1e-12:
            raise ValueError(f'{cell} metadata masking_ratio mismatch')
        _require_equal(cell, 'contract selected_for_final_family', bool(contract.get('selected_for_final_family')), cell == 'b512_ep30_m015_s1')
        if metadata.get('status') != 'PASS':
            raise ValueError(f"{cell} metadata status={metadata.get('status')!r}")
        val = float(metadata['best_val_loss'])
        if not pd.notna(val):
            raise ValueError(f'{cell} best_val_loss is not finite')
        rows.append({'cell_id': cell, 'batch_size': batch, 'epochs': epochs, 'masking_ratio': mask, 'seed': seed, 'train_mode': 'selection', 'best_val_loss': val, 'selected_for_final_family': cell == 'b512_ep30_m015_s1'})
        for kind, path in (('metadata', meta_path), ('cell_contract', contract_path)):
            inputs.append({'cell_id': cell, 'input_kind': kind, 'path': str(path)})
    frame = pd.DataFrame(rows)
    if frame['cell_id'].duplicated().any():
        raise ValueError('Duplicate encoder-sweep cell_id')
    table = out / 'rl_encoder_sweep_summary.csv'
    input_ledger = out / 'rl_encoder_sweep_input_files.csv'
    frame.to_csv(table, index=False, encoding='utf-8-sig')
    pd.DataFrame(inputs).to_csv(input_ledger, index=False, encoding='utf-8-sig')
    manifest = {'schema_version': 'rl_encoder_sweep_summary_v2', 'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'cell_count': len(frame), 'selected_cell': 'b512_ep30_m015_s1', 'selection_metric': 'best_val_loss is reported diagnostically; the final family follows the frozen thesis design rather than post-hoc minimum validation loss alone', 'cell_contract_required': True, 'outputs': {'summary': {'path': table.name, 'rows': len(frame)}, 'inputs': {'path': input_ledger.name, 'rows': len(inputs)}}, 'interpretation_boundary': 'Fresh execution of the five frozen Stage3 configurations. Historical development chronology remains provenance; numeric losses may differ across CUDA/driver builds. The intended mask/batch/epoch/seed contract is persisted independently of pipeline metadata for every archived cell.'}
    manifest_path = out / 'rl_encoder_sweep_manifest.json'
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return manifest

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-root', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    manifest = build(Path(args.run_root), Path(args.out))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
