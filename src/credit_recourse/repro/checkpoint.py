"""Durable checkpoints for long reproduction tasks.

Each task has one plainly named ``finalized`` checkpoint. Reuse compares the
declared contract and the output file paths and sizes.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "repro_cross_run_checkpoint_v1"
MANIFEST_NAME = "checkpoint_manifest.json"


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is incomplete or incompatible."""


def checkpoint_directory(store_root: Path, task_name: str) -> Path:
    safe_task = "".join(c if c.isalnum() or c in "-_." else "_" for c in task_name)
    return store_root.resolve() / safe_task / "finalized"


def output_inventory(root: Path) -> dict[str, Any]:
    root = root.resolve()
    rows = [
        {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size}
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    return {
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "files": rows,
    }


def inventory_differences(root: Path, expected: dict[str, Any]) -> list[str]:
    actual = output_inventory(root)
    expected_rows = {
        str(row.get("path", "")).replace("\\", "/"): int(row.get("size_bytes", -1))
        for row in expected.get("files", [])
        if isinstance(row, dict)
    }
    actual_rows = {str(row["path"]): int(row["size_bytes"]) for row in actual["files"]}
    errors: list[str] = []
    if int(expected.get("file_count", -1)) != actual["file_count"]:
        errors.append("output_file_count")
    if int(expected.get("total_bytes", -1)) != actual["total_bytes"]:
        errors.append("output_total_bytes")
    if expected_rows != actual_rows:
        errors.append("output_file_paths_or_sizes")
    return errors


def read_manifest(checkpoint_root: Path) -> dict[str, Any]:
    manifest_path = checkpoint_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise CheckpointError(f"checkpoint manifest is missing: {manifest_path}")
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CheckpointError(f"checkpoint manifest is unreadable: {manifest_path}: {exc}") from exc


def verify_checkpoint(
    checkpoint_root: Path,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_root = checkpoint_root.resolve()
    manifest = read_manifest(checkpoint_root)
    errors: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if manifest.get("status") != "PASS":
        errors.append("status")
    contract = manifest.get("contract")
    if not isinstance(contract, dict):
        errors.append("contract")
        contract = {}
    if expected_contract is not None and contract != expected_contract:
        errors.append("contract_mismatch")
    output = manifest.get("output")
    output_root = checkpoint_root / "output"
    if not isinstance(output, dict):
        errors.append("output_inventory")
    elif not output_root.is_dir():
        errors.append("output_directory")
    else:
        errors.extend(inventory_differences(output_root, output))
    if errors:
        raise CheckpointError(
            f"checkpoint verification failed ({', '.join(errors)}): {checkpoint_root}"
        )
    return manifest


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise CheckpointError(f"materialization destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination, copy_function=shutil.copy2)


def materialize_checkpoint(
    checkpoint_root: Path,
    destination: Path,
    expected_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest = verify_checkpoint(checkpoint_root, expected_contract)
    _copy_tree(checkpoint_root.resolve() / "output", destination.resolve())
    if inventory_differences(destination.resolve(), manifest["output"]):
        shutil.rmtree(destination.resolve(), ignore_errors=True)
        raise CheckpointError("checkpoint materialization path/size comparison failed")
    return manifest


def finalize_checkpoint(
    *,
    store_root: Path,
    task_name: str,
    contract: dict[str, Any],
    completed_output: Path,
    producer: dict[str, Any],
) -> Path:
    final_root = checkpoint_directory(store_root, task_name)
    if final_root.exists():
        verify_checkpoint(final_root, contract)
        return final_root
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=final_root.parent))
    try:
        _copy_tree(completed_output.resolve(), staging / "output")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "task_name": task_name,
            "contract": contract,
            "producer": producer,
            "output": output_inventory(staging / "output"),
        }
        (staging / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        verify_checkpoint(staging, contract)
        try:
            staging.rename(final_root)
        except FileExistsError:
            verify_checkpoint(final_root, contract)
        return final_root
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CheckpointError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cmd_verify(args: argparse.Namespace) -> int:
    contract = _load_json(Path(args.contract)) if args.contract else None
    manifest = verify_checkpoint(Path(args.checkpoint), contract)
    print(json.dumps({"status": "PASS", "task": manifest.get("task_name")}))
    return 0


def _cmd_finalize(args: argparse.Namespace) -> int:
    contract = _load_json(Path(args.contract))
    producer = _load_json(Path(args.producer))
    root = finalize_checkpoint(
        store_root=Path(args.store_root),
        task_name=args.task_name,
        contract=contract,
        completed_output=Path(args.output),
        producer=producer,
    )
    print(str(root))
    return 0


def _cmd_materialize(args: argparse.Namespace) -> int:
    contract = _load_json(Path(args.contract)) if args.contract else None
    manifest = materialize_checkpoint(Path(args.checkpoint), Path(args.destination), contract)
    if args.receipt:
        _write_json(
            Path(args.receipt),
            {
                "schema_version": "repro_checkpoint_reuse_receipt_v1",
                "status": "PASS",
                "task": manifest.get("task_name"),
                "reuse": True,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "destination": str(Path(args.destination).resolve()),
                "producer": manifest.get("producer"),
            },
        )
    print(json.dumps({"status": "PASS", "task": manifest.get("task_name")}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--contract")
    verify.set_defaults(func=_cmd_verify)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--store-root", required=True)
    finalize.add_argument("--task-name", required=True)
    finalize.add_argument("--contract", required=True)
    finalize.add_argument("--producer", required=True)
    finalize.add_argument("--output", required=True)
    finalize.set_defaults(func=_cmd_finalize)
    materialize = sub.add_parser("materialize")
    materialize.add_argument("--checkpoint", required=True)
    materialize.add_argument("--destination", required=True)
    materialize.add_argument("--contract")
    materialize.add_argument("--receipt")
    materialize.set_defaults(func=_cmd_materialize)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except CheckpointError as exc:
        print(f"checkpoint error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
