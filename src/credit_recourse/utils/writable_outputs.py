from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _writable_mode(mode: int) -> int:
    return mode | stat.S_IWUSR


def make_path_writable(path: Path) -> bool:
    path = Path(path)
    if not path.exists():
        return False
    path.chmod(_writable_mode(path.stat().st_mode))
    return True


def make_tree_writable(root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    paths = [root, *sorted(root.rglob('*'), key=lambda item: (len(item.parts), str(item)))]
    changed = 0
    checked = 0
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            before = path.stat().st_mode
            path.chmod(_writable_mode(before))
            after = path.stat().st_mode
            checked += 1
            changed += int(after != before)
            if path.is_file() and not after & stat.S_IWUSR:
                errors.append({'path': str(path), 'error': 'owner-write bit remains unset'})
        except Exception as exc:
            errors.append({'path': str(path), 'error': f'{type(exc).__name__}: {exc}'})
    probe_dir = root if root.is_dir() else root.parent
    probe_path: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix='.write_probe_', suffix='.tmp', dir=probe_dir)
        probe_path = Path(name)
        with os.fdopen(fd, 'wb') as handle:
            handle.write(b'PASS')
            handle.flush()
            os.fsync(handle.fileno())
        with probe_path.open('wb') as handle:
            handle.write(b'PASS_OVERWRITE')
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        errors.append({'path': str(probe_dir), 'error': f'write probe failed: {type(exc).__name__}: {exc}'})
    finally:
        if probe_path is not None and probe_path.exists():
            try:
                make_path_writable(probe_path)
                probe_path.unlink()
            except Exception as exc:
                errors.append({'path': str(probe_path), 'error': f'probe cleanup failed: {type(exc).__name__}: {exc}'})
    result = {
        'schema_version': 'generated_output_writability_v1',
        'status': 'PASS' if not errors else 'FAIL',
        'root': str(root),
        'checked_path_count': checked,
        'changed_mode_count': changed,
        'errors': errors,
    }
    if errors:
        raise PermissionError(f'Generated output tree is not writable: {errors[:3]}')
    return result


def atomic_write_generated_bytes(destination: Path, payload: bytes) -> dict[str, Any]:
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        make_path_writable(destination)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        make_path_writable(temp_path)
        os.replace(temp_path, destination)
        make_path_writable(destination)
    except Exception:
        if temp_path.exists():
            make_path_writable(temp_path)
            temp_path.unlink()
        raise
    if destination.stat().st_size != len(payload):
        raise OSError(f'Generated file size differs after write: {destination}')
    return {'status': 'WRITTEN', 'destination': str(destination), 'size_bytes': len(payload)}


def atomic_write_generated_text(destination: Path, text: str, *, encoding: str = 'utf-8') -> dict[str, Any]:
    return atomic_write_generated_bytes(destination, text.encode(encoding))


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as left_handle, right.open('rb') as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def atomic_copy_generated_file(source: Path, destination: Path, *, skip_if_identical: bool = True) -> dict[str, Any]:
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == destination:
        return {'status': 'SAME_PATH', 'source': str(source), 'destination': str(destination)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and skip_if_identical and _files_equal(source, destination):
        make_path_writable(destination)
        return {'status': 'IDENTICAL_SKIPPED', 'source': str(source), 'destination': str(destination)}
    if destination.exists():
        make_path_writable(destination)
    fd, temp_name = tempfile.mkstemp(prefix=f'.{destination.name}.', suffix='.tmp', dir=destination.parent)
    temp_path = Path(temp_name)
    try:
        with source.open('rb') as src, os.fdopen(fd, 'wb') as dst:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        make_path_writable(temp_path)
        os.replace(temp_path, destination)
        make_path_writable(destination)
    except Exception:
        if temp_path.exists():
            make_path_writable(temp_path)
            temp_path.unlink()
        raise
    if not _files_equal(source, destination):
        raise OSError(f'Copied file differs from source: {destination}')
    return {'status': 'COPIED', 'source': str(source), 'destination': str(destination)}
