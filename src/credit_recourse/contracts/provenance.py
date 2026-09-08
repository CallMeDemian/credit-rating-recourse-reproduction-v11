from __future__ import annotations
'Portable, role-relative provenance identities.\n\nRuntime paths remain available to producers that need to open files.  This\nmodule supplies a separate stable identity for manifests and catalogs.\n'
from pathlib import Path
from typing import Mapping

def portable_path_identity(path: str | Path, *, roots: Mapping[str, str | Path]) -> dict[str, str]:
    resolved = Path(path).resolve()
    matches: list[tuple[int, str, Path]] = []
    for role, root in roots.items():
        root_path = Path(root).resolve()
        try:
            relative = resolved.relative_to(root_path)
        except ValueError:
            continue
        matches.append((len(root_path.parts), str(role), relative))
    if not matches:
        return {'root_role': 'external', 'relative_path': resolved.as_posix()}
    _depth, role, relative = max(matches, key=lambda item: item[0])
    return {'root_role': role, 'relative_path': relative.as_posix()}
