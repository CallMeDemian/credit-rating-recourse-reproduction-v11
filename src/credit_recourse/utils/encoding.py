from __future__ import annotations

import io
import os
import sys


def configure_utf8_stdio() -> None:
    """Force UTF-8 console I/O when Python is launched from Windows/PowerShell.

    File writes in this repository already pass explicit encodings. This helper
    prevents Korean text in JSON/diagnostic prints from being mojibake when the
    host console uses a legacy code page. It is intentionally best-effort and
    side-effect limited to stdio encoding.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            # Older/wrapped streams may not support reconfigure. Leave them alone.
            pass
