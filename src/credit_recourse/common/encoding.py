# -*- coding: utf-8 -*-
"""Encoding helpers for Windows/PowerShell-safe thesis pipeline execution."""
from __future__ import annotations

import io
import locale
import os
import sys
from pathlib import Path
from typing import Any


def configure_utf8_stdio() -> None:
    """Force UTF-8 stdout/stderr and UTF-8-friendly process defaults.

    The project mostly runs on Windows via PowerShell + Tee-Object.  This helper
    is intentionally safe to call repeatedly and is also invoked by
    ``src/sitecustomize.py`` automatically when ``src`` is on PYTHONPATH.
    """
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("LC_ALL", "C.UTF-8")
    os.environ.setdefault("LANG", "C.UTF-8")

    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
            continue
        except Exception:
            pass
        buffer = getattr(stream, "buffer", None)
        if buffer is None:
            continue
        try:
            setattr(sys, name, io.TextIOWrapper(buffer, encoding="utf-8", errors="replace", line_buffering=True))
        except Exception:
            pass

    try:
        locale.getpreferredencoding = lambda do_setlocale=True: "UTF-8"  # type: ignore[assignment]
    except Exception:
        pass


def write_text_utf8(path: str | Path, text: str, **kwargs: Any) -> None:
    Path(path).write_text(text, encoding="utf-8", **kwargs)


def read_text_utf8(path: str | Path, **kwargs: Any) -> str:
    return Path(path).read_text(encoding="utf-8", **kwargs)
