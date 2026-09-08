from __future__ import annotations

import runpy
from pathlib import Path


def main() -> int:
    impl = Path(__file__).with_name("_pipeline_impl.py")
    runpy.run_path(str(impl), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
