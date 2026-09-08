from __future__ import annotations

import sys

MESSAGE = (
    "optuna_beta_optimizer.py is deprecated and disabled in the final clean-freeze "
    "production package. The active Stage1 backend is credit_recourse.oracle.backends.beta.pipeline. "
    "Do not use Optuna regeneration in clean rebuild runs."
)


def main(argv=None) -> int:
    print(MESSAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
