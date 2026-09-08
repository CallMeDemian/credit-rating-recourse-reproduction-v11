from __future__ import annotations
'Thesis-compatible Kruskal-Wallis edge-case contract for Stage00_04.\n\nSciPy versions disagree on the all-identical-input case: some raise\n``ValueError`` while others return NaN statistics.  The thesis pipeline catches\nthe failure and records all three statistics as NaN with ``kw_na=True``.\n'
from collections.abc import Sequence
import numpy as np
KW_CONSTANT_CONTRACT_VERSION = 'stage00_04_kw_constant_v2_thesis_nan'

def constant_kw_eta2_result(groups: Sequence[np.ndarray]) -> tuple[float, float, float, bool] | None:
    """Return the frozen thesis result when all KW observations are identical.

    ``None`` means the caller should continue with the ordinary SciPy
    Kruskal-Wallis calculation.  Empty/no-group cases remain owned by the
    caller's existing insufficient-data guards.
    """
    arrays = [np.asarray(group, dtype=float).reshape(-1) for group in groups]
    nonempty = [array for array in arrays if array.size > 0]
    if not nonempty:
        return None
    combined = np.concatenate(nonempty)
    finite = combined[np.isfinite(combined)]
    if finite.size == 0:
        return None
    if np.unique(finite).size <= 1:
        return (float('nan'), float('nan'), float('nan'), True)
    return None
