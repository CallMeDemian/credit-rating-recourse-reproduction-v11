from __future__ import annotations
'Scientific zero contract for score-difference diagnostics.\n\nThe thesis analysis uses exact floating-point zero for effect signs, nonzero\npair counts, and Wilcoxon preprocessing.  File-serialization comparison\ntolerances belong to the parity comparator and must never enter this contract.\n'
import math
from typing import Any, Iterable
import numpy as np
SCORE_TIE_CONTRACT_VERSION = 'score_tie_contract_v2_exact_zero'
SCIENTIFIC_ZERO_DEFINITION = 'exact_float_zero'

def finite_score_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Return a one-dimensional finite float array."""
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]

def score_positive_mask(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array > 0.0)

def score_negative_mask(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array < 0.0)

def score_nonzero_mask(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array != 0.0)

def score_tie_mask(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.isfinite(array) & (array == 0.0)

def score_positive_fraction(values: Iterable[float] | np.ndarray) -> float:
    """Positive fraction over the original row count.

    Non-finite rows remain in the denominator, matching the historical
    policy-summary denominator contract while no longer treating numerical
    epsilon as a positive effect.
    """
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return math.nan
    return float(score_positive_mask(array).mean())

def score_nonzero_count(values: Iterable[float] | np.ndarray) -> int:
    return int(score_nonzero_mask(values).sum())

def score_wilcoxon_values(values: Iterable[float] | np.ndarray) -> np.ndarray:
    array = finite_score_array(values)
    return array[array != 0.0]

def score_tie_contract_metadata() -> dict[str, Any]:
    return {'contract': SCORE_TIE_CONTRACT_VERSION, 'scientific_zero_definition': SCIENTIFIC_ZERO_DEFINITION, 'serialization_tolerance_applies': False, 'applies_to': ['positive_fraction', 'negative_fraction', 'zero_fraction', 'n_nonzero_pairs', 'wilcoxon_zero_discard'], 'denominator_policy': 'positive fractions retain the original row denominator; non-finite values are not positive'}

def score_tie_contract_status(metadata: dict[str, Any] | None) -> tuple[bool, str]:
    """Validate a producer metadata object against the active tie contract."""
    if not isinstance(metadata, dict):
        return (False, 'metadata is not a JSON object')
    contract = metadata.get('score_tie_contract')
    if not isinstance(contract, dict):
        return (False, 'score_tie_contract is absent')
    observed_version = contract.get('contract')
    if observed_version != SCORE_TIE_CONTRACT_VERSION:
        return (False, f'contract version mismatch: observed={observed_version!r}, expected={SCORE_TIE_CONTRACT_VERSION!r}')
    observed_definition = contract.get('scientific_zero_definition')
    if observed_definition != SCIENTIFIC_ZERO_DEFINITION:
        return (False, f'scientific zero definition mismatch: observed={observed_definition!r}, expected={SCIENTIFIC_ZERO_DEFINITION!r}')
    if contract.get('serialization_tolerance_applies') is not False:
        return (False, 'serialization tolerance must not apply to scientific ties')
    return (True, 'PASS')
