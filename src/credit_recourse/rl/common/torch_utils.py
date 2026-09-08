from __future__ import annotations

import numpy as np
import pandas as pd

FORBIDDEN_FEATURE_SUBSTRINGS = [
    "reward", "candidate", "projection_distance", "out_of_library", "action__",
    "oracle", "pv_r_score", "policy", "future", "rating_next", "next_rating",
]


def select_safe_numeric_features(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        lc = str(c).lower()
        if any(x in lc for x in FORBIDDEN_FEATURE_SUBSTRINGS):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(str(c))
    return cols


def build_preprocess_stats(df: pd.DataFrame, features: list[str]) -> dict:
    stats: dict[str, dict[str, float]] = {}
    for c in features:
        if c not in df.columns:
            raise KeyError(f"Missing feature column while building preprocess stats: {c}")
        s = pd.to_numeric(df[c], errors="coerce").astype("float64")
        med = float(s.median()) if s.notna().any() else 0.0
        q1 = float(s.quantile(0.25)) if s.notna().any() else 0.0
        q3 = float(s.quantile(0.75)) if s.notna().any() else 1.0
        iqr = q3 - q1
        if not np.isfinite(iqr) or abs(iqr) < 1e-12:
            iqr = 1.0
        lo = float(s.quantile(0.005)) if s.notna().any() else med - 5.0 * iqr
        hi = float(s.quantile(0.995)) if s.notna().any() else med + 5.0 * iqr
        if not np.isfinite(lo):
            lo = med - 5.0 * iqr
        if not np.isfinite(hi):
            hi = med + 5.0 * iqr
        stats[c] = {"median": med, "iqr": float(iqr), "lo": float(lo), "hi": float(hi), "missing_rate": float(s.isna().mean())}
    return stats


def _stat_for(stats: dict, c: str) -> dict:
    if c not in stats:
        raise KeyError(f"Missing preprocess stats for Stage3 feature: {c}")
    st = stats[c]
    if "median" not in st or "iqr" not in st:
        raise KeyError(f"Malformed preprocess stats for Stage3 feature {c}: expected median/iqr")
    return st


def transform(df: pd.DataFrame, features: list[str], stats: dict) -> np.ndarray:
    arr: list[np.ndarray] = []
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing Stage3 feature columns in downstream data. "
            "Broad SSL/BC and rated IQL/eval must share the same feature namespace; "
            f"first missing={missing[:20]}"
        )
    for c in features:
        st = _stat_for(stats, c)
        s = pd.to_numeric(df[c], errors="coerce").astype("float64").fillna(float(st["median"]))
        if "lo" in st and "hi" in st:
            s = s.clip(float(st["lo"]), float(st["hi"]))
        x = ((s - float(st["median"])) / (float(st["iqr"]) or 1.0)).clip(-5, 5).astype(float).to_numpy()
        arr.append(x)
    if not arr:
        raise ValueError("No safe numeric features selected for encoder")
    return np.vstack(arr).T.astype("float32")


def transform_with_missing_mask(df: pd.DataFrame, features: list[str], stats: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized feature matrix and true missing mask.

    The mask is True exactly where the original downstream dataframe has a
    missing value for the selected Stage3 feature.  It must be passed to the
    block-aware encoder at serving time to avoid train/serve skew.
    """
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(
            "Missing Stage3 feature columns in downstream data. "
            f"first missing={missing[:20]}"
        )
    X = transform(df, features, stats)
    M = np.vstack([pd.to_numeric(df[c], errors="coerce").isna().to_numpy() for c in features]).T
    return X.astype("float32"), M.astype(bool)


def transform_categorical(df: pd.DataFrame, categorical_columns: list[str], categorical_vocab: dict[str, dict[str, int]]) -> tuple[np.ndarray, dict[str, int]]:
    """Map Stage3 categorical fields to frozen token ids; unknown/missing -> 0."""
    if not categorical_columns:
        return np.zeros((len(df), 0), dtype=np.int64), {}
    missing = [c for c in categorical_columns if c not in df.columns]
    if missing:
        raise KeyError(f"Missing Stage3 categorical columns in downstream data: {missing}")
    arr = []
    oov_counts: dict[str, int] = {}
    for c in categorical_columns:
        vocab = categorical_vocab.get(c, {}) if isinstance(categorical_vocab, dict) else {}
        vals = df[c].astype(str)
        mapped = vals.map(vocab)
        oov_counts[c] = int(mapped.isna().sum())
        arr.append(mapped.fillna(0).astype(np.int64).to_numpy())
    return np.vstack(arr).T.astype(np.int64), oov_counts
