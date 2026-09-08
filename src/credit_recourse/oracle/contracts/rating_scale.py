from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np
import pandas as pd

GRADE_ORDER_10 = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
GRADE2NUM_10 = {g: i + 1 for i, g in enumerate(GRADE_ORDER_10)}
NUM2GRADE_10 = {v: k for k, v in GRADE2NUM_10.items()}

GRADE_ORDER_7 = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
GRADE2NUM_7 = {g: i + 1 for i, g in enumerate(GRADE_ORDER_7)}

NOTCH_ORDER = [
    "AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
    "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D",
]
GRADE2NUM_NOTCH = {g: i + 1 for i, g in enumerate(NOTCH_ORDER)}

PD_MAP_10 = {
    "AAA": 0.0001, "AA": 0.0001, "A": 0.0023, "BBB": 0.0031,
    "BB": 0.0070, "B": 0.0173, "CCC": 0.1389, "CC": 0.5805,
    "C": 0.5805, "D": 1.0,
}

ICR_ALLOWED_SECURITY_CODE = 40
# Final Stage0 sample contract: security type 40 (ICR) with agency codes 10/20/30/60/90.
# 60/90 are retained to reproduce the historical Oracle sample and to keep enough
# firm-year observations; dedup still prioritizes the three main CRAs first.
ALLOWED_MAIN_AGENCY_CODES = {10, 20, 30, 60, 90}
AGENCY_PRIORITY = {10: 1, 20: 2, 30: 3, 60: 4, 90: 5}
AGENCY_NAME = {
    10: "NICE신용평가",
    20: "한국신용평가",
    30: "한국기업평가",
    60: "NICE평가정보",
    90: "기타평가정보",
}

_REJECT_PATTERNS = [
    r"^A[1-3][+\-]?$", r"^P-?\d", r"^E-?\d", r"^E\d", r"^CR-?\d",
    r"^CFR\d", r"^BAA\d", r"^BA\d", r"^CAA\d", r"^CA\d", r"^B[1-9]$",
    r"정상|양호|보통|우수|미흡|없음|철회|취소|보류|NR|WR|N\.R|W\.R",
]
_OUTLOOK_WORDS = [
    "STABLE", "POSITIVE", "NEGATIVE", "DEVELOPING", "WATCH", "OUTLOOK", "CREDITWATCH",
    "안정적", "긍정적", "부정적", "유동적", "상향", "하향", "검토", "관찰",
]


def normalize_rating_text(x: Any) -> str:
    s = str(x).strip().upper().replace(" ", "")
    s = s.replace("−", "-").replace("－", "-").replace("–", "-")
    s = s.replace("°", "0").replace("º", "0").replace("*", "")
    for word in _OUTLOOK_WORDS:
        s = s.replace(word, "")
    return s.replace("↑", "").replace("↓", "")


def normalize_grade_components(x: Any) -> tuple[Any, Any, Any, str]:
    if pd.isna(x):
        return (pd.NA, pd.NA, pd.NA, "missing")
    raw = str(x).strip()
    if not raw:
        return (pd.NA, pd.NA, pd.NA, "empty")
    s = normalize_rating_text(raw)
    for pat in _REJECT_PATTERNS:
        if re.search(pat, s):
            return (pd.NA, pd.NA, pd.NA, f"excluded_pattern:{pat}")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.split(r"[/,;|]", s)[0]
    s = re.sub(r"[^A-Z0-9+\-]", "", s)
    s = re.sub(r"\++$", "+", s)
    s = re.sub(r"\-+$", "-", s)
    m = re.match(r"^(AAA|CCC|BBB|AA|BB|CC|A|B|C|D)([+\-0])?", s)
    if not m:
        return (pd.NA, pd.NA, pd.NA, "no_long_term_grade_match")
    base_raw = m.group(1)
    sign = m.group(2)
    if base_raw == "D":
        sign = pd.NA
        notch = "D"
    else:
        if sign == "0" or sign is None:
            sign = pd.NA
            notch = base_raw
        else:
            notch = base_raw + sign
    grade10 = fold_to_10(notch)
    return (base_raw, sign, notch, "ok") if grade10 is not pd.NA else (pd.NA, pd.NA, pd.NA, "invalid_base")


def fold_to_10(x: Any) -> Any:
    if pd.isna(x):
        return pd.NA
    s = str(x).strip().upper().replace("0", "")
    if s == "D":
        return "D"
    m = re.match(r"^(AAA|CCC|BBB|AA|BB|CC|A|B|C)([+\-])?$", s)
    if not m:
        return pd.NA
    base = m.group(1)
    return base if base in GRADE2NUM_10 else pd.NA


def fold_to_7(x: Any) -> Any:
    g10 = fold_to_10(x)
    if pd.isna(g10):
        return pd.NA
    return "CCC" if g10 in {"CCC", "CC", "C", "D"} else g10


def notch_grade(x: Any) -> Any:
    if pd.isna(x):
        return pd.NA
    _, _, notch, status = normalize_grade_components(x)
    return notch if status == "ok" else pd.NA


def add_rating_scale_columns(df: pd.DataFrame, source_col: str = "grade_base") -> pd.DataFrame:
    out = df.copy()
    src = out[source_col] if source_col in out.columns else pd.Series([pd.NA] * len(out), index=out.index)
    comps = src.apply(normalize_grade_components)
    out["grade_base_raw"] = comps.map(lambda t: t[0])
    out["grade_base_notch"] = comps.map(lambda t: t[2])
    out["rating_num_notch"] = out["grade_base_notch"].map(GRADE2NUM_NOTCH).astype("Int64")
    out["grade_base_10"] = out["grade_base_notch"].map(fold_to_10)
    out["rating_num_10"] = out["grade_base_10"].map(GRADE2NUM_10).astype("Int64")
    out["grade_base_7"] = out["grade_base_10"].map(fold_to_7)
    out["rating_num_7"] = out["grade_base_7"].map(GRADE2NUM_7).astype("Int64")
    return out


def ensure_10_grade_contract(boundaries: Mapping[str, float]) -> dict[str, float]:
    """Return robust monotone AAA..C thresholds; D is fallback below threshold_C.

    The final Oracle contract keeps CC/C/D in the master scale even when the
    sample cannot estimate their cutoffs directly.  This helper therefore
    extrapolates CC/C from the lower observed gaps, clips cutoffs to a valid
    0..100 score range, and enforces ``threshold_C > 0`` so that D remains a
    reachable fallback grade instead of being silently eliminated by a negative
    lower-tail cutoff.
    """
    ordered_no_d = [g for g in GRADE_ORDER_10 if g != "D"]
    b = {
        str(k): float(v)
        for k, v in dict(boundaries).items()
        if k in ordered_no_d and pd.notna(v) and np.isfinite(float(v))
    }

    known = [(g, b[g]) for g in ordered_no_d if g in b]
    gaps: list[float] = []
    for (_, hi), (_, lo) in zip(known, known[1:]):
        gap = hi - lo
        if np.isfinite(gap) and gap > 0:
            gaps.append(float(gap))
    if "B" in b and "CCC" in b and b["B"] > b["CCC"]:
        tail_gap = float(b["B"] - b["CCC"])
    elif gaps:
        tail_gap = float(np.median(gaps[-4:]))
    else:
        tail_gap = 5.0
    tail_gap = float(np.clip(tail_gap, 0.5, 25.0))

    if "CCC" not in b:
        if "B" in b:
            b["CCC"] = b["B"] - tail_gap
        elif "CC" in b:
            b["CCC"] = b["CC"] + tail_gap
        else:
            b["CCC"] = max(2.0 * tail_gap + 1.0, 10.0)
    if "CC" not in b:
        b["CC"] = b["CCC"] - tail_gap
    if "C" not in b:
        b["C"] = b["CC"] - tail_gap

    # Keep the extrapolated lower tail inside the valid score domain and keep D reachable.
    min_gap = min(0.5, max(tail_gap * 0.1, 1e-6))
    b["C"] = float(np.clip(b["C"], min_gap, 99.0))
    b["CC"] = float(np.clip(max(b["CC"], b["C"] + min_gap), b["C"] + min_gap, 99.5))
    b["CCC"] = float(np.clip(max(b["CCC"], b["CC"] + min_gap), b["CC"] + min_gap, 100.0))

    # Fill missing upper grades from the nearest lower anchor, then enforce strict descending order.
    prev = None
    for g in ordered_no_d:
        if g not in b:
            if prev is None:
                lower_vals = [b[x] for x in ordered_no_d[ordered_no_d.index(g)+1:] if x in b]
                b[g] = min(100.0, (lower_vals[0] + tail_gap) if lower_vals else 100.0)
            else:
                b[g] = prev - tail_gap
        b[g] = float(np.clip(b[g], 0.0, 100.0))
        if prev is not None and b[g] >= prev:
            b[g] = max(0.0, prev - min_gap)
        prev = b[g]

    # Re-assert the lower tail after the upper pass in case compression occurred.
    b["C"] = max(min_gap, min(b["C"], b["CC"] - min_gap))
    b["CC"] = max(b["C"] + min_gap, min(b["CC"], b["CCC"] - min_gap))
    return {g: float(b[g]) for g in ordered_no_d if g in b}


def assign_grade_10(scores: Any, boundaries: Mapping[str, float]) -> np.ndarray:
    b = ensure_10_grade_contract(boundaries)
    arr = np.asarray(scores, dtype=float)
    out = np.full(len(arr), "D", dtype=object)
    # Set from bad to good so high thresholds overwrite lower grades.
    for g in ["C", "CC", "CCC", "B", "BB", "BBB", "A", "AA", "AAA"]:
        if g in b:
            out[arr >= b[g]] = g
    return out


def assert_grade_order_10(order: list[str]) -> None:
    if list(order) != GRADE_ORDER_10:
        raise AssertionError(f"GRADE_ORDER must be {GRADE_ORDER_10}, got {order}")
