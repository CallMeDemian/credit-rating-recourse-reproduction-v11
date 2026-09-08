from __future__ import annotations

import re
from typing import Any

_KIS_ITEM_CODE_RE = re.compile(r"([A-Z]\d{2}[A-Z]\d{8,})")


def canonical_item_code(value: Any) -> str:
    """Return the canonical KIS/NICE financial statement item code.

    The raw Oracle/KIS statement panels can expose the same item identifier in
    several string forms, for example::

        U01B800000000
        [U01B800000000
        [U01B800000000]
        [U01B800000000]계속영업이익(손실)(IFRS)(천원)

    All of these must compare as the same exact item code.  This helper is not
    an alias or fuzzy matcher; it only removes presentation drift around the
    item identifier and returns the first canonical-looking KIS item code.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return ""
    match = _KIS_ITEM_CODE_RE.search(s)
    if match:
        return match.group(1).strip()
    return s.strip().strip("[]")
