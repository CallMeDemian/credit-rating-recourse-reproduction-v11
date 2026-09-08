import os
"""
Task 2 v2: Manual mapping override 적용
"""
import pandas as pd
import numpy as np
import re
import json
from pathlib import Path

UCODE_PATTERN = re.compile(r"(U\d{2}[A-Z]\d{9,12})")

def canonical_item_code_from_text(text) -> str:
    if pd.isna(text):
        return ""
    m = UCODE_PATTERN.search(str(text))
    return m.group(1) if m else ""

def _canonicalize_item_code_value(value, fallback="") -> str:
    return canonical_item_code_from_text(value) or canonical_item_code_from_text(fallback) or ("" if pd.isna(value) else str(value).strip())

ITEM_CODE_ALIASES = {
    "매출액": ["U01B100000000", "U02B100000000", "U03B100000000", "U04B100000000"],
    "계속사업이익": ["U01B800000000", "U02B800000000", "U03B800000000", "U04B800000000"],
    "무형자산상각비": ["U01B350014300", "U02B350012400", "U03B350012400", "U04B350012400", "U01B350012400"],
}

OUT = Path(os.environ['STAGE2_OUT'])

_cand_path = Path(os.environ['STAGE2_CANDIDATES_XLSX'])
if _cand_path.suffix.lower() == '.csv':
    candidates = pd.read_csv(_cand_path, encoding='utf-8-sig')
else:
    candidates = pd.read_excel(_cand_path, sheet_name='후보_재무비율')

    # [SUPPLEMENT PATCH 2026-05-07] 컬럼 구조: No, 평가항목, 세부영역, 후보비율명, 공식, 분자, 분모, 기대방향, 단위, 우선순위
    _supplement_path = Path(os.environ.get(
        'STAGE2_SUPPLEMENT_CSV',
        str(_cand_path.parent / 'supplemental_ratio_candidates.csv')
    ))
    if _supplement_path.is_file():
        _supplement = pd.read_csv(_supplement_path, encoding='utf-8-sig')
        existing_nos = set(candidates['No'].astype(str))
        _supplement = _supplement[~_supplement['No'].astype(str).isin(existing_nos)].copy()
        candidates = pd.concat([candidates, _supplement], ignore_index=True)
        print(f'  [SUPPLEMENT] +{len(_supplement)} candidates from {_supplement_path.name}')
    else:
        print(f'  [SUPPLEMENT] {_supplement_path.name} not found; base candidates only')

inventory = pd.read_csv(OUT / 'statement_item_inventory.csv')
inventory['item_code'] = inventory.apply(lambda r: _canonicalize_item_code_value(r.get('item_code', ''), r.get('original_column_name', '')), axis=1)
if inventory['item_code'].astype(str).str.startswith('[').any():
    bad = inventory[inventory['item_code'].astype(str).str.startswith('[')].head(20).to_dict(orient='records')
    raise RuntimeError(f'Task2 refuses broken inventory item_code starting with [: {bad}')

# ============================================================
# Manual override: 핵심 항목 정확 매핑
# ============================================================
# (role_hint, statement, item_code, item_name, original_column)
MANUAL_OVERRIDE = {
    '매출액': {
        'item_code': 'U01B100000000',
        'item_name': '매출액(수익)',
        'statement': '손익계산서',
    },
    '영업이익': {
        'item_code': 'U01B430000000',
        'item_name': '(정상)영업손익(계산수치)',
        'statement': '손익계산서',
        'note': '* (정상)영업손익(계산수치). U01B420000000(보고서기재)과 동일 의미.',
    },
    '정상영업이익': {  # 영업이익과 동일
        'item_code': 'U01B430000000',
        'item_name': '(정상)영업손익(계산수치)',
        'statement': '손익계산서',
    },
    '계속사업이익': {
        'item_code': 'U01B800000000',
        'item_name': '계속영업이익(손실)',
        'statement': '손익계산서',
    },
    '무형자산상각비': {
        'item_code': 'U01B350014300',
        'item_name': '기타무형자산상각비',
        'statement': '손익계산서',
        'note': 'NICE 표시면 무형자산상각비. U01B350014300이 91% 결측률로 main.',
    },
}

# ============================================================
# Duplicate ratio alias policy
# ============================================================
# R*** codes are internal engineered ratio identifiers. Some candidate
# definitions collapse to the same executable formula under the current
# KIS-VALUE account-code mapping (for example, 영업이익/정상영업이익/EBIT
# may map to the same source item). Keep those ratios in Stage 2 for
# audit/reproducibility, but mark aliases so Stage 3 and downstream
# simulator/evaluator do not treat them as independent intervention levers.
DUPLICATE_RATIO_ALIAS_MAP = {
    # profitability aliases caused by identical source mapping
    "R003": {"canonical": "R002", "type": "exact_value_alias", "reason": "정상영업이익률 maps to the same executable formula as 영업이익률 in the current source mapping."},
    "R004": {"canonical": "R002", "type": "exact_value_alias", "reason": "EBIT마진 maps to the same executable formula as 영업이익률 in the current source mapping."},
    "R014": {"canonical": "R013", "type": "exact_value_alias", "reason": "EBIT/총자산 maps to the same executable formula as 영업이익/총자산."},

    # explicit duplicate or appendix aliases
    "R216": {"canonical": "R010", "type": "exact_value_alias", "reason": "OCF/매출액 is identical to 영업현금흐름마진."},
    "R211": {"canonical": "R064", "type": "exact_value_alias", "reason": "Altman 이익잉여금/총자산 is identical to 이익잉여금/총자산."},
    "R118": {"canonical": "R076", "type": "exact_value_alias", "reason": "순운전자본비율 is identical to 순운전자본/총자산."},
    "R210": {"canonical": "R076", "type": "exact_value_alias", "reason": "Altman 운전자본/총자산 is identical to 순운전자본/총자산."},
    "R119": {"canonical": "R077", "type": "exact_value_alias", "reason": "순운전자본/매출액 is identical to 운전자본/매출액."},
    "R080": {"canonical": "R079", "type": "exact_value_alias", "reason": "EBIT 이자보상배율 is identical to 이자보상배율 under the current EBIT mapping."},
    "R086": {"canonical": "R085", "type": "exact_value_alias", "reason": "이자비용/매출액 uses the same numerator item code as 금융비용부담률; not an independent ratio in the current source mapping."},
    "R088": {"canonical": "R089", "type": "exact_value_alias", "reason": "이자비용/EBITDA uses the same numerator item code as 금융비용/EBITDA; keep the semantically accurate financial-cost ratio."},
    "R131": {"canonical": "R094", "type": "exact_value_alias", "reason": "OCF/유동부채 duplicates the debt-service OCF/유동부채 ratio."},
    "R114": {"canonical": "R102", "type": "exact_value_alias", "reason": "순차입금상환가능기간 is identical to 순차입금/EBITDA."},
    "R127": {"canonical": "R117", "type": "exact_value_alias", "reason": "현금성자산/유동부채 is identical to 현금비율."},
    "R164": {"canonical": "R158", "type": "exact_value_alias", "reason": "매출채권회수기간 is identical to 매출채권회전일수."},
    "R163": {"canonical": "R159", "type": "exact_value_alias", "reason": "재고보유기간 is identical to 재고자산회전일수."},
    "R165": {"canonical": "R160", "type": "exact_value_alias", "reason": "매입채무지급기간 is identical to 매입채무회전일수."},
    "R172": {"canonical": "R171", "type": "exact_value_alias", "reason": "정상영업이익증가율 maps to the same executable formula as 영업이익증가율."},
    "R173": {"canonical": "R171", "type": "exact_value_alias", "reason": "EBIT증가율 maps to the same executable formula as 영업이익증가율."},
}


def _load_duplicate_ratio_alias_master(default_map):
    """Load duplicate R-code alias policy from config/master data.

    Source of truth is configs/duplicate_ratio_alias_master.csv. The hard-coded
    map above is only a fallback so older bundles remain runnable.
    """
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(os.environ.get("STAGE2_DUPLICATE_ALIAS_MASTER", ""))
        if os.environ.get("STAGE2_DUPLICATE_ALIAS_MASTER") else None,
        root / "configs" / "duplicate_ratio_alias_master.csv",
        root / "duplicate_ratio_alias_master.csv",
    ]

    for path in candidates:
        if path is None:
            continue
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")

        required = {"alias_ratio_id", "canonical_ratio_id"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"duplicate alias master missing columns: {missing} ({path})")

        if "active" in df.columns:
            active = df["active"].astype(str).str.lower().isin(["true", "1", "yes", "y"])
            df = df[active].copy()

        alias_map = {}
        for _, r in df.iterrows():
            alias_id = str(r["alias_ratio_id"]).strip()
            canonical_id = str(r["canonical_ratio_id"]).strip()
            if not alias_id or alias_id.lower() == "nan":
                continue
            alias_map[alias_id] = {
                "canonical": canonical_id,
                "type": r.get("duplicate_alias_type", "exact_value_alias"),
                "reason": r.get("reason", "duplicate alias from master data"),
                "stage3_exclude": bool(str(r.get("stage3_exclude", True)).lower() in ["true", "1", "yes", "y"]),
                "source": r.get("source", path.name),
                "policy_note": r.get("policy_note", ""),
            }
        print(f"Duplicate alias master loaded from {path.name}: {len(alias_map)} active aliases")
        return alias_map

    print("Duplicate alias master not found; using built-in fallback alias map")
    return default_map


DUPLICATE_RATIO_ALIAS_MAP = _load_duplicate_ratio_alias_master(DUPLICATE_RATIO_ALIAS_MAP)

# ============================================================
# 1. inventory에서 항목 lookup helper (override 우선)
# ============================================================

# ============================================================
# External manual override from bundle-root account_item_master_full.csv
# ============================================================
def _load_external_manual_overrides():
    bundle_root = Path(__file__).resolve().parents[1]
    master_path = bundle_root / "account_item_master_full.csv"
    manual_map_path = bundle_root / "manual_standard_item_map.csv"

    src_path = master_path if master_path.exists() else manual_map_path
    if not src_path.exists():
        print("External manual override: no root master/manual map found")
        return {}

    df = pd.read_csv(src_path, encoding="utf-8-sig")

    role_col = "stage2_role" if "stage2_role" in df.columns else "standard_item"
    stmt_col = "statement_type" if "statement_type" in df.columns else "statement"

    required = [role_col, stmt_col, "item_code"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"External manual override skipped: missing columns={missing}")
        print(f"Available columns: {list(df.columns)}")
        return {}

    name_col = None
    for c in ["item_name_clean", "item_name", "stage2_name_match", "original_column_name"]:
        if c in df.columns:
            name_col = c
            break

    m = df.copy()
    m[role_col] = m[role_col].astype(str).str.strip()
    m = m[
        m[role_col].notna()
        & (m[role_col] != "")
        & (m[role_col].str.lower() != "nan")
    ]

    external = {}
    for _, r in m.iterrows():
        role = str(r.get(role_col, "")).strip()
        stmt = str(r.get(stmt_col, "")).strip()
        code = str(r.get("item_code", "")).strip()

        if not role or not stmt or not code or code.lower() == "nan":
            continue

        item_name = ""
        if name_col:
            item_name = str(r.get(name_col, "")).strip()
            if item_name.lower() == "nan":
                item_name = ""

        external[role] = {
            "item_code": _canonicalize_item_code_value(code),
            "item_name": item_name,
            "statement": stmt,
            "note": f"external_override_from_{src_path.name}",
        }

    print(f"External manual override loaded from {src_path.name}: {len(external)} roles")
    return external

_EXTERNAL_MANUAL_OVERRIDE = _load_external_manual_overrides()
MANUAL_OVERRIDE.update(_EXTERNAL_MANUAL_OVERRIDE)


def lookup_item(role_hint, prefer_stmt=None):
    """role_hint → inventory item dict 반환 (manual override 우선)"""
    # Manual override 우선: primary must win over lower-coverage aliases; aliases are row-level fallback only.
    if role_hint in MANUAL_OVERRIDE:
        ov = MANUAL_OVERRIDE[role_hint].copy()
        ov['item_code'] = _canonicalize_item_code_value(ov.get('item_code', ''))
        alias_order = ITEM_CODE_ALIASES.get(role_hint, [ov['item_code']])
        alias_order = [c for c in alias_order if c]
        inv_stmt = inventory[inventory['statement_type'] == ov['statement']] if 'statement_type' in inventory.columns else inventory
        for code in alias_order:
            match = inv_stmt[inv_stmt['item_code'] == code]
            if len(match) > 0:
                r = match.sort_values('non_missing_rate', ascending=False).iloc[0]
                return {'item_code': code, 'item_name': r['item_name'], 'statement': r['statement_type'], 'non_missing_rate': r['non_missing_rate']}
        # If the primary override is not present in inventory, return the primary explicitly as a hard semantic mapping.
        # Task6 performs row-level coalescing from aliases when the primary column is absent.
        return {'item_code': ov['item_code'], 'item_name': ov.get('item_name', role_hint), 'statement': ov.get('statement', prefer_stmt or ''), 'non_missing_rate': 0.0}

    # Heuristic: candidate_role_hint 사용
    cand = inventory[inventory['candidate_role_hint'] == role_hint]
    if prefer_stmt:
        cand_filtered = cand[cand['statement_type'] == prefer_stmt]
        if len(cand_filtered) > 0:
            cand = cand_filtered

    if len(cand) == 0:
        return None
    # high non_missing rate + summary_marker + lowest indent
    cand = cand.sort_values(
        ['non_missing_rate', 'is_summary_marker', 'indent_level'],
        ascending=[False, False, True]
    )
    r = cand.iloc[0]
    return {
        'item_code': r['item_code'] if pd.notna(r['item_code']) else '',
        'item_name': r['item_name'],
        'statement': r['statement_type'],
        'non_missing_rate': r['non_missing_rate'],
    }

# ============================================================
# 2. Term → role_hint 매핑 (확장)
# ============================================================
TERM_TO_ROLE = {
    '매출액': '매출액', '매출원가': '매출원가', '매출총이익': '매출총이익',
    '영업이익': '영업이익', '정상영업이익': '정상영업이익',
    '법인세차감전순이익': '법인세차감전순이익', '세전이익': '법인세차감전순이익',
    '당기순이익': '당기순이익', '순이익': '당기순이익',
    '계속사업이익': '계속사업이익', '계속사업손익': '계속사업이익', '계속영업이익': '계속사업이익', '계속영업이익(손실)': '계속사업이익', '총포괄이익': '총포괄이익',
    '금융비용': '금융비용', '이자비용': '금융비용',
    '감가상각비': '감가상각비', '무형자산상각비': '무형자산상각비',
    '총자산': '총자산', '유동자산': '유동자산', '비유동자산': '비유동자산',
    '재고자산': '재고자산', '매출채권': '매출채권',
    '현금및현금성자산': '현금및현금성자산', '현금성자산': '현금및현금성자산',
    '유동부채': '유동부채', '비유동부채': '비유동부채', '부채총계': '부채총계',
    '자기자본': '자본총계', '자본총계': '자본총계',
    '자본금': '자본금', '이익잉여금': '이익잉여금',
    '단기차입금': '단기차입금', '장기차입금': '장기차입금', '사채': '사채',
    '매입채무': '매입채무',
    '영업활동현금흐름': '영업활동현금흐름', 'OCF': '영업활동현금흐름',
    '투자활동현금흐름': '투자활동현금흐름', '재무활동현금흐름': '재무활동현금흐름',
}

# ============================================================
# 3. Derived items (계산식 정의)
# ============================================================
DERIVED = {
    'EBIT': {
        'formula': '영업이익',
        'components': ['영업이익'], 'fallback': None,
        'note': 'NICE 표시 영업이익(계산수치) 사용'},
    'EBITDA': {
        'formula': '영업이익 + 감가상각비 + 무형자산상각비',
        'components': ['영업이익', '감가상각비', '무형자산상각비'], 'fallback': None,
        'note': '무형자산상각비 결측 시 0 처리, 감가상각비 결측 시 EBITDA = NaN'},
    'NOPAT': {
        'formula': '영업이익 × (1 - 유효세율). 유효세율 = 1 - 당기순이익/세전이익',
        'components': ['영업이익', '법인세차감전순이익', '당기순이익'], 'fallback': None,
        'note': '세전이익≤0 firm은 default 25% 적용'},
    'CAPEX': {
        'formula': '현금흐름표 유형자산 취득',
        'components': ['CAPEX_proxy'], 'fallback': None,
        'note': 'NICE 항목명: 유형자산의 증가 (현금흐름표). 음수로 저장될 수도'},
    'FCF': {
        'formula': '영업활동현금흐름 - CAPEX',
        'components': ['영업활동현금흐름', 'CAPEX_proxy'], 'fallback': None,
        'note': 'CAPEX 부호 처리 주의 (NICE는 양수 표기)'},
    '총차입금': {
        'formula': '단기차입금 + 장기차입금 + 사채 + 유동성장기부채',
        'components': ['단기차입금', '장기차입금', '사채'],
        'extra_components': ['유동성장기부채'],  # 추가 항목
        'fallback': None,
        'note': '유동성장기부채 = U01A811027400. 결측 시 0 처리'},
    '순차입금': {
        'formula': '총차입금 - 현금및현금성자산',
        'components': ['단기차입금', '장기차입금', '사채', '현금및현금성자산'],
        'extra_components': ['유동성장기부채'],
        'fallback': None, 'note': '단기금융상품 추가 차감 옵션 (U01A111043200)'},
    '순운전자본': {
        'formula': '유동자산 - 유동부채',
        'components': ['유동자산', '유동부채'], 'fallback': None},
    '순영업운전자본': {
        'formula': '매출채권 + 재고자산 - 매입채무',
        'components': ['매출채권', '재고자산', '매입채무'], 'fallback': None},
    '영업운전자본': {
        'formula': '매출채권 + 재고자산 - 매입채무',
        'components': ['매출채권', '재고자산', '매입채무'], 'fallback': None},
    '유이자부채': {
        'formula': '단기차입금 + 장기차입금 + 사채',
        'components': ['단기차입금', '장기차입금', '사채'], 'fallback': None},
    '금융부채': {
        'formula': '단기차입금 + 장기차입금 + 사채 (= 유이자부채)',
        'components': ['단기차입금', '장기차입금', '사채'], 'fallback': None},
    '단기금융부채': {
        'formula': '단기차입금 + 유동성장기부채',
        'components': ['단기차입금'],
        'extra_components': ['유동성장기부채'], 'fallback': None},
    '장기금융부채': {
        'formula': '장기차입금 + 사채',
        'components': ['장기차입금', '사채'], 'fallback': None},
    '즉시가용유동성': {
        'formula': '현금및현금성자산 + 단기금융상품',
        'components': ['현금및현금성자산'],
        'extra_components': ['단기금융상품'], 'fallback': None,
        'note': '단기금융상품 = U01A111043200'},
    '당좌자산': {
        'formula': '유동자산 - 재고자산',
        'components': ['유동자산', '재고자산'], 'fallback': None,
        'note': 'NICE raw에 직접 항목 없음. 유동성 비율 계산용 derive'},
}

# Extra component manual mapping
EXTRA_MAPPING = {
    '유동성장기부채': {
        'item_code': 'U01A811027400', 'item_name': '유동성장기부채',
        'statement': '재무상태표'},
    '단기금융상품': {
        'item_code': 'U01A111043200', 'item_name': '단기금융상품(금융기관예치금)',
        'statement': '재무상태표'},
}

# Ambiguous (계산 불가/임의) terms
AMBIGUOUS_TERMS = {
    '배당금', '배당성향', 'ROA 3개년', '영업이익률 3개년',
    '흑자연도 수', 'DSO + DIO - DPO', 'max(OCF, EBITDA)',
    '연구개발비', '유형자산', '무형자산', '자본잉여금',
    '평균경영자본', '평균사용자본', '평균영업자산', '평균투하자본',
    '종업원수', '기간', '3', '-',
}

# ============================================================
# 4. ROLE_TO_ITEM 사전 (override 적용)
# ============================================================
ROLE_TO_STMT_PREFER = {
    '매출액': '손익계산서', '매출원가': '손익계산서', '매출총이익': '손익계산서',
    '영업이익': '손익계산서', '정상영업이익': '손익계산서',
    '법인세차감전순이익': '손익계산서', '당기순이익': '손익계산서',
    '계속사업이익': '손익계산서', '총포괄이익': '손익계산서',
    '금융비용': '손익계산서', '감가상각비': '손익계산서',
    '무형자산상각비': '손익계산서',
    '총자산': '재무상태표', '유동자산': '재무상태표', '비유동자산': '재무상태표',
    '재고자산': '재무상태표', '매출채권': '재무상태표',
    '현금및현금성자산': '재무상태표', '유동부채': '재무상태표',
    '비유동부채': '재무상태표', '부채총계': '재무상태표',
    '자본총계': '재무상태표', '자본금': '재무상태표', '이익잉여금': '재무상태표',
    '단기차입금': '재무상태표', '장기차입금': '재무상태표',
    '사채': '재무상태표', '매입채무': '재무상태표',
    '영업활동현금흐름': '현금흐름표', '투자활동현금흐름': '현금흐름표',
    '재무활동현금흐름': '현금흐름표', 'CAPEX_proxy': '현금흐름표',
}

ROLE_TO_ITEM = {}
print("표준 항목 매핑 결과:")
all_roles = set(TERM_TO_ROLE.values()) | set(ROLE_TO_STMT_PREFER.keys())
for role in sorted(all_roles):
    item = lookup_item(role, prefer_stmt=ROLE_TO_STMT_PREFER.get(role))
    if item:
        ROLE_TO_ITEM[role] = item
        flag = '✓' if item['non_missing_rate'] >= 0.5 else '⚠️'
        print(f"  {flag} {role:<20s} → [{item['statement']}] {item['item_code']:<15s} "
              f"{item['item_name']:<25s} ({item['non_missing_rate']:.1%})")
    else:
        ROLE_TO_ITEM[role] = None
        print(f"  ❌ {role:<20s} → NOT FOUND")

# Extra components
for role, info in EXTRA_MAPPING.items():
    match = inventory[
        (inventory['statement_type'] == info['statement']) &
        (inventory['item_code'] == info['item_code'])
    ]
    if len(match) > 0:
        r = match.iloc[0]
        ROLE_TO_ITEM[role] = {
            'item_code': r['item_code'], 'item_name': r['item_name'],
            'statement': r['statement_type'], 'non_missing_rate': r['non_missing_rate']
        }
        print(f"  ✓ {role:<20s} → [{info['statement']}] {info['item_code']:<15s} "
              f"{info['item_name']} ({r['non_missing_rate']:.1%})")

# Account item master audit required by Stage00_02 v10
_account_rows = []
for _role, _item in sorted(ROLE_TO_ITEM.items()):
    if _item:
        _account_rows.append({
            'standard_item': _role, 'stage2_role': _role, 'statement_type': _item.get('statement', ''),
            'item_code': _canonicalize_item_code_value(_item.get('item_code', '')),
            'item_name': _item.get('item_name', ''), 'non_missing_rate': _item.get('non_missing_rate', 0.0),
            'aliases': '|'.join(ITEM_CODE_ALIASES.get(_role, [])),
        })
account_master = pd.DataFrame(_account_rows)
account_master.to_csv(OUT / 'account_item_master_full.csv', index=False, encoding='utf-8-sig')
for _role, _primary in {'매출액':'U01B100000000', '계속사업이익':'U01B800000000', '무형자산상각비':'U01B350014300'}.items():
    _hit = account_master[account_master['stage2_role'] == _role] if not account_master.empty else pd.DataFrame()
    if _hit.empty or str(_hit.iloc[0]['item_code']) != _primary:
        raise RuntimeError(f'Stage00_02 primary item mapping violation: {_role} expected {_primary}, got {None if _hit.empty else _hit.iloc[0].to_dict()}')

# ============================================================
# 5. resolve_term + Dictionary 생성 (Task 2와 동일)
# ============================================================
def resolve_term(term):
    if pd.isna(term):
        return {'status': 'missing'}
    t = str(term).strip()
    lag_offset = 0
    growth_term = False
    if t.endswith('_t-1'):
        lag_offset = 1; t = t[:-4]
    elif t.endswith('_t-3'):
        lag_offset = 3; t = t[:-4]
    elif t.endswith('_t'):
        t = t[:-2]
    if '증가율' in t:
        growth_term = True; t = t.replace('증가율', '')
    is_average = t.startswith('평균')
    if is_average:
        t = t[2:]
        if t.endswith(' × 365'):
            t = t[:-6]

    if t in TERM_TO_ROLE:
        role = TERM_TO_ROLE[t]
        if role in ROLE_TO_ITEM and ROLE_TO_ITEM[role]:
            it = ROLE_TO_ITEM[role]
            return {
                'status': 'mapped',
                'item_code': it['item_code'], 'item_name': it['item_name'],
                'statement': it['statement'], 'non_missing_rate': it['non_missing_rate'],
                'lag': lag_offset, 'is_average': is_average,
                'is_growth': growth_term, 'is_derived': False,
            }
        else:
            return {'status': 'role_not_found_in_inventory', 'role': role,
                    'lag': lag_offset, 'is_average': is_average, 'is_growth': growth_term}

    if t in DERIVED:
        comp_codes = []
        comp_names = []
        comp_status = 'derivable'
        all_comps = list(DERIVED[t]['components']) + list(DERIVED[t].get('extra_components', []))
        for comp in all_comps:
            if comp in ROLE_TO_ITEM and ROLE_TO_ITEM[comp]:
                comp_codes.append(ROLE_TO_ITEM[comp]['item_code'])
                comp_names.append(ROLE_TO_ITEM[comp]['item_name'])
            else:
                if comp in DERIVED[t]['components']:  # 핵심 component 빠짐
                    comp_status = 'partial_components_missing'
        return {
            'status': comp_status,
            'item_code': '+'.join(comp_codes) if comp_codes else 'DERIVED',
            'item_name': t, 'statement': 'DERIVED',
            'derived_formula': DERIVED[t]['formula'],
            'derived_components': '|'.join(all_comps),
            'derived_note': DERIVED[t].get('note', ''),
            'lag': lag_offset, 'is_average': is_average,
            'is_growth': growth_term, 'is_derived': True,
        }

    if t in AMBIGUOUS_TERMS or any(amb in t for amb in AMBIGUOUS_TERMS):
        return {'status': 'ambiguous', 'item_name': t, 'lag': lag_offset,
                'is_average': is_average, 'is_growth': growth_term}

    if any(op in t for op in ['+', '-', '×', '/']):
        return {'status': 'compound_expression', 'item_name': t,
                'lag': lag_offset, 'is_average': is_average, 'is_growth': growth_term}

    return {'status': 'unknown', 'item_name': t, 'lag': lag_offset,
            'is_average': is_average, 'is_growth': growth_term}

ratio_rows = []
for idx, row in candidates.iterrows():
    rid = f"R{int(row['No']):03d}"
    cat = row['평가항목']; name_ko = row['후보비율명']
    formula = row['공식']; num_term = row['분자']; den_term = row['분모']
    direction = row['기대방향']; unit_orig = row['단위']; priority = row['우선순위']

    num = resolve_term(num_term); den = resolve_term(den_term)
    nstat, dstat = num.get('status'), den.get('status')

    if nstat == 'mapped' and dstat == 'mapped':
        if num.get('lag', 0) == 0 and den.get('lag', 0) == 0 \
           and not num.get('is_average') and not den.get('is_average') \
           and not num.get('is_growth'):
            availability = 'available_direct'
        else:
            availability = 'available_with_lag'
    elif nstat in ('mapped', 'derivable') and dstat in ('mapped', 'derivable'):
        availability = 'available_derived'
    elif 'ambiguous' in [nstat, dstat]:
        availability = 'ambiguous'
    elif 'compound_expression' in [nstat, dstat]:
        availability = 'compound_expression'
    elif 'partial_components_missing' in [nstat, dstat] or \
         'role_not_found_in_inventory' in [nstat, dstat]:
        availability = 'partial_data'
    elif nstat == 'missing' or dstat == 'missing':
        availability = 'missing_term'
    else:
        availability = 'unknown'

    requires_lag = (num.get('lag', 0) > 0) or (den.get('lag', 0) > 0) or \
                   num.get('is_average', False) or den.get('is_average', False) or \
                   num.get('is_growth', False) or den.get('is_growth', False)
    requires_avg_denom = den.get('is_average', False)
    requires_growth = num.get('is_growth', False) or den.get('is_growth', False) or \
                      '증가율' in name_ko

    unit_out = unit_orig if pd.notna(unit_orig) else '%'
    den_zero_policy = 'NaN'
    den_neg_policy = 'NaN' if cat == '성장성' else 'compute_with_flag'

    ratio_rows.append({
        'ratio_id': rid, 'category': cat, 'sub_category': row.get('세부영역', ''),
        'ratio_name_ko': name_ko, 'ratio_name_en': '', 'formula_text': formula,
        'numerator_term_orig': str(num_term),
        'numerator_item_code': num.get('item_code', ''),
        'numerator_item_name': num.get('item_name', ''),
        'numerator_statement_source': num.get('statement', ''),
        'numerator_lag': num.get('lag', 0),
        'numerator_is_average': num.get('is_average', False),
        'numerator_is_growth': num.get('is_growth', False),
        'numerator_status': num.get('status', ''),
        'numerator_derived_formula': num.get('derived_formula', ''),
        'numerator_derived_components': num.get('derived_components', ''),
        'denominator_term_orig': str(den_term),
        'denominator_item_code': den.get('item_code', ''),
        'denominator_item_name': den.get('item_name', ''),
        'denominator_statement_source': den.get('statement', ''),
        'denominator_lag': den.get('lag', 0),
        'denominator_is_average': den.get('is_average', False),
        'denominator_is_growth': den.get('is_growth', False),
        'denominator_status': den.get('status', ''),
        'denominator_derived_formula': den.get('derived_formula', ''),
        'denominator_derived_components': den.get('derived_components', ''),
        'requires_lag': requires_lag,
        'requires_average_denominator': requires_avg_denom,
        'requires_growth_base': requires_growth,
        'unit_output': unit_out, 'expected_good_direction': direction,
        'denominator_zero_policy': den_zero_policy,
        'negative_denominator_policy': den_neg_policy,
        'priority': priority, 'availability_status': availability,
        'notes': row.get('비고', ''),
    })

dict_df = pd.DataFrame(ratio_rows)

# ------------------------------------------------------------------
# R-code metadata / duplicate alias master outputs
# ------------------------------------------------------------------
# Stage 2 is the source-of-truth layer for internal R*** ratio codes. Do not
# delete duplicate aliases here: keep the calculable/auditable definition, but
# expose a machine-readable master so Stage 3, Oracle params, simulator, and
# evaluator can distinguish canonical ratios from aliases.
for _col, _default in {
    "duplicate_alias_of": "",
    "duplicate_alias_type": "",
    "duplicate_alias_reason": "",
    "duplicate_alias_source": "",
    "duplicate_alias_policy_note": "",
    "stage3_exclude": False,
}.items():
    if _col not in dict_df.columns:
        dict_df[_col] = _default

_alias_log = []
for _alias_id, _meta in DUPLICATE_RATIO_ALIAS_MAP.items():
    _mask = dict_df["ratio_id"].astype(str) == _alias_id
    if _mask.any():
        dict_df.loc[_mask, "duplicate_alias_of"] = _meta["canonical"]
        dict_df.loc[_mask, "duplicate_alias_type"] = _meta.get("type", "exact_value_alias")
        dict_df.loc[_mask, "duplicate_alias_reason"] = _meta["reason"]
        dict_df.loc[_mask, "duplicate_alias_source"] = _meta.get("source", "duplicate_ratio_alias_master")
        dict_df.loc[_mask, "duplicate_alias_policy_note"] = _meta.get("policy_note", "")
        dict_df.loc[_mask, "stage3_exclude"] = bool(_meta.get("stage3_exclude", True))
        _alias_log.append({
            "alias_ratio_id": _alias_id,
            "canonical_ratio_id": _meta["canonical"],
            "duplicate_alias_type": _meta.get("type", "exact_value_alias"),
            "stage3_exclude": bool(_meta.get("stage3_exclude", True)),
            "reason": _meta["reason"],
            "source": _meta.get("source", "duplicate_ratio_alias_master"),
            "policy_note": _meta.get("policy_note", ""),
            "n_rows_marked": int(_mask.sum()),
        })

# Stable formula/account fingerprint for audit and simulator grounding.
def _join_codes(*vals):
    codes = [str(v).strip() for v in vals if pd.notna(v) and str(v).strip() not in ("", "nan", "None")]
    return "|".join(codes)

dict_df["formula_item_code_fingerprint"] = dict_df.apply(
    lambda r: _join_codes(
        r.get("numerator_item_code", ""),
        r.get("denominator_item_code", ""),
        r.get("numerator_derived_components", ""),
        r.get("denominator_derived_components", ""),
    ),
    axis=1,
)
dict_df["is_canonical_ratio"] = dict_df["duplicate_alias_of"].astype(str).str.len().eq(0)
dict_df["canonical_ratio_id"] = dict_df.apply(
    lambda r: r["duplicate_alias_of"] if str(r.get("duplicate_alias_of", "")).strip() else r["ratio_id"],
    axis=1,
)

# Master data outputs: these are intended as metadata, not model outputs.
# They make the generated R-code system auditable and reusable downstream.
ratio_master_cols = [
    "ratio_id", "canonical_ratio_id", "is_canonical_ratio", "stage3_exclude",
    "duplicate_alias_of", "duplicate_alias_type", "duplicate_alias_reason",
    "duplicate_alias_source", "duplicate_alias_policy_note",
    "category", "ratio_name_ko", "numerator_term_orig", "numerator_item_code",
    "numerator_item_name", "numerator_statement_source", "numerator_lag",
    "numerator_is_average", "numerator_is_growth", "numerator_status",
    "numerator_derived_formula", "numerator_derived_components",
    "denominator_term_orig", "denominator_item_code", "denominator_item_name",
    "denominator_statement_source", "denominator_lag", "denominator_is_average",
    "denominator_is_growth", "denominator_status", "denominator_derived_formula",
    "denominator_derived_components", "requires_lag", "requires_average_denominator",
    "requires_growth_base", "unit_output", "expected_good_direction",
    "denominator_zero_policy", "negative_denominator_policy", "priority",
    "availability_status", "formula_item_code_fingerprint", "notes",
]
ratio_master_cols = [c for c in ratio_master_cols if c in dict_df.columns]
ratio_code_master = dict_df[ratio_master_cols].copy()
ratio_code_master.to_csv(OUT / "ratio_code_master_stage2.csv", index=False, encoding="utf-8-sig")
ratio_code_master.to_json(OUT / "ratio_code_master_stage2.json", orient="records", force_ascii=False, indent=2)

if _alias_log:
    alias_master_df = pd.DataFrame(_alias_log).sort_values(["canonical_ratio_id", "alias_ratio_id"])
    alias_master_df.to_csv(OUT / "duplicate_ratio_alias_log.csv", index=False, encoding="utf-8-sig")
    alias_master_df.to_csv(OUT / "duplicate_ratio_alias_master_resolved.csv", index=False, encoding="utf-8-sig")
    alias_map_json = {
        row["alias_ratio_id"]: {
            "canonical_ratio_id": row["canonical_ratio_id"],
            "duplicate_alias_type": row["duplicate_alias_type"],
            "stage3_exclude": bool(row["stage3_exclude"]),
            "reason": row["reason"],
            "source": row.get("source", ""),
            "policy_note": row.get("policy_note", ""),
        }
        for _, row in alias_master_df.iterrows()
    }
    with open(OUT / "duplicate_ratio_alias_map.json", "w", encoding="utf-8") as f:
        json.dump(alias_map_json, f, ensure_ascii=False, indent=2)
    with open(OUT / "simulator_ratio_alias_map.json", "w", encoding="utf-8") as f:
        json.dump({k: v["canonical_ratio_id"] for k, v in alias_map_json.items()}, f, ensure_ascii=False, indent=2)
    print(f"Duplicate alias policy: {len(alias_master_df)} aliases marked and master metadata written")
else:
    pd.DataFrame(columns=[
        "alias_ratio_id", "canonical_ratio_id", "duplicate_alias_type", "stage3_exclude",
        "reason", "source", "policy_note", "n_rows_marked"
    ]).to_csv(OUT / "duplicate_ratio_alias_master_resolved.csv", index=False, encoding="utf-8-sig")
    with open(OUT / "duplicate_ratio_alias_map.json", "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    with open(OUT / "simulator_ratio_alias_map.json", "w", encoding="utf-8") as f:
        json.dump({}, f, ensure_ascii=False, indent=2)

dict_df.to_csv(OUT / 'financial_ratio_formula_dictionary_draft.csv',
                index=False, encoding='utf-8-sig')

print(f"\nDictionary draft v2: {len(dict_df)}개 ratio")
print(f"\nCategory별:")
print(dict_df.groupby('category').size().to_string())
print(f"\nAvailability status:")
print(dict_df['availability_status'].value_counts().to_string())

calculable = dict_df[dict_df['availability_status'].isin(
    ['available_direct', 'available_with_lag', 'available_derived'])]
print(f"\n계산 가능 ratio: {len(calculable)}개")
print(calculable.groupby('category').size().to_string())
