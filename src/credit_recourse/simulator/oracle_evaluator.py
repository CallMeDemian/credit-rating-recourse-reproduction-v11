"""
oracle_evaluator.py — Oracle PV 평가.
raw value → bin → score → block aggregate → R_score → grade.

Input oracle_vars dict는 어떤 R코드 키가 들어와도 자동 canonicalize 후 처리.
예: {'R086': 0.003, ...} → evaluate 내부에서 {'R085': 0.003, ...}로 변환.
"""
import json
import bisect
from typing import Dict, Optional, List

try:
    from .ratio_alias import canonicalize_dict, canonicalize_list
except ImportError:  # direct script/local import fallback
    from ratio_alias import canonicalize_dict, canonicalize_list


GRADE_BOUNDARIES_ORDER = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'CC', 'C']
GRADE_TO_NUM = {'AAA': 1, 'AA': 2, 'A': 3, 'BBB': 4,
                'BB': 5, 'B': 6, 'CCC': 7, 'CC': 8, 'C': 9, 'D': 10}


def _normalize_var_list(raw):
    vals = []
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = list(raw.values())
    if not isinstance(raw, list):
        raw = [raw]
    for x in raw:
        if isinstance(x, str):
            v = x.strip()
        elif isinstance(x, dict):
            v = None
            for k in ['variable_id', 'variable', 'feature', 'name', 'column', 'id']:
                if x.get(k) is not None and str(x.get(k)).strip():
                    v = str(x.get(k)).strip(); break
        else:
            v = str(x).strip() if x is not None else None
        if v and v not in vals:
            vals.append(v)
    return vals


def _ensure_10_grade_boundaries(boundaries: Dict[str, float]) -> Dict[str, float]:
    """Return monotone AAA..C thresholds; D is fallback below C.

    This mirrors the final Oracle contract used by Alpha/Beta/Gamma. Legacy
    simulator callers may still pass only AAA..CCC boundaries; CC/C are then
    deterministically extrapolated from the lower-tail spacing.
    """
    b = {str(k): float(v) for k, v in dict(boundaries or {}).items() if k in GRADE_TO_NUM and k != 'D'}
    known = [(g, b[g]) for g in GRADE_BOUNDARIES_ORDER if g in b]
    gaps = []
    for (_, hi), (_, lo) in zip(known, known[1:]):
        gap = hi - lo
        if gap > 0:
            gaps.append(gap)
    tail_gap = sorted(gaps)[len(gaps)//2] if gaps else 5.0
    if 'CCC' not in b:
        b['CCC'] = b.get('B', 0.0) - tail_gap
    if 'CC' not in b:
        b['CC'] = b['CCC'] - tail_gap
    if 'C' not in b:
        b['C'] = b['CC'] - tail_gap
    prev = None
    for g in GRADE_BOUNDARIES_ORDER:
        if g not in b:
            if prev is None:
                continue
            b[g] = prev - tail_gap
        if prev is not None and b[g] >= prev:
            b[g] = prev - max(tail_gap, 1e-6)
        prev = b[g]
    return b


def r_score_to_grade(r_score: float, boundaries: Dict[str, float]) -> str:
    b = _ensure_10_grade_boundaries(boundaries)
    for g in GRADE_BOUNDARIES_ORDER:
        if r_score >= b[g]:
            return g
    return 'D'


class OracleEvaluator:
    # Final contract: selected variables are loaded dynamically from params.
    FINANCIAL_VARS = []
    NONFINANCIAL_VARS = []

    def __init__(self, params_path: str, bin_score_path: str):
        with open(params_path, encoding='utf-8') as f:
            self.params = json.load(f)
        with open(bin_score_path, encoding='utf-8') as f:
            self.bin_score_table = json.load(f)

        raw_vars = []
        for key in ['selected_variables', 'required_variables', 'features', 'feature_names', 'variables', 'variable_params']:
            raw_vars.extend(_normalize_var_list(self.params.get(key)))
        if not raw_vars and 'bin_edges' in self.params:
            raw_vars = list(self.params.get('bin_edges', {}).keys())
        selected = canonicalize_list(raw_vars)
        if not selected:
            raise ValueError('OracleEvaluator cannot resolve selected variables from params')
        self.FINANCIAL_VARS = [v for v in selected if str(v).startswith('R')]
        self.NONFINANCIAL_VARS = [v for v in selected if v not in self.FINANCIAL_VARS]
        missing_edges = [v for v in selected if v not in self.params.get('bin_edges', {})]
        if missing_edges:
            raise KeyError(f'OracleEvaluator params missing bin_edges for selected variables: {missing_edges}')

        self.bin_edges = {
            v: self.params['bin_edges'][v]['edges']
            for v in self.FINANCIAL_VARS + self.NONFINANCIAL_VARS
        }
        self.is_low_unique = {
            v: self.params['bin_edges'][v]['is_low_unique']
            for v in self.FINANCIAL_VARS + self.NONFINANCIAL_VARS
        }
        self.winsor = {
            w['variable_id']: (w['p01_dev'], w['p99_dev'])
            for w in self.params['winsorization_params']
        }
        self.weights = self.params['optimized_weights']
        self.block_norm = self.params['block_normalization']
        self.block_weights = {
            'financial': self.params['combined_weights']['financial'],
            'nonfinancial': self.params['combined_weights']['nonfinancial'],
        }
        self.boundaries = self.params['boundaries']
        self.imputation_map = self.params['imputation_map']

    def _value_to_bin(self, var: str, value: Optional[float]) -> Optional[int]:
        if value is None:
            return None
        edges = self.bin_edges[var]

        if self.is_low_unique[var]:
            for i, e in enumerate(edges):
                if abs(value - e) < 1e-9:
                    return i
            return min(range(len(edges)), key=lambda i: abs(edges[i] - value))

        p01, p99 = self.winsor[var]
        v = max(p01, min(p99, value))
        idx = bisect.bisect_right(edges, v) - 1
        idx = max(0, min(len(edges) - 2, idx))
        return idx

    def _bin_to_score(self, var: str, bin_idx: Optional[int]) -> float:
        if bin_idx is None:
            return self.imputation_map.get(var, 50.0)
        table = self.bin_score_table[var]
        if str(bin_idx) in table:
            return table[str(bin_idx)]
        avail = sorted(int(k) for k in table.keys())
        closest = min(avail, key=lambda x: abs(x - bin_idx))
        return table[str(closest)]

    def aggregate_block(self, var_scores, var_list, block_name) -> float:
        total_w = sum(self.weights[v] for v in var_list)
        weighted = sum(self.weights[v] * var_scores[v] for v in var_list) / total_w
        bn = self.block_norm[block_name]
        p01, p99 = bn['p01'], bn['p99']
        v = max(p01, min(p99, weighted))
        if p99 - p01 > 1e-9:
            return (v - p01) / (p99 - p01) * 100.0
        return weighted

    def evaluate(self, oracle_vars: Dict[str, Optional[float]]) -> Dict:
        # Input canonicalize — R086, R131 등 alias가 들어와도 자동 처리
        oracle_vars = canonicalize_dict(oracle_vars)

        var_scores = {}
        for v in self.FINANCIAL_VARS + self.NONFINANCIAL_VARS:
            bin_idx = self._value_to_bin(v, oracle_vars.get(v))
            var_scores[v] = self._bin_to_score(v, bin_idx)

        fin_score = self.aggregate_block(var_scores, self.FINANCIAL_VARS, 'financial')
        nonfin_score = self.aggregate_block(var_scores, self.NONFINANCIAL_VARS, 'nonfinancial')
        r_score = (
            self.block_weights['financial'] * fin_score
            + self.block_weights['nonfinancial'] * nonfin_score
        )
        grade = r_score_to_grade(r_score, self.boundaries)
        grade_num = GRADE_TO_NUM.get(grade, 10)

        return {
            'R_score': r_score,
            'R_grade': grade,
            'R_grade_num': grade_num,
            'financial_block_score': fin_score,
            'nonfinancial_block_score': nonfin_score,
            'var_scores': var_scores,
        }


def compute_pv(evaluator, oracle_vars_t, oracle_vars_t1) -> Dict:
    eval_t = evaluator.evaluate(oracle_vars_t)
    eval_t1 = evaluator.evaluate(oracle_vars_t1)
    return {
        'oracle_score_delta': eval_t1['R_score'] - eval_t['R_score'],
        'oracle_grade_delta': eval_t['R_grade_num'] - eval_t1['R_grade_num'],
        'R_score_t': eval_t['R_score'],
        'R_score_t1': eval_t1['R_score'],
        'R_grade_t': eval_t['R_grade'],
        'R_grade_t1': eval_t1['R_grade'],
        'eval_t': eval_t,
        'eval_t1': eval_t1,
    }
