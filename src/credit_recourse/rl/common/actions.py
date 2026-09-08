from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .io import config_root, final_root, load_yaml

@dataclass(frozen=True)
class ActionSpace:
    columns: list[str]
    bounds: dict[str, tuple[float, float]]
    fixed_candidates: dict[str, dict[str, float]]
    train_labels: list[str]
    row_conditional_baselines: list[str]
    final_rl_label: str
    scenario_candidates: dict[str, dict[str, float]]
    diagnostic_candidates: dict[str, dict[str, float]]
    candidate_library_hash: str = ""
    final_action_contract_hash: str = ""

    @property
    def names(self) -> list[str]:
        return list(self.fixed_candidates.keys())

    @property
    def stage6_fixed_policy_names(self) -> list[str]:
        # Main paper evaluation defaults to the 11 v32 main labels only.
        # Scenario/diagnostic candidates are opt-in robustness extras.
        return list(self.fixed_candidates.keys())

    def stage6_policy_names(self, include_extras: bool = False) -> list[str]:
        names = list(self.fixed_candidates.keys())
        if include_extras:
            names += list(self.scenario_candidates.keys()) + list(self.diagnostic_candidates.keys())
        return names

    def bound_width(self, col: str) -> float:
        lo, hi = self.bounds[col]
        return max(abs(lo), abs(hi), 1e-12)

    def _dict_for(self, name: str) -> dict[str, float]:
        if name in self.fixed_candidates:
            return self.fixed_candidates[name]
        if name in self.scenario_candidates:
            return self.scenario_candidates[name]
        if name in self.diagnostic_candidates:
            return self.diagnostic_candidates[name]
        raise KeyError(f"Unknown candidate: {name}")

    def candidate_vector(self, name: str) -> np.ndarray:
        vec = self._dict_for(name)
        return np.array([float(vec.get(c, 0.0) or 0.0) for c in self.columns], dtype=np.float32)

    def frame(self, include_stage6_extras: bool = False) -> pd.DataFrame:
        rows = []
        source = dict(self.fixed_candidates)
        if include_stage6_extras:
            source.update(self.scenario_candidates)
            source.update(self.diagnostic_candidates)
        for name, vec in source.items():
            row = {"candidate_id": name}
            row.update({c: float(vec.get(c, 0.0) or 0.0) for c in self.columns})
            if "tier" in vec: row["tier"] = vec.get("tier")
            if "paper_role" in vec: row["paper_role"] = vec.get("paper_role")
            rows.append(row)
        return pd.DataFrame(rows)

def _numeric_candidate_values(raw: dict, columns: list[str]) -> dict[str, float]:
    return {c: float(raw.get(c, 0.0) or 0.0) for c in columns} | {k: v for k, v in raw.items() if k not in columns}

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def active_config_hashes(project_root: Path) -> dict[str, str]:
    cfg = config_root(project_root)
    cand_path = cfg / "final_candidate_library.yaml"
    action_path = cfg / "final_action_contract.yaml"
    if not cand_path.exists():
        raise FileNotFoundError(f"Missing final candidate library: {cand_path}")
    if not action_path.exists():
        raise FileNotFoundError(f"Missing final action contract: {action_path}")
    return {
        "candidate_library_hash": _sha256_file(cand_path),
        "final_action_contract_hash": _sha256_file(action_path),
        "candidate_library_path": str(cand_path),
        "final_action_contract_path": str(action_path),
    }


def resolve_candidate_library_path(
    project_root: Path,
    *,
    magnitude_quantile: int | None = None,
    candidate_library_path: str | Path | None = None,
) -> Path:
    """Resolve the candidate-library YAML consumed by an action-space user.

    The canonical active config remains ``data/final_freeze/configs/final_candidate_library.yaml``.
    RL training/evaluation can consume Stage2's materialized magnitude-calibrated
    libraries (``final_candidate_library__P{q}.yaml``).  LLM stages use this helper
    so their prompt table, projection space, and simulator actions can be explicitly
    aligned to the same P50 library as the RL track without mutating the active base
    config or its hash.
    """
    root = Path(project_root).resolve()
    if candidate_library_path is not None:
        path = Path(candidate_library_path)
        if not path.is_absolute():
            path = root / path
    elif magnitude_quantile is not None:
        q = int(magnitude_quantile)
        if q not in {50, 65, 75, 85}:
            raise ValueError(f"Unsupported candidate-library magnitude quantile: {q}")
        path = final_root(root) / "stage2_candidate_projection" / f"final_candidate_library__P{q}.yaml"
    else:
        path = config_root(root) / "final_candidate_library.yaml"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Candidate library not found: {path}")
    return path

def assert_hashes_match(payload: dict, project_root: Path, *, context: str) -> None:
    active = active_config_hashes(project_root)
    for key in ["candidate_library_hash", "final_action_contract_hash"]:
        found = payload.get(key) or payload.get("schema", {}).get(key) or payload.get("config_hashes", {}).get(key)
        if found and str(found) != str(active[key]):
            raise ValueError(f"{context} hash mismatch for {key}: checkpoint={found} active={active[key]}")
        if not found:
            raise ValueError(f"{context} missing required embedded hash: {key}")

def _load_runtime_action_contract(cfg: Path) -> tuple[list[str], dict[str, tuple[float, float]], dict]:
    action = load_yaml(cfg / "final_action_contract.yaml")
    if "action_columns" in action:
        columns = list(action["action_columns"])
    else:
        columns = [f"action__{x}" for x in action["canonical_action_order"]]
    raw_bounds = action.get("action_bounds", {})
    bounds = {}
    for k, v in raw_bounds.items():
        kk = k if str(k).startswith("action__") else f"action__{k}"
        bounds[kk] = (float(v[0]), float(v[1]))
    return columns, bounds, action

def load_action_space(
    project_root: Path,
    *,
    candidate_library_path: str | Path | None = None,
    magnitude_quantile: int | None = None,
) -> ActionSpace:
    cfg = config_root(project_root)
    columns, bounds, action = _load_runtime_action_contract(cfg)
    cand_path = resolve_candidate_library_path(
        project_root,
        magnitude_quantile=magnitude_quantile,
        candidate_library_path=candidate_library_path,
    )
    cand = load_yaml(cand_path)
    missing_bounds = [c for c in columns if c not in bounds]
    if missing_bounds:
        raise ValueError(f"Action bounds missing for columns: {missing_bounds}")
    fixed_raw = cand.get("fixed_candidates", {}) or {}
    scenario_raw = cand.get("scenario_candidates", {}) or {}
    diagnostic_raw = cand.get("diagnostic_candidates", {}) or {}
    train_labels = list(action.get("candidate_label_rule", {}).get("train_labels") or cand.get("main_train_labels") or [])
    row_conditional = list(action.get("candidate_label_rule", {}).get("row_conditional_baselines") or list((cand.get("row_conditional_baselines") or {}).keys()))
    final_rl_label = str(action.get("candidate_label_rule", {}).get("final_rl_label") or cand.get("final_rl_label") or "C3_candidate_iql")
    if sorted(fixed_raw.keys()) != sorted(train_labels):
        raise ValueError(f"Fixed candidate names do not match train labels: fixed={sorted(fixed_raw)} train={sorted(train_labels)}")
    scenario_names = set(scenario_raw.keys())
    diagnostic_names = set(diagnostic_raw.keys())
    bad_train = sorted((scenario_names | diagnostic_names) & set(train_labels))
    if bad_train:
        raise ValueError(f"Scenario/diagnostic candidates must not be train labels: {bad_train}")
    forbidden_prefixes = ["H", "C2_", "SC", "AB", "OUT_OF_LIBRARY"]
    for name in fixed_raw:
        if any(name.startswith(p) for p in forbidden_prefixes):
            raise ValueError(f"Forbidden fixed/train candidate in active config: {name}")
    if any(str(x).startswith("H") for x in row_conditional + [final_rl_label]):
        raise ValueError("Legacy H-style labels are forbidden in final label rules")
    fixed = {k: _numeric_candidate_values(v, columns) for k, v in fixed_raw.items()}
    scenario = {k: _numeric_candidate_values(v, columns) for k, v in scenario_raw.items()}
    diagnostic = {k: _numeric_candidate_values(v, columns) for k, v in diagnostic_raw.items()}
    # Semantic sign guards from v32 design.
    for wc in [x for x in fixed if x.startswith("WC") or x == "MX2_liquidity_rescue"]:
        vec = fixed[wc]
        if vec.get("action__inv_turnover_chg", 0.0) < 0 or vec.get("action__ar_turnover_chg", 0.0) < 0:
            raise ValueError(f"Working-capital candidate must not reduce inventory/AR turnover: {wc}")
        if wc not in {"WC2_supplier_financing", "MX2_liquidity_rescue"} and vec.get("action__ap_turnover_chg", 0.0) < 0:
            raise ValueError(f"AP turnover decrease allowed only for supplier-financing candidates: {wc}")
    hashes = active_config_hashes(project_root)
    return ActionSpace(
        columns,
        bounds,
        fixed,
        train_labels,
        row_conditional,
        final_rl_label,
        scenario,
        diagnostic,
        _sha256_file(cand_path),
        hashes["final_action_contract_hash"],
    )

def validate_action_columns(df: pd.DataFrame, space: ActionSpace) -> None:
    missing = [c for c in space.columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required 10D action columns: {missing}")

def mean_normalized_l1(action_row: np.ndarray, candidate_vec: np.ndarray, widths: np.ndarray) -> float:
    return float(np.mean(np.abs((action_row - candidate_vec) / widths)))

WEIGHTED_L1_PRESETS: dict[str, dict[str, float]] = {
    "credit_recourse_v1": {
        # Debt-structure and margin axes receive higher weight because the
        # current simulator/oracle stack is most sensitive to leverage mix and
        # operating-cost relief.  This is an opt-in projection ablation only;
        # the default active-intent projection remains unchanged.
        "action__ppe_pct": 0.75,
        "action__inv_turnover_chg": 0.75,
        "action__ar_turnover_chg": 0.75,
        "action__ap_turnover_chg": 0.75,
        "action__short_debt_pct": 1.75,
        "action__long_debt_pct": 1.75,
        "action__bond_pct": 1.50,
        "action__revenue_growth": 1.00,
        "action__cogs_ratio_chg": 1.50,
        "action__sga_ratio_chg": 1.50,
    }
}

def projection_distance_description(projection_mode: str, *, a0_policy: str = "allow_nearest") -> str:
    mode = str(projection_mode or "active_intent")
    if mode == "active_intent":
        return "active_action_first_signed_intent_with_l1_diagnostics"
    if mode == "l1_best":
        base = "pure_normalized_l1_nearest_candidate_ablation"
    elif mode == "weighted_l1":
        base = "weighted_normalized_l1_nearest_candidate_ablation_credit_recourse_v1"
    else:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")
    if a0_policy == "margin":
        return base + "_with_a0_margin_threshold"
    return base

def resolve_weighted_l1_weights(
    space: ActionSpace,
    *,
    preset: str = "credit_recourse_v1",
    overrides: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return validated per-action-column weights for weighted-L1 projection.

    This helper is intentionally strict: every action column receives exactly one
    positive finite weight.  Unknown override keys hard-fail so a typo cannot
    silently change the projection experiment.
    """
    weights = {c: 1.0 for c in space.columns}
    if preset:
        if preset not in WEIGHTED_L1_PRESETS:
            raise ValueError(f"Unknown weighted L1 preset: {preset}. Available={sorted(WEIGHTED_L1_PRESETS)}")
        weights.update(WEIGHTED_L1_PRESETS[preset])
    if overrides:
        unknown = sorted(set(overrides) - set(space.columns))
        if unknown:
            raise ValueError(f"Weighted L1 overrides contain unknown action columns: {unknown}")
        for k, v in overrides.items():
            weights[k] = float(v)
    bad = {k: v for k, v in weights.items() if not np.isfinite(float(v)) or float(v) <= 0}
    if bad:
        raise ValueError(f"Weighted L1 weights must be positive finite numbers: {bad}")
    return {c: float(weights[c]) for c in space.columns}

def _distance_matrix(X: np.ndarray, C: np.ndarray, widths: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones_like(widths, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != widths.shape:
        raise ValueError(f"Projection weight shape mismatch: weights={weights.shape} widths={widths.shape}")
    denom = float(np.sum(weights))
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError("Projection weights must have positive finite sum")
    dists = np.zeros((len(X), len(C)), dtype=np.float32)
    for j in range(len(C)):
        dists[:, j] = np.sum(weights * np.abs((X - C[j]) / widths), axis=1) / denom
    return dists

def _apply_a0_margin_override(
    primary_order: np.ndarray,
    dists: np.ndarray,
    cand_names: list[str],
    *,
    a0_policy: str,
    a0_margin: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if a0_policy not in {"allow_nearest", "margin"}:
        raise ValueError(f"Unknown a0_policy: {a0_policy}")
    if not np.isfinite(float(a0_margin)) or float(a0_margin) < 0:
        raise ValueError(f"a0_margin must be a non-negative finite number: {a0_margin}")
    noop_idx = cand_names.index("A0_noop") if "A0_noop" in cand_names else None
    raw_best = primary_order[:, 0].astype(int)
    chosen = raw_best.copy()
    n = len(raw_best)
    a0_distance = np.full(n, np.nan, dtype=np.float32)
    best_non_noop_idx = np.full(n, -1, dtype=np.int32)
    best_non_noop_dist = np.full(n, np.nan, dtype=np.float32)
    raw_best_is_a0 = np.zeros(n, dtype=bool)
    keep_by_margin = np.zeros(n, dtype=bool)
    override_from_noop = np.zeros(n, dtype=bool)
    if noop_idx is not None:
        a0_distance = dists[:, noop_idx].astype(np.float32)
        for i in range(n):
            non = [int(j) for j in primary_order[i].tolist() if int(j) != int(noop_idx)]
            if non:
                best_non_noop_idx[i] = non[0]
                best_non_noop_dist[i] = float(dists[i, non[0]])
        raw_best_is_a0 = raw_best == int(noop_idx)
        if a0_policy == "margin":
            # Keep A0 only when it is decisively closer than the best non-A0.
            # Equality or near-tie is intentionally resolved to the intervention
            # candidate for this opt-in ablation.
            keep_by_margin = raw_best_is_a0 & ((a0_distance + float(a0_margin)) < best_non_noop_dist)
            override_from_noop = raw_best_is_a0 & (~keep_by_margin) & (best_non_noop_idx >= 0)
            chosen[override_from_noop] = best_non_noop_idx[override_from_noop]
        else:
            keep_by_margin = raw_best_is_a0.copy()
    diag = {
        "raw_best_is_a0": raw_best_is_a0,
        "a0_distance": a0_distance,
        "best_non_noop_idx": best_non_noop_idx,
        "best_non_noop_dist": best_non_noop_dist,
        "keep_by_margin": keep_by_margin,
        "override_from_noop": override_from_noop,
    }
    return chosen, diag

def project_actions_to_candidates(
    df: pd.DataFrame,
    space: ActionSpace,
    *,
    projection_mode: str = "active_intent",
    a0_policy: str = "allow_nearest",
    a0_margin: float = 0.0,
    weighted_l1_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Project continuous 10D pseudo-actions to v32 candidates.

    Modes:
    - active_intent: default final rule, active-action-first signed intent.
    - l1_best: opt-in pure normalized L1 nearest-candidate ablation.
    - weighted_l1: opt-in weighted normalized L1 nearest-candidate ablation.

    A0/no-op margin threshold is available only for l1_best/weighted_l1 and is
    disabled by default.  All modes preserve L1 and active-intent diagnostics so
    downstream audits can compare projection geometry without recomputation.
    """
    projection_mode = str(projection_mode or "active_intent")
    if projection_mode not in {"active_intent", "l1_best", "weighted_l1"}:
        raise ValueError(f"Unknown projection_mode: {projection_mode}")
    if projection_mode == "active_intent" and a0_policy != "allow_nearest":
        raise ValueError("a0-policy margin is only valid for l1_best/weighted_l1 projection modes")
    validate_action_columns(df, space)
    X = df[space.columns].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    widths = np.array([space.bound_width(c) for c in space.columns], dtype=np.float32)
    cand_names = space.names
    C = np.vstack([space.candidate_vector(n) for n in cand_names])
    l1_dists = _distance_matrix(X, C, widths, None)
    if projection_mode == "weighted_l1":
        if weighted_l1_weights is None:
            weighted_l1_weights = resolve_weighted_l1_weights(space)
        unknown = sorted(set(weighted_l1_weights) - set(space.columns))
        if unknown:
            raise ValueError(f"weighted_l1_weights contain unknown columns: {unknown}")
        missing = [c for c in space.columns if c not in weighted_l1_weights]
        if missing:
            raise ValueError(f"weighted_l1_weights missing action columns: {missing}")
        w = np.array([float(weighted_l1_weights[c]) for c in space.columns], dtype=np.float32)
        primary_dists = _distance_matrix(X, C, widths, w)
    else:
        w = np.ones_like(widths, dtype=np.float32)
        primary_dists = l1_dists

    noop_idx = cand_names.index("A0_noop") if "A0_noop" in cand_names else 0
    cand_active = np.abs(C) > 1e-12
    evidence_eps = np.maximum(widths * 1e-3, 1e-9)

    active_chosen = np.full(len(X), noop_idx, dtype=int)
    intent_scores = np.zeros(len(X), dtype=np.float32)
    active_match_counts = np.zeros(len(X), dtype=np.int32)
    active_mismatch_counts = np.zeros(len(X), dtype=np.int32)
    max_evidence = np.zeros(len(X), dtype=np.float32)
    fallback_to_noop = np.ones(len(X), dtype=bool)

    for i, row in enumerate(X):
        row_active = np.abs(row) > evidence_eps
        total_norm = float(np.mean(np.abs(row / widths)))
        if not np.any(row_active) or total_norm <= 1e-6:
            continue
        best_j = None
        best_tuple = None
        best_match_count = 0
        best_mismatch_count = 0
        best_evidence = 0.0
        for j, name in enumerate(cand_names):
            if j == noop_idx:
                continue
            active_dims = cand_active[j]
            if not np.any(active_dims):
                continue
            aligned = active_dims & row_active & ((row * C[j]) > 0)
            mismatched = active_dims & row_active & ((row * C[j]) < 0)
            match_count = int(aligned.sum())
            if match_count <= 0:
                continue
            mismatch_count = int(mismatched.sum())
            denom = int(active_dims.sum())
            evidence = np.minimum(np.abs(row[aligned]) / np.maximum(np.abs(C[j][aligned]), 1e-9), 1.0)
            evidence_mean = float(evidence.mean()) if len(evidence) else 0.0
            score = (match_count / max(denom, 1)) + 0.25 * evidence_mean - 0.20 * mismatch_count
            # Higher score is better; lower primary distance breaks ties.
            key = (score, -float(primary_dists[i, j]), match_count, -mismatch_count)
            if best_tuple is None or key > best_tuple:
                best_tuple = key
                best_j = j
                best_match_count = match_count
                best_mismatch_count = mismatch_count
                best_evidence = float(np.max(evidence)) if len(evidence) else 0.0
        if best_j is not None and best_tuple is not None and best_tuple[0] > 0:
            active_chosen[i] = int(best_j)
            intent_scores[i] = float(best_tuple[0])
            active_match_counts[i] = int(best_match_count)
            active_mismatch_counts[i] = int(best_mismatch_count)
            max_evidence[i] = float(best_evidence)
            fallback_to_noop[i] = False

    primary_order = np.argsort(primary_dists, axis=1)
    if projection_mode == "active_intent":
        chosen = active_chosen.copy()
        a0_diag = {
            "raw_best_is_a0": primary_order[:, 0] == noop_idx,
            "a0_distance": primary_dists[:, noop_idx] if "A0_noop" in cand_names else np.full(len(X), np.nan),
            "best_non_noop_idx": np.full(len(X), -1, dtype=np.int32),
            "best_non_noop_dist": np.full(len(X), np.nan, dtype=np.float32),
            "keep_by_margin": primary_order[:, 0] == noop_idx,
            "override_from_noop": np.zeros(len(X), dtype=bool),
        }
    else:
        chosen, a0_diag = _apply_a0_margin_override(
            primary_order,
            primary_dists,
            cand_names,
            a0_policy=a0_policy,
            a0_margin=float(a0_margin),
        )

    out = df.copy()
    # Soft target order: selected winner first, then nearest candidates by the
    # active primary distance for the selected ablation.
    order = np.zeros((len(X), min(3, len(cand_names))), dtype=int)
    for i in range(len(X)):
        ranked = [int(chosen[i])] + [int(j) for j in primary_order[i].tolist() if int(j) != int(chosen[i])]
        order[i, :] = ranked[: order.shape[1]]
    idx2 = order[:, 1] if order.shape[1] > 1 else order[:, 0]
    l1_order = np.argsort(l1_dists, axis=1)
    best_l1_idx = l1_order[:, 0]
    second_l1_idx = l1_order[:, 1] if l1_order.shape[1] > 1 else l1_order[:, 0]
    weighted_order = np.argsort(primary_dists, axis=1)
    best_primary_idx = weighted_order[:, 0]
    second_primary_idx = weighted_order[:, 1] if weighted_order.shape[1] > 1 else weighted_order[:, 0]

    best = primary_dists[np.arange(len(X)), chosen]
    second = primary_dists[np.arange(len(X)), idx2]
    best_l1 = l1_dists[np.arange(len(X)), best_l1_idx]
    second_l1 = l1_dists[np.arange(len(X)), second_l1_idx]
    best_primary_raw = primary_dists[np.arange(len(X)), best_primary_idx]
    second_primary_raw = primary_dists[np.arange(len(X)), second_primary_idx]

    out["projected_candidate_id"] = [cand_names[i] for i in chosen]
    out["nearest_candidate_id"] = out["projected_candidate_id"]
    out["second_best_candidate_id"] = [cand_names[i] for i in idx2]
    out["candidate_id"] = out["projected_candidate_id"]
    out["projection_method"] = projection_distance_description(projection_mode, a0_policy=a0_policy)
    out["projection_mode"] = projection_mode
    out["projection_distance"] = best
    out["second_best_distance"] = second
    out["projection_margin"] = second - best
    out["near_tie_flag"] = out["projection_margin"] < 0.025
    out["out_of_library_flag"] = out["projection_distance"] > 0.50
    out["out_of_library"] = out["out_of_library_flag"]
    out["projection_distance_l1_best"] = best_l1
    out["projection_l1_best_candidate_id"] = [cand_names[i] for i in best_l1_idx]
    out["projection_l1_second_distance"] = second_l1
    out["projection_primary_raw_best_candidate_id"] = [cand_names[i] for i in best_primary_idx]
    out["projection_primary_raw_best_distance"] = best_primary_raw
    out["projection_primary_raw_second_distance"] = second_primary_raw
    out["projection_active_intent_candidate_id"] = [cand_names[i] for i in active_chosen]
    out["projection_active_intent_distance"] = primary_dists[np.arange(len(X)), active_chosen]
    out["projection_intent_score"] = intent_scores
    out["projection_intent_max_evidence"] = max_evidence
    out["projection_active_match_count"] = active_match_counts
    out["projection_active_mismatch_count"] = active_mismatch_counts
    out["projection_total_action_norm"] = np.mean(np.abs(X / widths), axis=1)
    out["projection_fallback_to_noop"] = fallback_to_noop
    out["projection_weighted_l1_preset"] = "credit_recourse_v1" if projection_mode == "weighted_l1" else ""
    for col, val in zip(space.columns, w.tolist()):
        out[f"projection_weight__{col.replace('action__', '')}"] = float(val)
    out["projection_a0_policy"] = a0_policy
    out["projection_a0_margin"] = float(a0_margin)
    out["projection_a0_distance"] = a0_diag["a0_distance"]
    out["projection_a0_best_non_noop_candidate_id"] = [cand_names[i] if int(i) >= 0 else "" for i in a0_diag["best_non_noop_idx"]]
    out["projection_a0_best_non_noop_distance"] = a0_diag["best_non_noop_dist"]
    out["projection_a0_raw_best"] = a0_diag["raw_best_is_a0"]
    out["projection_a0_keep_by_margin"] = a0_diag["keep_by_margin"]
    out["projection_a0_override_from_noop"] = a0_diag["override_from_noop"]

    kernel = 1.0 / (primary_dists[np.arange(len(X))[:, None], order] + 1e-6)
    if projection_mode == "active_intent":
        active_boost = (~fallback_to_noop).astype(np.float32)
        kernel[:, 0] = kernel[:, 0] * (1.0 + active_boost)
    probs = kernel / kernel.sum(axis=1, keepdims=True)
    for k in range(order.shape[1]):
        out[f"soft_cand_id_{k+1}"] = [cand_names[i] for i in order[:, k]]
        out[f"soft_cand_prob_{k+1}"] = probs[:, k]
    for k in range(order.shape[1], 3):
        out[f"soft_cand_id_{k+1}"] = ""
        out[f"soft_cand_prob_{k+1}"] = 0.0
    return out

def assert_training_labels_allowed(df: pd.DataFrame, space: ActionSpace, label_col: str = "candidate_id") -> None:
    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")
    allowed = set(space.train_labels)
    labels = set(df[label_col].dropna().astype(str).unique().tolist())
    bad = sorted(labels - allowed)
    if bad:
        counts = df[df[label_col].astype(str).isin(bad)][label_col].astype(str).value_counts().to_dict()
        raise ValueError(f"Forbidden candidate labels found; refusing to silently drop: {counts}")
