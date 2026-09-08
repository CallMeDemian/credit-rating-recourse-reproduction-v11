from __future__ import annotations

"""Static + synthetic contract verifier for the N5M axis-swap intervention."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from credit_recourse.analysis.n5_7_10c_holm_inference import holm_adjust
from credit_recourse.analysis.n5m_axis_swap_intervention import (
    AXES,
    _fidelity_check,
    _oat_tables,
    _select_frozen_policy_pair,
    _select_policy_pair,
    _shapley_tables,
)

SCHEMA_VERSION = "n5m_axis_swap_contract_v3"


def verify(project_root: Path) -> dict:
    root = Path(project_root).resolve()
    src = root / "src" / "credit_recourse" / "analysis" / "n5m_axis_swap_intervention.py"
    journal_runner = root / "tools" / "run_c4r_journal_grid.ps1"
    errors: list[str] = []
    if not src.is_file():
        errors.append(f"required file missing: {src}")
    if src.is_file():
        text = src.read_text(encoding="utf-8-sig")
        for marker in ("validate=\"one_to_one\"", "SHAPLEY EFFICIENCY FAIL", "holm_adjust", "p_loss_vs_zero_holm", "--base-policy", "--target-policy", "_select_policy_pair", "_select_frozen_policy_pair"):
            if marker not in text:
                errors.append(f"axis-swap source marker missing: {marker}")
        forbidden = ("order = np.argsort(pvals)", 'rec["p_loss_holm"]')
        for marker in forbidden:
            if marker in text:
                errors.append(f"stale axis-swap implementation marker present: {marker}")
    # This verifier is scoped to the active journal-extension execution path.
    # Thesis-wide/legacy orchestration scripts are intentionally outside this
    # contract: their mere presence must not make a clean journal checkout fail.
    if not journal_runner.is_file():
        errors.append(f"required active journal runner missing: {journal_runner}")
    else:
        journal_text = journal_runner.read_text(encoding="utf-8-sig")
        for marker in (
            "RunAxisAttribution",
            "verify_n5m_axis_swap_contract",
            "credit_recourse.analysis.n5m_axis_swap_intervention",
            'name = "self_revision"; base = "C4"; target = "C4R"',
            'name = "reference_content"; base = "C4R"; target = "C6"',
            "--base-policy",
            "--target-policy",
        ):
            if marker not in journal_text:
                errors.append(f"journal axis-attribution runner marker missing: {marker}")

    # Contract-faithful synthetic construction smoke.
    c4 = pd.DataFrame({"row_id": [1, 2], "policy": ["C4", "C4"], "candidate_id": ["x", "x"], "mode": ["free_form_10d"] * 2})
    c6 = c4.copy(); c6["policy"] = "C6"
    for i, axis in enumerate(AXES):
        c4[axis] = [0.0, float(i) / 100.0]
        c6[axis] = [float(i + 1) / 100.0, 0.0]
    c4r = c4.copy(); c4r["policy"] = "C4R"
    for i, axis in enumerate(AXES):
        c4r[axis] = [(float(i + 1) / 200.0), (float(i + 1) / 300.0)]
    action_grid = pd.concat([c4, c4r, c6], ignore_index=True)
    try:
        pair_c4_c4r = _select_policy_pair(
            action_grid, base_policy="C4", target_policy="C4R", expected_n=2
        )
        pair_c4r_c6 = _select_policy_pair(
            action_grid, base_policy="C4R", target_policy="C6", expected_n=2
        )
        if not pair_c4_c4r[0]["policy"].astype(str).eq("C4").all():
            errors.append("synthetic C4->C4R base selection failed")
        if not pair_c4r_c6[0]["policy"].astype(str).eq("C4R").all():
            errors.append("synthetic C4R->C6 base selection failed")
    except Exception as exc:
        errors.append(f"synthetic policy-pair selection failed: {type(exc).__name__}: {exc}")

    # Regression for the real R2 failure: frozen Stage8 contains C4/C4R/C6,
    # while each fidelity rescore contains only the requested two-policy pair.
    frozen_rows = []
    for policy_idx, policy in enumerate(("C4", "C4R", "C6")):
        for row_id in (1, 2):
            frozen_rows.append({
                "row_id": row_id,
                "policy": policy,
                "mode": "free_form_10d",
                "delta_R_score_alpha": float(policy_idx + row_id),
                "delta_R_score_beta": float(policy_idx + row_id) / 10.0,
                "delta_R_score_gamma": float(policy_idx + row_id) / 5.0,
            })
    frozen_grid = pd.DataFrame(frozen_rows)
    try:
        frozen_c4_c4r = _select_frozen_policy_pair(
            frozen_grid,
            base_policy="C4",
            target_policy="C4R",
            expected_n_per_policy=2,
        )
        if len(frozen_c4_c4r) != 4 or set(frozen_c4_c4r["policy"].astype(str)) != {"C4", "C4R"}:
            errors.append("synthetic frozen C4->C4R filtering failed")
        scored_pair = frozen_c4_c4r.copy()
        fid = _fidelity_check(scored_pair, frozen_c4_c4r, 1e-12)
        if any(float(fid[o]["max_abs_diff"]) != 0.0 for o in ("alpha", "beta", "gamma")):
            errors.append("synthetic filtered fidelity identity check failed")
        frozen_c4r_c6 = _select_frozen_policy_pair(
            frozen_grid,
            base_policy="C4R",
            target_policy="C6",
            expected_n_per_policy=2,
        )
        if len(frozen_c4r_c6) != 4 or set(frozen_c4r_c6["policy"].astype(str)) != {"C4R", "C6"}:
            errors.append("synthetic frozen C4R->C6 filtering failed")
    except Exception as exc:
        errors.append(f"synthetic frozen policy-pair fidelity filtering failed: {type(exc).__name__}: {exc}")

    oat = _oat_tables(c4, c6)
    if len(oat) != len(c4) * len(AXES) or oat["policy"].nunique() != len(AXES):
        errors.append("synthetic OAT construction failed")
    shap = _shapley_tables(c4, c6, list(range(len(AXES))), 0)
    if len(shap) != len(c4) * len(AXES):
        errors.append("synthetic Shapley path construction failed")
    raw_p = [0.04, 0.01, 0.03]
    adjusted = holm_adjust(raw_p)
    order = np.argsort(raw_p)
    ordered_adjusted = [float(adjusted[int(idx)]) for idx in order]
    if any(ordered_adjusted[i] > ordered_adjusted[i + 1] for i in range(len(ordered_adjusted) - 1)):
        errors.append("Holm utility monotonicity smoke failed")
    if any(float(value) < float(raw_p[idx]) - 1e-15 for idx, value in enumerate(adjusted)):
        errors.append("Holm utility produced an adjusted p-value below its raw p-value")

    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL",
        "project_root": str(root),
        "synthetic_oat_rows": int(len(oat)),
        "synthetic_shapley_rows": int(len(shap)),
        "synthetic_policy_pairs": ["C4->C4R", "C4R->C6"],
        "synthetic_frozen_policy_pair_filtering": True if not any(
            "frozen" in error.lower() or "fidelity identity" in error.lower() for error in errors
        ) else False,
        "errors": errors,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", required=True)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args(argv)
    result = verify(Path(args.project_root))
    if args.out_json:
        out = Path(args.out_json); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
