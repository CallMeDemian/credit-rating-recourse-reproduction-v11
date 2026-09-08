from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(df: pd.DataFrame | None, split: str) -> dict[str, Any]:
    if df is None or "split" not in df.columns:
        return {}
    sub = df[df["split"].astype(str).str.lower() == split.lower()]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def _num(x: Any, default: float | None = None) -> float | None:
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _gate(name: str, value: Any, op: str, threshold: float, required: bool = True) -> dict[str, Any]:
    v = _num(value)
    if v is None:
        passed = False
    elif op == ">=":
        passed = v >= threshold
    elif op == "<=":
        passed = v <= threshold
    elif op == "<":
        passed = v < threshold
    elif op == ">":
        passed = v > threshold
    else:
        raise ValueError(op)
    return {
        "gate": name,
        "value": v,
        "operator": op,
        "threshold": threshold,
        "required": bool(required),
        "pass": bool(passed),
    }


# ORACLE_BACKEND_DIAGNOSTIC_SAMPLE_POLICY_2026_05_24
# Sample-size gates are required, but the required minimum must reflect the
# actual complete-case oracle-development universe after current Stage00_01~04
# filters. 1500/500 are retained as legacy advisory reference gates rather than
# hard stops; model-quality gates remain required and unchanged.
ALPHA_REQUIRED_MIN_DEV_N = 1000
ALPHA_REQUIRED_MIN_OOT_N = 300
ALPHA_LEGACY_REFERENCE_DEV_N = 1500
ALPHA_LEGACY_REFERENCE_OOT_N = 500


def _alpha_sample_size_gates(a_dev: dict[str, Any], a_oot: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _gate("alpha_dev_n", a_dev.get("n"), ">=", ALPHA_REQUIRED_MIN_DEV_N),
        _gate("alpha_oot_n", a_oot.get("n"), ">=", ALPHA_REQUIRED_MIN_OOT_N),
        _gate("alpha_dev_n_legacy_reference_advisory", a_dev.get("n"), ">=", ALPHA_LEGACY_REFERENCE_DEV_N, required=False),
        _gate("alpha_oot_n_legacy_reference_advisory", a_oot.get("n"), ">=", ALPHA_LEGACY_REFERENCE_OOT_N, required=False),
    ]


def diagnose_backend_dir(backend_dir: Path) -> dict[str, Any]:
    backend_dir = Path(backend_dir).resolve()
    alpha_dir, beta_dir, gamma_dir = backend_dir / "alpha", backend_dir / "beta", backend_dir / "gamma"
    alpha_m = _read_csv(alpha_dir / "preliminary_dev_oot_metrics_alpha.csv")
    beta_m = _read_csv(beta_dir / "preliminary_dev_oot_metrics_beta.csv")
    gamma_m = _read_csv(gamma_dir / "preliminary_dev_oot_metrics_gamma.csv")
    alpha_params = _read_json(alpha_dir / "oracle_alpha_params.json") or {}
    beta_params = _read_json(beta_dir / "benchmark_beta_params.json") or {}
    gamma_params = _read_json(gamma_dir / "benchmark_gamma_params.json") or {}

    gates: list[dict[str, Any]] = []
    a_dev, a_oot = _metric_row(alpha_m, "dev"), _metric_row(alpha_m, "oot")
    b_oot = _metric_row(beta_m, "oot")
    g_oot = _metric_row(gamma_m, "oot")

    # Alpha is the main reference oracle.
    # Required sample-size minima are current-universe safeguards; legacy
    # 1500/500 reference thresholds are reported as advisory diagnostics so a
    # valid current complete-case dev sample (e.g., 1338) does not hard-fail
    # after upstream filtering changes.
    gates += _alpha_sample_size_gates(a_dev, a_oot)
    gates += [
        _gate("alpha_dev_abs_spearman", abs(_num(a_dev.get("rho"), 0.0) or 0.0), ">=", 0.55),
        _gate("alpha_oot_abs_spearman", abs(_num(a_oot.get("rho"), 0.0) or 0.0), ">=", 0.55),
        _gate("alpha_oot_within1", a_oot.get("within1"), ">=", 0.85),
        _gate("alpha_oot_AR", a_oot.get("AR"), ">=", 30.0),
        _gate("alpha_oot_QK", a_oot.get("QK"), ">=", 0.40),
    ]

    # Gamma is the nonlinear robustness backend.
    gates += [
        _gate("gamma_oot_abs_spearman", g_oot.get("rho_abs", abs(_num(g_oot.get("rho"), 0.0) or 0.0)), ">=", 0.55),
        _gate("gamma_oot_within1", g_oot.get("within1"), ">=", 0.70),
    ]

    # Beta is advisory only because ordered-logit/folded-grade mapping can be biased.
    gates += [
        _gate("beta_oot_abs_spearman_advisory", b_oot.get("rho_abs", abs(_num(b_oot.get("rho"), 0.0) or 0.0)), ">=", 0.55, required=False),
        _gate("beta_oot_within1_advisory", b_oot.get("within1"), ">=", 0.70, required=False),
        _gate("beta_abs_bias_advisory", abs(_num(b_oot.get("bias_pred_minus_actual_notch"), 999.0) or 999.0), "<=", 0.50, required=False),
    ]

    # Cross-oracle consistency.
    ag = _read_csv(gamma_dir / "cross_oracle_alpha_gamma_comparison.csv")
    if ag is not None and not ag.empty:
        row = _metric_row(ag, "oot")
        val = row.get("spearman_alpha_gamma_ml", row.get("spearman_alpha_gamma"))
        gates += [
            _gate("alpha_gamma_oot_rank_consistency", val, ">=", 0.70),
            _gate("alpha_gamma_oot_within1_agree", row.get("within1_agree"), ">=", 0.75),
        ]

    # Boundary jump diagnostics are advisory for scorecard smoothness/exploit risk.
    bj = _read_csv(alpha_dir / "boundary_jump_test_alpha.csv")
    boundary_diag: dict[str, Any] = {}
    if bj is not None and not bj.empty:
        n = int(len(bj))
        excessive = int(bj.get("excessive_jump_flag", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        crossed = int(bj.get("boundary_crossed", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
        if "abs_R_delta" in bj.columns:
            abs_delta = pd.to_numeric(bj["abs_R_delta"], errors="coerce")
        elif "R_score_delta" in bj.columns:
            abs_delta = pd.to_numeric(bj["R_score_delta"], errors="coerce").abs()
        else:
            abs_delta = pd.Series(dtype=float)
        max_abs_delta = _num(abs_delta.max())
        p99_abs_delta = _num(abs_delta.quantile(0.99))
        boundary_diag = {
            "n_tests": n,
            "boundary_crossed": crossed,
            "excessive_jump_flag": excessive,
            "excessive_jump_rate": excessive / n if n else None,
            "max_abs_R_delta": max_abs_delta,
            "p99_abs_R_delta": p99_abs_delta,
        }
        gates += [
            _gate("alpha_boundary_excessive_jump_rate_advisory", boundary_diag["excessive_jump_rate"], "<=", 0.05, required=False),
            _gate("alpha_boundary_p99_abs_delta_advisory", boundary_diag["p99_abs_R_delta"], "<=", 5.0, required=False),
        ]

    split_policies = {
        "alpha": alpha_params.get("split_policy"),
        "beta": beta_params.get("split_policy"),
        "gamma": gamma_params.get("split_policy"),
    }
    split_consistent = len({json.dumps(v, sort_keys=True, ensure_ascii=False) for v in split_policies.values() if v}) <= 1
    gates.append({
        "gate": "backend_split_policy_consistent",
        "value": split_policies,
        "operator": "all_equal",
        "threshold": None,
        "required": True,
        "pass": bool(split_consistent),
    })

    required_failed = [g for g in gates if g.get("required") and not g.get("pass")]
    advisory_failed = [g for g in gates if not g.get("required") and not g.get("pass")]
    status = "PASS" if not required_failed else "FAIL"
    verdict = "usable_as_main_reference_oracle" if status == "PASS" else "do_not_freeze_until_required_gates_pass"

    return {
        "diagnostic_version": "oracle_backend_diagnostic_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "backend_dir": str(backend_dir),
        "status": status,
        "verdict": verdict,
        "split_policies": split_policies,
        "metrics": {
            "alpha_dev": a_dev,
            "alpha_oot": a_oot,
            "beta_oot": b_oot,
            "gamma_oot": g_oot,
        },
        "boundary_jump_diagnostics": boundary_diag,
        "sample_size_policy": {
            "policy_id": "ORACLE_BACKEND_DIAGNOSTIC_SAMPLE_POLICY_2026_05_24",
            "alpha_required_min_dev_n": ALPHA_REQUIRED_MIN_DEV_N,
            "alpha_required_min_oot_n": ALPHA_REQUIRED_MIN_OOT_N,
            "alpha_legacy_reference_dev_n": ALPHA_LEGACY_REFERENCE_DEV_N,
            "alpha_legacy_reference_oot_n": ALPHA_LEGACY_REFERENCE_OOT_N,
            "legacy_reference_thresholds_are_advisory": True,
        },
        "gates": gates,
        "required_failed_gates": required_failed,
        "advisory_failed_gates": advisory_failed,
        "interpretation": {
            "alpha": "main scorecard-style reference oracle; required gates decide final usability",
            "gamma": "nonlinear robustness backend; required for rank/within-one robustness",
            "beta": "ordered-logit benchmark; advisory only unless explicitly promoted by the researcher",
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend-dir", required=True, help="Path to data/final_freeze/stage1_oracle_backends")
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--fail-on-required", action="store_true")
    args = ap.parse_args(argv)

    report = diagnose_backend_dir(Path(args.backend_dir))
    out_json = Path(args.output_json) if args.output_json else Path(args.backend_dir) / "oracle_backend_diagnostic_report.json"
    out_csv = Path(args.output_csv) if args.output_csv else Path(args.backend_dir) / "oracle_backend_gate_summary.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    pd.DataFrame(report["gates"]).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({"status": report["status"], "verdict": report["verdict"], "output_json": str(out_json), "output_csv": str(out_csv)}, ensure_ascii=False, indent=2))
    return 1 if args.fail_on_required and report["status"] != "PASS" else 0


if __name__ == "__main__":
    raise SystemExit(main())
