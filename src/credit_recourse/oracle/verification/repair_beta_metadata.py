#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repair Stage4 Beta metadata after posterior MAP gradefix.

This utility updates benchmark_beta_params.json so that modeled_grades matches
actual prob_beta_grade_* output columns, and regenerates Beta confusion matrices
with all observed/predicted grades, including sparse C/D predictions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

GRADE_ORDER_10 = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "CC", "C", "D"]
GRADE2NUM_10 = {g: i + 1 for i, g in enumerate(GRADE_ORDER_10)}


def _sort_grades(grades: list[str]) -> list[str]:
    return [g for g in GRADE_ORDER_10 if g in set(grades)]


def _beta_dir(project_root: Path, beta_dir: str | None) -> Path:
    if beta_dir:
        return Path(beta_dir).resolve()
    return project_root / "data" / "final_freeze" / "stage1_oracle_backends" / "beta"


def _load_output(beta_dir: Path) -> pd.DataFrame:
    csv_path = beta_dir / "benchmark_firm_year_output_beta.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    parquet_path = beta_dir / "benchmark_firm_year_output_beta.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    raise FileNotFoundError(
        f"Neither benchmark_firm_year_output_beta.csv nor .parquet exists in {beta_dir}"
    )


def repair(beta_dir: Path, *, write: bool = True) -> dict[str, Any]:
    beta_dir = beta_dir.resolve()
    params_path = beta_dir / "benchmark_beta_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing {params_path}")

    with open(params_path, encoding="utf-8") as f:
        params = json.load(f)

    df = _load_output(beta_dir)
    prob_cols = [c for c in df.columns if c.startswith("prob_beta_grade_")]
    prob_grades = _sort_grades([c.replace("prob_beta_grade_", "") for c in prob_cols])
    prob_cols_sorted = [f"prob_beta_grade_{g}" for g in prob_grades]
    predicted_grades = _sort_grades([str(g) for g in df.get("R_grade_beta", pd.Series(dtype=str)).dropna().unique()])
    observed_col = "grade_base_10" if "grade_base_10" in df.columns else "grade_base"
    observed_grades = _sort_grades([str(g) for g in df.get(observed_col, pd.Series(dtype=str)).dropna().unique()])

    old_modeled = params.get("modeled_grades")
    old_modeled_nums = params.get("modeled_grade_nums")

    params["configured_modeled_grades"] = params.get("configured_modeled_grades", old_modeled)
    params["configured_modeled_grade_nums"] = params.get("configured_modeled_grade_nums", old_modeled_nums)
    params["modeled_grades"] = prob_grades
    params["modeled_grade_nums"] = [GRADE2NUM_10[g] for g in prob_grades]
    params["probability_output_grades"] = prob_grades
    params["probability_output_columns"] = prob_cols_sorted
    params["predicted_output_grades"] = predicted_grades
    params["observed_output_grades"] = observed_grades
    params["metadata_consistency_note"] = (
        "modeled_grades was repaired to match actual prob_beta_grade_* columns; "
        "configured_modeled_grades preserves the original configured metadata."
    )

    confusion_written: list[str] = []
    if {"split_stage4", observed_col, "R_grade_beta"}.issubset(df.columns):
        for split in ["dev", "oot"]:
            sub = df[(df["split_stage4"] == split) & df[observed_col].notna() & df["R_grade_beta"].notna()].copy()
            if sub.empty:
                continue
            present = set(sub[observed_col].astype(str)) | set(sub["R_grade_beta"].astype(str))
            labels = [g for g in GRADE_ORDER_10 if g in present]
            cm = pd.DataFrame(0, index=labels, columns=labels)
            for o, p in zip(sub[observed_col].astype(str), sub["R_grade_beta"].astype(str)):
                if o in cm.index and p in cm.columns:
                    cm.loc[o, p] += 1
            out_path = beta_dir / f"confusion_matrix_beta_{split}.csv"
            if write:
                cm.to_csv(out_path, encoding="utf-8-sig")
            confusion_written.append(str(out_path))

    if write:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False, indent=2, default=str)

    return {
        "status": "PASS",
        "beta_dir": str(beta_dir),
        "params_path": str(params_path),
        "old_modeled_grades": old_modeled,
        "new_modeled_grades": params["modeled_grades"],
        "probability_output_columns": params["probability_output_columns"],
        "predicted_output_grades": predicted_grades,
        "observed_output_grades": observed_grades,
        "confusion_matrices_written": confusion_written,
        "write": write,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".", help="Repository root; default current directory")
    ap.add_argument("--beta-dir", default=None, help="Optional explicit beta output directory")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    project_root = Path(args.project_root).resolve()
    beta_dir = _beta_dir(project_root, args.beta_dir)
    report = repair(beta_dir, write=not args.dry_run)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
