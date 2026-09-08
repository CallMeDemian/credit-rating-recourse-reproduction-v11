from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from credit_recourse.rl.pipelines.final_stage2_input_splits.pipeline import (
    ACTION_COLUMNS,
    _norm_firm_id,
    _read_raw_action_panel_for_stage2,
    join_cash_flow_substrate,
)


def _install_pickle_parquet_shim_if_needed() -> dict:
    """Allow the smoke to run in minimal CI sandboxes without pyarrow.

    The production pipeline still uses pandas parquet normally.  This shim is
    installed only inside this smoke process when pandas cannot find a parquet
    engine, so it tests Stage2 join semantics rather than optional IO packages.
    """
    try:
        probe = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as td:
            probe.to_parquet(Path(td) / "probe.parquet", index=False)
        return {"parquet_engine_available": True, "io_shim": "native_parquet"}
    except ImportError:
        def _to_parquet_pickle(self, path, *args, **kwargs):
            return self.to_pickle(path)
        def _read_parquet_pickle(path, *args, **kwargs):
            return pd.read_pickle(path)
        pd.DataFrame.to_parquet = _to_parquet_pickle  # type: ignore[method-assign]
        pd.read_parquet = _read_parquet_pickle  # type: ignore[assignment]
        return {"parquet_engine_available": False, "io_shim": "pickle_backed_parquet_filename_for_smoke_only"}


def run_smoke(out_dir: Path | None = None) -> dict:
    """Synthetic regression smoke for Stage2 cash-flow substrate joins.

    This intentionally uses Korean NICE raw-style ``거래소코드`` keys and
    ``YYYY/MM`` fiscal-year strings in the cash-flow panel, plus A-prefixed
    Stage2A panel keys and six-digit bridge keys in the Stage1-like input frame.  The
    test fails if Stage2 regresses to naive zfill-based joins, which previously
    produced degenerate OCF after --join-cash-flow-substrate.
    """
    io_meta = _install_pickle_parquet_shim_if_needed()
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        cf_dir = root / "data/final_freeze/stage1_oracle_inputs/stage00_01_rating_statement_integration/cleaned_statement_panels"
        cf_dir.mkdir(parents=True, exist_ok=True)
        cf_path = cf_dir / "현금흐름표_clean.parquet"
        pd.DataFrame(
            {
                "거래소코드": [1, 1, "A000002", "000002.0"],
                "회계년도": ["2022/12", "2023/12", "2022/12", "2023/12"],
                "cash_flow__[U01D100000000]영업활동현금흐름(IFRS)(천원)": [100.0, 130.0, 80.0, 90.0],
                "cash_flow__[U01D206012400]유형자산취득(IFRS)(천원)": [-30.0, -35.0, -20.0, -22.0],
            }
        ).to_parquet(cf_path, index=False)

        bridge = pd.DataFrame(
            {
                "firm_id": ["000001", "000001", "000002", "000002"],
                "fiscal_year": ["2022/12", "2023/12", 2022, 2023],
            }
        )
        joined, meta = join_cash_flow_substrate(root, bridge, encoder_mode="reward_only")
        if joined["reward_only__sim__operating_cf"].notna().sum() != 4:
            raise AssertionError("cash-flow join failed to match A-prefixed codes to six-digit bridge firm_id")
        if meta["operating_cf_nonzero_rate_after_join"] <= 0.95:
            raise AssertionError(f"unexpected degenerate OCF join: {meta}")
        if set(joined["firm_id"].tolist()) != {"000001", "000002"}:
            raise AssertionError(f"firm_id normalization drifted: {joined['firm_id'].tolist()}")

        panel_dir = root / "data/final_freeze/stage2_candidate_projection/action_sources"
        panel_dir.mkdir(parents=True, exist_ok=True)
        panel_path = panel_dir / "stage2_raw_action_source_panel.parquet"
        raw = pd.DataFrame({"firm_id": ["A000001"], "fiscal_year": [2022]})
        for col in ACTION_COLUMNS:
            raw[col] = 0.0
        raw.to_parquet(panel_path, index=False)
        normalized_panel = _read_raw_action_panel_for_stage2(root)
        if normalized_panel.loc[0, "firm_id"] != "000001":
            raise AssertionError("Stage2A action panel firm_id normalization failed")

        result = {
            "status": "PASS",
            "cash_flow_panel": str(cf_path),
            "joined_rows": int(len(joined)),
            "ocf_non_null": int(joined["reward_only__sim__operating_cf"].notna().sum()),
            "ocf_nonzero_rate": float(meta["operating_cf_nonzero_rate_after_join"]),
            "normalized_examples": {"A000001": _norm_firm_id("A000001"), "000001.0": _norm_firm_id("000001.0")},
            "cash_flow_key_alias_exercised": "거래소코드",
            "cash_flow_year_alias_exercised": "회계년도 as YYYY/MM",
            **io_meta,
        }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "smoke_stage2_cash_flow_contract.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    result = run_smoke(Path(args.out_dir) if args.out_dir else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
