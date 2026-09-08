
from __future__ import annotations
import argparse, os, sys, json, runpy, shutil, contextlib
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd, yaml
from credit_recourse.oracle.stage1.stage00_01_rating_statement.final_stage0_adapter import (
    materialize_stage00_01,
    ensure_stage00_01_context_contract,
)
from credit_recourse.oracle.stage0.rating_contract_repair import validate_stage0_contract, repair_stage0_canonical
from credit_recourse.utils.io_contract import configure_utf8_stdio, write_json, read_json, read_csv_korean_safe, resolve_selected_variables, selected_variables_from_backend_params
from credit_recourse.oracle.verification.diagnose_oracle_backends import diagnose_backend_dir

@contextlib.contextmanager
def pushd(path: Path):
    old=Path.cwd(); os.chdir(path)
    try: yield
    finally: os.chdir(old)

def write_yaml(p: Path, data: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding='utf-8')

def run_py(path: Path, argv: list[str]|None=None, cwd: Path|None=None, env: dict|None=None):
    old_argv=sys.argv[:]; old_env=os.environ.copy()
    if argv is not None: sys.argv=[str(path)]+argv
    if env: os.environ.update({k:str(v) for k,v in env.items()})
    try:
        with pushd(cwd or path.parent):
            try: runpy.run_path(str(path), run_name='__main__')
            except SystemExit as e:
                if e.code not in (None,0): raise
    finally:
        sys.argv=old_argv; os.environ.clear(); os.environ.update(old_env)

def stage_paths(root: Path):
    final=root/'data'/'final_freeze'
    return final, final/'stage1_oracle_inputs', final/'stage1_oracle_backends', final/'ledgers', final/'configs'

def oracle_component_config_dir(root: Path) -> Path:
    return root/'data'/'final_freeze'/'configs'/'oracle_components'

def prepare_stage2_config(root, src_dir, s1_out, s2_out):
    cfg={
      'inputs':{'stage1b':{'panel':str(s1_out/'firm_year_panel_v1.parquet'),'cleaned_dir':str(s1_out/'cleaned_statement_panels'),'raw_long':str(s1_out/'financial_statement_items_raw.parquet')},
                'reference':{'candidates_xlsx':str(src_dir/'candidate_ratio_master.xlsx'),'candidates_sheet':'후보_재무비율'},
                'raw':{'dir':str(root/'data'/'raw'/'raw_all'),'patterns':{'재무상태표':'*재무상태표*.xlsx','손익계산서':'*손익계산서*.xlsx','현금흐름표':'*현금흐름표*.xlsx','자본변동표':'*자본변동표*.xlsx','이익잉여금처분계산서':'*이익잉여금*.xlsx','재무비율':'*재무비율*.xlsx'}}},
      'outputs':{'results':str(s2_out),'run_log':str(root/'runs'/'final_freeze'/'stage00_02_financial_ratio_engineering')},
      'acceptance':{'ratios_calculated_min':150,'quality_pass_min':100,'candidates_total_draft_min':150},
      'growth_ratio':{'base_year_offset':1,'positive_base_only':True,'cap_extreme':100.0}}
    p=root/'data'/'final_freeze'/'configs'/'oracle_components'/'stage00_02'/'paths.final_freeze.generated.yaml'; write_yaml(p,cfg); return p

def prepare_stage3_config(root, src_dir, s1_out, s2_out, s3_out):
    cfg={
      'inputs':{'stage1b':{'panel':str(s1_out/'firm_year_panel_v1.parquet'), 'statements':{k:str(s1_out/'cleaned_statement_panels'/f'{k}_clean.parquet') for k in ['재무상태표','손익계산서','현금흐름표','자본변동표','이익잉여금처분계산서','재무비율']}},
                'stage2':{'engineered_ratios':str(s2_out/'engineered_financial_ratios.parquet'),'candidate_pool':str(s2_out/'candidate_ratio_pool_by_item.csv'),'lag_support_income':str(s2_out/'lag_support'/'손익계산서_lag_support.parquet')},
                'raw_nonfinancial':{'general_info':{'kospi':str(root/'data/raw/raw_nonfinancial/kospi_kosdaq/코스피_전업종_폐지사 포함_일반사항.xlsx'),'kosdaq':str(root/'data/raw/raw_nonfinancial/kospi_kosdaq/코스닥_전업종_폐지사 포함_일반사항.xlsx'),'konex':str(root/'data/raw/raw_nonfinancial/konex_optional/코넥스_전업종_일반사항.xlsx')},'capital_change':{'kospi':str(root/'data/raw/raw_nonfinancial/kospi_kosdaq/코스피_전업종_폐지사 포함_자본금 변동사항.xlsx'),'kosdaq':str(root/'data/raw/raw_nonfinancial/kospi_kosdaq/코스닥_전업종_폐지사 포함_자본금 변동사항.xlsx'),'konex':str(root/'data/raw/raw_nonfinancial/konex_optional/코넥스_전업종_자본금 변동사항.xlsx')},'ma_events':{},'lawsuit':{}}},
      'outputs_v3_2':{'results':str(s3_out),'run_log':str(root/'runs/final_freeze/stage00_03_nonfinancial_metadata')},'outputs':{'results':str(s3_out),'run_log':str(root/'runs/final_freeze/stage00_03_nonfinancial_metadata')}}
    p=root/'data'/'final_freeze'/'configs'/'oracle_components'/'stage00_03'/'paths.final_freeze.generated.yaml'; write_yaml(p,cfg); return p

def prepare_variable_work(src_dir, s1_out, s2_out, s3_out, s4_out):
    s4_out.mkdir(parents=True, exist_ok=True)
    for name in ['stage1b','stage2','stage1c']:
        d=s4_out/name; d.mkdir(exist_ok=True)
    shutil.copy2(s1_out/'firm_year_panel_v1.parquet', s4_out/'stage1b'/'firm_year_panel_v1.parquet')
    for f in ['engineered_financial_ratios.parquet','candidate_ratio_pool_by_item.csv','ratio_quality_report.csv','ratio_audit_against_precomputed_panel.csv']:
        if (s2_out/f).exists(): shutil.copy2(s2_out/f, s4_out/'stage2'/f)
    for f in ['nonfinancial_metadata_panel.parquet','nonfinancial_candidate_pool_by_item.csv','nonfinancial_variable_quality_report.csv']:
        if (s3_out/f).exists(): shutil.copy2(s3_out/f, s4_out/'stage1c'/f)
    # Stage00-04 reads its config from data/final_freeze/configs/oracle_components via ORACLE_STAGE00_04_CONFIG.
    # No src/**/configs runtime dependency is allowed in final entrypoint.

def prepare_oracle_input(input_root, s1_out, s2_out, s3_out, s4_out):
    # Alpha/Beta/Gamma expected folder names.
    for sub in ['stage1b','stage2','stage1c_v3','stage3_v2']:
        (input_root/sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(s1_out/'firm_year_panel_v1.parquet', input_root/'stage1b'/'firm_year_panel_v1.parquet')
    shutil.copy2(s2_out/'engineered_financial_ratios.parquet', input_root/'stage2'/'engineered_financial_ratios.parquet')
    shutil.copy2(s3_out/'nonfinancial_metadata_panel.parquet', input_root/'stage1c_v3'/'nonfinancial_metadata_panel.parquet')
    for f in ['selected_variables_v2.json','direction_encoding_v2.json','weights_prior_v3.json','selected_variable_master.csv','simulator_ratio_alias_map.json']:
        src=s4_out/'outputs'/f
        if src.exists(): shutil.copy2(src, input_root/'stage3_v2'/f)

def expose_alpha_for_cross_oracle(oracle_input: Path, alpha_out: Path) -> None:
    """Expose the generated Alpha output under oracle_backend_input/stage4_alpha.

    Beta/Gamma robustness backends compute cross-oracle diagnostics only if this
    folder exists. Making it explicit turns cross-backend comparison from an
    optional skipped diagnostic into a final Stage01 acceptance artifact.
    """
    target = oracle_input / "stage4_alpha"
    target.mkdir(parents=True, exist_ok=True)
    for name in ["oracle_firm_year_output_alpha.parquet", "oracle_firm_year_output_alpha.csv", "preliminary_dev_oot_metrics_alpha.csv"]:
        src = alpha_out / name
        if src.exists():
            shutil.copy2(src, target / name)


def publish_stage1_contract_artifacts(config_dir: Path, s4_target: Path) -> None:
    """Copy final Oracle selection/alias artifacts into the active config root."""
    config_dir.mkdir(parents=True, exist_ok=True)
    for name in ["simulator_ratio_alias_map.json", "stage3_ratio_alias_map_used.json", "stage3_ratio_alias_master_used.csv", "selected_variable_master.csv"]:
        src = s4_target / name
        if src.exists():
            shutil.copy2(src, config_dir / name)


def copy_required_outputs(s4_out, target):
    target.mkdir(parents=True, exist_ok=True)
    out=s4_out/'outputs'
    for f in ['selected_variables_v2.json','direction_encoding_v2.json','weights_prior_v3.json','selected_variable_master.csv','variable_screening_metrics.csv','selection_score_table.csv','stage3_verdict.md','simulator_ratio_alias_map.json','stage3_ratio_alias_map_used.json','stage3_ratio_alias_master_used.csv']:
        if (out/f).exists(): shutil.copy2(out/f, target/f)

def write_registry(config_dir, backend_dir):
    # Store project-root relative paths for portability. Stage6 also contains a
    # resolver for backward compatibility with older absolute-path registries.
    base = 'data/final_freeze/stage1_oracle_backends'
    reg={
        'schema_version':'oracle_backend_registry_stage1_final_v28',
        'status':'generated_by_stage1_oracle_development_verified',
        'final_result_allowed':True,
        'generated_by':'credit_recourse.oracle.stage1.run_stage1_oracle_development',
        'created_utc':datetime.now(timezone.utc).isoformat(),
        'allowed_backend_types':{'alpha':'alpha_vanilla_isotonic_scorecard','beta':'beta_ordered_logit','gamma':'gamma_ml_tree_boosting'},
        'backends':{
            'alpha':{'backend_type':'alpha_vanilla_isotonic_scorecard','path':f'{base}/alpha','params':f'{base}/alpha/oracle_alpha_params.json','output':f'{base}/alpha/oracle_firm_year_output_alpha.parquet','metrics':f'{base}/alpha/preliminary_dev_oot_metrics_alpha.csv'},
            'beta':{'backend_type':'beta_ordered_logit','path':f'{base}/beta','params':f'{base}/beta/benchmark_beta_params.json','output':f'{base}/beta/benchmark_firm_year_output_beta.parquet'},
            'gamma':{'backend_type':'gamma_ml_tree_boosting','path':f'{base}/gamma','params':f'{base}/gamma/benchmark_gamma_params.json','model':f'{base}/gamma/benchmark_gamma_model.joblib','output':f'{base}/gamma/benchmark_firm_year_output_gamma.parquet'}
        },
        'no_placeholder_policy':'hard_fail_unless_allow_unscored'
    }
    write_yaml(config_dir/'oracle_backend_registry.yaml', reg); return reg


def _norm_code(x):
    import re
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    digits = re.sub(r"[^0-9]", "", s)
    return digits.zfill(6) if digits else s


def _ensure_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "거래소코드" not in out.columns:
        for c in ["firm_id", "stock_code", "code", "corp_code"]:
            if c in out.columns:
                out["거래소코드"] = out[c]
                break
    if "year" not in out.columns:
        for c in ["fiscal_year", "회계년도"]:
            if c in out.columns:
                out["year"] = out[c]
                break
    if "거래소코드" not in out.columns or "year" not in out.columns:
        raise KeyError(f"Cannot infer 거래소코드/year columns; columns={list(df.columns)[:80]}")
    out["거래소코드"] = out["거래소코드"].map(_norm_code)
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    return out


def _merge_by_firm_year(base: pd.DataFrame, other_path: Path, label: str) -> pd.DataFrame:
    if not other_path.exists():
        raise FileNotFoundError(f"Missing bridge source {label}: {other_path}")
    other = _ensure_keys(pd.read_parquet(other_path))
    if other.duplicated(["거래소코드", "year"]).any():
        raise ValueError(f"Bridge source has duplicate firm-year rows: {label} {other_path}")
    keep = [c for c in other.columns if c not in base.columns or c in ["거래소코드", "year"]]
    return base.merge(other[keep], on=["거래소코드", "year"], how="left")


def _load_alpha_selected_variables(alpha_params: Path) -> list[str]:
    return selected_variables_from_backend_params(alpha_params)


def build_stage1_to_stage2_bridge(root: Path, inputs: Path, backends: Path, cfgdir: Path) -> dict:
    """Create the rich Stage1→Stage2 bridge from Stage1 contract artifacts.

    This is deliberately not a blind copy of Alpha scored output. It joins the
    firm-year panel, engineered ratios, nonfinancial metadata, selected variable
    metadata, and Alpha score columns, then validates the dynamic selected-variable
    contract before Stage2 can run.
    """
    s1 = inputs / "stage00_01_rating_statement_integration" / "firm_year_panel_v1.parquet"
    s2 = inputs / "stage00_02_financial_ratio_engineering" / "engineered_financial_ratios.parquet"
    s3 = inputs / "stage00_03_nonfinancial_metadata" / "nonfinancial_metadata_panel.parquet"
    alpha_out = backends / "alpha" / "oracle_firm_year_output_alpha.parquet"
    alpha_params = backends / "alpha" / "oracle_alpha_params.json"
    base = _ensure_keys(pd.read_parquet(s1))
    if base.duplicated(["거래소코드", "year"]).any():
        raise ValueError("Stage1→Stage2 bridge refuses duplicate firm-year rows in Stage00_01 output")
    base["firm_id"] = base["거래소코드"].map(_norm_code)
    base["fiscal_year"] = pd.to_numeric(base["year"], errors="coerce").astype("Int64")
    if "rating_num_10" not in base.columns:
        if "rating_num" in base.columns:
            base["rating_num_10"] = pd.to_numeric(base["rating_num"], errors="coerce").astype("Int64")
        else:
            raise KeyError("Stage1 bridge requires rating_num_10")
    base["rating_num"] = pd.to_numeric(base["rating_num_10"], errors="coerce").astype("Int64")
    if "grade_base_10" not in base.columns and "grade_base" in base.columns:
        base["grade_base_10"] = base["grade_base"]
    if "split" not in base.columns:
        base["split"] = "dev"
        base.loc[base["fiscal_year"].astype(float) >= 2020, "split"] = "oot"
    joined = _merge_by_firm_year(base, s2, "Stage00_02 engineered ratios")
    joined = _merge_by_firm_year(joined, s3, "Stage00_03 nonfinancial metadata")
    if alpha_out.exists():
        alpha = _ensure_keys(pd.read_parquet(alpha_out))
        score_cols = [c for c in alpha.columns if c in ["거래소코드", "year"] or any(tok in str(c).lower() for tok in ["alpha", "score", "grade", "pred", "prob", "r_score"])]
        alpha_small = alpha[score_cols].copy()
        if alpha_small.duplicated(["거래소코드", "year"]).any():
            raise ValueError("Alpha backend output has duplicate firm-year rows; bridge cannot proceed")
        rename = {c: f"alpha__{c}" for c in alpha_small.columns if c not in ["거래소코드", "year"] and c in joined.columns}
        alpha_small = alpha_small.rename(columns=rename)
        joined = joined.merge(alpha_small, on=["거래소코드", "year"], how="left")
    contract = resolve_selected_variables(root / "data" / "final_freeze", alpha_params if alpha_params.exists() else None)
    selected = contract["selected_variables"]
    if not selected:
        raise ValueError("Stage1→Stage2 bridge could not resolve dynamic selected variables from selected_variable_master/backend params")
    missing = [v for v in selected if v not in joined.columns]
    if missing:
        raise ValueError("Stage1→Stage2 bridge missing selected variables: " + json.dumps(missing, ensure_ascii=False))
    joined["selected_variables_all_complete"] = joined[selected].notna().all(axis=1)
    dup = int(joined.duplicated(["firm_id", "fiscal_year"]).sum())
    if dup:
        raise ValueError(f"Stage1→Stage2 bridge output duplicate firm-year rows: {dup}")
    out = inputs / "alpha_vanilla_input_candidate.parquet"
    meta_path = inputs / "alpha_vanilla_input_candidate_metadata.json"
    joined.to_parquet(out, index=False)
    meta = {
        "stage_name": "stage1_to_stage2_bridge",
        "contract_version": "stage1_to_stage2_dynamic_selected_variables_v1",
        "status": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_paths": {"firm_year": str(s1), "ratios": str(s2), "nonfinancial": str(s3), "alpha_output": str(alpha_out), "alpha_params": str(alpha_params)},
        "output_paths": {"bridge_panel": str(out), "metadata": str(meta_path)},
        "row_counts": {"bridge_rows": int(len(joined)), "selected_variables_all_complete_rows": int(joined["selected_variables_all_complete"].sum())},
        "key_columns": ["firm_id", "fiscal_year"],
        "duplicate_key_count": dup,
        "selected_variables_used": selected,
        "selected_variable_master_path": contract.get("selected_variable_master_path"),
        "alpha_params_selected_variables": _load_alpha_selected_variables(alpha_params) if alpha_params.exists() else [],
        "missing_selected_variables": [],
    }
    write_json(meta_path, meta)
    return meta

def verify_stage1(backend_dir, s4_target):
    req=[s4_target/'selected_variables_v2.json',s4_target/'direction_encoding_v2.json',s4_target/'weights_prior_v3.json',s4_target/'selected_variable_master.csv',backend_dir/'alpha'/'oracle_alpha_params.json',backend_dir/'beta'/'benchmark_beta_params.json',backend_dir/'gamma'/'benchmark_gamma_params.json',backend_dir/'gamma'/'benchmark_gamma_model.joblib']
    miss=[str(p) for p in req if not p.exists() or p.stat().st_size==0]
    if miss: raise FileNotFoundError('Stage1 missing required outputs: '+json.dumps(miss,ensure_ascii=False))


def _oracle_backend_env(args, oracle_input: Path, out: Path, config_path: Path) -> dict:
    """Common backend execution environment.

    This is intentionally the single source of truth for final Oracle year scope
    and Dev/OOT split. Previously Stage1 accepted --score-end-year but always
    passed Include2024 to backends, making cutoff/split policy silently drift
    from the thesis contract.
    """
    env = _oracle_split_env(args)
    env.update({
        'ORACLE_INPUT_DIR': str(oracle_input),
        'ORACLE_OUTPUT_DIR': str(out),
        'ORACLE_CONFIG': str(config_path),
    })
    return env


def _stage1_split_policy(args) -> dict:
    return {
        'dev': f'{int(args.dev_start_year)}-{int(args.dev_end_year)}',
        'oot': f'{int(args.oot_start_year)}-{int(args.score_end_year)}',
        'score_end_year': int(args.score_end_year),
    }


def _oracle_split_env(args) -> dict:
    score_end = int(args.score_end_year)
    return {
        'ORACLE_MAX_YEAR': str(score_end),
        'ORACLE_YEAR_SCOPE': f'year_le_{score_end}',
        'ORACLE_DEV_YEARS': f'{int(args.dev_start_year)},{int(args.dev_end_year)}',
        'ORACLE_OOT_YEARS': f'{int(args.oot_start_year)},{score_end}',
    }

def _files_ready(paths: list[Path]) -> bool:
    for path in paths:
        try:
            if not path.exists() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def _parquet_ready(path: Path) -> bool:
    if not _files_ready([path]):
        return False
    try:
        pd.read_parquet(path, columns=None).head(1)
        return True
    except Exception:
        return False


def _resume_step(report: dict, stage: str, output: Path, required: list[Path], *, parquet: Path | None = None) -> bool:
    if _files_ready(required) and (parquet is None or _parquet_ready(parquet)):
        report['steps'].append({'stage':stage,'status':'SKIP_REUSED','output':str(output),'resume_reason':'required outputs already exist and are readable'})
        return True
    return False


def _stage00_01_required(s1: Path) -> list[Path]:
    return [s1/'firm_year_panel_v1.parquet', s1/'financial_statement_items_raw.parquet']


def _stage00_02_required(s2: Path) -> list[Path]:
    return [s2/'engineered_financial_ratios.parquet', s2/'candidate_ratio_pool_by_item.csv', s2/'ratio_quality_report.csv']


def _stage00_03_required(s3: Path) -> list[Path]:
    return [s3/'nonfinancial_metadata_panel.parquet', s3/'nonfinancial_candidate_pool_by_item.csv', s3/'nonfinancial_variable_quality_report.csv']


def _stage00_04_required(s4: Path) -> list[Path]:
    return [s4/'selected_variables_v2.json', s4/'direction_encoding_v2.json', s4/'weights_prior_v3.json', s4/'selected_variable_master.csv']


def _backend_required(out: Path, backend: str) -> list[Path]:
    if backend == 'alpha':
        return [out/'oracle_alpha_params.json', out/'oracle_firm_year_output_alpha.parquet']
    if backend == 'beta':
        return [out/'benchmark_beta_params.json', out/'benchmark_firm_year_output_beta.parquet']
    if backend == 'gamma':
        return [out/'benchmark_gamma_params.json', out/'benchmark_gamma_model.joblib', out/'benchmark_firm_year_output_gamma.parquet']
    raise ValueError(backend)



STAGE1_STEP_ORDER = [
    "stage00_01",
    "stage00_02",
    "stage00_03",
    "stage00_04",
    "backend_alpha",
    "backend_beta",
    "backend_gamma",
    "bridge",
    "diagnostics",
    "substrate_validation",
]

def _stage_index(step: str) -> int:
    try:
        return STAGE1_STEP_ORDER.index(step)
    except ValueError as exc:
        raise ValueError(f"Unknown Stage1 step: {step}; allowed={STAGE1_STEP_ORDER}") from exc

def _normalize_step_window(args) -> tuple[str, str, int, int]:
    if getattr(args, "only_step", None):
        args.start_step = args.only_step
        args.end_step = args.only_step
    start = args.start_step or STAGE1_STEP_ORDER[0]
    end = args.end_step or STAGE1_STEP_ORDER[-1]
    start_i = _stage_index(start)
    end_i = _stage_index(end)
    if start_i > end_i:
        raise ValueError(f"Invalid Stage1 step window: start_step={start} is after end_step={end}")
    return start, end, start_i, end_i

def _step_in_window(step: str, start_i: int, end_i: int) -> bool:
    idx = _stage_index(step)
    return start_i <= idx <= end_i

def _step_before_window(step: str, start_i: int) -> bool:
    return _stage_index(step) < start_i

def _require_step_ready(report: dict, stage: str, output: Path, required: list[Path], *, parquet: Path | None = None) -> None:
    if _resume_step(report, stage, output, required, parquet=parquet):
        report['steps'][-1]['status'] = 'SKIP_REQUIRED_PREEXISTING'
        report['steps'][-1]['resume_reason'] = 'step is before requested start_step; artifacts must already exist and be readable'
        return
    missing = [str(p) for p in required if not p.exists() or p.stat().st_size <= 0]
    raise FileNotFoundError(f"Requested partial Stage1 run starts after {stage}, but required prerequisite artifacts are not ready. Missing/unreadable: {json.dumps(missing, ensure_ascii=False)}")

def _finish_if_window_ended(report: dict, ledgers: Path, step: str, end_i: int) -> bool:
    if _stage_index(step) == end_i:
        report['status'] = 'PASS_PARTIAL' if step != STAGE1_STEP_ORDER[-1] else 'PASS'
        report['final_result_allowed'] = (step == STAGE1_STEP_ORDER[-1])
        report['partial_stop_after'] = step
        (ledgers/'stage1_oracle_backends_full_development.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return True
    return False

def main(argv=None):
    configure_utf8_stdio()
    ap=argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--score-end-year', type=int, default=2023, help='Maximum fiscal year included in Oracle scoring/evaluation.')
    ap.add_argument('--dev-start-year', type=int, default=2002, help='First fiscal year in Oracle development split.')
    ap.add_argument('--dev-end-year', type=int, default=2019, help='Last fiscal year in Oracle development split.')
    ap.add_argument('--oot-start-year', type=int, default=2020, help='First fiscal year in Oracle OOT split.')
    ap.add_argument('--clean', action='store_true')
    ap.add_argument('--reuse-stage1-inputs', action='store_true', help='When used with --clean, preserve and reuse existing Stage00_01~Stage00_04 Oracle input artifacts instead of rematerializing wide statement/ratio/nonfinancial panels. Missing/corrupt input artifacts are rebuilt.')
    ap.add_argument('--clean-backends-only', action='store_true', help='Shortcut for rerunning only Oracle backends/diagnostics while preserving Stage1 input artifacts. Equivalent to --clean --reuse-stage1-inputs for Stage00_01~Stage00_04.')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--pending-ok', action='store_true')
    ap.add_argument('--raw-rating-dir', default=None)
    ap.add_argument('--force-stage0-repair', action='store_true')
    ap.add_argument('--start-step', choices=STAGE1_STEP_ORDER, default=STAGE1_STEP_ORDER[0], help='First Stage1 step to execute. Earlier steps are treated as required pre-existing prerequisites.')
    ap.add_argument('--end-step', choices=STAGE1_STEP_ORDER, default=STAGE1_STEP_ORDER[-1], help='Last Stage1 step to execute. Stops cleanly after this step with PASS_PARTIAL unless it is substrate_validation.')
    ap.add_argument('--only-step', choices=STAGE1_STEP_ORDER, default=None, help='Execute exactly one Stage1 step; equivalent to --start-step X --end-step X.')
    args=ap.parse_args(argv)
    if args.clean_backends_only:
        args.clean = True
        args.reuse_stage1_inputs = True
    if args.dev_end_year >= args.oot_start_year:
        raise ValueError(f'Invalid split: dev_end_year={args.dev_end_year} must be < oot_start_year={args.oot_start_year}')
    if args.score_end_year < args.oot_start_year:
        raise ValueError(f'Invalid split: score_end_year={args.score_end_year} must be >= oot_start_year={args.oot_start_year}')
    start_step, end_step, start_i, end_i = _normalize_step_window(args)
    if start_i > 0:
        args.resume = True
    root=Path(args.project_root).resolve(); final, inputs, backends, ledgers, cfgdir=stage_paths(root)
    stage0=final/'stage0_oracle_foundation'
    if not stage0.exists(): raise FileNotFoundError(f'Missing protected Stage0: {stage0}')
    if args.clean:
        clean_dirs = [backends/'alpha', backends/'beta', backends/'gamma']
        if not args.reuse_stage1_inputs:
            clean_dirs = [
                inputs/'stage00_01_rating_statement_integration',
                inputs/'stage00_02_financial_ratio_engineering',
                inputs/'stage00_03_nonfinancial_metadata',
                inputs/'stage00_04_variable_selection',
            ] + clean_dirs
        for d in clean_dirs:
            if d.exists(): shutil.rmtree(d)
    ledgers.mkdir(parents=True,exist_ok=True); inputs.mkdir(parents=True,exist_ok=True); backends.mkdir(parents=True,exist_ok=True); cfgdir.mkdir(parents=True,exist_ok=True)
    report={'stage':'stage1_oracle_development','created_utc':datetime.now(timezone.utc).isoformat(),'final_result_allowed':False,'steps':[],'methodology':'ported_old_oracle_core_in_final_package','split_policy':_stage1_split_policy(args),'step_window':{'start_step':start_step,'end_step':end_step,'only_step':args.only_step}}
    try:
        # Stage0 rating sampling contract is a prerequisite for every Oracle backend.
        # Validate first; if the protected Stage0 panel is stale/contaminated, rebuild
        # it explicitly from raw rating workbooks before Stage00-01 consumes it.
        raw_rating_dir = Path(args.raw_rating_dir).resolve() if args.raw_rating_dir else root/'data'/'raw'/'rating_sample'
        ok0, errors0, meta0 = validate_stage0_contract(stage0)
        if not ok0 and not args.force_stage0_repair:
            raise ValueError('Stage0 rating contract validation failed; Stage1 is read-only and will not repair Stage0 unless --force-stage0-repair is explicitly provided: ' + json.dumps(errors0, ensure_ascii=False))
        if args.force_stage0_repair:
            if not raw_rating_dir.exists():
                raise FileNotFoundError('Stage0 repair requested but raw rating dir is unavailable: ' + str(raw_rating_dir))
            repair_meta = repair_stage0_canonical(stage0, raw_rating_dir)
            report['steps'].append({'stage':'stage0_rating_contract_repair_explicit','status':'PASS','output':str(stage0/'canonical_panel'),'pre_repair_errors':errors0,'metadata':repair_meta})
        else:
            report['steps'].append({'stage':'stage0_rating_contract_validation','status':'PASS','output':str(stage0/'canonical_panel'),'metadata':meta0})

        # 00-01
        s1=inputs/'stage00_01_rating_statement_integration'
        if _step_before_window('stage00_01', start_i):
            _require_step_ready(report, 'stage00_01', s1, _stage00_01_required(s1), parquet=s1/'firm_year_panel_v1.parquet')
        elif _step_in_window('stage00_01', start_i, end_i):
            if not ((args.resume or args.reuse_stage1_inputs) and _resume_step(report, 'stage00_01', s1, _stage00_01_required(s1), parquet=s1/'firm_year_panel_v1.parquet')):
                report['steps'].append(materialize_stage00_01(stage0, s1))
            context_meta = ensure_stage00_01_context_contract(stage0, s1)
            report['steps'].append({'stage':'stage00_01_context_contract','status':'PASS','output':str(s1),'metadata':context_meta})
            if _finish_if_window_ended(report, ledgers, 'stage00_01', end_i): return 0
        else:
            _require_step_ready(report, 'stage00_01', s1, _stage00_01_required(s1), parquet=s1/'firm_year_panel_v1.parquet')
        stage1_dir=Path(__file__).resolve().parent
        oracle_dir=stage1_dir.parent
        src2=stage1_dir/'stage00_02_ratio_engineering'
        src3=stage1_dir/'stage00_03_nonfinancial'
        src4=stage1_dir/'stage00_04_variable_selection'
        backend_root=oracle_dir/'backends'
        required_dirs=[src2,src3,src4,backend_root/'alpha',backend_root/'beta',backend_root/'gamma']
        for rp in required_dirs:
            if not rp.exists():
                raise FileNotFoundError(f'Missing internal Oracle package path: {rp}')
        # 00-02
        s2=inputs/'stage00_02_financial_ratio_engineering'; cfgbase=oracle_component_config_dir(root)
        if _step_before_window('stage00_02', start_i):
            _require_step_ready(report, 'stage00_02', s2, _stage00_02_required(s2), parquet=s2/'engineered_financial_ratios.parquet')
        elif _step_in_window('stage00_02', start_i, end_i):
            if not ((args.resume or args.reuse_stage1_inputs) and _resume_step(report, 'stage00_02', s2, _stage00_02_required(s2), parquet=s2/'engineered_financial_ratios.parquet')):
                cfg2=prepare_stage2_config(root,src2,s1,s2); run_py(src2/'pipeline.py',['--config',str(cfg2),'--stage-config',str(cfgbase/'stage00_02'/'stage_config.yaml')], env={'ORACLE_STAGE00_02_BUNDLE_DIR': str(src2), 'STAGE2_DUPLICATE_ALIAS_MASTER': str(cfgbase/'stage00_02'/'duplicate_ratio_alias_master.csv')}); report['steps'].append({'stage':'stage00_02','status':'PASS','output':str(s2)})
            if _finish_if_window_ended(report, ledgers, 'stage00_02', end_i): return 0
        else:
            _require_step_ready(report, 'stage00_02', s2, _stage00_02_required(s2), parquet=s2/'engineered_financial_ratios.parquet')
        # 00-03
        s3=inputs/'stage00_03_nonfinancial_metadata'
        if _step_before_window('stage00_03', start_i):
            _require_step_ready(report, 'stage00_03', s3, _stage00_03_required(s3), parquet=s3/'nonfinancial_metadata_panel.parquet')
        elif _step_in_window('stage00_03', start_i, end_i):
            if not ((args.resume or args.reuse_stage1_inputs) and _resume_step(report, 'stage00_03', s3, _stage00_03_required(s3), parquet=s3/'nonfinancial_metadata_panel.parquet')):
                cfg3=prepare_stage3_config(root,src3,s1,s2,s3); run_py(src3/'pipeline.py',['--config',str(cfg3),'--stage-config',str(cfgbase/'stage00_03'/'stage_config.yaml'),'--var-mapping',str(cfgbase/'stage00_03'/'variable_mapping.yaml')]); report['steps'].append({'stage':'stage00_03','status':'PASS','output':str(s3)})
            if _finish_if_window_ended(report, ledgers, 'stage00_03', end_i): return 0
        else:
            _require_step_ready(report, 'stage00_03', s3, _stage00_03_required(s3), parquet=s3/'nonfinancial_metadata_panel.parquet')
        # 00-04
        s4=inputs/'stage00_04_variable_selection'
        if _step_before_window('stage00_04', start_i):
            _require_step_ready(report, 'stage00_04', s4, _stage00_04_required(s4))
        elif _step_in_window('stage00_04', start_i, end_i):
            if not ((args.resume or args.reuse_stage1_inputs) and _resume_step(report, 'stage00_04', s4, _stage00_04_required(s4))):
                prepare_variable_work(src4,s1,s2,s3,s4)
                env4 = _oracle_split_env(args)
                env4.update({'ORACLE_STAGE00_04_CONFIG': str(cfgbase/'stage00_04'/'stage3_config.yaml')})
                run_py(src4/'pipeline.py',cwd=s4, env=env4)
                copy_required_outputs(s4, s4); report['steps'].append({'stage':'stage00_04','status':'PASS','output':str(s4)})
            publish_stage1_contract_artifacts(cfgdir, s4)
            if _finish_if_window_ended(report, ledgers, 'stage00_04', end_i): return 0
        else:
            _require_step_ready(report, 'stage00_04', s4, _stage00_04_required(s4))
            publish_stage1_contract_artifacts(cfgdir, s4)
        # backends
        oracle_input=inputs/'oracle_backend_input'; prepare_oracle_input(oracle_input,s1,s2,s3,s4)

        # Alpha is the main frozen reference Oracle and must be generated first.
        alpha_out=backends/'alpha'; alpha_out.mkdir(parents=True,exist_ok=True)
        if _step_before_window('backend_alpha', start_i):
            _require_step_ready(report, 'backend_alpha', alpha_out, _backend_required(alpha_out, 'alpha'), parquet=alpha_out/'oracle_firm_year_output_alpha.parquet')
        elif _step_in_window('backend_alpha', start_i, end_i):
            if args.resume and _resume_step(report, 'backend_alpha', alpha_out, _backend_required(alpha_out, 'alpha'), parquet=alpha_out/'oracle_firm_year_output_alpha.parquet'):
                pass
            else:
                run_py(backend_root/'alpha'/'pipeline.py', env=_oracle_backend_env(args, oracle_input, alpha_out, cfgbase/'alpha'/'stage4_alpha_config.yaml'))
                if not (alpha_out/'oracle_alpha_params.json').exists():
                    raise FileNotFoundError(f'alpha backend params not created: {alpha_out/"oracle_alpha_params.json"}')
                report['steps'].append({'stage':'backend_alpha','status':'PASS','output':str(alpha_out)})
            if _finish_if_window_ended(report, ledgers, 'backend_alpha', end_i): return 0
        else:
            _require_step_ready(report, 'backend_alpha', alpha_out, _backend_required(alpha_out, 'alpha'), parquet=alpha_out/'oracle_firm_year_output_alpha.parquet')
        expose_alpha_for_cross_oracle(oracle_input, alpha_out)

        # Beta/Gamma are robustness backends. They now receive Alpha output for
        # mandatory cross-backend diagnostics.
        for b, sub, params in [('beta','beta','benchmark_beta_params.json'),('gamma','gamma','benchmark_gamma_params.json')]:
            step_name=f'backend_{b}'
            out=backends/b; out.mkdir(parents=True,exist_ok=True); srcb=backend_root/sub
            if _step_before_window(step_name, start_i):
                _require_step_ready(report, step_name, out, _backend_required(out, b), parquet=out/(f'benchmark_firm_year_output_{b}.parquet'))
                continue
            if _step_in_window(step_name, start_i, end_i):
                if args.resume and _resume_step(report, step_name, out, _backend_required(out, b), parquet=out/(f'benchmark_firm_year_output_{b}.parquet')):
                    pass
                else:
                    env=_oracle_backend_env(args, oracle_input, out, cfgbase/b/f'stage4_{b}_config.yaml')
                    run_py(srcb/'pipeline.py',env=env)
                    if not (out/params).exists(): raise FileNotFoundError(f'{b} backend params not created: {out/params}')
                    report['steps'].append({'stage':step_name,'status':'PASS','output':str(out)})
                if _finish_if_window_ended(report, ledgers, step_name, end_i): return 0
            else:
                _require_step_ready(report, step_name, out, _backend_required(out, b), parquet=out/(f'benchmark_firm_year_output_{b}.parquet'))

        publish_stage1_contract_artifacts(cfgdir, s4)
        reg=write_registry(cfgdir,backends); report['registry']=reg
        if not _step_in_window('bridge', start_i, end_i):
            _require_step_ready(report, 'bridge', inputs, [inputs/'alpha_vanilla_input_candidate.parquet', inputs/'alpha_vanilla_input_candidate_metadata.json'], parquet=inputs/'alpha_vanilla_input_candidate.parquet')
        else:
            bridge_meta = build_stage1_to_stage2_bridge(root, inputs, backends, cfgdir)
            report['steps'].append({'stage':'stage1_to_stage2_bridge','status':'PASS','output':bridge_meta['output_paths']['bridge_panel'],'metadata':bridge_meta})
            if _finish_if_window_ended(report, ledgers, 'bridge', end_i): return 0
        verify_stage1(backends,s4)
        if not _step_in_window('diagnostics', start_i, end_i):
            _require_step_ready(report, 'diagnostics', ledgers, [ledgers/'oracle_backend_diagnostic_report.json', ledgers/'oracle_backend_gate_summary.csv'])
            diagnostic_report = read_json(ledgers/'oracle_backend_diagnostic_report.json')
        else:
            diagnostic_report = diagnose_backend_dir(backends)
        diagnostic_path = ledgers/'oracle_backend_diagnostic_report.json'
        gate_path = ledgers/'oracle_backend_gate_summary.csv'
        diagnostic_path.write_text(json.dumps(diagnostic_report, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        pd.DataFrame(diagnostic_report.get('gates', [])).to_csv(gate_path, index=False, encoding='utf-8-sig')
        report['oracle_backend_diagnostic'] = {
            'status': diagnostic_report.get('status'),
            'verdict': diagnostic_report.get('verdict'),
            'output_json': str(diagnostic_path),
            'output_csv': str(gate_path),
            'required_failed_gates': diagnostic_report.get('required_failed_gates', []),
            'advisory_failed_gates': diagnostic_report.get('advisory_failed_gates', []),
        }
        if diagnostic_report.get('status') != 'PASS':
            raise ValueError('Oracle backend diagnostic required gates failed: ' + json.dumps(diagnostic_report.get('required_failed_gates', []), ensure_ascii=False, default=str))
        if _finish_if_window_ended(report, ledgers, 'diagnostics', end_i): return 0

        from credit_recourse.oracle.verification import verify_stage1_substrate_validation
        if not _step_in_window('substrate_validation', start_i, end_i):
            _require_step_ready(report, 'substrate_validation', ledgers, [ledgers/'stage1_substrate_validation_loopB1.json'])
            substrate_rc = 0
        else:
            substrate_rc = verify_stage1_substrate_validation.main(['--project-root', str(root)])
        substrate_path = ledgers / 'stage1_substrate_validation_loopB1.json'
        report['steps'].append({'stage':'stage1_substrate_validation_loopB1','status':'PASS' if substrate_rc == 0 else 'FAIL','output':str(substrate_path),'return_code':int(substrate_rc)})
        if substrate_rc != 0:
            raise ValueError(f'Stage1 substrate validation Loop B1 failed; see {substrate_path}')

        report['status']='PASS'; report['final_result_allowed']=True
    except Exception as exc:
        report['status']='FAIL'; report['final_result_allowed']=False; report['error']=repr(exc)
        (ledgers/'stage1_oracle_backends_full_development.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        if args.pending_ok:
            print(json.dumps(report,ensure_ascii=False,indent=2)); return 2
        raise
    (ledgers/'stage1_oracle_backends_full_development.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
