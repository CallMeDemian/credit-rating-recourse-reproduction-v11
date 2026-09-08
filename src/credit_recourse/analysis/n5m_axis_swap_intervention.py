"""
N5M/C4R axis-swap intervention: one-axis-at-a-time (OAT) base->target swaps and
Monte-Carlo Shapley decomposition of the policy-pair gain, scored
through the SAME frozen simulator + 3-Oracle path as Stage 8.

Thesis 10.5 후속검증 항목 2의 실행 스크립트. evaluator-only: LLM API 재호출 없음.
동결 규율: 지정한 base/target 정책을 먼저 재채점하여 frozen stage8 점수와의 per-row 일치를
assert(허용오차 --fidelity-tol)한 뒤에만 hybrid 행동을 채점한다. 불일치 시 즉시 중단.

Usage (from thesis_repo root, same venv as pipeline):
  python -m credit_recourse.analysis.n5m_axis_swap_intervention ^
      --project-root C:\\Users\\Demian\\Desktop\\thesis_repo ^
      --arm 0p75=data\\final_freeze\\archives\\N5M_C4C6_L1_0p75_ICb_gpt54mini_p50_main_seed1_20260712_161725 ^
      --arm unbounded=data\\final_freeze\\archives\\N5M_C4C6_L1_unbounded_ICb_gpt54mini_p50_main_seed1_20260712_161725 ^
      --shapley-permutations 64 --out data\\analysis\\n5m_axis_swap

(--arm은 4팔 모두 지정 가능. OAT는 지정된 모든 팔, Shapley는 --shapley-arms 기본 0p75,unbounded.)
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from credit_recourse.analysis.llm_action_budget_ablation import _score_action_table, _load_action_space_for_stage7
from credit_recourse.analysis.n5_7_10c_holm_inference import holm_adjust, p_to_stars, wilcoxon_paired_p
AXES = ['action__ppe_pct', 'action__inv_turnover_chg', 'action__ar_turnover_chg', 'action__ap_turnover_chg', 'action__short_debt_pct', 'action__long_debt_pct', 'action__bond_pct', 'action__revenue_growth', 'action__cogs_ratio_chg', 'action__sga_ratio_chg']
ORACLES = ['alpha', 'beta', 'gamma']

def _read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except ImportError:
        return pd.read_csv(path)

def _load_arm(archive: Path):
    inner = archive
    if not (inner / 'stage7_llm_action_generation').is_dir():
        sub = [d for d in archive.iterdir() if (d / 'stage7_llm_action_generation').is_dir()]
        if len(sub) != 1:
            raise FileNotFoundError(f'stage7 dir not found under {archive}')
        inner = sub[0]
    a7 = _read_table(inner / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet')
    s8 = _read_table(inner / 'stage8_llm_multi_oracle_eval' / 'llm_stage8_multi_oracle_scores.parquet')
    return (inner, a7, s8)

def _select_frozen_policy_pair(frozen: pd.DataFrame, *, base_policy: str, target_policy: str, mode: str='free_form_10d', expected_n_per_policy: int=575, required_row_ids: np.ndarray | None=None) -> pd.DataFrame:
    """Select the exact frozen Stage8 policy pair and optional fixed cohort."""
    required = {'row_id', 'policy'}
    missing = sorted(required - set(frozen.columns))
    if missing:
        raise ValueError(f'frozen Stage8 table missing columns: {missing}')
    base_policy = str(base_policy).strip()
    target_policy = str(target_policy).strip()
    if not base_policy or not target_policy or base_policy == target_policy:
        raise ValueError(f'base/target policies must be distinct non-empty names; base={base_policy!r}, target={target_policy!r}')
    mask = frozen['policy'].astype(str).isin([base_policy, target_policy])
    if 'mode' in frozen.columns:
        mask &= frozen['mode'].astype(str).eq(str(mode))
    selected = frozen.loc[mask].copy()
    selected['row_id'] = pd.to_numeric(selected['row_id'], errors='raise').astype(int)
    key = ['row_id', 'policy'] + (['mode'] if 'mode' in selected.columns else [])
    if selected.duplicated(key).any():
        raise ValueError(f'duplicate frozen Stage8 policy-pair keys: key={key}')
    if required_row_ids is not None:
        required_ids = np.asarray(required_row_ids, dtype=int)
        if len(required_ids) != len(np.unique(required_ids)):
            raise ValueError('required_row_ids contains duplicates')
        required_set = set(required_ids.tolist())
        selected = selected[selected['row_id'].isin(required_set)].copy()
        for policy in (base_policy, target_policy):
            observed_set = set(selected.loc[selected['policy'].astype(str).eq(policy), 'row_id'].astype(int).tolist())
            missing_ids = sorted(required_set - observed_set)
            extra_ids = sorted(observed_set - required_set)
            if missing_ids or extra_ids:
                raise ValueError(f'frozen Stage8 {policy} cohort mismatch; missing_sample={missing_ids[:10]}, extra_sample={extra_ids[:10]}')
    counts = selected['policy'].astype(str).value_counts().to_dict()
    expected = {base_policy: int(expected_n_per_policy), target_policy: int(expected_n_per_policy)}
    observed = {name: int(counts.get(name, 0)) for name in expected}
    if observed != expected:
        raise ValueError(f'frozen Stage8 policy-pair row count mismatch; expected={expected}, observed={observed}')
    return selected.sort_values(key).reset_index(drop=True)

def _load_required_row_ids(path: Path, expected_n: int) -> np.ndarray:
    """Read one fixed complete-case cohort from a CSV containing row_id."""
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'cohort row-id file missing: {path}')
    frame = pd.read_csv(path)
    if 'row_id' not in frame.columns:
        raise ValueError(f'cohort row-id file must contain row_id column: {path}')
    row_ids = pd.to_numeric(frame['row_id'], errors='raise').astype(int).to_numpy()
    if len(row_ids) != expected_n:
        raise ValueError(f'cohort row-id count mismatch: expected={expected_n}, observed={len(row_ids)}, path={path}')
    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError(f'cohort row-id file contains duplicates: {path}')
    return row_ids

def _fidelity_check(scored: pd.DataFrame, frozen: pd.DataFrame, tol: float) -> dict:
    key = ['row_id', 'policy'] + (['mode'] if 'mode' in scored.columns and 'mode' in frozen.columns else [])
    if scored.duplicated(key).any() or frozen.duplicated(key).any():
        raise AssertionError(f'FIDELITY KEY NOT UNIQUE: key={key}')
    m = scored.merge(frozen, on=key, suffixes=('_new', '_frz'), how='outer', validate='one_to_one', indicator=True)
    if not m['_merge'].eq('both').all():
        sample = m.loc[m['_merge'].ne('both'), key + ['_merge']].head(10).to_dict('records')
        raise AssertionError(f'FIDELITY ROW ALIGNMENT FAIL: key={key}, sample={sample}')
    m = m.drop(columns='_merge')
    rep = {}
    for o in ORACLES:
        d = (m[f'delta_R_score_{o}_new'] - m[f'delta_R_score_{o}_frz']).abs()
        rep[o] = {'max_abs_diff': float(d.max()), 'n': int(len(d))}
        if d.max() > tol:
            bad = m.loc[d.idxmax(), ['row_id', 'policy']].to_dict()
            raise AssertionError(f'FIDELITY FAIL oracle={o} max|diff|={d.max():.3e} > tol={tol} at {bad}. Frozen substrate not reproduced; aborting before any hybrid scoring.')
    return rep

def _select_policy_pair(action_table: pd.DataFrame, *, base_policy: str, target_policy: str, expected_n: int=575, required_row_ids: np.ndarray | None=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select and hard-validate one aligned free-form policy pair."""
    base_policy = str(base_policy).strip()
    target_policy = str(target_policy).strip()
    if not base_policy or not target_policy or base_policy == target_policy:
        raise ValueError(f'base/target policies must be distinct non-empty names; base={base_policy!r}, target={target_policy!r}')
    base = action_table[(action_table.policy.astype(str) == base_policy) & (action_table['mode'].astype(str) == 'free_form_10d')].sort_values('row_id').reset_index(drop=True)
    target = action_table[(action_table.policy.astype(str) == target_policy) & (action_table['mode'].astype(str) == 'free_form_10d')].sort_values('row_id').reset_index(drop=True)
    if required_row_ids is not None:
        required_ids = np.asarray(required_row_ids, dtype=int)
        required_set = set(required_ids.tolist())
        base = base[base['row_id'].isin(required_set)].copy()
        target = target[target['row_id'].isin(required_set)].copy()
        order = {int(row_id): idx for idx, row_id in enumerate(required_ids.tolist())}
        base['_cohort_order'] = base['row_id'].map(order)
        target['_cohort_order'] = target['row_id'].map(order)
        base = base.sort_values('_cohort_order').drop(columns='_cohort_order').reset_index(drop=True)
        target = target.sort_values('_cohort_order').drop(columns='_cohort_order').reset_index(drop=True)
    if len(base) != expected_n or len(target) != expected_n:
        raise ValueError(f'expected exactly {expected_n} {base_policy} and {expected_n} {target_policy} free-form rows; got {base_policy}={len(base)}, {target_policy}={len(target)}')
    if base.duplicated(['row_id', 'policy', 'mode']).any() or target.duplicated(['row_id', 'policy', 'mode']).any():
        raise ValueError(f'duplicate Stage7 {base_policy}/{target_policy} row keys')
    if not (base.row_id.to_numpy() == target.row_id.to_numpy()).all():
        raise ValueError(f'{base_policy}/{target_policy} row alignment broken')
    return (base, target)

def _oat_tables(c4: pd.DataFrame, c6: pd.DataFrame) -> pd.DataFrame:
    tabs = []
    base = c4.copy()
    for ax in AXES:
        t = base.copy()
        t[ax] = c6[ax].to_numpy()
        t['policy'] = f'OAT_{ax}'
        t['candidate_id'] = 'hybrid'
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True)

def _shapley_tables(c4: pd.DataFrame, c6: pd.DataFrame, perm: list[int], pidx: int) -> pd.DataFrame:
    tabs = []
    cur = c4.copy()
    for step, ai in enumerate(perm):
        cur = cur.copy()
        cur[AXES[ai]] = c6[AXES[ai]].to_numpy()
        t = cur.copy()
        t['policy'] = f'SHAP_p{pidx}_s{step}_a{ai}'
        t['candidate_id'] = 'hybrid'
        tabs.append(t)
    return pd.concat(tabs, ignore_index=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True, type=Path)
    ap.add_argument('--arm', action='append', required=True, help='label=path_to_archive (repeatable)')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--base-policy', default='C4', help='Stage7 free-form base policy; default preserves N5M C4.')
    ap.add_argument('--target-policy', default='C6', help='Stage7 free-form target policy; default preserves N5M C6.')
    ap.add_argument('--expected-n', type=int, default=575, help='Expected firms per selected policy after optional cohort filtering.')
    ap.add_argument('--row-id-file', type=Path, default=None, help='Optional CSV with a fixed row_id cohort applied to both policies and frozen Stage8.')
    ap.add_argument('--shapley-permutations', type=int, default=64)
    ap.add_argument('--shapley-arms', default='0p75,unbounded')
    ap.add_argument('--seed', type=int, default=20260714)
    ap.add_argument('--fidelity-tol', type=float, default=1e-09)
    ap.add_argument('--efficiency-tol', type=float, default=1e-08)
    ap.add_argument('--oat-only', action='store_true')
    args = ap.parse_args()
    if args.shapley_permutations <= 0:
        raise ValueError(f'--shapley-permutations must be positive; got {args.shapley_permutations}')
    if args.expected_n <= 0:
        raise ValueError(f'--expected-n must be positive; got {args.expected_n}')
    if args.fidelity_tol < 0 or args.efficiency_tol < 0:
        raise ValueError('fidelity and efficiency tolerances must be non-negative')
    root = args.project_root.resolve()
    required_row_ids = _load_required_row_ids(args.row_id_file, args.expected_n) if args.row_id_file is not None else None
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    shapley_arms = set((x.strip() for x in args.shapley_arms.split(',')))
    manifest = {'schema': 'n5m_axis_swap_intervention_v3', 'seed': args.seed, 'base_policy': args.base_policy, 'target_policy': args.target_policy, 'expected_n': int(args.expected_n), 'row_id_file': str(args.row_id_file.resolve()) if args.row_id_file is not None else None, 'fidelity_tol': args.fidelity_tol, 'efficiency_tol': args.efficiency_tol, 'arms': {}, 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    seen_labels: set[str] = set()
    for spec in args.arm:
        label, separator, pth = spec.partition('=')
        label = label.strip()
        pth = pth.strip()
        if separator != '=' or not label or (not pth):
            raise ValueError(f'--arm must use non-empty label=path syntax; got {spec!r}')
        if label in seen_labels:
            raise ValueError(f'duplicate --arm label: {label!r}')
        seen_labels.add(label)
        archive = (root / pth).resolve() if not Path(pth).is_absolute() else Path(pth).resolve()
        if not archive.is_dir():
            raise FileNotFoundError(f'N5M arm archive directory missing: {archive}')
        inner, a7, s8_frozen = _load_arm(archive)
        space, prov = _load_action_space_for_stage7(root, inner / 'stage7_llm_action_generation' / 'llm_stage7_action_table.parquet')
        keep = ['row_id', 'policy', 'candidate_id', 'mode'] + AXES
        missing = sorted(set(keep) - set(a7.columns))
        if missing:
            raise ValueError(f'{label}: Stage7 action table missing columns: {missing}')
        a7k = a7[keep].copy()
        a7k['row_id'] = pd.to_numeric(a7k['row_id'], errors='raise').astype(int)
        try:
            c4, c6 = _select_policy_pair(a7k, base_policy=args.base_policy, target_policy=args.target_policy, expected_n=args.expected_n, required_row_ids=required_row_ids)
        except Exception as exc:
            raise type(exc)(f'{label}: {exc}') from exc
        arm_out = out / label
        arm_out.mkdir(exist_ok=True)
        t0 = time.time()
        orig = pd.concat([c4, c6], ignore_index=True)
        scored0, _, _, sim_id = _score_action_table(project_root=root, action_table=orig, out=arm_out / '_fidelity', space=space, output_prefix='fidelity', write_outputs=False)
        s8_pair = _select_frozen_policy_pair(s8_frozen, base_policy=args.base_policy, target_policy=args.target_policy, mode='free_form_10d', expected_n_per_policy=args.expected_n, required_row_ids=required_row_ids)
        fid = _fidelity_check(scored0, s8_pair, args.fidelity_tol)
        oat = _oat_tables(c4, c6)
        s_oat, _, _, _ = _score_action_table(project_root=root, action_table=oat, out=arm_out / '_oat', space=space, output_prefix='oat', write_outputs=False)
        base4 = scored0[scored0.policy.astype(str) == str(args.base_policy)].set_index('row_id')
        base6 = scored0[scored0.policy.astype(str) == str(args.target_policy)].set_index('row_id')
        oat_rows = []
        for ax in AXES:
            h = s_oat[s_oat.policy == f'OAT_{ax}'].set_index('row_id')
            for o in ORACLES:
                m = h[f'delta_R_score_{o}'] - base4[f'delta_R_score_{o}']
                oat_rows.append(pd.DataFrame({'row_id': m.index, 'axis': ax, 'oracle': o, 'oat_marginal': m.to_numpy()}))
        OAT = pd.concat(oat_rows, ignore_index=True)
        OAT.to_parquet(arm_out / 'oat_marginals.parquet', index=False)
        gain = {o: base6[f'delta_R_score_{o}'] - base4[f'delta_R_score_{o}'] for o in ORACLES}
        oat_sum = OAT.pivot_table(index=['row_id', 'oracle'], columns='axis', values='oat_marginal').sum(axis=1).rename('oat_sum').reset_index()
        for o in ORACLES:
            oat_sum.loc[oat_sum.oracle == o, 'gain'] = oat_sum.loc[oat_sum.oracle == o, 'row_id'].map(gain[o])
        oat_sum['interaction_residual'] = oat_sum['gain'] - oat_sum['oat_sum']
        oat_sum.to_csv(arm_out / 'oat_additivity_audit.csv', index=False)
        shap_path = None
        if label in shapley_arms and (not args.oat_only):
            P = args.shapley_permutations
            phi = np.zeros((len(c4), len(AXES), len(ORACLES)))
            for p in range(P):
                perm = rng.permutation(len(AXES)).tolist()
                tab = _shapley_tables(c4, c6, perm, p)
                s_p, _, _, _ = _score_action_table(project_root=root, action_table=tab, out=arm_out / '_shap', space=space, output_prefix=f'shap_p{p}', write_outputs=False)
                prev = {o: base4[f'delta_R_score_{o}'].to_numpy() for o in ORACLES}
                for step, ai in enumerate(perm):
                    h = s_p[s_p.policy == f'SHAP_p{p}_s{step}_a{ai}'].sort_values('row_id')
                    for oi, o in enumerate(ORACLES):
                        cur = h[f'delta_R_score_{o}'].to_numpy()
                        phi[:, ai, oi] += cur - prev[o]
                        prev[o] = cur
            phi /= P
            recs = []
            rid = c4.row_id.to_numpy()
            for ai, ax in enumerate(AXES):
                for oi, o in enumerate(ORACLES):
                    recs.append(pd.DataFrame({'row_id': rid, 'axis': ax, 'oracle': o, 'phi': phi[:, ai, oi]}))
            PHI = pd.concat(recs, ignore_index=True)
            shap_path = arm_out / 'shapley_phi.parquet'
            PHI.to_parquet(shap_path, index=False)
            eff = PHI.pivot_table(index=['row_id', 'oracle'], values='phi', aggfunc='sum').reset_index()
            eff['gain'] = [gain[o].get(r, np.nan) for r, o in zip(eff.row_id, eff.oracle)]
            eff['efficiency_gap'] = eff['phi'] - eff['gain']
            max_efficiency_gap = float(eff['efficiency_gap'].abs().max())
            if max_efficiency_gap > args.efficiency_tol:
                raise AssertionError(f'SHAPLEY EFFICIENCY FAIL max|sum(phi)-gain|={max_efficiency_gap:.3e} > tol={args.efficiency_tol:.3e}')
            eff.to_csv(arm_out / 'shapley_efficiency_audit.csv', index=False)
            ga = gain['alpha']
            loss_ids = set(ga[ga < 0].index)
            win_ids = set(ga[ga > 0].index)
            summ = []
            pa = PHI[PHI.oracle == 'alpha']
            loss_pvals: list[float] = []
            win_pvals: list[float] = []
            for ax in AXES:
                s = pa[pa.axis == ax]
                sl = s[s.row_id.isin(loss_ids)].phi
                sw = s[s.row_id.isin(win_ids)].phi
                p_loss = wilcoxon_paired_p(sl)
                p_win = wilcoxon_paired_p(sw)
                loss_pvals.append(1.0 if not np.isfinite(p_loss) else float(p_loss))
                win_pvals.append(1.0 if not np.isfinite(p_win) else float(p_win))
                summ.append({'axis': ax, 'phi_mean_all': s.phi.mean(), 'n_loss': int(len(sl)), 'phi_mean_loss': sl.mean(), 'n_win': int(len(sw)), 'phi_mean_win': sw.mean(), 'phi_mean_loss_minus_win': sl.mean() - sw.mean(), 'p_loss_vs_zero_raw': p_loss, 'p_win_vs_zero_raw': p_win, 'evidence_tier': 'EVALUATOR_ONLY_POSTHOC_OUTCOME_DEFINED'})
            loss_holm = holm_adjust(loss_pvals)
            win_holm = holm_adjust(win_pvals)
            for rec, p_loss_holm, p_win_holm in zip(summ, loss_holm, win_holm):
                rec['p_loss_vs_zero_holm'] = p_loss_holm
                rec['sig_loss_vs_zero_holm'] = p_to_stars(p_loss_holm)
                rec['p_win_vs_zero_holm'] = p_win_holm
                rec['sig_win_vs_zero_holm'] = p_to_stars(p_win_holm)
            pd.DataFrame(summ).to_csv(arm_out / 'shapley_axis_summary_alpha.csv', index=False)
        manifest['arms'][label] = {'archive': str(inner), 'fidelity': fid, 'base_policy': args.base_policy, 'target_policy': args.target_policy, 'simulator_identity': sim_id, 'candidate_library': prov.get('candidate_library_path'), 'n_firms': int(len(c4)), 'shapley_permutations': args.shapley_permutations if label in shapley_arms and (not args.oat_only) else 0, 'shapley_efficiency_tolerance': float(args.efficiency_tol), 'elapsed_sec': round(time.time() - t0, 1), 'outputs': {'oat': str(arm_out / 'oat_marginals.parquet'), 'shapley': str(shap_path) if shap_path else None}}
        print(f"[{label}] fidelity OK {fid} elapsed={manifest['arms'][label]['elapsed_sec']}s")
    (out / 'intervention_manifest.json').write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding='utf-8')
    print('DONE ->', out)
if __name__ == '__main__':
    main()
