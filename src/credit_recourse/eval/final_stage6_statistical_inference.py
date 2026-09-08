from __future__ import annotations
import argparse, math, zlib
from pathlib import Path
import numpy as np
import pandas as pd
from credit_recourse.rl.common.io import final_root

# Bootstrap seed derivation scheme.
# v1 (legacy, pre-patch): abs(hash((backend,ref,pol))) % 2**32
#   — salted by Python process PYTHONHASHSEED; not reproducible across runs.
# v2 (current): zlib.crc32(f"{backend}|{ref}|{pol}".encode()) & 0xFFFFFFFF
#   — deterministic, no external state, same output across Python versions.
_BOOTSTRAP_SEED_SCHEME = "crc32_v2"


def _bootstrap_seed(backend: str, ref: str, pol: str) -> int:
    return zlib.crc32(f"{backend}|{ref}|{pol}".encode()) & 0xFFFFFFFF


def _holm_adjust(pvals):
    m=len(pvals); order=np.argsort([1.0 if not np.isfinite(p) else p for p in pvals]); out=[np.nan]*m; running=0.0
    for rank, idx in enumerate(order):
        p=1.0 if not np.isfinite(pvals[idx]) else float(pvals[idx])
        running=max(running, (m-rank)*p)
        out[idx]=min(1.0, running)
    return out


def _wilcoxon(diff):
    d=pd.to_numeric(diff, errors='coerce').dropna()
    d=d[np.abs(d)>1e-12]
    if len(d)==0: return 0.0,1.0
    try:
        from scipy.stats import wilcoxon
        r=wilcoxon(d.to_numpy(), zero_method='wilcox', alternative='two-sided')
        return float(r.statistic), float(r.pvalue)
    except Exception:
        ranks=pd.Series(np.abs(d)).rank(method='average').to_numpy(); vals=d.to_numpy()
        wpos=float(ranks[vals>0].sum()); n=len(ranks)
        mean=n*(n+1)/4.0; var=n*(n+1)*(2*n+1)/24.0
        z=(wpos-mean)/np.sqrt(max(var,1e-12))
        p=float(2.0*(1.0-0.5*(1.0+math.erf(abs(z)/np.sqrt(2.0)))))
        return wpos,p


def _cluster_bootstrap(diff, clusters, seed=42, n_boot=1000):
    d=pd.to_numeric(diff, errors='coerce'); c=clusters.astype(str)
    ok=d.notna() & c.notna(); d=d[ok]; c=c[ok]
    if len(d)==0: return np.nan,np.nan,np.nan
    g=d.groupby(c).mean().to_numpy(dtype=float)
    if len(g)<=1:
        se=float(d.std(ddof=1)/np.sqrt(max(len(d),1))) if len(d)>1 else 0.0; m=float(d.mean())
        return se,m-1.96*se,m+1.96*se
    rng=np.random.default_rng(seed); boots=[float(np.mean(rng.choice(g, size=len(g), replace=True))) for _ in range(n_boot)]
    se=float(np.std(boots, ddof=1)); lo,hi=np.quantile(boots,[0.025,0.975])
    return se,float(lo),float(hi)


def _build_row_cluster_map(df: pd.DataFrame) -> dict:
    """Return row_id -> cluster id for paired Stage6 inference.

    Stage6 paired inference is row-paired by the evaluation `row_id`.
    If `firm_id` exists, bootstrap clusters by firm; otherwise cluster by
    `row_id` itself. The fallback must not try to select `row_id` after
    setting it as the index, because that removes it from columns.
    """
    if 'row_id' not in df.columns:
        raise KeyError('Stage6 statistical inference requires row_id in multi_oracle_policy_eval.parquet')
    base = df.drop_duplicates('row_id').copy()
    if 'firm_id' in base.columns:
        return base.set_index('row_id')['firm_id'].to_dict()
    return {rid: rid for rid in base['row_id'].tolist()}


def run(project_root: Path) -> dict:
    final=final_root(project_root); out=final/'stage6_multi_oracle_eval'
    inp=out/'multi_oracle_policy_eval.parquet'
    if not inp.exists():
        raise FileNotFoundError(inp)
    df=pd.read_parquet(inp)
    required={'row_id','policy'}
    missing=sorted(required-set(df.columns))
    if missing:
        raise KeyError(f'Stage6 statistical inference missing required columns in {inp}: {missing}')
    cluster_map=_build_row_cluster_map(df)
    results={}
    for backend in ['alpha','beta','gamma']:
        col=f'R_score_{backend}'
        if col not in df.columns: continue
        piv=df.pivot_table(index='row_id', columns='policy', values=col, aggfunc='first')
        rows=[]; pvals=[]
        policies=list(piv.columns)
        for ref in policies:
            for pol in policies:
                if pol==ref: continue
                d=(piv[pol]-piv[ref]).dropna()
                if len(d)==0: continue
                clusters=pd.Series([cluster_map.get(i,i) for i in d.index], index=d.index)
                se,lo,hi=_cluster_bootstrap(d, clusters, seed=_bootstrap_seed(backend,ref,pol))
                w,p=_wilcoxon(d); pvals.append(p)
                rows.append({'policy_a':pol,'policy_b':ref,'comparison':'policy_a_minus_policy_b','n_pairs':int(len(d)),'mean_diff':float(d.mean()),'median_diff':float(d.median()),'cluster_se':se,'ci95_lo':lo,'ci95_hi':hi,'wilcoxon_W':w,'wilcoxon_p':p})
        adj=_holm_adjust(pvals)
        for r,a in zip(rows,adj): r['holm_adjusted_p']=a
        path=out/f'policy_paired_inference_{backend}.csv'
        pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig')
        results[backend]={'rows':len(rows),'output':str(path)}
    return {'status':'PASS','bootstrap_seed_scheme':_BOOTSTRAP_SEED_SCHEME,'outputs':results}


def main(argv=None):
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root', required=True)
    args=ap.parse_args(argv); res=run(Path(args.project_root).resolve()); print(res); return 0
if __name__=='__main__': raise SystemExit(main())
