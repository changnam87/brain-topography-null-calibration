from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from .config import save_config_snapshot

def run_stability(cfg: dict, label_npz: str | Path):
    z=np.load(label_npz,allow_pickle=True)
    ge=z["group_effects"].astype(np.float64)  # G x U
    obs=z["observed"].astype(np.float64)
    G,U=ge.shape
    rng=np.random.default_rng(int(cfg["random_seed"])+7000)

    # Leave-one-triad-out effects
    loto=np.empty((G,U),dtype=np.float64)
    for g in range(G):
        keep=np.arange(G)!=g
        loto[g]=ge[keep].mean(axis=0)

    sign_consistency=np.mean(np.sign(loto)==np.sign(obs)[None,:],axis=0)
    rank_corr=[]
    base_rank=np.argsort(np.argsort(np.abs(obs)))
    for g in range(G):
        r=np.argsort(np.argsort(np.abs(loto[g])))
        rank_corr.append(np.corrcoef(base_rank,r)[0,1])

    # Triad bootstrap CIs for effect, in chunks to control memory.
    B=2000
    low=np.empty(U); high=np.empty(U)
    chunk=1000
    for st in range(0,U,chunk):
        en=min(U,st+chunk)
        boot=np.empty((B,en-st),dtype=np.float32)
        for b in range(B):
            idx=rng.integers(0,G,size=G)
            boot[b]=ge[idx,st:en].mean(axis=0)
        low[st:en]=np.percentile(boot,2.5,axis=0)
        high[st:en]=np.percentile(boot,97.5,axis=0)

    outdir=Path(cfg["results_root"])/"stability"; outdir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        outdir/"stability_plv.npz",
        loto_effects=loto.astype(np.float32),
        sign_consistency=sign_consistency.astype(np.float32),
        bootstrap_ci_low=low.astype(np.float32),
        bootstrap_ci_high=high.astype(np.float32),
        loto_rank_correlation=np.asarray(rank_corr,dtype=np.float32),
    )
    summary={
        "mean_loto_rank_correlation":float(np.mean(rank_corr)),
        "min_loto_rank_correlation":float(np.min(rank_corr)),
        "median_sign_consistency":float(np.median(sign_consistency)),
        "fraction_sign_consistency_1":float(np.mean(sign_consistency==1.0)),
    }
    with (outdir/"stability_summary.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(summary.keys())); w.writeheader(); w.writerow(summary)
    save_config_snapshot(cfg,outdir)
    return outdir/"stability_plv.npz"
