#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve(); PKG=HERE.parents[1]; sys.path.insert(0,str(PKG/'lib'))
from bt_figstyle import setup_style, panel_label, clean_axes, save_all, COLORS, CANDIDATE_COLORS
EXPECTED=[3801,4994,8156]

def args():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument('--outdir',type=Path,default=None); return p.parse_args()

def main():
    a=args(); root=a.project.expanduser().resolve(); out=a.outdir or root/'figures'/'manuscript'
    m=pd.read_csv(root/'results'/'freeze'/'stage6C_master_evidence.csv').set_index('unit_index').loc[EXPECTED]
    loto=pd.read_csv(root/'results'/'stability'/'stage6B_final3_loto.csv')
    setup_style(); fig=plt.figure(figsize=(7.15,5.65)); gs=fig.add_gridspec(2,1,height_ratios=[1.12,1.0],hspace=.55)
    # A: stability
    ax=fig.add_subplot(gs[0]); panel_label(ax,'A',x=-.08,y=1.04); y=np.arange(3)
    for i,u in enumerate(EXPECTED):
        r=m.loc[u]; c=CANDIDATE_COLORS[u]; q=loto[loto.unit_index.astype(int)==u]
        yy=np.full(len(q),i)+np.linspace(-.13,.13,len(q))
        ax.scatter(q.LOTO_effect,yy,s=18,color=c,alpha=.58,edgecolor='none',zorder=3)
        ax.plot([r.bootstrap_95pct_low_descriptive,r.bootstrap_95pct_high_descriptive],[i,i],color=COLORS['black'],lw=3.2,solid_capstyle='round',zorder=2)
        ax.plot(r.PLV_effect,i,'D',ms=6.3,color=c,markeredgecolor='white',markeredgewidth=.5,zorder=4)
        ax.text(float(r.bootstrap_95pct_high_descriptive)+.004,i,
                f"LOTO {int(r.LOTO_same_direction_count)}/11 · bootstrap direction {float(r.bootstrap_same_direction_fraction):.0%}",
                va='center',fontsize=7.0,color=COLORS['gray'])
    labs=[f"{u}  {m.loc[u].task} {m.loc[u].band} · {m.loc[u].ch1}–{m.loc[u].ch2}" for u in EXPECTED]
    ax.set_yticks(y,labs); ax.invert_yaxis(); ax.axvline(0,color='#AAB1BA',lw=.8); ax.set_xlabel('CCC − Other ΔPLV'); clean_axes(ax,'x'); ax.set_xlim(-.005,.145)
    ax.set_title('Triad stability of the frozen PLV family',loc='left',pad=6)

    # B: iCOH boundary
    ax=fig.add_subplot(gs[1]); panel_label(ax,'B',x=-.08,y=1.04)
    pcols=['p_iCOH_label_maxT','p_iCOH_time_maxT','p_iCOH_partner_maxT','p_iCOH_within_triad_dyad_maxT']; labels=['Behavior-label','Temporal shift','Cross-partner','Within-triad dyad']; markers=['o','s','^','D']; offsets=np.array([-.18,-.06,.06,.18])
    for i,u in enumerate(EXPECTED):
        r=m.loc[u]; p=np.array([float(r[x]) for x in pcols]); x=-np.log10(p)
        for j,val in enumerate(x):
            ax.scatter(val,i+offsets[j],s=34,marker=markers[j],facecolor='white',edgecolor=CANDIDATE_COLORS[u],linewidth=1.2,zorder=3)
    thresh=-np.log10(.05); ax.axvline(thresh,color=COLORS['red'],ls='--',lw=1); ax.text(thresh+.015,-.43,'α=.05',fontsize=7,color=COLORS['red'],va='top')
    blabs=[f"{u}  ΔiCOH={float(m.loc[u].iCOH_effect):.3f} · same direction" for u in EXPECTED]
    ax.set_yticks(y,blabs); ax.invert_yaxis(); ax.set_xlabel('−log10(adjusted iCOH p)'); clean_axes(ax,'x'); ax.set_xlim(-.03,1.48); ax.set_title('Non-zero-lag metric boundary: directionally concordant, not confirmatory',loc='left',pad=6)
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker=mkr,color='none',markeredgecolor=COLORS['black'],markerfacecolor='white',label=lab) for mkr,lab in zip(markers,labels)]
    ax.legend(handles=handles,frameon=False,loc='upper right',ncol=2,columnspacing=.9,handletextpad=.35)
    save_all(fig,Path(out)/'Fig5_stability_metric_boundary'); plt.close(fig)
if __name__=='__main__': main()
