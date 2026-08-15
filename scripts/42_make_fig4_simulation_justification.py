#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve(); PKG=HERE.parents[1]; sys.path.insert(0,str(PKG/'lib'))
from bt_figstyle import setup_style, panel_label, clean_axes, save_all, COLORS

def args():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument('--outdir',type=Path,default=None); return p.parse_args()

def main():
    a=args(); root=a.project.expanduser().resolve(); out=a.outdir or root/'figures'/'manuscript'; path=root/'results'/'simulation'/'stage4C2_dyad_null_stress_summary.csv'; d=pd.read_csv(path)
    setup_style(); fig=plt.figure(figsize=(7.15,5.2)); gs=fig.add_gridspec(1,2,width_ratios=[1.05,1.35],wspace=.34)
    ax=fig.add_subplot(gs[0]); panel_label(ax,'A',x=-.14,y=1.03)
    z=d[d.n_true==0].copy(); order=['global_shared_event','group_shared_event']; z['scenario']=pd.Categorical(z.scenario,order,ordered=True); z=z.sort_values(['scenario','shared'])
    labs=[('Global shared\n'+f"s={r.shared:.2f}") if r.scenario=='global_shared_event' else ('Group shared\n'+f"s={r.shared:.2f}") for _,r in z.iterrows()]
    y=np.arange(len(z)); h=.34
    ax.barh(y-h/2,z.current_FWER,h,color='#BFC6CF',label='Before dyad null')
    ax.barh(y+h/2,z.augmented_FWER,h,color=COLORS['teal'],label='With dyad null')
    ax.axvline(.05,color=COLORS['red'],ls='--',lw=1,label='α=.05')
    ax.set_yticks(y,labs); ax.set_xlim(0,1.04); ax.set_xlabel('Family-wise false-positive rate'); ax.invert_yaxis(); clean_axes(ax,'x'); ax.legend(frameon=False,loc='lower right')
    ax.set_title('Confound stress test',loc='left')

    ax=fig.add_subplot(gs[1]); panel_label(ax,'B',x=-.13,y=1.03)
    t=d[d.n_true>0].copy(); t['type']=np.where(t.scenario=='sparse_true','True edges only','True edges + group-shared confound')
    markers={'True edges only':'o','True edges + group-shared confound':'s'}
    for typ,g in t.groupby('type',sort=False):
        for _,r in g.iterrows():
            ax.annotate('',xy=(r.augmented_sensitivity,r.augmented_precision),xytext=(r.current_sensitivity,r.current_precision),arrowprops=dict(arrowstyle='-|>',lw=1.0,color=COLORS['gray'],alpha=.8))
            ax.scatter(r.current_sensitivity,r.current_precision,s=24,facecolor='white',edgecolor=COLORS['gray'],marker=markers[typ],linewidth=1)
            ax.scatter(r.augmented_sensitivity,r.augmented_precision,s=30,color=COLORS['teal'],marker=markers[typ],edgecolor='white',linewidth=.4)
    ax.set_xlim(0,.5); ax.set_ylim(0,.36); ax.set_xlabel('Sensitivity'); ax.set_ylabel('Precision'); clean_axes(ax,'both')
    ax.set_title('Recovery trade-off after adding layer 4',loc='left')
    from matplotlib.lines import Line2D
    handles=[Line2D([0],[0],marker='o',color='none',markeredgecolor=COLORS['black'],markerfacecolor='white',label='True edges only'),Line2D([0],[0],marker='s',color='none',markeredgecolor=COLORS['black'],markerfacecolor='white',label='True edges + group-shared confound'),Line2D([0],[0],marker='o',color='none',markeredgecolor=COLORS['gray'],markerfacecolor='white',label='Before dyad null'),Line2D([0],[0],marker='o',color='none',markeredgecolor='white',markerfacecolor=COLORS['teal'],label='With dyad null')]
    ax.legend(handles=handles,frameon=False,loc='upper left')
    ax.text(.02,-.20,'Arrows show the change produced by the within-triad dyad null for each simulation cell.',transform=ax.transAxes,fontsize=7.1,color=COLORS['gray'])
    save_all(fig,Path(out)/'Fig4_simulation_justification'); plt.close(fig)
if __name__=='__main__': main()
