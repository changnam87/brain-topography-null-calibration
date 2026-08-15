#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

HERE=Path(__file__).resolve(); PKG=HERE.parents[1]; sys.path.insert(0,str(PKG/'lib'))
from bt_figstyle import setup_style, panel_label, clean_axes, save_all, COLORS

EXPECTED=[3801,4166,4877,4900,4994,5317,8156]
FINAL={3801,4994,8156}

def args():
    p=argparse.ArgumentParser()
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--outdir',type=Path,default=None)
    return p.parse_args()

def main():
    a=args(); root=a.project.expanduser().resolve(); out=(a.outdir or root/'figures'/'manuscript')
    n=root/'results'/'nulls'
    units=pd.read_csv(n/'label_null_plv_units.csv')
    s3=pd.read_csv(n/'candidate_nulls_plv.csv')
    s4=pd.read_csv(n/'stage4D2_empirical_within_triad_dyad_null.csv')
    cand=units[units['label_candidate'].astype(str).str.lower().isin(['true','1'])].copy()
    ids=cand['unit_index'].astype(int).tolist()
    if ids!=EXPECTED: raise RuntimeError(f'Expected fixed 7 {EXPECTED}, got {ids}')
    m=cand[['unit_index','task','band','dyad','ch1','ch2','observed_effect','p_label_maxT']].merge(
        s3[['unit_index','p_time_maxT','p_partner_maxT']],on='unit_index').merge(
        s4[['unit_index','p_within_triad_dyad_maxT','all_four_null_layers_pass']],on='unit_index')
    m['unit_index']=m['unit_index'].astype(int)
    if set(m.loc[m['all_four_null_layers_pass'].astype(str).str.lower().isin(['true','1']),'unit_index'])!=FINAL:
        raise RuntimeError('Final family does not match frozen [3801,4994,8156].')

    setup_style(); fig=plt.figure(figsize=(7.15,5.25)); gs=fig.add_gridspec(2,1,height_ratios=[0.9,2.2],hspace=0.38)
    ax=fig.add_subplot(gs[0]); ax.axis('off'); panel_label(ax,'A',x=-0.02,y=1.00)
    stages=['All eligible\nPLV units','Label maxT\nFWER','Temporal-shift\nmaxT','Cross-partner\nmaxT','Within-triad\ndyad maxT']
    counts=[len(units),len(cand),int((s3.p_time_maxT<.05).sum()),int((s3.p_partner_maxT<.05).sum()),int((s4.p_within_triad_dyad_maxT<.05).sum())]
    xs=np.linspace(.08,.92,5)
    for i,(x,lab,c) in enumerate(zip(xs,stages,counts)):
        fc=COLORS['navy'] if i<4 else COLORS['green']
        ax.add_patch(plt.Rectangle((x-.075,.28),.15,.38,transform=ax.transAxes,facecolor='white',edgecolor=fc,linewidth=1.6))
        ax.text(x,.51,f'{c:,}',transform=ax.transAxes,ha='center',va='center',fontsize=15,fontweight='bold',color=fc)
        ax.text(x,.20,lab,transform=ax.transAxes,ha='center',va='top',fontsize=7.8,color=COLORS['black'])
        if i<4:
            ax.annotate('',xy=(xs[i+1]-.085,.47),xytext=(x+.085,.47),xycoords=ax.transAxes,
                        arrowprops=dict(arrowstyle='-|>',lw=1.2,color=COLORS['gray']))
    ax.text(.5,.86,'Progressive calibration narrows the empirical family without redefining earlier stages',transform=ax.transAxes,ha='center',fontsize=8.2,color=COLORS['gray'])

    ax=fig.add_subplot(gs[1]); panel_label(ax,'B',x=-0.10,y=1.03)
    cols=['p_label_maxT','p_time_maxT','p_partner_maxT','p_within_triad_dyad_maxT']
    pvals=m.set_index('unit_index').loc[EXPECTED,cols].to_numpy(float)
    passed=(pvals<.05).astype(int)
    cmap=ListedColormap(['#ECEFF3','#D7EBDD']); norm=BoundaryNorm([-0.5,.5,1.5],cmap.N)
    ax.imshow(passed,aspect='auto',cmap=cmap,norm=norm)
    labels=[]
    for _,r in m.set_index('unit_index').loc[EXPECTED].iterrows():
        star='● ' if int(r.name) in FINAL else '○ '
        labels.append(f"{star}{int(r.name)}  {r['task']} {r['band']}  {r['dyad']}  {r['ch1']}–{r['ch2']}")
    ax.set_yticks(range(7),labels)
    ax.set_xticks(range(4),['Behavior-label\nmaxT','Temporal-shift\nmaxT','Cross-partner\nmaxT','Within-triad dyad\nmaxT'])
    for i in range(7):
        for j in range(4):
            p=pvals[i,j]
            txt='<.001' if p<.001 else f'{p:.3f}'
            ax.text(j,i,txt,ha='center',va='center',fontsize=7.2,fontweight='bold' if p<.05 else 'normal',color=COLORS['black'])
    for i,u in enumerate(EXPECTED):
        if u in FINAL: ax.add_patch(plt.Rectangle((-0.5,i-.5),4,1,fill=False,edgecolor=COLORS['green'],linewidth=1.1))
    ax.set_title('Adjusted empirical p-values for the fixed seven-candidate family',pad=8)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    save_all(fig,Path(out)/'Fig2_progressive_null_calibration'); plt.close(fig)

if __name__=='__main__': main()
