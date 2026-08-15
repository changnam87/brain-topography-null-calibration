#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Arc

HERE=Path(__file__).resolve(); PKG=HERE.parents[1]; sys.path.insert(0,str(PKG/'lib'))
from bt_figstyle import setup_style, panel_label, save_all, COLORS, CANDIDATE_COLORS

POS={'Fp1':(-.45,.88),'Fp2':(.45,.88),'F7':(-.88,.48),'F3':(-.45,.48),'Fz':(0,.53),'F4':(.45,.48),'F8':(.88,.48),
'T3':(-.98,0),'C3':(-.48,0),'Cz':(0,0),'C4':(.48,0),'T4':(.98,0),'T5':(-.82,-.46),'P3':(-.45,-.46),'Pz':(0,-.52),'P4':(.45,-.46),'T6':(.82,-.46),'O1':(-.38,-.88),'O2':(.38,-.88)}
EXPECTED=[3801,4994,8156]

def draw_head(ax,cx,cy,s,label,highlight=None,color='#2A7F9E'):
    ax.add_patch(Circle((cx,cy),s,fill=False,lw=1.0,ec='#4B5563'))
    ax.add_patch(Polygon([[cx-.10*s,cy+s*.98],[cx,cy+s*1.10],[cx+.10*s,cy+s*.98]],closed=False,fill=False,lw=.8,ec='#4B5563'))
    ax.add_patch(Arc((cx-s*1.02,cy),s*.20,s*.34,theta1=70,theta2=290,lw=.7,ec='#4B5563'))
    ax.add_patch(Arc((cx+s*1.02,cy),s*.20,s*.34,theta1=-110,theta2=110,lw=.7,ec='#4B5563'))
    for ch,(x,y) in POS.items():
        xx,yy=cx+x*s*.84,cy+y*s*.84
        ishi=ch==highlight
        ax.plot(xx,yy,'o',ms=4.0 if ishi else 1.8,mfc=color if ishi else '#9CA3AF',mec='white' if ishi else '#9CA3AF',mew=.5,zorder=4)
    ax.text(cx,cy-s*1.30,label,ha='center',va='top',fontsize=7.3,color='#4B5563')

def args():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]); p.add_argument('--outdir',type=Path,default=None); return p.parse_args()

def main():
    a=args(); root=a.project.expanduser().resolve(); out=a.outdir or root/'figures'/'manuscript'
    master=pd.read_csv(root/'results'/'freeze'/'stage6C_master_evidence.csv').set_index('unit_index').loc[EXPECTED]
    tri=pd.read_csv(root/'results'/'stability'/'stage6B_final3_triad_effects.csv')
    if set(tri.unit_index.astype(int))!=set(EXPECTED): raise RuntimeError('Triad file must contain only frozen final 3.')
    setup_style(); fig=plt.figure(figsize=(7.15,7.0)); gs=fig.add_gridspec(3,2,width_ratios=[1.15,1.75],hspace=.46,wspace=.28)
    for row,u in enumerate(EXPECTED):
        r=master.loc[u]; color=CANDIDATE_COLORS[u]
        axh=fig.add_subplot(gs[row,0]); axh.set_xlim(-2.15,2.15); axh.set_ylim(-1.35,1.45); axh.set_aspect('equal'); axh.axis('off'); panel_label(axh,chr(ord('A')+row),x=-.04,y=1.01)
        draw_head(axh,-.95,.10,.78,'Participant 1',r.ch1,color); draw_head(axh,.95,.10,.78,'Participant 3',r.ch2,color)
        p1=(-.95+POS[r.ch1][0]*.78*.84,.10+POS[r.ch1][1]*.78*.84); p2=(.95+POS[r.ch2][0]*.78*.84,.10+POS[r.ch2][1]*.78*.84)
        axh.annotate('',xy=p2,xytext=p1,arrowprops=dict(arrowstyle='-',lw=2.3,color=color,alpha=.9))
        axh.text(0,1.30,f"{r.task.capitalize()} · {r.band} · pair13",ha='center',va='top',fontsize=9,fontweight='bold',color=COLORS['black'])
        axh.text(0,1.08,f"{r.ch1}–{r.ch2}",ha='center',va='top',fontsize=8.5,color=color,fontweight='bold')

        ax=fig.add_subplot(gs[row,1]); q=tri[tri.unit_index.astype(int)==u].copy(); q['triad_num']=q['triad'].str.extract(r'(\d+)').astype(int)
        y=np.arange(len(q)); vals=q['triad_effect'].astype(float).to_numpy(); signs=q['same_direction_as_full'].astype(str).str.lower().isin(['true','1']).to_numpy()
        ax.axvline(0,color='#AAB1BA',lw=.8)
        ax.scatter(vals[signs],y[signs],s=31,color=color,edgecolor='white',linewidth=.5,zorder=3,label='same direction')
        if (~signs).any(): ax.scatter(vals[~signs],y[~signs],s=35,facecolor='white',edgecolor=color,linewidth=1.2,zorder=3,label='opposite direction')
        full=float(r.PLV_effect); lo=float(r.bootstrap_95pct_low_descriptive); hi=float(r.bootstrap_95pct_high_descriptive)
        ax.axvline(full,color=color,lw=1.5,ls='--')
        ax.plot([lo,hi],[-1.0,-1.0],lw=3,color=COLORS['black'],solid_capstyle='round'); ax.plot(full,-1.0,'D',ms=5,color=color)
        ax.text(hi+.003,-1.0,'10k bootstrap 95% interval',va='center',fontsize=7.0,color=COLORS['gray'])
        ax.set_yticks(y,q['triad']); ax.set_ylim(len(q)-.4,-1.65); ax.set_xlabel('CCC − Other ΔPLV'); ax.grid(axis='x',color='#E1E5EA',lw=.6); ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        same=int(r.individual_triads_same_direction); ax.set_title(f"ΔPLV={full:.3f} · individual triads {same}/11 same direction · LOTO 11/11",loc='left',pad=6)
    save_all(fig,Path(out)/'Fig3_final_topographies_triad_effects'); plt.close(fig)
if __name__=='__main__': main()
