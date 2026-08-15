from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

# Publication defaults: full-width figures are ~180 mm / 7.1 in.
COLORS = {
    'navy': '#243447',
    'teal': '#2A7F9E',
    'orange': '#C76D2A',
    'green': '#3D7A57',
    'red': '#A64B4B',
    'gray': '#6B7280',
    'light': '#E9EDF2',
    'lighter': '#F5F7F9',
    'black': '#111827',
    'white': '#FFFFFF',
}
CANDIDATE_COLORS = {3801:'#2A7F9E', 4994:'#C76D2A', 8156:'#6D5AA7'}


def setup_style():
    mpl.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 8.5,
        'axes.titlesize': 9.5,
        'axes.labelsize': 8.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5,
        'figure.titlesize': 10.5,
        'axes.linewidth': 0.8,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
    })


def panel_label(ax, label, x=-0.08, y=1.04):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5,
            fontweight='bold', ha='left', va='bottom', color=COLORS['black'])


def clean_axes(ax, grid_axis=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color='#D9DEE5', linewidth=0.6, alpha=0.8)
        ax.set_axisbelow(True)


def save_all(fig, outbase: Path, dpi=600):
    outbase = Path(outbase)
    outbase.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outbase.with_suffix('.pdf'))
    fig.savefig(outbase.with_suffix('.svg'))
    fig.savefig(outbase.with_suffix('.png'), dpi=dpi)
