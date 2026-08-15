#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1]); a=p.parse_args(); here=Path(__file__).resolve().parent
    for s in ['40_make_fig2_progressive_null_calibration.py','41_make_fig3_final_topographies.py','42_make_fig4_simulation_justification.py','43_make_fig5_stability_metric_boundary.py']:
        print(f'\n=== {s} ===',flush=True); subprocess.run([sys.executable,str(here/s),'--project',str(a.project)],check=True)
    print('\nDONE. Figures written to <project>/figures/manuscript as PDF, SVG, and 600-dpi PNG.')
if __name__=='__main__': main()
