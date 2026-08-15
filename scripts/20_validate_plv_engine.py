#!/usr/bin/env python3
"""
Validate the revised PLV engine WITHOUT using behavioral labels.

Checks:
  - synthetic identical phase -> PLV ~ 1
  - amplitude scaling leaves PLV unchanged
  - G01 pair12 all 9 task-band conditions:
      shape 40x19x19
      finite
      values within [0,1]
      reverse-pair symmetry M(A,B) = M(B,A)^T
  - exact analysis sample lengths:
      decision 1200 samples (4 s)
      feedback 600 samples (2 s)

No CCC/Other contrasts are computed.
"""

from pathlib import Path
import csv
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt.config import load_config, condition_list
from bt.connectivity import (
    analytic_unit_phase,
    plv_trial_matrices_from_phase,
    unit_phase_trials,
)
from bt.io import load_cleaned_subject


def main():
    cfg = load_config()
    outdir = Path(cfg["results_root"]) / "observed"
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "validation_G01_plv_engine.csv"
    out_txt = outdir / "validation_G01_plv_engine_summary.txt"

    # ---------- synthetic invariance checks ----------
    fs = 300.0
    t = np.arange(1500) / fs
    x = np.sin(2*np.pi*6*t)[None, :]
    z1 = analytic_unit_phase(
        x, fs, (4,7), 300, 0, 4,
        order=4, pad_seconds=2.0
    )
    z2 = analytic_unit_phase(
        10.0*x, fs, (4,7), 300, 0, 4,
        order=4, pad_seconds=2.0
    )
    synthetic_plv = float(
        plv_trial_matrices_from_phase(
            z1[None,:,:], z2[None,:,:]
        )[0,0,0]
    )

    synthetic_ok = abs(synthetic_plv - 1.0) < 1e-5

    # ---------- G01 no-label technical checks ----------
    s1 = load_cleaned_subject(cfg["cleaned_dir"], 1, 1)
    s2 = load_cleaned_subject(cfg["cleaned_dir"], 1, 2)

    rows = []
    all_ok = synthetic_ok

    for task, band_name in condition_list(cfg):
        band = tuple(cfg["bands_hz"][band_name])
        window = tuple(cfg["analysis_windows_seconds"][task])
        anchor = int(cfg["task_anchor_sample_zero_based"])
        order = int(cfg["connectivity"]["butterworth_order"])
        pad = float(cfg["connectivity"]["reflection_pad_seconds"])

        zA = unit_phase_trials(
            s1[task], s1["fs"], band, anchor, window,
            order=order, pad_seconds=pad
        )
        zB = unit_phase_trials(
            s2[task], s2["fs"], band, anchor, window,
            order=order, pad_seconds=pad
        )

        M = plv_trial_matrices_from_phase(zA, zB)

        # Reverse symmetry check on all trials.
        Mr = plv_trial_matrices_from_phase(zB, zA)
        symmetry_err = float(
            np.max(np.abs(M - np.transpose(Mr, (0,2,1))))
        )

        expected_n = int(round((window[1]-window[0]) * s1["fs"]))
        phase_n = int(zA.shape[2])

        ok = (
            M.shape == (40,19,19)
            and np.isfinite(M).all()
            and float(M.min()) >= -1e-7
            and float(M.max()) <= 1.0 + 1e-7
            and symmetry_err < 1e-5
            and phase_n == expected_n
        )
        all_ok = all_ok and ok

        rows.append({
            "task": task,
            "band": band_name,
            "phase_shape": "x".join(map(str, zA.shape)),
            "plv_shape": "x".join(map(str, M.shape)),
            "plv_min": float(M.min()),
            "plv_median": float(np.median(M)),
            "plv_max": float(M.max()),
            "reverse_symmetry_max_abs_error": symmetry_err,
            "expected_analysis_samples": expected_n,
            "actual_analysis_samples": phase_n,
            "validation_pass": ok,
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "Brain Topography Project — Stage 3A PLV Engine Validation",
        "="*66,
        f"Synthetic amplitude-invariance PLV: {synthetic_plv:.8f}",
        f"Synthetic check: {'PASS' if synthetic_ok else 'FAIL'}",
        "",
        "G01 PAIR12 TECHNICAL CHECKS (NO BEHAVIORAL LABELS USED)",
    ]
    for r in rows:
        lines.append(
            f"  {r['task']} {r['band']}: "
            f"phase={r['phase_shape']}, PLV={r['plv_shape']}, "
            f"range=[{r['plv_min']:.6f},{r['plv_max']:.6f}], "
            f"median={r['plv_median']:.6f}, "
            f"symerr={r['reverse_symmetry_max_abs_error']:.3e}, "
            f"samples={r['actual_analysis_samples']}/"
            f"{r['expected_analysis_samples']}, "
            f"pass={r['validation_pass']}"
        )

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if all_ok else 'CHECK REQUIRED'}",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
