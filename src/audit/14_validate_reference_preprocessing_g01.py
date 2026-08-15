#!/usr/bin/env python3
"""
Stage 2B1 — Validate the PUBLISHED reference preprocessing on G01 only.

This script deliberately imports the public Data in Brief preprocessing code:
  PD_EEG_hyperscan_processing/pd_eeg_analysis/preprocess_bids.py
  PD_EEG_hyperscan_processing/pd_eeg_analysis/preprocessing_core.py

It does not use the custom BT preprocessing.py.

Why
---
The downloaded EEGLAB files have an unusual numerical scale. The published
pipeline reads them with MNE's EEGLAB reader and then applies the exact
participant-local preprocessing used for the data article:
  average reference -> 1-100 Hz FIR -> extended Infomax ICA -> ICLabel.

We first validate that exact published pipeline on G01 S01-S03 before adapting
the BT codebase to consume its cleaned output.

Expected local locations
------------------------
Dataset:
  data/raw/openneuro_ds007822

Reference repository:
  data/reference/PD_EEG_hyperscan_processing

Outputs
-------
results/preprocessing/reference_G01_validation.csv
results/preprocessing/reference_G01_validation_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return p.parse_args()


def rms(x):
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    bids_dir = project / "data" / "raw" / "openneuro_ds007822"
    repo = project / "data" / "reference" / "PD_EEG_hyperscan_processing"
    code_dir = repo / "pd_eeg_analysis"
    outdir = project / "results" / "preprocessing"
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / "reference_G01_validation.csv"
    out_txt = outdir / "reference_G01_validation_summary.txt"

    if not bids_dir.is_dir():
        print(f"FAIL: BIDS dataset not found: {bids_dir}", file=sys.stderr)
        return 2
    if not code_dir.is_dir():
        print(
            "FAIL: published reference repository not found.\n"
            f"Expected: {repo}\n\n"
            "Clone it first with:\n"
            "  git clone https://github.com/heegyukim4043/"
            "PD_EEG_hyperscan_processing.git "
            f"{repo}\n",
            file=sys.stderr,
        )
        return 3

    sys.path.insert(0, str(code_dir))

    try:
        preprocess_bids = importlib.import_module("preprocess_bids")
        preprocessing_core = importlib.import_module("preprocessing_core")
    except Exception as e:
        print(f"FAIL importing published code: {type(e).__name__}: {e}", file=sys.stderr)
        return 4

    grp = preprocess_bids.load_group_from_bids(1, bids_dir)

    rows = []
    failures = []

    for task, key in (("decision", "decision_X"), ("feedback", "feedback_X")):
        eeg_raw = np.asarray(grp[key], dtype=float)

        expected_shape = (57, 1500, 40) if task == "decision" else (57, 900, 40)
        if eeg_raw.shape != expected_shape:
            failures.append(f"{task}: group array shape {eeg_raw.shape}, expected {expected_shape}")
            continue

        for si in range(3):
            sl = slice(si * 19, (si + 1) * 19)
            subject_numeric = eeg_raw[sl]

            try:
                cleaned, n_rej = preprocessing_core.preprocess_task(subject_numeric.copy())
            except Exception as e:
                failures.append(
                    f"{task} S{si+1}: {type(e).__name__}: {e}"
                )
                continue

            rows.append(
                {
                    "group": "G01",
                    "participant": f"S{si+1}",
                    "task": task,
                    "input_shape": "x".join(map(str, subject_numeric.shape)),
                    "input_numeric_rms_after_mne_reader": rms(subject_numeric),
                    "input_abs_p99_after_mne_reader": float(
                        np.percentile(np.abs(subject_numeric), 99)
                    ),
                    "cleaned_shape": "x".join(map(str, cleaned.shape)),
                    "cleaned_numeric_rms_microvolt_scale": rms(cleaned),
                    "cleaned_abs_p99_microvolt_scale": float(
                        np.percentile(np.abs(cleaned), 99)
                    ),
                    "n_rejected_ica_components": int(n_rej),
                    "cleaned_all_finite": bool(np.isfinite(cleaned).all()),
                }
            )

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # The published preprocess_task returns cleaned * 1e6, documented as
    # microvolt-scale numeric output. We do not impose a narrow physiological
    # amplitude cutoff; we only require finite, non-degenerate output and correct shape.
    shape_ok = all(
        (
            r["cleaned_shape"] == "19x1500x40"
            if r["task"] == "decision"
            else r["cleaned_shape"] == "19x900x40"
        )
        for r in rows
    )
    finite_ok = all(r["cleaned_all_finite"] for r in rows)
    nondegenerate = all(
        np.isfinite(r["cleaned_numeric_rms_microvolt_scale"])
        and r["cleaned_numeric_rms_microvolt_scale"] > 0
        for r in rows
    )

    pass_ok = (
        len(rows) == 6
        and not failures
        and shape_ok
        and finite_ok
        and nondegenerate
    )

    lines = [
        "Brain Topography Project — Stage 2B1 Published Reference Preprocessing",
        "=" * 74,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Python: {sys.version.split()[0]}",
        f"BIDS dataset: {bids_dir}",
        f"Reference repo: {repo}",
        "",
        "REFERENCE CONSTANTS",
        f"  CH_NAMES: {preprocessing_core.CH_NAMES}",
        f"  ICA_N_COMPONENTS: {preprocessing_core.ICA_N_COMPONENTS}",
        f"  ICA_RANDOM_STATE: {preprocessing_core.ICA_RANDOM_STATE}",
        f"  ICA_MAX_ITER: {preprocessing_core.ICA_MAX_ITER}",
        f"  ICA_REJECT_LABELS: {sorted(preprocessing_core.ICA_REJECT_LABELS)}",
        f"  ICA_REJECT_PROB: {preprocessing_core.ICA_REJECT_PROB}",
        "",
        "G01 RESULTS",
    ]

    for r in rows:
        lines.append(
            f"  {r['participant']} {r['task']}: "
            f"input RMS after MNE EEGLAB reader={r['input_numeric_rms_after_mne_reader']:.6g}; "
            f"cleaned RMS={r['cleaned_numeric_rms_microvolt_scale']:.6g}; "
            f"cleaned |x|p99={r['cleaned_abs_p99_microvolt_scale']:.6g}; "
            f"ICA rejected={r['n_rejected_ica_components']}; "
            f"finite={r['cleaned_all_finite']}"
        )

    if failures:
        lines.extend(["", "FAILURES"])
        for x in failures:
            lines.append(f"  {x}")

    lines.extend(
        [
            "",
            "RESULT",
            f"  {'PASS — published preprocessing reproduced on G01' if pass_ok else 'CHECK REQUIRED'}",
            "",
            "NEXT",
            "  If PASS, adapt the BT pipeline to use the published preprocessing path",
            "  rather than the custom Stage-1 preprocessing implementation.",
            "",
            f"CSV: {out_csv}",
            f"Summary: {out_txt}",
        ]
    )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if pass_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
