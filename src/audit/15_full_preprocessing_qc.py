#!/usr/bin/env python3
"""
Stage 2B2 — Full cleaned-EEG QC for the Brain Topography project.

Purpose
-------
Review all 33 participant-level cleaned NPZ files produced by the validated
published Data in Brief preprocessing wrapper BEFORE any connectivity analysis.

This script is diagnostic only. It does not delete subjects, channels, trials,
or samples.

For decision and feedback separately it computes:
  - overall RMS and |amplitude| percentiles
  - fraction of samples above 100/200/500 uV
  - per-channel RMS range and robust outlier count
  - per-trial RMS range and robust outlier count
  - subject-level robust z score of log(RMS) across all 33 participants

Primary review flag
-------------------
A subject/task is flagged if its subject-level log(RMS) robust z score exceeds
3.5 in absolute value. This is a relative QC flag, not an automatic exclusion.

Outputs
-------
results/preprocessing/stage2B2_subject_task_qc.csv
results/preprocessing/stage2B2_full_qc_summary.txt

Requirements
------------
Python 3, numpy
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
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


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6744897501960817 * (x - med) / mad


def rms_axis(x: np.ndarray, axis) -> np.ndarray:
    return np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=axis))


def summarize_task(x: np.ndarray) -> dict:
    # x: channels x time x trials, uV
    flat_abs = np.abs(x.astype(np.float64)).reshape(-1)
    overall_rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))

    # One RMS per channel across time/trials.
    ch_rms = rms_axis(x, axis=(1, 2))
    # One RMS per trial across channels/time.
    tr_rms = rms_axis(x, axis=(0, 1))

    ch_z = robust_z(np.log(np.maximum(ch_rms, 1e-12)))
    tr_z = robust_z(np.log(np.maximum(tr_rms, 1e-12)))

    return {
        "overall_rms_uV": overall_rms,
        "abs_p95_uV": float(np.percentile(flat_abs, 95)),
        "abs_p99_uV": float(np.percentile(flat_abs, 99)),
        "abs_p999_uV": float(np.percentile(flat_abs, 99.9)),
        "abs_max_uV": float(np.max(flat_abs)),
        "frac_abs_gt_100uV": float(np.mean(flat_abs > 100.0)),
        "frac_abs_gt_200uV": float(np.mean(flat_abs > 200.0)),
        "frac_abs_gt_500uV": float(np.mean(flat_abs > 500.0)),
        "channel_rms_min_uV": float(np.min(ch_rms)),
        "channel_rms_median_uV": float(np.median(ch_rms)),
        "channel_rms_max_uV": float(np.max(ch_rms)),
        "n_channel_logrms_robust_z_gt_3p5": int(np.sum(np.abs(ch_z) > 3.5)),
        "trial_rms_min_uV": float(np.min(tr_rms)),
        "trial_rms_median_uV": float(np.median(tr_rms)),
        "trial_rms_max_uV": float(np.max(tr_rms)),
        "n_trial_logrms_robust_z_gt_3p5": int(np.sum(np.abs(tr_z) > 3.5)),
    }


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    cleaned = project / "data" / "processed" / "cleaned_bt"
    outdir = project / "results" / "preprocessing"
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / "stage2B2_subject_task_qc.csv"
    out_txt = outdir / "stage2B2_full_qc_summary.txt"

    rows = []
    failures = []

    expected_subjects = [
        f"sub-G{g:02d}S{s:02d}"
        for g in range(1, 12)
        for s in range(1, 4)
    ]

    for sid in expected_subjects:
        path = cleaned / f"{sid}_cleaned_bt.npz"
        if not path.exists():
            failures.append(f"{sid}: missing {path.name}")
            continue

        try:
            z = np.load(path, allow_pickle=True)
            required = {
                "decision", "feedback", "fs", "units",
                "preprocessing_mode",
                "decision_n_rejected_components",
                "feedback_n_rejected_components",
            }
            missing = required - set(z.files)
            if missing:
                raise ValueError(f"missing keys {sorted(missing)}")

            units = str(np.asarray(z["units"]).squeeze())
            mode = str(np.asarray(z["preprocessing_mode"]).squeeze())
            fs = float(np.asarray(z["fs"]).squeeze())

            if units != "uV":
                raise ValueError(f"units={units!r}, expected 'uV'")
            if mode != "published_DataInBrief_reference":
                raise ValueError(f"preprocessing_mode={mode!r}")
            if abs(fs - 300.0) > 1e-9:
                raise ValueError(f"fs={fs}")

            for task, expected_shape, nrej_key in (
                ("decision", (19, 1500, 40), "decision_n_rejected_components"),
                ("feedback", (19, 900, 40), "feedback_n_rejected_components"),
            ):
                x = np.asarray(z[task], dtype=np.float32)
                if x.shape != expected_shape:
                    raise ValueError(f"{task} shape={x.shape}")
                if not np.isfinite(x).all():
                    raise ValueError(f"{task} contains non-finite values")

                s = summarize_task(x)
                row = {
                    "subject": sid,
                    "task": task,
                    "fs": fs,
                    "units": units,
                    "preprocessing_mode": mode,
                    "n_ica_rejected": int(np.asarray(z[nrej_key]).squeeze()),
                    **s,
                }
                rows.append(row)

        except Exception as e:
            failures.append(f"{sid}: {type(e).__name__}: {e}")

    # Subject-level robust z of log RMS within each task.
    for task in ("decision", "feedback"):
        idx = [i for i, r in enumerate(rows) if r["task"] == task]
        vals = np.array([rows[i]["overall_rms_uV"] for i in idx], dtype=float)
        zvals = robust_z(np.log(np.maximum(vals, 1e-12)))
        for i, rz in zip(idx, zvals):
            rows[i]["subject_logrms_robust_z"] = float(rz)
            rows[i]["subject_review_flag"] = bool(abs(rz) > 3.5)

    if rows:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Summary
    lines = [
        "Brain Topography Project — Stage 2B2 Full Cleaned-EEG QC",
        "=" * 66,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Cleaned directory: {cleaned}",
        "",
        "STRUCTURE / PROVENANCE",
        f"  Subject files successfully inspected: {len(rows)//2}/33",
        f"  Subject-task rows: {len(rows)}/66",
        f"  Failures: {len(failures)}",
        "",
        "TASK-LEVEL DISTRIBUTIONS",
    ]

    for task in ("decision", "feedback"):
        rr = [r for r in rows if r["task"] == task]
        vals = np.array([r["overall_rms_uV"] for r in rr])
        icas = np.array([r["n_ica_rejected"] for r in rr])
        p99 = np.array([r["abs_p99_uV"] for r in rr])
        flagged = [r for r in rr if r["subject_review_flag"]]

        lines.extend(
            [
                f"  {task}:",
                f"    RMS uV: median={np.median(vals):.3f}, "
                f"IQR=[{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}], "
                f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]",
                f"    |x| p99 uV: median={np.median(p99):.3f}, "
                f"IQR=[{np.percentile(p99,25):.3f}, {np.percentile(p99,75):.3f}], "
                f"range=[{np.min(p99):.3f}, {np.max(p99):.3f}]",
                f"    ICA rejected: median={np.median(icas):.1f}, "
                f"range=[{np.min(icas)}, {np.max(icas)}]",
                f"    Subject-level robust RMS review flags: {len(flagged)}",
            ]
        )

        if flagged:
            for r in sorted(
                flagged,
                key=lambda x: abs(x["subject_logrms_robust_z"]),
                reverse=True,
            ):
                lines.append(
                    f"      {r['subject']}: RMS={r['overall_rms_uV']:.3f} uV, "
                    f"robust-z={r['subject_logrms_robust_z']:.3f}, "
                    f"p99={r['abs_p99_uV']:.3f} uV, "
                    f"max={r['abs_max_uV']:.3f} uV, "
                    f"ICArej={r['n_ica_rejected']}, "
                    f"channel-flags={r['n_channel_logrms_robust_z_gt_3p5']}, "
                    f"trial-flags={r['n_trial_logrms_robust_z_gt_3p5']}"
                )

    # Top RMS subjects regardless of flag.
    lines.extend(["", "TOP 8 RMS SUBJECTS PER TASK"])
    for task in ("decision", "feedback"):
        rr = sorted(
            [r for r in rows if r["task"] == task],
            key=lambda x: x["overall_rms_uV"],
            reverse=True,
        )[:8]
        lines.append(f"  {task}:")
        for r in rr:
            lines.append(
                f"    {r['subject']}: RMS={r['overall_rms_uV']:.3f} uV, "
                f"p99={r['abs_p99_uV']:.3f}, "
                f"p999={r['abs_p999_uV']:.3f}, "
                f"max={r['abs_max_uV']:.3f}, "
                f"frac>|200|uV={100*r['frac_abs_gt_200uV']:.4f}%, "
                f"ICArej={r['n_ica_rejected']}"
            )

    if failures:
        lines.extend(["", "FAILURES"])
        for x in failures:
            lines.append(f"  {x}")

    n_review = sum(bool(r["subject_review_flag"]) for r in rows)
    pass_structure = (
        len(rows) == 66
        and not failures
    )

    lines.extend(
        [
            "",
            "RESULT",
            f"  {'STRUCTURAL PASS' if pass_structure else 'CHECK REQUIRED'}",
            f"  Subject-task review flags requiring inspection: {n_review}",
            "",
            "DECISION RULE",
            "  A review flag is NOT an exclusion. Inspect flagged subject/task",
            "  channel/trial QC before deciding whether any additional sensitivity",
            "  analysis is needed. Do not start connectivity analysis until this",
            "  diagnostic has been reviewed.",
            "",
            f"CSV: {out_csv}",
            f"Summary: {out_txt}",
        ]
    )

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if pass_structure else 1


if __name__ == "__main__":
    raise SystemExit(main())
