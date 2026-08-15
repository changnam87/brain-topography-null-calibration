#!/usr/bin/env python3
"""
Stage 2B3 — Drill-down QC for flagged cleaned EEG subject/tasks.

Purpose
-------
Stage 2B2 identified subject/task rows with unusually high cleaned-EEG RMS.
This script inspects ONLY those flagged rows and determines whether the excess
energy is concentrated in a few channels/trials or is diffuse across the record.

No data are modified or excluded.

For each flagged subject/task:
  - ranks all 19 channels by RMS, p99, max amplitude, >200/>500 uV fractions
  - ranks all 40 trials by RMS, p99, max amplitude, >200/>500 uV fractions
  - identifies within-subject channel/trial robust-z outliers
  - computes concentration of total squared energy in top 1/3 channels and trials
  - reports CCC vs Other composition of flagged high-RMS trials
  - compares decision/feedback top-channel overlap for the same participant

Inputs
------
results/preprocessing/stage2B2_subject_task_qc.csv
data/processed/cleaned_bt/sub-GxxSxx_cleaned_bt.npz
OpenNeuro decision events through bt.io for CCC/Other labels

Outputs
-------
results/preprocessing/stage2B3_flagged_channel_qc.csv
results/preprocessing/stage2B3_flagged_trial_qc.csv
results/preprocessing/stage2B3_flagged_summary.txt

Requirements
------------
Python 3, numpy
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
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


def parse_subject(sid: str):
    # sub-G02S03 -> group=2 participant=3
    import re
    m = re.fullmatch(r"sub-G(\d{2})S(\d{2})", sid)
    if not m:
        raise ValueError(f"Unexpected subject id: {sid}")
    return int(m.group(1)), int(m.group(2))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def rms(x: np.ndarray, axis=None):
    return np.sqrt(np.mean(np.square(x, dtype=np.float64), axis=axis))


def energy_contribution_by_channel(x):
    e = np.sum(np.square(x, dtype=np.float64), axis=(1,2))
    total = float(e.sum())
    return e / total if total > 0 else np.zeros_like(e)


def energy_contribution_by_trial(x):
    e = np.sum(np.square(x, dtype=np.float64), axis=(0,1))
    total = float(e.sum())
    return e / total if total > 0 else np.zeros_like(e)


def summarize_vector_abs(v):
    a = np.abs(np.asarray(v, dtype=np.float64)).reshape(-1)
    return (
        float(np.percentile(a, 99)),
        float(np.max(a)),
        float(np.mean(a > 200.0)),
        float(np.mean(a > 500.0)),
    )


def boolstr(v):
    return str(v).strip().lower() in {"1","true","yes","y"}


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.io import triad_labels

    cfg = load_config(project / "configs" / "bt_analysis_config.json")

    qc2 = project / "results" / "preprocessing" / "stage2B2_subject_task_qc.csv"
    cleaned = project / "data" / "processed" / "cleaned_bt"
    outdir = project / "results" / "preprocessing"
    outdir.mkdir(parents=True, exist_ok=True)

    channel_csv = outdir / "stage2B3_flagged_channel_qc.csv"
    trial_csv = outdir / "stage2B3_flagged_trial_qc.csv"
    summary_txt = outdir / "stage2B3_flagged_summary.txt"

    if not qc2.exists():
        print(f"FAIL: missing Stage 2B2 CSV: {qc2}", file=sys.stderr)
        return 2

    stage2b2 = read_csv(qc2)
    flagged = [r for r in stage2b2 if boolstr(r.get("subject_review_flag","false"))]

    if not flagged:
        print("No flagged subject/task rows in Stage 2B2.")
        summary_txt.write_text("No flagged subject/task rows.\n", encoding="utf-8")
        return 0

    channel_rows = []
    trial_rows = []
    summaries = []
    failures = []
    top_channels_by_subject_task = {}

    label_cache = {}

    for fr in flagged:
        sid = fr["subject"]
        task = fr["task"]
        group, participant = parse_subject(sid)
        npz = cleaned / f"{sid}_cleaned_bt.npz"

        try:
            z = np.load(npz, allow_pickle=True)
            x = np.asarray(z[task], dtype=np.float32)
            ch_names = [str(v) for v in z["channel_names"].tolist()]
            if x.shape[0] != 19 or x.shape[2] != 40:
                raise ValueError(f"{sid} {task}: unexpected shape {x.shape}")

            if group not in label_cache:
                y, choices = triad_labels(cfg["dataset_root"], group)
                label_cache[group] = y.astype(np.uint8)
            labels = label_cache[group]

            # -------- channel metrics --------
            ch_rms = rms(x, axis=(1,2))
            ch_z = robust_z(np.log(np.maximum(ch_rms, 1e-12)))
            ch_energy = energy_contribution_by_channel(x)
            ch_order = np.argsort(ch_rms)[::-1]
            ch_rank = np.empty(19, dtype=int)
            ch_rank[ch_order] = np.arange(1,20)

            for c in range(19):
                p99, mx, frac200, frac500 = summarize_vector_abs(x[c])
                channel_rows.append({
                    "subject": sid,
                    "task": task,
                    "channel_index": c,
                    "channel": ch_names[c],
                    "rms_uV": float(ch_rms[c]),
                    "logrms_robust_z_within_subject": float(ch_z[c]),
                    "review_flag_abs_robust_z_gt_3p5": bool(abs(ch_z[c]) > 3.5),
                    "p99_abs_uV": p99,
                    "max_abs_uV": mx,
                    "frac_abs_gt_200uV": frac200,
                    "frac_abs_gt_500uV": frac500,
                    "energy_fraction": float(ch_energy[c]),
                    "rms_rank_desc": int(ch_rank[c]),
                })

            top_channels = [ch_names[i] for i in ch_order[:5]]
            top_channels_by_subject_task[(sid,task)] = top_channels

            # -------- trial metrics --------
            tr_rms = rms(x, axis=(0,1))
            tr_z = robust_z(np.log(np.maximum(tr_rms, 1e-12)))
            tr_energy = energy_contribution_by_trial(x)
            tr_order = np.argsort(tr_rms)[::-1]
            tr_rank = np.empty(40, dtype=int)
            tr_rank[tr_order] = np.arange(1,41)

            high_trials = []

            for t in range(40):
                p99, mx, frac200, frac500 = summarize_vector_abs(x[:,:,t])
                flag = bool(abs(tr_z[t]) > 3.5)
                if flag:
                    high_trials.append(t)
                trial_rows.append({
                    "subject": sid,
                    "task": task,
                    "trial_1based": t+1,
                    "BT_label": "CCC" if labels[t] == 1 else "Other",
                    "rms_uV": float(tr_rms[t]),
                    "logrms_robust_z_within_subject": float(tr_z[t]),
                    "review_flag_abs_robust_z_gt_3p5": flag,
                    "p99_abs_uV": p99,
                    "max_abs_uV": mx,
                    "frac_abs_gt_200uV": frac200,
                    "frac_abs_gt_500uV": frac500,
                    "energy_fraction": float(tr_energy[t]),
                    "rms_rank_desc": int(tr_rank[t]),
                })

            flagged_ccc = int(sum(labels[t] == 1 for t in high_trials))
            flagged_other = int(sum(labels[t] == 0 for t in high_trials))

            top1_ch = float(np.max(ch_energy))
            top3_ch = float(np.sum(np.sort(ch_energy)[-3:]))
            top1_tr = float(np.max(tr_energy))
            top3_tr = float(np.sum(np.sort(tr_energy)[-3:]))

            # Descriptive concentration class only; never an exclusion rule.
            if top3_tr >= 0.50 and top3_tr >= top3_ch:
                concentration = "trial-concentrated"
            elif top3_ch >= 0.50 and top3_ch > top3_tr:
                concentration = "channel-concentrated"
            else:
                concentration = "diffuse/mixed"

            summaries.append({
                "subject": sid,
                "task": task,
                "overall_rms_uV": float(fr["overall_rms_uV"]),
                "subject_logrms_robust_z": float(fr["subject_logrms_robust_z"]),
                "n_channel_flags": int(sum(np.abs(ch_z)>3.5)),
                "n_trial_flags": int(sum(np.abs(tr_z)>3.5)),
                "top1_channel": top_channels[0],
                "top5_channels": ",".join(top_channels),
                "top1_channel_energy_pct": 100*top1_ch,
                "top3_channel_energy_pct": 100*top3_ch,
                "top1_trial_1based": int(tr_order[0]+1),
                "top5_trials_1based": ",".join(str(int(i+1)) for i in tr_order[:5]),
                "top1_trial_energy_pct": 100*top1_tr,
                "top3_trial_energy_pct": 100*top3_tr,
                "flagged_trials": ",".join(str(t+1) for t in high_trials),
                "flagged_trials_CCC": flagged_ccc,
                "flagged_trials_Other": flagged_other,
                "energy_concentration": concentration,
            })

        except Exception as e:
            failures.append(f"{sid} {task}: {type(e).__name__}: {e}")

    # decision-feedback top-channel overlap
    overlap_lines = []
    for sid in sorted(set(r["subject"] for r in summaries)):
        d = top_channels_by_subject_task.get((sid,"decision"))
        f = top_channels_by_subject_task.get((sid,"feedback"))
        if d and f:
            overlap = sorted(set(d) & set(f))
            overlap_lines.append(
                (sid, len(overlap), ",".join(overlap), ",".join(d), ",".join(f))
            )

    with channel_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(channel_rows[0].keys()))
        w.writeheader()
        w.writerows(channel_rows)

    with trial_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(trial_rows[0].keys()))
        w.writeheader()
        w.writerows(trial_rows)

    lines = [
        "Brain Topography Project — Stage 2B3 Flagged EEG Drill-Down",
        "="*70,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Flagged subject/task rows inspected: {len(summaries)}/{len(flagged)}",
        f"Failures: {len(failures)}",
        "",
        "FLAGGED SUBJECT/TASK SUMMARY",
    ]

    for s in sorted(
        summaries,
        key=lambda r: abs(r["subject_logrms_robust_z"]),
        reverse=True
    ):
        lines.extend([
            f"  {s['subject']} {s['task']}:",
            f"    RMS={s['overall_rms_uV']:.3f} uV; "
            f"subject robust-z={s['subject_logrms_robust_z']:.3f}",
            f"    channel flags={s['n_channel_flags']}; "
            f"trial flags={s['n_trial_flags']}",
            f"    top channels: {s['top5_channels']}",
            f"    top-1/top-3 channel energy: "
            f"{s['top1_channel_energy_pct']:.2f}% / "
            f"{s['top3_channel_energy_pct']:.2f}%",
            f"    top trials: {s['top5_trials_1based']}",
            f"    top-1/top-3 trial energy: "
            f"{s['top1_trial_energy_pct']:.2f}% / "
            f"{s['top3_trial_energy_pct']:.2f}%",
            f"    robust-z flagged trials: "
            f"{s['flagged_trials'] if s['flagged_trials'] else 'none'}",
            f"    flagged-trial labels: CCC={s['flagged_trials_CCC']}, "
            f"Other={s['flagged_trials_Other']}",
            f"    descriptive concentration: {s['energy_concentration']}",
        ])

    if overlap_lines:
        lines.extend(["", "DECISION/FEEDBACK TOP-5 CHANNEL OVERLAP"])
        for sid, n, overlap, d, f in overlap_lines:
            lines.append(
                f"  {sid}: overlap={n}/5 [{overlap}]"
            )
            lines.append(f"    decision: {d}")
            lines.append(f"    feedback: {f}")

    # Aggregate behavioral balance among all flagged trials.
    all_flag_trial_rows = [
        r for r in trial_rows
        if bool(r["review_flag_abs_robust_z_gt_3p5"])
    ]
    nccc = sum(r["BT_label"]=="CCC" for r in all_flag_trial_rows)
    nother = sum(r["BT_label"]=="Other" for r in all_flag_trial_rows)
    lines.extend([
        "",
        "FLAGGED-TRIAL BEHAVIORAL BALANCE (DESCRIPTIVE)",
        f"  Total flagged trial instances: {len(all_flag_trial_rows)}",
        f"  CCC: {nccc}",
        f"  Other: {nother}",
        "",
        "INTERPRETATION RULE",
        "  No automatic subject/channel/trial exclusion is made here.",
        "  If excess energy is concentrated in a small number of trials/channels,",
        "  preserve the full-sample primary analysis and add a pre-specified",
        "  sensitivity analysis excluding only objectively flagged segments.",
        "  If the elevation is diffuse across a participant, prefer robust",
        "  sensitivity at the triad/subject level rather than ad hoc trial deletion.",
    ])

    if failures:
        lines.extend(["", "FAILURES"])
        for x in failures:
            lines.append(f"  {x}")

    passed = len(summaries) == len(flagged) and not failures
    lines.extend([
        "",
        "RESULT",
        f"  {'DIAGNOSTIC COMPLETE' if passed else 'CHECK REQUIRED'}",
        "",
        f"Channel CSV: {channel_csv}",
        f"Trial CSV: {trial_csv}",
        f"Summary: {summary_txt}",
    ])

    summary_txt.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
