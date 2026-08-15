#!/usr/bin/env python3
"""
Stage 2B4 — Freeze artifact-sensitivity trial masks for BT analysis.

Purpose
-------
Stage 2B3 found residual high-energy EEG segments after the validated published
preprocessing. The primary BT analysis will retain ALL trials. This script
creates a separate, pre-specified sensitivity mask only.

Two-stage, label-blind rule
---------------------------
A participant/task contributes trial exclusions only if:
  1) the participant/task was a Stage 2B2 subject-level RMS review flag
     (|robust z of log RMS across subjects| > 3.5), AND
  2) the trial was a Stage 2B3 within-subject RMS review flag
     (|robust z of log RMS across that subject/task's 40 trials| > 3.5).

For a triad/task, the sensitivity analysis excludes the UNION of flagged trials
across the three participants. The same retained trial set is then used for all
three within-triad dyads, preserving comparable triad-level inference.

Behavioral labels are used ONLY AFTER the EEG-only mask has been frozen, to
summarize retained CCC/Other counts. They do not influence the mask.

Outputs
-------
results/preprocessing/stage2B4_sensitivity_trial_mask.csv
results/preprocessing/stage2B4_sensitivity_counts_by_group.csv
results/preprocessing/stage2B4_sensitivity_mask.npz
results/preprocessing/stage2B4_sensitivity_mask_summary.txt
"""

from __future__ import annotations

import argparse
import csv
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


def boolstr(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def parse_sid(sid: str):
    import re
    m = re.fullmatch(r"sub-G(\d{2})S(\d{2})", sid)
    if not m:
        raise ValueError(sid)
    return int(m.group(1)), int(m.group(2))


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.io import all_group_labels

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    pre = project / "results" / "preprocessing"

    b2 = pre / "stage2B2_subject_task_qc.csv"
    b3 = pre / "stage2B3_flagged_trial_qc.csv"

    if not b2.exists() or not b3.exists():
        raise FileNotFoundError(
            "Need Stage 2B2 and Stage 2B3 CSV outputs before freezing the mask."
        )

    rows2 = read_csv(b2)
    rows3 = read_csv(b3)

    # Subject/task rows eligible to contribute trial flags.
    subject_task_flag = {
        (r["subject"], r["task"])
        for r in rows2
        if boolstr(r.get("subject_review_flag", "false"))
    }

    # EEG-only flagged trial sets by group/task/participant.
    flagged = defaultdict(set)
    source_detail = defaultdict(list)

    for r in rows3:
        sid = r["subject"]
        task = r["task"]
        if (sid, task) not in subject_task_flag:
            continue
        if not boolstr(r.get("review_flag_abs_robust_z_gt_3p5", "false")):
            continue
        g, p = parse_sid(sid)
        t0 = int(r["trial_1based"]) - 1
        flagged[(g, task)].add(t0)
        source_detail[(g, task, t0)].append(sid)

    labels, choices = all_group_labels(cfg["dataset_root"], cfg["n_groups"])
    # True = retain for artifact-sensitivity analysis.
    masks = np.ones((cfg["n_groups"], 2, cfg["n_trials"]), dtype=np.uint8)
    tasks = ["decision", "feedback"]

    mask_rows = []
    count_rows = []

    for gi, g in enumerate(range(1, cfg["n_groups"] + 1)):
        for ti, task in enumerate(tasks):
            excluded = sorted(flagged.get((g, task), set()))
            for t0 in excluded:
                masks[gi, ti, t0] = 0

            keep = masks[gi, ti].astype(bool)
            y = labels[gi].astype(bool)

            n_ccc_total = int(y.sum())
            n_other_total = int((~y).sum())
            n_ccc_keep = int(np.sum(y & keep))
            n_other_keep = int(np.sum((~y) & keep))

            count_rows.append({
                "group": f"G{g:02d}",
                "task": task,
                "n_total": 40,
                "n_excluded": int((~keep).sum()),
                "n_retained": int(keep.sum()),
                "CCC_total": n_ccc_total,
                "Other_total": n_other_total,
                "CCC_excluded": n_ccc_total - n_ccc_keep,
                "Other_excluded": n_other_total - n_other_keep,
                "CCC_retained": n_ccc_keep,
                "Other_retained": n_other_keep,
                "both_classes_retained": bool(
                    n_ccc_keep > 0 and n_other_keep > 0
                ),
            })

            for t0 in range(40):
                mask_rows.append({
                    "group": f"G{g:02d}",
                    "task": task,
                    "trial_1based": t0 + 1,
                    "retain_sensitivity": bool(masks[gi, ti, t0]),
                    "BT_label": "CCC" if labels[gi, t0] == 1 else "Other",
                    "flag_source_subjects": ",".join(
                        source_detail.get((g, task, t0), [])
                    ),
                })

    mask_csv = pre / "stage2B4_sensitivity_trial_mask.csv"
    counts_csv = pre / "stage2B4_sensitivity_counts_by_group.csv"
    mask_npz = pre / "stage2B4_sensitivity_mask.npz"
    summary = pre / "stage2B4_sensitivity_mask_summary.txt"

    with mask_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(mask_rows[0].keys()))
        w.writeheader()
        w.writerows(mask_rows)

    with counts_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(count_rows[0].keys()))
        w.writeheader()
        w.writerows(count_rows)

    np.savez_compressed(
        mask_npz,
        retain_mask=masks,
        tasks=np.array(tasks, dtype=object),
        labels=labels.astype(np.uint8),
        rule=np.array(
            "Stage2B2 subject |robust-z(logRMS)|>3.5 AND "
            "Stage2B3 trial |robust-z(logRMS)|>3.5; "
            "union across participants within triad/task",
            dtype=object,
        ),
    )

    lines = [
        "Brain Topography Project — Stage 2B4 Frozen Artifact-Sensitivity Mask",
        "=" * 75,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "PRIMARY ANALYSIS",
        "  Retain all 440 triadic trials. No EEG-QC exclusions.",
        "",
        "SENSITIVITY MASK RULE",
        "  Stage 2B2 subject/task review flag AND Stage 2B3 trial review flag.",
        "  The mask is EEG-only and label-blind.",
        "  Within each triad/task, use the UNION across participants.",
        "  The same retained trial set is used for pair12, pair13, and pair23.",
        "",
        "COUNTS BY GROUP/TASK",
    ]

    for r in count_rows:
        if int(r["n_excluded"]) > 0:
            lines.append(
                f"  {r['group']} {r['task']}: "
                f"excluded={r['n_excluded']}, retained={r['n_retained']}; "
                f"CCC {r['CCC_total']}->{r['CCC_retained']} "
                f"(excluded {r['CCC_excluded']}), "
                f"Other {r['Other_total']}->{r['Other_retained']} "
                f"(excluded {r['Other_excluded']}); "
                f"both classes retained={r['both_classes_retained']}"
            )

    # Aggregate counts task-wise
    lines.extend(["", "AGGREGATE COUNTS"])
    for task in tasks:
        rr = [r for r in count_rows if r["task"] == task]
        excl = sum(int(r["n_excluded"]) for r in rr)
        ccc_excl = sum(int(r["CCC_excluded"]) for r in rr)
        oth_excl = sum(int(r["Other_excluded"]) for r in rr)
        ccc_keep = sum(int(r["CCC_retained"]) for r in rr)
        oth_keep = sum(int(r["Other_retained"]) for r in rr)
        lines.append(
            f"  {task}: excluded trial-instances={excl}; "
            f"CCC excluded={ccc_excl}, Other excluded={oth_excl}; "
            f"retained CCC={ccc_keep}, retained Other={oth_keep}"
        )

    bad = [r for r in count_rows if not bool(r["both_classes_retained"])]
    lines.extend([
        "",
        "CLASS RETENTION",
        f"  Group/task cells without both CCC and Other after masking: {len(bad)}",
    ])
    for r in bad:
        lines.append(
            f"    {r['group']} {r['task']}: "
            f"CCC={r['CCC_retained']}, Other={r['Other_retained']}"
        )

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS — sensitivity mask frozen' if not bad else 'CHECK REQUIRED'}",
        "",
        "INTERPRETATION",
        "  Because residual high-energy trials were disproportionately CCC in",
        "  Stage 2B3, the sensitivity analysis must be reported as a robustness",
        "  check rather than replacing the full-sample primary analysis.",
        "",
        f"Mask CSV: {mask_csv}",
        f"Counts CSV: {counts_csv}",
        f"Mask NPZ: {mask_npz}",
        f"Summary: {summary}",
    ])

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
