#!/usr/bin/env python3
"""
Stage 2B5 — Triad behavioral-balance and inference-feasibility audit.

Purpose
-------
Before computing connectivity, quantify how much within-triad information exists
for the CCC vs Other contrast. This is important because a triad with 39 CCC
and 1 Other trial should not automatically carry the same statistical weight
as a triad with a balanced 20/20 split.

This script is DIAGNOSTIC ONLY. It does not change labels, delete trials,
or select a statistical threshold.

For each triad it reports:
  - primary CCC / Other counts
  - minority-class count
  - imbalance ratio
  - contrast information weight n1*n0/(n1+n0)
  - normalized weight relative to a perfectly balanced 20/20 triad
  - sensitivity-mask counts for decision and feedback
  - the same information weights after the frozen artifact mask

Interpretation
--------------
The information weight w = n_CCC*n_Other/(n_CCC+n_Other) is proportional to
the inverse sampling variance of a two-group mean difference when per-trial
variances are comparable. It is reported here to compare:
  A) equal-triad aggregation, versus
  B) a stratified, information-weighted within-triad permutation statistic.

No final choice is made by this script.

Outputs
-------
results/preprocessing/stage2B5_behavior_balance.csv
results/preprocessing/stage2B5_behavior_balance_summary.txt
"""

from __future__ import annotations

import argparse
import csv
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


def info_weight(n1: int, n0: int) -> float:
    if n1 <= 0 or n0 <= 0:
        return 0.0
    return float(n1 * n0 / (n1 + n0))


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.io import all_group_labels

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    labels, _ = all_group_labels(cfg["dataset_root"], cfg["n_groups"])

    mask_path = (
        project / "results" / "preprocessing"
        / "stage2B4_sensitivity_mask.npz"
    )
    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    mz = np.load(mask_path, allow_pickle=True)
    mask = mz["retain_mask"].astype(bool)  # G x 2 x T
    tasks = [str(x) for x in mz["tasks"].tolist()]

    outdir = project / "results" / "preprocessing"
    out_csv = outdir / "stage2B5_behavior_balance.csv"
    out_txt = outdir / "stage2B5_behavior_balance_summary.txt"

    rows = []
    balanced_weight = info_weight(20, 20)  # 10.0

    for gi, g in enumerate(range(1, cfg["n_groups"] + 1)):
        y = labels[gi].astype(bool)
        n1 = int(y.sum())
        n0 = int((~y).sum())
        w = info_weight(n1, n0)
        minority = min(n1, n0)
        majority = max(n1, n0)

        row = {
            "group": f"G{g:02d}",
            "primary_CCC": n1,
            "primary_Other": n0,
            "primary_minority_count": minority,
            "primary_imbalance_ratio_majority_to_minority": (
                float(majority / minority) if minority > 0 else np.inf
            ),
            "primary_information_weight": w,
            "primary_information_weight_vs_20_20": w / balanced_weight,
        }

        for ti, task in enumerate(tasks):
            keep = mask[gi, ti]
            n1k = int(np.sum(y & keep))
            n0k = int(np.sum((~y) & keep))
            wk = info_weight(n1k, n0k)
            minority_k = min(n1k, n0k)
            majority_k = max(n1k, n0k)

            row.update({
                f"{task}_retained_CCC": n1k,
                f"{task}_retained_Other": n0k,
                f"{task}_retained_minority_count": minority_k,
                f"{task}_retained_imbalance_ratio_majority_to_minority": (
                    float(majority_k / minority_k)
                    if minority_k > 0 else np.inf
                ),
                f"{task}_information_weight": wk,
                f"{task}_information_weight_vs_20_20": wk / balanced_weight,
            })

        rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = [
        "Brain Topography Project — Stage 2B5 Behavioral-Balance Audit",
        "=" * 70,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "PRIMARY CCC VS OTHER COUNTS BY TRIAD",
    ]

    for r in rows:
        lines.append(
            f"  {r['group']}: CCC={r['primary_CCC']}, "
            f"Other={r['primary_Other']}, "
            f"minority={r['primary_minority_count']}, "
            f"imbalance={r['primary_imbalance_ratio_majority_to_minority']:.2f}:1, "
            f"information weight={r['primary_information_weight']:.3f} "
            f"({100*r['primary_information_weight_vs_20_20']:.1f}% of 20/20)"
        )

    lines.extend([
        "",
        "PRIMARY MINORITY-CLASS COUNTS",
    ])
    for threshold in (1, 2, 3, 5, 10):
        groups = [
            r["group"] for r in rows
            if int(r["primary_minority_count"]) <= threshold
        ]
        lines.append(
            f"  minority <= {threshold}: {len(groups)} triad(s)"
            + (f" [{','.join(groups)}]" if groups else "")
        )

    for task in tasks:
        lines.extend([
            "",
            f"ARTIFACT-SENSITIVITY COUNTS — {task.upper()}",
        ])
        for r in rows:
            lines.append(
                f"  {r['group']}: "
                f"CCC={r[f'{task}_retained_CCC']}, "
                f"Other={r[f'{task}_retained_Other']}, "
                f"minority={r[f'{task}_retained_minority_count']}, "
                f"information weight={r[f'{task}_information_weight']:.3f} "
                f"({100*r[f'{task}_information_weight_vs_20_20']:.1f}% of 20/20)"
            )

        zero = [
            r["group"] for r in rows
            if int(r[f"{task}_retained_minority_count"]) == 0
        ]
        lines.append(
            f"  Cells with one class absent: {len(zero)}"
            + (f" [{','.join(zero)}]" if zero else "")
        )

    # Weight concentration: how much of total contrast information comes from each triad.
    weights = np.array(
        [r["primary_information_weight"] for r in rows], dtype=float
    )
    total_w = float(weights.sum())
    lines.extend([
        "",
        "PRIMARY INFORMATION-WEIGHT SHARE",
    ])
    order = np.argsort(weights)[::-1]
    for i in order:
        share = 100 * weights[i] / total_w if total_w > 0 else 0.0
        lines.append(
            f"  {rows[i]['group']}: {share:.2f}%"
        )

    lines.extend([
        "",
        "RESULT",
        "  DIAGNOSTIC COMPLETE",
        "",
        "WHY THIS MATTERS",
        "  Equal-triad averaging gives a 39/1 triad the same nominal weight as",
        "  a 20/20 triad even though its within-triad contrast is estimated from",
        "  far less minority-class information. Before connectivity inference,",
        "  compare equal-triad aggregation with a stratified information-weighted",
        "  permutation statistic. The final choice should be frozen before viewing",
        "  connectivity results.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
