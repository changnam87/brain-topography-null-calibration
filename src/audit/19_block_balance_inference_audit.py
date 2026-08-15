#!/usr/bin/env python3
"""
Stage 2B6 — Block-level behavioral-balance audit before freezing inference.

Purpose
-------
The primary task contains 40 trials arranged as four 10-trial sessions/blocks.
Stage 2B5 showed strong triad-level CCC/Other imbalance in several groups.

Before viewing connectivity results, this script quantifies whether inference
should stratify only by triad or by triad x 10-trial block.

For each triad and 10-trial block it reports:
  - CCC / Other counts
  - minority-class count
  - information weight n1*n0/(n1+n0)
  - number of unique label reallocations C(10, n_CCC)
  - whether the block is informative for a within-block CCC-vs-Other contrast

It then compares:
  A) triad-stratified total contrast information from Stage 2B5 logic
  B) block-stratified contrast information, sum over informative 10-trial blocks

No EEG/connectivity data are read. No inferential rule is changed by this script.

Outputs
-------
results/preprocessing/stage2B6_block_balance.csv
results/preprocessing/stage2B6_block_balance_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
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

    outdir = project / "results" / "preprocessing"
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "stage2B6_block_balance.csv"
    out_txt = outdir / "stage2B6_block_balance_summary.txt"

    rows = []
    group_rows = []

    for gi, g in enumerate(range(1, cfg["n_groups"] + 1)):
        y = labels[gi].astype(np.uint8)

        n1_all = int(y.sum())
        n0_all = 40 - n1_all
        triad_w = info_weight(n1_all, n0_all)

        block_w_sum = 0.0
        informative_blocks = 0
        block_counts = []

        for b in range(4):
            yy = y[b * 10:(b + 1) * 10]
            n1 = int(yy.sum())
            n0 = 10 - n1
            w = info_weight(n1, n0)
            nalloc = math.comb(10, n1)
            informative = n1 > 0 and n0 > 0

            if informative:
                informative_blocks += 1
                block_w_sum += w

            block_counts.append(f"{n1}/{n0}")

            rows.append({
                "group": f"G{g:02d}",
                "block": b + 1,
                "trial_range": f"{b*10+1}-{(b+1)*10}",
                "CCC": n1,
                "Other": n0,
                "minority": min(n1, n0),
                "information_weight": w,
                "unique_within_block_label_allocations": nalloc,
                "informative_within_block": informative,
            })

        group_rows.append({
            "group": f"G{g:02d}",
            "triad_CCC": n1_all,
            "triad_Other": n0_all,
            "triad_information_weight": triad_w,
            "informative_blocks": informative_blocks,
            "block_information_weight_sum": block_w_sum,
            "block_vs_triad_information_ratio": (
                block_w_sum / triad_w if triad_w > 0 else np.nan
            ),
            "block_counts_CCC_Other": " | ".join(block_counts),
        })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    informative = [r for r in rows if r["informative_within_block"]]
    noninformative = [r for r in rows if not r["informative_within_block"]]

    total_triad_w = sum(r["triad_information_weight"] for r in group_rows)
    total_block_w = sum(r["block_information_weight_sum"] for r in group_rows)

    lines = [
        "Brain Topography Project — Stage 2B6 Block-Level Balance Audit",
        "=" * 72,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "BLOCK STRUCTURE",
        "  11 triads x 4 blocks x 10 trials = 44 block strata.",
        f"  Informative blocks (both CCC and Other): {len(informative)}/44",
        f"  Non-informative blocks (single class): {len(noninformative)}/44",
        "",
        "COUNTS BY TRIAD",
    ]

    for r in group_rows:
        lines.append(
            f"  {r['group']}: blocks CCC/Other = {r['block_counts_CCC_Other']}; "
            f"informative blocks={r['informative_blocks']}/4; "
            f"triad weight={r['triad_information_weight']:.3f}; "
            f"block-stratified weight sum={r['block_information_weight_sum']:.3f}; "
            f"ratio={100*r['block_vs_triad_information_ratio']:.1f}%"
        )

    lines.extend([
        "",
        "NON-INFORMATIVE BLOCKS",
    ])
    if noninformative:
        for r in noninformative:
            lines.append(
                f"  {r['group']} block {r['block']} "
                f"(trials {r['trial_range']}): CCC={r['CCC']}, Other={r['Other']}"
            )
    else:
        lines.append("  None")

    # Distribution of minority count among informative blocks.
    lines.extend([
        "",
        "INFORMATIVE-BLOCK MINORITY COUNTS",
    ])
    if informative:
        cnt = Counter(int(r["minority"]) for r in informative)
        for k in sorted(cnt):
            lines.append(f"  minority={k}: {cnt[k]} block(s)")

    # Label allocation support
    allocs = np.array(
        [int(r["unique_within_block_label_allocations"]) for r in informative],
        dtype=int
    ) if informative else np.array([], dtype=int)

    lines.extend([
        "",
        "WITHIN-BLOCK PERMUTATION SUPPORT",
    ])
    if len(allocs):
        lines.append(
            f"  Unique allocations per informative block: "
            f"median={np.median(allocs):.1f}, "
            f"range=[{np.min(allocs)}, {np.max(allocs)}]"
        )
        low = [r for r in informative
               if int(r["unique_within_block_label_allocations"]) <= 10]
        lines.append(
            f"  Informative blocks with <=10 unique allocations: {len(low)}"
        )
        for r in low:
            lines.append(
                f"    {r['group']} block {r['block']}: "
                f"{r['CCC']}/{r['Other']}, "
                f"allocations={r['unique_within_block_label_allocations']}"
            )

    lines.extend([
        "",
        "TOTAL CONTRAST INFORMATION",
        f"  Triad-stratified total information weight: {total_triad_w:.3f}",
        f"  Block-stratified total information weight: {total_block_w:.3f}",
        f"  Block/triad information ratio: "
        f"{100*total_block_w/total_triad_w:.1f}%"
        if total_triad_w > 0 else "  Block/triad information ratio: NA",
        "",
        "RESULT",
        "  DIAGNOSTIC COMPLETE",
        "",
        "DECISION GUIDE",
        "  If block stratification retains substantial contrast information,",
        "  a triad x block fixed-effect / restricted-permutation statistic is",
        "  preferable because it controls session/block-level drift and learning.",
        "  If it destroys most contrast information, retain triad-stratified",
        "  inference as primary and use block-aware inference as sensitivity.",
        "  Freeze the choice before computing connectivity.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
