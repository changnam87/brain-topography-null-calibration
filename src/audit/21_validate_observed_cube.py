#!/usr/bin/env python3
"""
Stage 3B — Full observed PLV cube technical audit (label-blind with respect to PLV).

Purpose
-------
Validate the full observed PLV cache before any CCC-vs-Other contrast or
permutation inference is run.

This script does NOT compare PLV by behavioral condition. Behavioral arrays are
checked only for structural consistency with the already frozen metadata.

Checks
------
- expected keys and array shapes
- PLV finite and bounded in [0, 1]
- exact condition order from config
- exact dyad order
- exact 19-channel order
- 19x19 directed inter-brain edge indexing, complete and non-duplicated
- statistic_design matches the frozen config
- labels/choices metadata remain internally consistent (CCC=251, Other=189)
- no degenerate condition/dyad cells
- no all-constant edge across the 440 trials
- descriptive PLV distributions by task-band/dyad only (no behavioral split)

Outputs
-------
results/observed/stage3B_observed_cube_qc.csv
results/observed/stage3B_observed_cube_summary.txt
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


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config, condition_list

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    cube_path = project / "results" / "observed" / "trial_connectivity_cube.npz"
    outdir = project / "results" / "observed"
    out_csv = outdir / "stage3B_observed_cube_qc.csv"
    out_txt = outdir / "stage3B_observed_cube_summary.txt"

    if not cube_path.exists():
        print(f"FAIL: missing {cube_path}", file=sys.stderr)
        return 2

    z = np.load(cube_path, allow_pickle=True)

    required = {
        "plv", "labels", "choices", "conditions", "dyads",
        "edge_i", "edge_j", "channel_names", "statistic_design"
    }
    missing = required - set(z.files)
    if missing:
        print(f"FAIL: missing keys {sorted(missing)}", file=sys.stderr)
        return 3

    plv = np.asarray(z["plv"], dtype=np.float32)
    labels = np.asarray(z["labels"], dtype=np.uint8)
    choices = np.asarray(z["choices"]).astype("U1")
    conditions = [str(x) for x in z["conditions"].tolist()]
    dyads = [str(x) for x in z["dyads"].tolist()]
    edge_i = np.asarray(z["edge_i"], dtype=int)
    edge_j = np.asarray(z["edge_j"], dtype=int)
    channels = [str(x) for x in z["channel_names"].tolist()]
    statistic_design = str(np.asarray(z["statistic_design"]).squeeze())

    expected_conditions = [f"{t}|{b}" for t, b in condition_list(cfg)]
    expected_dyads = ["pair12", "pair13", "pair23"]
    expected_channels = list(cfg["channel_names"])
    expected_shape = (
        int(cfg["n_groups"]),
        len(expected_conditions),
        3,
        int(cfg["n_trials"]),
        int(cfg["n_channels"]) ** 2,
    )

    checks = {}

    checks["plv_shape"] = plv.shape == expected_shape
    checks["labels_shape"] = labels.shape == (11, 40)
    checks["choices_shape"] = choices.shape == (11, 40, 3)
    checks["conditions_exact"] = conditions == expected_conditions
    checks["dyads_exact"] = dyads == expected_dyads
    checks["channels_exact"] = channels == expected_channels
    checks["statistic_design_exact"] = (
        statistic_design == cfg["inference"]["primary_statistic"]
    )
    checks["finite"] = bool(np.isfinite(plv).all())
    checks["bounded_0_1"] = bool(
        float(plv.min()) >= -1e-7 and float(plv.max()) <= 1.0 + 1e-7
    )

    # Behavioral metadata only; does not condition PLV on labels.
    choices_valid = bool(np.isin(choices, ["C", "D"]).all())
    reconstructed = np.all(choices == "C", axis=2).astype(np.uint8)
    checks["choices_only_C_D"] = choices_valid
    checks["labels_match_choices"] = bool(np.array_equal(labels, reconstructed))
    checks["CCC_251_Other_189"] = bool(
        int(labels.sum()) == 251 and int(labels.size - labels.sum()) == 189
    )

    # Edge indexing: row-major complete directed Cartesian product.
    n_ch = int(cfg["n_channels"])
    exp_i, exp_j = np.meshgrid(
        np.arange(n_ch), np.arange(n_ch), indexing="ij"
    )
    exp_i = exp_i.reshape(-1)
    exp_j = exp_j.reshape(-1)
    checks["edge_index_exact"] = bool(
        np.array_equal(edge_i, exp_i) and np.array_equal(edge_j, exp_j)
    )
    checks["edge_pairs_unique"] = len(set(zip(edge_i.tolist(), edge_j.tolist()))) == n_ch * n_ch

    # Degeneracy checks across all 440 trial instances for every unit.
    # Flatten G and trial while preserving C,D,E.
    x = plv.transpose(1, 2, 0, 3, 4).reshape(
        len(conditions), len(dyads), 11 * 40, n_ch * n_ch
    )
    unit_var = np.var(x.astype(np.float64), axis=2)
    n_zero_var_units = int(np.sum(unit_var <= 1e-15))
    checks["no_zero_variance_units"] = n_zero_var_units == 0

    rows = []
    for ci, cond in enumerate(conditions):
        task, band = cond.split("|")
        for di, dyad in enumerate(dyads):
            vals = x[ci, di].reshape(-1).astype(np.float64)
            per_trial_edge_sd = np.std(x[ci, di], axis=1)
            rows.append({
                "task": task,
                "band": band,
                "dyad": dyad,
                "n_values": int(vals.size),
                "min": float(np.min(vals)),
                "p01": float(np.percentile(vals, 1)),
                "median": float(np.median(vals)),
                "mean": float(np.mean(vals)),
                "p99": float(np.percentile(vals, 99)),
                "max": float(np.max(vals)),
                "fraction_exact_zero": float(np.mean(vals == 0.0)),
                "fraction_ge_0p95": float(np.mean(vals >= 0.95)),
                "median_trial_edge_sd": float(np.median(per_trial_edge_sd)),
                "min_trial_edge_sd": float(np.min(per_trial_edge_sd)),
            })

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    all_pass = all(checks.values())

    lines = [
        "Brain Topography Project — Stage 3B Full Observed PLV Cube Audit",
        "=" * 72,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Cube: {cube_path}",
        "",
        "STRUCTURE",
        f"  PLV shape: {plv.shape} expected={expected_shape}",
        f"  labels shape: {labels.shape}",
        f"  choices shape: {choices.shape}",
        f"  statistic_design: {statistic_design}",
        f"  conditions: {conditions}",
        f"  dyads: {dyads}",
        f"  channels: {channels}",
        "",
        "NUMERICAL",
        f"  PLV global range: [{float(plv.min()):.9g}, {float(plv.max()):.9g}]",
        f"  finite: {checks['finite']}",
        f"  bounded [0,1]: {checks['bounded_0_1']}",
        f"  zero-variance primary units across 440 trials: {n_zero_var_units}",
        "",
        "BEHAVIORAL METADATA CONSISTENCY (NO PLV CONDITION CONTRAST)",
        f"  CCC={int(labels.sum())}, Other={int(labels.size-labels.sum())}",
        f"  choices only C/D: {checks['choices_only_C_D']}",
        f"  labels exactly reconstructed from all-three-C: {checks['labels_match_choices']}",
        "",
        "EDGE INDEX",
        f"  edges: {len(edge_i)}",
        f"  exact row-major 19x19 Cartesian mapping: {checks['edge_index_exact']}",
        f"  unique edge pairs: {checks['edge_pairs_unique']}",
        "",
        "DESCRIPTIVE PLV DISTRIBUTIONS — NO BEHAVIORAL SPLIT",
    ]

    for r in rows:
        lines.append(
            f"  {r['task']} {r['band']} {r['dyad']}: "
            f"median={r['median']:.4f}, mean={r['mean']:.4f}, "
            f"p01={r['p01']:.4f}, p99={r['p99']:.4f}, "
            f"range=[{r['min']:.4f},{r['max']:.4f}], "
            f">=0.95={100*r['fraction_ge_0p95']:.4f}%"
        )

    lines.extend(["", "CHECKS"])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "NEXT",
        "  If PASS, validate the block-restricted label-null engine with a small",
        "  permutation run before launching the full 5,000-permutation maxT analysis.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
