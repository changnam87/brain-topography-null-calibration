#!/usr/bin/env python3
"""
Stage 4B — Production repeated ground-truth simulation.

This is the first simulation stage intended for scientific performance
estimation. It does NOT tune the generator based on Stage 4A/A2 outcomes.

Design
------
Null / confound calibration:
  independent_null:
    3 phase-noise levels x 100 repetitions
  global_shared_event:
    3 shared-drive strengths x 100 repetitions at phase-noise SD=0.10
  group_shared_event:
    3 shared-drive strengths x 100 repetitions at phase-noise SD=0.10

Known-edge recovery:
  sparse_true:
    3 coupling strengths x 3 phase-noise levels x 50 repetitions
  sparse_true_plus_group_shared:
    3 coupling strengths x 3 phase-noise levels x 50 repetitions,
    with fixed moderate group-shared strength=0.60

Total runs = 1,800.

Each run uses the frozen simulation/inference engine:
  - 500 block-restricted label permutations
  - 250 temporal-shift realizations
  - 250 partner-shuffle realizations
  - label global maxT FWER
  - candidate-family maxT for temporal and partner nulls

The script parallelizes repetitions within each cell, checkpoints after every
cell, and can resume after interruption.

Usage
-----
python3 scripts/30_run_groundtruth_simulation_production.py --n-jobs 4

Resume:
python3 scripts/30_run_groundtruth_simulation_production.py --n-jobs 4 --resume

Outputs
-------
results/simulation/stage4B_production_runs.csv
results/simulation/stage4B_production_cell_summary.csv
results/simulation/stage4B_production_scenario_summary.csv
results/simulation/stage4B_production_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed


HERE = Path(__file__).resolve()
PROJECT = next(
    (p for p in HERE.parents if (p / "src" / "bt").is_dir()),
    None,
)
if PROJECT is None:
    raise RuntimeError(
        f"Could not locate project root containing src/bt from {HERE}"
    )
sys.path.insert(0, str(PROJECT / "src"))

from bt.config import load_config, save_config_snapshot
from bt.io import all_group_labels
from bt.simulation import (
    balanced_identical_labels,
    detection_metrics,
    flatten_truth,
    full_framework_detection,
    generate_phase_dataset,
)


@dataclass(frozen=True)
class Cell:
    cell_id: str
    scenario: str
    labels_mode: str
    coupling: float
    shared: float
    noise: float
    reps: int


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=4,
        help="Parallel worker processes within each simulation cell (default 4).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from already checkpointed completed runs.",
    )
    return p.parse_args()


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    den = 1.0 + z*z/n
    center = (phat + z*z/(2*n)) / den
    half = (
        z
        * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
        / den
    )
    return center-half, center+half


def bootstrap_mean_ci(values, rng, B=2000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean
    idx = rng.integers(0, len(x), size=(B, len(x)))
    boot = x[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def f1(tp, fp, fn):
    den = 2*tp + fp + fn
    return (2*tp/den) if den else np.nan


def build_cells(scfg):
    noise_levels = [float(x) for x in scfg["phase_noise_sd_levels_full"]]
    coupling_levels = [float(x) for x in scfg["coupling_strengths_full"]]
    shared_levels = [float(x) for x in scfg["shared_strengths_full"]]

    cells = []

    # Null: 3 noise levels x 100.
    for ni, noise in enumerate(noise_levels):
        cells.append(
            Cell(
                f"N_IND_{ni+1}",
                "independent_null",
                "empirical",
                0.0,
                0.0,
                noise,
                100,
            )
        )

    # Global shared input: 3 strengths at central noise, 100 each.
    central_noise = 0.10
    for si, shared in enumerate(shared_levels):
        cells.append(
            Cell(
                f"N_GLOB_{si+1}",
                "global_shared_event",
                "balanced_identical",
                0.0,
                shared,
                central_noise,
                100,
            )
        )

    # Group-specific common drive: explicit limitation stress test.
    for si, shared in enumerate(shared_levels):
        cells.append(
            Cell(
                f"N_GROUP_{si+1}",
                "group_shared_event",
                "empirical",
                0.0,
                shared,
                central_noise,
                100,
            )
        )

    # Sparse recovery: 3 coupling x 3 noise x 50.
    for ci, coupling in enumerate(coupling_levels):
        for ni, noise in enumerate(noise_levels):
            cells.append(
                Cell(
                    f"R_TRUE_{ci+1}_{ni+1}",
                    "sparse_true",
                    "empirical",
                    coupling,
                    0.0,
                    noise,
                    50,
                )
            )

    # Sparse + fixed moderate group-specific common drive.
    moderate_shared = 0.60
    for ci, coupling in enumerate(coupling_levels):
        for ni, noise in enumerate(noise_levels):
            cells.append(
                Cell(
                    f"R_MIX_{ci+1}_{ni+1}",
                    "sparse_true_plus_group_shared",
                    "empirical",
                    coupling,
                    moderate_shared,
                    noise,
                    50,
                )
            )

    return cells


def read_existing(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_rows(path: Path, rows):
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)


def run_one(
    cfg,
    scfg,
    cell,
    rep,
    labels,
    B_label,
    B_sec,
):
    # Deterministic per-cell/per-repetition seed, independent of parallel order.
    cell_num = sum(ord(c) for c in cell.cell_id)
    seed = (
        int(cfg["random_seed"])
        + 700000
        + cell_num * 1000
        + int(rep)
    )

    Z, truth = generate_phase_dataset(
        labels,
        scfg,
        cell.scenario,
        cell.coupling,
        cell.shared,
        cell.noise,
        seed,
    )

    result = full_framework_detection(
        Z,
        labels,
        alpha=float(scfg["alpha"]),
        block_size=int(cfg["inference"]["block_size_trials"]),
        B_label=B_label,
        B_secondary=B_sec,
        seed=seed + 100000,
    )

    metrics = detection_metrics(
        flatten_truth(truth),
        result,
        alpha=float(scfg["alpha"]),
    )

    row = {
        "cell_id": cell.cell_id,
        "scenario": cell.scenario,
        "labels_mode": cell.labels_mode,
        "coupling_strength": cell.coupling,
        "shared_strength": cell.shared,
        "phase_noise_sd": cell.noise,
        "rep": rep,
        "seed": seed,
        "B_label": B_label,
        "B_secondary": B_sec,
        **metrics,
    }

    for stage in ("naive", "label", "full"):
        row[f"{stage}_f1"] = f1(
            int(row[f"{stage}_tp"]),
            int(row[f"{stage}_fp"]),
            int(row[f"{stage}_fn"]),
        )
        row[f"{stage}_all_true_recovered"] = (
            bool(int(row["n_true_edges"]) > 0)
            and int(row[f"{stage}_fn"]) == 0
        )
        row[f"{stage}_zero_false_positives"] = int(row[f"{stage}_fp"]) == 0
        row[f"{stage}_exact_recovery"] = (
            bool(int(row["n_true_edges"]) > 0)
            and int(row[f"{stage}_fn"]) == 0
            and int(row[f"{stage}_fp"]) == 0
        )

    return row


def summarize_cell(rows, rng):
    n = len(rows)
    scenario = rows[0]["scenario"]
    n_true = int(float(rows[0]["n_true_edges"]))

    out = {
        "cell_id": rows[0]["cell_id"],
        "scenario": scenario,
        "labels_mode": rows[0]["labels_mode"],
        "coupling_strength": float(rows[0]["coupling_strength"]),
        "shared_strength": float(rows[0]["shared_strength"]),
        "phase_noise_sd": float(rows[0]["phase_noise_sd"]),
        "n_repetitions": n,
        "n_true_edges": n_true,
    }

    for stage in ("naive", "label", "full"):
        any_det = np.array(
            [str(r[f"{stage}_any_detection"]).lower() == "true" for r in rows],
            dtype=bool,
        )
        k = int(any_det.sum())
        lo, hi = wilson_ci(k, n)
        out[f"{stage}_any_detection_rate"] = k/n
        out[f"{stage}_any_detection_ci_low"] = lo
        out[f"{stage}_any_detection_ci_high"] = hi

        # For null scenarios, any detection = family-wise false positive.
        if n_true == 0:
            out[f"{stage}_FWER"] = k/n
            out[f"{stage}_FWER_ci_low"] = lo
            out[f"{stage}_FWER_ci_high"] = hi
        else:
            out[f"{stage}_FWER"] = np.nan
            out[f"{stage}_FWER_ci_low"] = np.nan
            out[f"{stage}_FWER_ci_high"] = np.nan

        for field in (
            "sensitivity",
            "precision",
            "fpr",
            "f1",
            "n_detected",
        ):
            vals = [
                float(r[f"{stage}_{field}"])
                for r in rows
                if str(r[f"{stage}_{field}"]).lower() not in {"nan", ""}
            ]
            mean, clo, chi = bootstrap_mean_ci(vals, rng)
            out[f"mean_{stage}_{field}"] = mean
            out[f"mean_{stage}_{field}_ci_low"] = clo
            out[f"mean_{stage}_{field}_ci_high"] = chi

        for field in (
            "all_true_recovered",
            "zero_false_positives",
            "exact_recovery",
        ):
            vals = np.array(
                [str(r[f"{stage}_{field}"]).lower() == "true" for r in rows],
                dtype=bool,
            )
            kk = int(vals.sum())
            lo2, hi2 = wilson_ci(kk, n)
            out[f"{stage}_{field}_rate"] = kk/n
            out[f"{stage}_{field}_ci_low"] = lo2
            out[f"{stage}_{field}_ci_high"] = hi2

    for field in (
        "raw_auprc",
        "label_score_auprc",
        "raw_topk_recall",
    ):
        vals = [
            float(r[field])
            for r in rows
            if str(r[field]).lower() not in {"nan", ""}
        ]
        mean, clo, chi = bootstrap_mean_ci(vals, rng)
        out[f"mean_{field}"] = mean
        out[f"mean_{field}_ci_low"] = clo
        out[f"mean_{field}_ci_high"] = chi

    return out


def aggregate_scenario(cell_summaries, scenario):
    rr = [r for r in cell_summaries if r["scenario"] == scenario]
    if not rr:
        return None

    # Equal-cell aggregate; cell-level summaries are the primary object.
    out = {
        "scenario": scenario,
        "n_cells": len(rr),
        "total_repetitions": int(sum(int(r["n_repetitions"]) for r in rr)),
    }

    for field in (
        "naive_FWER",
        "label_FWER",
        "full_FWER",
        "mean_naive_sensitivity",
        "mean_label_sensitivity",
        "mean_full_sensitivity",
        "mean_naive_precision",
        "mean_label_precision",
        "mean_full_precision",
        "naive_exact_recovery_rate",
        "label_exact_recovery_rate",
        "full_exact_recovery_rate",
        "mean_raw_auprc",
        "mean_label_score_auprc",
        "mean_raw_topk_recall",
    ):
        vals = np.array(
            [float(r[field]) for r in rr if np.isfinite(float(r[field]))],
            dtype=float,
        )
        out[field] = float(np.mean(vals)) if len(vals) else np.nan

    return out


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    scfg = cfg["ground_truth_simulation"]

    outdir = Path(cfg["results_root"]) / "simulation"
    outdir.mkdir(parents=True, exist_ok=True)

    raw_csv = outdir / "stage4B_production_runs.csv"
    cell_csv = outdir / "stage4B_production_cell_summary.csv"
    scen_csv = outdir / "stage4B_production_scenario_summary.csv"
    summary_txt = outdir / "stage4B_production_summary.txt"

    if raw_csv.exists() and not args.resume:
        raise RuntimeError(
            f"{raw_csv} already exists. Use --resume to continue or move/delete "
            "the old file explicitly."
        )

    empirical, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    balanced_rng = np.random.default_rng(
        int(cfg["random_seed"]) + 600001
    )
    balanced = balanced_identical_labels(
        cfg["n_groups"],
        cfg["n_trials"],
        int(cfg["inference"]["block_size_trials"]),
        balanced_rng,
    )

    cells = build_cells(scfg)
    total_planned = sum(c.reps for c in cells)
    if total_planned != 1800:
        raise RuntimeError(f"Expected 1800 runs, got {total_planned}")

    B_label = int(scfg["label_permutations_full"])
    B_sec = int(scfg["secondary_nulls_full"])
    if B_label != 500 or B_sec != 250:
        raise RuntimeError(
            f"Frozen production B should be 500/250, got {B_label}/{B_sec}"
        )

    existing = read_existing(raw_csv) if args.resume else []
    completed = {
        (r["cell_id"], int(r["rep"])) for r in existing
    }

    start = time.time()
    done_before = len(existing)

    print(
        f"Stage 4B: {total_planned} planned runs; "
        f"{done_before} already checkpointed; "
        f"n_jobs={args.n_jobs}",
        flush=True,
    )

    for ci, cell in enumerate(cells, start=1):
        todo = [
            rep
            for rep in range(1, cell.reps + 1)
            if (cell.cell_id, rep) not in completed
        ]

        if not todo:
            print(
                f"[{ci}/{len(cells)}] {cell.cell_id} already complete",
                flush=True,
            )
            continue

        labels = (
            empirical
            if cell.labels_mode == "empirical"
            else balanced
        )

        print(
            f"[{ci}/{len(cells)}] {cell.cell_id} "
            f"{cell.scenario}: coupling={cell.coupling:.2f}, "
            f"shared={cell.shared:.2f}, noise={cell.noise:.3f}; "
            f"runs={len(todo)}",
            flush=True,
        )

        rows = Parallel(
            n_jobs=args.n_jobs,
            backend="loky",
            verbose=5,
        )(
            delayed(run_one)(
                cfg,
                scfg,
                cell,
                rep,
                labels,
                B_label,
                B_sec,
            )
            for rep in todo
        )

        append_rows(raw_csv, rows)
        for r in rows:
            completed.add((r["cell_id"], int(r["rep"])))

        elapsed = time.time() - start
        print(
            f"  checkpointed total={len(completed)}/{total_planned}; "
            f"elapsed={elapsed/60:.1f} min",
            flush=True,
        )

    # Reload full checkpoint for deterministic summaries.
    all_rows = read_existing(raw_csv)
    if len(all_rows) != total_planned:
        raise RuntimeError(
            f"Production incomplete: {len(all_rows)}/{total_planned} runs"
        )

    by_cell = {}
    for r in all_rows:
        by_cell.setdefault(r["cell_id"], []).append(r)

    rng = np.random.default_rng(int(cfg["random_seed"]) + 990000)
    cell_summaries = [
        summarize_cell(by_cell[c.cell_id], rng)
        for c in cells
    ]
    write_csv(cell_csv, cell_summaries)

    scenario_names = [
        "independent_null",
        "global_shared_event",
        "group_shared_event",
        "sparse_true",
        "sparse_true_plus_group_shared",
    ]
    scenario_summaries = [
        aggregate_scenario(cell_summaries, s)
        for s in scenario_names
    ]
    write_csv(scen_csv, scenario_summaries)

    save_config_snapshot(
        cfg, outdir, "stage4B_production_config_used.json"
    )

    elapsed = time.time() - start
    lines = [
        "Brain Topography Project — Stage 4B Production Ground-Truth Simulation",
        "=" * 82,
        f"Total runs: {len(all_rows)}",
        f"Elapsed minutes this invocation: {elapsed/60:.2f}",
        f"Parallel workers: {args.n_jobs}",
        f"Label permutations/run: {B_label}",
        f"Secondary null realizations/run: {B_sec}",
        "",
        "NULL / CONFOUND CALIBRATION CELLS",
    ]

    for r in cell_summaries:
        if int(r["n_true_edges"]) != 0:
            continue
        lines.extend([
            f"  {r['cell_id']} {r['scenario']} "
            f"shared={r['shared_strength']:.2f} "
            f"noise={r['phase_noise_sd']:.3f} n={r['n_repetitions']}:",
            f"    naive FWER={r['naive_FWER']:.3f} "
            f"[{r['naive_FWER_ci_low']:.3f},{r['naive_FWER_ci_high']:.3f}]",
            f"    label FWER={r['label_FWER']:.3f} "
            f"[{r['label_FWER_ci_low']:.3f},{r['label_FWER_ci_high']:.3f}]",
            f"    full FWER={r['full_FWER']:.3f} "
            f"[{r['full_FWER_ci_low']:.3f},{r['full_FWER_ci_high']:.3f}]",
        ])

    lines.extend(["", "KNOWN-EDGE RECOVERY CELLS"])
    for r in cell_summaries:
        if int(r["n_true_edges"]) == 0:
            continue
        lines.extend([
            f"  {r['cell_id']} {r['scenario']} "
            f"coupling={r['coupling_strength']:.2f} "
            f"shared={r['shared_strength']:.2f} "
            f"noise={r['phase_noise_sd']:.3f} n={r['n_repetitions']}:",
            f"    full sensitivity={r['mean_full_sensitivity']:.3f} "
            f"[{r['mean_full_sensitivity_ci_low']:.3f},"
            f"{r['mean_full_sensitivity_ci_high']:.3f}]",
            f"    full precision={r['mean_full_precision']:.3f} "
            f"[{r['mean_full_precision_ci_low']:.3f},"
            f"{r['mean_full_precision_ci_high']:.3f}]",
            f"    full exact-recovery={r['full_exact_recovery_rate']:.3f} "
            f"[{r['full_exact_recovery_ci_low']:.3f},"
            f"{r['full_exact_recovery_ci_high']:.3f}]",
            f"    raw AUPRC={r['mean_raw_auprc']:.3f}; "
            f"raw top-K recall={r['mean_raw_topk_recall']:.3f}",
        ])

    lines.extend([
        "",
        "INTERPRETATION FRAME",
        "  independent_null estimates ordinary false-positive control.",
        "  global_shared_event tests whether the full framework rejects a globally",
        "  shared event-locked drive that can inflate raw synchrony.",
        "  group_shared_event is an explicit limitation stress test: false positives",
        "  here quantify a confound that partner/time nulls may not distinguish from",
        "  interaction-specific coupling.",
        "  sparse_true cells quantify sensitivity/recovery with known edges.",
        "  sparse_true_plus_group_shared quantifies recovery under the hard confound.",
        "",
        "OUTPUTS",
        f"  Raw runs: {raw_csv}",
        f"  Cell summary: {cell_csv}",
        f"  Scenario summary: {scen_csv}",
        f"  Summary: {summary_txt}",
    ])

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
