#!/usr/bin/env python3
"""
Stage 4C2 — Repeated simulation stress test of the proposed within-triad dyad null.

Run ONLY after Stage 4C1 validation PASS.

Key question
------------
Does adding the triad-preserving dyad null materially reduce false positives
under group-specific common drive while retaining useful true-edge sensitivity?

This script is intentionally focused and cheaper than Stage 4B.
"""

from pathlib import Path
import csv
import sys
import time
import numpy as np
from joblib import Parallel, delayed

HERE = Path(__file__).resolve()
PROJECT = next(
    (p for p in HERE.parents if (p / "src" / "bt").is_dir()),
    None,
)
if PROJECT is None:
    raise RuntimeError(f"Cannot locate project root from {HERE}")
sys.path.insert(0, str(PROJECT / "src"))

from bt.config import load_config
from bt.io import all_group_labels
from bt.simulation import (
    balanced_identical_labels,
    flatten_truth,
    full_framework_detection,
    generate_phase_dataset,
    detection_metrics,
)
from bt.dyad_context_null import within_triad_dyad_maxT
from bt.statistics import block_contrast_coefficients


def run_one(cfg, scfg, scenario, labels, coupling, shared, noise, rep, code):
    seed = int(cfg["random_seed"]) + 920000 + code*1000 + rep

    Z, truth = generate_phase_dataset(
        labels, scfg, scenario, coupling, shared, noise, seed
    )

    current = full_framework_detection(
        Z,
        labels,
        alpha=0.05,
        block_size=int(cfg["inference"]["block_size_trials"]),
        B_label=300,
        B_secondary=150,
        seed=seed + 100000,
    )

    cand = np.where(current["label_detect"])[0].tolist()
    augmented = current["final_detect"].copy()

    if cand:
        coeff, _ = block_contrast_coefficients(
            labels,
            block_size=int(cfg["inference"]["block_size_trials"]),
        )
        p_dyad, _ = within_triad_dyad_maxT(
            current["observed"][cand],
            current["plv_cube"],
            cand,
            coeff,
            n_realizations=150,
            seed=seed + 200000,
        )
        for i, u in enumerate(cand):
            augmented[u] = bool(
                current["final_detect"][u] and p_dyad[i] < 0.05
            )

    truth_flat = flatten_truth(truth)

    cur = detection_metrics(truth_flat, current, 0.05)
    aug_result = dict(current)
    aug_result["final_detect"] = augmented
    aug = detection_metrics(truth_flat, aug_result, 0.05)

    return {
        "scenario": scenario,
        "coupling": coupling,
        "shared": shared,
        "noise": noise,
        "rep": rep,
        "n_true": int(truth_flat.sum()),
        "current_any": bool(cur["full_any_detection"]),
        "current_tp": int(cur["full_tp"]),
        "current_fp": int(cur["full_fp"]),
        "current_sensitivity": cur["full_sensitivity"],
        "current_precision": cur["full_precision"],
        "aug_any": bool(aug["full_any_detection"]),
        "aug_tp": int(aug["full_tp"]),
        "aug_fp": int(aug["full_fp"]),
        "aug_sensitivity": aug["full_sensitivity"],
        "aug_precision": aug["full_precision"],
    }


def mean_finite(rows, key):
    a = np.asarray([r[key] for r in rows], dtype=float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


def main():
    cfg = load_config()
    scfg = cfg["ground_truth_simulation"]
    empirical, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    brng = np.random.default_rng(int(cfg["random_seed"]) + 910000)
    balanced = balanced_identical_labels(
        cfg["n_groups"],
        cfg["n_trials"],
        int(cfg["inference"]["block_size_trials"]),
        brng,
    )

    jobs = []
    code = 0

    # Critical null/confound cells.
    for shared in (0.35, 0.60, 0.85):
        code += 1
        jobs.append((code, "group_shared_event", empirical, 0.0, shared, 0.10, 100))

    for shared in (0.35, 0.60):
        code += 1
        jobs.append((code, "global_shared_event", balanced, 0.0, shared, 0.10, 100))

    # True recovery cells around the useful/high-signal region.
    for coupling in (0.60, 0.85):
        for noise in (0.06, 0.10):
            code += 1
            jobs.append((code, "sparse_true", empirical, coupling, 0.0, noise, 75))

    # Hard mixed cells.
    for coupling in (0.60, 0.85):
        for noise in (0.06, 0.10):
            code += 1
            jobs.append((
                code, "sparse_true_plus_group_shared",
                empirical, coupling, 0.60, noise, 75
            ))

    start = time.time()
    all_rows = []

    for j, (code, scenario, labels, coupling, shared, noise, reps) in enumerate(jobs, 1):
        print(
            f"[{j}/{len(jobs)}] {scenario} coupling={coupling} "
            f"shared={shared} noise={noise} reps={reps}",
            flush=True,
        )
        rr = Parallel(n_jobs=4, backend="loky")(
            delayed(run_one)(
                cfg, scfg, scenario, labels, coupling, shared,
                noise, rep, code
            )
            for rep in range(1, reps+1)
        )
        all_rows.extend(rr)

    outdir = Path(cfg["results_root"]) / "simulation"
    raw_csv = outdir / "stage4C2_dyad_null_stress_runs.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Cell summary.
    keys = sorted(set(
        (r["scenario"], r["coupling"], r["shared"], r["noise"])
        for r in all_rows
    ))
    sum_rows = []

    for key in keys:
        rr = [
            r for r in all_rows
            if (r["scenario"], r["coupling"], r["shared"], r["noise"]) == key
        ]
        n_true = int(rr[0]["n_true"])

        row = {
            "scenario": key[0],
            "coupling": key[1],
            "shared": key[2],
            "noise": key[3],
            "n": len(rr),
            "n_true": n_true,
        }

        if n_true == 0:
            row["current_FWER"] = float(np.mean([r["current_any"] for r in rr]))
            row["augmented_FWER"] = float(np.mean([r["aug_any"] for r in rr]))
            row["current_sensitivity"] = np.nan
            row["augmented_sensitivity"] = np.nan
            row["current_precision"] = np.nan
            row["augmented_precision"] = np.nan
        else:
            row["current_FWER"] = np.nan
            row["augmented_FWER"] = np.nan
            row["current_sensitivity"] = mean_finite(rr, "current_sensitivity")
            row["augmented_sensitivity"] = mean_finite(rr, "aug_sensitivity")
            row["current_precision"] = mean_finite(rr, "current_precision")
            row["augmented_precision"] = mean_finite(rr, "aug_precision")

        sum_rows.append(row)

    sum_csv = outdir / "stage4C2_dyad_null_stress_summary.csv"
    with sum_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sum_rows[0].keys()))
        w.writeheader()
        w.writerows(sum_rows)

    txt = outdir / "stage4C2_dyad_null_stress_summary.txt"
    lines = [
        "Brain Topography Project — Stage 4C2 Within-Triad Dyad Null Stress Test",
        "=" * 82,
        f"Elapsed minutes: {(time.time()-start)/60:.2f}",
        "",
        "CELL RESULTS",
    ]
    for r in sum_rows:
        if r["n_true"] == 0:
            lines.append(
                f"  {r['scenario']} shared={r['shared']:.2f}: "
                f"current FWER={r['current_FWER']:.3f}, "
                f"+dyad-null FWER={r['augmented_FWER']:.3f}"
            )
        else:
            lines.append(
                f"  {r['scenario']} coupling={r['coupling']:.2f} "
                f"shared={r['shared']:.2f} noise={r['noise']:.2f}: "
                f"current sens/prec={r['current_sensitivity']:.3f}/"
                f"{r['current_precision']:.3f}; "
                f"+dyad-null sens/prec={r['augmented_sensitivity']:.3f}/"
                f"{r['augmented_precision']:.3f}"
            )

    lines.extend([
        "",
        "DECISION RULE",
        "  Integrate the new null into the empirical framework only if repeated",
        "  simulations show a substantial reduction of group_shared_event false",
        "  positives without collapsing sparse_true recovery to near zero.",
        "  Do not tune thresholds or generator parameters after seeing this test.",
        "",
        f"Raw CSV: {raw_csv}",
        f"Summary CSV: {sum_csv}",
        f"Summary: {txt}",
    ])
    txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
