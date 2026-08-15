#!/usr/bin/env python3
"""
Stage 4A2 — Corrected ground-truth simulation engine validation.

Why this replaces the Stage 4A pass criterion
----------------------------------------------
The original Stage 4A validator incorrectly required raw-PLV AUPRC > 0.50 in
the strong-coupling cases. That is not an engine-validity requirement and
would bias the simulation toward making the uncalibrated raw score look good.

This corrected validator leaves the simulation model unchanged and instead
checks:
  - deterministic reproducibility
  - array shape / unit-modulus / PLV bounds
  - exact truth-map size
  - end-to-end recovery path reaches at least one known true edge in each
    strong sparse-coupling smoke-test case

It also reports TP/FP/FN for naive, label-maxT, and full-framework detections
so the single-run behavior can be inspected without using it as an estimate of
false-positive rate or sensitivity.

No production grid is run here.
"""

from pathlib import Path
import csv
import sys
import time

import numpy as np

HERE = Path(__file__).resolve()
PROJECT = next(
    (p for p in HERE.parents if (p / "src" / "bt").is_dir()),
    None,
)
if PROJECT is None:
    raise RuntimeError(
        "Could not locate project root containing src/bt from "
        f"{HERE}"
    )
sys.path.insert(0, str(PROJECT / "src"))

from bt.config import load_config
from bt.io import all_group_labels
from bt.simulation import (
    balanced_identical_labels,
    compute_plv_cube_from_phase,
    detection_metrics,
    flatten_truth,
    full_framework_detection,
    generate_phase_dataset,
)


def main():
    cfg = load_config()
    scfg = cfg["ground_truth_simulation"]
    outdir = Path(cfg["results_root"]) / "simulation"
    outdir.mkdir(parents=True, exist_ok=True)

    empirical, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    rng = np.random.default_rng(int(cfg["random_seed"]) + 41000)
    balanced = balanced_identical_labels(
        cfg["n_groups"],
        cfg["n_trials"],
        int(cfg["inference"]["block_size_trials"]),
        rng,
    )

    cases = [
        (
            "independent_null",
            empirical,
            0.0,
            0.0,
            float(scfg["phase_noise_sd_validation"]),
        ),
        (
            "global_shared_event",
            balanced,
            0.0,
            float(scfg["shared_strength_validation"]),
            float(scfg["phase_noise_sd_validation"]),
        ),
        (
            "group_shared_event",
            empirical,
            0.0,
            float(scfg["shared_strength_validation"]),
            float(scfg["phase_noise_sd_validation"]),
        ),
        (
            "sparse_true",
            empirical,
            float(scfg["coupling_strength_validation"]),
            0.0,
            float(scfg["phase_noise_sd_validation"]),
        ),
        (
            "sparse_true_plus_group_shared",
            empirical,
            float(scfg["coupling_strength_validation"]),
            float(scfg["shared_strength_validation"]) * 0.5,
            float(scfg["phase_noise_sd_validation"]),
        ),
    ]

    rows = []
    checks = {}
    start = time.time()

    for ci, (scenario, labels, coupling, shared, noise) in enumerate(cases):
        seed = int(cfg["random_seed"]) + 42000 + ci

        Z, truth = generate_phase_dataset(
            labels,
            scfg,
            scenario,
            coupling,
            shared,
            noise,
            seed,
        )

        # Determinism check on the first scenario only.
        if ci == 0:
            Z2, truth2 = generate_phase_dataset(
                labels,
                scfg,
                scenario,
                coupling,
                shared,
                noise,
                seed,
            )
            checks["same_seed_reproducible"] = bool(
                np.array_equal(truth, truth2)
                and np.allclose(Z, Z2)
            )

        P = int(scfg["participants_per_group"])
        C = int(scfg["channels_per_participant"])
        L = int(scfg["n_time_samples"])

        phase_shape_ok = Z.shape == (
            cfg["n_groups"],
            cfg["n_trials"],
            P,
            C,
            L,
        )
        unit_modulus_ok = bool(
            np.max(np.abs(np.abs(Z) - 1.0)) < 2e-5
        )

        plv = compute_plv_cube_from_phase(Z)
        plv_ok = bool(
            plv.shape == (
                cfg["n_groups"],
                cfg["n_trials"],
                3,
                C,
                C,
            )
            and np.isfinite(plv).all()
            and float(plv.min()) >= -1e-7
            and float(plv.max()) <= 1.0 + 1e-7
        )

        result = full_framework_detection(
            Z,
            labels,
            alpha=float(scfg["alpha"]),
            block_size=int(cfg["inference"]["block_size_trials"]),
            B_label=int(scfg["label_permutations_validation"]),
            B_secondary=int(scfg["secondary_nulls_validation"]),
            seed=seed + 1000,
        )
        metrics = detection_metrics(
            flatten_truth(truth),
            result,
            alpha=float(scfg["alpha"]),
        )

        rows.append({
            "scenario": scenario,
            "coupling_strength": coupling,
            "shared_strength": shared,
            "phase_noise_sd": noise,
            "phase_shape_ok": phase_shape_ok,
            "phase_unit_modulus_ok": unit_modulus_ok,
            "plv_shape_range_ok": plv_ok,
            "n_true_edges": int(truth.sum()),
            **metrics,
        })

    elapsed = time.time() - start

    checks["all_phase_shapes_ok"] = all(
        bool(r["phase_shape_ok"]) for r in rows
    )
    checks["all_unit_modulus_ok"] = all(
        bool(r["phase_unit_modulus_ok"]) for r in rows
    )
    checks["all_plv_ok"] = all(
        bool(r["plv_shape_range_ok"]) for r in rows
    )
    checks["null_truth_zero"] = all(
        int(r["n_true_edges"]) == 0
        for r in rows
        if r["scenario"] in {
            "independent_null",
            "global_shared_event",
            "group_shared_event",
        }
    )
    checks["true_scenarios_have_3_true_edges"] = all(
        int(r["n_true_edges"]) == 3
        for r in rows
        if r["scenario"] in {
            "sparse_true",
            "sparse_true_plus_group_shared",
        }
    )

    # End-to-end smoke-test only: strong injected coupling should permit the
    # framework to recover at least one known true edge. We do NOT require a
    # particular AUPRC, sensitivity, or FP count in a single random run.
    strong_rows = [
        r for r in rows
        if r["scenario"] in {
            "sparse_true",
            "sparse_true_plus_group_shared",
        }
    ]
    checks["strong_true_end_to_end_recovery_nonzero"] = all(
        int(r["full_tp"]) >= 1 for r in strong_rows
    )

    all_pass = all(checks.values())

    out_csv = (
        outdir / "stage4A2_simulation_engine_validation.csv"
    )
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_txt = (
        outdir / "stage4A2_simulation_engine_validation_summary.txt"
    )
    lines = [
        "Brain Topography Project — Stage 4A2 Corrected Ground-Truth Validation",
        "=" * 82,
        f"Elapsed seconds: {elapsed:.2f}",
        f"Validation label permutations: "
        f"{scfg['label_permutations_validation']}",
        f"Validation secondary null realizations: "
        f"{scfg['secondary_nulls_validation']}",
        "",
        "WHY STAGE 4A WAS CHECK REQUIRED",
        "  The original validator incorrectly treated raw AUPRC > 0.50 as an",
        "  engine-validity requirement. Raw AUPRC is a scientific performance",
        "  outcome, not a software/simulation validity criterion. The simulation",
        "  model itself is unchanged here.",
        "",
        "VALIDATION CASES — SINGLE-RUN ENGINE DIAGNOSTIC ONLY",
    ]

    for r in rows:
        lines.extend([
            f"  {r['scenario']}: true edges={r['n_true_edges']}",
            f"    naive: TP={r['naive_tp']}, FP={r['naive_fp']}, "
            f"FN={r['naive_fn']}, detected={r['naive_n_detected']}",
            f"    label-maxT: TP={r['label_tp']}, FP={r['label_fp']}, "
            f"FN={r['label_fn']}, detected={r['label_n_detected']}",
            f"    full framework: TP={r['full_tp']}, FP={r['full_fp']}, "
            f"FN={r['full_fn']}, detected={r['full_n_detected']}",
            f"    raw AUPRC={r['raw_auprc']}; "
            f"label-score AUPRC={r['label_score_auprc']}",
        ])

    lines.extend(["", "CHECKS"])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "INTERPRETATION",
        "  Detection counts above are smoke-test observations only. They are",
        "  NOT estimates of FWER, sensitivity, specificity, or precision.",
        "  In particular, false positives in group_shared_event are an intended",
        "  stress-test of a known interpretive boundary, not an engine failure.",
        "",
        "NEXT",
        "  If PASS, freeze a computationally efficient production simulation grid",
        "  and estimate false-positive control and true-edge recovery over repeated",
        "  simulations. Do not tune the generator to improve raw AUPRC.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
