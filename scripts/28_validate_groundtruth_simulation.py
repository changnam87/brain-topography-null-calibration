#!/usr/bin/env python3
"""
Stage 4A — Ground-truth simulation engine validation.

No manuscript result should be taken from this run.
"""

from pathlib import Path
import csv
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

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

    rows = []
    checks = {}
    start = time.time()

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

    reproducibility_checked = False
    strong_true_ap = []

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

        P = int(scfg["participants_per_group"])
        C = int(scfg["channels_per_participant"])
        L = int(scfg["n_time_samples"])

        shape_ok = Z.shape == (
            cfg["n_groups"],
            cfg["n_trials"],
            P,
            C,
            L,
        )
        modulus_ok = bool(
            np.max(np.abs(np.abs(Z) - 1.0)) < 2e-5
        )
        truth_count = int(truth.sum())

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
            and plv.min() >= -1e-7
            and plv.max() <= 1 + 1e-7
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

        if scenario in ("sparse_true", "sparse_true_plus_group_shared"):
            strong_true_ap.append(metrics["raw_auprc"])

        if not reproducibility_checked:
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
            reproducibility_checked = True

        rows.append({
            "scenario": scenario,
            "coupling_strength": coupling,
            "shared_strength": shared,
            "phase_noise_sd": noise,
            "phase_shape_ok": shape_ok,
            "phase_unit_modulus_ok": modulus_ok,
            "plv_shape_range_ok": plv_ok,
            "n_true_edges": truth_count,
            **metrics,
        })

    elapsed = time.time() - start

    checks["all_phase_shapes_ok"] = all(r["phase_shape_ok"] for r in rows)
    checks["all_unit_modulus_ok"] = all(r["phase_unit_modulus_ok"] for r in rows)
    checks["all_plv_ok"] = all(r["plv_shape_range_ok"] for r in rows)
    checks["null_truth_zero"] = all(
        r["n_true_edges"] == 0
        for r in rows
        if r["scenario"] in (
            "independent_null",
            "global_shared_event",
            "group_shared_event",
        )
    )
    checks["true_scenarios_have_3_true_edges"] = all(
        r["n_true_edges"] == 3
        for r in rows
        if r["scenario"] in (
            "sparse_true",
            "sparse_true_plus_group_shared",
        )
    )
    checks["strong_true_raw_auprc_nontrivial"] = bool(
        len(strong_true_ap) == 2
        and all(np.isfinite(strong_true_ap))
        and min(strong_true_ap) > 0.50
    )

    all_pass = all(checks.values())

    out_csv = outdir / "stage4A_simulation_engine_validation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    out_txt = outdir / "stage4A_simulation_engine_validation_summary.txt"
    lines = [
        "Brain Topography Project — Stage 4A Ground-Truth Simulation Validation",
        "=" * 78,
        f"Elapsed seconds: {elapsed:.2f}",
        f"Phase process: G={cfg['n_groups']}, T={cfg['n_trials']}, "
        f"P={scfg['participants_per_group']}, C={scfg['channels_per_participant']}, "
        f"L={scfg['n_time_samples']}",
        f"Validation label permutations: {scfg['label_permutations_validation']}",
        f"Validation secondary null realizations: {scfg['secondary_nulls_validation']}",
        "",
        "VALIDATION CASES — ENGINE DIAGNOSTIC ONLY",
    ]

    for r in rows:
        lines.append(
            f"  {r['scenario']}: true={r['n_true_edges']}, "
            f"raw AUPRC={r['raw_auprc']}, "
            f"naive detected={r['naive_n_detected']}, "
            f"label detected={r['label_n_detected']}, "
            f"full detected={r['full_n_detected']}"
        )

    lines.extend(["", "CHECKS"])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "IMPORTANT",
        "  This validates the simulation machinery only. The single-run detection",
        "  counts above are not estimates of false-positive rate or sensitivity.",
        "",
        "NEXT",
        "  If PASS, benchmark runtime and freeze the production repetition/grid",
        "  before running the full false-positive / recovery experiment.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
