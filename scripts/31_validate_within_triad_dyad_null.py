#!/usr/bin/env python3
"""
Stage 4C1 — Validate the proposed triad-preserving dyad null.

This is a methodological stress-test only. It does not change the empirical
candidate evidence yet.

The new null randomizes dyad identity WITHIN the same triad and trial, thereby
preserving a group-specific common drive that the existing cross-group partner
shuffle destroys.

Validation checks software/statistical integrity only; single-run performance
is reported but is not used as a pass criterion.
"""

from pathlib import Path
import sys
import time
import numpy as np

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


def one_case(cfg, scfg, scenario, labels, coupling, shared, noise, seed):
    Z, truth = generate_phase_dataset(
        labels, scfg, scenario, coupling, shared, noise, seed
    )

    current = full_framework_detection(
        Z,
        labels,
        alpha=float(scfg["alpha"]),
        block_size=int(cfg["inference"]["block_size_trials"]),
        B_label=120,
        B_secondary=60,
        seed=seed + 1000,
    )

    label_candidates = np.where(current["label_detect"])[0].tolist()
    coeff, _ = block_contrast_coefficients(
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    if label_candidates:
        obs = current["observed"][label_candidates]
        p_dyad, null = within_triad_dyad_maxT(
            obs,
            current["plv_cube"],
            label_candidates,
            coeff,
            n_realizations=60,
            seed=seed + 2000,
        )
        dyad_pass = p_dyad < float(scfg["alpha"])
        augmented = current["final_detect"].copy()
        # Only fixed label candidates can be secondary candidates.
        for i, u in enumerate(label_candidates):
            augmented[u] = bool(
                current["final_detect"][u] and dyad_pass[i]
            )
        null_sd = np.std(null, axis=0, ddof=1)
    else:
        p_dyad = np.array([], dtype=float)
        augmented = current["final_detect"].copy()
        null_sd = np.array([], dtype=float)

    cur_metrics = detection_metrics(
        flatten_truth(truth), current, float(scfg["alpha"])
    )

    # Reuse metrics helper by replacing final detection only.
    augmented_result = dict(current)
    augmented_result["final_detect"] = augmented
    aug_metrics = detection_metrics(
        flatten_truth(truth), augmented_result, float(scfg["alpha"])
    )

    checks = {
        "candidate_count_consistent": (
            len(label_candidates) == int(current["label_detect"].sum())
        ),
        "p_finite": bool(np.isfinite(p_dyad).all()),
        "p_bounds": bool(
            np.all(p_dyad >= 1/61 - 1e-12)
            and np.all(p_dyad <= 1 + 1e-12)
        ) if len(p_dyad) else True,
        "null_sd_positive": bool(
            np.all(np.isfinite(null_sd)) and np.all(null_sd > 0)
        ) if len(null_sd) else True,
        "augmented_subset_current_full": bool(
            np.all(~augmented | current["final_detect"])
        ),
    }

    return {
        "scenario": scenario,
        "true_edges": int(truth.sum()),
        "label_candidates": len(label_candidates),
        "current_full_detected": int(current["final_detect"].sum()),
        "augmented_detected": int(augmented.sum()),
        "current_full_tp": int(cur_metrics["full_tp"]),
        "current_full_fp": int(cur_metrics["full_fp"]),
        "augmented_tp": int(aug_metrics["full_tp"]),
        "augmented_fp": int(aug_metrics["full_fp"]),
        "dyad_p_min": float(np.min(p_dyad)) if len(p_dyad) else np.nan,
        "dyad_p_max": float(np.max(p_dyad)) if len(p_dyad) else np.nan,
        "checks": checks,
    }


def main():
    cfg = load_config()
    scfg = cfg["ground_truth_simulation"]
    empirical, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    rng = np.random.default_rng(int(cfg["random_seed"]) + 880000)
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
            0.10,
        ),
        (
            "global_shared_event",
            balanced,
            0.0,
            0.60,
            0.10,
        ),
        (
            "group_shared_event",
            empirical,
            0.0,
            0.60,
            0.10,
        ),
        (
            "sparse_true",
            empirical,
            0.85,
            0.0,
            0.06,
        ),
        (
            "sparse_true_plus_group_shared",
            empirical,
            0.85,
            0.60,
            0.06,
        ),
    ]

    start = time.time()
    rows = []
    all_checks = []

    for i, case in enumerate(cases):
        row = one_case(
            cfg,
            scfg,
            case[0],
            case[1],
            case[2],
            case[3],
            case[4],
            int(cfg["random_seed"]) + 881000 + i,
        )
        rows.append(row)
        all_checks.extend(row["checks"].values())

    outdir = Path(cfg["results_root"]) / "simulation"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "stage4C1_within_triad_dyad_null_validation_summary.txt"

    lines = [
        "Brain Topography Project — Stage 4C1 Within-Triad Dyad Null Validation",
        "=" * 82,
        f"Elapsed seconds: {time.time()-start:.2f}",
        "",
        "RATIONALE",
        "  Existing cross-group partner shuffling destroys group-specific common",
        "  drive. The proposed null instead randomizes dyad identity while preserving",
        "  the same triad, trial, timing, and channel pair.",
        "",
        "SINGLE-RUN DIAGNOSTICS — NOT PERFORMANCE ESTIMATES",
    ]

    for r in rows:
        lines.extend([
            f"  {r['scenario']}: true={r['true_edges']}, "
            f"label candidates={r['label_candidates']}",
            f"    current full: detected={r['current_full_detected']}, "
            f"TP={r['current_full_tp']}, FP={r['current_full_fp']}",
            f"    + within-triad dyad null: detected={r['augmented_detected']}, "
            f"TP={r['augmented_tp']}, FP={r['augmented_fp']}",
            f"    dyad-null p range=[{r['dyad_p_min']}, {r['dyad_p_max']}]",
        ])
        for k, v in r["checks"].items():
            lines.append(f"      {k}: {'PASS' if v else 'FAIL'}")

    passed = all(all_checks)
    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if passed else 'CHECK REQUIRED'}",
        "",
        "IMPORTANT",
        "  PASS validates only the new null engine. Whether it actually reduces",
        "  group-shared false positives without unacceptable sensitivity loss must",
        "  be estimated over repeated simulations before any empirical use.",
        "",
        f"Summary: {out}",
    ])

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
