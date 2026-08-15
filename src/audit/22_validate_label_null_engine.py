#!/usr/bin/env python3
"""
Stage 3C — Block-restricted label-null / maxT engine validation.

Purpose
-------
Validate the frozen primary inferential engine on the real observed PLV cube
using only the configured validation number of permutations (default 200).

This is an ENGINE VALIDATION run, not the final inferential analysis.

It verifies:
  1) frozen block-information weights and contrast coefficients;
  2) restricted permutations preserve CCC/Other counts inside every 10-trial block;
  3) the primary statistic has the expected algebraic normalization;
  4) the 200-permutation studentized global maxT run completes for all 9,747 units;
  5) all null SDs, studentized statistics, and p-values are finite/valid;
  6) maxT-adjusted p-values are never smaller than unadjusted permutation p-values;
  7) empirical p-values lie on the expected Monte Carlo grid 1/(B+1).

The validation outputs are renamed so they cannot be confused with the later
5,000-permutation production results.

No scientific interpretation of candidate edges should be made from this run.

Outputs
-------
results/nulls/stage3C_validation_label_null_plv.npz
results/nulls/stage3C_validation_label_null_plv_units.csv
results/nulls/stage3C_label_null_validation_summary.txt
"""

from __future__ import annotations

import argparse
import math
import shutil
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


def check_empirical_grid(p: np.ndarray, B: int, tol: float = 1e-10) -> bool:
    p = np.asarray(p, dtype=float)
    scaled = p * (B + 1)
    return bool(np.all(np.abs(scaled - np.rint(scaled)) < tol))


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.null_label import run_label_null
    from bt.statistics import (
        block_contrast_coefficients,
        block_information_weights,
        draw_within_block_permuted_coefficients_for_group,
    )

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    cube = project / "results" / "observed" / "trial_connectivity_cube.npz"
    outdir = project / "results" / "nulls"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / "stage3C_label_null_validation_summary.txt"

    if not cube.exists():
        print(f"FAIL: missing observed cube {cube}", file=sys.stderr)
        return 2

    z = np.load(cube, allow_pickle=True)
    labels = z["labels"].astype(np.uint8)

    B = int(cfg["inference"]["n_label_permutations_validation"])
    block_size = int(cfg["inference"]["block_size_trials"])
    alpha = float(cfg["inference"]["candidate_alpha"])

    checks = {}

    # ------------------------------------------------------------------
    # 1) Frozen observed weights / coefficient algebra
    # ------------------------------------------------------------------
    weights = block_information_weights(labels, block_size=block_size)
    coeff, total_w = block_contrast_coefficients(
        labels, block_size=block_size
    )

    informative = weights > 0
    n_informative = int(informative.sum())

    checks["informative_blocks_23"] = (n_informative == 23)
    checks["total_information_weight_37p9"] = (
        abs(float(total_w) - 37.9) < 1e-9
    )
    checks["coeff_shape"] = coeff.shape == (11, 40)
    checks["coeff_total_sum_zero"] = abs(float(coeff.sum())) < 1e-10
    checks["positive_coeff_sum_one"] = (
        abs(float(coeff[coeff > 0].sum()) - 1.0) < 1e-9
    )
    checks["negative_coeff_sum_minus_one"] = (
        abs(float(coeff[coeff < 0].sum()) + 1.0) < 1e-9
    )

    block_coeff_ok = True
    for g in range(11):
        for b in range(4):
            sl = slice(b * block_size, (b + 1) * block_size)
            y = labels[g, sl]
            c = coeff[g, sl]
            n1 = int((y == 1).sum())
            n0 = int((y == 0).sum())
            if n1 > 0 and n0 > 0:
                if int(np.sum(c > 0)) != n1 or int(np.sum(c < 0)) != n0:
                    block_coeff_ok = False
            else:
                if not np.allclose(c, 0.0):
                    block_coeff_ok = False
    checks["observed_coeff_preserves_block_counts"] = block_coeff_ok

    # ------------------------------------------------------------------
    # 2) Restricted permutation coefficient integrity
    # ------------------------------------------------------------------
    rng = np.random.default_rng(int(cfg["random_seed"]) + 88001)
    perm_coeff_ok = True
    test_draws = 25

    for g in range(11):
        Wg = draw_within_block_permuted_coefficients_for_group(
            labels[g],
            test_draws,
            rng,
            global_total_weight=total_w,
            block_size=block_size,
        )
        if Wg.shape != (test_draws, 40):
            perm_coeff_ok = False
            break

        for r in range(test_draws):
            for b in range(4):
                sl = slice(b * block_size, (b + 1) * block_size)
                y = labels[g, sl]
                c = Wg[r, sl]
                n1 = int((y == 1).sum())
                n0 = int((y == 0).sum())

                if n1 > 0 and n0 > 0:
                    if int(np.sum(c > 0)) != n1:
                        perm_coeff_ok = False
                    if int(np.sum(c < 0)) != n0:
                        perm_coeff_ok = False
                else:
                    if not np.allclose(c, 0.0):
                        perm_coeff_ok = False

    checks["permuted_coeff_preserves_within_block_counts"] = perm_coeff_ok

    # ------------------------------------------------------------------
    # 3) Real 200-permutation full-unit validation
    # ------------------------------------------------------------------
    production_npz = outdir / "label_null_plv.npz"
    production_csv = outdir / "label_null_plv_units.csv"

    # Remove stale validation/production-named outputs before this validation.
    for p in (production_npz, production_csv):
        if p.exists():
            p.unlink()

    run_npz, run_csv = run_label_null(
        cfg,
        cube,
        full=False,
        metric="plv",
    )

    validation_npz = outdir / "stage3C_validation_label_null_plv.npz"
    validation_csv = outdir / "stage3C_validation_label_null_plv_units.csv"

    if validation_npz.exists():
        validation_npz.unlink()
    if validation_csv.exists():
        validation_csv.unlink()

    shutil.move(str(run_npz), str(validation_npz))
    shutil.move(str(run_csv), str(validation_csv))

    q = np.load(validation_npz, allow_pickle=True)

    observed = q["observed"].astype(float)
    obs_z = q["observed_studentized"].astype(float)
    p_unadj = q["p_label_unadjusted"].astype(float)
    p_maxT = q["p_label_maxT"].astype(float)
    candidate = q["candidate"].astype(bool)
    null_mean = q["null_mean"].astype(float)
    null_sd = q["null_sd"].astype(float)
    null_max = q["null_max_abs_studentized"].astype(float)
    nperm_saved = int(np.asarray(q["n_permutations"]).squeeze())

    U = 9 * 3 * 361

    checks["saved_nperm_matches_validation"] = nperm_saved == B
    checks["unit_count_9747"] = len(observed) == U
    checks["observed_finite"] = bool(np.isfinite(observed).all())
    checks["observed_studentized_finite"] = bool(np.isfinite(obs_z).all())
    checks["null_mean_finite"] = bool(np.isfinite(null_mean).all())
    checks["null_sd_finite_positive"] = bool(
        np.isfinite(null_sd).all() and np.all(null_sd > 0)
    )
    checks["null_max_finite_nonnegative"] = bool(
        np.isfinite(null_max).all() and np.all(null_max >= 0)
    )
    checks["p_unadjusted_valid"] = bool(
        np.isfinite(p_unadj).all()
        and np.all(p_unadj >= 1.0 / (B + 1) - 1e-12)
        and np.all(p_unadj <= 1.0 + 1e-12)
    )
    checks["p_maxT_valid"] = bool(
        np.isfinite(p_maxT).all()
        and np.all(p_maxT >= 1.0 / (B + 1) - 1e-12)
        and np.all(p_maxT <= 1.0 + 1e-12)
    )
    checks["maxT_not_smaller_than_unadjusted"] = bool(
        np.all(p_maxT + 1e-12 >= p_unadj)
    )
    checks["p_unadjusted_on_empirical_grid"] = check_empirical_grid(
        p_unadj, B
    )
    checks["p_maxT_on_empirical_grid"] = check_empirical_grid(
        p_maxT, B
    )
    checks["candidate_matches_maxT_alpha"] = bool(
        np.array_equal(candidate, p_maxT < alpha)
    )

    # This count is diagnostic only; do not interpret scientifically.
    n_candidates_validation = int(candidate.sum())
    min_p_unadj = float(np.min(p_unadj))
    min_p_maxT = float(np.min(p_maxT))
    max_null_z_q95 = float(np.percentile(null_max, 95))
    max_null_z_q99 = float(np.percentile(null_max, 99))

    all_pass = all(checks.values())

    lines = [
        "Brain Topography Project — Stage 3C Label-Null Engine Validation",
        "=" * 74,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Validation permutations: {B}",
        "",
        "FROZEN BLOCK STATISTIC",
        f"  Informative triad x block strata: {n_informative}/44",
        f"  Total block information weight: {total_w:.6f}",
        f"  Positive coefficient sum: {coeff[coeff > 0].sum():.12f}",
        f"  Negative coefficient sum: {coeff[coeff < 0].sum():.12f}",
        "",
        "REAL-DATA NULL ENGINE",
        f"  Units: {len(observed)}",
        f"  Minimum unadjusted permutation p: {min_p_unadj:.8f}",
        f"  Minimum maxT-FWER p: {min_p_maxT:.8f}",
        f"  Null max-|studentized statistic| 95th percentile: {max_null_z_q95:.4f}",
        f"  Null max-|studentized statistic| 99th percentile: {max_null_z_q99:.4f}",
        f"  Validation-run maxT candidates at alpha={alpha}: "
        f"{n_candidates_validation}  [ENGINE DIAGNOSTIC ONLY]",
        "",
        "CHECKS",
    ]

    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "IMPORTANT",
        "  The 200-permutation output is for engine validation only.",
        "  Do not interpret or report its candidate count or individual p-values.",
        "  If PASS, run the frozen 5,000-permutation production maxT analysis.",
        "",
        f"Validation NPZ: {validation_npz}",
        f"Validation units CSV: {validation_csv}",
        f"Summary: {summary}",
    ])

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
