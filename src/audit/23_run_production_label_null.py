#!/usr/bin/env python3
"""
Stage 3D — Production primary label-null inference (5,000 permutations).

Purpose
-------
Run the frozen primary PLV inference after the Stage 3C engine validation PASS.

Frozen primary analysis
-----------------------
- Statistic: triad x 10-trial-block information-weighted CCC - Other contrast
- Label null: restricted permutation within each 10-trial block
- Multiple-comparison control: studentized global maxT FWER
- Primary metric: PLV
- Full EEG-QC primary sample: all 440 triadic trial instances

This script calls the already validated bt.null_label.run_label_null(..., full=True)
and then performs production-output integrity checks.

Outputs
-------
results/nulls/label_null_plv.npz
results/nulls/label_null_plv_units.csv
results/nulls/stage3D_production_label_null_summary.txt
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
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


def sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def empirical_grid_ok(p: np.ndarray, B: int, tol: float = 1e-9) -> bool:
    scaled = np.asarray(p, dtype=float) * (B + 1)
    return bool(np.all(np.abs(scaled - np.rint(scaled)) < tol))


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.null_label import run_label_null

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    cube = project / "results" / "observed" / "trial_connectivity_cube.npz"
    outdir = project / "results" / "nulls"
    outdir.mkdir(parents=True, exist_ok=True)
    summary = outdir / "stage3D_production_label_null_summary.txt"

    if not cube.exists():
        print(f"FAIL: missing {cube}", file=sys.stderr)
        return 2

    B = int(cfg["inference"]["n_label_permutations_full"])
    if B != 5000:
        print(
            f"FAIL: frozen production permutation count should be 5000, got {B}",
            file=sys.stderr,
        )
        return 3

    start = time.time()
    out_npz, out_csv = run_label_null(
        cfg,
        cube,
        full=True,
        metric="plv",
    )
    elapsed = time.time() - start

    z = np.load(out_npz, allow_pickle=True)

    observed = z["observed"].astype(float)
    obs_z = z["observed_studentized"].astype(float)
    p_unadj = z["p_label_unadjusted"].astype(float)
    p_maxT = z["p_label_maxT"].astype(float)
    candidate = z["candidate"].astype(bool)
    null_sd = z["null_sd"].astype(float)
    null_max = z["null_max_abs_studentized"].astype(float)

    nperm = int(np.asarray(z["n_permutations"]).squeeze())
    primary_stat = str(np.asarray(z["primary_statistic"]).squeeze())
    mc_method = str(np.asarray(z["multiple_comparison"]).squeeze())

    checks = {
        "nperm_5000": nperm == 5000,
        "units_9747": len(observed) == 9747,
        "observed_finite": bool(np.isfinite(observed).all()),
        "studentized_finite": bool(np.isfinite(obs_z).all()),
        "null_sd_positive_finite": bool(
            np.isfinite(null_sd).all() and np.all(null_sd > 0)
        ),
        "p_unadjusted_valid": bool(
            np.isfinite(p_unadj).all()
            and np.all(p_unadj >= 1/(B+1) - 1e-12)
            and np.all(p_unadj <= 1 + 1e-12)
        ),
        "p_maxT_valid": bool(
            np.isfinite(p_maxT).all()
            and np.all(p_maxT >= 1/(B+1) - 1e-12)
            and np.all(p_maxT <= 1 + 1e-12)
        ),
        "maxT_ge_unadjusted": bool(
            np.all(p_maxT + 1e-12 >= p_unadj)
        ),
        "p_unadjusted_grid": empirical_grid_ok(p_unadj, B),
        "p_maxT_grid": empirical_grid_ok(p_maxT, B),
        "candidate_matches_alpha": bool(
            np.array_equal(
                candidate,
                p_maxT < float(cfg["inference"]["candidate_alpha"])
            )
        ),
        "primary_statistic_exact": (
            primary_stat == cfg["inference"]["primary_statistic"]
        ),
        "multiple_comparison_exact": (
            mc_method == cfg["inference"]["multiple_comparison_primary"]
        ),
    }

    all_pass = all(checks.values())

    q = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    pmax_quant = np.percentile(p_maxT, q)
    z_quant = np.percentile(np.abs(obs_z), q)
    nullmax_quant = np.percentile(null_max, [90, 95, 99, 99.5, 100])

    lines = [
        "Brain Topography Project — Stage 3D Production Label-Null Inference",
        "=" * 76,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Elapsed seconds: {elapsed:.2f}",
        f"Permutations: {nperm}",
        f"Primary statistic: {primary_stat}",
        f"Multiple-comparison method: {mc_method}",
        "",
        "OUTPUT INTEGRITY",
        f"  Units: {len(observed)}",
        f"  Minimum attainable empirical p: {1/(B+1):.8f}",
        f"  Minimum unadjusted p: {float(np.min(p_unadj)):.8f}",
        f"  Minimum maxT-FWER p: {float(np.min(p_maxT)):.8f}",
        f"  maxT-FWER candidates at alpha="
        f"{float(cfg['inference']['candidate_alpha']):.3f}: "
        f"{int(candidate.sum())}",
        "",
        "MAXT-FWER P-VALUE QUANTILES",
    ]

    for qq, vv in zip(q, pmax_quant):
        lines.append(f"  p{qq:>3}: {vv:.8f}")

    lines.extend([
        "",
        "ABS(OBSERVED STUDENTIZED STATISTIC) QUANTILES",
    ])
    for qq, vv in zip(q, z_quant):
        lines.append(f"  p{qq:>3}: {vv:.6f}")

    lines.extend([
        "",
        "NULL MAX-|STUDENTIZED| QUANTILES",
    ])
    for qq, vv in zip([90,95,99,99.5,100], nullmax_quant):
        lines.append(f"  p{qq:>4}: {vv:.6f}")

    lines.extend(["", "CHECKS"])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "FILE PROVENANCE",
        f"  NPZ: {out_npz}",
        f"  NPZ SHA256: {sha256(Path(out_npz))}",
        f"  CSV: {out_csv}",
        f"  CSV SHA256: {sha256(Path(out_csv))}",
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "NEXT",
        "  If PASS, inspect the production candidate set and secondary statistics.",
        "  Do NOT run temporal-shift or partner-shuffle candidate nulls until the",
        "  production candidate table itself has been audited.",
        "",
        f"Summary: {summary}",
    ])

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
