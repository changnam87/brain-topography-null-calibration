#!/usr/bin/env python3
"""
Stage 5A — Validate absolute imaginary-coherency robustness engine.

No new candidate discovery is performed.

The fixed candidate family is the 3 PLV candidates that passed all four
empirical null layers in Stage 4D2.

Validation:
  - synthetic zero-lag iCOH ~ 0
  - synthetic pi/2 phase-lag iCOH ~ 1
  - amplitude scaling leaves iCOH unchanged
  - exact final-3 candidate metadata
  - empirical trial values finite and in [0,1]
  - 40-realization validation of label/time/partner/within-triad-dyad nulls
  - p-values finite, bounded, and on 1/41 grid

Validation p-values are diagnostics only.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
PROJECT = next(
    (p for p in HERE.parents if (p / "src" / "bt").is_dir()),
    None,
)
if PROJECT is None:
    raise RuntimeError(f"Could not locate project root from {HERE}")
sys.path.insert(0, str(PROJECT / "src"))

from bt.config import load_config
from bt.io import all_group_labels
from bt.imcoh_robustness import (
    abs_imag_coherency_trials,
    run_fixed_candidate_imcoh,
)


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def boolstr(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def grid_ok(p, B, tol=1e-9):
    p = np.asarray(p, dtype=float)
    return bool(
        np.all(
            np.abs(p * (B + 1) - np.rint(p * (B + 1))) < tol
        )
    )


def main():
    cfg = load_config()
    outdir = Path(cfg["results_root"]) / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)

    stage4d2 = (
        Path(cfg["results_root"])
        / "nulls"
        / "stage4D2_empirical_within_triad_dyad_null.csv"
    )
    rows4 = read_csv(stage4d2)
    fixed = [
        r for r in rows4
        if boolstr(r["all_four_null_layers_pass"])
    ]
    if len(fixed) != 3:
        raise RuntimeError(
            f"Expected 3 final all-four-layer PLV candidates, got {len(fixed)}"
        )

    # Synthetic metric checks.
    L = 600
    t = np.arange(L) / 300.0
    base = np.exp(1j * 2*np.pi*10*t)[None, :]
    same = base.copy()
    lag90 = base * np.exp(-1j * np.pi/2)

    zero = float(abs_imag_coherency_trials(base, same)[0])
    ninety = float(abs_imag_coherency_trials(base, lag90)[0])
    scaled = float(
        abs_imag_coherency_trials(7.0*base, 0.2*lag90)[0]
    )

    labels, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )

    B = 40
    result = run_fixed_candidate_imcoh(
        cfg,
        fixed,
        labels,
        n_label=B,
        n_time=B,
        n_partner=B,
        n_dyad=B,
        seed=int(cfg["random_seed"]) + 970000,
    )

    X = result["trial_values"]
    observed = np.asarray(result["observed"], dtype=float)

    checks = {
        "fixed_candidate_count_3": len(fixed) == 3,
        "synthetic_zero_lag_near_zero": abs(zero) < 1e-6,
        "synthetic_90deg_near_one": abs(ninety - 1.0) < 1e-6,
        "synthetic_amplitude_scaling_invariant": abs(scaled - 1.0) < 1e-6,
        "trial_values_shape_11x40x3": X.shape == (11, 40, 3),
        "trial_values_finite": bool(np.isfinite(X).all()),
        "trial_values_bounded_0_1": bool(
            float(X.min()) >= -1e-7
            and float(X.max()) <= 1.0 + 1e-7
        ),
        "observed_finite": bool(np.isfinite(observed).all()),
        "information_weight_37p9": abs(
            float(result["total_weight"]) - 37.9
        ) < 1e-9,
    }

    for name in ("label", "time", "partner", "dyad"):
        p = result[name]["p_maxT"]
        sd = result[name]["null_sd"]
        checks[f"{name}_p_finite"] = bool(np.isfinite(p).all())
        checks[f"{name}_p_bounds"] = bool(
            np.all(p >= 1/(B+1) - 1e-12)
            and np.all(p <= 1 + 1e-12)
        )
        checks[f"{name}_p_grid"] = grid_ok(p, B)
        checks[f"{name}_null_sd_positive"] = bool(
            np.isfinite(sd).all() and np.all(sd > 0)
        )

    out_rows = []
    for i, c in enumerate(fixed):
        out_rows.append({
            "unit_index": int(c["unit_index"]),
            "task": c["task"],
            "band": c["band"],
            "dyad": c["dyad"],
            "ch1": c["ch1"],
            "ch2": c["ch2"],
            "PLV_effect": float(c["observed_effect"]),
            "iCOH_effect": float(observed[i]),
            "validation_p_label_maxT": float(
                result["label"]["p_maxT"][i]
            ),
            "validation_p_time_maxT": float(
                result["time"]["p_maxT"][i]
            ),
            "validation_p_partner_maxT": float(
                result["partner"]["p_maxT"][i]
            ),
            "validation_p_dyad_maxT": float(
                result["dyad"]["p_maxT"][i]
            ),
        })

    out_csv = outdir / "stage5A_imcoh_validation.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    passed = all(checks.values())
    out_txt = outdir / "stage5A_imcoh_validation_summary.txt"

    lines = [
        "Brain Topography Project — Stage 5A Imaginary-Coherency Validation",
        "=" * 80,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "METRIC DEFINITION",
        "  Absolute imaginary part of narrow-band complex coherency:",
        "  |Im(sum z1*conj(z2) / sqrt(sum|z1|^2 sum|z2|^2))|.",
        "  Analytic amplitude is preserved; this is not phase-only PLV.",
        "",
        "SYNTHETIC CHECKS",
        f"  zero-lag iCOH={zero:.8f}",
        f"  90-degree-lag iCOH={ninety:.8f}",
        f"  scaled 90-degree-lag iCOH={scaled:.8f}",
        "",
        "FINAL-3 FIXED CANDIDATES — VALIDATION P-VALUES ARE DIAGNOSTIC ONLY",
    ]

    for r in out_rows:
        lines.extend([
            f"  unit {r['unit_index']}: {r['task']} {r['band']} "
            f"{r['dyad']} {r['ch1']}-{r['ch2']}",
            f"    PLV Δ={r['PLV_effect']:.6f}; "
            f"iCOH Δ={r['iCOH_effect']:.6f}",
            f"    validation maxT p: label={r['validation_p_label_maxT']:.8f}, "
            f"time={r['validation_p_time_maxT']:.8f}, "
            f"partner={r['validation_p_partner_maxT']:.8f}, "
            f"dyad={r['validation_p_dyad_maxT']:.8f}",
        ])

    lines.extend(["", "CHECKS"])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if passed else 'CHECK REQUIRED'}",
        "",
        "IMPORTANT",
        "  The 40-realization null p-values are engine-validation diagnostics.",
        "  Do not use them to judge imaginary-coherency robustness.",
        "",
        "NEXT",
        "  If PASS, run Stage 5B with 1,000 realizations for each of the four",
        "  fixed-candidate null families. No additional discovery analysis is needed.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
