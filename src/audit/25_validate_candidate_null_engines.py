#!/usr/bin/env python3
"""
Stage 3F — Candidate temporal-shift / partner-shuffle null engine validation.

Purpose
-------
Validate the two secondary null engines on the FIXED production candidate set
from Stage 3D/3E before launching the 1,000-realization production runs.

Validation configuration
------------------------
- fixed candidate family: production maxT-FWER candidates only
- randomized temporal-shift null: configured validation count (default 50)
- randomized partner-shuffle null: configured validation count (default 50)
- candidate-family studentized maxT FWER for each null family
- block-information-weighted statistic is retained exactly

This is an ENGINE VALIDATION run. Its p-values are not scientific results.

Checks
------
- exactly the fixed production candidates are analyzed
- candidate metadata are preserved
- p-values are finite and on the Monte Carlo grid 1/(B+1)
- p-values are bounded [1/(B+1), 1]
- no production candidate is added/dropped/reordered
- validation output is renamed so it cannot be confused with production output

Outputs
-------
results/nulls/stage3F_validation_candidate_nulls_plv.csv
results/nulls/stage3F_candidate_null_validation_summary.txt
"""

from __future__ import annotations

import argparse
import csv
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


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def boolstr(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def grid_ok(values, B, tol=1e-9):
    a = np.asarray(values, dtype=float)
    scaled = a * (B + 1)
    return bool(np.all(np.abs(scaled - np.rint(scaled)) < tol))


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.candidate_nulls import run_candidate_nulls

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    outdir = project / "results" / "nulls"

    units_csv = outdir / "label_null_plv_units.csv"
    if not units_csv.exists():
        print(f"FAIL: missing {units_csv}", file=sys.stderr)
        return 2

    production_units = read_csv(units_csv)
    fixed = [r for r in production_units if boolstr(r["label_candidate"])]
    fixed_ids = [int(r["unit_index"]) for r in fixed]

    if not fixed:
        print("FAIL: no fixed production candidates", file=sys.stderr)
        return 3

    Btime = int(cfg["inference"]["n_time_shift_validation"])
    Bpartner = int(cfg["inference"]["n_partner_shuffle_validation"])

    # Remove stale generic validation output before calling the engine.
    generic = outdir / "candidate_nulls_plv.csv"
    if generic.exists():
        generic.unlink()

    produced = run_candidate_nulls(
        cfg,
        units_csv,
        full=False,
    )

    validation_csv = (
        outdir / "stage3F_validation_candidate_nulls_plv.csv"
    )
    if validation_csv.exists():
        validation_csv.unlink()
    shutil.move(str(produced), str(validation_csv))

    rows = read_csv(validation_csv)
    got_ids = [int(r["unit_index"]) for r in rows]

    ptime = np.array([float(r["p_time_maxT"]) for r in rows])
    ppartner = np.array([float(r["p_partner_maxT"]) for r in rows])

    checks = {
        "fixed_candidate_count_preserved": len(rows) == len(fixed),
        "fixed_candidate_ids_exact_order": got_ids == fixed_ids,
        "time_validation_count_50": Btime == 50,
        "partner_validation_count_50": Bpartner == 50,
        "time_p_finite": bool(np.isfinite(ptime).all()),
        "partner_p_finite": bool(np.isfinite(ppartner).all()),
        "time_p_bounds": bool(
            np.all(ptime >= 1/(Btime+1) - 1e-12)
            and np.all(ptime <= 1 + 1e-12)
        ),
        "partner_p_bounds": bool(
            np.all(ppartner >= 1/(Bpartner+1) - 1e-12)
            and np.all(ppartner <= 1 + 1e-12)
        ),
        "time_p_empirical_grid": grid_ok(ptime, Btime),
        "partner_p_empirical_grid": grid_ok(ppartner, Bpartner),
        "saved_time_counts_exact": all(
            int(r["n_time_shift"]) == Btime for r in rows
        ),
        "saved_partner_counts_exact": all(
            int(r["n_partner_shuffle"]) == Bpartner for r in rows
        ),
    }

    # Metadata preservation.
    fixed_by_id = {int(r["unit_index"]): r for r in fixed}
    metadata_ok = True
    for r in rows:
        u = int(r["unit_index"])
        srcrow = fixed_by_id[u]
        for key in ("task", "band", "dyad", "ch1", "ch2"):
            if r[key] != srcrow[key]:
                metadata_ok = False
    checks["candidate_metadata_preserved"] = metadata_ok

    all_pass = all(checks.values())

    summary = outdir / "stage3F_candidate_null_validation_summary.txt"
    lines = [
        "Brain Topography Project — Stage 3F Candidate-Null Engine Validation",
        "=" * 76,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed production candidate family: {len(fixed)}",
        f"Temporal-shift validation realizations: {Btime}",
        f"Partner-shuffle validation realizations: {Bpartner}",
        "",
        "VALIDATION P-VALUE RANGES — ENGINE DIAGNOSTIC ONLY",
        f"  time maxT: min={float(ptime.min()):.8f}, "
        f"max={float(ptime.max()):.8f}",
        f"  partner maxT: min={float(ppartner.min()):.8f}, "
        f"max={float(ppartner.max()):.8f}",
        f"  time validation passes at alpha=0.05: "
        f"{int(np.sum(ptime < 0.05))}/{len(rows)}",
        f"  partner validation passes at alpha=0.05: "
        f"{int(np.sum(ppartner < 0.05))}/{len(rows)}",
        f"  both validation nulls pass: "
        f"{int(np.sum((ptime < 0.05) & (ppartner < 0.05)))}/{len(rows)}",
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
        "  These 50-realization p-values are validation diagnostics only.",
        "  Do not interpret candidate survival scientifically.",
        "  If PASS, run the frozen 1,000-realization temporal-shift and",
        "  partner-shuffle production candidate nulls.",
        "",
        f"Validation CSV: {validation_csv}",
        f"Summary: {summary}",
    ])

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
