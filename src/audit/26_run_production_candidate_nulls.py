#!/usr/bin/env python3
"""
Stage 3G — Production temporal-shift and partner-shuffle candidate nulls.

Purpose
-------
Run the frozen secondary null analyses on the FIXED production candidate family
after Stage 3F engine validation PASS.

Frozen secondary validation
---------------------------
Candidate family:
  production PLV units that survived Stage 3D label-null global maxT FWER.

Temporal-shift null:
  randomized circular temporal misalignment, with the configured minimum shift.

Partner-shuffle null:
  randomized cross-triad partner reassignment using derangements.

For each null family:
  - 1,000 realizations
  - studentized candidate-family maxT FWER
  - the block-information-weighted primary statistic remains unchanged

A candidate passes secondary validation only if BOTH:
  p_time_maxT < alpha
  p_partner_maxT < alpha

Outputs
-------
results/nulls/candidate_nulls_plv.csv
results/nulls/stage3G_production_candidate_null_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import Counter
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


def empirical_grid_ok(values, B, tol=1e-9):
    a = np.asarray(values, dtype=float)
    scaled = a * (B + 1)
    return bool(np.all(np.abs(scaled - np.rint(scaled)) < tol))


def sha256(path: Path, block_size: int = 1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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

    if len(fixed) != 7:
        print(
            f"FAIL: expected frozen production candidate family of 7, got {len(fixed)}",
            file=sys.stderr,
        )
        return 3

    Btime = int(cfg["inference"]["n_time_shift_full"])
    Bpartner = int(cfg["inference"]["n_partner_shuffle_full"])
    alpha = float(cfg["inference"]["alpha"])

    if Btime != 1000 or Bpartner != 1000:
        print(
            f"FAIL: expected 1000/1000 production realizations, got "
            f"{Btime}/{Bpartner}",
            file=sys.stderr,
        )
        return 4

    generic = outdir / "candidate_nulls_plv.csv"
    if generic.exists():
        generic.unlink()

    start = time.time()
    out_csv = run_candidate_nulls(
        cfg,
        units_csv,
        full=True,
    )
    elapsed = time.time() - start

    rows = read_csv(out_csv)
    got_ids = [int(r["unit_index"]) for r in rows]

    ptime = np.array([float(r["p_time_maxT"]) for r in rows], dtype=float)
    ppartner = np.array([float(r["p_partner_maxT"]) for r in rows], dtype=float)
    secondary = np.array(
        [boolstr(r["secondary_validation_pass"]) for r in rows],
        dtype=bool,
    )

    checks = {
        "fixed_candidate_count_preserved": len(rows) == 7,
        "fixed_candidate_ids_exact_order": got_ids == fixed_ids,
        "time_production_count_1000": Btime == 1000,
        "partner_production_count_1000": Bpartner == 1000,
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
        "time_p_empirical_grid": empirical_grid_ok(ptime, Btime),
        "partner_p_empirical_grid": empirical_grid_ok(ppartner, Bpartner),
        "saved_time_counts_exact": all(
            int(r["n_time_shift"]) == Btime for r in rows
        ),
        "saved_partner_counts_exact": all(
            int(r["n_partner_shuffle"]) == Bpartner for r in rows
        ),
        "secondary_pass_logic_exact": bool(
            np.array_equal(
                secondary,
                (ptime < alpha) & (ppartner < alpha),
            )
        ),
    }

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

    n_time = int(np.sum(ptime < alpha))
    n_partner = int(np.sum(ppartner < alpha))
    n_both = int(np.sum(secondary))

    summary = outdir / "stage3G_production_candidate_null_summary.txt"

    lines = [
        "Brain Topography Project — Stage 3G Production Candidate Nulls",
        "=" * 74,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Elapsed seconds: {elapsed:.2f}",
        f"Fixed production candidate family: {len(rows)}",
        f"Temporal-shift realizations: {Btime}",
        f"Partner-shuffle realizations: {Bpartner}",
        f"Candidate-family alpha: {alpha:.3f}",
        "",
        "SECONDARY NULL RESULTS",
        f"  Temporal-shift maxT passes: {n_time}/{len(rows)}",
        f"  Partner-shuffle maxT passes: {n_partner}/{len(rows)}",
        f"  BOTH secondary nulls pass: {n_both}/{len(rows)}",
        "",
        "CANDIDATES",
    ]

    for r in sorted(rows, key=lambda x: int(x["unit_index"])):
        lines.extend([
            f"  unit {r['unit_index']}: "
            f"{r['task']} {r['band']} {r['dyad']} "
            f"{r['ch1']}-{r['ch2']}",
            f"    primary ΔPLV={float(r['observed_effect']):.6f}; "
            f"label p_maxT={float(r['p_label_maxT']):.8f}",
            f"    time p_maxT={float(r['p_time_maxT']):.8f}; "
            f"partner p_maxT={float(r['p_partner_maxT']):.8f}; "
            f"both-pass={r['secondary_validation_pass']}",
        ])

    lines.extend([
        "",
        "CHECKS",
    ])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "FILE PROVENANCE",
        f"  CSV: {out_csv}",
        f"  CSV SHA256: {sha256(Path(out_csv))}",
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "INTERPRETATION RULE",
        "  The fixed Stage 3D maxT result defines the label-supported candidate set.",
        "  Stage 3G evaluates whether each fixed candidate also survives temporal",
        "  misalignment and partner-identity alternatives. A candidate that fails",
        "  either secondary null should not be described as robustly interaction-",
        "  specific. A candidate that passes both remains a null-calibrated candidate,",
        "  not proof of causal inter-brain coupling.",
        "",
        "NEXT",
        "  If PASS, freeze the final candidate-evidence classes, then proceed to",
        "  ground-truth simulation before any manuscript drafting.",
        "",
        f"Summary: {summary}",
    ])

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
