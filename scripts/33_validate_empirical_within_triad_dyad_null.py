#!/usr/bin/env python3
"""
Stage 4D1 — Empirical adapter validation for the within-triad dyad null.

This validates ONLY the empirical-cube adapter on the frozen 7 production
candidates. It uses 50 realizations and must not be interpreted scientifically.

Checks
------
- exact frozen candidate family and metadata
- unit decoding back to condition/dyad/edge is exact
- 50x7 null matrix finite with nonzero SD
- studentized maxT p-values on the 1/51 Monte Carlo grid
- same-triad/same-trial dyad randomization engine completes for all candidates

Outputs
-------
results/nulls/stage4D1_validation_empirical_dyad_null.csv
results/nulls/stage4D1_validation_empirical_dyad_null_summary.txt
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
from bt.statistics import block_contrast_coefficients
from bt.empirical_dyad_null import (
    decode_empirical_unit,
    empirical_within_triad_dyad_null,
    studentized_candidate_family_maxT,
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
    outdir = Path(cfg["results_root"]) / "nulls"
    cube_path = (
        Path(cfg["results_root"])
        / "observed"
        / "trial_connectivity_cube.npz"
    )
    units_csv = outdir / "label_null_plv_units.csv"

    cube = np.load(cube_path, allow_pickle=True)
    X = cube["plv"].astype(np.float32)
    labels = cube["labels"].astype(np.uint8)

    units = read_csv(units_csv)
    cand = [r for r in units if boolstr(r["label_candidate"])]
    if len(cand) != 7:
        raise RuntimeError(f"Expected 7 frozen candidates, got {len(cand)}")

    ids = [int(r["unit_index"]) for r in cand]
    observed = np.array(
        [float(r["observed_effect"]) for r in cand],
        dtype=float,
    )

    coeff, total_w = block_contrast_coefficients(
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    # Metadata decoding audit.
    conditions = [str(x) for x in cube["conditions"].tolist()]
    dyads = [str(x) for x in cube["dyads"].tolist()]
    edge_i = cube["edge_i"].astype(int)
    edge_j = cube["edge_j"].astype(int)
    channels = [str(x) for x in cube["channel_names"].tolist()]

    decode_ok = True
    decoded = []
    for r in cand:
        u = int(r["unit_index"])
        ci, di, ei = decode_empirical_unit(
            u, X.shape[1], X.shape[2], X.shape[4]
        )
        task, band = conditions[ci].split("|")
        rec = {
            "unit_index": u,
            "task": task,
            "band": band,
            "dyad": dyads[di],
            "ch1": channels[edge_i[ei]],
            "ch2": channels[edge_j[ei]],
        }
        decoded.append(rec)
        for key in ("task", "band", "dyad", "ch1", "ch2"):
            if str(rec[key]) != str(r[key]):
                decode_ok = False

    B = 50
    null = empirical_within_triad_dyad_null(
        X,
        ids,
        coeff,
        n_realizations=B,
        seed=int(cfg["random_seed"]) + 940001,
    )
    stats = studentized_candidate_family_maxT(
        observed, null
    )

    p = stats["p_maxT"]
    sd = stats["null_sd"]
    oz = stats["observed_studentized"]

    checks = {
        "candidate_count_7": len(ids) == 7,
        "candidate_metadata_decodes_exactly": decode_ok,
        "information_weight_37p9": abs(total_w - 37.9) < 1e-9,
        "null_shape_50x7": null.shape == (50, 7),
        "null_all_finite": bool(np.isfinite(null).all()),
        "null_sd_positive_finite": bool(
            np.isfinite(sd).all() and np.all(sd > 0)
        ),
        "observed_studentized_finite": bool(np.isfinite(oz).all()),
        "p_finite": bool(np.isfinite(p).all()),
        "p_bounds": bool(
            np.all(p >= 1/51 - 1e-12)
            and np.all(p <= 1 + 1e-12)
        ),
        "p_empirical_grid": grid_ok(p, B),
    }

    rows = []
    for r, d, pv, z, mu, s in zip(
        cand,
        decoded,
        p,
        oz,
        stats["null_mean"],
        stats["null_sd"],
    ):
        rows.append({
            **d,
            "observed_effect": float(r["observed_effect"]),
            "validation_dyad_null_mean": float(mu),
            "validation_dyad_null_sd": float(s),
            "validation_observed_z": float(z),
            "validation_p_dyad_maxT": float(pv),
        })

    out_csv = outdir / "stage4D1_validation_empirical_dyad_null.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    passed = all(checks.values())
    out_txt = (
        outdir
        / "stage4D1_validation_empirical_dyad_null_summary.txt"
    )

    lines = [
        "Brain Topography Project — Stage 4D1 Empirical Dyad-Null Validation",
        "=" * 80,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed candidates: {len(ids)}",
        f"Validation realizations: {B}",
        "",
        "VALIDATION P-VALUES — ENGINE DIAGNOSTIC ONLY",
    ]
    for r in rows:
        lines.append(
            f"  unit {r['unit_index']} {r['task']} {r['band']} "
            f"{r['dyad']} {r['ch1']}-{r['ch2']}: "
            f"z={r['validation_observed_z']:.3f}, "
            f"p_dyad_maxT={r['validation_p_dyad_maxT']:.8f}"
        )

    lines.extend(["", "CHECKS"])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if passed else 'CHECK REQUIRED'}",
        "",
        "IMPORTANT",
        "  These 50-realization p-values are adapter-validation diagnostics only.",
        "  Do not interpret candidate survival scientifically.",
        "",
        "NEXT",
        "  If PASS, run the 1,000-realization empirical within-triad dyad null",
        "  on the same fixed 7 candidates.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
