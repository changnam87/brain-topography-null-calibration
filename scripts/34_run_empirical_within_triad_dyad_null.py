#!/usr/bin/env python3
"""
Stage 4D2 — Production empirical within-triad dyad null.

Run only after Stage 4D1 validation PASS.

Fixed family:
  the 7 production label-maxT candidates from Stage 3D.

Inference:
  1,000 triad-preserving dyad-identity randomizations
  + candidate-family studentized maxT FWER.

The result is an additional robustness layer. It does not redefine the original
Stage 3D label-supported candidate set.
"""

from __future__ import annotations

import csv
import hashlib
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


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    cfg = load_config()
    alpha = float(cfg["inference"]["alpha"])
    B = 1000

    outdir = Path(cfg["results_root"]) / "nulls"
    cube_path = (
        Path(cfg["results_root"])
        / "observed"
        / "trial_connectivity_cube.npz"
    )
    units_csv = outdir / "label_null_plv_units.csv"
    stage3g_csv = outdir / "candidate_nulls_plv.csv"

    cube = np.load(cube_path, allow_pickle=True)
    X = cube["plv"].astype(np.float32)
    labels = cube["labels"].astype(np.uint8)

    units = read_csv(units_csv)
    cand = [r for r in units if boolstr(r["label_candidate"])]
    if len(cand) != 7:
        raise RuntimeError(f"Expected 7 fixed candidates, got {len(cand)}")

    ids = [int(r["unit_index"]) for r in cand]
    observed = np.array(
        [float(r["observed_effect"]) for r in cand], dtype=float
    )

    coeff, _ = block_contrast_coefficients(
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    null = empirical_within_triad_dyad_null(
        X,
        ids,
        coeff,
        n_realizations=B,
        seed=int(cfg["random_seed"]) + 950001,
    )
    stats = studentized_candidate_family_maxT(observed, null)

    p = stats["p_maxT"]
    pass_dyad = p < alpha

    # Merge the already frozen Stage 3G secondary-null evidence if available.
    stage3g = {}
    if stage3g_csv.exists():
        stage3g = {
            int(r["unit_index"]): r for r in read_csv(stage3g_csv)
        }

    conditions = [str(x) for x in cube["conditions"].tolist()]
    dyads = [str(x) for x in cube["dyads"].tolist()]
    edge_i = cube["edge_i"].astype(int)
    edge_j = cube["edge_j"].astype(int)
    channels = [str(x) for x in cube["channel_names"].tolist()]

    rows = []
    for i, r in enumerate(cand):
        u = int(r["unit_index"])
        ci, di, ei = decode_empirical_unit(
            u, X.shape[1], X.shape[2], X.shape[4]
        )
        task, band = conditions[ci].split("|")

        g = stage3g.get(u, {})
        time_p = float(g["p_time_maxT"]) if g else np.nan
        partner_p = float(g["p_partner_maxT"]) if g else np.nan
        prior_both = (
            boolstr(g["secondary_validation_pass"])
            if g else False
        )

        rows.append({
            "unit_index": u,
            "task": task,
            "band": band,
            "dyad": dyads[di],
            "ch1": channels[edge_i[ei]],
            "ch2": channels[edge_j[ei]],
            "observed_effect": float(r["observed_effect"]),
            "p_label_maxT": float(r["p_label_maxT"]),
            "p_time_maxT": time_p,
            "p_crossgroup_partner_maxT": partner_p,
            "prior_time_partner_pass": prior_both,
            "dyad_null_mean": float(stats["null_mean"][i]),
            "dyad_null_sd": float(stats["null_sd"][i]),
            "dyad_observed_studentized": float(
                stats["observed_studentized"][i]
            ),
            "p_within_triad_dyad_maxT": float(p[i]),
            "within_triad_dyad_pass": bool(pass_dyad[i]),
            "all_four_null_layers_pass": bool(
                prior_both and pass_dyad[i]
            ),
        })

    out_csv = outdir / "stage4D2_empirical_within_triad_dyad_null.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    checks = {
        "candidate_count_7": len(rows) == 7,
        "null_shape_1000x7": null.shape == (1000, 7),
        "null_all_finite": bool(np.isfinite(null).all()),
        "null_sd_positive": bool(
            np.isfinite(stats["null_sd"]).all()
            and np.all(stats["null_sd"] > 0)
        ),
        "p_valid": bool(
            np.isfinite(p).all()
            and np.all(p >= 1/1001 - 1e-12)
            and np.all(p <= 1 + 1e-12)
        ),
    }
    passed = all(checks.values())

    out_txt = (
        outdir
        / "stage4D2_empirical_within_triad_dyad_null_summary.txt"
    )
    lines = [
        "Brain Topography Project — Stage 4D2 Empirical Within-Triad Dyad Null",
        "=" * 82,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed label-supported candidates: {len(rows)}",
        f"Dyad-null realizations: {B}",
        f"Candidate-family alpha: {alpha:.3f}",
        "",
        "CANDIDATES",
    ]

    for r in rows:
        lines.extend([
            f"  unit {r['unit_index']}: {r['task']} {r['band']} "
            f"{r['dyad']} {r['ch1']}-{r['ch2']}",
            f"    ΔPLV={r['observed_effect']:.6f}; "
            f"label={r['p_label_maxT']:.8f}; "
            f"time={r['p_time_maxT']:.8f}; "
            f"cross-group partner={r['p_crossgroup_partner_maxT']:.8f}",
            f"    within-triad dyad p_maxT="
            f"{r['p_within_triad_dyad_maxT']:.8f}; "
            f"pass={r['within_triad_dyad_pass']}; "
            f"all-four-layers-pass={r['all_four_null_layers_pass']}",
        ])

    lines.extend([
        "",
        "COUNTS",
        f"  Within-triad dyad-null passes: "
        f"{sum(r['within_triad_dyad_pass'] for r in rows)}/7",
        f"  All four null layers pass: "
        f"{sum(r['all_four_null_layers_pass'] for r in rows)}/7",
        "",
        "CHECKS",
    ])
    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if passed else 'CHECK REQUIRED'}",
        "",
        "INTERPRETATION",
        "  Stage 3D still defines the fixed label-supported candidate family.",
        "  The within-triad dyad null is an additional common-drive robustness",
        "  diagnostic justified by the frozen Stage 4B/4C simulation evidence.",
        "  Passing all layers does not establish causal neural coupling.",
        "",
        "FILE PROVENANCE",
        f"  CSV: {out_csv}",
        f"  CSV SHA256: {sha256(out_csv)}",
        "",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
