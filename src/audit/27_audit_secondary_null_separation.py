#!/usr/bin/env python3
"""
Stage 3H — Secondary-null distribution audit for the fixed 7 candidates.

Why this audit
--------------
Stage 3G returned the Monte Carlo floor p=1/1001 for ALL seven candidates under
both temporal-shift and partner-shuffle maxT tests. That is encouraging, but
before freezing the evidence classes we should verify that the result reflects
clear separation between the observed effects and the underlying null
distributions rather than a scaling/centering artifact.

This script re-generates the SAME configured 1,000 temporal-shift and 1,000
partner-shuffle null distributions for the fixed production candidates and
reports distributional diagnostics. It does not change candidate membership or
perform any new selection.

For each candidate:
  - observed primary Delta PLV
  - label-null mean/SD from Stage 3D
  - temporal-null mean/SD, 95th/99th/max absolute deviation
  - partner-null mean/SD, 95th/99th/max absolute deviation
  - observed standardized separation from each secondary null
  - empirical exceedance count

It also reports correlations among candidate null distributions to help confirm
that candidate-family maxT is operating on non-degenerate null variability.

Outputs
-------
results/nulls/stage3H_secondary_null_distribution_audit.csv
results/nulls/stage3H_secondary_null_distribution_summary.txt
"""

from __future__ import annotations

import argparse
import csv
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


def summarize_null(null: np.ndarray, obs: float):
    null = np.asarray(null, dtype=float)
    mu = float(np.mean(null))
    sd = float(np.std(null, ddof=1))
    dev = np.abs(null - mu)
    obs_dev = abs(obs - mu)
    exceed = int(np.sum(dev >= obs_dev))
    p_two = (1 + exceed) / (len(null) + 1)
    z = (obs - mu) / sd if sd > 0 else np.nan
    return {
        "mean": mu,
        "sd": sd,
        "q95_abs_dev": float(np.percentile(dev, 95)),
        "q99_abs_dev": float(np.percentile(dev, 99)),
        "max_abs_dev": float(np.max(dev)),
        "obs_abs_dev": obs_dev,
        "obs_z": float(z),
        "exceed_count": exceed,
        "p_unadjusted_two_sided": float(p_two),
    }


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.candidate_nulls import (
        _time_null_candidate,
        _partner_null_candidate,
    )
    from bt.io import all_group_labels
    from bt.statistics import block_contrast_coefficients

    cfg = load_config(project / "configs" / "bt_analysis_config.json")
    outdir = project / "results" / "nulls"

    units_csv = outdir / "label_null_plv_units.csv"
    label_npz = outdir / "label_null_plv.npz"

    if not units_csv.exists() or not label_npz.exists():
        print("FAIL: missing Stage 3D outputs", file=sys.stderr)
        return 2

    all_units = read_csv(units_csv)
    candidates = [r for r in all_units if boolstr(r["label_candidate"])]
    if len(candidates) != 7:
        print(
            f"FAIL: expected fixed candidate family of 7, got {len(candidates)}",
            file=sys.stderr,
        )
        return 3

    labels_all, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    coeff, _ = block_contrast_coefficients(
        labels_all,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    Btime = int(cfg["inference"]["n_time_shift_full"])
    Bpartner = int(cfg["inference"]["n_partner_shuffle_full"])

    labelz = np.load(label_npz, allow_pickle=True)
    label_null_mean = labelz["null_mean"].astype(float)
    label_null_sd = labelz["null_sd"].astype(float)

    time_matrix = np.empty((Btime, len(candidates)), dtype=np.float32)
    partner_matrix = np.empty((Bpartner, len(candidates)), dtype=np.float32)

    rows = []

    for k, cand in enumerate(candidates):
        print(
            f"Secondary-null audit {k+1}/{len(candidates)}: "
            f"{cand['task']} {cand['band']} {cand['dyad']} "
            f"{cand['ch1']}-{cand['ch2']}",
            flush=True,
        )
        tn = _time_null_candidate(
            cfg, cand, Btime, labels_all, coeff
        )
        pn = _partner_null_candidate(
            cfg, cand, Bpartner, labels_all, coeff
        )

        time_matrix[:, k] = tn
        partner_matrix[:, k] = pn

        obs = float(cand["observed_effect"])
        u = int(cand["unit_index"])

        ts = summarize_null(tn, obs)
        ps = summarize_null(pn, obs)

        label_mu = float(label_null_mean[u])
        label_sd = float(label_null_sd[u])
        label_z = (
            (obs - label_mu) / label_sd
            if label_sd > 0 else np.nan
        )

        rows.append({
            "unit_index": u,
            "task": cand["task"],
            "band": cand["band"],
            "dyad": cand["dyad"],
            "ch1": cand["ch1"],
            "ch2": cand["ch2"],
            "observed_effect": obs,
            "label_null_mean": label_mu,
            "label_null_sd": label_sd,
            "label_null_obs_z": float(label_z),

            "time_null_mean": ts["mean"],
            "time_null_sd": ts["sd"],
            "time_null_q95_abs_dev": ts["q95_abs_dev"],
            "time_null_q99_abs_dev": ts["q99_abs_dev"],
            "time_null_max_abs_dev": ts["max_abs_dev"],
            "time_obs_abs_dev": ts["obs_abs_dev"],
            "time_obs_z": ts["obs_z"],
            "time_exceed_count": ts["exceed_count"],
            "time_p_unadjusted_two_sided": ts["p_unadjusted_two_sided"],

            "partner_null_mean": ps["mean"],
            "partner_null_sd": ps["sd"],
            "partner_null_q95_abs_dev": ps["q95_abs_dev"],
            "partner_null_q99_abs_dev": ps["q99_abs_dev"],
            "partner_null_max_abs_dev": ps["max_abs_dev"],
            "partner_obs_abs_dev": ps["obs_abs_dev"],
            "partner_obs_z": ps["obs_z"],
            "partner_exceed_count": ps["exceed_count"],
            "partner_p_unadjusted_two_sided": ps["p_unadjusted_two_sided"],
        })

    out_csv = outdir / "stage3H_secondary_null_distribution_audit.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Pairwise null correlations across the fixed candidate family.
    def offdiag_corr(mat):
        c = np.corrcoef(mat, rowvar=False)
        vals = c[np.triu_indices_from(c, k=1)]
        return (
            float(np.min(vals)),
            float(np.median(vals)),
            float(np.max(vals)),
        )

    t_corr = offdiag_corr(time_matrix)
    p_corr = offdiag_corr(partner_matrix)

    checks = {
        "seven_candidates": len(rows) == 7,
        "time_null_all_finite": bool(np.isfinite(time_matrix).all()),
        "partner_null_all_finite": bool(np.isfinite(partner_matrix).all()),
        "time_null_all_nonzero_sd": bool(
            np.all(np.std(time_matrix, axis=0, ddof=1) > 0)
        ),
        "partner_null_all_nonzero_sd": bool(
            np.all(np.std(partner_matrix, axis=0, ddof=1) > 0)
        ),
        "time_all_zero_exceedances": all(
            int(r["time_exceed_count"]) == 0 for r in rows
        ),
        "partner_all_zero_exceedances": all(
            int(r["partner_exceed_count"]) == 0 for r in rows
        ),
        "time_observed_beyond_null_max_abs_dev": all(
            float(r["time_obs_abs_dev"]) > float(r["time_null_max_abs_dev"])
            for r in rows
        ),
        "partner_observed_beyond_null_max_abs_dev": all(
            float(r["partner_obs_abs_dev"]) > float(r["partner_null_max_abs_dev"])
            for r in rows
        ),
    }

    all_pass = all(checks.values())

    summary = (
        outdir / "stage3H_secondary_null_distribution_summary.txt"
    )
    lines = [
        "Brain Topography Project — Stage 3H Secondary-Null Distribution Audit",
        "=" * 78,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed candidates: {len(rows)}",
        f"Temporal-shift realizations: {Btime}",
        f"Partner-shuffle realizations: {Bpartner}",
        "",
        "CANDIDATE DISTRIBUTIONAL SEPARATION",
    ]

    for r in rows:
        lines.extend([
            f"  unit {r['unit_index']}: "
            f"{r['task']} {r['band']} {r['dyad']} "
            f"{r['ch1']}-{r['ch2']}",
            f"    observed ΔPLV={r['observed_effect']:.6f}",
            f"    label null: mean={r['label_null_mean']:.6f}, "
            f"SD={r['label_null_sd']:.6f}, "
            f"obs z={r['label_null_obs_z']:.3f}",
            f"    time null: mean={r['time_null_mean']:.6f}, "
            f"SD={r['time_null_sd']:.6f}, "
            f"99% |dev|={r['time_null_q99_abs_dev']:.6f}, "
            f"max |dev|={r['time_null_max_abs_dev']:.6f}, "
            f"obs |dev|={r['time_obs_abs_dev']:.6f}, "
            f"obs z={r['time_obs_z']:.3f}, "
            f"exceed={r['time_exceed_count']}/{Btime}",
            f"    partner null: mean={r['partner_null_mean']:.6f}, "
            f"SD={r['partner_null_sd']:.6f}, "
            f"99% |dev|={r['partner_null_q99_abs_dev']:.6f}, "
            f"max |dev|={r['partner_null_max_abs_dev']:.6f}, "
            f"obs |dev|={r['partner_obs_abs_dev']:.6f}, "
            f"obs z={r['partner_obs_z']:.3f}, "
            f"exceed={r['partner_exceed_count']}/{Bpartner}",
        ])

    lines.extend([
        "",
        "NULL CORRELATION ACROSS CANDIDATES",
        f"  temporal off-diagonal correlation: "
        f"min={t_corr[0]:.3f}, median={t_corr[1]:.3f}, max={t_corr[2]:.3f}",
        f"  partner off-diagonal correlation: "
        f"min={p_corr[0]:.3f}, median={p_corr[1]:.3f}, max={p_corr[2]:.3f}",
        "",
        "CHECKS",
    ])

    for k, v in checks.items():
        lines.append(f"  {k}: {'PASS' if v else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if all_pass else 'CHECK REQUIRED'}",
        "",
        "INTERPRETATION",
        "  PASS means the Monte Carlo floor p-values from Stage 3G reflect",
        "  genuine observed-vs-null distributional separation for all fixed",
        "  candidates under the implemented temporal and partner alternatives.",
        "  It does not establish causal neural coupling.",
        "",
        "NEXT",
        "  If PASS, freeze all 7 as robust null-calibrated candidates and move",
        "  to ground-truth simulation of false-positive control and edge recovery.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {summary}",
    ])

    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
