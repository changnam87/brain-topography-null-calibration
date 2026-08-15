#!/usr/bin/env python3
"""
Stage 5B — Production imaginary-coherency robustness of the fixed final 3.

Run only after Stage 5A validation PASS.

This is NOT a new discovery analysis. The candidate family is fixed by the PLV
pipeline before any iCOH result is examined.

For each fixed candidate:
  - block-information-weighted CCC-Other absolute imaginary-coherency effect
  - candidate-family label maxT
  - candidate-family temporal-shift maxT
  - candidate-family cross-group partner maxT
  - candidate-family within-triad dyad maxT
  - direction agreement with the primary PLV effect

All four iCOH null families use 1,000 realizations.
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
from bt.io import all_group_labels
from bt.imcoh_robustness import run_fixed_candidate_imcoh


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
            f"Expected 3 final PLV candidates, got {len(fixed)}"
        )

    labels, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )

    result = run_fixed_candidate_imcoh(
        cfg,
        fixed,
        labels,
        n_label=B,
        n_time=B,
        n_partner=B,
        n_dyad=B,
        seed=int(cfg["random_seed"]) + 980000,
    )

    observed = np.asarray(result["observed"], dtype=float)

    rows = []
    for i, c in enumerate(fixed):
        plv = float(c["observed_effect"])
        icoh = float(observed[i])

        p_label = float(result["label"]["p_maxT"][i])
        p_time = float(result["time"]["p_maxT"][i])
        p_partner = float(result["partner"]["p_maxT"][i])
        p_dyad = float(result["dyad"]["p_maxT"][i])

        rows.append({
            "unit_index": int(c["unit_index"]),
            "task": c["task"],
            "band": c["band"],
            "dyad": c["dyad"],
            "ch1": c["ch1"],
            "ch2": c["ch2"],
            "PLV_effect": plv,
            "iCOH_effect": icoh,
            "direction_agrees_with_PLV": bool(
                np.sign(icoh) == np.sign(plv)
            ),
            "p_iCOH_label_maxT": p_label,
            "p_iCOH_time_maxT": p_time,
            "p_iCOH_partner_maxT": p_partner,
            "p_iCOH_dyad_maxT": p_dyad,
            "iCOH_label_pass": bool(p_label < alpha),
            "iCOH_all_four_nulls_pass": bool(
                p_label < alpha
                and p_time < alpha
                and p_partner < alpha
                and p_dyad < alpha
            ),
        })

    out_csv = outdir / "stage5B_imcoh_final3_robustness.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    checks = {
        "fixed_candidate_count_3": len(rows) == 3,
        "all_effects_finite": bool(
            np.isfinite([r["iCOH_effect"] for r in rows]).all()
        ),
        "all_pvalues_valid": bool(
            np.isfinite([
                x
                for r in rows
                for x in (
                    r["p_iCOH_label_maxT"],
                    r["p_iCOH_time_maxT"],
                    r["p_iCOH_partner_maxT"],
                    r["p_iCOH_dyad_maxT"],
                )
            ]).all()
        ),
    }

    passed = all(checks.values())
    out_txt = outdir / "stage5B_imcoh_final3_robustness_summary.txt"

    lines = [
        "Brain Topography Project — Stage 5B Final-3 Imaginary-Coherency Robustness",
        "=" * 86,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed PLV candidates: {len(rows)}",
        f"Realizations per iCOH null family: {B}",
        "",
        "CANDIDATES",
    ]

    for r in rows:
        lines.extend([
            f"  unit {r['unit_index']}: {r['task']} {r['band']} "
            f"{r['dyad']} {r['ch1']}-{r['ch2']}",
            f"    PLV Δ={r['PLV_effect']:.6f}; "
            f"iCOH Δ={r['iCOH_effect']:.6f}; "
            f"direction agreement={r['direction_agrees_with_PLV']}",
            f"    iCOH maxT p: label={r['p_iCOH_label_maxT']:.8f}; "
            f"time={r['p_iCOH_time_maxT']:.8f}; "
            f"partner={r['p_iCOH_partner_maxT']:.8f}; "
            f"within-triad dyad={r['p_iCOH_dyad_maxT']:.8f}",
            f"    iCOH label-pass={r['iCOH_label_pass']}; "
            f"iCOH all-four-null-pass={r['iCOH_all_four_nulls_pass']}",
        ])

    lines.extend([
        "",
        "COUNTS",
        f"  Direction agrees with PLV: "
        f"{sum(r['direction_agrees_with_PLV'] for r in rows)}/3",
        f"  iCOH label-maxT passes: "
        f"{sum(r['iCOH_label_pass'] for r in rows)}/3",
        f"  iCOH all-four-null passes: "
        f"{sum(r['iCOH_all_four_nulls_pass'] for r in rows)}/3",
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
        "INTERPRETATION RULE",
        "  PLV remains the primary metric and defines the fixed candidate family.",
        "  iCOH is a pre-specified non-zero-lag robustness metric only.",
        "  Lack of iCOH significance does not retroactively invalidate the PLV",
        "  candidates, but it limits claims that the effect is robust to zero-lag",
        "  or common-source contributions.",
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
