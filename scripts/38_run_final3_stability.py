#!/usr/bin/env python3
"""
Stage 6B — Production triad-level stability analysis of the fixed final 3.

Run only after Stage 6A validation PASS.

Outputs:
  - one candidate-level summary CSV
  - one triad-level effect CSV
  - one leave-one-triad-out CSV
  - compressed NPZ containing the 10,000 triad-bootstrap draws
  - human-readable summary TXT

No stability result is allowed to redefine the frozen Stage-4D2 candidate set.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
PROJECT = next((p for p in HERE.parents if (p / "src" / "bt").is_dir()), None)
if PROJECT is None:
    raise RuntimeError(f"Could not locate project root from {HERE}")
sys.path.insert(0, str(PROJECT / "src"))

from bt.config import load_config
from bt.final3_stability import compute_final3_stability, load_final3_inputs


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    cfg = load_config()
    results = Path(cfg["results_root"])
    outdir = results / "stability"
    outdir.mkdir(parents=True, exist_ok=True)

    validation = outdir / "stage6A_final3_stability_validation_summary.txt"
    if not validation.exists():
        raise RuntimeError(
            "Stage 6A validation summary not found. Run "
            "python3 scripts/37_validate_final3_stability.py first."
        )
    text = validation.read_text(encoding="utf-8")
    if "PASS — production Stage 6B may be run" not in text:
        raise RuntimeError("Stage 6A did not record a production PASS. Do not run Stage 6B.")

    label_npz = results / "nulls" / "label_null_plv.npz"
    stage4d2 = results / "nulls" / "stage4D2_empirical_within_triad_dyad_null.csv"
    fixed, inp = load_final3_inputs(label_npz, stage4d2)

    B = 10_000
    seed = int(cfg["random_seed"]) + 990200
    s = compute_final3_stability(
        inp["group_effects"], inp["group_information_weights"], inp["observed"], B, seed
    )
    valid_idx = s["valid_group_indices"].astype(int)
    n = len(valid_idx)

    # Candidate-level summary.
    candidate_rows = []
    for k, c in enumerate(fixed):
        influence_g = int(s["influential_group_index"][k])
        candidate_rows.append({
            "unit_index": int(c["unit_index"]),
            "task": c["task"],
            "band": c["band"],
            "dyad": c["dyad"],
            "ch1": c["ch1"],
            "ch2": c["ch2"],
            "observed_PLV_effect": float(inp["observed"][k]),
            "reconstructed_primary_effect": float(s["reconstructed_primary"][k]),
            "informative_triads": n,
            "individual_triads_same_direction": int(np.sum(s["individual_same_direction"][:, k])),
            "individual_triads_same_direction_fraction": float(np.mean(s["individual_same_direction"][:, k])),
            "LOTO_same_direction_count": int(s["loto_same_direction_count"][k]),
            "LOTO_same_direction_fraction": float(s["loto_same_direction_fraction"][k]),
            "LOTO_min_effect": float(s["loto_min"][k]),
            "LOTO_max_effect": float(s["loto_max"][k]),
            "most_influential_omitted_triad": f"G{influence_g + 1:02d}",
            "most_influential_abs_effect_change": float(s["influential_abs_change"][k]),
            "bootstrap_draws": B,
            "bootstrap_median_effect": float(s["bootstrap_median"][k]),
            "bootstrap_95pct_low": float(s["bootstrap_ci_low"][k]),
            "bootstrap_95pct_high": float(s["bootstrap_ci_high"][k]),
            "bootstrap_same_direction_fraction": float(s["bootstrap_same_direction_fraction"][k]),
            "bootstrap_95pct_interval_excludes_zero_descriptive": bool(s["bootstrap_ci_excludes_zero"][k]),
            "equal_triad_full_effect_secondary": float(s["equal_triad_full"][k]),
            "equal_triad_direction_agrees_secondary": bool(s["equal_triad_direction_agrees"][k]),
        })

    candidate_csv = outdir / "stage6B_final3_stability.csv"
    write_csv(candidate_csv, candidate_rows)

    # Triad-level effects.
    triad_rows = []
    for j, g0 in enumerate(valid_idx):
        for k, c in enumerate(fixed):
            triad_rows.append({
                "triad": f"G{g0 + 1:02d}",
                "group_index_zero_based": int(g0),
                "information_weight": float(s["group_information_weights"][j]),
                "unit_index": int(c["unit_index"]),
                "task": c["task"],
                "band": c["band"],
                "dyad": c["dyad"],
                "ch1": c["ch1"],
                "ch2": c["ch2"],
                "triad_effect": float(s["group_effects"][j, k]),
                "same_direction_as_full": bool(s["individual_same_direction"][j, k]),
            })
    triad_csv = outdir / "stage6B_final3_triad_effects.csv"
    write_csv(triad_csv, triad_rows)

    # Leave-one-triad-out effects.
    loto_rows = []
    for j, g0 in enumerate(valid_idx):
        for k, c in enumerate(fixed):
            loto_rows.append({
                "omitted_triad": f"G{g0 + 1:02d}",
                "omitted_group_index_zero_based": int(g0),
                "unit_index": int(c["unit_index"]),
                "task": c["task"],
                "band": c["band"],
                "dyad": c["dyad"],
                "ch1": c["ch1"],
                "ch2": c["ch2"],
                "full_effect": float(inp["observed"][k]),
                "LOTO_effect": float(s["loto"][j, k]),
                "LOTO_minus_full": float(s["loto"][j, k] - inp["observed"][k]),
                "same_direction_as_full": bool(s["loto_same_direction"][j, k]),
            })
    loto_csv = outdir / "stage6B_final3_loto.csv"
    write_csv(loto_csv, loto_rows)

    out_npz = outdir / "stage6B_final3_stability.npz"
    np.savez_compressed(
        out_npz,
        **s,
        unit_ids=inp["unit_ids"],
        n_bootstrap=np.array(B),
        bootstrap_seed=np.array(seed),
    )

    reconstruction_error = float(np.max(np.abs(s["reconstructed_primary"] - inp["observed"])))
    checks = {
        "fixed_candidate_count_3": len(fixed) == 3,
        "candidate_unit_ids_expected": inp["unit_ids"].tolist() == [3801, 4994, 8156],
        "primary_reconstruction_error_lt_2e-6": reconstruction_error < 2e-6,
        "all_loto_finite": bool(np.isfinite(s["loto"]).all()),
        "all_bootstrap_finite": bool(np.isfinite(s["bootstrap"]).all()),
        "all_candidate_rows_written": len(candidate_rows) == 3,
        "bootstrap_draw_count_10000": s["bootstrap"].shape == (B, 3),
    }
    technical_pass = all(checks.values())

    out_txt = outdir / "stage6B_final3_stability_summary.txt"
    lines = [
        "Brain Topography Project — Stage 6B Final-3 Triad Stability",
        "=" * 76,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed PLV candidates: {len(fixed)}",
        f"Informative triads: {n}",
        f"Triad-bootstrap draws: {B}",
        f"Maximum primary reconstruction error: {reconstruction_error:.10g}",
        "",
        "CANDIDATES",
    ]

    for r in candidate_rows:
        lines.extend([
            f"  unit {r['unit_index']}: {r['task']} {r['band']} {r['dyad']} {r['ch1']}-{r['ch2']}",
            f"    full ΔPLV={r['observed_PLV_effect']:.6f}; "
            f"individual triads same direction={r['individual_triads_same_direction']}/{n}",
            f"    LOTO same direction={r['LOTO_same_direction_count']}/{n}; "
            f"range=[{r['LOTO_min_effect']:.6f}, {r['LOTO_max_effect']:.6f}]",
            f"    largest LOTO change when omitting {r['most_influential_omitted_triad']}: "
            f"|Δ change|={r['most_influential_abs_effect_change']:.6f}",
            f"    bootstrap median={r['bootstrap_median_effect']:.6f}; "
            f"95% percentile interval=[{r['bootstrap_95pct_low']:.6f}, {r['bootstrap_95pct_high']:.6f}]; "
            f"same-direction fraction={r['bootstrap_same_direction_fraction']:.4f}",
            f"    equal-triad secondary effect={r['equal_triad_full_effect_secondary']:.6f}; "
            f"direction agrees={r['equal_triad_direction_agrees_secondary']}",
        ])

    lines.extend(["", "TECHNICAL CHECKS"])
    for name, ok in checks.items():
        lines.append(f"  {name}: {'PASS' if ok else 'FAIL'}")

    lines.extend([
        "",
        "RESULT",
        f"  {'PASS' if technical_pass else 'CHECK REQUIRED'}",
        "",
        "INTERPRETATION RULE",
        "  These results quantify dependence on the 11 triads after the final PLV",
        "  family was already frozen. They must not be used to discover a new edge",
        "  or silently drop a candidate. LOTO sign retention and bootstrap intervals",
        "  are sensitivity descriptors. A bootstrap percentile interval excluding",
        "  zero is not a replacement for the Stage-3/4 null-calibrated inference.",
        "",
        "  The equal-triad estimate is a secondary weighting-sensitivity diagnostic",
        "  only. The primary stability calculations preserve the same block-information",
        "  weighting used by the primary PLV effect.",
        "",
        "PROVENANCE",
        f"  Stage 4D2 CSV: {stage4d2}",
        f"  Stage 4D2 SHA256: {sha256(stage4d2)}",
        f"  Candidate CSV: {candidate_csv}",
        f"  Candidate CSV SHA256: {sha256(candidate_csv)}",
        f"  Triad CSV: {triad_csv}",
        f"  LOTO CSV: {loto_csv}",
        f"  NPZ: {out_npz}",
        f"  Summary: {out_txt}",
    ])
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if technical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
