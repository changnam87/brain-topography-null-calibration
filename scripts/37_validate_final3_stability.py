#!/usr/bin/env python3
"""
Stage 6A — Validate final-3 triad-level stability engine.

This stage does NOT discover or remove candidates.  It verifies that the
stability estimator is aligned with the frozen primary statistic and that
LOTO/bootstrap calculations behave deterministically and sensibly for the
three Stage-4D2 PLV candidates.
"""
from __future__ import annotations

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


def main() -> int:
    cfg = load_config()
    results = Path(cfg["results_root"])
    label_npz = results / "nulls" / "label_null_plv.npz"
    stage4d2 = results / "nulls" / "stage4D2_empirical_within_triad_dyad_null.csv"
    outdir = results / "stability"
    outdir.mkdir(parents=True, exist_ok=True)

    fixed, inp = load_final3_inputs(label_npz, stage4d2)
    B = 500
    seed = int(cfg["random_seed"]) + 990100

    a = compute_final3_stability(
        inp["group_effects"], inp["group_information_weights"], inp["observed"], B, seed
    )
    b = compute_final3_stability(
        inp["group_effects"], inp["group_information_weights"], inp["observed"], B, seed
    )

    valid_n = len(a["valid_group_indices"])
    reconstruction_error = np.max(
        np.abs(a["reconstructed_primary"] - inp["observed"])
    )

    checks = {
        "fixed_candidate_count_3": len(fixed) == 3,
        "informative_triad_count_at_least_10": valid_n >= 10,
        "primary_reconstruction_matches": bool(reconstruction_error < 2e-6),
        "loto_shape_correct": a["loto"].shape == (valid_n, 3),
        "bootstrap_shape_correct": a["bootstrap"].shape == (B, 3),
        "all_core_outputs_finite": bool(np.isfinite(np.concatenate([
            a["reconstructed_primary"].ravel(), a["loto"].ravel(),
            a["bootstrap"].ravel(), a["bootstrap_ci_low"].ravel(),
            a["bootstrap_ci_high"].ravel(),
        ])).all()),
        "ci_order_valid": bool(np.all(a["bootstrap_ci_low"] <= a["bootstrap_ci_high"])),
        "deterministic_same_seed": bool(np.array_equal(a["bootstrap"], b["bootstrap"])),
        "candidate_unit_ids_expected": inp["unit_ids"].tolist() == [3801, 4994, 8156],
    }
    passed = all(checks.values())

    out_npz = outdir / "stage6A_final3_stability_validation.npz"
    np.savez_compressed(out_npz, **a, unit_ids=inp["unit_ids"], n_bootstrap=np.array(B))

    out_txt = outdir / "stage6A_final3_stability_validation_summary.txt"
    lines = [
        "Brain Topography Project — Stage 6A Final-3 Stability Validation",
        "=" * 78,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Fixed candidate units: {inp['unit_ids'].tolist()}",
        f"Informative triads: {valid_n}",
        f"Validation bootstrap draws: {B}",
        f"Maximum primary reconstruction error: {reconstruction_error:.10g}",
        "",
        "CANDIDATE PREVIEW (descriptive only)",
    ]
    for k, c in enumerate(fixed):
        lines.append(
            f"  unit {c['unit_index']}: {c['task']} {c['band']} {c['dyad']} "
            f"{c['ch1']}-{c['ch2']} | ΔPLV={inp['observed'][k]:.6f} | "
            f"LOTO same-direction={int(a['loto_same_direction_count'][k])}/{valid_n} | "
            f"bootstrap 95% percentile CI=[{a['bootstrap_ci_low'][k]:.6f}, "
            f"{a['bootstrap_ci_high'][k]:.6f}]"
        )

    lines.extend(["", "CHECKS"])
    for name, ok in checks.items():
        lines.append(f"  {name}: {'PASS' if ok else 'FAIL'}")
    lines.extend([
        "",
        "RESULT",
        f"  {'PASS — production Stage 6B may be run' if passed else 'CHECK REQUIRED — do not run Stage 6B'}",
        "",
        "INTERPRETATION RULE",
        "  Stage 6 is a fixed-family sensitivity/stability analysis, not a new",
        "  discovery or confirmatory candidate-selection stage. LOTO and bootstrap",
        "  preserve the primary block-information-weighted estimator. Bootstrap",
        "  percentile intervals are descriptive because the inferential unit count",
        "  is only 11 triads.",
        "",
        "PROVENANCE",
        f"  Stage 4D2 CSV: {stage4d2}",
        f"  Stage 4D2 SHA256: {sha256(stage4d2)}",
        f"  Label NPZ: {label_npz}",
        f"  Validation NPZ: {out_npz}",
        f"  Summary: {out_txt}",
    ])
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
