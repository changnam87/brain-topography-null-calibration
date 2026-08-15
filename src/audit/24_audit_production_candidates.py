#!/usr/bin/env python3
"""
Stage 3E — Production candidate audit + pre-null robustness diagnostics.

Purpose
-------
Audit the 7 (or however many) primary label-null candidates that survived the
frozen 5,000-permutation studentized global maxT FWER analysis BEFORE running
temporal-shift or partner-shuffle nulls.

This script does NOT perform any new candidate selection.

For each production candidate it reports:
  - task / band / dyad / channel-pair metadata
  - primary block-adjusted PLV effect and maxT p-value
  - unblocked triad-information-weighted effect
  - equal-triad effect
  - frozen EEG-artifact-sensitivity effect
  - direction agreement across these descriptive robustness estimators
  - leave-one-triad-out (LOTO) primary-effect range and sign consistency
  - dominant triad contribution
  - number of informative 10-trial blocks whose effect has the primary sign

The purpose is to detect candidates driven by one triad, one estimator choice,
or the frozen artifact-QC sensitivity mask before secondary null testing.

No manuscript claims should be made from this audit alone.

Outputs
-------
results/nulls/stage3E_candidate_audit.csv
results/nulls/stage3E_candidate_audit_summary.txt
"""

from __future__ import annotations

import argparse
import csv
import sys
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


def sgn(x: float, tol: float = 1e-12) -> int:
    if x > tol:
        return 1
    if x < -tol:
        return -1
    return 0


def fmt_sign(x: float) -> str:
    return "CCC>Other" if x > 0 else ("CCC<Other" if x < 0 else "zero")


def main():
    args = parse_args()
    project = args.project.expanduser().resolve()
    src = project / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from bt.config import load_config
    from bt.statistics import (
        apply_trial_coefficients,
        block_contrast_coefficients,
        equal_triad_coefficients,
        triad_information_coefficients,
    )

    cfg = load_config(project / "configs" / "bt_analysis_config.json")

    cube_path = project / "results" / "observed" / "trial_connectivity_cube.npz"
    label_npz = project / "results" / "nulls" / "label_null_plv.npz"
    units_csv = project / "results" / "nulls" / "label_null_plv_units.csv"
    mask_path = (
        project / "results" / "preprocessing"
        / "stage2B4_sensitivity_mask.npz"
    )
    outdir = project / "results" / "nulls"
    out_csv = outdir / "stage3E_candidate_audit.csv"
    out_txt = outdir / "stage3E_candidate_audit_summary.txt"

    for p in (cube_path, label_npz, units_csv, mask_path):
        if not p.exists():
            print(f"FAIL: missing {p}", file=sys.stderr)
            return 2

    cube = np.load(cube_path, allow_pickle=True)
    label = np.load(label_npz, allow_pickle=True)
    maskz = np.load(mask_path, allow_pickle=True)

    X5 = cube["plv"].astype(np.float64)  # G,C,D,T,E
    labels = cube["labels"].astype(np.uint8)
    G, C, D, T, E = X5.shape
    U = C * D * E
    X = X5.transpose(0, 3, 1, 2, 4).reshape(G, T, U)

    unit_rows = read_csv(units_csv)
    candidates = [r for r in unit_rows if boolstr(r["label_candidate"])]

    saved_candidate = label["candidate"].astype(bool)
    checks = {}
    checks["candidate_count_matches_npz"] = (
        len(candidates) == int(saved_candidate.sum())
    )
    checks["production_candidate_count_7"] = (len(candidates) == 7)
    checks["unit_csv_rows_9747"] = len(unit_rows) == 9747

    # ---------- recompute secondary/global estimators ----------
    block_c, total_block_w = block_contrast_coefficients(
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )
    triad_c, total_triad_w = triad_information_coefficients(labels)
    equal_c = equal_triad_coefficients(labels)

    primary_recomputed = apply_trial_coefficients(X, block_c)
    triad_weighted = apply_trial_coefficients(X, triad_c)
    equal_triad = apply_trial_coefficients(X, equal_c)

    stored_primary = label["observed"].astype(np.float64)
    checks["stored_primary_matches_recompute"] = bool(
        np.allclose(stored_primary, primary_recomputed, atol=1e-7, rtol=1e-6)
    )

    # ---------- frozen artifact-sensitivity effects ----------
    mask = maskz["retain_mask"].astype(bool)  # G x 2 x T
    mask_tasks = [str(x) for x in maskz["tasks"].tolist()]
    artifact_effect_by_task = {}

    for task in ("decision", "feedback"):
        ti = mask_tasks.index(task)
        coeff_sens, _ = block_contrast_coefficients(
            labels,
            block_size=int(cfg["inference"]["block_size_trials"]),
            retain_mask=mask[:, ti, :],
        )
        artifact_effect_by_task[task] = apply_trial_coefficients(X, coeff_sens)

    # ---------- group/block detail ----------
    block_size = int(cfg["inference"]["block_size_trials"])
    n_blocks = T // block_size

    # Exact per-triad block-adjusted effects and information weights.
    group_effects = np.full((G, U), np.nan, dtype=np.float64)
    group_weights = np.zeros(G, dtype=np.float64)
    block_effects = [[] for _ in range(U)]  # only candidates will be filled below

    candidate_indices = [int(r["unit_index"]) for r in candidates]
    candidate_set = set(candidate_indices)

    # Build group-level effects only for candidate units to stay cheap.
    ge_cand = np.full((G, len(candidate_indices)), np.nan, dtype=np.float64)
    gw = np.zeros(G, dtype=np.float64)

    # Also collect candidate block effects: list of (g,b,w,effect)
    block_details = {u: [] for u in candidate_indices}

    for g in range(G):
        pieces = {u: [] for u in candidate_indices}
        weights = []
        block_ids = []

        for b in range(n_blocks):
            sl = slice(b * block_size, (b + 1) * block_size)
            y = labels[g, sl]
            n1 = int(np.sum(y == 1))
            n0 = int(np.sum(y == 0))
            if n1 == 0 or n0 == 0:
                continue

            w = float(n1 * n0 / (n1 + n0))
            weights.append(w)
            block_ids.append(b)

            xb = X[g, sl, :][:, candidate_indices]
            d = xb[y == 1].mean(axis=0) - xb[y == 0].mean(axis=0)

            for k, u in enumerate(candidate_indices):
                pieces[u].append(float(d[k]))
                block_details[u].append((g, b, w, float(d[k])))

        if weights:
            ww = np.asarray(weights, dtype=float)
            gw[g] = float(ww.sum())
            for k, u in enumerate(candidate_indices):
                dd = np.asarray(pieces[u], dtype=float)
                ge_cand[g, k] = float(np.dot(ww / ww.sum(), dd))

    checks["all_triads_have_primary_block_information"] = bool(np.all(gw > 0))
    checks["group_information_weight_sum_37p9"] = (
        abs(float(gw.sum()) - 37.9) < 1e-9
    )

    rows = []

    for k, cand in enumerate(candidates):
        u = int(cand["unit_index"])
        task = cand["task"]
        pmax = float(cand["p_label_maxT"])
        punadj = float(cand["p_label_unadjusted"])
        primary = float(stored_primary[u])
        triad_eff = float(triad_weighted[u])
        equal_eff = float(equal_triad[u])
        artifact_eff = float(artifact_effect_by_task[task][u])

        # LOTO exact recombination of group block-adjusted effects.
        contrib_num = gw * ge_cand[:, k]
        total_num = float(np.nansum(contrib_num))
        total_w = float(gw.sum())
        primary_from_groups = total_num / total_w

        loto = []
        for g in range(G):
            denom = total_w - gw[g]
            if denom <= 0:
                continue
            val = (total_num - contrib_num[g]) / denom
            loto.append(float(val))

        loto = np.asarray(loto, dtype=float)
        loto_sign_consistency = float(
            np.mean(np.sign(loto) == np.sign(primary))
        )

        # Dominant absolute contribution.
        abs_contrib = np.abs(contrib_num / total_w)
        dom_g = int(np.argmax(abs_contrib))
        denom_abs = float(abs_contrib.sum())
        dom_share = (
            float(abs_contrib[dom_g] / denom_abs)
            if denom_abs > 0 else np.nan
        )

        # Block-sign support.
        bd = block_details[u]
        effects = np.asarray([x[3] for x in bd], dtype=float)
        weights_bd = np.asarray([x[2] for x in bd], dtype=float)
        same = np.sign(effects) == np.sign(primary)
        n_same = int(np.sum(same))
        n_blocks_inf = len(effects)
        weighted_same = (
            float(np.sum(weights_bd[same]) / np.sum(weights_bd))
            if np.sum(weights_bd) > 0 else np.nan
        )

        sign_agree_triad = sgn(triad_eff) == sgn(primary)
        sign_agree_equal = sgn(equal_eff) == sgn(primary)
        sign_agree_artifact = sgn(artifact_eff) == sgn(primary)

        row = {
            "unit_index": u,
            "task": task,
            "band": cand["band"],
            "dyad": cand["dyad"],
            "ch1": cand["ch1"],
            "ch2": cand["ch2"],
            "direction": fmt_sign(primary),
            "primary_effect": primary,
            "observed_studentized": float(cand["observed_studentized"]),
            "p_label_unadjusted": punadj,
            "p_label_maxT": pmax,
            "triad_information_weighted_effect": triad_eff,
            "equal_triad_effect": equal_eff,
            "artifact_sensitivity_effect": artifact_eff,
            "triad_weighted_sign_agrees": sign_agree_triad,
            "equal_triad_sign_agrees": sign_agree_equal,
            "artifact_sensitivity_sign_agrees": sign_agree_artifact,
            "all_three_secondary_signs_agree": bool(
                sign_agree_triad and sign_agree_equal and sign_agree_artifact
            ),
            "primary_from_group_recombination": primary_from_groups,
            "loto_min_effect": float(np.min(loto)),
            "loto_max_effect": float(np.max(loto)),
            "loto_sign_consistency_fraction": loto_sign_consistency,
            "dominant_triad": f"G{dom_g+1:02d}",
            "dominant_triad_abs_contribution_share": dom_share,
            "informative_blocks": n_blocks_inf,
            "blocks_same_direction": n_same,
            "block_sign_consistency_fraction": (
                float(n_same / n_blocks_inf) if n_blocks_inf else np.nan
            ),
            "information_weight_same_direction_fraction": weighted_same,
        }
        rows.append(row)

    checks["primary_group_recombination_matches"] = bool(
        all(
            abs(r["primary_effect"] - r["primary_from_group_recombination"])
            < 1e-7
            for r in rows
        )
    )
    checks["candidate_pmax_lt_alpha"] = bool(
        all(
            r["p_label_maxT"] < float(cfg["inference"]["candidate_alpha"])
            for r in rows
        )
    )
    checks["all_candidate_values_finite"] = bool(
        all(
            np.isfinite([
                r["primary_effect"],
                r["observed_studentized"],
                r["p_label_unadjusted"],
                r["p_label_maxT"],
                r["triad_information_weighted_effect"],
                r["equal_triad_effect"],
                r["artifact_sensitivity_effect"],
                r["loto_min_effect"],
                r["loto_max_effect"],
                r["loto_sign_consistency_fraction"],
                r["dominant_triad_abs_contribution_share"],
                r["block_sign_consistency_fraction"],
                r["information_weight_same_direction_fraction"],
            ]).all()
            for r in rows
        )
    )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---------- summary ----------
    task_counts = Counter(r["task"] for r in rows)
    band_counts = Counter(r["band"] for r in rows)
    dyad_counts = Counter(r["dyad"] for r in rows)
    dir_counts = Counter(r["direction"] for r in rows)

    all_pass = all(checks.values())

    lines = [
        "Brain Topography Project — Stage 3E Production Candidate Audit",
        "=" * 72,
        f"Run time: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Production candidates: {len(rows)}",
        "",
        "CANDIDATE DISTRIBUTION",
        f"  task: {dict(task_counts)}",
        f"  band: {dict(band_counts)}",
        f"  dyad: {dict(dyad_counts)}",
        f"  direction: {dict(dir_counts)}",
        "",
        "CANDIDATES",
    ]

    for r in sorted(rows, key=lambda x: x["p_label_maxT"]):
        lines.extend([
            f"  unit {r['unit_index']}: "
            f"{r['task']} {r['band']} {r['dyad']} "
            f"{r['ch1']}-{r['ch2']} ({r['direction']})",
            f"    primary ΔPLV={r['primary_effect']:.6f}; "
            f"z={r['observed_studentized']:.4f}; "
            f"p_unadj={r['p_label_unadjusted']:.8f}; "
            f"p_maxT={r['p_label_maxT']:.8f}",
            f"    secondary effects: "
            f"triad-weighted={r['triad_information_weighted_effect']:.6f}, "
            f"equal-triad={r['equal_triad_effect']:.6f}, "
            f"artifact-sensitivity={r['artifact_sensitivity_effect']:.6f}",
            f"    secondary sign agreement: "
            f"triad={r['triad_weighted_sign_agrees']}, "
            f"equal={r['equal_triad_sign_agrees']}, "
            f"artifact={r['artifact_sensitivity_sign_agrees']}",
            f"    LOTO range=[{r['loto_min_effect']:.6f}, "
            f"{r['loto_max_effect']:.6f}], "
            f"sign consistency={100*r['loto_sign_consistency_fraction']:.1f}%",
            f"    dominant triad={r['dominant_triad']} "
            f"(abs contribution share="
            f"{100*r['dominant_triad_abs_contribution_share']:.1f}%)",
            f"    informative blocks same direction="
            f"{r['blocks_same_direction']}/{r['informative_blocks']} "
            f"({100*r['block_sign_consistency_fraction']:.1f}%); "
            f"information-weight same direction="
            f"{100*r['information_weight_same_direction_fraction']:.1f}%",
        ])

    robust_sign_all = sum(r["all_three_secondary_signs_agree"] for r in rows)
    loto_100 = sum(
        abs(r["loto_sign_consistency_fraction"] - 1.0) < 1e-12
        for r in rows
    )

    lines.extend([
        "",
        "ROBUSTNESS COUNTS",
        f"  Candidates with all 3 secondary estimator signs agreeing: "
        f"{robust_sign_all}/{len(rows)}",
        f"  Candidates with 100% LOTO sign consistency: "
        f"{loto_100}/{len(rows)}",
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
        "INTERPRETATION RULE",
        "  These are robustness diagnostics, not additional selection filters.",
        "  The production maxT result defines the candidate set. Temporal-shift",
        "  and partner-shuffle nulls should next be applied to this fixed set,",
        "  with candidate-family multiple-comparison control as frozen in config.",
        "",
        f"CSV: {out_csv}",
        f"Summary: {out_txt}",
    ])

    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
