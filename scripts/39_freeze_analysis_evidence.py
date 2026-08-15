#!/usr/bin/env python3
"""
Stage 6C — Freeze the final evidence chain for the Brain Topography project.

This stage performs NO new statistical analysis and NO candidate selection.
It cross-checks the already-frozen Stage 4D2 final PLV family against Stage 3G,
Stage 5B iCOH robustness, and Stage 6B triad stability, then writes:

  results/freeze/stage6C_master_evidence.csv
  results/freeze/stage6C_analysis_freeze_record.md

The freeze fails closed if candidate IDs, metadata, primary PLV effects, or the
expected pass/fail evidence pattern do not agree across stages.
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve()
PROJECT = next((p for p in HERE.parents if (p / "src" / "bt").is_dir()), None)
if PROJECT is None:
    raise RuntimeError(f"Could not locate project root from {HERE}")
sys.path.insert(0, str(PROJECT / "src"))

from bt.config import load_config

EXPECTED_IDS = [3801, 4994, 8156]
EXPECTED_META = {
    3801: ("decision", "beta", "pair13", "Fp2", "C3"),
    4994: ("decision", "gamma", "pair13", "F7", "F8"),
    8156: ("feedback", "beta", "pair13", "T3", "C4"),
}
ALPHA = 0.05
EFFECT_TOL = 2e-6


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"CSV is empty: {path}")
    return rows


def boolstr(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def fnum(v) -> float:
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(f"Non-finite numeric value: {v}")
    return x


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def index_by_unit(rows: list[dict[str, str]], label: str) -> dict[int, dict[str, str]]:
    out: dict[int, dict[str, str]] = {}
    for r in rows:
        u = int(r["unit_index"])
        if u in out:
            raise RuntimeError(f"Duplicate unit {u} in {label}")
        out[u] = r
    return out


def meta_tuple(r: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (r["task"], r["band"], r["dyad"], r["ch1"], r["ch2"])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fail_if(name: str, condition: bool, details: str = "") -> tuple[str, bool, str]:
    return (name, bool(condition), details)


def main() -> int:
    cfg = load_config()
    results = Path(cfg["results_root"])
    outdir = results / "freeze"
    outdir.mkdir(parents=True, exist_ok=True)

    stage3g = results / "nulls" / "candidate_nulls_plv.csv"
    stage4d2 = results / "nulls" / "stage4D2_empirical_within_triad_dyad_null.csv"
    stage5b = results / "robustness" / "stage5B_imcoh_final3_robustness.csv"
    stage6b = results / "stability" / "stage6B_final3_stability.csv"
    stage6b_summary = results / "stability" / "stage6B_final3_stability_summary.txt"

    source_paths = [stage3g, stage4d2, stage5b, stage6b, stage6b_summary]
    missing = [p for p in source_paths if not p.exists()]
    if missing:
        raise RuntimeError("Missing required frozen source files:\n  " + "\n  ".join(map(str, missing)))

    if "RESULT\n  PASS" not in stage6b_summary.read_text(encoding="utf-8"):
        raise RuntimeError("Stage 6B summary does not contain RESULT PASS")

    r3 = read_csv(stage3g)
    r4 = read_csv(stage4d2)
    r5 = read_csv(stage5b)
    r6 = read_csv(stage6b)

    i3 = index_by_unit(r3, "Stage 3G")
    i4_all = index_by_unit(r4, "Stage 4D2")
    i5 = index_by_unit(r5, "Stage 5B")
    i6 = index_by_unit(r6, "Stage 6B")

    final4 = [r for r in r4 if boolstr(r.get("all_four_null_layers_pass", False))]
    final_ids = [int(r["unit_index"]) for r in final4]

    checks: list[tuple[str, bool, str]] = []
    checks.append(fail_if("stage4D2_final_family_exactly_expected_3", final_ids == EXPECTED_IDS, str(final_ids)))
    checks.append(fail_if("stage5B_family_exactly_expected_3", list(i5.keys()) == EXPECTED_IDS, str(list(i5.keys()))))
    checks.append(fail_if("stage6B_family_exactly_expected_3", list(i6.keys()) == EXPECTED_IDS, str(list(i6.keys()))))
    checks.append(fail_if("stage3G_contains_all_final3", all(u in i3 for u in EXPECTED_IDS)))

    master_rows: list[dict] = []

    for u in EXPECTED_IDS:
        if u not in i3 or u not in i4_all or u not in i5 or u not in i6:
            checks.append(fail_if(f"unit_{u}_present_in_all_stages", False))
            continue

        a, b, c, d = i3[u], i4_all[u], i5[u], i6[u]
        expected_meta = EXPECTED_META[u]
        meta_ok = all(meta_tuple(x) == expected_meta for x in (a, b, c, d))
        checks.append(fail_if(f"unit_{u}_metadata_consistent", meta_ok, str([meta_tuple(x) for x in (a,b,c,d)])))

        # Primary PLV effect: Stage 3G may call it observed_effect; Stage 4D2 does too.
        e3 = fnum(a["observed_effect"])
        e4 = fnum(b["observed_effect"])
        e5 = fnum(c["PLV_effect"])
        e6 = fnum(d["observed_PLV_effect"])
        eff_ok = max(abs(e3-e4), abs(e3-e5), abs(e3-e6)) < EFFECT_TOL
        checks.append(fail_if(f"unit_{u}_PLV_effect_consistent_across_stages", eff_ok,
                              f"S3={e3:.9f}, S4={e4:.9f}, S5={e5:.9f}, S6={e6:.9f}"))

        # Stage 3G: fixed label-supported family and both secondary nulls pass.
        p_label = fnum(a["p_label_maxT"])
        p_time = fnum(a["p_time_maxT"])
        p_partner = fnum(a["p_partner_maxT"])
        s3_ok = (p_label < ALPHA and p_time < ALPHA and p_partner < ALPHA
                 and boolstr(a["secondary_validation_pass"]))
        checks.append(fail_if(f"unit_{u}_stage3G_expected_pass_pattern", s3_ok))

        # Stage 4D2: the additional within-triad dyad null passes for the final 3.
        p_dyad = fnum(b["p_within_triad_dyad_maxT"])
        s4_ok = (p_dyad < ALPHA and boolstr(b["within_triad_dyad_pass"])
                 and boolstr(b["all_four_null_layers_pass"]))
        checks.append(fail_if(f"unit_{u}_stage4D2_all_four_PLV_nulls_pass", s4_ok))

        # Stage 5B: direction agrees, but the pre-specified iCOH robustness test
        # does not reproduce label-maxT significance or all-four-null significance.
        icoh = fnum(c["iCOH_effect"])
        pic_label = fnum(c["p_iCOH_label_maxT"])
        pic_time = fnum(c["p_iCOH_time_maxT"])
        pic_partner = fnum(c["p_iCOH_partner_maxT"])
        pic_dyad = fnum(c["p_iCOH_dyad_maxT"])
        s5_ok = (boolstr(c["direction_agrees_with_PLV"])
                 and not boolstr(c["iCOH_label_pass"])
                 and not boolstr(c["iCOH_all_four_nulls_pass"]))
        checks.append(fail_if(f"unit_{u}_stage5B_direction_only_expected_pattern", s5_ok))

        # Stage 6B: all LOTO estimates and all bootstrap draws retain the full
        # direction, and equal-triad weighting agrees. Individual triad signs are
        # reported descriptively and are NOT a candidate-retention rule.
        informative = int(d["informative_triads"])
        loto_n = int(d["LOTO_same_direction_count"])
        loto_frac = fnum(d["LOTO_same_direction_fraction"])
        boot_frac = fnum(d["bootstrap_same_direction_fraction"])
        equal_agree = boolstr(d["equal_triad_direction_agrees_secondary"])
        s6_ok = (informative == 11 and loto_n == 11 and abs(loto_frac - 1.0) < 1e-12
                 and abs(boot_frac - 1.0) < 1e-12 and equal_agree)
        checks.append(fail_if(f"unit_{u}_stage6B_stability_expected_pattern", s6_ok))

        master_rows.append({
            "unit_index": u,
            "task": expected_meta[0],
            "band": expected_meta[1],
            "dyad": expected_meta[2],
            "ch1": expected_meta[3],
            "ch2": expected_meta[4],
            "PLV_effect": e4,
            "p_PLV_label_maxT": p_label,
            "p_PLV_time_maxT": p_time,
            "p_PLV_crossgroup_partner_maxT": p_partner,
            "p_PLV_within_triad_dyad_maxT": p_dyad,
            "PLV_all_four_null_layers_pass": True,
            "iCOH_effect": icoh,
            "iCOH_direction_agrees_with_PLV": boolstr(c["direction_agrees_with_PLV"]),
            "p_iCOH_label_maxT": pic_label,
            "p_iCOH_time_maxT": pic_time,
            "p_iCOH_partner_maxT": pic_partner,
            "p_iCOH_within_triad_dyad_maxT": pic_dyad,
            "iCOH_label_pass": boolstr(c["iCOH_label_pass"]),
            "iCOH_all_four_nulls_pass": boolstr(c["iCOH_all_four_nulls_pass"]),
            "informative_triads": informative,
            "individual_triads_same_direction": int(d["individual_triads_same_direction"]),
            "LOTO_same_direction_count": loto_n,
            "LOTO_min_effect": fnum(d["LOTO_min_effect"]),
            "LOTO_max_effect": fnum(d["LOTO_max_effect"]),
            "most_influential_omitted_triad": d["most_influential_omitted_triad"],
            "most_influential_abs_effect_change": fnum(d["most_influential_abs_effect_change"]),
            "bootstrap_draws": int(d["bootstrap_draws"]),
            "bootstrap_median_effect": fnum(d["bootstrap_median_effect"]),
            "bootstrap_95pct_low_descriptive": fnum(d["bootstrap_95pct_low"]),
            "bootstrap_95pct_high_descriptive": fnum(d["bootstrap_95pct_high"]),
            "bootstrap_same_direction_fraction": boot_frac,
            "equal_triad_effect_secondary": fnum(d["equal_triad_full_effect_secondary"]),
            "equal_triad_direction_agrees_secondary": equal_agree,
            "frozen_evidence_class": "PLV all-four-null supported; triad-stable; iCOH direction-only",
            "claim_ceiling": "Null-calibrated PLV topographic association; do not claim causal or zero-lag-insensitive inter-brain coupling",
        })

    all_pass = all(ok for _, ok, _ in checks) and len(master_rows) == 3

    # Fail closed: do not write a freeze record that looks authoritative unless
    # every cross-stage consistency check passes.
    if not all_pass:
        report = ["Stage 6C pre-freeze checks FAILED", ""]
        for name, ok, details in checks:
            report.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f" :: {details}" if details else ""))
        print("\n".join(report))
        return 1

    master_csv = outdir / "stage6C_master_evidence.csv"
    write_csv(master_csv, master_rows)

    source_hashes = {p: sha256(p) for p in source_paths}
    master_hash = sha256(master_csv)
    now = datetime.now().astimezone().isoformat(timespec="seconds")

    md = outdir / "stage6C_analysis_freeze_record.md"
    lines = [
        "# Brain Topography Project — Stage 6C Analysis Freeze",
        "",
        f"**Freeze time:** {now}",
        "",
        "**Status: FROZEN — PASS**",
        "",
        "This record freezes the evidence chain after Stage 6B. Stage 6C performs no new",
        "statistical inference and no candidate selection; it verifies consistency across",
        "the already-completed Stage 3G, Stage 4D2, Stage 5B, and Stage 6B outputs.",
        "",
        "## Frozen inferential framework",
        "",
        "- Dataset: OpenNeuro ds007822 v1.0.0.",
        "- Inferential unit: triad (n = 11).",
        "- Primary connectivity metric: PLV.",
        "- Stage 3D label-maxT defines the fixed label-supported family; Stage 3G evaluates temporal-shift and cross-group partner nulls.",
        "- Stage 4D2 adds the empirical within-triad dyad-identity null motivated by the Stage 4 simulation work.",
        "- Final PLV family is fixed at exactly three candidates: 3801, 4994, and 8156.",
        "- iCOH is a pre-specified non-zero-lag robustness metric, not the primary metric.",
        "- Stage 6 LOTO/bootstrap quantities are post-selection sensitivity descriptors, not replacement significance tests.",
        "",
        "## Frozen final evidence table",
        "",
        "| Unit | Context | ΔPLV | PLV null p-values (label / time / partner / dyad) | iCOH Δ | iCOH label p | Individual triads same direction | LOTO sign retention | 10k bootstrap 95% interval |",
        "|---:|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for r in master_rows:
        context = f"{r['task']} / {r['band']} / {r['dyad']} / {r['ch1']}–{r['ch2']}"
        nulls = (f"{r['p_PLV_label_maxT']:.6f} / {r['p_PLV_time_maxT']:.6f} / "
                 f"{r['p_PLV_crossgroup_partner_maxT']:.6f} / {r['p_PLV_within_triad_dyad_maxT']:.6f}")
        ci = f"[{r['bootstrap_95pct_low_descriptive']:.6f}, {r['bootstrap_95pct_high_descriptive']:.6f}]"
        lines.append(
            f"| {r['unit_index']} | {context} | {r['PLV_effect']:.6f} | {nulls} | "
            f"{r['iCOH_effect']:.6f} | {r['p_iCOH_label_maxT']:.6f} | "
            f"{r['individual_triads_same_direction']}/11 | {r['LOTO_same_direction_count']}/11 | {ci} |"
        )

    lines.extend([
        "",
        "## Evidence interpretation frozen for manuscript use",
        "",
        "All three final PLV candidates pass the four empirical PLV null layers and show strong triad-level stability. Units 3801 and 4994 have the same PLV-effect direction in 11/11 individual triads; unit 8156 does so in 9/11. All three retain the PLV direction in 11/11 leave-one-triad-out estimates, all 10,000 triad-bootstrap draws retain the same direction, and equal-triad weighting agrees with the primary direction.",
        "",
        "For iCOH, all three effects agree in direction with PLV, but none passes iCOH label-maxT and none passes all four iCOH null layers. Therefore the iCOH analysis is directionally concordant but not inferentially confirmatory.",
        "",
        "## Claim ceiling",
        "",
        "The manuscript may describe these as **null-calibrated PLV topographic associations that are stable to triad omission/resampling and weighting sensitivity**. It may report that iCOH effects are directionally concordant.",
        "",
        "The manuscript must **not** claim that the surviving candidates establish causal inter-brain neural coupling, demonstrate information flow, or are robustly insensitive to zero-lag/common-source contributions. The lack of iCOH significance explicitly limits those interpretations.",
        "",
        "## Analysis-freeze rule",
        "",
        "The three-candidate PLV family is now frozen for the current manuscript. Subsequent visualization, reporting, and manuscript drafting must not silently add, remove, or redefine candidates based on new exploratory results. Any future external validation or additional exploratory analysis must be labeled as a separate post-freeze extension and must not retroactively alter the frozen primary evidence chain without an explicit documented amendment.",
        "",
        "## Technical checks",
        "",
    ])
    for name, ok, details in checks:
        suffix = f" — {details}" if details else ""
        lines.append(f"- **{'PASS' if ok else 'FAIL'}** — `{name}`{suffix}")

    lines.extend([
        "",
        "## Provenance / SHA256",
        "",
    ])
    for p in source_paths:
        lines.append(f"- `{p}`  ")
        lines.append(f"  SHA256: `{source_hashes[p]}`")
    lines.extend([
        f"- `{master_csv}`  ",
        f"  SHA256: `{master_hash}`",
        "",
        "## Stage 6C result",
        "",
        "**PASS — analysis evidence frozen for manuscript planning.**",
        "",
    ])
    md.write_text("\n".join(lines), encoding="utf-8")

    print("Brain Topography Project — Stage 6C Analysis Freeze")
    print("=" * 64)
    print("RESULT: PASS — analysis evidence frozen for manuscript planning")
    print(f"Master evidence CSV: {master_csv}")
    print(f"Freeze record:       {md}")
    print(f"Master CSV SHA256:   {master_hash}")
    print("")
    for r in master_rows:
        print(
            f"unit {r['unit_index']}: {r['task']} {r['band']} {r['dyad']} "
            f"{r['ch1']}-{r['ch2']} | ΔPLV={r['PLV_effect']:.6f} | "
            f"LOTO={r['LOTO_same_direction_count']}/11 | "
            f"iCOH label p={r['p_iCOH_label_maxT']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
