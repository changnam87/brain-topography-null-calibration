# Stage 7C — Brain Topography manuscript figure package

## Design rule

This package separates **data-driven figures** from the **conceptual framework figure**.

- **Figure 1** is a static publication graphic (PDF/SVG/600-dpi PNG). It contains no empirical result values and therefore is not regenerated from the results hierarchy.
- **Figures 2–5** are generated exclusively from saved project results. The scripts do not simulate, approximate, or reconstruct missing empirical distributions.
- In particular, no normal approximation is used to invent null distributions that were not saved by the original analysis. Adjusted p-values, simulation results, triad effects, LOTO estimates, and bootstrap summaries are plotted directly from the frozen output files.
- The final candidate family is fail-closed at **[3801, 4994, 8156]** where appropriate.

All figures are exported as **vector PDF**, **editable SVG**, and **600-dpi PNG**.

## Figure set

### Figure 1 — Experimental design and null-calibrated hyperscanning framework
Static vector image supplied in `figures/manuscript/`.

Panels:
- **A**: triadic interaction and the CCC versus Other behavioral contrast.
- **B**: decision (0–4 s) and feedback (0–2 s) analysis windows.
- **C**: sensor-level three-dyad hyperscanning topology with triad-level inference.
- **D**: four-layer progressive null-calibration logic; PLV is primary and iCOH is a non-zero-lag robustness boundary.

### Figure 2 — Progressive null calibration
Generated from:
- `results/nulls/label_null_plv_units.csv`
- `results/nulls/candidate_nulls_plv.csv`
- `results/nulls/stage4D2_empirical_within_triad_dyad_null.csv`

Panels:
- **A**: 9,747 → 7 → 7 → 7 → 3 candidate attrition.
- **B**: adjusted empirical p-values for all seven fixed label-supported candidates across the four null layers. The three final rows receive a border; the four candidates rejected by the within-triad dyad null remain visible.

### Figure 3 — Final PLV topographies and triad-level effects
Generated from:
- `results/freeze/stage6C_master_evidence.csv`
- `results/stability/stage6B_final3_triad_effects.csv`

Each row shows one frozen candidate:
- 3801: decision beta, pair13, Fp2–C3
- 4994: decision gamma, pair13, F7–F8
- 8156: feedback beta, pair13, T3–C4

The left side is a sensor-space cross-brain topology. The right side plots the actual 11 triad effects, the full block-information-weighted effect, and the descriptive 10,000-bootstrap interval. Opposite-direction individual triads are drawn as open markers rather than hidden.

### Figure 4 — Simulation evidence motivating layer 4
Generated from:
- `results/simulation/stage4C2_dyad_null_stress_summary.csv`

Panels:
- **A**: family-wise false-positive rates before versus after the within-triad dyad null under global- and group-shared confounds.
- **B**: sensitivity–precision movement caused by adding the dyad null in true-edge and true-edge-plus-confound cells. Arrows explicitly show the trade-off rather than presenting the new null as uniformly beneficial.

### Figure 5 — Triad stability and iCOH boundary
Generated from:
- `results/freeze/stage6C_master_evidence.csv`
- `results/stability/stage6B_final3_loto.csv`

Panels:
- **A**: full PLV effect, all 11 leave-one-triad-out estimates, and descriptive 10,000-bootstrap intervals.
- **B**: the four adjusted iCOH p-values for each frozen candidate, with the alpha=.05 threshold shown explicitly. Candidate labels report ΔiCOH and directional agreement with PLV. The panel is intentionally designed to show non-confirmation rather than visually minimizing it.

## Installation / execution

The ZIP is structured to be extracted **directly into the project root** (there is no extra wrapper directory).

```bash
cd .
unzip -o ~/Downloads/BT_stage7C_manuscript_figures.zip -d .
source .venv/bin/activate
python3 scripts/44_make_all_manuscript_figures.py
```

Outputs are written to:

```text
figures/manuscript/
```

Expected generated files for each data figure:

```text
Fig2_progressive_null_calibration.pdf/.svg/.png
Fig3_final_topographies_triad_effects.pdf/.svg/.png
Fig4_simulation_justification.pdf/.svg/.png
Fig5_stability_metric_boundary.pdf/.svg/.png
```

Figure 1 is already present there as PDF/SVG/PNG.

## Publication-figure decisions

- No figure title or figure number is embedded inside the artwork; the journal caption should carry that information.
- Panel labels are embedded.
- Full-width layout is approximately 180 mm / 7.15 inches.
- Text is kept at publication-readable size rather than shrinking labels to fit more information.
- Sensor diagrams are explicitly **sensor-space**, not anatomical/source-localization claims.
- PLV and iCOH magnitudes are not plotted as directly comparable effect sizes because they are different metrics.
- Bootstrap intervals are labeled/designed as descriptive stability summaries, not replacement inferential tests.
- The fourth null is shown together with its simulation trade-off so the manuscript does not imply that it was added arbitrarily after inspecting the empirical final candidates.

## Caption language (recommended)

**Figure 1. Experimental design and null-calibrated hyperscanning framework.** The analysis contrasts unanimous triadic cooperation (CCC) with all remaining triadic outcomes and evaluates sensor-level inter-brain phase locking across decision and feedback windows. The inferential framework progressively tests behavioral-label exchangeability, temporal alignment, partner structure, and within-triad dyad context. PLV is the primary connectivity metric; iCOH is used only as a non-zero-lag robustness analysis.

**Figure 2. Progressive null calibration of the sensor-level PLV search.** Global studentized maxT control reduced 9,747 eligible PLV units to seven label-supported candidates. All seven survived temporal-shift and cross-partner calibration, whereas the simulation-motivated within-triad dyad null retained three. Adjusted empirical p-values are shown for the fixed seven-candidate family; bordered rows denote the frozen final PLV family.

**Figure 3. Final null-calibrated PLV topographies and triad-level stability.** The three frozen sensor-space associations were decision beta Fp2–C3, decision gamma F7–F8, and feedback beta T3–C4, all for pair13. Triad-level CCC-minus-Other effects are shown together with the full block-information-weighted estimate and descriptive 10,000-bootstrap interval. Units 3801 and 4994 showed the same effect direction in 11/11 triads; unit 8156 did so in 9/11, while all three retained direction under every leave-one-triad-out estimate.

**Figure 4. Simulation evidence motivating the within-triad dyad null.** A group-specific shared-drive failure mode remained after the earlier null framework. Adding the within-triad dyad null markedly reduced false-positive rates under group-shared confounds. Sensitivity–precision trajectories show the corresponding recovery trade-off in simulations containing true edges with and without a group-shared confound.

**Figure 5. Triad stability and the non-zero-lag metric boundary.** All three frozen PLV associations retained direction in 11/11 leave-one-triad-out estimates and in all 10,000 triad-bootstrap draws. iCOH effects agreed in direction with PLV, but none reached adjusted significance under the behavior-label or complete four-null framework. The iCOH result therefore provides directional concordance, not confirmatory evidence of zero-lag-insensitive inter-brain coupling.
