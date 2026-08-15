# Brain Topography Project — Stage 6C Analysis Freeze

**Freeze time:** 2026-08-15T09:50:24-04:00

**Status: FROZEN — PASS**

This record freezes the evidence chain after Stage 6B. Stage 6C performs no new
statistical inference and no candidate selection; it verifies consistency across
the already-completed Stage 3G, Stage 4D2, Stage 5B, and Stage 6B outputs.

## Frozen inferential framework

- Dataset: OpenNeuro ds007822 v1.0.0.
- Inferential unit: triad (n = 11).
- Primary connectivity metric: PLV.
- Stage 3D label-maxT defines the fixed label-supported family; Stage 3G evaluates temporal-shift and cross-group partner nulls.
- Stage 4D2 adds the empirical within-triad dyad-identity null motivated by the Stage 4 simulation work.
- Final PLV family is fixed at exactly three candidates: 3801, 4994, and 8156.
- iCOH is a pre-specified non-zero-lag robustness metric, not the primary metric.
- Stage 6 LOTO/bootstrap quantities are post-selection sensitivity descriptors, not replacement significance tests.

## Frozen final evidence table

| Unit | Context | ΔPLV | PLV null p-values (label / time / partner / dyad) | iCOH Δ | iCOH label p | Individual triads same direction | LOTO sign retention | 10k bootstrap 95% interval |
|---:|---|---:|---|---:|---:|---:|---:|---|
| 3801 | decision / beta / pair13 / Fp2–C3 | 0.066665 | 0.009598 / 0.000999 / 0.000999 / 0.036963 | 0.014314 | 0.381618 | 11/11 | 11/11 | [0.045936, 0.097500] |
| 4994 | decision / gamma / pair13 / F7–F8 | 0.078445 | 0.017197 / 0.000999 / 0.000999 / 0.001998 | 0.014489 | 0.261738 | 11/11 | 11/11 | [0.047832, 0.119343] |
| 8156 | feedback / beta / pair13 / T3–C4 | 0.088978 | 0.017996 / 0.000999 / 0.000999 / 0.011988 | 0.012420 | 0.776224 | 9/11 | 11/11 | [0.052807, 0.123148] |

## Evidence interpretation frozen for manuscript use

All three final PLV candidates pass the four empirical PLV null layers and show strong triad-level stability. Units 3801 and 4994 have the same PLV-effect direction in 11/11 individual triads; unit 8156 does so in 9/11. All three retain the PLV direction in 11/11 leave-one-triad-out estimates, all 10,000 triad-bootstrap draws retain the same direction, and equal-triad weighting agrees with the primary direction.

For iCOH, all three effects agree in direction with PLV, but none passes iCOH label-maxT and none passes all four iCOH null layers. Therefore the iCOH analysis is directionally concordant but not inferentially confirmatory.

## Claim ceiling

The manuscript may describe these as **null-calibrated PLV topographic associations that are stable to triad omission/resampling and weighting sensitivity**. It may report that iCOH effects are directionally concordant.

The manuscript must **not** claim that the surviving candidates establish causal inter-brain neural coupling, demonstrate information flow, or are robustly insensitive to zero-lag/common-source contributions. The lack of iCOH significance explicitly limits those interpretations.

## Analysis-freeze rule

The three-candidate PLV family is now frozen for the current manuscript. Subsequent visualization, reporting, and manuscript drafting must not silently add, remove, or redefine candidates based on new exploratory results. Any future external validation or additional exploratory analysis must be labeled as a separate post-freeze extension and must not retroactively alter the frozen primary evidence chain without an explicit documented amendment.

## Technical checks

- **PASS** — `stage4D2_final_family_exactly_expected_3` — [3801, 4994, 8156]
- **PASS** — `stage5B_family_exactly_expected_3` — [3801, 4994, 8156]
- **PASS** — `stage6B_family_exactly_expected_3` — [3801, 4994, 8156]
- **PASS** — `stage3G_contains_all_final3`
- **PASS** — `unit_3801_metadata_consistent` — [('decision', 'beta', 'pair13', 'Fp2', 'C3'), ('decision', 'beta', 'pair13', 'Fp2', 'C3'), ('decision', 'beta', 'pair13', 'Fp2', 'C3'), ('decision', 'beta', 'pair13', 'Fp2', 'C3')]
- **PASS** — `unit_3801_PLV_effect_consistent_across_stages` — S3=0.066665195, S4=0.066665195, S5=0.066665195, S6=0.066665195
- **PASS** — `unit_3801_stage3G_expected_pass_pattern`
- **PASS** — `unit_3801_stage4D2_all_four_PLV_nulls_pass`
- **PASS** — `unit_3801_stage5B_direction_only_expected_pattern`
- **PASS** — `unit_3801_stage6B_stability_expected_pattern`
- **PASS** — `unit_4994_metadata_consistent` — [('decision', 'gamma', 'pair13', 'F7', 'F8'), ('decision', 'gamma', 'pair13', 'F7', 'F8'), ('decision', 'gamma', 'pair13', 'F7', 'F8'), ('decision', 'gamma', 'pair13', 'F7', 'F8')]
- **PASS** — `unit_4994_PLV_effect_consistent_across_stages` — S3=0.078444600, S4=0.078444600, S5=0.078444600, S6=0.078444600
- **PASS** — `unit_4994_stage3G_expected_pass_pattern`
- **PASS** — `unit_4994_stage4D2_all_four_PLV_nulls_pass`
- **PASS** — `unit_4994_stage5B_direction_only_expected_pattern`
- **PASS** — `unit_4994_stage6B_stability_expected_pattern`
- **PASS** — `unit_8156_metadata_consistent` — [('feedback', 'beta', 'pair13', 'T3', 'C4'), ('feedback', 'beta', 'pair13', 'T3', 'C4'), ('feedback', 'beta', 'pair13', 'T3', 'C4'), ('feedback', 'beta', 'pair13', 'T3', 'C4')]
- **PASS** — `unit_8156_PLV_effect_consistent_across_stages` — S3=0.088977635, S4=0.088977635, S5=0.088977635, S6=0.088977635
- **PASS** — `unit_8156_stage3G_expected_pass_pattern`
- **PASS** — `unit_8156_stage4D2_all_four_PLV_nulls_pass`
- **PASS** — `unit_8156_stage5B_direction_only_expected_pattern`
- **PASS** — `unit_8156_stage6B_stability_expected_pattern`

## Provenance / SHA256

- `results/nulls/candidate_nulls_plv.csv`  
  SHA256: `2b9dcc8e650323bcce5a55cffd50201fbea0d40459e2155f263ac33330dd0356`
- `results/nulls/stage4D2_empirical_within_triad_dyad_null.csv`  
  SHA256: `cde80283b2229bf2e8f95f3585d501bc8376379d37407f42c2996d642292c599`
- `results/robustness/stage5B_imcoh_final3_robustness.csv`  
  SHA256: `013a8928372618398a73116cbf81c8b955fda486609380a955b1e40cc54b9c3b`
- `results/stability/stage6B_final3_stability.csv`  
  SHA256: `13dce9876c650bdfebd1739b2311faa68363dff7596bf29e70b52b80d0f2f8a0`
- `results/stability/stage6B_final3_stability_summary.txt`  
  SHA256: `4a541e3dd1bad2f94780926bdf8a8459c098b9a8ac86bb3c0b2a0d209ddc52fe`
- `results/freeze/stage6C_master_evidence.csv`  
  SHA256: `14881d1e4a2fe463e870262314ef4eb81d676f8149b25881bf7ca5d4fde9b98c`

## Stage 6C result

**PASS — analysis evidence frozen for manuscript planning.**
