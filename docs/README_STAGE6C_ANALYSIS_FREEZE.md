# Stage 6C — Analysis Freeze & Evidence Integration

Stage 6C performs **no new statistical analysis** and **no candidate selection**.
It verifies the already-completed evidence chain and freezes the final three PLV
candidates for manuscript planning.

## What it verifies

The script cross-checks these existing production outputs:

- `results/nulls/candidate_nulls_plv.csv` — Stage 3G
- `results/nulls/stage4D2_empirical_within_triad_dyad_null.csv` — Stage 4D2
- `results/robustness/stage5B_imcoh_final3_robustness.csv` — Stage 5B
- `results/stability/stage6B_final3_stability.csv` — Stage 6B
- `results/stability/stage6B_final3_stability_summary.txt` — Stage 6B PASS record

It fails closed unless all of the following remain true:

- fixed final units are exactly `3801, 4994, 8156`;
- task/band/dyad/channel metadata agree across stages;
- the primary PLV effect agrees numerically across stages;
- all four PLV null layers pass for all final three;
- all three iCOH effects agree in direction with PLV but none passes iCOH label-maxT or all four iCOH nulls;
- Stage 6B uses 11 informative triads, all 11 LOTO estimates retain direction, all bootstrap draws retain direction, and equal-triad weighting agrees.

Individual-triad sign consistency is recorded descriptively but is **not** a
candidate-retention rule (unit 8156 is 9/11, whereas 3801 and 4994 are 11/11).

## Install

Unzip the patch into the project root. It adds:

- `scripts/39_freeze_analysis_evidence.py`
- `README_STAGE6C_ANALYSIS_FREEZE.md`

## Run

```bash
cd .
source .venv/bin/activate  # use your existing environment if different
python3 scripts/39_freeze_analysis_evidence.py
```

## Expected outputs

If every cross-stage check passes:

- `results/freeze/stage6C_master_evidence.csv`
- `results/freeze/stage6C_analysis_freeze_record.md`

The terminal should contain:

`RESULT: PASS — analysis evidence frozen for manuscript planning`

## Interpretation rule

The frozen manuscript-level claim is limited to null-calibrated PLV topographic
associations with strong triad-level stability. iCOH is directionally concordant
but not inferentially confirmatory. Do not claim causal coupling, information
flow, or demonstrated robustness to zero-lag/common-source contributions.
