# Stage 6 — fixed-final-3 triad stability

This patch supersedes the earlier generic `scripts/06_run_stability.py` for the
current frozen analysis.

## Why a new Stage 6 is needed

The old generic stability implementation averaged triad effects equally for
LOTO and bootstrap. That does **not** reproduce the primary estimator, which is
block-information-weighted across triads. The new implementation preserves the
same primary weighting rule and restricts stability analysis to the three PLV
candidates already frozen by Stage 4D2:

- 3801 — decision / beta / pair13 / Fp2-C3
- 4994 — decision / gamma / pair13 / F7-F8
- 8156 — feedback / beta / pair13 / T3-C4

Stage 6 is a post-selection stability/sensitivity analysis. It does not create,
remove, or redefine candidates.

## Install

Unzip this patch into the project root so that these paths are created:

- `src/bt/final3_stability.py`
- `scripts/37_validate_final3_stability.py`
- `scripts/38_run_final3_stability.py`
- `README_STAGE6_FINAL3_STABILITY.md`

## Run Stage 6A first

```bash
cd .
source .venv/bin/activate  # use your existing project environment if different
python3 scripts/37_validate_final3_stability.py
```

Expected final line in the summary:

`PASS — production Stage 6B may be run`

If Stage 6A does not pass, do not run production; share the complete terminal
output and the generated `results/stability/stage6A_final3_stability_validation_summary.txt`.

## Then run Stage 6B

```bash
python3 scripts/38_run_final3_stability.py
```

Production uses 10,000 triad-cluster bootstrap draws for the fixed three.

## Production outputs

- `results/stability/stage6B_final3_stability.csv`
- `results/stability/stage6B_final3_triad_effects.csv`
- `results/stability/stage6B_final3_loto.csv`
- `results/stability/stage6B_final3_stability.npz`
- `results/stability/stage6B_final3_stability_summary.txt`

## What to send back

The most useful files for the next decision are:

1. `stage6A_final3_stability_validation_summary.txt`
2. after PASS, `stage6B_final3_stability_summary.txt`
3. `stage6B_final3_stability.csv`

Do not interpret a bootstrap interval as a replacement significance test. The
formal evidence remains the frozen null-calibrated PLV pipeline; Stage 6 only
quantifies sensitivity to the finite set of triads and weighting choices.
