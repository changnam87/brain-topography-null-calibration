# Stage 4D — empirical within-triad dyad null

Stage 4C2 satisfied the pre-specified integration rule:

- group-specific shared-event FWER was strongly reduced;
- sparse-true sensitivity did not collapse;
- precision improved especially in the hard mixed-confound conditions.

Therefore the triad-preserving dyad null is now eligible to be applied as an
additional empirical robustness layer to the FIXED seven Stage-3D candidates.

Run validation first:

```bash
python3 scripts/33_validate_empirical_within_triad_dyad_null.py
```

Do not run production until validation passes.

Production:

```bash
python3 scripts/34_run_empirical_within_triad_dyad_null.py
```
