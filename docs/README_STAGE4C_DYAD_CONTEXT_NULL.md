# Stage 4C — triad-preserving dyad null

Stage 4B revealed a critical limitation: cross-group partner shuffling does not
rule out a group-specific common drive because that shuffle destroys the group
context itself.

The proposed new null operates within each triad and trial. For a fixed
candidate channel pair it randomizes which of the three dyads occupies the
candidate-dyad role. Therefore triad-specific common timing and common drive are
retained, while stable dyad identity is broken.

This null is tested in simulation before any empirical use.

Run only:

```bash
python3 scripts/31_validate_within_triad_dyad_null.py
```

If and only if Stage 4C1 passes, then run:

```bash
python3 scripts/32_stress_test_within_triad_dyad_null.py
```
