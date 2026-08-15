# Stage 5 — fixed-final-3 imaginary-coherency robustness

PLV remains the primary metric.

The robustness metric is the absolute imaginary part of narrow-band complex
coherency:

    |Im(sum z1 conj(z2) / sqrt(sum |z1|^2 sum |z2|^2))|

The analytic signal preserves amplitude. Filtering, explicit reflection
padding, Hilbert transform, and task-window cropping follow the same window
logic as the validated PLV engine.

No new units are screened. The candidate family is fixed to the three PLV
candidates that survived all four empirical null layers in Stage 4D2.

Validation first:

```bash
python3 scripts/35_validate_imcoh_robustness.py
```

Production only after validation passes:

```bash
python3 scripts/36_run_imcoh_robustness.py
```
