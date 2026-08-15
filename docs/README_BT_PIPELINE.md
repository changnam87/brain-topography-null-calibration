# Brain Topography EEG Hyperscanning Experiment Codebase

This codebase is for the new Brain Topography submission project. It does **not**
reuse the legacy CBM numerical outputs as inputs.

## Frozen analysis principles

- Authoritative primary dataset: OpenNeuro `ds007822` snapshot `v1.0.0`.
- Behavioral label: triad-level `CCC` vs `Other`, reconstructed from the three
  participant-specific `choice` columns in decision `events.tsv`.
- Primary task windows:
  - decision: 0-4 s after the published 1-s pre-event anchor;
  - feedback: 0-2 s after the published 1-s pre-event anchor.
- Combined decision-feedback window is excluded from primary analysis.
- Primary connectivity metric: PLV.
- Neurophysiological robustness metric: magnitude of imaginary coherency.
- PLV and imaginary coherency are never combined into one arbitrary score.
- Primary frequency bands:
  - delta 1-3 Hz
  - theta 4-7 Hz
  - alpha 8-12 Hz
  - beta 14-25 Hz
  - gamma 30-45 Hz
- Feedback-delta is excluded from primary inference because the 2-s feedback
  window is too short for a defensible low-frequency primary analysis.
- Inferential unit: triad (n=11), not pooled trials.
- Label permutation is the all-unit primary screening null.
- Randomized temporal-shift and partner-shuffle nulls are applied to
  label-supported candidates, avoiding arbitrary top-K screening.
- Ground-truth simulation is required.
- No manuscript-body drafting should begin until all planned analyses are done.

## Directory assumptions

Primary public dataset:

`data/raw/openneuro_ds007822`

Outputs:

`results/...`

## Install

Activate the project virtual environment, then:

```bash
python3 -m pip install -r requirements_bt.txt
```

For reproducibility after successful installation:

```bash
python3 scripts/freeze_environment.py
```

## Validation-first execution order

Do not run the full experiment first.

1. Validate preprocessing on G01 only:

```bash
python3 scripts/00_validate_preprocessing_g01.py
```

2. If validation passes, preprocess all 33 participants:

```bash
python3 scripts/01_run_preprocessing_all.py
```

3. Compute observed trial-level connectivity:

```bash
python3 scripts/02_compute_observed.py
```

4. Run all-unit label-permutation null:

```bash
python3 scripts/03_run_label_null.py
```

5. Run temporal-shift and partner-shuffle nulls on label-supported candidates:

```bash
python3 scripts/04_run_candidate_nulls.py
```

6. Run ground-truth simulation:

```bash
python3 scripts/05_run_simulation.py
```

7. Run triad-level stability analyses:

```bash
python3 scripts/06_run_stability.py
```

8. External validation is adapter-based and is intentionally not run until the
external dataset location/structure is audited:

```bash
python3 scripts/07_run_external_validation.py
```

After every module has been individually validated, the full orchestrator is:

```bash
python3 scripts/run_all_bt.py
```

## Computational strategy

The expensive temporal and partner nulls are **candidate-level**, not run on all
9,747 primary PLV units. This is deliberate:

1. all units undergo label-permutation screening;
2. candidates are defined by pre-specified FDR, not top-K;
3. only those candidates undergo temporal-alignment and partner-specificity
   diagnostics.

This is both statistically cleaner and substantially more efficient.

## Important preprocessing note

The OpenNeuro task `.set` files are treated as the released task-epoch input.
Participants are always processed separately (19 channels each). The code never
average-references all 57 hyperscanning channels together.

The default cleaning stage applies participant-local average reference and
ICA/ICLabel artifact rejection. Optional extra broad-band task filtering is
available but is OFF by default; downstream connectivity uses explicit
band-specific filtering with reflection padding. This avoids silently applying
a second broad-band filter to an already released public task dataset.

## Outputs

Key outputs are written under:

- `results/preprocessing/`
- `results/observed/`
- `results/nulls/`
- `results/simulation/`
- `results/stability/`
- `results/external/`
- `results/logs/`

The analysis configuration used for every run is copied into the corresponding
result directory.
