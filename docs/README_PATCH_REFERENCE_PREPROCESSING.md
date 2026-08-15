# BT reference-preprocessing patch

This patch replaces the initial custom BT ICA/ICLabel layer with the exact
public preprocessing path that was validated from the Data in Brief repository.

## Important changes

1. `src/bt/preprocessing.py` now imports:
   - `preprocess_bids.py`
   - `preprocessing_core.py`
   from the locally cloned public reference repository.

2. The reference pipeline is checked against the validated constants:
   - 19-channel order:
     `P3,C3,F3,Fz,F4,C4,P4,Cz,Pz,Fp1,Fp2,T3,T5,O1,O2,F7,F8,T6,T4`
   - ICA components = 15
   - random state = 42
   - max iterations = 500
   - reject labels = eye blink / muscle artifact
   - reject probability = 0.80

3. Decision and feedback are processed separately, exactly as in the validated
   reference workflow.

4. Saved cleaned numeric values are explicitly tagged as `uV`.

5. Old custom cleaned NPZ files are rejected by `load_cleaned_subject()` unless
   they contain `preprocessing_mode=published_DataInBrief_reference`.

## Before applying the patch

Archive the three old custom G01 cleaned files:

```bash
cd .
mkdir -p data/processed/cleaned_bt_legacy_custom
mv data/processed/cleaned_bt/sub-G01S0*_cleaned_bt* \
   data/processed/cleaned_bt_legacy_custom/ 2>/dev/null || true
```

## Apply

Unzip this patch at the project root, overwriting the named files.

## Validate

```bash
source .venv/bin/activate
python3 scripts/00_validate_preprocessing_g01.py
```

Expected deterministic ICA-rejection counts from the already validated public
reference run:

- S01 decision 1, feedback 2
- S02 decision 3, feedback 2
- S03 decision 2, feedback 2

Do not run all 33 participants until this wrapper validation passes.
