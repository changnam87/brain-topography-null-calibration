# Simulation-Informed Progressive Null Calibration of Inter-Brain PLV Topographies in Triadic Cooperation

This repository contains the analysis code and frozen reference outputs for the manuscript:

**Simulation-Informed Progressive Null Calibration of Inter-Brain PLV Topographies in Triadic Cooperation**

## Analysis overview

The primary PLV search contains 9,747 predefined units spanning decision and feedback windows, conventional frequency bands, three within-triad participant pairs, and all 19 x 19 cross-brain sensor pairs.

The inferential workflow consists of:

1. global block-restricted behavior-label maxT inference;
2. temporal-shift candidate nulls;
3. cross-partner candidate nulls;
4. ground-truth simulation stress testing;
5. a simulation-motivated within-triad dyad-identity null;
6. imaginary-coherency robustness analysis;
7. triad-level stability, leave-one-triad-out analysis, and bootstrap analysis; and
8. a fail-closed evidence-freeze stage.

The frozen final PLV family contains units 3801, 4994, and 8156.

## Data

The EEG dataset is publicly available from OpenNeuro under accession `ds007822`.

Raw EEG data are not duplicated in this repository.

Place the downloaded dataset at:

`data/raw/openneuro_ds007822/`

## Reference preprocessing implementation

The preprocessing wrapper used here reproduces and validates the published preprocessing implementation distributed separately with the dataset paper.

Clone the repository `heegyukim4043/PD_EEG_hyperscan_processing` into:

`data/reference/PD_EEG_hyperscan_processing/`

## Installation

Python 3.11 is recommended.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

The portable analysis configuration is:

`configs/bt_analysis_config.json`

All paths are repository-relative by default.

## Full manuscript pipeline

```bash
python scripts/run_manuscript_pipeline.py --n-jobs 4
```

To resume the production ground-truth simulation:

```bash
python scripts/run_manuscript_pipeline.py --n-jobs 4 --resume-simulation
```

If preprocessing outputs already exist:

```bash
python scripts/run_manuscript_pipeline.py --skip-preprocessing
```

## Frozen reference outputs

Small manuscript-level reference outputs are included under `reference_results/`.

The final evidence record is:

`reference_results/freeze/stage6C_analysis_freeze_record.md`

The corresponding candidate summary is:

`reference_results/freeze/stage6C_master_evidence.csv`

## Reproducibility

The master random seed is `20260814`.

Stage-specific deterministic seed offsets are used throughout stochastic analyses.

## Interpretation boundary

The final findings are null-calibrated sensor-level PLV associations.

They should not be interpreted as source-localized anatomical connectivity, directed neural information flow, causal inter-brain coupling, or demonstrated immunity to all zero-lag/common-source contributions.

## Manuscript figures

Figures 2-5 are generated directly from saved production results:

```bash
python scripts/44_make_all_manuscript_figures.py --project .
```

Figure 1 is a conceptual vector schematic and is not generated from empirical result values.

## License

This software is released under the MIT License. See `LICENSE`.

## Citation

Citation metadata are provided in `CITATION.cff`. The archived release DOI will be added after the first GitHub release is deposited in Zenodo.
