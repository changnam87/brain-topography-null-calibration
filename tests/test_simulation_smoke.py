from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt.simulation import (
    balanced_identical_labels,
    compute_plv_cube_from_phase,
    flatten_truth,
    flatten_units,
    generate_phase_dataset,
)


def test_groundtruth_simulation_smoke():
    rng = np.random.default_rng(123)

    labels = balanced_identical_labels(
        n_groups=2,
        n_trials=6,
        block_size=2,
        rng=rng,
    )

    simcfg = {
        "participants_per_group": 3,
        "channels_per_participant": 4,
        "n_time_samples": 120,
        "phase_sampling_hz": 60.0,
        "carrier_hz": 8.0,
        "frequency_jitter_hz": 0.15,
        "true_edges": [
            {
                "dyad": "pair13",
                "ch1": 0,
                "ch2": 1,
                "lag_rad": 0.5,
            }
        ],
    }

    Z, truth = generate_phase_dataset(
        labels=labels,
        simcfg=simcfg,
        scenario="sparse_true",
        coupling_strength=0.7,
        shared_strength=0.0,
        phase_noise_sd=0.10,
        seed=456,
    )

    assert Z.shape == (2, 6, 3, 4, 120)
    assert truth.shape == (3, 4, 4)
    assert int(truth.sum()) == 1

    plv = compute_plv_cube_from_phase(Z)
    assert plv.shape == (2, 6, 3, 4, 4)
    assert np.isfinite(plv).all()
    assert np.all((plv >= 0.0) & (plv <= 1.0 + 1e-6))

    flat = flatten_units(plv)
    flat_truth = flatten_truth(truth)

    assert flat.shape == (2, 6, 48)
    assert flat_truth.shape == (48,)
    assert int(flat_truth.sum()) == 1
