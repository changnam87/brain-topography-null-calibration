from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt.connectivity import plv_trial_matrices_from_phase, edge_index


def test_plv_identical_unit_phase_is_one():
    t = np.linspace(0.0, 1.0, 500, endpoint=False)
    z = np.exp(1j * 2 * np.pi * 6 * t).astype(np.complex64)

    z1 = z[None, None, :]
    z2 = z[None, None, :]

    plv = plv_trial_matrices_from_phase(z1, z2)

    assert plv.shape == (1, 1, 1)
    assert np.allclose(plv, 1.0, atol=1e-6)


def test_edge_index_covers_all_cross_brain_pairs():
    ii, jj = edge_index(4)

    assert ii.shape == (16,)
    assert jj.shape == (16,)
    assert set(zip(ii.tolist(), jj.tolist())) == {
        (i, j) for i in range(4) for j in range(4)
    }
