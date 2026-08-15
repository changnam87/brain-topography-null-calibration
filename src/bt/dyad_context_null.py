from __future__ import annotations

import numpy as np

from .connectivity import DYAD_NAMES
from .simulation import candidate_family_maxT


def decode_unit(unit_index: int, n_channels: int):
    """
    Simulation unit ordering:
      dyad-major, then ch1-major, then ch2.
    """
    C = int(n_channels)
    u = int(unit_index)
    dyad_i = u // (C * C)
    rem = u % (C * C)
    ch1 = rem // C
    ch2 = rem % C
    if not (0 <= dyad_i < 3 and 0 <= ch1 < C and 0 <= ch2 < C):
        raise ValueError(f"Bad unit index {unit_index} for C={C}")
    return dyad_i, ch1, ch2


def within_triad_dyad_null(
    plv_cube: np.ndarray,
    candidate_units: list[int],
    coeff: np.ndarray,
    n_realizations: int,
    seed: int,
):
    """
    Triad-preserving dyad-label randomization null.

    Parameters
    ----------
    plv_cube : G x T x 3 x C x C
        Trial-level PLV for all three dyads.
    candidate_units : fixed label-supported candidate unit indices
    coeff : G x T
        Frozen block-information-weighted trial coefficients.
    n_realizations : number of null draws
    seed : RNG seed

    Null logic
    ----------
    For each candidate, group, trial, and realization, randomly assign one of
    the three within-triad dyads to occupy the candidate dyad role while keeping:
      - the same triad,
      - the same trial,
      - the same task/condition timing,
      - the same channel pair.

    Thus a triad-wide common drive remains present in the surrogate data,
    unlike cross-group partner shuffling. Stable dyad identity is broken.

    The observed dyad is deliberately allowed among the 3 random assignments.
    This makes the null conservative and corresponds to exchangeability of dyad
    identity under the null hypothesis of no dyad-specific effect.

    Returns
    -------
    null_matrix : R x K
    """
    plv = np.asarray(plv_cube, dtype=np.float32)
    coeff = np.asarray(coeff, dtype=np.float32)

    if plv.ndim != 5 or plv.shape[2] != 3 or plv.shape[3] != plv.shape[4]:
        raise ValueError(f"Expected GxTx3xCxC PLV cube, got {plv.shape}")
    G, T, _, C, _ = plv.shape
    if coeff.shape != (G, T):
        raise ValueError(f"coeff {coeff.shape} != {(G,T)}")

    K = len(candidate_units)
    R = int(n_realizations)
    rng = np.random.default_rng(int(seed))
    out = np.zeros((R, K), dtype=np.float32)

    for k, u in enumerate(candidate_units):
        _, ch1, ch2 = decode_unit(u, C)
        # G x T x 3
        vals = plv[:, :, :, ch1, ch2]

        total = np.zeros(R, dtype=np.float32)
        for g in range(G):
            # R x T random dyad identities, preserving same triad/trial.
            draw = rng.integers(0, 3, size=(R, T))
            selected = vals[g][
                np.arange(T)[None, :],
                draw,
            ]
            total += selected @ coeff[g]

        out[:, k] = total

    return out


def within_triad_dyad_maxT(
    observed: np.ndarray,
    plv_cube: np.ndarray,
    candidate_units: list[int],
    coeff: np.ndarray,
    n_realizations: int,
    seed: int,
):
    null = within_triad_dyad_null(
        plv_cube,
        candidate_units,
        coeff,
        n_realizations,
        seed,
    )
    p = candidate_family_maxT(
        np.asarray(observed, dtype=float),
        null,
    )
    return p, null
