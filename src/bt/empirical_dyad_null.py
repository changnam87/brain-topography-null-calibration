from __future__ import annotations

import numpy as np


def decode_empirical_unit(
    unit_index: int,
    n_conditions: int,
    n_dyads: int,
    n_edges: int,
):
    """
    Unit ordering used by the empirical BT cube:
      condition-major -> dyad-major -> edge-major.
    """
    u = int(unit_index)
    per_condition = int(n_dyads) * int(n_edges)
    ci = u // per_condition
    rem = u % per_condition
    di = rem // int(n_edges)
    ei = rem % int(n_edges)

    if not (
        0 <= ci < int(n_conditions)
        and 0 <= di < int(n_dyads)
        and 0 <= ei < int(n_edges)
    ):
        raise ValueError(
            f"Bad unit {unit_index} for "
            f"C={n_conditions}, D={n_dyads}, E={n_edges}"
        )
    return ci, di, ei


def empirical_within_triad_dyad_null(
    trial_cube: np.ndarray,
    candidate_units: list[int],
    coeff: np.ndarray,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    """
    Triad-preserving dyad-identity randomization for the empirical PLV cube.

    Parameters
    ----------
    trial_cube : G x C x D x T x E
        Primary trial-level PLV cube.
    candidate_units : fixed Stage-3D candidate unit indices
    coeff : G x T
        Frozen block-information-weighted contrast coefficients.
    n_realizations : number of dyad-randomization draws
    seed : RNG seed

    Null logic
    ----------
    For each candidate, same triad + same trial + same task-band condition +
    same channel pair are retained. Only which of the three within-triad dyads
    occupies the candidate-dyad role is randomized.

    This preserves triad-specific common drive and trial timing while breaking
    stable dyad identity.

    Returns
    -------
    R x K null-statistic matrix
    """
    X = np.asarray(trial_cube, dtype=np.float32)
    if X.ndim != 5:
        raise ValueError(f"Expected GxCxDxTxE cube, got {X.shape}")

    G, C, D, T, E = X.shape
    if D != 3:
        raise ValueError(f"Expected 3 dyads, got {D}")

    coeff = np.asarray(coeff, dtype=np.float32)
    if coeff.shape != (G, T):
        raise ValueError(f"coeff {coeff.shape} != {(G,T)}")

    R = int(n_realizations)
    K = len(candidate_units)
    rng = np.random.default_rng(int(seed))
    out = np.zeros((R, K), dtype=np.float32)

    for k, u in enumerate(candidate_units):
        ci, _di, ei = decode_empirical_unit(
            u, C, D, E
        )

        # G x D x T for same condition + channel pair.
        vals = X[:, ci, :, :, ei]

        total = np.zeros(R, dtype=np.float32)
        for g in range(G):
            # R x T random within-triad dyad identities.
            draw = rng.integers(0, D, size=(R, T))
            # vals[g]: D x T -> transpose T x D for advanced indexing.
            v = vals[g].T  # T x D
            selected = v[
                np.arange(T)[None, :],
                draw,
            ]  # R x T
            total += selected @ coeff[g]

        out[:, k] = total

    return out


def studentized_candidate_family_maxT(
    observed: np.ndarray,
    null_matrix: np.ndarray,
):
    """
    Candidate-family studentized maxT FWER p-values.
    """
    obs = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null_matrix, dtype=np.float64)

    if null.ndim != 2 or null.shape[1] != len(obs):
        raise ValueError(
            f"null shape {null.shape} incompatible with observed {obs.shape}"
        )

    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.nan)

    oz = (obs - mu) / sd
    nz = (null - mu[None, :]) / sd[None, :]
    nz = np.where(np.isfinite(nz), nz, 0.0)

    max_abs = np.max(np.abs(nz), axis=1)
    sorted_max = np.sort(max_abs)
    left = np.searchsorted(
        sorted_max, np.abs(oz), side="left"
    )
    p = (1 + (len(max_abs) - left)) / (len(max_abs) + 1)

    return {
        "p_maxT": p.astype(np.float64),
        "observed_studentized": oz.astype(np.float64),
        "null_mean": mu.astype(np.float64),
        "null_sd": sd.astype(np.float64),
        "null_max_abs_studentized": max_abs.astype(np.float64),
    }
