from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def boolstr(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def load_fixed_final3(stage4d2_csv: str | Path) -> list[dict[str, str]]:
    """
    Load the final PLV family frozen by Stage 4D2.

    The family is defined ONLY by all_four_null_layers_pass=True.  Stage 6
    never re-selects candidates from stability results.
    """
    rows = read_csv(stage4d2_csv)
    fixed = [r for r in rows if boolstr(r.get("all_four_null_layers_pass", False))]
    if len(fixed) != 3:
        raise RuntimeError(
            "Expected exactly 3 Stage-4D2 final PLV candidates with "
            f"all_four_null_layers_pass=True; got {len(fixed)}"
        )
    return fixed


def weighted_mean_rows(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean across rows of values (N x K)."""
    x = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"values must be 2D, got {x.shape}")
    if w.shape != (x.shape[0],):
        raise ValueError(f"weights {w.shape} incompatible with values {x.shape}")
    if not np.isfinite(x).all() or not np.isfinite(w).all():
        raise ValueError("values/weights must be finite")
    if np.any(w < 0) or np.sum(w) <= 0:
        raise ValueError("weights must be nonnegative with positive total")
    return np.sum(x * w[:, None], axis=0) / np.sum(w)


def _same_direction(x: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """True only for nonzero estimates with the same sign as reference."""
    x = np.asarray(x)
    r = np.asarray(reference)
    return (np.sign(x) == np.sign(r)) & (np.sign(r) != 0)


def compute_final3_stability(
    group_effects: np.ndarray,
    group_information_weights: np.ndarray,
    observed_effects: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Triad-level stability for a FIXED candidate family.

    Parameters
    ----------
    group_effects : G x K
        Within-triad block-information-weighted effects from Stage 3D.
    group_information_weights : G
        Triad total information weights used by the primary estimator.
    observed_effects : K
        Frozen primary block-information-weighted effects.
    n_bootstrap : int
        Number of triad-cluster bootstrap draws.
    seed : int
        RNG seed.

    Notes
    -----
    The primary estimator is a group-information-weighted average.  Both LOTO
    and bootstrap therefore retain the same weighting rule, rather than using
    an unweighted mean of triad effects.

    Bootstrap resamples triads (clusters) with replacement; when a triad is
    sampled multiple times, its information weight is duplicated with it.
    This is a descriptive stability/sensitivity analysis, not a new candidate
    selection or confirmatory significance test.
    """
    ge = np.asarray(group_effects, dtype=np.float64)
    gw = np.asarray(group_information_weights, dtype=np.float64)
    obs = np.asarray(observed_effects, dtype=np.float64)

    if ge.ndim != 2:
        raise ValueError(f"group_effects must be GxK, got {ge.shape}")
    G, K = ge.shape
    if gw.shape != (G,):
        raise ValueError(f"group_information_weights {gw.shape} != {(G,)}")
    if obs.shape != (K,):
        raise ValueError(f"observed_effects {obs.shape} != {(K,)}")
    if G < 3:
        raise ValueError("Need at least 3 informative triads for stability analysis")
    if int(n_bootstrap) < 1:
        raise ValueError("n_bootstrap must be >= 1")

    informative = (
        np.isfinite(gw)
        & (gw > 0)
        & np.all(np.isfinite(ge), axis=1)
    )
    valid_idx = np.where(informative)[0]
    if len(valid_idx) < 3:
        raise ValueError(
            f"Only {len(valid_idx)} triads have positive information weight and "
            "finite effects"
        )

    x = ge[valid_idx]
    w = gw[valid_idx]
    n = len(valid_idx)

    reconstructed = weighted_mean_rows(x, w)

    # Leave-one-triad-out, preserving the primary information-weighting rule.
    loto = np.empty((n, K), dtype=np.float64)
    for j in range(n):
        keep = np.arange(n) != j
        loto[j] = weighted_mean_rows(x[keep], w[keep])

    loto_same = _same_direction(loto, obs[None, :])
    individual_same = _same_direction(x, obs[None, :])

    abs_dev = np.abs(loto - obs[None, :])
    influential_pos = np.argmax(abs_dev, axis=0)
    influential_group_index = valid_idx[influential_pos]
    influential_abs_change = abs_dev[influential_pos, np.arange(K)]

    # Cluster bootstrap over triads. K is only 3, so retaining all draws is cheap.
    rng = np.random.default_rng(int(seed))
    B = int(n_bootstrap)
    boot = np.empty((B, K), dtype=np.float64)
    for b in range(B):
        draw = rng.integers(0, n, size=n)
        boot[b] = weighted_mean_rows(x[draw], w[draw])

    ci_low = np.percentile(boot, 2.5, axis=0)
    ci_high = np.percentile(boot, 97.5, axis=0)
    boot_same = _same_direction(boot, obs[None, :])

    # Equal-triad summaries are included only as a secondary weighting-sensitivity
    # diagnostic; they do not replace the primary estimator.
    equal_full = np.mean(x, axis=0)
    equal_loto = np.empty((n, K), dtype=np.float64)
    for j in range(n):
        keep = np.arange(n) != j
        equal_loto[j] = np.mean(x[keep], axis=0)

    return {
        "informative_mask": informative.astype(np.uint8),
        "valid_group_indices": valid_idx.astype(np.int16),
        "group_effects": x,
        "group_information_weights": w,
        "observed": obs,
        "reconstructed_primary": reconstructed,
        "individual_same_direction": individual_same.astype(np.uint8),
        "loto": loto,
        "loto_same_direction": loto_same.astype(np.uint8),
        "loto_min": np.min(loto, axis=0),
        "loto_max": np.max(loto, axis=0),
        "loto_same_direction_count": np.sum(loto_same, axis=0).astype(np.int16),
        "loto_same_direction_fraction": np.mean(loto_same, axis=0),
        "influential_group_index": influential_group_index.astype(np.int16),
        "influential_abs_change": influential_abs_change,
        "bootstrap": boot,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "bootstrap_median": np.median(boot, axis=0),
        "bootstrap_same_direction_fraction": np.mean(boot_same, axis=0),
        "bootstrap_ci_excludes_zero": ((ci_low > 0) | (ci_high < 0)).astype(np.uint8),
        "equal_triad_full": equal_full,
        "equal_triad_loto": equal_loto,
        "equal_triad_direction_agrees": _same_direction(equal_full, obs).astype(np.uint8),
    }


def load_final3_inputs(
    label_npz: str | Path,
    stage4d2_csv: str | Path,
) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    """Load and align Stage-3D group effects with the frozen Stage-4D2 final 3."""
    fixed = load_fixed_final3(stage4d2_csv)
    ids = np.array([int(r["unit_index"]) for r in fixed], dtype=int)

    z = np.load(label_npz, allow_pickle=True)
    required = {"group_effects", "group_information_weights", "observed"}
    missing = sorted(required.difference(z.files))
    if missing:
        raise RuntimeError(f"{label_npz} missing arrays: {missing}")

    ge_all = z["group_effects"].astype(np.float64)
    gw = z["group_information_weights"].astype(np.float64)
    obs_all = z["observed"].astype(np.float64)

    if np.any(ids < 0) or np.any(ids >= ge_all.shape[1]):
        raise RuntimeError(
            f"Final candidate indices {ids.tolist()} out of range for "
            f"group_effects shape {ge_all.shape}"
        )

    obs_from_csv = np.array(
        [float(r["observed_effect"]) for r in fixed], dtype=np.float64
    )
    obs_from_npz = obs_all[ids]
    if not np.allclose(obs_from_csv, obs_from_npz, rtol=1e-5, atol=1e-7):
        raise RuntimeError(
            "Stage-4D2 observed effects do not match label_null_plv.npz for "
            f"the fixed units. CSV={obs_from_csv}, NPZ={obs_from_npz}"
        )

    return fixed, {
        "unit_ids": ids,
        "group_effects": ge_all[:, ids],
        "group_information_weights": gw,
        "observed": obs_from_npz,
    }
