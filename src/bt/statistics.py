from __future__ import annotations

import numpy as np


def information_weight(n1: int, n0: int) -> float:
    if n1 <= 0 or n0 <= 0:
        return 0.0
    return float(n1 * n0 / (n1 + n0))


def _normalize_mask(labels: np.ndarray, retain_mask: np.ndarray | None):
    labels = np.asarray(labels, dtype=np.uint8)
    if labels.ndim != 2:
        raise ValueError(f"labels must be GxT, got {labels.shape}")
    if retain_mask is None:
        return np.ones_like(labels, dtype=bool)
    m = np.asarray(retain_mask, dtype=bool)
    if m.shape != labels.shape:
        raise ValueError(f"mask shape {m.shape} != labels shape {labels.shape}")
    return m


def block_information_weights(
    labels: np.ndarray,
    block_size: int = 10,
    retain_mask: np.ndarray | None = None,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.uint8)
    keep = _normalize_mask(labels, retain_mask)
    G, T = labels.shape
    if T % block_size != 0:
        raise ValueError(f"T={T} not divisible by block_size={block_size}")
    B = T // block_size
    out = np.zeros((G, B), dtype=np.float64)
    for g in range(G):
        for b in range(B):
            sl = slice(b * block_size, (b + 1) * block_size)
            use = keep[g, sl]
            y = labels[g, sl][use]
            n1 = int(np.sum(y == 1))
            n0 = int(np.sum(y == 0))
            out[g, b] = information_weight(n1, n0)
    return out


def block_contrast_coefficients(
    labels: np.ndarray,
    block_size: int = 10,
    retain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Return normalized trial coefficients C (G x T) such that

        statistic = sum_{g,t} C[g,t] * X[g,t]

    equals the information-weighted mean of within-block CCC-Other contrasts.

    Non-informative blocks and excluded trials receive zero weight.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    keep = _normalize_mask(labels, retain_mask)
    G, T = labels.shape
    if T % block_size != 0:
        raise ValueError(f"T={T} not divisible by block_size={block_size}")

    weights = block_information_weights(labels, block_size, keep)
    total_w = float(np.sum(weights))
    if total_w <= 0:
        raise ValueError("No informative block contrast remains")

    coeff = np.zeros((G, T), dtype=np.float64)
    n_blocks = T // block_size

    for g in range(G):
        for b in range(n_blocks):
            w = float(weights[g, b])
            if w <= 0:
                continue
            sl = slice(b * block_size, (b + 1) * block_size)
            use = keep[g, sl]
            y = labels[g, sl]
            idx_local = np.arange(block_size)
            pos = idx_local[use & (y == 1)]
            neg = idx_local[use & (y == 0)]
            n1, n0 = len(pos), len(neg)
            scale = w / total_w
            coeff[g, b * block_size + pos] = scale / n1
            coeff[g, b * block_size + neg] = -scale / n0

    return coeff, total_w


def triad_information_coefficients(
    labels: np.ndarray,
    retain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Secondary unblocked information-weighted CCC-Other contrast.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    keep = _normalize_mask(labels, retain_mask)
    G, T = labels.shape
    w = np.zeros(G, dtype=np.float64)

    for g in range(G):
        y = labels[g][keep[g]]
        w[g] = information_weight(int(np.sum(y == 1)), int(np.sum(y == 0)))

    total_w = float(np.sum(w))
    if total_w <= 0:
        raise ValueError("No informative triad contrast remains")

    coeff = np.zeros((G, T), dtype=np.float64)
    for g in range(G):
        if w[g] <= 0:
            continue
        pos = np.where(keep[g] & (labels[g] == 1))[0]
        neg = np.where(keep[g] & (labels[g] == 0))[0]
        scale = w[g] / total_w
        coeff[g, pos] = scale / len(pos)
        coeff[g, neg] = -scale / len(neg)

    return coeff, total_w


def equal_triad_coefficients(
    labels: np.ndarray,
    retain_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Secondary equal-triad mean of within-triad CCC-Other differences.
    Triads missing either class receive zero weight; remaining informative
    triads are normalized to equal weight.
    """
    labels = np.asarray(labels, dtype=np.uint8)
    keep = _normalize_mask(labels, retain_mask)
    G, T = labels.shape
    informative = []

    for g in range(G):
        y = labels[g][keep[g]]
        informative.append(np.any(y == 1) and np.any(y == 0))

    n_inf = int(np.sum(informative))
    if n_inf == 0:
        raise ValueError("No informative triads")

    coeff = np.zeros((G, T), dtype=np.float64)
    for g in range(G):
        if not informative[g]:
            continue
        pos = np.where(keep[g] & (labels[g] == 1))[0]
        neg = np.where(keep[g] & (labels[g] == 0))[0]
        coeff[g, pos] = (1.0 / n_inf) / len(pos)
        coeff[g, neg] = -(1.0 / n_inf) / len(neg)

    return coeff


def apply_trial_coefficients(
    trial_values: np.ndarray,
    coeff: np.ndarray,
) -> np.ndarray:
    """
    trial_values: G x T x ... (any trailing unit dimensions)
    coeff:        G x T
    """
    x = np.asarray(trial_values)
    c = np.asarray(coeff, dtype=np.float64)
    if x.shape[:2] != c.shape:
        raise ValueError(f"trial values {x.shape[:2]} != coeff {c.shape}")
    return np.tensordot(c, x, axes=([0, 1], [0, 1]))


def group_block_effects(
    trial_values: np.ndarray,
    labels: np.ndarray,
    block_size: int = 10,
    retain_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For stability analysis, return one block-adjusted effect per triad plus
    each triad's total within-block information weight.

    trial_values: G x T x U
    returns:
      effects: G x U (NaN for triads with no informative block)
      weights: G
    """
    x = np.asarray(trial_values)
    labels = np.asarray(labels, dtype=np.uint8)
    keep = _normalize_mask(labels, retain_mask)
    G, T = labels.shape
    if x.shape[:2] != (G, T):
        raise ValueError(f"X shape {x.shape[:2]} != labels {(G,T)}")

    n_blocks = T // block_size
    effects = np.full((G,) + x.shape[2:], np.nan, dtype=np.float64)
    gw = np.zeros(G, dtype=np.float64)

    for g in range(G):
        pieces = []
        weights = []
        for b in range(n_blocks):
            sl = slice(b * block_size, (b + 1) * block_size)
            use = keep[g, sl]
            y = labels[g, sl][use]
            if len(y) == 0:
                continue
            n1 = int(np.sum(y == 1))
            n0 = int(np.sum(y == 0))
            w = information_weight(n1, n0)
            if w <= 0:
                continue
            xb = x[g, sl][use]
            d = xb[y == 1].mean(axis=0) - xb[y == 0].mean(axis=0)
            pieces.append(d)
            weights.append(w)

        if weights:
            ww = np.asarray(weights, dtype=np.float64)
            stack = np.stack(pieces, axis=0)
            effects[g] = np.tensordot(ww / ww.sum(), stack, axes=(0, 0))
            gw[g] = ww.sum()

    return effects, gw


def draw_within_block_permuted_coefficients_for_group(
    labels_g: np.ndarray,
    n_draws: int,
    rng: np.random.Generator,
    global_total_weight: float,
    block_size: int = 10,
    retain_mask_g: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate B x T normalized contrast coefficients for one triad under
    within-block restricted label permutation. Class counts and retained trial
    positions are preserved within every block.

    global_total_weight must be the observed sum of information weights across
    all triad x block strata. Because blockwise counts are fixed, this
    denominator is constant under permutation.
    """
    y0 = np.asarray(labels_g, dtype=np.uint8)
    T = len(y0)
    if T % block_size != 0:
        raise ValueError("T must be divisible by block size")
    keep = (
        np.ones(T, dtype=bool)
        if retain_mask_g is None
        else np.asarray(retain_mask_g, dtype=bool)
    )
    W = np.zeros((n_draws, T), dtype=np.float32)

    for b in range(T // block_size):
        st = b * block_size
        en = st + block_size
        local_keep = np.where(keep[st:en])[0]
        if len(local_keep) == 0:
            continue
        local_y = y0[st:en][local_keep]
        n1 = int(np.sum(local_y == 1))
        n0 = int(np.sum(local_y == 0))
        w = information_weight(n1, n0)
        if w <= 0:
            continue
        scale = w / global_total_weight

        for r in range(n_draws):
            perm = rng.permutation(local_y)
            pos = local_keep[perm == 1]
            neg = local_keep[perm == 0]
            W[r, st + pos] = scale / n1
            W[r, st + neg] = -scale / n0

    return W
