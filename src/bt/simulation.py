from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score

from .connectivity import DYADS, DYAD_NAMES
from .statistics import (
    apply_trial_coefficients,
    block_contrast_coefficients,
    draw_within_block_permuted_coefficients_for_group,
)


@dataclass(frozen=True)
class SimScenario:
    name: str
    labels_mode: str
    has_true_edges: bool
    shared_mode: str  # "none", "global", "group"


def scenario_spec(name: str) -> SimScenario:
    table = {
        "independent_null": SimScenario(
            name, "empirical", False, "none"
        ),
        "global_shared_event": SimScenario(
            name, "balanced_identical", False, "global"
        ),
        "group_shared_event": SimScenario(
            name, "empirical", False, "group"
        ),
        "sparse_true": SimScenario(
            name, "empirical", True, "none"
        ),
        "sparse_true_plus_group_shared": SimScenario(
            name, "empirical", True, "group"
        ),
    }
    if name not in table:
        raise ValueError(f"Unknown simulation scenario: {name}")
    return table[name]


def balanced_identical_labels(
    n_groups: int,
    n_trials: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_trials % block_size != 0:
        raise ValueError("n_trials must be divisible by block_size")
    if block_size % 2 != 0:
        raise ValueError("balanced identical labels require even block size")
    one = []
    for _ in range(n_trials // block_size):
        block = np.array(
            [1] * (block_size // 2) + [0] * (block_size // 2),
            dtype=np.uint8,
        )
        rng.shuffle(block)
        one.append(block)
    y = np.concatenate(one)
    return np.tile(y[None, :], (n_groups, 1))


def unit_complex(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z)
    mag = np.abs(z)
    mag = np.where(mag == 0, 1.0, mag)
    return (z / mag).astype(np.complex64)


def mix_toward(
    base: np.ndarray,
    target: np.ndarray,
    strength: float,
) -> np.ndarray:
    s = float(strength)
    if s <= 0:
        return base
    if s >= 1:
        return unit_complex(target)
    return unit_complex((1.0 - s) * base + s * target)


def _latent_phase(
    rng: np.random.Generator,
    n_time: int,
    fs: float,
    carrier_hz: float,
    phase_noise_sd: float,
    initial_phase: float | None = None,
) -> np.ndarray:
    t = np.arange(n_time, dtype=np.float64) / float(fs)
    phi0 = (
        float(initial_phase)
        if initial_phase is not None
        else float(rng.uniform(0, 2 * np.pi))
    )
    drift = np.cumsum(
        rng.normal(
            loc=0.0,
            scale=float(phase_noise_sd) * 0.35,
            size=n_time,
        )
    )
    phase = 2 * np.pi * float(carrier_hz) * t + phi0 + drift
    return np.exp(1j * phase).astype(np.complex64)


def generate_phase_dataset(
    labels: np.ndarray,
    simcfg: dict,
    scenario: str,
    coupling_strength: float,
    shared_strength: float,
    phase_noise_sd: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    Z : complex64, shape G x T x P x C x L
        Unit phase processes.
    truth : uint8, shape D x C x C
        True causal/specified sparse edge map. Shared-drive scenarios have no
        true edges unless the scenario also includes sparse_true.
    """
    spec = scenario_spec(scenario)
    rng = np.random.default_rng(int(seed))

    labels = np.asarray(labels, dtype=np.uint8)
    G, T = labels.shape
    P = int(simcfg["participants_per_group"])
    C = int(simcfg["channels_per_participant"])
    L = int(simcfg["n_time_samples"])
    fs = float(simcfg["phase_sampling_hz"])
    f0 = float(simcfg["carrier_hz"])
    fj = float(simcfg["frequency_jitter_hz"])

    # Independent narrow-band phase processes.
    tt = np.arange(L, dtype=np.float64) / fs
    freq = rng.normal(
        loc=f0,
        scale=fj,
        size=(G, T, P, C, 1),
    )
    phi0 = rng.uniform(
        0, 2 * np.pi, size=(G, T, P, C, 1)
    )
    drift = np.cumsum(
        rng.normal(
            loc=0.0,
            scale=float(phase_noise_sd),
            size=(G, T, P, C, L),
        ),
        axis=-1,
    )
    phase = (
        2 * np.pi * freq * tt.reshape(1, 1, 1, 1, -1)
        + phi0
        + drift
    )
    Z = np.exp(1j * phase).astype(np.complex64)

    # Shared event/common-drive injection on positive-label trials.
    if spec.shared_mode != "none" and float(shared_strength) > 0:
        channel_offsets = np.linspace(
            -0.40, 0.40, C, dtype=np.float64
        )

        if spec.shared_mode == "global":
            # One latent phase per trial index, shared across groups.
            global_latent = [
                _latent_phase(
                    rng, L, fs, f0, phase_noise_sd * 0.25
                )
                for _ in range(T)
            ]

        for g in range(G):
            for tr in range(T):
                if labels[g, tr] != 1:
                    continue

                if spec.shared_mode == "global":
                    latent = global_latent[tr]
                elif spec.shared_mode == "group":
                    latent = _latent_phase(
                        rng, L, fs, f0, phase_noise_sd * 0.25
                    )
                else:
                    raise AssertionError(spec.shared_mode)

                for p in range(P):
                    participant_offset = 0.08 * (p - 1)
                    for c in range(C):
                        target = latent * np.exp(
                            1j * (
                                participant_offset
                                + channel_offsets[c]
                            )
                        )
                        Z[g, tr, p, c] = mix_toward(
                            Z[g, tr, p, c],
                            target,
                            shared_strength,
                        )

    # Sparse true coupling injection on positive-label trials.
    truth = np.zeros((3, C, C), dtype=np.uint8)

    if spec.has_true_edges and float(coupling_strength) > 0:
        for edge in simcfg["true_edges"]:
            dyad_name = str(edge["dyad"])
            di = DYAD_NAMES.index(dyad_name)
            a, b = DYADS[di]
            c1 = int(edge["ch1"])
            c2 = int(edge["ch2"])
            lag = float(edge["lag_rad"])

            if c1 >= C or c2 >= C:
                raise ValueError(
                    f"True edge channel outside C={C}: {edge}"
                )

            truth[di, c1, c2] = 1

            for g in range(G):
                for tr in range(T):
                    if labels[g, tr] != 1:
                        continue

                    latent = _latent_phase(
                        rng,
                        L,
                        fs,
                        f0,
                        phase_noise_sd * 0.20,
                    )
                    Z[g, tr, a, c1] = mix_toward(
                        Z[g, tr, a, c1],
                        latent,
                        coupling_strength,
                    )
                    Z[g, tr, b, c2] = mix_toward(
                        Z[g, tr, b, c2],
                        latent * np.exp(1j * lag),
                        coupling_strength,
                    )

    return Z, truth


def compute_plv_cube_from_phase(
    Z: np.ndarray,
) -> np.ndarray:
    """
    Z: G x T x P x C x L
    return: G x T x D x C x C
    """
    Z = np.asarray(Z)
    G, T, P, C, L = Z.shape
    out = np.empty((G, T, 3, C, C), dtype=np.float32)

    for di, (a, b) in enumerate(DYADS):
        cross = np.einsum(
            "gtci,gtdi->gtcd",
            Z[:, :, a],
            np.conj(Z[:, :, b]),
            optimize=True,
        ) / L
        out[:, :, di] = np.abs(cross).astype(np.float32)

    return out


def flatten_units(plv_cube: np.ndarray) -> np.ndarray:
    """
    G x T x D x C x C -> G x T x U
    """
    G, T, D, C, _ = plv_cube.shape
    return plv_cube.reshape(G, T, D * C * C)


def flatten_truth(truth: np.ndarray) -> np.ndarray:
    return np.asarray(truth, dtype=np.uint8).reshape(-1)


def label_null_maxT(
    X: np.ndarray,
    labels: np.ndarray,
    block_size: int,
    B: int,
    seed: int,
) -> dict:
    """
    X: G x T x U
    """
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    G, T, U = X.shape

    coeff, total_w = block_contrast_coefficients(
        labels, block_size=block_size
    )
    observed = apply_trial_coefficients(
        X.astype(np.float64), coeff
    ).astype(np.float64)

    rng = np.random.default_rng(int(seed))
    null = np.zeros((B, U), dtype=np.float32)

    for g in range(G):
        Wg = draw_within_block_permuted_coefficients_for_group(
            labels[g],
            B,
            rng,
            global_total_weight=total_w,
            block_size=block_size,
        )
        null += Wg @ X[g]

    mu = null.mean(axis=0, dtype=np.float64)
    sd = null.std(axis=0, ddof=1, dtype=np.float64)
    sd = np.where(sd > 1e-12, sd, np.nan)

    oz = (observed - mu) / sd
    nz = (null.astype(np.float64) - mu[None, :]) / sd[None, :]
    nz = np.where(np.isfinite(nz), nz, 0.0)

    unadj_exceed = np.sum(
        np.abs(nz) >= np.abs(oz)[None, :], axis=0
    )
    p_unadj = (1 + unadj_exceed) / (B + 1)

    maxz = np.max(np.abs(nz), axis=1)
    sorted_max = np.sort(maxz)
    left = np.searchsorted(
        sorted_max, np.abs(oz), side="left"
    )
    p_maxT = (1 + (B - left)) / (B + 1)

    return {
        "observed": observed,
        "observed_z": oz,
        "p_unadjusted": p_unadj,
        "p_maxT": p_maxT,
        "coeff": coeff,
        "null_mean": mu,
        "null_sd": sd,
        "null_max_abs_z": maxz,
    }


def _valid_lags(n_time: int, min_fraction: float = 0.15):
    min_lag = max(1, int(round(n_time * min_fraction)))
    lags = np.arange(n_time)
    d = np.minimum(lags, n_time - lags)
    valid = lags[d >= min_lag]
    if len(valid) == 0:
        raise ValueError("No valid circular lags")
    return valid


def _candidate_time_null(
    Z: np.ndarray,
    candidate_units: list[int],
    C: int,
    coeff: np.ndarray,
    B: int,
    seed: int,
) -> np.ndarray:
    """
    Return B x K candidate null statistics.
    """
    G, T, P, C0, L = Z.shape
    if C0 != C:
        raise ValueError("C mismatch")
    rng = np.random.default_rng(int(seed))
    valid_lags = _valid_lags(L)
    K = len(candidate_units)
    out = np.zeros((B, K), dtype=np.float32)

    for k, u in enumerate(candidate_units):
        di = u // (C * C)
        rem = u % (C * C)
        c1 = rem // C
        c2 = rem % C
        a, b = DYADS[di]

        total = np.zeros(B, dtype=np.float32)

        for g in range(G):
            z1 = Z[g, :, a, c1, :]  # T x L
            z2 = Z[g, :, b, c2, :]

            A = np.fft.fft(z1, axis=1)
            Bf = np.fft.fft(z2, axis=1)
            corr = np.fft.ifft(
                A * np.conj(Bf), axis=1
            ) / L
            lag_plv = np.abs(corr).astype(np.float32)

            lag_idx = rng.choice(
                valid_lags, size=(B, T), replace=True
            )
            vals = lag_plv[
                np.arange(T)[None, :],
                lag_idx,
            ]
            total += vals @ coeff[g].astype(np.float32)

        out[:, k] = total

    return out


def _derangement(rng: np.random.Generator, n: int):
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    raise RuntimeError("Could not draw derangement")


def _candidate_partner_null(
    Z: np.ndarray,
    candidate_units: list[int],
    C: int,
    coeff: np.ndarray,
    B: int,
    seed: int,
) -> np.ndarray:
    G, T, P, C0, L = Z.shape
    if C0 != C:
        raise ValueError("C mismatch")
    rng = np.random.default_rng(int(seed))
    K = len(candidate_units)
    out = np.empty((B, K), dtype=np.float32)

    # Precompute cross-group candidate PLVs.
    cross_all = []
    for u in candidate_units:
        di = u // (C * C)
        rem = u % (C * C)
        c1 = rem // C
        c2 = rem % C
        a, b = DYADS[di]

        cross = np.empty((G, G, T), dtype=np.float32)
        for g in range(G):
            za = Z[g, :, a, c1, :]
            for h in range(G):
                zb = Z[h, :, b, c2, :]
                cross[g, h] = np.abs(
                    np.mean(za * np.conj(zb), axis=1)
                ).astype(np.float32)
        cross_all.append(cross)

    for r in range(B):
        perm = _derangement(rng, G)
        for k, cross in enumerate(cross_all):
            val = 0.0
            for g in range(G):
                val += float(
                    np.dot(cross[g, perm[g]], coeff[g])
                )
            out[r, k] = val

    return out


def candidate_family_maxT(
    observed: np.ndarray,
    null_matrix: np.ndarray,
) -> np.ndarray:
    observed = np.asarray(observed, dtype=float)
    null = np.asarray(null_matrix, dtype=float)
    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, np.nan)

    oz = (observed - mu) / sd
    nz = (null - mu[None, :]) / sd[None, :]
    nz = np.where(np.isfinite(nz), nz, 0.0)
    maxz = np.max(np.abs(nz), axis=1)
    sorted_max = np.sort(maxz)
    left = np.searchsorted(
        sorted_max, np.abs(oz), side="left"
    )
    return (1 + (len(maxz) - left)) / (len(maxz) + 1)


def full_framework_detection(
    Z: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    block_size: int,
    B_label: int,
    B_secondary: int,
    seed: int,
) -> dict:
    plv = compute_plv_cube_from_phase(Z)
    X = flatten_units(plv)
    C = plv.shape[-1]

    lab = label_null_maxT(
        X,
        labels,
        block_size=block_size,
        B=B_label,
        seed=seed + 100,
    )

    label_detect = lab["p_maxT"] < alpha
    candidate_units = np.where(label_detect)[0].tolist()

    p_time_full = np.ones(X.shape[-1], dtype=float)
    p_partner_full = np.ones(X.shape[-1], dtype=float)
    final_detect = np.zeros(X.shape[-1], dtype=bool)

    if candidate_units:
        coeff = lab["coeff"]
        tnull = _candidate_time_null(
            Z,
            candidate_units,
            C,
            coeff,
            B=B_secondary,
            seed=seed + 200,
        )
        pnull = _candidate_partner_null(
            Z,
            candidate_units,
            C,
            coeff,
            B=B_secondary,
            seed=seed + 300,
        )
        obs_c = lab["observed"][candidate_units]

        pt = candidate_family_maxT(obs_c, tnull)
        pp = candidate_family_maxT(obs_c, pnull)

        p_time_full[candidate_units] = pt
        p_partner_full[candidate_units] = pp

        for i, u in enumerate(candidate_units):
            final_detect[u] = (
                pt[i] < alpha and pp[i] < alpha
            )

    return {
        "plv_cube": plv,
        "X": X,
        "observed": lab["observed"],
        "p_unadjusted": lab["p_unadjusted"],
        "p_label_maxT": lab["p_maxT"],
        "label_detect": label_detect,
        "p_time_maxT": p_time_full,
        "p_partner_maxT": p_partner_full,
        "final_detect": final_detect,
    }


def detection_metrics(
    truth: np.ndarray,
    result: dict,
    alpha: float,
) -> dict:
    truth = np.asarray(truth, dtype=np.uint8).reshape(-1)
    raw = np.abs(result["observed"])
    p_unadj = np.asarray(result["p_unadjusted"])
    p_label = np.asarray(result["p_label_maxT"])
    label_detect = np.asarray(result["label_detect"], dtype=bool)
    final_detect = np.asarray(result["final_detect"], dtype=bool)

    naive = p_unadj < alpha

    def counts(det):
        det = np.asarray(det, dtype=bool)
        tp = int(np.sum(det & (truth == 1)))
        fp = int(np.sum(det & (truth == 0)))
        fn = int(np.sum((~det) & (truth == 1)))
        tn = int(np.sum((~det) & (truth == 0)))
        sens = tp / (tp + fn) if tp + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        fpr = fp / (fp + tn) if fp + tn else np.nan
        return tp, fp, fn, tn, sens, spec, prec, fpr

    nt = int(truth.sum())
    raw_ap = (
        float(average_precision_score(truth, raw))
        if nt > 0 else np.nan
    )
    label_score = -np.log10(
        np.maximum(p_label, 1e-12)
    )
    label_ap = (
        float(average_precision_score(truth, label_score))
        if nt > 0 else np.nan
    )

    if nt > 0:
        topk = np.argsort(raw)[-nt:]
        raw_topk_recall = float(
            np.sum(truth[topk] == 1) / nt
        )
    else:
        raw_topk_recall = np.nan

    out = {
        "n_true_edges": nt,
        "raw_auprc": raw_ap,
        "label_score_auprc": label_ap,
        "raw_topk_recall": raw_topk_recall,
        "naive_any_detection": bool(np.any(naive)),
        "label_any_detection": bool(np.any(label_detect)),
        "full_any_detection": bool(np.any(final_detect)),
    }

    for name, det in (
        ("naive", naive),
        ("label", label_detect),
        ("full", final_detect),
    ):
        tp, fp, fn, tn, sens, spec, prec, fpr = counts(det)
        out.update({
            f"{name}_tp": tp,
            f"{name}_fp": fp,
            f"{name}_fn": fn,
            f"{name}_tn": tn,
            f"{name}_sensitivity": sens,
            f"{name}_specificity": spec,
            f"{name}_precision": prec,
            f"{name}_fpr": fpr,
            f"{name}_n_detected": int(np.sum(det)),
        })

    return out
