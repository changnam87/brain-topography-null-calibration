from __future__ import annotations

import numpy as np
from scipy import signal

from .connectivity import DYADS, DYAD_NAMES
from .io import load_cleaned_subject
from .statistics import (
    apply_trial_coefficients,
    block_contrast_coefficients,
    draw_within_block_permuted_coefficients_for_group,
)


def _reflect_pad_1d_trials(
    x: np.ndarray,
    fs: float,
    pad_seconds: float,
) -> tuple[np.ndarray, int]:
    """
    x: trials x time
    """
    x = np.asarray(x, dtype=np.float64)
    pad = int(round(float(pad_seconds) * float(fs)))
    pad = max(1, min(pad, x.shape[1] - 2))
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    return xp, pad


def analytic_band_trials_1ch(
    epochs_ch_time_trial: np.ndarray,
    fs: float,
    band: tuple[float, float],
    anchor: int,
    window: tuple[float, float],
    order: int = 4,
    pad_seconds: float = 2.0,
) -> np.ndarray:
    """
    One channel, all trials.

    Input
    -----
    epochs_ch_time_trial : time x trials

    Output
    ------
    trials x analysis_time complex analytic signal, amplitude preserved.

    This mirrors the PLV engine's explicit reflection-padding logic, but unlike
    PLV it does NOT normalize the analytic signal to unit phase because
    coherency requires the complex amplitude.
    """
    x = np.asarray(epochs_ch_time_trial, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected time x trials, got {x.shape}")

    xt = x.T  # trials x time
    xp, pad = _reflect_pad_1d_trials(xt, fs, pad_seconds)

    sos = signal.butter(
        int(order),
        [float(band[0]), float(band[1])],
        btype="bandpass",
        fs=float(fs),
        output="sos",
    )
    xf = signal.sosfiltfilt(
        sos,
        xp,
        axis=1,
        padtype=None,
    )
    zp = signal.hilbert(xf, axis=1)
    z = zp[:, pad:-pad]

    start = int(anchor) + int(round(float(window[0]) * fs))
    end = int(anchor) + int(round(float(window[1]) * fs))
    if start < 0 or end > z.shape[1] or end <= start:
        raise ValueError(
            f"Bad crop {start}:{end} for epoch length {z.shape[1]}"
        )

    return z[:, start:end].astype(np.complex64)


def abs_imag_coherency_trials(
    z1: np.ndarray,
    z2: np.ndarray,
) -> np.ndarray:
    """
    Absolute imaginary part of narrow-band complex coherency, trial by trial.

      Cxy = sum_t z1(t) conj(z2(t))
            / sqrt(sum_t |z1(t)|^2 sum_t |z2(t)|^2)

      iCOH = |Im(Cxy)|

    Returns values in [0,1] up to floating-point tolerance.
    """
    a = np.asarray(z1)
    b = np.asarray(z2)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError(f"Need equal trials x time arrays, got {a.shape}, {b.shape}")

    num = np.sum(a * np.conj(b), axis=1, dtype=np.complex128)
    e1 = np.sum(np.abs(a) ** 2, axis=1, dtype=np.float64)
    e2 = np.sum(np.abs(b) ** 2, axis=1, dtype=np.float64)
    den = np.sqrt(e1 * e2)

    c = np.zeros_like(num, dtype=np.complex128)
    good = den > 1e-20
    c[good] = num[good] / den[good]

    out = np.abs(np.imag(c))
    out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def _candidate_channel_indices(cfg: dict, candidate: dict):
    ch = list(cfg["channel_names"])
    return ch.index(candidate["ch1"]), ch.index(candidate["ch2"])


def build_candidate_signals(
    cfg: dict,
    candidate: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build analytic signals for the two channel roles required by a fixed
    candidate, for every group and participant.

    Returns
    -------
    role1, role2 : G x 3 x T x L complex64
        role1 uses candidate ch1 for every participant;
        role2 uses candidate ch2 for every participant.

    The three within-triad dyads are then:
      pair12 = role1(P1) vs role2(P2)
      pair13 = role1(P1) vs role2(P3)
      pair23 = role1(P2) vs role2(P3)
    """
    task = candidate["task"]
    band_name = candidate["band"]
    band = tuple(cfg["bands_hz"][band_name])
    window = tuple(cfg["analysis_windows_seconds"][task])
    anchor = int(cfg["task_anchor_sample_zero_based"])
    order = int(cfg["connectivity"]["butterworth_order"])
    pad = float(cfg["connectivity"]["reflection_pad_seconds"])
    c1, c2 = _candidate_channel_indices(cfg, candidate)

    G = int(cfg["n_groups"])
    T = int(cfg["n_trials"])

    role1_groups = []
    role2_groups = []

    for g in range(1, G + 1):
        r1_p = []
        r2_p = []
        for p in (1, 2, 3):
            s = load_cleaned_subject(cfg["cleaned_dir"], g, p)
            fs = float(s["fs"])

            z1 = analytic_band_trials_1ch(
                s[task][c1],
                fs,
                band,
                anchor,
                window,
                order=order,
                pad_seconds=pad,
            )
            z2 = analytic_band_trials_1ch(
                s[task][c2],
                fs,
                band,
                anchor,
                window,
                order=order,
                pad_seconds=pad,
            )
            if z1.shape[0] != T or z2.shape[0] != T:
                raise ValueError(
                    f"G{g:02d} P{p}: unexpected trial count "
                    f"{z1.shape[0]}/{z2.shape[0]}"
                )
            r1_p.append(z1)
            r2_p.append(z2)

        role1_groups.append(np.stack(r1_p, axis=0))
        role2_groups.append(np.stack(r2_p, axis=0))

    return (
        np.stack(role1_groups, axis=0),
        np.stack(role2_groups, axis=0),
    )


def candidate_observed_trials(
    role1: np.ndarray,
    role2: np.ndarray,
    dyad_name: str,
) -> np.ndarray:
    """
    G x T trial-level |imag coherency| for the candidate's actual dyad.
    """
    di = DYAD_NAMES.index(str(dyad_name))
    a, b = DYADS[di]
    G = role1.shape[0]
    vals = []
    for g in range(G):
        vals.append(
            abs_imag_coherency_trials(
                role1[g, a],
                role2[g, b],
            )
        )
    return np.stack(vals, axis=0)


def candidate_all_dyad_trials(
    role1: np.ndarray,
    role2: np.ndarray,
) -> np.ndarray:
    """
    G x T x 3 trial-level |imag coherency| for all within-triad dyads,
    keeping the same candidate channel-role pair.
    """
    G = role1.shape[0]
    out = []
    for g in range(G):
        dd = []
        for a, b in DYADS:
            dd.append(
                abs_imag_coherency_trials(
                    role1[g, a],
                    role2[g, b],
                )
            )
        out.append(np.stack(dd, axis=1))  # T x 3
    return np.stack(out, axis=0)  # G x T x 3


def candidate_family_studentized_maxT(
    observed: np.ndarray,
    null_matrix: np.ndarray,
) -> dict:
    obs = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null_matrix, dtype=np.float64)

    if null.ndim != 2 or null.shape[1] != len(obs):
        raise ValueError(
            f"null {null.shape} incompatible with observed {obs.shape}"
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
        sorted_max,
        np.abs(oz),
        side="left",
    )
    p = (1 + (len(max_abs) - left)) / (len(max_abs) + 1)

    return {
        "p_maxT": p.astype(np.float64),
        "observed_studentized": oz.astype(np.float64),
        "null_mean": mu.astype(np.float64),
        "null_sd": sd.astype(np.float64),
        "null_max_abs_studentized": max_abs.astype(np.float64),
    }


def label_null_fixed_candidates(
    trial_values: np.ndarray,
    labels: np.ndarray,
    coeff: np.ndarray,
    total_weight: float,
    block_size: int,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    """
    trial_values: G x T x K fixed-candidate values.

    The same block-restricted label permutation realization is applied across
    all K candidates, preserving candidate dependence for maxT.
    """
    X = np.asarray(trial_values, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.uint8)
    G, T, K = X.shape
    rng = np.random.default_rng(int(seed))
    B = int(n_realizations)
    null = np.zeros((B, K), dtype=np.float32)

    for g in range(G):
        Wg = draw_within_block_permuted_coefficients_for_group(
            labels[g],
            B,
            rng,
            global_total_weight=float(total_weight),
            block_size=int(block_size),
        )
        null += Wg @ X[g]

    return null


def _valid_lags(n_time: int, fs: float, min_shift_seconds: float):
    min_s = int(round(float(min_shift_seconds) * float(fs)))
    lag = np.arange(n_time)
    d = np.minimum(lag, n_time - lag)
    valid = lag[d >= min_s]
    if len(valid) == 0:
        raise ValueError(
            f"No valid circular lags for n={n_time}, min={min_s}"
        )
    return valid


def circular_abs_imag_coherency_all_lags(
    z1: np.ndarray,
    z2: np.ndarray,
) -> np.ndarray:
    """
    z1,z2: T x L analytic signals.
    Returns T x L absolute imaginary coherency for every circular lag.

    Circular shifting leaves auto-energies unchanged, so the denominator is
    constant over lag within a trial.
    """
    a = np.asarray(z1)
    b = np.asarray(z2)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")

    A = np.fft.fft(a, axis=1)
    B = np.fft.fft(b, axis=1)
    num = np.fft.ifft(A * np.conj(B), axis=1)

    e1 = np.sum(np.abs(a) ** 2, axis=1, dtype=np.float64)
    e2 = np.sum(np.abs(b) ** 2, axis=1, dtype=np.float64)
    den = np.sqrt(e1 * e2)

    coh = np.zeros_like(num, dtype=np.complex128)
    good = den > 1e-20
    coh[good] = num[good] / den[good, None]

    return np.clip(
        np.abs(np.imag(coh)),
        0.0,
        1.0,
    ).astype(np.float32)


def temporal_null_one_candidate(
    role1: np.ndarray,
    role2: np.ndarray,
    dyad_name: str,
    coeff: np.ndarray,
    fs: float,
    min_shift_seconds: float,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    di = DYAD_NAMES.index(str(dyad_name))
    a, b = DYADS[di]
    G, _, T, _ = role1.shape
    B = int(n_realizations)
    rng = np.random.default_rng(int(seed))
    out = np.zeros(B, dtype=np.float32)

    for g in range(G):
        lagvals = circular_abs_imag_coherency_all_lags(
            role1[g, a],
            role2[g, b],
        )
        valid = _valid_lags(
            lagvals.shape[1],
            fs,
            min_shift_seconds,
        )
        draw = rng.choice(valid, size=(B, T), replace=True)
        vals = lagvals[
            np.arange(T)[None, :],
            draw,
        ]
        out += vals @ coeff[g].astype(np.float32)

    return out


def _derangements(
    rng: np.random.Generator,
    n_groups: int,
    n_realizations: int,
) -> np.ndarray:
    out = []
    for _ in range(int(n_realizations)):
        for _attempt in range(10000):
            p = rng.permutation(n_groups)
            if np.all(p != np.arange(n_groups)):
                out.append(p)
                break
        else:
            raise RuntimeError("Could not draw derangement")
    return np.stack(out, axis=0)


def partner_null_fixed_candidates(
    packages: list[dict],
    coeff: np.ndarray,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    """
    Cross-group partner-shuffle null for all fixed candidates using the same
    derangement per realization across candidates.
    """
    B = int(n_realizations)
    G = coeff.shape[0]
    rng = np.random.default_rng(int(seed))
    perms = _derangements(rng, G, B)

    out = np.empty((B, len(packages)), dtype=np.float32)

    for k, pkg in enumerate(packages):
        role1 = pkg["role1"]
        role2 = pkg["role2"]
        di = DYAD_NAMES.index(pkg["candidate"]["dyad"])
        a, b = DYADS[di]

        cross = np.empty((G, G, coeff.shape[1]), dtype=np.float32)
        for g in range(G):
            for h in range(G):
                cross[g, h] = abs_imag_coherency_trials(
                    role1[g, a],
                    role2[h, b],
                )

        vals = np.empty(B, dtype=np.float32)
        for r in range(B):
            total = 0.0
            for g in range(G):
                total += float(
                    np.dot(cross[g, perms[r, g]], coeff[g])
                )
            vals[r] = total
        out[:, k] = vals

    return out


def within_triad_dyad_null_fixed_candidates(
    packages: list[dict],
    coeff: np.ndarray,
    n_realizations: int,
    seed: int,
) -> np.ndarray:
    """
    Same-triad/same-trial dyad identity randomization for all fixed candidates.
    The same random dyad assignment is used across candidates per realization,
    group, and trial to preserve candidate dependence for maxT.
    """
    B = int(n_realizations)
    G, T = coeff.shape
    rng = np.random.default_rng(int(seed))
    draw = rng.integers(
        0,
        3,
        size=(B, G, T),
    )

    out = np.zeros((B, len(packages)), dtype=np.float32)

    for k, pkg in enumerate(packages):
        all_d = pkg["all_dyad_trials"]  # G x T x 3
        total = np.zeros(B, dtype=np.float32)
        for g in range(G):
            selected = all_d[g][
                np.arange(T)[None, :],
                draw[:, g, :],
            ]
            total += selected @ coeff[g].astype(np.float32)
        out[:, k] = total

    return out


def build_fixed_candidate_packages(
    cfg: dict,
    candidates: list[dict],
) -> list[dict]:
    """
    Build/caches all analytic-signal inputs for the fixed candidate family.
    """
    packages = []
    for i, c in enumerate(candidates, start=1):
        print(
            f"iCOH signals {i}/{len(candidates)}: "
            f"{c['task']} {c['band']} {c['dyad']} "
            f"{c['ch1']}-{c['ch2']}",
            flush=True,
        )
        r1, r2 = build_candidate_signals(cfg, c)
        obs_trials = candidate_observed_trials(
            r1, r2, c["dyad"]
        )
        all_d = candidate_all_dyad_trials(r1, r2)
        packages.append(
            {
                "candidate": c,
                "role1": r1,
                "role2": r2,
                "observed_trials": obs_trials,
                "all_dyad_trials": all_d,
            }
        )
    return packages


def run_fixed_candidate_imcoh(
    cfg: dict,
    candidates: list[dict],
    labels: np.ndarray,
    n_label: int,
    n_time: int,
    n_partner: int,
    n_dyad: int,
    seed: int,
) -> dict:
    """
    Complete fixed-candidate imaginary-coherency robustness analysis.
    """
    packages = build_fixed_candidate_packages(cfg, candidates)

    coeff, total_w = block_contrast_coefficients(
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    trial_values = np.stack(
        [p["observed_trials"] for p in packages],
        axis=2,
    )  # G x T x K

    observed = apply_trial_coefficients(
        trial_values.astype(np.float64),
        coeff,
    )

    label_null = label_null_fixed_candidates(
        trial_values,
        labels,
        coeff,
        total_w,
        int(cfg["inference"]["block_size_trials"]),
        n_label,
        seed + 10,
    )

    time_null = np.empty(
        (int(n_time), len(packages)),
        dtype=np.float32,
    )
    for k, pkg in enumerate(packages):
        # same sampling rate for all cleaned subjects
        fs = float(cfg["sampling_rate_hz"])
        time_null[:, k] = temporal_null_one_candidate(
            pkg["role1"],
            pkg["role2"],
            pkg["candidate"]["dyad"],
            coeff,
            fs,
            float(cfg["inference"]["min_time_shift_seconds"]),
            n_time,
            seed + 100 + k,
        )

    partner_null = partner_null_fixed_candidates(
        packages,
        coeff,
        n_partner,
        seed + 200,
    )
    dyad_null = within_triad_dyad_null_fixed_candidates(
        packages,
        coeff,
        n_dyad,
        seed + 300,
    )

    return {
        "packages": packages,
        "trial_values": trial_values,
        "observed": observed,
        "coeff": coeff,
        "total_weight": total_w,
        "label": candidate_family_studentized_maxT(
            observed, label_null
        ),
        "time": candidate_family_studentized_maxT(
            observed, time_null
        ),
        "partner": candidate_family_studentized_maxT(
            observed, partner_null
        ),
        "dyad": candidate_family_studentized_maxT(
            observed, dyad_null
        ),
    }
