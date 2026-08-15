from __future__ import annotations

import numpy as np
from scipy import signal

DYADS = [(0, 1), (0, 2), (1, 2)]
DYAD_NAMES = ["pair12", "pair13", "pair23"]


def _explicit_reflect_pad(
    x: np.ndarray,
    fs: float,
    pad_seconds: float,
) -> tuple[np.ndarray, int]:
    pad = int(round(float(pad_seconds) * float(fs)))
    pad = max(1, min(pad, x.shape[1] - 2))
    xp = np.pad(x, ((0, 0), (pad, pad)), mode="reflect")
    return xp, pad


def analytic_unit_phase(
    full_epoch: np.ndarray,
    fs: float,
    band: tuple[float, float],
    anchor: int,
    start_sec: float,
    end_sec: float,
    order: int = 4,
    pad_seconds: float = 2.0,
) -> np.ndarray:
    """
    Band-pass + Hilbert phase on an explicitly reflection-padded full epoch.

    Important: the Hilbert transform is computed BEFORE removing the explicit
    padding. This reduces endpoint contamination compared with applying Hilbert
    only to the unpadded 3-5 s released epoch.
    """
    x = np.asarray(full_epoch, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected channels x time, got {x.shape}")

    xp, pad = _explicit_reflect_pad(x, fs, pad_seconds)
    sos = signal.butter(
        int(order),
        [float(band[0]), float(band[1])],
        btype="bandpass",
        fs=float(fs),
        output="sos",
    )
    xf = signal.sosfiltfilt(sos, xp, axis=1, padtype=None)
    zp = signal.hilbert(xf, axis=1)
    z = zp[:, pad:-pad]

    start = int(anchor) + int(round(float(start_sec) * fs))
    end = int(anchor) + int(round(float(end_sec) * fs))

    if start < 0 or end > z.shape[1] or end <= start:
        raise ValueError(
            f"Bad crop {start}:{end} for epoch length {z.shape[1]}"
        )

    z = z[:, start:end]
    mag = np.abs(z)
    mag[mag == 0] = 1.0
    return (z / mag).astype(np.complex64)


def unit_phase_trials(
    epochs: np.ndarray,
    fs: float,
    band: tuple[float, float],
    anchor: int,
    window: tuple[float, float],
    order: int = 4,
    pad_seconds: float = 2.0,
) -> np.ndarray:
    """
    epochs: channels x time x trials
    returns: trials x channels x analysis_time
    """
    epochs = np.asarray(epochs)
    out = []
    for tr in range(epochs.shape[2]):
        out.append(
            analytic_unit_phase(
                epochs[:, :, tr],
                fs,
                band,
                anchor,
                window[0],
                window[1],
                order=order,
                pad_seconds=pad_seconds,
            )
        )
    return np.stack(out, axis=0)


def plv_trial_matrices_from_phase(
    z1: np.ndarray,
    z2: np.ndarray,
) -> np.ndarray:
    """
    z1,z2: trials x channels x time, unit magnitude complex phase.
    returns: trials x channels x channels
    """
    if z1.shape[0] != z2.shape[0] or z1.shape[2] != z2.shape[2]:
        raise ValueError(f"Phase shape mismatch {z1.shape} vs {z2.shape}")

    n_time = z1.shape[2]
    c = np.einsum(
        "tci,tdi->tcd",
        z1,
        np.conj(z2),
        optimize=True,
    ) / n_time
    return np.abs(c).astype(np.float32)


def edge_index(n_ch: int = 19):
    ii, jj = np.meshgrid(
        np.arange(n_ch), np.arange(n_ch), indexing="ij"
    )
    return ii.reshape(-1), jj.reshape(-1)
