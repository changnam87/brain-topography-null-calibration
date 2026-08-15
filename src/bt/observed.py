from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import condition_list, save_config_snapshot
from .connectivity import (
    DYADS,
    DYAD_NAMES,
    edge_index,
    plv_trial_matrices_from_phase,
    unit_phase_trials,
)
from .io import all_group_labels, load_cleaned_subject
from .statistics import (
    apply_trial_coefficients,
    block_contrast_coefficients,
    equal_triad_coefficients,
    group_block_effects,
    triad_information_coefficients,
)


def compute_trial_metric_cube(
    cfg: dict,
    overwrite: bool = False,
):
    """
    Compute PRIMARY metric only (PLV) and cache trial-level edge values.

    Efficiency:
      participant-specific band/Hilbert phase is computed once per
      group x task-band and reused for both dyads containing that participant.
    """
    outdir = Path(cfg["results_root"]) / "observed"
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "trial_connectivity_cube.npz"

    if out.exists() and not overwrite:
        return out

    conds = condition_list(cfg)
    G = int(cfg["n_groups"])
    C = len(conds)
    D = 3
    T = int(cfg["n_trials"])
    E = int(cfg["n_channels"]) ** 2

    plv = np.empty((G, C, D, T, E), dtype=np.float32)
    labels, choices = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    edge_i, edge_j = edge_index(cfg["n_channels"])

    for gi, g in enumerate(range(1, G + 1)):
        print(f"Observed PLV: G{g:02d}", flush=True)
        subj = [
            load_cleaned_subject(cfg["cleaned_dir"], g, p)
            for p in (1, 2, 3)
        ]
        fs = subj[0]["fs"]
        if any(abs(s["fs"] - fs) > 1e-9 for s in subj):
            raise ValueError(f"G{g:02d}: sampling-rate mismatch")

        for ci, (task, band_name) in enumerate(conds):
            band = tuple(cfg["bands_hz"][band_name])
            window = tuple(cfg["analysis_windows_seconds"][task])
            anchor = int(cfg["task_anchor_sample_zero_based"])
            order = int(cfg["connectivity"]["butterworth_order"])
            pad = float(cfg["connectivity"]["reflection_pad_seconds"])

            phases = [
                unit_phase_trials(
                    s[task],
                    fs,
                    band,
                    anchor,
                    window,
                    order=order,
                    pad_seconds=pad,
                )
                for s in subj
            ]

            for di, (a, b) in enumerate(DYADS):
                pm = plv_trial_matrices_from_phase(phases[a], phases[b])
                plv[gi, ci, di] = pm.reshape(T, E)

    np.savez_compressed(
        out,
        plv=plv,
        labels=labels.astype(np.uint8),
        choices=choices.astype("U1"),
        conditions=np.array(
            [f"{t}|{b}" for t, b in conds], dtype=object
        ),
        dyads=np.array(DYAD_NAMES, dtype=object),
        edge_i=edge_i.astype(np.int16),
        edge_j=edge_j.astype(np.int16),
        channel_names=np.array(cfg["channel_names"], dtype=object),
        statistic_design=np.array(
            cfg["inference"]["primary_statistic"], dtype=object
        ),
    )

    save_config_snapshot(cfg, outdir)
    return out


def _flatten_trial_cube(X: np.ndarray) -> np.ndarray:
    # G,C,D,T,E -> G,T,U
    G, C, D, T, E = X.shape
    return X.transpose(0, 3, 1, 2, 4).reshape(G, T, C * D * E)


def observed_effects(
    npz_path: str | Path,
    cfg: dict,
    metric: str = "plv",
):
    z = np.load(npz_path, allow_pickle=True)
    X = _flatten_trial_cube(z[metric].astype(np.float64))
    labels = z["labels"].astype(np.uint8)

    block_c, _ = block_contrast_coefficients(
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )
    triad_c, _ = triad_information_coefficients(labels)
    equal_c = equal_triad_coefficients(labels)

    primary = apply_trial_coefficients(X, block_c)
    triad_weighted = apply_trial_coefficients(X, triad_c)
    equal_triad = apply_trial_coefficients(X, equal_c)

    ge, gw = group_block_effects(
        X,
        labels,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    return {
        "primary_block_weighted": primary,
        "secondary_triad_weighted": triad_weighted,
        "secondary_equal_triad": equal_triad,
        "group_block_effects": ge,
        "group_block_information_weights": gw,
    }
