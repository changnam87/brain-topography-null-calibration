from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import condition_list, save_config_snapshot
from .connectivity import DYADS, DYAD_NAMES, analytic_unit_phase
from .io import all_group_labels, load_cleaned_subject
from .statistics import block_contrast_coefficients


def _load_candidates(unit_csv: str | Path):
    with Path(unit_csv).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        r for r in rows
        if str(r["label_candidate"]).lower() == "true"
    ]


def _unit_phase_1ch(
    epoch_1ch: np.ndarray,
    fs: float,
    band,
    anchor,
    window,
    cfg,
):
    out = []
    for tr in range(epoch_1ch.shape[1]):
        z = analytic_unit_phase(
            epoch_1ch[:, tr][None, :],
            fs,
            band,
            anchor,
            window[0],
            window[1],
            order=int(cfg["connectivity"]["butterworth_order"]),
            pad_seconds=float(
                cfg["connectivity"]["reflection_pad_seconds"]
            ),
        )[0]
        out.append(z)
    return np.stack(out)


def _valid_lags(T: int, fs: float, min_shift_sec: float):
    min_s = int(round(min_shift_sec * fs))
    lags = np.arange(T)
    circular_distance = np.minimum(lags, T - lags)
    valid = lags[circular_distance >= min_s]
    if len(valid) == 0:
        raise ValueError(
            f"No valid lags T={T}, min_shift={min_s}"
        )
    return valid


def _circular_plv_all_lags(z1: np.ndarray, z2: np.ndarray):
    A = np.fft.fft(z1, axis=1)
    B = np.fft.fft(z2, axis=1)
    corr = np.fft.ifft(A * np.conj(B), axis=1) / z1.shape[1]
    return np.abs(corr).astype(np.float32)


def _time_null_candidate(
    cfg,
    cand,
    B,
    labels_all,
    coeff,
):
    dyad_i = DYAD_NAMES.index(cand["dyad"])
    a, b = DYADS[dyad_i]
    ch1 = int(cand["ch1_index"])
    ch2 = int(cand["ch2_index"])
    band = tuple(cfg["bands_hz"][cand["band"]])
    window = tuple(cfg["analysis_windows_seconds"][cand["task"]])
    anchor = int(cfg["task_anchor_sample_zero_based"])
    rng = np.random.default_rng(
        int(cfg["random_seed"]) + int(cand["unit_index"]) * 17 + 1
    )

    total = np.zeros(B, dtype=np.float32)

    for gi, g in enumerate(range(1, cfg["n_groups"] + 1)):
        s1 = load_cleaned_subject(cfg["cleaned_dir"], g, a + 1)
        s2 = load_cleaned_subject(cfg["cleaned_dir"], g, b + 1)
        fs = s1["fs"]
        z1 = _unit_phase_1ch(
            s1[cand["task"]][ch1], fs, band, anchor, window, cfg
        )
        z2 = _unit_phase_1ch(
            s2[cand["task"]][ch2], fs, band, anchor, window, cfg
        )
        lag_plv = _circular_plv_all_lags(z1, z2)
        valid = _valid_lags(
            lag_plv.shape[1],
            fs,
            float(cfg["inference"]["min_time_shift_seconds"]),
        )

        lag_idx = rng.choice(
            valid,
            size=(B, cfg["n_trials"]),
            replace=True,
        )
        vals = lag_plv[
            np.arange(cfg["n_trials"])[None, :],
            lag_idx,
        ]
        total += vals @ coeff[gi].astype(np.float32)

    return total


def _derangement(rng, n):
    for _ in range(10000):
        p = rng.permutation(n)
        if np.all(p != np.arange(n)):
            return p
    raise RuntimeError("Failed to draw derangement")


def _partner_null_candidate(
    cfg,
    cand,
    B,
    labels_all,
    coeff,
):
    dyad_i = DYAD_NAMES.index(cand["dyad"])
    a, b = DYADS[dyad_i]
    ch1 = int(cand["ch1_index"])
    ch2 = int(cand["ch2_index"])
    band = tuple(cfg["bands_hz"][cand["band"]])
    window = tuple(cfg["analysis_windows_seconds"][cand["task"]])
    anchor = int(cfg["task_anchor_sample_zero_based"])
    G = int(cfg["n_groups"])
    rng = np.random.default_rng(
        int(cfg["random_seed"]) + int(cand["unit_index"]) * 17 + 2
    )

    ZA, ZB = [], []
    for g in range(1, G + 1):
        sa = load_cleaned_subject(cfg["cleaned_dir"], g, a + 1)
        sb = load_cleaned_subject(cfg["cleaned_dir"], g, b + 1)
        fs = sa["fs"]
        ZA.append(
            _unit_phase_1ch(
                sa[cand["task"]][ch1],
                fs,
                band,
                anchor,
                window,
                cfg,
            )
        )
        ZB.append(
            _unit_phase_1ch(
                sb[cand["task"]][ch2],
                fs,
                band,
                anchor,
                window,
                cfg,
            )
        )

    cross = np.empty(
        (G, G, cfg["n_trials"]), dtype=np.float32
    )
    for g in range(G):
        for h in range(G):
            cross[g, h] = np.abs(
                np.mean(ZA[g] * np.conj(ZB[h]), axis=1)
            ).astype(np.float32)

    null = np.empty(B, dtype=np.float32)
    for r in range(B):
        perm = _derangement(rng, G)
        val = 0.0
        for g in range(G):
            val += float(np.dot(cross[g, perm[g]], coeff[g]))
        null[r] = val

    return null


def _studentized_candidate_maxT(
    observed: np.ndarray,
    null_matrix: np.ndarray,
):
    mean = null_matrix.mean(axis=0)
    sd = null_matrix.std(axis=0, ddof=1)
    sd[sd <= 1e-12] = np.nan

    oz = (observed - mean) / sd
    nz = (null_matrix - mean[None, :]) / sd[None, :]
    nz = np.where(np.isfinite(nz), nz, 0.0)
    maxz = np.max(np.abs(nz), axis=1)

    p = np.empty(len(observed), dtype=float)
    sorted_max = np.sort(maxz)
    for i, v in enumerate(np.abs(oz)):
        if not np.isfinite(v):
            p[i] = 1.0
        else:
            left = np.searchsorted(sorted_max, v, side="left")
            p[i] = (1 + (len(maxz) - left)) / (len(maxz) + 1)

    return oz, p


def run_candidate_nulls(
    cfg: dict,
    unit_csv: str | Path,
    full: bool = True,
):
    candidates = _load_candidates(unit_csv)
    outdir = Path(cfg["results_root"]) / "nulls"
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / "candidate_nulls_plv.csv"

    if not candidates:
        out_csv.write_text(
            "unit_index,task,band,dyad,ch1,ch2,"
            "p_time_maxT,p_partner_maxT,secondary_validation_pass\n",
            encoding="utf-8",
        )
        return out_csv

    Btime = int(
        cfg["inference"][
            "n_time_shift_full"
            if full
            else "n_time_shift_validation"
        ]
    )
    Bpartner = int(
        cfg["inference"][
            "n_partner_shuffle_full"
            if full
            else "n_partner_shuffle_validation"
        ]
    )

    labels_all, _ = all_group_labels(
        cfg["dataset_root"], cfg["n_groups"]
    )
    coeff, _ = block_contrast_coefficients(
        labels_all,
        block_size=int(cfg["inference"]["block_size_trials"]),
    )

    K = len(candidates)
    time_null = np.empty((Btime, K), dtype=np.float32)
    partner_null = np.empty((Bpartner, K), dtype=np.float32)
    observed = np.array(
        [float(c["observed_effect"]) for c in candidates],
        dtype=float,
    )

    for k, cand in enumerate(candidates):
        print(
            f"Candidate nulls {k+1}/{K}: "
            f"{cand['task']} {cand['band']} {cand['dyad']} "
            f"{cand['ch1']}-{cand['ch2']}",
            flush=True,
        )
        time_null[:, k] = _time_null_candidate(
            cfg, cand, Btime, labels_all, coeff
        )
        partner_null[:, k] = _partner_null_candidate(
            cfg, cand, Bpartner, labels_all, coeff
        )

    _, ptime = _studentized_candidate_maxT(
        observed, time_null
    )
    _, ppartner = _studentized_candidate_maxT(
        observed, partner_null
    )

    alpha = float(cfg["inference"]["alpha"])
    rows = []

    for cand, pt, pp in zip(candidates, ptime, ppartner):
        row = dict(cand)
        row.update(
            {
                "p_time_maxT": float(pt),
                "p_partner_maxT": float(pp),
                "secondary_validation_pass": bool(
                    pt < alpha and pp < alpha
                ),
                "n_time_shift": Btime,
                "n_partner_shuffle": Bpartner,
            }
        )
        rows.append(row)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_config_snapshot(cfg, outdir)
    return out_csv
