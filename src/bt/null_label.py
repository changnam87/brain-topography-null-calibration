from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .config import save_config_snapshot
from .statistics import (
    apply_trial_coefficients,
    block_contrast_coefficients,
    draw_within_block_permuted_coefficients_for_group,
    group_block_effects,
)


def _flatten_trial_cube(X: np.ndarray) -> np.ndarray:
    G, C, D, T, E = X.shape
    return X.transpose(0, 3, 1, 2, 4).reshape(G, T, C * D * E)


def _studentized_maxT_from_memmap(
    mmap_path: Path,
    shape: tuple[int, int],
    observed: np.ndarray,
    unit_chunk: int = 1000,
):
    B, U = shape
    mm = np.memmap(mmap_path, mode="r", dtype="float32", shape=shape)

    null_mean = np.empty(U, dtype=np.float64)
    null_sd = np.empty(U, dtype=np.float64)

    for st in range(0, U, unit_chunk):
        en = min(U, st + unit_chunk)
        x = np.asarray(mm[:, st:en], dtype=np.float64)
        null_mean[st:en] = x.mean(axis=0)
        null_sd[st:en] = x.std(axis=0, ddof=1)

    null_sd[~np.isfinite(null_sd) | (null_sd <= 1e-12)] = np.nan
    obs_z = (observed - null_mean) / null_sd

    max_abs_z = np.zeros(B, dtype=np.float64)
    unadj_exceed = np.zeros(U, dtype=np.int64)

    for st in range(0, U, unit_chunk):
        en = min(U, st + unit_chunk)
        x = np.asarray(mm[:, st:en], dtype=np.float64)
        z = (x - null_mean[st:en]) / null_sd[st:en]
        z = np.where(np.isfinite(z), z, 0.0)

        max_abs_z = np.maximum(
            max_abs_z, np.max(np.abs(z), axis=1)
        )
        unadj_exceed[st:en] = np.sum(
            np.abs(z) >= np.abs(obs_z[st:en])[None, :],
            axis=0,
        )

    p_unadj = (1.0 + unadj_exceed) / (B + 1.0)

    sorted_max = np.sort(max_abs_z)
    abs_obs = np.abs(obs_z)
    left = np.searchsorted(sorted_max, abs_obs, side="left")
    p_maxT = (1.0 + (B - left)) / (B + 1.0)

    return (
        obs_z.astype(np.float32),
        null_mean.astype(np.float32),
        null_sd.astype(np.float32),
        p_unadj.astype(np.float64),
        p_maxT.astype(np.float64),
        max_abs_z.astype(np.float32),
    )


def run_label_null(
    cfg: dict,
    trial_cube_path: str | Path,
    full: bool = True,
    metric: str = "plv",
):
    z = np.load(trial_cube_path, allow_pickle=True)
    X5 = z[metric].astype(np.float32)  # G,C,D,T,E
    labels = z["labels"].astype(np.uint8)

    G, C, D, T, E = X5.shape
    U = C * D * E
    X = _flatten_trial_cube(X5)

    block_size = int(cfg["inference"]["block_size_trials"])
    coeff, total_w = block_contrast_coefficients(
        labels, block_size=block_size
    )
    observed = apply_trial_coefficients(
        X.astype(np.float64), coeff
    ).astype(np.float32)

    group_eff, group_w = group_block_effects(
        X.astype(np.float64),
        labels,
        block_size=block_size,
    )

    B = int(
        cfg["inference"][
            "n_label_permutations_full"
            if full
            else "n_label_permutations_validation"
        ]
    )

    outdir = Path(cfg["results_root"]) / "nulls"
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = outdir / f"_tmp_label_null_{metric}_{B}x{U}.dat"

    mm = np.memmap(
        tmp, mode="w+", dtype="float32", shape=(B, U)
    )

    rng = np.random.default_rng(int(cfg["random_seed"]) + 3100)
    batch = min(250, B)

    for st in range(0, B, batch):
        en = min(B, st + batch)
        bb = en - st
        null_b = np.zeros((bb, U), dtype=np.float32)

        for g in range(G):
            Wg = draw_within_block_permuted_coefficients_for_group(
                labels[g],
                bb,
                rng,
                global_total_weight=total_w,
                block_size=block_size,
            )
            null_b += Wg @ X[g]

        mm[st:en] = null_b
        mm.flush()
        print(f"Label null permutations: {en}/{B}", flush=True)

    del mm

    (
        obs_z,
        null_mean,
        null_sd,
        p_unadj,
        p_maxT,
        max_abs_z,
    ) = _studentized_maxT_from_memmap(
        tmp, (B, U), observed
    )

    try:
        os.remove(tmp)
    except OSError:
        pass

    alpha = float(cfg["inference"]["candidate_alpha"])
    candidate = p_maxT < alpha

    out_npz = outdir / f"label_null_{metric}.npz"
    np.savez_compressed(
        out_npz,
        observed=observed,
        observed_studentized=obs_z,
        p_label_unadjusted=p_unadj,
        p_label_maxT=p_maxT,
        candidate=candidate.astype(np.uint8),
        null_mean=null_mean,
        null_sd=null_sd,
        null_max_abs_studentized=max_abs_z,
        group_effects=group_eff.astype(np.float32),
        group_information_weights=group_w.astype(np.float32),
        conditions=z["conditions"],
        dyads=z["dyads"],
        edge_i=z["edge_i"],
        edge_j=z["edge_j"],
        channel_names=z["channel_names"],
        n_permutations=np.array(B),
        primary_statistic=np.array(
            cfg["inference"]["primary_statistic"], dtype=object
        ),
        multiple_comparison=np.array(
            cfg["inference"]["multiple_comparison_primary"],
            dtype=object,
        ),
    )

    conds = [str(x) for x in z["conditions"].tolist()]
    dyads = [str(x) for x in z["dyads"].tolist()]
    ei = z["edge_i"].astype(int)
    ej = z["edge_j"].astype(int)
    ch = [str(x) for x in z["channel_names"].tolist()]

    rows = []
    idx = 0
    for cond in conds:
        task, band = cond.split("|")
        for dyad in dyads:
            for e in range(E):
                rows.append(
                    {
                        "unit_index": idx,
                        "task": task,
                        "band": band,
                        "dyad": dyad,
                        "ch1_index": int(ei[e]),
                        "ch2_index": int(ej[e]),
                        "ch1": ch[int(ei[e])],
                        "ch2": ch[int(ej[e])],
                        "observed_effect": float(observed[idx]),
                        "observed_studentized": float(obs_z[idx]),
                        "p_label_unadjusted": float(p_unadj[idx]),
                        "p_label_maxT": float(p_maxT[idx]),
                        "label_candidate": bool(candidate[idx]),
                    }
                )
                idx += 1

    out_csv = outdir / f"label_null_{metric}_units.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    save_config_snapshot(cfg, outdir)
    return out_npz, out_csv
