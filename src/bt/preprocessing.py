from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np

from .io import subject_id


def _reference_modules(cfg: dict):
    """
    Import the exact public Data in Brief preprocessing modules from the
    locally cloned reference repository.

    Expected:
      <repo>/pd_eeg_analysis/preprocess_bids.py
      <repo>/pd_eeg_analysis/preprocessing_core.py
    """
    repo = Path(cfg["reference_preprocessing_repo"]).expanduser().resolve()
    code_dir = repo / "pd_eeg_analysis"
    if not code_dir.is_dir():
        raise FileNotFoundError(
            "Published reference preprocessing repository not found.\n"
            f"Expected: {repo}"
        )

    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))

    preprocess_bids = importlib.import_module("preprocess_bids")
    preprocessing_core = importlib.import_module("preprocessing_core")
    return repo, preprocess_bids, preprocessing_core


def _assert_reference_constants(preprocessing_core, cfg: dict):
    """
    Fail early if the checked-out public preprocessing code no longer matches
    the validated Data in Brief implementation.
    """
    expected = cfg["preprocessing"]

    if int(preprocessing_core.ICA_N_COMPONENTS) != int(
        expected["expected_ica_n_components"]
    ):
        raise RuntimeError(
            f"Reference ICA_N_COMPONENTS changed: "
            f"{preprocessing_core.ICA_N_COMPONENTS}"
        )

    if int(preprocessing_core.ICA_RANDOM_STATE) != int(
        expected["expected_ica_random_state"]
    ):
        raise RuntimeError(
            f"Reference ICA_RANDOM_STATE changed: "
            f"{preprocessing_core.ICA_RANDOM_STATE}"
        )

    if int(preprocessing_core.ICA_MAX_ITER) != int(
        expected["expected_ica_max_iter"]
    ):
        raise RuntimeError(
            f"Reference ICA_MAX_ITER changed: {preprocessing_core.ICA_MAX_ITER}"
        )

    got_labels = {str(x).lower() for x in preprocessing_core.ICA_REJECT_LABELS}
    exp_labels = {str(x).lower() for x in expected["expected_reject_labels"]}
    if got_labels != exp_labels:
        raise RuntimeError(
            f"Reference ICA_REJECT_LABELS changed: {sorted(got_labels)}"
        )

    if abs(
        float(preprocessing_core.ICA_REJECT_PROB)
        - float(expected["expected_reject_probability"])
    ) > 1e-12:
        raise RuntimeError(
            f"Reference ICA_REJECT_PROB changed: "
            f"{preprocessing_core.ICA_REJECT_PROB}"
        )

    got_channels = list(preprocessing_core.CH_NAMES)
    if got_channels != list(cfg["channel_names"]):
        raise RuntimeError(
            "Reference CH_NAMES differ from BT config.\n"
            f"Reference: {got_channels}\n"
            f"BT config: {cfg['channel_names']}"
        )


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def preprocess_group_reference(
    cfg: dict,
    group: int,
    overwrite: bool = False,
) -> list[tuple[Path, dict]]:
    """
    Preprocess one triad using the exact public Data in Brief code path.

    Efficiency:
      - load_group_from_bids() is called ONCE per triad;
      - each participant is then processed separately;
      - decision and feedback are processed separately, matching the validated
        public reference workflow.

    Saved numeric EEG units:
      microvolt-scale numeric output, exactly as returned by the published
      preprocess_task() implementation.
    """
    repo, preprocess_bids, preprocessing_core = _reference_modules(cfg)
    _assert_reference_constants(preprocessing_core, cfg)

    bids_dir = Path(cfg["dataset_root"]).expanduser().resolve()
    outdir = Path(cfg["cleaned_dir"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    expected_paths = [
        outdir / f"{subject_id(group, p)}_cleaned_bt.npz" for p in (1, 2, 3)
    ]
    expected_qc = [
        outdir / f"{subject_id(group, p)}_cleaned_bt_qc.json" for p in (1, 2, 3)
    ]

    if (
        not overwrite
        and all(p.exists() for p in expected_paths)
        and all(p.exists() for p in expected_qc)
    ):
        return [
            (p, json.loads(q.read_text(encoding="utf-8")))
            for p, q in zip(expected_paths, expected_qc)
        ]

    grp = preprocess_bids.load_group_from_bids(group, bids_dir)
    decision = np.asarray(grp["decision_X"], dtype=float)
    feedback = np.asarray(grp["feedback_X"], dtype=float)

    if decision.shape != (57, 1500, 40):
        raise ValueError(
            f"G{group:02d}: decision group shape {decision.shape}; "
            "expected (57,1500,40)"
        )
    if feedback.shape != (57, 900, 40):
        raise ValueError(
            f"G{group:02d}: feedback group shape {feedback.shape}; "
            "expected (57,900,40)"
        )

    results = []

    for p in (1, 2, 3):
        sid = subject_id(group, p)
        sl = slice((p - 1) * 19, p * 19)

        d_in = decision[sl].copy()
        f_in = feedback[sl].copy()

        d_clean, d_nrej = preprocessing_core.preprocess_task(d_in)
        f_clean, f_nrej = preprocessing_core.preprocess_task(f_in)

        d_clean = np.asarray(d_clean, dtype=np.float32)
        f_clean = np.asarray(f_clean, dtype=np.float32)

        if d_clean.shape != (19, 1500, 40):
            raise ValueError(f"{sid}: cleaned decision shape {d_clean.shape}")
        if f_clean.shape != (19, 900, 40):
            raise ValueError(f"{sid}: cleaned feedback shape {f_clean.shape}")
        if not np.isfinite(d_clean).all():
            raise ValueError(f"{sid}: non-finite decision values")
        if not np.isfinite(f_clean).all():
            raise ValueError(f"{sid}: non-finite feedback values")

        out = outdir / f"{sid}_cleaned_bt.npz"
        qc_path = outdir / f"{sid}_cleaned_bt_qc.json"

        np.savez_compressed(
            out,
            decision=d_clean,
            feedback=f_clean,
            fs=np.array(float(cfg["sampling_rate_hz"]), dtype=np.float64),
            channel_names=np.array(
                list(preprocessing_core.CH_NAMES), dtype=object
            ),
            units=np.array("uV", dtype=object),
            preprocessing_mode=np.array(
                "published_DataInBrief_reference", dtype=object
            ),
            reference_repo=np.array(str(repo), dtype=object),
            decision_n_rejected_components=np.array(
                int(d_nrej), dtype=np.int16
            ),
            feedback_n_rejected_components=np.array(
                int(f_nrej), dtype=np.int16
            ),
        )

        qc = {
            "subject": sid,
            "group": int(group),
            "participant": int(p),
            "decision_shape": list(d_clean.shape),
            "feedback_shape": list(f_clean.shape),
            "fs": float(cfg["sampling_rate_hz"]),
            "units": "uV",
            "channel_names": list(preprocessing_core.CH_NAMES),
            "preprocessing_mode": "published_DataInBrief_reference",
            "reference_repo": str(repo),
            "decision_n_rejected_components": int(d_nrej),
            "feedback_n_rejected_components": int(f_nrej),
            "decision_finite": True,
            "feedback_finite": True,
            "decision_rms_uV": _rms(d_clean),
            "feedback_rms_uV": _rms(f_clean),
            "decision_abs_p99_uV": float(
                np.percentile(np.abs(d_clean), 99)
            ),
            "feedback_abs_p99_uV": float(
                np.percentile(np.abs(f_clean), 99)
            ),
        }
        qc_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")
        results.append((out, qc))

    return results


def preprocess_subject(
    cfg: dict,
    group: int,
    participant: int,
    overwrite: bool = False,
):
    """
    Compatibility wrapper for callers that need one subject.

    For efficiency, new scripts should call preprocess_group_reference()
    once per triad instead of calling preprocess_subject three times.
    """
    results = preprocess_group_reference(cfg, group, overwrite=overwrite)
    return results[participant - 1]
