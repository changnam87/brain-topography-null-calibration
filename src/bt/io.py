from __future__ import annotations
import csv
import re
from pathlib import Path
import numpy as np
from scipy.io import loadmat

SUBJECT_RE = re.compile(r"(sub-G(\d{2})S(\d{2}))", re.IGNORECASE)

def subject_id(group: int, participant: int) -> str:
    return f"sub-G{group:02d}S{participant:02d}"

def find_task_set(dataset_root: str | Path, subject: str, task: str) -> Path:
    root = Path(dataset_root)
    matches = list(root.rglob(f"{subject}_task-pd{task}_eeg.set"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"{subject} {task}: expected 1 .set, found {len(matches)}"
        )
    return matches[0]

def events_sidecar_for_set(set_path: Path) -> Path:
    name = set_path.name
    if not name.endswith("_eeg.set"):
        raise ValueError(f"Unexpected EEGLAB filename: {name}")
    return set_path.with_name(name[:-8] + "_events.tsv")

def read_events(events_tsv: str | Path) -> list[dict]:
    with Path(events_tsv).open(
        "r", encoding="utf-8-sig", errors="replace", newline=""
    ) as f:
        return list(csv.DictReader(f, delimiter="\t"))

def normalize_trial(v) -> int | None:
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(round(float(s)))
    except Exception:
        m = re.search(r"(\d+)", s)
        return int(m.group(1)) if m else None

def normalize_choice(v) -> str | None:
    s = str(v).strip().upper()
    if s in {"C","COOP","COOPERATE","COOPERATION","1","1.0"}:
        return "C"
    if s in {"D","DEFECT","DEFECTION","2","2.0","0","0.0"}:
        return "D"
    if "COOP" in s:
        return "C"
    if "DEFECT" in s:
        return "D"
    return None

def subject_choice_vector(
    dataset_root: str | Path, group: int, participant: int
) -> np.ndarray:
    sid = subject_id(group, participant)
    set_path = find_task_set(dataset_root, sid, "decision")
    rows = read_events(events_sidecar_for_set(set_path))
    choices = {}
    for row in rows:
        t = normalize_trial(row.get("trial",""))
        c = normalize_choice(row.get("choice",""))
        if t is not None and c is not None:
            if t in choices and choices[t] != c:
                raise ValueError(f"{sid}: conflicting choice at trial {t}")
            choices[t] = c
    if sorted(choices) != list(range(1,41)):
        raise ValueError(
            f"{sid}: expected trials 1..40, got {sorted(choices)}"
        )
    return np.array(
        [choices[t] for t in range(1,41)], dtype="U1"
    )

def triad_labels(
    dataset_root: str | Path, group: int
) -> tuple[np.ndarray, np.ndarray]:
    c = [
        subject_choice_vector(dataset_root, group, p)
        for p in (1,2,3)
    ]
    mat = np.column_stack(c)
    label = np.all(mat == "C", axis=1).astype(np.uint8)
    return label, mat

def all_group_labels(
    dataset_root: str | Path, n_groups: int = 11
):
    labels = []
    choices = []
    for g in range(1, n_groups+1):
        y, m = triad_labels(dataset_root, g)
        labels.append(y)
        choices.append(m)
    return np.stack(labels), np.stack(choices)

def load_cleaned_subject(
    cleaned_dir: str | Path, group: int, participant: int
):
    path = (
        Path(cleaned_dir)
        / f"{subject_id(group, participant)}_cleaned_bt.npz"
    )
    if not path.exists():
        raise FileNotFoundError(path)

    z = np.load(path, allow_pickle=True)
    required = {
        "decision","feedback","fs","channel_names","units",
        "preprocessing_mode",
        "decision_n_rejected_components",
        "feedback_n_rejected_components",
    }
    missing = required - set(z.files)
    if missing:
        raise ValueError(
            f"{path}: old/incompatible cleaned file; missing {sorted(missing)}"
        )

    mode = str(np.asarray(z["preprocessing_mode"]).squeeze())
    if mode != "published_DataInBrief_reference":
        raise ValueError(
            f"{path}: incompatible preprocessing mode {mode!r}"
        )

    return {
        "decision": z["decision"].astype(np.float32),
        "feedback": z["feedback"].astype(np.float32),
        "fs": float(np.asarray(z["fs"]).squeeze()),
        "channel_names": [
            str(x) for x in z["channel_names"].tolist()
        ],
        "units": str(np.asarray(z["units"]).squeeze()),
        "preprocessing_mode": mode,
        "decision_n_rejected_components": int(
            np.asarray(z["decision_n_rejected_components"]).squeeze()
        ),
        "feedback_n_rejected_components": int(
            np.asarray(z["feedback_n_rejected_components"]).squeeze()
        ),
    }
