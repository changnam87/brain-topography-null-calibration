#!/usr/bin/env python3
from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bt.config import load_config, ensure_result_dirs, save_config_snapshot
from bt.preprocessing import preprocess_group_reference

cfg = load_config()
ensure_result_dirs(cfg)

results = preprocess_group_reference(cfg, 1, overwrite=True)

expected = {
    "sub-G01S01": {"decision": 1, "feedback": 2},
    "sub-G01S02": {"decision": 3, "feedback": 2},
    "sub-G01S03": {"decision": 2, "feedback": 2},
}

rows = []
ok = True

for _, qc in results:
    sid = qc["subject"]
    exp = expected[sid]

    row = {
        "subject": sid,
        "decision_shape": "x".join(map(str, qc["decision_shape"])),
        "feedback_shape": "x".join(map(str, qc["feedback_shape"])),
        "fs": qc["fs"],
        "units": qc["units"],
        "preprocessing_mode": qc["preprocessing_mode"],
        "decision_ica_rejected": qc["decision_n_rejected_components"],
        "feedback_ica_rejected": qc["feedback_n_rejected_components"],
        "expected_decision_ica_rejected": exp["decision"],
        "expected_feedback_ica_rejected": exp["feedback"],
        "decision_rms_uV": qc["decision_rms_uV"],
        "feedback_rms_uV": qc["feedback_rms_uV"],
        "decision_abs_p99_uV": qc["decision_abs_p99_uV"],
        "feedback_abs_p99_uV": qc["feedback_abs_p99_uV"],
        "decision_finite": qc["decision_finite"],
        "feedback_finite": qc["feedback_finite"],
    }

    row_ok = (
        row["decision_shape"] == "19x1500x40"
        and row["feedback_shape"] == "19x900x40"
        and abs(float(row["fs"]) - 300.0) < 1e-9
        and row["units"] == "uV"
        and row["preprocessing_mode"]
            == "published_DataInBrief_reference"
        and row["decision_ica_rejected"]
            == row["expected_decision_ica_rejected"]
        and row["feedback_ica_rejected"]
            == row["expected_feedback_ica_rejected"]
        and row["decision_finite"]
        and row["feedback_finite"]
    )
    row["validation_pass"] = row_ok
    ok = ok and row_ok
    rows.append(row)

outdir = Path(cfg["results_root"]) / "preprocessing"
outdir.mkdir(parents=True, exist_ok=True)

with (outdir / "validation_G01_reference_wrapper.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

save_config_snapshot(cfg, outdir)

print(
    "PASS: BT wrapper reproduces published G01 preprocessing"
    if ok
    else "CHECK REQUIRED"
)
for r in rows:
    print(r)

raise SystemExit(0 if ok else 1)
