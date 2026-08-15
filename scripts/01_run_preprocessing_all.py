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

rows = []

for g in range(1, cfg["n_groups"] + 1):
    print(f"Preprocessing G{g:02d} ...", flush=True)
    results = preprocess_group_reference(cfg, g, overwrite=False)
    for _, qc in results:
        rows.append(
            {
                "subject": qc["subject"],
                "units": qc["units"],
                "preprocessing_mode": qc["preprocessing_mode"],
                "decision_ica_rejected":
                    qc["decision_n_rejected_components"],
                "feedback_ica_rejected":
                    qc["feedback_n_rejected_components"],
                "decision_rms_uV": qc["decision_rms_uV"],
                "feedback_rms_uV": qc["feedback_rms_uV"],
                "decision_finite": qc["decision_finite"],
                "feedback_finite": qc["feedback_finite"],
            }
        )

outdir = Path(cfg["results_root"]) / "preprocessing"
with (outdir / "preprocessing_all_summary.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

save_config_snapshot(cfg, outdir)
print(f"DONE: published reference preprocessing completed for {len(rows)} participants")
