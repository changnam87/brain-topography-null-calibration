#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

def run(path: Path, *args: str):
    cmd = [PYTHON, str(path), *args]
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)

def require(path: Path, message: str):
    if not path.exists():
        raise SystemExit(message)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--resume-simulation", action="store_true")
    p.add_argument("--skip-preprocessing", action="store_true")
    p.add_argument("--skip-figures", action="store_true")
    args = p.parse_args()

    require(ROOT / "configs" / "bt_analysis_config.json",
            "Missing configs/bt_analysis_config.json")
    require(ROOT / "data" / "raw" / "openneuro_ds007822",
            "Missing OpenNeuro dataset at data/raw/openneuro_ds007822")
    require(ROOT / "data" / "reference" / "PD_EEG_hyperscan_processing",
            "Missing reference preprocessing repository at "
            "data/reference/PD_EEG_hyperscan_processing")

    if not args.skip_preprocessing:
        run(ROOT / "src/audit/14_validate_reference_preprocessing_g01.py",
            "--project", str(ROOT))
        run(ROOT / "scripts/00_validate_preprocessing_g01.py")
        run(ROOT / "scripts/01_run_preprocessing_all.py")
        run(ROOT / "src/audit/15_full_preprocessing_qc.py",
            "--project", str(ROOT))
        run(ROOT / "src/audit/16_flagged_preprocessing_drilldown.py",
            "--project", str(ROOT))
        run(ROOT / "src/audit/17_freeze_artifact_sensitivity_mask.py",
            "--project", str(ROOT))
        run(ROOT / "src/audit/18_behavior_balance_inference_audit.py",
            "--project", str(ROOT))
        run(ROOT / "src/audit/19_block_balance_inference_audit.py",
            "--project", str(ROOT))

    run(ROOT / "scripts/20_validate_plv_engine.py")
    run(ROOT / "scripts/02_compute_observed.py")

    run(ROOT / "src/audit/21_validate_observed_cube.py",
        "--project", str(ROOT))
    run(ROOT / "src/audit/22_validate_label_null_engine.py",
        "--project", str(ROOT))
    run(ROOT / "src/audit/23_run_production_label_null.py",
        "--project", str(ROOT))
    run(ROOT / "src/audit/24_audit_production_candidates.py",
        "--project", str(ROOT))
    run(ROOT / "src/audit/25_validate_candidate_null_engines.py",
        "--project", str(ROOT))
    run(ROOT / "src/audit/26_run_production_candidate_nulls.py",
        "--project", str(ROOT))
    run(ROOT / "src/audit/27_audit_secondary_null_separation.py",
        "--project", str(ROOT))

    run(ROOT / "scripts/28_validate_groundtruth_simulation.py")

    sim_args = ["--project", str(ROOT), "--n-jobs", str(args.n_jobs)]
    if args.resume_simulation:
        sim_args.append("--resume")

    run(ROOT / "scripts/30_run_groundtruth_simulation_production.py", *sim_args)
    run(ROOT / "scripts/31_validate_within_triad_dyad_null.py")
    run(ROOT / "scripts/32_stress_test_within_triad_dyad_null.py")
    run(ROOT / "scripts/33_validate_empirical_within_triad_dyad_null.py")
    run(ROOT / "scripts/34_run_empirical_within_triad_dyad_null.py")
    run(ROOT / "scripts/35_validate_imcoh_robustness.py")
    run(ROOT / "scripts/36_run_imcoh_robustness.py")
    run(ROOT / "scripts/37_validate_final3_stability.py")
    run(ROOT / "scripts/38_run_final3_stability.py")
    run(ROOT / "scripts/39_freeze_analysis_evidence.py")

    if not args.skip_figures:
        run(ROOT / "scripts/44_make_all_manuscript_figures.py",
            "--project", str(ROOT))

    print("\nDONE: manuscript analysis pipeline completed.", flush=True)

if __name__ == "__main__":
    main()
