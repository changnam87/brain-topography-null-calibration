from __future__ import annotations
import json
from pathlib import Path

def _resolve_path(value: str | Path, base: Path) -> str:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = base / p
    return str(p.resolve())

def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "bt_analysis_config.json"

    path = Path(path).expanduser().resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))

    project = path.parent.parent

    for key in ("project_root", "dataset_root", "cleaned_dir", "results_root"):
        cfg[key] = _resolve_path(cfg[key], project)

    if "reference_preprocessing_repo" in cfg:
        cfg["reference_preprocessing_repo"] = _resolve_path(
            cfg["reference_preprocessing_repo"], project
        )

    mask = cfg.get("inference", {}).get("artifact_sensitivity_mask")
    if mask:
        cfg["inference"]["artifact_sensitivity_mask"] = _resolve_path(mask, project)

    return cfg

def condition_list(cfg: dict) -> list[tuple[str, str]]:
    return [tuple(x) for x in cfg["eligible_task_bands"]]

def ensure_result_dirs(cfg: dict) -> None:
    root = Path(cfg["results_root"])
    for name in (
        "preprocessing",
        "observed",
        "nulls",
        "simulation",
        "robustness",
        "stability",
        "freeze",
        "logs",
    ):
        (root / name).mkdir(parents=True, exist_ok=True)

def save_config_snapshot(
    cfg: dict,
    dest_dir: str | Path,
    filename: str = "config_used.json",
):
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / filename).write_text(
        json.dumps(cfg, indent=2),
        encoding="utf-8",
    )
