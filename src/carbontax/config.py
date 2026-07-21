"""Per-run YAML config. No CLI arguments anywhere: edit config/run.yaml, then run the stage."""

from __future__ import annotations

import yaml

def load_run_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    # run_name is the key that ties all stages to the same batch_folder/<run>/ folder
    if not isinstance(cfg, dict) or "run_name" not in cfg:
        raise KeyError(f"{path} must have a top-level 'run_name' key")
    return cfg


def stage_section(cfg: dict, stage: str) -> dict:
    # section names match the stage package names (acquire_pdfs, prepare_batch_input, ...)
    return cfg.get(stage) or {}
