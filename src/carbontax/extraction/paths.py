"""
paths.py — per-run folder convention for the extraction pipeline.

Every run lives in its own subfolder under ``batch_folder/<run_name>/`` and holds
that run's batch input, reference parquet, BatchManager DuckDB state, raw/parsed
output. Centralizing the names here keeps build/run/parse in agreement.

Layout per run:
  batch_folder/<run_name>/
    pilot_batch_combined.jsonl          — batch input        (build)
    pilot_batch_combined_ref.parquet    — join key           (build)
    batch_status.db                     — BatchManager state  (run)
    output_pilot_batch_combined.jsonl   — raw OpenAI output   (run)
    output_pilot_batch_combined.csv     — regulated output    (run)
    parsed_v2_combined.csv              — analysis-ready      (parse)
"""

from __future__ import annotations

import os

BATCH_ROOT = "batch_folder"
JOB_ID     = "pilot_batch_combined"


def run_dir(run_name: str) -> str:
    return os.path.join(BATCH_ROOT, run_name)


def batch_jsonl(run_name: str) -> str:
    return os.path.join(run_dir(run_name), f"{JOB_ID}.jsonl")


def combined_ref(run_name: str) -> str:
    return os.path.join(run_dir(run_name), f"{JOB_ID}_ref.parquet")


def output_csv(run_name: str) -> str:
    return os.path.join(run_dir(run_name), f"output_{JOB_ID}.csv")


def parsed_csv(run_name: str) -> str:
    return os.path.join(run_dir(run_name), "parsed_v2_combined.csv")
