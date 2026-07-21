"""Where every run's files live: batch_folder/<run_name>/. All stages agree via this module."""

from __future__ import annotations

import os

BATCH_ROOT = "batch_folder"
# historical filename stem, kept so existing run folders (batch_folder/pilot/) stay readable
JOB_ID = "pilot_batch_combined"
DEFAULT_RUN_NAME = "pilot"


def run_dir(run_name: str) -> str:
    return os.path.join(BATCH_ROOT, run_name)


def batch_jsonl(run_name: str) -> str:  # batch input        (written by prepare)
    return os.path.join(run_dir(run_name), f"{JOB_ID}.jsonl")


def combined_ref(run_name: str) -> str:  # chunks + join key  (written by prepare)
    return os.path.join(run_dir(run_name), f"{JOB_ID}_ref.parquet")


def batch_jsonl_summary(run_name: str) -> str:  # human-readable batch report (written by prepare)
    return os.path.join(run_dir(run_name), "batch_jsonl_summary.md")


def skipped_pdfs_json(run_name: str) -> str:  # fileids skipped (missing/corrupt) during chunking
    return os.path.join(run_dir(run_name), "skipped_pdfs.json")


def output_csv(run_name: str) -> str:  # regulated output    (written by download)
    return os.path.join(run_dir(run_name), f"output_{JOB_ID}.csv")


def parsed_csv(run_name: str) -> str:  # analysis-ready      (written by parse)
    return os.path.join(run_dir(run_name), "parsed_v2_combined.csv")


def report_xlsx(run_name: str) -> str:  # review workbook     (written by report)
    return os.path.join(run_dir(run_name), f"combined_{run_name}_batch.xlsx")
