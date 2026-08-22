"""Where every run's files live: batch_folder/<run_name>/. All stages agree via this module."""

from __future__ import annotations

import glob
import os

BATCH_ROOT = "batch_folder"
FILE_STEM = "batch"  # stem for the files WE write (batch.jsonl, batch_ref.parquet)
# BatchManager's job id: the tracking key in batch_status.db AND the stem the wrapper
# uses to name the files IT writes (output_<JOB_ID>.csv/.jsonl). Keep stable, or an
# already-submitted batch loses its record.
JOB_ID = "pilot_batch_combined"


def run_dir(run_name: str) -> str:
    return os.path.join(BATCH_ROOT, run_name)


# batch input is sharded: OpenAI caps one batch at 50k requests AND ~200MB per file,
# so prepare writes an indexed folder of shards instead of a single batch.jsonl.
def batch_files_dir(run_name: str) -> str:  # holds the shards   (written by prepare)
    return os.path.join(run_dir(run_name), f"{FILE_STEM}_files")


def batch_shard_jsonl(run_name: str, idx: int) -> str:  # one shard file
    return os.path.join(batch_files_dir(run_name), f"{FILE_STEM}_{idx:03d}.jsonl")


def batch_shards(run_name: str) -> list[str]:  # all shards, index order (read by submit)
    return sorted(glob.glob(os.path.join(batch_files_dir(run_name), f"{FILE_STEM}_*.jsonl")))


def shard_job_id(idx: int) -> str:  # per-shard tracking key in batch_status.db
    return f"{JOB_ID}_p{idx:03d}"


def combined_ref(run_name: str) -> str:  # chunks + join key  (written by prepare)
    return os.path.join(run_dir(run_name), f"{FILE_STEM}_ref.parquet")


def batch_jsonl_summary(run_name: str) -> str:  # human-readable batch report (written by prepare)
    return os.path.join(run_dir(run_name), "batch_jsonl_summary.md")


def skipped_pdfs_json(run_name: str) -> str:  # fileids skipped (missing/corrupt) during chunking
    return os.path.join(run_dir(run_name), "skipped_pdfs.json")


# preview=True appends __preview so a partial (completed-shards-only) run never clobbers the
# canonical artifact — see openai_batch.allow_partial in the run config.
def output_csv(run_name: str, preview: bool = False) -> str:  # regulated output (written by download)
    suffix = "__preview" if preview else ""
    return os.path.join(run_dir(run_name), f"aggregated_output_batch_results{suffix}.csv")


# per-shard outputs the wrapper writes during download, tucked into batch_output/{csv,jsonl}/ so
# a many-shard run doesn't clutter the run root (the merged aggregated_output_batch_results.csv still lives above)
def shard_output_csv_dir(run_name: str) -> str:  # per-shard regulated CSVs
    return os.path.join(run_dir(run_name), "batch_output", "csv")


def shard_output_jsonl_dir(run_name: str) -> str:  # per-shard raw JSONL
    return os.path.join(run_dir(run_name), "batch_output", "jsonl")


def parsed_csv(run_name: str, preview: bool = False) -> str:  # analysis-ready flags, one row per chunk (written by parse)
    suffix = "__preview" if preview else ""
    return os.path.join(run_dir(run_name), f"parsed_aggregated_batch_output{suffix}.csv")


def report_xlsx(run_name: str, preview: bool = False) -> str:  # review workbook (written by report)
    suffix = "__preview" if preview else ""
    return os.path.join(run_dir(run_name), f"report{suffix}.xlsx")


# stage 5 keeps its own subfolder: regression output accumulates across specs rather
# than overwriting one canonical artifact the way report.xlsx does
def regression_dir(run_name: str) -> str:
    return os.path.join(run_dir(run_name), "regression")


def panel_csv(run_name: str) -> str:  # one row per company-year (written by build_panel)
    return os.path.join(regression_dir(run_name), "panel.csv")


def regression_results_csv(run_name: str) -> str:  # tidy: one row per spec × term
    return os.path.join(regression_dir(run_name), "results.csv")


def regression_summary_md(run_name: str) -> str:  # headline specs, human-readable
    return os.path.join(regression_dir(run_name), "summary.md")
