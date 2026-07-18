"""
download_batch.py — poll a submitted batch and download its output when done.

This fills the gap between submit (run_batch) and parse (parse_output): OpenAI
runs the batch asynchronously, and the regulated CSV that parse_output expects is
only produced once BatchManager.get_output_file() pulls it down.

The BatchManager is rebuilt from the same JOB_ID and run folder, so it reloads the
batch id from batch_status.db. get_batch_status() is called first because it is
what refreshes openai_output_file_id (required by get_output_file). The output is
only fetched once the batch reports "completed".

Writes into batch_folder/<run_name>/:
  output_pilot_batch_combined.jsonl   — raw OpenAI output
  output_pilot_batch_combined.csv     — regulated output (parse_output input)

Pipeline order (run from repo root):
  carbontax-build    --run-name <name>   # parquet → batch JSONL
  carbontax-run      --run-name <name>   # upload + submit
  carbontax-download --run-name <name>   # poll + download output   (this file)
  carbontax-parse    --run-name <name>   # output → flat CSV
"""

import argparse
import sys

from openai_batch_wrapper.batch_manager import BatchManager

from carbontax.extraction.paths import DEFAULT_RUN_NAME, JOB_ID, batch_jsonl, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll and download the submitted batch output")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run folder name under batch_folder/")
    args = parser.parse_args()

    batch_manager = BatchManager(
        job_id=JOB_ID,
        input_jsonl_path=batch_jsonl(args.run_name),
        output_path=run_dir(args.run_name),
        batch_task_reset=False,
    )

    # get_batch_status() refreshes openai_output_file_id, which get_output_file() needs.
    status, status_df = batch_manager.get_batch_status()
    print(f"Batch status: {status}")
    print(status_df.to_string(index=False))

    if status != "completed":
        print(f"Batch not finished yet (status={status}); rerun once OpenAI reports 'completed'.")
        sys.exit(1)

    paths = batch_manager.get_output_file()
    print("Downloaded output:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
