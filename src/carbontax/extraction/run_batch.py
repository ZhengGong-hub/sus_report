"""
run_batch.py — upload and submit the combined-call batch to OpenAI.

Each run is isolated in its own folder, batch_folder/<run_name>/, which holds the
batch input JSONL, the BatchManager DuckDB state (batch_status.db), and the raw +
regulated output. A fresh run name = a fresh DB, so reruns never collide with a
previous (possibly failed) submission.

Pipeline order (run from repo root):
  carbontax-chunk                       # PDF → reference parquet   (chunking.pipeline)
  carbontax-build --run-name <name>     # parquet → batch JSONL     (extraction.build_batch)
  carbontax-run   --run-name <name>     # upload + submit           (this file)
  (wait for OpenAI to finish the batch)
  carbontax-download --run-name <name>  # poll + download output    (extraction.download_batch)
  carbontax-parse --run-name <name>     # output → flat CSV         (extraction.parse_output)
  carbontax-report --run-name <name>    # flat CSV → styled .xlsx   (analysis.excel)
  carbontax-compare                     # adoption-rate delta vs v1 (analysis.compare)
"""

import argparse

from openai_batch_wrapper.batch_manager import BatchManager

from carbontax.extraction.paths import DEFAULT_RUN_NAME, JOB_ID, batch_jsonl, run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload and submit the combined batch")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run folder name under batch_folder/")
    args = parser.parse_args()

    batch_manager = BatchManager(
        job_id=JOB_ID,
        input_jsonl_path=batch_jsonl(args.run_name),
        output_path=run_dir(args.run_name),
        batch_task_reset=False,
    )
    batch_manager.upload_file()
    batch_manager.create_batch()


if __name__ == "__main__":
    main()
