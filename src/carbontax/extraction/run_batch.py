"""
run_batch.py — upload and submit the combined-call batch to OpenAI.

Pipeline order (run from repo root):
  carbontax-chunk            # PDF → reference parquet      (chunking.pipeline)
  carbontax-build            # parquet → batch JSONL        (extraction.build_batch)
  carbontax-run              # upload + submit              (this file)
  (wait for OpenAI to finish the batch)
  carbontax-parse            # output → flat CSV            (extraction.parse_output)
  carbontax-compare          # adoption-rate delta vs v1    (analysis.compare)
"""

from openai_batch_wrapper.batch_manager import BatchManager

JOB_ID     = "pilot_batch_combined"
INPUT_PATH = f"to_batch_pilot/{JOB_ID}.jsonl"


def main() -> None:
    batch_manager = BatchManager(
        job_id=JOB_ID,
        input_jsonl_path=INPUT_PATH,
        batch_task_reset=False,
    )
    batch_manager.upload_file()
    batch_manager.create_batch()


if __name__ == "__main__":
    main()
