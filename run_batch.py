"""
run_batch.py — upload and submit the v2 combined batch to OpenAI.

Run order:
  1. python build_batch.py     # generate the JSONL (already done if parquet exists)
  2. python run_batch.py       # this file — upload + submit
  3. (wait for OpenAI)
  4. python parse_output.py    # flatten output to CSV
  5. python compare_v1_v2.py   # adoption-rate delta vs v1
"""

from openai_batch_wrapper.batch_manager import BatchManager

JOB_ID     = "pilot_batch_combined"
INPUT_PATH = f"to_batch_pilot/{JOB_ID}.jsonl"

batch_manager = BatchManager(
    job_id=JOB_ID,
    input_jsonl_path=INPUT_PATH,
    batch_task_reset=False,
)

batch_manager.upload_file()
batch_manager.create_batch()
