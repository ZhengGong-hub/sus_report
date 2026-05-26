import pandas as pd 
import time
import glob

# relative import 
from openai_batch_wrapper.batch_manager import BatchManager

job_path = "to_batch/test_batch.jsonl"

batch_manager = BatchManager(
    job_id="test_batch",
    input_jsonl_path=job_path,
    batch_task_reset=False
)

# batch_manager.delete_all_files()

batch_manager.upload_file()
batch_manager.create_batch()