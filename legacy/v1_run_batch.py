import pandas as pd 
import time
import glob

# relative import 
from openai_batch_wrapper.batch_manager import BatchManager


TIERS = ["tier1", "tier2"]

for tier in TIERS:
    job_path = f"to_batch_pilot/pilot_batch_{tier}.jsonl"

    batch_manager = BatchManager(
        job_id=f"pilot_batch_{tier}",
        input_jsonl_path=job_path,
        batch_task_reset=False
    )

    # batch_manager.delete_all_files()

    batch_manager.upload_file()
    batch_manager.create_batch()