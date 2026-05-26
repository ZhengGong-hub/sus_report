import pandas as pd 
import time
import glob

# relative import 
from openai_batch_wrapper.batch_manager import BatchManager

job_id = "test_batch"
job_path = "batch.jsonl"
output_path = "to_batch"

batch_manager = BatchManager(
    job_id=job_id,
    input_jsonl_path=job_path,
    batch_task_reset=False,
    output_path=output_path
)

# batch_manager.delete_all_files()

batch_manager.get_batch_status()

if batch_manager.get_batch_status()[0] == "completed" or "expired":
    print("Batch completed or expired")
    print(batch_manager.get_output_file())
else:
    print("Batch not completed or expired")

# get the final clean and informational output file
ref_df = pd.read_parquet(f"{output_path}/{job_id}_ref.parquet")
output = pd.read_csv(f"{output_path}/output_{job_id}.csv").rename(columns={"custom_id": "chunk_ids"})

output_df = pd.merge(ref_df, output, left_on="chunk_ids", right_on="chunk_ids", how="left")
output_df.to_csv(f"{output_path}/clean_and_aggregated_output_{job_id}.csv", index=False)
print(f"Wrote {len(output_df)} rows to {output_path}/clean_and_aggregated_output_{job_id}.csv")
print(f"dealt with {ref_df['filingId'].nunique()} files")