"""OpenAIBatchJob: the batch lifecycle for one run — submit → status → download."""

from __future__ import annotations

import logging

from openai_batch_wrapper.batch_manager import BatchManager

from carbontax.paths import JOB_ID, batch_jsonl, run_dir

logger = logging.getLogger(__name__)


class OpenAIBatchJob:

    def __init__(self, run_name: str):
        self.run_name = run_name
        # rebuilt from the run folder every time: batch_status.db inside it
        # remembers the OpenAI batch id across submit/status/download calls
        self.manager = BatchManager(
            job_id=JOB_ID,
            input_jsonl_path=batch_jsonl(run_name),
            output_path=run_dir(run_name),
            batch_task_reset=False,
        )

    def submit(self) -> None:
        """Upload the batch JSONL and create the OpenAI batch."""
        self.manager.upload_file()
        self.manager.create_batch()

    def status(self) -> str:
        """Print batch state + request counts; also refreshes the output-file id."""
        status, status_df = self.manager.get_batch_status()
        print(f"Batch status: {status}")
        print(status_df.to_string(index=False))
        return status

    def download(self) -> list[str]:
        """Fetch raw + regulated output once the batch is completed."""
        # get_batch_status() must run first: it refreshes openai_output_file_id,
        # which get_output_file() needs
        status = self.status()
        if status != "completed":
            raise RuntimeError(
                f"Batch not finished yet (status={status}); rerun once OpenAI reports 'completed'."
            )
        paths = self.manager.get_output_file()
        print("Downloaded output:")
        for p in paths:
            print(f"  {p}")
        return paths
