"""OpenAIBatchJob: the batch lifecycle for one run, sharded — submit → status → download.

OpenAI caps a batch at 50k requests AND ~200MB per input file, so prepare writes an indexed
folder of shards. Here each shard is its own BatchManager job (job_id = pilot_batch_combined_pNNN);
submit/status/download loop over the shards and the per-shard outputs are merged back into the
single output_{JOB_ID}.csv that the parse stage already reads.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
from openai_batch_wrapper.batch_manager import BatchManager

from carbontax.paths import batch_files_dir, batch_shards, output_csv, run_dir, shard_job_id

logger = logging.getLogger(__name__)


class OpenAIBatchJob:

    def __init__(self, run_name: str):
        self.run_name = run_name
        shard_paths = batch_shards(run_name)
        if not shard_paths:
            raise FileNotFoundError(
                f"No batch shards in {batch_files_dir(run_name)} — run prepare_batch_input first."
            )
        # (shard_index, jsonl_path, job_id): one BatchManager job per shard, all sharing
        # batch_status.db in the run folder — its rows are keyed by job_id, so they stay isolated
        self.shards = [(i, p, shard_job_id(i)) for i, p in enumerate(shard_paths)]

    def _manager(self, path: str, job_id: str) -> BatchManager:
        # built fresh per shard per call so we never hold many duckdb/file handles at once
        return BatchManager(
            job_id=job_id,
            input_jsonl_path=path,
            output_path=run_dir(self.run_name),
            batch_task_reset=False,
        )

    def submit(self, wait_s: int) -> None:
        """Upload + create one batch per shard, sequentially.

        The enqueued-token limit is a *concurrent* budget across all in-progress batches, so
        OpenAI rejects a shard once the in-flight total would exceed it. That raises and aborts
        here (fail-fast); because already-submitted shards carry their batch id in batch_status.db,
        re-running once some batches complete resumes at the first unsubmitted shard. We only pause
        wait_s after a shard we actually created, so resumes skip past finished shards instantly.
        """
        for pos, (i, path, job_id) in enumerate(self.shards):
            with self._manager(path, job_id) as m:
                if m.openai_batch_id:  # submitted in an earlier run — skip, no wait
                    print(f"Shard {i} ({job_id}) already submitted ({m.openai_batch_id})")
                    continue
                print(f"Submitting shard {i} ({job_id}) ← {path}")
                m.upload_file()
                m.create_batch()
            if pos < len(self.shards) - 1:
                print(f"  waiting {wait_s}s before next shard…")
                time.sleep(wait_s)

    def status(self) -> str:
        """Print each shard's state + counts; return 'completed' only when every shard is."""
        states = []
        for i, path, job_id in self.shards:
            with self._manager(path, job_id) as m:
                state, state_df = m.get_batch_status()
            print(f"Shard {i} ({job_id}): {state}")
            print(state_df.to_string(index=False))
            states.append(state)

        if all(s == "completed" for s in states):
            agg = "completed"
        else:
            # surface a terminal failure over a still-running shard so it isn't masked
            bad = [s for s in states if s in ("failed", "expired", "cancelled")]
            agg = bad[0] if bad else "in_progress"
        print(f"Aggregate across {len(states)} shards: {agg}")
        return agg

    def download(self) -> list[str]:
        """Fetch each shard's output once all are completed, then merge into output_{JOB_ID}.csv."""
        # get_batch_status() runs inside status(): it refreshes each shard's output-file id,
        # which get_output_file() needs
        state = self.status()
        if state != "completed":
            raise RuntimeError(
                f"Not all shards finished (aggregate={state}); rerun once every shard reports 'completed'."
            )

        shard_csvs = []
        for i, path, job_id in self.shards:
            with self._manager(path, job_id) as m:
                paths = m.get_output_file()  # writes output_<job_id>.csv/.jsonl into the run folder
            shard_csvs.append(paths[0])  # the regulated .csv
            print(f"Shard {i} output: {paths}")

        merged = self._merge(shard_csvs)
        print(f"Merged {len(shard_csvs)} shard outputs → {merged}")
        return [merged]

    def _merge(self, shard_csvs: list[str]) -> str:
        # concatenate the per-shard regulated CSVs into the single output_{JOB_ID}.csv that parse reads
        frames = [pd.read_csv(p, dtype={"custom_id": "string"}) for p in shard_csvs]
        merged_df = pd.concat(frames, ignore_index=True)
        dest = output_csv(self.run_name)
        merged_df.to_csv(dest, index=False)
        logger.info("Merged %d shard rows → %s", len(merged_df), dest)
        return dest
