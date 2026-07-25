"""OpenAIBatchJob: the batch lifecycle for one run, sharded — submit → status → download.

OpenAI caps a batch at 50k requests AND ~200MB per input file, so prepare writes an indexed
folder of shards. Here each shard is its own BatchManager job (job_id = pilot_batch_combined_pNNN);
submit/status/download loop over the shards and the per-shard outputs are merged back into the
single aggregated_output_batch_results.csv that the parse stage already reads.
"""

from __future__ import annotations

import logging
import os
import time

import pandas as pd
from openai_batch_wrapper.batch_manager import BatchManager

from carbontax.paths import (
    batch_files_dir,
    batch_shards,
    output_csv,
    run_dir,
    shard_job_id,
    shard_output_csv_dir,
    shard_output_jsonl_dir,
)

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
        # built fresh per shard per call so we never hold many duckdb/file handles at once;
        # verbose=False silences the wrapper's per-shard INFO dumps — status() prints its own summary
        return BatchManager(
            job_id=job_id,
            input_jsonl_path=path,
            output_path=run_dir(self.run_name),
            batch_task_reset=False,
            verbose=False,
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

    @staticmethod
    def _parse_progress(progress) -> tuple[int | None, int | None, int | None]:
        # "Completed: 6873;Failed: 0;Total: 6873" → (6873, 0, 6873); None before a shard starts running
        if not isinstance(progress, str):
            return None, None, None
        counts = dict(kv.split(": ") for kv in progress.split(";"))
        return int(counts["Completed"]), int(counts["Failed"]), int(counts["Total"])

    @staticmethod
    def _aggregate_state(states: list[str]) -> str:
        # 'completed' only when every shard is; else surface a terminal failure over a still-running shard
        if all(s == "completed" for s in states):
            return "completed"
        bad = [s for s in states if s in ("failed", "expired", "cancelled")]
        return bad[0] if bad else "in_progress"

    def status(self) -> pd.DataFrame:
        """Print one combined table (the latest row per shard); return that per-shard summary frame."""
        logging.getLogger("httpx").setLevel(logging.WARNING)  # drop the per-shard HTTP request lines

        rows = []
        for i, path, job_id in self.shards:
            with self._manager(path, job_id) as m:
                state, state_df = m.get_batch_status()
            # state_df is this shard's recent history; its last row (by updated_at) is the live one,
            # carrying the request-count progress and output-file id
            latest = state_df.sort_values("updated_at").iloc[-1]
            completed, failed, total = self._parse_progress(latest["progress"])
            rows.append(
                {
                    "shard": i,
                    "status": state,
                    "completed": completed,
                    "failed": failed,
                    "total": total,
                    "updated_at": latest["updated_at"],
                }
            )

        summary = pd.DataFrame(rows)  # one row per shard, in shard order (matches self.shards)
        # Int64 keeps counts as clean integers while showing <NA> for shards not yet running
        summary[["completed", "failed", "total"]] = summary[["completed", "failed", "total"]].astype("Int64")
        summary["pct"] = (summary["completed"] / summary["total"] * 100).round(1)  # per-shard % done
        summary["updated_at"] = pd.to_datetime(summary["updated_at"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        print(summary.to_string(index=False))

        states = summary["status"].tolist()
        agg = self._aggregate_state(states)
        done = sum(s == "completed" for s in states)
        tot = summary[["completed", "failed", "total"]].sum()
        pct = 100 * int(tot["completed"]) / int(tot["total"]) if int(tot["total"]) else 0.0
        print(
            f"\n{done}/{len(states)} shards completed — "
            f"{int(tot['completed'])} done, {int(tot['failed'])} failed of {int(tot['total'])} requests"
            f" ({pct:.1f}%)  →  aggregate: {agg}"
        )
        return summary

    def download(self, allow_partial: bool = False) -> list[str]:
        """Fetch each shard's output and merge it into one CSV that parse reads.

        Full run (allow_partial=False): requires every shard 'completed' and writes the canonical
        aggregated_output_batch_results.csv. allow_partial=True: pulls only the shards OpenAI already
        reports completed and writes aggregated_output_batch_results__preview.csv instead, leaving the
        canonical file untouched — for
        peeking while stragglers finish. The preview is a biased subset (shards are contiguous filingId
        blocks), not a random sample, so treat its numbers as indicative only.
        """
        # get_batch_status() runs inside status(): it refreshes each shard's output-file id,
        # which get_output_file() needs. summary is one row per shard, in self.shards order.
        summary = self.status()

        if allow_partial:
            selected = [s for s, st in zip(self.shards, summary["status"]) if st == "completed"]
            if not selected:
                raise RuntimeError("No shard has completed yet — nothing to preview.")
            print(f"Partial preview: downloading {len(selected)}/{len(self.shards)} completed shards")
        else:
            agg = self._aggregate_state(summary["status"].tolist())
            if agg != "completed":
                raise RuntimeError(
                    f"Not all shards finished (aggregate={agg}); rerun once every shard reports 'completed', "
                    f"or set openai_batch.allow_partial: true to preview just the completed shards."
                )
            selected = self.shards

        shard_csvs = []
        for i, path, job_id in selected:
            with self._manager(path, job_id) as m:
                paths = m.get_output_file()  # writes output_<job_id>.csv/.jsonl flat into the run folder
            csv_dest = self._relocate_shard_output(paths[0], paths[1])
            shard_csvs.append(csv_dest)  # the regulated .csv, now under batch_output/csv/
            print(f"Shard {i} output → {csv_dest}")

        merged = self._merge(shard_csvs, preview=allow_partial)
        print(f"Merged {len(shard_csvs)} shard outputs → {merged}")
        return [merged]

    def _relocate_shard_output(self, csv_src: str, jsonl_src: str) -> str:
        # the wrapper writes both files flat into the run root; move them under batch_output/{csv,jsonl}/
        # so many shards don't clutter it. os.replace overwrites, so reruns are idempotent. returns the
        # moved csv path (what _merge reads)
        csv_dir = shard_output_csv_dir(self.run_name)
        jsonl_dir = shard_output_jsonl_dir(self.run_name)
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(jsonl_dir, exist_ok=True)
        csv_dest = os.path.join(csv_dir, os.path.basename(csv_src))
        os.replace(csv_src, csv_dest)
        os.replace(jsonl_src, os.path.join(jsonl_dir, os.path.basename(jsonl_src)))
        return csv_dest

    def _merge(self, shard_csvs: list[str], preview: bool = False) -> str:
        # concatenate the per-shard regulated CSVs into the single output CSV that parse reads
        frames = [pd.read_csv(p, dtype={"custom_id": "string"}) for p in shard_csvs]
        merged_df = pd.concat(frames, ignore_index=True)
        dest = output_csv(self.run_name, preview=preview)
        merged_df.to_csv(dest, index=False)
        logger.info("Merged %d shard rows → %s", len(merged_df), dest)
        return dest
