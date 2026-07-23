# Stage 3 — openai_batch

Reads the shards in `batch_folder/<run>/batch_files/` — one OpenAI batch per shard.

1. `carbontax-submit`   → upload + create one batch per shard, paced by `openai_batch.submit_wait_seconds`.
   Aborts if OpenAI's concurrent enqueued-token cap is hit; re-run once some batches finish and it resumes.
2. `carbontax-status`   (rerun until `completed`) → per-shard + aggregate state.
3. `carbontax-download` → fetch per-shard outputs, merged into `batch_folder/<run>/output_<JOB_ID>.csv`.
4. `carbontax-parse`    → `batch_folder/<run>/parsed.csv` (one row per chunk).
