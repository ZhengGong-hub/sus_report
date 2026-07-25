# Stage 3 — openai_batch

Reads the shards in `batch_folder/<run>/batch_files/` — one OpenAI batch per shard.

1. `carbontax-submit`   → upload + create one batch per shard, paced by `openai_batch.submit_wait_seconds`.
   Aborts if OpenAI's concurrent enqueued-token cap is hit; re-run once some batches finish and it resumes.
2. `carbontax-status`   (rerun until `completed`) → per-shard + aggregate state.
3. `carbontax-download` → fetch per-shard outputs, merged into `batch_folder/<run>/aggregated_output_batch_results.csv`.
4. `carbontax-parse`    → `batch_folder/<run>/parsed_aggregated_batch_output.csv` (one row per chunk).

`carbontax-download-and-parse` runs steps 3+4 in one command (same `allow_partial` switch).

`openai_batch.allow_partial: true` → download/parse/report use only the completed shards and write
`*__preview` files (never the canonical ones), for peeking before every shard finishes. Set back to `false`.
