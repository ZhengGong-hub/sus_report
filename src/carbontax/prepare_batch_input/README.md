# Stage 2 — prepare_batch_input

1. Run `carbontax-prepare` (one command, local only) → get, in `batch_folder/<run>/`:
   - `pilot_batch_combined_ref.parquet` — filtered chunks + company join keys,
   - `pilot_batch_combined.jsonl` — one OpenAI batch request per chunk.
