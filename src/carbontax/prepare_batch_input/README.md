# Stage 2 — prepare_batch_input

1. Run `carbontax-prepare` (one command, local only) → get, in `batch_folder/<run>/`:
   - `batch_ref.parquet` — filtered chunks + company join keys,
   - `batch.jsonl` — one OpenAI batch request per chunk,
   - `batch_jsonl_summary.md` — composition/token/cost report + skipped PDFs.
