# Stage 3 — openai_batch

1. Run `carbontax-submit` → uploads the stage-2 JSONL, creates the OpenAI batch.
2. Run `carbontax-status` (rerun until `completed`) → get the batch state.
3. Run `carbontax-download` → get the raw + regulated output files in `batch_folder/<run>/`.
4. Run `carbontax-parse` → get `batch_folder/<run>/parsed.csv` (one row per chunk).
