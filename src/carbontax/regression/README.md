# Stage 5 — regression

1. Run `carbontax-panel` → get `batch_folder/<run>/regression/panel.csv`, one row per
   (company, year, window): adoption dummies (`_any`) and chunk shares (`_share`) for every
   flag, plus exposure controls (`n_filings`, `n_years`, `n_chunks`, `chunk_tokens`, `total_pages`).
   In: `batch_folder/<run>/parsed_aggregated_batch_output.csv` and
   `data/output/ciq_filing_mapping/company_esgfiling_mapping.csv`. Outcomes merge in a later step.
