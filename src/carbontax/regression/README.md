# Stage 5 — regression

1. Run `carbontax-panel` → get `batch_folder/<run>/regression/panel.csv`, one row per
   (company, year, window): adoption dummies (`_any`) and chunk shares (`_share`) for every
   flag, plus exposure controls (`n_filings`, `n_years`, `n_chunks`, `chunk_tokens`, `total_pages`).
   In: `batch_folder/<run>/parsed_aggregated_batch_output.csv` and
   `data/output/ciq_filing_mapping/company_esgfiling_mapping.csv`. Outcomes merge in a later step.
2. Run `carbontax-regress` → get `regression/results.csv` and `regression/summary.md`: two
   pyfixest regressions, `ln(absolute)` and `ln(intensity)` on all 30 `tier2_*_any` dummies
   with firm + year FE and SEs clustered on the firm. In: `regression/panel.csv`.
