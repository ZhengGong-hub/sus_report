# Stage 1 — acquire_pdfs

1. Run `carbontax-acquire` with `init_folders: true` (once) → get `data/output/intermed/<n>/` batch folders (10 companies each).
2. Run `carbontax-acquire` → get the PDFs in `data/output/sustain_reports_pdfs/` and search hits in `intermed/<n>/fileids.csv`.
3. Run `carbontax-mapping` → get `data/output/ciq_filing_mapping/company_esgfiling_mapping.csv` (company↔filing mapping).
