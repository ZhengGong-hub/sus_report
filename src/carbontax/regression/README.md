# Stage 5 — regression

1. Run `carbontax-panel` → get `batch_folder/<run>/regression/panel.csv`, one row per
   (company, year, window): adoption dummies (`_any`) and chunk shares (`_share`) for every
   flag, plus exposure controls (`n_filings`, `n_years`, `n_chunks`, `chunk_tokens`, `total_pages`).
   In: `batch_folder/<run>/parsed_aggregated_batch_output.csv` and
   `data/output/ciq_filing_mapping/company_esgfiling_mapping.csv`. Outcomes merge in a later step.
2. Run `carbontax-regress` → get `regression/results.csv` and `regression/summary.md`: two
   pyfixest regressions, `ln(absolute)` and `ln(intensity)` on all 30 `tier2_*_any` dummies
   with firm + year FE and SEs clustered on the firm. In: `regression/panel.csv`.

`regression.specs.fixed_effects` in `config/run_test_trucost.yaml` runs each set separately:

```yaml
sector_mapping_csv: data/input/ciq_simple_industries_by_sector.csv
fixed_effects:
  - [companyid, year]
  - [sector, year]
  - [sector_year]
  - []
```

Sector names are mapped from the existing panel's `simpleindustry` descriptions when
regressions load; no panel rebuild is needed. Unmapped industries are logged and excluded
only from specifications using sector FE. `[sector, year]` absorbs separate sector and year
effects. `[sector_year]` absorbs one effect per sector–year combination, allowing each
sector its own annual shocks. `[companyid, sector_year]` additionally absorbs company
effects; a separate year FE is already covered by the sector–year effects. Interactions
use the outcome year even when regressors are lagged. `cluster` remains independently configurable.
The `min_switchers` filter applies only to specifications containing company FE.
