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

Use `outcome_transform: [log, delta_log]` to estimate both levels in logs and annual
log changes for every configured specification. A single value (e.g. `log`) still works.
Results include a `transform` column; individual folders and summary columns identify
both the transformation and the exact FE set. `delta_log` requires previous-year
outcome columns in the panel and can use a smaller sample than `log`.

Each specification's `summary.md` ends with a concise sample audit. A sentence above the table gives retained report counts
and their aggregation into window-1 firm-years. The table starts with panel firm-years, period alignment, outcome availability, log eligibility,
lag availability, complete cases, and final estimation sample. Counts include companies.
Alignment is shown separately because rejected matches are blanked in the saved panel;
outcome availability here is not a measure of raw Trucost coverage. Lagged report variables
come from the full selected-window panel, independently of prior outcome eligibility.
The sequential counts are also stored as JSON in the results CSV's `sample_audit` column.

Each regression run also writes PNG coefficient plots under `regression/results/graphs/`.
One plot per family, outcome, transformation, window and lag overlays the fixed-effect
models, with distinct markers/colors, sample counts, model settings, and estimator-provided
95% confidence intervals. Measure coefficients are plotted jointly; controls are included
in estimation and named in the caption. Filenames include every grouping parameter.

Per-specification `results.csv` and `summary.md` files live in
`regression/results/tables/<outcome>/<transform>/<family>/w<window>__lag<lag>__fe_<effects>/`; PNG comparisons live alongside in
`regression/results/graphs/`.

Both output trees are grouped by outcome → transformation → regressor family.
Graphs end in `w<window>__lag<lag>.png`; tables have one window/lag/FE folder
containing `results.csv` and `summary.md`. Earlier outputs with old names are preserved
under `tables/_legacy/` and are not refreshed by new runs.
