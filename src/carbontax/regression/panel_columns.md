# `panel.csv` — column dictionary

One row per **(companyid, year, window)**. Written by `PanelBuilder.run()` (stage 5a) to
`batch_folder/<run_name>/regression/panel.csv`. Current shape for `test_trucost`: **17,933 × 313**
(windows 1, 2, 3).

Three sources feed it:

| source | file | supplies |
|---|---|---|
| **LLM flags** | `parsed_aggregated_batch_output.csv` | one row per scored chunk → the 39 taxonomy booleans |
| **CIQ mapping** | `data/output/ciq_filing_mapping/company_esgfiling_mapping.csv` | filing → companyid, periodDate, fileType, noOfPages |
| **Trucost** | `data/input/trucost_environment.csv` | emissions outcomes, disclosure scores, firm metadata |
| **stage-2 ref** | `batch_folder/<run_name>/batch_ref.parquet` | `n_chunks_total` (chunks per filing before the keyword filter) |

A cell exists for every year between a firm's first and last report, not only years it filed —
in a gap year the firm still carries the stock of what it has already disclosed. Only filings
with `fileType == "Sustainability Report"` contribute.

---

## Keys and structure (3)

| column | type | meaning |
|---|---|---|
| `companyid` | string | CIQ company id. Read as string everywhere — as int64 it stops matching upstream keys. |
| `year` | int | The cell's anchor year. For a filing matched to a Trucost period (±31d) this **is** that row's `fiscalyear`; otherwise the calendar year of the report's `periodDate`. |
| `window` | int | Backward pooling width in years, inclusive of the anchor. `1` = that year's report alone; `3` = the anchor plus the two before it. The same company-year appears once per window. |

## Filing-level aggregates (7)

Computed over the **distinct filings** pooled into the cell (deduped on companyid-year-filingId,
so a filing reused across anchors in a wide window isn't double-counted).

| column | type | meaning |
|---|---|---|
| `n_filings` | int | Reports pooled into this cell. |
| `n_years` | int | Distinct source years covered. `1` for window=1; less than the window when the firm skipped a year or sits at the panel's left edge. Used as the `n_years` control. |
| `filed_this_year` | bool | `False` = a gap-year cell, carried entirely by earlier reports (window>1 only). |
| `anchor_matched` | bool | Did a report **in the anchor year** pair with a real Trucost period? Only an anchor-year filing can vouch for the cell's year label. |
| `n_chunks_total` | float | Every chunk the window's reports split into, keyword-matched or not. **Denominator for `_share`** when `share_denominator: all_chunks`. From the stage-2 ref parquet. |
| `total_pages` | float | Sum of `noOfPages` over the cell's filings. NaN when no filing reports pages (~17% of cells). |
| `pages_missing` | bool | At least one pooled filing has no page count — read `total_pages` as a lower bound. |

## Chunk-level aggregates (2)

| column | type | meaning |
|---|---|---|
| `n_chunks` | int | Chunks actually sent to the LLM, i.e. those that passed the keyword filter. This is the **sampling-exposure** measure: a flag can only fire in a chunk that was read. Feeds the `log_n_chunks` control. |
| `chunk_tokens` | int | Sum of `prompt_tokens` over those chunks. |

Note `n_chunks ≤ n_chunks_total` always; their ratio is keyword retention (median 0.33).

## Taxonomy flags — `_any` (39)

`tier1_{S1,S2,S3U,S3D,CDR}`, `tier2_{30 measures}`, `governance_{sbti, internal_carbon_price,
exec_comp_linked, third_party_assurance}`. Defined in `taxonomy.py`.

`<flag>_any` = 1 if the measure appears in **at least one** chunk pooled into the cell. This is the
dummy form, unaffected by any denominator choice.

**Interpretation caveat:** a flag means *"the report says the firm does X"*, not *"the firm does X"*.
It is disclosure text, filtered by `prepare_batch_input.filter_keywords` and sampled at chunk level.
Measured persistence between adjacent reports: P(1 | 1 last year) = 59% for `tier2_ppa_any` against
a 12% base rate. Strongly stock-like, but the 41% of 1→0 transitions are mostly sampling misses
rather than firms abandoning measures — classical measurement error, which attenuates coefficients
toward zero.

## Taxonomy flags — `_share` (39)

```
<flag>_share = chunks in the cell carrying the flag / denominator
```

Denominator is set by `regression.panel.share_denominator`:
`all_chunks` → `n_chunks_total` (current), `keyword_chunks` → `n_chunks`.

`all_chunks` is used because the keyword list contains measure terms (`renewable`, `solar`,
`energy efficiency`), so the keyword-matched count grows with the treatment itself — a firm-year
where `tier1_S2_any` fires has 2.6× the kept chunks. Switching denominators does **not** simply
rescale: rank correlation between the two bases is 0.77–0.87.

Recoverable either way: `share_keyword = share_all × n_chunks_total / n_chunks`.

## Trucost outcomes (15 of the 197 data-item columns)

Five scopes × three columns each. `absolute` is tCO2e; `intensity` is tCO2e per $M of revenue;
`score` is disclosure provenance.

`intensity` equals `absolute / trucost_total_revenue` for 96.1% of rows where all three are
present, not all of them — Trucost computes intensity on its own revenue basis, so treat the
identity as close but not definitional. The regression uses whichever column the spec names
rather than deriving one from the other.

| scope | absolute | intensity | score |
|---|---|---|---|
| S1 | `absolute_ghg_scope_1` | `intensity_ghg_scope_1` | `ghg_scope_1_score` |
| S2L | `absolute_ghg_scope_2_location_based` | `intensity_ghg_scope_2_location_based` | `ghg_scope_2_location_based_score` |
| S2M | `absolute_ghg_scope_2_market_based` | `intensity_ghg_scope_2_market_based` | `ghg_scope_2_market_based_score` |
| S3U | `absolute_ghg_scope_3_upstream_total` | `intensity_ghg_scope_3_upstream` | `ghg_scope_3_upstream_score` |
| S3D | `absolute_ghg_scope_3_downstream_total` | `intensity_ghg_scope_3_downstream_total` | `ghg_scope_3_downstream_total_score` |

**Score scale** (undocumented by WRDS — read off the score × disclosure-text crosstab):

| score | provenance |
|---|---|
| 1.0 / 2.0 | the firm's own disclosure (CDP, CSR/environmental report, annual report) |
| 2.4 | partial disclosure, or extrapolated from the prior year |
| 3.0 | Trucost model from physical intensity factors (S3 downstream only) |
| 4.0 | Trucost model from revenue intensity factors, or disclosure rejected |

Trucost writes **exact zeros** for not-applicable categories — absences, not measurements. The
regression drops `y <= 0` before logging.

Coverage across all windows, as the build logs it: S1/S2L/S3U 92.6% of cells, S3D 87.9%,
**S2M only 23.5%** (24.1% at window=1).

## Other Trucost data items (182)

Every `di_*` item in the WRDS export, renamed to readable names via
`regression/trucost_vars.py`. Families: `absolute_*` / `intensity_*` / `impact_ratio_*` /
`weighted_disclosure_*` cost and quantity measures for air pollutants, water, waste, land use and
natural resources, plus energy and revenue breakdowns.

They ride along unused by the current specs — `OUTCOMES` picks the y candidates, but controls,
heterogeneity splits and robustness rows may want others. `trucost_total_revenue` is the one the
code depends on (it builds `log_revenue`).

To recover the original WRDS code for any of them, invert `TRUCOST_VARS`.

## Previous-year values — `_prev` (11)

The 10 outcome columns plus `trucost_total_revenue`, each carrying **year t−1's** value.

Taken off the **Trucost** frame, which is near-continuous (12.9 years per firm, 91% of its cells
have a real t−1), not off this panel, which only has years the firm filed. Matched by explicit
`year + 1` arithmetic, never a positional shift — Trucost has gaps and a shift would silently pass
off t−2 as the previous year.

Feeds the `intensity_growth` / `absolute_growth` outcome forms, which compute
`ln(y_t) − ln(y_t−1)` and require both years strictly positive. That is a smaller sample than the
level specs: S1 89.6% of cells vs 92.3%, **S2M 13.7% vs 23.2%**.

## Disclosure flags — `_disclosed` (5)

`S1_disclosed`, `S2L_disclosed`, `S2M_disclosed`, `S3U_disclosed`, `S3D_disclosed`.

`<scope>_disclosed = score <= regression.panel.disclosed_max_score` (currently **2.4**). A missing
score counts as **not** disclosed — provenance unknown is not firm-verified.

This only *labels*; it drops nothing. The sample cut is made by `regression.specs.disclosed_only`,
whose first entry is currently `false`, so the headline spec runs on all data. The raw `*_score`
columns stay in the panel either way.

## Join trust — `outcome_trusted` (1)

`True` if a matched report period vouches for this cell's year:

- cell filed in its anchor year → `anchor_matched`
- gap-year cell (no anchor filing) → the firm paired with a Trucost period somewhere else

When `False`, **every Trucost-sourced column is nulled** — outcomes, metadata, `_prev`, all of it.
Without this the `(companyid, year)` merge still reaches a Trucost row for cells the date join
rejected, pairing a report with a period months away. Costs 1,284 cells and 534 S1 outcomes
(95.6% → 92.6% coverage): a missing outcome drops the cell, a wrong one biases the coefficient.

## Trucost firm metadata (9)

`periodenddate`, `fiscalyear`, `gvkey`, `ticker`, `companyname`, `country`, `simpleindustry`,
`tcprimarysectorid`, `reportedcurrencyisocode`.

Carried for sample description, heterogeneity splits and sector×year variants — firm FE absorbs all
of them in the main spec. Non-null exactly where the outcome merge landed (92.6%). Note the universe
is essentially all-US: 2,038 of 2,040 firms.

---

## Reading notes

- **Not a regression sample.** The estimator further drops `y <= 0`, rows with missing regressors,
  singleton firms (iterated, by pyfixest), and flags identified off fewer than `min_switchers` firms.
- **Rows are not independent.** The same company-year appears once per window, and within a wide
  window neighbouring cells reuse the same filings. Never pool windows in one regression.
- **`year` is fiscal, not calendar**, wherever the date join matched — it is the Trucost row's own
  `fiscalyear`.
- **Gap-year cells** (`filed_this_year == False`) reuse an earlier report's flags against a fresh
  outcome. 7.1% of window=3 cells; they add no new firms.
