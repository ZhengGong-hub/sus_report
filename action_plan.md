# Carbon-Reduction Measure Project — Summary & Coding Plan
 
---
 
## Part 1 — Proposal summary
 
### The research question
Which categories of corporate carbon-reduction measures are actually associated with realised emission reductions? Panel regression of annual firm-level emission changes on binary adoption indicators, one per measure category.
 
### The data approach
Large-scale LLM extraction of measure adoption from raw PDF sustainability reports, covering a firm universe wider than CDP and across multiple years.
 
### The taxonomy — two tiers
 
**Tier 1 (5 buckets, anchored in GHG Protocol + IPCC AR6 WGIII):**
- `S1` — Scope 1, direct emissions
- `S2` — Scope 2, purchased energy
- `S3U` — Scope 3 upstream (Categories 1–8)
- `S3D` — Scope 3 downstream (Categories 9–15)
- `CDR` — carbon removal and offsets
**Tier 2 (27 specific measures):** 6 under S1, 4 under S2, 8 under S3U (one per Scope 3 upstream category C1–C8), 7 under S3D (C9–C15), 3 under CDR.
 
**Governance (4 parallel regressors):** SBTi commitment, internal carbon price, executive comp linked to emissions, third-party assurance.
 
### Pilot
~100 reports, stratified by sector. Tests extractability, vague/aspirational share, and inter-year consistency. Categories with poor agreement get merged or demoted to Tier-1-only.
 
### Open issue (separate memo)
Identification — adoption is not random. To be handled with fixed effects, staggered-adoption designs around CSRD / EU ETS, and propensity-score matching.
 
---
 
## Part 2 — Coding plan
 
**Stack:** Python 3.11+, OpenAI API, local folder, Parquet outputs.
 
### Repo layout
 
```
carbon-measures/
├── data/
│   ├── reports/              # raw PDFs: {ticker}_{year}.pdf
│   ├── emissions/            # firm-year emissions panel (CSV from CDP/Refinitiv/etc.)
│   └── extracted/            # Parquet outputs from the LLM pipeline
├── src/
│   ├── taxonomy.py           # the 27 Tier-2 categories + 4 governance flags as constants
│   ├── prompt.py             # the extraction prompt template
│   ├── pdf_utils.py          # PDF → text (chunked)
│   ├── extract.py            # per-report LLM call → structured row
│   ├── run_pilot.py          # orchestrates the pilot run
│   └── analyze_pilot.py      # agreement, vague-rate, inter-year consistency
├── notebooks/
│   └── pilot_results.ipynb   # tables + plots for the pilot memo
├── prompts/
│   └── extraction_v1.txt     # the actual prompt (versioned)
├── .env                      # OPENAI_API_KEY
└── requirements.txt
```
 
### Dependencies
 
```
openai>=1.40
pypdf>=4.0
pdfplumber>=0.11        # fallback for tricky PDFs
pandas>=2.2
pyarrow>=15
pydantic>=2.7           # schema validation on LLM output
tenacity>=8.2           # retry on API errors
tqdm>=4.66
python-dotenv>=1.0
```
 
### Step 1 — Codify the taxonomy
 
`src/taxonomy.py`: one source of truth, used by the prompt and the schema validator.
 
```python
TIER1 = ["S1", "S2", "S3U", "S3D", "CDR"]
 
TIER2 = {
    "S1": ["energy_efficiency", "fuel_switching", "onsite_renewables",
           "fgas_substitution", "methane_fugitive", "process_emissions"],
    "S2": ["ppa", "rec_goo", "247_cfe", "low_carbon_heat"],
    "S3U": ["c1_purchased_goods", "c2_capital_goods", "c3_fuel_energy",
            "c4_upstream_transport", "c5_waste_ops", "c6_business_travel",
            "c7_commuting", "c8_upstream_leased"],
    "S3D": ["c9_downstream_transport", "c10_processing", "c11_use_phase",
            "c12_eol", "c13_downstream_leased", "c14_franchises", "c15_investments"],
    "CDR": ["nbs", "tech_cdr", "voluntary_offsets"],
}
 
GOVERNANCE = ["sbti", "internal_carbon_price", "exec_comp_linked", "third_party_assurance"]
```
 
### Step 2 — Sample selection
 
`data/sample.csv`: 100 firms × report-year, with columns `ticker, year, sector, cdp_status, pdf_path`. Pull tickers from a clean universe (e.g. STOXX 600, S&P 500, plus a non-CDP tail). Stratify across the six sectors named in the proposal so the pilot tests extractability across language styles.
 
### Step 3 — PDF → text
 
`pdf_utils.py`: extract text with `pypdf`, fall back to `pdfplumber` for failures. Sustainability reports are often 80–200 pages. Two choices:
 
- **Whole document, one call:** simple but costs more and risks truncation on huge reports.
- **Chunked + map-reduce:** split into ~10-page chunks, extract per chunk, then merge by OR-ing the binaries and concatenating the source quotes.
Recommend chunked — more robust and cheaper. Keep page numbers in chunk metadata so source quotes are traceable.
 
### Step 4 — The prompt
 
`prompts/extraction_v1.txt`. Two principles:
 
1. Force structured JSON output matching the schema below.
2. Require a **source quote** (verbatim, ≤30 words, with page number) for every positive hit. This is your audit trail and the input to manual spot-checking.
```
You are extracting corporate carbon-reduction measures from sustainability
report text. For each of the 27 Tier-2 categories and 4 governance flags,
decide whether the company describes adopting or actively pursuing that
measure in the reporting year.
 
Return JSON matching this schema exactly:
{
  "tier2": {"energy_efficiency": {"adopted": bool, "quote": str|null, "page": int|null}, ...},
  "governance": {"sbti": {"adopted": bool, "quote": str|null, "page": int|null}, ...},
  "notes": "any sector-specific measure mentioned that didn't fit"
}
 
Rules:
- "Adopted" means concrete actions in the reporting year, not aspirational
  language ("we are committed to..." alone = false).
- "quote" must be verbatim, ≤30 words, copied from the text.
- If unsure, set adopted=false. Do not infer beyond the text.
```
 
Version the prompt. When you change it, bump to `extraction_v2.txt` and store the version in every output row.
 
### Step 5 — Schema validation with Pydantic
 
```python
from pydantic import BaseModel
from typing import Optional
 
class Measure(BaseModel):
    adopted: bool
    quote: Optional[str] = None
    page: Optional[int] = None
 
class Extraction(BaseModel):
    tier2: dict[str, Measure]
    governance: dict[str, Measure]
    notes: str = ""
```
 
After every API call, parse the JSON into `Extraction`. Reject and retry on schema mismatch. Cheap insurance against malformed LLM output.
 
### Step 6 — The extraction function
 
`extract.py`:
 
```python
@retry(stop=stop_after_attempt(3), wait=wait_exponential())
def extract_report(text_chunks: list[str], firm: str, year: int) -> Extraction:
    # one call per chunk, then merge
    chunk_results = [call_openai(chunk) for chunk in text_chunks]
    return merge(chunk_results)
```
 
Use OpenAI's structured outputs / JSON mode (response_format with the schema). Set temperature=0 for reproducibility. Log every call (input tokens, output tokens, latency, cost) to a JSONL file so you can audit later.
 
### Step 7 — Output to Parquet
 
One row per firm-year. Columns:
 
| Column | Type | Notes |
|---|---|---|
| `firm_id` | str | ticker or LEI |
| `year` | int | report year |
| `sector` | str | stratification variable |
| `s1_energy_efficiency` | bool | one column per Tier-2 measure (27 total) |
| `s1_energy_efficiency_quote` | str | source quote |
| `s1_energy_efficiency_page` | int | page number |
| `... × 26 more ...` | | |
| `gov_sbti` | bool | one column per governance flag (4 total) |
| `... × 3 more ...` | | |
| `prompt_version` | str | e.g. "v1" |
| `model` | str | e.g. "gpt-4o-2024-11" |
| `extraction_timestamp` | datetime | for replication |
 
Save to `data/extracted/pilot_v1.parquet`. Parquet is small, typed, and pandas/R/Stata can all read it.
 
### Step 8 — Pilot analysis
 
`analyze_pilot.py` computes:
 
1. **Adoption rate per category** — how often each binary fires. Categories firing <10% (or >95%) are candidates for cutting; no regression power either way.
2. **Sector × category heatmap** — sanity check that financials show C15 (investments) and industrials show process emissions.
3. **Vague-rate** — share of reports where Tier-2 binaries are mostly false but the report does mention climate action somewhere. Flag for manual review.
4. **Inter-year consistency** — for firms with ≥2 years in the pilot, compute the share of categories that flip on/off year-to-year. High flip rates = either the LLM is noisy or the firm genuinely changed disclosure; either way, you need to know.
5. **Spot-check sample** — randomly draw 20 positive hits, check the source quotes really do support the binary. This is your manual QC step.
Output: a notebook with tables and small plots, ready to drop into the pilot memo for your supervisor.
 
### Step 9 — Cost & time budget for the pilot
 
Back-of-envelope, GPT-4o, chunked PDFs averaging 100 pages:
- ~10 chunks × ~3,000 input tokens + ~500 output tokens per chunk = ~35k tokens per report
- 100 reports = ~3.5M tokens total
- At GPT-4o pricing, roughly $15–25 for the full pilot extraction
- Wall time: ~2 hours with sensible parallelism (5–10 concurrent requests)
Cheap enough to re-run the whole pilot if you change the prompt.
 
### Step 10 — Versioning and reproducibility
 
- Git the repo from day one.
- Every output Parquet has `prompt_version` and `model` columns — never silently overwrite.
- Pin the OpenAI model snapshot (e.g. `gpt-4o-2024-11-20`), not the floating `gpt-4o`.
- Store the random seed for sample selection.
### What this gets you for the supervisor meeting
 
After the pilot run:
- A Parquet panel of 100 firm-years × 31 binary indicators with source quotes.
- A short notebook showing adoption rates, sector splits, vague-rate, and inter-year flips.
- A clear list of categories that should be merged, cut, or demoted to Tier-1-only.
That becomes the pilot memo. Once approved, scale to the full universe by running the same `extract_report` function over the bigger sample.