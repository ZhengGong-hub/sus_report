# carbontax

LLM-based extraction of corporate carbon-reduction measures from sustainability-report
PDFs, classified into a sourced two-tier taxonomy (GHG Protocol boundaries × IPCC AR6
WGIII mitigation options), to be linked to firm-level emission outcomes. US-exclusive sample.

This repo covers the **taxonomy + extraction** layer. The current ("v2") approach runs a
single combined LLM call per text chunk; the earlier two-call approach is archived under
[`legacy/`](legacy/README.md).

## Layout

```
src/carbontax/            installed package (editable)
├── taxonomy.py           SINGLE SOURCE OF TRUTH — measures, schema, prompt (v2)
├── schemas.py            generic LLM call schemas (filter, summary)
├── utils/
│   ├── logger.py         Logger class
│   └── llm.py            LLMWrapper (OpenAI/Gemini), DEFAULT_MODELS
├── acquire/              get raw data
│   ├── tokens.py         S&P CIQ token refresh
│   ├── pdfs.py           search + download filing PDFs
│   └── mapping.py        build company↔filing mapping CSV
├── chunking/             PDF → filtered text chunks → reference parquet
│   ├── pdf_parser.py     PDF → cleaned text blocks
│   ├── splitter.py       recursive token-window chunker
│   ├── filter.py         keyword / LLM relevance filter
│   └── pipeline.py       orchestrates the above → ref parquet
├── extraction/           chunks → batch → OpenAI → parsed CSV
│   ├── build_batch.py    ref parquet → combined-call batch JSONL
│   ├── run_batch.py      upload + submit batch
│   └── parse_output.py   batch output → flat analysis CSV
└── analysis/
    └── compare.py        per-measure adoption-rate delta (v2 vs v1)

config/pilot.yaml         which companies to chunk for the pilot
legacy/                   frozen v1 two-call implementation (see legacy/README.md)
```

The measure list lives in `taxonomy.py` **only**; the prompt and JSON schema both derive
from it. Never hardcode measures elsewhere — prompt/schema drift is the main failure mode.

## Setup

```bash
uv pip install -e .       # registers the carbontax-* console scripts
```

Requires `OPENAI_API_KEY` in `.env` (and S&P CIQ credentials for acquisition).

## Pipeline

Run from the repo root (scripts use relative data paths: `files/`, `mapping_data/`,
`to_batch_pilot/`, `output/`).

| Step | Command | In → Out |
|------|---------|----------|
| 1. Acquire PDFs | `carbontax-acquire` | CIQ API → `files/*.pdf`, `intermed/*/fileids.csv` |
| 2. Build mapping | `carbontax-mapping` | `intermed/` → `mapping_data/company_esgfiling_mapping.csv` |
| 3. Chunk | `carbontax-chunk config/pilot.yaml` | PDFs → `to_batch_pilot/pilot_batch_ref.parquet` |
| 4. Build batch | `carbontax-build` | ref parquet → `to_batch_pilot/pilot_batch_combined.jsonl` |
| 5. Submit | `carbontax-run` | JSONL → OpenAI batch (24h window) |
| 6. Parse | `carbontax-parse` | batch output → `output/parsed_v2_combined.csv` |
| 7. Compare | `carbontax-compare` | v1 vs v2 → `output/compare_v1_v2.csv` |

Each script is also runnable as a module, e.g. `python -m carbontax.extraction.build_batch --help`.

## v2 extraction design (one combined call per chunk)

- **One call** fills Tier-1 buckets, Tier-2 measures, and governance flags together — no
  Tier-1 gate that can misfire before Tier-2 runs.
- **Flat schema**: measures are bare booleans plus a single `evidence[]` array (populated
  only for `true` measures). This keeps the schema at ~46 properties, under OpenAI's
  100-property structured-output limit (per-measure `{adopted,quote,page}` objects would
  blow past it at ~160).
- **Tradeoff**: Tier-1 and Tier-2 are no longer independent, so the v1 Tier1↔Tier2
  agreement check is gone. `carbontax-compare` measures any per-measure adoption-rate
  drift (v2 − v1) on the same chunk set as the attention-degradation check.

See [`legacy/README.md`](legacy/README.md) for the full v1→v2 diff.
