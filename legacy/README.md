# legacy/ — archived v1 extraction approach

This directory is a **frozen archive** of the first-generation ("v1") extraction
pipeline. It is kept for reference and reproducibility only. Code here predates
the package refactor and its imports reference the old `utils.*` / `src.*`
layout — it is **not expected to run as-is** against the current tree.

The live pipeline is the `carbontax` package under `src/`.

## What changed v1 → v2

| Aspect            | v1 (here)                                   | v2 (`carbontax`)                          |
|-------------------|---------------------------------------------|-------------------------------------------|
| Calls per chunk   | **Two** (separate Tier-1 and Tier-2)        | **One** combined call                     |
| Schema            | `{adopted, quote, page}` per measure (~160 props) | flat booleans + one `evidence[]` array (46 props) |
| Taxonomy          | 27 measures incl. `247_cfe`, `c1_purchased_goods` | 30 measures: dropped `247_cfe`, split c1, added `renewable_electricity_general` + `packaging` |
| Output            | Excel with Tier1↔Tier2 coverage/accuracy sheets | flat CSV (tier1↔tier2 consistency check intentionally dropped) |
| Prompt            | two prompts, ≤30-word quotes                | one scope-ordered prompt, ≤15-word quotes |

## File map

| File                    | Was                              | Role |
|-------------------------|----------------------------------|------|
| `taxonomy.py`           | `src/utils/taxonomy.py`          | v1 taxonomy (TIER1/TIER2/GOVERNANCE), `PROMPT_VERSION="v1"` |
| `carbon_schemas.py`     | `src/utils/llm_schemas.py`       | v1 two-call carbon prompts + schemas (also held the generic schemas now in `carbontax.schemas`) |
| `research_question.py`  | `src/utils/research_question.py` | binds a schema to chunks → JSONL (superseded by `extraction.build_batch`) |
| `research_run.py`       | `src/runner/research_run.py`     | v1 orchestrator; its PDF→chunk half was rescued into `carbontax.chunking.pipeline` |
| `sample_companies.py`   | `src/runner/sample_companies.py` | stratified company sampler → config |
| `excel.py`              | `src/reporting/excel.py`         | v1 Excel reporter (coverage + T1-accuracy sheets) |
| `run_batch.py`          | root `run_batch.py`              | v1 two-call submit (tier1 + tier2) |
| `parse_export.py`       | root `batch_status_check.py`     | v1 fetch + flatten + Excel export |
| `config/*.yaml`         | `src/research_config/*.yaml`     | v1 run configs |

The v1 **results** themselves live in `output/clean_output_pilot_batch_tier{1,2}.csv`
and are still consumed by `carbontax.analysis.compare` for the v1-vs-v2 delta.
