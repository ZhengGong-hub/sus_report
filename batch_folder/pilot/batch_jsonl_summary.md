# Batch JSONL summary — `pilot`

Generated 2026-07-21 19:27 UTC · model `gpt-5.4-mini` · prompt `v2`

## Composition & cost

| Metric | Value |
|---|---|
| Companies | 28 |
| Filings | 144 |
| Chunks (= requests = JSONL lines) | 1,722 |
| Total input tokens | 2,401,597 |
| Avg chunk token length | 450.7 |
| Input price | $0.375 / 1M tokens |
| **Estimated input cost** | **$0.90** |
| Cost per company | $0.0322 |
| Cost per filing | $0.0063 |

## Token detail

| Metric | Value |
|---|---|
| System-prompt tokens (per request) | 944 |
| System-prompt share of input | 67.7% |
| Total chunk tokens | 776,029 |
| Chunk tokens — median · min · max | 484 · 14 · 602 |
| Chunk tokens — p90 · p95 | 600 · 600 |
| Max single-request tokens (system + largest chunk) | 1,546 |

## Spread (mean · median · min · max)

| Metric | mean · median · min · max |
|---|---|
| Chunks per filing | 12.0 · 12 · 1 · 20 |
| Chunks per company | 61.5 · 28 · 2 · 413 |
| Filings per company | 5.1 · 4 · 1 · 27 |

## Run config context

- `max_chunks_per_file`: 20
- `filter_keywords`: carbon, emission

---
*Estimates. Tokens counted with `cl100k_base`; the real model may tokenize differently. Input tokens
cover the system prompt (×1,722 requests) plus chunk text, and exclude the response JSON schema
and per-message envelope overhead, so true billed input is modestly higher. Cost is inputs only, at
the configured (already batch-discounted) rate.*
