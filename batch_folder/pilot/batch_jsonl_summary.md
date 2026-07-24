# Batch JSONL summary — `pilot`

Generated 2026-07-23 23:27 UTC · model `gpt-5.4-nano` · prompt `v2.2`

## Composition & cost

| Metric | Value |
|---|---|
| Companies | 28 |
| Filings | 144 |
| Chunks (= requests = JSONL lines) | 1,722 |
| Total input tokens | 2,298,638 |
| Avg chunk token length | 445.9 |
| Input price | $0.375 / 1M tokens |
| **Estimated input cost** | **$0.86** |
| Cost per company | $0.0308 |
| Cost per filing | $0.0060 |

## Token detail

| Metric | Value |
|---|---|
| System-prompt tokens (per request) | 889 |
| System-prompt share of input | 66.6% |
| Total chunk tokens | 767,780 |
| Chunk tokens — median · min · max | 482 · 14 · 602 |
| Chunk tokens — p90 · p95 | 600 · 600 |
| Max single-request tokens (system + largest chunk) | 1,491 |

## Spread (mean · median · min · max)

| Metric | mean · median · min · max |
|---|---|
| Chunks per filing | 12.0 · 12 · 1 · 20 |
| Chunks per company | 61.5 · 28 · 2 · 413 |
| Filings per company | 5.1 · 4 · 1 · 27 |

## Sharding

| Metric | Value |
|---|---|
| Shards written | 1 |
| Requests per shard — min · max | 1,722 · 1,722 |
| Shard size MiB — min · max | 14.8 · 14.8 |
| Shard input tokens (M) — min · max | 2.3 · 2.3 |
| Caps (requests / MiB / M-tokens per shard) | 45,000 / 180 / 15 |

## Skipped PDFs

| Category | Count |
|---|---|
| Requested filings | 167 |
| With readable PDF | 167 |
| Missing — no PDF downloaded | 0 |
| Corrupt — present but unreadable | 0 |

## Run config context

- `max_chunks_per_file`: 20
- `chunk_selection`: random (seed 42)
- `filter_keywords`: carbon, emission

---
*Estimates. Tokens counted with `cl100k_base`; the real model may tokenize differently. Input tokens
cover the system prompt (×1,722 requests) plus chunk text, and exclude the response JSON schema
and per-message envelope overhead, so true billed input is modestly higher. Cost is inputs only, at
the configured (already batch-discounted) rate.*
