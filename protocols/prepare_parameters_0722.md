# Prepare-stage parameter deliberation — 2026-07-22

Parsed 60 PDFs and swept every knob. Here's the complete picture — what each parameter
does, its measured cost effect, and its quality/recall effect — so you can dial with
eyes open. All figures are sample-based (60 filings, seed 42) then scaled to the full
run; the method validates well (replicated chunks@50 = 1,011 vs the parquet's actual
1,051 for the same files, ~96%).

## First, the cost model

Input cost is driven by **number of requests** and **tokens per request**:

`input$ ≈ (768 prompt + ~avg chunk tokens) × n_requests × $0.375/M`

Two things follow: (1) the 768-token prompt is paid on **every** request, so anything
that cuts request count cuts the biggest line item; (2) **outputs are extra and also
scale with n_requests** — so cutting requests helps twice (outputs not yet priced).

## The five knobs, measured

### 1. `filter_keywords` — the recall gate (most important for quality)

This runs **before** the LLM. Right now `[carbon, emission]` keeps only **34% of
chunks** — the other 66% never reach the model. What expanding it recovers (sample):

| keyword set | chunks kept | vs base |
|---|---|---|
| base `[carbon, emission]` | 1,071 | 100% |
| + ghg/greenhouse | 1,094 | +2% |
| **+ energy (renewable/solar/wind/efficiency)** | 1,229 | **+15%** |
| + targets (net zero/sbti/science-based) | 1,101 | +3% |
| + gases (co2/methane/n2o/hfc/f-gas) | 1,082 | +1% |
| + scope 1/2/3 | 1,082 | +1% |
| expanded (all) | 1,278 | +19% |

The critical finding: ghg/gases/scope add almost nothing (they co-occur with
"emission" already, and "carbon" substring-matches "decarbonization", "low-carbon",
etc.). But **energy vocabulary adds +15%** — chunks like *"installed 40 MW of rooftop
solar"* that never say carbon/emission. Since a big share of the Tier-2 measures are
energy actions (`onsite_renewables`, `ppa`, `rec_goo`, `renewable_electricity_general`,
`energy_efficiency`, `low_carbon_heat`), the current gate is most likely silently
dropping the very S1/S2 measures you want. This is the one knob where the failure mode
is **invisible recall loss the prompt can't recover** — widen it (energy terms at
minimum) and spend the ~+15% cost.

### 2. `min_page_tokens` — drop thin pages

Page-token distribution: p25 = 174, p50 = 328, p75 = 513. The dial:

| min_page_tokens | vs current | est. full chunks | ~input $ |
|---|---|---|---|
| 50 (baseline) | 100% | 216,832 | $104 |
| 100 | 96% | 208,000 | $100 |
| 150 | 89% | 193,000 | $93 |
| 200 | 82% | 179,000 | $86 |
| 300 | 66% | 144,000 | $69 |

Fairly gentle up to ~150 and mostly safe on quality — thin pages are usually dividers,
title pages, image-only pages. The risk is small dense **data tables** (low token
count, high value); at 300 you're dropping a third of content, which starts to bite.

### 3. `chunk_max_tokens` — window ceiling

Effective size = setting + overlap, so 800 → chunks up to ~900 (matches the batch's
p95 899). Effect on real data:

| chunk_max | chunks kept |
|---|---|
| 500 | 1,255 |
| 800 | 1,071 (−15% vs 500) |
| 1200 | 1,018 (−5% vs 800) |

500 → 800 was a real ~15% cut; above 800 it flattens (most chunks are set by natural
page boundaries, not the ceiling). **800 is a good stopping point** — going higher
buys little and dilutes per-chunk focus.

### 4. `chunk_overlap_tokens` — boundary duplication

Surprisingly expensive, because overlap both enlarges chunks and creates more
force-split windows:

| overlap | chunks | total chunk tokens | vs 100 |
|---|---|---|---|
| 100 | 1,071 | 553,983 | 100% |
| 50 | 1,028 | 481,101 | 87% |
| 0 | 926 | 396,265 | 72% |

Dropping 100 → 50 saves ~13% of chunk tokens and ~4% of requests at low risk; 100 → 0
saves ~28% tokens + 14% requests but risks splitting a measure statement across a
boundary. **50 is a sensible middle** for boolean extraction.

### 5. `max_chunks_per_file` + `chunk_selection` — tail cap

Chunks/filing is long-tailed (median 11, p99 100, max 259). With random sampling
already wired in:

| cap | chunks kept | ~input $ | filings affected |
|---|---|---|---|
| 30 | 75% | $79 | 2,168 |
| 50 | 89% | $93 | 930 |
| 100 | 99% | $103 | 114 |

A cap of 30–50 bounds runaway reports representatively; the risk is a rare measure
appearing only in a dropped chunk of a long report.

## How to think about it (leverage vs risk)

- **Highest quality leverage:** `filter_keywords`. It's the only gate that causes
  silent recall loss, and energy-measure coverage is probably the weak spot. This is
  the one to get right — even at higher cost.
- **Safest cost cuts:** `chunk_overlap_tokens` 100→50, `min_page_tokens` 50→100/150.
  Low quality risk, real savings.
- **Moderate cost cut:** `chunk_max_tokens` 800 (already done); don't chase higher.
- **Tunable cost dial with a coverage trade:** `max_chunks_per_file` cap — use only if
  you need to bound spend.

## Three concrete starting points

Effects roughly compound on request count (they act on different dimensions), so these
are ballpark:

- **Recall-first (spend for quality):** keywords += energy terms, `min_page_tokens` 50,
  overlap 100, no cap → +15% cost (~$120 input) but materially better measure coverage.
  My lean, given the goal is measure extraction.
- **Balanced:** keywords += energy, `min_page_tokens` 100, overlap 50, cap 50 → recall
  up on energy, cost roughly flat vs today because the savings offset the keyword
  expansion.
- **Cost-min:** keep base keywords, `min_page_tokens` 150, overlap 50, cap 30 →
  ~$60–65 input, but you keep the energy-recall blind spot.

## The meta-point

There genuinely is no right answer — it's a recall/cost frontier. The two moves that
dominate everything else: **(a)** widen `filter_keywords` to close the energy-measure
gap (quality you can't buy back later), and **(b)** cut overlap/`min_page_tokens` to
pay for it. And since each full run is ~$100+, do parameter iteration on a small
sampled config and validate with the prompt A/B before the one full production run.

## The test that would settle the keyword decision

Measure the energy-keyword recall gain at the **measure level** — sample some of the
chunks that the energy keywords newly catch and check whether they really contain
extractable measures (confirming the +15% is signal, not noise). That's the test that
settles whether to widen the gate.
