# Carbon-Measure Extraction — v2 vs v1 (Pilot)

**To:** [Supervisor] **From:** Zheng **Re:** `combined_pilot_batch.xlsx`

## What changed

**1. One pass instead of two.** v1 asked the model two separate questions per text chunk (first "which emission scope?", then "which specific measure?"). v2 asks both in a single pass. Same task, half the queries.

**2. Refined categories.** The measure list grew 28 → 30. We dropped one rarely-used tag, split the broad "purchased goods" into *supplier engagement* vs *material substitution*, and added *packaging* and *general renewable electricity* — sharper, more economically meaningful buckets. Every quote now also records its **page number** for auditing.

## Results

**Cost fell ~68%.** v1 used ≈6,850 tokens per chunk; v2 uses ≈2,200. (Tokens are the unit we pay for, so this is a direct ~3× cost and time saving — important as we scale to thousands of reports.)

**Quality rose.** Two internal-consistency checks:

| Check | v1 | v2 |
|---|---|---|
| Self-contradictions* (avg) | 31% | **9%** |
| Vague flags** — Scope 2 | 77% | **37%** |
| Vague flags** — Scope 1 | 30% | **19%** |

\* A measure tagged without its parent scope — a logical error. \** A scope flagged but no concrete measure named.

## Why v2 is better

Reasoning about scope and measure *together* lets the model stay consistent: contradictions dropped roughly 3.5×, and far more high-level claims are now backed by a specific, quoted measure. Adoption rates for some scopes are lower (e.g. Scope 2: 32% → 15%) — this reflects fewer loose positives, not lost signal.

So: cheaper, faster, more internally coherent, and better evidenced.

*Note: chunk sets differ slightly (1,729 vs 1,653 text passages), so compare rates, not raw counts. Full detail in the workbook's Stats tabs.*
