# System-prompt clarity improvements — 2026-07-23

Clarity review of the `v2.1` extraction prompt (`build_combined_system_prompt` in
`taxonomy.py`), read from the built `test_trucost/batch.jsonl`. The structure is good
(scope-ordered, glosses only where needed, rules stated once), but five ambiguities
could cause systematic extraction errors. These are **candidate edits, not yet
applied** — if adopted they should ship as `PROMPT_VERSION = v2.2` and be validated
with the planned v2.1-vs-v2.2 A/B before a full run. Ranked by likely impact.

## 1. Action-vs-mention ambiguity (highest impact) — ✅ IMPLEMENTED in v2.2 (2026-07-23)

Applied: the Tier-2 header now reads *"Mark a measure true only when the chunk
describes an action that reduces this source — not a mere mention that the source
exists."* The duplicate "concrete/aspiration" line was removed from the Tier-2 header
so that rule now lives once (in Decision rules). Net +11 tokens (768 → 779).

**Issue.** The task is to detect emission-reduction *actions*, but ~18 fields are bare
category names that read like *types of emissions* — `process_emissions`,
`methane_fugitive`, `c4_upstream_transport`, `c11_use_phase`, `c2_capital_goods`, etc.
A model can mark `process_emissions = true` because a chunk *mentions* process
emissions, not because the firm *reduced* them.

**Why it matters.** Highest false-positive risk. The "concrete actions, not
aspirations" rule is stated three times, but the field naming works against it.

**Proposed fix.** One line near the Tier-2 header:
> "Mark true only when the chunk describes an action reducing this source — not a mere
> mention that the source exists."

## 2. Governance section has no instruction — ✅ IMPLEMENTED in v2.2 (2026-07-23)

Applied (kept the section, did not cut it — it is a full downstream dimension: schema
`governance` object + `parse_output` `governance_*` columns + analysis report). Added a
trigger header ("Mark true if the chunk shows the firm has this mechanism in place.
Unlike Tier-2 measures, a stated commitment or target counts (e.g. an SBTi target =
true) — the aspiration rule below applies to measures, not to governance.") plus a
dedicated `GOVERNANCE_GLOSS` table (own guard) for the four flags. The Decision-rules
aspiration line was scoped to "Tier-1 buckets & Tier-2 measures" so it no longer reads
as gating governance. `GOVERNANCE_FLAGS` and the schema are unchanged, so downstream is
unaffected.

**Issue.** Tier-1 and Tier-2 each get a "mark true if…" header, but `## Governance
flags` is just four bare ids — `sbti`, `internal_carbon_price`, `exec_comp_linked`,
`third_party_assurance` — with no trigger rule and no glosses. The model must infer
both meaning and trigger condition. `exec_comp_linked` is especially ambiguous (linked
to climate targets? any ESG metric?).

**Why it matters.** Least-specified part of the prompt; governance flags are likely the
noisiest outputs.

**Proposed fix.** Add a header line ("Mark true if the chunk states the firm has this
governance mechanism in place.") plus short glosses, e.g.:
- `sbti`: has an SBTi-validated or -committed target
- `internal_carbon_price`: applies an internal price/shadow cost on carbon
- `exec_comp_linked`: executive pay tied to climate/emissions targets
- `third_party_assurance`: emissions data externally assured/verified

## 3. Tier-1 ↔ Tier-2 relationship never stated — ✅ IMPLEMENTED in v2.2 (2026-07-23)

Applied: the Tier-1 header now reads *"Set a bucket true if any of its Tier-2 measures
below is true, or if the chunk describes a scope-level emission-reduction action not
itemised as a measure."* (replaced the old "mark true if … ANY … action in that scope"
line, so no overlapping instruction).

**Issue.** Nothing says whether a Tier-1 bucket is a rollup of its Tier-2 measures or an
independent judgment. If `energy_efficiency = true`, must `S1 = true`?

**Why it matters.** Produces inconsistent rows (bucket false while a measure under it is
true), which corrupts scope-level aggregates.

**Proposed fix.** One sentence:
> "Set a Tier-1 bucket true if any of its Tier-2 measures is true, or if the chunk
> describes a scope-level action not itemised below."

## 4. `evidence[]` scope is ambiguous — ✅ RESOLVED by removing evidence (v2.2, 2026-07-23)

Decision: rather than clarify the scope, **cut the `evidence` array entirely** — the
output is flags only. Rationale: a flag with an ad-hoc quote/page is a half-designed
feature; if traceability is needed later it deserves a real design, not a bolted-on
array. Removed from all three stages: schema (`evidence` dropped from properties +
`required`), `parse_output` (`_parse_evidence` and all `*_quote`/`*_page` columns gone),
and the analysis workbook (`_write_data_sheet` no longer renders/collapses evidence
columns; unused `Outline` import dropped). Bonus: the prompt lost its 4 evidence lines
(902 → 848 tokens), and each response no longer emits quotes — a real **output**-token
saving too. The source chunk text (`chunks` column, Raw sheet) still allows manual
verification.

**Original issue (for the record).** "one entry per measure you mark true" doesn't say whether "measure" includes
the Tier-1 buckets and governance flags, or only Tier-2 measures. If `sbti = true`, is
an evidence quote expected?

**Why it matters.** Inconsistent evidence coverage; complicates downstream evidence
joins.

**Proposed fix.** State the scope explicitly, e.g. "evidence[]: one entry per **Tier-2
measure and governance flag** you mark true (not Tier-1 buckets)." (Exact scope is a
design choice — pick and state it.)

## 5. Stale "24/7 CFE" reference — ✅ IMPLEMENTED in v2.2 (2026-07-23)

Applied: dropped the "or 24/7 CFE contract" clause; the gloss now reads "renewable
electricity purchased with no named instrument (not a PPA or REC/GO)". The exclusion
list now names only fields that exist, and a CFE chunk falls naturally into
`renewable_electricity_general`.

**Issue.** The `renewable_electricity_general` gloss says "not a PPA, REC/GO, or 24/7
CFE contract", but there is no CFE field (dropped in v2 for 0% adoption). A chunk about
a 24/7 CFE contract is excluded from `renewable_electricity_general` with nowhere to
go.

**Why it matters.** Minor, but a dangling instruction referencing a non-existent field.

**Proposed fix.** Either drop the "or 24/7 CFE contract" clause, or fold CFE explicitly
into `renewable_electricity_general` (treat it as general renewable purchase).

## Minor

Cryptic bare ids — `nbs`, `tech_cdr`, `c12_eol` — were terse. ✅ Glossed in v2.2
(2026-07-23): `nbs` = nature-based carbon removal (afforestation, reforestation, soil
carbon); `tech_cdr` = engineered carbon removal (direct air capture, BECCS, biochar);
`c12_eol` = end-of-life treatment of sold products (disposal, recycling, incineration).
The `nbs`/`tech_cdr` examples also sharpen the CDR split.

## Status

- **#1 — implemented in `v2.2`** (2026-07-23).
- **#2 — implemented in `v2.2`** (2026-07-23); kept the section, added trigger + glosses.
- **#3 — implemented in `v2.2`** (2026-07-23); Tier-1 rollup rule.
- **#4 — resolved in `v2.2`** (2026-07-23) by **removing** the evidence array (flags-only
  output), across schema + parse_output + analysis.
- **#5 — implemented in `v2.2`** (2026-07-23); dropped stale CFE reference.

All five addressed (plus the minor CDR/S3D glosses). Prompt is now 889 tokens (v2.1 was
768). The output schema changed
shape (no `evidence`), so v2.2 is a harder break from v2.1 than a prose edit — confirm
with the A/B before the full run, and note parsed CSVs lose the `*_quote`/`*_page` columns.
