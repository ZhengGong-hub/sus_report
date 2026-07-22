"""
taxonomy.py — single source of truth for the carbon-reduction taxonomy (v2).

All measure IDs, Tier-1 bucket definitions, selective glosses, JSON schemas,
and the system prompt are derived from this file. Never hardcode the measure
list elsewhere — drift between prompt and schema is the primary failure mode.
"""

from __future__ import annotations

PROMPT_VERSION = "v2.2"  # v2.2: action-vs-mention rule at Tier-2; governance trigger + glosses; Tier-1 rollup rule; evidence array removed (flags only); dropped stale CFE reference; taxonomy unchanged from v2

# ── Tier-1 scope buckets ──────────────────────────────────────────────────────

TIER1_BUCKETS: dict[str, str] = {
    "S1": (
        "Scope 1 — direct emissions from owned or controlled sources "
        "(combustion, industrial processes, fugitive releases). "
        "On-site energy generation (solar, wind, CHP) belongs here, not S2."
    ),
    "S2": (
        "Scope 2 — indirect emissions from *purchased* electricity, heat, steam, "
        "or cooling only. Energy efficiency and on-site generation are S1, not S2. "
        "A utility selling renewable power = its product (S3D), not the utility's own S2."
    ),
    "S3U": (
        "Scope 3 upstream — indirect emissions in the supply chain upstream of the firm "
        "(GHG Protocol Categories 1–8; itemised as c1–c8 below)."
    ),
    "S3D": (
        "Scope 3 downstream — indirect emissions from the use and end-of-life of products "
        "sold by the firm (Categories 9–15; itemised as c9–c15 below)."
    ),
    "CDR": (
        "Carbon removal and offsets — nature-based solutions, engineered carbon dioxide "
        "removal, or voluntary offset purchases."
    ),
}

# ── Tier-2 measures (scope-ordered, v2 taxonomy) ─────────────────────────────
#
# Changes from v1:
#   DROP  247_cfe            (0.0% adoption in pilot)
#   SPLIT c1_purchased_goods → c1_supplier_engagement + c1_material_substitution
#   ADD   renewable_electricity_general  (addresses 77% S2 coverage gap)
#   ADD   packaging          (recurring S3D→S3U leak; scope provisional S3U)

MEASURES: list[tuple[str, str]] = [
    # Scope 1
    ("energy_efficiency",             "S1"),
    ("fuel_switching",                "S1"),
    ("onsite_renewables",             "S1"),
    ("fgas_substitution",             "S1"),
    ("methane_fugitive",              "S1"),
    ("process_emissions",             "S1"),
    # Scope 2
    ("ppa",                           "S2"),
    ("rec_goo",                       "S2"),
    ("renewable_electricity_general", "S2"),
    ("low_carbon_heat",               "S2"),
    # Scope 3 upstream
    ("c1_supplier_engagement",        "S3U"),
    ("c1_material_substitution",      "S3U"),
    ("c2_capital_goods",              "S3U"),
    ("c3_fuel_energy",                "S3U"),
    ("c4_upstream_transport",         "S3U"),
    ("c5_waste_ops",                  "S3U"),
    ("c6_business_travel",            "S3U"),
    ("c7_commuting",                  "S3U"),
    ("c8_upstream_leased",            "S3U"),
    ("packaging",                     "S3U"),
    # Scope 3 downstream
    ("c9_downstream_transport",       "S3D"),
    ("c10_processing",                "S3D"),
    ("c11_use_phase",                 "S3D"),
    ("c12_eol",                       "S3D"),
    ("c13_downstream_leased",         "S3D"),
    ("c14_franchises",                "S3D"),
    ("c15_investments",               "S3D"),
    # Carbon removal / offsets
    ("nbs",                           "CDR"),
    ("tech_cdr",                      "CDR"),
    ("voluntary_offsets",             "CDR"),
]

MEASURE_IDS: list[str] = [m[0] for m in MEASURES]
MEASURE_SCOPE: dict[str, str] = {m[0]: m[1] for m in MEASURES}

# ── Governance flags ──────────────────────────────────────────────────────────

GOVERNANCE_FLAGS: list[str] = [
    "sbti",
    "internal_carbon_price",
    "exec_comp_linked",
    "third_party_assurance",
]

# ── Selective glosses ─────────────────────────────────────────────────────────
# Annotate ONLY where the plain field name is insufficient.
# Most names are self-documenting — over-glossing wastes tokens and flattens signal.

GLOSS: dict[str, str] = {
    "ppa":                           "renewable electricity via a named power purchase agreement",
    "rec_goo":                       "renewable electricity via a named REC or Guarantee of Origin certificate",
    "c1_supplier_engagement":        "engaging, auditing, or requiring suppliers to reduce emissions",
    "c1_material_substitution":      "switching to lower-carbon input materials or ingredients (independent of c1_supplier_engagement — both may be true)",
    "renewable_electricity_general": "renewable electricity purchased with no named instrument (not a PPA or REC/GO)",
    "onsite_renewables":             "firm-OWNED on-site generation (rooftop solar, wind, CHP) — Scope 1, not Scope 2",
    "c3_fuel_energy":                "upstream extraction, production, and transport of fuels/energy purchased by the firm but not resold",
    "c10_processing":                "downstream processor's emissions from further processing of the firm's sold intermediate products",
    "c12_eol":                       "end-of-life treatment of the firm's sold products (disposal, recycling, incineration)",
    "c13_downstream_leased":         "emissions from assets leased OUT by the firm to lessees",
    "packaging":                     "reducing packaging weight or switching to lower-carbon packaging materials",
    "nbs":                           "nature-based carbon removal (afforestation, reforestation, soil carbon)",
    "tech_cdr":                      "engineered carbon removal (direct air capture, BECCS, biochar)",
}

# Guard: every GLOSS key must be a valid measure id
assert all(k in MEASURE_IDS for k in GLOSS), (
    f"GLOSS keys not in MEASURE_IDS: {set(GLOSS) - set(MEASURE_IDS)}"
)

# Governance flags need their own framing (existence of the mechanism, not a reduction
# action), so they get a small dedicated gloss table rather than sharing GLOSS.
GOVERNANCE_GLOSS: dict[str, str] = {
    "sbti":                  "an SBTi-validated or -committed emissions target",
    "internal_carbon_price": "applies an internal carbon price or shadow cost",
    "exec_comp_linked":      "executive pay tied to climate/emissions targets",
    "third_party_assurance": "emissions data externally assured or verified",
}

# Guard: every GOVERNANCE_GLOSS key must be a valid governance flag
assert all(k in GOVERNANCE_FLAGS for k in GOVERNANCE_GLOSS), (
    f"GOVERNANCE_GLOSS keys not in GOVERNANCE_FLAGS: {set(GOVERNANCE_GLOSS) - set(GOVERNANCE_FLAGS)}"
)


# ── Schema builders ───────────────────────────────────────────────────────────

def _bool_prop() -> dict:
    return {"type": "boolean"}


def build_combined_schema() -> dict:
    """
    Combined v2 schema: flat booleans for all measures + governance flags.

    Property count across all objects:
      root(3) + tier1(5) + tier2(30) + governance(4) = 42
    Well under OpenAI's 100-property hard limit.
    Nesting depth ≤ 3 (under the 5-level limit).
    """
    tier1_props    = {bucket: _bool_prop() for bucket in TIER1_BUCKETS}
    tier2_props    = {mid: _bool_prop() for mid in MEASURE_IDS}
    gov_props      = {flag: _bool_prop() for flag in GOVERNANCE_FLAGS}

    return {
        "name": "carbon_extraction",
        "schema": {
            "type": "object",
            "properties": {
                "tier1": {
                    "type": "object",
                    "properties": tier1_props,
                    "required": list(tier1_props),
                    "additionalProperties": False,
                },
                "tier2": {
                    "type": "object",
                    "properties": tier2_props,
                    "required": list(tier2_props),
                    "additionalProperties": False,
                },
                "governance": {
                    "type": "object",
                    "properties": gov_props,
                    "required": list(gov_props),
                    "additionalProperties": False,
                },
            },
            "required": ["tier1", "tier2", "governance"],
            "additionalProperties": False,
        },
        "strict": True,
    }


# ── System prompt builder ─────────────────────────────────────────────────────

def _measures_by_scope() -> dict[str, list[str]]:
    by_scope: dict[str, list[str]] = {s: [] for s in TIER1_BUCKETS}
    for mid, scope in MEASURES:
        by_scope[scope].append(mid)
    return by_scope


def build_combined_system_prompt() -> str:
    """
    Single combined extraction prompt (v2).
    Scope-ordered, selective glosses carry the disambiguation, rules stated once.
    Target: < 900 words so chunk text gets full model attention.
    """
    by_scope = _measures_by_scope()

    lines: list[str] = [
        "You are extracting corporate carbon-reduction measures from sustainability report text.",
        "",
        "## Tier-1 scope buckets",
        "Set a bucket true if any of its Tier-2 measures below is true, or if the chunk "
        "describes a scope-level emission-reduction action not itemised as a measure.",
        "",
    ]
    for bucket, defn in TIER1_BUCKETS.items():
        lines.append(f"  {bucket} — {defn}")

    lines += [
        "",
        "## Tier-2 measures (by scope)",
        "Mark a measure true only when the chunk describes an action that reduces this "
        "source — not a mere mention that the source exists.",
        "",
    ]
    for scope, mids in by_scope.items():
        if not mids:
            continue
        lines.append(f"**{scope}**")
        for mid in mids:
            gloss = GLOSS.get(mid)
            if gloss:
                lines.append(f"  {mid}: {gloss}")
            else:
                lines.append(f"  {mid}")
        lines.append("")

    lines += [
        "## Governance flags",
        "Mark true if the chunk shows the firm has this mechanism in place. Unlike Tier-2 "
        "measures, a stated commitment or target counts (e.g. an SBTi target = true) — the "
        "aspiration rule below applies to measures, not to governance.",
        "",
    ]
    for flag in GOVERNANCE_FLAGS:
        gloss = GOVERNANCE_GLOSS.get(flag)
        lines.append(f"  {flag}: {gloss}" if gloss else f"  {flag}")

    lines += [
        "",
        "## Decision rules",
        "- Tier-1 buckets & Tier-2 measures: true only for a concrete action in the reporting year.",
        "  Pure aspiration ('committed to...', 'target to...') = false. (Governance flags: see above.)",
        "- If unsure → false. Do not infer beyond the text.",
    ]

    return "\n".join(lines)
