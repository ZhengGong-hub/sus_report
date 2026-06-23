"""
Single source of truth for the carbon-reduction measure taxonomy.
Used by the extraction prompt, JSON schema, and downstream analysis.
"""

TIER1 = ["S1", "S2", "S3U", "S3D", "CDR"]

TIER2 = {
    "S1": [
        "energy_efficiency", "fuel_switching", "onsite_renewables",
        "fgas_substitution", "methane_fugitive", "process_emissions",
    ],
    "S2": ["ppa", "rec_goo", "247_cfe", "low_carbon_heat"],
    "S3U": [
        "c1_purchased_goods", "c2_capital_goods", "c3_fuel_energy",
        "c4_upstream_transport", "c5_waste_ops", "c6_business_travel",
        "c7_commuting", "c8_upstream_leased",
    ],
    "S3D": [
        "c9_downstream_transport", "c10_processing", "c11_use_phase",
        "c12_eol", "c13_downstream_leased", "c14_franchises", "c15_investments",
    ],
    "CDR": ["nbs", "tech_cdr", "voluntary_offsets"],
}

GOVERNANCE = ["sbti", "internal_carbon_price", "exec_comp_linked", "third_party_assurance"]

PROMPT_VERSION = "v1"

def _measure_schema(include_evidence: bool) -> dict:
    if include_evidence:
        return {
            "type": "object",
            "properties": {
                "adopted": {"type": "boolean"},
                "quote": {"type": ["string", "null"]},
                "page":  {"type": ["integer", "null"]},
            },
            "required": ["adopted", "quote", "page"],
            "additionalProperties": False,
        }
    # flat boolean — no wrapper object, minimises output tokens
    return {"type": "boolean"}


def build_tier1_schema(include_evidence: bool = True) -> dict:
    """JSON schema for Tier-1 bucket-level extraction (5 scope buckets)."""
    ms = _measure_schema(include_evidence)
    tier1_props = {bucket: ms for bucket in TIER1}
    props: dict = {
        "tier1": {
            "type": "object",
            "properties": tier1_props,
            "required": list(tier1_props.keys()),
            "additionalProperties": False,
        }
    }
    return {"type": "object", "properties": props,
            "required": ["tier1"], "additionalProperties": False}


def build_tier2_schema(include_evidence: bool = True) -> dict:
    """JSON schema for Tier-2 measure-level extraction (27 measures + 4 governance flags)."""
    ms = _measure_schema(include_evidence)
    tier2_props = {m: ms for measures in TIER2.values() for m in measures}
    gov_props   = {flag: ms for flag in GOVERNANCE}
    props: dict = {
        "tier2": {
            "type": "object",
            "properties": tier2_props,
            "required": list(tier2_props.keys()),
            "additionalProperties": False,
        },
        "governance": {
            "type": "object",
            "properties": gov_props,
            "required": list(gov_props.keys()),
            "additionalProperties": False,
        },
    }
    return {"type": "object", "properties": props,
            "required": ["tier2", "governance"], "additionalProperties": False}
