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

_MEASURE_SCHEMA = {
    "type": "object",
    "properties": {
        "adopted": {"type": "boolean"},
        "quote": {"type": ["string", "null"]},
        "page": {"type": ["integer", "null"]},
    },
    "required": ["adopted", "quote", "page"],
    "additionalProperties": False,
}


def build_tier1_schema() -> dict:
    """JSON schema for Tier-1 bucket-level extraction (5 scope buckets)."""
    tier1_props = {bucket: _MEASURE_SCHEMA for bucket in TIER1}
    return {
        "type": "object",
        "properties": {
            "tier1": {
                "type": "object",
                "properties": tier1_props,
                "required": list(tier1_props.keys()),
                "additionalProperties": False,
            },
            "notes": {"type": "string"},
        },
        "required": ["tier1", "notes"],
        "additionalProperties": False,
    }


def build_tier2_schema() -> dict:
    """JSON schema for Tier-2 measure-level extraction (27 measures + 4 governance flags)."""
    tier2_props = {
        measure: _MEASURE_SCHEMA
        for measures in TIER2.values()
        for measure in measures
    }
    gov_props = {flag: _MEASURE_SCHEMA for flag in GOVERNANCE}
    return {
        "type": "object",
        "properties": {
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
            "notes": {"type": "string"},
        },
        "required": ["tier2", "governance", "notes"],
        "additionalProperties": False,
    }
