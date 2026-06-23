"""
Shared structured-output schemas for LLM calls.
"""
from pydantic import BaseModel

from utils.taxonomy import TIER1, TIER2, GOVERNANCE, build_tier1_schema, build_tier2_schema


class LLMCallSchema(BaseModel):
    prompt: str
    schema: dict


# ---------------------------------------------------------------------------
# Generic utility schemas
# ---------------------------------------------------------------------------

SUMMARY_SCHEMA = LLMCallSchema(
    prompt="Summarize the following text and extract key points. Respond using the JSON schema.",
    schema={
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
        },
        "required": ["summary"],
    },
)

FILTER_SCHEMA = LLMCallSchema(
    prompt=(
        "Decide whether the text discusses specific corporate efforts to reduce "
        "carbon emissions. It can be either setting a target for emissions or "
        "describing concrete actions they are taking to reduce emissions. "
        "Answer using the JSON schema."
    ),
    schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["answer"],
    },
)


# ---------------------------------------------------------------------------
# Carbon extraction schemas (Tier-1 and Tier-2)
# ---------------------------------------------------------------------------

_INTRO = (
    "You are extracting corporate carbon-reduction measures from sustainability report text.\n"
)

_TIER1_DESCRIPTIONS = "\n".join([
    "  S1  — Scope 1: direct emissions from owned or controlled sources (combustion, industrial processes, fugitive releases).",
    "  S2  — Scope 2: indirect emissions from purchased electricity, heat, steam, or cooling.",
    "  S3U — Scope 3 upstream: indirect emissions in the upstream supply chain (Categories 1–8, e.g. purchased goods, business travel, commuting).",
    "  S3D — Scope 3 downstream: indirect emissions from use and end-of-life of sold products (Categories 9–15, e.g. use-phase emissions, downstream logistics, investments).",
    "  CDR — Carbon removal and offsets: nature-based solutions, engineered carbon dioxide removal, or voluntary offset purchases.",
])

_TIER2_LISTING = "\n".join(
    f"  {bucket}: {', '.join(measures)}" for bucket, measures in TIER2.items()
)

def _shared_rules(include_evidence: bool) -> str:
    rules = [
        "Rules:",
        "- 'adopted' = true only for concrete actions in the reporting year. Aspirational language alone ('we are committed to...') = false.",
    ]
    if include_evidence:
        rules += [
            "- 'quote' must be verbatim, ≤30 words, copied from the text. null if not adopted.",
            "- 'page' is the page number where the quote appears (use [PAGE N] markers). null if not adopted.",
        ]
    rules += [
        "- If unsure, set adopted=false. Do not infer beyond the text.",
        "",
        "Answer using the JSON schema.",
    ]
    return "\n".join(rules)


def build_carbon_tier1_schema(include_evidence: bool = True) -> LLMCallSchema:
    return LLMCallSchema(
        prompt="\n".join([
            _INTRO,
            "For each Tier-1 scope bucket below, decide whether the company describes adopting or actively pursuing any emission-reduction actions in that scope.",
            "",
            "Tier-1 buckets:",
            _TIER1_DESCRIPTIONS,
            "",
            _shared_rules(include_evidence),
        ]),
        schema=build_tier1_schema(include_evidence),
    )


def build_carbon_tier2_schema(include_evidence: bool = True) -> LLMCallSchema:
    return LLMCallSchema(
        prompt="\n".join([
            _INTRO,
            "For each Tier-2 category and governance flag below, decide whether the company describes adopting or actively pursuing that measure in the reporting year.",
            "",
            "Tier-2 categories:",
            _TIER2_LISTING,
            "",
            f"Governance flags: {', '.join(GOVERNANCE)}",
            "",
            _shared_rules(include_evidence),
        ]),
        schema=build_tier2_schema(include_evidence),
    )


# backwards-compatible constants (full evidence mode)
CARBON_TIER1_SCHEMA = build_carbon_tier1_schema()
CARBON_TIER2_SCHEMA = build_carbon_tier2_schema()
