"""
Generic, taxonomy-agnostic LLM call schemas.

Carbon-extraction schemas live in `carbontax.taxonomy`; these are the
general-purpose helpers reused across the pipeline (e.g. the chunk filter).
"""
from pydantic import BaseModel


class LLMCallSchema(BaseModel):
    prompt: str
    schema: dict  # noqa: A003 — intentional; shadows BaseModel.schema() (cosmetic warning)


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
