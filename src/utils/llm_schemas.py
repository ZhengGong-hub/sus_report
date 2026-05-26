"""
Shared structured-output schemas for LLM calls.
"""
from pydantic import BaseModel


class LLMCallSchema(BaseModel):
    prompt: str
    schema: dict


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

CARBON_ACTION_SCHEMA = LLMCallSchema(
    prompt=(
        "Analyze the text and determine whether the company discusses efforts to reduce carbon emissions. "
        "If yes, describe the specific measures the company is taking and the goal of the carbon emission target.\n\n"
        "Instructions:\n"
        "1. Describe the carbon reduction measures in one concise sentence.\n"
        "2. If carbon reduction measures are not mentioned, write 'none'.\n"
        "3. If a carbon emission target is not mentioned, write 'none'.\n"
        "4. When writing 'none', write exactly 'none' and nothing else.\n\n"
        "Answer using the JSON schema."
    ),
    schema={
        "type": "object",
        "properties": {
            "mentions_carbon_reduction": {
                "type": "string",
                "enum": ["yes", "no"]
            },
            "carbon_reduction_measures": {
                "type": "string",
                "description": "One concise sentence describing the company's carbon reduction measures."
            },
            "carbon_emission_target": {
                "type": "string",
                "description": "The stated carbon emission reduction goal or target."
            }
        },
        "required": [
            "mentions_carbon_reduction",
            "carbon_reduction_measures",
            "carbon_emission_target"
        ],
    },
)