"""
Semantic filter: keyword prefilter → LLM classification with structured output.
Keeps only chunks that mention keywords and are about corporate carbon-reduction efforts.
"""
from typing import List, Optional
from pydantic import BaseModel

from utils.llm_wrapper import LLMWrapper, Provider

class LLMCallSchema(BaseModel):
    prompt: str
    schema: dict

# Structured summary schema for call_structured
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
    prompt="Decide whether the text discusses specific corporate efforts to reduce carbon emissions. It can be either set a target of their carbon emission or what actions they are taking to reduce their carbon emission. Answer using the JSON schema.",
    schema={
        "type": "object",
        "properties": {
            "answer": {"type": "string", "enum": ["yes", "no"]},
        },
        "required": ["answer"],
    },
)


class SemanticFilter:
    """
    Pipeline: raw chunks → keyword prefilter → LLM classification.
    Discards chunks with no keyword match; sends keyword-matched chunks to LLM
    to keep only those about corporate carbon-reduction efforts.
    """

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        logger=None,
        llm: Optional[LLMWrapper] = None,
    ):
        self.keywords = keywords or ["carbon", "emission"]
        self.logger = logger
        self._llm = llm or LLMWrapper(provider=Provider.OPENAI, logger=logger)

    def _keyword_prefilter(self, chunks: List[str]) -> List[str]:
        """Keep only chunks that contain at least one keyword (case-insensitive)."""
        lower_keywords = [k.lower() for k in self.keywords]
        matched = [
            c for c in chunks
            if any(kw in c.lower() for kw in lower_keywords)
        ]
        if self.logger:
            self.logger.info(
                "Keyword prefilter: %d → %d chunks (keywords: %s)",
                len(chunks), len(matched), self.keywords,
            )
        return matched

    def _llm_classify(self, chunks: List[str]) -> List[str]:
        """Keep only chunks the LLM classifies as about corporate carbon-reduction efforts."""
        kept: List[str] = []
        for chunk in chunks:
            result = self._llm.call_structured(
                FILTER_SCHEMA.prompt,
                chunk,
                json_schema=FILTER_SCHEMA.schema,
                schema_name="filter",
            )
            answer = str(result.get("answer", "")).lower()
            if answer == "yes":
                kept.append(chunk)
        if self.logger:
            self.logger.info(
                "LLM classification: %d → %d chunks (corporate carbon-reduction)",
                len(chunks), len(kept),
            )
        return kept

    def filter(self, chunks: List[str]) -> List[str]:
        """
        Run pipeline: keyword prefilter then LLM classification.
        Returns only chunks that match keywords and are deemed about corporate
        efforts to reduce carbon/emissions.
        """
        if not chunks:
            return []
        after_keyword = self._keyword_prefilter(chunks)
        if not after_keyword:
            return []
        return self._llm_classify(after_keyword)
