"""
Semantic filter: keyword prefilter → LLM classification with structured output.
Keeps only chunks that mention keywords and are about corporate carbon-reduction efforts.
"""
from typing import Optional

from utils.llm_wrapper import LLMWrapper, Provider
from utils.llm_schemas import FILTER_SCHEMA

class SemanticFilter:
    """
    Pipeline: raw chunks → keyword prefilter → LLM classification.
    Discards chunks with no keyword match; sends keyword-matched chunks to LLM
    to keep only those about corporate carbon-reduction efforts.
    """

    def __init__(
        self,
        keywords: Optional[list[str]] = None,
        logger=None,
        llm: Optional[LLMWrapper] = None,
    ):
        self.keywords = keywords or ["carbon", "emission"]
        self.logger = logger
        self._llm = llm or LLMWrapper(provider=Provider.OPENAI, logger=logger)

    def _keyword_prefilter(self, chunks: list[str]) -> list[str]:
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

    def _llm_classify(self, chunks: list[str]) -> list[str]:
        """Keep only chunks the LLM classifies as about corporate carbon-reduction efforts."""
        kept: list[str] = []
        for chunk in chunks:
            result = self._llm.call_structured(
                FILTER_SCHEMA.prompt,
                chunk,
                json_schema=FILTER_SCHEMA.schema,
                schema_name="filter",
            )
            self.logger.debug("LLM classification result: %s", result)
            answer = str(result.get("answer", "")).lower()
            if answer == "yes":
                kept.append(chunk)
        if self.logger:
            self.logger.info(
                "LLM classification: %d → %d chunks (corporate carbon-reduction)",
                len(chunks), len(kept),
            )
        return kept

    def filter(self, chunks: list[str]) -> list[str]:
        """
        Run pipeline: keyword prefilter then LLM classification.
        Returns only chunks that match keywords and are deemed about corporate
        efforts to reduce carbon/emissions.
        """
        if not chunks:
            return []
        chunks_after_keyword: list[str] = self._keyword_prefilter(chunks)
        if not chunks_after_keyword:
            return []
        return self._llm_classify(chunks_after_keyword)

    def filter_async(self, chunks: list[str], cid: str, output_path: str = "batch.jsonl") -> str:
        """
        Run with batch processing.
        """
        self._llm.create_jsonl_for_batch(chunks, cid=cid, output_path=output_path)
        return "batch is created under %s" % output_path