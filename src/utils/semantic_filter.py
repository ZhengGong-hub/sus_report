"""
Semantic filter: keyword prefilter → LLM classification.
Keeps only chunks that mention keywords and are about corporate carbon-reduction efforts.
"""
from typing import List, Optional

from utils.llm_wrapper import LLMWrapper, Provider


CLASSIFICATION_PROMPT = (
    "Does this text discuss corporate efforts to reduce carbon or emissions? "
    "Answer only YES or NO."
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
        classification_prompt: Optional[str] = None,
        logger=None,
        llm: Optional[LLMWrapper] = None,
    ):
        self.keywords = keywords or ["carbon", "emission"]
        self._prompt = classification_prompt or CLASSIFICATION_PROMPT
        self._log = logger
        self._llm = llm or LLMWrapper(provider=Provider.OPENAI, logger=logger)

    def _keyword_prefilter(self, chunks: List[str]) -> List[str]:
        """Keep only chunks that contain at least one keyword (case-insensitive)."""
        lower_keywords = [k.lower() for k in self.keywords]
        matched = [
            c for c in chunks
            if any(kw in c.lower() for kw in lower_keywords)
        ]
        if self._log:
            self._log.info(
                "Keyword prefilter: %d → %d chunks (keywords: %s)",
                len(chunks), len(matched), self.keywords,
            )
        return matched

    def _llm_classify(self, chunks: List[str]) -> List[str]:
        """Keep only chunks the LLM classifies as about corporate carbon-reduction efforts."""
        kept = []
        for chunk in chunks:
            reply = self._llm.call(self._prompt, chunk).strip().upper()
            if reply.startswith("YES"):
                kept.append(chunk)
        if self._log:
            self._log.info(
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
