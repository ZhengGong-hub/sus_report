"""
Semantic filter: keyword prefilter → optional LLM classification.

Works on a DataFrame of chunks produced by `RecursiveTextSplitter`, with
columns:
  - chunk_id: unique identifier per chunk
  - chunk:    text content
"""
from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from carbontax.schemas import FILTER_SCHEMA
from carbontax.utils.llm import LLMWrapper, Provider

logger = logging.getLogger(__name__)


class SemanticFilter:
    """
    Pipeline on chunk DataFrames:

    raw chunks (DataFrame)
      → keyword prefilter on `chunk` column
      → optional LLM classification using FILTER_SCHEMA.

    The LLM client is created lazily, so keyword-only filtering needs no API key.
    """

    def __init__(
        self,
        keywords: Optional[list[str]] = None,
        llm: Optional[LLMWrapper] = None,
    ):
        self.keywords = keywords or ["carbon", "emission"]
        self._llm = llm

    def _keyword_prefilter(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Keep only chunks that contain at least one keyword (case-insensitive)."""
        if chunks_df.empty:
            return chunks_df

        pattern = "|".join(re.escape(k) for k in self.keywords)
        mask = chunks_df["chunk"].str.contains(pattern, case=False, na=False)
        filtered = chunks_df[mask].reset_index(drop=True)

        logger.info(
            "Keyword prefilter: %d → %d chunks (keywords: %s)",
            len(chunks_df), len(filtered), self.keywords,
        )
        return filtered

    def _llm_classify(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only chunks the LLM classifies as about corporate carbon-reduction efforts.
        """
        if chunks_df.empty:
            return chunks_df

        if self._llm is None:
            self._llm = LLMWrapper(provider=Provider.OPENAI)

        keep_ids: list[str] = []
        for row in chunks_df.itertuples(index=False):
            result = self._llm.call_structured(
                FILTER_SCHEMA.prompt,
                row.chunk,
                json_schema=FILTER_SCHEMA.schema,
                schema_name="filter",
            )
            logger.debug("LLM classification result for %s: %s", row.chunk_id, result)

            answer = str(result.get("answer", "")).strip().lower()
            if answer == "yes":
                keep_ids.append(row.chunk_id)

        filtered = chunks_df[chunks_df["chunk_id"].isin(keep_ids)].reset_index(drop=True)

        logger.info(
            "LLM classification: %d → %d chunks (corporate carbon-reduction)",
            len(chunks_df), len(filtered),
        )
        return filtered

    def filter(
        self,
        chunks_df: pd.DataFrame,
        use_llm_classification: bool = False,
    ) -> pd.DataFrame:
        """
        Run pipeline: keyword prefilter then optional LLM classification.

        - `chunks_df` must have columns `chunk_id` and `chunk`.
        - If `use_llm_classification` is False, only the keyword prefilter is applied.
        """
        if chunks_df is None or chunks_df.empty:
            return chunks_df

        filtered = self._keyword_prefilter(chunks_df)
        if use_llm_classification and not filtered.empty:
            filtered = self._llm_classify(filtered)
        return filtered
