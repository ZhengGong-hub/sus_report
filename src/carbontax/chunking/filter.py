"""
Semantic filter: keyword prefilter → optional LLM classification.

Works on a DataFrame of chunks produced by `RecursiveTextSplitter`, with
columns:
  - chunk_id: unique identifier per chunk
  - chunk:    text content
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from carbontax.schemas import FILTER_SCHEMA
from carbontax.utils.llm import LLMWrapper, Provider


class SemanticFilter:
    """
    Pipeline on chunk DataFrames:

    raw chunks (DataFrame)
      → keyword prefilter on `chunk` column
      → optional LLM classification using FILTER_SCHEMA.
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

    def _keyword_prefilter(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """Keep only chunks that contain at least one keyword (case-insensitive)."""
        if chunks_df.empty:
            return chunks_df

        # Simple OR pattern over escaped keywords
        pattern = "|".join(self.keywords)
        mask = chunks_df["chunk"].str.contains(pattern, case=True, na=False)
        filtered = chunks_df[mask].reset_index(drop=True)

        if self.logger:
            self.logger.info(
                "Keyword prefilter: %d → %d chunks (keywords: %s)",
                len(chunks_df),
                len(filtered),
                self.keywords,
            )
        return filtered

    def _llm_classify(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only chunks the LLM classifies as about corporate carbon-reduction efforts.
        """
        if chunks_df.empty:
            return chunks_df

        keep_ids: list[str] = []
        for row in chunks_df.itertuples(index=False):
            chunk_id = row.chunk_id
            chunk_text = row.chunk

            result = self._llm.call_structured(
                FILTER_SCHEMA.prompt,
                chunk_text,
                json_schema=FILTER_SCHEMA.schema,
                schema_name="filter",
            )
            if self.logger:
                self.logger.debug("LLM classification result for %s: %s", chunk_id, result)

            answer = str(result.get("answer", "")).strip().lower()
            if answer == "yes":
                keep_ids.append(chunk_id)

        filtered = chunks_df[chunks_df["chunk_id"].isin(keep_ids)].reset_index(drop=True)

        if self.logger:
            self.logger.info(
                "LLM classification: %d → %d chunks (corporate carbon-reduction)",
                len(chunks_df),
                len(filtered),
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

        after_keyword = self._keyword_prefilter(chunks_df)
        if after_keyword.empty:
            return after_keyword

        if use_llm_classification:
            return self._llm_classify(after_keyword)

        return after_keyword