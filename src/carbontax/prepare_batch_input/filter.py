"""
Semantic filter: keyword filter on chunk DataFrames.

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

logger = logging.getLogger(__name__)


class SemanticFilter:
    """Keep only chunks that contain at least one keyword (case-insensitive)."""

    def __init__(self, keywords: Optional[list[str]] = None):
        self.keywords = keywords or ["carbon", "emission"]

    def filter(self, chunks_df: pd.DataFrame) -> pd.DataFrame:
        """`chunks_df` must have columns `chunk_id` and `chunk`."""
        if chunks_df is None or chunks_df.empty:
            return chunks_df

        pattern = "|".join(re.escape(k) for k in self.keywords)
        mask = chunks_df["chunk"].str.contains(pattern, case=False, na=False)
        filtered = chunks_df[mask].reset_index(drop=True)

        logger.info(
            "Keyword filter: %d → %d chunks (keywords: %s)",
            len(chunks_df), len(filtered), self.keywords,
        )
        return filtered
