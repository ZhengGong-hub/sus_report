"""
ResearchQuestion: binds an LLM schema to a set of chunks
and writes an OpenAI batch-consumable JSONL file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from utils.llm_schemas import LLMCallSchema
from utils.llm_wrapper import LLMWrapper, Provider


class ResearchQuestion:
    """
    A research question defined by:
      - an LLMCallSchema (prompt + JSON schema)
      - an LLMWrapper (for model / API details)

    Given a DataFrame of chunks (`chunk_id`, `chunk`), it can create a JSONL
    file suitable for OpenAI batch ingestion.
    """

    def __init__(
        self,
        schema: LLMCallSchema,
        logger=None,
        llm: Optional[LLMWrapper] = None,
    ):
        self.schema = schema
        self.logger = logger
        self._llm = llm or LLMWrapper(provider=Provider.OPENAI, logger=logger)

    def create_batch_jsonl(
        self,
        chunks_df: pd.DataFrame,
        schema_name: str = "research_question",
    ) -> str:
        """
        Write one JSON object per chunk to `output_path`, for OpenAI batch.

        - `chunks_df` must have a `chunk` column containing text.
        - `cid` is used as the `custom_id` prefix (e.g., company or filing id).
        """
        jsonl_list = []

        for row in chunks_df.itertuples(index=False):
            chunk_id = row.chunk_id
            chunk = row.chunk
            obj = self._llm.create_jsonl_for_batch(
                chunk=chunk,
                custom_id=chunk_id,
                prompt=self.schema.prompt,
                schema=self.schema.schema,
                schema_name=schema_name
            )
            jsonl_list.append(obj)
        
        return jsonl_list

