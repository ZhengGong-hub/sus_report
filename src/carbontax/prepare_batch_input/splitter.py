from __future__ import annotations

import logging

import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)


class RecursiveTextSplitter:
    def __init__(
        self,
        max_length: int = 500,
        overlap: int = 100,
        separators: list[str] | None = None,
        encoding_name: str = "cl100k_base",
    ):
        if not 0 <= overlap < max_length:
            raise ValueError(f"overlap ({overlap}) must be >= 0 and < max_length ({max_length})")
        self.max_length = max_length
        self.overlap = overlap
        self.separators = separators or ["\n\n"]
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def split(self, text: str, chunk_id_prefix: str = "chunk_") -> pd.DataFrame:
        """
        Splits text into chunks with up to max_length tokens, using the given separators in order.
        This is done iteratively (not recursively), with fallback forced splitting at the token level.
        """
        texts_to_split = [text]
        for sep in self.separators:
            next_texts = []
            for t in texts_to_split:
                # Only split if t is longer than max_length tokens
                if self.count_tokens(t) > self.max_length and sep:
                    next_texts.extend([x for x in t.split(sep) if x.strip()])
                else:
                    next_texts.append(t)
            texts_to_split = next_texts

        # We've split on all separators; now force split any remaining long chunks.
        # Overlap lives in _force_split only: separator splits are semantic
        # boundaries and get no overlap, so every chunk stays <= max_length.
        all_chunks = []
        for t in texts_to_split:
            if self.count_tokens(t) > self.max_length:
                all_chunks.extend(self._force_split(t))
            else:
                all_chunks.append(t.strip())

        result_chunks_df = pd.DataFrame(all_chunks, columns=["chunk"])
        result_chunks_df["chunk_id"] = [f"{chunk_id_prefix}_{i}" for i in range(len(result_chunks_df))]
        return result_chunks_df

    def _force_split(self, text: str) -> list[str]:
        """Split into windows of max_length tokens; consecutive windows share `overlap` tokens."""
        tokens = self.tokenizer.encode(text)
        chunks = []
        start = 0
        while start < len(tokens):
            end = start + self.max_length
            chunks.append(self.tokenizer.decode(tokens[start:end]).strip())
            if end >= len(tokens):
                break
            start = end - self.overlap
        return chunks