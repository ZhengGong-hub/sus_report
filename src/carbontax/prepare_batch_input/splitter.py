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

        # We've split on all separators; now force split any remaining long chunks
        all_chunks = []
        for t in texts_to_split:
            if self.count_tokens(t) > self.max_length:
                all_chunks.extend(self._force_split(t))
            else:
                all_chunks.append(t.strip())

        # Apply overlap if necessary
        result_chunks = self._apply_overlap(all_chunks)

        # Logging check for '\n\n'
        total_newline_double = sum(chunk.count("\n\n") for chunk in result_chunks)
        logger.info("Total '\\n\\n' occurrences in all chunks: %d", total_newline_double)

        result_chunks_df = pd.DataFrame(result_chunks, columns=["chunk"])
        result_chunks_df["chunk_id"] = [f"{chunk_id_prefix}_{i}" for i in range(len(result_chunks_df))]
        return result_chunks_df

    def _force_split(self, text: str):
        """
        Forcefully splits text into chunks of max_length tokens with specified overlap.
        """
        chunks = []
        tokens = self.tokenizer.encode(text)
        start = 0
        while start < len(tokens):
            end = start + self.max_length
            chunk_tokens = tokens[start:end]
            chunk = self.tokenizer.decode(chunk_tokens).strip()
            chunks.append(chunk)
            start = end - self.overlap
            if start < 0:
                start = 0
                if end >= len(tokens):
                    break
        return chunks

    def _apply_overlap(self, chunks):
        if self.overlap <= 0 or len(chunks) <= 1:
            return chunks

        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
                continue

            prev_chunk = overlapped_chunks[-1]
            prev_tokens = self.tokenizer.encode(prev_chunk)
            curr_tokens = self.tokenizer.encode(chunk)

            overlap_tokens = prev_tokens[-self.overlap:] if len(prev_tokens) >= self.overlap else prev_tokens
            new_chunk_tokens = overlap_tokens + curr_tokens
            # Prevent duplication if overlap already includes the start of curr_tokens
            if len(overlap_tokens) >= len(curr_tokens):
                new_chunk_tokens = curr_tokens
            overlapped_chunk = self.tokenizer.decode(new_chunk_tokens)
            overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks