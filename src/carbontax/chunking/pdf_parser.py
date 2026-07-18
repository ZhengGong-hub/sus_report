"""
PDF parser: extract text blocks from PDFs, clean and prune.
Returns a block-level DataFrame (page_ind, block_ind, text, token_length).
"""
import logging
import re

import fitz
import pandas as pd
import tiktoken

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parse PDFs to block-level text with cleaning and pruning.
    """

    def __init__(
        self,
        min_block_tokens: int = 5,
        min_block_chars: int = 30,
    ):
        self.min_block_tokens = min_block_tokens
        self.min_block_chars = min_block_chars
        self._enc = tiktoken.get_encoding("cl100k_base")

    @staticmethod
    def clean_text_chunk(text: str) -> str:
        """Fix hyphenation, merge broken lines, collapse newlines, normalize spaces."""
        text = re.sub(r"-\n(\w)", r"\1", text)
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        return text.strip()

    def _prune_blocks(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """Drop footer/header repeats (via text without numbers) and too-short blocks."""
        text_df = text_df.copy()
        text_df["_no_num"] = text_df["text"].str.replace(r"\d+", "", regex=True)
        freq = text_df["_no_num"].value_counts()
        repeated = freq[freq > 2].index
        text_df = text_df[~text_df["_no_num"].isin(repeated)]
        text_df["_n"] = text_df["_no_num"].str.len()
        text_df["_t"] = text_df["text"].apply(lambda t: len(self._enc.encode(t)))
        text_df = text_df[
            (text_df["_n"] > self.min_block_chars) & (text_df["_t"] > self.min_block_tokens)
        ]
        return text_df.drop(columns=["_no_num", "_n", "_t"]).reset_index(drop=True)

    def add_token_length(self, text_df: pd.DataFrame) -> pd.DataFrame:
        """Add token_length column (e.g. for use after aggregation)."""
        out = text_df.copy()
        out["token_length"] = out["text"].apply(lambda t: len(self._enc.encode(t)))
        return out

    def parse(self, pdf_path: str) -> pd.DataFrame:
        """
        Extract blocks from PDF, clean and prune.
        Returns DataFrame with columns: page_ind, block_ind, text, token_length.
        """
        logger.info("Parsing PDF: %s", pdf_path)

        doc = fitz.open(pdf_path)
        rows = []
        for page_ind, page in enumerate(doc, start=1):
            for block_ind, block in enumerate(page.get_text("blocks"), start=1):
                raw = (block[4] or "").strip()
                if raw:
                    rows.append((page_ind, block_ind, self.clean_text_chunk(raw)))

        df = pd.DataFrame(rows, columns=["page_ind", "block_ind", "text"])
        df = self.add_token_length(self._prune_blocks(df))
        logger.info("Kept %d blocks after cleaning", len(df))
        return df
