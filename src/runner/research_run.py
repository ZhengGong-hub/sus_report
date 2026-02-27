"""
Research pipeline: PDF → parse → aggregate → chunk → LLM.
"""

from pathlib import Path
from pydantic import BaseModel
import os
import pandas as pd
import yaml

from utils.llm_wrapper import LLMWrapper, Provider
from utils.logger import Logger
from utils.pdf_parser import PDFParser
from utils.recursive_splitter import RecursiveTextSplitter
from utils.semantic_filter import SemanticFilter

logger = Logger.get("research_run")


def run_research(research_config: dict) -> list[str]:
    """Simplified pipeline: PDF → parse → aggregate → chunk → filter → LLM."""
    logger.info("Loaded research config: %s", research_config)
    pdf_path = f"files/{research_config['fileid']}.pdf"

    # Parse PDF and aggregate text by page
    parser = PDFParser(logger=logger)
    blocks = parser.parse(pdf_path)
    agg_df = blocks.groupby("page_ind", as_index=False).agg({"text": " ".join})

    # Add token length to each page, filter out short ones
    agg_df = parser.add_token_length(agg_df)
    agg_df = agg_df[agg_df["token_length"] > 50].reset_index(drop=True)

    # Combine all page texts into a single string
    flat_text = "\n\n".join(agg_df["text"].tolist())

    # Split text into overlapping chunks
    splitter = RecursiveTextSplitter(max_length=500, overlap=100, logger=logger)
    chunks = splitter.split(flat_text)
    logger.info("Recursive split produced %d chunks", len(chunks))

    # Filter semantically relevant chunks
    filtered_chunks = SemanticFilter(logger=logger).filter(chunks)
    logger.info("Semantic filter: %d chunks remaining", len(filtered_chunks))

    return filtered_chunks


if __name__ == "__main__":
    config_path = "src/research_config/test.yaml"
    with open(config_path, "r") as f:
        research_config = yaml.safe_load(f)
    run_research(research_config)
