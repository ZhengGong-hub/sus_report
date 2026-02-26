"""
Research pipeline: PDF → parse → aggregate → chunk → LLM.
"""
import logging
from pathlib import Path

import pandas as pd
import yaml

from utils.llm_wrapper import LLMWrapper, Provider
from utils.logger import Logger
from utils.pdf_parser import PDFParser
from utils.recursive_splitter import RecursiveTextSplitter
from utils.semantic_filter import SemanticFilter

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = Logger.get("research_run")


# -----------------------------------------------------------------------------
# Aggregate and flatten (operate on parsed block DataFrame)
# -----------------------------------------------------------------------------
def _agg_by_page(text_df: pd.DataFrame) -> pd.DataFrame:
    return text_df.groupby("page_ind", as_index=False).agg({"text": " ".join})


def _prune_pages(
    agg_df: pd.DataFrame,
    min_tokens: int = 50,
    add_token_length=None,
) -> pd.DataFrame:
    if add_token_length is None:
        parser = PDFParser(logger=logger)
        add_token_length = parser.add_token_length
    agg_df = add_token_length(agg_df)
    return agg_df[agg_df["token_length"] > min_tokens].reset_index(drop=True)


def _flatten_pages(chunks_per_page: pd.DataFrame) -> str:
    return "\n\n".join(chunks_per_page["text"].tolist())


# -----------------------------------------------------------------------------
# Chunking and LLM
# -----------------------------------------------------------------------------
def split_into_chunks(flat_text: str, max_length: int = 500, overlap: int = 100) -> list[str]:
    splitter = RecursiveTextSplitter(max_length=max_length, overlap=overlap)
    return splitter.split(flat_text)


def process_chunks_with_llm(
    chunks: list[str],
    prompt: str = "Summarize the following text and extract key points.",
    log: Logger = logger,
) -> list[str]:
    log.info("Processing %d chunks with LLM (OpenAI)...", len(chunks))
    llm = LLMWrapper(provider=Provider.OPENAI, logger=log)
    return [llm.call(prompt, c) for c in chunks]


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------
def run_research(research_config: dict) -> list[str]:
    """Run full pipeline: PDF → parse → aggregate → chunk → LLM. Returns LLM outputs."""
    logger.info("Loaded research config: %s", research_config)
    pdf_path = f"files/{research_config['fileid']}.pdf"

    parser = PDFParser(logger=logger)
    blocks = parser.parse(pdf_path)
    by_page = _prune_pages(_agg_by_page(blocks), add_token_length=parser.add_token_length)
    flat_text = _flatten_pages(by_page)

    chunks = split_into_chunks(flat_text)
    logger.info("Recursive split produced %d chunks", len(chunks))

    semantic_filter = SemanticFilter(logger=logger)
    filtered_chunks = semantic_filter.filter(chunks)
    logger.info("Semantic filter: %d chunks remaining", len(filtered_chunks))
    assert False, "Stop here"

    return process_chunks_with_llm(filtered_chunks)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "research_config" / "test.yaml"
    with open(config_path, "r") as f:
        research_config = yaml.safe_load(f)
    run_research(research_config)
