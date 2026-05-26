"""
Research pipeline: PDF → parse → aggregate → chunk → LLM.
"""

import os
from pathlib import Path
import json

import pandas as pd
import yaml

from utils.logger import Logger
from utils.pdf_parser import PDFParser
from utils.recursive_splitter import RecursiveTextSplitter
from utils.research_question import ResearchQuestion
from utils.semantic_filter import SemanticFilter
from utils.llm_schemas import FILTER_SCHEMA, CARBON_ACTION_SCHEMA

logger = Logger.get("research_run")


def _build_batch_entries_for_file(fileid: str) -> tuple[list[dict], pd.DataFrame]:
    """
    Run the full pipeline for a single file id and return JSONL-ready
    request objects for OpenAI batch.
    """
    logger.info("Processing fileid=%s", fileid)
    pdf_path = f"files/{fileid}.pdf"

    # Parse PDF and aggregate text by page
    parser = PDFParser(logger=logger)
    blocks = parser.parse(pdf_path)
    agg_df = blocks.groupby("page_ind", as_index=False).agg({"text": " ".join})

    # Add token length to each page, filter out short ones
    agg_df = parser.add_token_length(agg_df)
    agg_df = agg_df[agg_df["token_length"] > 50].reset_index(drop=True)

    # Combine all page texts into a single string
    flat_text = "\n\n".join(agg_df["text"].tolist())

    # Split text into overlapping chunks (returned as DataFrame with chunk_id, chunk)
    splitter = RecursiveTextSplitter(max_length=500, overlap=100, logger=logger)
    chunks_df = splitter.split(flat_text, chunk_id_prefix=f"{fileid}")
    logger.info("Recursive split produced %d chunks", len(chunks_df))

    # Filter semantically relevant chunks
    filtered_chunks = SemanticFilter(logger=logger).filter(
        chunks_df, use_llm_classification=False
    )
    logger.info("Semantic filter: %d chunks remaining", len(filtered_chunks))

    # Create batch entries for the configured research question (using FILTER_SCHEMA)
    rq = ResearchQuestion(schema=CARBON_ACTION_SCHEMA, logger=logger)
    entries = rq.create_batch_jsonl(filtered_chunks)
    logger.info("Created %d batch entries for fileid=%s", len(entries), fileid)

    reference_df = pd.DataFrame({
        "filingId": fileid,
        "chunks": filtered_chunks["chunk"].tolist(),
        "chunk_ids": filtered_chunks["chunk_id"].tolist()
    })
    reference_df["filingId"] = reference_df["filingId"].astype(int)
    return [entries, reference_df]


def run_research(research_config: dict) -> str:
    """Simplified pipeline: apply research question over many fileids and write batch JSONL."""

    logger.info("Loaded research config: %s", research_config)

    if research_config["identifier"] == "fileid":
        fileids = research_config["fileid"]
    elif research_config["identifier"] == "companyid":
        companyids = research_config["companyid"]
        esgfiling_mapping = _load_esgfiling_mapping(companyids)
        fileids = esgfiling_mapping["filingId"].tolist()
    else:
        raise ValueError(f"Invalid identifier: {research_config['identifier']}")

    all_entries: list[dict] = []
    reference_df = pd.DataFrame()
    for fileid in fileids:
        entries, _reference_df = _build_batch_entries_for_file(str(fileid))
        all_entries.extend(entries)
        reference_df = pd.concat([reference_df, _reference_df])
    
    esgfiling_mapping = _load_esgfiling_mapping(fileids=reference_df["filingId"].unique().tolist())
    reference_df = pd.merge(reference_df, esgfiling_mapping[["companyid", "companyname", "filingDate", "filingId"]], on="filingId", how="left")

    output_path = research_config["output_folder"]
    os.makedirs(output_path, exist_ok=True)
    with open(f"{output_path}/{research_config['jsonl_path']}.jsonl", "w", encoding="utf-8") as f:
        for obj in all_entries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    logger.info("Wrote %d total batch entries to %s", len(all_entries), f"{output_path}/{research_config['jsonl_path']}.jsonl")

    reference_df.to_parquet(f"{output_path}/{research_config['ref_data_path']}.parquet")
    logger.info("Wrote %d total reference data to %s", len(reference_df), f"{output_path}/{research_config['ref_data_path']}.parquet")
    return output_path

def _load_esgfiling_mapping(companyids: list[int] = None, fileids: list[int] = None) -> pd.DataFrame:
    esgfiling_mapping = pd.read_csv("mapping_data/company_esgfiling_mapping.csv")
    if companyids is not None:
        esgfiling_mapping = esgfiling_mapping[esgfiling_mapping["companyid"].isin(companyids)]
    if fileids is not None:
        esgfiling_mapping = esgfiling_mapping[esgfiling_mapping["filingId"].isin(fileids)]
    return esgfiling_mapping

if __name__ == "__main__":
    config_path = "src/research_config/test.yaml"
    with open(config_path, "r") as f:
        research_config = yaml.safe_load(f)
    run_research(research_config)
