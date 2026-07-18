"""
chunking/pipeline.py — PDF filings → filtered text chunks → reference parquet.

This is the version-agnostic front half of the extraction pipeline (formerly
the chunking part of the v1 `research_run.py`). It produces the reference
parquet that `carbontax.extraction.build_batch` consumes; it does NOT build any
LLM request itself, so it is shared by every prompt/schema version.

Per filing:
  1. Parse PDF → block-level text (PDFParser), drop headers/footers.
  2. Aggregate to page level, drop pages with ≤ 50 tokens.
  3. Join pages into one string with [PAGE N] markers, then chunk
     (RecursiveTextSplitter: 500-token windows, 100-token overlap).
  4. Keyword-filter to carbon/emission-relevant chunks (SemanticFilter).

Output (written to `output_folder`):
  <ref_data_path>.parquet — columns:
    filingId, chunks, chunk_ids, prompt_version, model,
    companyid, companyname, filingDate

Config (YAML / dict):
  identifier:          "fileid" | "companyid"
  fileid:              [215978062, ...]   # when identifier == "fileid"
  companyid:           [32307, ...]       # when identifier == "companyid"
  output_folder:       "to_batch_pilot"
  ref_data_path:       "pilot_batch_ref"  # stem; .parquet appended
  max_chunks_per_file: 20                 # optional; null/omit for no cap

Run from the repo root (relative paths: files/, mapping_data/):
  python -m carbontax.chunking.pipeline [config.yaml]
  carbontax-chunk [config.yaml]          # console entry point
"""

from __future__ import annotations

import logging
import os
import sys

import pandas as pd
import yaml

from carbontax.chunking.filter import SemanticFilter
from carbontax.chunking.pdf_parser import PDFParser
from carbontax.chunking.splitter import RecursiveTextSplitter
from carbontax.taxonomy import PROMPT_VERSION
from carbontax.utils.llm import DEFAULT_MODELS, Provider
from carbontax.utils.logger import setup_logging

logger = logging.getLogger(__name__)

MAPPING_CSV = "mapping_data/company_esgfiling_mapping.csv"
DEFAULT_CONFIG = "config/pilot.yaml"


def chunk_filing(fileid: str, max_chunks: int | None = None) -> pd.DataFrame:
    """
    Run the per-filing chunking pipeline and return a reference DataFrame.

    Columns: filingId, chunks, chunk_ids, prompt_version, model.
    One row per filtered chunk; company metadata is joined later by
    `build_reference`.
    """
    logger.info("Processing fileid=%s", fileid)

    parser = PDFParser()
    blocks = parser.parse(f"files/{fileid}.pdf")
    agg_df = blocks.groupby("page_ind", as_index=False).agg({"text": " ".join})
    agg_df = parser.add_token_length(agg_df)
    agg_df = agg_df[agg_df["token_length"] > 50].reset_index(drop=True)

    flat_text = "\n\n".join(
        f"[PAGE {row.page_ind}]\n{row.text}" for row in agg_df.itertuples(index=False)
    )
    chunks_df = RecursiveTextSplitter(max_length=500, overlap=100).split(
        flat_text, chunk_id_prefix=fileid
    )
    logger.info("Recursive split produced %d chunks", len(chunks_df))

    filtered = SemanticFilter().filter(chunks_df, use_llm_classification=False)
    logger.info("Semantic filter: %d chunks remaining", len(filtered))

    if max_chunks is not None and len(filtered) > max_chunks:
        filtered = filtered.head(max_chunks)
        logger.info("Capped to %d chunks for fileid=%s", max_chunks, fileid)

    return pd.DataFrame({
        "filingId": int(fileid),
        "chunks": filtered["chunk"].tolist(),
        "chunk_ids": filtered["chunk_id"].tolist(),
        "prompt_version": PROMPT_VERSION,
        "model": DEFAULT_MODELS[Provider.OPENAI],
    })


def _load_mapping(companyids: list[int] = None, fileids: list[int] = None) -> pd.DataFrame:
    """Load the company↔filing mapping CSV, optionally filtered."""
    df = pd.read_csv(MAPPING_CSV)
    if companyids is not None:
        df = df[df["companyid"].isin(companyids)]
    if fileids is not None:
        df = df[df["filingId"].isin(fileids)]
    return df


def build_reference(config: dict) -> str:
    """
    Build the reference parquet for a set of filings.

    Returns the path to the written parquet file.
    """
    logger.info("Loaded chunking config: %s", config)

    identifier = config["identifier"]
    if identifier == "fileid":
        fileids = config["fileid"]
    elif identifier == "companyid":
        fileids = _load_mapping(companyids=config["companyid"])["filingId"].tolist()
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    # Drop filings that map to >1 companyid in the mapping CSV: the metadata
    # re-join below would fan every chunk out into one row per company,
    # producing duplicate chunk_ids (custom_ids) in the batch. Tiny sample, so
    # we exclude them rather than disambiguate.
    counts = _load_mapping(fileids=fileids).groupby("filingId")["companyid"].nunique()
    ambiguous = set(counts[counts > 1].index)
    if ambiguous:
        logger.warning("Dropping %d filing(s) mapped to multiple companyids: %s",
                       len(ambiguous), sorted(ambiguous))
        fileids = [f for f in fileids if f not in ambiguous]

    ref_frames: list[pd.DataFrame] = []
    for fileid in fileids:
        if not os.path.exists(f"files/{fileid}.pdf"):
            logger.warning("PDF not found for fileid=%s — skipping", fileid)
            continue
        ref_frames.append(chunk_filing(str(fileid), max_chunks=config.get("max_chunks_per_file")))

    if not ref_frames:
        raise RuntimeError("No filings produced chunks — check files/ and the config.")

    reference_df = pd.concat(ref_frames, ignore_index=True)
    mapping = _load_mapping(fileids=reference_df["filingId"].unique().tolist())
    reference_df = reference_df.merge(
        mapping[["companyid", "companyname", "filingDate", "filingId"]],
        on="filingId", how="left",
    )

    output_path = config["output_folder"]
    os.makedirs(output_path, exist_ok=True)
    parquet_path = f"{output_path}/{config['ref_data_path']}.parquet"
    reference_df.to_parquet(parquet_path, index=False)
    logger.info("Wrote %d reference rows → %s", len(reference_df), parquet_path)
    return parquet_path


def main() -> None:
    setup_logging()
    config_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG
    with open(config_path) as f:
        build_reference(yaml.safe_load(f))


if __name__ == "__main__":
    main()
