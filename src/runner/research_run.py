"""
Research pipeline: ESG PDF filings → OpenAI batch JSONL + reference parquet.

High-level flow
---------------
For each filing (identified by fileid or companyid):

  1. PDF parsing   – extract text blocks page-by-page, drop headers/footers
                     and pages with < 50 tokens (PDFParser).
  2. Chunking      – split the full document into overlapping token windows
                     (max 500 tokens, 100-token overlap) so each LLM call
                     fits within the context limit (RecursiveTextSplitter).
  3. Filtering     – keep only chunks that mention carbon / emissions keywords;
                     optionally re-score with an LLM classifier (SemanticFilter).
  4. Batch prep    – format each surviving chunk as two sets of OpenAI batch
                     requests: one using CARBON_TIER1_SCHEMA (scope-bucket level)
                     and one using CARBON_TIER2_SCHEMA (measure level).

Outputs (written to `output_folder` from the config):
  - <jsonl_path>_tier1.jsonl  – Tier-1 batch requests, one line per chunk.
  - <jsonl_path>_tier2.jsonl  – Tier-2 batch requests, one line per chunk.
  - <ref_data_path>.parquet   – chunk text + ids joined with company metadata
                                (companyid, companyname, filingDate, filingId).

Config shape (YAML / dict)
--------------------------
  identifier:    "fileid" | "companyid"
  fileid:        [215978062, ...]   # used when identifier == "fileid"
  companyid:     [32307, ...]       # used when identifier == "companyid"
  output_folder:       "to_batch"
  jsonl_path:          "test_batch"  # stem; _tier1.jsonl / _tier2.jsonl appended
  ref_data_path:       "test_batch_ref"  # stem; .parquet is appended
  max_chunks_per_file: 50            # optional; omit or set null for no cap
"""

import os
import json

import pandas as pd
import yaml

from utils.llm_schemas import CARBON_TIER1_SCHEMA, CARBON_TIER2_SCHEMA
from utils.llm_wrapper import DEFAULT_MODELS, Provider
from utils.logger import Logger
from utils.pdf_parser import PDFParser
from utils.recursive_splitter import RecursiveTextSplitter
from utils.research_question import ResearchQuestion
from utils.semantic_filter import SemanticFilter
from utils.taxonomy import PROMPT_VERSION

logger = Logger.get("research_run")


def _build_batch_entries_for_file(
    fileid: str,
    max_chunks: int | None = None,
) -> tuple[list[dict], list[dict], pd.DataFrame]:
    """
    Run the full per-filing pipeline and return Tier-1 entries, Tier-2 entries,
    and a reference table.

    Parameters
    ----------
    fileid : str
        Numeric filing ID (used to locate `files/<fileid>.pdf`).

    Returns
    -------
    tier1_entries : list[dict]
        OpenAI batch request objects using the Tier-1 scope-bucket schema.
    tier2_entries : list[dict]
        OpenAI batch request objects using the Tier-2 measure-level schema.
    reference_df : pd.DataFrame
        Columns: filingId, chunks, chunk_ids, prompt_version, model.
        One row per filtered chunk; used downstream to join company metadata.

    Pipeline steps
    --------------
    1. Parse PDF → block-level DataFrame (page_ind, block_ind, text, token_length).
    2. Aggregate blocks by page, drop pages with ≤ 50 tokens (likely empty/noise).
    3. Join all page text into one string (with [PAGE N] markers), then chunk.
    4. Keyword-filter chunks to those mentioning carbon/emission topics.
    5. Format each surviving chunk as batch entries for both tier schemas.
    """
    logger.info("Processing fileid=%s", fileid)

    # --- Step 1-2: parse PDF, aggregate to page level, drop sparse pages ---
    parser = PDFParser(logger=logger)
    blocks = parser.parse(f"files/{fileid}.pdf")
    agg_df = blocks.groupby("page_ind", as_index=False).agg({"text": " ".join})
    agg_df = parser.add_token_length(agg_df)
    agg_df = agg_df[agg_df["token_length"] > 50].reset_index(drop=True)

    # --- Step 3: flatten to single string with page markers, then chunk ---
    flat_text = "\n\n".join(
        f"[PAGE {row.page_ind}]\n{row.text}" for row in agg_df.itertuples(index=False)
    )
    chunks_df = RecursiveTextSplitter(max_length=500, overlap=100, logger=logger).split(
        flat_text, chunk_id_prefix=fileid
    )
    logger.info("Recursive split produced %d chunks", len(chunks_df))

    # --- Step 4: keep only chunks relevant to carbon/emissions ---
    filtered_chunks = SemanticFilter(logger=logger).filter(chunks_df, use_llm_classification=False)
    logger.info("Semantic filter: %d chunks remaining", len(filtered_chunks))

    if max_chunks is not None and len(filtered_chunks) > max_chunks:
        filtered_chunks = filtered_chunks.head(max_chunks)
        logger.info("Capped to %d chunks for fileid=%s", max_chunks, fileid)

    # --- Step 5: format for OpenAI batch (both tiers) ---
    tier1_entries = ResearchQuestion(schema=CARBON_TIER1_SCHEMA, logger=logger).create_batch_jsonl(filtered_chunks)
    tier2_entries = ResearchQuestion(schema=CARBON_TIER2_SCHEMA, logger=logger).create_batch_jsonl(filtered_chunks)
    logger.info("Created %d tier1 / %d tier2 batch entries for fileid=%s", len(tier1_entries), len(tier2_entries), fileid)

    reference_df = pd.DataFrame({
        "filingId": int(fileid),
        "chunks": filtered_chunks["chunk"].tolist(),
        "chunk_ids": filtered_chunks["chunk_id"].tolist(),
        "prompt_version": PROMPT_VERSION,
        "model": DEFAULT_MODELS[Provider.OPENAI],
    })
    return tier1_entries, tier2_entries, reference_df


def run_research(research_config: dict) -> str:
    """
    Orchestrate the research pipeline over a list of filings and write outputs.

    Accepts either a direct list of filing IDs or a list of company IDs (which
    are resolved to filing IDs via the ESG filing mapping CSV).

    Parameters
    ----------
    research_config : dict
        Must contain the keys described in this module's docstring.

    Returns
    -------
    str
        Path to the output folder where JSONL and parquet files were written.
    """
    logger.info("Loaded research config: %s", research_config)

    # --- Resolve filing IDs from whichever identifier type was provided ---
    identifier = research_config["identifier"]
    if identifier == "fileid":
        fileids = research_config["fileid"]
    elif identifier == "companyid":
        fileids = _load_esgfiling_mapping(companyids=research_config["companyid"])["filingId"].tolist()
    else:
        raise ValueError(f"Invalid identifier: {identifier}")

    # --- Process each filing; accumulate batch entries and reference rows ---
    all_tier1: list[dict] = []
    all_tier2: list[dict] = []
    ref_frames: list[pd.DataFrame] = []
    for fileid in fileids:
        if not os.path.exists(f"files/{fileid}.pdf"):
            logger.warning("PDF not found for fileid=%s — skipping", fileid)
            continue
        t1, t2, ref = _build_batch_entries_for_file(str(fileid), max_chunks=research_config.get("max_chunks_per_file"))
        all_tier1.extend(t1)
        all_tier2.extend(t2)
        ref_frames.append(ref)

    # --- Enrich reference table with company metadata for downstream use ---
    reference_df = pd.concat(ref_frames, ignore_index=True)
    mapping = _load_esgfiling_mapping(fileids=reference_df["filingId"].unique().tolist())
    reference_df = reference_df.merge(
        mapping[["companyid", "companyname", "filingDate", "filingId"]], on="filingId", how="left"
    )

    # --- Write outputs ---
    output_path = research_config["output_folder"]
    os.makedirs(output_path, exist_ok=True)

    jsonl_stem = f"{output_path}/{research_config['jsonl_path']}"
    _write_jsonl(f"{jsonl_stem}_tier1.jsonl", all_tier1)
    _write_jsonl(f"{jsonl_stem}_tier2.jsonl", all_tier2)

    parquet_path = f"{output_path}/{research_config['ref_data_path']}.parquet"
    reference_df.to_parquet(parquet_path)
    logger.info("Wrote %d total reference rows to %s", len(reference_df), parquet_path)

    return output_path


def _write_jsonl(path: str, entries: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in entries:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    logger.info("Wrote %d batch entries to %s", len(entries), path)


def _load_esgfiling_mapping(companyids: list[int] = None, fileids: list[int] = None) -> pd.DataFrame:
    """
    Load `mapping_data/company_esgfiling_mapping.csv`, optionally filtered.

    Parameters
    ----------
    companyids : list[int], optional
        If provided, keep only rows whose `companyid` is in this list.
    fileids : list[int], optional
        If provided, keep only rows whose `filingId` is in this list.

    Returns
    -------
    pd.DataFrame
        Columns include at minimum: companyid, companyname, filingDate, filingId.
    """
    df = pd.read_csv("mapping_data/company_esgfiling_mapping.csv")
    if companyids is not None:
        df = df[df["companyid"].isin(companyids)]
    if fileids is not None:
        df = df[df["filingId"].isin(fileids)]
    return df


if __name__ == "__main__":
    with open("src/research_config/pilot.yaml") as f:
        run_research(yaml.safe_load(f))
