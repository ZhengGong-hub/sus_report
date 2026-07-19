"""BatchInputPreparer: PDFs → filtered chunks → reference parquet → batch JSONL."""

from __future__ import annotations

import json
import logging
import os

import pandas as pd

from carbontax.paths import batch_jsonl, combined_ref, run_dir
from carbontax.prepare_batch_input.filter import SemanticFilter
from carbontax.prepare_batch_input.pdf_parser import PDFParser
from carbontax.prepare_batch_input.splitter import RecursiveTextSplitter
from carbontax.taxonomy import PROMPT_VERSION, build_combined_schema, build_combined_system_prompt

logger = logging.getLogger(__name__)


class BatchInputPreparer:

    def __init__(self, run_name: str, section: dict, data: dict):
        self.run_name = run_name
        self.section = section
        # all knobs come explicitly from config/run.yaml — a missing key fails loudly
        # model is stamped into the ref parquet AND into every batch request, so both agree
        self.pdfs_dir = data["output"]["pdfs_dir"]
        self.mapping_csv = data["output"]["mapping_csv"]
        self.model = section["model"]
        self.min_page_tokens = section["min_page_tokens"]
        self.parser = PDFParser()
        self.splitter = RecursiveTextSplitter(
            max_length=section["chunk_max_tokens"],
            overlap=section["chunk_overlap_tokens"],
        )
        self.filter = SemanticFilter(keywords=section["filter_keywords"])

    def run(self) -> None:
        self.chunk_filings()
        self.build_jsonl()

    # ── step 1: PDFs → reference parquet ──────────────────────────────────────

    def chunk_filings(self) -> str:
        """Chunk every configured filing; write the reference parquet to the run folder."""
        fileids = self._resolve_fileids()

        ref_frames: list[pd.DataFrame] = []
        for fileid in fileids:
            if not os.path.exists(f"{self.pdfs_dir}/{fileid}.pdf"):
                logger.warning("PDF not found for fileid=%s — skipping", fileid)
                continue
            ref_frames.append(self._chunk_one_filing(str(fileid)))

        if not ref_frames:
            raise RuntimeError(f"No filings produced chunks — check {self.pdfs_dir}/ and the config.")

        # join company metadata (companyid, companyname, filingDate) onto every chunk
        reference_df = pd.concat(ref_frames, ignore_index=True)
        mapping = self._load_mapping(fileids=reference_df["filingId"].unique().tolist())
        reference_df = reference_df.merge(
            mapping[["companyid", "companyname", "filingDate", "filingId"]],
            on="filingId", how="left",
        )

        os.makedirs(run_dir(self.run_name), exist_ok=True)
        parquet_path = combined_ref(self.run_name)
        reference_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %d reference rows → %s", len(reference_df), parquet_path)
        return parquet_path

    def _resolve_fileids(self) -> list[int]:
        identifier = self.section["identifier"]
        if identifier == "fileid":
            fileids = self.section["fileid"]
        elif identifier == "companyid":
            fileids = self._load_mapping(companyids=self.section["companyid"])["filingId"].tolist()
        else:
            raise ValueError(f"Invalid identifier: {identifier}")

        # Drop filings that map to >1 companyid in the mapping CSV: the metadata
        # re-join in chunk_filings would fan every chunk out into one row per
        # company, producing duplicate chunk_ids (custom_ids) in the batch.
        # Tiny sample, so we exclude them rather than disambiguate.
        counts = self._load_mapping(fileids=fileids).groupby("filingId")["companyid"].nunique()
        ambiguous = set(counts[counts > 1].index)
        if ambiguous:
            logger.warning("Dropping %d filing(s) mapped to multiple companyids: %s",
                           len(ambiguous), sorted(ambiguous))
            fileids = [f for f in fileids if f not in ambiguous]
        return fileids

    def _chunk_one_filing(self, fileid: str) -> pd.DataFrame:
        logger.info("Processing fileid=%s", fileid)

        # PDF → page-level text, dropping headers/footers and near-empty pages
        blocks = self.parser.parse(f"{self.pdfs_dir}/{fileid}.pdf")
        agg_df = blocks.groupby("page_ind", as_index=False).agg({"text": " ".join})
        agg_df = self.parser.add_token_length(agg_df)
        agg_df = agg_df[agg_df["token_length"] > self.min_page_tokens].reset_index(drop=True)

        # one string with [PAGE N] markers → token-window chunks with overlap
        flat_text = "\n\n".join(
            f"[PAGE {row.page_ind}]\n{row.text}" for row in agg_df.itertuples(index=False)
        )
        chunks_df = self.splitter.split(flat_text, chunk_id_prefix=fileid)
        logger.info("Recursive split produced %d chunks", len(chunks_df))

        # keep only carbon/emission-relevant chunks
        filtered = self.filter.filter(chunks_df, use_llm_classification=False)
        logger.info("Semantic filter: %d chunks remaining", len(filtered))

        max_chunks = self.section["max_chunks_per_file"]  # null in YAML = no cap
        if max_chunks is not None and len(filtered) > max_chunks:
            filtered = filtered.head(max_chunks)
            logger.info("Capped to %d chunks for fileid=%s", max_chunks, fileid)

        return pd.DataFrame({
            "filingId": int(fileid),
            "chunks": filtered["chunk"].tolist(),
            "chunk_ids": filtered["chunk_id"].tolist(),
            "prompt_version": PROMPT_VERSION,
            "model": self.model,
        })

    def _load_mapping(self, companyids: list[int] = None, fileids: list[int] = None) -> pd.DataFrame:
        df = pd.read_csv(self.mapping_csv)
        if companyids is not None:
            df = df[df["companyid"].isin(companyids)]
        if fileids is not None:
            df = df[df["filingId"].isin(fileids)]
        return df

    # ── step 2: reference parquet → batch JSONL ───────────────────────────────

    def build_jsonl(self) -> str:
        """One combined-schema request per chunk, written to the run folder."""
        ref_path = combined_ref(self.run_name)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference parquet not found: {ref_path} — run chunk_filings first.")

        ref_df = pd.read_parquet(ref_path)
        logger.info("Loaded %d chunks from %s", len(ref_df), ref_path)

        schema = build_combined_schema()
        system_prompt = build_combined_system_prompt()

        out_path = batch_jsonl(self.run_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            for _, row in ref_df.iterrows():
                request = self._build_request(row["chunk_ids"], row["chunks"], schema, system_prompt)
                fh.write(json.dumps(request, ensure_ascii=False) + "\n")

        logger.info("Wrote %d requests → %s", len(ref_df), out_path)
        return out_path

    def _build_request(self, chunk_id: str, chunk_text: str, schema: dict, system_prompt: str) -> dict:
        return {
            "custom_id": chunk_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": chunk_text},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": schema,
                },
            },
        }
