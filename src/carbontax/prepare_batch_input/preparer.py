"""BatchInputPreparer: PDFs → filtered chunks → reference parquet → batch JSONL."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import tiktoken
from tqdm import tqdm

from carbontax.paths import batch_jsonl, batch_jsonl_summary, combined_ref, run_dir, skipped_pdfs_json
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
        self.input = data["input"]  # read lazily; only some identifiers need input files
        self.model = section["model"]
        self.min_page_tokens = section["min_page_tokens"]
        self.input_price_per_1m_usd = section["input_price_per_1m_usd"]  # for the batch cost estimate
        # how to pick chunks when a filing exceeds max_chunks_per_file; seed makes random reproducible
        self.chunk_selection = section["chunk_selection"]
        self.chunk_sample_seed = section["chunk_sample_seed"]
        if self.chunk_selection not in ("head", "random"):
            raise ValueError(f"Invalid chunk_selection: {self.chunk_selection}")
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
        missing: list[int] = []          # no PDF on disk (expected: not every filing was downloaded)
        corrupt: list[dict] = []         # PDF present but unreadable — skip so a long run isn't lost
        for fileid in tqdm(fileids, desc="Chunking filings", unit="pdf"):
            if not os.path.exists(f"{self.pdfs_dir}/{fileid}.pdf"):
                logger.warning("PDF not found for fileid=%s — skipping", fileid)
                missing.append(int(fileid))
                continue
            try:
                ref_frames.append(self._chunk_one_filing(str(fileid)))
            except Exception as err:  # corrupt/unreadable PDF (e.g. pymupdf.FileDataError)
                logger.warning("Failed to parse fileid=%s (%s: %s) — skipping",
                               fileid, type(err).__name__, err)
                corrupt.append({"fileid": int(fileid), "error": f"{type(err).__name__}: {err}"})

        os.makedirs(run_dir(self.run_name), exist_ok=True)
        self._write_skipped(len(fileids), missing, corrupt)
        if not ref_frames:
            raise ValueError(
                f"No filings could be chunked: {len(missing)} missing, {len(corrupt)} corrupt "
                f"out of {len(fileids)} requested — see {skipped_pdfs_json(self.run_name)}")

        # join company metadata (companyid, companyname, filingDate) onto every chunk
        reference_df = pd.concat(ref_frames, ignore_index=True)
        mapping = self._load_mapping(fileids=reference_df["filingId"].unique().tolist())
        reference_df = reference_df.merge(
            mapping[["companyid", "companyname", "filingDate", "filingId"]],
            on="filingId", how="left",
        )

        parquet_path = combined_ref(self.run_name)  # run folder already created above
        reference_df.to_parquet(parquet_path, index=False)
        logger.info("Wrote %d reference rows → %s", len(reference_df), parquet_path)
        return parquet_path

    def _write_skipped(self, requested: int, missing: list[int], corrupt: list[dict]) -> None:
        """Persist the missing/corrupt fileids so the summary is correct even if build_jsonl reruns."""
        payload = {"requested": requested, "missing": missing, "corrupt": corrupt}
        with open(skipped_pdfs_json(self.run_name), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info("Skipped %d missing + %d corrupt of %d requested filings",
                    len(missing), len(corrupt), requested)

    def _resolve_fileids(self) -> list[int]:
        identifier = self.section["identifier"]
        if identifier == "fileid":
            fileids = self.section["fileid"]
        elif identifier == "companyid":
            companyids = self.section.get("companyid")  # empty/omitted = every companyid in the mapping
            if companyids:
                fileids = self._load_mapping(companyids=companyids)["filingId"].tolist()
            else:
                fileids = self._load_mapping()["filingId"].tolist()
        elif identifier == "trucost_companyid":
            # every company in the trucost export; all their filings, all years
            trucost = pd.read_csv(self.input["trucost_csv"], usecols=["companyid"])
            companyids = trucost["companyid"].dropna().astype("int64").unique().tolist()
            fileids = self._load_mapping(companyids=companyids)["filingId"].tolist()
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
            if self.chunk_selection == "head":
                filtered = filtered.head(max_chunks)
            else:  # "random" — seeded so re-chunking reproduces the same draw
                filtered = filtered.sample(n=max_chunks, random_state=self.chunk_sample_seed).sort_index()
            logger.info("Capped to %d chunks (%s) for fileid=%s", max_chunks, self.chunk_selection, fileid)

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
        self._write_summary(ref_df, system_prompt)
        return out_path

    def _write_summary(self, ref_df: pd.DataFrame, system_prompt: str) -> str:
        """Human-readable markdown report on the batch just written (composition, tokens, cost)."""
        enc = tiktoken.get_encoding("cl100k_base")  # repo-standard tokenizer; model may differ slightly
        tok = ref_df["chunks"].map(lambda c: len(enc.encode(c)))
        system_tokens = len(enc.encode(system_prompt))

        n_chunks = len(ref_df)
        n_companies = int(ref_df["companyid"].dropna().nunique())
        n_filings = int(ref_df["filingId"].nunique())

        total_chunk_tokens = int(tok.sum())
        total_input_tokens = total_chunk_tokens + system_tokens * n_chunks
        avg_chunk_tokens = total_chunk_tokens / n_chunks
        system_share = system_tokens * n_chunks / total_input_tokens * 100
        max_request_tokens = system_tokens + int(tok.max())

        est_cost = total_input_tokens / 1e6 * self.input_price_per_1m_usd
        cost_per_company = est_cost / n_companies if n_companies else 0.0
        cost_per_filing = est_cost / n_filings if n_filings else 0.0

        # spreads: mean · median · min · max, formatted in one cell each
        chunks_per_filing = ref_df.groupby("filingId").size()
        chunks_per_company = ref_df.groupby("companyid").size()
        filings_per_company = ref_df.groupby("companyid")["filingId"].nunique()

        def spread(s: pd.Series) -> str:
            return f"{s.mean():.1f} · {s.median():.0f} · {s.min()} · {s.max()}"

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        keywords = ", ".join(self.filter.keywords)
        cap = self.section["max_chunks_per_file"]
        chunk_sel = self.chunk_selection
        if self.chunk_selection == "random":
            chunk_sel += f" (seed {self.chunk_sample_seed})"
        skip_section = self._skip_section()

        md = f"""# Batch JSONL summary — `{self.run_name}`

Generated {now} · model `{self.model}` · prompt `{PROMPT_VERSION}`

## Composition & cost

| Metric | Value |
|---|---|
| Companies | {n_companies:,} |
| Filings | {n_filings:,} |
| Chunks (= requests = JSONL lines) | {n_chunks:,} |
| Total input tokens | {total_input_tokens:,} |
| Avg chunk token length | {avg_chunk_tokens:,.1f} |
| Input price | ${self.input_price_per_1m_usd:.3f} / 1M tokens |
| **Estimated input cost** | **${est_cost:,.2f}** |
| Cost per company | ${cost_per_company:.4f} |
| Cost per filing | ${cost_per_filing:.4f} |

## Token detail

| Metric | Value |
|---|---|
| System-prompt tokens (per request) | {system_tokens:,} |
| System-prompt share of input | {system_share:.1f}% |
| Total chunk tokens | {total_chunk_tokens:,} |
| Chunk tokens — median · min · max | {tok.median():.0f} · {tok.min()} · {tok.max()} |
| Chunk tokens — p90 · p95 | {tok.quantile(0.90):.0f} · {tok.quantile(0.95):.0f} |
| Max single-request tokens (system + largest chunk) | {max_request_tokens:,} |

## Spread (mean · median · min · max)

| Metric | mean · median · min · max |
|---|---|
| Chunks per filing | {spread(chunks_per_filing)} |
| Chunks per company | {spread(chunks_per_company)} |
| Filings per company | {spread(filings_per_company)} |

{skip_section}
## Run config context

- `max_chunks_per_file`: {cap}
- `chunk_selection`: {chunk_sel}
- `filter_keywords`: {keywords}

---
*Estimates. Tokens counted with `cl100k_base`; the real model may tokenize differently. Input tokens
cover the system prompt (×{n_chunks:,} requests) plus chunk text, and exclude the response JSON schema
and per-message envelope overhead, so true billed input is modestly higher. Cost is inputs only, at
the configured (already batch-discounted) rate.*
"""
        out_path = batch_jsonl_summary(self.run_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(md)
        logger.info("Wrote batch summary → %s", out_path)
        return out_path

    def _skip_section(self) -> str:
        """Markdown block on PDFs skipped during chunking, read from the persisted sidecar."""
        path = skipped_pdfs_json(self.run_name)
        if not os.path.exists(path):
            return "## Skipped PDFs\n\n_No skip record found (chunking not run this session)._\n"
        with open(path, encoding="utf-8") as fh:
            skip = json.load(fh)
        requested, missing, corrupt = skip["requested"], skip["missing"], skip["corrupt"]
        readable = requested - len(missing) - len(corrupt)

        lines = [
            "## Skipped PDFs",
            "",
            "| Category | Count |",
            "|---|---|",
            f"| Requested filings | {requested:,} |",
            f"| With readable PDF | {readable:,} |",
            f"| Missing — no PDF downloaded | {len(missing):,} |",
            f"| Corrupt — present but unreadable | {len(corrupt):,} |",
            "",
        ]
        if corrupt:
            preview = 50  # keep the .md readable; full list lives in skipped_pdfs.json
            lines.append("Corrupt fileids jumped during chunking:")
            lines.append("")
            lines += [f"- `{c['fileid']}` — {c['error']}" for c in corrupt[:preview]]
            if len(corrupt) > preview:
                lines.append(f"- … +{len(corrupt) - preview:,} more (full list in `skipped_pdfs.json`)")
            lines.append("")
        return "\n".join(lines)

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
