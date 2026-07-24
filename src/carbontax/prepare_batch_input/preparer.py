"""BatchInputPreparer: PDFs → filtered chunks → reference parquet → batch JSONL."""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import tiktoken
from tqdm import tqdm

from carbontax.paths import (
    batch_files_dir,
    batch_jsonl_summary,
    batch_shard_jsonl,
    combined_ref,
    run_dir,
    skipped_pdfs_json,
)
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
        # request temperature; null in config = omit it (gpt-5 reasoning models reject any non-default value)
        self.temperature = section["temperature"]
        self.chunk_workers = section["chunk_workers"]  # parallel PDF-chunking processes
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
        """Chunk every configured filing across chunk_workers processes; write the reference parquet.

        Each PDF is independent and the work is CPU-bound (pymupdf parse + tiktoken), so we fan the
        filings out over a process pool. Missing/corrupt PDFs are reported by the workers, not raised,
        so one bad file never sinks a long run.
        """
        fileids = self._resolve_fileids()

        missing: list[int] = []          # no PDF on disk (expected: not every filing was downloaded)
        corrupt: list[dict] = []         # PDF present but unreadable — recorded so a long run isn't lost
        results: list[tuple[int, pd.DataFrame]] = []
        initargs = (
            self.pdfs_dir, self.min_page_tokens, self.section["max_chunks_per_file"],
            self.chunk_selection, self.chunk_sample_seed, self.model,
            self.section["chunk_max_tokens"], self.section["chunk_overlap_tokens"],
            self.section["filter_keywords"],
        )
        with ProcessPoolExecutor(max_workers=self.chunk_workers,
                                 initializer=_init_chunk_worker, initargs=initargs) as ex:
            futures = [ex.submit(_chunk_worker, int(fid)) for fid in fileids]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Chunking filings", unit="pdf"):
                r = fut.result()  # the worker never raises — it returns missing/corrupt as data
                if r["missing"]:
                    logger.warning("PDF not found for fileid=%s — skipping", r["fileid"])
                    missing.append(r["fileid"])
                elif r["error"]:
                    logger.warning("Failed to parse fileid=%s (%s) — skipping", r["fileid"], r["error"])
                    corrupt.append({"fileid": r["fileid"], "error": r["error"]})
                else:
                    results.append((r["fileid"], r["df"]))

        os.makedirs(run_dir(self.run_name), exist_ok=True)
        self._write_skipped(len(fileids), missing, corrupt)
        if not results:
            raise ValueError(
                f"No filings could be chunked: {len(missing)} missing, {len(corrupt)} corrupt "
                f"out of {len(fileids)} requested — see {skipped_pdfs_json(self.run_name)}")

        # sort by fileid so the parquet is byte-stable regardless of worker completion order
        results.sort(key=lambda p: p[0])
        reference_df = pd.concat([df for _, df in results], ignore_index=True)
        # join company metadata (companyid, companyname, filingDate) onto every chunk
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
        # thin wrapper over the module-level impl (which the parallel workers also call)
        return _chunk_one_filing_impl(
            fileid, self.parser, self.splitter, self.filter, self.pdfs_dir, self.min_page_tokens,
            self.section["max_chunks_per_file"], self.chunk_selection, self.chunk_sample_seed, self.model,
        )

    def _load_mapping(self, companyids: list[int] = None, fileids: list[int] = None) -> pd.DataFrame:
        df = pd.read_csv(self.mapping_csv)
        if companyids is not None:
            df = df[df["companyid"].isin(companyids)]
        if fileids is not None:
            df = df[df["filingId"].isin(fileids)]
        return df

    # ── step 2: reference parquet → batch JSONL ───────────────────────────────

    def build_jsonl(self) -> str:
        """One combined-schema request per chunk, sharded under OpenAI's per-batch limits.

        OpenAI caps a batch at 50k requests, ~200MB per input file, AND an enqueued-token
        budget per model — so we roll to a fresh indexed shard whenever the next request
        would breach any configured cap. The token cap is usually the one that binds.

        Each shard is written to a .partial file and atomically renamed to batch_NNN.jsonl
        only once complete. batch_shards() globs batch_*.jsonl, so it never sees a half-written
        shard — you can run submit repeatedly while this is still generating, and each run picks
        up whatever shards have finished so far.
        """
        ref_path = combined_ref(self.run_name)
        if not os.path.exists(ref_path):
            raise FileNotFoundError(f"Reference parquet not found: {ref_path} — run chunk_filings first.")

        ref_df = pd.read_parquet(ref_path)
        logger.info("Loaded %d chunks from %s", len(ref_df), ref_path)

        schema = build_combined_schema()
        system_prompt = build_combined_system_prompt()

        # per-request input tokens = system prompt (fixed) + chunk. The chunk count is done
        # lazily inside the loop (NOT eagerly over the whole frame) so the first shard is
        # written within seconds instead of after tokenizing all rows. tiktoken/cl100k_base is
        # an estimate — the token cap keeps headroom under OpenAI's real enqueued-token limit.
        enc = tiktoken.get_encoding("cl100k_base")
        system_tokens = len(enc.encode(system_prompt))

        max_rows = self.section["max_requests_per_shard"]
        max_bytes = self.section["max_bytes_per_shard"]
        max_tokens = self.section["max_tokens_per_shard"]

        shards_dir = batch_files_dir(self.run_name)
        # start clean so a re-run never leaves stale shards alongside the new ones
        if os.path.isdir(shards_dir):
            shutil.rmtree(shards_dir)
        os.makedirs(shards_dir, exist_ok=True)

        shard_stats: list[dict] = []  # rows + bytes + tokens per shard, for the summary report
        chunk_tokens_acc: list[int] = []  # per-chunk token counts, collected for _write_summary

        def finalize(i: int, fh, rows: int, nbytes: int, ntok: int) -> None:
            # close, then atomically publish: the .jsonl name appears only when the shard is whole
            fh.close()
            os.replace(batch_shard_jsonl(self.run_name, i) + ".partial", batch_shard_jsonl(self.run_name, i))
            shard_stats.append({"rows": rows, "bytes": nbytes, "tokens": ntok})
            logger.info("Shard %d ready: %d requests · %.1f MiB · %.1fM tokens", i, rows, nbytes / 1024 / 1024, ntok / 1e6)

        idx, rows, nbytes, ntok = 0, 0, 0, 0
        fh = open(batch_shard_jsonl(self.run_name, idx) + ".partial", "w", encoding="utf-8")
        for _, row in ref_df.iterrows():
            request = self._build_request(row["chunk_ids"], row["chunks"], schema, system_prompt)
            line = json.dumps(request, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode("utf-8"))
            ctok = len(enc.encode(row["chunks"]))
            chunk_tokens_acc.append(ctok)
            rtok = system_tokens + ctok
            # roll before writing when this line would breach any cap; never split a request,
            # and rows > 0 guarantees a fresh shard always gets at least one request
            if rows > 0 and (rows + 1 > max_rows or nbytes + line_bytes > max_bytes or ntok + rtok > max_tokens):
                finalize(idx, fh, rows, nbytes, ntok)
                idx += 1
                fh = open(batch_shard_jsonl(self.run_name, idx) + ".partial", "w", encoding="utf-8")
                rows, nbytes, ntok = 0, 0, 0
            fh.write(line)
            rows += 1
            nbytes += line_bytes
            ntok += rtok
        finalize(idx, fh, rows, nbytes, ntok)

        chunk_tokens = pd.Series(chunk_tokens_acc, index=ref_df.index)  # for the summary token stats

        logger.info("Wrote %d requests across %d shards → %s", len(ref_df), len(shard_stats), shards_dir)
        self._write_summary(ref_df, chunk_tokens, system_tokens, shard_stats)
        return shards_dir

    def _write_summary(self, ref_df: pd.DataFrame, chunk_tokens: pd.Series,
                       system_tokens: int, shard_stats: list[dict]) -> str:
        """Human-readable markdown report on the batch just written (composition, tokens, cost).

        chunk_tokens / system_tokens are counted once in build_jsonl and passed in, so the
        report and the shard token accounting use exactly the same numbers.
        """
        tok = chunk_tokens  # per-chunk token counts (cl100k_base; the model may differ slightly)

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

        # sharding: OpenAI-limit-driven split of the batch into indexed files
        n_shards = len(shard_stats)
        shard_rows = [s["rows"] for s in shard_stats]
        shard_mib = [s["bytes"] / 1024 / 1024 for s in shard_stats]
        shard_mtok = [s["tokens"] / 1e6 for s in shard_stats]
        cap_rows = self.section["max_requests_per_shard"]
        cap_mib = self.section["max_bytes_per_shard"] / 1024 / 1024
        cap_mtok = self.section["max_tokens_per_shard"] / 1e6

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

## Sharding

| Metric | Value |
|---|---|
| Shards written | {n_shards} |
| Requests per shard — min · max | {min(shard_rows):,} · {max(shard_rows):,} |
| Shard size MiB — min · max | {min(shard_mib):.1f} · {max(shard_mib):.1f} |
| Shard input tokens (M) — min · max | {min(shard_mtok):.1f} · {max(shard_mtok):.1f} |
| Caps (requests / MiB / M-tokens per shard) | {cap_rows:,} / {cap_mib:.0f} / {cap_mtok:.0f} |

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
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": chunk_text},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": schema,
            },
        }
        # some models (e.g. gpt-5-mini) reject any non-default temperature — null in config omits it
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return {
            "custom_id": chunk_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }


# ── parallel chunking ─────────────────────────────────────────────────────────
# ProcessPoolExecutor uses the 'spawn' start method on macOS, so the worker and its
# initializer must be importable module-level functions, not closures over the instance.

def _chunk_one_filing_impl(fileid: str, parser: PDFParser, splitter: RecursiveTextSplitter,
                           filt: SemanticFilter, pdfs_dir: str, min_page_tokens: int,
                           max_chunks: int | None, chunk_selection: str,
                           chunk_sample_seed: int, model: str) -> pd.DataFrame:
    """One filing → its filtered, capped chunks as a reference frame (parser/splitter/filter injected)."""
    logger.info("Processing fileid=%s", fileid)

    # PDF → page-level text, dropping headers/footers and near-empty pages
    blocks = parser.parse(f"{pdfs_dir}/{fileid}.pdf")
    agg_df = blocks.groupby("page_ind", as_index=False).agg({"text": " ".join})
    agg_df = parser.add_token_length(agg_df)
    agg_df = agg_df[agg_df["token_length"] > min_page_tokens].reset_index(drop=True)

    # one string with [PAGE N] markers → token-window chunks with overlap
    flat_text = "\n\n".join(
        f"[PAGE {row.page_ind}]\n{row.text}" for row in agg_df.itertuples(index=False)
    )
    chunks_df = splitter.split(flat_text, chunk_id_prefix=fileid)
    logger.info("Recursive split produced %d chunks", len(chunks_df))

    # keep only carbon/emission-relevant chunks
    filtered = filt.filter(chunks_df, use_llm_classification=False)
    logger.info("Semantic filter: %d chunks remaining", len(filtered))

    if max_chunks is not None and len(filtered) > max_chunks:  # null in YAML = no cap
        if chunk_selection == "head":
            filtered = filtered.head(max_chunks)
        else:  # "random" — seeded so re-chunking reproduces the same draw
            filtered = filtered.sample(n=max_chunks, random_state=chunk_sample_seed).sort_index()
        logger.info("Capped to %d chunks (%s) for fileid=%s", max_chunks, chunk_selection, fileid)

    return pd.DataFrame({
        "filingId": int(fileid),
        "chunks": filtered["chunk"].tolist(),
        "chunk_ids": filtered["chunk_id"].tolist(),
        "prompt_version": PROMPT_VERSION,
        "model": model,
    })


_WORKER: dict = {}  # per-process chunking context, populated once per worker by the initializer


def _init_chunk_worker(pdfs_dir, min_page_tokens, max_chunks, chunk_selection, chunk_sample_seed,
                       model, chunk_max_tokens, chunk_overlap_tokens, filter_keywords) -> None:
    # build one parser/splitter/filter per process — they hold tiktoken encoders that don't
    # pickle cleanly, so we construct them here rather than ship them from the parent
    _WORKER.update(
        parser=PDFParser(),
        splitter=RecursiveTextSplitter(max_length=chunk_max_tokens, overlap=chunk_overlap_tokens),
        filter=SemanticFilter(keywords=filter_keywords),
        pdfs_dir=pdfs_dir,
        min_page_tokens=min_page_tokens,
        max_chunks=max_chunks,
        chunk_selection=chunk_selection,
        chunk_sample_seed=chunk_sample_seed,
        model=model,
    )


def _chunk_worker(fileid: int) -> dict:
    """Chunk one filing in a worker process; report missing/corrupt PDFs as data, never raise."""
    if not os.path.exists(f"{_WORKER['pdfs_dir']}/{fileid}.pdf"):
        return {"fileid": int(fileid), "df": None, "missing": True, "error": None}
    try:
        df = _chunk_one_filing_impl(
            str(fileid), _WORKER["parser"], _WORKER["splitter"], _WORKER["filter"],
            _WORKER["pdfs_dir"], _WORKER["min_page_tokens"], _WORKER["max_chunks"],
            _WORKER["chunk_selection"], _WORKER["chunk_sample_seed"], _WORKER["model"],
        )
        return {"fileid": int(fileid), "df": df, "missing": False, "error": None}
    except Exception as err:  # corrupt/unreadable PDF (e.g. pymupdf.FileDataError)
        return {"fileid": int(fileid), "df": None, "missing": False, "error": f"{type(err).__name__}: {err}"}
