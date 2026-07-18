# Refactor protocol — 2026-07-18

Two passes over `src/carbontax`: first a cleanup of the existing code, then a
restructure into four explicit pipeline stages. All decisions below were made
interactively; behavior was verified unchanged against the completed pilot run.

## Pass 1 — cleanup

**Bugs fixed**
- `carbontax-acquire` entry point pointed at a nonexistent module (`acquire.pdfs`).
- `RecursiveTextSplitter` crashed with its own default `logger=None`.
- `SemanticFilter` built an `LLMWrapper` in `__init__`, requiring `OPENAI_API_KEY`
  even for keyword-only filtering → now lazy. Keywords regex-escaped and matching
  made case-insensitive (as documented; affects future chunking runs only).
- Model default unified to `gpt-5.4-mini` (what the pilot batch actually ran with);
  previously `utils/llm.py` said `gpt-5-mini` while `build_batch` said `gpt-5.4-mini`.
- `tokens.py` logged raw access/refresh tokens into `logs/` — removed.
  (Old log files may still contain tokens; consider deleting them.)

**try/except removed everywhere** (decision: fail fast)
- Batch-output cells are Python reprs → parsed with `ast.literal_eval` only;
  malformed cells raise instead of silently becoming None/{}.
- Bare `except:` in the acquire script → `os.path.exists` check.
- `call_structured` lets `json.JSONDecodeError` propagate.

**Dead code deleted**: `create_jsonl_for_batch`, `SUMMARY_SCHEMA`, taxonomy's v1
schema builders + `V1_MEASURE_IDS` (~130 lines).

**Logging**: custom `Logger` class (one log file per module per call) replaced by
`setup_logging()` — one timestamped file per run under `logs/` + stderr; modules
use standard `logging.getLogger(__name__)`.

**Excel reporter**: three near-identical stats-sheet writers collapsed into one
generic `_write_stats_sheet` + small per-sheet style functions.

## Pass 2 — restructure into four stages

Stages are split by *what they talk to*, with maximally explicit names:

| Package | Talks to | Class | Commands |
|---|---|---|---|
| `acquire_pdfs/` | S&P CIQ API | `PdfAcquirer` | `carbontax-acquire`, `carbontax-mapping` |
| `prepare_batch_input/` | local CPU only | `BatchInputPreparer` | `carbontax-prepare` (chunk + JSONL in one go) |
| `openai_batch/` | OpenAI batch API | `OpenAIBatchJob` | `carbontax-submit`, `-status`, `-download`, `-parse` |
| `analysis/` | pandas/openpyxl | (`ExcelReporter`) | `carbontax-report` |

Shared top-level modules:
- `paths.py` — the data-layout contract: everything for a run lives in
  `batch_folder/<run_name>/`. (`JOB_ID` filename stem kept for compatibility with
  the existing pilot folder.)
- `config.py` — loads `config/run.yaml`.
- `taxonomy.py` — what gets extracted (untouched).

**Working conventions decided**
- **No CLI arguments.** Edit `config/run.yaml` (one canonical file, `run_name` +
  one section per stage), then run the bare command — or run the module file
  from the IDE. Runnable files are thin 3-line `main()`s.
- **No hidden defaults.** Every tunable knob is an explicit required YAML key
  (model, `max_chunks_per_file`, `min_page_tokens`, `chunk_max_tokens`,
  `chunk_overlap_tokens`, `filter_keywords`, dates, …); the code reads
  `section["key"]` so a missing key fails loudly. Only run-folder file paths are
  derived from `run_name` (writing them literally would drift on rename).
- **Classes only where state is shared** (the three stage classes above);
  `paths`/`config`/`taxonomy` stay function-based.
- **Short docstrings** (≤ 2 lines); naming + inline comments carry readability.

**Data-flow simplification**: the chunk step now writes the reference parquet
directly into the run folder; the JSONL builder reads it from there. The old
`to_batch_pilot/` intermediary is no longer written.

**New capability**: `carbontax-status` (`openai_batch/check_status.py`) — inspect
batch state and request counts without downloading.

**Deleted** (all recoverable from git history):
- `analysis/compare_v1_v2.py` — its one-time v1→v2 validation job is done
  (conclusions in `batch_folder/pilot/RESULTS_COMPARISON.md`).
- `legacy/` — the entire v1-era folder (11 files).
- `config/pilot.yaml` (old format) → replaced by `config/run.yaml`, same values.

**Renames**: `analysis/excel.py` → `build_report.py`; `run_batch.py` →
`submit_batch.py`; `pipeline.py` + `build_batch.py` absorbed into `preparer.py`.
Version bumped to 0.3.0.

## Verification (both passes)

- All modules import; all 8 console commands installed and runnable.
- `carbontax-parse` re-run on the pilot output → **byte-identical** to the
  existing `parsed_v2_combined.csv` (1,653 rows).
- `carbontax-report` re-run → all 11 sheets **cell-by-cell identical** to the
  existing workbook (values, fills, fonts).
- `carbontax-status` → pilot batch: completed, 1653/1653, 0 failed.
- `carbontax-prepare` smoke-tested on a scratch run name (2 PDFs → chunks →
  JSONL with correct model), then removed.
- Greps confirm: zero `try`/`except`, zero `argparse`, zero old-package imports.

## Open items

- `README.md` still describes the pre-refactor layout (old package names,
  `--run-name` flags, `carbontax-compare`) — needs a rewrite.
- ~~Old files under `logs/` may contain API tokens logged by the previous code.~~
  Done: `log/` and `logs/` contents deleted (token purge included); `log/*` added
  to .gitignore. Note `log/` is created by the external openai_batch_wrapper
  library (not configurable), so the folder reappears on every batch command.
- `JOB_ID = "pilot_batch_combined"` still bakes "pilot" into every run's
  filenames; renaming it would orphan the existing pilot folder, so deferred
  until a genuinely new run starts.
- Per-stage part-by-part review (agreed iteration style) not started yet.
