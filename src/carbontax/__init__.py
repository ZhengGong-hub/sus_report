"""
carbontax — LLM-based extraction of corporate carbon-reduction measures
from sustainability-report PDFs, classified into a sourced two-tier taxonomy.

Pipeline stages (all configured via config/run.yaml, no CLI arguments):
  acquire_pdfs        → download filings + build the company↔filing mapping
  prepare_batch_input → PDFs → filtered chunks → reference parquet → batch JSONL
  openai_batch        → submit → status → download → parse to flat CSV
  analysis            → parsed CSV → styled review workbook

What gets extracted lives in `carbontax.taxonomy`; where files live in `carbontax.paths`.
"""

__version__ = "0.3.0"
