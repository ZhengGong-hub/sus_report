"""
carbontax — LLM-based extraction of corporate carbon-reduction measures
from sustainability-report PDFs, classified into a sourced two-tier taxonomy.

Pipeline stages (see README):
  acquire    → download filings + build the company↔filing mapping
  chunking   → PDF → filtered text chunks → reference parquet
  extraction → chunks → combined-call batch → OpenAI → parsed CSV
  analysis   → v1/v2 comparison and downstream stats

The taxonomy (single source of truth) lives in `carbontax.taxonomy`.
"""

__version__ = "0.2.0"
