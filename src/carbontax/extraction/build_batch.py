"""
build_batch.py — build the combined-call batch JSONL (v2 extraction).

One request per chunk; combined tier1 + tier2 + governance + evidence in a
single schema. Reads the same reference parquet used for v1 so the chunk set
is identical (required for the v1/v2 comparison in compare_v1_v2.py).

Output:
  batch_folder/pilot_batch_combined.jsonl        — batch input
  batch_folder/pilot_batch_combined_ref.parquet  — reference (join key)

Usage:
  python build_batch.py
  python build_batch.py --ref batch_folder/pilot_batch_ref.parquet
                        --out batch_folder/pilot_batch_combined.jsonl
                        --model gpt-5-mini
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from carbontax.taxonomy import (
    PROMPT_VERSION,
    build_combined_schema,
    build_combined_system_prompt,
)

DEFAULT_REF     = "batch_folder/pilot_batch_ref.parquet"
DEFAULT_OUT     = "batch_folder/pilot_batch_combined.jsonl"
DEFAULT_REF_OUT = "batch_folder/pilot_batch_combined_ref.parquet"
DEFAULT_MODEL   = "gpt-5.4-mini"


def build_request(chunk_id: str, chunk_text: str, model: str, schema: dict, system_prompt: str) -> dict:
    return {
        "custom_id": chunk_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
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


def build_batch(
    ref_path: str = DEFAULT_REF,
    out_path: str = DEFAULT_OUT,
    ref_out_path: str = DEFAULT_REF_OUT,
    model: str = DEFAULT_MODEL,
) -> None:
    ref_df = pd.read_parquet(ref_path)
    print(f"Loaded {len(ref_df)} chunks from {ref_path}")

    schema        = build_combined_schema()
    system_prompt = build_combined_system_prompt()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as fh:
        for _, row in ref_df.iterrows():
            req = build_request(
                chunk_id=row["chunk_ids"],
                chunk_text=row["chunks"],
                model=model,
                schema=schema,
                system_prompt=system_prompt,
            )
            fh.write(json.dumps(req, ensure_ascii=False) + "\n")
            n_written += 1

    ref_out = ref_df.copy()
    ref_out["prompt_version"] = PROMPT_VERSION
    ref_out["model"]          = model
    ref_out.to_parquet(ref_out_path, index=False)

    print(f"Wrote {n_written} requests → {out_path}")
    print(f"Wrote reference parquet  → {ref_out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined-call batch JSONL")
    parser.add_argument("--ref",   default=DEFAULT_REF,     help="Input reference parquet")
    parser.add_argument("--out",   default=DEFAULT_OUT,     help="Output JSONL path")
    parser.add_argument("--model", default=DEFAULT_MODEL,   help="OpenAI model id")
    args = parser.parse_args()

    build_batch(
        ref_path=args.ref,
        out_path=args.out,
        model=args.model,
    )


if __name__ == "__main__":
    main()
