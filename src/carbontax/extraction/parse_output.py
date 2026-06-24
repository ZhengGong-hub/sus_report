"""
parse_output.py — flatten combined batch output to analysis-ready columns.

Input:
  output/output_pilot_batch_combined.csv   (produced by BatchManager.get_output_file)
  to_batch_pilot/pilot_batch_combined_ref.parquet

Output columns (one row per chunk):
  [ref columns]  filingId, companyid, companyname, filingDate, chunks, chunk_ids,
                 prompt_version, model

  [tier1]        tier1_S1, tier1_S2, tier1_S3U, tier1_S3D, tier1_CDR

  [tier2]        tier2_{mid}           — boolean adoption flag
                 tier2_{mid}_quote     — verbatim quote (≤15 words)
                 tier2_{mid}_page      — page number

  [governance]   governance_{flag}           — boolean
                 governance_{flag}_quote
                 governance_{flag}_page

  [meta]         model_y, prompt_tokens, completion_tokens

Usage:
  python parse_v2_output.py
  python parse_v2_output.py --output output/output_pilot_batch_combined.csv
                            --ref    to_batch_pilot/pilot_batch_combined_ref.parquet
                            --dest   output/parsed_v2_combined.csv
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re

import pandas as pd

from carbontax.taxonomy import MEASURE_IDS, GOVERNANCE_FLAGS, TIER1_BUCKETS

DEFAULT_OUTPUT = "output/output_pilot_batch_combined.csv"
DEFAULT_REF    = "to_batch_pilot/pilot_batch_combined_ref.parquet"
DEFAULT_DEST   = "output/parsed_v2_combined.csv"

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(v: object) -> object:
    if isinstance(v, str):
        return _ILLEGAL.sub("", v)
    return v


def _parse_json_col(v: object) -> dict:
    if pd.isna(v) or v == "":
        return {}
    if isinstance(v, dict):
        return v
    try:
        return json.loads(str(v))
    except Exception:
        try:
            return ast.literal_eval(str(v))
        except Exception:
            return {}


def _parse_evidence(v: object) -> dict[str, dict]:
    """Return {measure_id: {quote, page}} from the evidence array."""
    if pd.isna(v) or v == "":
        return {}
    try:
        arr = json.loads(str(v)) if isinstance(v, str) else v
    except Exception:
        try:
            arr = ast.literal_eval(str(v))
        except Exception:
            return {}
    result: dict[str, dict] = {}
    if not isinstance(arr, list):
        return result
    for item in arr:
        if isinstance(item, dict) and "measure" in item:
            mid = item["measure"]
            result[mid] = {"quote": item.get("quote"), "page": item.get("page")}
    return result


def flatten_row(row: pd.Series) -> dict:
    """Expand a single output row into flat analysis columns."""
    t1  = _parse_json_col(row.get("tier1",      {}))
    t2  = _parse_json_col(row.get("tier2",      {}))
    gov = _parse_json_col(row.get("governance", {}))
    ev  = _parse_evidence(row.get("evidence",   []))

    flat: dict = {}

    # tier1 — flat booleans
    for bucket in TIER1_BUCKETS:
        flat[f"tier1_{bucket}"] = t1.get(bucket)

    # tier2 — boolean + evidence
    for mid in MEASURE_IDS:
        flat[f"tier2_{mid}"] = t2.get(mid)
        ev_entry = ev.get(mid, {})
        flat[f"tier2_{mid}_quote"] = ev_entry.get("quote")
        flat[f"tier2_{mid}_page"]  = ev_entry.get("page")

    # governance — boolean + evidence
    for flag in GOVERNANCE_FLAGS:
        flat[f"governance_{flag}"] = gov.get(flag)
        ev_entry = ev.get(flag, {})
        flat[f"governance_{flag}_quote"] = ev_entry.get("quote")
        flat[f"governance_{flag}_page"]  = ev_entry.get("page")

    return flat


def parse_output(
    output_path: str = DEFAULT_OUTPUT,
    ref_path:    str = DEFAULT_REF,
    dest_path:   str = DEFAULT_DEST,
) -> pd.DataFrame:
    if not os.path.exists(output_path):
        raise FileNotFoundError(
            f"Output CSV not found: {output_path}\n"
            "Run BatchManager.get_output_file() first, or check the path."
        )

    ref_df = pd.read_parquet(ref_path)
    out_df = pd.read_csv(output_path, dtype={"custom_id": "string"})

    print(f"Ref rows: {len(ref_df)}, output rows: {len(out_df)}")

    flat_rows = [flatten_row(row) for _, row in out_df.iterrows()]
    flat_df   = pd.DataFrame(flat_rows, index=out_df.index)

    meta_cols = ["custom_id", "model", "prompt_tokens", "completion_tokens"]
    meta_df   = out_df[[c for c in meta_cols if c in out_df.columns]].rename(
        columns={"custom_id": "chunk_ids", "model": "model_y"}
    )

    result = ref_df.merge(
        pd.concat([meta_df, flat_df], axis=1),
        on="chunk_ids",
        how="left",
    )

    # clean illegal chars in string columns
    result = result.apply(
        lambda col: col.map(_clean) if col.dtype == object else col
    )

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    result.to_csv(dest_path, index=False)
    print(f"Wrote {len(result)} rows → {dest_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse combined batch output")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="BatchManager output CSV")
    parser.add_argument("--ref",    default=DEFAULT_REF,    help="Reference parquet")
    parser.add_argument("--dest",   default=DEFAULT_DEST,   help="Destination CSV path")
    args = parser.parse_args()

    parse_output(output_path=args.output, ref_path=args.ref, dest_path=args.dest)


if __name__ == "__main__":
    main()
