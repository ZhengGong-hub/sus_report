"""
parse_output.py — flatten combined batch output to analysis-ready columns.

Input (under the run folder batch_folder/<run_name>/):
  output_pilot_batch_combined.csv   (produced by BatchManager.get_output_file)
  pilot_batch_combined_ref.parquet

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
  python parse_v2_output.py --run-name pilot
  python parse_v2_output.py --run-name pilot
                            --output batch_folder/pilot/output_pilot_batch_combined.csv
                            --ref    batch_folder/pilot/pilot_batch_combined_ref.parquet
                            --dest   batch_folder/pilot/parsed_v2_combined.csv
"""

from __future__ import annotations

import argparse
import ast
import os
import re

import pandas as pd

from carbontax.extraction.paths import DEFAULT_RUN_NAME, combined_ref, output_csv, parsed_csv
from carbontax.taxonomy import MEASURE_IDS, GOVERNANCE_FLAGS, TIER1_BUCKETS

_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(v: object) -> object:
    if isinstance(v, str):
        return _ILLEGAL.sub("", v)
    return v


def _parse_json_col(v: object) -> dict:
    """Parse a dict cell. The batch output stores dicts as Python reprs
    (single quotes, True/False), so ast.literal_eval is the right parser;
    a malformed cell raises rather than being silently dropped."""
    if pd.isna(v) or v == "":
        return {}
    if isinstance(v, dict):
        return v
    return ast.literal_eval(str(v))


def _parse_evidence(v: object) -> dict[str, dict]:
    """Return {measure_id: {quote, page}} from the evidence array."""
    if pd.isna(v) or v == "":
        return {}
    arr = v if isinstance(v, list) else ast.literal_eval(str(v))
    result: dict[str, dict] = {}
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
    output_path: str | None = None,
    ref_path:    str | None = None,
    dest_path:   str | None = None,
    run_name:    str = DEFAULT_RUN_NAME,
) -> pd.DataFrame:
    output_path = output_path or output_csv(run_name)
    ref_path    = ref_path    or combined_ref(run_name)
    dest_path   = dest_path   or parsed_csv(run_name)

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

    # page numbers are ints; keep them as nullable Int64 so missing pages stay
    # blank instead of rendering as "11.0" via float NaN promotion.
    page_cols = [c for c in result.columns if c.endswith("_page")]
    for c in page_cols:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype("Int64")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    result.to_csv(dest_path, index=False)
    print(f"Wrote {len(result)} rows → {dest_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse combined batch output")
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME, help="Run folder name under batch_folder/")
    parser.add_argument("--output", default=None, help="BatchManager output CSV (defaults to run folder)")
    parser.add_argument("--ref",    default=None, help="Reference parquet (defaults to run folder)")
    parser.add_argument("--dest",   default=None, help="Destination CSV path (defaults to run folder)")
    args = parser.parse_args()

    parse_output(
        output_path=args.output,
        ref_path=args.ref,
        dest_path=args.dest,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
