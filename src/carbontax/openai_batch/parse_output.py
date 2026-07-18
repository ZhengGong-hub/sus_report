"""Flatten the downloaded batch output into one analysis-ready CSV row per chunk."""

from __future__ import annotations

import ast
import logging
import os
import re

import pandas as pd

from carbontax.config import load_run_config
from carbontax.paths import combined_ref, output_csv, parsed_csv
from carbontax.taxonomy import MEASURE_IDS, GOVERNANCE_FLAGS, TIER1_BUCKETS
from carbontax.utils.logger import setup_logging

logger = logging.getLogger(__name__)

# control characters that Excel/openpyxl reject in cell values
_ILLEGAL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _clean(v: object) -> object:
    if isinstance(v, str):
        return _ILLEGAL.sub("", v)
    return v


def _parse_dict_cell(v: object) -> dict:
    # batch output stores dicts as Python reprs (single quotes, True/False),
    # so ast.literal_eval is the right parser; a malformed cell raises (fail fast)
    if pd.isna(v) or v == "":
        return {}
    if isinstance(v, dict):
        return v
    return ast.literal_eval(str(v))


def _parse_evidence(v: object) -> dict[str, dict]:
    # evidence array → {measure_id: {quote, page}}
    if pd.isna(v) or v == "":
        return {}
    arr = v if isinstance(v, list) else ast.literal_eval(str(v))
    result: dict[str, dict] = {}
    for item in arr:
        if isinstance(item, dict) and "measure" in item:
            result[item["measure"]] = {"quote": item.get("quote"), "page": item.get("page")}
    return result


def flatten_row(row: pd.Series) -> dict:
    """One output row → flat columns: tier1_*, tier2_*(+_quote/_page), governance_*(+_quote/_page)."""
    t1  = _parse_dict_cell(row.get("tier1",      {}))
    t2  = _parse_dict_cell(row.get("tier2",      {}))
    gov = _parse_dict_cell(row.get("governance", {}))
    ev  = _parse_evidence(row.get("evidence",    []))

    flat: dict = {}
    for bucket in TIER1_BUCKETS:
        flat[f"tier1_{bucket}"] = t1.get(bucket)
    for mid in MEASURE_IDS:
        flat[f"tier2_{mid}"] = t2.get(mid)
        flat[f"tier2_{mid}_quote"] = ev.get(mid, {}).get("quote")
        flat[f"tier2_{mid}_page"]  = ev.get(mid, {}).get("page")
    for flag in GOVERNANCE_FLAGS:
        flat[f"governance_{flag}"] = gov.get(flag)
        flat[f"governance_{flag}_quote"] = ev.get(flag, {}).get("quote")
        flat[f"governance_{flag}_page"]  = ev.get(flag, {}).get("page")
    return flat


def parse_output(run_name: str) -> pd.DataFrame:
    output_path = output_csv(run_name)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Output CSV not found: {output_path} — run the download step first.")

    ref_df = pd.read_parquet(combined_ref(run_name))
    out_df = pd.read_csv(output_path, dtype={"custom_id": "string"})
    logger.info("Ref rows: %d, output rows: %d", len(ref_df), len(out_df))

    flat_df = pd.DataFrame([flatten_row(row) for _, row in out_df.iterrows()], index=out_df.index)

    meta_cols = ["custom_id", "model", "prompt_tokens", "completion_tokens"]
    meta_df = out_df[[c for c in meta_cols if c in out_df.columns]].rename(
        columns={"custom_id": "chunk_ids", "model": "model_y"}
    )

    result = ref_df.merge(pd.concat([meta_df, flat_df], axis=1), on="chunk_ids", how="left")

    result = result.apply(lambda col: col.map(_clean) if col.dtype == object else col)

    # keep page numbers as nullable Int64 so missing pages stay blank
    # instead of rendering as "11.0" via float NaN promotion
    for c in [c for c in result.columns if c.endswith("_page")]:
        result[c] = pd.to_numeric(result[c], errors="coerce").astype("Int64")

    dest_path = parsed_csv(run_name)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    result.to_csv(dest_path, index=False)
    logger.info("Wrote %d rows → %s", len(result), dest_path)
    return result


def main() -> None:
    setup_logging()
    parse_output(load_run_config()["run_name"])


if __name__ == "__main__":
    main()
