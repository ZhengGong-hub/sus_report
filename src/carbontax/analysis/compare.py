"""
compare_v1_v2.py — per-measure adoption-rate delta (v2 − v1) on the same chunk set.

This is the attention-degradation check (§4 of the handoff): if combining tier1
and tier2 into a single call degrades rare-category recall, it shows up as a
systematic downward shift in adoption rates for those measures.

Flags any measure where |delta| > THRESHOLD_PP (default 5 pp).

Inputs:
  v1 tier2 parsed CSV  (output/clean_output_pilot_batch_tier2.csv)
  v2 combined parsed CSV (output/parsed_v2_combined.csv)  — must exist first

Output:
  output/compare_v1_v2.csv   — one row per measure/flag with adoption rates and delta
  Prints flagged measures to stdout.

Usage:
  python compare_v1_v2.py
  python compare_v1_v2.py --v1  output/clean_output_pilot_batch_tier2.csv
                          --v2  output/parsed_v2_combined.csv
                          --threshold 5
"""

from __future__ import annotations

import argparse
import ast
import os

import pandas as pd

from carbontax.extraction.paths import DEFAULT_RUN_NAME, parsed_csv
from carbontax.taxonomy import MEASURE_IDS, GOVERNANCE_FLAGS

DEFAULT_V1        = "output/clean_output_pilot_batch_tier2.csv"
DEFAULT_V2        = parsed_csv(DEFAULT_RUN_NAME)
DEFAULT_DEST      = "output/compare_v1_v2.csv"
DEFAULT_THRESHOLD = 5.0

# v1 measures that no longer exist in v2 (dropped or split)
V1_DROPPED  = {"247_cfe"}
V1_SPLIT    = {"c1_purchased_goods"}  # split into c1_supplier_engagement + c1_material_substitution
# v2 measures that are new (no v1 counterpart to compare against)
V2_NEW      = {"renewable_electricity_general", "c1_supplier_engagement",
               "c1_material_substitution", "packaging"}


def _parse_dict_col(series: pd.Series) -> pd.DataFrame:
    """Expand a column of stringified dicts into a flat boolean DataFrame.
    Cells are Python reprs; a malformed cell raises rather than being dropped."""
    def parse(v: object) -> dict:
        if pd.isna(v) or v == "":
            return {}
        return ast.literal_eval(str(v)) if isinstance(v, str) else v

    parsed = series.apply(parse)
    keys: dict[str, None] = {}
    for d in parsed:
        keys.update(dict.fromkeys(d.keys()))

    rows: dict[str, list] = {}
    for key in keys:
        rows[key] = parsed.apply(lambda d, k=key: d.get(k, {}).get("adopted") if isinstance(d.get(k), dict) else d.get(k))
    return pd.DataFrame(rows, index=series.index)


def adoption_rate(series: pd.Series) -> float:
    """% of non-null rows where the value is True."""
    valid = series.dropna()
    if valid.empty:
        return float("nan")
    return 100.0 * valid.astype(bool).sum() / len(valid)


def load_v1_rates(v1_path: str) -> pd.Series:
    """Return adoption rates (%) per measure from the v1 tier2 output."""
    df = pd.read_csv(v1_path, dtype={"chunk_ids": "string"})

    rates: dict[str, float] = {}

    # tier2 dict column
    if "tier2" in df.columns:
        t2 = _parse_dict_col(df["tier2"])
        for col in t2.columns:
            rates[col] = adoption_rate(t2[col])

    # governance dict column
    if "governance" in df.columns:
        gov = _parse_dict_col(df["governance"])
        for col in gov.columns:
            rates[col] = adoption_rate(gov[col])

    return pd.Series(rates, name="v1_rate_pct")


def load_v2_rates(v2_path: str) -> pd.Series:
    """Return adoption rates (%) per measure from the parsed v2 output."""
    df = pd.read_csv(v2_path, dtype={"chunk_ids": "string"})

    rates: dict[str, float] = {}
    for mid in MEASURE_IDS:
        col = f"tier2_{mid}"
        if col in df.columns:
            rates[mid] = adoption_rate(df[col])
    for flag in GOVERNANCE_FLAGS:
        col = f"governance_{flag}"
        if col in df.columns:
            rates[flag] = adoption_rate(df[col])

    return pd.Series(rates, name="v2_rate_pct")


def compare(
    v1_path:   str   = DEFAULT_V1,
    v2_path:   str   = DEFAULT_V2,
    dest_path: str   = DEFAULT_DEST,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    if not os.path.exists(v1_path):
        raise FileNotFoundError(f"v1 CSV not found: {v1_path}")
    if not os.path.exists(v2_path):
        raise FileNotFoundError(
            f"v2 CSV not found: {v2_path}\n"
            "Run parse_v2_output.py first after the v2 batch completes."
        )

    v1_rates = load_v1_rates(v1_path)
    v2_rates = load_v2_rates(v2_path)

    # canonical taxonomy order first; v1-only measures (dropped/split) at the end
    order = {m: i for i, m in enumerate(MEASURE_IDS + GOVERNANCE_FLAGS)}
    all_measures = sorted(
        set(v1_rates.index) | set(v2_rates.index),
        key=lambda m: (order.get(m, len(order)), m),
    )

    rows = []
    for mid in all_measures:
        r1 = v1_rates.get(mid, float("nan"))
        r2 = v2_rates.get(mid, float("nan"))
        delta = r2 - r1 if not (pd.isna(r1) or pd.isna(r2)) else float("nan")
        note = ""
        if mid in V1_DROPPED:
            note = "dropped_in_v2"
        elif mid in V1_SPLIT:
            note = "split_in_v2"
        elif mid in V2_NEW:
            note = "new_in_v2"
        rows.append({
            "measure":      mid,
            "v1_rate_pct":  round(r1, 2) if not pd.isna(r1) else float("nan"),
            "v2_rate_pct":  round(r2, 2) if not pd.isna(r2) else float("nan"),
            "delta_pp":     round(delta, 2) if not pd.isna(delta) else float("nan"),
            "note":         note,
        })

    result = pd.DataFrame(rows)

    flagged = result[result["delta_pp"].abs() > threshold]
    if not flagged.empty:
        print(f"\n⚠  {len(flagged)} measure(s) shifted >|{threshold}| pp (v2 − v1):")
        print(flagged[["measure", "v1_rate_pct", "v2_rate_pct", "delta_pp"]].to_string(index=False))
    else:
        print(f"\n✓  No measure shifted >|{threshold}| pp — attention degradation not detected.")

    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    result.to_csv(dest_path, index=False)
    print(f"\nFull comparison → {dest_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1 vs v2 adoption rates")
    parser.add_argument("--v1",        default=DEFAULT_V1,        help="v1 tier2 output CSV")
    parser.add_argument("--v2",        default=DEFAULT_V2,        help="v2 parsed output CSV")
    parser.add_argument("--dest",      default=DEFAULT_DEST,      help="Output comparison CSV")
    parser.add_argument("--threshold", default=DEFAULT_THRESHOLD, type=float,
                        help="Flag delta > this many percentage points")
    args = parser.parse_args()

    compare(
        v1_path=args.v1,
        v2_path=args.v2,
        dest_path=args.dest,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
