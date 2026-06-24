"""
acquire/mapping.py — aggregate per-batch fileid CSVs into one company↔filing map.

Scans intermed/*/fileids.csv (written during PDF acquisition), concatenates
them, joins company names, de-duplicates, and writes the canonical mapping that
the chunking pipeline reads.

Output:
  mapping_data/company_esgfiling_mapping.csv

Run from the repo root:
  python -m carbontax.acquire.mapping
  carbontax-mapping            # console entry point
"""

import glob
import os

import pandas as pd
import tqdm

OUTPUT_DIR = "mapping_data"
OUTPUT_CSV = f"{OUTPUT_DIR}/company_esgfiling_mapping.csv"


def aggregate_mapping() -> pd.DataFrame:
    to_agg = []
    for intermed in tqdm.tqdm(glob.glob("intermed/*")):
        addr = f"{intermed}/fileids.csv"
        if not os.path.exists(addr):
            continue
        to_agg.append(pd.read_csv(addr))

    res_df = pd.concat(to_agg).reset_index(drop=True)
    res_df.rename(columns={"companyId": "companyid"}, inplace=True)

    company_ref = pd.read_csv("company.csv").drop_duplicates(subset=["companyid"])
    res_df = pd.merge(res_df, company_ref[["companyid", "companyname"]], on="companyid", how="left")

    # some filingIds are duplicated because different companyids claim the same
    # filing (e.g. AMD and Xilinx, Viacom and Paramount Skydance).
    res_df.drop_duplicates(subset=["filingId", "companyid"], inplace=True)
    return res_df


def main() -> None:
    res_df = aggregate_mapping()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    res_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(res_df)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
