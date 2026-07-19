"""Aggregate <intermed_dir>/*/fileids.csv into the canonical company↔filing mapping CSV."""

import glob
import os

import pandas as pd
import tqdm

from carbontax.config import load_run_config


def aggregate_mapping(intermed_dir: str, company_csv: str) -> pd.DataFrame:
    to_agg = []
    for intermed in tqdm.tqdm(glob.glob(f"{intermed_dir}/*")):
        addr = f"{intermed}/fileids.csv"
        if not os.path.exists(addr):
            continue
        to_agg.append(pd.read_csv(addr))

    res_df = pd.concat(to_agg).reset_index(drop=True)
    res_df.rename(columns={"companyId": "companyid"}, inplace=True)

    company_ref = pd.read_csv(company_csv).drop_duplicates(subset=["companyid"])
    res_df = pd.merge(res_df, company_ref[["companyid", "companyname"]], on="companyid", how="left")

    # some filingIds are duplicated because different companyids claim the same
    # filing (e.g. AMD and Xilinx, Viacom and Paramount Skydance).
    res_df.drop_duplicates(subset=["filingId", "companyid"], inplace=True)
    return res_df


def main() -> None:
    data = load_run_config()["data"]
    res_df = aggregate_mapping(data["output"]["intermed_dir"], data["input"]["company_csv"])
    mapping_csv = data["output"]["mapping_csv"]
    os.makedirs(os.path.dirname(mapping_csv), exist_ok=True)
    res_df.to_csv(mapping_csv, index=False)
    print(f"Wrote {len(res_df)} rows → {mapping_csv}")


if __name__ == "__main__":
    main()
