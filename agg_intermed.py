import pandas as pd 
import glob
import tqdm
import os

def agg_intermed():
    glob_intermed = glob.glob("intermed/*")
    # reorder glob_intermed by the number in the folder name

    to_agg = []

    for intermed in tqdm.tqdm(glob_intermed):

        addr = f"{intermed}/fileids.csv"
        if not os.path.exists(addr):
            continue
        fileids_df = pd.read_csv(addr)
        to_agg.append(fileids_df)
    res_df = pd.concat(to_agg)

    res_df.reset_index(drop=True, inplace=True)
    
    res_df.rename(columns={"companyId": "companyid"}, inplace=True)

    company_ref = pd.read_csv("company.csv").drop_duplicates(subset=["companyid"])


    res_df = pd.merge(res_df, company_ref[['companyid', 'companyname']], on="companyid", how="left")
    res_df.drop_duplicates(subset=["filingId", "companyid"], inplace=True) # note: some filingid are duplicated becasue of different companyids claim the same filings, for example AMD and Xilix, Viacom and Paramount Skydance.
    return res_df

if __name__ == "__main__":
    res_df = agg_intermed()
    output_addr = "outputs"
    os.makedirs(output_addr, exist_ok=True)
    res_df.to_csv(f"{output_addr}/company_esgfiling_mapping.csv", index=False)