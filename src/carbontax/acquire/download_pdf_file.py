# the strategy is such that 
#   I first have the companyids of all the US companies which are a lot: let's say more than 10 thousands
#   then I basically split them into a list of 10 elements each.
#   and then for each 10 companies, i pull down their filings of sustainability reports with an rolling time window of 3 months, starting from 2005-01-01 to today.
import requests
import json
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import glob

from carbontax.utils.logger import Logger
# https://help.edgar-online.com/edgar/formtypes.asp

from carbontax.acquire.tokens import quick_refresh_and_save_token

logger = Logger.get("acquire.pdfs")

# --- CONFIGURATION ---
BASE_URL = "https://api-ciq.marketintelligence.spglobal.com/gds/documents/api/v1"
COMPANY_ID = [29002, 288502,139677, 247483, 285467, 162270, 25798, 24937] # Example ID for NVIDIA
INCL_FILETYPEID = [
    432, # Sustainability Report
    1010, # Corporate Social Responsibility Report
    1091, # Environmental Report
    1090, # Corporate Governance Report
    1105, # TCFD Report
]

def load_company_df():
    company_df = pd.read_csv("company.csv").dropna(subset=["simpleindustryid", "tradingitemstatusid"]).query("companytypeid in [4, 5] and securitysubtypeid == 1")
    company_df = company_df.query("importancelevel <= 5") # keep nasdaq global market, nyse, nyse american llc, nasdaq capital market, nasdaq global select
    company_df = company_df[(company_df["securityenddate"].isna()) | (company_df["securityenddate"] >= "2000-01-01")]
    company_df = company_df.drop(columns=["countryid.1", "countryid", "officefaxvalue", "city", "exchangeid.1", "officephonevalue", "otherphonevalue", "streetaddress", "streetaddress2", "streetaddress3", "streetaddress4", "yearfounded", "monthfounded", "dayfounded", "zipcode", "securitysubtypeid", "primaryflag.1", "primaryflag", "currencyid.1", "currencyid", "importancelevel", "exchangename", "exchangeid", "companyid.1", "incorporationcountryid", "incorporationstateid", "reportingtemplatetypeid", "stateid", "securityid.1", "securityid", "securitystartdate", "webpage"])
    company_df = company_df.drop_duplicates(subset=["companyid"])
    return company_df


def create_intermed_folders():
    company_df = load_company_df()
    company_df_list = [company_df.iloc[i:i+10] for i in range(0, len(company_df), 10)]
    for i, company_df in enumerate(company_df_list):
        os.makedirs(f"intermed/{i}", exist_ok=True)
        company_df.to_csv(f"intermed/{i}/company_df.csv", index=False)
        with open(f"intermed/{i}/companyids.txt", "w") as f:
            for companyid in company_df['companyid'].tolist():
                f.write(f"{companyid}\n")
        with open(f"intermed/{i}/finished.txt", "w") as f:
            f.write("not finished")


def load_tokens():
    tokens_path = os.path.join("tokens", "token_current.json")
    with open(tokens_path, "r") as f:
        current_tokens = json.load(f)
    return current_tokens["access_token"]

def gen_start_end_dates(
    start_date: str,
    end_date: str,
):
    windows = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while start < end:
        windows.append((start.strftime("%Y-%m-%d"), (start + timedelta(days=89)).strftime("%Y-%m-%d")))
        start = start + timedelta(days=90)
    return windows

def catch_fileids(
    headers: dict,
    companyids: list[int] = COMPANY_ID,
    min_filing_date: str = "2023-07-01",
    max_filing_date: str = "2023-09-28",
    intermed: str = None,
):
    # 1. SEARCH: Find document IDs for the company
    search_payload = {
        "properties": {
            "companyId": companyids,
            "minFilingDate": min_filing_date,
            "maxFilingDate": max_filing_date,
            "fileTypeId": INCL_FILETYPEID,
            "pageNumber": 1,
            "pageSize": 500,
        }
    }

    # The docType is required as a query parameter
    search_response = requests.post(
        f"{BASE_URL}/search?docType=FILINGS_DOCUMENTS_API", 
        headers=headers, 
        json=search_payload
    )
    if search_response.status_code == 400 or search_response.status_code == 401:
        raise RuntimeError(f"Search API returned error 400 or 401: {search_response.text}")

    response_data = search_response.json()
    if not response_data.get('numRows'):
        logger.info("No documents found")
        return
    else:
        logger.info(f"Found {response_data.get('numRows', 0)} documents.")

    # Convert to DataFrame
    column_headers = response_data.get("headers", [])
    rows_data = [row.get("row", []) for row in response_data.get("rows", [])]

    df = pd.DataFrame(rows_data, columns=column_headers)
    df = df.drop(columns=["filingVersionId", "language", "languageId", "documentVersionId","supplierDocumentId", "institutionId"])# .query("fileTypeId != @DROP_FILETYPEID")
    logger.info("\n%s", df)

    # i need to maintain a csv that contains this df
    #  every row if the filingid is not in the csv, i need to update the csv with the new row
    if not os.path.exists(f"{intermed}/fileids.csv"):
        # create the csv with the columns
        df.to_csv(f"{intermed}/fileids.csv", index=False)
    else:
        # read the csv
        df_existing = pd.read_csv(f"{intermed}/fileids.csv")
        # merge the two dfs
        df_ = pd.concat([df_existing, df])
        # drop duplicates
        df_ = df_.drop_duplicates(subset=["filingId", "periodDate", "filingDate", "processedDate", "documentId"])
        # save the csv
        df_.to_csv(f"{intermed}/fileids.csv", index=False)
    return df 


def download_files(
    df: pd.DataFrame = None,
    headers: dict = None,
    addr: str = "files/",
):
    if df is None:
        logger.info("No fileids given! Skipping download.")
        return

    # 2. DOWNLOAD: Loop through results and save each file
    for _, row in df.iterrows():
        # Pulling the unique filingId from the search results
        filing_id = row["filingId"]

        if os.path.exists(f"{addr}{filing_id}.pdf"):
            logger.info(f"File {filing_id} already exists! Skipping download.")
            continue

        download_payload = {
            "properties": {
                "filingId": filing_id
            }
        }

        # Call the download endpoint
        file_response = requests.post(
            f"{BASE_URL}/download?docType=FILINGS_DOCUMENTS_API",
            headers=headers,
            json=download_payload
        )
        
        if file_response.status_code == 200:
            filename = f"{addr}{filing_id}.pdf"
            with open(filename, "wb") as f:
                f.write(file_response.content)  # Save binary content to disk
            logger.info(f"Successfully saved {filename}")
        else:
            logger.info(f"Failed to download ID {filing_id}: {file_response.status_code}")


def main():
    # create_intermed_folders()

    # Roll every 2 months from 2005-01-01 to today, in window [start, end), e.g., Jan 1 to Mar 1 (exclusive)
    windows = gen_start_end_dates(
        start_date='2015-01-01',
        end_date='2025-01-01',
    )

    # ------------------------------- MAIN LOOP -------------------------------
    glob_intermed = glob.glob("intermed/*")
    # reorder glob_intermed by the number in the folder name
    glob_intermed = sorted(glob_intermed, key=lambda x: int(x.split("/")[-1]))
    # jump the first 1000
    glob_intermed = glob_intermed[1050:]

    # load the companyids
    for intermed in glob_intermed:

        AUTH_TOKEN = quick_refresh_and_save_token(token_path="tokens/token_current.json")["access_token"]
        headers = {
            "Authorization": f"Bearer {AUTH_TOKEN}",
            "Content-Type": "application/json",
            "accept": "application/json"
        }

        logger.info("--------------------------------")
        logger.info(f"Processing {intermed}")

        # check if the finished.txt is finished
        if os.path.exists(f"{intermed}/finished.txt"):
            with open(f"{intermed}/finished.txt", "r") as f:
                if f.read() == "already finished!":
                    logger.info("already finished!")
                    continue

        # load the companyids
        companyids = []
        with open(f"{intermed}/companyids.txt", "r") as f:
            for line in f:
                companyids.append(int(line.strip()))

        # load the windows
        # start the loop
        for start, end in windows:
            logger.info(f"Processing from {start} to {end}")
            fileids_df = catch_fileids(
                headers=headers,
                companyids=companyids,
                min_filing_date=start, 
                max_filing_date=end,
                intermed=intermed,
            )

            time.sleep(0.5)

            logger.info(f"Downloading files for {start} to {end}")
            download_files(
                headers=headers,
                df=fileids_df,
            )

        # set the finished.txt to finished
        with open(f"{intermed}/finished.txt", "w") as f:
            f.write("already finished!")
        logger.info(f"Finished processing {intermed}")
        try:
            logger.info(f"with {len(pd.read_csv(f'{intermed}/fileids.csv'))} fileids")
        except:
            logger.info("no fileids found")
        logger.info(f"from {start} to {end}")
        logger.info(f"companyids: {companyids}")
        logger.info("--------------------------------")


if __name__ == "__main__":
    main()