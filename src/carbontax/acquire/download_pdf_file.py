"""
acquire/download_pdf_file.py — bulk-download sustainability-report PDFs from S&P CIQ.

Strategy: the universe of US companies (company.csv) is split into batches of 10,
one folder each under intermed/<n>/ (created by `create_intermed_folders`). For
every batch we roll a ~3-month window from --start-date to --end-date, search the
CIQ filings API for sustainability-type documents, append hits to
intermed/<n>/fileids.csv, and download each PDF to files/<filingId>.pdf.

A batch folder is marked done via intermed/<n>/finished.txt, so the loop is
resumable; --resume-from skips the first N batch folders outright.

File-type reference: https://help.edgar-online.com/edgar/formtypes.asp

Run from the repo root:
  carbontax-acquire --start-date 2015-01-01 --end-date 2025-01-01 --resume-from 1050
"""
import argparse
import glob
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from carbontax.acquire.tokens import quick_refresh_and_save_token
from carbontax.utils.logger import setup_logging

logger = logging.getLogger(__name__)

BASE_URL = "https://api-ciq.marketintelligence.spglobal.com/gds/documents/api/v1"
TOKEN_PATH = "tokens/token_current.json"
FINISHED_MARKER = "already finished!"

INCL_FILETYPEID = [
    432,   # Sustainability Report
    1010,  # Corporate Social Responsibility Report
    1091,  # Environmental Report
    1090,  # Corporate Governance Report
    1105,  # TCFD Report
]

# Search-result columns we never use downstream.
DROP_SEARCH_COLS = [
    "filingVersionId", "language", "languageId", "documentVersionId",
    "supplierDocumentId", "institutionId",
]


def load_company_df() -> pd.DataFrame:
    """US-listed operating companies from company.csv, one row per companyid."""
    company_df = (
        pd.read_csv("company.csv")
        .dropna(subset=["simpleindustryid", "tradingitemstatusid"])
        .query("companytypeid in [4, 5] and securitysubtypeid == 1")
    )
    # keep nasdaq global market, nyse, nyse american llc, nasdaq capital market,
    # nasdaq global select
    company_df = company_df.query("importancelevel <= 5")
    company_df = company_df[
        (company_df["securityenddate"].isna())
        | (company_df["securityenddate"] >= "2000-01-01")
    ]
    company_df = company_df.drop(columns=[
        "countryid.1", "countryid", "officefaxvalue", "city", "exchangeid.1",
        "officephonevalue", "otherphonevalue", "streetaddress", "streetaddress2",
        "streetaddress3", "streetaddress4", "yearfounded", "monthfounded",
        "dayfounded", "zipcode", "securitysubtypeid", "primaryflag.1",
        "primaryflag", "currencyid.1", "currencyid", "importancelevel",
        "exchangename", "exchangeid", "companyid.1", "incorporationcountryid",
        "incorporationstateid", "reportingtemplatetypeid", "stateid",
        "securityid.1", "securityid", "securitystartdate", "webpage",
    ])
    return company_df.drop_duplicates(subset=["companyid"])


def create_intermed_folders(batch_size: int = 10) -> None:
    """Split the company universe into intermed/<n>/ batch folders."""
    company_df = load_company_df()
    for i in range(0, len(company_df), batch_size):
        batch = company_df.iloc[i:i + batch_size]
        folder = f"intermed/{i // batch_size}"
        os.makedirs(folder, exist_ok=True)
        batch.to_csv(f"{folder}/company_df.csv", index=False)
        with open(f"{folder}/companyids.txt", "w") as f:
            for companyid in batch["companyid"]:
                f.write(f"{companyid}\n")
        with open(f"{folder}/finished.txt", "w") as f:
            f.write("not finished")


def gen_date_windows(start_date: str, end_date: str) -> list[tuple[str, str]]:
    """Rolling ~3-month [start, end) windows between the two dates."""
    windows = []
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while start < end:
        windows.append((
            start.strftime("%Y-%m-%d"),
            (start + timedelta(days=89)).strftime("%Y-%m-%d"),
        ))
        start = start + timedelta(days=90)
    return windows


def search_fileids(
    headers: dict,
    companyids: list[int],
    min_filing_date: str,
    max_filing_date: str,
) -> pd.DataFrame | None:
    """Search the CIQ filings API; return the hits as a DataFrame (None if empty)."""
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
    response = requests.post(
        f"{BASE_URL}/search?docType=FILINGS_DOCUMENTS_API",
        headers=headers,
        json=search_payload,
    )
    if response.status_code in (400, 401):
        raise RuntimeError(f"Search API returned {response.status_code}: {response.text}")

    data = response.json()
    if not data.get("numRows"):
        logger.info("No documents found")
        return None
    logger.info("Found %d documents.", data["numRows"])

    columns = data.get("headers", [])
    rows = [row.get("row", []) for row in data.get("rows", [])]
    df = pd.DataFrame(rows, columns=columns).drop(columns=DROP_SEARCH_COLS)
    logger.info("\n%s", df)
    return df


def append_fileids_csv(df: pd.DataFrame, intermed: str) -> None:
    """Append new search hits to <intermed>/fileids.csv, de-duplicated."""
    csv_path = f"{intermed}/fileids.csv"
    if os.path.exists(csv_path):
        df = pd.concat([pd.read_csv(csv_path), df])
        df = df.drop_duplicates(
            subset=["filingId", "periodDate", "filingDate", "processedDate", "documentId"]
        )
    df.to_csv(csv_path, index=False)


def download_files(df: pd.DataFrame | None, headers: dict, files_dir: str) -> None:
    """Download each filing in df to <files_dir>/<filingId>.pdf (skip existing)."""
    if df is None:
        logger.info("No fileids given! Skipping download.")
        return

    for _, row in df.iterrows():
        filing_id = row["filingId"]
        filename = os.path.join(files_dir, f"{filing_id}.pdf")

        if os.path.exists(filename):
            logger.info("File %s already exists! Skipping download.", filing_id)
            continue

        response = requests.post(
            f"{BASE_URL}/download?docType=FILINGS_DOCUMENTS_API",
            headers=headers,
            json={"properties": {"filingId": filing_id}},
        )
        if response.status_code == 200:
            with open(filename, "wb") as f:
                f.write(response.content)
            logger.info("Successfully saved %s", filename)
        else:
            logger.info("Failed to download ID %s: %s", filing_id, response.status_code)


def _is_finished(intermed: str) -> bool:
    marker = f"{intermed}/finished.txt"
    if not os.path.exists(marker):
        return False
    with open(marker) as f:
        return f.read() == FINISHED_MARKER


def _load_companyids(intermed: str) -> list[int]:
    with open(f"{intermed}/companyids.txt") as f:
        return [int(line.strip()) for line in f if line.strip()]


def process_batch_folder(
    intermed: str,
    windows: list[tuple[str, str]],
    files_dir: str,
) -> None:
    """Search + download every date window for one intermed/<n>/ batch folder."""
    # refresh the token per batch folder: one folder's windows fit within a
    # token's lifetime, a whole run does not.
    access_token = quick_refresh_and_save_token(token_path=TOKEN_PATH)["access_token"]
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "accept": "application/json",
    }

    companyids = _load_companyids(intermed)

    for start, end in windows:
        logger.info("Processing from %s to %s", start, end)
        fileids_df = search_fileids(
            headers=headers,
            companyids=companyids,
            min_filing_date=start,
            max_filing_date=end,
        )
        if fileids_df is not None:
            append_fileids_csv(fileids_df, intermed)

        time.sleep(0.5)

        logger.info("Downloading files for %s to %s", start, end)
        download_files(fileids_df, headers=headers, files_dir=files_dir)

    with open(f"{intermed}/finished.txt", "w") as f:
        f.write(FINISHED_MARKER)

    fileids_csv = f"{intermed}/fileids.csv"
    if os.path.exists(fileids_csv):
        logger.info("Finished %s with %d fileids", intermed, len(pd.read_csv(fileids_csv)))
    else:
        logger.info("Finished %s with no fileids", intermed)
    logger.info("companyids: %s", companyids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download sustainability-report PDFs from S&P CIQ")
    parser.add_argument("--start-date", default="2015-01-01", help="Earliest filing date (YYYY-MM-DD)")
    parser.add_argument("--end-date",   default="2025-01-01", help="Latest filing date (YYYY-MM-DD)")
    parser.add_argument("--resume-from", type=int, default=0,
                        help="Skip the first N intermed/ batch folders")
    parser.add_argument("--files-dir", default="files", help="Destination folder for PDFs")
    parser.add_argument("--init-folders", action="store_true",
                        help="(Re)create the intermed/ batch folders from company.csv and exit")
    args = parser.parse_args()

    setup_logging()

    if args.init_folders:
        create_intermed_folders()
        return

    windows = gen_date_windows(args.start_date, args.end_date)
    os.makedirs(args.files_dir, exist_ok=True)

    batch_folders = sorted(glob.glob("intermed/*"), key=lambda x: int(x.split("/")[-1]))
    batch_folders = batch_folders[args.resume_from:]

    for intermed in batch_folders:
        logger.info("--------------------------------")
        logger.info("Processing %s", intermed)
        if _is_finished(intermed):
            logger.info("already finished!")
            continue
        process_batch_folder(intermed, windows, files_dir=args.files_dir)
        logger.info("--------------------------------")


if __name__ == "__main__":
    main()
