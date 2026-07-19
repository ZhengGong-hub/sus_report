"""PdfAcquirer: bulk-download sustainability-report PDFs from S&P CIQ into files/."""

import glob
import logging
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

from carbontax.acquire_pdfs.tokens import quick_refresh_and_save_token
from carbontax.config import load_run_config, stage_section
from carbontax.utils.logger import setup_logging

logger = logging.getLogger(__name__)

BASE_URL = "https://api-ciq.marketintelligence.spglobal.com/gds/documents/api/v1"
TOKEN_PATH = "tokens/token_current.json"
FINISHED_MARKER = "already finished!"

# file-type reference: https://help.edgar-online.com/edgar/formtypes.asp
INCL_FILETYPEID = [
    432,   # Sustainability Report
    1010,  # Corporate Social Responsibility Report
    1091,  # Environmental Report
    1090,  # Corporate Governance Report
    1105,  # TCFD Report
]

# search-result columns we never use downstream
DROP_SEARCH_COLS = [
    "filingVersionId", "language", "languageId", "documentVersionId",
    "supplierDocumentId", "institutionId",
]


class PdfAcquirer:
    """The company universe is pre-split into <intermed_dir>/<n>/ folders of 10
    companies each (see create_intermed_folders); every folder is searched over
    rolling ~3-month windows and marked done via finished.txt, so the loop is
    resumable."""

    def __init__(self, section: dict, data: dict):
        # all knobs come explicitly from config/run.yaml — a missing key fails loudly
        self.start_date = section["start_date"]
        self.end_date = section["end_date"]
        self.resume_from = section["resume_from"]
        self.pdfs_dir = data["output"]["pdfs_dir"]
        self.intermed_dir = data["output"]["intermed_dir"]
        self.headers: dict = {}  # set per batch folder by _refresh_token

    def run(self) -> None:
        windows = self._gen_date_windows()
        os.makedirs(self.pdfs_dir, exist_ok=True)

        batch_folders = sorted(glob.glob(f"{self.intermed_dir}/*"),
                               key=lambda x: int(x.split("/")[-1]))
        batch_folders = batch_folders[self.resume_from:]

        for intermed in batch_folders:
            logger.info("--------------------------------")
            logger.info("Processing %s", intermed)
            if self._is_finished(intermed):
                logger.info("already finished!")
                continue
            self._process_batch_folder(intermed, windows)
            logger.info("--------------------------------")

    # ── per-batch-folder loop ─────────────────────────────────────────────────

    def _process_batch_folder(self, intermed: str, windows: list[tuple[str, str]]) -> None:
        # refresh the token per batch folder: one folder's windows fit within a
        # token's lifetime, a whole run does not
        self._refresh_token()
        companyids = self._load_companyids(intermed)

        for start, end in windows:
            logger.info("Processing from %s to %s", start, end)
            fileids_df = self._search_fileids(companyids, start, end)
            if fileids_df is not None:
                self._append_fileids_csv(fileids_df, intermed)

            time.sleep(0.5)  # stay polite to the API

            logger.info("Downloading files for %s to %s", start, end)
            self._download_files(fileids_df)

        with open(f"{intermed}/finished.txt", "w") as f:
            f.write(FINISHED_MARKER)

        fileids_csv = f"{intermed}/fileids.csv"
        if os.path.exists(fileids_csv):
            logger.info("Finished %s with %d fileids", intermed, len(pd.read_csv(fileids_csv)))
        else:
            logger.info("Finished %s with no fileids", intermed)
        logger.info("companyids: %s", companyids)

    def _refresh_token(self) -> None:
        access_token = quick_refresh_and_save_token(token_path=TOKEN_PATH)["access_token"]
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

    def _gen_date_windows(self) -> list[tuple[str, str]]:
        # rolling ~3-month [start, end) windows between start_date and end_date
        windows = []
        start = datetime.strptime(self.start_date, "%Y-%m-%d")
        end = datetime.strptime(self.end_date, "%Y-%m-%d")
        while start < end:
            windows.append((
                start.strftime("%Y-%m-%d"),
                (start + timedelta(days=89)).strftime("%Y-%m-%d"),
            ))
            start = start + timedelta(days=90)
        return windows

    @staticmethod
    def _is_finished(intermed: str) -> bool:
        marker = f"{intermed}/finished.txt"
        if not os.path.exists(marker):
            return False
        with open(marker) as f:
            return f.read() == FINISHED_MARKER

    @staticmethod
    def _load_companyids(intermed: str) -> list[int]:
        with open(f"{intermed}/companyids.txt") as f:
            return [int(line.strip()) for line in f if line.strip()]

    # ── CIQ API calls ─────────────────────────────────────────────────────────

    def _search_fileids(self, companyids: list[int], min_filing_date: str,
                        max_filing_date: str) -> pd.DataFrame | None:
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
            headers=self.headers,
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

    @staticmethod
    def _append_fileids_csv(df: pd.DataFrame, intermed: str) -> None:
        # accumulate search hits per batch folder, de-duplicated
        csv_path = f"{intermed}/fileids.csv"
        if os.path.exists(csv_path):
            df = pd.concat([pd.read_csv(csv_path), df])
            df = df.drop_duplicates(
                subset=["filingId", "periodDate", "filingDate", "processedDate", "documentId"]
            )
        df.to_csv(csv_path, index=False)

    def _download_files(self, df: pd.DataFrame | None) -> None:
        if df is None:
            logger.info("No fileids given! Skipping download.")
            return

        for _, row in df.iterrows():
            filing_id = row["filingId"]
            filename = os.path.join(self.pdfs_dir, f"{filing_id}.pdf")

            if os.path.exists(filename):
                logger.info("File %s already exists! Skipping download.", filing_id)
                continue

            response = requests.post(
                f"{BASE_URL}/download?docType=FILINGS_DOCUMENTS_API",
                headers=self.headers,
                json={"properties": {"filingId": filing_id}},
            )
            if response.status_code == 200:
                with open(filename, "wb") as f:
                    f.write(response.content)
                logger.info("Successfully saved %s", filename)
            else:
                logger.info("Failed to download ID %s: %s", filing_id, response.status_code)


# ── one-time setup: split the company universe into <intermed_dir>/<n>/ ───────

def load_company_df(company_csv: str) -> pd.DataFrame:
    """US-listed operating companies from the CIQ company export, one row per companyid."""
    company_df = (
        pd.read_csv(company_csv)
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


def create_intermed_folders(company_csv: str, intermed_dir: str, batch_size: int = 10) -> None:
    company_df = load_company_df(company_csv)
    for i in range(0, len(company_df), batch_size):
        batch = company_df.iloc[i:i + batch_size]
        folder = f"{intermed_dir}/{i // batch_size}"
        os.makedirs(folder, exist_ok=True)
        batch.to_csv(f"{folder}/company_df.csv", index=False)
        with open(f"{folder}/companyids.txt", "w") as f:
            for companyid in batch["companyid"]:
                f.write(f"{companyid}\n")
        with open(f"{folder}/finished.txt", "w") as f:
            f.write("not finished")


def main() -> None:
    setup_logging()
    cfg = load_run_config()
    section = stage_section(cfg, "acquire_pdfs")
    data = cfg["data"]
    if section["init_folders"]:
        # one-time setup; flip init_folders back to false afterwards
        create_intermed_folders(data["input"]["company_csv"], data["output"]["intermed_dir"])
        return
    PdfAcquirer(section, data).run()


if __name__ == "__main__":
    main()
