"""PanelBuilder: chunk-level flags + filing metadata → one row per company-year."""

from __future__ import annotations

import logging
import os

import pandas as pd

from carbontax.paths import panel_csv, parsed_csv
from carbontax.regression.outcomes import OUTCOMES, add_disclosure_flags, load_trucost
from carbontax.taxonomy import GOVERNANCE_FLAGS, MEASURE_IDS, TIER1_BUCKETS

logger = logging.getLogger(__name__)

# the 39 extracted booleans, in taxonomy order — the panel's regressors
FLAG_COLS: list[str] = (
    [f"tier1_{b}" for b in TIER1_BUCKETS]
    + [f"tier2_{m}" for m in MEASURE_IDS]
    + [f"governance_{g}" for g in GOVERNANCE_FLAGS]
)

# everything else in the parsed CSV is dropped: `chunks` is ~550MB of raw report text,
# and prompt_version/model/model_y/completion_tokens are single-valued across the run.
CHUNK_COLS: list[str] = ["filingId", "chunk_ids", "prompt_tokens"] + FLAG_COLS

# filingDate is when the report was published, periodDate the period it covers — they
# differ for 16% of filings, so which one dates the panel is a config choice.
MAPPING_COLS: list[str] = ["filingId", "companyid", "filingDate", "periodDate", "fileType", "noOfPages"]

CELL: list[str] = ["companyid", "year"]


class PanelBuilder:
    def __init__(self, run_name: str, section: dict, data: dict):
        self.run_name = run_name
        self.file_type = section["file_type"]
        self.year_from = section["year_from"]
        self.windows = section["windows"]
        self.trucost_year_from = section["trucost_year_from"]
        self.disclosed_max_score = section["disclosed_max_score"]
        self.mapping_csv = data["output"]["mapping_csv"]
        self.trucost_csv = data["input"]["trucost_csv"]

        if self.year_from not in ("periodDate", "filingDate"):
            raise ValueError(f"regression.panel.year_from must be periodDate or filingDate, got {self.year_from!r}")
        if not self.windows or any(w < 1 for w in self.windows):
            raise ValueError(f"regression.panel.windows must be positive year counts, got {self.windows!r}")
        if not 1.0 <= self.disclosed_max_score < 4.0:
            raise ValueError("regression.panel.disclosed_max_score must sit on the Trucost score scale "
                             f"[1.0, 4.0) — 4.0 would count Trucost's own model as firm data; got {self.disclosed_max_score!r}")
        for path in (parsed_csv(self.run_name), self.mapping_csv, self.trucost_csv):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Panel input not found: {path}")

    def run(self) -> pd.DataFrame:
        panel = self.build()
        dest = panel_csv(self.run_name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        panel.to_csv(dest, index=False)
        logger.info("Wrote %d rows across windows %s → %s", len(panel), self.windows, dest)
        return panel

    def build(self) -> pd.DataFrame:
        filings = self._load_filings()
        chunks = self._load_chunks()

        # inner join drops every chunk whose filing is not the kept genre
        df = chunks.merge(filings, on="filingId", how="inner")
        logger.info("Kept %d chunks across %d filings, %d companies",
                    len(df), df.filingId.nunique(), df.companyid.nunique())

        # long panel: the same company-year cells repeated once per pooling window
        panel = pd.concat([self._window_panel(df, w) for w in self.windows], ignore_index=True)
        return self._merge_outcomes(panel)

    def _merge_outcomes(self, panel: pd.DataFrame) -> pd.DataFrame:
        # every di_* item comes along, not just the y candidates: the regression stage
        # picks its outcome from OUTCOMES, but controls and robustness rows may want others
        tc = load_trucost(self.trucost_csv, self.trucost_year_from)
        merged = panel.merge(tc, on=["companyid", "year"], how="left")
        if len(merged) != len(panel):
            raise ValueError(f"Outcome merge changed the row count ({len(panel)} → {len(merged)}); "
                             "the Trucost frame is not unique on companyid-year")

        merged = add_disclosure_flags(merged, self.disclosed_max_score)
        logger.info("Outcome quality: score <= %s counts as the firm's own number", self.disclosed_max_score)
        for scope, cols in OUTCOMES.items():
            have = merged[cols["intensity"]].notna()
            disclosed = merged.loc[have, f"{scope}_disclosed"].mean()  # share OF THOSE, not of all cells
            logger.info("  %-4s %s: %5.1f%% of cells have an outcome, %4.1f%% of those are the firm's own number",
                        scope, cols["intensity"], 100 * have.mean(), 100 * disclosed)
        return merged

    @staticmethod
    def _anchors(df: pd.DataFrame) -> pd.DataFrame:
        # every year between a firm's first and last report, not only the years it filed:
        # in a gap year the firm still carries the stock of what it has already disclosed.
        # Bounded by the last report — past it there is nothing to pool, only extrapolation.
        span = df.groupby("companyid").year.agg(["min", "max"])
        years = [(cid, y) for cid, (lo, hi) in span.iterrows() for y in range(lo, hi + 1)]
        return pd.DataFrame(years, columns=CELL)

    def _window_panel(self, df: pd.DataFrame, window: int) -> pd.DataFrame:
        # the window only reaches backwards, so a cell never uses information the firm had
        # not yet published at its anchor year. Anchors whose window catches no report drop
        # out of the groupby below, so a gap year is filled only when there is content.
        anchors = self._anchors(df)
        src = df.rename(columns={"year": "src_year"})
        pooled = anchors.merge(src, on="companyid")
        pooled = pooled[(pooled.src_year <= pooled.year) & (pooled.src_year > pooled.year - window)]

        panel = pd.concat([self._filing_level(pooled), self._chunk_level(pooled)], axis=1)
        panel = panel.reset_index().sort_values(CELL, ignore_index=True)
        panel.insert(2, "window", window)
        logger.info("window=%d: %d cells, median %d chunks pooled",
                    window, len(panel), int(panel.n_chunks.median()))
        return panel

    def _load_filings(self) -> pd.DataFrame:
        mp = pd.read_csv(self.mapping_csv, usecols=MAPPING_COLS,
                         dtype={"companyid": "string", "filingId": "string"})
        # 29 filingIds appear twice in the mapping; keep one so the chunk join cannot fan out
        mp = mp.drop_duplicates("filingId")

        if self.file_type not in set(mp.fileType):
            raise ValueError(f"fileType {self.file_type!r} not in mapping; have {sorted(set(mp.fileType))}")
        mp = mp[mp.fileType == self.file_type].copy()

        mp["year"] = pd.to_datetime(mp[self.year_from]).dt.year
        logger.info("Mapping: %d %s filings", len(mp), self.file_type)
        return mp[["filingId", "companyid", "year", "noOfPages"]]

    def _load_chunks(self) -> pd.DataFrame:
        path = parsed_csv(self.run_name)
        header = pd.read_csv(path, nrows=0).columns
        missing = [c for c in CHUNK_COLS if c not in header]
        if missing:
            raise KeyError(f"{path} is missing expected columns: {missing}")

        logger.info("Reading %s (%d of %d columns)", path, len(CHUNK_COLS), len(header))
        return pd.read_csv(path, usecols=CHUNK_COLS,
                           dtype={"filingId": "string", "chunk_ids": "string"})

    @staticmethod
    def _filing_level(df: pd.DataFrame) -> pd.DataFrame:
        # noOfPages is a filing attribute, so it must be summed over filings — aggregating
        # it off the chunk frame would count each report once per chunk. Dedupe within the
        # cell, not globally: a window>1 cell legitimately reuses a filing across anchors.
        fil = df[CELL + ["filingId", "src_year", "noOfPages"]].drop_duplicates(CELL + ["filingId"])
        fil = fil.assign(is_anchor=fil.src_year == fil.year)
        g = fil.groupby(CELL)
        return pd.DataFrame({
            "n_filings": g.filingId.size(),
            # how much of the window the firm actually covered — 1 for window=1, and less
            # than the window whenever the firm skipped a year or sits at the panel's left edge
            "n_years": g.src_year.nunique(),
            # False = a gap year carried entirely by earlier reports (window>1 only)
            "filed_this_year": g.is_anchor.any(),
            "total_pages": g.noOfPages.sum(min_count=1),  # NaN when no filing reports pages
            "pages_missing": g.noOfPages.apply(lambda s: s.isna().any()),
        })

    @staticmethod
    def _chunk_level(df: pd.DataFrame) -> pd.DataFrame:
        g = df.groupby(CELL)
        counts = pd.DataFrame({"n_chunks": g.chunk_ids.size(), "chunk_tokens": g.prompt_tokens.sum()})
        # _any = the dummy (measure appears anywhere in the firm-year), _share = fraction of
        # chunks carrying it. Both are emitted so the spec choice happens at estimation time.
        anys = g[FLAG_COLS].any().astype("int8").add_suffix("_any")
        shares = g[FLAG_COLS].mean().astype("float32").add_suffix("_share")
        return pd.concat([counts, anys, shares], axis=1)
