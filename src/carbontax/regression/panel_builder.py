"""PanelBuilder: chunk-level flags + filing metadata → one row per company-year."""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from carbontax.paths import combined_ref, panel_csv, parsed_csv
from carbontax.regression.outcomes import OUTCOMES, TOTAL_REVENUE, add_disclosure_flags, load_trucost
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
        self.trucost_join_tolerance_days = section["trucost_join_tolerance_days"]
        self.disclosed_max_score = section["disclosed_max_score"]
        self.share_denominator = section["share_denominator"]
        self.mapping_csv = data["output"]["mapping_csv"]
        self.trucost_csv = data["input"]["trucost_csv"]

        if self.year_from not in ("periodDate", "filingDate"):
            raise ValueError(f"regression.panel.year_from must be periodDate or filingDate, got {self.year_from!r}")
        if not self.windows or any(w < 1 for w in self.windows):
            raise ValueError(f"regression.panel.windows must be positive year counts, got {self.windows!r}")
        if not 1.0 <= self.disclosed_max_score < 4.0:
            raise ValueError("regression.panel.disclosed_max_score must sit on the Trucost score scale "
                             f"[1.0, 4.0) — 4.0 would count Trucost's own model as firm data; got {self.disclosed_max_score!r}")
        if self.share_denominator not in ("all_chunks", "keyword_chunks"):
            raise ValueError("regression.panel.share_denominator must be all_chunks or keyword_chunks, "
                             f"got {self.share_denominator!r}")
        if self.trucost_join_tolerance_days < 1:
            raise ValueError("regression.panel.trucost_join_tolerance_days must be a positive day count, "
                             f"got {self.trucost_join_tolerance_days!r}")
        for path in (parsed_csv(self.run_name), self.mapping_csv, self.trucost_csv, combined_ref(self.run_name)):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Panel input not found: {path}")
        # n_chunks_total is written by stage 2; a ref parquet predating that change lacks the
        # column and every _share would come out NaN. Schema only — the file is ~230MB.
        ref_cols = pq.read_schema(combined_ref(self.run_name)).names
        if "n_chunks_total" not in ref_cols:
            raise KeyError(f"{combined_ref(self.run_name)} has no n_chunks_total column — re-run "
                           "stage 2 chunking (BatchInputPreparer.chunk_filings) to record it")

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

        # every surviving filing has chunks, so it must appear in the ref parquet. If it does
        # not, its total is NaN and _filing_level's sum would quietly return a short denominator
        # for the whole cell rather than failing.
        orphans = df.loc[df.n_chunks_total.isna(), "filingId"].unique()
        if len(orphans):
            raise ValueError(f"{len(orphans)} filings have chunks but no n_chunks_total in "
                             f"{combined_ref(self.run_name)} (e.g. {list(orphans[:5])}) — the ref "
                             "parquet is stale relative to the parsed CSV; re-run stage 2 chunking")

        # whether a firm's years sit in Trucost's fiscal-year space at all — gap-year cells have
        # no filing of their own to vouch for their year label, so they fall back to this
        self.firm_year_verified = df.groupby("companyid").trucost_matched.any()

        # long panel: the same company-year cells repeated once per pooling window
        panel = pd.concat([self._window_panel(df, w) for w in self.windows], ignore_index=True)
        return self._merge_outcomes(panel)

    @staticmethod
    def _add_previous_year(merged: pd.DataFrame, tc: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Attach year t-1's outcome as <col>_prev, so the regression can form ln(y_t / y_t-1).

        Taken off the Trucost frame, which is near-continuous (12.9 years per firm, 91% of cells
        have a real t-1), not off the panel, which only has years the firm filed a report. Only
        the y candidates and revenue get a lag — the other ~290 di_* items would triple the file.
        """
        base = sorted({c for cols in OUTCOMES.values() for k, c in cols.items() if k != "score"}
                      | {TOTAL_REVENUE})
        prev = tc[["companyid", "year"] + base].copy()
        # t-1 by explicit year arithmetic, never a positional shift: Trucost has gaps, and a shift
        # would silently pass off t-2 (or t-5) as the previous year
        prev["year"] = prev["year"] + 1
        prev = prev.rename(columns={c: f"{c}_prev" for c in base})

        n = len(merged)
        merged = merged.merge(prev, on=["companyid", "year"], how="left")
        if len(merged) != n:
            raise ValueError(f"Previous-year merge changed the row count ({n} → {len(merged)})")
        return merged, [f"{c}_prev" for c in base]

    def _outcome_trusted(self, panel: pd.DataFrame) -> pd.Series:
        """Is this cell's year founded on a report that really sits inside the Trucost period?

        A cell that filed in its anchor year is vouched for by that filing. A gap-year cell has no
        filing to vouch for it, so it falls back to the firm: if the firm's reports line up with
        Trucost periods elsewhere, its calendar and Trucost's agree and the enumerated year is safe.
        """
        filed = panel.filed_this_year.astype(bool)
        firm_ok = panel.companyid.map(self.firm_year_verified).fillna(False).astype(bool)
        return (filed & panel.anchor_matched.astype(bool)) | (~filed & firm_ok)

    def _merge_outcomes(self, panel: pd.DataFrame) -> pd.DataFrame:
        # every di_* item comes along, not just the y candidates: the regression stage
        # picks its outcome from OUTCOMES, but controls and robustness rows may want others
        # fiscalyear, always: a filing's year IS the matched Trucost row's fiscalyear, so keying
        # the outcome frame any other way would re-open the mismatch the date join closes
        tc = load_trucost(self.trucost_csv, "fiscalyear")
        outcome_cols = [c for c in tc.columns if c not in ("companyid", "year")]
        merged = panel.merge(tc, on=["companyid", "year"], how="left")
        if len(merged) != len(panel):
            raise ValueError(f"Outcome merge changed the row count ({len(panel)} → {len(merged)}); "
                             "the Trucost frame is not unique on companyid-year")

        merged, prev_cols = self._add_previous_year(merged, tc)
        outcome_cols = outcome_cols + prev_cols

        # keep only outcomes whose year a real report period vouches for. The merge above is on
        # (companyid, year), so it still reaches a Trucost row for cells the date join rejected —
        # a period months away from the report — which is the pairing the date join exists to
        # refuse. A missing outcome drops the cell; a wrong one biases the coefficient toward zero.
        merged["outcome_trusted"] = self._outcome_trusted(merged)
        untrusted = ~merged.outcome_trusted
        lost = merged.loc[untrusted, OUTCOMES["S1"]["intensity"]].notna().sum()
        merged.loc[untrusted, outcome_cols] = np.nan
        logger.info("%d cells have no matched report period vouching for their year; %d of them "
                    "would otherwise have carried an S1 outcome", int(untrusted.sum()), int(lost))

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
        panel = self._add_shares(panel)
        panel = panel.reset_index().sort_values(CELL, ignore_index=True)
        panel.insert(2, "window", window)
        logger.info("window=%d: %d cells, median %d chunks pooled (of %d split), _share over %s",
                    window, len(panel), int(panel.n_chunks.median()),
                    int(panel.n_chunks_total.median()), self.share_denominator)
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
        mp = self._adopt_trucost_year(mp)
        logger.info("Mapping: %d %s filings", len(mp), self.file_type)
        return mp[["filingId", "companyid", "year", "noOfPages", "trucost_matched"]].merge(
            self._load_chunk_totals(), on="filingId", how="left")

    def _adopt_trucost_year(self, mp: pd.DataFrame) -> pd.DataFrame:
        """Year a filing by the Trucost period its report period actually falls in, not by a rule.

        Labelling each side separately and joining on the label only works if the two rules agree;
        where they disagree the outcome lands a year off, which a lag spec cannot distinguish from
        the lag itself. Pairing on the date and taking the matched row's own fiscalyear removes the
        second rule entirely — the later merge on (companyid, year) is then exact by construction.
        """
        tc = pd.read_csv(self.trucost_csv, dtype={"companyid": "string"}, low_memory=False,
                         usecols=["companyid", "periodenddate", "fiscalyear"])
        tc["periodenddate"] = pd.to_datetime(tc["periodenddate"], errors="coerce")
        tc = tc.dropna(subset=["periodenddate", "fiscalyear"])
        # restatements repeat a (companyid, periodenddate); merge_asof would take an arbitrary one
        tc = tc.drop_duplicates(["companyid", "periodenddate"]).sort_values("periodenddate")

        left = mp.assign(_period=pd.to_datetime(mp[self.year_from])).sort_values("_period")
        m = pd.merge_asof(left, tc, left_on="_period", right_on="periodenddate", by="companyid",
                          direction="nearest",
                          tolerance=pd.Timedelta(days=self.trucost_join_tolerance_days))

        matched = m["fiscalyear"].notna()
        shifted = (m.loc[matched, "fiscalyear"].astype("int64") != m.loc[matched, "year"]).sum()
        logger.info("Trucost date join (±%dd): %d of %d filings matched a period (%.1f%%), "
                    "%d of those move to a different year than %s implies",
                    self.trucost_join_tolerance_days, matched.sum(), len(m),
                    100 * matched.mean(), shifted, self.year_from)
        # an unmatched filing keeps its own year so its chunks still pool into windows; it simply
        # carries no outcome, which is already true — nothing sits within tolerance of it
        year = m["fiscalyear"].fillna(m["year"])
        if year.isna().any():
            raise ValueError(f"{year.isna().sum()} filings have neither a matched Trucost period nor "
                             f"a parseable {self.year_from}")
        m["year"] = year.astype("int64")
        m["trucost_matched"] = matched.to_numpy()
        return m.drop(columns=["_period", "periodenddate", "fiscalyear"])

    def _load_chunk_totals(self) -> pd.DataFrame:
        # filingId → how many chunks the report split into before the keyword filter. Constant
        # within a filing in the ref parquet, so first() is the value, not an aggregate.
        ref = pd.read_parquet(combined_ref(self.run_name), columns=["filingId", "n_chunks_total"])
        totals = ref.groupby("filingId", as_index=False).n_chunks_total.first()
        totals["filingId"] = totals.filingId.astype("string")  # mapping keys filingId as string
        return totals

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
        fil = df[CELL + ["filingId", "src_year", "noOfPages", "n_chunks_total", "trucost_matched"]] \
            .drop_duplicates(CELL + ["filingId"])
        fil = fil.assign(is_anchor=fil.src_year == fil.year)
        # did a report *in the anchor year* pair with a real Trucost period? Only an anchor-year
        # filing can vouch for the cell's year label; an earlier pooled report dates a different one
        fil = fil.assign(anchor_match=fil.is_anchor & fil.trucost_matched.astype(bool))
        g = fil.groupby(CELL)
        return pd.DataFrame({
            "n_filings": g.filingId.size(),
            "anchor_matched": g.anchor_match.any(),
            # every chunk the window's reports split into, keyword-matched or not — the
            # all_chunks denominator. Summed over filings for the same reason as total_pages.
            "n_chunks_total": g.n_chunks_total.sum(min_count=1),
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
        # _any = the dummy (measure appears anywhere in the firm-year). _flagged is the raw
        # count of chunks carrying the measure; _add_shares turns it into _share once the
        # denominator is known, since that lives on the filing-level frame.
        anys = g[FLAG_COLS].any().astype("int8").add_suffix("_any")
        flagged = g[FLAG_COLS].sum().astype("int32").add_suffix("_flagged")
        return pd.concat([counts, anys, flagged], axis=1)

    def _add_shares(self, panel: pd.DataFrame) -> pd.DataFrame:
        """_flagged counts → _share, on whichever denominator the config names."""
        # keyword_chunks divides by the chunks the LLM actually read. That denominator grows with
        # the treatment — filter_keywords contains "renewable", "solar", "energy efficiency", so a
        # firm doing more of those matches more chunks and every _share is deflated for exactly
        # the firms doing the most. all_chunks divides by the whole report instead, which breaks
        # that link. n_chunks (kept chunks) stays the exposure control either way: a flag can
        # only fire in a chunk that was actually read.
        denom = panel.n_chunks if self.share_denominator == "keyword_chunks" else panel.n_chunks_total
        if denom.isna().any() or (denom <= 0).any():
            raise ValueError(f"share_denominator={self.share_denominator} produced {denom.isna().sum()} "
                             f"missing and {(denom <= 0).sum()} non-positive denominators")
        if (panel.n_chunks > panel.n_chunks_total).any():
            raise ValueError("some cells have more kept chunks than total chunks — the ref parquet "
                             "and the parsed CSV disagree about which chunks exist")
        flagged = panel[[f"{f}_flagged" for f in FLAG_COLS]]
        shares = flagged.div(denom, axis=0).astype("float32")
        shares.columns = [f"{f}_share" for f in FLAG_COLS]
        return pd.concat([panel.drop(columns=list(flagged.columns)), shares], axis=1)
