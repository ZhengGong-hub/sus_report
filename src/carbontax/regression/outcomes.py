"""Trucost outcome variables: which WRDS data items are the y candidates, and the merge-ready loader."""

from __future__ import annotations

import logging

import pandas as pd

from carbontax.regression.trucost_vars import TRUCOST_VARS

logger = logging.getLogger(__name__)

# Column names below are the readable ones from TRUCOST_VARS, not the raw di_* codes: the
# loader renames every data item on the way in. Semantics verified against the data:
#   absolute  — tCO2e
#   intensity — tCO2e per $M of TOTAL_REVENUE (absolute / revenue, exactly)
#   score     — disclosure quality, see the scale below
TOTAL_REVENUE = "trucost_total_revenue"

OUTCOMES: dict[str, dict[str, str]] = {
    "S1":  {"absolute": "absolute_ghg_scope_1",
            "intensity": "intensity_ghg_scope_1",
            "score": "ghg_scope_1_score"},
    "S2L": {"absolute": "absolute_ghg_scope_2_location_based",
            "intensity": "intensity_ghg_scope_2_location_based",
            "score": "ghg_scope_2_location_based_score"},
    "S2M": {"absolute": "absolute_ghg_scope_2_market_based",
            "intensity": "intensity_ghg_scope_2_market_based",
            "score": "ghg_scope_2_market_based_score"},
    "S3U": {"absolute": "absolute_ghg_scope_3_upstream_total",
            "intensity": "intensity_ghg_scope_3_upstream",
            "score": "ghg_scope_3_upstream_score"},
    "S3D": {"absolute": "absolute_ghg_scope_3_downstream_total",
            "intensity": "intensity_ghg_scope_3_downstream_total",
            "score": "ghg_scope_3_downstream_total_score"},
}

# Score → did this number come from the firm, or did Trucost supply it?
#   1.0, 2.0  the firm's own disclosure (CDP, CSR/environmental report, annual report)
#   2.4       partial disclosure or extrapolated from the prior year — still firm-anchored
#   3.0       Trucost model from PHYSICAL intensity factors (S3 downstream only, 352 rows)
#   4.0       Trucost model from REVENUE intensity factors, or the firm's disclosure rejected
# The WRDS dictionary documents no scale for these — it was read off the score × disclosure-
# text crosstab, in which every text category falls in exactly one score level.
# Where to cut is a research choice, so the threshold lives in the run config, not here.

# firm attributes carried alongside the outcomes. Firm FE absorbs all of them in the main
# spec; they are here for sample description, heterogeneity splits and sector×year variants.
META_COLS: list[str] = ["periodenddate", "fiscalyear", "gvkey", "ticker", "companyname",
                        "country", "simpleindustry", "tcprimarysectorid", "reportedcurrencyisocode"]


def load_trucost(csv_path: str, year_from: str) -> pd.DataFrame:
    """Trucost environment CSV → one row per companyid-year, all di_* data items kept."""
    if year_from not in ("fiscalyear", "periodenddate"):
        raise ValueError(f"trucost_year_from must be fiscalyear or periodenddate, got {year_from!r}")

    tc = pd.read_csv(csv_path, dtype={"companyid": "string"}, low_memory=False)
    data_cols = [c for c in tc.columns if c.startswith("di_")]
    if not data_cols:
        raise KeyError(f"{csv_path} has no di_* data columns")
    # an unmapped code means WRDS added an item — refresh TRUCOST_VARS from the data
    # dictionary rather than letting a bare di_* code through to the panel
    unmapped = [c for c in data_cols if c not in TRUCOST_VARS]
    if unmapped:
        raise KeyError(f"{len(unmapped)} Trucost items are missing from TRUCOST_VARS: {unmapped[:5]}")

    # fiscalyear is Trucost's own period assignment and is correct for non-December year
    # ends (a period ending 2021-01-01 is FY2020); the calendar year of periodenddate
    # misdates those and collapses two adjacent fiscal years into one cell.
    tc["year"] = (tc["fiscalyear"] if year_from == "fiscalyear"
                  else pd.to_datetime(tc["periodenddate"]).dt.year)
    tc = tc[tc.year.notna()].copy()
    tc["year"] = tc["year"].astype("int64")

    # some companyid-year cells appear twice — a restatement, or one full row plus a mostly
    # empty duplicate. Keep the most complete row, latest period end breaking ties.
    tc["_filled"] = tc[data_cols].notna().sum(axis=1)
    tc = tc.sort_values(["_filled", "periodenddate"], ascending=False)
    before = len(tc)
    tc = tc.drop_duplicates(["companyid", "year"]).drop(columns="_filled")
    logger.info("Trucost: %d rows → %d companyid-year cells (%d duplicates dropped), keyed on %s",
                before, len(tc), before - len(tc), year_from)

    meta = [c for c in META_COLS if c in tc.columns]
    out = tc[["companyid", "year"] + meta + data_cols].rename(columns=TRUCOST_VARS)

    wanted = [TOTAL_REVENUE] + [c for cols in OUTCOMES.values() for c in cols.values()]
    absent = [c for c in wanted if c not in out.columns]
    if absent:
        raise KeyError(f"OUTCOMES/TOTAL_REVENUE name columns absent after rename: {absent}")
    return out


def add_disclosure_flags(panel: pd.DataFrame, max_score: float) -> pd.DataFrame:
    """Add <scope>_disclosed booleans. The raw score columns stay — this only names the rule."""
    for scope, cols in OUTCOMES.items():
        # a missing score counts as not-disclosed: provenance unknown is not firm-verified
        panel[f"{scope}_disclosed"] = panel[cols["score"]].le(max_score).fillna(False)
    return panel
