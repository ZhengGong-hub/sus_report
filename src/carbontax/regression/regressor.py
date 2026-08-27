"""Regressor: company-year panel → a spec grid of two-way FE regressions, as one tidy table."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pandas as pd

from carbontax.paths import panel_csv, regression_results_csv, regression_summary_md
from carbontax.regression.estimator import count_switchers, fe_ols
from carbontax.regression.outcomes import OUTCOME_TAXONOMY_SCOPE, OUTCOMES, TOTAL_REVENUE
from carbontax.taxonomy import GOVERNANCE_FLAGS, MEASURE_SCOPE, TIER1_BUCKETS

logger = logging.getLogger(__name__)

CELL = ["companyid", "year"]
FE_COLS = ["companyid", "year"]  # firm FE + year FE

# exposure controls. log_n_chunks matters because a flag can only fire in a chunk we
# sampled; n_years because a pooled window covers more history for some cells than others.
CONTROL_BUILDERS = {
    "log_n_chunks": lambda d: np.log(d.n_chunks),
    "log_revenue": lambda d: np.log(d[TOTAL_REVENUE].where(d[TOTAL_REVENUE] > 0)),
    # revenue in growth units, for the *_growth outcomes — same transform as the outcome
    "log_revenue_growth": lambda d: (np.log(d[TOTAL_REVENUE].where(d[TOTAL_REVENUE] > 0))
                                     - np.log(d[f"{TOTAL_REVENUE}_prev"].where(d[f"{TOTAL_REVENUE}_prev"] > 0))),
    "n_years": lambda d: d.n_years.astype(float),
    "n_filings": lambda d: d.n_filings.astype(float),
    "filed_this_year": lambda d: d.filed_this_year.astype(float),
}


def regressor_set(name: str, scope: str) -> list[str]:
    """Which flags go on the right-hand side for a given outcome scope."""
    taxonomy_scope = OUTCOME_TAXONOMY_SCOPE[scope]
    matched = [m for m, s in MEASURE_SCOPE.items() if s == taxonomy_scope]
    governance = list(GOVERNANCE_FLAGS)
    sets = {
        "scope_matched": [f"tier2_{m}" for m in matched],
        "scope_matched_plus_gov": [f"tier2_{m}" for m in matched] + [f"governance_{g}" for g in governance],
        "all_tier2": [f"tier2_{m}" for m in MEASURE_SCOPE],
        "tier1": [f"tier1_{b}" for b in TIER1_BUCKETS],          # all five buckets together
        "tier1_matched": [f"tier1_{taxonomy_scope}"],            # only the bucket matching the outcome
        "governance": [f"governance_{g}" for g in governance],
    }
    if name not in sets:
        raise ValueError(f"unknown regressor_set {name!r}; have {sorted(sets)}")
    return sets[name]


def _markdown(wide: pd.DataFrame) -> str:
    # hand-rolled so the package does not need tabulate just to print one table
    header = [str(wide.index.name or "term")] + [f"lag {c}" for c in wide.columns]
    rows = [[str(i)] + ["" if pd.isna(v) else str(v) for v in row] for i, row in zip(wide.index, wide.to_numpy())]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


class Regressor:
    def __init__(self, run_name: str, section: dict):
        self.run_name = run_name
        s = section
        self.scopes = s["scopes"]
        self.outcome_forms = s["outcome_forms"]
        self.windows = s["windows"]
        self.regressor_forms = s["regressor_forms"]
        self.regressor_sets = s["regressor_sets"]
        self.lags = s["lags"]
        self.disclosed_only = s["disclosed_only"]
        self.controls = s["controls"]
        self.min_switchers = s["min_switchers"]
        self.cluster = s["cluster"]

        unknown = [c for c in self.controls if c not in CONTROL_BUILDERS]
        if unknown:
            raise ValueError(f"unknown controls {unknown}; have {sorted(CONTROL_BUILDERS)}")
        for scope in self.scopes:
            if scope not in OUTCOMES:
                raise ValueError(f"unknown scope {scope!r}; have {sorted(OUTCOMES)}")
        legal_forms = ("intensity", "absolute", "intensity_growth", "absolute_growth")
        for form in self.outcome_forms:
            if form not in legal_forms:
                raise ValueError(f"outcome_forms must be one of {legal_forms}, got {form!r}")

    def run(self) -> pd.DataFrame:
        # CSV carries no dtypes: companyid is an all-digit id that would otherwise come back
        # as int64 and no longer match the string ids everything upstream is keyed on
        panel = pd.read_csv(panel_csv(self.run_name), dtype={"companyid": "string"}, low_memory=False)
        logger.info("Panel: %d rows, %d companies", len(panel), panel.companyid.nunique())

        results = [r for spec in self._grid() for r in self._estimate(panel, spec)]
        table = pd.DataFrame(results)

        dest = regression_results_csv(self.run_name)
        table.to_csv(dest, index=False)
        logger.info("Wrote %d coefficient rows over %d specs → %s",
                    len(table), table.spec_id.nunique() if len(table) else 0, dest)
        self._write_summary(table)
        return table

    def _grid(self) -> list[dict]:
        keys = ["scope", "outcome_form", "window", "regressor_form", "regressor_set", "lag", "disclosed"]
        values = [self.scopes, self.outcome_forms, self.windows, self.regressor_forms,
                  self.regressor_sets, self.lags, self.disclosed_only]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _estimate(self, panel: pd.DataFrame, spec: dict) -> list[dict]:
        cols = OUTCOMES[spec["scope"]]
        form = spec["outcome_form"]
        growth = form.endswith("_growth")
        y_col = cols[form[: -len("_growth")] if growth else form]

        d = panel[panel.window == spec["window"]].copy()
        if spec["disclosed"]:
            d = d[d[f"{spec['scope']}_disclosed"]]
        # logs need a strictly positive outcome; Trucost writes exact zeros for
        # not-applicable categories, which are absences rather than measurements
        if growth:
            # ln(y_t / y_t-1). Both years must be positive, so a growth spec is a strictly
            # smaller sample than the matching level spec — and a firm-year with no t-1 in
            # Trucost drops out rather than borrowing a further-back year.
            prev = f"{y_col}_prev"
            d = d[(d[y_col] > 0) & (d[prev] > 0)]
            d["_y"] = np.log(d[y_col]) - np.log(d[prev])
        else:
            d = d[d[y_col] > 0]
            d["_y"] = np.log(d[y_col])

        flags = [f"{f}_{spec['regressor_form']}" for f in regressor_set(spec["regressor_set"], spec["scope"])]
        flags = [f for f in flags if f in d.columns]
        d = self._apply_lag(d, flags, spec["lag"])
        if d.empty:
            return []

        controls = list(self.controls)
        # ln(intensity) already divides by revenue; with an absolute outcome the revenue
        # term has to enter explicitly rather than being forced to a coefficient of -1.
        # A growth outcome needs the control in the same units, so revenue enters as its
        # own log change rather than its level.
        if form == "absolute" and "log_revenue" not in controls:
            controls = controls + ["log_revenue"]
        elif form == "absolute_growth" and "log_revenue_growth" not in controls:
            controls = controls + ["log_revenue_growth"]
        for c in controls:
            d[c] = CONTROL_BUILDERS[c](d)

        rhs = flags + controls
        d = d.dropna(subset=["_y"] + rhs)
        # a within estimator only learns from firms observed more than once
        d = d[d.groupby("companyid").companyid.transform("size") > 1]
        if len(d) < 50 or d.companyid.nunique() < 10:
            return []

        # a dummy that never changes within any firm is absorbed by the firm FE; with only a
        # handful of switchers the coefficient is identified off too few firms to mean anything
        switchers = {f: count_switchers(d, f, "companyid") for f in flags}
        kept = [f for f in flags if switchers[f] >= self.min_switchers]
        if not kept:
            return []
        rhs = kept + [c for c in controls if d[c].std() > 0]

        terms, diag = fe_ols(d, "_y", rhs, FE_COLS, self.cluster)
        spec_id = "|".join(f"{k}={spec[k]}" for k in spec)
        return [{"spec_id": spec_id, **spec, "y": y_col,
                 "n_dropped_flags": len(flags) - len(kept),
                 **row, "switchers": switchers.get(row["term"], np.nan), **diag}
                for row in terms.to_dict("records")]

    @staticmethod
    def _apply_lag(d: pd.DataFrame, flags: list[str], lag: int) -> pd.DataFrame:
        if lag == 0:
            return d
        # match on year explicitly rather than shifting rows: the panel has gaps, and a
        # positional shift would silently pair non-adjacent years
        past = d[CELL + flags].copy()
        past["year"] = past["year"] + lag
        return d.drop(columns=flags).merge(past, on=CELL, how="inner")

    def _write_summary(self, table: pd.DataFrame) -> None:
        dest = regression_summary_md(self.run_name)
        if table.empty:
            open(dest, "w").write("# Regression results\n\nNo spec produced an estimate.\n")
            return

        flags_only = table[table.term.str.startswith(("tier1_", "tier2_", "governance_"))].copy()
        flags_only["stars"] = np.select(
            [flags_only.p < 0.01, flags_only.p < 0.05, flags_only.p < 0.10], ["***", "**", "*"], "")
        flags_only["est"] = (flags_only.coef.map("{:+.4f}".format) + flags_only.stars
                             + " (" + flags_only.se.map("{:.4f}".format) + ")")

        lines = ["# Regression results", "",
                 f"`{table.spec_id.nunique()}` specs · outcome is ln(y) · firm FE + year FE · "
                 f"SEs clustered on `{self.cluster}`", "",
                 "Coefficient (standard error). `***` p<0.01, `**` p<0.05, `*` p<0.10.", ""]

        for (scope, form), block in flags_only.groupby(["scope", "outcome_form"], sort=False):
            head = block[(block.window == self.windows[0])
                         & (block.regressor_form == self.regressor_forms[0])
                         & (block.regressor_set == self.regressor_sets[0])
                         & (block.disclosed == self.disclosed_only[0])]
            if head.empty:
                continue
            wide = head.pivot_table(index="term", columns="lag", values="est", aggfunc="first")
            n = head.groupby("lag").n_obs.first().to_dict()
            firms = head.groupby("lag").n_clusters.first().to_dict()
            lines += [f"## {scope} — ln({form})",
                      f"window={self.windows[0]} · {self.regressor_form_label()} · "
                      f"disclosed_only={self.disclosed_only[0]} · "
                      + " · ".join(f"lag{k}: n={n.get(k,0):,}/{firms.get(k,0):,} firms" for k in sorted(n)),
                      "", _markdown(wide), ""]

        lines += ["## Full grid", "",
                  f"All {table.spec_id.nunique()} specs and every coefficient are in "
                  f"`{regression_results_csv(self.run_name)}`."]
        open(dest, "w").write("\n".join(lines) + "\n")
        logger.info("Wrote summary → %s", dest)

    def regressor_form_label(self) -> str:
        return f"regressors={self.regressor_forms[0]}, set={self.regressor_sets[0]}"
