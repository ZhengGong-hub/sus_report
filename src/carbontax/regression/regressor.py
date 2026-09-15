"""Regressor: company-year panel → one two-way FE regression per configured outcome, as one tidy table."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import pandas as pd

from carbontax.paths import (panel_csv, regression_results_csv, regression_results_dir,
                             regression_spec_dir, regression_summary_md, regression_tables_dir)
from carbontax.regression.estimator import count_switchers, fe_ols
from carbontax.regression.outcomes import TOTAL_REVENUE
from carbontax.taxonomy import GOVERNANCE_FLAGS, MEASURE_SCOPE, TIER1_BUCKETS

logger = logging.getLogger(__name__)

CELL = ["companyid", "year"]     # panel key, and the join key the lag merge uses

# what makes one grid cell — and so one row of results.csv, one subfolder, one regression.
# Each outcome transformation is a separate model and output folder.
SPEC_KEYS = ["regressors", "y", "window", "lag", "fe", "transform"]

# outcome transforms. delta_log needs a <col>_prev in the panel — regression.panel.previous_year_extra
# controls which columns get one. It differences away the firm's level, so with company FE also on
# the spec the fixed effect becomes a firm-specific TREND rather than a firm-specific level.
TRANSFORMS = ("log", "level", "delta_log")

# how the dependent variable is written wherever a summary names it
LHS_LABEL = {"log": "ln(y)", "level": "y", "delta_log": "\u0394 ln(y)"}

# how the standard errors are described wherever a summary names them
SE_LABEL = {True: "clustered by firm", False: "heteroskedasticity-robust, NOT clustered"}


def fe_label(fe_cols: list[str]) -> str:
    """How a fixed-effect set is named in results.csv, folder names and summaries."""
    return "+".join(fe_cols) if fe_cols else "none"


def spec_name(fam: str, y_col: str, window: int, lag: int, fe: list[str], transform: str) -> str:
    """Folder name for one grid cell. Every part is already lowercase and underscored."""
    return f"{fam}__{y_col}__{transform}__w{window}__lag{lag}__fe_{fe_label(fe).replace('+', '-')}"

# exposure controls, available to config.controls — an empty list is legal and means the
# dummies and the two fixed effects are the whole model. log_n_chunks matters because a flag
# can only fire in a chunk we sampled; n_years because a pooled window covers more history
# for some cells than others. Every outcome gets the same set: the config names y columns,
# not what kind of quantity they are, so a ratio outcome cannot be handed a different list.
CONTROL_BUILDERS = {
    "log_n_chunks": lambda d: np.log(d.n_chunks),
    "log_revenue": lambda d: np.log(d[TOTAL_REVENUE].where(d[TOTAL_REVENUE] > 0)),
    "n_years": lambda d: d.n_years.astype(float),
    "n_filings": lambda d: d.n_filings.astype(float),
    "filed_this_year": lambda d: d.filed_this_year.astype(float),
}

# which family of adoption dummies goes on the right-hand side, all of it in one regression.
# Exactly one family per run, by config.regressors: tier1 and tier2 measure the same thing at
# different resolutions, so combining them would identify each tier1 bucket off the cells where
# it fires and none of its own tier2 members do — extraction disagreement, not economics.
REGRESSOR_SETS = {
    "tier2": [f"tier2_{m}_any" for m in MEASURE_SCOPE],           # 30 specific measures
    "tier1": [f"tier1_{b}_any" for b in TIER1_BUCKETS],           # 5 scope-level buckets
    "governance": [f"governance_{g}_any" for g in GOVERNANCE_FLAGS],  # 4 commitment flags
}

# families whose flag list depends on which outcome is being estimated, so they cannot be a
# static entry above. scope_matched keeps only the measures that can physically move this y
# (S1 measures for a Scope 1 outcome); scope_mismatched keeps exactly the rest, which is the
# falsification — those measures must NOT move it, and if they do the flags are picking up a
# firm characteristic rather than an abatement channel.
SCOPED_SETS = ("scope_matched", "scope_mismatched")


def flags_for(fam: str, y_col: str, outcome_scope: dict) -> list[str]:
    """Which dummies go on the right-hand side for this family × outcome."""
    if fam not in SCOPED_SETS:
        return REGRESSOR_SETS[fam]
    matched = [f"tier2_{m}_any" for m, s in MEASURE_SCOPE.items() if s == outcome_scope[y_col]]
    if fam == "scope_matched":
        return matched
    return [f for f in REGRESSOR_SETS["tier2"] if f not in matched]


def _md_table(frame: pd.DataFrame) -> str:
    # hand-rolled so the package does not need tabulate just to print a few tables
    header = [str(c) for c in frame.columns]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
            for row in frame.to_numpy()]
    return "\n".join(out)


def _markdown(wide: pd.DataFrame) -> str:
    """A pivot table (terms down the side, spec values across) as markdown."""
    flat = wide.reset_index()
    flat.columns = [str(wide.index.name or "term")] + [str(c) for c in wide.columns]
    return _md_table(flat)


class Regressor:
    def __init__(self, run_name: str, section: dict):
        self.run_name = run_name
        self.windows = section["window"]
        self.outcomes = section["outcomes"]
        transform = section["outcome_transform"]
        self.outcome_transforms = [transform] if isinstance(transform, str) else transform
        self.regressors = section["regressors"]
        self.outcome_scope = section["outcome_scope"]
        self.lags = section["lags"]
        self.fixed_effects = section["fixed_effects"]
        self.sector_mapping_csv = section.get("sector_mapping_csv")
        self.controls = section["controls"]
        self.min_switchers = section["min_switchers"]
        self.cluster = section["cluster"]

        if not isinstance(self.regressors, list) or not self.regressors:
            raise ValueError(f"regressors must be a non-empty list, got {self.regressors!r}")
        unknown = [f for f in self.regressors if f not in REGRESSOR_SETS and f not in SCOPED_SETS]
        if unknown:
            raise ValueError(f"unknown regressors {unknown}; "
                             f"have {sorted(set(REGRESSOR_SETS) | set(SCOPED_SETS))}")
        legal_scopes = set(TIER1_BUCKETS)
        bad_scope = {y: s for y, s in self.outcome_scope.items() if s not in legal_scopes}
        if bad_scope:
            raise ValueError(f"outcome_scope values must be taxonomy scopes {sorted(legal_scopes)}, "
                             f"got {bad_scope}")
        unknown = [c for c in self.controls if c not in CONTROL_BUILDERS]
        if unknown:
            raise ValueError(f"unknown controls {unknown}; have {sorted(CONTROL_BUILDERS)}")
        if not isinstance(self.outcomes, list) or not self.outcomes:
            raise ValueError(f"outcomes must be a non-empty list of panel columns, got {self.outcomes!r}")
        if not isinstance(self.lags, list) or not self.lags:
            raise ValueError(f"lags must be a non-empty list of whole years, got {self.lags!r}")
        bad = [k for k in self.lags if not isinstance(k, int) or k < 0]
        if bad:
            raise ValueError(f"lags must be non-negative whole years, got {bad}")
        if not isinstance(self.windows, list) or not self.windows:
            raise ValueError(f"window must be a non-empty list of pooling widths, got {self.windows!r}")
        if (not isinstance(self.outcome_transforms, list) or not self.outcome_transforms
                or any(t not in TRANSFORMS for t in self.outcome_transforms)):
            raise ValueError(f"outcome_transform must be a string or non-empty list from {TRANSFORMS}")
        if len(set(self.outcome_transforms)) != len(self.outcome_transforms):
            raise ValueError("outcome_transform must not contain duplicates")
        if not isinstance(self.fixed_effects, list) or not self.fixed_effects:
            raise ValueError("fixed_effects must be a non-empty list of fixed-effect sets; "
                             f"use [] for a set with no fixed effects, got {self.fixed_effects!r}")
        bad_fe = [fe for fe in self.fixed_effects if not isinstance(fe, list)]
        if bad_fe:
            raise ValueError(f"each entry of fixed_effects must itself be a list, got {bad_fe}")

    def run(self) -> pd.DataFrame:
        # CSV carries no dtypes: companyid is an all-digit id that would otherwise come back
        # as int64 and no longer match the string ids everything upstream is keyed on
        panel = pd.read_csv(panel_csv(self.run_name), dtype={"companyid": "string"}, low_memory=False)
        if any(c in {"sector", "sector_year"} for fe in self.fixed_effects for c in fe):
            panel = self._add_sector(panel)
        absent = [c for c in self.outcomes if c not in panel.columns]
        if absent:
            raise KeyError(f"outcome columns absent from the panel: {absent}")
        logger.info("Panel: %d rows, %d companies", len(panel), panel.companyid.nunique())

        # one regression per (family, outcome, window, lag) — separate models, never a shared RHS
        table = pd.DataFrame([r for fam in self.regressors for y_col in self.outcomes
                              for window in self.windows for lag in self.lags
                              for fe in self.fixed_effects for transform in self.outcome_transforms
                              for r in self._estimate(panel, y_col, fam, window, lag, fe, transform)])
        dest = regression_results_csv(self.run_name)
        table.to_csv(dest, index=False)
        # the estimated count, not the grid product: thin cells are skipped and never appear
        planned = (len(self.regressors) * len(self.outcomes) * len(self.windows)
                   * len(self.lags) * len(self.fixed_effects) * len(self.outcome_transforms))
        estimated = table.groupby(SPEC_KEYS).ngroups
        logger.info("Wrote %d coefficient rows over %d of %d planned regressions → %s",
                    len(table), estimated, planned, dest)
        self._write_spec_folders(table)
        self._write_summary(table)
        from carbontax.regression.plots import write_coefficient_plots
        graphs = write_coefficient_plots(
            table, os.path.join(regression_results_dir(self.run_name), "graphs"),
            controls=self.controls, fe_order=[fe_label(fe) for fe in self.fixed_effects],
            min_switchers=self.min_switchers)
        logger.info("Wrote %d coefficient plots", len(graphs))
        return table

    def _add_sector(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Map CIQ simple-industry descriptions to sector names on existing panels."""
        if not self.sector_mapping_csv:
            raise ValueError("sector FE requires regression.specs.sector_mapping_csv")
        if "simpleindustry" not in panel:
            raise KeyError("sector FE requires simpleindustry in the panel; rebuild the panel")
        mapping = pd.read_csv(self.sector_mapping_csv, dtype="string")
        keys = mapping["SIMPLEINDUSTRYDESCRIPTION"].str.strip()
        sectors = mapping["SECTORDESCRIPTION"].str.strip()
        if keys.isna().any() or keys.duplicated().any() or sectors.isna().any():
            raise ValueError("sector mapping must have unique, non-missing industries and sectors")
        panel = panel.copy()
        panel["sector"] = panel.simpleindustry.astype("string").str.strip().map(
            pd.Series(sectors.to_numpy(), index=keys))
        missing = panel.sector.isna()
        if missing.any():
            logger.warning("Sector mapping missing for %d panel rows; excluded only from sector FE "
                           "specifications. Industries: %s", missing.sum(),
                           panel.loc[missing, "simpleindustry"].unique().tolist())
        if any("sector_year" in fe for fe in self.fixed_effects):
            # Factorize pairs rather than concatenating labels, avoiding delimiter collisions.
            # Missing components stay missing so the specification's dropna excludes them.
            valid = panel[["sector", "year"]].notna().all(axis=1)
            panel["sector_year"] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
            pairs = pd.MultiIndex.from_frame(panel.loc[valid, ["sector", "year"]])
            panel.loc[valid, "sector_year"] = pd.factorize(pairs)[0]
        return panel

    def _write_spec_folders(self, table: pd.DataFrame) -> None:
        """One subfolder per grid cell, each holding that regression's own results.csv and summary.md."""
        root = regression_tables_dir(self.run_name)
        os.makedirs(root, exist_ok=True)

        written = set()
        for (fam, y_col, window, lag, fe, transform), block in table.groupby(SPEC_KEYS, sort=False):
            name = spec_name(fam, y_col, int(window), int(lag), fe.split("+") if fe != "none" else [], transform)
            folder = regression_spec_dir(self.run_name, name)
            os.makedirs(folder, exist_ok=True)
            block.to_csv(os.path.join(folder, "results.csv"), index=False)
            open(os.path.join(folder, "summary.md"), "w").write(
                self._spec_summary(fam, y_col, int(window), int(lag), fe, block))
            written.add(os.path.relpath(folder, root))

        # a folder this run did not write is left over from an earlier config and would read as
        # current output. Named rather than deleted — these are results, not scratch files.
        stale = sorted(os.path.relpath(folder, root)
                       for folder, dirs, files in os.walk(root)
                       if "results.csv" in files and "summary.md" in files
                       and "_legacy" not in os.path.relpath(folder, root).split(os.sep)
                       and os.path.relpath(folder, root) not in written)
        if stale:
            logger.warning("%d spec folder(s) in %s are left over from an earlier config and were "
                           "NOT refreshed: %s", len(stale), root, ", ".join(stale))
        logger.info("Wrote %d spec folders → %s", len(written), root)

    def _spec_summary(self, fam: str, y_col: str, window: int, lag: int, fe: str,
                      block: pd.DataFrame) -> str:
        b = block.copy()
        b["stars"] = np.select([b.p < 0.01, b.p < 0.05, b.p < 0.10], ["***", "**", "*"], "")
        flags = flags_for(fam, y_col, self.outcome_scope) if fam in SCOPED_SETS else REGRESSOR_SETS[fam]
        # with no fixed effects pyfixest returns an Intercept row, which is neither
        b["kind"] = np.where(b.term.isin(flags), "flag", "control")

        show = pd.DataFrame({
            "term": b.term, "": b.kind.map({"flag": "", "control": "control"}),
            "coef": b.coef.map("{:+.4f}".format), "se": b.se.map("{:.4f}".format),
            "t": b.t.map("{:+.2f}".format), "p": b.p.map("{:.4f}".format), "sig": b.stars,
            "switchers": b.switchers.map(lambda v: "" if pd.isna(v) else f"{int(v)}"),
        }).sort_values(["", "p"])

        r = block.iloc[0]
        lhs = LHS_LABEL[r["transform"]]
        controls = f"{', '.join(self.controls)} + " if self.controls else ""
        fe_txt = " + ".join(f"{c} FE" for c in fe.split("+")) if fe != "none" else "no fixed effects"
        r2_txt = "Within-R²" if fe != "none" else "R²"
        return "\n".join([
            f"# {fam} · `{y_col}` · {r['transform']} · window {window} · lag {lag} · fe {fe}", "",
            f"`{lhs} ~ {len(flags)} {fam} dummies + {controls}{fe_txt}`", "",
            f"- Sample: **{int(r.n_obs):,}** company-years over **{int(r.n_firms):,}** firms",
            f"- Standard errors: {SE_LABEL[bool(self.cluster)]} (`{r.vcov}`)",
            f"- Flags: {len(flags) - int(r.n_dropped_flags)} estimated, {int(r.n_dropped_flags)} "
            f"dropped, {int(r.n_collinear)} collinear"
            + (f" (`{r.collinear}`)" if isinstance(r.collinear, str) and r.collinear else ""),
            f"- {r2_txt}: {r.r2_within:.4f}", "",
            "`***` p<0.01, `**` p<0.05, `*` p<0.10. `switchers` = firms whose flag changes over "
            "time; the min_switchers filter applies only with company fixed effects.", "",
            _md_table(show), "",
            "Identical content in `results.csv` beside this file.", "",
            "## Sample audit", "",
            *([r.sample_source, ""] if isinstance(r.sample_source, str) and r.sample_source else []),
            _md_table(pd.DataFrame(json.loads(r.sample_audit))), "",
            "Counts are sequential. Reports are aggregated into firm-years; wider windows can fill gap years. "
            "Outcome availability is measured after report–Trucost alignment, not raw Trucost coverage. "
            "Zero-valued measures stay in; variable filtering does not remove non-switching firms.", "",
        ])

    def _estimate(self, panel: pd.DataFrame, y_col: str, fam: str, window: int, lag: int,
                  fe: list[str], transform: str | None = None) -> list[dict]:
        if transform is None:
            if len(self.outcome_transforms) != 1:
                raise ValueError("Specify transform when estimating a multi-transform run")
            transform = self.outcome_transforms[0]
        if transform not in TRANSFORMS:
            raise ValueError(f"Unknown outcome transform: {transform!r}")
        d = panel[panel.window == window].copy()
        audit = []
        def record(label, frame):
            audit.append({"Step": label, "Firm-years": f"{len(frame):,}",
                          "Companies": f"{frame.companyid.nunique():,}"})

        # Window 1 counts each retained filing once, unlike pooled report counts.
        base = panel[panel.window == 1]
        sample_source = ""
        if not base.empty:
            sample_source = (f"{int(base.n_filings.sum()):,} retained reports aggregated into "
                             f"{len(base):,} firm-years across {base.companyid.nunique():,} companies "
                             "(window 1).")
        record(f"Panel (window {window})", d)
        controls = list(self.controls)
        for c in controls:
            d[c] = CONTROL_BUILDERS[c](d)
        # Source lagged X from all report rows, independent of outcome eligibility.
        lag_source = d.copy()
        if "outcome_trusted" in d:
            d = d[d.outcome_trusted.fillna(False).astype(bool)]
            record("Report–carbon period aligned", d)
        d = d[d[y_col].notna()]
        record("Current outcome available", d)
        if transform == "delta_log":
            prev = f"{y_col}_prev"
            if prev not in d.columns:
                raise KeyError(f"outcome_transform=delta_log needs {prev!r} in the panel; add "
                               f"{y_col!r} to regression.panel.previous_year_extra and rebuild")
            d = d[d[prev].notna()]
            record("Previous-year outcome available", d)
        if transform != "level":
            d = d[d[y_col] > 0]
            if transform == "delta_log":
                d = d[d[prev] > 0]
            record("Positive outcomes for logs", d)
        d["dep_y"] = (d[y_col].astype(float) if transform == "level" else np.log(d[y_col]))
        if transform == "delta_log":
            d["dep_y"] -= np.log(d[prev])

        # an outcome with no taxonomy scope (water, waste, the cross-scope aggregates) has no
        # matched or mismatched measure set, so those families simply do not apply to it
        if fam in SCOPED_SETS and y_col not in self.outcome_scope:
            logger.info("SKIP %s / %s / lag %d / fe %s: outcome has no scope in outcome_scope",
                        fam, y_col, lag, fe_label(fe))
            return []
        flags = flags_for(fam, y_col, self.outcome_scope)
        missing = [f for f in flags if f not in d.columns]
        if missing:
            raise KeyError(f"{fam} flags absent from the panel: {missing}")
        # the controls move with the flags, not with y: log_n_chunks is the sampling exposure of
        # the report the flags were read from, so at lag k it has to be that year's chunk count.
        # A control describing the outcome instead (log_revenue) would want the opposite.
        d = self._apply_lag(d, flags + controls, lag, source=lag_source)
        if lag:
            record(f"Report variables available at lag {lag}", d)
        # not for pyfixest, which drops NaN rows itself — this is so the switcher counts and the
        # zero-variance control check below are computed on the rows the regression actually uses
        d = d.dropna(subset=["dep_y"] + flags + controls + fe
                        + ([self.cluster] if self.cluster else []))
        record("Complete measures, controls and FE", d)
        # A thin cell is a fact about the data, not a broken config, so it is skipped rather than
        # raised on: a grid is expected to have corners a sparse outcome cannot fill. Structural
        # problems above (absent columns, unknown config) still raise.
        # singleton firms are left to pyfixest (fixef_rm="singleton"), which removes them iteratively
        if len(d) < 50 or d.companyid.nunique() < 10:
            logger.warning("SKIP %s / %s / lag %d / fe %s: sample too small — %d rows, %d firms",
                           fam, y_col, lag, fe_label(fe), len(d), d.companyid.nunique())
            return []

        # switcher counts are reported either way, but only gate the regressor set when the firm
        # FE is actually on: a dummy constant within firm is absorbed by that FE and contributes
        # nothing, whereas with no firm FE it still identifies off variation across firms.
        switchers = {f: count_switchers(d, f, "companyid") for f in flags}
        if "companyid" in fe:
            kept = [f for f in flags if switchers[f] >= self.min_switchers]
            reason = f"min_switchers={self.min_switchers}"
        else:
            kept = [f for f in flags if d[f].std() > 0]
            reason = "zero variance"
        if not kept:
            logger.warning("SKIP %s / %s / lag %d / fe %s: no flag survives %s",
                           fam, y_col, lag, fe_label(fe), reason)
            return []
        rhs = kept + [c for c in controls if d[c].std() > 0]

        terms, diag = fe_ols(d, "dep_y", rhs, fe, self.cluster)
        audit.append({"Step": "Final sample (after singleton removal)" if fe else "Final sample",
                      "Firm-years": f"{diag['n_obs']:,}", "Companies": f"{diag['n_firms']:,}"})
        logger.info("%s / %s / lag %d / fe %s: n=%d, %d firms, %d/%d flags, %d collinear",
                    fam, y_col, lag, fe_label(fe), diag["n_obs"], diag["n_firms"], len(kept),
                    len(flags), diag["n_collinear"])
        return [{"y": y_col, "transform": transform, "regressors": fam,
                 "window": window, "lag": lag,
                 "sample_source": sample_source,
                 "sample_audit": json.dumps(audit, ensure_ascii=False),
                 "n_dropped_flags": len(flags) - len(kept),
                 **row, "switchers": switchers.get(row["term"], np.nan), **diag}
                for row in terms.to_dict("records")]

    @staticmethod
    def _apply_lag(d: pd.DataFrame, cols: list[str], lag: int,
                   source: pd.DataFrame | None = None) -> pd.DataFrame:
        """Pair y at year t with cols from year t-lag. Replaces cols, it does not add to them."""
        if lag == 0:
            return d
        # match on year explicitly rather than shifting rows: the panel has gaps, and a
        # positional shift would silently pair non-adjacent years
        past = (d if source is None else source)[CELL + cols].copy()
        past["year"] = past["year"] + lag
        return d.drop(columns=cols).merge(past, on=CELL, how="inner")

    def _write_summary(self, table: pd.DataFrame) -> None:
        # an empty controls list is legal, so the term only appears when there is one
        controls = f"{', '.join(self.controls)} + " if self.controls else ""
        lhs = " / ".join(LHS_LABEL[t] for t in self.outcome_transforms)
        lines = ["# Regression results", "",
                 f"`{lhs} ~ <adoption dummies at t-lag> + {controls}<fixed effects>`. "
                 f"Standard errors {SE_LABEL[bool(self.cluster)]}. One regression per "
                 f"family × outcome × transformation × window × lag × fixed-effect set — the families are separate "
                 f"models, never a shared right-hand side, and a lag replaces the contemporaneous "
                 f"flags rather than joining them.", "",
                 "Coefficient (standard error). `***` p<0.01, `**` p<0.05, `*` p<0.10."]

        # column order follows the config, so the specs read left to right the way they were asked for
        fe_labels = [fe_label(fe) for fe in self.fixed_effects]
        spec_order = [(t, w, k, e) for t in self.outcome_transforms
                      for w in self.windows for k in self.lags for e in fe_labels]

        # one table per (family, outcome), every window × lag side by side — the comparison to read
        for fam in self.regressors:
            for y_col in self.outcomes:
                block = table[(table.regressors == fam) & (table.y == y_col)
                              # flags only: a scoped family's set varies by outcome, so the
                              # controls are excluded by name rather than the flags by membership
                              # flags only: drop the controls by name and the no-FE intercept
                              & ~table.term.isin(self.controls + ["Intercept"])].copy()
                if block.empty:  # every spec for this pair was skipped as too thin
                    lines += ["", f"## {fam} — `{y_col}`", "", "_no spec produced an estimate._"]
                    continue
                block["stars"] = np.select(
                    [block.p < 0.01, block.p < 0.05, block.p < 0.10], ["***", "**", "*"], "")
                block["est"] = (block.coef.map("{:+.4f}".format) + block.stars
                                + " (" + block.se.map("{:.4f}".format) + ")")
                block["spec"] = (block["transform"] + " w" + block.window.astype(int).astype(str)
                                 + " lag" + block.lag.astype(int).astype(str)
                                 + " fe=" + block.fe)
                labels = [f"{t} w{w} lag{k} fe={e}" for t, w, k, e in spec_order]

                wide = block.pivot_table(index="term", columns="spec", values="est", aggfunc="first")
                wide = wide.reindex(columns=[c for c in labels if c in wide.columns])

                head = block.groupby("spec").first()
                lines += ["", f"## {fam} — `{y_col}`", ""]
                for lab in [c for c in labels if c in head.index]:
                    r = head.loc[lab]
                    lines.append(f"- {lab}: n={int(r.n_obs):,} over {int(r.n_firms):,} firms, "
                                 f"{'R²' if r.fe == 'none' else 'within-R²'}={r.r2_within:.3f}, "
                                 f"{int(r.n_dropped_flags)} flags filtered, "
                                 f"{int(r.n_collinear)} dropped collinear")
                lines += ["", _markdown(wide)]

        lines += ["", "Each specification’s `summary.md` in `results/tables/` ends with a sample audit "
                  "showing retained reports, usable firm-years, and companies at each restriction.", "",
                  f"Controls and full diagnostics are in `{regression_results_csv(self.run_name)}`."]
        open(dest := regression_summary_md(self.run_name), "w").write("\n".join(lines) + "\n")
        logger.info("Wrote summary → %s", dest)
