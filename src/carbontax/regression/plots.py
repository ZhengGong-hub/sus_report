"""PNG coefficient plots comparing fixed effects within an otherwise identical spec."""
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROUP_KEYS = ["regressors", "y", "transform", "window", "lag"]
FE_NAMES = {"none": "No fixed effects", "companyid+year": "Company + year FE",
            "sector+year": "Sector + year FE", "sector_year": "Sector × year FE",
            "companyid+sector_year": "Company + sector × year FE"}
COLORS = ["#194f78", "#d77920", "#25836d", "#9064a8", "#b84254", "#626262"]
MARKERS = ["o", "s", "D", "^", "v", "P"]


def term_label(term):
    if term.startswith("tier1_"):
        return term.removeprefix("tier1_").removesuffix("_any")
    for prefix in ("tier2_", "governance_"):
        term = term.removeprefix(prefix)
    label = term.removesuffix("_any").replace("_", " ").capitalize()
    for old, new in {"Ppa": "PPA", "Rec goo": "RECs / guarantees of origin", "Nbs": "Nature-based solutions",
                     "Tech cdr": "Technological carbon removal", "Fgas": "F-gas", " eol": " end of life"}.items():
        label = label.replace(old, new)
    return label


def write_coefficient_plots(table, directory, *, controls, fe_order, min_switchers):
    """Use estimator-provided 95% intervals, never reconstruct inference from rounded CSVs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    required = {"ci_low", "ci_high", *GROUP_KEYS, "fe", "term", "coef"}
    if required - set(table):
        raise ValueError("Plotting requires fresh regression results with confidence intervals")
    paths = []
    for (family, outcome, transform, window, lag), block in table.groupby(GROUP_KEYS, sort=False):
        # Measures are the focus; controls are named in the caption and remain in the model.
        measures = block[~block.term.isin(list(controls) + ["Intercept"])].copy()
        if measures.empty:
            continue
        effects = [e for e in fe_order if e in set(block.fe)]
        effects += [e for e in block.fe.unique() if e not in effects]
        # One shared ordering for all models, analogous to the reference's sorted rows.
        terms = measures.groupby("term", sort=False).coef.mean().sort_values().index.tolist()
        height = max(6.8, 3.8 + .36 * len(terms) + .22 * len(effects))
        fig, ax = plt.subplots(figsize=(12.5, height))
        fig.subplots_adjust(left=.36, right=.97, top=1-2.25/height, bottom=1.45/height)
        fig.suptitle("Reported carbon measures and emissions", x=.03, ha="left", y=1-.18/height,
                     fontsize=16, fontweight="bold")
        lhs = {"log": "ln(Y)", "delta_log": "ln(Y[t]) − ln(Y[t−1])", "level": "Y"}[transform]
        caption = (f"Outcome: {outcome}\n"
                   f"Y transform: {lhs}  |  Regressors: {family}  |  Window: {window} year(s)  |  Lag: {lag} year(s)\n"
                   f"Controls (included, not plotted): {', '.join(controls) or 'none'}\n"
                   f"Standard errors: {', '.join('HC1 (heteroskedasticity-robust)' if v == 'HC1' else v.replace('CRV1:', 'Clustered by ') for v in block.vcov.unique())}  |  Intervals: 95% (estimator-provided)")
        fig.text(.03, 1-.55/height, caption, ha="left", va="top", fontsize=10, linespacing=1.6)
        offsets = np.linspace(-.27, .27, len(effects)) if len(effects)>1 else [0]
        for i, (fe, offset) in enumerate(zip(effects, offsets)):
            rows = measures[measures.fe == fe].set_index("term").reindex(terms)
            valid = rows[["coef", "ci_low", "ci_high"]].notna().all(axis=1)
            rows = rows[valid]
            meta = block[block.fe == fe].iloc[0]
            label = (f"{FE_NAMES.get(fe, fe)}  ·  N={int(meta.n_obs):,}, firms={int(meta.n_firms):,}"
                     f"  ·  {int(meta.n_dropped_flags)} filtered, {int(meta.n_collinear)} collinear")
            ax.errorbar(rows.coef, np.flatnonzero(valid)+offset,
                        xerr=np.vstack([rows.coef-rows.ci_low, rows.ci_high-rows.coef]),
                        fmt=MARKERS[i % len(MARKERS)], color=COLORS[i % len(COLORS)],
                        markersize=5, elinewidth=1.3, capsize=2, label=label)
        ax.axvline(0, color="#c33954", lw=1.1, zorder=0)
        ax.set_yticks(range(len(terms)), [textwrap.fill(term_label(t), 37) for t in terms], fontsize=10)
        ax.set_ylim(len(terms)-.4, -.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#e4edf1", linewidth=.9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")
        ax.set_xlabel("Estimated coefficient (log points)" if transform != "level" else "Estimated coefficient (outcome units)", fontsize=10)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=9, frameon=False, borderaxespad=0)
        fig.text(.03, .16/height,
                 "All measure coefficients are estimated jointly within each model. Zero-valued measures remain in the sample.\n"
                 f"Missing dots indicate filtered/absorbed terms. Company FE uses min_switchers={min_switchers}; "
                 "other FE uses a variance check.\nSamples can differ across FE models. Rows ordered by mean coefficient across the models shown.",
                 fontsize=9, color="#555555", va="bottom", linespacing=1.5)
        folder = directory / outcome / transform / family
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"w{window}__lag{lag}.png"
        fig.savefig(path, dpi=180, facecolor="white")
        plt.close(fig)
        paths.append(path)
    return paths
