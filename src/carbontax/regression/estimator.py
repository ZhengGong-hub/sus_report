"""Two-way fixed-effects OLS with cluster-robust standard errors, via pyfixest. No config, no I/O."""

from __future__ import annotations

import pandas as pd
import pyfixest as pf


def fe_ols(frame: pd.DataFrame, y_col: str, x_cols: list[str],
           fe_cols: list[str], cluster_col: str | None) -> tuple[pd.DataFrame, dict]:
    """Absorb fe_cols, regress y on x. cluster_col=None gives heteroskedasticity-robust SEs instead."""
    fml = f"{y_col} ~ {' + '.join(x_cols)} | {' + '.join(fe_cols)}"
    # HC1 assumes observations are independent once the fixed effects are out, which a panel of
    # repeat observations per firm generally violates — clustering is the safer default
    vcov = {"CRV1": cluster_col} if cluster_col else "HC1"
    # fixef_rm="singleton" iterates: dropping a one-observation firm can strand a year with one
    # observation, and a singleton adds nothing to a within estimator while inflating the cluster count
    fit = pf.feols(fml, data=frame, vcov=vcov, fixef_rm="singleton")

    # pyfixest drops perfectly collinear regressors rather than solving through them, so terms
    # can come back shorter than x_cols — the caller carries the names, not the row positions
    tidy = fit.tidy().reset_index()
    terms = pd.DataFrame({"term": tidy["Coefficient"], "coef": tidy["Estimate"],
                          "se": tidy["Std. Error"], "t": tidy["t value"], "p": tidy["Pr(>|t|)"]})

    collinear = fit._collin_vars or []
    # _k_fe counts the levels of each absorbed dimension, keyed by name — look the firm dimension
    # up rather than taking the first entry, whose order pyfixest does not guarantee. Works for
    # either vcov, unlike _G, which comes back empty under HC1.
    unit_key = next(k for k in fit._k_fe.index if fe_cols[0] in str(k))
    diagnostics = {"n_obs": int(fit._N), "n_firms": int(fit._k_fe[unit_key]),
                   "vcov": f"CRV1:{cluster_col}" if cluster_col else "HC1",
                   "n_clusters": int(fit._G[0]) if cluster_col else 0,
                   "n_absorbed": int(fit._k_fe.sum()), "r2_within": float(fit._r2_within),
                   "n_collinear": len(collinear), "collinear": ";".join(collinear)}
    return terms, diagnostics


def count_switchers(frame: pd.DataFrame, col: str, unit: str) -> int:
    """Firms whose regressor changes over time — the only ones a within estimator uses."""
    per_unit = frame.groupby(unit)[col].nunique()
    return int((per_unit > 1).sum())
