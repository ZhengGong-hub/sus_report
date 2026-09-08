"""OLS with optional absorbed fixed effects and robust standard errors, via pyfixest. No config, no I/O."""

from __future__ import annotations

import pandas as pd
import pyfixest as pf

UNIT = "companyid"  # the panel's unit dimension, used for the firm count when nothing is absorbed


def fe_ols(frame: pd.DataFrame, y_col: str, x_cols: list[str],
           fe_cols: list[str], cluster_col: str | None) -> tuple[pd.DataFrame, dict]:
    """Regress y on x, absorbing fe_cols. Empty fe_cols = pooled OLS with an intercept;
    cluster_col=None = heteroskedasticity-robust SEs rather than clustered."""
    fml = f"{y_col} ~ {' + '.join(x_cols)}"
    if fe_cols:
        fml += f" | {' + '.join(fe_cols)}"
    # HC1 assumes observations are independent once the fixed effects are out, which a panel of
    # repeat observations per firm generally violates — clustering is the safer default
    vcov = {"CRV1": cluster_col} if cluster_col else "HC1"
    # fixef_rm="singleton" iterates: dropping a one-observation firm can strand a year with one
    # observation, and a singleton adds nothing to a within estimator. Meaningless with no FE,
    # where a firm seen once still carries cross-sectional information.
    kwargs = {"fixef_rm": "singleton"} if fe_cols else {}
    fit = pf.feols(fml, data=frame, vcov=vcov, **kwargs)

    # pyfixest drops perfectly collinear regressors rather than solving through them, so terms
    # can come back shorter than x_cols — the caller carries the names, not the row positions
    tidy = fit.tidy().reset_index()
    terms = pd.DataFrame({"term": tidy["Coefficient"], "coef": tidy["Estimate"],
                          "se": tidy["Std. Error"], "t": tidy["t value"], "p": tidy["Pr(>|t|)"]})

    collinear = fit._collin_vars or []
    if fe_cols:
        # _k_fe counts the levels of each absorbed dimension, keyed by name — look the unit
        # dimension up rather than taking the first entry, whose order pyfixest does not
        # guarantee. Works for either vcov, unlike _G, which comes back empty under HC1.
        unit_key = next(k for k in fit._k_fe.index if fe_cols[0] in str(k))
        n_firms, n_absorbed = int(fit._k_fe[unit_key]), int(fit._k_fe.sum())
        # with no FE there is no within transformation, so pyfixest returns nan for r2_within
        # and plain R² is the only fit statistic that means anything
        r2 = float(fit._r2_within)
    else:
        n_firms, n_absorbed = int(frame[UNIT].nunique()), 0
        r2 = float(fit._r2)

    diagnostics = {"n_obs": int(fit._N), "n_firms": n_firms,
                   "fe": "+".join(fe_cols) if fe_cols else "none",
                   "vcov": f"CRV1:{cluster_col}" if cluster_col else "HC1",
                   "n_clusters": int(fit._G[0]) if cluster_col else 0,
                   "n_absorbed": n_absorbed, "r2_within": r2,
                   "n_collinear": len(collinear), "collinear": ";".join(collinear)}
    return terms, diagnostics


def count_switchers(frame: pd.DataFrame, col: str, unit: str) -> int:
    """Firms whose regressor changes over time — the only ones a within estimator uses."""
    per_unit = frame.groupby(unit)[col].nunique()
    return int((per_unit > 1).sum())
