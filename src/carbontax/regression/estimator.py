"""Two-way fixed-effects OLS with cluster-robust standard errors. Pure numpy: no config, no I/O."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _codes(frame: pd.DataFrame, col: str) -> np.ndarray:
    # int64 explicitly: pandas hands back int8 codes when a column has few categories,
    # and the absorbed-parameter count then overflows on small samples
    return frame[col].astype("category").cat.codes.to_numpy().astype(np.int64)


def _demean(values: np.ndarray, groups: list[np.ndarray], iters: int) -> np.ndarray:
    # alternating projections: sweeping out each group's means repeatedly converges on the
    # two-way within transformation, without ever building the firm/year dummy matrix
    out = values.astype(float).copy()
    for _ in range(iters):
        for idx in groups:
            sums = np.zeros((idx.max() + 1, out.shape[1]))
            counts = np.zeros(idx.max() + 1)
            np.add.at(sums, idx, out)
            np.add.at(counts, idx, 1)
            out -= (sums / counts[:, None])[idx]
    return out


def fe_ols(frame: pd.DataFrame, y_col: str, x_cols: list[str],
           fe_cols: list[str], cluster_col: str, iters: int = 50) -> tuple[pd.DataFrame, dict]:
    """Absorb fe_cols, regress y on x, cluster SEs on cluster_col. Returns (terms, diagnostics)."""
    groups = [_codes(frame, c) for c in fe_cols]
    demeaned = _demean(frame[[y_col] + x_cols].to_numpy(), groups, iters)
    y, X = demeaned[:, 0], demeaned[:, 1:]

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    # absorbed parameters: one per firm plus one per year, less the shared intercept
    absorbed = sum(idx.max() + 1 for idx in groups) - (len(groups) - 1)
    n, k = len(frame), X.shape[1]
    cid = _codes(frame, cluster_col)
    n_clusters = cid.max() + 1

    meat = np.zeros((k, k))
    for g in range(n_clusters):
        sel = cid == g
        s = X[sel].T @ resid[sel]
        meat += np.outer(s, s)
    # standard finite-sample correction for clustered SEs, counting the absorbed FE
    dof = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / max(n - k - absorbed, 1))
    vcov = xtx_inv @ meat @ xtx_inv * dof
    se = np.sqrt(np.clip(np.diag(vcov), 0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(se > 0, beta / se, np.nan)
    terms = pd.DataFrame({"term": x_cols, "coef": beta, "se": se, "t": t})
    # two-sided normal p-value; the cluster count is what limits inference here, not n
    terms["p"] = 2 * (1 - _norm_cdf(np.abs(terms.t.to_numpy())))

    tss = float((y**2).sum())
    diagnostics = {"n_obs": n, "n_clusters": int(n_clusters), "n_absorbed": int(absorbed),
                   "r2_within": 1 - float((resid**2).sum()) / tss if tss > 0 else np.nan}
    return terms, diagnostics


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # Abramowitz & Stegun 7.1.26 style erf, so scipy is not a dependency
    t = 1 / (1 + 0.2316419 * x)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
    return 1 - np.exp(-x * x / 2) / np.sqrt(2 * np.pi) * poly


def count_switchers(frame: pd.DataFrame, col: str, unit: str) -> int:
    """Firms whose regressor changes over time — the only ones a within estimator uses."""
    per_unit = frame.groupby(unit)[col].nunique()
    return int((per_unit > 1).sum())
