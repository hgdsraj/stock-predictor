"""Phase 17: Fama-MacBeth cross-sectional regression as an alternative
to GBM.

Why bother:
  - GBM is the modeling workhorse but is famously prone to over-fitting
    on weak tabular signals with hundreds of trees and dozens of
    features. We've seen 13 phases of HOLDOUT-CI-straddles-zero.
  - Fama-MacBeth is the standard cross-sectional asset-pricing tool:
    one OLS per day (so trivial to fit; no hyperparameter zoo), and
    the factor-return time-series averaging is naturally robust.
  - This is also a useful sanity check: if FM produces materially
    better HOLDOUT Sharpe than GBM with the same features, GBM is
    over-fitting; if worse, the marginal value of tree non-linearity
    is real.

Procedure (textbook):

  1. First-stage (per-date OLS):
     For each training date t with at least `min_obs_per_date` rows:
         y_t  = X_t @ lambda_t  + epsilon
     gives a vector lambda_t in R^k (k = #features).

  2. Time-series averaging:
         lambda_hat = mean over training dates of lambda_t

  3. Out-of-sample prediction:
         y_hat_oos = X_oos @ lambda_hat

  4. Optional shrinkage: lambda_hat * (1 - shrinkage) toward zero, which
     is equivalent to ridge in expectation.

This module exposes `fit_predict_fama_macbeth(X_tr, y_tr, X_te)` that
plugs into `pipeline._fit_and_predict_fold` when `model='fama_macbeth'`.

Memory profile: O(k) factor weights + O(#train_dates) per-date lambdas.
For 150 tickers x 11 yr x 18 ranked features, k ~= 18 and #train_dates
~= 2200; total state ~= 40 KB. The per-date regression matrices are
small (~150x18) so float32 throughout.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Numerical guardrails
_MIN_OBS_PER_DATE = 10  # require >= 10 names per date to fit a regression
_RIDGE_LAMBDA = 1e-4  # tiny ridge to avoid singularity when X is rank-deficient


def _fit_one_day(
    X_day: np.ndarray,
    y_day: np.ndarray,
    *,
    ridge: float = _RIDGE_LAMBDA,
) -> np.ndarray | None:
    """One-day cross-sectional OLS with tiny ridge.

    Returns a vector of length k (no intercept by convention; the model
    is run on z-scored / ranked features so a const offset wouldn't
    matter for the cross-sectional ranking we ultimately use).

    `None` on degenerate cases (too few rows, all-NaN inputs).
    """
    if X_day.shape[0] < _MIN_OBS_PER_DATE:
        return None
    # Median-impute NaN columns; drop all-NaN rows
    nan_cols = np.isnan(X_day).all(axis=0)
    if nan_cols.any():
        X_day = X_day[:, ~nan_cols]
    if X_day.shape[1] == 0:
        return None
    col_medians = np.nanmedian(X_day, axis=0)
    inds = np.where(np.isnan(X_day))
    X_day = X_day.copy()
    X_day[inds] = np.take(col_medians, inds[1])

    # Drop rows with NaN y
    mask = ~np.isnan(y_day)
    if mask.sum() < _MIN_OBS_PER_DATE:
        return None
    X = X_day[mask].astype(np.float32, copy=False)
    y = y_day[mask].astype(np.float32, copy=False)

    # Ridge-regularised normal equations: lambda = (X'X + lam*I)^-1 X'y
    k = X.shape[1]
    XtX = X.T @ X + ridge * np.eye(k, dtype=np.float32)
    Xty = X.T @ y
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lam = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        return None
    # Restore zeros for dropped all-NaN columns
    if nan_cols.any():
        out = np.zeros(nan_cols.shape[0], dtype=np.float32)
        out[~nan_cols] = lam
        return out
    return lam.astype(np.float32)


def fit_predict_fama_macbeth(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    *,
    shrinkage: float = 0.0,
) -> pd.Series:
    """Fit per-date OLS on training rows; predict on test rows.

    Both X_tr/X_te must be indexed by (date, ticker); columns must be
    the same set of features.

    Returns a Series indexed like X_te with the predicted score.
    """
    if X_tr.empty or X_te.empty:
        return pd.Series(index=X_te.index, dtype=float, name="prediction")

    feats = X_tr.columns.tolist()
    X_te = X_te[feats]  # enforce column alignment

    # First-stage: per-date lambdas
    lambdas: list[np.ndarray] = []
    train_dates = X_tr.index.get_level_values("date").unique()
    for d in train_dates:
        X_day_df = X_tr.xs(d, level="date")
        y_day = y_tr.xs(d, level="date")
        X_day = X_day_df.to_numpy(dtype=np.float32, copy=False)
        y_day_arr = y_day.to_numpy(dtype=np.float32, copy=False)
        lam = _fit_one_day(X_day, y_day_arr)
        if lam is not None:
            lambdas.append(lam)

    if not lambdas:
        log.warning(
            "fama_macbeth: zero successful per-day regressions on %d train dates; "
            "returning zero predictions.",
            len(train_dates),
        )
        return pd.Series(0.0, index=X_te.index, name="prediction")

    lambda_arr = np.stack(lambdas, axis=0)  # (n_train_dates, k)
    lambda_hat = lambda_arr.mean(axis=0)  # (k,)
    if shrinkage > 0.0:
        lambda_hat = lambda_hat * (1.0 - shrinkage)
    log.info(
        "fama_macbeth: averaged %d daily lambdas over %d features; "
        "|lambda_hat|_max=%.4f, shrinkage=%.2f",
        len(lambdas),
        len(feats),
        float(np.max(np.abs(lambda_hat))),
        shrinkage,
    )

    # Predict OOS: y_hat = X_te @ lambda_hat
    X_te_arr = X_te.to_numpy(dtype=np.float32, copy=False)
    # Median-impute NaN cells using train medians (could leak past->future,
    # but the bias is small and uniform; honest comparison vs gbm which
    # does its own imputation internally).
    nan_mask = np.isnan(X_te_arr)
    if nan_mask.any():
        col_medians = np.nanmedian(X_tr.to_numpy(dtype=np.float32, copy=False), axis=0)
        X_te_arr = X_te_arr.copy()
        for c in range(X_te_arr.shape[1]):
            X_te_arr[np.isnan(X_te_arr[:, c]), c] = (
                col_medians[c] if not np.isnan(col_medians[c]) else 0.0
            )
    y_hat = X_te_arr @ lambda_hat
    return pd.Series(y_hat, index=X_te.index, name="prediction")
