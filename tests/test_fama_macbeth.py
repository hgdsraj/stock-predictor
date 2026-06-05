"""Phase 17 tests: Fama-MacBeth cross-sectional regression model.

Verifies:
  - Per-day OLS produces correct lambdas in the noiseless 2-factor case
  - Time-series averaging works
  - NaN handling (NaN feature cells imputed, NaN y rows dropped)
  - Empty inputs return empty predictions (no crash)
  - Degenerate per-day folds (<min_obs) are skipped
  - Out-of-sample predictions match the linear formula
  - Pipeline dispatch wires model='fama_macbeth' through _fit_and_predict_fold
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.models.fama_macbeth import (
    _fit_one_day,
    fit_predict_fama_macbeth,
)


def _make_two_factor_panel(
    n_days: int = 30,
    n_tickers: int = 20,
    true_lambdas: tuple[float, float] = (0.5, -0.3),
    noise: float = 0.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a synthetic (date, ticker) panel where
    y = lambda_1 * f1 + lambda_2 * f2 + epsilon.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    f1 = rng.normal(0, 1, len(idx))
    f2 = rng.normal(0, 1, len(idx))
    eps = rng.normal(0, noise, len(idx))
    y = true_lambdas[0] * f1 + true_lambdas[1] * f2 + eps
    X = pd.DataFrame({"f1": f1, "f2": f2}, index=idx)
    return X, pd.Series(y, index=idx, name="y")


def test_fit_one_day_recovers_lambda_in_noiseless_case():
    """Per-day OLS on a clean linear y must recover the true coefficients."""
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (30, 2)).astype(np.float32)
    true = np.array([0.5, -0.3], dtype=np.float32)
    y = X @ true
    lam = _fit_one_day(X, y)
    assert lam is not None
    np.testing.assert_allclose(lam, true, atol=0.01)


def test_fit_one_day_returns_none_for_too_few_rows():
    """Fewer than _MIN_OBS_PER_DATE rows -> None."""
    X = np.zeros((5, 3), dtype=np.float32)
    y = np.zeros(5, dtype=np.float32)
    assert _fit_one_day(X, y) is None


def test_fama_macbeth_recovers_average_lambda_noiseless():
    """In the noiseless case, the averaged lambda should be the true value."""
    X, y = _make_two_factor_panel(n_days=20, n_tickers=15, noise=0.0)
    # Use first 15 days as train, last 5 as test
    train_dates = X.index.get_level_values("date").unique()[:15]
    test_dates = X.index.get_level_values("date").unique()[15:]
    X_tr = X.loc[(X.index.get_level_values("date").isin(train_dates))]
    y_tr = y.loc[(y.index.get_level_values("date").isin(train_dates))]
    X_te = X.loc[(X.index.get_level_values("date").isin(test_dates))]
    pred = fit_predict_fama_macbeth(X_tr, y_tr, X_te)
    assert len(pred) == len(X_te)
    # Predictions in the noiseless case should be ~exactly y_te
    expected = 0.5 * X_te["f1"] - 0.3 * X_te["f2"]
    np.testing.assert_allclose(pred.to_numpy(), expected.to_numpy(), atol=0.05)


def test_fama_macbeth_with_noise_correlates_with_truth():
    """With noise, lambda recovery is imperfect but predictions should
    still correlate strongly with the true generative function."""
    X, y = _make_two_factor_panel(n_days=60, n_tickers=30, noise=0.5, seed=1)
    train_dates = X.index.get_level_values("date").unique()[:50]
    test_dates = X.index.get_level_values("date").unique()[50:]
    X_tr = X.loc[(X.index.get_level_values("date").isin(train_dates))]
    y_tr = y.loc[(y.index.get_level_values("date").isin(train_dates))]
    X_te = X.loc[(X.index.get_level_values("date").isin(test_dates))]
    pred = fit_predict_fama_macbeth(X_tr, y_tr, X_te)
    expected = 0.5 * X_te["f1"] - 0.3 * X_te["f2"]
    # Correlation should be strong (the true r-squared is ~0.5/0.5 + noise^2
    # = ~0.6; we're aiming for empirical Pearson > 0.6).
    corr = np.corrcoef(pred.to_numpy(), expected.to_numpy())[0, 1]
    assert corr > 0.6, f"FM pred should correlate strongly with truth; got {corr:.3f}"


def test_fama_macbeth_handles_empty_inputs():
    empty_idx = pd.MultiIndex.from_tuples([], names=["date", "ticker"])
    X = pd.DataFrame(columns=["f1"], index=empty_idx)
    y = pd.Series(dtype=float, index=empty_idx)
    out = fit_predict_fama_macbeth(X, y, X)
    assert out.empty


def test_fama_macbeth_handles_nan_features():
    """NaN cells should be median-imputed; NaN y rows dropped."""
    X, y = _make_two_factor_panel(n_days=20, n_tickers=20, noise=0.1, seed=2)
    # Inject some NaNs
    X_nan = X.copy()
    X_nan.iloc[::5, 0] = np.nan
    y_nan = y.copy()
    y_nan.iloc[10::20] = np.nan
    train_dates = X.index.get_level_values("date").unique()[:15]
    test_dates = X.index.get_level_values("date").unique()[15:]
    X_tr = X_nan.loc[(X_nan.index.get_level_values("date").isin(train_dates))]
    y_tr = y_nan.loc[(y_nan.index.get_level_values("date").isin(train_dates))]
    X_te = X.loc[(X.index.get_level_values("date").isin(test_dates))]
    pred = fit_predict_fama_macbeth(X_tr, y_tr, X_te)
    assert not pred.isna().all()
    assert len(pred) == len(X_te)


def test_fama_macbeth_shrinkage_scales_predictions():
    """shrinkage=0.5 should halve the predictions (relative to shrinkage=0)."""
    X, y = _make_two_factor_panel(n_days=20, n_tickers=15, noise=0.0)
    train_dates = X.index.get_level_values("date").unique()[:15]
    test_dates = X.index.get_level_values("date").unique()[15:]
    X_tr = X.loc[(X.index.get_level_values("date").isin(train_dates))]
    y_tr = y.loc[(y.index.get_level_values("date").isin(train_dates))]
    X_te = X.loc[(X.index.get_level_values("date").isin(test_dates))]
    pred_no_shrink = fit_predict_fama_macbeth(X_tr, y_tr, X_te, shrinkage=0.0)
    pred_half = fit_predict_fama_macbeth(X_tr, y_tr, X_te, shrinkage=0.5)
    np.testing.assert_allclose(pred_half.to_numpy(), 0.5 * pred_no_shrink.to_numpy(), atol=1e-5)


def test_pipeline_dispatch_routes_fama_macbeth():
    """REGRESSION: _fit_and_predict_fold must accept model='fama_macbeth'
    and return the right shape."""
    from stockpred.config import CVConfig
    from stockpred.models.gbm import GBMConfig
    from stockpred.pipeline import _fit_and_predict_fold

    X, y = _make_two_factor_panel(n_days=20, n_tickers=15, noise=0.1, seed=7)
    X_tr = X.iloc[: int(0.7 * len(X))]
    y_tr = y.iloc[: int(0.7 * len(y))]
    X_te = X.iloc[int(0.7 * len(X)) :]
    pred = _fit_and_predict_fold("fama_macbeth", X_tr, y_tr, X_te, GBMConfig())
    assert len(pred) == len(X_te)
    # Predictions are not all zero
    assert pred.abs().sum() > 0
