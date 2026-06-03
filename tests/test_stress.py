"""Phase 4 stress-test utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.validation.stress import (
    bootstrap_sharpe,
    holdout_split_dates,
    regime_breakdown,
    sensitivity_grid,
    spy_regime,
    vix_regime,
)


def test_holdout_split_dates_partitions_chronologically():
    dates = pd.bdate_range("2015-01-01", "2024-12-31")
    dev, hold = holdout_split_dates(dates, holdout_years=2)
    assert len(dev) + len(hold) == len(dates)
    assert dev.max() < hold.min()
    # ~2 years of business days = ~500 trading days
    assert 450 < len(hold) < 540


def test_bootstrap_sharpe_zero_returns_yield_nan():
    r = pd.Series([0.0] * 200)
    out = bootstrap_sharpe(r, n_resamples=200)
    assert np.isnan(out["sharpe"]) or out["sharpe"] == 0


def test_bootstrap_sharpe_recovers_known_mean():
    """A series with strong positive mean should produce CI containing the
    point estimate, and a clearly positive Sharpe."""
    rng = np.random.default_rng(42)
    # Strong daily Sharpe: mean=0.003, sd=0.01 -> annualised Sharpe ~4.7.
    # Bootstrap CI should easily contain the point estimate.
    r = pd.Series(rng.normal(0.003, 0.01, size=2000))
    out = bootstrap_sharpe(r, n_resamples=500, rng_seed=1)
    assert out["sharpe_lo"] <= out["sharpe"] <= out["sharpe_hi"]
    assert out["sharpe"] > 2.0
    # CI width is reasonable (not a single point or implausibly tight).
    assert (out["sharpe_hi"] - out["sharpe_lo"]) > 0.2


def test_sensitivity_grid_iterates_all_combos():
    """Make sure the grid expands correctly and metrics are collected."""

    def fake_run(a, b, c):
        return {"metrics": {"ann_return": a + b * c, "sharpe": float(a), "max_drawdown": -0.1}}

    df = sensitivity_grid(
        fake_run,
        base_kwargs={"a": 0},
        param_grid={"b": [1, 2], "c": [3, 4]},
    )
    assert len(df) == 4
    assert set(df.columns) >= {"b", "c", "ann_return", "sharpe", "max_drawdown"}


def test_regime_breakdown_splits_returns_by_label():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=400)
    returns = pd.Series(rng.normal(0.0002, 0.01, size=400), index=dates)
    # Synthetic regime: first half "low", second half "high".
    regime = pd.Series(["low"] * 200 + ["high"] * 200, index=dates, name="regime")
    out = regime_breakdown(returns, regime)
    assert set(out.index) == {"low", "high"}
    assert "sharpe" in out.columns and "hit" in out.columns


def test_vix_regime_buckets():
    rng = np.random.default_rng(1)
    vix = pd.Series(rng.uniform(10, 50, size=1000))
    r = vix_regime(vix, q=4)
    assert r.nunique() == 4
