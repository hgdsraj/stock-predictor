"""Sanity tests for technical features and cross-sectional ranks."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.features.cross_sectional import add_cross_sectional_ranks
from stockpred.features.technical import compute_technical_features


def _toy_panel(n: int = 300, k: int = 4, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    cols = [f"T{i}" for i in range(k)]
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(n, k)), axis=0)),
        index=idx,
        columns=cols,
    )
    return px


def test_technical_features_are_lag_safe():
    """Mutating future prices must not change features at earlier dates."""
    px = _toy_panel()
    feats = compute_technical_features(px)
    snap = feats.xs(px.index[100], level="date").copy()

    px2 = px.copy()
    px2.iloc[150:] = px2.iloc[150:] * 5.0  # mutate the future
    feats2 = compute_technical_features(px2)
    snap2 = feats2.xs(px.index[100], level="date")

    pd.testing.assert_frame_equal(snap, snap2)


def test_cross_sectional_ranks_centered_at_zero():
    px = _toy_panel()
    feats = compute_technical_features(px)
    ranked = add_cross_sectional_ranks(feats, cols=["ret_5d"])
    # Average rank across the cross-section each day should be ~0.
    daily_mean = ranked["ret_5d_rank"].groupby(level="date").mean().dropna()
    assert daily_mean.abs().max() < 1e-9


def test_cross_sectional_rank_is_bounded():
    px = _toy_panel()
    feats = compute_technical_features(px)
    ranked = add_cross_sectional_ranks(feats, cols=["ret_5d"])
    r = ranked["ret_5d_rank"].dropna()
    assert r.min() >= -0.5 - 1e-9
    assert r.max() <= 0.5 + 1e-9


def test_cross_sectional_rank_hits_exact_bounds_with_full_cross_section():
    """Regression test for review finding C3.

    For a date with a full cross-section, the rank should map MIN -> -0.5 and
    MAX -> +0.5 exactly, not just approximately. The previous implementation
    used `rank(pct=True) - 0.5` which can never produce exactly -0.5.
    """
    # Toy panel with k=5 distinct values per day for 10 days.
    idx = pd.bdate_range("2020-01-01", periods=10)
    tickers = ["A", "B", "C", "D", "E"]
    # Strictly increasing values per day: A<B<C<D<E.
    rows = []
    for d in idx:
        for i, t in enumerate(tickers):
            rows.append((d, t, float(i)))
    feats = pd.DataFrame(rows, columns=["date", "ticker", "v"]).set_index(["date", "ticker"])
    ranked = add_cross_sectional_ranks(feats, cols=["v"])
    daily = ranked["v_rank"].unstack("ticker")
    # Lowest ticker A must be exactly -0.5, highest E exactly +0.5, every day.
    np.testing.assert_allclose(daily["A"].values, -0.5, atol=1e-12)
    np.testing.assert_allclose(daily["E"].values, 0.5, atol=1e-12)
    # The mean per day must be exactly 0.
    np.testing.assert_allclose(daily.mean(axis=1).values, 0.0, atol=1e-12)


def test_cross_sectional_rank_handles_single_observation_day():
    """A day with only one ticker should produce 0.0 (no cross-section)."""
    df = pd.DataFrame(
        {"v": [1.5]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2020-01-01"), "A")], names=["date", "ticker"]
        ),
    )
    ranked = add_cross_sectional_ranks(df, cols=["v"])
    assert ranked["v_rank"].iloc[0] == 0.0
