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
