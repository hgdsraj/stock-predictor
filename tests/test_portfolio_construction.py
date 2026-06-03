"""Phase 3 portfolio-construction tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest.portfolio import (
    apply_min_trade_threshold,
    apply_sector_caps,
    ic_ir_weighted_ensemble,
    top_bottom_k_weights,
    vol_scaled_weights,
)


@pytest.fixture
def synthetic_score_and_vol():
    """Random scores and vols across 30 names x 5 days."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2020-01-01", periods=5)
    tickers = [f"T{i:02d}" for i in range(30)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx, name="score")
    vol = pd.DataFrame(
        rng.uniform(0.005, 0.05, size=(len(dates), len(tickers))),
        index=dates,
        columns=tickers,
    )
    return score, vol


def test_top_bottom_k_is_dollar_neutral(synthetic_score_and_vol):
    score, _ = synthetic_score_and_vol
    w = top_bottom_k_weights(score, k=5)
    # Net per day should be ~0.
    np.testing.assert_allclose(w.sum(axis=1).values, 0.0, atol=1e-12)
    # Gross per day should be ~ 2 * leverage_per_side (default 1.0) = 2.0
    np.testing.assert_allclose(w.abs().sum(axis=1).values, 2.0, atol=1e-12)


def test_vol_scaled_weights_normalise_per_leg(synthetic_score_and_vol):
    score, vol = synthetic_score_and_vol
    w = vol_scaled_weights(score, vol, leverage_per_side=1.0, top_fraction=0.2)
    # Each side gross ≈ 1.0 per day
    long_gross = w.clip(lower=0).sum(axis=1)
    short_gross = (-w.clip(upper=0)).sum(axis=1)
    np.testing.assert_allclose(long_gross.values, 1.0, atol=1e-12)
    np.testing.assert_allclose(short_gross.values, 1.0, atol=1e-12)


def test_vol_scaled_weights_low_vol_gets_bigger_weight():
    """A low-vol name on the same side gets a bigger weight than a high-vol one."""
    dates = pd.bdate_range("2020-01-01", periods=1)
    tickers = [f"T{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    # Scores: top 3 are T7, T8, T9 (long). T7 has 1/2 the vol of T8 and T9.
    scores = np.array([1, 2, 3, 4, 5, 6, 7, 100, 50, 50], dtype=float)
    score = pd.Series(scores, index=idx)
    vols_row = np.array([0.02] * 10)
    vols_row[7] = 0.01  # T7 has half the vol
    vol = pd.DataFrame([vols_row], index=dates, columns=tickers)

    w = vol_scaled_weights(score, vol, leverage_per_side=1.0, top_fraction=0.3)
    long_w = w.loc[dates[0]][w.loc[dates[0]] > 0]
    # T7 should have ~double the weight of T8 and T9 (because 1/0.01 = 2 * 1/0.02).
    # Normalisation makes T7 weight = 2/(2+1+1) = 0.5; T8 = T9 = 0.25.
    np.testing.assert_allclose(long_w["T7"], 0.5)
    np.testing.assert_allclose(long_w["T8"], 0.25)
    np.testing.assert_allclose(long_w["T9"], 0.25)


def test_sector_caps_shrink_overweight_sector():
    """If a sector's gross exceeds the cap, every weight in that sector
    shrinks proportionally; weights in other sectors are unchanged."""
    dates = pd.bdate_range("2020-01-01", periods=2)
    weights = pd.DataFrame(
        {"AA": 0.6, "AB": 0.4, "BA": 0.5},
        index=dates,  # AA+AB = sector "X", BA = sector "Y"
    )
    sector_map = {"AA": "X", "AB": "X", "BA": "Y"}
    capped = apply_sector_caps(weights, sector_map, max_per_sector_gross=0.5)
    # Sector X originally 1.0, capped at 0.5 -> scale 0.5
    np.testing.assert_allclose(capped["AA"].values, 0.6 * 0.5)
    np.testing.assert_allclose(capped["AB"].values, 0.4 * 0.5)
    # Sector Y under the cap -> unchanged
    np.testing.assert_allclose(capped["BA"].values, 0.5)


def test_min_trade_threshold_suppresses_small_moves():
    dates = pd.bdate_range("2020-01-01", periods=5)
    # Differences chosen to be unambiguous w.r.t. floating point.
    # day 1 vs 0: delta 0.001  (< 0.01)  -> keep prev
    # day 2 vs prev: delta 0.100 (>= 0.01) -> trade
    # day 3 vs prev: delta 0.020 (>= 0.01) -> trade
    # day 4 vs prev: delta 0.005 (<  0.01) -> keep prev
    weights = pd.DataFrame({"A": [0.100, 0.101, 0.200, 0.220, 0.225]}, index=dates)
    capped = apply_min_trade_threshold(weights, min_abs_delta=0.01)
    np.testing.assert_allclose(capped["A"].values, [0.100, 0.100, 0.200, 0.220, 0.220])


def test_ic_ir_weighted_ensemble_zeros_negative_horizons():
    """Horizons with negative IC IR should contribute zero weight."""
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=3), ["A", "B"]],
        names=["date", "ticker"],
    )
    preds = {
        1: pd.Series([1.0, -1.0, 0.5, -0.5, 0.3, -0.3], index=idx),
        5: pd.Series([1.0, -1.0, 0.5, -0.5, 0.3, -0.3], index=idx),
    }
    ic_ir = {1: 1.5, 5: -0.4}
    ens = ic_ir_weighted_ensemble(preds, ic_ir)
    # Per-day mean of cross-section should be 0 (because per-day z-scoring).
    per_day = ens.groupby(level="date").mean()
    np.testing.assert_allclose(per_day.values, 0.0, atol=1e-12)
