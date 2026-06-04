"""Phase 6 tests: Tier-2 features, regime features, beta neutralisation,
block bootstrap, and the leakage-fix regression for vol-scaled labels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest.portfolio import neutralise_portfolio_beta
from stockpred.features.regime import (
    broadcast_to_panel,
    cross_sectional_dispersion,
)
from stockpred.features.tier2 import (
    amihud_illiquidity,
    beta_vs_bench,
    compute_tier2_features,
    idio_vol_vs_bench,
    max_daily_return,
    momentum_12_1,
    short_term_reversal,
)
from stockpred.labels import compute_vol_scaled_forward_returns
from stockpred.validation.stress import bootstrap_sharpe


def _panel(n_dates: int = 400, n_tickers: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    cols = [f"T{i}" for i in range(n_tickers)]
    px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(n_dates, n_tickers)), axis=0))
    return pd.DataFrame(px, index=dates, columns=cols)


# -------- Tier-2 features ---------------------------------------------------


def test_momentum_12_1_is_lag_safe():
    """Mutating future prices must not change momentum_12_1 at earlier dates."""
    close = _panel(400)
    mom = momentum_12_1(close)
    snap = mom.iloc[300].copy()
    close2 = close.copy()
    close2.iloc[310:] = close2.iloc[310:] * 2.0
    mom2 = momentum_12_1(close2)
    pd.testing.assert_series_equal(mom.iloc[300], snap, check_names=False)
    pd.testing.assert_series_equal(mom2.iloc[300], snap, check_names=False)


def test_st_reversal_sign_is_inverted_return():
    close = _panel(50)
    rev = short_term_reversal(close, window=5)
    # rev[t] should be the negative of log(close[t]/close[t-5])
    expected = -(np.log(close).diff(5))
    pd.testing.assert_frame_equal(rev, expected)


def test_max_daily_return_picks_largest_in_window():
    close = pd.DataFrame({"A": [100, 110, 99, 100, 100, 100]})
    m = max_daily_return(close, window=5)
    # Day 1: +10%; that's the max. After enough warmup it should be log(110/100).
    expected_max = np.log(110 / 100)
    assert abs(m["A"].iloc[5] - expected_max) < 1e-12


def test_amihud_high_when_volume_low():
    close = _panel(60)
    volume = pd.DataFrame(1e7, index=close.index, columns=close.columns)
    volume["T0"] = 1e5  # one illiquid name
    illiq = amihud_illiquidity(close, volume, window=21)
    # T0 should have systematically higher Amihud than the rest.
    avg = illiq.iloc[30:].mean()
    assert avg["T0"] > avg.drop("T0").max()


def test_beta_vs_bench_recovers_known_beta():
    """If asset returns are exactly 2 * bench returns + small noise, rolling
    beta should be ~2."""
    rng = np.random.default_rng(1)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    bench_ret = rng.normal(0.0005, 0.01, size=n)
    bench_close = pd.Series(100 * np.exp(np.cumsum(bench_ret)), index=dates, name="SPY")
    asset_ret = 2 * bench_ret + rng.normal(0, 0.001, size=n)
    asset_close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(asset_ret))}, index=dates)
    beta = beta_vs_bench(asset_close, bench_close, window=60)
    # Pick a date well past the warmup window.
    val = beta["A"].iloc[150]
    assert 1.8 < val < 2.2, f"beta {val} should be ~2"


def test_idio_vol_smaller_than_total_vol():
    """idio vol (residual std after removing beta*bench) should typically be
    smaller than total std for a name correlated with the bench. Pointwise
    residual isn't guaranteed to satisfy OLS orthogonality at every date
    (we use rolling beta then pointwise residual), so we relax to 'mean
    over a window'."""
    rng = np.random.default_rng(2)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    bench_ret = rng.normal(0, 0.01, size=n)
    bench_close = pd.Series(100 * np.exp(np.cumsum(bench_ret)), index=dates, name="SPY")
    asset_ret = 1.5 * bench_ret + rng.normal(0, 0.005, size=n)
    asset_close = pd.DataFrame({"A": 100 * np.exp(np.cumsum(asset_ret))}, index=dates)
    idio = idio_vol_vs_bench(asset_close, bench_close, window=60)
    total = np.log(asset_close).diff().rolling(60, min_periods=60).std()
    # Average idio across the post-warmup window should be smaller than total.
    mean_idio = idio["A"].iloc[120:].mean()
    mean_total = total["A"].iloc[120:].mean()
    assert mean_idio < mean_total, f"mean idio {mean_idio} not < mean total {mean_total}"


def test_compute_tier2_features_shape_and_columns():
    close = _panel(300)
    vol = pd.DataFrame(1e6, index=close.index, columns=close.columns)
    bench = close["T0"].rename("SPY")
    out = compute_tier2_features(close, vol, bench_close=bench)
    expected = {"mom_12_1", "st_reversal_5", "max_ret_21", "amihud_21", "beta_60", "idio_vol_60"}
    assert expected <= set(out.columns)


# -------- Regime features ---------------------------------------------------


def test_cross_sectional_dispersion_positive_with_random_panel():
    close = _panel(60, n_tickers=10, seed=42)
    disp = cross_sectional_dispersion(close)
    # Dispersion is std of daily returns across tickers; for random data > 0.
    assert disp.iloc[5:].mean() > 0


def test_broadcast_to_panel_aligns_dates_to_multiindex():
    dates = pd.bdate_range("2020-01-01", periods=5)
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    regime_wide = pd.DataFrame({"vix_level": [10, 11, 12, 13, 14]}, index=dates)
    out = broadcast_to_panel(regime_wide, idx)
    assert out.shape[0] == 10
    assert "reg_vix_level" in out.columns
    # On day 2, both A and B should see vix=12.
    np.testing.assert_array_equal(
        out.loc[(dates[2], slice(None)), "reg_vix_level"].values, [12, 12]
    )


# -------- Beta neutralisation ----------------------------------------------


def test_neutralise_portfolio_beta_reduces_portfolio_beta():
    dates = pd.bdate_range("2020-01-01", periods=3)
    weights = pd.DataFrame(
        {"A": [0.5, 0.5, 0.5], "B": [-0.5, -0.5, -0.5], "C": [0.0, 0.0, 0.0]},
        index=dates,
    )
    # A has high beta, B has low beta -> portfolio beta is positive.
    betas = pd.DataFrame(
        {"A": [1.5, 1.5, 1.5], "B": [0.5, 0.5, 0.5], "C": [1.0, 1.0, 1.0]},
        index=dates,
    )
    pre_beta = (weights * betas).sum(axis=1)
    out = neutralise_portfolio_beta(weights, betas, target=0.0)
    post_beta = (out * betas).sum(axis=1)
    # Neutralisation must materially reduce portfolio beta toward zero.
    assert abs(post_beta.mean()) < abs(pre_beta.mean()) / 2


# -------- Block bootstrap --------------------------------------------------


def test_block_bootstrap_returns_proper_dict():
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.001, 0.01, size=500))
    out = bootstrap_sharpe(r, n_resamples=200, method="block", block_length=10)
    assert out["method"] == "block"
    assert out["block_length"] == 10.0
    assert out["sharpe_lo"] <= out["sharpe"] <= out["sharpe_hi"]


def test_block_bootstrap_wider_than_iid_for_autocorrelated_series():
    """A series with strong positive autocorr should produce a wider CI with
    block bootstrap than with iid (the iid CI is artificially narrow)."""
    rng = np.random.default_rng(8)
    # AR(1) with phi=0.7 -> high autocorr
    n = 1000
    eps = rng.normal(0, 0.01, size=n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.7 * r[i - 1] + eps[i]
    s = pd.Series(r)
    iid = bootstrap_sharpe(s, n_resamples=400, method="iid", rng_seed=1)
    blk = bootstrap_sharpe(s, n_resamples=400, method="block", block_length=20, rng_seed=1)
    iid_width = iid["sharpe_hi"] - iid["sharpe_lo"]
    blk_width = blk["sharpe_hi"] - blk["sharpe_lo"]
    assert blk_width > iid_width, (
        f"block width {blk_width} should be > iid width {iid_width} for autocorr series"
    )


# -------- Leakage regression ------------------------------------------------


def test_vol_scaled_label_denominator_is_lag_safe():
    """Phase 6 leakage fix (P6L1): mutating close[t] must NOT change the
    vol-scaled target at date t.

    Previously the denominator used rolling std through close-of-t, sharing
    close[t] with features and creating a same-day leak."""
    rng = np.random.default_rng(99)
    n = 200
    dates = pd.bdate_range("2020-01-01", periods=n)
    px = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(n, 3)), axis=0))
    close = pd.DataFrame(px, index=dates, columns=["A", "B", "C"])

    vs = compute_vol_scaled_forward_returns(close, horizons=(5,), vol_window=21)[5]
    snap_at_100 = vs.iloc[100].copy()

    # Mutate close at date 100 (the signal date).
    close2 = close.copy()
    close2.iloc[100] = close2.iloc[100] * 1.5
    vs2 = compute_vol_scaled_forward_returns(close2, horizons=(5,), vol_window=21)[5]

    # Target at date 100 must be UNCHANGED if denominator is strictly t-1 lagged
    # AND numerator uses close[t+1] onwards. (The numerator does use close[t+1]
    # so a mutation at t+1 would change the label — but at t, no change.)
    # Note: numerator log(close[t+1+h]) - log(close[t+1]) does NOT touch close[t].
    pd.testing.assert_series_equal(vs.iloc[100], snap_at_100, check_names=False)
    pd.testing.assert_series_equal(vs2.iloc[100], snap_at_100, check_names=False)
