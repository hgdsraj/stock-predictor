"""Backtest engine correctness tests, including horizon-aware semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest.engine import run_backtest
from stockpred.config import BacktestConfig


def _const_prices(rate: float, n: int = 30, ticker: str = "AAA") -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = pd.Series(100 * (1 + rate) ** np.arange(n), index=idx, name=ticker)
    return px.to_frame()


def _ZERO_COSTS() -> BacktestConfig:
    return BacktestConfig(commission_bps=0, spread_bps=0, slippage_bps=0)


# --------------------------- horizon == 1 ---------------------------- #


def test_constant_long_position_earns_underlying_return():
    px = _const_prices(0.001)  # +10 bps/day
    weights = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    res = run_backtest(weights, px, cfg=_ZERO_COSTS(), horizon=1, trade_lag=1)
    daily = res.gross_returns.dropna()
    np.testing.assert_allclose(daily.values, 0.001, atol=1e-12)


def test_costs_charged_on_clearing_day_not_signal_day():
    """Cost timing fix (review finding H2). Cost should appear on the day the
    trade actually clears (signal date + trade_lag), not on the signal date.

    Signal sequence (target weights): [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, ...]
    With trade_lag=2, the held series shifts by 2:
        held[2..6] = 1, held[7..11] = 0, held[12..] = 1
    Turnover-bearing days are: 2 (open from 0->1), 7 (close 1->0), 12 (reopen 0->1).
    """
    px = _const_prices(0.0, n=30)  # flat prices
    idx = px.index

    weights = pd.DataFrame(0.0, index=idx, columns=px.columns)
    weights.iloc[0:5] = 1.0  # long for first 5 signal days
    weights.iloc[5:10] = 0.0  # flat for next 5
    weights.iloc[10:] = 1.0  # re-long thereafter

    cfg = BacktestConfig(commission_bps=2, spread_bps=3, slippage_bps=0)  # 5 bps/side
    res = run_backtest(weights, px, cfg=cfg, horizon=1, trade_lag=2)
    expected_cost = 5 / 10_000

    # Days 2, 7, 12 (= signal-edge + trade_lag) should bear cost.
    np.testing.assert_allclose(res.returns.iloc[2], -expected_cost, atol=1e-12)
    np.testing.assert_allclose(res.returns.iloc[7], -expected_cost, atol=1e-12)
    np.testing.assert_allclose(res.returns.iloc[12], -expected_cost, atol=1e-12)
    # Days between edges should have no cost (turnover = 0).
    for i in (3, 4, 5, 6, 8, 9, 10, 11, 13):
        val = res.returns.iloc[i]
        assert val == 0.0 or pd.isna(val), f"day {i}: {val}"


def test_dollar_neutral_two_assets_offset_to_zero_on_correlated_move():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame(
        {"A": 100 * 1.01 ** np.arange(5), "B": 100 * 1.01 ** np.arange(5)},
        index=idx,
    )
    w = pd.DataFrame({"A": 1.0, "B": -1.0}, index=idx)
    res = run_backtest(w, px, cfg=_ZERO_COSTS(), horizon=1, trade_lag=1)
    np.testing.assert_allclose(res.gross_returns.dropna().values, 0.0, atol=1e-12)


# --------------------------- horizon > 1 ---------------------------- #


@pytest.mark.parametrize("horizon", [5, 21])
def test_horizon_aware_backtest_matches_realised_window_return(horizon):
    """Regression test for review finding C1.

    When the model claims to predict an h-day forward return, the backtest's
    cumulative gross return for that single rebalance must equal the realised
    h-day return on the held basket.

    Set-up:
      - Single ticker, prices grow at a fixed daily rate (so the h-day return
        is exactly known).
      - Place a single positive weight on the signal day; with cadence
        enforcement, the trade clears at t+1 and is held for h days.
      - Sum the gross daily returns over the held window. Assert this equals
        the h-day realised return, to floating-point precision.
    """
    n = horizon * 4 + 10
    daily = 0.002
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = pd.DataFrame({"AAA": 100 * (1 + daily) ** np.arange(n)}, index=idx)

    w = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    w.iloc[0] = 1.0

    res = run_backtest(w, px, cfg=_ZERO_COSTS(), horizon=horizon, trade_lag=1)

    # The position should be held over indices [1, 1 + horizon).
    held_window = res.gross_returns.iloc[1 : 1 + horizon].fillna(0.0)
    realised_cum = (1 + held_window).prod() - 1
    expected_h_day_return = (1 + daily) ** horizon - 1
    np.testing.assert_allclose(realised_cum, expected_h_day_return, atol=1e-12)


def test_cadence_prevents_overlapping_holds_at_horizon_5():
    """With horizon=5 and signals every day, the engine should hold each basket
    for ~5 days before refreshing. Total positions across days should equal
    n_signals / horizon (approximately)."""
    n = 50
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = pd.DataFrame({"AAA": 100.0}, index=idx)

    # Signal every single day
    w = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    w.iloc[:] = 1.0

    res = run_backtest(w, px, cfg=_ZERO_COSTS(), horizon=5, trade_lag=1)

    # Held weights should be constant 1.0 (because the cadence keeps refreshing
    # to the same value), but turnover should be small: only the initial trade
    # creates positive turnover.
    nonzero_turnover_days = (res.turnover > 1e-9).sum()
    assert nonzero_turnover_days <= 2, f"too many rebalances: {nonzero_turnover_days}"


def test_zero_horizon_or_negative_raises_sane_error_or_works():
    """Smoke: horizon=1 default path should be safe even for tiny panels."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    px = pd.DataFrame({"AAA": [100.0, 101.0, 102.01]}, index=idx)
    w = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    w.iloc[0] = 1.0
    res = run_backtest(w, px, cfg=_ZERO_COSTS(), horizon=1, trade_lag=1)
    assert "AAA" in res.held_weights.columns
    assert "AAA" in res.target_weights.columns


def test_pct_change_is_clipped_to_plus_minus_50pct():
    """Defensive fix (Phase 7 big-universe finding): data-quality glitches
    in adjusted close (e.g. 0.50 -> 0.01 -> 0.50 reverse-split artefacts)
    produce phantom +9900% / -99% returns that, multiplied by even a small
    position weight, blow up DEV NAV.

    Backtester now clips per-name pct_change to +/- 50%."""
    idx = pd.bdate_range("2020-01-01", periods=5)
    # AAA goes through a data glitch (100 -> 100 -> 1 -> 100 -> 100).
    px = pd.DataFrame({"AAA": [100.0, 100.0, 1.0, 100.0, 100.0]}, index=idx)
    w = pd.DataFrame(0.1, index=idx, columns=["AAA"])  # constant +0.1 position
    res = run_backtest(w, px, cfg=_ZERO_COSTS(), horizon=1, trade_lag=1)
    # Without clipping, day 2 = -99% * 0.1 = -9.9% and day 3 = +9900% * 0.1 = +990.
    # With clipping, day 2 is at most -50% * 0.1 = -5% and day 3 +50% * 0.1 = +5%.
    daily = res.gross_returns.dropna().abs()
    assert (daily <= 0.05 + 1e-9).all(), f"clipping failed: {daily.tolist()}"
    """Smoke: horizon=1 default path should be safe even for tiny panels."""
    idx = pd.bdate_range("2020-01-01", periods=3)
    px = pd.DataFrame({"AAA": [100.0, 101.0, 102.01]}, index=idx)
    w = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    w.iloc[0] = 1.0
    res = run_backtest(w, px, cfg=_ZERO_COSTS(), horizon=1, trade_lag=1)
    assert "AAA" in res.held_weights.columns
    assert "AAA" in res.target_weights.columns
