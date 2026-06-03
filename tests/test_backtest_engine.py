"""Backtest engine sanity tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.backtest.engine import run_backtest
from stockpred.config import BacktestConfig


def _const_prices(rate: float, n: int = 30, ticker: str = "AAA") -> pd.DataFrame:
    idx = pd.bdate_range("2020-01-01", periods=n)
    px = pd.Series(100 * (1 + rate) ** np.arange(n), index=idx, name=ticker)
    return px.to_frame()


def test_constant_long_position_earns_underlying_return():
    px = _const_prices(0.001)  # +10 bps/day
    weights = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    cfg = BacktestConfig(commission_bps=0, spread_bps=0, slippage_bps=0)
    # trade_lag=1: assume fill at signal close (cleaner for this unit test).
    res = run_backtest(weights, px, cfg=cfg, trade_lag=1)
    # First day has no realised return (weight set EOD t, applies on t+1).
    daily = res.gross_returns.dropna()
    np.testing.assert_allclose(daily.values, 0.001, atol=1e-12)


def test_costs_reduce_return_on_turnover():
    """A full turnover day should be penalised by exactly cost_per_side bps * |dW|."""
    px = _const_prices(0.0)  # flat price
    idx = px.index
    # Day 0: +1; Day 1: 0; Day 2: +1; ... causes turnover of 1 on flip days.
    weights = pd.DataFrame(0.0, index=idx, columns=px.columns)
    weights.iloc[0] = 1.0
    weights.iloc[1] = 0.0  # close => turnover 1
    weights.iloc[2] = 1.0  # open  => turnover 1

    cfg = BacktestConfig(commission_bps=2, spread_bps=3, slippage_bps=0)  # 5 bps/side
    res = run_backtest(weights, px, cfg=cfg, trade_lag=1)
    # Gross returns are 0 (flat prices). Net should be -cost on turnover days.
    expected_cost = 5 / 10_000
    # Cost charged on the day the weight changes.
    assert res.returns.iloc[0] == -expected_cost  # initial position from 0->1
    assert res.returns.iloc[1] == -expected_cost  # 1 -> 0
    assert res.returns.iloc[2] == -expected_cost  # 0 -> 1


def test_trade_lag_2_matches_label_alignment():
    """When labels are computed with trade_next_open=True (the project default),
    the backtest must use trade_lag=2 so realisation == label window.

    Construct a clean setup: ticker AAA gets a positive weight on date 0. With
    trade_lag=2, the first non-zero P&L appears on day 2 (= close[1] -> close[2]
    return), matching the label horizon-1 forward return at date 0.
    """
    idx = pd.bdate_range("2020-01-01", periods=6)
    # Engineered returns: 0%, +1%, -2%, 0%, 0%, 0% (day 1 vs day 0 is +1%, etc.)
    rates = [0.0, 0.01, -0.02, 0.0, 0.0, 0.0]
    px = pd.Series(100.0, index=idx, name="AAA")
    for i in range(1, len(idx)):
        px.iloc[i] = px.iloc[i - 1] * (1 + rates[i])
    prices = px.to_frame()

    # Signal: long on day 0 only, then flat.
    w = pd.DataFrame(0.0, index=idx, columns=["AAA"])
    w.iloc[0] = 1.0

    cfg = BacktestConfig(commission_bps=0, spread_bps=0, slippage_bps=0)
    res = run_backtest(w, prices, cfg=cfg, trade_lag=2)
    gross = res.gross_returns

    # With trade_lag=2: weights from day 0 -> realised on day 2, i.e. the
    # close[1]->close[2] return = -2%. Days 0 and 1 should have no realised P&L.
    assert pd.isna(gross.iloc[0]) or gross.iloc[0] == 0.0
    assert pd.isna(gross.iloc[1]) or gross.iloc[1] == 0.0
    np.testing.assert_allclose(gross.iloc[2], -0.02, atol=1e-12)


def test_dollar_neutral_two_assets_offset_to_zero_on_correlated_move():
    idx = pd.bdate_range("2020-01-01", periods=5)
    px = pd.DataFrame(
        {"A": 100 * 1.01 ** np.arange(5), "B": 100 * 1.01 ** np.arange(5)},
        index=idx,
    )
    w = pd.DataFrame({"A": 1.0, "B": -1.0}, index=idx)
    cfg = BacktestConfig(commission_bps=0, spread_bps=0, slippage_bps=0)
    res = run_backtest(w, px, cfg=cfg, trade_lag=1)
    # Both assets move identically -> long/short offsets exactly.
    np.testing.assert_allclose(res.gross_returns.dropna().values, 0.0, atol=1e-12)
