"""Vectorised backtester for daily-rebalanced cross-sectional strategies.

Conventions:
- Weights at end of day t are intended positions for trading day t+1.
- PnL on day t+1 = sum(weights_t * return_t+1) - turnover_cost.
- Turnover cost = |w_t - w_{t-1}| * cost_per_side / 10000  (cost in bps).
- We assume close-to-close execution at adjusted prices. This is an
  approximation: real fills happen at the open or via MOC orders.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpred.config import BacktestConfig


@dataclass
class BacktestResult:
    returns: pd.Series  # daily strategy net returns
    gross_returns: pd.Series  # before costs
    turnover: pd.Series  # daily one-way turnover, fraction of gross
    weights: pd.DataFrame  # wide weights actually held (date-by-ticker)


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cfg: BacktestConfig | None = None,
    trade_lag: int = 2,
) -> BacktestResult:
    """Run a long/short backtest.

    Parameters
    ----------
    weights : wide [date x ticker], weights set at end of date `t`
              (the "signal date").
    prices  : wide [date x ticker] adjusted close.
    trade_lag : how many days separate signal generation from the realisation
                window. Must match the label definition used to train the model.
                * trade_lag=1 -> realise return from close[t] to close[t+1]
                  (lookahead: requires trading at signal close, generally not
                  achievable in live trading).
                * trade_lag=2 -> realise from close[t+1] to close[t+2]
                  (matches labels.compute_forward_returns(trade_next_open=True)).
    """
    cfg = cfg or BacktestConfig()
    prices = prices.sort_index()
    common_tickers = sorted(set(weights.columns) & set(prices.columns))
    if not common_tickers:
        raise ValueError("No overlapping tickers between weights and prices.")

    prices = prices[common_tickers]
    weights = weights.reindex(columns=common_tickers).fillna(0.0)

    # Daily returns: pct_change at date d represents close[d]/close[d-1]-1.
    ret = prices.pct_change()

    # Align weights to trading-day index, forward-fill until next rebalance.
    weights = weights.reindex(prices.index).ffill().fillna(0.0)

    # Shift weights by trade_lag: signal at EOD t, realised over the t+trade_lag
    # bar. With trade_lag=2 this means weights set on date t are used to compute
    # P&L on date t+2 (= return from close[t+1] to close[t+2]).
    held = weights.shift(trade_lag)
    no_position = held.isna().all(axis=1)
    held_filled = held.fillna(0.0)
    gross = (held_filled * ret).sum(axis=1)
    gross[no_position] = np.nan

    # Turnover and costs: weight change between rebalances. Costs are real on
    # every day weights change, including day 0 when we open the position.
    turnover_daily = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_per_unit = cfg.total_cost_per_side_bps / 10_000.0
    costs = turnover_daily * cost_per_unit

    # Net return: gross minus costs. If gross is NaN (e.g. day 0) but costs are
    # nonzero, charge the cost to NAV: returns reflect the cost of opening.
    net = gross.copy()
    net = net.where(~no_position, -costs[no_position])  # day-0-style rows
    has_position = ~no_position
    net[has_position] = gross[has_position] - costs[has_position]

    return BacktestResult(
        returns=net.rename("strategy_return"),
        gross_returns=gross.rename("gross_return"),
        turnover=turnover_daily.rename("turnover"),
        weights=held,
    )
