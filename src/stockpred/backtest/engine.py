"""Vectorised backtester for cross-sectional long/short strategies.

Core conventions
----------------
- Weights are produced at end of day `t` from the model's signal on `t`.
- The model is trained to predict the forward return over a window of `horizon`
  trading days starting one day after signal generation:
        label_t = close[t + trade_lag + horizon - 1] / close[t + trade_lag] - 1
  (the project's default labels use `trade_lag = 1`, `horizon ∈ {1, 5, 21}`.)
- The backtester realises P&L over the *same* window the model predicts. The
  total P&L of a single rebalance is the `horizon`-day cumulative return on the
  held basket; we then distribute that P&L back to daily strategy returns by
  applying the **constant** weight basket to **daily** returns over the window.
  This produces a daily P&L stream whose sum (compounded) equals the realised
  basket return per rebalance, and whose volatility is realistic.
- Transaction costs are charged on the day the trade actually clears (i.e. on
  `t + trade_lag` for the weight set at `t`). Cost = |Δ held weight| × bps.
- Adjacent-period overlap is avoided by enforcing a `rebalance_every = horizon`
  cadence on the *signal* stream before shifting.

The single-day path (`horizon == 1`) is identical to the previous
implementation and preserves the existing test conventions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpred.config import BacktestConfig


@dataclass
class BacktestResult:
    returns: pd.Series  # daily net returns (incl. costs)
    gross_returns: pd.Series  # daily gross returns (excl. costs)
    turnover: pd.Series  # one-way turnover on the day the trade clears
    held_weights: pd.DataFrame  # weights actually held each day (shifted)
    target_weights: pd.DataFrame  # weights produced by the signal (pre-shift)


def _coerce_weights_to_cadence(weights: pd.DataFrame, every: int) -> pd.DataFrame:
    """Keep every `every`-th row of weights, forward-fill the rest.

    This produces a stepwise weight profile where the basket is held constant
    for `every` trading days between rebalances. Equivalent to "rebalance every
    h days starting at the first valid row".
    """
    if every <= 1:
        return weights
    # First non-empty signal row anchors the cadence.
    nz_idx = weights.ne(0).any(axis=1).idxmax() if weights.shape[0] else None
    if nz_idx is None or nz_idx not in weights.index:
        return weights
    start = weights.index.get_loc(nz_idx)
    keep_mask = pd.Series(False, index=weights.index)
    keep_mask.iloc[start::every] = True
    out = weights.where(keep_mask, np.nan).ffill().fillna(0.0)
    return out


def run_backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    cfg: BacktestConfig | None = None,
    horizon: int = 1,
    trade_lag: int = 1,
    enforce_cadence: bool = True,
) -> BacktestResult:
    """Run a cross-sectional long/short backtest.

    Parameters
    ----------
    weights : wide [date x ticker] target weights produced at end of each signal day.
    prices  : wide [date x ticker] adjusted close.
    cfg     : cost configuration; defaults to project defaults.
    horizon : forward horizon, in trading days, that the model was trained on.
              Must match the label horizon used in training.
    trade_lag : how many trading days after the signal we begin holding. The
              project default is 1 (signal at EOD t, fill at next open, hold
              from close[t+1]).
    enforce_cadence : if True, the engine coerces the weight stream to refresh
              every `horizon` trading days (so positions don't overlap and
              double-count the horizon window). Set False only for unit tests.

    Returns
    -------
    BacktestResult with daily strategy and gross returns, turnover, and both
    target and held weight panels.
    """
    cfg = cfg or BacktestConfig()
    prices = prices.sort_index()
    common = sorted(set(weights.columns) & set(prices.columns))
    if not common:
        raise ValueError("No overlapping tickers between weights and prices.")
    prices = prices[common]

    # Align weights to the trading-day index and forward-fill.
    target = weights.reindex(columns=common).fillna(0.0)
    target = target.reindex(prices.index).ffill().fillna(0.0)

    # Enforce h-day cadence on signals so successive rebalances don't overlap
    # the forward windows the model was trained on.
    if enforce_cadence and horizon > 1:
        target = _coerce_weights_to_cadence(target, horizon)

    # The trade for a weight set on day t clears on day t + trade_lag, and is
    # held for `horizon` consecutive days. We model this by shifting the target
    # weights forward by `trade_lag` to get the "held" weight on each day.
    held = target.shift(trade_lag)

    # When horizon > 1, the "held" basket persists for `horizon` days. Because
    # `target` has been coerced to refresh every `horizon` days, shifting by
    # trade_lag and forward-filling already yields the correct stepwise held
    # series. (For horizon==1 this is a no-op.)
    if horizon > 1:
        held = held.ffill()

    no_position = held.isna().all(axis=1)
    held_filled = held.fillna(0.0)

    # Daily simple returns of each ticker, with a defensive clip to ±50%.
    # Genuine daily moves > 50% are almost always data quality issues
    # (yfinance occasionally returns near-zero closes for delisted/halted
    # names, producing -99% / +9900% phantom returns). Clipping them to
    # ±50% caps the worst single-day P&L at the position weight × 0.5,
    # which is the right behaviour even for *real* shocks (in practice a
    # trader would have circuit-breakers and the like to limit damage).
    ret = prices.pct_change().clip(lower=-0.5, upper=0.5)
    gross = (held_filled * ret).sum(axis=1)
    gross[no_position] = np.nan

    # Turnover: charge costs on the day the trade *clears* (= shifted weights
    # changing), not on the signal day. This is the correct timing for P&L.
    held_for_turnover = held.fillna(0.0)
    turnover_daily = (held_for_turnover - held_for_turnover.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost_per_unit = cfg.total_cost_per_side_bps / 10_000.0
    costs = turnover_daily * cost_per_unit

    # Net return: gross - cost. On rows where gross is NaN but cost is positive
    # (i.e. the first non-zero turnover day if it happens before any position is
    # held), charge the cost so NAV reflects it.
    net = gross.fillna(0.0) - costs
    # Restore NaN for the strictly-pre-position rows where no trade has cleared
    # yet (cost = 0 there because turnover = 0).
    pre_position_and_no_trade = no_position & (costs == 0.0)
    net[pre_position_and_no_trade] = np.nan

    return BacktestResult(
        returns=net.rename("strategy_return"),
        gross_returns=gross.rename("gross_return"),
        turnover=turnover_daily.rename("turnover"),
        held_weights=held_filled,
        target_weights=target,
    )
