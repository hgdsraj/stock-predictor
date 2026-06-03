"""Forward-return labels.

A label at time t for horizon h is the return realised over (t, t+h] using
adjusted close prices. We compute both:

  fwd_return_h:    log return over the horizon
  fwd_direction_h: binary 1 if return > 0 else 0

We deliberately exclude the *open of day t+1* in our label computation; for a
strategy that trades end-of-day on signal generation, we'd actually realise the
return from close[t+1] to close[t+1+h]. We expose `t_plus_one` shift in helper
to make this explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.config import DEFAULT


def compute_forward_returns(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT.horizons.horizons,
    *,
    trade_next_open: bool = True,
) -> dict[int, pd.DataFrame]:
    """Compute log forward returns per horizon.

    Parameters
    ----------
    prices : wide DataFrame of adjusted close (index=date, columns=tickers)
    horizons : tuple of forward-window lengths in trading days
    trade_next_open : if True, returns are close[t+1+h] / close[t+1].
                      If False, close[t+h] / close[t] (look-ahead at the
                      *signal-generation* timestamp itself; bad for live use).

    Returns
    -------
    {h: DataFrame of log forward returns, same shape as prices}
    """
    if prices.empty:
        return {h: prices.copy() for h in horizons}

    log_p = np.log(prices)
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        if trade_next_open:
            # Realised return from t+1 to t+1+h, attributed to signal at t.
            r = log_p.shift(-(1 + h)) - log_p.shift(-1)
        else:
            r = log_p.shift(-h) - log_p
        out[h] = r
    return out


def to_binary(returns: pd.DataFrame) -> pd.DataFrame:
    """1 if return > 0 else 0; NaN preserved."""
    binary = (returns > 0).astype("float32")
    binary[returns.isna()] = np.nan
    return binary


def long_labels(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT.horizons.horizons,
    *,
    trade_next_open: bool = True,
) -> pd.DataFrame:
    """Long-form labels: index=[date, ticker], columns=[fwd_return_{h}, fwd_dir_{h}]."""
    fwd = compute_forward_returns(prices, horizons, trade_next_open=trade_next_open)
    frames = []
    for h, ret_df in fwd.items():
        ret_long = ret_df.stack(future_stack=True).rename(f"fwd_return_{h}").to_frame()
        ret_long[f"fwd_dir_{h}"] = (ret_long[f"fwd_return_{h}"] > 0).astype("float32")
        ret_long.loc[ret_long[f"fwd_return_{h}"].isna(), f"fwd_dir_{h}"] = np.nan
        frames.append(ret_long)
    out = pd.concat(frames, axis=1)
    out.index = out.index.set_names(["date", "ticker"])
    return out
