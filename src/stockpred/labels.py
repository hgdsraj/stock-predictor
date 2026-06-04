"""Forward-return labels.

For each (date `t`, ticker) and horizon `h`, we produce the realised return
over the forward window. With `trade_next_open=True` (the project default and
the only honest choice for live trading), the window is `(close[t+1], close[t+1+h]]`
— i.e. the model "predicts" a return it could actually capture by trading at
the next session's open after generating the signal.

Three label shapes are provided:

* **fwd_return_h** — raw log return over the horizon. Used as the regression
  target for LightGBM (signal-strength preserved → better portfolio ranking).
* **fwd_dir_h** — binary 1 if return > 0 else 0. Used by the logistic baseline
  for diagnostics / hit-rate metrics.
* **fwd_vol_scaled_h** — log return divided by trailing volatility. Targets
  are unit-comparable across stocks and across horizons, which lets us train
  a single ensemble across horizons without one horizon dominating.
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
    prices : wide DataFrame of adjusted close (index=date, columns=tickers).
    horizons : tuple of forward-window lengths in trading days.
    trade_next_open :
        * True  -> realise close[t+1+h] / close[t+1]
                   (correct for live trading; the project default).
        * False -> realise close[t+h] / close[t]
                   (uses the signal-day close as the entry price — a form of
                    same-day lookahead, OK for analytical comparisons only).
    """
    if prices.empty:
        return {h: prices.copy() for h in horizons}

    log_p = np.log(prices)
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        if trade_next_open:
            r = log_p.shift(-(1 + h)) - log_p.shift(-1)
        else:
            r = log_p.shift(-h) - log_p
        out[h] = r
    return out


def compute_vol_scaled_forward_returns(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT.horizons.horizons,
    *,
    vol_window: int = 21,
    trade_next_open: bool = True,
) -> dict[int, pd.DataFrame]:
    """Forward log returns divided by trailing realised vol (per ticker).

    *** Phase 6 leakage fix (P6L1): the trailing-vol denominator is shifted
    by +1 day so it uses only returns strictly through close-of-(t-1). ***

    The previous version used returns through close-of-t in the denominator,
    which created a same-day shared-input leak with features like `ret_1d`
    that also use close-of-t. A tree model could learn "when the most recent
    return is large, the divisor is large, so the target is smaller in
    magnitude" — getting most of its 'predictive' power from the divisor
    rather than the forward return. The Phase 6 leakage audit (scripts/
    leakage_audit.py) found the h=5d IC IR fell from +2.45 to -0.58 after
    applying a strict t-1 cutoff to features; this was the cause.

    With this fix, the denominator only uses information known one day
    before the signal date, eliminating the shared-input correlation.
    """
    if prices.empty:
        return {h: prices.copy() for h in horizons}

    log_p = np.log(prices)
    daily_log_ret = log_p.diff()
    # Lag-safe denominator: shift forward 1 day so the value at date t uses
    # only daily returns through close-of-(t-1).
    trailing_vol = daily_log_ret.rolling(vol_window, min_periods=vol_window).std().shift(1)
    trailing_vol = trailing_vol.where(trailing_vol > 1e-6)

    fwd = compute_forward_returns(prices, horizons, trade_next_open=trade_next_open)
    return {h: r / (trailing_vol * np.sqrt(h)) for h, r in fwd.items()}


def to_binary(returns: pd.DataFrame) -> pd.DataFrame:
    """1 if return > 0 else 0; NaN preserved; exact-zero returns left as NaN
    so they don't bias the binary toward 'down' (review finding H6)."""
    binary = (returns > 0).astype("float32")
    # Mark exact zero as NaN to be conservative.
    binary[returns.isna() | (returns == 0)] = np.nan
    return binary


def long_labels(
    prices: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT.horizons.horizons,
    *,
    trade_next_open: bool = True,
    include_vol_scaled: bool = True,
    vol_window: int = 21,
) -> pd.DataFrame:
    """Long-form labels: index=[date, ticker], columns=[fwd_return_{h}, fwd_dir_{h}, fwd_vs_{h}]."""
    fwd = compute_forward_returns(prices, horizons, trade_next_open=trade_next_open)
    vs = (
        compute_vol_scaled_forward_returns(
            prices, horizons, vol_window=vol_window, trade_next_open=trade_next_open
        )
        if include_vol_scaled
        else {}
    )
    frames: list[pd.DataFrame] = []
    for h, ret_df in fwd.items():
        ret_long = ret_df.stack(future_stack=True).rename(f"fwd_return_{h}").to_frame()
        ret_long[f"fwd_dir_{h}"] = (ret_long[f"fwd_return_{h}"] > 0).astype("float32")
        zero_or_nan = ret_long[f"fwd_return_{h}"].isna() | (ret_long[f"fwd_return_{h}"] == 0)
        ret_long.loc[zero_or_nan, f"fwd_dir_{h}"] = np.nan
        if h in vs:
            vs_long = vs[h].stack(future_stack=True).rename(f"fwd_vs_{h}")
            ret_long[f"fwd_vs_{h}"] = vs_long
        frames.append(ret_long)
    out = pd.concat(frames, axis=1)
    out.index = out.index.set_names(["date", "ticker"])
    return out
