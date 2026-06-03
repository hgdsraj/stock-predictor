"""Lag-safe technical features.

Every feature computed for date `t` uses information available at or before
close of day `t`. We never use t+1 data; this is enforced by always shifting any
rolling computation by 0 (we *are* allowed to use the close on the signal day
itself, but the realised label refers to t+1 onward).

Features produced (long-form, index=[date, ticker]):
- ret_1d, ret_5d, ret_21d, ret_63d, ret_252d : trailing log returns
- vol_5d, vol_21d, vol_63d                   : rolling std of daily returns
- rsi_14                                     : Relative Strength Index
- macd, macd_signal, macd_hist               : 12/26/9 MACD
- bb_z_20                                    : z-score within 20d Bollinger band
- dist_high_252                              : (price - 252d high) / 252d high
- dist_low_252                               : (price - 252d low) / 252d low
- adv_21                                     : 21d avg dollar volume (log)
- skew_21, kurt_63                           : higher moments of daily returns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(
    close: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd - sig
    # Normalise by price for cross-sectional comparability.
    macd_n = macd / close
    sig_n = sig / close
    hist_n = hist / close
    return macd_n, sig_n, hist_n


def _bollinger_z(close: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    m = close.rolling(window, min_periods=window).mean()
    s = close.rolling(window, min_periods=window).std()
    return (close - m) / s


def compute_technical_features(
    close: pd.DataFrame, volume: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Return long-form DataFrame indexed by [date, ticker]."""
    if close.empty:
        return pd.DataFrame()

    log_close = np.log(close)
    log_ret = log_close.diff()

    feats: dict[str, pd.DataFrame] = {}

    # Trailing returns (already lag-safe — uses close up to and including t).
    for w in (1, 5, 21, 63, 252):
        feats[f"ret_{w}d"] = log_close - log_close.shift(w)

    # Realised volatility.
    for w in (5, 21, 63):
        feats[f"vol_{w}d"] = log_ret.rolling(w, min_periods=w).std()

    feats["rsi_14"] = _rsi(close, 14) / 100.0  # scale to ~[0,1]

    macd_n, sig_n, hist_n = _macd(close)
    feats["macd"] = macd_n
    feats["macd_signal"] = sig_n
    feats["macd_hist"] = hist_n

    feats["bb_z_20"] = _bollinger_z(close, 20)

    hi252 = close.rolling(252, min_periods=60).max()
    lo252 = close.rolling(252, min_periods=60).min()
    feats["dist_high_252"] = (close - hi252) / hi252
    feats["dist_low_252"] = (close - lo252) / lo252

    if volume is not None and not volume.empty:
        dv = (close * volume).replace(0, np.nan)
        feats["adv_21"] = np.log(dv.rolling(21, min_periods=10).mean())

    feats["skew_21"] = log_ret.rolling(21, min_periods=21).skew()
    feats["kurt_63"] = log_ret.rolling(63, min_periods=63).kurt()

    long_frames = []
    for name, df in feats.items():
        long_frames.append(df.stack(future_stack=True).rename(name))
    out = pd.concat(long_frames, axis=1)
    out.index = out.index.set_names(["date", "ticker"])
    return out
