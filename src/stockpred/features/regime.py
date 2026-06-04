"""Phase 6: cross-asset regime features.

A single value per date broadcast to every (date, ticker) in the panel. The
tree model uses these as conditioning variables to learn when momentum vs
reversal dominates (Daniel & Moskowitz "Momentum Crashes", JFE 2016).

Sources (all free):
  - VIX (^VIX) from yfinance — equity vol regime
  - 10y-3m Treasury spread (FRED T10Y3M) — recession watch
  - USD index proxy via DTWEXBGS (FRED) — trade-weighted broad dollar
  - Cross-sectional return dispersion (computed from the panel itself)

The "broadcast" is implemented by merging on date in long-form.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from stockpred.data import macro as macro_mod
from stockpred.data import prices as prices_mod

log = logging.getLogger(__name__)


def fetch_vix(start: str, end: str | None = None) -> pd.Series:
    """Pull ^VIX close as a daily series."""
    try:
        df = prices_mod.fetch_one("^VIX", start=start, end=end)
        if df.empty or "adj_close" not in df.columns:
            return pd.Series(dtype=float, name="VIX")
        return df["adj_close"].rename("VIX")
    except Exception as e:  # noqa: BLE001
        log.warning("fetch_vix failed: %s", e)
        return pd.Series(dtype=float, name="VIX")


def cross_sectional_dispersion(close: pd.DataFrame) -> pd.Series:
    """Cross-sectional std of *daily* log returns across the universe.

    Lag-safe: at date t we use returns through close-of-t. (When fed to the
    model alongside features at date t, the time alignment is the same — we
    broadcast this scalar to every ticker for date t.)
    """
    daily = np.log(close).diff()
    return daily.std(axis=1).rename("xs_disp")


def compute_regime_features(
    close: pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return a wide DataFrame indexed by trading date with regime columns.

    Columns produced (all forward-filled to the trading-day index):
      - vix_level    : ^VIX close
      - vix_chg_5    : 5-day change in VIX
      - vix_chg_21   : 21-day change in VIX
      - term_spread  : 10y minus 3m Treasury (T10Y3M from FRED)
      - usd_chg_21   : 21-day pct change in DTWEXBGS
      - xs_disp_21   : 21-day rolling mean of cross-sectional return dispersion
    """
    if close.empty:
        return pd.DataFrame()

    start = start or str(close.index[0].date())
    end = end or str(close.index[-1].date())

    out = pd.DataFrame(index=close.index)

    # VIX
    vix = fetch_vix(start, end)
    if not vix.empty:
        vix_aligned = vix.reindex(close.index).ffill()
        out["vix_level"] = vix_aligned
        out["vix_chg_5"] = vix_aligned - vix_aligned.shift(5)
        out["vix_chg_21"] = vix_aligned - vix_aligned.shift(21)

    # FRED macro
    try:
        macro = macro_mod.fetch_macro(("T10Y3M", "DTWEXBGS"), start=start, end=end, refresh=refresh)
        if "T10Y3M" in macro.columns:
            out["term_spread"] = macro["T10Y3M"].reindex(close.index).ffill()
        if "DTWEXBGS" in macro.columns:
            usd = macro["DTWEXBGS"].reindex(close.index).ffill()
            out["usd_chg_21"] = usd.pct_change(21)
    except Exception as e:  # noqa: BLE001
        log.warning("macro fetch failed for regime features: %s", e)

    # Cross-sectional dispersion from the panel itself
    xs_raw = cross_sectional_dispersion(close)
    out["xs_disp_21"] = xs_raw.rolling(21, min_periods=10).mean()

    return out


def broadcast_to_panel(regime_wide: pd.DataFrame, panel_index: pd.MultiIndex) -> pd.DataFrame:
    """Take a date-indexed regime DataFrame and broadcast its columns to a
    [date, ticker] MultiIndex. Returns long-form DataFrame.

    Columns are prefixed with `reg_` so they don't collide with ticker-level
    feature names.
    """
    if regime_wide.empty:
        return pd.DataFrame(index=panel_index)
    aligned = regime_wide.copy()
    aligned.columns = [f"reg_{c}" for c in aligned.columns]
    # Build a long-form DataFrame: take the panel's (date, ticker) tuples,
    # look up regime values by date.
    dates = panel_index.get_level_values("date")
    long_df = aligned.reindex(dates).set_index(panel_index)
    return long_df
