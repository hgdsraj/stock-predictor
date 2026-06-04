"""Triple-barrier labels (López de Prado, *Advances in Financial Machine
Learning*, Ch. 3).

For each (date, ticker) we set three barriers from the date-t price:
  - upper:  +k_up * sigma_t (profit target)
  - lower:  -k_dn * sigma_t (stop loss)
  - vertical: t + max_horizon trading days (time limit)

The label is:
  +1 if upper hit first
   0 if vertical hit first (no clear direction)
  -1 if lower hit first

`sigma_t` is the trailing realised daily-return vol over `vol_window` days,
STRICTLY through close-of-(t-1) (lag-safe; same convention as
compute_vol_scaled_forward_returns in labels.py after the P6L1 fix).

Implementation notes:
  - We iterate per-ticker (vectorising the path-dependent first-touch is
    non-trivial and the per-ticker loop is fast enough for daily bars).
  - The barriers are measured in *log returns from close[t+1]*, matching
    the trade_next_open convention. (Entry at close[t+1], exit at close
    of whichever bar first crosses a barrier or at close[t + max_horizon].)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripleBarrierConfig:
    max_horizon: int = 10
    k_up: float = 2.0
    k_dn: float = 2.0
    vol_window: int = 21
    min_vol: float = 1e-6


def _per_ticker_triple_barrier(log_prices: pd.Series, cfg: TripleBarrierConfig) -> pd.DataFrame:
    """Compute triple-barrier labels for one ticker.

    Parameters
    ----------
    log_prices : Series indexed by date (sorted asc) of log(close).

    Returns
    -------
    DataFrame with columns:
      label       : -1, 0, +1
      barrier     : 'up', 'down', 'vert'
      hit_offset  : number of trading days from t+1 to first hit (or max_horizon)
      hit_return  : realised log return from close[t+1] to close[t+1+hit_offset]
    """
    log_p = log_prices.dropna().sort_index()
    n = len(log_p)
    if n < cfg.vol_window + cfg.max_horizon + 2:
        return pd.DataFrame(
            index=log_p.index, columns=["label", "barrier", "hit_offset", "hit_return"]
        )

    daily_ret = log_p.diff()
    # Lag-safe vol (through close of t-1)
    vol = daily_ret.rolling(cfg.vol_window, min_periods=cfg.vol_window).std().shift(1)
    vol = vol.where(vol > cfg.min_vol)

    arr_logp = log_p.to_numpy()
    arr_vol = vol.to_numpy()

    labels = np.full(n, np.nan)
    barriers = np.array([""] * n, dtype=object)
    offsets = np.full(n, np.nan)
    hit_rets = np.full(n, np.nan)

    H = cfg.max_horizon
    for t in range(n - H - 1):
        sigma = arr_vol[t]
        if not np.isfinite(sigma):
            continue
        entry = arr_logp[t + 1]
        if not np.isfinite(entry):
            continue
        # Path of log-returns from t+1 forward.
        path = arr_logp[t + 2 : t + 1 + H + 1] - entry
        if path.size == 0:
            continue
        up = cfg.k_up * sigma
        dn = -cfg.k_dn * sigma
        # First crossing index in the path (1-indexed offset from t+1).
        first_up = np.argmax(path >= up) if np.any(path >= up) else -1
        first_dn = np.argmax(path <= dn) if np.any(path <= dn) else -1
        # argmax returns 0 if no True; use np.any to disambiguate.
        if first_up < 0 and first_dn < 0:
            # Vertical barrier
            offset = path.size  # H days
            labels[t] = 0
            barriers[t] = "vert"
            offsets[t] = offset
            hit_rets[t] = path[-1]
        elif first_dn < 0 or (first_up >= 0 and first_up < first_dn):
            offset = first_up + 1  # 1-indexed
            labels[t] = 1
            barriers[t] = "up"
            offsets[t] = offset
            hit_rets[t] = path[first_up]
        else:
            offset = first_dn + 1
            labels[t] = -1
            barriers[t] = "down"
            offsets[t] = offset
            hit_rets[t] = path[first_dn]

    return pd.DataFrame(
        {
            "label": labels,
            "barrier": barriers,
            "hit_offset": offsets,
            "hit_return": hit_rets,
        },
        index=log_p.index,
    )


def compute_triple_barrier_labels(
    prices: pd.DataFrame, cfg: TripleBarrierConfig | None = None
) -> pd.DataFrame:
    """Triple-barrier labels for a wide [date x ticker] price panel.

    Returns long-form DataFrame indexed by [date, ticker] with columns
    `tb_label`, `tb_barrier`, `tb_offset`, `tb_return`.
    """
    cfg = cfg or TripleBarrierConfig()
    if prices.empty:
        return pd.DataFrame()
    log_prices = np.log(prices)
    pieces: list[pd.DataFrame] = []
    for ticker in prices.columns:
        out = _per_ticker_triple_barrier(log_prices[ticker], cfg)
        out["ticker"] = ticker
        out.index.name = "date"
        pieces.append(out.reset_index())
    long_df = pd.concat(pieces, ignore_index=True).set_index(["date", "ticker"]).sort_index()
    long_df = long_df.rename(
        columns={
            "label": "tb_label",
            "barrier": "tb_barrier",
            "hit_offset": "tb_offset",
            "hit_return": "tb_return",
        }
    )
    return long_df
