"""Portfolio construction from cross-sectional predictions.

Strategy: dollar-neutral long/short, equal-weighted within each side.
- Each day, rank all stocks by predicted score.
- Long the top-k (highest predicted), short the bottom-k (lowest predicted).
- Each leg is equal-weighted to total $1; net = $0, gross = $2.

This is the canonical evaluation harness for cross-sectional equity signals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def top_bottom_k_weights(
    preds: pd.Series,
    *,
    k: int | float = 50,
    leverage_per_side: float = 1.0,
) -> pd.DataFrame:
    """Build a wide weights DataFrame [date x ticker] from long predictions.

    Parameters
    ----------
    preds : Series indexed by [date, ticker] with predicted scores.
    k     : if int, count of names per side. If float in (0, 0.5), fraction of
            universe per side (e.g. 0.1 = top/bottom decile).
    leverage_per_side : gross exposure per leg (default 1.0 => 2x gross, 0 net).
    """
    if isinstance(preds, pd.DataFrame):
        preds = preds.iloc[:, 0]
    df = preds.dropna().to_frame("pred")
    df.index = df.index.set_names(["date", "ticker"])

    def _per_day(group: pd.DataFrame) -> pd.Series:
        if isinstance(k, float) and 0 < k < 0.5:
            kk = max(1, int(len(group) * k))
        else:
            kk = int(k)
        if len(group) < 2 * kk:
            return pd.Series(dtype=float)
        ranked = group["pred"].rank(method="first")
        # Bottom kk: short; top kk: long.
        long_thresh = len(group) - kk
        short_thresh = kk
        weights = pd.Series(0.0, index=group.index)
        weights[ranked > long_thresh] = leverage_per_side / kk
        weights[ranked <= short_thresh] = -leverage_per_side / kk
        return weights

    w = df.groupby(level="date", group_keys=False).apply(_per_day)
    if w.empty:
        return pd.DataFrame()
    # Convert to wide [date x ticker].
    w.index = w.index.set_names(["date", "ticker"])
    wide = w.unstack("ticker").fillna(0.0)
    return wide
