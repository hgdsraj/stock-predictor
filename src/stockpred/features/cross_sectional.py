"""Cross-sectional ranking.

Many equity ML signals work much better when transformed to *daily cross-sectional
ranks* — e.g. "which stocks today have the highest momentum vs all others today".
This neutralises broad market regime shifts.

For each numeric column we add a `<col>_rank` column scaled to [-0.5, 0.5].
NaN inputs stay NaN.
"""

from __future__ import annotations

import pandas as pd


def add_cross_sectional_ranks(
    features: pd.DataFrame,
    cols: list[str] | None = None,
    *,
    suffix: str = "_rank",
) -> pd.DataFrame:
    """Add per-date rank columns for the given numeric columns.

    Parameters
    ----------
    features : long-form DataFrame indexed by [date, ticker]
    cols     : columns to rank (default: all numeric)
    """
    if features.empty:
        return features

    if cols is None:
        cols = features.select_dtypes("number").columns.tolist()

    grouped = features.groupby(level="date")
    # rank(pct=True) -> (0, 1]. Subtracting the *per-day mean* of the same
    # transform centers each day's distribution at 0 exactly, regardless of
    # the per-day count of non-NaN observations.
    pct_ranks = grouped[cols].rank(pct=True)
    day_means = pct_ranks.groupby(level="date").transform("mean")
    ranks = pct_ranks - day_means
    ranks.columns = [f"{c}{suffix}" for c in ranks.columns]
    return features.join(ranks)


def neutralise_by_sector(
    features: pd.DataFrame, sector_map: dict[str, str], cols: list[str]
) -> pd.DataFrame:
    """Subtract per-date per-sector mean from `cols`. Returns a copy.

    `sector_map`: ticker -> sector string.
    """
    if features.empty:
        return features
    out = features.copy()
    sectors = out.index.get_level_values("ticker").map(sector_map)
    out = out.assign(_sector=sectors)
    means = out.groupby([out.index.get_level_values("date"), "_sector"])[cols].transform("mean")
    for c in cols:
        out[c] = out[c] - means[c]
    return out.drop(columns=["_sector"])
