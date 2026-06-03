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

    # Bounded-exact rank centering. For each (date, column), if there are n
    # non-NaN observations, we want the values to be uniformly spread over
    # the closed interval [-0.5, 0.5]: rank 1 -> -0.5, rank n -> +0.5.
    # The formula `(rank - 1) / (n - 1) - 0.5` achieves this exactly.
    #
    # n == 1 days are returned as 0.0 (no cross-section to rank against).
    grouped = features.groupby(level="date")
    raw_ranks = grouped[cols].rank(method="average")
    counts = grouped[cols].transform("count")
    # Avoid division by zero for n=1 days.
    denom = (counts - 1).where(counts > 1, 1.0)
    centered = (raw_ranks - 1) / denom - 0.5
    centered = centered.where(counts > 1, 0.0)
    centered.columns = [f"{c}{suffix}" for c in centered.columns]
    return features.join(centered)


def neutralise_by_sector(
    features: pd.DataFrame,
    sector_map: dict[str, str],
    cols: list[str],
    *,
    suffix: str = "_sn",
) -> pd.DataFrame:
    """Add sector-neutralised versions of `cols`.

    For each (date, sector) we subtract the cross-sectional mean of `cols`
    within that bucket. The original columns are preserved; neutralised
    versions are added with the configured suffix.

    Parameters
    ----------
    features : long-form DataFrame indexed by [date, ticker]
    sector_map : ticker -> sector string. Tickers absent from the map are
        treated as their own sector ("__OTHER__") which is a no-op group.
    cols : columns to neutralise.
    """
    if features.empty:
        return features
    out = features.copy()
    sectors = (
        pd.Series(out.index.get_level_values("ticker"), index=out.index)
        .map(sector_map)
        .fillna("__OTHER__")
    )
    dates = out.index.get_level_values("date")
    means = out[cols].groupby([dates, sectors]).transform("mean")
    for c in cols:
        out[f"{c}{suffix}"] = out[c] - means[c]
    return out


def add_sector_dummies(
    features: pd.DataFrame, sector_map: dict[str, str], *, prefix: str = "sec"
) -> pd.DataFrame:
    """Append one-hot sector membership columns (binary). Tree models can use
    these as splits; linear models will benefit from imputed sector bias."""
    if features.empty:
        return features
    out = features.copy()
    sectors = (
        pd.Series(out.index.get_level_values("ticker"), index=out.index)
        .map(sector_map)
        .fillna("__OTHER__")
    )
    dummies = pd.get_dummies(sectors, prefix=prefix, dtype="float32")
    dummies.index = out.index
    return out.join(dummies)
