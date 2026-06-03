"""Purged + embargoed walk-forward cross-validation.

Reference: Marcos López de Prado, *Advances in Financial Machine Learning*
(Chapter 7), simplified for daily forward-horizon labels.

Why this matters
----------------
When labels are *forward-looking* over h trading days, the label for date t
depends on prices through date t+h. A naive train/test split where train ends
at t and test starts at t+1 leaks: the last label in train uses prices that
fall inside the test window. We defend with two devices:

1. **Purging.** Drop from training any observation whose label window overlaps
   the test window.
2. **Embargo.** Drop a small buffer between the end of training and the start
   of testing, to also prevent leakage from serial correlation in features.

**Embargo must be in trading days, not calendar days.** A 10 *calendar* day
buffer is only ~7 trading days; for a 21-trading-day horizon, that leaves
~14 days of overlap → real leakage. We use positional offsets into the sorted
unique trading-date index, which is unambiguous.

The splitter yields (train_idx, test_idx) tuples of positional indices into
the sorted unique date set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """Expanding-window walk-forward splitter.

    Attributes
    ----------
    train_years : initial training window length, in calendar years.
    test_months : test window length per fold, in calendar months.
    embargo_days : buffer between end of train and start of test, in
        **trading days**. Should be >= max forward-label horizon. Default 25
        comfortably covers horizons up to 21 trading days plus safety margin.
    min_train_obs : minimum number of training observations needed for a fold
        to be yielded.
    """

    train_years: int = 5
    test_months: int = 6
    embargo_days: int = 25  # trading days; should be >= max label horizon
    min_train_obs: int = 1000

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) into `dates` (sorted unique trading days)."""
        dates = pd.DatetimeIndex(dates).unique().sort_values()
        n = len(dates)
        if n == 0:
            return

        first = dates[0]
        train_end_calendar = first + pd.DateOffset(years=self.train_years)

        while train_end_calendar < dates[-1]:
            test_start_calendar = train_end_calendar + pd.Timedelta(days=1)
            test_end_calendar = (
                test_start_calendar + pd.DateOffset(months=self.test_months) - pd.Timedelta(days=1)
            )
            if test_end_calendar > dates[-1]:
                test_end_calendar = dates[-1]

            # Locate test window in trading-day positions.
            test_start_pos = int(dates.searchsorted(test_start_calendar, side="left"))
            test_end_pos = int(dates.searchsorted(test_end_calendar, side="right")) - 1

            if test_start_pos > test_end_pos:
                # Test window contains no trading days; advance.
                train_end_calendar = test_end_calendar
                continue

            # Embargo: cut N TRADING DAYS before the first test day.
            purge_pos = max(0, test_start_pos - self.embargo_days) - 1
            if purge_pos < 0:
                train_idx = np.array([], dtype=np.int64)
            else:
                train_idx = np.arange(0, purge_pos + 1, dtype=np.int64)

            test_idx = np.arange(test_start_pos, test_end_pos + 1, dtype=np.int64)

            if len(train_idx) >= self.min_train_obs and len(test_idx) > 0:
                yield train_idx, test_idx

            train_end_calendar = test_end_calendar


def time_aware_filter(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    date_level: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Given a multi-indexed df and integer positions over unique dates, return
    the row-subsets corresponding to those date positions."""
    all_dates = df.index.get_level_values(date_level).unique().sort_values()
    train_dates = all_dates[train_idx]
    test_dates = all_dates[test_idx]
    train_df = df[df.index.get_level_values(date_level).isin(train_dates)]
    test_df = df[df.index.get_level_values(date_level).isin(test_dates)]
    return train_df, test_df
