"""Purged + embargoed walk-forward cross-validation.

Reference: Marcos López de Prado, *Advances in Financial Machine Learning*
(Chapter 7), with simplifications appropriate for daily horizon labels.

Why this matters:

When labels are *forward-looking* over h days, the label for date t depends on
prices through date t+h. A naive train/test split where train ends at t and
test starts at t+1 leaks: the last label in train uses prices that are also in
the test window. We fix this two ways:

1. **Purging**: drop from the training set any observation whose label window
   overlaps the test window.
2. **Embargo**: drop a small buffer after the test window before the next
   training fold begins, to prevent serial correlation leakage in the other
   direction.

The splitter yields (train_idx, test_idx) tuples for use with sklearn-style
fit/predict loops.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """Expanding-window walk-forward splitter."""

    train_years: int = 5
    test_months: int = 6
    embargo_days: int = 10  # should be >= max label horizon
    min_train_obs: int = 1000

    def split(self, dates: pd.DatetimeIndex) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) into `dates`.

        `dates` is the *sorted unique* set of observation dates.
        """
        dates = pd.DatetimeIndex(dates).unique().sort_values()
        if len(dates) == 0:
            return

        first = dates[0]
        # Initial train window end.
        train_end = first + pd.DateOffset(years=self.train_years)

        while train_end < dates[-1]:
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=self.test_months) - pd.Timedelta(days=1)
            if test_end > dates[-1]:
                test_end = dates[-1]

            # Embargo: cut a buffer at the end of train to prevent label leakage
            # from train into test.
            purge_cut = test_start - pd.Timedelta(days=self.embargo_days)

            train_mask = dates <= purge_cut
            test_mask = (dates >= test_start) & (dates <= test_end)

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) >= self.min_train_obs and len(test_idx) > 0:
                yield train_idx, test_idx

            # Slide forward by one test window.
            train_end = test_end


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
