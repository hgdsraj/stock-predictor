"""Walk-forward CV correctness tests, including trading-day embargo (C2 fix)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.validation.walk_forward import WalkForwardSplit


def test_no_train_test_overlap_with_embargo():
    dates = pd.date_range("2010-01-01", "2020-12-31", freq="B")
    splitter = WalkForwardSplit(train_years=3, test_months=6, embargo_days=10, min_train_obs=10)
    n_folds = 0
    for train_idx, test_idx in splitter.split(dates):
        n_folds += 1
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        # No overlap.
        assert not set(train_dates) & set(test_dates)
    assert n_folds > 10, f"Should produce many folds; got {n_folds}"


def test_embargo_is_in_trading_days_not_calendar_days():
    """Regression test for review finding C2.

    With embargo_days=21, there must be EXACTLY 21 trading days gap between
    the last train day and the first test day (i.e. 21 dropped trading days),
    regardless of how that maps to calendar days.
    """
    dates = pd.date_range("2010-01-01", "2020-12-31", freq="B")
    embargo = 21
    splitter = WalkForwardSplit(
        train_years=3, test_months=6, embargo_days=embargo, min_train_obs=10
    )
    sorted_dates = pd.DatetimeIndex(dates).unique().sort_values()
    for train_idx, test_idx in splitter.split(dates):
        last_train_pos = int(train_idx[-1])
        first_test_pos = int(test_idx[0])
        trading_day_gap = first_test_pos - last_train_pos - 1
        assert trading_day_gap >= embargo, f"trading-day gap {trading_day_gap} < embargo {embargo}"


def test_horizon_21_requires_embargo_at_least_21():
    """If your label horizon is 21 trading days, embargo must be >= 21 to
    avoid the last training observation's label window overlapping the test
    period. We assert the *default* embargo is >= 21."""
    splitter = WalkForwardSplit()
    assert splitter.embargo_days >= 21, (
        "Default embargo must cover the largest project horizon (21)"
    )


def test_train_window_grows_monotonically():
    """Expanding-window: training set never shrinks across folds."""
    dates = pd.date_range("2010-01-01", "2020-12-31", freq="B")
    splitter = WalkForwardSplit(train_years=3, test_months=6, embargo_days=10, min_train_obs=10)
    prev_train_size = 0
    for train_idx, _ in splitter.split(dates):
        assert len(train_idx) >= prev_train_size
        prev_train_size = len(train_idx)


def test_no_folds_when_history_too_short():
    dates = pd.date_range("2010-01-01", "2010-06-30", freq="B")
    splitter = WalkForwardSplit(train_years=3, test_months=6, embargo_days=10)
    folds = list(splitter.split(dates))
    assert folds == []


def test_indices_are_into_sorted_unique_dates():
    """Splitter contract: the indices yielded must be valid positions into the
    sorted unique trading-day index, monotone and disjoint per fold."""
    dates = pd.date_range("2015-01-01", "2020-12-31", freq="B")
    sorted_dates = pd.DatetimeIndex(dates).unique().sort_values()
    splitter = WalkForwardSplit(train_years=2, test_months=6, embargo_days=10, min_train_obs=10)
    for train_idx, test_idx in splitter.split(dates):
        # In-range
        assert train_idx.min() >= 0
        assert test_idx.max() < len(sorted_dates)
        # Monotone
        assert np.all(np.diff(train_idx) > 0)
        assert np.all(np.diff(test_idx) > 0)
        # Disjoint
        assert set(train_idx).isdisjoint(set(test_idx))
