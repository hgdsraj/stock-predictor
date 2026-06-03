"""Walk-forward CV correctness tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

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
        # Embargo: latest train date is at least embargo_days before earliest test.
        gap = (test_dates.min() - train_dates.max()).days
        assert gap >= 10, f"Embargo not respected; gap={gap}"
    assert n_folds > 10, f"Should produce many folds; got {n_folds}"


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
