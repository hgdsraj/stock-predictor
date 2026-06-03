"""Regression test for review finding M3.

A test-set row with no usable features must NOT receive a base-rate
prediction silently — it must be NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.models.baseline import fit_predict_proba, make_baseline_pipeline


def test_all_nan_test_row_returns_nan():
    rng = np.random.default_rng(0)
    n_train, n_features = 200, 4
    X_train = pd.DataFrame(
        rng.normal(size=(n_train, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    y_train = (rng.normal(size=n_train) > 0).astype(int)
    y_train = pd.Series(y_train, index=X_train.index)

    # Test set: 3 normal rows, 1 row of all-NaN.
    X_test = pd.DataFrame(
        rng.normal(size=(4, n_features)),
        columns=X_train.columns,
        index=["a", "b", "c", "d"],
    )
    X_test.loc["d"] = np.nan

    pipe = make_baseline_pipeline()
    proba = fit_predict_proba(pipe, X_train, pd.Series(y_train), X_test)

    assert proba.notna().sum() == 3
    assert pd.isna(proba.loc["d"])
