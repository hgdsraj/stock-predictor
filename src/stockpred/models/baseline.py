"""Logistic regression baseline.

The whole point of this module is to give us a *transparent* baseline that
exposes leakage early. If the baseline scores >55% accuracy out-of-sample, we
should suspect a bug before celebrating.

Pipeline: median-impute -> standard-scale -> logistic regression.
Predicts probability of fwd_dir_h == 1 (i.e. positive forward return).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_baseline_pipeline(C: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs", random_state=42),
            ),
        ]
    )


def fit_predict_proba(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
) -> pd.Series:
    """Fit on (X_train, y_train), return predicted P(y=1) on X_test.

    Test rows that are entirely NaN in the original `X_test` (i.e. the model
    has no real features to act on) produce NaN predictions instead of
    base-rate predictions from the median imputer (review finding M3).
    """
    mask = y_train.notna()
    Xt = X_train.loc[mask]
    yt = y_train.loc[mask].astype(int)

    keep_cols = Xt.columns[Xt.notna().any(axis=0)]
    Xt = Xt[keep_cols]
    Xs = X_test[keep_cols]

    if Xt.empty or yt.nunique() < 2:
        return pd.Series(np.nan, index=X_test.index)

    # Identify test rows that have no usable signal at all -- they will get
    # NaN below regardless of imputer behaviour.
    all_nan_mask = Xs.isna().all(axis=1)

    pipe.fit(Xt.values, yt.values)
    proba = pipe.predict_proba(Xs.values)[:, 1]
    out = pd.Series(proba, index=X_test.index, name="proba_up")
    out[all_nan_mask] = np.nan
    return out
