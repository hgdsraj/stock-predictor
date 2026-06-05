"""LightGBM regressor for forward returns (used as a soft predictor).

Why regress on forward return instead of classifying direction?
- Magnitude carries signal: a 5% predicted return is much stronger evidence than 0.1%.
- The portfolio constructor can rank by predicted return directly.
- We can derive binary direction from sign(prediction) when needed.

We use LightGBM because:
- Handles missing values natively (no imputation needed).
- Fast on tabular data, robust to feature scaling.
- Industry workhorse for tabular finance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import lightgbm as lgb

# NOTE: ``lightgbm`` is imported lazily inside the functions that actually
# train a model (see train_gbm). LightGBM dlopen's a native OpenMP runtime
# (libomp) at import time; importing it at module top would make the whole
# backend fail to boot on machines without libomp installed — even for paths
# that never train a model (e.g. serving the dashboard, or seeding synthetic
# data). Keeping the import lazy lets everything except actual training run
# without the native dependency.


@dataclass
class GBMConfig:
    num_leaves: int = 63
    learning_rate: float = 0.03
    n_estimators: int = 800
    min_data_in_leaf: int = 200
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    reg_lambda: float = 1.0
    objective: str = "regression"
    metric: str = "l2"
    verbose: int = -1
    random_state: int = 42
    early_stopping_rounds: int | None = 50
    extra_params: dict = field(default_factory=dict)


def train_gbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_valid: pd.DataFrame | None = None,
    y_valid: pd.Series | None = None,
    cfg: GBMConfig | None = None,
) -> lgb.Booster:
    import lightgbm as lgb

    cfg = cfg or GBMConfig()
    mask = y_train.notna()
    Xt = X_train.loc[mask]
    yt = y_train.loc[mask]

    train_set = lgb.Dataset(Xt.values, label=yt.values, feature_name=list(Xt.columns))
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = []
    if X_valid is not None and y_valid is not None:
        vmask = y_valid.notna()
        Xv = X_valid.loc[vmask]
        yv = y_valid.loc[vmask]
        if not Xv.empty:
            valid_sets.append(
                lgb.Dataset(
                    Xv.values, label=yv.values, feature_name=list(Xv.columns), reference=train_set
                )
            )
            valid_names.append("valid")
            if cfg.early_stopping_rounds:
                callbacks.append(lgb.early_stopping(cfg.early_stopping_rounds, verbose=False))

    params = {
        "objective": cfg.objective,
        "metric": cfg.metric,
        "num_leaves": cfg.num_leaves,
        "learning_rate": cfg.learning_rate,
        "min_data_in_leaf": cfg.min_data_in_leaf,
        "feature_fraction": cfg.feature_fraction,
        "bagging_fraction": cfg.bagging_fraction,
        "bagging_freq": cfg.bagging_freq,
        "lambda_l2": cfg.reg_lambda,
        "verbose": cfg.verbose,
        "seed": cfg.random_state,
        **cfg.extra_params,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=cfg.n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    return booster


def predict_gbm(booster: lgb.Booster, X: pd.DataFrame) -> pd.Series:
    pred = booster.predict(X.values, num_iteration=booster.best_iteration or None)
    return pd.Series(np.asarray(pred), index=X.index, name="pred")
