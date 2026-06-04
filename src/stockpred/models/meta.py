"""Meta-labelling (López de Prado Ch. 3.6).

Idea: train a *primary* model that decides direction (long/short). Then train
a *secondary* binary classifier that decides whether to *act* on the primary
signal — i.e. it predicts P(the primary signal will be correct | features).

Use:
  - Primary: any signed score from the existing pipeline (e.g. the IC-IR
    ensemble score, or a single-horizon GBM regressor).
  - Meta target: 1 if sign(primary_pred) == sign(realised_return), else 0.
  - Meta features: usually the primary score itself plus the regime / volatility
    features the primary model used (or a subset).

When the meta classifier's P(correct) is high we size up; when it's low we
skip the trade. The net effect is usually higher precision at the cost of
recall — exactly what you want when transaction costs dominate.

This module provides:
  - `build_meta_dataset` : assemble (X, y_meta) from primary preds and labels.
  - `fit_meta` and `predict_meta` : a LightGBM classifier for P(correct).
  - `meta_filter_signal` : apply a P(correct) threshold to gate trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpred.models.gbm import GBMConfig, predict_gbm, train_gbm

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetaConfig:
    p_threshold: float = 0.55  # only act when P(correct) > threshold
    use_primary_score: bool = True  # include primary score as a meta feature


_FORBIDDEN_META_COLS = frozenset(
    {
        "primary",
        "primary_pred",
        "primary_score",
        "realised",
        "realised_return",
        "fwd_return",
        "fwd_vs",
        "label",
    }
)


def build_meta_dataset(
    primary_pred: pd.Series,
    realised_return: pd.Series,
    features: pd.DataFrame,
    *,
    use_primary_score: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X_meta, y_meta) for meta-labelling.

    Parameters
    ----------
    primary_pred : Series indexed by [date, ticker] — signed signal from the
        primary model.
    realised_return : Series indexed by [date, ticker] — realised forward
        return for the same observations.
    features : long-form DataFrame indexed by [date, ticker] — the features
        you want the meta-classifier to use. **MUST NOT** contain the
        primary signal itself or any forward-looking column (raises
        ValueError on detected leakage names).
    use_primary_score : if True, append the absolute primary score as a
        meta feature (it usually adds signal: bigger conviction tends to be
        right more often).
    """
    # Defensive: reject obviously leaky feature names.
    bad = sorted(c for c in features.columns if c.lower() in _FORBIDDEN_META_COLS)
    if bad:
        raise ValueError(
            f"build_meta_dataset: features contain forbidden columns {bad}. "
            f"These would leak the primary signal or the realised target into "
            f"the meta model. Drop them before calling."
        )
    # Also reject any column that starts with 'fwd_' (forward-looking).
    fwd_like = [c for c in features.columns if c.startswith("fwd_")]
    if fwd_like:
        raise ValueError(
            f"build_meta_dataset: features contain forward-looking columns "
            f"{fwd_like}. These leak the label into the meta model."
        )
    aligned = pd.concat(
        [
            primary_pred.rename("primary"),
            realised_return.rename("realised"),
        ],
        axis=1,
    ).dropna()
    y_meta = (np.sign(aligned["primary"]) == np.sign(aligned["realised"])).astype(int)
    # Drop ties (realised == 0) — they're noise for the binary classifier.
    keep = aligned["realised"] != 0
    aligned = aligned[keep]
    y_meta = y_meta[keep]

    X_meta = features.loc[aligned.index].copy()
    if use_primary_score:
        X_meta["primary_abs"] = aligned["primary"].abs()
    return X_meta, y_meta


def fit_meta(X_train: pd.DataFrame, y_train: pd.Series, *, cfg: GBMConfig | None = None):
    """Fit a binary LightGBM classifier predicting P(primary signal is correct).

    Reuses the GBMConfig but switches the objective to 'binary'.
    """
    cfg = cfg or GBMConfig()
    cfg_binary = GBMConfig(
        num_leaves=cfg.num_leaves,
        learning_rate=cfg.learning_rate,
        n_estimators=cfg.n_estimators,
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        reg_lambda=cfg.reg_lambda,
        objective="binary",
        metric="binary_logloss",
        verbose=cfg.verbose,
        random_state=cfg.random_state,
        early_stopping_rounds=cfg.early_stopping_rounds,
        extra_params=cfg.extra_params,
    )
    return train_gbm(X_train, y_train.astype(float), cfg=cfg_binary)


def predict_meta(booster, X: pd.DataFrame) -> pd.Series:
    return predict_gbm(booster, X).clip(0.0, 1.0)


def meta_filter_signal(
    primary_pred: pd.Series,
    meta_proba: pd.Series,
    *,
    p_threshold: float = 0.55,
) -> pd.Series:
    """Zero out primary predictions where meta P(correct) < threshold.

    The surviving predictions keep their sign and magnitude; the model can
    still rank them in the portfolio constructor as usual. The threshold
    just refuses to bet on low-conviction signals.
    """
    aligned = pd.concat([primary_pred.rename("p"), meta_proba.rename("m")], axis=1)
    out = aligned["p"].where(aligned["m"] >= p_threshold, 0.0)
    return out.rename(primary_pred.name or "filtered_pred")
