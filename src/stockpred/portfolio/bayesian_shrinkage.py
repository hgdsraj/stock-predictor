"""Phase 19: per-ticker Bayesian shrinkage of signal scores toward zero.

Motivation: top-k cross-sectional selection treats every ticker equally
when deciding which day's signals are "best". But our walk-forward
diagnostics show that some tickers' signals are reliably useful (the
model has been right about them 55-60% of the time) while others are
basically noise (50-52% hit rate). A robust aggregation should DOWN-
WEIGHT the noisy tickers and UP-WEIGHT the reliable ones.

Approach (a James-Stein / empirical-Bayes-flavored shrinkage):

  For each ticker t and a fitted lookback window, define the per-ticker
  empirical sign-precision:
      p_t = P(sign(pred_t) == sign(realised_t))     in [0, 1]
  Cross-sectional mean p_bar = mean_t(p_t). The shrunken score is:
      shrunk_score_t,d = raw_score_t,d * shrink_t
  where
      shrink_t = max(0, (p_t - 0.5) / (0.5)) * alpha    (clipped)
  Thus a ticker with p_t = 0.50 (random) gets weight 0; a ticker with
  p_t = 0.60 gets weight 0.2 * alpha; a ticker with p_t > 0.60 gets
  more (capped at 1 * alpha).

  Alpha is a global blend coefficient in [0, 1]: alpha=0 -> raw scores
  unchanged; alpha=1 -> aggressive shrinkage.

The shrinkage table p_t is computed from a TRAINING window (never
including holdout). At predict time, today's (date, ticker)-indexed
raw scores are multiplied by the ticker's pre-computed shrink factor.

This is a SIGNAL transform, not a portfolio transform. It is applied
BEFORE the existing top-k / HRP / vol-scaled portfolio constructors.

Memory profile: O(#tickers) shrinkage factors. Trivial.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def compute_per_ticker_sign_precision(
    pred: pd.Series,
    realised: pd.Series,
    *,
    min_obs: int = 30,
) -> pd.Series:
    """Per-ticker P(sign(pred) == sign(realised)) over the input window.

    Inputs are (date, ticker)-indexed Series.

    Tickers with fewer than `min_obs` non-NaN paired observations get a
    NaN (and downstream code treats NaN as "use default = 0.5 = no edge").
    """
    df = pd.DataFrame({"pred": pred, "realised": realised}).dropna()
    if df.empty:
        empty = pd.Series(dtype=float, name="sign_precision")
        empty.index.name = "ticker"
        return empty
    df["sign_match"] = (np.sign(df["pred"]) == np.sign(df["realised"])).astype(float)
    # Per-ticker mean (sign_match) AND count, filter <min_obs
    g = df.groupby(level="ticker", observed=True)
    counts = g.size()
    means = g["sign_match"].mean()
    out = means.where(counts >= min_obs)
    out.name = "sign_precision"
    out.index.name = "ticker"  # H4: be explicit for downstream merges
    return out


def compute_shrinkage_factors(
    sign_precision: pd.Series,
    *,
    alpha: float = 1.0,
    default: float = 0.5,
) -> pd.Series:
    """Convert per-ticker sign-precision to per-ticker shrinkage factor.

    Formula:
        shrink_t = clip( (p_t - 0.5) / 0.5, 0, 1 ) * alpha
    p_t < 0.5  -> shrink = 0    (worse than random; drop the ticker)
    p_t = 0.5  -> shrink = 0    (no edge)
    p_t = 0.6  -> shrink = 0.2 * alpha
    p_t = 0.75 -> shrink = 0.5 * alpha
    p_t = 1.0  -> shrink = 1.0 * alpha
    NaN -> use `default - 0.5` ratio (default 0.5 -> 0 shrink)
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    filled = sign_precision.fillna(default)
    excess = filled - 0.5
    shrink = (excess / 0.5).clip(lower=0.0, upper=1.0) * alpha
    shrink.name = "shrink_factor"
    shrink.index.name = "ticker"  # H4: preserve index name for merge
    return shrink


def apply_shrinkage_to_panel(
    score: pd.Series,
    shrink_factors: pd.Series,
) -> pd.Series:
    """Multiply each (date, ticker) score by its ticker's shrink_factor.

    Returns a Series with the same index as `score`. Missing tickers
    (no row in `shrink_factors`) get factor 0 (i.e. dropped from the
    cross-section).

    Reviewer H4: enforce that `shrink_factors` is indexed by ticker
    name; without this guard the downstream merge silently produces a
    column-name collision when the caller has accidentally relabelled
    the index.
    """
    if score.empty:
        return score
    if shrink_factors.index.name != "ticker":
        raise ValueError(
            f"shrink_factors.index.name must be 'ticker', got "
            f"{shrink_factors.index.name!r}. Set it explicitly with "
            f"shrink_factors.index.name = 'ticker'."
        )
    # Defensive: dedupe duplicate ticker rows (reviewer M6).
    if shrink_factors.index.duplicated().any():
        log.warning(
            "apply_shrinkage_to_panel: %d duplicate ticker(s) in shrink_factors; "
            "keeping first occurrence of each.",
            int(shrink_factors.index.duplicated().sum()),
        )
        shrink_factors = shrink_factors.loc[~shrink_factors.index.duplicated()]
    # score has MultiIndex (date, ticker); join shrink_factors by ticker
    df = score.to_frame("score").reset_index()
    df = df.merge(
        shrink_factors.rename("shrink").reset_index(),
        on="ticker",
        how="left",
    )
    df["shrink"] = df["shrink"].fillna(0.0)
    df["shrunk_score"] = df["score"] * df["shrink"]
    out = df.set_index(["date", "ticker"])["shrunk_score"]
    out.name = score.name
    log.info(
        "apply_shrinkage_to_panel: %d rows, mean(|shrink|)=%.3f, "
        "n_zeroed=%d (no-edge tickers dropped)",
        len(out),
        float(shrink_factors.abs().mean()),
        int((shrink_factors == 0).sum()),
    )
    return out


def fit_apply_bayesian_shrinkage(
    train_pred: pd.Series,
    train_realised: pd.Series,
    apply_pred: pd.Series,
    *,
    alpha: float = 1.0,
    min_obs: int = 30,
) -> pd.Series:
    """Fit shrinkage factors on the train window, apply to `apply_pred`.

    `train_pred` and `train_realised` are (date, ticker)-indexed Series
    spanning the TRAINING period only. `apply_pred` is the (date,
    ticker)-indexed score to shrink (typically the dev or holdout
    ensemble score). Returns the shrunk score with the same index as
    `apply_pred`.
    """
    sp = compute_per_ticker_sign_precision(train_pred, train_realised, min_obs=min_obs)
    sf = compute_shrinkage_factors(sp, alpha=alpha)
    return apply_shrinkage_to_panel(apply_pred, sf)
