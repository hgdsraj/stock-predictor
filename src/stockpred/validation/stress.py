"""Phase 4 — stress tests and honest evaluation.

This module provides utilities for the kind of analysis that separates "the
backtest looks great" from "the strategy might actually work":

1. **Holdout split** — partition the date range so that the last `holdout_years`
   are never touched by any prior code (CV, model selection, hyperparameter
   tuning). The pipeline is then evaluated ONLY on those held-out predictions.

2. **Bootstrap Sharpe confidence interval** — Sharpe is a noisy estimate. A
   strategy reporting Sharpe = 1.0 over 3 years often has a 95% CI of
   (-0.3, 2.3). We compute this honestly.

3. **Sensitivity grid** — run the pipeline across a grid of (horizon, k,
   cost_bps, universe_size) and report a table. If results swing wildly with
   small parameter changes, the "edge" is probably overfitting.

4. **Regime breakdown** — split the realised return series by external regime
   (VIX quintile, bull vs. bear S&P, etc.) and report metrics per regime.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Holdout
# --------------------------------------------------------------------- #


def holdout_split_dates(
    dates: pd.DatetimeIndex, holdout_years: int = 2
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Return (development_dates, holdout_dates) split chronologically."""
    dates = pd.DatetimeIndex(dates).unique().sort_values()
    if len(dates) == 0:
        return dates, dates
    split_at = dates[-1] - pd.DateOffset(years=holdout_years)
    dev = dates[dates < split_at]
    hold = dates[dates >= split_at]
    return dev, hold


# --------------------------------------------------------------------- #
# Bootstrap Sharpe CI
# --------------------------------------------------------------------- #


def bootstrap_sharpe(
    returns: pd.Series,
    *,
    n_resamples: int = 1000,
    periods_per_year: int = 252,
    confidence: float = 0.95,
    rng_seed: int = 0,
    method: str = "iid",
    block_length: int | None = None,
) -> dict[str, float | str]:
    """Bootstrap a confidence interval for annualised Sharpe.

    Methods:
      "iid"   — i.i.d. resampling with replacement. Suitable only when daily
                returns have negligible autocorrelation. With overlapping
                multi-day-horizon strategies, this narrows the CI artificially.
      "block" — moving-block bootstrap (Künsch 1989). Preserves short-range
                autocorrelation by sampling contiguous blocks. Use this for
                any strategy with horizon > 1 day or visible autocorr in
                daily returns. `block_length` defaults to ~horizon.

    Returns dict: sharpe, sharpe_lo, sharpe_hi, ci_pct, method, block_length.
    """
    r = returns.dropna().to_numpy(dtype=float)
    n = len(r)
    out_base: dict[str, float | str] = {
        "sharpe": float("nan"),
        "sharpe_lo": float("nan"),
        "sharpe_hi": float("nan"),
        "ci_pct": float(confidence),
        "method": method,
        "block_length": float(block_length) if block_length else float("nan"),
    }
    if n < 30:
        return out_base

    rng = np.random.default_rng(rng_seed)
    if method == "block":
        L = max(2, int(block_length) if block_length else max(2, int(np.sqrt(n))))
        # Number of blocks to fill ~n observations.
        n_blocks = int(np.ceil(n / L))
        # Sample starting indices uniformly over [0, n - L].
        max_start = max(1, n - L)
        starts = rng.integers(0, max_start, size=(n_resamples, n_blocks))
        # Build (n_resamples, n_blocks * L) index array, then truncate to n.
        offsets = np.arange(L)
        idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)[:, :n]
        samples = r[idx]
        out_base["block_length"] = float(L)
    elif method == "iid":
        idx = rng.integers(0, n, size=(n_resamples, n))
        samples = r[idx]
    else:
        raise ValueError(f"unknown bootstrap method: {method!r}")

    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    stds = np.where(stds == 0, np.nan, stds)
    sharpes = means / stds * np.sqrt(periods_per_year)
    alpha = (1 - confidence) / 2
    lo, hi = np.nanpercentile(sharpes, [100 * alpha, 100 * (1 - alpha)])
    point = (
        (r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) else float("nan")
    )
    out_base["sharpe"] = float(point)
    out_base["sharpe_lo"] = float(lo)
    out_base["sharpe_hi"] = float(hi)
    return out_base
    rng = np.random.default_rng(rng_seed)
    # Sample with replacement
    idx = rng.integers(0, n, size=(n_resamples, n))
    samples = r[idx]
    means = samples.mean(axis=1)
    stds = samples.std(axis=1, ddof=1)
    stds = np.where(stds == 0, np.nan, stds)
    sharpes = means / stds * np.sqrt(periods_per_year)
    alpha = (1 - confidence) / 2
    lo, hi = np.nanpercentile(sharpes, [100 * alpha, 100 * (1 - alpha)])
    point = (
        (r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) else float("nan")
    )
    return {
        "sharpe": float(point),
        "sharpe_lo": float(lo),
        "sharpe_hi": float(hi),
        "ci_pct": confidence,
    }


# --------------------------------------------------------------------- #
# Sensitivity grid
# --------------------------------------------------------------------- #


@dataclass
class GridResult:
    params: dict
    metrics: dict[str, float]


def sensitivity_grid(
    run_fn: Callable[..., dict],
    base_kwargs: dict,
    param_grid: dict[str, list],
    *,
    metric_keys: tuple[str, ...] = ("ann_return", "sharpe", "max_drawdown"),
) -> pd.DataFrame:
    """Run `run_fn(**base_kwargs, **combo)` for every combination of the grid.

    `run_fn` is expected to return a dict with a "metrics" sub-dict (the
    pipeline returns this shape). Returns a long-form DataFrame with one row
    per combination.
    """
    rows: list[dict] = []
    keys = list(param_grid)
    values = [param_grid[k] for k in keys]
    n_combos = int(np.prod([len(v) for v in values]))
    log.info("Sensitivity grid: %d combinations", n_combos)
    for i, combo in enumerate(itertools.product(*values), 1):
        kw = dict(zip(keys, combo))
        log.info("  combo %d/%d: %s", i, n_combos, kw)
        try:
            result = run_fn(**{**base_kwargs, **kw})
            metrics = result.get("metrics", {})
            row = {**kw, **{k: metrics.get(k) for k in metric_keys}}
        except Exception as e:  # noqa: BLE001
            log.warning("  combo failed: %s", e)
            row = {**kw, **{k: float("nan") for k in metric_keys}, "error": str(e)}
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- #
# Regime breakdown
# --------------------------------------------------------------------- #


def vix_regime(vix: pd.Series, *, q: int = 4) -> pd.Series:
    """Bucket VIX into quantile regimes per date. Returns a Series of regime labels."""
    return pd.qcut(vix.dropna(), q=q, labels=[f"vix_q{i + 1}" for i in range(q)])


def spy_regime(spy_close: pd.Series, *, ma_window: int = 200) -> pd.Series:
    """Bull/bear regime based on whether SPY > its trailing N-day moving average."""
    ma = spy_close.rolling(ma_window, min_periods=ma_window).mean()
    return pd.Series(
        np.where(spy_close > ma, "bull", "bear"),
        index=spy_close.index,
        name="spy_regime",
    )


def regime_breakdown(
    returns: pd.Series, regime: pd.Series, *, periods_per_year: int = 252
) -> pd.DataFrame:
    """Per-regime mean, std, Sharpe, hit-rate, count."""
    df = pd.concat(
        [returns.rename("r"), regime.reindex(returns.index).rename("regime")], axis=1
    ).dropna()
    out = df.groupby("regime")["r"].agg(
        n="count",
        mean="mean",
        std="std",
    )
    out["sharpe"] = out["mean"] / out["std"] * np.sqrt(periods_per_year)
    out["hit"] = df.groupby("regime")["r"].apply(lambda x: (x > 0).mean())
    out["ann_return"] = out["mean"] * periods_per_year
    return out
