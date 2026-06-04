"""Phase 6: Tier-2 features.

These are the canonical price-only factors the research literature consistently
finds in cross-sectional US equity returns. We add them as features so the
LightGBM model can compose them; combined IC typically beats any one alone.

Every feature here uses prices through close-of-t (no look-ahead). All return
calculations are log returns. SPY-relative features take an `spy` Series.

Citations (see docs/PROJECT_LOG.md "Session 3 strategy research"):
  - 12-1 momentum: Jegadeesh & Titman (JF 1993)
  - Short-term reversal: Lehmann (QJE 1990); Lo & MacKinlay (RFS 1990)
  - Idiosyncratic volatility: Ang, Hodrick, Xing & Zhang (JF 2006)
  - Beta (BAB): Frazzini & Pedersen (JFE 2014)
  - Max daily return (lottery): Bali, Cakici & Whitelaw (JFE 2011)
  - Amihud illiquidity: Amihud (JFM 2002)
  - 52-week high: George & Hwang (JF 2004) — already partially present
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _log_returns(close: pd.DataFrame) -> pd.DataFrame:
    return np.log(close).diff()


# -------------------------------------------------------------------- #
# Single-asset features
# -------------------------------------------------------------------- #


def momentum_12_1(close: pd.DataFrame) -> pd.DataFrame:
    """12-month minus 1-month momentum: log-return from t-252 to t-21.

    Skipping the most recent month is standard practice — short-term reversal
    dominates that window and would noise up the longer-horizon momentum
    signal.
    """
    log_p = np.log(close)
    return log_p.shift(21) - log_p.shift(252)


def short_term_reversal(close: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """Negative of the recent return — high returns expected to revert."""
    log_p = np.log(close)
    return -(log_p - log_p.shift(window))


def max_daily_return(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Maximum single-day return in the trailing `window` days. Bali et al.
    show high-max names underperform (lottery preference effect)."""
    daily = _log_returns(close)
    return daily.rolling(window, min_periods=window).max()


def amihud_illiquidity(close: pd.DataFrame, volume: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Average of |return| / dollar-volume over the trailing window.

    `close * volume` is the dollar-volume *proxy* (true raw dollar-volume
    requires unadjusted prices; we don't have those for free, see H5 in
    technical.py). Amihud is a relative cross-sectional measure, so the
    monotonic distortion is mostly OK for ranking purposes.
    """
    daily = _log_returns(close).abs()
    dollar_vol = (close * volume).replace(0, np.nan)
    illiq = daily / dollar_vol
    return illiq.rolling(window, min_periods=int(window * 0.8)).mean()


# -------------------------------------------------------------------- #
# Two-asset (vs benchmark) features
# -------------------------------------------------------------------- #


def _rolling_ols_beta_resid(
    asset_ret: pd.DataFrame,
    bench_ret: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-ticker rolling OLS of asset_ret on bench_ret.

    Returns (beta, idio_vol):
      beta(t)     — slope of asset on bench over the trailing `window` days
                    computed using returns through close-of-t.
      idio_vol(t) — std of residuals over the same window.

    Implementation: closed-form rolling beta = cov(a, b) / var(b). Residuals
    are computed pointwise then their rolling std is the idio_vol.
    """
    if min_periods is None:
        min_periods = window
    bench = bench_ret.reindex(asset_ret.index)
    bench_var = bench.rolling(window, min_periods=min_periods).var()
    # cov(a, b) per ticker
    a_minus_mean = asset_ret.subtract(asset_ret.rolling(window, min_periods=min_periods).mean())
    b_minus_mean = bench.subtract(bench.rolling(window, min_periods=min_periods).mean())
    # cross-product mean; align over the rolling window
    cov = (a_minus_mean.mul(b_minus_mean, axis=0)).rolling(window, min_periods=min_periods).mean()
    # Adjust for ddof=0 vs ddof=1 — we use sample stats so multiply by N/(N-1)
    cov = cov * window / max(window - 1, 1)
    beta = cov.div(bench_var, axis=0)
    # Pointwise residual = a - beta_t * b_t. We use the per-day beta as best
    # estimate (a "constant-window" residual would smooth out idio shocks).
    resid = asset_ret.sub(beta.mul(bench, axis=0))
    idio_vol = resid.rolling(window, min_periods=min_periods).std()
    return beta, idio_vol


def beta_vs_bench(close: pd.DataFrame, bench_close: pd.Series, *, window: int = 60) -> pd.DataFrame:
    asset_ret = _log_returns(close)
    bench_ret = _log_returns(bench_close.to_frame()).iloc[:, 0]
    beta, _ = _rolling_ols_beta_resid(asset_ret, bench_ret, window)
    return beta


def idio_vol_vs_bench(
    close: pd.DataFrame, bench_close: pd.Series, *, window: int = 60
) -> pd.DataFrame:
    asset_ret = _log_returns(close)
    bench_ret = _log_returns(bench_close.to_frame()).iloc[:, 0]
    _, idio = _rolling_ols_beta_resid(asset_ret, bench_ret, window)
    return idio


# -------------------------------------------------------------------- #
# Composer
# -------------------------------------------------------------------- #


def compute_tier2_features(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    bench_close: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute all Tier-2 features and return a long-form DataFrame indexed
    by [date, ticker]. Missing benchmark disables benchmark-relative features.
    """
    feats: dict[str, pd.DataFrame] = {}
    feats["mom_12_1"] = momentum_12_1(close)
    feats["st_reversal_5"] = short_term_reversal(close, 5)
    feats["max_ret_21"] = max_daily_return(close, 21)
    if volume is not None and not volume.empty:
        feats["amihud_21"] = amihud_illiquidity(close, volume, 21)
    if bench_close is not None and not bench_close.dropna().empty:
        feats["beta_60"] = beta_vs_bench(close, bench_close, window=60)
        feats["idio_vol_60"] = idio_vol_vs_bench(close, bench_close, window=60)

    if not feats:
        return pd.DataFrame()
    long_frames = []
    for name, df in feats.items():
        long_frames.append(df.stack(future_stack=True).rename(name))
    out = pd.concat(long_frames, axis=1)
    out.index = out.index.set_names(["date", "ticker"])
    return out
