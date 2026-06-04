"""Hierarchical Risk Parity (HRP) portfolio construction.

Reference: López de Prado, *Building Diversified Portfolios that Outperform
Out of Sample*, JPM 2016.

HRP avoids the matrix-inversion instability of mean-variance optimisation
on small samples by:
  1. Computing a correlation-based distance between assets.
  2. Hierarchically clustering them.
  3. Quasi-diagonalising the covariance matrix per the cluster order.
  4. Recursively splitting clusters, allocating inverse-variance weight to
     each side, and bisecting until single-asset leaves.

This implementation:
  - Pure numpy/scipy + scikit-learn. No new heavy deps.
  - Works on the per-day-selected long and short cohorts independently
    (the cross-sectional pipeline picks K names per side; we then compute
    HRP weights *within* each side from a trailing covariance estimate).
  - Falls back to inverse-vol equal weighting if the covariance estimate
    is rank-deficient (which happens with very small clusters or short
    windows).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

log = logging.getLogger(__name__)


# -------------------------------------------------------------------- #
# Core HRP helpers
# -------------------------------------------------------------------- #


def _corr_distance(corr: np.ndarray) -> np.ndarray:
    """Lopez de Prado distance: sqrt((1 - corr) / 2). Bounded to [0, 1]."""
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
    np.fill_diagonal(d, 0.0)
    return d


def _quasi_diag(link: np.ndarray) -> list[int]:
    """López de Prado's quasi-diagonalisation: walk the linkage tree to put
    similar items adjacent in the order."""
    link = link.astype(int)
    sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
    num_items = link[-1, 3]  # number of original items
    while sort_ix.max() >= num_items:
        sort_ix.index = range(0, sort_ix.shape[0] * 2, 2)  # spread
        df0 = sort_ix[sort_ix >= num_items]  # find clusters
        i = df0.index
        j = df0.values - num_items
        sort_ix[i] = link[j, 0]
        df1 = pd.Series(link[j, 1], index=i + 1)
        sort_ix = pd.concat([sort_ix, df1])
        sort_ix = sort_ix.sort_index()
        sort_ix.index = range(sort_ix.shape[0])
    return sort_ix.tolist()


def _cluster_var(cov: np.ndarray, items: list[int]) -> float:
    """Inverse-variance portfolio variance for a sub-cluster.

    Review H5 fix: guards against zero or non-finite diagonal entries
    (which can come out of Ledoit-Wolf shrinkage on near-constant or all-
    NaN return columns). Returns NaN if any diagonal entry is ill-conditioned;
    the caller treats NaN as a skip signal.
    """
    cov_ = cov[np.ix_(items, items)]
    diag = np.diag(cov_)
    if not np.all(np.isfinite(diag)) or np.any(diag <= 1e-20):
        return float("nan")
    ivp = 1.0 / diag
    ivp /= ivp.sum()
    return float(ivp @ cov_ @ ivp)


def _recursive_bisection(cov: np.ndarray, sort_ix: list[int]) -> np.ndarray:
    """Allocate weights via top-down recursive bisection through the sorted
    linkage order."""
    n = len(sort_ix)
    weights = np.ones(n)
    clusters: list[list[int]] = [list(range(n))]
    while clusters:
        new_clusters: list[list[int]] = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            mid = len(cl) // 2
            left = cl[:mid]
            right = cl[mid:]
            v_left = _cluster_var(cov, [sort_ix[i] for i in left])
            v_right = _cluster_var(cov, [sort_ix[i] for i in right])
            denom = v_left + v_right
            # Use a tolerance rather than exact zero; numerically tiny var
            # post-shrinkage can otherwise cause meaningless alphas.
            if not np.isfinite(denom) or denom < 1e-20:
                continue
            alpha = 1.0 - v_left / denom
            for i in left:
                weights[i] *= alpha
            for i in right:
                weights[i] *= 1.0 - alpha
            new_clusters.extend([left, right])
        clusters = new_clusters
    return weights


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Compute HRP weights for an N-asset covariance matrix.

    Returns a length-N array summing to 1 with all-positive entries.
    """
    n = cov.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])
    # Correlation matrix
    std = np.sqrt(np.diag(cov))
    std = np.where(std == 0, 1e-12, std)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    dist = _corr_distance(corr)
    # Linkage requires a condensed distance matrix.
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="single")
    sort_ix = _quasi_diag(link)
    w = _recursive_bisection(cov, sort_ix)
    # Reorder back to original indexing.
    out = np.zeros(n)
    for i, j in enumerate(sort_ix):
        out[j] = w[i]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    s = out.sum()
    if s == 0 or not np.isfinite(s):
        return np.full(n, 1.0 / n)
    return out / s


# -------------------------------------------------------------------- #
# Daily portfolio constructor
# -------------------------------------------------------------------- #


@dataclass(frozen=True)
class HRPConfig:
    cov_window: int = 60  # trailing days for covariance estimate
    top_fraction: float = 0.15  # per-side fraction of cross-section
    leverage_per_side: float = 1.0
    use_ledoit_wolf: bool = True  # shrinkage estimator (almost always better)
    min_names_per_side: int = 3  # below this, fall back to inverse-vol equal


def _trailing_returns_panel(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Daily log returns, indexed by date, columns by ticker."""
    return np.log(close).diff()


def _cov_estimate(
    returns_window: pd.DataFrame, use_ledoit_wolf: bool
) -> tuple[np.ndarray, list[str]]:
    """Estimate covariance from a window of returns. Returns (cov, columns)
    after dropping all-NaN columns and forward-filling residual NaNs.

    NOTE: we use ffill but NOT bfill (review CRIT-caveat). bfill would
    propagate future values backward, technically still inside the
    strictly-past window but distorting the regime mix toward the
    most-recent observation.
    """
    sub = returns_window.dropna(axis=1, how="all").ffill()
    # Drop rows that still have NaN in any column (typically the early
    # warmup before all columns have data).
    sub = sub.dropna(how="any")
    cols = sub.columns.tolist()
    arr = sub.to_numpy(dtype=float)
    if arr.shape[0] < 5 or arr.shape[1] < 2:
        return np.zeros((len(cols), len(cols))), cols
    if use_ledoit_wolf:
        try:
            lw = LedoitWolf().fit(arr)
            return lw.covariance_, cols
        except Exception as e:  # noqa: BLE001
            log.debug("LedoitWolf failed (%s); falling back to sample cov", e)
    return np.cov(arr, rowvar=False), cols


def hrp_long_short_weights(
    score: pd.Series,
    close: pd.DataFrame,
    *,
    cfg: HRPConfig | None = None,
) -> pd.DataFrame:
    """Build long/short HRP weights per date.

    Steps per date `t`:
      1. Pick top/bottom `top_fraction` of cross-section by score.
      2. For each side, estimate a covariance from the trailing `cov_window`
         days of returns *strictly through close-of-(t-1)* (lag-safe).
      3. Compute HRP weights within each side and scale to leverage_per_side.

    If a side has fewer than `min_names_per_side` viable names that day, we
    fall back to inverse-vol equal weighting for that side (HRP needs >=2).
    """
    cfg = cfg or HRPConfig()
    if isinstance(score, pd.DataFrame):
        score = score.iloc[:, 0]

    score = score.dropna()
    score.index = score.index.set_names(["date", "ticker"])
    ret = _trailing_returns_panel(close, cfg.cov_window)
    out_pieces: list[pd.Series] = []

    for date, sub in score.groupby(level="date"):
        per_day = sub.copy()
        per_day.index = per_day.index.droplevel("date")
        n = len(per_day)
        # Review fix: clamp kk <= n//2 so long and short cohorts are always
        # disjoint regardless of top_fraction (a top_fraction > 0.5 would
        # otherwise produce overlap, and the groupby.sum() would silently
        # net positions to a near-zero residual — breaking dollar-neutrality).
        kk = max(1, min(int(n * cfg.top_fraction), n // 2))
        if n < 2 * cfg.min_names_per_side:
            continue
        ranks = per_day.rank(method="first")
        long_names = ranks[ranks > n - kk].index.tolist()
        short_names = ranks[ranks <= kk].index.tolist()

        # Lag-safe trailing returns: rows strictly before `date`.
        ret_before = ret.loc[ret.index < date].tail(cfg.cov_window)
        if ret_before.empty:
            continue

        for names, sign in ((long_names, 1.0), (short_names, -1.0)):
            if not names:
                continue
            sub_ret = ret_before.reindex(columns=names)
            cov, cols = _cov_estimate(sub_ret, cfg.use_ledoit_wolf)
            if not cols:
                continue
            if len(cols) < cfg.min_names_per_side:
                # Fall back to inverse-vol equal-weight on this side.
                vols = sub_ret.std()
                inv = 1.0 / vols.replace(0, np.nan)
                inv = inv.dropna()
                if inv.empty:
                    continue
                w = inv / inv.sum() * cfg.leverage_per_side
            else:
                hw = hrp_weights(cov) * cfg.leverage_per_side
                w = pd.Series(hw, index=cols)
            piece = (sign * w).reindex(per_day.index, fill_value=0.0)
            idx = pd.MultiIndex.from_product([[date], piece.index], names=["date", "ticker"])
            piece.index = idx
            out_pieces.append(piece)

    if not out_pieces:
        return pd.DataFrame()
    long_form = pd.concat(out_pieces).groupby(level=["date", "ticker"]).sum()
    wide = long_form.unstack("ticker").fillna(0.0)
    return wide
