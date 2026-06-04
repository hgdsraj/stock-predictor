"""Portfolio construction from cross-sectional predictions.

Two construction styles are supported:

1. **Top-K equal-weight (legacy / simple)**
   `top_bottom_k_weights(score, k=50)` — long the top k names, short the
   bottom k names, equal-weighted within each side. Dollar-neutral, easy to
   reason about.

2. **Signal × inverse-volatility, sector-capped (Phase 3)**
   `vol_scaled_weights(score, vol)` followed by `apply_sector_caps(...)` —
   weight each name by `score / vol`, normalise so each side sums to a target
   gross exposure, then cap per-sector gross exposure to avoid concentration.
   This is what you'd run in production for a tradable signal.

Both functions return a wide [date x ticker] DataFrame of target weights.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Style 1: simple top-K equal-weight
# --------------------------------------------------------------------- #


def top_bottom_k_weights(
    preds: pd.Series,
    *,
    k: int | float = 50,
    leverage_per_side: float = 1.0,
) -> pd.DataFrame:
    """Build a wide weights DataFrame [date x ticker] from long predictions.

    Parameters
    ----------
    preds : Series indexed by [date, ticker] with predicted scores.
    k     : if int, count of names per side. If float in (0, 0.5), fraction of
            universe per side (e.g. 0.1 = top/bottom decile).
    leverage_per_side : gross exposure per leg (default 1.0 => 2x gross, 0 net).
    """
    if isinstance(preds, pd.DataFrame):
        preds = preds.iloc[:, 0]
    df = preds.dropna().to_frame("pred")
    df.index = df.index.set_names(["date", "ticker"])

    def _per_day(group: pd.DataFrame) -> pd.Series:
        if isinstance(k, float) and 0 < k < 0.5:
            kk = max(1, int(len(group) * k))
        else:
            kk = int(k)
        if len(group) < 2 * kk:
            return pd.Series(dtype=float)
        ranked = group["pred"].rank(method="first")
        long_thresh = len(group) - kk
        short_thresh = kk
        weights = pd.Series(0.0, index=group.index)
        weights[ranked > long_thresh] = leverage_per_side / kk
        weights[ranked <= short_thresh] = -leverage_per_side / kk
        return weights

    w = df.groupby(level="date", group_keys=False).apply(_per_day)
    if w.empty:
        return pd.DataFrame()
    w.index = w.index.set_names(["date", "ticker"])
    wide = w.unstack("ticker").fillna(0.0)
    return wide


# --------------------------------------------------------------------- #
# Style 2: signal × inverse-vol, sector-capped
# --------------------------------------------------------------------- #


def vol_scaled_weights(
    score: pd.Series,
    vol: pd.DataFrame,
    *,
    leverage_per_side: float = 1.0,
    top_fraction: float = 0.2,
) -> pd.DataFrame:
    """Convert a score into vol-scaled long/short weights.

    For each date:
      * Take the top `top_fraction` by score and the bottom `top_fraction`.
      * Within each side, weight ∝ |score| / vol (so high-conviction low-vol
        names get more weight than low-conviction high-vol names).
      * Normalise so each side's gross is `leverage_per_side`.

    Parameters
    ----------
    score : Series indexed by [date, ticker]; higher = more bullish.
    vol   : wide [date x ticker] DataFrame of estimated daily volatility used
            at signal time (e.g. rolling realised vol). Lag-safe.
    leverage_per_side : gross dollar per leg.
    top_fraction : fraction of the cross-section per side.
    """
    if isinstance(score, pd.DataFrame):
        score = score.iloc[:, 0]
    df = score.dropna().to_frame("score")
    df.index = df.index.set_names(["date", "ticker"])

    # Look up the corresponding vol on the same (date, ticker). stack() may
    # produce a 1- or 2-level index depending on shape; align via explicit
    # multi-index construction.
    vol_long = vol.stack(future_stack=True)
    if isinstance(vol_long.index, pd.MultiIndex) and vol_long.index.nlevels == 2:
        vol_long.index = vol_long.index.set_names(["date", "ticker"])
    df["vol"] = vol_long.reindex(df.index)
    # Clip to avoid /0; vol that's effectively zero means the name has no
    # signal-time information and we shouldn't trade it.
    df["vol"] = df["vol"].where(df["vol"] > 1e-6)

    def _per_day(group: pd.DataFrame) -> pd.Series:
        g = group.dropna(subset=["vol"])
        if len(g) < 10:
            return pd.Series(dtype=float)
        # Clamp kk <= len(g)//2 so long and short cohorts are disjoint
        # even when top_fraction > 0.5 (review fix matching the HRP path).
        kk = max(1, min(int(len(g) * top_fraction), len(g) // 2))
        ranked = g["score"].rank(method="first")
        long_mask = ranked > len(g) - kk
        short_mask = ranked <= kk
        w = pd.Series(0.0, index=g.index)
        # Inverse-vol within side.
        for mask, sign in ((long_mask, 1.0), (short_mask, -1.0)):
            if not mask.any():
                continue
            sub = g.loc[mask]
            inv_vol = 1.0 / sub["vol"]
            normalised = inv_vol / inv_vol.sum() * leverage_per_side
            w.loc[mask] = sign * normalised
        return w

    # Apply per day, then materialise a clean (date, ticker) MultiIndex.
    pieces: list[pd.Series] = []
    for date, sub in df.groupby(level="date"):
        sub2 = sub.copy()
        # Drop the date level so _per_day sees a ticker-only index.
        sub2.index = sub2.index.droplevel("date")
        piece = _per_day(sub2)
        if piece.empty:
            continue
        # Re-attach date.
        piece.index = pd.MultiIndex.from_product([[date], piece.index], names=["date", "ticker"])
        pieces.append(piece)

    if not pieces:
        return pd.DataFrame()
    w = pd.concat(pieces)
    return w.unstack("ticker").fillna(0.0)


def apply_sector_caps(
    weights: pd.DataFrame,
    sector_map: dict[str, str],
    *,
    max_per_sector_gross: float = 0.30,
) -> pd.DataFrame:
    """Scale down weights so no single sector's gross exposure exceeds the cap.

    Per-date, per-sector: if the sector's |weight| sum exceeds the cap, shrink
    every weight in that sector proportionally. We do not redistribute the
    truncated weight to other sectors (which would change the long/short
    balance); the strategy simply runs at lower gross when one sector is over.

    `sector_map` maps ticker -> sector string. Missing tickers go to a special
    "__OTHER__" sector that is also capped.
    """
    if weights.empty:
        return weights

    sectors = pd.Series(
        [sector_map.get(t, "__OTHER__") for t in weights.columns],
        index=weights.columns,
        name="sector",
    )

    capped = weights.copy()
    for sector, ticker_list in sectors.groupby(sectors):
        cols = [c for c in ticker_list.index if c in capped.columns]
        if not cols:
            continue
        sec_gross = capped[cols].abs().sum(axis=1)
        over = sec_gross > max_per_sector_gross
        if not over.any():
            continue
        scale = pd.Series(1.0, index=capped.index)
        scale[over] = max_per_sector_gross / sec_gross[over]
        capped[cols] = capped[cols].multiply(scale, axis=0)
        n_capped = int(over.sum())
        if n_capped:
            log.debug(
                "Sector cap engaged: %s on %d / %d days",
                sector,
                n_capped,
                len(capped),
            )
    return capped


def apply_min_trade_threshold(
    weights: pd.DataFrame, *, min_abs_delta: float = 0.005
) -> pd.DataFrame:
    """Suppress small day-to-day rebalances that don't earn back their cost.

    If |w_t - w_{t-1}| < min_abs_delta for a name, we keep w_{t-1} instead.
    This roughly halves turnover for noisy day-to-day signals; the cost
    saving usually swamps the small alpha loss from not trading tiny moves.
    """
    if weights.empty:
        return weights
    out = weights.copy()
    for i in range(1, len(out)):
        prev = out.iloc[i - 1]
        curr = out.iloc[i]
        delta = (curr - prev).abs()
        keep_prev = delta < min_abs_delta
        out.iloc[i] = curr.where(~keep_prev, prev)
    return out


# --------------------------------------------------------------------- #
# IC-IR weighted ensemble (Phase 3 model-side helper)
# --------------------------------------------------------------------- #


def neutralise_portfolio_beta(
    weights: pd.DataFrame,
    asset_betas: pd.DataFrame,
    *,
    target: float = 0.0,
) -> pd.DataFrame:
    """Phase 6: rescale weights so portfolio beta ≈ `target`.

    For each date with a non-zero portfolio beta we *shrink* the long and
    short sides toward each other (not toward zero, which would change
    leverage). Specifically: long-side := long * (1 - alpha), short-side :=
    short * (1 + alpha) where alpha is chosen to drive portfolio beta to
    target. If alpha hits a sensible cap (|alpha| > 0.5), we leave that day
    untouched and the gross exposure shifts a bit; this is a soft constraint.

    Parameters
    ----------
    weights : wide [date x ticker] target weights
    asset_betas : wide [date x ticker] beta of each name vs the benchmark
    """
    if weights.empty:
        return weights
    aligned_betas = asset_betas.reindex_like(weights).fillna(1.0)
    port_beta = (weights * aligned_betas).sum(axis=1)
    long_mask = weights > 0
    short_mask = weights < 0
    long_beta = (weights.where(long_mask, 0.0) * aligned_betas).sum(axis=1)
    short_beta = (weights.where(short_mask, 0.0) * aligned_betas).sum(axis=1)

    out = weights.copy()
    for date in weights.index:
        pb = port_beta.loc[date]
        if abs(pb - target) < 1e-6:
            continue
        lb = long_beta.loc[date]
        sb = short_beta.loc[date]
        # Solve: long * (1-a) * lb + short * (1+a) * sb = target
        # Net beta in pb = lb + sb; we want lb*(1-a) + sb*(1+a) = target.
        # => lb - a*lb + sb + a*sb = target  =>  a*(sb - lb) = target - (lb+sb)
        denom = sb - lb
        if abs(denom) < 1e-9:
            continue
        alpha = (target - (lb + sb)) / denom
        if not np.isfinite(alpha) or abs(alpha) > 0.5:
            continue  # too aggressive; skip this day
        # Apply: scale longs by (1-alpha), shorts by (1+alpha).
        row = out.loc[date]
        row = row.where(~long_mask.loc[date], row * (1 - alpha))
        row = row.where(~short_mask.loc[date], row * (1 + alpha))
        out.loc[date] = row
    return out


def ic_ir_weighted_ensemble(
    per_horizon_predictions: dict[int, pd.Series],
    per_horizon_ic_ir: dict[int, float],
    *,
    min_weight: float = 0.0,
) -> pd.Series:
    """Average per-horizon predictions weighted by their |IC IR|.

    Horizons with negative IC IR get zero weight (we don't anti-sample them
    because we lack out-of-sample validation that the sign-flip is stable;
    the safer move is to ignore them entirely).
    """
    if not per_horizon_predictions:
        return pd.Series(dtype=float, name="ensemble_score")

    weights: dict[int, float] = {}
    for h in per_horizon_predictions:
        ir = per_horizon_ic_ir.get(h, 0.0)
        weights[h] = max(min_weight, float(ir)) if np.isfinite(ir) else 0.0

    total = sum(weights.values())
    if total <= 0:
        # Fall back to equal weights so the pipeline still produces something
        # rather than NaN; log loudly.
        log.warning(
            "ic_ir_weighted_ensemble: all horizons have IC IR <= 0; falling back to equal weights"
        )
        weights = {h: 1.0 / len(per_horizon_predictions) for h in per_horizon_predictions}
        total = 1.0
    weights = {h: w / total for h, w in weights.items()}

    # Z-score per date per horizon, then weighted sum.
    parts = []
    for h, pred in per_horizon_predictions.items():
        g = pred.groupby(level="date")
        mu = g.transform("mean")
        sd = g.transform("std").replace(0, np.nan)
        z = ((pred - mu) / sd).fillna(0.0)
        parts.append(z * weights[h])
    out = pd.concat(parts, axis=1).sum(axis=1)
    out.name = "ensemble_score"
    return out
