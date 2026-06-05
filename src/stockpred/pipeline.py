"""End-to-end pipeline for the stock-predictor.

Phase 2 design (this module supersedes the original Phase 1 driver):

1. Load S&P 500 historical membership (no survivorship).
2. Fetch adjusted prices for the chosen tickers.
3. Optionally fetch fundamentals for sector tagging.
4. Build features: lag-safe technicals + cross-sectional ranks + sector
   neutralisation + sector one-hot dummies.
5. Build labels for one or more horizons: forward return, binary direction,
   vol-scaled forward return.
6. For each horizon in `cfg.horizons`, walk-forward-train the configured model
   (GBM by default, logistic baseline if requested) using vol-scaled returns
   as the regression target.
7. Convert each horizon's OOS prediction into a centred score, average across
   horizons to produce a single ensemble score per (date, ticker).
8. Construct a long/short portfolio from the ensemble score, run a
   horizon-aware backtest, build a tearsheet.

The pipeline is intentionally pure-Python with no caching of intermediates
between phases — caching happens at the data layer (`data.cache/`). This keeps
the pipeline easy to reason about and easy to re-run after a code change.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.backtest.engine import run_backtest
from stockpred.backtest.portfolio import top_bottom_k_weights
from stockpred.config import (
    REPORTS_DIR,
    BacktestConfig,
    CVConfig,
)
from stockpred.data import fundamentals as fundamentals_mod
from stockpred.data import prices as prices_mod
from stockpred.data import universe as universe_mod
from stockpred.features.cross_sectional import (
    add_cross_sectional_ranks,
    add_sector_dummies,
    neutralise_by_sector,
)
from stockpred.features.technical import compute_technical_features
from stockpred.labels import long_labels
from stockpred.models.baseline import fit_predict_proba, make_baseline_pipeline
from stockpred.models.gbm import GBMConfig, predict_gbm, train_gbm
from stockpred.reports.tearsheet import build_tearsheet
from stockpred.validation.metrics import (
    ic_summary,
    information_coefficient,
    tearsheet_metrics,
)
from stockpred.validation.walk_forward import WalkForwardSplit

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    """All knobs for an end-to-end run."""

    # Universe / history
    start_date: str = "2010-01-01"
    end_date: str | None = None
    n_tickers: int | None = 100
    universe_sampling: str = "random"  # {"random", "current", "first"}
    refresh_data: bool = False

    # Horizons (trading days). The model is trained per-horizon and the OOS
    # scores are averaged into an ensemble score for the portfolio.
    horizons: tuple[int, ...] = (1, 5, 21)

    # Portfolio
    k_per_side: int = 20

    # Validation
    cv: CVConfig = field(
        default_factory=lambda: CVConfig(
            train_years=3,
            test_months=6,
            embargo_days=25,  # >= max horizon
            min_train_obs=1000,
        )
    )

    # Model selection
    model: str = "gbm"  # {"gbm", "logistic"}
    gbm: GBMConfig = field(default_factory=GBMConfig)

    # Feature engineering
    use_sector_features: bool = True
    feature_cols: list[str] | None = None  # if None, use everything

    # Reports
    tearsheet_path: Path | None = None  # if None, auto


# --------------------------------------------------------------------- #
# Universe selection (M4 fix preserved)
# --------------------------------------------------------------------- #


def select_universe(cfg: PipelineConfig) -> tuple[list[str], pd.DataFrame]:
    """Choose tickers per `cfg.universe_sampling`. See PipelineConfig docs."""
    membership = universe_mod.fetch_sp500_membership(refresh=cfg.refresh_data)
    all_tickers = universe_mod.all_tickers_in_range(
        cfg.start_date, cfg.end_date, membership=membership
    )

    if cfg.n_tickers is None or len(all_tickers) <= cfg.n_tickers:
        return all_tickers, membership

    sampling = cfg.universe_sampling
    if sampling == "current":
        log.warning(
            "Universe sampling='current' selects only currently-listed names. "
            "This introduces SURVIVORSHIP BIAS; positive backtest results "
            "should be discounted heavily."
        )
        current = sorted(membership[membership["end_date"].isna()]["ticker"].unique())
        if len(current) >= cfg.n_tickers:
            return current[: cfg.n_tickers], membership
        return all_tickers[: cfg.n_tickers], membership
    if sampling == "first":
        log.info("Universe sampling='first' (alphabetical, mildly biased).")
        return all_tickers[: cfg.n_tickers], membership

    seed = int(
        hashlib.sha256(f"{cfg.start_date}|{cfg.end_date}".encode()).hexdigest()[:8],
        16,
    )
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(all_tickers, size=cfg.n_tickers, replace=False).tolist())
    log.info(
        "Universe sampling='random' (seed=%d): chose %d / %d tickers",
        seed,
        len(chosen),
        len(all_tickers),
    )
    return chosen, membership


# --------------------------------------------------------------------- #
# Feature pipeline
# --------------------------------------------------------------------- #


def build_feature_matrix(
    close: pd.DataFrame,
    volume: pd.DataFrame | None,
    *,
    sector_map: dict[str, str] | None = None,
    use_sector_features: bool = True,
) -> pd.DataFrame:
    """Compute long-form features.

    Adds: lag-safe technicals -> cross-sectional ranks -> (optional) sector-
    neutralised versions of the numeric features -> (optional) one-hot sector
    dummies. All NaN handling is deferred to the model layer.
    """
    feats = compute_technical_features(close, volume=volume)
    feats = add_cross_sectional_ranks(feats)

    if use_sector_features and sector_map:
        # Sector-neutralise the raw numeric features (not the ranks).
        numeric_cols = [c for c in feats.columns if not c.endswith("_rank")]
        feats = neutralise_by_sector(feats, sector_map, numeric_cols)
        feats = add_sector_dummies(feats, sector_map)
    return feats


# --------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------- #


def assemble_dataset(
    feats: pd.DataFrame, labels: pd.DataFrame, horizon: int, *, target: str = "vs"
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Join features + labels for one horizon, drop label-NaN rows.

    Parameters
    ----------
    target : {"vs", "return", "dir"}
        Which label column to regress on:
        * "vs"     -> fwd_vs_{h} (vol-scaled forward return; default)
        * "return" -> fwd_return_{h} (raw log forward return)
        * "dir"    -> fwd_dir_{h} (binary; only valid for logistic baseline)

    Returns
    -------
    (X, y_target, y_return, y_bin) — y_return is always raw log fwd return for
    IC diagnostics; y_bin is the binary direction for hit-rate.
    """
    dir_col = f"fwd_dir_{horizon}"
    ret_col = f"fwd_return_{horizon}"
    vs_col = f"fwd_vs_{horizon}"

    needed = {dir_col, ret_col}
    if target == "vs" and vs_col in labels.columns:
        needed.add(vs_col)
    needed = sorted(needed)

    joined = feats.join(labels[needed], how="inner")
    joined = joined.dropna(subset=[ret_col])  # need at least the raw return

    if target == "vs":
        col = vs_col
    elif target == "return":
        col = ret_col
    else:
        col = dir_col

    if col not in joined.columns:
        # Fall back: requested target unavailable, use raw return.
        col = ret_col

    y_target = joined[col]
    y_return = joined[ret_col]
    y_bin = joined[dir_col].astype("float32")
    X = joined.drop(columns=[c for c in (dir_col, ret_col, vs_col) if c in joined.columns])
    return X, y_target, y_return, y_bin


# --------------------------------------------------------------------- #
# Walk-forward training (model-agnostic)
# --------------------------------------------------------------------- #


def _fit_and_predict_fold(
    model: str,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_te: pd.DataFrame,
    gbm_cfg: GBMConfig,
) -> pd.Series:
    if model == "logistic":
        # y_tr for logistic is already the binary direction label (0/1 with
        # NaN where return was zero or undefined). Pass through to the
        # baseline pipeline, which returns P(up).
        pipe = make_baseline_pipeline()
        return fit_predict_proba(pipe, X_tr, y_tr, X_te)

    if model == "fama_macbeth":
        # Phase 17: cross-sectional OLS per training date, average the
        # daily factor returns, predict OOS as X_te @ lambda_hat.
        # Different model class than tree-based; useful as a robustness
        # check against GBM over-fitting on weak signals.
        from stockpred.models.fama_macbeth import fit_predict_fama_macbeth

        return fit_predict_fama_macbeth(X_tr, y_tr, X_te)

    # Default: LightGBM regressor on the (vol-scaled) forward return.
    # Use the last 10% of train as an internal validation slice for early stop.
    n_tr = len(X_tr)
    if n_tr < 200:
        booster = train_gbm(X_tr, y_tr, cfg=gbm_cfg)
    else:
        split = int(n_tr * 0.9)
        booster = train_gbm(
            X_tr.iloc[:split],
            y_tr.iloc[:split],
            X_valid=X_tr.iloc[split:],
            y_valid=y_tr.iloc[split:],
            cfg=gbm_cfg,
        )
    return predict_gbm(booster, X_te)


def walk_forward_predict(
    X: pd.DataFrame,
    y_target: pd.Series,
    cv_cfg: CVConfig,
    *,
    model: str = "gbm",
    gbm_cfg: GBMConfig | None = None,
) -> pd.Series:
    """Run walk-forward CV; return concatenated OOS predictions.

    For "gbm" the predictions are continuous (vol-scaled return forecasts).
    For "logistic" they are probabilities of fwd direction == 1 (P(up)).
    """
    gbm_cfg = gbm_cfg or GBMConfig()
    splitter = WalkForwardSplit(
        train_years=cv_cfg.train_years,
        test_months=cv_cfg.test_months,
        embargo_days=cv_cfg.embargo_days,
        min_train_obs=cv_cfg.min_train_obs,
    )
    dates = X.index.get_level_values("date").unique().sort_values()
    preds: list[pd.Series] = []
    fold = 0
    for train_idx, test_idx in splitter.split(dates):
        fold += 1
        train_dates = dates[train_idx]
        test_dates = dates[test_idx]
        train_mask = X.index.get_level_values("date").isin(train_dates)
        test_mask = X.index.get_level_values("date").isin(test_dates)
        X_tr = X[train_mask]
        X_te = X[test_mask]
        y_tr = y_target[train_mask]

        log.info(
            "Fold %d: train [%s..%s] n=%d, test [%s..%s] n=%d",
            fold,
            train_dates.min().date(),
            train_dates.max().date(),
            len(X_tr),
            test_dates.min().date(),
            test_dates.max().date(),
            len(X_te),
        )
        try:
            pred = _fit_and_predict_fold(model, X_tr, y_tr, X_te, gbm_cfg)
        except Exception as e:  # noqa: BLE001
            log.warning("Fold %d failed: %s -- skipping", fold, e)
            continue
        preds.append(pred)

    if not preds:
        return pd.Series(dtype=float, name="prediction")
    out = pd.concat(preds).sort_index()
    out.name = "prediction"
    return out


# --------------------------------------------------------------------- #
# Multi-horizon ensemble
# --------------------------------------------------------------------- #


def _to_cross_sectional_zscore(series: pd.Series, by: str = "date") -> pd.Series:
    """Per-date z-score so different horizons are unit-comparable."""
    g = series.groupby(level=by)
    mu = g.transform("mean")
    sd = g.transform("std").replace(0, np.nan)
    return ((series - mu) / sd).fillna(0.0)


def ensemble_predictions(
    per_horizon_preds: dict[int, pd.Series], weights: dict[int, float] | None = None
) -> pd.Series:
    """Average per-date z-scores across horizons.

    Returns a Series indexed by [date, ticker] of the ensemble score.
    """
    if not per_horizon_preds:
        return pd.Series(dtype=float, name="ensemble_score")
    weights = weights or {h: 1.0 / len(per_horizon_preds) for h in per_horizon_preds}
    parts = []
    for h, pred in per_horizon_preds.items():
        z = _to_cross_sectional_zscore(pred)
        parts.append(z * weights.get(h, 0.0))
    out = pd.concat(parts, axis=1).sum(axis=1)
    out.name = "ensemble_score"
    return out


# --------------------------------------------------------------------- #
# End-to-end run
# --------------------------------------------------------------------- #


def _diagnostics(
    pred: pd.Series, y_return: pd.Series, y_bin: pd.Series
) -> tuple[float, dict[str, float]]:
    """Return (hit_rate, ic_summary) for one horizon's OOS predictions."""
    aligned = pd.concat(
        [pred.rename("p"), y_return.rename("r"), y_bin.rename("d")], axis=1
    ).dropna()
    if aligned.empty:
        return float("nan"), {
            "ic_mean": float("nan"),
            "ic_std": float("nan"),
            "ic_ir": float("nan"),
        }
    ic = information_coefficient(aligned["p"], aligned["r"])
    stats = ic_summary(ic)
    # Hit rate: did our sign match the realised sign?
    hit = float((np.sign(aligned["p"]) == np.sign(aligned["r"])).mean())
    return hit, stats


def run_pipeline(cfg: PipelineConfig | None = None) -> dict:
    """End-to-end run. Returns a dict of artefacts (predictions, metrics, paths)."""
    cfg = cfg or PipelineConfig()
    t0 = time.time()
    log.info(
        "Pipeline starting: model=%s horizons=%s n_tickers=%s [%s..%s]",
        cfg.model,
        cfg.horizons,
        cfg.n_tickers,
        cfg.start_date,
        cfg.end_date or "today",
    )

    tickers, _ = select_universe(cfg)
    log.info("Universe size: %d", len(tickers))

    log.info("Fetching prices (cached)...")
    raw_panel = prices_mod.long_panel(
        tickers, start=cfg.start_date, end=cfg.end_date, refresh=cfg.refresh_data
    )
    if raw_panel.empty:
        raise RuntimeError("No price data retrieved. Check connectivity / yfinance.")
    close = raw_panel["adj_close"].unstack("ticker").sort_index()
    volume = raw_panel["volume"].unstack("ticker").sort_index()
    log.info("Loaded prices: %d dates x %d tickers", close.shape[0], close.shape[1])

    sector_map: dict[str, str] = {}
    if cfg.use_sector_features:
        try:
            funds = fundamentals_mod.fetch_fundamentals(tickers, refresh=cfg.refresh_data)
            sector_map = fundamentals_mod.sector_map(funds)
            n_with_sector = sum(1 for t in tickers if t in sector_map)
            log.info("Sector tags available for %d / %d tickers", n_with_sector, len(tickers))
        except Exception as e:  # noqa: BLE001
            log.warning("Failed to load sector tags (%s); proceeding without.", e)

    log.info("Building features...")
    feats = build_feature_matrix(
        close, volume, sector_map=sector_map, use_sector_features=cfg.use_sector_features
    )
    log.info("Feature matrix: %s rows x %s cols", *feats.shape)

    log.info("Building labels for horizons %s...", cfg.horizons)
    labels = long_labels(close, horizons=tuple(cfg.horizons), include_vol_scaled=True)

    per_horizon_preds: dict[int, pd.Series] = {}
    per_horizon_diag: dict[int, dict] = {}
    per_horizon_returns: dict[int, pd.Series] = {}
    for h in cfg.horizons:
        log.info("=== Horizon %d ===", h)
        target = "vs" if cfg.model == "gbm" else "dir"
        X, y_target, y_return, y_bin = assemble_dataset(feats, labels, h, target=target)
        if cfg.feature_cols is not None:
            X = X[[c for c in cfg.feature_cols if c in X.columns]]
        log.info("Dataset h=%d: X=%s, y=%s", h, X.shape, y_target.shape)

        pred = walk_forward_predict(X, y_target, cfg.cv, model=cfg.model, gbm_cfg=cfg.gbm)
        if pred.empty:
            log.warning("Horizon %d produced no predictions; skipping", h)
            continue
        # For the logistic path, centre proba at 0.5 so the score is a return-like signed number.
        if cfg.model == "logistic":
            pred = pred - 0.5

        hit, ic_stats = _diagnostics(pred, y_return, y_bin)
        log.info(
            "Horizon %d OOS: hit=%.4f  ic_mean=%+.5f  ic_ir=%+.3f",
            h,
            hit,
            ic_stats["ic_mean"],
            ic_stats["ic_ir"],
        )
        per_horizon_preds[h] = pred
        per_horizon_returns[h] = y_return
        per_horizon_diag[h] = {"hit_rate": hit, **ic_stats}

    if not per_horizon_preds:
        raise RuntimeError("All horizons failed; no predictions produced.")

    # Ensemble score.
    score = ensemble_predictions(per_horizon_preds)
    log.info("Ensemble score: %d obs across %d horizons", len(score), len(per_horizon_preds))

    # Portfolio: top/bottom k by ensemble score per day.
    weights = top_bottom_k_weights(score, k=cfg.k_per_side)
    if weights.empty:
        raise RuntimeError("Portfolio is empty: too few tickers per day for k.")

    bt_cfg = BacktestConfig()
    # Use the smallest horizon for the backtest cadence; the ensemble is biased
    # toward the highest-confidence end of the predictive distribution at that
    # horizon, refreshed at the slowest cadence we can while still trading.
    bt_horizon = min(per_horizon_preds.keys())
    res = run_backtest(weights, close, cfg=bt_cfg, horizon=bt_horizon, trade_lag=1)
    metrics = tearsheet_metrics(res.returns)
    log.info("Backtest metrics: %s", metrics)

    out_path = cfg.tearsheet_path or REPORTS_DIR / (
        f"run_{cfg.model}_h{'-'.join(str(h) for h in cfg.horizons)}_k{cfg.k_per_side}.html"
    )
    bench_ret = close.pct_change().mean(axis=1)
    build_tearsheet(
        res.returns,
        out_path,
        benchmark=bench_ret,
        cost_bps_per_side=bt_cfg.total_cost_per_side_bps,
    )

    elapsed = time.time() - t0
    log.info("Tearsheet -> %s (elapsed %.1fs)", out_path, elapsed)

    return {
        "tickers": tickers,
        "feature_matrix_shape": feats.shape,
        "per_horizon_predictions": per_horizon_preds,
        "per_horizon_diagnostics": per_horizon_diag,
        "ensemble_score": score,
        "weights": weights,
        "backtest": res,
        "metrics": metrics,
        "tearsheet_path": out_path,
        "elapsed_s": elapsed,
    }


# Back-compat: keep the old name available for scripts/tests that imported it.
run_phase1 = run_pipeline
