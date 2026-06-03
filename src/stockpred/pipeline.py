"""Phase 1 end-to-end pipeline, importable from scripts and notebooks.

Pipeline:
    1. Load universe (S&P 500 historical constituents).
    2. Download adjusted prices for the chosen tickers.
    3. Compute lag-safe features (technical + cross-sectional ranks).
    4. Compute forward-return labels for each horizon.
    5. Walk-forward train baseline (logistic regression) per fold.
    6. Concatenate out-of-sample predictions into a single Series.
    7. Build long/short portfolio, backtest, generate tearsheet.
"""

from __future__ import annotations

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
    DEFAULT,
)
from stockpred.data import prices as prices_mod
from stockpred.data import universe as universe_mod
from stockpred.features.cross_sectional import add_cross_sectional_ranks
from stockpred.features.technical import compute_technical_features
from stockpred.labels import long_labels
from stockpred.models.baseline import fit_predict_proba, make_baseline_pipeline
from stockpred.reports.tearsheet import build_tearsheet
from stockpred.validation.metrics import (
    ic_summary,
    information_coefficient,
    tearsheet_metrics,
)
from stockpred.validation.walk_forward import WalkForwardSplit

log = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Knobs for the Phase 1 run."""

    start_date: str = "2010-01-01"  # default narrower for faster first run
    end_date: str | None = None
    n_tickers: int | None = 100  # most liquid current S&P 500 names; None = all
    horizon: int = 1  # which horizon to train baseline on
    k_per_side: int = 20  # long-short top-k
    cost_bps_per_side: float = 6.0
    rebalance_every: int | None = None  # None = same as horizon
    cv: CVConfig = field(default_factory=lambda: CVConfig(train_years=3, test_months=6))
    feature_cols: list[str] | None = None  # if None, use all numeric
    refresh_data: bool = False


def select_universe(
    cfg: PipelineConfig,
) -> tuple[list[str], pd.DataFrame]:
    """Choose tickers and return their membership frame."""
    membership = universe_mod.fetch_sp500_membership(refresh=cfg.refresh_data)
    tickers = universe_mod.all_tickers_in_range(cfg.start_date, cfg.end_date, membership=membership)
    if cfg.n_tickers is not None and len(tickers) > cfg.n_tickers:
        # Prefer current constituents (open intervals end_date NaT)
        current = sorted(membership[membership["end_date"].isna()]["ticker"].unique())
        if len(current) >= cfg.n_tickers:
            tickers = current[: cfg.n_tickers]
        else:
            tickers = tickers[: cfg.n_tickers]
    return tickers, membership


def build_feature_matrix(close: pd.DataFrame, volume: pd.DataFrame | None) -> pd.DataFrame:
    """Compute long-form features (technical + cross-sectional ranks)."""
    feats = compute_technical_features(close, volume=volume)
    feats = add_cross_sectional_ranks(feats)
    return feats


def assemble_dataset(
    feats: pd.DataFrame, labels: pd.DataFrame, horizon: int
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Join features + labels, drop rows where label is NaN.

    Returns (X, y_binary, y_continuous).
    """
    label_col = f"fwd_dir_{horizon}"
    ret_col = f"fwd_return_{horizon}"
    joined = feats.join(labels[[label_col, ret_col]], how="inner")
    joined = joined.dropna(subset=[label_col])
    X = joined.drop(columns=[label_col, ret_col])
    y_bin = joined[label_col].astype(int)
    y_cont = joined[ret_col]
    return X, y_bin, y_cont


def walk_forward_predict(
    X: pd.DataFrame,
    y_bin: pd.Series,
    cv_cfg: CVConfig,
) -> pd.Series:
    """Run walk-forward CV with the baseline pipeline; return concatenated OOS preds."""
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
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr = y_bin[train_mask]
        log.info(
            "Fold %d: train [%s .. %s] (n=%d), test [%s .. %s] (n=%d)",
            fold,
            train_dates.min().date(),
            train_dates.max().date(),
            len(X_tr),
            test_dates.min().date(),
            test_dates.max().date(),
            len(X_te),
        )
        pipe = make_baseline_pipeline()
        proba = fit_predict_proba(pipe, X_tr, y_tr, X_te)
        preds.append(proba)
    if not preds:
        return pd.Series(dtype=float, name="proba_up")
    return pd.concat(preds).sort_index()


def run_phase1(cfg: PipelineConfig | None = None) -> dict:
    """End-to-end Phase 1. Returns a dict with key artefacts."""
    cfg = cfg or PipelineConfig()
    t0 = time.time()
    log.info(
        "Phase 1 starting: start=%s end=%s n_tickers=%s",
        cfg.start_date,
        cfg.end_date,
        cfg.n_tickers,
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

    log.info("Building features...")
    feats = build_feature_matrix(close, volume)
    log.info("Feature matrix: %s rows x %s cols", *feats.shape)

    log.info("Building labels for horizon %d...", cfg.horizon)
    labels = long_labels(close, horizons=(cfg.horizon,))

    X, y_bin, y_cont = assemble_dataset(feats, labels, cfg.horizon)
    if cfg.feature_cols is not None:
        X = X[[c for c in cfg.feature_cols if c in X.columns]]
    log.info("Dataset assembled: X=%s, y=%s", X.shape, y_bin.shape)

    log.info("Walk-forward training baseline...")
    proba = walk_forward_predict(X, y_bin, cfg.cv)
    log.info("OOS predictions: %d rows", len(proba))

    # IC + hit-rate diagnostics: use continuous return for IC, binary for hit-rate.
    aligned = pd.concat([proba.rename("p"), y_cont.rename("r"), y_bin.rename("d")], axis=1).dropna()
    ic = information_coefficient(aligned["p"], aligned["r"])
    ic_stats = ic_summary(ic)
    hit = float(((aligned["p"] > 0.5).astype(int) == aligned["d"]).mean())
    log.info("Hit rate (OOS, baseline): %.4f", hit)
    log.info("IC summary: %s", ic_stats)

    # Convert probability to centred score for ranking.
    score = (proba - 0.5).rename("score")
    weights = top_bottom_k_weights(score, k=cfg.k_per_side)
    if weights.empty:
        raise RuntimeError("Portfolio is empty: too few tickers per day for k.")

    # Rebalance only every R trading days. With daily rebalancing and 60+ bps
    # round-trip cost, even a strong daily-horizon signal is buried by turnover.
    # Hold the latest signal for (rebalance_every) days before refreshing.
    R = cfg.rebalance_every if cfg.rebalance_every is not None else cfg.horizon
    if R > 1:
        # Reindex weights to a coarser cadence: keep every R-th row, ffill the rest.
        keep_rows = weights.index[::R]
        weights = weights.loc[keep_rows].reindex(weights.index).ffill().fillna(0.0)

    bt_cfg = BacktestConfig()
    # trade_lag=2 because labels are computed with trade_next_open=True
    # (forward return from close[t+1] to close[t+1+h]). The backtester must
    # realise P&L over the SAME window the model was trained to predict.
    res = run_backtest(weights, close, cfg=bt_cfg, trade_lag=2)
    metrics = tearsheet_metrics(res.returns)
    log.info("Backtest metrics: %s", metrics)

    out_path = REPORTS_DIR / f"phase1_h{cfg.horizon}_k{cfg.k_per_side}.html"
    # Benchmark: equal-weight long-only on the universe.
    bench_ret = close.pct_change().mean(axis=1)
    build_tearsheet(
        res.returns,
        out_path,
        benchmark=bench_ret,
        cost_bps_per_side=bt_cfg.total_cost_per_side_bps,
    )
    elapsed = time.time() - t0
    log.info("Tearsheet written -> %s (elapsed %.1fs)", out_path, elapsed)

    return {
        "tickers": tickers,
        "feature_matrix_shape": X.shape,
        "predictions": proba,
        "ic_summary": ic_stats,
        "hit_rate": hit,
        "metrics": metrics,
        "weights": weights,
        "backtest": res,
        "tearsheet_path": out_path,
        "elapsed_s": elapsed,
    }
