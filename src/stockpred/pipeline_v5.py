"""Phase 5 pipeline — assembled from the Phase 3/4 building blocks.

Improvements over `pipeline.py` (Phase 2):

  1. IC-IR-weighted ensemble instead of equal-weight. Horizons with
     out-of-sample IC IR <= 0 are dropped entirely. (The Phase 2 run
     showed h=21d has no signal; the equal-weight ensemble was dragging
     down the strong h=5d signal.)

  2. Vol-scaled top-K position sizing (signal x inverse-vol, normalised
     per side) instead of equal-weight top-K.

  3. Sector exposure caps (default 30% gross per sector).

  4. Minimum trade threshold (skip rebalances below 0.5%) to suppress
     noise-trading that just pays costs.

  5. Held-out window: the last N years are NEVER used in CV, model
     selection, or any pipeline decision; they are only used to compute
     final out-of-sample metrics.

  6. Bootstrap Sharpe confidence interval reported alongside point
     estimate. If 0 sits inside the CI, the strategy is not statistically
     distinguishable from random.

  7. Per-regime breakdown (VIX quintile) so the user can see whether the
     strategy works equally in calm and stressed markets.

These are the changes the strategy-research sub-agent identified as
"Tier 1 highest-ROI fixes" (see docs/PROJECT_LOG.md "Phase 5 research").

This module shares the data-loading and feature-building code with
pipeline.py via direct imports; it only changes the ensemble, portfolio
construction, and reporting layers.
"""

from __future__ import annotations

import dataclasses as _dc
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stockpred.backtest.engine import run_backtest
from stockpred.backtest.portfolio import (
    apply_min_trade_threshold,
    apply_sector_caps,
    ic_ir_weighted_ensemble,
    top_bottom_k_weights,
    vol_scaled_weights,
)
from stockpred.config import (
    REPORTS_DIR,
    BacktestConfig,
    CVConfig,
)
from stockpred.data import fundamentals as fundamentals_mod
from stockpred.data import macro as macro_mod
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
from stockpred.pipeline import (  # reuse Phase 2 helpers
    _diagnostics,
    assemble_dataset,
    build_feature_matrix,
    select_universe,
    walk_forward_predict,
)
from stockpred.reports.tearsheet import build_tearsheet
from stockpred.validation.metrics import tearsheet_metrics
from stockpred.validation.stress import (
    bootstrap_sharpe,
    holdout_split_dates,
    regime_breakdown,
    vix_regime,
)

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #


@dataclass
class PipelineV5Config:
    """All knobs for a Phase 5 run."""

    # Universe / history
    start_date: str = "2010-01-01"
    end_date: str | None = None
    n_tickers: int | None = 100
    universe_sampling: str = "random"
    refresh_data: bool = False

    # Horizons + model
    horizons: tuple[int, ...] = (1, 5)  # 21d dropped by default (Phase 2 showed no signal)
    model: str = "gbm"
    gbm: GBMConfig = field(default_factory=GBMConfig)
    use_sector_features: bool = True

    # Validation
    cv: CVConfig = field(
        default_factory=lambda: CVConfig(
            train_years=3,
            test_months=6,
            embargo_days=25,
            min_train_obs=1000,
        )
    )
    holdout_years: int = 2  # last N years untouched by CV / model selection

    # Portfolio construction (the heart of Phase 5)
    position_sizing: str = "vol_scaled"  # {"vol_scaled", "top_k"}
    k_per_side_pct: float = 0.15  # top/bottom 15% per side
    leverage_per_side: float = 1.0
    sector_cap_gross: float | None = 0.30
    min_trade_threshold: float = 0.005

    # Ensemble
    ensemble_weighting: str = "ic_ir"  # {"ic_ir", "equal"}

    # Stress
    bootstrap_n: int = 500

    # Output
    tearsheet_path: Path | None = None


# --------------------------------------------------------------------- #
# Helper: compute trailing vol per ticker (lag-safe)
# --------------------------------------------------------------------- #


def _trailing_vol(close: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """Per-ticker daily-return std over `window` trading days, computed using
    only past returns. Aligned to the close index.

    Critical: returned at date `t` reflects vol *through close of t-1*, so
    when consumed at signal time t it is lag-safe.
    """
    log_ret = np.log(close).diff()
    vol = log_ret.rolling(window, min_periods=window).std()
    return vol.shift(1)  # ensure no same-day vol leakage


# --------------------------------------------------------------------- #
# Ensemble + portfolio
# --------------------------------------------------------------------- #


def _build_weights(
    cfg: PipelineV5Config,
    per_horizon_preds: dict[int, pd.Series],
    per_horizon_diag: dict[int, dict],
    close: pd.DataFrame,
    sector_map: dict[str, str],
) -> pd.DataFrame:
    """Compose per-horizon predictions into the final wide weights frame."""
    # 1) Ensemble.
    if cfg.ensemble_weighting == "ic_ir":
        ic_ir = {h: float(per_horizon_diag.get(h, {}).get("ic_ir", 0.0)) for h in per_horizon_preds}
        log.info("IC-IR ensemble weights (pre-normalisation): %s", ic_ir)
        score = ic_ir_weighted_ensemble(per_horizon_preds, ic_ir)
    else:
        # equal-weight z-scores across horizons (Phase 2 default)
        from stockpred.pipeline import ensemble_predictions

        score = ensemble_predictions(per_horizon_preds)

    # 2) Position sizing.
    if cfg.position_sizing == "vol_scaled":
        vol = _trailing_vol(close, window=21)
        weights = vol_scaled_weights(
            score,
            vol,
            leverage_per_side=cfg.leverage_per_side,
            top_fraction=cfg.k_per_side_pct,
        )
    else:
        # interpret k_per_side_pct as a fraction of the universe per side
        kk = max(1, int(close.shape[1] * cfg.k_per_side_pct))
        weights = top_bottom_k_weights(score, k=kk, leverage_per_side=cfg.leverage_per_side)

    # 3) Sector caps.
    if cfg.sector_cap_gross is not None and sector_map:
        weights = apply_sector_caps(weights, sector_map, max_per_sector_gross=cfg.sector_cap_gross)

    # 4) Minimum trade threshold.
    if cfg.min_trade_threshold > 0:
        weights = apply_min_trade_threshold(weights, min_abs_delta=cfg.min_trade_threshold)

    return weights


# --------------------------------------------------------------------- #
# Holdout helpers
# --------------------------------------------------------------------- #


def _split_holdout(
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    holdout_years: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = feats.index.get_level_values("date").unique().sort_values()
    dev_dates, hold_dates = holdout_split_dates(dates, holdout_years=holdout_years)
    dev_mask = feats.index.get_level_values("date").isin(dev_dates)
    hold_mask = feats.index.get_level_values("date").isin(hold_dates)
    return (
        feats[dev_mask],
        labels[labels.index.get_level_values("date").isin(dev_dates)],
        feats[hold_mask],
        labels[labels.index.get_level_values("date").isin(hold_dates)],
    )


# --------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------- #


def run_pipeline_v5(cfg: PipelineV5Config | None = None) -> dict:
    """End-to-end Phase 5 pipeline. Returns artefacts + holdout metrics."""
    cfg = cfg or PipelineV5Config()
    t0 = time.time()
    log.info(
        "Phase 5 pipeline starting: model=%s horizons=%s n_tickers=%s [%s..%s] holdout=%dy",
        cfg.model,
        cfg.horizons,
        cfg.n_tickers,
        cfg.start_date,
        cfg.end_date or "today",
        cfg.holdout_years,
    )

    # Reuse Phase 2 universe selection (already de-survivored).
    from stockpred.pipeline import PipelineConfig as _PCfg

    universe_cfg = _PCfg(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        n_tickers=cfg.n_tickers,
        universe_sampling=cfg.universe_sampling,
        refresh_data=cfg.refresh_data,
    )
    tickers, _ = select_universe(universe_cfg)
    log.info("Universe size: %d", len(tickers))

    log.info("Fetching prices (cached)...")
    raw_panel = prices_mod.long_panel(
        tickers, start=cfg.start_date, end=cfg.end_date, refresh=cfg.refresh_data
    )
    if raw_panel.empty:
        raise RuntimeError("No price data retrieved.")
    close = raw_panel["adj_close"].unstack("ticker").sort_index()
    volume = raw_panel["volume"].unstack("ticker").sort_index()
    log.info("Loaded prices: %d dates x %d tickers", close.shape[0], close.shape[1])

    sector_map: dict[str, str] = {}
    if cfg.use_sector_features:
        try:
            funds = fundamentals_mod.fetch_fundamentals(tickers, refresh=cfg.refresh_data)
            sector_map = fundamentals_mod.sector_map(funds)
            log.info(
                "Sectors tagged for %d / %d",
                sum(1 for t in tickers if t in sector_map),
                len(tickers),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Sector load failed (%s); continuing without.", e)

    log.info("Building features...")
    feats = build_feature_matrix(
        close, volume, sector_map=sector_map, use_sector_features=cfg.use_sector_features
    )
    log.info("Feature matrix: %s rows x %s cols", *feats.shape)

    log.info("Building labels for horizons %s...", cfg.horizons)
    labels = long_labels(close, horizons=tuple(cfg.horizons), include_vol_scaled=True)

    # ---------------- Holdout split (touch dev only for training) -----
    dev_feats, dev_labels, hold_feats, hold_labels = _split_holdout(
        feats, labels, cfg.holdout_years
    )
    log.info(
        "Holdout split: dev dates %d, holdout dates %d",
        dev_feats.index.get_level_values("date").nunique(),
        hold_feats.index.get_level_values("date").nunique(),
    )

    # ---------------- Train per-horizon on dev with walk-forward CV ---
    per_horizon_preds: dict[int, pd.Series] = {}
    per_horizon_diag: dict[int, dict] = {}
    per_horizon_returns: dict[int, pd.Series] = {}
    for h in cfg.horizons:
        log.info("=== Horizon %d ===", h)
        target = "vs" if cfg.model == "gbm" else "dir"
        X, y_target, y_return, y_bin = assemble_dataset(dev_feats, dev_labels, h, target=target)
        log.info("Dev dataset h=%d: X=%s", h, X.shape)

        pred = walk_forward_predict(X, y_target, cfg.cv, model=cfg.model, gbm_cfg=cfg.gbm)
        if pred.empty:
            log.warning("Horizon %d produced no predictions; skipping", h)
            continue
        if cfg.model == "logistic":
            pred = pred - 0.5

        hit, ic_stats = _diagnostics(pred, y_return, y_bin)
        log.info(
            "Horizon %d DEV OOS: hit=%.4f ic_mean=%+.5f ic_ir=%+.3f",
            h,
            hit,
            ic_stats["ic_mean"],
            ic_stats["ic_ir"],
        )
        per_horizon_preds[h] = pred
        per_horizon_returns[h] = y_return
        per_horizon_diag[h] = {"hit_rate": hit, **ic_stats}

    if not per_horizon_preds:
        raise RuntimeError("All horizons failed.")

    # ---------------- Portfolio + backtest on the DEV span -----------
    dev_weights = _build_weights(cfg, per_horizon_preds, per_horizon_diag, close, sector_map)
    if dev_weights.empty:
        raise RuntimeError("Dev portfolio is empty.")
    bt_cfg = BacktestConfig()
    bt_horizon = min(per_horizon_preds.keys())
    dev_bt = run_backtest(dev_weights, close, cfg=bt_cfg, horizon=bt_horizon, trade_lag=1)
    dev_metrics = tearsheet_metrics(dev_bt.returns)
    log.info("DEV backtest metrics: %s", dev_metrics)

    # ---------------- Score the holdout window with the SAME model ---
    # We do NOT re-train on holdout. We use the predictions the walk-forward
    # CV already produced for dev, plus a final fold that uses ALL dev data
    # to produce holdout-period predictions.
    hold_preds: dict[int, pd.Series] = {}
    for h in cfg.horizons:
        if h not in per_horizon_preds:
            continue
        log.info("Scoring holdout for h=%d...", h)
        target = "vs" if cfg.model == "gbm" else "dir"
        X_dev, y_dev, _, _ = assemble_dataset(dev_feats, dev_labels, h, target=target)
        X_hold, y_hold, y_ret_hold, y_bin_hold = assemble_dataset(
            hold_feats, hold_labels, h, target=target
        )
        if X_hold.empty:
            log.warning("h=%d holdout feature matrix empty; skipping", h)
            continue
        if cfg.model == "logistic":
            pipe = make_baseline_pipeline()
            pred = fit_predict_proba(pipe, X_dev, y_dev, X_hold) - 0.5
        else:
            # Fix C2 (review finding): chronological train/valid split, not
            # positional. We sort dev by date so the last 10% is the most
            # recent dev period — never interleaved with earlier training.
            X_dev_sorted = X_dev.sort_index(level="date")
            y_dev_sorted = y_dev.reindex(X_dev_sorted.index)
            split = max(1, int(len(X_dev_sorted) * 0.9))
            tr_dates = X_dev_sorted.iloc[:split].index.get_level_values("date").max()
            va_dates = X_dev_sorted.iloc[split:].index.get_level_values("date").min()
            assert tr_dates < va_dates, (
                f"Internal valid split is interleaved: train.max={tr_dates} >= valid.min={va_dates}"
            )
            booster = train_gbm(
                X_dev_sorted.iloc[:split],
                y_dev_sorted.iloc[:split],
                X_valid=X_dev_sorted.iloc[split:],
                y_valid=y_dev_sorted.iloc[split:],
                cfg=cfg.gbm,
            )
            pred = predict_gbm(booster, X_hold)

        hit_h, ic_h = _diagnostics(pred, y_ret_hold, y_bin_hold)
        log.info(
            "Horizon %d HOLDOUT: hit=%.4f ic_mean=%+.5f ic_ir=%+.3f",
            h,
            hit_h,
            ic_h["ic_mean"],
            ic_h["ic_ir"],
        )
        per_horizon_diag.setdefault(h, {})["holdout_hit_rate"] = hit_h
        per_horizon_diag[h]["holdout_ic_mean"] = ic_h["ic_mean"]
        per_horizon_diag[h]["holdout_ic_ir"] = ic_h["ic_ir"]
        hold_preds[h] = pred

    # Build holdout weights and run holdout backtest.
    hold_metrics: dict = {}
    hold_bt = None
    hold_score = pd.Series(dtype=float, name="ensemble_score")
    if hold_preds:
        hold_weights = _build_weights(cfg, hold_preds, per_horizon_diag, close, sector_map)
        if not hold_weights.empty:
            hold_bt = run_backtest(hold_weights, close, cfg=bt_cfg, horizon=bt_horizon, trade_lag=1)
            hold_metrics = tearsheet_metrics(hold_bt.returns)
            log.info("HOLDOUT backtest metrics: %s", hold_metrics)
            from stockpred.pipeline import ensemble_predictions

            if cfg.ensemble_weighting == "ic_ir":
                ic_ir = {
                    h: float(per_horizon_diag.get(h, {}).get("ic_ir", 0.0)) for h in hold_preds
                }
                hold_score = ic_ir_weighted_ensemble(hold_preds, ic_ir)
            else:
                hold_score = ensemble_predictions(hold_preds)

    # ---------------- Bootstrap Sharpe CI on HOLDOUT -----------------
    ci = {}
    if hold_bt is not None and not hold_bt.returns.dropna().empty:
        ci = bootstrap_sharpe(hold_bt.returns, n_resamples=cfg.bootstrap_n)
        log.info(
            "HOLDOUT bootstrap Sharpe: %.3f  [%.3f, %.3f] @ %.0f%%",
            ci["sharpe"],
            ci["sharpe_lo"],
            ci["sharpe_hi"],
            ci["ci_pct"] * 100,
        )

    # ---------------- Regime breakdown on HOLDOUT --------------------
    regimes_df: pd.DataFrame | None = None
    if hold_bt is not None:
        try:
            macro = macro_mod.fetch_macro(("VIXCLS",))
            vix = macro["VIXCLS"].dropna()
            vix_reindexed = vix.reindex(hold_bt.returns.index).ffill().dropna()
            reg = vix_regime(vix_reindexed, q=4)
            regimes_df = regime_breakdown(hold_bt.returns, reg)
            log.info("HOLDOUT regime breakdown:\n%s", regimes_df.to_string())
        except Exception as e:  # noqa: BLE001
            log.warning("Regime breakdown failed (%s); continuing.", e)

    # ---------------- Tearsheet on the DEV span (so equity has length) -
    out_path = cfg.tearsheet_path or REPORTS_DIR / (
        f"phase5_{cfg.model}_h{'-'.join(str(h) for h in cfg.horizons)}_"
        f"{cfg.position_sizing}_{cfg.ensemble_weighting}.html"
    )
    bench_ret = close.pct_change().mean(axis=1)
    build_tearsheet(
        dev_bt.returns,
        out_path,
        benchmark=bench_ret,
        cost_bps_per_side=bt_cfg.total_cost_per_side_bps,
    )

    elapsed = time.time() - t0
    log.info("Phase 5 complete in %.1fs", elapsed)

    # Make the score schema compatible with the snapshot writer.
    if cfg.ensemble_weighting == "ic_ir":
        ic_ir_d = {
            h: float(per_horizon_diag.get(h, {}).get("ic_ir", 0.0)) for h in per_horizon_preds
        }
        ensemble_score = ic_ir_weighted_ensemble(per_horizon_preds, ic_ir_d)
    else:
        from stockpred.pipeline import ensemble_predictions

        ensemble_score = ensemble_predictions(per_horizon_preds)

    return {
        "tickers": tickers,
        "feature_matrix_shape": feats.shape,
        "per_horizon_predictions": per_horizon_preds,
        "per_horizon_diagnostics": per_horizon_diag,
        "ensemble_score": ensemble_score,
        "weights": dev_weights,
        "backtest": dev_bt,
        "metrics": dev_metrics,
        "holdout_metrics": hold_metrics,
        "holdout_backtest": hold_bt,
        "holdout_ensemble_score": hold_score,
        "bootstrap_sharpe": ci,
        "regime_breakdown": regimes_df.to_dict() if regimes_df is not None else {},
        "tearsheet_path": out_path,
        "elapsed_s": elapsed,
        "config": _dc.asdict(cfg),
    }
