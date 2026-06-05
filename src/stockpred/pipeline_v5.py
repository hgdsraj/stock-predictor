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
    neutralise_portfolio_beta,
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
from stockpred.features.regime import broadcast_to_panel, compute_regime_features
from stockpred.features.technical import compute_technical_features
from stockpred.features.tier2 import compute_tier2_features
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
    use_tier2_features: bool = True  # Phase 6: 12-1 momentum, IVOL, beta, max ret, Amihud
    use_regime_features: bool = True  # Phase 6: VIX, term spread, USD, xs dispersion
    beta_neutralise: bool = False  # Phase 6: portfolio-level beta-vs-SPY neutralisation
    bootstrap_method: str = "block"  # Phase 6: 'block' or 'iid'

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

    # Portfolio construction (the heart of Phase 5/6/7)
    position_sizing: str = "vol_scaled"  # {"vol_scaled", "top_k", "hrp"} — "hrp" is Phase 7
    k_per_side_pct: float = 0.15  # top/bottom 15% per side
    leverage_per_side: float = 1.0
    sector_cap_gross: float | None = 0.30
    min_trade_threshold: float = 0.005

    # Ensemble
    ensemble_weighting: str = "ic_ir"  # {"ic_ir", "equal"}

    # Phase 8: meta-labelling (López de Prado Ch. 3.6)
    # When enabled, a secondary binary GBM is trained per fold predicting
    # P(primary ensemble score has correct sign). The primary score is
    # then gated: rows with P(correct) < meta_threshold are zeroed out.
    # This typically reduces turnover and improves precision at the cost
    # of recall — useful when transaction costs dominate.
    use_meta_labelling: bool = False
    meta_threshold: float = 0.55
    # Phase 9a: confidence-weighted sizing.
    # 'binary'     -> traditional gate (zero if P < threshold, full size otherwise)
    # 'confidence' -> scale signal by clip(2*(P-floor)/(cap-floor), 0, 1).
    # 'confidence' preserves magnitude info; 'binary' is simpler.
    meta_mode: str = "binary"
    meta_conf_floor: float = 0.5
    meta_conf_cap: float = 1.0
    # Phase 9c: number of walk-forward folds for the meta-gate.
    # K=1 -> Phase 8 single-pass (50/50 train/predict).
    # K>1 -> proper expanding-window CV; cleaner but K* slower per gate.
    meta_walk_forward_folds: int = 1
    # Phase 9d: sector-conditional meta classifiers. When True, one meta
    # GBM is trained per sector; each ticker is gated by its sector's
    # classifier. Requires sector_map to be populated (i.e. fundamentals
    # successfully loaded). Falls back to global meta if not.
    meta_per_sector: bool = False

    # Phase 8: triple-barrier labels (López de Prado Ch. 3)
    # When enabled, the regression target switches from `fwd_vs_h` to the
    # triple-barrier signed return `tb_return * tb_label` per horizon.
    # The barriers are set at ±tb_k_sigma trailing-vol units; the vertical
    # barrier is at the horizon's max_horizon (defaults to h).
    use_triple_barrier_labels: bool = False
    tb_k_sigma: float = 2.0

    # Phase 8: drop raw feature columns, keep only cross-sectional ranks
    # (and sector dummies + regime broadcasts). Per-feature audit showed
    # raw columns degrade much more under hard-cutoff than their ranked
    # versions, suggesting the ranks carry the stable signal and the raw
    # versions add noise + same-day regime-level dependence.
    ranks_only: bool = False

    # Phase 11: explicit feature-name blocklist. Any column whose name is
    # in this tuple is dropped from `feats` right after `ranks_only` is
    # applied. Used by `scripts/phase11_feature_pruning.py` to test whether
    # removing low-information features (per per-feature audit) improves
    # holdout. Empty tuple = no pruning (default).
    feature_exclude: tuple[str, ...] = ()

    # Phase 12: SEC EDGAR 8-K event features. When True, fetches all
    # 8-K filings in the backtest range, maps to per-ticker daily
    # has_8k flag + rolling counts (5/21/63d windows). Columns are
    # prefixed `edgar_` so they survive `ranks_only` filtering. Free,
    # no API key required; respects SEC's 10 req/sec rate limit and
    # User-Agent rule (set EDGAR_USER_AGENT env var to override).
    use_edgar_features: bool = False

    # Phase 13: SEC EDGAR 8-K item-code features. When True, fetches
    # per-company 8-K item history (item 5.02 = CEO change, item 2.02
    # = earnings, etc.) and builds per-item-family flags + rolling
    # counts. Columns are prefixed `edgaritem_` (still matches the
    # `edgar` prefix kept by ranks_only). This is the "with sentiment
    # direction" version of Phase 12's raw counts. Same SEC rate limit
    # / User-Agent rules apply.
    use_edgar_item_features: bool = False

    # Phase 14: GDELT GKG daily tone + mention features. Reads ONLY
    # from per-day parquet caches (bulk fetch done overnight by
    # `scripts/phase14_gdelt_bulk_fetch.py`). Columns prefixed `gdelt_`
    # so they survive ranks_only via the shared `gdelt` prefix.
    use_gdelt_features: bool = False

    # Phase 19: per-ticker Bayesian shrinkage of ensemble score by
    # historical sign-precision. alpha=0 disables (raw scores pass
    # through). alpha=1.0 is full shrinkage. Tickers with worse-than-
    # random sign-precision are dropped entirely (factor = 0).
    bayesian_shrinkage_alpha: float = 0.0
    bayesian_shrinkage_min_obs: int = 30

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
    asset_betas: pd.DataFrame | None = None,
    precomputed_score: pd.Series | None = None,
) -> pd.DataFrame:
    """Compose per-horizon predictions into the final wide weights frame.

    Phase 8 (review C1 fix): callers that have already gated / processed an
    ensemble score (e.g. via meta-labelling) can pass `precomputed_score`
    directly. We then skip the per-horizon ensemble step entirely so that
    gated zeros aren't re-z-scored against survivors (which would silently
    flip "don't trade" into "active short" via the top-k ranking).
    """
    # 1) Ensemble (skipped when caller supplied a precomputed score).
    if precomputed_score is not None:
        score = precomputed_score
        log.info("Using precomputed score (ensemble step skipped)")
    elif cfg.ensemble_weighting == "ic_ir":
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
    elif cfg.position_sizing == "hrp":
        from stockpred.backtest.hrp import HRPConfig, hrp_long_short_weights

        hcfg = HRPConfig(
            cov_window=60,
            top_fraction=cfg.k_per_side_pct,
            leverage_per_side=cfg.leverage_per_side,
            use_ledoit_wolf=True,
        )
        weights = hrp_long_short_weights(score, close, cfg=hcfg)
    else:
        kk = max(1, int(close.shape[1] * cfg.k_per_side_pct))
        weights = top_bottom_k_weights(score, k=kk, leverage_per_side=cfg.leverage_per_side)

    # 3) Sector caps.
    if cfg.sector_cap_gross is not None and sector_map:
        weights = apply_sector_caps(weights, sector_map, max_per_sector_gross=cfg.sector_cap_gross)

    # 4) Minimum trade threshold.
    if cfg.min_trade_threshold > 0:
        weights = apply_min_trade_threshold(weights, min_abs_delta=cfg.min_trade_threshold)

    # 5) Beta-neutralisation vs SPY (Phase 6).
    if cfg.beta_neutralise and asset_betas is not None and not asset_betas.empty:
        weights = neutralise_portfolio_beta(weights, asset_betas, target=0.0)

    return weights


# --------------------------------------------------------------------- #
# Holdout helpers
# --------------------------------------------------------------------- #


def _apply_meta_gate_per_sector(
    ensemble_score: pd.Series,
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    sector_map: dict[str, str],
    *,
    bt_horizon_for_meta: int,
    threshold: float,
    gbm_cfg,
    log,
    mode: str = "binary",
    conf_floor: float = 0.5,
    conf_cap: float = 1.0,
    walk_forward_folds: int = 1,
) -> pd.Series:
    """Phase 9d: per-sector meta-gating. For each sector, recurse into
    _apply_meta_gate on the subset of tickers in that sector. Tickers
    without a sector tag are gated using a global meta classifier.

    Returns a single Series with the same index as `ensemble_score`,
    constructed by concatenating per-sector gated outputs.
    """
    if not sector_map:
        log.warning(
            "meta-per-sector requested but sector_map is empty; falling back to global meta"
        )
        return _apply_meta_gate(
            ensemble_score,
            feats,
            labels,
            bt_horizon_for_meta=bt_horizon_for_meta,
            threshold=threshold,
            gbm_cfg=gbm_cfg,
            log=log,
            mode=mode,
            conf_floor=conf_floor,
            conf_cap=conf_cap,
            walk_forward_folds=walk_forward_folds,
        )

    # Review C1 fix: per-sector meta should learn sector-conditional patterns
    # in isolation. Drop columns that encode universe-wide cross-sectional
    # information (ranks computed across the full universe, regime
    # broadcasts) so the sector-specific classifier doesn't see what other
    # sectors did at date t. Sector dummies and ticker-local technicals stay.
    cross_sectional_suffixes = ("_rank",)
    cross_sectional_prefixes = ("reg_",)
    feats_sector_local = feats[
        [
            c
            for c in feats.columns
            if not c.endswith(cross_sectional_suffixes)
            and not c.startswith(cross_sectional_prefixes)
        ]
    ]
    n_dropped = feats.shape[1] - feats_sector_local.shape[1]
    if n_dropped > 0:
        log.info(
            "meta-per-sector: dropped %d cross-sectional cols (ranks/regime) "
            "to isolate sector-specific learning",
            n_dropped,
        )

    pieces: list[pd.Series] = []
    # Group tickers by sector.
    tickers_by_sector: dict[str, list[str]] = {}
    untagged: list[str] = []
    all_tickers = ensemble_score.index.get_level_values("ticker").unique()
    for t in all_tickers:
        sec = sector_map.get(t)
        if sec is None:
            untagged.append(t)
        else:
            tickers_by_sector.setdefault(sec, []).append(t)

    if untagged:
        log.info("meta-per-sector: %d untagged tickers will use global meta", len(untagged))

    for sector, tickers in tickers_by_sector.items():
        if len(tickers) < 5:
            log.info(
                "meta-per-sector: sector '%s' has %d tickers; skipping per-sector "
                "meta (insufficient data), passing through",
                sector,
                len(tickers),
            )
            sub_mask = ensemble_score.index.get_level_values("ticker").isin(tickers)
            pieces.append(ensemble_score[sub_mask])
            continue
        sub_mask = ensemble_score.index.get_level_values("ticker").isin(tickers)
        feat_mask = feats_sector_local.index.get_level_values("ticker").isin(tickers)
        label_mask = labels.index.get_level_values("ticker").isin(tickers)
        sub_score = ensemble_score[sub_mask]
        sub_feats = feats_sector_local[feat_mask]
        sub_labels = labels[label_mask]
        log.info(
            "meta-per-sector: sector '%s' n_tickers=%d n_obs=%d",
            sector,
            len(tickers),
            len(sub_score),
        )
        gated_sub = _apply_meta_gate(
            sub_score,
            sub_feats,
            sub_labels,
            bt_horizon_for_meta=bt_horizon_for_meta,
            threshold=threshold,
            gbm_cfg=gbm_cfg,
            log=log,
            mode=mode,
            conf_floor=conf_floor,
            conf_cap=conf_cap,
            walk_forward_folds=walk_forward_folds,
        )
        pieces.append(gated_sub)

    if untagged:
        u_mask = ensemble_score.index.get_level_values("ticker").isin(untagged)
        feat_mask = feats_sector_local.index.get_level_values("ticker").isin(untagged)
        label_mask = labels.index.get_level_values("ticker").isin(untagged)
        sub_score = ensemble_score[u_mask]
        sub_feats = feats_sector_local[feat_mask]
        sub_labels = labels[label_mask]
        gated_u = _apply_meta_gate(
            sub_score,
            sub_feats,
            sub_labels,
            bt_horizon_for_meta=bt_horizon_for_meta,
            threshold=threshold,
            gbm_cfg=gbm_cfg,
            log=log,
            mode=mode,
            conf_floor=conf_floor,
            conf_cap=conf_cap,
            walk_forward_folds=walk_forward_folds,
        )
        pieces.append(gated_u)

    if not pieces:
        return ensemble_score
    return pd.concat(pieces).sort_index()


def _apply_meta_gate(
    ensemble_score: pd.Series,
    feats: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    bt_horizon_for_meta: int,
    threshold: float,
    gbm_cfg,
    log,
    mode: str = "binary",
    conf_floor: float = 0.5,
    conf_cap: float = 1.0,
    walk_forward_folds: int = 1,
) -> pd.Series:
    """Phase 8: train a binary GBM to predict P(ensemble_score is correct)
    and gate the score on that probability.

    Splits the input window into 80% train / 20% predict by date. Trains
    meta on train, predicts on predict. The early-train portion (no meta
    prediction) is returned unmodified — we only gate where we have a
    prediction.

    Phase 9a — `mode` parameter:
      "binary"     -> hard threshold (zero if P(correct) < threshold).
      "confidence" -> scale signal by clip((P - floor) / (cap - floor), 0, 1).
                      Preserves magnitude information instead of discarding it.

    This is a single-pass implementation: for stricter walk-forward you
    would loop folds inside here. The 80/20 single-pass is enough to give
    a sense of whether the meta layer helps at all without quintupling
    runtime.
    """
    from stockpred.models.meta import (
        build_meta_dataset,
        fit_meta,
        meta_confidence_weight_signal,
        meta_filter_signal,
        predict_meta,
    )

    realised_col = f"fwd_return_{bt_horizon_for_meta}"
    if realised_col not in labels.columns:
        log.warning(
            "meta-gate: realised return column %s not in labels; skipping gate",
            realised_col,
        )
        return ensemble_score

    realised = labels[realised_col]
    # Forbid any column that obviously leaks. The meta module also enforces
    # this, but pre-filtering keeps the error message clean.
    forbidden_prefixes = ("fwd_", "tb_")
    safe_feats = feats[[c for c in feats.columns if not c.startswith(forbidden_prefixes)]]

    all_dates = ensemble_score.index.get_level_values("date").unique().sort_values()
    if len(all_dates) < 20:
        log.warning("meta-gate: not enough dates; skipping gate")
        return ensemble_score

    # Phase 9c: walk-forward meta-CV. K folds, each rolling-train on dates
    # prior to a test slice. K=1 reduces to the Phase 8 single 80/20 split.
    K = max(1, int(walk_forward_folds))
    # Reserve the first 50% as initial training history; partition the
    # remaining 50% into K equally-sized prediction slices.
    train_end_idx = max(1, int(len(all_dates) * 0.5))
    remaining = len(all_dates) - train_end_idx
    if remaining <= 0:
        log.warning("meta-gate: not enough dates for walk-forward; skipping")
        return ensemble_score
    slice_size = max(1, remaining // K)

    try:
        gated_pieces: list[pd.Series] = []
        # Always include the initial training window unchanged.
        init_train_mask = ensemble_score.index.get_level_values("date").isin(
            all_dates[:train_end_idx]
        )
        gated_pieces.append(ensemble_score[init_train_mask])

        for k in range(K):
            tr_end = train_end_idx + k * slice_size
            pr_start = tr_end
            pr_end = pr_start + slice_size if k < K - 1 else len(all_dates)
            tr_dates = all_dates[:tr_end]
            pr_dates = all_dates[pr_start:pr_end]
            if len(tr_dates) < 10 or len(pr_dates) < 1:
                continue
            tr_mask = ensemble_score.index.get_level_values("date").isin(tr_dates)
            pr_mask = ensemble_score.index.get_level_values("date").isin(pr_dates)
            score_train = ensemble_score[tr_mask]
            score_pred = ensemble_score[pr_mask]
            X_meta_train, y_meta_train = build_meta_dataset(
                score_train,
                realised.reindex(score_train.index),
                safe_feats.loc[score_train.index],
                use_primary_score=True,
            )
            if y_meta_train.empty or y_meta_train.nunique() < 2:
                log.warning(
                    "meta-gate fold %d/%d: insufficient class diversity; passing through",
                    k + 1,
                    K,
                )
                gated_pieces.append(score_pred)
                continue
            booster = fit_meta(X_meta_train, y_meta_train, cfg=gbm_cfg)
            X_meta_pred = safe_feats.loc[score_pred.index].copy()
            X_meta_pred["primary_abs"] = score_pred.abs()
            proba = predict_meta(booster, X_meta_pred)
            if mode == "confidence":
                gated = meta_confidence_weight_signal(
                    score_pred,
                    proba,
                    floor=conf_floor,
                    cap=conf_cap,
                )
            else:
                gated = meta_filter_signal(score_pred, proba, p_threshold=threshold)
            log.info(
                "meta-gate fold %d/%d (mode=%s): train n=%d, pred n=%d, %.1f%% survived gate",
                k + 1,
                K,
                mode,
                len(y_meta_train),
                len(score_pred),
                100 * (gated != 0).mean(),
            )
            gated_pieces.append(gated)

        if not gated_pieces:
            return ensemble_score
        # Review C2 fix: verify_integrity=True surfaces any silent fold-
        # boundary bug as a loud failure rather than a silent duplicate.
        out = pd.concat(gated_pieces, verify_integrity=True).sort_index()
        return out
    except Exception as e:  # noqa: BLE001
        log.warning("meta-gate failed (%s); returning ungated score", e)
        return ensemble_score


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
    del raw_panel  # free long-format panel (~300 MB for full S&P 500)
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

    log.info("Building features (tier-1 technicals + sector)...")
    feats = build_feature_matrix(
        close, volume, sector_map=sector_map, use_sector_features=cfg.use_sector_features
    )

    # Optional benchmark (SPY) for tier-2 + regime features.
    spy_close: pd.Series | None = None
    if cfg.use_tier2_features or cfg.use_regime_features or cfg.beta_neutralise:
        try:
            spy_df = prices_mod.fetch_one("SPY")
            if not spy_df.empty:
                spy_close = spy_df["adj_close"].reindex(close.index).ffill().rename("SPY")
        except Exception as e:  # noqa: BLE001
            log.warning("SPY fetch failed (%s); benchmark-relative features disabled.", e)

    if cfg.use_tier2_features:
        log.info("Building tier-2 features (12-1 momentum, IVOL, beta, max ret, Amihud)...")
        t2 = compute_tier2_features(close, volume, bench_close=spy_close)
        if not t2.empty:
            feats = feats.join(t2, how="left")
            # Review H1 fix: also produce *_rank versions of Tier-2 columns so
            # `ranks_only=True` doesn't silently drop them all.
            from stockpred.features.cross_sectional import add_cross_sectional_ranks

            t2_cols = list(t2.columns)
            feats = add_cross_sectional_ranks(feats, cols=t2_cols)
            del t2  # free tier-2 intermediate (~200 MB for full S&P 500)
            log.info("After tier-2 (with ranks): %s rows x %s cols", *feats.shape)

    if cfg.use_regime_features:
        log.info("Building regime features (VIX, term spread, USD, xs dispersion)...")
        try:
            regime_wide = compute_regime_features(close, refresh=False)
            if not regime_wide.empty:
                reg_long = broadcast_to_panel(regime_wide, feats.index)
                feats = feats.join(reg_long, how="left")
                del reg_long, regime_wide  # free regime intermediates
                log.info("After regime: %s rows x %s cols", *feats.shape)
        except Exception as e:  # noqa: BLE001
            log.warning("Regime features failed (%s); continuing.", e)

    # Phase 12: SEC EDGAR 8-K event features. Joined into the master
    # panel as `edgar_has_8k`, `edgar_count_8k_5d`, etc. The prefix
    # `edgar_` is recognised by the ranks_only filter below so these
    # discrete event flags survive (they have no _rank counterpart by
    # design — event indicators don't need cross-sectional ranking).
    if cfg.use_edgar_features:
        log.info("Building EDGAR 8-K event features...")
        # Reviewer CRITICAL #2: narrow the except so MemoryError and
        # other "you really should hear about this" errors propagate.
        # MemoryError, KeyboardInterrupt are intentionally not caught.
        import requests as _requests

        try:
            from stockpred.data import edgar as edgar_mod

            edgar_panel = edgar_mod.build_8k_features(
                tickers=list(close.columns),
                trading_days=close.index,
                start=cfg.start_date,
                end=cfg.end_date,
                refresh=cfg.refresh_data,
            )
            if not edgar_panel.empty:
                # Prefix columns with `edgar_` so they survive ranks_only.
                edgar_panel = edgar_panel.add_prefix("edgar_")
                feats = feats.join(edgar_panel, how="left")
                # Fill missing (ticker not in SEC map) with 0 — i.e.
                # treat absent data as "no filings reported".
                edgar_cols = [c for c in feats.columns if c.startswith("edgar_")]
                for c in edgar_cols:
                    feats[c] = feats[c].fillna(0).astype("int16")
                log.info("After EDGAR: %s rows x %s cols", *feats.shape)
                # Memory hygiene: explicit del + gc per RAM constraint #8.
                del edgar_panel
                import gc as _gc

                _gc.collect()
            else:
                # When EDGAR is explicitly requested but yields empty,
                # this is suspicious enough to WARN (not just skip).
                log.warning(
                    "EDGAR features: empty panel returned. Check that "
                    "tickers have SEC CIK matches and the date range has filings."
                )
        except (
            _requests.RequestException,  # network / 403 / timeout
            OSError,  # disk / cache I/O
            ValueError,  # parser / dtype
            RuntimeError,  # explicit raise from edgar module (validation fail)
        ) as e:
            log.warning("EDGAR features failed (%s); continuing without.", e)

    # Phase 13: SEC EDGAR 8-K item-code features. Independent from
    # Phase 12; both can be enabled together. Uses SEC's per-company
    # submissions JSON endpoint (one request per ticker, not per
    # quarter -- much faster than Phase 12).
    if cfg.use_edgar_item_features:
        log.info("Building EDGAR 8-K item-code features...")
        try:
            from stockpred.data import edgar as edgar_mod

            item_panel = edgar_mod.build_8k_item_features(
                tickers=list(close.columns),
                trading_days=close.index,
                refresh=cfg.refresh_data,
            )
            if not item_panel.empty:
                # Columns already prefixed `edgaritem_` by the builder.
                feats = feats.join(item_panel, how="left")
                # Fill NaN -> 0 (treat absent as no filings reported).
                item_cols = [c for c in feats.columns if c.startswith("edgaritem_")]
                for c in item_cols:
                    feats[c] = feats[c].fillna(0).astype("int16")
                log.info("After EDGAR items: %s rows x %s cols", *feats.shape)
                del item_panel
                import gc as _gc

                _gc.collect()
            else:
                log.warning("EDGAR item-code features: empty panel returned.")
        except (
            _requests.RequestException,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            log.warning("EDGAR item-code features failed (%s); continuing.", e)

    # Phase 14: GDELT GKG daily tone + mention features. Reads ONLY
    # from per-day parquet caches; operator runs `scripts/phase14_
    # gdelt_bulk_fetch.py` overnight to populate them. If caches are
    # missing the join silently fills zeros + emits a coverage warning.
    if cfg.use_gdelt_features:
        log.info("Building GDELT GKG tone+mention features...")
        try:
            from stockpred.data import edgar as edgar_mod
            from stockpred.data import gdelt as gdelt_mod

            ticker_to_cik = edgar_mod.fetch_ticker_to_cik(refresh=cfg.refresh_data)
            gdelt_panel = gdelt_mod.build_gdelt_features(
                tickers=list(close.columns),
                trading_days=close.index,
                start=cfg.start_date,
                end=cfg.end_date,
                refresh=cfg.refresh_data,
                ticker_to_cik=ticker_to_cik,
            )
            if not gdelt_panel.empty:
                feats = feats.join(gdelt_panel, how="left")
                gdelt_cols = [c for c in feats.columns if c.startswith("gdelt_")]
                for c in gdelt_cols:
                    feats[c] = feats[c].fillna(0)
                log.info("After GDELT: %s rows x %s cols", *feats.shape)
                del gdelt_panel
                import gc as _gc

                _gc.collect()
            else:
                log.warning(
                    "GDELT features: empty panel returned. Did you run "
                    "scripts/phase14_gdelt_bulk_fetch.py to populate the cache?"
                )
        except (
            _requests.RequestException,
            OSError,
            ValueError,
            RuntimeError,
        ) as e:
            log.warning("GDELT features failed (%s); continuing without.", e)

    # Phase 8: optionally drop the raw (non-rank) numeric columns. Per-feature
    # audit on the medium universe showed raw versions degrade ~100% under
    # the hard-cutoff audit while their _rank versions degrade only ~15-50%;
    # keeping only ranks is a defensible noise-reduction move.
    if cfg.ranks_only:
        # Keep: anything ending in _rank, anything prefixed sec_ (sector
        # dummies), reg_ (regime broadcasts), edgar (Phase 12 has_8k /
        # count_8k OR Phase 13 edgaritem_), gdelt_ (Phase 14 GDELT).
        # Drop the rest.
        keep_cols = [
            c
            for c in feats.columns
            if c.endswith("_rank") or c.startswith(("sec_", "reg_", "edgar", "gdelt_"))
        ]
        if keep_cols:
            log.info(
                "ranks_only: dropping %d raw cols, keeping %d",
                feats.shape[1] - len(keep_cols),
                len(keep_cols),
            )
            feats = feats[keep_cols]
        else:
            log.warning("ranks_only: no rank/sec/reg/edgar columns found; keeping all")

    # Phase 11: explicit feature blocklist. Applied AFTER ranks_only so the
    # blocklist can target either raw or rank columns regardless of whether
    # ranks_only is set.
    if cfg.feature_exclude:
        present = [c for c in cfg.feature_exclude if c in feats.columns]
        missing = [c for c in cfg.feature_exclude if c not in feats.columns]
        if missing:
            log.warning(
                "feature_exclude: %d names not present in feats (ignored): %s",
                len(missing),
                missing,
            )
        if present:
            log.info("feature_exclude: dropping %d cols: %s", len(present), present)
            feats = feats.drop(columns=present)
            if feats.shape[1] == 0:
                raise RuntimeError(
                    "feature_exclude removed ALL feature columns; refusing to "
                    "train on an empty matrix."
                )

    log.info("Final feature matrix: %s rows x %s cols", *feats.shape)

    log.info("Building labels for horizons %s...", cfg.horizons)
    labels = long_labels(close, horizons=tuple(cfg.horizons), include_vol_scaled=True)

    # Phase 8: optional triple-barrier labels. We compute per-horizon and
    # join into `labels` under the name `tb_target_{h}` (signed return,
    # i.e. tb_label * |tb_return|, clipped to ±k_sigma window so the
    # magnitude is bounded by construction).
    if cfg.use_triple_barrier_labels:
        from stockpred.labels_triple_barrier import (
            TripleBarrierConfig,
            compute_triple_barrier_labels,
        )

        for h in cfg.horizons:
            tb_cfg = TripleBarrierConfig(
                max_horizon=h,
                k_up=cfg.tb_k_sigma,
                k_dn=cfg.tb_k_sigma,
                vol_window=21,
            )
            tb = compute_triple_barrier_labels(close, tb_cfg)
            if tb.empty:
                continue
            # Signed target: positive when upper barrier hit, negative when
            # lower hit; vertical-barrier rows take the realised path return
            # (already in tb_return). NaN preserved.
            target = tb["tb_return"].copy()
            target.name = f"tb_target_{h}"
            labels = labels.join(target, how="left")
        log.info("Triple-barrier targets added for horizons %s", cfg.horizons)

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
        # Phase 8: if triple-barrier labels are enabled, swap y_target for
        # the triple-barrier signed return. Realised return (y_return) and
        # binary direction (y_bin) stay on the original forward-return
        # convention so IC and hit-rate diagnostics remain comparable.
        if cfg.use_triple_barrier_labels:
            tb_col = f"tb_target_{h}"
            if tb_col in dev_labels.columns:
                tb_target = dev_labels[tb_col].reindex(X.index)
                # Drop rows where the TB target is NaN (insufficient warmup).
                keep = tb_target.notna()
                X = X[keep]
                y_target = tb_target[keep]
                y_return = y_return.reindex(X.index)
                y_bin = y_bin.reindex(X.index)
                log.info("h=%d using triple-barrier target; %d rows after NaN drop", h, len(X))
            else:
                log.warning("h=%d: requested triple-barrier but column %s not found", h, tb_col)
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

    # ---------------- Compute SPY-relative betas for neutralisation -----
    asset_betas: pd.DataFrame | None = None
    if cfg.beta_neutralise and spy_close is not None and not spy_close.dropna().empty:
        try:
            from stockpred.features.tier2 import beta_vs_bench

            asset_betas = beta_vs_bench(close, spy_close, window=60)
            log.info("Computed asset betas vs SPY for neutralisation: %s", asset_betas.shape)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not compute asset betas (%s); skipping beta-neutralise.", e)

    # ---------------- Build DEV ensemble score (pre-gating) ----------
    if cfg.ensemble_weighting == "ic_ir":
        ic_ir_d = {
            h: float(per_horizon_diag.get(h, {}).get("ic_ir", 0.0)) for h in per_horizon_preds
        }
        dev_ensemble_score = ic_ir_weighted_ensemble(per_horizon_preds, ic_ir_d)
    else:
        from stockpred.pipeline import ensemble_predictions

        dev_ensemble_score = ensemble_predictions(per_horizon_preds)

    # ---------------- Phase 19: Bayesian per-ticker shrinkage ----------
    # Drop tickers with worse-than-random historical sign-precision;
    # downweight tickers proportional to their (precision - 0.5).
    # Fit shrinkage on the FIRST 80% of dev, apply to the FULL dev
    # ensemble. This is leakage-safe because we never look at holdout
    # to choose the factors.
    _shrink_factors_for_holdout = None
    if cfg.bayesian_shrinkage_alpha > 0.0:
        try:
            from stockpred.portfolio.bayesian_shrinkage import (
                compute_per_ticker_sign_precision,
                compute_shrinkage_factors,
                apply_shrinkage_to_panel,
            )

            dev_dates_sorted = (
                dev_ensemble_score.index.get_level_values("date").unique().sort_values()
            )
            n_split = max(1, int(0.8 * len(dev_dates_sorted)))
            fit_dates = dev_dates_sorted[:n_split]
            fit_mask = dev_ensemble_score.index.get_level_values("date").isin(fit_dates)
            shortest_h = min(per_horizon_preds.keys())
            fit_realised = per_horizon_returns[shortest_h].reindex(dev_ensemble_score.index)
            sp = compute_per_ticker_sign_precision(
                dev_ensemble_score[fit_mask],
                fit_realised[fit_mask],
                min_obs=cfg.bayesian_shrinkage_min_obs,
            )
            sf = compute_shrinkage_factors(sp, alpha=cfg.bayesian_shrinkage_alpha)
            dev_ensemble_score = apply_shrinkage_to_panel(dev_ensemble_score, sf)
            _shrink_factors_for_holdout = sf  # reuse on holdout below
            log.info(
                "Phase 19 shrinkage (alpha=%.2f): n_tickers=%d, n_active=%d, n_dropped=%d",
                cfg.bayesian_shrinkage_alpha,
                len(sf),
                int((sf > 0).sum()),
                int((sf == 0).sum()),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("Phase 19 shrinkage failed (%s); continuing with raw scores.", e)

    # ---------------- Phase 8: optional meta-labelling gate -----------
    # Train a binary GBM predicting P(primary signal is correct) on the
    # first 80% of dev, evaluate on last 20% + use that fit to gate the
    # dev portfolio. Holdout gets its own fit using all of dev.
    # Review C2 fix: keep an UNGATED copy of the dev score for the holdout
    # meta-fit; otherwise the holdout classifier trains on data where
    # gated-out rows have primary=0 (which always look "incorrect" because
    # sign(0) != sign(realised)), biasing the meta toward "predict wrong".
    dev_ensemble_score_ungated = dev_ensemble_score.copy()
    if cfg.use_meta_labelling:
        meta_kwargs = dict(
            bt_horizon_for_meta=min(per_horizon_preds.keys()),
            threshold=cfg.meta_threshold,
            gbm_cfg=cfg.gbm,
            log=log,
            mode=cfg.meta_mode,
            conf_floor=cfg.meta_conf_floor,
            conf_cap=cfg.meta_conf_cap,
            walk_forward_folds=cfg.meta_walk_forward_folds,
        )
        if cfg.meta_per_sector:
            dev_ensemble_score = _apply_meta_gate_per_sector(
                dev_ensemble_score,
                dev_feats,
                dev_labels,
                sector_map,
                **meta_kwargs,
            )
        else:
            dev_ensemble_score = _apply_meta_gate(
                dev_ensemble_score,
                dev_feats,
                dev_labels,
                **meta_kwargs,
            )

    # ---------------- Portfolio + backtest on the DEV span -----------
    # Review C1 fix: when we have a gated score, pass it directly via
    # precomputed_score so _build_weights doesn't re-z-score the zeros.
    dev_weights = _build_weights(
        cfg,
        per_horizon_preds,
        per_horizon_diag,
        close,
        sector_map,
        asset_betas=asset_betas,
        precomputed_score=dev_ensemble_score if cfg.use_meta_labelling else None,
    )
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
            # positional. We sort dev by date and split at a *date* boundary
            # so train and valid never share a date across tickers.
            X_dev_sorted = X_dev.sort_index(level="date")
            y_dev_sorted = y_dev.reindex(X_dev_sorted.index)
            all_dates = X_dev_sorted.index.get_level_values("date").unique().sort_values()
            split_date = all_dates[max(1, int(len(all_dates) * 0.9))]
            tr_mask = X_dev_sorted.index.get_level_values("date") < split_date
            va_mask = ~tr_mask
            if not va_mask.any():
                # No room for a validation split; train on everything.
                booster = train_gbm(X_dev_sorted, y_dev_sorted, cfg=cfg.gbm)
            else:
                tr_max = X_dev_sorted.loc[tr_mask].index.get_level_values("date").max()
                va_min = X_dev_sorted.loc[va_mask].index.get_level_values("date").min()
                assert tr_max < va_min, (
                    f"Internal valid split overlaps: train.max={tr_max} >= valid.min={va_min}"
                )
                booster = train_gbm(
                    X_dev_sorted.loc[tr_mask],
                    y_dev_sorted.loc[tr_mask],
                    X_valid=X_dev_sorted.loc[va_mask],
                    y_valid=y_dev_sorted.loc[va_mask],
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
        # Phase 8: holdout meta-gating. Train meta on ALL of dev (using
        # the dev ensemble score we already built), then apply to the
        # holdout ensemble score before weight construction.
        if cfg.use_meta_labelling:
            # Build holdout ensemble score first.
            if cfg.ensemble_weighting == "ic_ir":
                ic_ir_h = {
                    h: float(per_horizon_diag.get(h, {}).get("ic_ir", 0.0)) for h in hold_preds
                }
                hold_ensemble_pre = ic_ir_weighted_ensemble(hold_preds, ic_ir_h)
            else:
                from stockpred.pipeline import ensemble_predictions

                hold_ensemble_pre = ensemble_predictions(hold_preds)

            # Phase 19: apply shrinkage factors fit on DEV to the holdout
            # ensemble. Leakage-safe: factors were computed from the dev
            # window (first 80% of dev) and never see holdout.
            if cfg.bayesian_shrinkage_alpha > 0.0 and _shrink_factors_for_holdout is not None:
                try:
                    from stockpred.portfolio.bayesian_shrinkage import (
                        apply_shrinkage_to_panel,
                    )

                    hold_ensemble_pre = apply_shrinkage_to_panel(
                        hold_ensemble_pre, _shrink_factors_for_holdout
                    )
                    log.info("Phase 19: applied dev-fit shrinkage factors to holdout.")
                except Exception as e:  # noqa: BLE001
                    log.warning("Phase 19 holdout shrinkage failed (%s); using raw scores.", e)

            from stockpred.models.meta import (
                build_meta_dataset,
                fit_meta,
                meta_confidence_weight_signal,
                meta_filter_signal,
                predict_meta,
            )

            realised_col = f"fwd_return_{bt_horizon}"
            if realised_col in dev_labels.columns and realised_col in hold_labels.columns:
                forbidden_prefixes = ("fwd_", "tb_")
                safe_dev = dev_feats[
                    [c for c in dev_feats.columns if not c.startswith(forbidden_prefixes)]
                ]
                safe_hold = hold_feats[
                    [c for c in hold_feats.columns if not c.startswith(forbidden_prefixes)]
                ]
                try:
                    # Review C2 fix: train the holdout meta on the UNGATED
                    # dev score (not the gated one, which has zeros that
                    # always look "wrong" to the binary classifier).
                    X_dev_m, y_dev_m = build_meta_dataset(
                        dev_ensemble_score_ungated,
                        dev_labels[realised_col].reindex(dev_ensemble_score_ungated.index),
                        safe_dev.loc[dev_ensemble_score_ungated.index],
                        use_primary_score=True,
                    )
                    if y_dev_m.nunique() >= 2:
                        booster_h = fit_meta(X_dev_m, y_dev_m, cfg=cfg.gbm)
                        X_hold_m = safe_hold.reindex(hold_ensemble_pre.index).copy()
                        X_hold_m["primary_abs"] = hold_ensemble_pre.abs()
                        proba_h = predict_meta(booster_h, X_hold_m)
                        # Phase 9a: confidence vs binary mode
                        if cfg.meta_mode == "confidence":
                            gated_hold = meta_confidence_weight_signal(
                                hold_ensemble_pre,
                                proba_h,
                                floor=cfg.meta_conf_floor,
                                cap=cfg.meta_conf_cap,
                            )
                        else:
                            gated_hold = meta_filter_signal(
                                hold_ensemble_pre,
                                proba_h,
                                p_threshold=cfg.meta_threshold,
                            )
                        log.info(
                            "HOLDOUT meta-gate(mode=%s): %.1f%% of obs survived gate",
                            cfg.meta_mode,
                            100 * (gated_hold != 0).mean(),
                        )
                        if cfg.meta_per_sector:
                            log.warning(
                                "meta-per-sector is dev-only: HOLDOUT meta uses "
                                "a single global classifier trained on full dev. "
                                "DEV/HOLDOUT metrics are not directly comparable "
                                "under this flag."
                            )
                        # Store the gated score for downstream snapshot
                        # writers; we pass it directly to _build_weights via
                        # precomputed_score below (Review C1 fix).
                        hold_gated_score = gated_hold
                    else:
                        hold_gated_score = None
                except Exception as e:  # noqa: BLE001
                    log.warning("HOLDOUT meta-gate failed (%s); ungated", e)
                    hold_gated_score = None
            else:
                hold_gated_score = None
        else:
            hold_gated_score = None

        hold_weights = _build_weights(
            cfg,
            hold_preds,
            per_horizon_diag,
            close,
            sector_map,
            asset_betas=asset_betas,
            precomputed_score=hold_gated_score,
        )
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
        # Block bootstrap by default (review H1). Block length ~ horizon so we
        # preserve the short-range autocorrelation overlapping-horizon strategies
        # induce in daily returns.
        block_len = bt_horizon if cfg.bootstrap_method == "block" else None
        ci = bootstrap_sharpe(
            hold_bt.returns,
            n_resamples=cfg.bootstrap_n,
            method=cfg.bootstrap_method,
            block_length=block_len,
        )
        log.info(
            "HOLDOUT bootstrap Sharpe (%s, block=%s): %.3f  [%.3f, %.3f] @ %.0f%%",
            ci.get("method"),
            ci.get("block_length"),
            ci["sharpe"],
            ci["sharpe_lo"],
            ci["sharpe_hi"],
            float(ci["ci_pct"]) * 100,
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
    # User direction 2026-06-04: log peak RSS so 8 GB RAM budget can be
    # verified by smoke tests. psutil is a transitive dep (yfinance).
    # Logged at WARNING level (always visible) because RSS is an SLO.
    try:
        import psutil

        rss_gb = psutil.Process().memory_info().rss / 1024**3
        log.warning("Phase 5 complete in %.1fs (peak RSS: %.2f GB)", elapsed, rss_gb)
        if rss_gb > 6.0:
            log.warning(
                "Peak RSS %.2f GB exceeds 6 GB budget (8 GB box, 2 GB headroom).",
                rss_gb,
            )
    except Exception:  # noqa: BLE001
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
