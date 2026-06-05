"""Pydantic response models for the API. Designed to match the frontend's needs."""

from __future__ import annotations

import datetime as dt
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# Strict ticker pattern. Allows:
#   - Optional leading `^` (indices like ^VIX)
#   - 1–16 chars from [A-Z0-9.-=]
#   - Optional `.SUFFIX` for international (e.g. HND.TO, BABA.HK)
# Disallows `/`, `..`, spaces, NUL, query strings, etc.
_TICKER_RE = re.compile(r"^\^?[A-Z][A-Z0-9.\-=]{0,15}$")


def _validate_ticker(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("ticker must be a string")
    v = value.strip().upper()
    if not _TICKER_RE.fullmatch(v):
        raise ValueError(f"invalid ticker: {value!r}")
    return v


class TickerSummary(BaseModel):
    ticker: str
    sector: str | None = None
    industry: str | None = None
    last_price: float | None = None
    market_cap: float | None = None
    last_updated: dt.date | None = None


class PriceBarOut(BaseModel):
    date: dt.date
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    adj_close: float | None
    volume: float | None


class PredictionOut(BaseModel):
    date: dt.date
    ticker: str
    score: float
    rank: int | None
    side: str | None
    weight: float | None
    per_horizon: dict[str, float | None]


class TickerDetail(BaseModel):
    ticker: str
    sector: str | None
    industry: str | None
    market_cap: float | None
    beta: float | None
    trailing_pe: float | None
    forward_pe: float | None
    dividend_yield: float | None
    short_ratio: float | None
    short_percent_of_float: float | None
    fifty_two_week_high: float | None
    fifty_two_week_low: float | None
    long_business_summary: str | None
    prices: list[PriceBarOut]
    predictions: list[PredictionOut]


class TopMovers(BaseModel):
    date: dt.date | None
    long: list[PredictionOut]
    short: list[PredictionOut]


class EquityPoint(BaseModel):
    date: dt.date
    daily_return: float | None
    cumulative_return: float | None
    drawdown: float | None
    turnover: float | None
    benchmark_return: float | None = None


class RunSummary(BaseModel):
    id: int
    started_at: dt.datetime
    completed_at: dt.datetime | None
    status: str
    metrics: dict
    per_horizon_diagnostics: dict
    tickers_count: int
    note: str | None


class BacktestSummary(BaseModel):
    run: RunSummary
    equity_curve: list[EquityPoint]


class CVParams(BaseModel):
    train_years: int = Field(default=3, ge=1)
    test_months: int = Field(default=6, ge=1)
    embargo_days: int = Field(default=25, ge=0)
    min_train_obs: int = Field(default=1000, ge=1)


class GBMParams(BaseModel):
    num_leaves: int = Field(default=63, ge=2)
    learning_rate: float = Field(default=0.03, gt=0)
    n_estimators: int = Field(default=800, ge=1)
    min_data_in_leaf: int = Field(default=200, ge=1)
    feature_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_fraction: float = Field(default=0.8, gt=0, le=1)
    bagging_freq: int = Field(default=5, ge=0)
    reg_lambda: float = Field(default=1.0, ge=0)
    early_stopping_rounds: int | None = Field(default=50)


class RefreshRequest(BaseModel):
    """Body for POST /jobs/refresh. All fields are optional; defaults mirror the pipeline configs."""

    phase: Literal[1, 5] = Field(
        default=5,
        description="Which pipeline to run. 1 = Phase 1 (basic GBM), 5 = Phase 5 (vol-scaled, regime-aware).",
    )

    # --- Universe / history ---
    start_date: str = Field(default="2013-01-01", description="ISO date, e.g. '2015-01-01'")
    end_date: str | None = Field(default=None, description="ISO date; None = today")
    n_tickers: int | None = Field(default=None, ge=1, description="Universe size; None = all")
    universe_sampling: Literal["random", "current", "first"] = Field(
        default="current",
        description="How tickers are sampled from S&P 500 membership history.",
    )
    refresh_data: bool = Field(default=False, description="Force-refetch cached price/fundamental data")

    # --- Horizons + model ---
    horizons: list[int] | None = Field(
        default=[5],
        description=(
            "Forecast horizons in trading days. "
            "Defaults to [1, 5, 21] for phase 1 and [1, 5] for phase 5 "
            "(21d showed no signal in Phase 2 evaluation)."
        ),
    )
    model: Literal["gbm", "logistic"] = Field(default="gbm")
    use_sector_features: bool = Field(default=False)

    # --- CV ---
    cv: CVParams = Field(default_factory=CVParams)

    # --- GBM hyper-params (ignored when model='logistic') ---
    gbm: GBMParams = Field(default_factory=GBMParams)

    # --- Phase 1 only ---
    k_per_side: int = Field(
        default=20, ge=1, description="[Phase 1] Number of longs and shorts in portfolio."
    )
    feature_cols: list[str] | None = Field(
        default=None, description="[Phase 1] Explicit feature list; None = use all."
    )

    # --- Phase 5 only ---
    use_tier2_features: bool = Field(
        default=False,
        description="[Phase 5] Include 12-1 momentum, IVOL, beta, max-ret, Amihud features.",
    )
    use_regime_features: bool = Field(
        default=False,
        description="[Phase 5] Include VIX, term spread, USD, cross-sectional dispersion features.",
    )
    beta_neutralise: bool = Field(
        default=False, description="[Phase 5] Apply portfolio-level beta-vs-SPY neutralisation."
    )
    bootstrap_method: Literal["block", "iid"] = Field(
        default="block", description="[Phase 5] Stress-test bootstrap method."
    )
    holdout_years: int = Field(
        default=2, ge=0, description="[Phase 5] Years held out from CV / model selection."
    )
    position_sizing: Literal["vol_scaled", "top_k", "hrp"] = Field(
        default="vol_scaled",
        description="[Phase 5] Portfolio construction method. hrp = Hierarchical Risk Parity (Phase 7).",
    )
    k_per_side_pct: float = Field(
        default=0.15,
        gt=0,
        le=1,
        description="[Phase 5] Fraction of universe selected per side (vol_scaled mode).",
    )
    leverage_per_side: float = Field(default=1.0, gt=0, description="[Phase 5] Gross leverage per side.")
    sector_cap_gross: float | None = Field(
        default=0.30,
        description="[Phase 5] Max gross exposure per GICS sector; None = uncapped.",
    )
    min_trade_threshold: float = Field(
        default=0.005, ge=0, description="[Phase 5] Ignore weight changes smaller than this."
    )
    ensemble_weighting: Literal["ic_ir", "equal"] = Field(
        default="equal",
        description="[Phase 5] How to weight horizons in the ensemble score.",
    )
    bootstrap_n: int = Field(
        default=500, ge=1, description="[Phase 5] Number of bootstrap samples for stress test."
    )

    # --- Phase 8: meta-labelling (López de Prado Ch. 3.6) ---
    use_meta_labelling: bool = Field(
        default=False,
        description="[Phase 8] Train a secondary classifier to gate signals by P(primary score is correct).",
    )
    meta_threshold: float = Field(
        default=0.55, ge=0, le=1,
        description="[Phase 8] Min P(correct) required to pass the meta-gate.",
    )
    meta_mode: Literal["binary", "confidence"] = Field(
        default="binary",
        description="[Phase 9] 'binary' = hard gate; 'confidence' = scale signal by P.",
    )
    meta_conf_floor: float = Field(
        default=0.5, ge=0, le=1,
        description="[Phase 9] Lower bound for confidence scaling (P below this → weight 0).",
    )
    meta_conf_cap: float = Field(
        default=1.0, ge=0, le=1,
        description="[Phase 9] Upper bound for confidence scaling (P above this → full weight).",
    )
    meta_walk_forward_folds: int = Field(
        default=1, ge=1,
        description="[Phase 9] CV folds for meta classifier. 1 = single-pass (faster); >1 = proper walk-forward.",
    )
    meta_per_sector: bool = Field(
        default=False,
        description="[Phase 9] Train one meta classifier per GICS sector instead of globally.",
    )

    # --- Phase 8: triple-barrier labels ---
    use_triple_barrier_labels: bool = Field(
        default=False,
        description="[Phase 8] Use triple-barrier labels instead of forward returns as the regression target.",
    )
    tb_k_sigma: float = Field(
        default=2.0, gt=0,
        description="[Phase 8] Barrier width in trailing-volatility units.",
    )

    # --- Phase 8: feature transformations ---
    ranks_only: bool = Field(
        default=False,
        description="[Phase 8] Drop raw feature values; keep only cross-sectional rank columns + EDGAR/regime.",
    )

    # --- Phase 11: feature pruning ---
    feature_exclude: list[str] = Field(
        default=[],
        description="[Phase 11] Feature names to exclude from training (e.g. low-IC features from audit).",
    )

    # --- Phase 12: SEC EDGAR 8-K event features ---
    use_edgar_features: bool = Field(
        default=False,
        description="[Phase 12] Add SEC 8-K filing event counts (has_8k, count_5d/21d/63d) as features.",
    )

    # --- Phase 13: SEC EDGAR 8-K per-item features ---
    use_edgar_item_features: bool = Field(
        default=False,
        description="[Phase 13] Add per-item-code 8-K features (CEO changes, earnings releases, etc.).",
    )


class JobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class JobDetail(BaseModel):
    """Full detail for a single in-flight or completed job (GET /jobs/{job_id})."""

    job_id: str
    status: str
    job_type: str = "pipeline"  # "pipeline" | "hypersearch"
    started_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    config: dict = {}
    logs: list[str] = []
    run_id: int | None = None
    elapsed_s: float | None = None
    error: str | None = None


# ─── Hyperparameter search ────────────────────────────────────────────────────


class HypersearchRequest(BaseModel):
    """Body for POST /jobs/queue when job_type = 'hypersearch'."""

    n_trials: int = Field(default=50, ge=1, le=500, description="Number of Optuna trials")
    n_tickers: int = Field(default=25, ge=5, le=500, description="Universe size per trial")
    start_date: str = Field(default="2015-01-01", description="History start date")
    end_date: str | None = Field(default=None, description="History end date; None = today")
    holdout_years: int = Field(default=2, ge=1, le=5, description="Years withheld from tuning")
    bootstrap_n: int = Field(
        default=50, ge=10, le=500,
        description="Bootstrap resamples for Sharpe CI (50=fast, 500=honest)",
    )
    universe_sampling: Literal["current", "first", "random"] = Field(
        default="current",
        description="Ticker selection strategy (current = same set every trial)",
    )
    seed: int = Field(default=42, description="Optuna sampler seed for reproducibility")


class HypersearchTrialOut(BaseModel):
    """One trial result row returned by GET /hypersearch/runs/{id}."""

    trial: int
    value: float | None = None
    hold_sharpe: float | None = None
    hold_ci_lo: float | None = None
    hold_ci_hi: float | None = None
    hold_dd: float | None = None
    hold_hit: float | None = None
    hold_ann_return: float | None = None
    dev_sharpe: float | None = None
    elapsed_s: float | None = None
    error: str | None = None
    params: dict = {}


class HypersearchRunOut(BaseModel):
    """GET /hypersearch/runs or GET /hypersearch/runs/{id}."""

    id: int
    job_id: str | None = None
    started_at: dt.datetime
    completed_at: dt.datetime | None = None
    status: str
    config: dict = {}
    n_trials_requested: int
    n_trials_done: int
    best_sharpe: float | None = None
    best_params: dict | None = None
    trials: list[HypersearchTrialOut] = []


class QueuedJobOut(BaseModel):
    """A pending job queued via POST /jobs/queue, awaiting password-protected launch."""

    id: str
    created_at: dt.datetime
    config: dict
    label: str | None = None
    status: str  # pending | launched | cancelled
    launched_at: dt.datetime | None = None
    job_id: str | None = None  # populated once launched


class HealthResponse(BaseModel):
    status: str = "ok"
    db: str
    scheduler: str


class QuoteOut(BaseModel):
    """Latest (delayed) quote from yfinance fast_info. ~15 min delayed; only
    moves during market hours. Fields are None when unavailable."""

    ticker: str
    price: float | None = None
    previous_close: float | None = None
    open: float | None = None
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    market_cap: float | None = None
    change: float | None = None  # price - previous_close
    change_pct: float | None = None  # change / previous_close
    as_of: dt.datetime  # server time the quote was fetched
    delayed: bool = True


class WatchedItem(BaseModel):
    ticker: str
    label: str | None = None
    category: str | None = None
    note: str | None = None
    last_price: float | None = None
    last_updated: dt.date | None = None


class WatchedAdd(BaseModel):
    ticker: str = Field(..., description="Strict pattern; see _validate_ticker")
    label: str | None = Field(default=None, max_length=128)
    category: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=512)

    @field_validator("ticker")
    @classmethod
    def _check_ticker(cls, v: str) -> str:
        return _validate_ticker(v)


class NewsHeadline(BaseModel):
    uuid: str
    title: str | None
    publisher: str | None
    link: str | None
    type: str | None
    published_at: dt.datetime | None
