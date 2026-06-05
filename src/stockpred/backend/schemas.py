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
        default=1,
        description="Which pipeline to run. 1 = Phase 1 (basic GBM), 5 = Phase 5 (vol-scaled, regime-aware).",
    )

    # --- Universe / history ---
    start_date: str = Field(default="2010-01-01", description="ISO date, e.g. '2015-01-01'")
    end_date: str | None = Field(default=None, description="ISO date; None = today")
    n_tickers: int | None = Field(default=100, ge=1, description="Universe size; None = all")
    universe_sampling: Literal["random", "current", "first"] = Field(
        default="random",
        description="How tickers are sampled from S&P 500 membership history.",
    )
    refresh_data: bool = Field(
        default=False, description="Force-refetch cached price/fundamental data"
    )

    # --- Horizons + model ---
    horizons: list[int] | None = Field(
        default=None,
        description=(
            "Forecast horizons in trading days. "
            "Defaults to [1, 5, 21] for phase 1 and [1, 5] for phase 5 "
            "(21d showed no signal in Phase 2 evaluation)."
        ),
    )
    model: Literal["gbm", "logistic"] = Field(default="gbm")
    use_sector_features: bool = Field(default=True)

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
        default=True,
        description="[Phase 5] Include 12-1 momentum, IVOL, beta, max-ret, Amihud features.",
    )
    use_regime_features: bool = Field(
        default=True,
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
        description=(
            "[Phase 5] Portfolio construction method. "
            "'hrp' (Phase 7) = Hierarchical Risk Parity per-cluster "
            "inverse-variance allocation; was the best non-meta option on the "
            "production sweep."
        ),
    )
    k_per_side_pct: float = Field(
        default=0.15,
        gt=0,
        le=1,
        description="[Phase 5] Fraction of universe selected per side (vol_scaled mode).",
    )
    leverage_per_side: float = Field(
        default=1.0, gt=0, description="[Phase 5] Gross leverage per side."
    )
    sector_cap_gross: float | None = Field(
        default=0.30,
        description="[Phase 5] Max gross exposure per GICS sector; None = uncapped.",
    )
    min_trade_threshold: float = Field(
        default=0.005, ge=0, description="[Phase 5] Ignore weight changes smaller than this."
    )
    ensemble_weighting: Literal["ic_ir", "equal"] = Field(
        default="ic_ir",
        description="[Phase 5] How to weight horizons in the ensemble score.",
    )
    bootstrap_n: int = Field(
        default=500, ge=1, description="[Phase 5] Number of bootstrap samples for stress test."
    )

    # --- Phase 8: meta-labelling + ranks-only -----------------------------
    use_meta_labelling: bool = Field(
        default=False,
        description=(
            "[Phase 8] Train a binary meta-classifier per fold predicting "
            "P(primary score has correct sign). Gates the primary score to "
            "reduce turnover and improve precision."
        ),
    )
    meta_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="[Phase 8] Gate threshold for meta-labelling (binary mode).",
    )
    ranks_only: bool = Field(
        default=False,
        description=(
            "[Phase 8] Drop raw feature columns, keep only cross-sectional "
            "rank columns (plus sec_/reg_/edgar_ prefixes). Per-feature audit "
            "showed raw versions degrade ~100% under hard-cutoff vs ~15-50% "
            "for the ranked versions."
        ),
    )

    # --- Phase 9: confidence-weighted sizing + walk-forward meta-CV ------
    meta_mode: Literal["binary", "confidence"] = Field(
        default="binary",
        description=(
            "[Phase 9] 'binary' = hard meta-gate at meta_threshold; "
            "'confidence' = scale signal by clip((P-floor)/(cap-floor), 0, 1). "
            "Phase 10 sweep: floor 0.60 has the best point estimate; "
            "DO NOT use the default 0.50 (reproducibly worse than binary)."
        ),
    )
    meta_conf_floor: float = Field(
        default=0.60,
        ge=0.0,
        lt=1.0,
        description=(
            "[Phase 9] Lower bound for confidence-weighted sizing. "
            "0.60 is the Phase 10 sweet spot; 0.50 was the Phase 9 default "
            "that reproducibly hurt the strategy."
        ),
    )
    meta_conf_cap: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="[Phase 9] Upper bound for confidence-weighted sizing.",
    )
    meta_walk_forward_folds: int = Field(
        default=1,
        ge=1,
        description=(
            "[Phase 9] K-fold expanding-window meta-classifier CV. "
            "1 = single-pass Phase 8 behaviour."
        ),
    )
    meta_per_sector: bool = Field(
        default=False,
        description=(
            "[Phase 9] One meta classifier per sector (requires fundamentals "
            "loaded; falls back to global meta if not)."
        ),
    )

    # --- Phase 7 triple-barrier labels -----------------------------------
    use_triple_barrier_labels: bool = Field(
        default=False,
        description=(
            "[Phase 7] Switch from simple forward-return label to López de "
            "Prado triple-barrier signed return per horizon."
        ),
    )

    # --- Phase 11 feature pruning ----------------------------------------
    feature_exclude: list[str] = Field(
        default_factory=list,
        description=(
            "[Phase 11] Explicit feature-name blocklist applied after ranks_only. "
            "On the 150-name x 11yr universe, dropping these 5 was the best: "
            "['adv_proxy_21', 'dist_low_252_rank', 'ret_252d_rank', 'kurt_63', "
            "'dist_low_252']."
        ),
    )

    # --- Phase 12 EDGAR 8-K event flags + counts --------------------------
    use_edgar_features: bool = Field(
        default=False,
        description=(
            "[Phase 12] Enable SEC EDGAR 8-K event features: has_8k + "
            "rolling 5d/21d/63d counts. Free, no API key, respects SEC's 10 "
            "req/sec + User-Agent rule (set EDGAR_USER_AGENT env var). "
            "WARNING: Phase 12 production smoke showed this HURT the strategy "
            "(Sharpe -0.16 -> -0.38). Prefer use_edgar_item_features (Phase 13)."
        ),
    )

    # --- Phase 13 EDGAR 8-K item codes -----------------------------------
    use_edgar_item_features: bool = Field(
        default=False,
        description=(
            "[Phase 13] Enable SEC EDGAR 8-K item-code features (earnings, "
            "CEO change, M&A, guidance, going-concern). Best honest result "
            "across all 13 phases: Sharpe +0.17 with CI [-0.32, +0.58] and "
            "smallest holdout DD (-8.2%)."
        ),
    )

    # --- Phase 14 GDELT GKG tone + mentions ------------------------------
    use_gdelt_features: bool = Field(
        default=False,
        description=(
            "[Phase 14] Enable GDELT GKG daily tone + mention features. "
            "Reads ONLY from per-day parquet caches; operator must populate "
            "them first via `scripts/phase14_gdelt_bulk_fetch.py` overnight. "
            "If caches are missing, emits a coverage warning + fills zeros."
        ),
    )


class JobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class JobDetail(BaseModel):
    """Full detail for a single in-flight or completed job (GET /jobs/{job_id})."""

    job_id: str
    status: str
    started_at: dt.datetime | None = None
    updated_at: dt.datetime | None = None
    config: dict = {}
    logs: list[str] = []
    run_id: int | None = None
    elapsed_s: float | None = None
    error: str | None = None


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
