"""FastAPI application.

Routes:
  GET  /healthz                     — DB + scheduler health
  GET  /tickers                     — list all tickers with summary
  GET  /tickers/{ticker}/details    — fundamentals + extended history (?run_id= optional)
  GET  /tickers/{ticker}/news       — headlines
  GET  /predictions/latest          — top-k long / bottom-k short (?run_id= optional)
  GET  /runs                        — recent runs (metadata + config + job_id + is_active)
  GET  /runs/{run_id}               — single run summary
  GET  /runs/{run_id}/equity        — backtest equity curve for that run
  GET  /runs/{run_id}/backtest      — full BacktestSummary for arbitrary run
  POST /runs/{run_id}/activate      — pin a run as the active data source (X-Password)
  POST /runs/deactivate             — clear the active-run pin (X-Password)
  GET  /backtest/summary            — backtest tearsheet (?run_id= optional; default = active/latest)

  POST /jobs/refresh                — trigger run (requires X-API-Key)
  GET  /jobs                        — list recent jobs (in-memory)
  GET  /jobs/{job_id}               — job detail + logs

  POST /jobs/queue                  — queue a pipeline job (no auth, max 5)
  GET  /jobs/queue                  — list queued jobs
  POST /jobs/run/{queue_id}         — launch a queued job (requires X-Password)
  DELETE /jobs/queue/{queue_id}     — delete a queued job (requires X-Password)
  DELETE /jobs/{job_id}/cancel      — cancel a running job (requires X-Password)

  POST /hypersearch/queue           — queue a hypersearch job (no auth, max 5)
  GET  /hypersearch/runs            — list all hypersearch run results
  GET  /hypersearch/runs/{run_id}   — detail for one hypersearch run (trials, best params)
  GET  /hypersearch/runs/by-job/{job_id} — hypersearch run linked to a specific job

  GET  /watchlist                   — watched tickers
  POST /watchlist                   — add ticker (requires X-API-Key)
  DELETE /watchlist/{ticker}        — remove ticker (requires X-API-Key)
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import threading
import uuid as _uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from sqlalchemy import func, select
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from stockpred.backend import jobs as jobs_mod
from stockpred.backend import schemas, store
from stockpred.backend.db import create_all, make_engine, make_session_factory, session_scope
from stockpred.pipeline import PipelineConfig
from stockpred.pipeline_v5 import PipelineV5Config
from stockpred.config import CVConfig
from stockpred.models.gbm import GBMConfig

log = logging.getLogger(__name__)

# ----- App state ---------------------------------------------------------

DB_PATH = os.environ.get("STOCKPRED_DB", None)
WEB_DIST = os.environ.get(
    "STOCKPRED_WEB_DIST",
    str((Path(__file__).resolve().parent.parent.parent.parent / "web" / "dist").resolve()),
)

WRITE_API_KEY = os.environ.get("STOCKPRED_API_KEY")
WRITE_PW = os.environ.get("STOCKPRED_PW")

_refresh_lock = threading.Lock()

# Server-side quote cache: {ticker: (fetched_at, payload)}. TTL throttles
# outbound yfinance calls so frontend polling can't rate-limit our IP.
_QUOTE_TTL_S = 8.0
_quote_cache: dict[str, tuple[float, dict]] = {}
_quote_lock = threading.Lock()


def _is_pipeline_running() -> bool:
    return _refresh_lock.locked()


class AppState:
    engine = None
    SessionLocal = None
    scheduler = None


# ----- JSON response: NaN/Inf -> null ------------------------------------


def _sanitize(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


class SafeJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        import json

        return json.dumps(
            _sanitize(content),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    AppState.engine = make_engine(DB_PATH)
    create_all(AppState.engine)
    AppState.SessionLocal = make_session_factory(AppState.engine)
    try:
        with session_scope(AppState.SessionLocal) as s:
            store.seed_default_watchlist(s)
    except Exception as e:  # noqa: BLE001
        log.warning("watchlist seed failed: %s", e)
    try:
        with session_scope(AppState.SessionLocal) as s:
            crashed = store.mark_stale_jobs_crashed(s)
            if crashed:
                log.info("startup: marked %d stale job(s) as crashed", crashed)
    except Exception as e:  # noqa: BLE001
        log.warning("stale job cleanup failed: %s", e)
    AppState.scheduler = jobs_mod.make_scheduler(AppState.SessionLocal)
    if os.environ.get("STOCKPRED_DISABLE_SCHEDULER") != "1":
        AppState.scheduler.start()
        log.info("scheduler started")
    yield
    if AppState.scheduler and AppState.scheduler.running:
        AppState.scheduler.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(
        title="stock-predictor",
        description=(
            "Cross-sectional directional forecaster for S&P 500 equities. "
            "Free data, honest backtests, fully portable."
        ),
        version="0.2.0",
        lifespan=lifespan,
        default_response_class=SafeJSONResponse,
    )

    raw = os.environ.get("STOCKPRED_CORS", "http://localhost:5173,http://127.0.0.1:8000")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    wildcard = origins == ["*"]
    if wildcard:
        log.warning("STOCKPRED_CORS='*' — only safe for local development with no auth.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=not wildcard,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Password"],
    )

    register_routes(app)
    register_static(app)
    return app


# ----- Auth dependencies -------------------------------------------------


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Guard original write endpoints (POST /jobs/refresh, watchlist)."""
    if WRITE_API_KEY is None:
        raise HTTPException(
            403,
            "Write endpoints are disabled. Set STOCKPRED_API_KEY env var to enable.",
        )
    if not x_api_key or x_api_key != WRITE_API_KEY:
        raise HTTPException(401, "X-API-Key header required")


def _require_password(x_password: str | None = Header(default=None)) -> None:
    """Guard launch / cancel / delete-queued endpoints (STOCKPRED_PW)."""
    if WRITE_PW is None:
        raise HTTPException(
            403,
            "Password not configured. Set STOCKPRED_PW env var to enable.",
        )
    if not x_password or x_password != WRITE_PW:
        raise HTTPException(401, "X-Password header required")


def get_db():
    if AppState.SessionLocal is None:
        raise HTTPException(status_code=500, detail="DB not initialised")
    with session_scope(AppState.SessionLocal) as s:
        yield s


# ----- Run serialisation -------------------------------------------------


def _job_id_for_run(s: Session, run_id: int) -> str | None:
    """Reverse-lookup the JobRecord that owns this run, if any.

    Cheap — JobRecord rows are bounded by retention and we only call this
    when serialising a small number of RunSummary rows at a time.
    """
    from stockpred.backend.models import JobRecord

    return s.execute(
        select(JobRecord.job_id).where(JobRecord.run_id == run_id).limit(1)
    ).scalar_one_or_none()


def _run_to_summary(s: Session, run) -> "schemas.RunSummary":
    """Build a RunSummary payload from an ORM Run, including config + job_id."""
    summary = run.summary_json or {}
    return schemas.RunSummary(
        id=run.id,
        started_at=run.started_at,
        completed_at=run.completed_at,
        status=run.status,
        metrics=summary.get("metrics", {}),
        per_horizon_diagnostics=summary.get("per_horizon_diagnostics", {}),
        tickers_count=summary.get("tickers_count", 0),
        note=run.note,
        config=run.config_json or {},
        job_id=_job_id_for_run(s, run.id),
        is_active=bool(getattr(run, "is_active", False)),
        has_report=bool(getattr(run, "report_html", None)),
    )


def _equity_to_payload(rows) -> list["schemas.EquityPoint"]:
    return [
        schemas.EquityPoint(
            date=r.date,
            daily_return=r.daily_return,
            cumulative_return=r.cumulative_return,
            drawdown=r.drawdown,
            turnover=r.turnover,
            benchmark_return=r.benchmark_return,
        )
        for r in rows
    ]


# ----- Pipeline config builder (shared by /refresh and /jobs/run) --------


def _build_pipeline_cfg(
    body: schemas.RefreshRequest,
) -> PipelineConfig | PipelineV5Config:
    cv_cfg = CVConfig(
        train_years=body.cv.train_years,
        test_months=body.cv.test_months,
        embargo_days=body.cv.embargo_days,
        min_train_obs=body.cv.min_train_obs,
    )
    gbm_cfg = GBMConfig(
        num_leaves=body.gbm.num_leaves,
        learning_rate=body.gbm.learning_rate,
        n_estimators=body.gbm.n_estimators,
        min_data_in_leaf=body.gbm.min_data_in_leaf,
        feature_fraction=body.gbm.feature_fraction,
        bagging_fraction=body.gbm.bagging_fraction,
        bagging_freq=body.gbm.bagging_freq,
        reg_lambda=body.gbm.reg_lambda,
        early_stopping_rounds=body.gbm.early_stopping_rounds,
    )
    if body.phase == 5:
        horizons = tuple(body.horizons) if body.horizons is not None else (1, 5)
        return PipelineV5Config(
            start_date=body.start_date,
            end_date=body.end_date,
            n_tickers=body.n_tickers,
            universe_sampling=body.universe_sampling,
            refresh_data=body.refresh_data,
            horizons=horizons,
            model=body.model,
            gbm=gbm_cfg,
            use_sector_features=body.use_sector_features,
            use_tier2_features=body.use_tier2_features,
            use_regime_features=body.use_regime_features,
            beta_neutralise=body.beta_neutralise,
            bootstrap_method=body.bootstrap_method,
            cv=cv_cfg,
            holdout_years=body.holdout_years,
            position_sizing=body.position_sizing,
            k_per_side_pct=body.k_per_side_pct,
            leverage_per_side=body.leverage_per_side,
            sector_cap_gross=body.sector_cap_gross,
            min_trade_threshold=body.min_trade_threshold,
            ensemble_weighting=body.ensemble_weighting,
            bootstrap_n=body.bootstrap_n,
            # Phase 8 meta-labelling + ranks-only
            use_meta_labelling=body.use_meta_labelling,
            meta_threshold=body.meta_threshold,
            ranks_only=body.ranks_only,
            # Phase 9 confidence-weighted sizing + walk-forward meta-CV
            meta_mode=body.meta_mode,
            meta_conf_floor=body.meta_conf_floor,
            meta_conf_cap=body.meta_conf_cap,
            meta_walk_forward_folds=body.meta_walk_forward_folds,
            meta_per_sector=body.meta_per_sector,
            # Phase 7/8 triple-barrier
            use_triple_barrier_labels=body.use_triple_barrier_labels,
            tb_k_sigma=body.tb_k_sigma,
            # Phase 11 feature pruning
            feature_exclude=tuple(body.feature_exclude),
            # Phase 12 / 13 EDGAR
            use_edgar_features=body.use_edgar_features,
            use_edgar_item_features=body.use_edgar_item_features,
            # Phase 14 GDELT
            use_gdelt_features=body.use_gdelt_features,
            # Phase 19 Bayesian shrinkage
            bayesian_shrinkage_alpha=body.bayesian_shrinkage_alpha,
            # Turnover control
            weight_smoothing_alpha=body.weight_smoothing_alpha,
            rebalance_every=body.rebalance_every,
        )
    else:
        horizons = tuple(body.horizons) if body.horizons is not None else (1, 5, 21)
        return PipelineConfig(
            start_date=body.start_date,
            end_date=body.end_date,
            n_tickers=body.n_tickers,
            universe_sampling=body.universe_sampling,
            refresh_data=body.refresh_data,
            horizons=horizons,
            k_per_side=body.k_per_side,
            cv=cv_cfg,
            model=body.model,
            gbm=gbm_cfg,
            use_sector_features=body.use_sector_features,
            feature_cols=body.feature_cols,
        )


def _launch_pipeline(pipeline_cfg, job_id: str) -> None:
    """Start a background thread that acquires _refresh_lock and runs the pipeline."""

    def _run():
        with _refresh_lock:
            jobs_mod.run_pipeline_job(
                AppState.SessionLocal, pipeline_cfg=pipeline_cfg, job_id=job_id
            )

    t = threading.Thread(target=_run, daemon=True)
    jobs_mod.register_job_thread(job_id, t)
    t.start()


def _launch_hypersearch(hypersearch_cfg, job_id: str) -> None:
    """Start a background thread that acquires _refresh_lock and runs a hypersearch."""
    def _run():
        with _refresh_lock:
            jobs_mod.run_hypersearch_job(
                AppState.SessionLocal, hypersearch_cfg=hypersearch_cfg, job_id=job_id
            )
    t = threading.Thread(target=_run, daemon=True)
    jobs_mod.register_job_thread(job_id, t)
    t.start()


# ----- Routes ------------------------------------------------------------


def register_routes(app: FastAPI) -> None:

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #

    @app.get("/healthz", response_model=schemas.HealthResponse, tags=["meta"])
    def healthz():
        db_status = "ok" if AppState.engine is not None else "uninitialised"
        sched_status = "running" if (AppState.scheduler and AppState.scheduler.running) else "off"
        return schemas.HealthResponse(db=db_status, scheduler=sched_status)

    # ------------------------------------------------------------------ #
    # Tickers
    # ------------------------------------------------------------------ #

    @app.get("/tickers", response_model=list[schemas.TickerSummary], tags=["tickers"])
    def list_tickers(s: Session = Depends(get_db)):
        from stockpred.backend.models import Fundamental, PriceBar

        latest_dates = (
            select(PriceBar.ticker, func.max(PriceBar.date).label("max_date"))
            .group_by(PriceBar.ticker)
            .subquery()
        )
        rows = s.execute(
            select(
                PriceBar.ticker,
                PriceBar.adj_close,
                PriceBar.date,
                Fundamental.sector,
                Fundamental.industry,
                Fundamental.market_cap,
            )
            .join(
                latest_dates,
                (PriceBar.ticker == latest_dates.c.ticker)
                & (PriceBar.date == latest_dates.c.max_date),
            )
            .outerjoin(Fundamental, Fundamental.ticker == PriceBar.ticker)
            .order_by(PriceBar.ticker)
        ).all()
        return [
            schemas.TickerSummary(
                ticker=r.ticker,
                sector=r.sector,
                industry=r.industry,
                last_price=float(r.adj_close) if r.adj_close is not None else None,
                market_cap=float(r.market_cap) if r.market_cap is not None else None,
                last_updated=r.date,
            )
            for r in rows
        ]

    @app.get("/tickers/{ticker}/details", response_model=schemas.TickerDetail, tags=["tickers"])
    def ticker_details(
        ticker: str,
        days: int = Query(default=365, ge=1, le=2000),
        run_id: int | None = Query(
            default=None,
            description=(
                "If set, return predictions from this exact run instead of the "
                "active/latest run. Lets the UI compare historical model "
                "predictions for the same ticker across runs."
            ),
        ),
        s: Session = Depends(get_db),
    ):
        fund = store.fundamental_for(s, ticker)
        if fund is None:
            raise HTTPException(404, f"unknown ticker {ticker}")
        start = dt.date.today() - dt.timedelta(days=days)
        prices = store.prices_for_ticker(s, ticker, start=start)
        run = store.resolve_run(s, run_id)
        preds_out: list[schemas.PredictionOut] = []
        if run:
            for p in store.predictions_for_ticker(s, run.id, ticker):
                if p.date >= start:
                    preds_out.append(
                        schemas.PredictionOut(
                            date=p.date,
                            ticker=p.ticker,
                            score=p.score,
                            rank=p.rank,
                            side=p.side,
                            weight=p.weight,
                            per_horizon=p.per_horizon_json or {},
                        )
                    )
        return schemas.TickerDetail(
            ticker=fund.ticker,
            sector=fund.sector,
            industry=fund.industry,
            market_cap=fund.market_cap,
            beta=fund.beta,
            trailing_pe=fund.trailing_pe,
            forward_pe=fund.forward_pe,
            dividend_yield=fund.dividend_yield,
            short_ratio=fund.short_ratio,
            short_percent_of_float=fund.short_percent_of_float,
            fifty_two_week_high=fund.fifty_two_week_high,
            fifty_two_week_low=fund.fifty_two_week_low,
            long_business_summary=fund.long_business_summary,
            prices=[
                schemas.PriceBarOut(
                    date=p.date,
                    open=p.open,
                    high=p.high,
                    low=p.low,
                    close=p.close,
                    adj_close=p.adj_close,
                    volume=p.volume,
                )
                for p in prices
            ],
            predictions=preds_out,
        )

    @app.get("/quote/{ticker}", response_model=schemas.QuoteOut, tags=["tickers"])
    def quote(ticker: str):
        """Latest delayed quote (~15 min, market hours only). Server-cached
        for a few seconds so frontend polling can't rate-limit our IP."""
        import time as _time

        from stockpred.data import prices as prices_mod

        try:
            ticker = schemas._validate_ticker(ticker)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None

        now = _time.monotonic()
        with _quote_lock:
            cached = _quote_cache.get(ticker)
            if cached is not None and (now - cached[0]) < _QUOTE_TTL_S:
                return cached[1]

        q = prices_mod.latest_quote(ticker)
        price = q.get("price")
        prev = q.get("previous_close")
        change = (price - prev) if (price is not None and prev is not None) else None
        change_pct = (change / prev) if (change is not None and prev) else None
        payload = schemas.QuoteOut(
            ticker=ticker,
            price=price,
            previous_close=prev,
            open=q.get("open"),
            day_high=q.get("day_high"),
            day_low=q.get("day_low"),
            volume=q.get("volume"),
            market_cap=q.get("market_cap"),
            change=change,
            change_pct=change_pct,
            as_of=dt.datetime.now(dt.timezone.utc),
        ).model_dump()
        with _quote_lock:
            _quote_cache[ticker] = (now, payload)
        return payload

    # ------------------------------------------------------------------ #
    # Predictions
    # ------------------------------------------------------------------ #

    @app.get("/predictions/latest", response_model=schemas.TopMovers, tags=["predictions"])
    def latest_movers(
        top_k: int = Query(default=10, ge=1, le=100),
        run_id: int | None = Query(
            default=None,
            description=(
                "If set, return predictions from this exact run instead of the "
                "active/latest run. Lets the UI switch its data source live."
            ),
        ),
        s: Session = Depends(get_db),
    ):
        run = store.resolve_run(s, run_id)
        if run is None:
            return schemas.TopMovers(date=None, long=[], short=[])
        data = store.latest_predictions(s, run.id, top_k=top_k)
        date_val = data.get("date")
        return schemas.TopMovers(
            date=date_val,
            long=[
                schemas.PredictionOut(
                    date=p.date,
                    ticker=p.ticker,
                    score=p.score,
                    rank=p.rank,
                    side=p.side,
                    weight=p.weight,
                    per_horizon=p.per_horizon_json or {},
                )
                for p in data["long"]
            ],
            short=[
                schemas.PredictionOut(
                    date=p.date,
                    ticker=p.ticker,
                    score=p.score,
                    rank=p.rank,
                    side=p.side,
                    weight=p.weight,
                    per_horizon=p.per_horizon_json or {},
                )
                for p in data["short"]
            ],
        )

    # ------------------------------------------------------------------ #
    # Runs / backtest
    # ------------------------------------------------------------------ #

    @app.get("/runs", response_model=list[schemas.RunSummary], tags=["runs"])
    def runs(limit: int = Query(default=20, ge=1, le=100), s: Session = Depends(get_db)):
        rows = store.list_runs(s, limit=limit)
        return [_run_to_summary(s, r) for r in rows]

    @app.get("/runs/{run_id}", response_model=schemas.RunSummary, tags=["runs"])
    def get_run_endpoint(run_id: int, s: Session = Depends(get_db)):
        """Single run by id, including config + linked job_id + is_active."""
        run = store.get_run(s, run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return _run_to_summary(s, run)

    @app.get("/runs/{run_id}/equity", response_model=list[schemas.EquityPoint], tags=["runs"])
    def equity_curve(run_id: int, s: Session = Depends(get_db)):
        return _equity_to_payload(store.equity_for_run(s, run_id))

    @app.get("/runs/{run_id}/report", tags=["runs"])
    def run_report(run_id: int, s: Session = Depends(get_db)):
        """Serve the tearsheet HTML for a run. Returns 404 if no report was
        stored (old runs, or run produced no tearsheet)."""
        run = store.get_run(s, run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        html = getattr(run, "report_html", None)
        if not html:
            raise HTTPException(404, f"no report stored for run {run_id}")
        return HTMLResponse(content=html)

    @app.get(
        "/runs/{run_id}/backtest",
        response_model=schemas.BacktestSummary,
        tags=["runs"],
    )
    def run_backtest(run_id: int, s: Session = Depends(get_db)):
        """Full BacktestSummary for an arbitrary run. Used by the Runs history
        page and by `Backtest.tsx` when the user pins / switches runs."""
        run = store.get_run(s, run_id)
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return schemas.BacktestSummary(
            run=_run_to_summary(s, run),
            equity_curve=_equity_to_payload(store.equity_for_run(s, run.id)),
        )

    @app.post(
        "/runs/{run_id}/activate",
        response_model=schemas.ActivateRunResponse,
        tags=["runs"],
        dependencies=[Depends(_require_password)],
    )
    def activate_run_endpoint(run_id: int, s: Session = Depends(get_db)):
        """Pin `run_id` as the global default data source for /predictions/latest,
        /backtest/summary, etc. Requires X-Password (same as job launch — this
        affects every viewer of the live site)."""
        try:
            run = store.activate_run(s, run_id)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        if run is None:
            raise HTTPException(404, f"run {run_id} not found")
        return schemas.ActivateRunResponse(
            active_run_id=run.id,
            message=f"run {run.id} is now the active data source",
        )

    @app.post(
        "/runs/deactivate",
        response_model=schemas.ActivateRunResponse,
        tags=["runs"],
        dependencies=[Depends(_require_password)],
    )
    def deactivate_runs_endpoint(s: Session = Depends(get_db)):
        """Clear the active-run pin. The API reverts to "latest ok run" behaviour.
        Requires X-Password."""
        cleared = store.deactivate_all_runs(s)
        return schemas.ActivateRunResponse(
            active_run_id=None,
            message=f"cleared {cleared} active-run pin(s); reverting to latest-ok",
        )

    @app.get("/backtest/summary", response_model=schemas.BacktestSummary, tags=["backtest"])
    def backtest_summary(
        run_id: int | None = Query(
            default=None,
            description=(
                "If set, return this exact run's backtest instead of the "
                "active/latest run. Lets the UI switch its data source live."
            ),
        ),
        s: Session = Depends(get_db),
    ):
        run = store.resolve_run(s, run_id)
        if run is None:
            raise HTTPException(404, "no runs yet")
        return schemas.BacktestSummary(
            run=_run_to_summary(s, run),
            equity_curve=_equity_to_payload(store.equity_for_run(s, run.id)),
        )

    # ------------------------------------------------------------------ #
    # Jobs — existing refresh + new queue/run/cancel endpoints
    # ------------------------------------------------------------------ #

    @app.post(
        "/jobs/refresh",
        response_model=schemas.JobResponse,
        tags=["jobs"],
        dependencies=[Depends(_require_api_key)],
    )
    def refresh(body: schemas.RefreshRequest | None = None):
        """Trigger a pipeline run immediately. Requires X-API-Key."""
        if AppState.SessionLocal is None:
            raise HTTPException(500, "DB not initialised")
        if _is_pipeline_running():
            raise HTTPException(409, "A pipeline run is already in flight")
        body = body or schemas.RefreshRequest()
        pipeline_cfg = _build_pipeline_cfg(body)
        job_id = str(_uuid.uuid4())
        jobs_mod._record_job(job_id, "queued")
        _launch_pipeline(pipeline_cfg, job_id)
        return schemas.JobResponse(job_id=job_id, status="queued")

    @app.post(
        "/jobs/queue",
        response_model=schemas.QueuedJobOut,
        tags=["jobs"],
    )
    def queue_job(body: schemas.RefreshRequest, s: Session = Depends(get_db)):
        """Queue a pipeline job for later password-protected launch. No auth required.
        Up to 5 pending jobs at once. For hypersearch, use POST /hypersearch/queue."""
        try:
            qj = store.create_queued_job(s, config=body.model_dump())
        except ValueError as e:
            raise HTTPException(429, str(e)) from None
        return schemas.QueuedJobOut(
            id=qj.id, created_at=qj.created_at, config=qj.config_json,
            label=qj.label, status=qj.status,
        )

    @app.get(
        "/jobs/queue",
        response_model=list[schemas.QueuedJobOut],
        tags=["jobs"],
    )
    def list_queued_jobs(s: Session = Depends(get_db)):
        """List all queued jobs (pending, launched, cancelled)."""
        return [
            schemas.QueuedJobOut(
                id=j.id,
                created_at=j.created_at,
                config=j.config_json,
                label=j.label,
                status=j.status,
                launched_at=j.launched_at,
                job_id=j.job_id,
            )
            for j in store.list_queued_jobs(s)
        ]

    @app.post(
        "/jobs/run/{queue_id}",
        response_model=schemas.JobResponse,
        tags=["jobs"],
        dependencies=[Depends(_require_password)],
    )
    def run_queued_job(queue_id: str, s: Session = Depends(get_db)):
        """Launch a pending queued job (pipeline or hypersearch). Requires X-Password header.

        The queue entry transitions `pending` → `launched` and stays in the
        listing as an audit trail (with `job_id` populated so the UI can
        cross-link to the live job). It does NOT count toward the 5-pending
        cap any more because `count_pending_queued_jobs` filters by
        `status='pending'`.
        """
        if AppState.SessionLocal is None:
            raise HTTPException(500, "DB not initialised")
        qj = store.get_queued_job(s, queue_id)
        if qj is None:
            raise HTTPException(404, "queued job not found")
        if qj.status != "pending":
            raise HTTPException(409, f"job is already {qj.status}")
        if _is_pipeline_running():
            raise HTTPException(409, "A job is already in flight")

        job_id = str(_uuid.uuid4())

        # Keep the queue row (status='launched', job_id set) so the Jobs UI
        # can show the trail "queued at X → launched at Y → job <uuid>".
        store.mark_queued_launched(s, queue_id, job_id)
        jobs_mod._record_job(job_id, "queued")

        job_type = qj.config_json.get("job_type", "pipeline")
        if job_type == "hypersearch":
            from stockpred.hypersearch import HypersearchConfig
            cfg_data = {k: v for k, v in qj.config_json.items() if k != "job_type"}
            hs_cfg = HypersearchConfig(**{
                f: cfg_data[f] for f in HypersearchConfig.__dataclass_fields__ if f in cfg_data
            })
            _launch_hypersearch(hs_cfg, job_id)
        else:
            body = schemas.RefreshRequest(**qj.config_json)
            pipeline_cfg = _build_pipeline_cfg(body)
            _launch_pipeline(pipeline_cfg, job_id)

        return schemas.JobResponse(job_id=job_id, status="queued")

    @app.delete(
        "/jobs/queue/{queue_id}",
        tags=["jobs"],
        dependencies=[Depends(_require_password)],
    )
    def delete_queued_job(queue_id: str, s: Session = Depends(get_db)):
        """Delete a pending queued job. Requires X-Password header."""
        deleted = store.delete_queued_job(s, queue_id)
        if not deleted:
            raise HTTPException(404, "queued job not found")
        return {"ok": True}

    @app.delete(
        "/jobs/{job_id}/cancel",
        tags=["jobs"],
        dependencies=[Depends(_require_password)],
    )
    def cancel_job(job_id: str):
        """Soft-cancel a running job. The thread finishes but results are discarded.
        Requires X-Password header."""
        cancelled = jobs_mod.request_cancel(job_id)
        if not cancelled:
            raise HTTPException(404, "job not found or not running")
        return {"ok": True}

    @app.get("/jobs/{job_id}", response_model=schemas.JobDetail, tags=["jobs"])
    def job_status(job_id: str):
        rec = jobs_mod.get_job_status(job_id, AppState.SessionLocal)
        if rec is None:
            raise HTTPException(404, "unknown job")
        cfg = rec.get("config", {})
        return schemas.JobDetail(
            job_id=job_id,
            status=rec["status"],
            job_type=cfg.get("job_type", "pipeline"),
            started_at=rec.get("started_at"),
            updated_at=rec.get("updated_at"),
            config=cfg,
            logs=rec.get("logs", []),
            run_id=rec.get("run_id"),
            elapsed_s=rec.get("elapsed_s"),
            error=rec.get("error"),
        )

    @app.get("/jobs", response_model=list[schemas.JobDetail], tags=["jobs"])
    def jobs_list(limit: int = Query(default=25, ge=1, le=100)):
        return [
            schemas.JobDetail(
                job_id=item["job_id"],
                status=item["status"],
                job_type=(item.get("config") or {}).get("job_type", "pipeline"),
                started_at=item.get("started_at"),
                updated_at=item.get("updated_at"),
                config=item.get("config", {}),
                logs=[],
                run_id=item.get("run_id"),
                elapsed_s=item.get("elapsed_s"),
                error=item.get("error"),
            )
            for item in jobs_mod.list_jobs(limit, AppState.SessionLocal)
        ]

    # ------------------------------------------------------------------ #
    # Hypersearch results
    # ------------------------------------------------------------------ #

    def _hs_trial_out(row: dict) -> schemas.HypersearchTrialOut:
        """Convert a flat trial dict to the API schema."""
        param_keys = {
            "position_sizing", "k_per_side_pct", "leverage_per_side", "sector_cap_gross",
            "min_trade_threshold", "horizons", "ensemble_weighting", "use_tier2_features",
            "use_regime_features", "use_sector_features", "ranks_only", "beta_neutralise",
            "use_meta_labelling", "meta_threshold", "meta_mode", "meta_conf_floor",
            "num_leaves", "learning_rate", "n_estimators", "train_years",
        }
        params = {k: v for k, v in row.items() if k in param_keys}
        return schemas.HypersearchTrialOut(
            trial=row.get("trial", 0),
            value=row.get("value"),
            hold_sharpe=row.get("hold_sharpe"),
            hold_ci_lo=row.get("hold_ci_lo"),
            hold_ci_hi=row.get("hold_ci_hi"),
            hold_dd=row.get("hold_dd"),
            hold_hit=row.get("hold_hit"),
            hold_ann_return=row.get("hold_ann_return"),
            dev_sharpe=row.get("dev_sharpe"),
            elapsed_s=row.get("elapsed_s"),
            error=row.get("error") or None,
            params=params,
        )

    def _hs_run_out(run, *, include_trials: bool = True) -> schemas.HypersearchRunOut:
        trials = []
        if include_trials and run.trials_json:
            trials = [_hs_trial_out(r) for r in run.trials_json]
        return schemas.HypersearchRunOut(
            id=run.id,
            job_id=run.job_id,
            started_at=run.started_at,
            completed_at=run.completed_at,
            status=run.status,
            config=_sanitize(run.config_json or {}),
            n_trials_requested=run.n_trials_requested,
            n_trials_done=run.n_trials_done,
            best_sharpe=run.best_sharpe,
            best_params=run.best_params_json,
            trials=trials,
        )

    @app.post("/hypersearch/queue", response_model=schemas.QueuedJobOut, tags=["hypersearch"])
    def queue_hypersearch(body: schemas.HypersearchRequest, s: Session = Depends(get_db)):
        """Queue a hyperparameter search job. No auth required. Up to 5 pending at once.
        Launch via POST /jobs/run/{queue_id} with X-Password."""
        config = body.model_dump()
        config["job_type"] = "hypersearch"
        try:
            qj = store.create_queued_job(s, config=config)
        except ValueError as e:
            raise HTTPException(429, str(e)) from None
        return schemas.QueuedJobOut(
            id=qj.id, created_at=qj.created_at, config=qj.config_json,
            label=qj.label, status=qj.status,
        )

    @app.get("/hypersearch/runs", response_model=list[schemas.HypersearchRunOut], tags=["hypersearch"])
    def list_hypersearch_runs(
        limit: int = Query(default=25, ge=1, le=100),
        s: Session = Depends(get_db),
    ):
        """List all hypersearch runs (metadata only, no trials)."""
        runs = store.list_hypersearch_runs(s, limit=limit)
        return [_hs_run_out(r, include_trials=False) for r in runs]

    @app.get("/hypersearch/runs/{run_id}", response_model=schemas.HypersearchRunOut, tags=["hypersearch"])
    def get_hypersearch_run(run_id: int, s: Session = Depends(get_db)):
        """Full detail for one hypersearch run including all trial results."""
        run = store.get_hypersearch_run(s, run_id)
        if run is None:
            raise HTTPException(404, "hypersearch run not found")
        return _hs_run_out(run, include_trials=True)

    @app.get("/hypersearch/runs/by-job/{job_id}", response_model=schemas.HypersearchRunOut, tags=["hypersearch"])
    def get_hypersearch_run_by_job(job_id: str, s: Session = Depends(get_db)):
        """Get the hypersearch run linked to a specific job_id."""
        run = store.get_hypersearch_run_by_job(s, job_id)
        if run is None:
            raise HTTPException(404, "no hypersearch run found for this job")
        return _hs_run_out(run, include_trials=True)

    # ------------------------------------------------------------------ #
    # Watchlist
    # ------------------------------------------------------------------ #

    @app.get("/watchlist", response_model=list[schemas.WatchedItem], tags=["watchlist"])
    def watchlist(s: Session = Depends(get_db)):
        from stockpred.backend.models import PriceBar

        rows = store.list_watched(s)
        latest = dict(
            s.execute(
                select(PriceBar.ticker, func.max(PriceBar.date))
                .where(PriceBar.ticker.in_([r.ticker for r in rows]))
                .group_by(PriceBar.ticker)
            ).all()
        )
        out = []
        for r in rows:
            last_price = None
            last_dt = latest.get(r.ticker)
            if last_dt is not None:
                bar = s.execute(
                    select(PriceBar.adj_close).where(
                        PriceBar.ticker == r.ticker, PriceBar.date == last_dt
                    )
                ).scalar_one_or_none()
                last_price = float(bar) if bar is not None else None
            out.append(
                schemas.WatchedItem(
                    ticker=r.ticker,
                    label=r.label,
                    category=r.category,
                    note=r.note,
                    last_price=last_price,
                    last_updated=last_dt,
                )
            )
        return out

    @app.post(
        "/watchlist",
        response_model=schemas.WatchedItem,
        tags=["watchlist"],
        dependencies=[Depends(_require_api_key)],
    )
    def watchlist_add(item: schemas.WatchedAdd, s: Session = Depends(get_db)):
        from stockpred.data import prices as prices_mod

        try:
            df = prices_mod.fetch_one(item.ticker)
            if not df.empty:
                rows = (
                    df.reset_index()
                    .assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)
                    .assign(ticker=item.ticker)
                    .to_dict("records")
                )
                store.upsert_prices(s, rows)
        except Exception as e:  # noqa: BLE001
            log.warning("watchlist add: prices fetch failed for %s: %s", item.ticker, e)
        added = store.add_watched(
            s, item.ticker, label=item.label, category=item.category, note=item.note
        )
        return schemas.WatchedItem(
            ticker=added.ticker,
            label=added.label,
            category=added.category,
            note=added.note,
        )

    @app.delete(
        "/watchlist/{ticker}",
        tags=["watchlist"],
        dependencies=[Depends(_require_api_key)],
    )
    def watchlist_remove(ticker: str, s: Session = Depends(get_db)):
        try:
            ticker = schemas._validate_ticker(ticker)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        removed = store.remove_watched(s, ticker)
        if not removed:
            raise HTTPException(404, "not in watchlist")
        return {"ok": True}

    # ------------------------------------------------------------------ #
    # News
    # ------------------------------------------------------------------ #

    @app.get(
        "/tickers/{ticker}/news",
        response_model=list[schemas.NewsHeadline],
        tags=["tickers"],
    )
    def ticker_news(
        ticker: str,
        limit: int = Query(default=20, ge=1, le=100),
        refresh: bool = Query(default=False),
        with_sentiment: bool = Query(
            default=True,
            description=(
                "Phase 15: when True (default), each headline is scored "
                "with FinBERT (cached). If FinBERT isn't installed the "
                "score fields are None and the rest of the payload is "
                "unchanged. Set False to skip sentiment lookup entirely."
            ),
        ),
        s: Session = Depends(get_db),
    ):
        try:
            ticker = schemas._validate_ticker(ticker)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if refresh:
            from stockpred.data import news as news_mod

            items = news_mod.fetch_one(ticker, max_items=limit, refresh=True)
            store.upsert_news(s, ticker, items)
        rows = store.news_for_ticker(s, ticker, limit=limit)

        # Phase 15: score headlines via FinBERT (lazy-loaded; cached
        # per headline). If not installed, all sentiment_* fields stay
        # None and the legacy payload is preserved exactly.
        #
        # Reviewer C3 (2026-06-05): the try block wraps BOTH the
        # scorer call AND the serialization step, since malformed
        # results (truncated list, non-dict entries) would otherwise
        # raise AttributeError in the list comprehension below.
        try:
            scores: list[dict | None] = [None] * len(rows)
            if with_sentiment and rows:
                from stockpred.data import sentiment as sent_mod

                titles = [r.title or "" for r in rows]
                scored = sent_mod.score_headlines(titles)
                # Defensive: scorer contract says len(scored) == len(titles)
                # but a future drift would silently truncate / extend.
                if len(scored) != len(rows):
                    log.warning(
                        "Sentiment scorer returned %d items for %d input headlines; "
                        "skipping sentiment fields.",
                        len(scored),
                        len(rows),
                    )
                else:
                    scores = [
                        (s if isinstance(s, dict) and s.get("label") != "unavailable" else None)
                        for s in scored
                    ]

            return [
                schemas.NewsHeadline(
                    uuid=r.uuid,
                    title=r.title,
                    publisher=r.publisher,
                    link=r.link,
                    type=r.type,
                    published_at=r.published_at,
                    sentiment_label=(scores[i] or {}).get("label") if scores[i] else None,
                    sentiment_net=(scores[i] or {}).get("net") if scores[i] else None,
                    sentiment_positive=(scores[i] or {}).get("positive") if scores[i] else None,
                    sentiment_neutral=(scores[i] or {}).get("neutral") if scores[i] else None,
                    sentiment_negative=(scores[i] or {}).get("negative") if scores[i] else None,
                )
                for i, r in enumerate(rows)
            ]
        except Exception as e:  # noqa: BLE001
            log.warning(
                "Sentiment scoring or serialization failed (%s); "
                "returning headlines without sentiment.",
                e,
            )
            return [
                schemas.NewsHeadline(
                    uuid=r.uuid,
                    title=r.title,
                    publisher=r.publisher,
                    link=r.link,
                    type=r.type,
                    published_at=r.published_at,
                )
                for r in rows
            ]


# ----- Static frontend ---------------------------------------------------


def register_static(app: FastAPI) -> None:
    dist = Path(WEB_DIST).resolve()
    if not dist.exists():
        log.info("Static frontend not found at %s; SPA disabled", dist)
        return

    assets_dir = dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(dist / "index.html"))

    def _safe_target(full_path: str) -> Path | None:
        target = (dist / full_path).resolve()
        try:
            target.relative_to(dist)
        except ValueError:
            return None
        if not target.exists() or not target.is_file():
            return None
        return target

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str):
        target = _safe_target(full_path)
        if target is not None:
            return FileResponse(str(target))
        return FileResponse(str(dist / "index.html"))


app = create_app()
