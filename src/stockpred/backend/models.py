"""ORM models for the stock-predictor backend.

Conceptual entities:

* **Run**     — one execution of the pipeline. Has a started_at, completed_at,
                status, config snapshot, summary metrics.
* **Prediction** — one row per (run, date, ticker). Score and per-horizon
                breakdown stored as a small JSON blob.
* **PriceBar** — one row per (date, ticker). Adjusted OHLCV. Shared across
                runs so the table stays small.
* **Fundamental** — one row per ticker. Static-ish info from yfinance .info.
* **EquitySample** — one row per (run, date) of the backtest equity curve.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from stockpred.backend.db import Base


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    note: Mapped[str | None] = mapped_column(String(512), nullable=True)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )
    equity_samples: Mapped[list["EquitySample"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(String(16), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)  # 'long' / 'short' / None
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    per_horizon_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    run: Mapped[Run] = relationship(back_populates="predictions")

    __table_args__ = (
        Index("ix_predictions_run_date", "run_id", "date"),
        Index("ix_predictions_run_ticker", "run_id", "ticker"),
        UniqueConstraint("run_id", "date", "ticker", name="uq_predictions_run_date_ticker"),
    )


class PriceBar(Base):
    __tablename__ = "price_bars"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    date: Mapped[dt.date] = mapped_column(Date, primary_key=True)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    adj_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)


class Fundamental(Base):
    __tablename__ = "fundamentals"

    ticker: Mapped[str] = mapped_column(String(16), primary_key=True)
    sector: Mapped[str | None] = mapped_column(String(64))
    industry: Mapped[str | None] = mapped_column(String(128))
    market_cap: Mapped[float | None] = mapped_column(Float)
    beta: Mapped[float | None] = mapped_column(Float)
    trailing_pe: Mapped[float | None] = mapped_column(Float)
    forward_pe: Mapped[float | None] = mapped_column(Float)
    dividend_yield: Mapped[float | None] = mapped_column(Float)
    short_ratio: Mapped[float | None] = mapped_column(Float)
    short_percent_of_float: Mapped[float | None] = mapped_column(Float)
    fifty_two_week_high: Mapped[float | None] = mapped_column(Float)
    fifty_two_week_low: Mapped[float | None] = mapped_column(Float)
    long_business_summary: Mapped[str | None] = mapped_column(String(8192))
    updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class EquitySample(Base):
    __tablename__ = "equity_samples"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(
        ForeignKey("runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    date: Mapped[dt.date] = mapped_column(Date, nullable=False)
    daily_return: Mapped[float | None] = mapped_column(Float)
    cumulative_return: Mapped[float | None] = mapped_column(Float)
    drawdown: Mapped[float | None] = mapped_column(Float)
    turnover: Mapped[float | None] = mapped_column(Float)
    benchmark_return: Mapped[float | None] = mapped_column(Float)

    run: Mapped[Run] = relationship(back_populates="equity_samples")

    __table_args__ = (
        Index("ix_equity_run_date", "run_id", "date"),
        UniqueConstraint("run_id", "date", name="uq_equity_run_date"),
    )


class WatchedTicker(Base):
    """Arbitrary tickers tracked outside the model's universe (e.g. HND.TO,
    HNU.TO, BTC-USD). Charted on the screener and ticker pages but NEVER used
    as model targets or features — different instruments behave differently
    (leveraged ETFs decay, crypto trades 24/7, etc.)."""

    __tablename__ = "watched_tickers"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    label: Mapped[str | None] = mapped_column(String(128))
    category: Mapped[str | None] = mapped_column(
        String(64)
    )  # 'equity', 'leveraged_etf', 'commodity', ...
    note: Mapped[str | None] = mapped_column(String(512))
    added_at: Mapped[dt.datetime | None] = mapped_column(DateTime)


class QueuedJob(Base):
    """A pipeline job submitted by the UI and waiting for password-protected approval
    before it actually runs.  Max 5 pending at once; launched by POST /jobs/run/{id}."""

    __tablename__ = "queued_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    label: Mapped[str | None] = mapped_column(String(256), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    launched_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class JobRecord(Base):
    """Persistent job execution record. Survives server restarts so crashed/
    completed jobs remain queryable via GET /jobs/{job_id}."""

    __tablename__ = "job_records"

    job_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    elapsed_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error: Mapped[str | None] = mapped_column(String(4096), nullable=True)
    config_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    logs_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)


class HypersearchRun(Base):
    """Persistent record for one hyperparameter search job.

    Created when a hypersearch job starts; updated after each trial via
    `store.update_hypersearch_run`; finalised when the job completes.
    Linked to `JobRecord` by `job_id` (nullable — can exist standalone).
    """

    __tablename__ = "hypersearch_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, nullable=False)
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    config_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    n_trials_requested: Mapped[int] = mapped_column(Integer, nullable=False, default=50)
    n_trials_done: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    best_sharpe: Mapped[float | None] = mapped_column(Float, nullable=True)
    best_params_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    trials_json: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)


class NewsItem(Base):
    """Headline-level news from yfinance Ticker.news. Free, no API key.

    Persisted with a (ticker, uuid) composite primary key so re-fetches are
    idempotent. The model does NOT consume this table; it's purely a
    presentation feed for the ticker detail page (CONCEPTS.md §7e).
    """

    __tablename__ = "news_items"

    ticker: Mapped[str] = mapped_column(String(32), primary_key=True)
    uuid: Mapped[str] = mapped_column(String(128), primary_key=True)
    title: Mapped[str | None] = mapped_column(String(1024))
    publisher: Mapped[str | None] = mapped_column(String(256))
    link: Mapped[str | None] = mapped_column(String(2048))
    type: Mapped[str | None] = mapped_column(String(64))
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    fetched_at: Mapped[dt.datetime | None] = mapped_column(DateTime)

    __table_args__ = (Index("ix_news_ticker_published", "ticker", "published_at"),)
