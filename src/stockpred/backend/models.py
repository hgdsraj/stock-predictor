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
