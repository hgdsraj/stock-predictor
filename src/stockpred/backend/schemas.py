"""Pydantic response models for the API. Designed to match the frontend's needs."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel


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


class JobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    db: str
    scheduler: str
