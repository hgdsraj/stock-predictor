"""Pydantic response models for the API. Designed to match the frontend's needs."""

from __future__ import annotations

import datetime as dt
import re

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


class JobResponse(BaseModel):
    job_id: str
    status: str
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str = "ok"
    db: str
    scheduler: str


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
