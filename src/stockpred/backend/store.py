"""Repository-pattern read/write helpers over the ORM models.

Routes call these instead of the ORM directly, keeping the API layer thin and
testable. Every function takes a `Session` so it composes with `session_scope`.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Iterable, Sequence

import pandas as pd
from sqlalchemy import delete, desc, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from stockpred.backend.models import (
    EquitySample,
    Fundamental,
    JobRecord,
    NewsItem,
    PriceBar,
    Prediction,
    QueuedJob,
    Run,
    WatchedTicker,
)

log = logging.getLogger(__name__)

# SQLite hard-limits bound variables to 999 per statement.
# Use this helper to split any bulk payload into safe chunks.
def _chunks(payload: list, n_cols: int):
    size = 999 // n_cols
    for i in range(0, len(payload), size):
        yield payload[i : i + size]


# --------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------- #


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)


def create_run(s: Session, *, config: dict, note: str | None = None) -> Run:
    run = Run(
        started_at=_now(),
        status="running",
        config_json=config,
        summary_json={},
        note=note,
    )
    s.add(run)
    s.flush()  # populate run.id
    return run


def complete_run(s: Session, run: Run, *, summary: dict, status: str = "ok") -> None:
    run.completed_at = _now()
    run.status = status
    run.summary_json = summary
    s.add(run)


def fail_run(s: Session, run: Run, *, error: str) -> None:
    run.completed_at = _now()
    run.status = "failed"
    run.summary_json = {"error": error}
    s.add(run)


def latest_run(s: Session, *, status: str | None = "ok") -> Run | None:
    stmt = select(Run).order_by(desc(Run.completed_at))
    if status:
        stmt = stmt.where(Run.status == status)
    return s.execute(stmt.limit(1)).scalar_one_or_none()


def list_runs(s: Session, *, limit: int = 20) -> list[Run]:
    return list(s.execute(select(Run).order_by(desc(Run.started_at)).limit(limit)).scalars().all())


# --------------------------------------------------------------------- #
# Predictions
# --------------------------------------------------------------------- #


def upsert_predictions(s: Session, run: Run, rows: Iterable[dict]) -> int:
    """Bulk insert predictions for a run. Returns count inserted."""
    payload = [{**r, "run_id": run.id} for r in rows]
    if not payload:
        return 0
    for chunk in _chunks(payload, 8):
        stmt = sqlite_insert(Prediction).values(chunk)
        stmt = stmt.on_conflict_do_nothing(index_elements=["run_id", "date", "ticker"])
        s.execute(stmt)
    return len(payload)


def predictions_for_run(
    s: Session, run_id: int, *, date: dt.date | None = None
) -> list[Prediction]:
    stmt = select(Prediction).where(Prediction.run_id == run_id)
    if date is not None:
        stmt = stmt.where(Prediction.date == date)
    return list(s.execute(stmt).scalars().all())


def predictions_for_ticker(s: Session, run_id: int, ticker: str) -> list[Prediction]:
    stmt = (
        select(Prediction)
        .where(Prediction.run_id == run_id, Prediction.ticker == ticker)
        .order_by(Prediction.date)
    )
    return list(s.execute(stmt).scalars().all())


def latest_predictions(s: Session, run_id: int, *, top_k: int = 10) -> dict[str, list[Prediction]]:
    """Most recent date's predictions, split into long / short top-k."""
    # Find max date for the run.
    max_date = s.execute(
        select(Prediction.date)
        .where(Prediction.run_id == run_id)
        .order_by(desc(Prediction.date))
        .limit(1)
    ).scalar_one_or_none()
    if max_date is None:
        return {"long": [], "short": []}
    rows = predictions_for_run(s, run_id, date=max_date)
    longs = sorted(
        [r for r in rows if r.weight and r.weight > 0],
        key=lambda r: -float(r.score),
    )[:top_k]
    shorts = sorted(
        [r for r in rows if r.weight and r.weight < 0],
        key=lambda r: float(r.score),
    )[:top_k]
    return {"long": longs, "short": shorts, "date": max_date}


# --------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------- #


def upsert_prices(s: Session, rows: Iterable[dict]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    for chunk in _chunks(payload, 8):
        stmt = sqlite_insert(PriceBar).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "date"],
            set_={
                "open": stmt.excluded.open,
                "high": stmt.excluded.high,
                "low": stmt.excluded.low,
                "close": stmt.excluded.close,
                "adj_close": stmt.excluded.adj_close,
                "volume": stmt.excluded.volume,
            },
        )
        s.execute(stmt)
    return len(payload)


def prices_for_ticker(
    s: Session,
    ticker: str,
    *,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> list[PriceBar]:
    stmt = select(PriceBar).where(PriceBar.ticker == ticker)
    if start:
        stmt = stmt.where(PriceBar.date >= start)
    if end:
        stmt = stmt.where(PriceBar.date <= end)
    stmt = stmt.order_by(PriceBar.date)
    return list(s.execute(stmt).scalars().all())


def all_tickers(s: Session) -> list[str]:
    return list(s.execute(select(PriceBar.ticker).distinct()).scalars().all())


# --------------------------------------------------------------------- #
# Fundamentals
# --------------------------------------------------------------------- #


def upsert_fundamentals(s: Session, rows: Iterable[dict]) -> int:
    payload = list(rows)
    if not payload:
        return 0
    # Fundamental has 14 columns → chunk_size = 71
    for chunk in _chunks(payload, 14):
        stmt = sqlite_insert(Fundamental).values(chunk)
        excluded_cols = {
            c.name: stmt.excluded[c.name]
            for c in Fundamental.__table__.columns
            if c.name != "ticker"
        }
        stmt = stmt.on_conflict_do_update(index_elements=["ticker"], set_=excluded_cols)
        s.execute(stmt)
    return len(payload)


def fundamental_for(s: Session, ticker: str) -> Fundamental | None:
    return s.execute(select(Fundamental).where(Fundamental.ticker == ticker)).scalar_one_or_none()


# --------------------------------------------------------------------- #
# Equity samples
# --------------------------------------------------------------------- #


def upsert_equity(s: Session, run: Run, rows: Iterable[dict]) -> int:
    payload = [{**r, "run_id": run.id} for r in rows]
    if not payload:
        return 0
    # EquitySample has 7 non-autoincrement columns → chunk_size = 142
    for chunk in _chunks(payload, 7):
        stmt = sqlite_insert(EquitySample).values(chunk)
        stmt = stmt.on_conflict_do_nothing(index_elements=["run_id", "date"])
        s.execute(stmt)
    return len(payload)


def equity_for_run(s: Session, run_id: int) -> list[EquitySample]:
    return list(
        s.execute(
            select(EquitySample).where(EquitySample.run_id == run_id).order_by(EquitySample.date)
        )
        .scalars()
        .all()
    )


# --------------------------------------------------------------------- #
# Watchlist
# --------------------------------------------------------------------- #


def list_watched(s: Session) -> list[WatchedTicker]:
    return list(s.execute(select(WatchedTicker).order_by(WatchedTicker.ticker)).scalars().all())


def add_watched(
    s: Session,
    ticker: str,
    *,
    label: str | None = None,
    category: str | None = None,
    note: str | None = None,
) -> WatchedTicker:
    stmt = sqlite_insert(WatchedTicker).values(
        {
            "ticker": ticker,
            "label": label,
            "category": category,
            "note": note,
            "added_at": _now(),
        }
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["ticker"],
        set_={
            "label": stmt.excluded.label,
            "category": stmt.excluded.category,
            "note": stmt.excluded.note,
        },
    )
    s.execute(stmt)
    return s.get(WatchedTicker, ticker)


def remove_watched(s: Session, ticker: str) -> bool:
    row = s.get(WatchedTicker, ticker)
    if row is None:
        return False
    s.delete(row)
    return True


def is_watched(s: Session, ticker: str) -> bool:
    return s.get(WatchedTicker, ticker) is not None


def seed_default_watchlist(s: Session) -> None:
    """Seed a small default watchlist on first boot. Idempotent."""
    defaults = [
        (
            "HND.TO",
            "Horizons NaturalGas Bear (2x)",
            "leveraged_etf",
            "WARNING: 2x leveraged daily-reset; volatility decay over time",
        ),
        (
            "HNU.TO",
            "Horizons NaturalGas Bull (2x)",
            "leveraged_etf",
            "WARNING: 2x leveraged daily-reset; volatility decay over time",
        ),
        ("UNG", "United States Natural Gas Fund", "commodity_etf", None),
        ("SPY", "SPDR S&P 500 ETF", "index_etf", "Benchmark"),
        (
            "^VIX",
            "CBOE Volatility Index",
            "regime",
            "Implied vol of S&P 500 options; regime indicator",
        ),
    ]
    for tkr, label, cat, note in defaults:
        if not is_watched(s, tkr):
            add_watched(s, tkr, label=label, category=cat, note=note)


# --------------------------------------------------------------------- #
# News
# --------------------------------------------------------------------- #


def upsert_news(s: Session, ticker: str, items: Iterable[dict]) -> int:
    """Bulk-upsert news items for a ticker. (ticker, uuid) primary key
    makes this naturally idempotent."""
    payload = []
    now = _now()
    for it in items:
        if not it.get("uuid"):
            continue
        payload.append(
            {
                "ticker": ticker,
                "uuid": it["uuid"],
                "title": it.get("title"),
                "publisher": it.get("publisher"),
                "link": it.get("link"),
                "type": it.get("type"),
                "published_at": it.get("published_at"),
                "fetched_at": now,
            }
        )
    if not payload:
        return 0
    for chunk in _chunks(payload, 8):
        stmt = sqlite_insert(NewsItem).values(chunk)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "uuid"],
            set_={
                "title": stmt.excluded.title,
                "publisher": stmt.excluded.publisher,
                "link": stmt.excluded.link,
                "type": stmt.excluded.type,
                "published_at": stmt.excluded.published_at,
                "fetched_at": stmt.excluded.fetched_at,
            },
        )
        s.execute(stmt)
    return len(payload)


def news_for_ticker(s: Session, ticker: str, *, limit: int = 20) -> list[NewsItem]:
    return list(
        s.execute(
            select(NewsItem)
            .where(NewsItem.ticker == ticker)
            .order_by(desc(NewsItem.published_at))
            .limit(limit)
        )
        .scalars()
        .all()
    )


# --------------------------------------------------------------------- #
# Queued jobs
# --------------------------------------------------------------------- #

_MAX_PENDING_JOBS = 5


def count_pending_queued_jobs(s: Session) -> int:
    from sqlalchemy import func
    return s.execute(
        select(func.count()).select_from(QueuedJob).where(QueuedJob.status == "pending")
    ).scalar_one()


def create_queued_job(
    s: Session, config: dict, *, label: str | None = None
) -> QueuedJob:
    """Create a pending queued job. Raises ValueError if the pending cap is hit."""
    import uuid as _uuid

    if count_pending_queued_jobs(s) >= _MAX_PENDING_JOBS:
        raise ValueError(f"Maximum of {_MAX_PENDING_JOBS} pending queued jobs reached")
    job = QueuedJob(
        id=str(_uuid.uuid4()),
        created_at=_now(),
        config_json=config,
        label=label,
        status="pending",
    )
    s.add(job)
    s.flush()
    return job


def list_queued_jobs(s: Session) -> list[QueuedJob]:
    return list(
        s.execute(select(QueuedJob).order_by(desc(QueuedJob.created_at))).scalars().all()
    )


def get_queued_job(s: Session, queue_id: str) -> QueuedJob | None:
    return s.get(QueuedJob, queue_id)


def mark_queued_launched(s: Session, queue_id: str, job_id: str) -> QueuedJob | None:
    job = s.get(QueuedJob, queue_id)
    if job is None:
        return None
    job.status = "launched"
    job.launched_at = _now()
    job.job_id = job_id
    s.add(job)
    return job


def delete_queued_job(s: Session, queue_id: str) -> bool:
    job = s.get(QueuedJob, queue_id)
    if job is None:
        return False
    s.delete(job)
    return True


# --------------------------------------------------------------------- #
# Job records (persistent across restarts)
# --------------------------------------------------------------------- #


def upsert_job_record(
    s: Session,
    job_id: str,
    status: str,
    *,
    started_at=None,
    updated_at=None,
    elapsed_s=None,
    run_id=None,
    error=None,
    config=None,
) -> None:
    stmt = sqlite_insert(JobRecord).values(
        job_id=job_id,
        status=status,
        started_at=started_at,
        updated_at=updated_at,
        elapsed_s=elapsed_s,
        run_id=run_id,
        error=error,
        config_json=config,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=["job_id"],
        set_={
            "status": stmt.excluded.status,
            "started_at": stmt.excluded.started_at,
            "updated_at": stmt.excluded.updated_at,
            "elapsed_s": stmt.excluded.elapsed_s,
            "run_id": stmt.excluded.run_id,
            "error": stmt.excluded.error,
            "config_json": stmt.excluded.config_json,
        },
    )
    s.execute(stmt)


def get_job_record(s: Session, job_id: str) -> JobRecord | None:
    return s.get(JobRecord, job_id)


def list_job_records(s: Session, *, limit: int = 50) -> list[JobRecord]:
    return list(
        s.execute(
            select(JobRecord).order_by(desc(JobRecord.updated_at)).limit(limit)
        )
        .scalars()
        .all()
    )


def mark_stale_jobs_crashed(s: Session) -> int:
    """On startup, any job_records still in 'running'/'queued' state were
    interrupted by a server crash/restart. Mark them crashed."""
    from sqlalchemy import update

    result = s.execute(
        update(JobRecord)
        .where(JobRecord.status.in_(["running", "queued", "cancelling"]))
        .values(status="crashed", updated_at=_now())
    )
    return result.rowcount
