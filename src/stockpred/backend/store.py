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
    PriceBar,
    Prediction,
    Run,
)

log = logging.getLogger(__name__)


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
    stmt = sqlite_insert(Prediction).values(payload)
    # If the same run re-inserts (unlikely), ignore conflicts.
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
    stmt = sqlite_insert(PriceBar).values(payload)
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
    stmt = sqlite_insert(Fundamental).values(payload)
    excluded_cols = {
        c.name: stmt.excluded[c.name] for c in Fundamental.__table__.columns if c.name != "ticker"
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
    stmt = sqlite_insert(EquitySample).values(payload)
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
