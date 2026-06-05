"""SQLAlchemy engine + declarative base. SQLite by default; portable, no infra."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from stockpred.config import DATA_DIR

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = DATA_DIR / "app.db"


class Base(DeclarativeBase):
    """Common declarative base for all ORM models."""


def make_engine(db_path: Path | str | None = None, *, echo: bool = False) -> Engine:
    """Create a SQLite engine with sensible defaults (WAL, foreign keys ON)."""
    if db_path is None:
        db_path = DEFAULT_DB_PATH
    if isinstance(db_path, str) and db_path.startswith("sqlite"):
        url = db_path
    else:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        url = f"sqlite:///{db_path}"
    engine = create_engine(url, echo=echo, future=True)

    @event.listens_for(engine, "connect")
    def _pragmas(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(SessionLocal: sessionmaker[Session]) -> Iterator[Session]:
    """Transactional scope: commit on success, rollback on exception."""
    s: Session = SessionLocal()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def create_all(engine: Engine) -> None:
    """Create every table defined on Base. Idempotent.
    Also applies lightweight column migrations for tables that already exist."""
    Base.metadata.create_all(engine)
    _migrate_columns(engine)
    log.info("DB ready at %s", engine.url)


def _migrate_columns(engine: Engine) -> None:
    """Add any columns that exist in the ORM models but not yet in the DB.
    SQLite doesn't support ALTER TABLE DROP/MODIFY, only ADD COLUMN, so this
    is safe to run on every startup."""
    _add_column_if_missing = []
    # job_records.logs_json — added when log persistence was introduced
    _add_column_if_missing.append(
        ("job_records", "logs_json", "JSON")
    )

    with engine.connect() as conn:
        for table, col, col_type in _add_column_if_missing:
            try:
                existing = [
                    row[1]
                    for row in conn.execute(
                        __import__("sqlalchemy").text(f"PRAGMA table_info({table})")
                    )
                ]
                if col not in existing:
                    conn.execute(
                        __import__("sqlalchemy").text(
                            f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"
                        )
                    )
                    conn.commit()
                    log.info("migration: added column %s.%s", table, col)
            except Exception as exc:  # noqa: BLE001
                log.warning("migration: could not add %s.%s: %s", table, col, exc)
