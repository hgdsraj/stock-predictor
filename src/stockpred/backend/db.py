"""SQLAlchemy engine + declarative base. SQLite by default; portable, no infra."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from stockpred.config import DATA_DIR

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = DATA_DIR / "app.db"


# Lightweight idempotent migrations applied on every startup (after
# create_all). We keep this here instead of pulling in Alembic since the schema
# is small and SQLite-only. Each entry: (table_name, column_name, DDL fragment).
#
# Rules:
#   - Never include `NOT NULL` without a `DEFAULT` (SQLite rejects it for
#     existing rows).
#   - Adding a column is always safe and idempotent (we check `PRAGMA
#     table_info` first).
#   - Schema removals/renames must be done manually with a release note.
_LIGHTWEIGHT_MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    # 2026-06: model-run history feature.
    ("runs", "is_active", "BOOLEAN NOT NULL DEFAULT 0"),
    # 2026-06: persist pipeline logs across server restarts.
    ("job_records", "logs_json", "JSON"),
)


def apply_lightweight_migrations(engine: Engine) -> int:
    """Add any missing columns from _LIGHTWEIGHT_MIGRATIONS. Idempotent.

    Returns the number of columns added (0 = schema already up to date).
    Logs each addition. Failures on a single column are logged and skipped;
    they do not abort startup, since the table may not exist yet on a brand
    new DB (`create_all` runs first so this is only theoretical defence).
    """
    insp = inspect(engine)
    added = 0
    for table, column, ddl in _LIGHTWEIGHT_MIGRATIONS:
        try:
            cols = {c["name"] for c in insp.get_columns(table)}
        except Exception as e:  # noqa: BLE001 — table may not exist
            log.warning("migration skip %s.%s: cannot inspect (%s)", table, column, e)
            continue
        if column in cols:
            continue
        try:
            with engine.begin() as conn:
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))
            log.info("migration: added %s.%s (%s)", table, column, ddl)
            added += 1
        except Exception as e:  # noqa: BLE001
            log.error("migration FAILED for %s.%s: %s", table, column, e)
    return added


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
    """Create every table defined on Base, then run lightweight migrations.

    Idempotent. Safe on both fresh and existing databases.
    """
    Base.metadata.create_all(engine)
    apply_lightweight_migrations(engine)
    log.info("DB ready at %s", engine.url)


