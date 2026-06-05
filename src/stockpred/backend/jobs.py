"""APScheduler-based job runner. In-process, single-node.

Two job kinds:

* `run_pipeline_job` — fetches data, runs the pipeline, snapshots to DB.
  Used by both the daily cron and the on-demand `POST /jobs/refresh` /
  `POST /jobs/run/{queue_id}`.

* `cleanup_old_runs` — periodically prunes runs older than retention.

Log capture: every pipeline thread attaches a thread-local job_id.  A
module-level logging.Handler reads that value and appends log lines to
`_job_logs[job_id]`, capped at _MAX_LOG_LINES to bound memory.

Cancel: `request_cancel(job_id)` sets a flag.  The flag is checked after
the pipeline finishes — if set, the snapshot is skipped and the job is
marked "cancelled".  This is a soft-cancel (the thread still runs to
completion; we just discard the result).
"""

from __future__ import annotations

import dataclasses
import logging
import threading
import uuid
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from stockpred.backend import store
from stockpred.backend.db import session_scope
from stockpred.backend.snapshot import snapshot_run
from stockpred.pipeline import PipelineConfig, run_pipeline
from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# In-memory job registry
# ------------------------------------------------------------------ #

_job_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _record_job(job_id: str, status: str, **extra) -> None:
    with _job_lock:
        existing = _jobs.get(job_id, {})
        _jobs[job_id] = {
            **existing,
            "status": status,
            "updated_at": _now(),
            **extra,
        }


# ------------------------------------------------------------------ #
# Per-job log capture
# ------------------------------------------------------------------ #

_MAX_LOG_LINES = 2000
_job_logs: dict[str, list[str]] = {}
_log_lock = threading.Lock()
_current_job_id = threading.local()          # per-thread: .value = job_id | None
_log_handler_installed = False
_log_handler_install_lock = threading.Lock()


class _JobLogHandler(logging.Handler):
    """Captures log records emitted on the current job thread."""

    def emit(self, record: logging.LogRecord) -> None:
        job_id = getattr(_current_job_id, "value", None)
        if job_id is None:
            return
        try:
            msg = self.format(record)
        except Exception:  # noqa: BLE001
            return
        with _log_lock:
            bucket = _job_logs.get(job_id)
            if bucket is not None and len(bucket) < _MAX_LOG_LINES:
                bucket.append(msg)


def _ensure_log_handler() -> None:
    global _log_handler_installed
    with _log_handler_install_lock:
        if _log_handler_installed:
            return
        handler = _JobLogHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
        )
        handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(handler)
        _log_handler_installed = True


def get_job_logs(job_id: str) -> list[str]:
    with _log_lock:
        return list(_job_logs.get(job_id, []))


# ------------------------------------------------------------------ #
# Cancel flags
# ------------------------------------------------------------------ #

_cancel_flags: dict[str, bool] = {}


def request_cancel(job_id: str) -> bool:
    """Mark a running job for soft-cancel.  Returns True if the job existed."""
    with _job_lock:
        rec = _jobs.get(job_id)
        if rec is None or rec["status"] not in ("running", "queued"):
            return False
        _cancel_flags[job_id] = True
        _jobs[job_id] = {**rec, "status": "cancelling", "updated_at": _now()}
        return True


def _is_cancelled(job_id: str) -> bool:
    return _cancel_flags.get(job_id, False)


# ------------------------------------------------------------------ #
# Public helpers
# ------------------------------------------------------------------ #


def get_job_status(job_id: str, session_factory=None) -> dict | None:
    with _job_lock:
        rec = _jobs.get(job_id)
    if rec is not None:
        result = dict(rec)
        result["logs"] = get_job_logs(job_id)
        return result
    if session_factory is not None:
        try:
            with session_scope(session_factory) as s:
                db_rec = store.get_job_record(s, job_id)
                if db_rec is not None:
                    return {
                        "status": db_rec.status,
                        "started_at": db_rec.started_at,
                        "updated_at": db_rec.updated_at,
                        "elapsed_s": db_rec.elapsed_s,
                        "run_id": db_rec.run_id,
                        "error": db_rec.error,
                        "config": db_rec.config_json or {},
                        "logs": [],
                    }
        except Exception:  # noqa: BLE001
            pass
    return None


def list_jobs(limit: int = 25, session_factory=None) -> list[dict]:
    with _job_lock:
        memory_items = [{"job_id": k, **v} for k, v in _jobs.items()]
        memory_ids = {item["job_id"] for item in memory_items}

    db_items: list[dict] = []
    if session_factory is not None:
        try:
            with session_scope(session_factory) as s:
                for rec in store.list_job_records(s, limit=limit):
                    if rec.job_id not in memory_ids:
                        db_items.append({
                            "job_id": rec.job_id,
                            "status": rec.status,
                            "started_at": rec.started_at,
                            "updated_at": rec.updated_at,
                            "elapsed_s": rec.elapsed_s,
                            "run_id": rec.run_id,
                            "error": rec.error,
                            "config": rec.config_json or {},
                        })
        except Exception:  # noqa: BLE001
            pass

    combined = sorted(
        memory_items + db_items,
        key=lambda x: x.get("updated_at") or datetime.min,
        reverse=True,
    )
    return combined[:limit]


def is_any_job_running() -> bool:
    """True if any job is currently queued or running on this process."""
    with _job_lock:
        return any(v["status"] in ("queued", "running", "cancelling") for v in _jobs.values())


# ------------------------------------------------------------------ #
# Pipeline execution
# ------------------------------------------------------------------ #


def run_pipeline_job(
    session_factory,
    *,
    pipeline_cfg: PipelineConfig | PipelineV5Config | None = None,
    job_id: str | None = None,
) -> str:
    """Execute the pipeline and snapshot to DB.

    Synchronous — meant to be called from a background thread or APScheduler.
    """
    job_id = job_id or str(uuid.uuid4())
    pipeline_cfg = pipeline_cfg or PipelineConfig()

    # Set up log capture for this thread.
    _ensure_log_handler()
    with _log_lock:
        _job_logs[job_id] = []
    _current_job_id.value = job_id

    started_at = _now()
    phase = 5 if isinstance(pipeline_cfg, PipelineV5Config) else 1
    cfg_dict = {"phase": phase, **dataclasses.asdict(pipeline_cfg)}
    _record_job(job_id, "running", config=cfg_dict, started_at=started_at)

    # Persist running state so a crash/restart leaves a "crashed" record.
    try:
        with session_scope(session_factory) as s:
            store.upsert_job_record(
                s, job_id, "running",
                started_at=started_at, updated_at=started_at, config=cfg_dict,
            )
    except Exception:  # noqa: BLE001
        pass

    try:
        if isinstance(pipeline_cfg, PipelineV5Config):
            result = run_pipeline_v5(pipeline_cfg)
        else:
            result = run_pipeline(pipeline_cfg)

        elapsed = (_now() - started_at).total_seconds()

        if _is_cancelled(job_id):
            log.info("job %s cancelled after pipeline finished (result discarded)", job_id)
            _record_job(job_id, "cancelled", started_at=started_at, elapsed_s=elapsed)
            try:
                with session_scope(session_factory) as s:
                    store.upsert_job_record(
                        s, job_id, "cancelled",
                        started_at=started_at, updated_at=_now(),
                        elapsed_s=elapsed, config=cfg_dict,
                    )
            except Exception:  # noqa: BLE001
                pass
            return job_id

        with session_scope(session_factory) as s:
            run = snapshot_run(s, result, config=cfg_dict, note=f"job {job_id}")
            store.upsert_job_record(
                s, job_id, "ok",
                started_at=started_at, updated_at=_now(),
                elapsed_s=elapsed, run_id=run.id, config=cfg_dict,
            )
        _record_job(job_id, "ok", run_id=run.id, started_at=started_at, elapsed_s=elapsed)
        log.info("job %s → run %d ok (%.0fs)", job_id, run.id, elapsed)
        return job_id

    except Exception as e:  # noqa: BLE001
        elapsed = (_now() - started_at).total_seconds()
        log.exception("job %s failed", job_id)
        _record_job(job_id, "failed", error=str(e), started_at=started_at, elapsed_s=elapsed)
        try:
            with session_scope(session_factory) as s:
                run = store.create_run(s, config=cfg_dict, note=f"job {job_id} (failed)")
                store.fail_run(s, run, error=str(e))
                store.upsert_job_record(
                    s, job_id, "failed",
                    started_at=started_at, updated_at=_now(),
                    elapsed_s=elapsed, run_id=run.id, error=str(e), config=cfg_dict,
                )
        except Exception:  # noqa: BLE001
            pass
        return job_id

    finally:
        _current_job_id.value = None
        _cancel_flags.pop(job_id, None)


# ------------------------------------------------------------------ #
# Cleanup
# ------------------------------------------------------------------ #


def cleanup_old_runs(session_factory, *, keep_n: int = 50) -> None:
    """Delete all but the most recent `keep_n` runs."""
    from stockpred.backend.models import Run
    from sqlalchemy import delete, select

    with session_scope(session_factory) as s:
        ids_keep = [
            r.id
            for r in s.execute(select(Run).order_by(Run.started_at.desc()).limit(keep_n))
            .scalars()
            .all()
        ]
        if ids_keep:
            s.execute(delete(Run).where(Run.id.notin_(ids_keep)))
            log.info("cleanup: kept top %d runs", len(ids_keep))


# ------------------------------------------------------------------ #
# Scheduler
# ------------------------------------------------------------------ #


def _run_daily_if_idle(session_factory, pipeline_cfg=None) -> None:
    """Daily cron wrapper: skip if another pipeline is already running."""
    if is_any_job_running():
        log.info("daily cron: skipping — a pipeline job is already in flight")
        return
    run_pipeline_job(session_factory, pipeline_cfg=pipeline_cfg)


def make_scheduler(
    session_factory,
    *,
    pipeline_cfg: PipelineConfig | None = None,
    cron: str = "0 0 * * *",          # midnight UTC every day
    cleanup_cron: str = "0 3 * * 0",  # 03:00 UTC Sundays
    timezone_name: str = "UTC",
) -> BackgroundScheduler:
    """Create a BackgroundScheduler with the daily pipeline + weekly cleanup jobs."""
    scheduler = BackgroundScheduler(timezone=timezone_name)
    scheduler.add_job(
        _run_daily_if_idle,
        trigger=CronTrigger.from_crontab(cron, timezone=timezone_name),
        kwargs={"session_factory": session_factory, "pipeline_cfg": pipeline_cfg},
        id="daily_pipeline",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        cleanup_old_runs,
        trigger=CronTrigger.from_crontab(cleanup_cron, timezone=timezone_name),
        kwargs={"session_factory": session_factory},
        id="weekly_cleanup",
        replace_existing=True,
    )
    return scheduler
