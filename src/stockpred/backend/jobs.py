"""APScheduler-based job runner. In-process, single-node.

Three job kinds:

* `run_pipeline_job`    — fetches data, runs the pipeline, snapshots to DB.
  Used by both the daily cron and the on-demand `POST /jobs/refresh` /
  `POST /jobs/run/{queue_id}`.

* `run_hypersearch_job` — runs an Optuna hyperparameter search, persisting
  trial results to `HypersearchRun` after each trial. Launched the same way
  as pipeline jobs via `POST /jobs/run/{queue_id}` when the queued config
  includes `job_type = "hypersearch"`.

* `cleanup_old_runs`    — periodically prunes runs older than retention.

Log capture: every pipeline thread attaches a thread-local job_id.  A
module-level logging.Handler reads that value and appends log lines to
`_job_logs[job_id]`, capped at _MAX_LOG_LINES to bound memory.

Cancel: `request_cancel(job_id)` sets a flag.  The flag is checked after
the pipeline finishes — if set, the snapshot is skipped and the job is
marked "cancelled".  This is a soft-cancel (the thread still runs to
completion; we just discard the result).
"""

from __future__ import annotations

import ctypes
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
# Thread interrupt for hard-cancel
# ------------------------------------------------------------------ #


class _CancelInterrupt(BaseException):
    """Raised inside the pipeline thread when cancel is requested.
    Inherits from BaseException so it propagates through most try/except
    blocks in dependencies (lightgbm, pandas, etc.)."""


def _raise_in_thread(ident: int) -> None:
    """Inject _CancelInterrupt into thread `ident` at the next Python bytecode."""
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(ident),
        ctypes.py_object(_CancelInterrupt),
    )
    if res > 1:  # shouldn't happen; undo to be safe
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(ident), None)


# ------------------------------------------------------------------ #
# In-memory job registry
# ------------------------------------------------------------------ #

_job_lock = threading.Lock()
_jobs: dict[str, dict] = {}

# Thread references for hard-cancel
_job_threads: dict[str, threading.Thread] = {}
_thread_lock = threading.Lock()


def register_job_thread(job_id: str, thread: threading.Thread) -> None:
    with _thread_lock:
        _job_threads[job_id] = thread


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
        logging.getLogger("stockpred").setLevel(logging.INFO)
        _log_handler_installed = True


def get_job_logs(job_id: str) -> list[str]:
    with _log_lock:
        return list(_job_logs.get(job_id, []))


# ------------------------------------------------------------------ #
# Cancel flags
# ------------------------------------------------------------------ #

_cancel_flags: dict[str, bool] = {}


def request_cancel(job_id: str) -> bool:
    """Hard-cancel a running job by injecting _CancelInterrupt into its thread.
    Returns True if the job existed and was active."""
    with _job_lock:
        rec = _jobs.get(job_id)
        if rec is None or rec["status"] not in ("running", "queued"):
            return False
        _cancel_flags[job_id] = True
        _jobs[job_id] = {**rec, "status": "cancelling", "updated_at": _now()}

    # Fire the hard interrupt — raises at the next Python bytecode in the thread.
    # If the pipeline is deep in C code (e.g. LightGBM training) this won't take
    # effect until the C extension yields back to Python, which may still take
    # seconds to minutes per fold. But it's far faster than waiting for the whole run.
    with _thread_lock:
        t = _job_threads.get(job_id)
    if t is not None and t.is_alive() and t.ident is not None:
        try:
            _raise_in_thread(t.ident)
        except Exception:  # noqa: BLE001
            pass
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
                        "logs": db_rec.logs_json or [],
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

    def _sort_key(x):
        v = x.get("updated_at") or datetime.min
        # Normalise to naive UTC so timezone-aware (in-memory) and
        # timezone-naive (DB) datetimes can be compared together.
        if getattr(v, "tzinfo", None) is not None:
            v = v.replace(tzinfo=None)
        return v

    combined = sorted(memory_items + db_items, key=_sort_key, reverse=True)
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

    def _current_logs() -> list[str]:
        return get_job_logs(job_id)

    def _persist_cancelled(elapsed: float) -> None:
        _record_job(job_id, "cancelled", started_at=started_at, elapsed_s=elapsed)
        try:
            with session_scope(session_factory) as s:
                store.upsert_job_record(
                    s, job_id, "cancelled",
                    started_at=started_at, updated_at=_now(),
                    elapsed_s=elapsed, config=cfg_dict, logs=_current_logs(),
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
            _persist_cancelled(elapsed)
            return job_id

        with session_scope(session_factory) as s:
            run = snapshot_run(s, result, config=cfg_dict, note=f"job {job_id}")
            store.upsert_job_record(
                s, job_id, "ok",
                started_at=started_at, updated_at=_now(),
                elapsed_s=elapsed, run_id=run.id, config=cfg_dict, logs=_current_logs(),
            )
        _record_job(job_id, "ok", run_id=run.id, started_at=started_at, elapsed_s=elapsed)
        log.info("job %s → run %d ok (%.0fs)", job_id, run.id, elapsed)
        return job_id

    except _CancelInterrupt:
        elapsed = (_now() - started_at).total_seconds()
        log.info("job %s hard-cancelled via thread interrupt (%.0fs)", job_id, elapsed)
        _persist_cancelled(elapsed)
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
                    elapsed_s=elapsed, run_id=run.id, error=str(e),
                    config=cfg_dict, logs=_current_logs(),
                )
        except Exception:  # noqa: BLE001
            pass
        return job_id

    finally:
        _current_job_id.value = None
        _cancel_flags.pop(job_id, None)
        with _thread_lock:
            _job_threads.pop(job_id, None)


# ------------------------------------------------------------------ #
# Hypersearch execution
# ------------------------------------------------------------------ #


def run_hypersearch_job(
    session_factory,
    *,
    hypersearch_cfg,        # stockpred.hypersearch.HypersearchConfig
    job_id: str | None = None,
) -> str:
    """Execute a hyperparameter search, persisting trial results to DB.

    Shares the same log-capture, cancel, and thread-registry infrastructure
    as `run_pipeline_job`. The `HypersearchRun` row is updated after every
    completed trial so partial results survive a cancel.
    """
    from stockpred.hypersearch import run_hypersearch

    job_id = job_id or str(uuid.uuid4())

    _ensure_log_handler()
    with _log_lock:
        _job_logs[job_id] = []
    _current_job_id.value = job_id

    started_at = _now()
    cfg_dict = {
        "job_type": "hypersearch",
        **dataclasses.asdict(hypersearch_cfg),
    }
    _record_job(job_id, "running", config=cfg_dict, started_at=started_at)

    try:
        with session_scope(session_factory) as s:
            store.upsert_job_record(
                s, job_id, "running",
                started_at=started_at, updated_at=started_at, config=cfg_dict,
            )
    except Exception:  # noqa: BLE001
        pass

    # Create the HypersearchRun row immediately so the UI can see it.
    hs_run_id: int | None = None
    try:
        with session_scope(session_factory) as s:
            hs_run = store.create_hypersearch_run(
                s,
                job_id=job_id,
                config=cfg_dict,
                n_trials=hypersearch_cfg.n_trials,
            )
            hs_run_id = hs_run.id
    except Exception:  # noqa: BLE001
        pass

    def _current_logs() -> list[str]:
        return get_job_logs(job_id)

    # Accumulate trial rows so we can pass the full list to store on each update.
    _trial_rows: list[dict] = []

    def _on_trial_done(row: dict) -> None:
        _trial_rows.append(row)
        if hs_run_id is None:
            return
        # best so far — filter out penalty values
        valid = [r for r in _trial_rows if (r.get("hold_sharpe") or -99) > -9]
        best_s: float | None = None
        best_p: dict | None = None
        if valid:
            best_row = max(valid, key=lambda r: r.get("hold_sharpe") or float("-inf"))
            best_s = best_row.get("hold_sharpe")
            best_p = {
                k: best_row[k] for k in best_row
                if k not in ("trial", "value", "hold_sharpe", "hold_ci_lo",
                             "hold_ci_hi", "hold_dd", "hold_hit", "hold_ann_return",
                             "dev_sharpe", "elapsed_s", "error")
            }
        try:
            with session_scope(session_factory) as s:
                store.update_hypersearch_run(
                    s, hs_run_id,
                    n_trials_done=len(_trial_rows),
                    best_sharpe=best_s,
                    best_params=best_p,
                    trials=list(_trial_rows),
                )
        except Exception:  # noqa: BLE001
            pass

    def _should_stop() -> bool:
        return _is_cancelled(job_id)

    def _persist_end(status: str, elapsed: float, error: str | None = None) -> None:
        _record_job(job_id, status, started_at=started_at, elapsed_s=elapsed, error=error)
        try:
            with session_scope(session_factory) as s:
                store.upsert_job_record(
                    s, job_id, status,
                    started_at=started_at, updated_at=_now(),
                    elapsed_s=elapsed, error=error,
                    config=cfg_dict, logs=_current_logs(),
                )
                if hs_run_id is not None:
                    store.finalize_hypersearch_run(
                        s, hs_run_id,
                        status=status,
                        trials=list(_trial_rows),
                    )
        except Exception:  # noqa: BLE001
            pass

    try:
        run_hypersearch(
            hypersearch_cfg,
            on_trial_done=_on_trial_done,
            should_stop=_should_stop,
        )
        elapsed = (_now() - started_at).total_seconds()

        if _is_cancelled(job_id):
            log.info("hypersearch job %s cancelled", job_id)
            _persist_end("cancelled", elapsed)
            return job_id

        # Finalise with best params from accumulated rows.
        valid = [r for r in _trial_rows if (r.get("hold_sharpe") or -99) > -9]
        best_s = None
        best_p = None
        if valid:
            best_row = max(valid, key=lambda r: r.get("hold_sharpe") or float("-inf"))
            best_s = best_row.get("hold_sharpe")
            best_p = {
                k: best_row[k] for k in best_row
                if k not in ("trial", "value", "hold_sharpe", "hold_ci_lo",
                             "hold_ci_hi", "hold_dd", "hold_hit", "hold_ann_return",
                             "dev_sharpe", "elapsed_s", "error")
            }

        _record_job(job_id, "ok", started_at=started_at, elapsed_s=elapsed)
        try:
            with session_scope(session_factory) as s:
                store.upsert_job_record(
                    s, job_id, "ok",
                    started_at=started_at, updated_at=_now(),
                    elapsed_s=elapsed, config=cfg_dict, logs=_current_logs(),
                )
                if hs_run_id is not None:
                    store.finalize_hypersearch_run(
                        s, hs_run_id,
                        status="ok",
                        best_sharpe=best_s,
                        best_params=best_p,
                        trials=list(_trial_rows),
                    )
        except Exception:  # noqa: BLE001
            pass

        log.info("hypersearch job %s ok (%.0fs, %d trials)", job_id, elapsed, len(_trial_rows))
        return job_id

    except _CancelInterrupt:
        elapsed = (_now() - started_at).total_seconds()
        log.info("hypersearch job %s hard-cancelled (%.0fs)", job_id, elapsed)
        _persist_end("cancelled", elapsed)
        return job_id

    except Exception as e:  # noqa: BLE001
        elapsed = (_now() - started_at).total_seconds()
        log.exception("hypersearch job %s failed", job_id)
        _persist_end("failed", elapsed, error=str(e))
        return job_id

    finally:
        _current_job_id.value = None
        _cancel_flags.pop(job_id, None)
        with _thread_lock:
            _job_threads.pop(job_id, None)


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
