"""APScheduler-based job runner. In-process, single-node.

Two job kinds:

* `run_pipeline_job` — fetches data, runs the pipeline, snapshots to DB.
  Used by both the daily cron and the on-demand `POST /jobs/refresh`.

* `cleanup_old_runs` — periodically prunes runs older than retention.
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

log = logging.getLogger(__name__)

# Module-level state for tracking in-flight jobs (single-node assumption).
_job_lock = threading.Lock()
_jobs: dict[str, dict] = {}


def _record_job(job_id: str, status: str, **extra) -> None:
    with _job_lock:
        _jobs[job_id] = {"status": status, "updated_at": datetime.now(timezone.utc), **extra}


def get_job_status(job_id: str) -> dict | None:
    with _job_lock:
        return _jobs.get(job_id)


def list_jobs(limit: int = 25) -> list[dict]:
    with _job_lock:
        items = sorted(
            [{"job_id": k, **v} for k, v in _jobs.items()],
            key=lambda x: x.get("updated_at") or datetime.min,
            reverse=True,
        )
        return items[:limit]


def run_pipeline_job(
    session_factory,
    *,
    pipeline_cfg: PipelineConfig | None = None,
    job_id: str | None = None,
) -> str:
    """Execute the pipeline and snapshot to DB. Synchronous; meant to be
    invoked from a background scheduler or an on-demand thread."""
    job_id = job_id or str(uuid.uuid4())
    pipeline_cfg = pipeline_cfg or PipelineConfig()
    _record_job(job_id, "running", config=dataclasses.asdict(pipeline_cfg))

    try:
        result = run_pipeline(pipeline_cfg)
        with session_scope(session_factory) as s:
            run = snapshot_run(
                s,
                result,
                config=dataclasses.asdict(pipeline_cfg),
                note=f"job {job_id}",
            )
        _record_job(job_id, "ok", run_id=run.id)
        log.info("job %s -> run %d ok", job_id, run.id)
        return job_id
    except Exception as e:  # noqa: BLE001
        log.exception("job %s failed", job_id)
        _record_job(job_id, "failed", error=str(e))
        # Also mark a failed Run row so dashboards can show the failure.
        try:
            with session_scope(session_factory) as s:
                run = store.create_run(
                    s, config=dataclasses.asdict(pipeline_cfg), note=f"job {job_id} (failed)"
                )
                store.fail_run(s, run, error=str(e))
        except Exception:  # noqa: BLE001
            pass
        return job_id


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


def make_scheduler(
    session_factory,
    *,
    pipeline_cfg: PipelineConfig | None = None,
    cron: str = "0 22 * * 1-5",  # 22:00 weekdays
    cleanup_cron: str = "0 3 * * 0",  # 03:00 Sundays
    timezone_name: str = "America/New_York",
) -> BackgroundScheduler:
    """Create and start a BackgroundScheduler with the configured jobs."""
    scheduler = BackgroundScheduler(timezone=timezone_name)
    scheduler.add_job(
        run_pipeline_job,
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
