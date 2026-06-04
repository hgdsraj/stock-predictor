"""Tests for the job queue endpoints and the password-protected launch/cancel/delete.

Covers:
  POST /jobs/queue       — create a queued job (no auth)
  GET  /jobs/queue       — list queued jobs
  POST /jobs/run/{id}    — launch a queued job (requires X-Password)
  DELETE /jobs/queue/{id} — delete a queued job (requires X-Password)
  DELETE /jobs/{id}/cancel — cancel a running job (requires X-Password)
  GET  /jobs             — list in-memory jobs (JobDetail list)
  GET  /jobs/{id}        — get detail for a single job
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    """Fresh app with isolated DB, no scheduler, API key and password set."""
    db_path = tmp_path / "app.db"
    monkeypatch.setenv("STOCKPRED_DB", str(db_path))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    monkeypatch.setenv("STOCKPRED_API_KEY", "test-api-key")
    monkeypatch.setenv("STOCKPRED_PW", "test-pw")
    monkeypatch.setenv("STOCKPRED_CORS", "*")

    import stockpred.backend.api as api_mod

    importlib.reload(api_mod)
    with TestClient(api_mod.app) as client:
        yield client, api_mod


# ── POST /jobs/queue ─────────────────────────────────────────────────────────


def test_queue_job_no_auth_required(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 1})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "pending"
    assert "id" in body
    assert body["config"]["phase"] == 1


def test_queue_job_phase5(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 5, "n_tickers": 50})
    assert r.status_code == 200
    body = r.json()
    assert body["config"]["phase"] == 5
    assert body["config"]["n_tickers"] == 50


def test_queue_job_max_5_pending(app_client):
    client, _ = app_client
    for _ in range(5):
        r = client.post("/jobs/queue", json={"phase": 1})
        assert r.status_code == 200
    # 6th should fail
    r = client.post("/jobs/queue", json={"phase": 1})
    assert r.status_code == 429


# ── GET /jobs/queue ──────────────────────────────────────────────────────────


def test_list_queued_empty(app_client):
    client, _ = app_client
    r = client.get("/jobs/queue")
    assert r.status_code == 200
    assert r.json() == []


def test_list_queued_after_creating(app_client):
    client, _ = app_client
    client.post("/jobs/queue", json={"phase": 1})
    client.post("/jobs/queue", json={"phase": 5})
    r = client.get("/jobs/queue")
    assert r.status_code == 200
    items = r.json()
    assert len(items) == 2
    phases = {item["config"]["phase"] for item in items}
    assert phases == {1, 5}


# ── DELETE /jobs/queue/{id} ──────────────────────────────────────────────────


def test_delete_queued_requires_password(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 1})
    qid = r.json()["id"]
    # No password → 401
    r = client.delete(f"/jobs/queue/{qid}")
    assert r.status_code == 401


def test_delete_queued_wrong_password(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 1})
    qid = r.json()["id"]
    r = client.delete(f"/jobs/queue/{qid}", headers={"X-Password": "wrong"})
    assert r.status_code == 401


def test_delete_queued_success(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 1})
    qid = r.json()["id"]
    r = client.delete(f"/jobs/queue/{qid}", headers={"X-Password": "test-pw"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    # Should now be gone
    queued = client.get("/jobs/queue").json()
    assert all(j["id"] != qid for j in queued)


def test_delete_queued_not_found(app_client):
    client, _ = app_client
    r = client.delete("/jobs/queue/nonexistent-id", headers={"X-Password": "test-pw"})
    assert r.status_code == 404


# ── POST /jobs/run/{id} ──────────────────────────────────────────────────────


def test_run_queued_requires_password(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 1, "n_tickers": 5})
    qid = r.json()["id"]
    r = client.post(f"/jobs/run/{qid}")
    assert r.status_code == 401


def test_run_queued_wrong_password(app_client):
    client, _ = app_client
    r = client.post("/jobs/queue", json={"phase": 1})
    qid = r.json()["id"]
    r = client.post(f"/jobs/run/{qid}", headers={"X-Password": "bad"})
    assert r.status_code == 401


def test_run_queued_not_found(app_client):
    client, _ = app_client
    r = client.post("/jobs/run/does-not-exist", headers={"X-Password": "test-pw"})
    assert r.status_code == 404


def test_run_queued_returns_job_id(app_client, monkeypatch):
    """Launch a queued job; the endpoint should return a job_id immediately.
    We monkeypatch run_pipeline_job so we don't actually run the full pipeline."""
    client, api_mod = app_client

    import stockpred.backend.jobs as jobs_mod

    # Patch the pipeline runner to be a no-op so the test doesn't hit data layer.
    def _noop(session_factory, *, pipeline_cfg=None, job_id=None):
        jid = job_id or "fake"
        jobs_mod._record_job(jid, "ok", elapsed_s=0.1)
        return jid

    monkeypatch.setattr(jobs_mod, "run_pipeline_job", _noop)

    r = client.post("/jobs/queue", json={"phase": 1})
    qid = r.json()["id"]

    r = client.post(f"/jobs/run/{qid}", headers={"X-Password": "test-pw"})
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body
    assert body["status"] == "queued"

    # The queued entry should now be "launched"
    queued = client.get("/jobs/queue").json()
    entry = next(j for j in queued if j["id"] == qid)
    assert entry["status"] == "launched"
    assert entry["job_id"] == body["job_id"]


def test_run_already_launched_is_409(app_client, monkeypatch):
    client, api_mod = app_client
    import stockpred.backend.jobs as jobs_mod

    def _noop(session_factory, *, pipeline_cfg=None, job_id=None):
        jid = job_id or "fake"
        jobs_mod._record_job(jid, "ok", elapsed_s=0.1)
        return jid

    monkeypatch.setattr(jobs_mod, "run_pipeline_job", _noop)

    r = client.post("/jobs/queue", json={"phase": 1})
    qid = r.json()["id"]

    client.post(f"/jobs/run/{qid}", headers={"X-Password": "test-pw"})
    # Launching again should fail
    r = client.post(f"/jobs/run/{qid}", headers={"X-Password": "test-pw"})
    assert r.status_code == 409


# ── DELETE /jobs/{id}/cancel ─────────────────────────────────────────────────


def test_cancel_requires_password(app_client, monkeypatch):
    client, api_mod = app_client
    import stockpred.backend.jobs as jobs_mod

    # Record a fake running job
    jobs_mod._record_job("fake-job-1", "running")

    r = client.delete("/jobs/fake-job-1/cancel")
    assert r.status_code == 401


def test_cancel_running_job(app_client, monkeypatch):
    client, _ = app_client
    import stockpred.backend.jobs as jobs_mod

    jobs_mod._record_job("fake-job-2", "running")

    r = client.delete("/jobs/fake-job-2/cancel", headers={"X-Password": "test-pw"})
    assert r.status_code == 200
    assert r.json()["ok"] is True

    # Status should now be cancelling
    detail = client.get("/jobs/fake-job-2").json()
    assert detail["status"] == "cancelling"


def test_cancel_nonexistent_job(app_client):
    client, _ = app_client
    r = client.delete("/jobs/does-not-exist/cancel", headers={"X-Password": "test-pw"})
    assert r.status_code == 404


# ── GET /jobs and GET /jobs/{id} ─────────────────────────────────────────────


def test_jobs_list_empty(app_client):
    client, _ = app_client
    import stockpred.backend.jobs as jobs_mod
    # Clear in-memory state
    with jobs_mod._job_lock:
        jobs_mod._jobs.clear()

    r = client.get("/jobs")
    assert r.status_code == 200
    assert r.json() == []


def test_jobs_detail_has_logs_field(app_client):
    client, _ = app_client
    import stockpred.backend.jobs as jobs_mod

    jobs_mod._record_job("log-job", "ok", elapsed_s=1.0)
    with jobs_mod._log_lock:
        jobs_mod._job_logs["log-job"] = ["line one", "line two"]

    r = client.get("/jobs/log-job")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["logs"] == ["line one", "line two"]
    assert body["elapsed_s"] == 1.0


def test_jobs_list_omits_logs(app_client):
    """List endpoint should not include logs to keep payload small."""
    client, _ = app_client
    import stockpred.backend.jobs as jobs_mod

    jobs_mod._record_job("list-job", "ok")
    with jobs_mod._log_lock:
        jobs_mod._job_logs["list-job"] = ["lots", "of", "lines"]

    r = client.get("/jobs")
    assert r.status_code == 200
    items = r.json()
    target = next((j for j in items if j["job_id"] == "list-job"), None)
    assert target is not None
    assert target["logs"] == []  # omitted in list


def test_job_detail_not_found(app_client):
    client, _ = app_client
    r = client.get("/jobs/no-such-job")
    assert r.status_code == 404


# ── Password not configured ──────────────────────────────────────────────────


def test_password_not_configured_returns_403(tmp_path, monkeypatch):
    """When STOCKPRED_PW is unset, all password-gated endpoints return 403."""
    db_path = tmp_path / "app.db"
    monkeypatch.setenv("STOCKPRED_DB", str(db_path))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    monkeypatch.delenv("STOCKPRED_PW", raising=False)
    monkeypatch.setenv("STOCKPRED_CORS", "*")

    import stockpred.backend.api as api_mod

    importlib.reload(api_mod)
    with TestClient(api_mod.app) as client:
        # Create a queued job first (no auth needed)
        r = client.post("/jobs/queue", json={"phase": 1})
        assert r.status_code == 200
        qid = r.json()["id"]

        # Launch → 403 because PW not configured
        r = client.post(f"/jobs/run/{qid}", headers={"X-Password": "anything"})
        assert r.status_code == 403

        # Delete → 403
        r = client.delete(f"/jobs/queue/{qid}", headers={"X-Password": "anything"})
        assert r.status_code == 403
