"""Backend / FastAPI contract tests.

Run against an isolated temp-file SQLite DB so we don't touch the real one.
Scheduler is disabled to keep test invocations deterministic.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    """Spin up a fresh FastAPI app with an isolated SQLite DB and no scheduler.

    Sets STOCKPRED_API_KEY=test so refresh endpoint can be exercised.
    Sets STOCKPRED_CORS='*' to widen for tests that don't care about origin.
    """
    db_path = tmp_path / "app.db"
    monkeypatch.setenv("STOCKPRED_DB", str(db_path))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    monkeypatch.setenv("STOCKPRED_API_KEY", "test")
    monkeypatch.setenv("STOCKPRED_CORS", "*")
    # Ensure fresh module state (api uses module-level AppState + envs).
    import importlib

    import stockpred.backend.api as api_mod

    importlib.reload(api_mod)
    with TestClient(api_mod.app) as client:
        yield client, api_mod


def test_healthz_reports_db_and_scheduler(app_client):
    client, _ = app_client
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["db"] == "ok"
    assert body["scheduler"] in ("off", "running")


def test_tickers_empty_initially(app_client):
    client, _ = app_client
    r = client.get("/tickers")
    assert r.status_code == 200
    assert r.json() == []


def test_runs_empty_initially(app_client):
    client, _ = app_client
    r = client.get("/runs")
    assert r.status_code == 200
    assert r.json() == []


def test_latest_predictions_empty(app_client):
    client, _ = app_client
    r = client.get("/predictions/latest")
    assert r.status_code == 200
    body = r.json()
    assert body["long"] == []
    assert body["short"] == []
    assert body["date"] is None


def test_backtest_summary_404_when_no_runs(app_client):
    client, _ = app_client
    r = client.get("/backtest/summary")
    assert r.status_code == 404


def test_ticker_details_404_unknown(app_client):
    client, _ = app_client
    r = client.get("/tickers/NOPE/details")
    assert r.status_code == 404


def test_full_snapshot_round_trip(app_client, tmp_path):
    """Simulate a pipeline result -> snapshot_run -> read it back via API."""
    import numpy as np
    import pandas as pd

    from stockpred.backend import store
    from stockpred.backend.db import session_scope
    from stockpred.backend.models import Fundamental, PriceBar
    from stockpred.backend.snapshot import snapshot_run

    client, api_mod = app_client
    SessionLocal = api_mod.AppState.SessionLocal

    # Build a tiny synthetic pipeline result.
    dates = pd.bdate_range("2024-01-01", periods=10)
    tickers = ["AAA", "BBB", "CCC"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(np.random.RandomState(0).randn(len(idx)), index=idx)
    # Weights: long top, short bottom on each date.
    weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    for d in dates:
        s_day = score.loc[d]
        top = s_day.idxmax()
        bot = s_day.idxmin()
        weights.loc[d, top] = 0.5
        weights.loc[d, bot] = -0.5

    # Fake a BacktestResult-like object.
    from stockpred.backtest.engine import BacktestResult

    daily_returns = pd.Series(
        np.random.RandomState(1).randn(len(dates)) * 0.005, index=dates, name="strategy_return"
    )
    bt = BacktestResult(
        returns=daily_returns,
        gross_returns=daily_returns,
        turnover=pd.Series(0.1, index=dates),
        held_weights=weights,
        target_weights=weights,
    )

    pipeline_result = {
        "tickers": tickers,
        "feature_matrix_shape": (30, 5),
        "per_horizon_predictions": {1: score},
        "per_horizon_diagnostics": {
            1: {"hit_rate": 0.52, "ic_mean": 0.01, "ic_std": 0.1, "ic_ir": 0.5}
        },
        "ensemble_score": score,
        "weights": weights,
        "backtest": bt,
        "metrics": {"ann_return": 0.05, "ann_vol": 0.10, "sharpe": 0.5, "max_drawdown": -0.1},
        "tearsheet_path": str(tmp_path / "ts.html"),
        "elapsed_s": 1.2,
    }

    # Seed prices + fundamentals directly (avoid hitting yfinance).
    with session_scope(SessionLocal) as s:
        store.upsert_prices(
            s,
            [
                {
                    "ticker": t,
                    "date": d.date(),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "adj_close": 100.5,
                    "volume": 1_000_000,
                }
                for t in tickers
                for d in dates
            ],
        )
        store.upsert_fundamentals(
            s,
            [{"ticker": t, "sector": "Tech", "industry": "X", "market_cap": 1e9} for t in tickers],
        )
        # Run snapshot WITHOUT touching yfinance.
        snapshot_run(
            s,
            pipeline_result,
            config={"horizons": [1]},
            persist_prices=False,
            persist_fundamentals=False,
        )

    # Now hit the API.
    r = client.get("/tickers")
    assert r.status_code == 200
    tickers_out = r.json()
    assert len(tickers_out) == 3
    assert {t["ticker"] for t in tickers_out} == set(tickers)

    r = client.get("/runs")
    runs = r.json()
    assert len(runs) >= 1
    run_id = runs[0]["id"]
    assert runs[0]["status"] == "ok"

    r = client.get(f"/runs/{run_id}/equity")
    eq = r.json()
    assert len(eq) > 0
    assert "cumulative_return" in eq[0]

    r = client.get("/predictions/latest")
    movers = r.json()
    assert movers["date"] is not None
    assert len(movers["long"]) >= 1
    assert len(movers["short"]) >= 1

    # Request a wide enough window to include the seeded synthetic dates.
    r = client.get("/tickers/AAA/details", params={"days": 2000})
    detail = r.json()
    assert detail["ticker"] == "AAA"
    assert len(detail["prices"]) > 0
    assert detail["sector"] == "Tech"

    r = client.get("/backtest/summary")
    summary = r.json()
    assert "run" in summary
    assert "equity_curve" in summary
    assert summary["run"]["metrics"]["sharpe"] == 0.5


# --------------------------- security regressions ---------------------------


def test_spa_fallback_rejects_path_traversal(app_client, tmp_path):
    """Review finding C3: an attacker requesting /..%2F..%2Fetc%2Fpasswd must
    NOT receive arbitrary file contents — they should get the SPA index."""
    client, api_mod = app_client
    # Create a synthetic SPA dist with a single index.html so the fallback fires.
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<html>spa</html>")
    # Re-init the app with this dist path.
    import importlib
    import os

    os.environ["STOCKPRED_WEB_DIST"] = str(dist)
    importlib.reload(api_mod)
    with TestClient(api_mod.app) as c:
        # First, the index works.
        r = c.get("/")
        assert r.status_code == 200
        assert "spa" in r.text

        # Now an attempted traversal must fall through to the SPA index, not
        # serve /etc/passwd or any other file outside the dist.
        for attack in (
            "../../../etc/passwd",
            "..%2F..%2F..%2Fetc%2Fpasswd",
            "..%5C..%5Cwindows%5Cwin.ini",
            "%2e%2e/%2e%2e/etc/passwd",
        ):
            r = c.get(f"/{attack}")
            # SPA fallback should serve index.html, not error out and not leak files.
            assert r.status_code == 200
            assert "root:" not in r.text  # /etc/passwd canary
            assert "spa" in r.text


def test_refresh_endpoint_requires_api_key(app_client):
    """Review finding C1: POST /jobs/refresh must be gated."""
    client, _ = app_client
    # Without header: 401 because API key configured but not provided.
    r = client.post("/jobs/refresh")
    assert r.status_code == 401

    # Wrong key: 401.
    r = client.post("/jobs/refresh", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_refresh_endpoint_disabled_when_no_api_key(tmp_path, monkeypatch):
    """If STOCKPRED_API_KEY is unset, write endpoints return 403 (not 500)."""
    monkeypatch.setenv("STOCKPRED_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    monkeypatch.delenv("STOCKPRED_API_KEY", raising=False)
    import importlib

    import stockpred.backend.api as api_mod

    importlib.reload(api_mod)
    with TestClient(api_mod.app) as c:
        r = c.post("/jobs/refresh")
        assert r.status_code == 403
        assert "STOCKPRED_API_KEY" in r.text


def test_nan_metrics_serialise_as_null_not_NaN(app_client):
    """Review finding H1: NaN/Inf in summary metrics must produce valid JSON
    (null), not the non-RFC NaN token that breaks browser JSON parsers."""
    import json as _json

    from stockpred.backend import store
    from stockpred.backend.db import session_scope

    client, api_mod = app_client
    SessionLocal = api_mod.AppState.SessionLocal

    with session_scope(SessionLocal) as s:
        run = store.create_run(s, config={}, note="nan-test")
        store.complete_run(
            s,
            run,
            summary={
                "metrics": {
                    "sharpe": float("nan"),
                    "max_drawdown": float("-inf"),
                    "ann_return": 0.1,
                }
            },
        )

    r = client.get("/backtest/summary")
    assert r.status_code == 200
    raw = r.text
    # The literal "NaN" / "Infinity" tokens are not RFC-7159 valid JSON; their
    # presence would crash browsers. We coerce to null in SafeJSONResponse.
    assert "NaN" not in raw
    assert "Infinity" not in raw
    # Body must still be standard-parseable.
    body = _json.loads(raw)
    assert body["run"]["metrics"]["sharpe"] is None
    assert body["run"]["metrics"]["max_drawdown"] is None
    assert body["run"]["metrics"]["ann_return"] == 0.1


def test_concurrent_refresh_returns_409(app_client):
    """Review finding H2: a second refresh while one is in-flight must 409."""
    client, api_mod = app_client
    # Manually take the lock to simulate an in-flight job.
    locked = api_mod._refresh_lock.acquire()
    try:
        assert locked
        r = client.post("/jobs/refresh", headers={"X-API-Key": "test"})
        assert r.status_code == 409
    finally:
        api_mod._refresh_lock.release()


def test_refresh_request_includes_phase_8_through_13_flags():
    """REGRESSION: the Phase 8-13 flags (meta-labelling, ranks_only, HRP,
    confidence sizing, triple-barrier, feature pruning, EDGAR events/items)
    must be exposed by RefreshRequest so the HTTP API can drive the
    documented best config without operators having to SSH and run the
    CLI directly."""
    from stockpred.backend.schemas import RefreshRequest

    # Build the documented Phase 13 best config via the schema.
    body = RefreshRequest(
        phase=5,
        start_date="2014-01-01",
        n_tickers=150,
        universe_sampling="current",
        horizons=[5],
        model="gbm",
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        ensemble_weighting="equal",
        position_sizing="hrp",
        k_per_side_pct=0.15,
        sector_cap_gross=0.30,
        min_trade_threshold=0.005,
        holdout_years=2,
        bootstrap_method="block",
        bootstrap_n=500,
        # Phase 8
        use_meta_labelling=True,
        meta_threshold=0.55,
        ranks_only=True,
        # Phase 9 (defaults are safe: binary mode, conf-floor 0.60)
        meta_mode="binary",
        # Phase 13
        use_edgar_item_features=True,
    )
    assert body.use_meta_labelling is True
    assert body.ranks_only is True
    assert body.position_sizing == "hrp"
    assert body.meta_conf_floor == 0.60  # default must be the Phase 10 sweet spot
    assert body.use_edgar_item_features is True


def test_build_pipeline_cfg_passes_phase_8_through_13_fields_through():
    """REGRESSION: every new RefreshRequest field must reach the
    PipelineV5Config _build_pipeline_cfg constructs."""
    from stockpred.backend.api import _build_pipeline_cfg
    from stockpred.backend.schemas import RefreshRequest
    from stockpred.pipeline_v5 import PipelineV5Config

    body = RefreshRequest(
        phase=5,
        use_meta_labelling=True,
        meta_threshold=0.60,
        ranks_only=True,
        meta_mode="confidence",
        meta_conf_floor=0.65,
        meta_conf_cap=0.95,
        meta_walk_forward_folds=3,
        meta_per_sector=True,
        use_triple_barrier_labels=True,
        feature_exclude=["adv_proxy_21", "kurt_63"],
        use_edgar_features=False,
        use_edgar_item_features=True,
        position_sizing="hrp",
    )
    cfg = _build_pipeline_cfg(body)
    assert isinstance(cfg, PipelineV5Config)
    assert cfg.use_meta_labelling is True
    assert cfg.meta_threshold == 0.60
    assert cfg.ranks_only is True
    assert cfg.meta_mode == "confidence"
    assert cfg.meta_conf_floor == 0.65
    assert cfg.meta_conf_cap == 0.95
    assert cfg.meta_walk_forward_folds == 3
    assert cfg.meta_per_sector is True
    assert cfg.use_triple_barrier_labels is True
    assert cfg.feature_exclude == ("adv_proxy_21", "kurt_63")
    assert cfg.use_edgar_features is False
    assert cfg.use_edgar_item_features is True
    assert cfg.position_sizing == "hrp"
