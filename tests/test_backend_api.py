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
    """Spin up a fresh FastAPI app with an isolated SQLite DB and no scheduler."""
    db_path = tmp_path / "app.db"
    monkeypatch.setenv("STOCKPRED_DB", str(db_path))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    # Ensure fresh module state (api uses module-level AppState).
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
