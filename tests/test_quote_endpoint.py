"""Tests for the live (delayed) quote endpoint GET /quote/{ticker}.

yfinance is monkeypatched so tests never hit the network. We verify the
change/change_pct computation, the validation guard, and the server-side
cache (a second call within TTL must not re-invoke the fetcher).
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "app.db"
    monkeypatch.setenv("STOCKPRED_DB", str(db_path))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    monkeypatch.setenv("STOCKPRED_CORS", "*")

    import stockpred.backend.api as api_mod

    importlib.reload(api_mod)
    with TestClient(api_mod.app) as client:
        yield client, api_mod


def test_quote_computes_change(app_client, monkeypatch):
    client, _ = app_client
    from stockpred.data import prices as prices_mod

    monkeypatch.setattr(
        prices_mod, "latest_quote",
        lambda t: {
            "ticker": t, "price": 105.0, "previous_close": 100.0,
            "open": 101.0, "day_high": 106.0, "day_low": 100.5,
            "volume": 1_000_000.0, "market_cap": 2.0e12,
        },
    )

    r = client.get("/quote/AAPL")
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "AAPL"
    assert body["price"] == 105.0
    assert body["change"] == 5.0
    assert abs(body["change_pct"] - 0.05) < 1e-9
    assert body["delayed"] is True
    assert body["as_of"] is not None


def test_quote_handles_missing_previous_close(app_client, monkeypatch):
    client, _ = app_client
    from stockpred.data import prices as prices_mod

    monkeypatch.setattr(
        prices_mod, "latest_quote",
        lambda t: {"ticker": t, "price": 50.0, "previous_close": None,
                   "open": None, "day_high": None, "day_low": None,
                   "volume": None, "market_cap": None},
    )

    r = client.get("/quote/MSFT")
    assert r.status_code == 200
    body = r.json()
    assert body["price"] == 50.0
    assert body["change"] is None
    assert body["change_pct"] is None


def test_quote_invalid_ticker_422(app_client):
    client, _ = app_client
    r = client.get("/quote/not!a!ticker")
    assert r.status_code == 422


def test_quote_cache_throttles_fetch(app_client, monkeypatch):
    """Two calls within TTL should only invoke latest_quote once."""
    client, _ = app_client
    from stockpred.data import prices as prices_mod

    calls = {"n": 0}

    def _counting_quote(t):
        calls["n"] += 1
        return {"ticker": t, "price": 10.0 + calls["n"], "previous_close": 10.0,
                "open": None, "day_high": None, "day_low": None,
                "volume": None, "market_cap": None}

    monkeypatch.setattr(prices_mod, "latest_quote", _counting_quote)

    r1 = client.get("/quote/NVDA").json()
    r2 = client.get("/quote/NVDA").json()
    assert calls["n"] == 1            # second served from cache
    assert r1["price"] == r2["price"]  # identical payload
