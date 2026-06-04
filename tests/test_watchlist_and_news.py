"""Backend tests for watchlist and news endpoints."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def app_client(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("STOCKPRED_DB", str(tmp_path / "app.db"))
    monkeypatch.setenv("STOCKPRED_DISABLE_SCHEDULER", "1")
    monkeypatch.setenv("STOCKPRED_API_KEY", "test")
    import importlib
    import stockpred.backend.api as api_mod

    importlib.reload(api_mod)
    with TestClient(api_mod.app) as c:
        yield c, api_mod


def test_default_watchlist_seeded_on_first_boot(app_client):
    """The seed should populate HND.TO / HNU.TO / UNG / SPY / ^VIX."""
    client, _ = app_client
    r = client.get("/watchlist")
    assert r.status_code == 200
    body = r.json()
    tickers = {item["ticker"] for item in body}
    assert {"HND.TO", "HNU.TO", "UNG", "SPY", "^VIX"} <= tickers


def test_watchlist_add_requires_api_key(app_client):
    client, _ = app_client
    r = client.post("/watchlist", json={"ticker": "BTC-USD"})
    assert r.status_code == 401  # missing key


def test_watchlist_remove_requires_api_key(app_client):
    client, _ = app_client
    r = client.delete("/watchlist/HND.TO")
    assert r.status_code == 401


def test_watchlist_add_and_remove(app_client, monkeypatch):
    """Add a synthetic ticker (patching out the network) then remove it."""
    client, _ = app_client

    # Patch the price loader so the add path doesn't hit yfinance.
    import pandas as pd

    def fake_fetch_one(ticker, **kw):
        idx = pd.date_range("2024-01-01", periods=5, freq="B")
        return pd.DataFrame(
            {
                "open": [10.0] * 5,
                "high": [11.0] * 5,
                "low": [9.0] * 5,
                "close": [10.5] * 5,
                "adj_close": [10.5] * 5,
                "volume": [1e6] * 5,
            },
            index=idx,
        )

    monkeypatch.setattr("stockpred.data.prices.fetch_one", fake_fetch_one)

    headers = {"X-API-Key": "test"}
    r = client.post(
        "/watchlist",
        json={"ticker": "BTC-USD", "label": "Bitcoin", "category": "crypto"},
        headers=headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["ticker"] == "BTC-USD"

    # Now visible in the listing.
    r = client.get("/watchlist")
    assert any(item["ticker"] == "BTC-USD" for item in r.json())

    # Remove.
    r = client.delete("/watchlist/BTC-USD", headers=headers)
    assert r.status_code == 200
    r = client.delete("/watchlist/BTC-USD", headers=headers)
    assert r.status_code == 404


def test_news_endpoint_returns_persisted_items(app_client):
    """Inject news rows directly into the DB, hit the endpoint, verify shape."""
    from stockpred.backend import store
    from stockpred.backend.db import session_scope

    client, api_mod = app_client
    SessionLocal = api_mod.AppState.SessionLocal

    items = [
        {
            "uuid": "u1",
            "title": "Test headline 1",
            "publisher": "Test Wire",
            "link": "https://example.com/1",
            "type": "STORY",
            "published_at": dt.datetime(2024, 5, 1, 12, 0),
        },
        {
            "uuid": "u2",
            "title": "Test headline 2",
            "publisher": "Other Wire",
            "link": "https://example.com/2",
            "type": "VIDEO",
            "published_at": dt.datetime(2024, 5, 2, 12, 0),
        },
    ]
    with session_scope(SessionLocal) as s:
        store.upsert_news(s, "AAPL", items)

    r = client.get("/tickers/AAPL/news")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 2
    # Order: most recent first.
    assert body[0]["uuid"] == "u2"
    assert body[1]["uuid"] == "u1"
    assert body[0]["title"] == "Test headline 2"


def test_news_endpoint_does_not_use_model(app_client):
    """Sanity: the news endpoint is independent of any pipeline run."""
    client, _ = app_client
    # No predictions exist, no runs exist; the news endpoint should still
    # return an empty list (not error).
    r = client.get("/tickers/NOFINDER/news")
    assert r.status_code == 200
    assert r.json() == []


# -------- security regressions for the review's CRITICAL findings ----------


def test_watchlist_add_rejects_path_traversal_ticker(app_client):
    """C1 fix: an attacker-supplied ticker that's not a valid ticker pattern
    must be rejected with 422 BEFORE the loader ever touches the filesystem
    or yfinance."""
    client, _ = app_client
    headers = {"X-API-Key": "test"}
    for bad in ("../../etc/passwd", "foo/bar", "..", "ABC?inject=1", "a b c", ""):
        r = client.post("/watchlist", json={"ticker": bad}, headers=headers)
        assert r.status_code == 422, f"expected 422 for {bad!r}, got {r.status_code}"


def test_watchlist_delete_rejects_bad_ticker_chars(app_client):
    """C1 fix: characters that DO reach our handler must be rejected with 422.

    (URL-encoded `/` is rejected by Starlette at routing time with 405 —
    also fine.)
    """
    client, _ = app_client
    headers = {"X-API-Key": "test"}
    # Lowercase + spaces + quotes — characters that survive URL routing but
    # fail our ticker validator.
    for bad in ("a b c", "ABC;DROP", "hello'world"):
        r = client.delete(f"/watchlist/{bad}", headers=headers)
        assert r.status_code == 422, f"expected 422 for {bad!r}, got {r.status_code}"


def test_ticker_news_rejects_bad_ticker_chars(app_client):
    """Path-parameter validation on the news endpoint.

    Note: lowercase is *permitted* (we uppercase it). We reject chars that
    aren't part of any legitimate ticker.
    """
    client, _ = app_client
    for bad in ("a b c", "ABC;DROP", "AAA*BBB"):
        r = client.get(f"/tickers/{bad}/news")
        assert r.status_code == 422, f"expected 422 for {bad!r}, got {r.status_code}"


def test_news_normaliser_drops_dangerous_links():
    """Defence in depth: javascript:/data: URLs must NOT be persisted."""
    from stockpred.data.news import _normalise_one

    item = {
        "uuid": "u1",
        "title": "click me",
        "link": "javascript:alert(1)",
        "providerPublishTime": 1700000000,
    }
    out = _normalise_one(item)
    assert out["link"] is None
    assert out["title"] == "click me"  # title is preserved, React escapes on render

    safe_item = {**item, "link": "https://example.com/news"}
    assert _normalise_one(safe_item)["link"] == "https://example.com/news"


def test_news_title_with_html_is_safely_json_encoded(app_client):
    """Even if a publisher slips HTML into a title, the JSON response must be
    safe (React escapes by default; we also rely on JSON encoding here)."""
    import json as _json
    from stockpred.backend import store
    from stockpred.backend.db import session_scope

    client, api_mod = app_client
    SessionLocal = api_mod.AppState.SessionLocal

    with session_scope(SessionLocal) as s:
        store.upsert_news(
            s,
            "AAPL",
            [{"uuid": "x1", "title": "<script>alert(1)</script>", "link": "https://e.com"}],
        )

    r = client.get("/tickers/AAPL/news")
    assert r.status_code == 200
    raw = r.text
    # The title appears JSON-encoded; literal <script> is fine inside JSON
    # strings — the SPA renders it via React {value} which HTML-escapes.
    body = _json.loads(raw)
    assert body[0]["title"] == "<script>alert(1)</script>"
    assert body[0]["link"] == "https://e.com"
