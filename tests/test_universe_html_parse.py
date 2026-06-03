"""Regression test: `pd.read_html` must be fed a StringIO, not a raw string.

Bug: a previous version called `pd.read_html(resp.text)`. Pandas interpreted
the body string as a *file path*, blew up with an OSError, and the lxml driver
echoed the whole HTML body to stderr.

We assert the parser path by monkey-patching `requests.get` to return a
known-good Wikipedia-like HTML snippet, then call the loader and expect it to
produce a usable membership DataFrame without OSError.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from stockpred.data import universe as universe_mod


WIKI_FIXTURE = """<!doctype html>
<html><body>
<table id="constituents">
  <thead>
    <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th></tr>
  </thead>
  <tbody>
    <tr><td>AAA</td><td>Alpha Corp</td><td>Tech</td></tr>
    <tr><td>BBB</td><td>Beta Co</td><td>Health</td></tr>
    <tr><td>BRK.B</td><td>Berkshire B</td><td>Financials</td></tr>
  </tbody>
</table>

<table id="changes">
  <thead>
    <tr>
      <th>Date</th>
      <th>Added Ticker</th><th>Added Security</th>
      <th>Removed Ticker</th><th>Removed Security</th>
      <th>Reason</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>January 1, 2015</td><td>AAA</td><td>Alpha Corp</td><td></td><td></td><td>added</td></tr>
    <tr><td>March 5, 2018</td><td>BBB</td><td>Beta Co</td><td>ZZZ</td><td>Old Co</td><td>swap</td></tr>
  </tbody>
</table>
</body></html>
"""


def test_universe_parses_fixture_html(monkeypatch, tmp_path):
    """Universe loader must produce a DataFrame from a Wikipedia-like page."""

    def fake_get(url, headers=None, timeout=None):
        return SimpleNamespace(text=WIKI_FIXTURE, raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", fake_get)
    cache = tmp_path / "membership.parquet"

    out = universe_mod.fetch_sp500_membership(cache_file=cache, refresh=True)
    assert isinstance(out, pd.DataFrame)
    assert {"ticker", "start_date", "end_date"}.issubset(out.columns)
    # Three current tickers plus one historical-only "ZZZ" => 4 unique tickers.
    assert set(out["ticker"]) >= {"AAA", "BBB", "BRK-B"}
    # `members_on(today)` should include current tickers.
    today = pd.Timestamp("2024-01-01")
    members = universe_mod.members_on(today, membership=out)
    assert "AAA" in members
    assert "BBB" in members


def test_read_html_takes_stringio_not_raw_string(monkeypatch):
    """The bug was: pd.read_html(resp.text) is interpreted as a path.
    Assert the loader doesn't crash on a body that *looks* like a file path.
    """

    def fake_get(url, headers=None, timeout=None):
        return SimpleNamespace(text=WIKI_FIXTURE, raise_for_status=lambda: None)

    monkeypatch.setattr("requests.get", fake_get)
    # Should not raise OSError.
    tables = universe_mod._read_html_tables("http://example.com")
    assert isinstance(tables, list) and len(tables) >= 1


def test_members_on_strict_boundary_for_removal_date():
    """Regression test for review finding H3.

    A ticker removed on date D must NOT appear in members_on(D). Using `>=`
    would leak forward-looking information about the delisting.
    """
    removal_date = pd.Timestamp("2020-06-15")
    fake_membership = pd.DataFrame(
        {
            "ticker": ["ZZZ", "AAA"],
            "start_date": [pd.Timestamp("2010-01-01"), pd.NaT],  # ZZZ joined 2010, AAA always
            "end_date": [removal_date, pd.NaT],
        }
    )

    # On removal_date itself: ZZZ is GONE.
    members = universe_mod.members_on(removal_date, membership=fake_membership)
    assert "AAA" in members
    assert "ZZZ" not in members, "ticker removed on date D must not be in members_on(D)"

    # On removal_date - 1: ZZZ is still in.
    prev = universe_mod.members_on(removal_date - pd.Timedelta(days=1), membership=fake_membership)
    assert "ZZZ" in prev
