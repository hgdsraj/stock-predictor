"""Phase 12 tests: SEC EDGAR 8-K event-feature builder.

All HTTP is mocked. Tests verify:
  - User-Agent header is sent on every request
  - Rate-limit sleep is invoked (test the call, not the duration)
  - Ticker -> CIK parsing handles SEC's JSON shape
  - form.idx parsing extracts only 8-K rows, ignoring headers and
    non-8-K forms (10-K, 10-Q, S-1, etc.)
  - Quarter cache parquet round-trip works
  - build_8k_features handles missing tickers, weekend/holiday filings,
    and zero-event quarters
  - Output dtypes are memory-efficient (int8 / int16)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from stockpred.data import edgar


# A realistic minimal form.idx slice (header + 4 filings: two 8-K, one 10-K,
# one S-1). Whitespace is FIXED-WIDTH (real SEC format).
SAMPLE_FORM_IDX = """Description:           Master Index of EDGAR Dissemination Feed by Form Type
Last Data Received:    March 31, 2024
Comments:              webmaster@sec.gov
Anonymous FTP:         ftp://ftp.sec.gov/edgar/

 
Form Type        Company Name                                                  CIK         Date Filed  Filename
---------------------------------------------------------------------------------------------------------------
8-K              APPLE INC                                                     320193      2024-01-25  edgar/data/320193/0000320193-24-000003-index.htm
8-K              MICROSOFT CORP                                                789019      2024-01-30  edgar/data/789019/0000789019-24-000007-index.htm
10-K             APPLE INC                                                     320193      2024-02-02  edgar/data/320193/0000320193-24-000007-index.htm
S-1              SOME NEW IPO INC                                              999999      2024-03-15  edgar/data/999999/0000999999-24-000001-index.htm
"""


SAMPLE_TICKER_JSON = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1652044, "ticker": "GOOG", "title": "Alphabet Inc."},
}


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Redirect EDGAR cache to a temp dir so tests don't see each other.

    Also lowers `_MIN_TICKER_CIK_ENTRIES` to 1 so the tiny test
    fixtures (3 tickers) pass validation. Production code keeps the
    1000-entry threshold to catch stub responses.
    """
    monkeypatch.setattr(edgar, "CACHE_DIR_EDGAR", tmp_path)
    monkeypatch.setattr(edgar, "TICKER_CIK_CACHE", tmp_path / "ticker_to_cik.json")
    monkeypatch.setattr(edgar, "EVENTS_CACHE", tmp_path / "8k_events.parquet")
    monkeypatch.setattr(edgar, "_MIN_TICKER_CIK_ENTRIES", 1)
    monkeypatch.setattr(
        edgar,
        "_quarter_cache_path",
        lambda y, q: tmp_path / f"8k_{y}Q{q}.parquet",
    )
    return tmp_path


def _make_mock_response(text: str = "", json_obj: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.text = text
    r.raise_for_status = MagicMock()
    if json_obj is not None:
        r.json = MagicMock(return_value=json_obj)
    return r


def test_split_ws2_handles_company_name_spaces():
    """Company names contain single spaces; tokens are separated by 2+ spaces."""
    line = "8-K              APPLE INC        320193   2024-01-25    edgar/foo.htm"
    parts = edgar._split_ws2(line)
    assert parts == ["8-K", "APPLE INC", "320193", "2024-01-25", "edgar/foo.htm"]


def test_split_ws2_handles_multi_word_company():
    """JOHNSON & JOHNSON should stay as one token."""
    line = "8-K              JOHNSON & JOHNSON          200406    2024-04-01   x.htm"
    parts = edgar._split_ws2(line)
    assert parts[0] == "8-K"
    assert parts[1] == "JOHNSON & JOHNSON"
    assert parts[2] == "200406"


def test_quarters_in_range_int_back_compat():
    """Bare-year API returns every quarter of the year range."""
    assert edgar._quarters_in_range(2023, 2024) == [
        (2023, 1),
        (2023, 2),
        (2023, 3),
        (2023, 4),
        (2024, 1),
        (2024, 2),
        (2024, 3),
        (2024, 4),
    ]
    assert edgar._quarters_in_range(2024, 2024) == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
        (2024, 4),
    ]


def test_quarters_in_range_timestamp_narrows_to_overlapping_quarters():
    """Timestamp API restricts to quarters that overlap the range."""
    # 2024-01-01 to 2024-03-31 = Q1 only
    assert edgar._quarters_in_range(pd.Timestamp("2024-01-01"), pd.Timestamp("2024-03-31")) == [
        (2024, 1)
    ]
    # 2024-02-15 to 2024-08-15 = Q1, Q2, Q3
    assert edgar._quarters_in_range(pd.Timestamp("2024-02-15"), pd.Timestamp("2024-08-15")) == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
    ]
    # 2023-11-01 to 2024-02-01 = 2023Q4, 2024Q1
    assert edgar._quarters_in_range(pd.Timestamp("2023-11-01"), pd.Timestamp("2024-02-01")) == [
        (2023, 4),
        (2024, 1),
    ]
    # Single day in a single quarter
    assert edgar._quarters_in_range(pd.Timestamp("2024-05-15"), pd.Timestamp("2024-05-15")) == [
        (2024, 2)
    ]


def test_http_get_sends_user_agent_and_sleeps(monkeypatch):
    """Every HTTP call must include User-Agent and sleep first."""
    sleep_calls: list[float] = []
    monkeypatch.setattr(edgar.time, "sleep", lambda s: sleep_calls.append(s))

    captured = {}

    def mock_get(url, headers=None, timeout=None, **kw):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _make_mock_response(text="ok")

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    edgar._http_get("https://www.sec.gov/foo")

    assert "User-Agent" in captured["headers"]
    assert captured["headers"]["User-Agent"]  # non-empty
    assert sleep_calls and sleep_calls[0] == edgar._RATE_LIMIT_SLEEP_S
    assert captured["timeout"] == 30


def test_fetch_ticker_to_cik_parses_sec_format(monkeypatch):
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        edgar.requests,
        "get",
        lambda url, **kw: _make_mock_response(json_obj=SAMPLE_TICKER_JSON),
    )

    out = edgar.fetch_ticker_to_cik(refresh=True)

    assert out == {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "GOOG": "0001652044",
    }
    # All CIKs are 10-digit zero-padded strings
    assert all(len(c) == 10 and c.isdigit() for c in out.values())


def test_validate_ticker_cik_map_catches_bad_structure():
    """REGRESSION (Phase 12 reviewer CRITICAL #1): semantically bad
    cache must fail validation even when JSON is well-formed."""
    # All structure OK except one CIK is wrong shape
    bad = {"AAPL": "0000320193", "MSFT": "0000789019", "BAD": "320193"}
    assert not edgar._validate_ticker_cik_map(bad, min_entries=1)
    # Empty dict fails
    assert not edgar._validate_ticker_cik_map({}, min_entries=1)
    # Non-dict input fails
    assert not edgar._validate_ticker_cik_map([], min_entries=1)
    # Size threshold enforced
    good_small = {"AAPL": "0000320193"}
    assert edgar._validate_ticker_cik_map(good_small, min_entries=1)
    assert not edgar._validate_ticker_cik_map(good_small, min_entries=1000)


def test_fetch_ticker_to_cik_rejects_corrupt_cache(monkeypatch, isolate_cache):
    """REGRESSION (Phase 12 reviewer CRITICAL #1): a corrupt cached
    JSON (well-formed JSON, but missing CIKs) must trigger a refetch
    rather than silently propagating zeros."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    # Write a structurally-bad cache file
    edgar.TICKER_CIK_CACHE.write_text(json.dumps({"AAPL": "not-a-cik"}))

    call_count = {"n": 0}

    def mock_get(url, **kw):
        call_count["n"] += 1
        return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)

    monkeypatch.setattr(edgar.requests, "get", mock_get)
    out = edgar.fetch_ticker_to_cik(refresh=False)
    # Must have refetched despite refresh=False
    assert call_count["n"] == 1
    # Must have produced a valid result
    assert "AAPL" in out
    assert out["AAPL"] == "0000320193"


def test_fetch_ticker_to_cik_uses_cache(monkeypatch, isolate_cache):
    """A second call with refresh=False must NOT hit the network."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    call_count = {"n": 0}

    def mock_get(url, **kw):
        call_count["n"] += 1
        return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    edgar.fetch_ticker_to_cik(refresh=True)
    assert call_count["n"] == 1

    # Second call should hit cache.
    edgar.fetch_ticker_to_cik(refresh=False)
    assert call_count["n"] == 1  # Still 1 — no extra fetch


def test_fetch_quarter_8k_filters_to_8k_only(monkeypatch):
    """form.idx contains 10-K, S-1, etc. We must keep ONLY 8-K rows."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    monkeypatch.setattr(
        edgar.requests,
        "get",
        lambda url, **kw: _make_mock_response(text=SAMPLE_FORM_IDX),
    )

    df = edgar._fetch_quarter_8k(2024, 1, refresh=True)

    assert len(df) == 2  # Only the two 8-K rows
    assert set(df["cik"]) == {"0000320193", "0000789019"}
    assert all(df["filing_date"].dt.year == 2024)


def test_fetch_quarter_8k_404_returns_empty(monkeypatch):
    """A 404 (e.g. very-old quarter) must return an empty DF, not crash."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)

    def mock_get_404(url, **kw):
        r = MagicMock()
        r.raise_for_status = MagicMock(side_effect=edgar.requests.HTTPError("404"))
        return r

    monkeypatch.setattr(edgar.requests, "get", mock_get_404)
    df = edgar._fetch_quarter_8k(1995, 1, refresh=True)
    assert df.empty
    assert list(df.columns) == ["cik", "filing_date"]


def test_fetch_quarter_8k_caches_parquet(monkeypatch, isolate_cache):
    """First call hits HTTP and caches; second call reads cache."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    call_count = {"n": 0}

    def mock_get(url, **kw):
        call_count["n"] += 1
        return _make_mock_response(text=SAMPLE_FORM_IDX)

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    df1 = edgar._fetch_quarter_8k(2024, 1, refresh=True)
    assert call_count["n"] == 1
    cache_path = edgar._quarter_cache_path(2024, 1)
    assert cache_path.exists()

    df2 = edgar._fetch_quarter_8k(2024, 1, refresh=False)
    assert call_count["n"] == 1  # No extra HTTP
    # Round-trip equivalence
    assert df1.equals(df2)


def test_build_8k_features_returns_correct_panel_shape(monkeypatch, isolate_cache):
    """Full pipeline: ticker map + 8-K events -> per-(date, ticker) features."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)

    def mock_get(url, **kw):
        if "company_tickers.json" in url:
            return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)
        if "form.idx" in url:
            return _make_mock_response(text=SAMPLE_FORM_IDX)
        raise ValueError(f"Unexpected URL: {url}")

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    trading_days = pd.bdate_range("2024-01-22", "2024-02-09")  # 15 bdays
    tickers = ["AAPL", "MSFT", "FAKE"]

    out = edgar.build_8k_features(
        tickers, trading_days, start="2024-01-01", end="2024-03-31", refresh=True
    )

    # Should be indexed by (date, ticker); FAKE has no CIK -> dropped from columns
    assert "has_8k" in out.columns
    assert "count_8k_5d" in out.columns
    assert "count_8k_21d" in out.columns
    assert "count_8k_63d" in out.columns
    # Memory-efficient dtypes
    assert out["has_8k"].dtype.name == "int8"
    assert out["count_8k_5d"].dtype.name == "int16"

    # AAPL had a filing on 2024-01-25; that day's has_8k must be 1.
    # MSFT had a filing on 2024-01-30; that day's has_8k must be 1.
    aapl_jan25 = out.loc[(pd.Timestamp("2024-01-25"), "AAPL")]
    assert int(aapl_jan25["has_8k"]) == 1
    msft_jan30 = out.loc[(pd.Timestamp("2024-01-30"), "MSFT")]
    assert int(msft_jan30["has_8k"]) == 1
    # AAPL on a day with NO filing: has_8k = 0
    aapl_feb05 = out.loc[(pd.Timestamp("2024-02-05"), "AAPL")]
    assert int(aapl_feb05["has_8k"]) == 0


def test_build_8k_features_rolling_counts_accumulate(monkeypatch, isolate_cache):
    """count_8k_5d for AAPL on 2024-01-26 should be 1 (the Jan 25 filing)."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)

    def mock_get(url, **kw):
        if "company_tickers.json" in url:
            return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)
        return _make_mock_response(text=SAMPLE_FORM_IDX)

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    trading_days = pd.bdate_range("2024-01-22", "2024-02-09")
    out = edgar.build_8k_features(
        ["AAPL"], trading_days, start="2024-01-01", end="2024-03-31", refresh=True
    )
    # Jan 25 (Thu): filing day, count_5d = 1
    assert int(out.loc[(pd.Timestamp("2024-01-25"), "AAPL"), "count_8k_5d"]) == 1
    # Jan 26 (Fri): count_5d still 1 (Jan 25 in window)
    assert int(out.loc[(pd.Timestamp("2024-01-26"), "AAPL"), "count_8k_5d"]) == 1
    # 6+ trading days later, count_5d drops back to 0
    later = pd.Timestamp("2024-02-05")
    assert int(out.loc[(later, "AAPL"), "count_8k_5d"]) == 0
    # count_21d still includes the Jan 25 filing on Feb 5
    assert int(out.loc[(later, "AAPL"), "count_8k_21d"]) == 1


def test_build_8k_features_no_ticker_overlap_returns_empty(monkeypatch, isolate_cache):
    """If none of our tickers have a CIK match, return empty without crashing."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)

    def mock_get(url, **kw):
        if "company_tickers.json" in url:
            return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)
        return _make_mock_response(text=SAMPLE_FORM_IDX)

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    trading_days = pd.bdate_range("2024-01-22", "2024-02-09")
    out = edgar.build_8k_features(
        ["FAKE1", "FAKE2"], trading_days, start="2024-01-01", end="2024-03-31", refresh=True
    )
    assert out.empty


def test_parse_idx_header_finds_known_field_starts():
    """Detect column starts from a real SEC header line."""
    header = (
        "Form Type        Company Name                                                  "
        "CIK         Date Filed  Filename"
    )
    starts = edgar._parse_idx_header(header)
    assert len(starts) == 5
    # Form Type is at 0
    assert starts[0] == 0
    # Company Name appears after some spaces
    assert header[starts[1] : starts[1] + len("Company Name")] == "Company Name"
    assert header[starts[2] : starts[2] + 3] == "CIK"
    assert header[starts[3] : starts[3] + len("Date Filed")] == "Date Filed"
    assert header[starts[4] : starts[4] + len("Filename")] == "Filename"


def test_parse_idx_header_missing_field_raises():
    """A header without one of the expected fields raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        edgar._parse_idx_header("Random  header  with  no  matching  words")


def test_slice_fixed_width_returns_correct_cells():
    """Fixed-width slicing must produce the exact field strings.

    Build the line by placing each field at the column the header
    defines, then slice it back. This is the same shape SEC actually
    publishes.
    """
    header = (
        "Form Type        Company Name                                                  "
        "CIK         Date Filed  Filename"
    )
    starts = edgar._parse_idx_header(header)

    # Construct a line: pad each field with spaces to land at the next
    # column position.
    def cell(text: str, width: int) -> str:
        return text.ljust(width)

    widths = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    parts = [
        cell("8-K", widths[0]),
        cell("PROCTER  &  GAMBLE CO", widths[1]),  # Multi-space name!
        cell("80424", widths[2]),
        cell("2024-01-22", widths[3]),
        "edgar/data/80424/0000080424-24-000007-index.htm",
    ]
    line = "".join(parts)
    cells = edgar._slice_fixed_width(line, starts)
    assert len(cells) == 5
    assert cells[0].strip() == "8-K"
    # The fixed-width parser preserves multi-space company names exactly.
    assert cells[1].strip() == "PROCTER  &  GAMBLE CO"
    assert cells[2].strip() == "80424"
    assert cells[3].strip() == "2024-01-22"
    assert cells[4].strip().startswith("edgar/")


def _build_idx_line(form_type: str, company: str, cik: str, date_str: str) -> str:
    """Construct a single form.idx data line using the SAMPLE_FORM_IDX
    column positions (so test fixtures stay aligned with the header).

    Mirrors the real SEC fixed-width format used in tests.
    """
    header_line = (
        "Form Type        Company Name                                                  "
        "CIK         Date Filed  Filename"
    )
    starts = edgar._parse_idx_header(header_line)
    widths = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    parts = [
        form_type.ljust(widths[0]),
        company.ljust(widths[1]),
        cik.ljust(widths[2]),
        date_str.ljust(widths[3]),
        f"edgar/data/{cik}/{cik}-24-000001-index.htm",
    ]
    return "".join(parts)


def _build_idx_doc(lines: list[str]) -> str:
    """Wrap data lines with a header + separator that matches the parser."""
    header = (
        "Form Type        Company Name                                                  "
        "CIK         Date Filed  Filename"
    )
    sep = "-" * 110
    return "\n".join([header, sep, *lines]) + "\n"


def test_fetch_quarter_8k_handles_procter_and_gamble_multi_space_name(monkeypatch, isolate_cache):
    """REGRESSION (Phase 12 reviewer HIGH #3): a company name with multiple
    consecutive spaces between words (like real-SEC "PROCTER  &  GAMBLE CO")
    must be parsed correctly, not silently dropped.
    """
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    idx_with_pg = _build_idx_doc(
        [
            _build_idx_line("8-K", "APPLE INC", "320193", "2024-01-25"),
            _build_idx_line("8-K", "PROCTER  &  GAMBLE CO", "80424", "2024-01-22"),
        ]
    )
    monkeypatch.setattr(
        edgar.requests,
        "get",
        lambda url, **kw: _make_mock_response(text=idx_with_pg),
    )

    df = edgar._fetch_quarter_8k(2024, 1, refresh=True)
    # MUST find BOTH filings (the old whitespace-split parser would
    # silently drop the P&G row).
    assert len(df) == 2
    assert "0000080424" in set(df["cik"])  # P&G's CIK
    assert "0000320193" in set(df["cik"])  # AAPL


def test_fetch_quarter_8k_handles_weekend_filing(monkeypatch, isolate_cache):
    """REGRESSION (Phase 12 reviewer MEDIUM #11): a Saturday 8-K filing
    must be carried forward to the next trading day (Monday) in the
    feature panel, NOT applied to the prior Friday."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)
    # Sat 2024-01-27 -> should appear on Mon 2024-01-29
    weekend_idx = _build_idx_doc(
        [
            _build_idx_line("8-K", "APPLE INC", "320193", "2024-01-27"),
        ]
    )

    def mock_get(url, **kw):
        if "company_tickers.json" in url:
            return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)
        return _make_mock_response(text=weekend_idx)

    monkeypatch.setattr(edgar.requests, "get", mock_get)

    trading_days = pd.bdate_range("2024-01-22", "2024-02-02")
    out = edgar.build_8k_features(
        ["AAPL"],
        trading_days,
        start="2024-01-01",
        end="2024-03-31",
        refresh=True,
    )
    # Fri 2024-01-26: NO filing (would be leakage if it appeared here)
    fri = pd.Timestamp("2024-01-26")
    assert int(out.loc[(fri, "AAPL"), "has_8k"]) == 0
    # Mon 2024-01-29: must be 1 (weekend filing forwarded to next bday)
    mon = pd.Timestamp("2024-01-29")
    assert int(out.loc[(mon, "AAPL"), "has_8k"]) == 1


def test_fetch_quarter_8k_zero_events_in_range(monkeypatch, isolate_cache):
    """If the date range has no 8-Ks at all, return empty without crashing."""
    monkeypatch.setattr(edgar.time, "sleep", lambda s: None)

    # form.idx with only non-8-K rows
    nonidx = """Form Type        Company Name                                                  CIK         Date Filed  Filename
---------------------------------------------------------------------------------------------------------------
10-K             APPLE INC                                                     320193      2024-02-02  x.htm
"""

    def mock_get(url, **kw):
        if "company_tickers.json" in url:
            return _make_mock_response(json_obj=SAMPLE_TICKER_JSON)
        return _make_mock_response(text=nonidx)

    monkeypatch.setattr(edgar.requests, "get", mock_get)
    trading_days = pd.bdate_range("2024-01-22", "2024-02-09")
    out = edgar.build_8k_features(
        ["AAPL"], trading_days, start="2024-01-01", end="2024-03-31", refresh=True
    )
    assert out.empty
