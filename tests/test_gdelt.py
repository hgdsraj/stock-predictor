"""Phase 14 tests: GDELT GKG daily-tone feature builder.

All HTTP is mocked; tests verify:
  - Per-day cache write/read round-trip
  - 404 caches an empty marker (no infinite retries)
  - Tab-delimited GKG parser handles legacy positional schema
  - ORGANIZATIONS field with multi-name + tone slot is aggregated correctly
  - Same-row dedup (one article mentioning a company twice -> 1 mention)
  - build_gdelt_features handles missing-day caches gracefully (no crash)
  - Weekend filings forward-shift to next trading day
  - Memory-friendly dtypes (int16, float32, category) on the output
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from stockpred.data import gdelt


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Redirect GDELT cache to a temp dir + zero-sleep so tests fly."""
    monkeypatch.setattr(gdelt, "CACHE_DIR_GDELT", tmp_path)
    monkeypatch.setattr(
        gdelt,
        "_day_cache_path",
        lambda d: tmp_path / f"gkg_{d.strftime('%Y%m%d')}.parquet",
    )
    monkeypatch.setattr(gdelt.time, "sleep", lambda s: None)
    return tmp_path


def _make_gkg_zip(rows: list[list[str]]) -> bytes:
    """Build an in-memory GKG zip with tab-delimited rows.

    Schema is the legacy positional one:
      col 0 = DATE (YYYYMMDD)
      col 1 = NUMARTS
      col 2 = COUNTS
      col 3 = THEMES
      col 4 = LOCATIONS
      col 5 = PERSONS
      col 6 = ORGANIZATIONS
      col 7 = TONE
    """
    csv_text_lines = []
    for r in rows:
        # Pad to 11 columns to mimic the real legacy schema width.
        padded = (r + ["", "", "", "", "", "", "", "", "", "", ""])[:11]
        csv_text_lines.append("\t".join(padded))
    csv_text = "\n".join(csv_text_lines).encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("20240125.gkg.csv", csv_text)
    return buf.getvalue()


def _make_response(content: bytes, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.raise_for_status = MagicMock()
    r.iter_content = MagicMock(return_value=[content])
    r.__enter__ = MagicMock(return_value=r)
    r.__exit__ = MagicMock(return_value=False)
    return r


def test_parse_gkg_zip_aggregates_one_org_mention(isolate_cache):
    rows = [
        # DATE  NUMARTS  COUNTS  THEMES  LOCATIONS  PERSONS  ORGS         TONE
        ["20240125", "3", "", "", "", "", "APPLE INC", "1.5, 4.0, 2.5, 5.0, 0, 0, 100"],
    ]
    zip_bytes = _make_gkg_zip(rows)
    df = gdelt._parse_gkg_zip(zip_bytes, {"APPLE INC": ["AAPL"]})
    assert len(df) == 1
    row = df.iloc[0]
    assert row["ticker"] == "AAPL"
    assert row["gdelt_mention_count"] == 1
    assert row["gdelt_article_count"] == 3
    assert abs(row["gdelt_tone_mean"] - 1.5) < 1e-6


def test_parse_gkg_zip_dedupes_same_row_repeated_org(isolate_cache):
    """A single GKG row that mentions 'APPLE INC; APPLE INC' must
    count as ONE mention, not two."""
    rows = [
        ["20240125", "1", "", "", "", "", "APPLE INC;APPLE INC", "2.0, 0, 0, 0, 0, 0, 0"],
    ]
    df = gdelt._parse_gkg_zip(_make_gkg_zip(rows), {"APPLE INC": ["AAPL"]})
    assert len(df) == 1
    assert df.iloc[0]["gdelt_mention_count"] == 1


def test_parse_gkg_zip_aggregates_multi_day(isolate_cache):
    """Two rows on different days aggregate as two ticker rows."""
    rows = [
        ["20240125", "1", "", "", "", "", "APPLE INC", "1.0, 0, 0, 0, 0, 0, 0"],
        ["20240126", "1", "", "", "", "", "APPLE INC", "3.0, 0, 0, 0, 0, 0, 0"],
    ]
    df = gdelt._parse_gkg_zip(_make_gkg_zip(rows), {"APPLE INC": ["AAPL"]})
    assert len(df) == 2
    by_date = df.set_index("date")
    assert abs(by_date.loc[pd.Timestamp("2024-01-25"), "gdelt_tone_mean"] - 1.0) < 1e-6
    assert abs(by_date.loc[pd.Timestamp("2024-01-26"), "gdelt_tone_mean"] - 3.0) < 1e-6


def test_parse_gkg_zip_aggregates_multi_articles_same_day(isolate_cache):
    """Two rows same day -> mention_count=2, tone_mean = avg of the two tones."""
    rows = [
        ["20240125", "1", "", "", "", "", "APPLE INC", "2.0, 0, 0, 0, 0, 0, 0"],
        ["20240125", "1", "", "", "", "", "APPLE INC", "-4.0, 0, 0, 0, 0, 0, 0"],
    ]
    df = gdelt._parse_gkg_zip(_make_gkg_zip(rows), {"APPLE INC": ["AAPL"]})
    assert len(df) == 1
    row = df.iloc[0]
    assert row["gdelt_mention_count"] == 2
    # mean of 2.0 and -4.0 = -1.0
    assert abs(row["gdelt_tone_mean"] - (-1.0)) < 1e-6


def test_parse_gkg_zip_filters_unknown_orgs(isolate_cache):
    rows = [
        ["20240125", "1", "", "", "", "", "RANDOM CORP", "1.0, 0, 0, 0, 0, 0, 0"],
        ["20240125", "1", "", "", "", "", "APPLE INC", "1.0, 0, 0, 0, 0, 0, 0"],
    ]
    df = gdelt._parse_gkg_zip(_make_gkg_zip(rows), {"APPLE INC": ["AAPL"]})
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "AAPL"


def test_fetch_gdelt_day_caches_parquet(isolate_cache, monkeypatch):
    rows = [["20240125", "2", "", "", "", "", "APPLE INC", "1.0, 0, 0, 0, 0, 0, 0"]]
    zip_bytes = _make_gkg_zip(rows)
    call_count = {"n": 0}

    def mock_get(url, **kw):
        call_count["n"] += 1
        return _make_response(zip_bytes)

    monkeypatch.setattr(gdelt.requests, "get", mock_get)

    df1 = gdelt.fetch_gdelt_day(date(2024, 1, 25), {"APPLE INC": ["AAPL"]}, refresh=True)
    assert call_count["n"] == 1
    assert len(df1) == 1

    # Second call uses cache.
    df2 = gdelt.fetch_gdelt_day(date(2024, 1, 25), {"APPLE INC": ["AAPL"]}, refresh=False)
    assert call_count["n"] == 1  # No new HTTP


def test_fetch_gdelt_day_404_caches_empty_marker(isolate_cache, monkeypatch):
    """A 404 (file not yet published) must cache an empty marker so we
    don't retry endlessly."""

    def mock_get_404(url, **kw):
        return _make_response(b"", status_code=404)

    monkeypatch.setattr(gdelt.requests, "get", mock_get_404)
    df = gdelt.fetch_gdelt_day(date(2025, 12, 31), {"APPLE INC": ["AAPL"]}, refresh=True)
    assert df.empty
    cache_path = gdelt._day_cache_path(date(2025, 12, 31))
    assert cache_path.exists()


def test_build_name_to_tickers_strips_legal_suffixes(isolate_cache, monkeypatch, tmp_path):
    """SEC titles end in INC / CORP / etc.; the GDELT matcher needs the
    bare brand name. Verify common suffixes are stripped."""
    # Write a fake company_tickers.json next to the EDGAR cache.
    from stockpred.data import edgar
    import json

    src_dir = tmp_path / "edgar"
    src_dir.mkdir()
    monkeypatch.setattr(edgar, "TICKER_CIK_CACHE", src_dir / "ticker_to_cik.json")
    (src_dir / "company_tickers.json").write_text(
        json.dumps(
            {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
                "1": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORPORATION"},
                "2": {"cik_str": 200406, "ticker": "JNJ", "title": "JOHNSON & JOHNSON"},
                "3": {"cik_str": 99999, "ticker": "FAKE", "title": "INC"},  # too short after strip
            }
        )
    )
    ticker_to_cik = {
        "AAPL": "0000320193",
        "MSFT": "0000789019",
        "JNJ": "0000200406",
        "FAKE": "0000099999",
    }
    name_map = gdelt._build_name_to_tickers(ticker_to_cik)
    # Apple Inc. -> APPLE
    assert "APPLE" in name_map
    assert name_map["APPLE"] == ["AAPL"]
    # MICROSOFT CORPORATION -> MICROSOFT
    assert "MICROSOFT" in name_map
    # JOHNSON & JOHNSON has no legal suffix; stays
    assert "JOHNSON & JOHNSON" in name_map
    # 'INC' -> stripped to empty -> dropped
    assert "INC" not in name_map
    assert "" not in name_map


def test_build_gdelt_features_handles_no_cache(isolate_cache, monkeypatch):
    """If NO per-day caches exist, build_gdelt_features must return an
    empty DataFrame without crashing and warn the operator."""
    trading_days = pd.bdate_range("2024-01-01", "2024-01-05")
    # Don't write any caches; ticker_to_cik exists but no GDELT data.
    out = gdelt.build_gdelt_features(
        ["AAPL"],
        trading_days,
        ticker_to_cik={"AAPL": "0000320193"},
    )
    assert out.empty


def test_build_gdelt_features_aggregates_from_cache(isolate_cache, monkeypatch):
    """Write a few per-day caches by hand, then verify build_gdelt_features
    aggregates them onto trading days with rolling windows."""
    name_to_tickers = {"APPLE INC": ["AAPL"]}

    # Write 3 days of cache: 2024-01-23 (tone=2), 2024-01-24 (tone=-1), 2024-01-25 (tone=3)
    for d, tone, mentions in [
        (date(2024, 1, 23), 2.0, 4),
        (date(2024, 1, 24), -1.0, 2),
        (date(2024, 1, 25), 3.0, 5),
    ]:
        cache_df = pd.DataFrame(
            [
                {
                    "date": pd.Timestamp(d),
                    "ticker": "AAPL",
                    "gdelt_mention_count": mentions,
                    "gdelt_article_count": mentions * 2,
                    "gdelt_tone_mean": tone,
                    "gdelt_tone_std": 0.0,
                }
            ]
        )
        cache_df.to_parquet(gdelt._day_cache_path(d), compression="snappy", index=False)

    trading_days = pd.bdate_range("2024-01-22", "2024-01-26")
    out = gdelt.build_gdelt_features(
        ["AAPL"],
        trading_days,
        ticker_to_cik={"AAPL": "0000320193"},
        rolling_windows=(2,),
    )
    assert not out.empty
    # Output dtype must be memory-friendly
    assert out["gdelt_mention_count"].dtype.name == "int16"
    assert out["gdelt_tone_mean"].dtype.name == "float32"
    # On 2024-01-25 (Thu): tone_mean should be ~3.0
    row = out.loc[(pd.Timestamp("2024-01-25"), "AAPL")]
    assert abs(float(row["gdelt_tone_mean"]) - 3.0) < 1e-3
    # Rolling 2d mention sum on 2024-01-25 should include 24 + 25 mentions = 2 + 5 = 7
    assert int(row["gdelt_mention_2d"]) == 7


def test_build_gdelt_features_weekend_filing_forwards_to_monday(isolate_cache):
    """A GDELT row dated Saturday must land on Monday's row, not Friday's."""
    # Sat 2024-01-27
    cache_df = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2024-01-27"),
                "ticker": "AAPL",
                "gdelt_mention_count": 5,
                "gdelt_article_count": 10,
                "gdelt_tone_mean": 1.0,
                "gdelt_tone_std": 0.0,
            }
        ]
    )
    cache_df.to_parquet(gdelt._day_cache_path(date(2024, 1, 27)), compression="snappy", index=False)
    trading_days = pd.bdate_range("2024-01-22", "2024-02-02")
    out = gdelt.build_gdelt_features(
        ["AAPL"],
        trading_days,
        ticker_to_cik={"AAPL": "0000320193"},
    )
    # Fri 2024-01-26: no news (would be leakage)
    fri = (pd.Timestamp("2024-01-26"), "AAPL")
    assert int(out.loc[fri, "gdelt_mention_count"]) == 0
    # Mon 2024-01-29: news landed here (forward-shifted)
    mon = (pd.Timestamp("2024-01-29"), "AAPL")
    assert int(out.loc[mon, "gdelt_mention_count"]) == 5
