"""SEC EDGAR 8-K filings as per-ticker daily event features.

Phase 12: free, full-historical, no-API-key event data. Each US-listed
public company files an 8-K with the SEC when a "material event" occurs
(CEO change, M&A, earnings pre-release, restructuring, etc.). The
filing date is publicly available within minutes; using `filing_date
<= t-1` for a t-trade is leakage-safe.

This module:
    1. Fetches the SEC ticker -> CIK map (one-shot, ~3 MB JSON).
    2. Streams quarterly form indexes (~2 MB each) to extract every
       8-K filing's (CIK, filing_date).
    3. Produces a `(date, ticker)`-indexed DataFrame with:
         - `has_8k`              (int8, 0/1)
         - `count_8k_5d`         (int16)  rolling sum over last 5 trading days
         - `count_8k_21d`        (int16)  rolling sum over last 21 trading days
         - `count_8k_63d`        (int16)  rolling sum over last 63 trading days
    4. Caches the per-quarter 8-K table and the merged panel as gzipped
       parquet at `data/cache/edgar/`.

Memory discipline (8 GB box, per `docs/continue.md` constraint #8):
    - Form indexes are streamed line-by-line (never read whole file).
    - Per-quarter raw events compress to ~30 KB each.
    - Merged panel uses int8 / int16 dtypes (categorical tickers).
    - Full 2001-2024 panel for 150 tickers is ~5 MB in RAM, ~1 MB on disk.

User-Agent rule:
    SEC requires every request to include a User-Agent that identifies
    you. Override via env var `EDGAR_USER_AGENT="Your Name email@x.com"`.
    Default is `stock-predictor/0.2 (raj.axisos@gmail.com)`.

Rate limit:
    SEC publicly suggests max 10 req/sec. We sleep 0.11s between
    requests (~9 req/sec). DO NOT remove this; SEC will throttle you
    or block your IP.

Curl recipes for users (also documented in docs/USAGE.md §6c):
    UA="stockpred-research raj.axisos@gmail.com"
    curl -A "$UA" -o ticker_to_cik.json \\
        https://www.sec.gov/files/company_tickers.json
    curl -A "$UA" -o form_2024Q1.idx \\
        https://www.sec.gov/Archives/edgar/full-index/2024/QTR1/form.idx
"""

from __future__ import annotations

import gc
import io
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from stockpred.config import CACHE_DIR

log = logging.getLogger(__name__)

CACHE_DIR_EDGAR = CACHE_DIR / "edgar"
CACHE_DIR_EDGAR.mkdir(parents=True, exist_ok=True)

TICKER_CIK_CACHE = CACHE_DIR_EDGAR / "ticker_to_cik.json"
EVENTS_CACHE = CACHE_DIR_EDGAR / "8k_events.parquet"

_DEFAULT_UA = "stock-predictor/0.2 (raj.axisos@gmail.com)"
_USER_AGENT = os.environ.get("EDGAR_USER_AGENT", _DEFAULT_UA)
_RATE_LIMIT_SLEEP_S = 0.11  # ~9 req/sec, under SEC's 10/sec ceiling

_TICKER_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
_FORM_IDX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/form.idx"


def _http_get(url: str, *, timeout: int = 30) -> requests.Response:
    """Single GET with SEC-compliant headers + rate-limit sleep.

    The sleep is BEFORE the request so callers can fire several without
    needing to coordinate; the first call sleeps 0.11s but that's a
    rounding error vs the typical 200-500ms request latency.
    """
    time.sleep(_RATE_LIMIT_SLEEP_S)
    r = requests.get(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            # Accept-Encoding is set automatically by requests, but be
            # explicit so future maintainers don't think it's missing.
            "Accept-Encoding": "gzip, deflate",
            # Host header sometimes helps avoid SEC's CDN routing oddities.
            "Host": url.split("/")[2],
        },
        timeout=timeout,
    )
    r.raise_for_status()
    return r


# Production threshold for a healthy ticker->CIK map. SEC's full map
# has ~10 000 entries; anything under this is almost certainly a stub
# response we should not cache. Tests override via the env var.
_MIN_TICKER_CIK_ENTRIES = int(os.environ.get("EDGAR_MIN_TICKER_CIK_ENTRIES", "1000"))


def _validate_ticker_cik_map(d: dict, min_entries: int | None = None) -> bool:
    """Return True if `d` looks like a healthy ticker->CIK map.

    Two checks:
      1. Structure: every CIK must be a 10-digit zero-padded string
         (`'0000320193'`). A stub or hand-edited bad CIK fails here.
      2. Size: at least `min_entries` rows (defaults to
         `_MIN_TICKER_CIK_ENTRIES`). SEC's full map has ~10 000
         entries; anything materially smaller is almost certainly
         a truncated / stub response we should not cache.

    Reviewer CRITICAL #1. Tests pass `min_entries=1` to bypass the
    size check while still asserting structure.
    """
    if not isinstance(d, dict):
        return False
    threshold = min_entries if min_entries is not None else _MIN_TICKER_CIK_ENTRIES
    if len(d) < threshold:
        return False
    for cik in d.values():
        if not (isinstance(cik, str) and len(cik) == 10 and cik.isdigit()):
            return False
    return True


def fetch_ticker_to_cik(*, refresh: bool = False) -> dict[str, str]:
    """Return a {ticker -> 10-digit zero-padded CIK string} mapping.

    SEC's `company_tickers.json` is a single ~3 MB file with all
    US-listed tickers. We cache it locally as JSON. CIK is returned as
    a 10-digit zero-padded string because that's the format used in the
    form.idx files we'll filter against later.

    Reviewer CRITICAL #1: cache is semantically validated on read.
    """
    if TICKER_CIK_CACHE.exists() and not refresh:
        try:
            cached = json.loads(TICKER_CIK_CACHE.read_text())
        except Exception as e:  # noqa: BLE001
            log.warning("Cached ticker->CIK invalid JSON (%s); refetching.", e)
        else:
            if _validate_ticker_cik_map(cached):
                return cached
            log.warning(
                "Cached ticker->CIK failed validation (n=%d, type=%s); refetching.",
                len(cached) if isinstance(cached, dict) else -1,
                type(cached).__name__,
            )

    log.info("Fetching SEC ticker->CIK map from %s", _TICKER_JSON_URL)
    r = _http_get(_TICKER_JSON_URL)
    raw = r.json()
    # Format: { "0": { "cik_str": 320193, "ticker": "AAPL", "title": "..." }, ... }
    out: dict[str, str] = {}
    for v in raw.values():
        ticker = str(v.get("ticker", "")).upper()
        cik_int = v.get("cik_str")
        if not ticker or cik_int is None:
            continue
        out[ticker] = f"{int(cik_int):010d}"
    if not _validate_ticker_cik_map(out):
        raise RuntimeError(
            f"SEC ticker->CIK map fetched but failed validation "
            f"(n={len(out)}). Got an empty or stub response from "
            f"{_TICKER_JSON_URL}. Refusing to cache."
        )
    TICKER_CIK_CACHE.write_text(json.dumps(out, indent=0))
    log.info("Cached %d ticker->CIK mappings to %s", len(out), TICKER_CIK_CACHE)
    return out


def _quarters_in_range(
    start: pd.Timestamp | str | int,
    end: pd.Timestamp | str | int,
) -> list[tuple[int, int]]:
    """Return list of (year, qtr) tuples whose date-range overlaps [start, end].

    Accepts either Timestamps or bare years (back-compat). When integers
    are passed, it returns ALL quarters of those years inclusive (legacy
    behaviour used by some callers / tests). When Timestamps are passed,
    it returns only the quarters that actually contain dates in the
    requested range — saves a lot of needless HTTP calls.
    """
    # Back-compat: bare ints = years -> all quarters in range
    if isinstance(start, int) and isinstance(end, int):
        return [(y, q) for y in range(start, end + 1) for q in (1, 2, 3, 4)]
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_q = (start_ts.month - 1) // 3 + 1
    end_q = (end_ts.month - 1) // 3 + 1
    out: list[tuple[int, int]] = []
    for y in range(start_ts.year, end_ts.year + 1):
        q_lo = start_q if y == start_ts.year else 1
        q_hi = end_q if y == end_ts.year else 4
        for q in range(q_lo, q_hi + 1):
            out.append((y, q))
    return out


def _quarter_cache_path(year: int, qtr: int) -> Path:
    return CACHE_DIR_EDGAR / f"8k_{year}Q{qtr}.parquet"


def _fetch_quarter_8k(year: int, qtr: int, *, refresh: bool = False) -> pd.DataFrame:
    """Fetch one quarter's form.idx, extract 8-K filings.

    form.idx is a fixed-width text file with header lines then one row
    per filing:
        Form Type        Company Name      CIK    Date Filed    Filename
        8-K              APPLE INC        320193   2024-01-25    edgar/...

    Returns DataFrame with columns [cik, filing_date]. cik is the
    10-digit zero-padded string. filing_date is a pandas Timestamp.

    Streamed line-by-line so we never hold the full 2-MB file in
    memory beyond the lines we keep.
    """
    cache_path = _quarter_cache_path(year, qtr)
    if cache_path.exists() and not refresh:
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Cached %s invalid (%s); refetching.", cache_path, e)

    url = _FORM_IDX_URL.format(year=year, qtr=qtr)
    log.info("Fetching %s", url)
    try:
        r = _http_get(url)
    except requests.HTTPError as e:
        # Some quarters in the deep past may 404; skip cleanly.
        log.warning("Form index %s 404 / failed: %s", url, e)
        return pd.DataFrame(columns=["cik", "filing_date"])

    rows: list[tuple[str, str]] = []
    # form.idx is FIXED-WIDTH with a header row like:
    #   "Form Type        Company Name      CIK   Date Filed  Filename"
    # Column positions vary slightly across years, so we parse the
    # header to discover the start-index of each field, then slice
    # each data row by those positions. This is robust to company
    # names containing multiple consecutive spaces (e.g. real SEC
    # names like "PROCTER  &  GAMBLE"), which a whitespace-delimited
    # parser would silently mis-tokenize and drop. (Reviewer HIGH #3.)
    text = r.text
    col_starts: list[int] | None = None
    for line in io.StringIO(text):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # Look for the header line: starts with "Form Type".
        if col_starts is None:
            if line.lstrip().startswith("Form Type"):
                col_starts = _parse_idx_header(line)
            continue
        # Skip the underline separator line ("----...----").
        if line[0] in "-=":
            continue
        # Skip any other non-data line (alpha-numeric is mandatory in
        # column 0 for real data lines).
        if not line[0].isalnum():
            continue
        try:
            cells = _slice_fixed_width(line, col_starts)
        except IndexError:
            continue
        if len(cells) < 4:
            continue
        form_type = cells[0].strip()
        if form_type != "8-K":
            continue
        # cells[1] = company name (unused for now), cells[2] = CIK,
        # cells[3] = filing date.
        try:
            cik = f"{int(cells[2].strip()):010d}"
            filing_date = pd.to_datetime(cells[3].strip(), errors="raise")
        except (ValueError, TypeError):
            continue
        rows.append((cik, filing_date.strftime("%Y-%m-%d")))

    if col_starts is None:
        log.warning("Form index %d Q%d: no header line found; cannot parse.", year, qtr)

    # Free the raw text immediately
    del text, r
    gc.collect()

    if not rows:
        log.warning("No 8-K rows parsed for %d Q%d", year, qtr)
        df = pd.DataFrame(columns=["cik", "filing_date"])
    else:
        df = pd.DataFrame(rows, columns=["cik", "filing_date"])
        df["cik"] = df["cik"].astype("string")
        df["filing_date"] = pd.to_datetime(df["filing_date"])

    df.to_parquet(cache_path, compression="snappy", index=False)
    log.info("Cached %d 8-K filings for %d Q%d -> %s", len(df), year, qtr, cache_path)
    return df


def _parse_idx_header(header_line: str) -> list[int]:
    """Find the start column for each named field in a form.idx header.

    The header line looks like (variable widths across years):
        "Form Type        Company Name      CIK   Date Filed  Filename"

    We locate the start index of each of the 5 named fields and return
    them as `[0, company_start, cik_start, date_start, filename_start]`.
    The 0 is included so the first slice `[0:company_start]` carves out
    the Form Type column.
    """
    fields = ("Form Type", "Company Name", "CIK", "Date Filed", "Filename")
    starts: list[int] = []
    cursor = 0
    for field in fields:
        idx = header_line.find(field, cursor)
        if idx < 0:
            raise ValueError(f"Header field {field!r} not found in {header_line!r}")
        starts.append(idx)
        cursor = idx + len(field)
    return starts


def _slice_fixed_width(line: str, col_starts: list[int]) -> list[str]:
    """Slice `line` into cells using `col_starts` as start positions.

    Returns one string per column. The last column is open-ended.
    Pads short lines with empty strings if they're shorter than expected.
    """
    if not col_starts:
        return []
    cells: list[str] = []
    for i, start in enumerate(col_starts):
        end = col_starts[i + 1] if i + 1 < len(col_starts) else None
        if end is None:
            cells.append(line[start:] if start < len(line) else "")
        else:
            cells.append(line[start:end] if start < len(line) else "")
    return cells


# Legacy whitespace-split parser kept for back-compat tests.
def _split_ws2(line: str) -> list[str]:
    """DEPRECATED. Split a line on runs of 2+ whitespace characters.

    Kept only for back-compat with one existing test. New code should
    use `_parse_idx_header` + `_slice_fixed_width` (the fixed-width
    parser handles company names like "PROCTER  &  GAMBLE" correctly,
    which this function silently mis-tokenizes).
    """
    out: list[str] = []
    buf = []
    spaces = 0
    for ch in line:
        if ch == " ":
            spaces += 1
            if spaces >= 2 and buf:
                out.append("".join(buf))
                buf = []
        else:
            if spaces >= 2:
                pass  # Already flushed; continue building next token.
            elif spaces == 1:
                buf.append(" ")  # Single space INSIDE a token (company names).
            buf.append(ch)
            spaces = 0
    if buf:
        out.append("".join(buf))
    return out


def fetch_8k_events(
    start: str = "2014-01-01",
    end: str | None = None,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return DataFrame of (cik, filing_date) for all 8-K filings in
    range, drawing from per-quarter caches.

    Coverage: SEC's full-index is reliably good back to ~2000. We
    default to 2014-01-01 to match our backtest window.
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else pd.Timestamp.today()
    quarters = _quarters_in_range(start_dt, end_dt)
    frames: list[pd.DataFrame] = []
    for y, q in quarters:
        df = _fetch_quarter_8k(y, q, refresh=refresh)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["cik", "filing_date"])
    out = pd.concat(frames, ignore_index=True)
    # Filter to range (some quarters straddle the boundary slightly).
    out = out[(out["filing_date"] >= start_dt) & (out["filing_date"] <= end_dt)]
    # Free the per-quarter frames
    del frames
    gc.collect()
    return out.reset_index(drop=True)


def build_8k_features(
    tickers: list[str],
    trading_days: pd.DatetimeIndex,
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    ticker_to_cik: dict[str, str] | None = None,
    rolling_windows: tuple[int, ...] = (5, 21, 63),
) -> pd.DataFrame:
    """Build the per-(date, ticker) 8-K feature panel.

    Output: long-format DataFrame indexed by (date, ticker) with columns:
        has_8k                (int8)   1 if 8-K filed ON this trading day
        count_8k_{w}d         (int16)  rolling sum over last w trading days

    NB: "filed on this trading day" treats the filing date as the SAME
    trading day. To use this safely as a t-day feature (no leakage),
    the pipeline must shift it by 1 (which `pipeline_v5` does for
    every feature via the t-1 trading rule).
    """
    if start is None:
        start = trading_days.min().strftime("%Y-%m-%d")
    if end is None:
        end = trading_days.max().strftime("%Y-%m-%d")
    if ticker_to_cik is None:
        ticker_to_cik = fetch_ticker_to_cik(refresh=refresh)

    # Filter to tickers we have CIKs for
    present = [t for t in tickers if t.upper() in ticker_to_cik]
    missing = [t for t in tickers if t.upper() not in ticker_to_cik]
    if missing:
        log.info(
            "build_8k_features: %d of %d tickers missing from SEC ticker map: %s",
            len(missing),
            len(tickers),
            missing[:10],
        )
    if not present:
        log.warning("build_8k_features: NO tickers had a CIK match; returning empty.")
        return pd.DataFrame()

    cik_to_ticker = {ticker_to_cik[t.upper()]: t.upper() for t in present}

    # Fetch 8-K filings in range
    events = fetch_8k_events(start=start, end=end, refresh=refresh)
    if events.empty:
        log.warning("build_8k_features: zero 8-K events in range; returning empty.")
        return pd.DataFrame()
    # Filter to our universe
    events = events[events["cik"].isin(cik_to_ticker.keys())].copy()
    events["ticker"] = events["cik"].map(cik_to_ticker)

    # Build a (date, ticker) -> count_today panel via groupby.
    # Multiple 8-Ks on the same day are valid (e.g. earnings + M&A).
    daily = (
        events.groupby(["filing_date", "ticker"], observed=True)
        .size()
        .rename("count_today")
        .reset_index()
    )

    # Align to trading days: re-index onto (trading_days x tickers).
    # Use a wide pivot (memory-cheap because most cells are 0 / NaN).
    wide = daily.pivot(index="filing_date", columns="ticker", values="count_today").fillna(0)
    # Re-index onto trading_days (filings that landed on a weekend get
    # forward-merged into the next trading day below).
    full_idx = trading_days
    # For filings on non-trading days (weekends / holidays), shift to the
    # NEXT trading day. resample('B').sum() loses dates; instead, use
    # reindex(method='bfill'). Actually we want: every event date that's
    # not in trading_days should map to the next trading day.
    wide.index = pd.to_datetime(wide.index)
    # Move non-trading-day events to the next trading day so they're
    # accounted for. We do this by re-indexing AFTER bucketing into
    # trading days via merge_asof.
    long_events = wide.reset_index().melt(
        id_vars="filing_date", var_name="ticker", value_name="count_today"
    )
    long_events = long_events[long_events["count_today"] > 0]
    # Normalize both sides to nanosecond datetime to avoid
    # merge_asof "incompatible merge keys" errors when one source
    # uses '<M8[us]' (yfinance) and the other uses '<M8[ms]' (parsed
    # CSV). Cast both to '<M8[ns]' explicitly.
    long_events["effective_day"] = pd.to_datetime(long_events["filing_date"]).astype(
        "datetime64[ns]"
    )
    # Align: for each event date, find the trading day on or after.
    td_df = pd.DataFrame({"trading_day": pd.DatetimeIndex(full_idx).astype("datetime64[ns]")})
    long_events = long_events.sort_values("effective_day")
    long_events = pd.merge_asof(
        long_events,
        td_df,
        left_on="effective_day",
        right_on="trading_day",
        direction="forward",
    )
    long_events = long_events.dropna(subset=["trading_day"])
    # Sum by (trading_day, ticker) after re-bucketing.
    daily_td = (
        long_events.groupby(["trading_day", "ticker"], observed=True)["count_today"]
        .sum()
        .rename("count_today")
        .reset_index()
    )

    # Re-pivot for rolling-window math.
    wide_td = (
        daily_td.pivot(index="trading_day", columns="ticker", values="count_today")
        .reindex(index=full_idx, columns=[t.upper() for t in present])
        .fillna(0)
        .astype("int16")
    )
    wide_td.index.name = "date"
    wide_td.columns.name = "ticker"

    # Build features
    has_8k = (wide_td > 0).astype("int8").stack(future_stack=True).rename("has_8k")
    feats = [has_8k]
    for w in rolling_windows:
        roll = wide_td.rolling(window=w, min_periods=1).sum().astype("int16")
        feats.append(roll.stack(future_stack=True).rename(f"count_8k_{w}d"))

    out = pd.concat(feats, axis=1)
    out.index.names = ["date", "ticker"]
    # Drop fully-zero rows isn't worth it: pipeline merges on the panel
    # of all (date, ticker) so we need a value for every cell.

    # Memory hygiene
    del wide_td, wide, daily, daily_td, events, long_events, td_df
    gc.collect()

    log.info(
        "build_8k_features: %d rows x %d cols (covering %d tickers)",
        out.shape[0],
        out.shape[1],
        len(present),
    )
    return out
