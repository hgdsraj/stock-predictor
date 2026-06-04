"""yfinance OHLCV loader with on-disk parquet caching.

Design:
- One parquet per ticker under data/cache/prices/{TICKER}.parquet
- Columns: open, high, low, close, adj_close, volume (lowercase)
- Adjusted close handles splits + dividends; we use adj_close to compute returns.
- Long-form helper `panel(...)` returns a tidy DataFrame [date, ticker, ...].
- All requests are batched + throttled politely; failures are recorded and skipped.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

from stockpred.config import CACHE_DIR

log = logging.getLogger(__name__)

PRICES_DIR = CACHE_DIR / "prices"
PRICES_DIR.mkdir(parents=True, exist_ok=True)

OHLCV_COLS = ["open", "high", "low", "close", "adj_close", "volume"]


def _ticker_path(ticker: str) -> Path:
    return PRICES_DIR / f"{ticker}.parquet"


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLS)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    # yfinance >= 0.2.50 returns 'Adj Close' only when auto_adjust=False
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]
    df = df[[c for c in OHLCV_COLS if c in df.columns]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "date"
    return df.sort_index()


def fetch_one(
    ticker: str,
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    retries: int = 3,
    sleep_s: float = 0.3,
) -> pd.DataFrame:
    """Fetch a single ticker (cached). Returns empty DF on persistent failure."""
    cache = _ticker_path(ticker)
    if cache.exists() and not refresh:
        df = pd.read_parquet(cache)
        if not df.empty:
            return df

    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
            )
            if isinstance(raw.columns, pd.MultiIndex):
                # When passing a single ticker yfinance sometimes returns a multiindex.
                raw.columns = raw.columns.get_level_values(0)
            df = _normalise(raw)
            if not df.empty:
                df.to_parquet(cache)
                return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(sleep_s * (2**attempt))
    log.warning("Failed to fetch %s: %s", ticker, last_err)
    return pd.DataFrame(columns=OHLCV_COLS)


def fetch_many(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    max_workers: int = 8,
) -> dict[str, pd.DataFrame]:
    """Fetch many tickers in parallel; return dict[ticker -> OHLCV DataFrame]."""
    out: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(fetch_one, t, start=start, end=end, refresh=refresh): t for t in tickers
        }
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                out[t] = fut.result()
            except Exception as e:  # noqa: BLE001
                log.warning("Exception for %s: %s", t, e)
                out[t] = pd.DataFrame(columns=OHLCV_COLS)
            if i % 25 == 0:
                log.info("Fetched %d / %d", i, len(tickers))
    return out


def panel(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    field: str = "adj_close",
) -> pd.DataFrame:
    """Wide DataFrame of a single field, columns=tickers, index=date."""
    data = fetch_many(tickers, start=start, end=end, refresh=refresh)
    series = {t: df[field] for t, df in data.items() if not df.empty and field in df.columns}
    if not series:
        return pd.DataFrame()
    out = pd.concat(series, axis=1).sort_index()
    return out


def long_panel(
    tickers: list[str],
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Long-form DataFrame: index=[date, ticker], columns=OHLCV."""
    data = fetch_many(tickers, start=start, end=end, refresh=refresh)
    frames = []
    for t, df in data.items():
        if df.empty:
            continue
        df = df.copy()
        df["ticker"] = t
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).reset_index().set_index(["date", "ticker"]).sort_index()
