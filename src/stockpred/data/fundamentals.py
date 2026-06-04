"""Per-ticker static fundamentals + sector lookup via yfinance.

This is a *best-effort* loader. yfinance `info` is unreliable: it returns
inconsistent shapes, frequently throttles, sometimes silently returns an empty
dict. We tolerate all failures.

Cached to data/cache/fundamentals.parquet so subsequent runs are offline.

Fields we try to capture (when available):
    sector, industry, marketCap, shortRatio, shortPercentOfFloat,
    beta, trailingPE, forwardPE, dividendYield, fiftyTwoWeekHigh,
    fiftyTwoWeekLow, longBusinessSummary
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

CACHE_FILE = CACHE_DIR / "fundamentals.parquet"

FIELDS: tuple[str, ...] = (
    "sector",
    "industry",
    "marketCap",
    "shortRatio",
    "shortPercentOfFloat",
    "beta",
    "trailingPE",
    "forwardPE",
    "dividendYield",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "longBusinessSummary",
)


def _fetch_one(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:  # noqa: BLE001
        log.debug("fundamentals %s failed: %s", ticker, e)
        info = {}
    out = {"ticker": ticker}
    for f in FIELDS:
        v = info.get(f)
        out[f] = v if v is not None else pd.NA
    return out


def fetch_fundamentals(
    tickers: list[str],
    *,
    refresh: bool = False,
    cache_file: Path = CACHE_FILE,
    max_workers: int = 8,
) -> pd.DataFrame:
    """Return DataFrame indexed by ticker with the FIELDS columns.

    Falls back to cached data on errors. Refreshes per-ticker rows individually
    so a single failure doesn't poison the whole cache.
    """
    cached = pd.DataFrame()
    if cache_file.exists():
        cached = pd.read_parquet(cache_file)
        if "ticker" in cached.columns:
            cached = cached.set_index("ticker")

    if not refresh and not cached.empty:
        missing = [t for t in tickers if t not in cached.index]
    else:
        missing = list(tickers)

    if not missing:
        return cached.reindex(tickers)

    log.info("Fetching fundamentals for %d tickers...", len(missing))
    rows: list[dict] = []
    # L7 fix: rate-limit by *submitting* slowly, not by sleeping after results
    # are already in flight.
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures: dict = {}
        for j, t in enumerate(missing):
            if j and j % max_workers == 0:
                time.sleep(0.2)  # spread submissions
            futures[ex.submit(_fetch_one, t)] = t
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                rows.append(fut.result())
            except Exception as e:  # noqa: BLE001
                log.warning("fundamentals exception for %s: %s", futures[fut], e)
            if i % 25 == 0:
                log.info("  ... %d / %d", i, len(missing))

    new_df = pd.DataFrame(rows).set_index("ticker")
    if not cached.empty:
        merged = pd.concat([cached, new_df])
        merged = merged[~merged.index.duplicated(keep="last")]
    else:
        merged = new_df

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    merged.reset_index().to_parquet(cache_file, index=False)
    return merged.reindex(tickers)


def sector_map(fundamentals: pd.DataFrame) -> dict[str, str]:
    if "sector" not in fundamentals.columns:
        return {}
    s = fundamentals["sector"].dropna().astype(str)
    return s.to_dict()
