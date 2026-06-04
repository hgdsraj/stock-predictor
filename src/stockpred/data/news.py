"""Headline-level news from yfinance `Ticker.news` — free, no API key.

What this gives you:
    title, publisher, link, providerPublishTime (UTC ts), type

What this does NOT give you (intentionally — free-data ceiling):
    body text, entity tags, sentiment score, ground-truth event labels

We persist a per-ticker rolling window of the most recent N items. The
backend can surface them on the ticker detail page; the model does NOT use
them as features — see docs/CONCEPTS.md §7 on look-ahead in features.

If you want to add news as a feature, you must:
    1. Use only items with `providerPublishTime < signal_timestamp`.
    2. Confirm the timestamp is when the article became publicly visible,
       not when yfinance scraped it (the latter can be hours later).
    3. Build event flags (earnings, downgrade, M&A) rather than raw
       sentiment — see the strategy-research sub-agent's report
       (docs/PROJECT_LOG.md "Phase 5 research note") for what's
       empirically defensible.
"""

from __future__ import annotations

import datetime as dt
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import yfinance as yf

from stockpred.config import CACHE_DIR

log = logging.getLogger(__name__)

CACHE_DIR_NEWS = CACHE_DIR / "news"
CACHE_DIR_NEWS.mkdir(parents=True, exist_ok=True)


def _normalise_one(item: dict) -> dict:
    """yfinance `Ticker.news` returns items in two different shapes depending
    on the version. We handle both."""
    link = item.get("link") or (item.get("content", {}).get("canonicalUrl") or {}).get("url")
    # Defence in depth: only persist http(s) links so a malicious
    # `javascript:` URL can't make it into the SPA's <a href=...>.
    if link and not (isinstance(link, str) and link.startswith(("http://", "https://"))):
        link = None
    out = {
        "title": item.get("title") or item.get("content", {}).get("title"),
        "publisher": (
            item.get("publisher")
            or (item.get("content", {}).get("provider") or {}).get("displayName")
        ),
        "link": link,
        "type": item.get("type") or item.get("content", {}).get("contentType"),
        "uuid": item.get("uuid") or item.get("id"),
    }
    ts = item.get("providerPublishTime")
    if ts is None:
        pub = item.get("content", {}).get("pubDate")
        if pub:
            try:
                out["published_at"] = pd.Timestamp(pub).to_pydatetime()
            except Exception:  # noqa: BLE001
                out["published_at"] = None
        else:
            out["published_at"] = None
    else:
        try:
            out["published_at"] = dt.datetime.fromtimestamp(int(ts), dt.timezone.utc).replace(
                tzinfo=None
            )
        except Exception:  # noqa: BLE001
            out["published_at"] = None
    return out


def fetch_one(
    ticker: str,
    *,
    max_items: int = 20,
    refresh: bool = False,
    sleep_s: float = 0.2,
) -> list[dict]:
    """Return up to `max_items` recent news items for one ticker.

    Cached per ticker to a parquet file. The cache is overwritten each
    refresh (we don't accumulate; yfinance only exposes recent items
    anyway).
    """
    cache_file = CACHE_DIR_NEWS / f"{ticker}.parquet"
    if cache_file.exists() and not refresh:
        try:
            df = pd.read_parquet(cache_file)
            if not df.empty:
                return df.head(max_items).to_dict("records")
        except Exception as e:  # noqa: BLE001
            log.warning("news cache read failed for %s: %s", ticker, e)

    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:  # noqa: BLE001
        log.warning("news fetch failed for %s: %s", ticker, e)
        return []

    items = [_normalise_one(x) for x in raw[:max_items]]
    if items:
        try:
            pd.DataFrame(items).to_parquet(cache_file, index=False)
        except Exception as e:  # noqa: BLE001
            log.warning("news cache write failed for %s: %s", ticker, e)
    time.sleep(sleep_s)
    return items


def fetch_many(
    tickers: list[str],
    *,
    max_items: int = 20,
    refresh: bool = False,
    max_workers: int = 4,
) -> dict[str, list[dict]]:
    """Parallel news fetch. Polite (4 workers, 200 ms sleep per call)."""
    out: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(fetch_one, t, max_items=max_items, refresh=refresh): t for t in tickers
        }
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                out[t] = fut.result()
            except Exception as e:  # noqa: BLE001
                log.warning("news exception for %s: %s", t, e)
                out[t] = []
    return out
