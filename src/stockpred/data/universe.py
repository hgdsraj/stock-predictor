"""S&P 500 historical constituents.

We reconstruct a point-in-time view of the index by parsing the current
constituents and the change log from Wikipedia, then walking the log to determine
membership at any past date.

Caveats:
- Wikipedia's change log is reliable from ~1995 onward but is not authoritative.
- Ticker symbol changes (e.g. FB -> META) are normalised to the *current* yfinance
  symbol; this is sufficient for back-prices via yfinance which generally maps the
  history correctly.
- We persist a snapshot to data/cache/sp500_membership.parquet so runs are
  reproducible offline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from stockpred.config import CACHE_DIR, DEFAULT

log = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "sp500_membership.parquet"

Interval = tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]


def _read_html_tables(url: str) -> list[pd.DataFrame]:
    """Fetch `url` and parse all HTML tables.

    `pd.read_html` interprets bare strings as file paths, so we wrap the
    response body in a StringIO. We also send a friendly User-Agent because
    Wikipedia and similar sites block default urllib UAs.
    """
    import io as _io

    import requests

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; stock-predictor/0.1; "
            "+https://github.com/hgdsraj/stock-predictor)"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return pd.read_html(_io.StringIO(resp.text))


def fetch_sp500_membership(
    url: str | None = None,
    *,
    refresh: bool = False,
    cache_file: Path = CACHE_FILE,
) -> pd.DataFrame:
    """Return a long DataFrame with columns: [ticker, start_date, end_date].

    `end_date` is `NaT` for currently-included tickers.
    `start_date` is `NaT` if the ticker pre-dates the Wikipedia change log.
    """
    if cache_file.exists() and not refresh:
        return pd.read_parquet(cache_file)

    url = url or DEFAULT.universe.sp500_changes_url
    log.info("Fetching S&P 500 constituents from %s", url)
    tables = _read_html_tables(url)

    if len(tables) < 2:
        raise RuntimeError(
            "Unexpected Wikipedia layout: expected at least 2 tables, "
            f"got {len(tables)}. The page may have changed."
        )

    current = tables[0].copy()
    changes = tables[1].copy()

    # ----- normalise current constituents -----
    # Some snapshots have MultiIndex columns; flatten first.
    if isinstance(current.columns, pd.MultiIndex):
        current.columns = ["_".join(str(c) for c in tup if c).strip() for tup in current.columns]
    if "Symbol" in current.columns:
        sym_col = "Symbol"
    else:
        sym_col = next(
            (c for c in current.columns if "symbol" in str(c).lower()),
            current.columns[0],
        )
    current = current.rename(columns={sym_col: "ticker"})
    current["ticker"] = current["ticker"].astype(str).str.strip().str.replace(".", "-", regex=False)
    current_tickers = set(current["ticker"].unique())

    # ----- normalise changes table -----
    if isinstance(changes.columns, pd.MultiIndex):
        changes.columns = [
            "_".join(str(c) for c in tup if c).strip().lower() for tup in changes.columns
        ]
    else:
        changes.columns = [str(c).strip().lower() for c in changes.columns]

    date_col = next((c for c in changes.columns if "date" in c), None)
    added_col = next((c for c in changes.columns if "added" in c and "ticker" in c), None)
    removed_col = next((c for c in changes.columns if "removed" in c and "ticker" in c), None)
    if not (date_col and added_col and removed_col):
        raise RuntimeError(
            "Could not find expected columns in S&P 500 changes table. "
            f"Saw columns: {list(changes.columns)}"
        )

    changes = changes[[date_col, added_col, removed_col]].rename(
        columns={date_col: "date", added_col: "added", removed_col: "removed"}
    )
    changes["date"] = pd.to_datetime(changes["date"], errors="coerce")
    changes = changes.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for c in ("added", "removed"):
        changes[c] = (
            changes[c]
            .astype(str)
            .str.strip()
            .str.replace(".", "-", regex=False)
            .replace({"nan": pd.NA, "": pd.NA, "None": pd.NA})
        )

    # ----- reconstruct lifetimes -----
    add_by_t: dict[str, list[pd.Timestamp]] = (
        changes.dropna(subset=["added"]).groupby("added")["date"].apply(list).to_dict()
    )
    rem_by_t: dict[str, list[pd.Timestamp]] = (
        changes.dropna(subset=["removed"]).groupby("removed")["date"].apply(list).to_dict()
    )

    all_tickers = set(current_tickers) | set(add_by_t) | set(rem_by_t)
    rows: list[dict] = []
    for t in sorted(all_tickers):
        adds = sorted(add_by_t.get(t, []))
        rems = sorted(rem_by_t.get(t, []))
        is_current = t in current_tickers

        intervals: list[Interval] = []
        open_start: pd.Timestamp | None = None
        events: list[tuple[pd.Timestamp, str]] = [(d, "add") for d in adds] + [
            (d, "rem") for d in rems
        ]
        events.sort()
        for d, kind in events:
            if kind == "add" and open_start is None:
                open_start = d
            elif kind == "rem" and open_start is not None:
                intervals.append((open_start, d))
                open_start = None
            elif kind == "rem" and open_start is None:
                # Removed without a recorded add => assume in index since
                # before our change log began.
                intervals.append((None, d))

        if open_start is not None:
            # An add with no subsequent remove.
            intervals.append((open_start, None if is_current else None))
        elif is_current and not intervals:
            # In index now, no events at all -> in since before the log.
            intervals.append((None, None))
        elif is_current and intervals and intervals[-1][1] is not None:
            # Currently in index but log shows we were removed -> re-added
            # without a recorded event. Open a new interval from the removal
            # date (conservative; underestimates true tenure).
            intervals.append((intervals[-1][1], None))

        for s, e in intervals:
            rows.append({"ticker": t, "start_date": s, "end_date": e})

    out = pd.DataFrame(rows)
    out["start_date"] = pd.to_datetime(out["start_date"])
    out["end_date"] = pd.to_datetime(out["end_date"])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_file, index=False)
    log.info(
        "Cached %d membership intervals for %d tickers -> %s",
        len(out),
        out["ticker"].nunique(),
        cache_file,
    )
    return out


def members_on(date: str | pd.Timestamp, membership: pd.DataFrame | None = None) -> list[str]:
    """Return tickers that were in the S&P 500 at the close of `date`.

    Boundary convention (H3 fix):
      * `start_date <= date` (inclusive): a ticker added on date d is in the
        index at the close of d.
      * `end_date > date` (strict): a ticker removed on date d is NOT in the
        index at the close of d.

    Using `>=` here would leak forward information about delistings, because
    the membership change is announced before market open of the removal day.
    """
    if membership is None:
        membership = fetch_sp500_membership()
    d = pd.Timestamp(date)
    mask = (membership["start_date"].isna() | (membership["start_date"] <= d)) & (
        membership["end_date"].isna() | (membership["end_date"] > d)
    )
    return sorted(membership.loc[mask, "ticker"].unique().tolist())


def all_tickers_in_range(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    membership: pd.DataFrame | None = None,
) -> list[str]:
    """Return every ticker that was a member at any point in [start, end]."""
    if membership is None:
        membership = fetch_sp500_membership()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else pd.Timestamp(datetime.now(timezone.utc).date())
    mask = (membership["start_date"].isna() | (membership["start_date"] <= end_ts)) & (
        membership["end_date"].isna() | (membership["end_date"] >= start_ts)
    )
    return sorted(membership.loc[mask, "ticker"].unique().tolist())
