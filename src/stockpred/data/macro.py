"""Macro / market-state series from FRED via pandas-datareader (free, no key).

We pull a handful of series commonly useful for regime features:
- VIXCLS: CBOE VIX (equity vol)
- DGS10:  10y treasury yield
- DGS2:   2y treasury yield   (slope = DGS10 - DGS2)
- T10Y2Y: 10y-2y term spread (recession-watch)
- DFF:    federal funds effective rate
- DTWEXBGS: trade-weighted USD broad index
- DCOILWTICO: WTI crude

All series are forward-filled to trading days; only past values are ever used
(no lookahead).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from stockpred.config import CACHE_DIR

log = logging.getLogger(__name__)

CACHE_FILE = CACHE_DIR / "macro.parquet"

DEFAULT_SERIES: tuple[str, ...] = (
    "VIXCLS",
    "DGS10",
    "DGS2",
    "T10Y2Y",
    "DFF",
    "DTWEXBGS",
    "DCOILWTICO",
)


_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _fetch_one(series: str) -> pd.DataFrame | None:
    """Pull one FRED series as CSV directly. No auth, no key, no deps."""
    try:
        r = requests.get(
            _FRED_CSV.format(series=series),
            headers={
                "User-Agent": "stock-predictor/0.2 (+https://github.com/hgdsraj/stock-predictor)"
            },
            timeout=30,
        )
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        # Column names vary; the date is "DATE" or "observation_date".
        date_col = next((c for c in df.columns if c.upper() in ("DATE", "OBSERVATION_DATE")), None)
        val_col = next((c for c in df.columns if c != date_col), None)
        if date_col is None or val_col is None:
            return None
        df = df.rename(columns={date_col: "date", val_col: series})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df[series] = pd.to_numeric(df[series], errors="coerce")
        return df
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to fetch FRED %s: %s", series, e)
        return None


def fetch_macro(
    series: tuple[str, ...] = DEFAULT_SERIES,
    *,
    start: str = "2000-01-01",
    end: str | None = None,
    refresh: bool = False,
    cache_file: Path = CACHE_FILE,
) -> pd.DataFrame:
    """Return wide DataFrame indexed by date with one column per FRED series.

    Uses a direct CSV pull from fredgraph (no API key, no third-party libs).
    """
    if cache_file.exists() and not refresh:
        cached = pd.read_parquet(cache_file)
        return cached

    frames = [df for df in (_fetch_one(s) for s in series) if df is not None]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, axis=1).sort_index()
    out.index.name = "date"
    if start:
        out = out.loc[start:]
    if end:
        out = out.loc[:end]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache_file)
    return out


def align_to_trading_days(macro: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    """Forward-fill macro values onto a trading-day index.

    Critical: ffill only, never bfill. We must use macro values *known* on each
    trading day, not future values back-filled to past.
    """
    return macro.reindex(trading_days).ffill()
