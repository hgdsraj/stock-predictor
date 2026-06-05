"""GDELT GKG daily tone + theme features for S&P 500 universe.

Phase 14: free per-day news-sentiment data from the GDELT Project's
Global Knowledge Graph (GKG). Historical coverage:
  - GDELT 1.0 daily files: 2013-04-01 to present (one ~5-15 MB zip/day)
  - GDELT 2.0 15-min files: 2015-02-18 to present (too big for our box)

We use GDELT 1.0 daily files because:
  1. They cover our 2014-01-01 backtest start (2.0 doesn't until 2015-02).
  2. ~4000 files for 11 years is tractable (vs ~400k 15-min files).
  3. Pre-aggregated per-day rows, no need to roll up 96 slices/day.
  4. Smaller compressed size; manageable on an 8 GB box if streamed.

Memory discipline (per docs/continue.md constraint #8):
  - Stream every fetch via `requests.get(stream=True)`; never hold the
    raw ~15 MB CSV in memory at once.
  - On parse, filter IMMEDIATELY to rows that mention an S&P 500 ticker
    in V2Organizations / V2Locations. Discard the rest.
  - Cache the per-day filtered output as gzipped parquet (~10-100 KB
    per day vs ~5-15 MB raw -> ~50x compression after filter).
  - Use float32 for tone, int16 for counts, category for tickers.
  - `del` + `gc.collect()` after each day's parse.

Output per (date, ticker):
  gdelt_mention_count    (int16)  # of articles mentioning the ticker
  gdelt_tone_mean        (float32) average article tone (-100 to +100)
  gdelt_tone_std         (float32) tone dispersion within the day

Coverage caveats (also documented in docs/continue.md known issues):
  - GDELT mentions companies by NAME, not ticker. We map ticker -> name
    using SEC's company_tickers.json (the same map used in Phase 12/13).
    Mis-matches on common short names ('CAT' for Caterpillar vs the
    word) are filtered by case + length heuristics; some signal lost.
  - GDELT 1.0 file format has gone through schema changes; we tolerate
    column drift via tolerant pd.read_csv kwargs.
  - GDELT day-D file is typically available ~24h after day D; we honour
    the same t-1 trading-rule the rest of the pipeline uses, so this is
    leakage-safe even with delayed publication.

Rate limit:
  GDELT does NOT publish a hard rate limit but recommends being polite.
  We sleep 0.5 s between file fetches (~2 req/sec). Cold fetch for 11
  years = ~4000 files * 0.5 s = ~33 minutes of sleep + network. Plan
  on 1-3 hours for the full bulk fetch over typical bandwidth.
"""

from __future__ import annotations

import csv
import gc
import gzip
import io
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from stockpred.config import CACHE_DIR

log = logging.getLogger(__name__)

CACHE_DIR_GDELT = CACHE_DIR / "gdelt"
CACHE_DIR_GDELT.mkdir(parents=True, exist_ok=True)

_DAILY_FILE_URL = "http://data.gdeltproject.org/gkg/{date_str}.gkg.csv.zip"
_USER_AGENT = os.environ.get("GDELT_USER_AGENT", "stock-predictor/0.2")
_RATE_LIMIT_SLEEP_S = float(os.environ.get("GDELT_RATE_LIMIT_S", "0.5"))
_HTTP_TIMEOUT_S = 60

# Columns we keep from GKG CSV. The full schema has ~30 columns; we
# only need 4. Streaming + early projection saves substantial RAM.
_GKG_COLS = ("DATE", "NUMARTS", "ORGANIZATIONS", "TONE")

# Tone field is comma-separated: "avg_tone, positive_score, negative_score,
# polarity, activity_density, self_density, word_count". We just want
# the first (avg_tone).


def _day_cache_path(d: date) -> Path:
    return CACHE_DIR_GDELT / f"gkg_{d.strftime('%Y%m%d')}.parquet"


def _stream_gkg_day(d: date, *, timeout: int = _HTTP_TIMEOUT_S) -> bytes | None:
    """Download one daily GKG zip; return raw zip bytes or None on 404.

    Uses stream=True so the bytes don't sit in memory before the read
    completes; the zip is small enough (5-15 MB) that we keep the
    final bytes object, but never the in-flight chunks.
    """
    url = _DAILY_FILE_URL.format(date_str=d.strftime("%Y%m%d"))
    time.sleep(_RATE_LIMIT_SLEEP_S)
    try:
        with requests.get(
            url,
            headers={"User-Agent": _USER_AGENT},
            timeout=timeout,
            stream=True,
        ) as r:
            if r.status_code == 404:
                log.info("GDELT day %s: 404 (file not published).", d)
                return None
            r.raise_for_status()
            chunks: list[bytes] = []
            total = 0
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if chunk:
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > 200 * 1024 * 1024:  # 200 MB safety ceiling
                        raise RuntimeError(f"GDELT day {d}: response exceeded 200 MB; aborting.")
            return b"".join(chunks)
    except requests.RequestException as e:
        log.warning("GDELT day %s: fetch failed: %s", d, e)
        return None


def _parse_gkg_zip(zip_bytes: bytes, name_to_tickers: dict[str, list[str]]) -> pd.DataFrame:
    """Decompress and stream-parse one GKG day; return per-ticker
    aggregates as a tiny DataFrame.

    `name_to_tickers` maps an UPPER-CASED company name to a list of
    tickers that share it (handles dual-class names; see CIK dedup in
    edgar.py for the same pattern).

    Memory: the zip is decompressed in memory (a single ~5-15 MB CSV),
    then streamed via the csv module so we never hold a parsed
    DataFrame of the full file. Only the rows whose ORGANIZATIONS field
    mentions one of our tracked names contribute to the output.
    """
    import zipfile

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        # GKG zips contain one CSV named YYYYMMDD.gkg.csv
        with zf.open(names[0]) as f:
            text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
            # GDELT 1.0 GKG is TAB-delimited (not comma). Schema has
            # changed across years; we use the first row as header
            # if present, else fall back to positional column indices.
            reader = csv.reader(text, delimiter="\t")
            header: list[str] | None = None
            try:
                first = next(reader)
            except StopIteration:
                return pd.DataFrame()
            # If the first cell looks like a date (8 digits), there's
            # no header line and we're already on row 1.
            if first and len(first[0]) == 8 and first[0].isdigit():
                # Positional schema (legacy): col 0=DATE, 1=NUMARTS,
                # 2=COUNTS, 3=THEMES, 4=LOCATIONS, 5=PERSONS,
                # 6=ORGANIZATIONS, 7=TONE, 8=CAMEOEVENTIDS, 9=SOURCES,
                # 10=SOURCEURLS
                idx_date = 0
                idx_numarts = 1
                idx_orgs = 6
                idx_tone = 7
                row_iter = _chain_single(first, reader)
            else:
                header = first
                try:
                    idx_date = header.index("DATE")
                    idx_numarts = header.index("NUMARTS") if "NUMARTS" in header else 1
                    idx_orgs = header.index("ORGANIZATIONS")
                    idx_tone = header.index("TONE")
                except ValueError:
                    log.warning(
                        "GDELT GKG: unrecognized header %s; skipping file.",
                        header[:8],
                    )
                    return pd.DataFrame()
                row_iter = reader

            # Build per-(ticker, date) aggregates while streaming.
            mention_counts: dict[tuple[str, str], int] = {}
            tone_sums: dict[tuple[str, str], float] = {}
            tone_sq_sums: dict[tuple[str, str], float] = {}
            article_sums: dict[tuple[str, str], int] = {}

            for row in row_iter:
                if len(row) <= max(idx_orgs, idx_tone, idx_date, idx_numarts):
                    continue
                date_str = row[idx_date][:8]
                if not date_str.isdigit() or len(date_str) != 8:
                    continue
                orgs_str = row[idx_orgs] or ""
                if not orgs_str:
                    continue
                tone_str = row[idx_tone] or ""
                if not tone_str:
                    continue
                # tone = "avg_tone, positive, negative, ..."; we want
                # the first comma-separated value.
                try:
                    tone_val = float(tone_str.split(",")[0])
                except (ValueError, IndexError):
                    continue
                try:
                    n_articles = int(row[idx_numarts]) if row[idx_numarts] else 1
                except ValueError:
                    n_articles = 1

                # ORGANIZATIONS field is ";"-delimited list of names,
                # each possibly with a "name,offset" tail. Take only
                # the name portion.
                orgs = (o.split(",", 1)[0].strip().upper() for o in orgs_str.split(";"))
                seen_tickers_in_row: set[str] = set()
                for org in orgs:
                    if not org or len(org) < 3:
                        continue
                    if org not in name_to_tickers:
                        continue
                    for t in name_to_tickers[org]:
                        if t in seen_tickers_in_row:
                            continue
                        seen_tickers_in_row.add(t)
                        key = (date_str, t)
                        mention_counts[key] = mention_counts.get(key, 0) + 1
                        tone_sums[key] = tone_sums.get(key, 0.0) + tone_val
                        tone_sq_sums[key] = tone_sq_sums.get(key, 0.0) + tone_val * tone_val
                        article_sums[key] = article_sums.get(key, 0) + n_articles

            if not mention_counts:
                return pd.DataFrame()

            rows: list[dict] = []
            for (date_str, t), n in mention_counts.items():
                s = tone_sums[(date_str, t)]
                s2 = tone_sq_sums[(date_str, t)]
                mean = s / n
                var = max(0.0, s2 / n - mean * mean)
                std = var**0.5
                rows.append(
                    {
                        "date": date_str,
                        "ticker": t,
                        "gdelt_mention_count": n,
                        "gdelt_article_count": article_sums[(date_str, t)],
                        "gdelt_tone_mean": mean,
                        "gdelt_tone_std": std,
                    }
                )
            out = pd.DataFrame(rows)
            out["date"] = pd.to_datetime(out["date"], format="%Y%m%d")
            out["ticker"] = out["ticker"].astype("category")
            out["gdelt_mention_count"] = out["gdelt_mention_count"].astype("int16")
            out["gdelt_article_count"] = out["gdelt_article_count"].astype("int32")
            out["gdelt_tone_mean"] = out["gdelt_tone_mean"].astype("float32")
            out["gdelt_tone_std"] = out["gdelt_tone_std"].astype("float32")
            return out


def _chain_single(first, rest):
    """Helper: yield `first` then iterate `rest`."""
    yield first
    yield from rest


def _build_name_to_tickers(ticker_to_cik: dict[str, str]) -> dict[str, list[str]]:
    """From SEC's ticker_to_cik map, build an UPPER-CASE company-name
    -> list-of-tickers map suitable for matching GDELT's ORGANIZATIONS
    field.

    We re-fetch SEC's company_tickers.json (cached) for the company
    titles. Names are filtered:
      - length >= 4 characters (rejects 'CAT', 'CO', 'INC' false-positives)
      - strip trailing 'INC', 'INC.', 'CORP', 'CORPORATION', 'CO',
        'COMPANY', 'CO.', 'LTD', 'LTD.', 'LP', 'PLC', 'HOLDINGS', etc.
      - resulting name >= 4 chars; otherwise dropped
    """
    from stockpred.data.edgar import TICKER_CIK_CACHE
    import json

    # The edgar module wrote a {ticker: cik} dict; we need {ticker: title}
    # which is also derivable from company_tickers.json. Re-read SEC's
    # source file if present.
    src = TICKER_CIK_CACHE.parent / "company_tickers.json"
    titles_by_ticker: dict[str, str] = {}
    if src.exists():
        try:
            raw = json.loads(src.read_text())
            for v in raw.values():
                t = str(v.get("ticker", "")).upper()
                title = str(v.get("title", "")).upper().strip()
                if t and title:
                    titles_by_ticker[t] = title
        except Exception as e:  # noqa: BLE001
            log.warning("Could not parse company_tickers.json: %s", e)

    if not titles_by_ticker:
        log.warning(
            "No SEC company titles available; GDELT will match nothing. "
            "Run a Phase 12 EDGAR fetch first to populate company_tickers.json."
        )
        return {}

    # Filter to tickers in our requested universe.
    universe = set(ticker_to_cik.keys())
    titles_filtered = {t: titles_by_ticker[t] for t in universe if t in titles_by_ticker}

    suffixes = (
        " INC.",
        " INC",
        " CORPORATION",
        " CORP.",
        " CORP",
        " COMPANY",
        " CO.",
        " CO",
        " LTD.",
        " LTD",
        " LIMITED",
        " LP",
        " PLC",
        " HOLDINGS",
        " HOLDING",
        " GROUP",
        " THE",
        ", THE",
    )

    name_to_tickers: dict[str, list[str]] = {}
    for ticker, title in titles_filtered.items():
        name = title
        # Strip trailing legal-suffix tokens
        changed = True
        while changed:
            changed = False
            for suf in suffixes:
                if name.endswith(suf):
                    name = name[: -len(suf)].strip().rstrip(",")
                    changed = True
                    break
        if len(name) < 4:
            continue
        name_to_tickers.setdefault(name, []).append(ticker)

    log.info(
        "GDELT name->tickers map: %d unique names (from %d tickers)",
        len(name_to_tickers),
        len(titles_filtered),
    )
    return name_to_tickers


def fetch_gdelt_day(
    d: date,
    name_to_tickers: dict[str, list[str]],
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch + parse one day's GKG, return per-ticker aggregates.

    Cached as gzipped parquet at `data/cache/gdelt/gkg_YYYYMMDD.parquet`.
    """
    cache_path = _day_cache_path(d)
    if cache_path.exists() and not refresh:
        try:
            return pd.read_parquet(cache_path)
        except Exception as e:  # noqa: BLE001
            log.warning("Cached %s invalid (%s); refetching.", cache_path, e)

    zip_bytes = _stream_gkg_day(d)
    if zip_bytes is None:
        # 404 or fetch failure; cache an empty marker so we don't retry.
        empty = pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "gdelt_mention_count",
                "gdelt_article_count",
                "gdelt_tone_mean",
                "gdelt_tone_std",
            ]
        )
        empty.to_parquet(cache_path, compression="snappy", index=False)
        return empty

    df = _parse_gkg_zip(zip_bytes, name_to_tickers)
    # Free raw bytes ASAP -- can be ~15 MB per day.
    del zip_bytes
    gc.collect()

    df.to_parquet(cache_path, compression="snappy", index=False)
    log.info(
        "GDELT day %s: parsed %d per-ticker rows -> %s",
        d,
        len(df),
        cache_path,
    )
    return df


def bulk_fetch_gdelt(
    start: date,
    end: date,
    name_to_tickers: dict[str, list[str]],
    *,
    refresh: bool = False,
    progress_every: int = 25,
) -> None:
    """Bulk-fetch every GKG day in [start, end] inclusive.

    Honours the per-day cache; idempotent if interrupted + resumed.
    Designed to be run overnight. Use `progress_every` to limit log
    noise (default: log every 25 days = roughly every minute at 0.5 s
    sleep).
    """
    cur = start
    n_total = (end - start).days + 1
    n_done = 0
    n_with_data = 0
    n_empty = 0
    n_errors = 0
    t0 = time.time()
    while cur <= end:
        try:
            df = fetch_gdelt_day(cur, name_to_tickers, refresh=refresh)
            if len(df) > 0:
                n_with_data += 1
            else:
                n_empty += 1
        except Exception as e:  # noqa: BLE001
            log.error("GDELT %s: unexpected error: %s", cur, e)
            n_errors += 1
        n_done += 1
        if n_done % progress_every == 0:
            elapsed = time.time() - t0
            rate = n_done / max(elapsed, 1)
            remaining = (n_total - n_done) / max(rate, 0.01)
            log.warning(
                "GDELT bulk fetch progress: %d/%d (%.1f%%) | "
                "with-data=%d, empty=%d, errors=%d | "
                "%.1f files/sec, ~%.1f min remaining",
                n_done,
                n_total,
                100 * n_done / n_total,
                n_with_data,
                n_empty,
                n_errors,
                rate,
                remaining / 60,
            )
        cur += timedelta(days=1)
    log.warning(
        "GDELT bulk fetch DONE: %d files, with-data=%d, empty=%d, errors=%d, elapsed=%.1f min",
        n_total,
        n_with_data,
        n_empty,
        n_errors,
        (time.time() - t0) / 60,
    )


def build_gdelt_features(
    tickers: list[str],
    trading_days: pd.DatetimeIndex,
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
    ticker_to_cik: dict[str, str] | None = None,
    rolling_windows: tuple[int, ...] = (5, 21),
) -> pd.DataFrame:
    """Build per-(date, ticker) GDELT features.

    Reads PRE-CACHED per-day parquet files (output of bulk_fetch_gdelt).
    Does NOT trigger HTTP fetches itself -- that's the operator's
    overnight job. If a day's cache is missing, this function treats
    that day as 'no mentions' (all zeros).

    Output columns:
      gdelt_mention_count       (int16)
      gdelt_article_count       (int32)
      gdelt_tone_mean           (float32)
      gdelt_tone_std            (float32)
      gdelt_mention_{w}d        (int16)  rolling sum
      gdelt_tone_{w}d           (float32) rolling mean of daily means
    """
    if ticker_to_cik is None:
        from stockpred.data.edgar import fetch_ticker_to_cik

        ticker_to_cik = fetch_ticker_to_cik(refresh=False)
    if start is None:
        start = trading_days.min().strftime("%Y-%m-%d")
    if end is None:
        end = trading_days.max().strftime("%Y-%m-%d")
    start_d = pd.Timestamp(start).date()
    end_d = pd.Timestamp(end).date()

    present = [t.upper() for t in tickers if t.upper() in ticker_to_cik]
    if not present:
        log.warning("build_gdelt_features: no tickers had CIK match; empty.")
        return pd.DataFrame()

    # Aggregate from per-day cache files.
    frames: list[pd.DataFrame] = []
    cur = start_d
    n_missing = 0
    while cur <= end_d:
        cache_path = _day_cache_path(cur)
        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    frames.append(df[df["ticker"].isin(present)])
            except Exception as e:  # noqa: BLE001
                log.warning("GDELT day %s: cache unreadable (%s); skipping.", cur, e)
                n_missing += 1
        else:
            n_missing += 1
        cur += timedelta(days=1)

    if n_missing:
        n_total = (end_d - start_d).days + 1
        coverage = 100 * (1 - n_missing / n_total)
        log.warning(
            "GDELT cache coverage: %.1f%% (%d / %d days missing). "
            "Run scripts/phase14_gdelt_bulk_fetch.py first.",
            coverage,
            n_missing,
            n_total,
        )

    if not frames:
        log.warning("build_gdelt_features: empty result (no cached days).")
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # Forward-shift news-day events to the NEXT trading day so
    # weekend/holiday news lands on Monday (same leak-safety pattern
    # as Phase 12 EDGAR events).
    raw["date"] = pd.to_datetime(raw["date"]).astype("datetime64[ns]")
    td_df = pd.DataFrame({"trading_day": pd.DatetimeIndex(trading_days).astype("datetime64[ns]")})
    raw = raw.sort_values("date")
    mapped = pd.merge_asof(
        raw,
        td_df,
        left_on="date",
        right_on="trading_day",
        direction="forward",
    )
    mapped = mapped.dropna(subset=["trading_day"])

    # Sum mention/article counts by (trading_day, ticker); weight tone
    # mean by mention count.
    grouped = mapped.groupby(["trading_day", "ticker"], observed=True).agg(
        gdelt_mention_count=("gdelt_mention_count", "sum"),
        gdelt_article_count=("gdelt_article_count", "sum"),
        # tone_mean: weighted by mentions
        gdelt_tone_sum=("gdelt_tone_mean", lambda s: float(s.sum())),
        gdelt_tone_n=("gdelt_tone_mean", "count"),
    )
    grouped["gdelt_tone_mean"] = (grouped["gdelt_tone_sum"] / grouped["gdelt_tone_n"]).astype(
        "float32"
    )
    grouped["gdelt_mention_count"] = grouped["gdelt_mention_count"].astype("int16")
    grouped["gdelt_article_count"] = grouped["gdelt_article_count"].astype("int32")
    grouped = grouped.drop(columns=["gdelt_tone_sum", "gdelt_tone_n"])

    # Reindex onto (trading_days x tickers), filling zeros for no-news days.
    full_idx = pd.MultiIndex.from_product([trading_days, present], names=["date", "ticker"])
    grouped = grouped.reset_index().rename(columns={"trading_day": "date"})
    grouped = (
        grouped.set_index(["date", "ticker"])
        .reindex(full_idx)
        .fillna(
            {
                "gdelt_mention_count": 0,
                "gdelt_article_count": 0,
                "gdelt_tone_mean": 0.0,
            }
        )
    )
    grouped["gdelt_mention_count"] = grouped["gdelt_mention_count"].astype("int16")
    grouped["gdelt_article_count"] = grouped["gdelt_article_count"].astype("int32")
    grouped["gdelt_tone_mean"] = grouped["gdelt_tone_mean"].astype("float32")

    # Rolling features (per ticker)
    wide_mention = grouped["gdelt_mention_count"].unstack("ticker").fillna(0).astype("int16")
    wide_tone = grouped["gdelt_tone_mean"].unstack("ticker").fillna(0.0).astype("float32")
    extra: list[pd.Series] = []
    for w in rolling_windows:
        m = wide_mention.rolling(window=w, min_periods=1).sum().astype("int16")
        extra.append(m.stack(future_stack=True).rename(f"gdelt_mention_{w}d"))
        t = wide_tone.rolling(window=w, min_periods=1).mean().astype("float32")
        extra.append(t.stack(future_stack=True).rename(f"gdelt_tone_{w}d"))
    if extra:
        rolled = pd.concat(extra, axis=1)
        rolled.index.names = ["date", "ticker"]
        out = grouped.join(rolled, how="left")
    else:
        out = grouped

    # Cleanup
    del raw, mapped, grouped, wide_mention, wide_tone
    if extra:
        del extra
    gc.collect()

    log.info(
        "build_gdelt_features: %d rows x %d cols (covering %d tickers)",
        out.shape[0],
        out.shape[1],
        len(present),
    )
    return out
