#!/usr/bin/env python
"""Phase 14: bulk-fetch GDELT GKG daily files for the backtest range.

Designed to be run OVERNIGHT or in the background. Each daily file is
~5-15 MB compressed, ~4000 files for 2014-2024; with the 0.5 sec
rate-limit sleep that's ~33 min of sleep + ~1-2 hr of network at
typical bandwidth.

The fetch is idempotent: per-day caches are written immediately and
re-fetched only when missing or forced. Safe to interrupt + resume.

Usage (overnight, default 2014-2024 range):
    nohup uv run python scripts/phase14_gdelt_bulk_fetch.py \\
        --tickers-from-edgar \\
        > logs/phase14_bulk_fetch.log 2>&1 &

Usage (smaller smoke test, 60 days):
    uv run python scripts/phase14_gdelt_bulk_fetch.py \\
        --tickers-from-edgar \\
        --start 2024-10-01 --end 2024-11-30
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

from stockpred.data import edgar as edgar_mod
from stockpred.data import gdelt as gdelt_mod

log = logging.getLogger("phase14")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--start",
        default="2014-01-01",
        help="ISO date (default: 2014-01-01, the backtest start)",
    )
    p.add_argument(
        "--end",
        default=None,
        help="ISO date (default: today minus 1)",
    )
    p.add_argument(
        "--tickers-from-edgar",
        action="store_true",
        help=(
            "Build the ticker->name map from SEC's company_tickers.json "
            "(populated by Phase 12 fetch). Required for filtering "
            "GDELT mentions to your universe."
        ),
    )
    p.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch days even if cached.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    start_d = datetime.fromisoformat(args.start).date()
    end_d = datetime.fromisoformat(args.end).date() if args.end else date.today()

    if not args.tickers_from_edgar:
        raise SystemExit(
            "--tickers-from-edgar is required so we know which company "
            "names to filter for. Run a Phase 12 EDGAR fetch first."
        )

    log.warning("Loading SEC ticker->CIK map...")
    ticker_to_cik = edgar_mod.fetch_ticker_to_cik(refresh=False)
    log.warning("Building name->tickers map for GDELT matching...")
    name_to_tickers = gdelt_mod._build_name_to_tickers(ticker_to_cik)
    if not name_to_tickers:
        raise SystemExit(
            "name_to_tickers is empty. Did Phase 12 EDGAR fetch run and "
            "cache company_tickers.json? Check data/cache/edgar/."
        )
    log.warning(
        "Will match GDELT mentions against %d company names covering %d tickers.",
        len(name_to_tickers),
        sum(len(v) for v in name_to_tickers.values()),
    )

    log.warning(
        "Bulk-fetching GDELT GKG daily files: %s to %s (%d days).",
        start_d,
        end_d,
        (end_d - start_d).days + 1,
    )
    log.warning(
        "ETA at default 0.5s sleep: ~%.0f min of pure sleep, plus ~1-2 hr "
        "of network at typical bandwidth.",
        (end_d - start_d).days * 0.5 / 60,
    )

    gdelt_mod.bulk_fetch_gdelt(
        start_d,
        end_d,
        name_to_tickers,
        refresh=args.refresh,
    )

    cache_dir = gdelt_mod.CACHE_DIR_GDELT
    total_size_mb = sum(p.stat().st_size for p in cache_dir.glob("*.parquet")) / (1024**2)
    n_files = len(list(cache_dir.glob("*.parquet")))
    log.warning(
        "GDELT cache directory %s: %d files, %.1f MB total.",
        cache_dir,
        n_files,
        total_size_mb,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
