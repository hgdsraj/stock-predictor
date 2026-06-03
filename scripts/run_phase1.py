#!/usr/bin/env python
"""End-to-end Phase 1 entrypoint.

Usage:
    uv run python scripts/run_phase1.py
    uv run python scripts/run_phase1.py --n-tickers 50 --start 2015-01-01

The script will:
  - Fetch S&P 500 historical constituents (cached to data/cache/)
  - Download adjusted prices via yfinance (cached per-ticker parquet)
  - Build features + labels
  - Run walk-forward CV with the logistic baseline
  - Construct top-k long/short portfolio, backtest with realistic costs
  - Write an HTML tearsheet to reports/
"""

from __future__ import annotations

import argparse
import logging
import sys

from stockpred.pipeline import PipelineConfig, run_phase1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2010-01-01", help="History start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="History end date (YYYY-MM-DD)")
    p.add_argument("--n-tickers", type=int, default=100, help="Universe size (None = all)")
    p.add_argument("--horizon", type=int, default=1, help="Forecast horizon in trading days")
    p.add_argument("--k", type=int, default=20, help="Top/bottom k per side for portfolio")
    p.add_argument("--refresh", action="store_true", help="Bypass data cache")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        horizon=args.horizon,
        k_per_side=args.k,
        refresh_data=args.refresh,
    )
    result = run_phase1(cfg)

    print()
    print("=" * 60)
    print(" Phase 1 complete")
    print("=" * 60)
    print(f"  Universe size      : {len(result['tickers'])}")
    print(f"  Feature matrix     : {result['feature_matrix_shape']}")
    print(f"  Hit rate (OOS)     : {result['hit_rate']:.4f}")
    print(f"  IC mean (OOS)      : {result['ic_summary']['ic_mean']:+.5f}")
    print(f"  IC IR (OOS)        : {result['ic_summary']['ic_ir']:+.3f}")
    metrics = result["metrics"]
    print(f"  Ann return (net)   : {metrics['ann_return']:+.2%}")
    print(f"  Ann vol            : {metrics['ann_vol']:.2%}")
    print(f"  Sharpe (net)       : {metrics['sharpe']:+.3f}")
    print(f"  Max drawdown       : {metrics['max_drawdown']:.2%}")
    print(f"  Tearsheet          : {result['tearsheet_path']}")
    print(f"  Elapsed            : {result['elapsed_s']:.1f}s")
    print()
    print("Reminder: hit-rate of ~50–54% on real walk-forward is normal.")
    print("Anything >55% probably means a bug. Investigate, don't celebrate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
