#!/usr/bin/env python
"""End-to-end pipeline entrypoint.

Usage:
    uv run python scripts/run_phase1.py
    uv run python scripts/run_phase1.py --n-tickers 100 --horizons 1 5 21 --k 20

The script will:
  - Fetch S&P 500 historical constituents (cached to data/cache/)
  - Download adjusted prices via yfinance (cached per-ticker parquet)
  - Optionally fetch sector tags via yfinance .info
  - Build features (technicals + cross-sectional ranks + sector-neutralised)
  - Train walk-forward CV per horizon with LightGBM (default) or logistic baseline
  - Ensemble per-horizon scores and construct top-k long/short portfolio
  - Backtest horizon-aware, with realistic costs
  - Write an HTML tearsheet to reports/
"""

from __future__ import annotations

import argparse
import logging
import sys

from stockpred.pipeline import PipelineConfig, run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2010-01-01", help="History start date (YYYY-MM-DD)")
    p.add_argument("--end", default=None, help="History end date (YYYY-MM-DD)")
    p.add_argument("--n-tickers", type=int, default=100, help="Universe size (None = all)")
    p.add_argument(
        "--universe-sampling",
        choices=("random", "current", "first"),
        default="random",
        help="How to subset the historical universe (random is unbiased; current is SURVIVORSHIP-BIASED)",
    )
    p.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 21],
        help="One or more forecast horizons (trading days)",
    )
    p.add_argument("--k", type=int, default=20, help="Top/bottom k per side for portfolio")
    p.add_argument(
        "--model",
        choices=("gbm", "logistic"),
        default="gbm",
        help="Model used per horizon",
    )
    p.add_argument(
        "--no-sector",
        action="store_true",
        help="Disable sector-neutralised features (skip yfinance .info call)",
    )
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
        universe_sampling=args.universe_sampling,
        horizons=tuple(args.horizons),
        k_per_side=args.k,
        model=args.model,
        use_sector_features=not args.no_sector,
        refresh_data=args.refresh,
    )
    result = run_pipeline(cfg)

    print()
    print("=" * 64)
    print(" Pipeline complete")
    print("=" * 64)
    print(f"  Universe size      : {len(result['tickers'])}")
    print(f"  Feature matrix     : {result['feature_matrix_shape']}")
    print()
    print("  Per-horizon OOS:")
    for h, d in result["per_horizon_diagnostics"].items():
        print(
            f"    h={h:>2}d   hit={d['hit_rate']:.4f}   "
            f"ic_mean={d['ic_mean']:+.5f}   ic_ir={d['ic_ir']:+.3f}"
        )
    metrics = result["metrics"]
    print()
    print("  Backtest (ensemble):")
    print(f"    Ann return (net)   : {metrics['ann_return']:+.2%}")
    print(f"    Ann vol            : {metrics['ann_vol']:.2%}")
    print(f"    Sharpe (net)       : {metrics['sharpe']:+.3f}")
    print(f"    Max drawdown       : {metrics['max_drawdown']:.2%}")
    print(f"  Tearsheet          : {result['tearsheet_path']}")
    print(f"  Elapsed            : {result['elapsed_s']:.1f}s")
    print()
    print("Reminder: hit-rate of ~50–54% on real walk-forward is normal.")
    print("Anything >55% probably means a bug. Investigate, don't celebrate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
