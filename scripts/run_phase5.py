#!/usr/bin/env python
"""Phase 5 pipeline CLI.

Builds on Phase 1/2 with: IC-IR-weighted ensemble + vol-scaled sizing +
sector caps + minimum trade threshold + held-out window + bootstrap
Sharpe CI + per-regime breakdown.

Usage:
    uv run python scripts/run_phase5.py \\
        --start 2018-01-01 \\
        --n-tickers 100 \\
        --horizons 1 5 \\
        --weighting ic_ir \\
        --position-sizing vol_scaled \\
        --sector-cap 0.30 \\
        --min-trade-threshold 0.005 \\
        --holdout-years 2

See docs/USAGE.md §6 for what these flags actually do.
"""

from __future__ import annotations

import argparse
import logging
import sys

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--n-tickers", type=int, default=100)
    p.add_argument(
        "--universe-sampling",
        choices=("random", "current", "first"),
        default="random",
    )
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    p.add_argument("--model", choices=("gbm", "logistic"), default="gbm")
    p.add_argument(
        "--weighting",
        choices=("ic_ir", "equal"),
        default="ic_ir",
        help="How to combine per-horizon predictions",
    )
    p.add_argument(
        "--position-sizing",
        choices=("vol_scaled", "top_k"),
        default="vol_scaled",
    )
    p.add_argument("--k-pct", type=float, default=0.15, help="Top/bottom fraction per side")
    p.add_argument(
        "--sector-cap",
        type=float,
        default=0.30,
        help="Max gross exposure per sector (set 0 to disable)",
    )
    p.add_argument("--min-trade-threshold", type=float, default=0.005)
    p.add_argument("--holdout-years", type=int, default=2)
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--no-sector", action="store_true")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineV5Config(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling=args.universe_sampling,
        horizons=tuple(args.horizons),
        model=args.model,
        ensemble_weighting=args.weighting,
        position_sizing=args.position_sizing,
        k_per_side_pct=args.k_pct,
        sector_cap_gross=args.sector_cap if args.sector_cap > 0 else None,
        min_trade_threshold=args.min_trade_threshold,
        holdout_years=args.holdout_years,
        bootstrap_n=args.bootstrap_n,
        use_sector_features=not args.no_sector,
        refresh_data=args.refresh,
    )
    r = run_pipeline_v5(cfg)

    m = r["metrics"]
    hm = r.get("holdout_metrics", {}) or {}
    ci = r.get("bootstrap_sharpe", {}) or {}

    print()
    print("=" * 64)
    print(" Phase 5 pipeline complete")
    print("=" * 64)
    print(f"  Universe size      : {len(r['tickers'])}")
    print(f"  Feature matrix     : {r['feature_matrix_shape']}")
    print()
    print("  Per-horizon (DEV walk-forward OOS):")
    for h, d in r["per_horizon_diagnostics"].items():
        print(
            f"    h={h:>2}d  hit={d.get('hit_rate', float('nan')):.4f}  "
            f"ic_ir={d.get('ic_ir', float('nan')):+.3f}  "
            f"holdout_ic_ir={d.get('holdout_ic_ir', float('nan')):+.3f}"
        )
    print()
    print("  DEV backtest:")
    print(f"    Ann return (net) : {m.get('ann_return', float('nan')):+.2%}")
    print(f"    Sharpe (net)     : {m.get('sharpe', float('nan')):+.3f}")
    print(f"    Max drawdown     : {m.get('max_drawdown', float('nan')):.2%}")
    print()
    print("  HOLDOUT backtest (never seen during training):")
    print(f"    Ann return (net) : {hm.get('ann_return', float('nan')):+.2%}")
    print(f"    Sharpe (net)     : {hm.get('sharpe', float('nan')):+.3f}")
    print(f"    Max drawdown     : {hm.get('max_drawdown', float('nan')):.2%}")
    if ci:
        print(
            f"    Bootstrap Sharpe : {ci.get('sharpe', float('nan')):+.3f} "
            f"[{ci.get('sharpe_lo', float('nan')):+.3f}, "
            f"{ci.get('sharpe_hi', float('nan')):+.3f}] @ "
            f"{int(ci.get('ci_pct', 0) * 100)}%"
        )
        if ci.get("sharpe_lo", 0) > 0:
            print("    → 95% CI excludes 0: strategy is statistically distinguishable from random.")
        elif ci.get("sharpe_hi", 0) < 0:
            print("    → 95% CI is entirely negative: strategy loses statistically significantly.")
        else:
            print(
                "    → 95% CI straddles 0: NOT distinguishable from random. Treat any 'positive' result as luck."
            )
    print()
    print(f"  Tearsheet          : {r['tearsheet_path']}")
    print(f"  Elapsed            : {r['elapsed_s']:.1f}s")
    print()
    print("Reminder: holdout Sharpe is the honest number. DEV is in-sample-ish.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
