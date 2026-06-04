#!/usr/bin/env python
"""Phase 6 sensitivity grid.

Run the Phase 5 pipeline across a grid of cost / k_per_side / horizon and
report holdout Sharpe + holdout-bootstrap-CI for each combination. If
results swing wildly with a 2 -> 6 bps cost change or with k 10% -> 20%,
any 'edge' is overfitting.

Usage:
    uv run python scripts/sensitivity.py --start 2018-01-01 --end 2024-12-31 \\
        --n-tickers 60 --universe-sampling current
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys

import pandas as pd

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-tickers", type=int, default=60)
    p.add_argument("--universe-sampling", default="current")
    p.add_argument("--horizons-grid", type=str, nargs="+", default=["1,5", "5", "1"])
    p.add_argument("--k-grid", type=float, nargs="+", default=[0.10, 0.15, 0.20])
    p.add_argument("--cost-grid", type=float, nargs="+", default=[2.0, 6.0, 12.0])
    p.add_argument("--sector-cap-grid", type=float, nargs="+", default=[0.0, 0.30])
    p.add_argument("--beta-neutralise-grid", type=int, nargs="+", default=[0, 1])
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if not args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    horizons_options = [tuple(int(x) for x in h.split(",")) for h in args.horizons_grid]
    combos = list(
        itertools.product(
            horizons_options,
            args.k_grid,
            args.cost_grid,
            args.sector_cap_grid,
            args.beta_neutralise_grid,
        )
    )
    print(f"Running {len(combos)} combinations...")
    print()

    # Pre-build column header.
    cols = [
        "horizons",
        "k_pct",
        "cost_bps",
        "sector_cap",
        "beta_neut",
        "dev_sharpe",
        "hold_sharpe",
        "hold_ci_lo",
        "hold_ci_hi",
        "hold_dd",
    ]
    rows: list[dict] = []
    for i, (horizons, k, cost, scap, bn) in enumerate(combos, 1):
        # Patch the cost into the global config dataclass (BacktestConfig is
        # immutable so the cleanest path is to swap defaults via monkey-patch
        # inside the pipeline call).
        from stockpred import config as cfg_mod

        original_bcfg = cfg_mod.BacktestConfig
        cost_per_side = cost / 3.0  # split into commission/spread/slippage roughly
        new_bcfg = cfg_mod.BacktestConfig(
            commission_bps=cost_per_side,
            spread_bps=cost_per_side,
            slippage_bps=cost_per_side,
        )
        try:
            cfg = PipelineV5Config(
                start_date=args.start,
                end_date=args.end,
                n_tickers=args.n_tickers,
                universe_sampling=args.universe_sampling,
                horizons=horizons,
                k_per_side_pct=k,
                sector_cap_gross=(scap if scap > 0 else None),
                beta_neutralise=bool(bn),
                bootstrap_method="block",
            )
            print(
                f"[{i}/{len(combos)}] horizons={horizons} k={k} cost={cost}bps scap={scap} beta_neut={bn} ..."
            )
            # Monkey-patch BacktestConfig inside the engine for this run.
            from stockpred.backtest import engine as engine_mod

            engine_mod.BacktestConfig = lambda: new_bcfg  # type: ignore[assignment]
            res = run_pipeline_v5(cfg)
            engine_mod.BacktestConfig = original_bcfg  # type: ignore[assignment]
            dm = res.get("metrics", {})
            hm = res.get("holdout_metrics", {}) or {}
            ci = res.get("bootstrap_sharpe", {}) or {}
            rows.append(
                {
                    "horizons": str(horizons),
                    "k_pct": k,
                    "cost_bps": cost,
                    "sector_cap": scap,
                    "beta_neut": bn,
                    "dev_sharpe": dm.get("sharpe", float("nan")),
                    "hold_sharpe": hm.get("sharpe", float("nan")),
                    "hold_ci_lo": ci.get("sharpe_lo", float("nan")),
                    "hold_ci_hi": ci.get("sharpe_hi", float("nan")),
                    "hold_dd": hm.get("max_drawdown", float("nan")),
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"  combo failed: {e}")
            rows.append(
                {
                    "horizons": str(horizons),
                    "k_pct": k,
                    "cost_bps": cost,
                    "sector_cap": scap,
                    "beta_neut": bn,
                    **{k2: float("nan") for k2 in cols[5:]},
                }
            )
        finally:
            engine_mod.BacktestConfig = original_bcfg  # type: ignore[assignment]

    df = pd.DataFrame(rows)[cols]
    print()
    print("=" * 90)
    print("Sensitivity grid")
    print("=" * 90)
    print(
        df.to_string(
            index=False, float_format=lambda x: f"{x:+.3f}" if -10 < x < 10 else f"{x:.0f}"
        )
    )
    print()
    # Save CSV.
    out_path = "reports/sensitivity_grid.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    # Diagnostics.
    print()
    print("Holdout-Sharpe spread across the grid:")
    print(f"  best  : {df['hold_sharpe'].max():+.3f}")
    print(f"  worst : {df['hold_sharpe'].min():+.3f}")
    print(f"  median: {df['hold_sharpe'].median():+.3f}")
    n_positive = int((df["hold_sharpe"] > 0).sum())
    n_significant = int((df["hold_ci_lo"] > 0).sum())
    print(f"  combos with hold_sharpe > 0          : {n_positive} / {len(df)}")
    print(f"  combos with bootstrap CI lower > 0   : {n_significant} / {len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
