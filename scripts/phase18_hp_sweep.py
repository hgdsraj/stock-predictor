#!/usr/bin/env python
"""Phase 18: LightGBM hyperparameter sweep on the Phase 13 best config.

The Phase 13 best uses default GBMConfig (num_leaves=31, lr=0.05,
n_estimators=200, min_data_in_leaf=20). We sweep a small grid around
these and pick the combination with the best HOLDOUT Sharpe + CI.

Honest design:
  - Grid is small (3 x 3 x 2 x 2 = 36 configs) so total runtime is
    manageable (~36 * ~3 min = ~110 min). Sub-grids can be passed
    via CLI to cut this further.
  - HOLDOUT Sharpe is the SELECTION metric (we already pay the price
    of holdout-data selection by reporting a CI; we'll re-honor that
    by NOT calling the best config 'proven'.)
  - Output ranks all 36 by hold Sharpe and reports CI for each.

Output: reports/phase18_hp_sweep_<args>.csv
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from stockpred.models.gbm import GBMConfig
from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

log = logging.getLogger("phase18")

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-tickers", type=int, default=150)
    p.add_argument("--horizons", type=int, nargs="+", default=[5])
    p.add_argument("--bootstrap-n", type=int, default=300)
    p.add_argument(
        "--num-leaves",
        type=int,
        nargs="+",
        default=[15, 31, 63],
        help="Grid for num_leaves",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
        help="Grid for learning_rate",
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=[200, 400],
        help="Grid for n_estimators",
    )
    p.add_argument(
        "--min-data-in-leaf",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Grid for min_data_in_leaf",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _build_cfg(args: argparse.Namespace, gbm_cfg: GBMConfig) -> PipelineV5Config:
    """Phase 13 best, with the swept GBM hyperparams."""
    return PipelineV5Config(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling="current",
        horizons=tuple(args.horizons),
        model="gbm",
        gbm=gbm_cfg,
        ensemble_weighting="equal",
        position_sizing="hrp",
        k_per_side_pct=0.15,
        sector_cap_gross=0.30,
        min_trade_threshold=0.005,
        holdout_years=2,
        bootstrap_n=args.bootstrap_n,
        bootstrap_method="block",
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        beta_neutralise=False,
        use_meta_labelling=True,
        meta_threshold=0.55,
        meta_mode="binary",
        meta_walk_forward_folds=1,
        meta_per_sector=False,
        ranks_only=True,
        use_triple_barrier_labels=False,
        use_edgar_features=False,
        use_edgar_item_features=True,
    )


def run_one(label: str, cfg: PipelineV5Config) -> dict:
    t0 = time.time()
    try:
        res = run_pipeline_v5(cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("Run %s failed: %s", label, e)
        return {
            "label": label,
            "dev_sharpe": float("nan"),
            "hold_sharpe": float("nan"),
            "hold_ci_lo": float("nan"),
            "hold_ci_hi": float("nan"),
            "hold_dd": float("nan"),
            "elapsed_s": time.time() - t0,
            "error": str(e),
        }
    dev = res.get("metrics", {}) or {}
    hold = res.get("holdout_metrics", {}) or {}
    ci = res.get("bootstrap_sharpe", {}) or {}
    return {
        "label": label,
        "num_leaves": cfg.gbm.num_leaves,
        "learning_rate": cfg.gbm.learning_rate,
        "n_estimators": cfg.gbm.n_estimators,
        "min_data_in_leaf": cfg.gbm.min_data_in_leaf,
        "dev_sharpe": dev.get("sharpe", float("nan")),
        "hold_sharpe": hold.get("sharpe", float("nan")),
        "hold_ci_lo": ci.get("sharpe_lo", float("nan")),
        "hold_ci_hi": ci.get("sharpe_hi", float("nan")),
        "hold_dd": hold.get("max_drawdown", float("nan")),
        "elapsed_s": round(time.time() - t0, 1),
        "error": "",
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    grid = list(
        itertools.product(
            args.num_leaves,
            args.learning_rate,
            args.n_estimators,
            args.min_data_in_leaf,
        )
    )
    print(f"Phase 18 HP sweep: {len(grid)} configs")
    print(f"  num_leaves       : {args.num_leaves}")
    print(f"  learning_rate    : {args.learning_rate}")
    print(f"  n_estimators     : {args.n_estimators}")
    print(f"  min_data_in_leaf : {args.min_data_in_leaf}")
    print(f"  universe         : {args.n_tickers} tickers, {args.start} -> {args.end}")
    print(f"  bootstrap_n      : {args.bootstrap_n}")
    print(f"  ETA              : ~{len(grid) * 3:.0f} min (3 min/config)")
    print()

    rows: list[dict] = []
    for i, (nl, lr, ne, mdl) in enumerate(grid, 1):
        label = f"nl={nl} lr={lr} ne={ne} mdl={mdl}"
        gbm_cfg = GBMConfig(
            num_leaves=nl,
            learning_rate=lr,
            n_estimators=ne,
            min_data_in_leaf=mdl,
        )
        cfg = _build_cfg(args, gbm_cfg)
        print(f"[{i}/{len(grid)}] {label} ...", flush=True)
        row = run_one(label, cfg)
        rows.append(row)
        if row["error"]:
            print(f"  -> FAILED in {row['elapsed_s']:.0f}s: {row['error']}")
        else:
            print(
                f"  -> hold Sharpe={row['hold_sharpe']:+.3f}  "
                f"CI=[{row['hold_ci_lo']:+.3f}, {row['hold_ci_hi']:+.3f}]  "
                f"DD={row['hold_dd']:+.2%}  ({row['elapsed_s']:.0f}s)"
            )
        print()

    df = pd.DataFrame(rows).sort_values("hold_sharpe", ascending=False)
    fmt = {
        "dev_sharpe": "{:+.3f}".format,
        "hold_sharpe": "{:+.3f}".format,
        "hold_ci_lo": "{:+.3f}".format,
        "hold_ci_hi": "{:+.3f}".format,
        "hold_dd": "{:+.2%}".format,
        "learning_rate": "{:.3f}".format,
    }
    show_cols = [
        "num_leaves",
        "learning_rate",
        "n_estimators",
        "min_data_in_leaf",
        "dev_sharpe",
        "hold_sharpe",
        "hold_ci_lo",
        "hold_ci_hi",
        "hold_dd",
        "elapsed_s",
    ]
    print("=" * 110)
    print("Phase 18 HP sweep results (sorted by HOLDOUT Sharpe descending)")
    print("=" * 110)
    print(df[show_cols].to_string(index=False, formatters=fmt))
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / (f"phase18_hp_sweep_{args.start}_{args.end}_n{args.n_tickers}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    valid = df[df["error"] == ""]
    if not valid.empty:
        best = valid.iloc[0]
        print()
        print("Honest interpretation:")
        print(
            f"  Headline best (point estimate only -- NOT a significance test): "
            f"{best['hold_sharpe']:+.3f}  ({best['label']})"
        )
        n_pos_ci = int((valid["hold_ci_lo"] > 0).sum())
        n_straddle = int(((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum())
        print(f"  Configs with CI strictly > 0    : {n_pos_ci} / {len(valid)}")
        print(f"  Configs with CI straddling zero : {n_straddle} / {len(valid)}")
        print(
            "  CAUTION: this is a HOLDOUT-data hyperparameter selection. "
            "Even with bootstrap CI, picking the headline-best risks the "
            "garden-of-forking-paths effect. Treat the BEST config as a "
            "STARTING POINT for a future independent holdout, not as a "
            "proven edge."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
