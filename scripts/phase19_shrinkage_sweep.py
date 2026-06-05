#!/usr/bin/env python
"""Phase 19: sweep `bayesian_shrinkage_alpha` on top of the Phase 13
best config to see whether per-ticker shrinkage improves holdout.

Runs alpha in {0.0, 0.25, 0.5, 0.75, 1.0} (5 configs, ~15 min total
on warm cache). alpha=0 reproduces the Phase 13 baseline exactly;
alpha=1.0 drops every below-random ticker and downweights noisy ones
proportional to (precision - 0.5).

Output: reports/phase19_shrinkage_sweep_<args>.csv

CAUTION (same as Phase 10, 16, 18): this is HOLDOUT-data selection.
Even with bootstrap CI, treat the best alpha as a STARTING POINT for
an independent holdout, not as a proven edge.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

log = logging.getLogger("phase19")

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-tickers", type=int, default=150)
    p.add_argument("--horizons", type=int, nargs="+", default=[5])
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help="Shrinkage alpha values to sweep (default: 0.0, 0.25, 0.5, 0.75, 1.0)",
    )
    p.add_argument(
        "--with-gdelt",
        action="store_true",
        help="Also include GDELT features (requires bulk fetch done).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _build_cfg(args: argparse.Namespace, alpha: float) -> PipelineV5Config:
    """Phase 13 best, plus shrinkage."""
    return PipelineV5Config(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling="current",
        horizons=tuple(args.horizons),
        model="gbm",
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
        use_gdelt_features=args.with_gdelt,
        bayesian_shrinkage_alpha=alpha,
    )


def run_one(label: str, cfg: PipelineV5Config) -> dict:
    t0 = time.time()
    try:
        res = run_pipeline_v5(cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("Run %s failed: %s", label, e)
        return {
            "label": label,
            "alpha": cfg.bayesian_shrinkage_alpha,
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
        "alpha": cfg.bayesian_shrinkage_alpha,
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

    print(f"Phase 19 shrinkage sweep: {len(args.alphas)} configs")
    print(f"  universe        : {args.n_tickers} tickers, {args.start} -> {args.end}")
    print(f"  horizons        : {args.horizons}")
    print(f"  alphas          : {args.alphas}")
    print(f"  with_gdelt      : {args.with_gdelt}")
    print(f"  ETA             : ~{len(args.alphas) * 3:.0f} min (3 min/config)")
    print()

    rows: list[dict] = []
    for i, alpha in enumerate(args.alphas, 1):
        label = f"alpha={alpha:.2f}"
        cfg = _build_cfg(args, alpha)
        print(f"[{i}/{len(args.alphas)}] {label} ...", flush=True)
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

    df = pd.DataFrame(rows)
    fmt = {
        "alpha": "{:.2f}".format,
        "dev_sharpe": "{:+.3f}".format,
        "hold_sharpe": "{:+.3f}".format,
        "hold_ci_lo": "{:+.3f}".format,
        "hold_ci_hi": "{:+.3f}".format,
        "hold_dd": "{:+.2%}".format,
    }
    show_cols = [c for c in df.columns if c != "error" or df["error"].any()]
    print("=" * 110)
    print("Phase 19 shrinkage sweep results")
    print("=" * 110)
    print(df[show_cols].to_string(index=False, formatters=fmt))
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / (
        f"phase19_shrinkage_sweep_{args.start}_{args.end}_n{args.n_tickers}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Honest interpretation
    valid = df[df["error"] == ""]
    if not valid.empty:
        baseline_row = valid[valid["alpha"] == 0.0]
        baseline_s = (
            float(baseline_row["hold_sharpe"].iloc[0]) if not baseline_row.empty else float("nan")
        )
        best = valid.loc[valid["hold_sharpe"].idxmax()]
        n_pos_ci = int((valid["hold_ci_lo"] > 0).sum())
        n_straddle = int(((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum())
        sig_positive = valid[valid["hold_ci_lo"] > 0]
        print()
        print("Honest interpretation:")
        print(
            f"  Headline best (point estimate only -- NOT a significance test): "
            f"{best['hold_sharpe']:+.3f}  (alpha={best['alpha']:.2f})"
        )
        if not pd.isna(baseline_s):
            print(f"  Baseline (alpha=0.0, no shrinkage): {baseline_s:+.3f}")
            delta = float(best["hold_sharpe"]) - baseline_s
            print(f"  Best - baseline                   : {delta:+.3f}")
        print(f"  Configs with CI strictly > 0    : {n_pos_ci} / {len(valid)}")
        print(f"  Configs with CI straddling zero : {n_straddle} / {len(valid)}")
        if not sig_positive.empty:
            best_sig = sig_positive.loc[sig_positive["hold_sharpe"].idxmax()]
            print(f"  Best config with CI > 0         : alpha={best_sig['alpha']:.2f}")
        else:
            print(
                "  Best config with CI > 0         : NONE (no statistically significant edge found)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
