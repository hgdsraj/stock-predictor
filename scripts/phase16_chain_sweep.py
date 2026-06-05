#!/usr/bin/env python
"""Phase 16: chain triple-barrier + confidence-sizing on top of the
Phase 13 best config and see if any combination improves on it.

The Phase 13 best is binary meta + ranks_only + HRP + EDGAR items
(hold Sharpe +0.173, CI [-0.32, +0.58], DD -8.2%). Open questions:

  Q1. Does swapping the simple forward-return label for the Lopez de
      Prado triple-barrier label help (the original Phase 7 motivation)?
  Q2. Does the confidence-floor=0.60 sweet spot from Phase 10 still
      apply on top of the Phase 13 baseline?
  Q3. Do both stack additively, or is one strictly better?

The sweep runs 4 configs (baseline + TB + conf + TB&conf) on the
production 150-ticker x 11-yr universe and reports HOLDOUT Sharpe +
95% block-bootstrap CI per config. Reproducibility check: the
baseline run should match the documented Phase 13 result (+0.173).

Output: reports/phase16_chain_sweep_<start>_<end>_n<N>.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

log = logging.getLogger("phase16")

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-tickers", type=int, default=150)
    p.add_argument("--universe-sampling", default="current")
    p.add_argument("--horizons", type=int, nargs="+", default=[5])
    p.add_argument("--k-pct", type=float, default=0.15)
    p.add_argument("--sector-cap", type=float, default=0.30)
    p.add_argument("--min-trade-threshold", type=float, default=0.005)
    p.add_argument("--holdout-years", type=int, default=2)
    p.add_argument("--meta-threshold", type=float, default=0.55)
    p.add_argument(
        "--conf-floor",
        type=float,
        default=0.60,
        help="Phase 10 sweet spot; sweep tries binary + confidence(floor).",
    )
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument(
        "--with-gdelt",
        action="store_true",
        help="Also include GDELT features (requires bulk fetch done).",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def _base_cfg(args: argparse.Namespace, **overrides) -> PipelineV5Config:
    """Phase 13 best config; per-test overrides applied on top."""
    base = dict(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling=args.universe_sampling,
        horizons=tuple(args.horizons),
        model="gbm",
        ensemble_weighting="equal",
        position_sizing="hrp",
        k_per_side_pct=args.k_pct,
        sector_cap_gross=(args.sector_cap if args.sector_cap > 0 else None),
        min_trade_threshold=args.min_trade_threshold,
        holdout_years=args.holdout_years,
        bootstrap_n=args.bootstrap_n,
        bootstrap_method="block",
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        beta_neutralise=False,
        use_meta_labelling=True,
        meta_threshold=args.meta_threshold,
        # default Phase 13: binary meta
        meta_mode="binary",
        meta_conf_floor=args.conf_floor,
        meta_conf_cap=1.0,
        meta_walk_forward_folds=1,
        meta_per_sector=False,
        ranks_only=True,
        # default Phase 13: simple labels (NOT triple-barrier)
        use_triple_barrier_labels=False,
        use_edgar_features=False,
        use_edgar_item_features=True,
        use_gdelt_features=args.with_gdelt,
    )
    base.update(overrides)
    return PipelineV5Config(**base)


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
            "hold_hit": float("nan"),
            "hold_vol": float("nan"),
            "elapsed_s": time.time() - t0,
            "error": str(e),
        }
    dev = res.get("metrics", {}) or {}
    hold = res.get("holdout_metrics", {}) or {}
    ci = res.get("bootstrap_sharpe", {}) or {}
    return {
        "label": label,
        "dev_sharpe": dev.get("sharpe", float("nan")),
        "hold_sharpe": hold.get("sharpe", float("nan")),
        "hold_ci_lo": ci.get("sharpe_lo", float("nan")),
        "hold_ci_hi": ci.get("sharpe_hi", float("nan")),
        "hold_dd": hold.get("max_drawdown", float("nan")),
        "hold_hit": hold.get("hit_ratio", float("nan")),
        "hold_vol": hold.get("ann_vol", float("nan")),
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

    # 4 configs to compare
    runs: list[tuple[str, PipelineV5Config]] = [
        ("baseline (Phase 13)", _base_cfg(args)),
        (
            f"+TB",
            _base_cfg(args, use_triple_barrier_labels=True),
        ),
        (
            f"+conf(floor={args.conf_floor})",
            _base_cfg(args, meta_mode="confidence"),
        ),
        (
            f"+TB +conf(floor={args.conf_floor})",
            _base_cfg(
                args,
                use_triple_barrier_labels=True,
                meta_mode="confidence",
            ),
        ),
    ]

    print(f"Phase 16 chain sweep: {len(runs)} configs")
    print(f"  universe: {args.n_tickers} tickers, {args.start} -> {args.end}")
    print(f"  horizons: {args.horizons}")
    print(f"  with_gdelt: {args.with_gdelt}")
    print()

    rows: list[dict] = []
    for i, (label, cfg) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {label} ...", flush=True)
        row = run_one(label, cfg)
        rows.append(row)
        s = row["hold_sharpe"]
        lo = row["hold_ci_lo"]
        hi = row["hold_ci_hi"]
        dd = row["hold_dd"]
        if row["error"]:
            print(f"  -> FAILED in {row['elapsed_s']:.0f}s: {row['error']}")
        else:
            print(
                f"  -> hold Sharpe={s:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  "
                f"DD={dd:+.2%}  ({row['elapsed_s']:.0f}s)"
            )
        print()

    df = pd.DataFrame(rows)
    fmt = {
        "dev_sharpe": "{:+.3f}".format,
        "hold_sharpe": "{:+.3f}".format,
        "hold_ci_lo": "{:+.3f}".format,
        "hold_ci_hi": "{:+.3f}".format,
        "hold_dd": "{:+.2%}".format,
        "hold_hit": "{:+.3f}".format,
        "hold_vol": "{:+.3f}".format,
    }
    show_cols = [c for c in df.columns if c != "error" or df["error"].any()]
    print("=" * 110)
    print("Phase 16 chain sweep results")
    print("=" * 110)
    print(df[show_cols].to_string(index=False, formatters=fmt))
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / (f"phase16_chain_sweep_{args.start}_{args.end}_n{args.n_tickers}.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Honest interpretation
    valid = df[df["error"] == ""]
    if not valid.empty:
        baseline_s = (
            float(valid[valid["label"].str.startswith("baseline")]["hold_sharpe"].iloc[0])
            if (valid["label"].str.startswith("baseline")).any()
            else float("nan")
        )
        best = valid.loc[valid["hold_sharpe"].idxmax()]
        n_pos_ci = int((valid["hold_ci_lo"] > 0).sum())
        n_straddle = int(((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum())
        sig_positive = valid[valid["hold_ci_lo"] > 0]
        sig_best_label = (
            sig_positive.loc[sig_positive["hold_sharpe"].idxmax(), "label"]
            if not sig_positive.empty
            else None
        )

        print()
        print("Honest interpretation:")
        print(
            f"  Headline best (point estimate only -- NOT a significance test): "
            f"{best['hold_sharpe']:+.3f}  ({best['label']})"
        )
        if not pd.isna(baseline_s):
            print(f"  Baseline (Phase 13): {baseline_s:+.3f}")
            delta = float(best["hold_sharpe"]) - baseline_s
            print(f"  Best - baseline    : {delta:+.3f}  (delta has its own uncertainty)")
        print(f"  Configs with CI strictly > 0    : {n_pos_ci} / {len(valid)}")
        print(f"  Configs with CI straddling zero : {n_straddle} / {len(valid)}")
        if sig_best_label is not None:
            print(f"  Best config with CI > 0         : {sig_best_label}")
        else:
            print(
                "  Best config with CI > 0         : NONE (no statistically significant edge found)"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
