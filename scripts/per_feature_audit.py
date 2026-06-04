#!/usr/bin/env python
"""Per-feature leakage audit.

Phase 6 found a label-leak via the shared close[t] between features and the
vol-scaled denominator. The fix was global, but it raises the question:
*which features* still contribute most of the delta between an as-is run
and a hard-cutoff (feature-shifted-by-1) run?

This script answers that by:
  1. Training a single LightGBM on the as-is features + target.
  2. Training a second LightGBM where ONE feature is shifted +1 day while
     all others remain as-is.
  3. Comparing IC IR between the two. If a feature's per-feature shift
     drops IC IR by more than ~30%, that feature is doing disproportionate
     same-day-leakage work and is a candidate for removal or further
     leakage scrutiny.

For each feature we record:
  - as-is OOS IC IR (constant across rows; computed once for reference)
  - shifted OOS IC IR
  - delta
  - importance from the as-is model (LightGBM feature importance)

Output: a markdown table sorted by largest IC IR drop, plus a CSV at
reports/per_feature_audit.csv.

This is NOT cheap (one CV per feature). With ~40 features and the full
historical universe it's ~30-45 min. On a small smoke-test universe
(30 names, 5 years) it's ~5 min.
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np
import pandas as pd

from stockpred.config import CVConfig
from stockpred.data import prices as prices_mod
from stockpred.features.cross_sectional import add_cross_sectional_ranks
from stockpred.features.technical import compute_technical_features
from stockpred.labels import long_labels
from stockpred.models.gbm import GBMConfig
from stockpred.pipeline import (
    PipelineConfig,
    _diagnostics,
    assemble_dataset,
    select_universe,
    walk_forward_predict,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-tickers", type=int, default=30)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--universe-sampling", default="current")
    p.add_argument(
        "--top-n", type=int, default=20, help="Only audit the top-N features by importance"
    )
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.WARNING if not args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("per_feature_audit")

    log.warning("Loading universe...")
    universe_cfg = PipelineConfig(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling=args.universe_sampling,
    )
    tickers, _ = select_universe(universe_cfg)
    log.warning("Universe: %d tickers", len(tickers))

    raw = prices_mod.long_panel(tickers, start=args.start, end=args.end)
    close = raw["adj_close"].unstack("ticker").sort_index()
    volume = raw["volume"].unstack("ticker").sort_index()

    feats = add_cross_sectional_ranks(compute_technical_features(close, volume=volume))
    labels = long_labels(close, horizons=(args.horizon,), include_vol_scaled=True)
    cv = CVConfig(train_years=3, test_months=6, embargo_days=25)
    gbm = GBMConfig()

    X, y_target, y_return, y_bin = assemble_dataset(feats, labels, args.horizon, target="vs")
    log.warning("Dataset: %s rows x %s cols", *X.shape)

    # Baseline (as-is)
    log.warning("Training as-is baseline...")
    baseline_preds = walk_forward_predict(X, y_target, cv, model="gbm", gbm_cfg=gbm)
    baseline_hit, baseline_ic = _diagnostics(baseline_preds, y_return, y_bin)
    base_ic_ir = baseline_ic["ic_ir"]
    log.warning("Baseline IC IR: %+.3f", base_ic_ir)

    feature_cols = list(X.columns)
    if args.top_n and len(feature_cols) > args.top_n:
        # Score features by a single quick fit to pick the top-N to audit.
        # (Saves time vs auditing all 36+.)
        from stockpred.models.gbm import train_gbm

        booster = train_gbm(X, y_target, cfg=gbm)
        try:
            imp = pd.Series(booster.feature_importance(importance_type="gain"), index=feature_cols)
            feature_cols = imp.sort_values(ascending=False).head(args.top_n).index.tolist()
            log.warning("Auditing top-%d features by gain importance", args.top_n)
        except Exception:  # noqa: BLE001
            log.warning("Could not get feature importances; auditing all")

    # For each feature, shift it by 1 day per ticker, re-train, compare.
    rows: list[dict] = []
    for i, col in enumerate(feature_cols, 1):
        log.warning("[%d/%d] auditing feature %s ...", i, len(feature_cols), col)
        X_shifted = X.copy()
        # Group by ticker so shift is per-asset chronological.
        X_shifted[col] = X[col].groupby(level="ticker").shift(1)
        # Drop rows where the shift introduced NaN (first date per ticker).
        valid = X_shifted[col].notna()
        X_use = X_shifted.loc[valid]
        y_use = y_target.loc[valid]
        try:
            preds = walk_forward_predict(X_use, y_use, cv, model="gbm", gbm_cfg=gbm)
        except Exception as e:  # noqa: BLE001
            log.warning("  fold failed: %s", e)
            rows.append(
                {
                    "feature": col,
                    "shifted_ic_ir": float("nan"),
                    "delta_ic_ir": float("nan"),
                    "pct_drop": float("nan"),
                }
            )
            continue
        _, ic = _diagnostics(preds, y_return.loc[valid], y_bin.loc[valid])
        shifted_ir = ic["ic_ir"]
        delta = shifted_ir - base_ic_ir
        # Gate the percentage on baseline magnitude; near-zero baselines
        # turn small absolute deltas into meaningless huge percentages.
        if abs(base_ic_ir) < 0.05:
            pct = float("nan")
        else:
            # Sign convention: "drop %" should be positive when shifting
            # HURT the signal (legitimate same-day info loss).
            pct = -delta / abs(base_ic_ir) * 100
        rows.append(
            {
                "feature": col,
                "shifted_ic_ir": shifted_ir,
                "delta_ic_ir": delta,
                "pct_drop": pct,
            }
        )
        log.warning("  shifted IC IR=%+.3f  delta=%+.3f  pct_drop=%+.0f%%", shifted_ir, delta, pct)

    df = pd.DataFrame(rows).sort_values("delta_ic_ir")  # most-leaky first (largest drop)
    out_csv = "reports/per_feature_audit.csv"
    df.to_csv(out_csv, index=False)

    print()
    print("=" * 72)
    print(f"Per-feature leakage audit  (baseline IC IR = {base_ic_ir:+.3f})")
    print("=" * 72)
    print(f"{'feature':<30} {'shifted_ic_ir':<14} {'delta':<10} {'pct':<8}")
    print("-" * 72)
    for _, r in df.iterrows():
        print(
            f"{r['feature']:<30} {r['shifted_ic_ir']:+.3f}        "
            f"{r['delta_ic_ir']:+.3f}    {r['pct_drop']:+.0f}%"
        )
    print()
    print(f"Saved to {out_csv}")
    print()
    print(
        "Reading: a feature with pct_drop > 30% (i.e. removing one day of "
        "its info dropped IC IR by >30%) is doing a lot of same-day work. "
        "That's usually short-term reversal — a legitimate effect — but "
        "features whose drop is much larger than peers are worth a "
        "closer look for leakage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
