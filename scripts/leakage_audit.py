#!/usr/bin/env python
"""Phase 6 leakage audit.

This script runs the same model and CV twice on the same universe:

  variant_a — "as-is": features computed at close-of-t are used to predict
              fwd_return at t (the standard path).
  variant_b — "hard cutoff": every feature is additionally shifted forward
              by 1 day, so feature_at_t only contains information from
              close-of-(t-1).

Reads the IC IR for both variants. Some drop between as-is and hard-cutoff
is EXPECTED and legitimate (e.g. short-term reversal: today's big move
predicts tomorrow's opposite-sign return; that's a real factor, not a leak).
A *catastrophic* drop or a sign flip is the signature of real leakage.

History of findings:
  - Phase 5 (before P6L1 fix): h=5d as-is IC IR +2.45, hard t-1 IC IR
    -0.58 (sign-flipped). Diagnosed: vol-scaled label denominator was
    computed through close-of-t and shared `close[t]` with features like
    `ret_1d`. Fixed in labels.py::compute_vol_scaled_forward_returns by
    shifting the denominator by +1 day.

  - Phase 6 (after P6L1): h=5d as-is IC IR ~+3.7, hard t-1 IC IR ~+1.8.
    The Δ is no longer catastrophic. Reading: ~+1.8 is the conservative
    lower bound on the true signal (assumes all the extra information in
    the as-is run is leakage); ~+3.7 is the upper bound (assumes all of
    it is legitimate short-term reversal). Reality is somewhere in
    between, probably closer to the lower bound. Treat all production
    backtest IR numbers as needing similar discount.

No claims; the user reads the result and decides.
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


def _shift_feats_one_day(feats: pd.DataFrame) -> pd.DataFrame:
    """Apply an extra .shift(1) per ticker so feature_at_t reflects
    information strictly through close-of-(t-1)."""
    out_chunks: list[pd.DataFrame] = []
    for tkr, sub in feats.groupby(level="ticker"):
        sub2 = sub.copy()
        sub2.index = sub2.index.droplevel("ticker")
        sub2 = sub2.sort_index().shift(1)
        sub2["ticker"] = tkr
        sub2 = sub2.set_index("ticker", append=True).swaplevel("ticker", -1)
        out_chunks.append(sub2)
    out = pd.concat(out_chunks).sort_index()
    out.index = out.index.set_names(["date", "ticker"])
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--n-tickers", type=int, default=60)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5])
    p.add_argument("--universe-sampling", default="current")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("leakage_audit")

    universe_cfg = PipelineConfig(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling=args.universe_sampling,
    )
    tickers, _ = select_universe(universe_cfg)
    log.info("Universe: %d tickers", len(tickers))

    raw = prices_mod.long_panel(tickers, start=args.start, end=args.end)
    close = raw["adj_close"].unstack("ticker").sort_index()
    volume = raw["volume"].unstack("ticker").sort_index()

    feats_a = add_cross_sectional_ranks(compute_technical_features(close, volume=volume))
    feats_b = _shift_feats_one_day(feats_a)
    labels = long_labels(close, horizons=tuple(args.horizons), include_vol_scaled=True)
    cv = CVConfig(train_years=3, test_months=6, embargo_days=25)
    gbm = GBMConfig()

    print()
    print("=" * 72)
    print(f"{'Horizon':<10} {'variant':<20} {'ic_mean':<12} {'ic_ir':<10} {'hit':<8}")
    print("-" * 72)
    rows: list[dict] = []
    for h in args.horizons:
        for variant, feats in (("a_as_is", feats_a), ("b_hard_t-1", feats_b)):
            X, y_target, y_return, y_bin = assemble_dataset(feats, labels, h, target="vs")
            preds = walk_forward_predict(X, y_target, cv, model="gbm", gbm_cfg=gbm)
            if preds.empty:
                print(f"h={h:<8} {variant:<20} (no preds)")
                continue
            hit, ic = _diagnostics(preds, y_return, y_bin)
            print(
                f"h={h:<8} {variant:<20} {ic['ic_mean']:+.5f}    {ic['ic_ir']:+.3f}     {hit:.4f}"
            )
            rows.append({"horizon": h, "variant": variant, **ic, "hit": hit})
    print("=" * 72)

    # Verdict: compare IC IR by horizon between the two variants.
    print()
    df = pd.DataFrame(rows)
    if df.empty:
        return 0
    for h in df["horizon"].unique():
        sub = df[df["horizon"] == h]
        if len(sub) < 2:
            continue
        a = sub[sub["variant"] == "a_as_is"]["ic_ir"].iloc[0]
        b = sub[sub["variant"] == "b_hard_t-1"]["ic_ir"].iloc[0]
        delta = b - a
        pct = (delta / a * 100) if a else float("nan")
        verdict = (
            "OK (signal survives strict cutoff)"
            if abs(b) > 0.5 * abs(a) and np.sign(a) == np.sign(b)
            else "SUSPECT (IC IR collapses under strict cutoff — likely same-day leak)"
            if abs(b) < 0.25 * abs(a)
            else "PARTIAL (large drop; investigate)"
        )
        print(
            f"h={h}: as_is ic_ir={a:+.3f}, hard_t-1 ic_ir={b:+.3f} (Δ={delta:+.3f}, {pct:+.0f}%) → {verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
