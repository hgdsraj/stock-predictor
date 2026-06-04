#!/usr/bin/env python
"""Phase 11: feature pruning driven by per-feature audit.

The Phase 6 per-feature-audit (`scripts/per_feature_audit.py`) reports,
for each feature `f`, what happens to OOS IC IR when `f` is shifted +1
day while all others are held as-is. Two interpretations of the
resulting `pct_drop`:

  - HIGH pct_drop  -> shifting hurt a lot  -> feature was doing real
    same-day work (legitimate short-term effect, or possibly residual
    leakage candidate).
  - LOW pct_drop   -> shifting barely mattered -> feature has little
    same-day information to lose. Could be noise, could be a robust
    slowly-changing variable.

The roadmap calls for testing whether dropping the **bottom-quartile**
by pct_drop (i.e. the lowest-impact features) reduces overfitting and
improves holdout. As a sanity check we also test dropping the
**top-quartile** (the most-impact features — usually a stress test
that strips real signal). The reference Phase 8 best config is the
baseline.

Output: reports/phase11_feature_pruning_<start>_<end>_n<N>.csv with one
row per config (baseline + 2 prunings), including the list of dropped
features.

Requires: an up-to-date per-feature audit at the path passed via
`--audit-csv` (defaults to reports/per_feature_audit.csv). Run the
audit on the same universe FIRST:

    uv run python scripts/per_feature_audit.py \\
        --start 2014-01-01 --end 2024-12-31 \\
        --n-tickers 150 --horizon 5 --top-n 20

Then run this sweep:

    uv run python scripts/phase11_feature_pruning.py \\
        --start 2014-01-01 --end 2024-12-31 --n-tickers 150
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

log = logging.getLogger("phase11")

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
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument(
        "--audit-csv",
        default=str(REPORTS_DIR / "per_feature_audit.csv"),
        help="Per-feature audit CSV (output of scripts/per_feature_audit.py)",
    )
    p.add_argument(
        "--quartile",
        type=float,
        default=0.25,
        help=(
            "Quantile fraction to drop. 0.25 drops the bottom/top quarter; "
            "0.50 drops the bottom/top half (more aggressive)."
        ),
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def load_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Audit CSV not found at {path}. Run per_feature_audit.py first.")
    df = pd.read_csv(path)
    required = {"feature", "pct_drop"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Audit CSV missing required columns: {missing}")
    # Drop rows with NaN pct_drop (audit failed for that feature)
    n_before = len(df)
    df = df.dropna(subset=["pct_drop"]).reset_index(drop=True)
    n_after = len(df)
    if n_after < n_before:
        log.warning("Dropped %d audit rows with NaN pct_drop", n_before - n_after)
    if n_after == 0:
        raise SystemExit("Audit CSV has no valid (non-NaN) pct_drop rows.")
    return df


def select_quartile(audit: pd.DataFrame, quartile: float, *, which: str) -> list[str]:
    """Return the feature names in the bottom (`which='bottom'`) or
    top (`which='top'`) `quartile` fraction by pct_drop."""
    if which not in ("bottom", "top"):
        raise ValueError(f"which must be 'bottom' or 'top', got {which}")
    if not 0.0 < quartile < 1.0:
        raise ValueError(f"quartile must be in (0,1); got {quartile}")
    sorted_df = audit.sort_values("pct_drop", ascending=True).reset_index(drop=True)
    n = len(sorted_df)
    k = max(1, int(round(n * quartile)))
    if which == "bottom":
        return sorted_df.head(k)["feature"].tolist()
    else:
        return sorted_df.tail(k)["feature"].tolist()


def build_config(
    args: argparse.Namespace,
    *,
    feature_exclude: tuple[str, ...],
) -> PipelineV5Config:
    """Phase 8 best config with a feature blocklist applied."""
    return PipelineV5Config(
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
        meta_mode="binary",
        # Phase 8 pin (matches Phase 10 baseline)
        meta_walk_forward_folds=1,
        meta_per_sector=False,
        ranks_only=True,
        use_triple_barrier_labels=False,
        feature_exclude=feature_exclude,
    )


def run_one(cfg: PipelineV5Config, label: str, excluded: list[str]) -> dict:
    t0 = time.time()
    try:
        res = run_pipeline_v5(cfg)
    except Exception as e:  # noqa: BLE001
        log.exception("Run %s failed: %s", label, e)
        return {
            "label": label,
            "n_excluded": len(excluded),
            "excluded_features": ";".join(excluded),
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
        "n_excluded": len(excluded),
        "excluded_features": ";".join(excluded),
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

    audit_path = Path(args.audit_csv)
    audit = load_audit(audit_path)
    print(f"Loaded audit: {len(audit)} features from {audit_path}")
    print()

    bottom = select_quartile(audit, args.quartile, which="bottom")
    top = select_quartile(audit, args.quartile, which="top")
    print(f"Bottom {args.quartile:.0%} by pct_drop (least same-day work; candidates for noise):")
    for f in bottom:
        row = audit[audit["feature"] == f].iloc[0]
        print(f"  {f:30s}  pct_drop = {row['pct_drop']:+7.2f}%")
    print()
    print(f"Top {args.quartile:.0%} by pct_drop (most same-day work; usually real signal):")
    for f in top:
        row = audit[audit["feature"] == f].iloc[0]
        print(f"  {f:30s}  pct_drop = {row['pct_drop']:+7.2f}%")
    print()

    runs: list[tuple[str, list[str], PipelineV5Config]] = [
        ("baseline (no pruning)", [], build_config(args, feature_exclude=())),
        (
            f"drop bottom {args.quartile:.0%}",
            bottom,
            build_config(args, feature_exclude=tuple(bottom)),
        ),
        (
            f"drop top {args.quartile:.0%}",
            top,
            build_config(args, feature_exclude=tuple(top)),
        ),
    ]

    print(f"Phase 11 sweep: {len(runs)} runs")
    print(f"  universe: {args.n_tickers} tickers, {args.start} -> {args.end}")
    print(f"  horizons: {args.horizons}")
    print()

    rows: list[dict] = []
    for i, (label, excluded, cfg) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {label} (n_excluded={len(excluded)}) ...", flush=True)
        row = run_one(cfg, label, excluded)
        rows.append(row)
        s = row["hold_sharpe"]
        lo = row["hold_ci_lo"]
        hi = row["hold_ci_hi"]
        dd = row["hold_dd"]
        elapsed = row["elapsed_s"]
        if row["error"]:
            print(f"  -> FAILED in {elapsed:.0f}s: {row['error']}")
        else:
            print(
                f"  -> hold Sharpe={s:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  "
                f"DD={dd:+.2%}  ({elapsed:.0f}s)"
            )
        print()

    df = pd.DataFrame(rows)
    print("=" * 110)
    print("Phase 11 feature-pruning sweep results")
    print("=" * 110)
    fmt = {
        "dev_sharpe": "{:+.3f}".format,
        "hold_sharpe": "{:+.3f}".format,
        "hold_ci_lo": "{:+.3f}".format,
        "hold_ci_hi": "{:+.3f}".format,
        "hold_dd": "{:+.2%}".format,
        "hold_hit": "{:+.3f}".format,
        "hold_vol": "{:+.3f}".format,
    }
    # Don't print the long excluded_features list to stdout
    show_cols = [c for c in df.columns if c != "excluded_features"]
    print(df[show_cols].to_string(index=False, formatters=fmt))
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / (
        f"phase11_feature_pruning_{args.start}_{args.end}_n{args.n_tickers}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}  (full excluded_features included in CSV)")

    # Diagnostics
    valid = df[df["error"] == ""]
    if not valid.empty:
        baseline_row = valid[valid["label"] == "baseline (no pruning)"]
        baseline_s = (
            float(baseline_row["hold_sharpe"].iloc[0]) if not baseline_row.empty else float("nan")
        )
        best = valid.loc[valid["hold_sharpe"].idxmax()]
        n_pos_ci = int((valid["hold_ci_lo"] > 0).sum())
        n_straddle = int(((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum())
        print()
        print("Honest interpretation:")
        print(
            f"  Headline best (point estimate only — NOT a significance test): "
            f"{best['hold_sharpe']:+.3f}  ({best['label']})"
        )
        if not pd.isna(baseline_s):
            print(f"  Baseline (no pruning): {baseline_s:+.3f}")
            delta = float(best["hold_sharpe"]) - baseline_s
            print(
                f"  Best - baseline   : {delta:+.3f}  "
                "(delta has its own uncertainty; do not over-interpret)"
            )
        print(f"  Configs with CI strictly > 0    : {n_pos_ci} / {len(valid)}")
        print(f"  Configs with CI straddling zero : {n_straddle} / {len(valid)}")
        print()
        print(
            "Expected interpretations:"
            "\n  - 'drop bottom' WORSE than baseline -> low-pct_drop features"
            "\n    were carrying some signal after all; do not prune them."
            "\n  - 'drop bottom' ~SAME or BETTER than baseline -> safe to drop."
            "\n  - 'drop top' MUCH WORSE -> sanity check passes; top features"
            "\n    were doing real work."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
