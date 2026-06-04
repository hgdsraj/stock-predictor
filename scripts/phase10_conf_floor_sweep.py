#!/usr/bin/env python
"""Phase 10: confidence-floor sweep on the best Phase 8 config.

Phase 9 introduced confidence-weighted sizing as an alternative to the
Phase 8 binary meta-gate. With the default floor=0.5 it made things worse
(HOLDOUT Sharpe -0.57 vs Phase 8's -0.16). Hypothesis: a higher floor
should approximate the binary gate's hard-refusal behaviour. If a higher
floor smoothly recovers the Phase 8 result, that's a useful default; if
not, we deprecate confidence mode in favour of binary.

The sweep:
- a fixed-binary baseline (= reproduces Phase 8 best config exactly,
  with `meta_walk_forward_folds=1` and `meta_per_sector=False` explicitly
  pinned so the baseline can't drift if Phase 9 defaults move)
- confidence mode at floor ∈ {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}
  (cap fixed at 1.0)

Output: reports/phase10_conf_floor_sweep_<start>_<end>_n<N>.csv with one
row per run, including dev Sharpe, holdout Sharpe, holdout 95% block-
bootstrap CI, holdout max drawdown, holdout hit ratio, and annualised vol.
The filename is stamped with the sweep's universe so repeat runs don't
clobber each other.

Usage (small smoke test):
    uv run python scripts/phase10_conf_floor_sweep.py \\
        --start 2018-01-01 --end 2024-12-31 --n-tickers 60

Usage (production, matches Phase 8 best-config grid):
    uv run python scripts/phase10_conf_floor_sweep.py \\
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

log = logging.getLogger("phase10")

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
        "--floors",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75],
        help="Confidence-mode floor values to sweep",
    )
    p.add_argument(
        "--cap",
        type=float,
        default=1.0,
        help="Confidence-mode cap (fixed across the sweep)",
    )
    # NB: the binary Phase 8 baseline is ALWAYS included; there is no flag
    # to suppress it. Without the baseline this sweep has nothing to compare
    # the confidence runs against, so making it optional would be a footgun.
    p.add_argument("--bootstrap-n", type=int, default=500)
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def build_config(
    args: argparse.Namespace,
    *,
    meta_mode: str,
    meta_conf_floor: float,
    meta_conf_cap: float,
) -> PipelineV5Config:
    """Best Phase 8 config + the meta-mode/floor under test.

    The Phase 9 fields (`meta_walk_forward_folds`, `meta_per_sector`) are
    pinned to their Phase 8 single-pass-global defaults so the binary
    baseline is guaranteed to reproduce the documented Phase 8 best result
    even if the pipeline defaults change in a future phase. Reviewer C1.
    """
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
        # Phase 8 best: no sector/regime/tier2 (raw ranks-only), HRP, meta on
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        beta_neutralise=False,
        use_meta_labelling=True,
        meta_threshold=args.meta_threshold,
        meta_mode=meta_mode,
        meta_conf_floor=meta_conf_floor,
        meta_conf_cap=meta_conf_cap,
        # Pin Phase 9 fields to their Phase 8 single-pass-global defaults so
        # the baseline can't silently drift if pipeline defaults change.
        meta_walk_forward_folds=1,
        meta_per_sector=False,
        ranks_only=True,
        use_triple_barrier_labels=False,
    )


def run_one(cfg: PipelineV5Config, label: str) -> dict:
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

    # Validate cap first, then each floor against cap (reviewer H3).
    if not 0.0 < args.cap <= 1.0:
        raise SystemExit(f"--cap must be in (0, 1]; got {args.cap}")
    for f in args.floors:
        if not 0.0 <= f < args.cap:
            raise SystemExit(
                f"--floors values must be in [0, {args.cap}) (i.e. >= 0 and < cap); got {f}"
            )

    runs: list[tuple[str, PipelineV5Config]] = []
    # Phase 8 binary baseline (always included; see arg-parser comment).
    # meta_conf_floor/cap are passed but the pipeline ignores them when
    # meta_mode='binary' (see pipeline_v5.py:152-157).
    cfg_b = build_config(
        args,
        meta_mode="binary",
        meta_conf_floor=0.5,
        meta_conf_cap=1.0,
    )
    runs.append(("binary (Phase 8 baseline)", cfg_b))
    for floor in args.floors:
        cfg = build_config(
            args,
            meta_mode="confidence",
            meta_conf_floor=floor,
            meta_conf_cap=args.cap,
        )
        runs.append((f"confidence floor={floor:.2f} cap={args.cap:.2f}", cfg))

    print(f"Phase 10 sweep: {len(runs)} runs")
    print(f"  universe: {args.n_tickers} tickers, {args.start} -> {args.end}")
    print(f"  horizons: {args.horizons}")
    print(f"  meta_threshold: {args.meta_threshold}")
    print(f"  floors: {args.floors}  cap: {args.cap}")
    print()

    rows: list[dict] = []
    for i, (label, cfg) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {label} ...", flush=True)
        row = run_one(cfg, label)
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
    cols = [
        "label",
        "dev_sharpe",
        "hold_sharpe",
        "hold_ci_lo",
        "hold_ci_hi",
        "hold_dd",
        "hold_hit",
        "hold_vol",
        "elapsed_s",
        "error",
    ]
    df = df[cols]

    print("=" * 100)
    print("Phase 10 confidence-floor sweep results")
    print("=" * 100)
    fmt = {
        "dev_sharpe": "{:+.3f}".format,
        "hold_sharpe": "{:+.3f}".format,
        "hold_ci_lo": "{:+.3f}".format,
        "hold_ci_hi": "{:+.3f}".format,
        "hold_dd": "{:+.2%}".format,
        "hold_hit": "{:+.3f}".format,
        "hold_vol": "{:+.3f}".format,
    }
    print(df.to_string(index=False, formatters=fmt))
    print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Stamp the filename with universe params so reruns with different
    # configs don't clobber each other (reviewer M3).
    out_csv = REPORTS_DIR / (
        f"phase10_conf_floor_sweep_{args.start}_{args.end}_n{args.n_tickers}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Diagnostics (reviewer M2: be careful not to invite headline-only
    # readings; always frame "best" in CI terms).
    valid = df[df["error"] == ""]
    if not valid.empty:
        best = valid.loc[valid["hold_sharpe"].idxmax()]
        binaries = valid[valid["label"].str.startswith("binary")]
        binary_s = float(binaries["hold_sharpe"].iloc[0]) if not binaries.empty else float("nan")
        n_pos_ci = int((valid["hold_ci_lo"] > 0).sum())
        n_straddle = int(((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum())
        # Statistical-significance frame: only configs whose CI excludes 0
        # are 'real' positives; everything else is noise around zero.
        sig_positive = valid[valid["hold_ci_lo"] > 0]
        sig_best_label = (
            sig_positive.loc[sig_positive["hold_sharpe"].idxmax(), "label"]
            if not sig_positive.empty
            else None
        )

        print()
        print("Honest interpretation:")
        print(
            f"  Headline best (point estimate only — NOT a significance test): "
            f"{best['hold_sharpe']:+.3f}  ({best['label']})"
        )
        if not pd.isna(binary_s):
            print(f"  Binary Phase 8 baseline: {binary_s:+.3f}")
            delta = float(best["hold_sharpe"]) - binary_s
            print(
                f"  Best - binary  : {delta:+.3f}  "
                "(this delta has its OWN uncertainty; do not over-interpret)"
            )
        print(f"  Configs with CI strictly > 0       : {n_pos_ci} / {len(valid)}")
        print(f"  Configs with CI straddling zero    : {n_straddle} / {len(valid)}")
        if sig_best_label is not None:
            print(f"  Best config with CI > 0            : {sig_best_label}")
        else:
            print("  Best config with CI > 0            : NONE (no real edge found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
