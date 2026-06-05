#!/usr/bin/env python
"""CLI for the Phase 5+ hyperparameter search (Optuna TPE).

Delegates all search logic to `stockpred.hypersearch`; this script handles
CLI arguments, console progress output, and saving results to reports/.

Fast-mode defaults (tuned for ~2-4 min/trial on a laptop):
  --n-tickers 25   --start 2015-01-01   --bootstrap-n 50

50 trials ≈ 2-4 hours. Increase --n-tickers for production-quality results.

Note: tuning uses a fixed universe every trial (universe_sampling "current"
= current S&P 500 constituents). Results may not fully generalise to the
complete universe; run the best config at full scale to validate.

Usage:
    uv run python scripts/run_hypersearch.py                       # 50 trials, ~2-4h
    uv run python scripts/run_hypersearch.py --n-trials 20         # quick smoke test
    uv run python scripts/run_hypersearch.py --n-trials 100 --n-tickers 40

Parallel / persistent study (run in N separate terminals):
    uv run python scripts/run_hypersearch.py \\
        --storage sqlite:///reports/hypersearch.db \\
        --study-name my_study --n-trials 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import optuna
import pandas as pd

from stockpred.hypersearch import (
    HypersearchConfig,
    PENALTY,
    best_trial_params,
    best_trial_sharpe,
    run_hypersearch,
    suggest_pipeline_config,
    trials_to_records,
)

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
log = logging.getLogger("hypersearch")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials (default: 50)")
    p.add_argument(
        "--n-tickers",
        type=int,
        default=25,
        help="Universe size per trial. 25 ≈ 2-4 min/trial; 50 ≈ 5-10 min/trial (default: 25)",
    )
    p.add_argument("--start", default="2015-01-01", help="History start date (default: 2015-01-01)")
    p.add_argument("--end", default=None, help="History end date (default: today)")
    p.add_argument("--holdout-years", type=int, default=2, help="Holdout window in years (default: 2)")
    p.add_argument(
        "--bootstrap-n",
        type=int,
        default=50,
        help="Bootstrap resamples for Sharpe CI (50=fast, 500=honest; default: 50)",
    )
    p.add_argument(
        "--universe-sampling",
        default="current",
        choices=["current", "first", "random"],
        help="How to pick tickers (default: current — same set every trial for consistency)",
    )
    p.add_argument("--seed", type=int, default=42, help="Optuna sampler seed (default: 42)")
    p.add_argument(
        "--study-name",
        default=None,
        help="Optuna study name. Auto-generated from timestamp if omitted.",
    )
    p.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL for persistence/parallelism (e.g. sqlite:///reports/hypersearch.db)",
    )
    p.add_argument(
        "--server-url",
        default="http://localhost:8000",
        help="Server base URL for generated curl command (default: http://localhost:8000)",
    )
    p.add_argument("--verbose", "-v", action="store_true", help="Show pipeline log output per trial")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Progress printer
# ─────────────────────────────────────────────────────────────────────────────


def _make_on_trial_done(n_trials: int):
    """Return a callback that prints one line per trial."""
    def on_trial_done(row: dict) -> None:
        num = row.get("trial", "?")
        sharpe = row.get("hold_sharpe", float("nan"))
        ci_lo = row.get("hold_ci_lo", float("nan"))
        ci_hi = row.get("hold_ci_hi", float("nan"))
        dd = row.get("hold_dd", float("nan"))
        elapsed = row.get("elapsed_s", float("nan"))
        error = row.get("error", "")
        tag = f"[{int(num) + 1:>3}/{n_trials}]"
        if error:
            print(f"{tag} FAILED ({elapsed:.0f}s): {error}", flush=True)
        else:
            s = f"{sharpe:+.3f}" if pd.notna(sharpe) else "  nan "
            lo = f"{ci_lo:+.3f}" if pd.notna(ci_lo) else "   nan"
            hi = f"{ci_hi:+.3f}" if pd.notna(ci_hi) else "   nan"
            d = f"{dd:+.2%}" if pd.notna(dd) else "   nan"
            print(f"{tag} Sharpe={s}  CI=[{lo},{hi}]  DD={d}  ({elapsed:.0f}s)", flush=True)
    return on_trial_done


# ─────────────────────────────────────────────────────────────────────────────
# curl / RefreshRequest body generation
# ─────────────────────────────────────────────────────────────────────────────


def _to_refresh_request_body(params: dict, cfg: HypersearchConfig) -> dict:
    """Convert best Optuna params to a RefreshRequest-compatible JSON body."""
    horizons = [int(h) for h in params.get("horizons", "5").split(",")]
    sc = params.get("sector_cap_gross", "0.30")
    sector_cap_gross = None if sc == "none" else float(sc)
    return {
        "phase": 5,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "n_tickers": None,  # full universe for production run
        "universe_sampling": cfg.universe_sampling,
        "horizons": horizons,
        "model": "gbm",
        "gbm": {
            "num_leaves": params.get("num_leaves", 63),
            "learning_rate": round(params.get("learning_rate", 0.03), 6),
            "n_estimators": params.get("n_estimators", 800),
        },
        "cv": {
            "train_years": params.get("train_years", 3),
            "test_months": 6,
            "embargo_days": 25,
            "min_train_obs": 1000,
        },
        "holdout_years": cfg.holdout_years,
        "position_sizing": params.get("position_sizing", "vol_scaled"),
        "k_per_side_pct": round(params.get("k_per_side_pct", 0.15), 4),
        "leverage_per_side": round(params.get("leverage_per_side", 1.0), 4),
        "sector_cap_gross": sector_cap_gross,
        "min_trade_threshold": round(params.get("min_trade_threshold", 0.005), 6),
        "ensemble_weighting": params.get("ensemble_weighting", "ic_ir"),
        "use_sector_features": params.get("use_sector_features", True),
        "use_tier2_features": params.get("use_tier2_features", True),
        "use_regime_features": params.get("use_regime_features", True),
        "beta_neutralise": params.get("beta_neutralise", False),
        "ranks_only": params.get("ranks_only", False),
        "use_meta_labelling": params.get("use_meta_labelling", False),
        "meta_threshold": round(params.get("meta_threshold", 0.55), 4),
        "meta_mode": params.get("meta_mode", "binary"),
        "meta_conf_floor": round(params.get("meta_conf_floor", 0.5), 4),
        "meta_conf_cap": 1.0,
        "meta_walk_forward_folds": 1,
        "bootstrap_n": 500,
        "bootstrap_method": "block",
    }


def _build_curl(body: dict, server_url: str) -> str:
    url = server_url.rstrip("/") + "/jobs/queue"
    body_json = json.dumps(body, separators=(", ", ": "))
    return f"curl -X POST \\\n  '{url}' \\\n  -H 'Content-Type: application/json' \\\n  -d '{body_json}'"


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


def _write_md_report(
    study: optuna.Study,
    rows: list[dict],
    out_md: Path,
    cfg: HypersearchConfig,
    curl_cmd: str,
) -> None:
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = [r for r in rows if not r.get("error")]
    n_total = len(valid)

    lines: list[str] = [
        "# Hyperparameter Search Results",
        "",
        f"Generated: {now}  ",
        f"Study: `{study.study_name}`  ",
        f"Universe: {cfg.n_tickers} tickers (`{cfg.universe_sampling}`), "
        f"{cfg.start_date} → {cfg.end_date or 'today'}  ",
        f"Trials completed: {len(study.trials)} ({n_total} succeeded)  ",
        "Objective: maximise **holdout Sharpe** "
        f"(last {cfg.holdout_years} years, never seen during tuning)  ",
        "",
        "## Top 10 Configs",
        "",
    ]

    if not valid:
        lines.append("_No successful trials._")
    else:
        top10 = valid[:10]
        cols = [
            ("Rank", None), ("Sharpe", "hold_sharpe"), ("CI lo", "hold_ci_lo"),
            ("CI hi", "hold_ci_hi"), ("Max DD", "hold_dd"), ("Ann Ret", "hold_ann_return"),
            ("Dev Sh", "dev_sharpe"), ("Sizing", "position_sizing"),
            ("Horizons", "horizons"), ("Meta", "use_meta_labelling"),
            ("Ranks", "ranks_only"), ("s", "elapsed_s"),
        ]
        header = " | ".join(h for h, _ in cols)
        sep = " | ".join("---" for _ in cols)
        lines += [f"| {header} |", f"| {sep} |"]
        for rank, row in enumerate(top10, 1):
            def fv(col, r=row, rk=rank):
                if col is None:
                    return str(rk)
                v = r.get(col, float("nan"))
                if col in ("hold_sharpe", "hold_ci_lo", "hold_ci_hi", "dev_sharpe"):
                    return f"{v:+.3f}" if pd.notna(v) else "nan"
                if col in ("hold_dd", "hold_ann_return"):
                    return f"{v:+.1%}" if pd.notna(v) else "nan"
                if col == "elapsed_s":
                    return f"{int(v)}s" if pd.notna(v) else "nan"
                return str(v) if pd.notna(v) else "nan"
            cells = " | ".join(fv(col) for _, col in cols)
            lines.append(f"| {cells} |")
        lines.append("")

    # Honest interpretation
    lines += ["## Honest Interpretation", ""]
    if valid:
        best = valid[0]
        s = best.get("hold_sharpe", float("nan"))
        lo = best.get("hold_ci_lo", float("nan"))
        hi = best.get("hold_ci_hi", float("nan"))
        n_pos_ci = sum(1 for r in valid if (r.get("hold_ci_lo") or float("nan")) > 0)
        n_straddle = sum(
            1 for r in valid
            if (r.get("hold_ci_lo") or float("nan")) <= 0
            and (r.get("hold_ci_hi") or float("nan")) >= 0
        )
        sig = [r for r in valid if (r.get("hold_ci_lo") or float("nan")) > 0]
        lines += [
            f"- **Best point estimate**: Sharpe = `{s:+.3f}` (trial {best.get('trial', '?')})",
            f"  95% block-bootstrap CI: [`{lo:+.3f}`, `{hi:+.3f}`]",
            f"- Configs with CI **strictly > 0** (real edge): **{n_pos_ci} / {n_total}**",
            f"- Configs with CI straddling zero (noise): {n_straddle} / {n_total}",
        ]
        if sig:
            bs = sig[0]
            lines.append(
                f"- **Best config with CI > 0**: trial {bs.get('trial', '?')}, "
                f"Sharpe = `{bs.get('hold_sharpe', float('nan')):+.3f}` "
                f"CI = [`{bs.get('hold_ci_lo', float('nan')):+.3f}`, "
                f"`{bs.get('hold_ci_hi', float('nan')):+.3f}`]"
            )
        else:
            lines.append("- **Best config with CI > 0**: _none — no statistically reliable edge found_")
        lines += [
            "",
            "> CI computed on a fast-mode universe. Validate the best config on the full "
            "S&P 500 before drawing conclusions.",
        ]
    else:
        lines.append("_No successful trials to interpret._")
    lines.append("")

    # Best config JSON
    lines += ["## Best Config Parameters", ""]
    best_params = best_trial_params(study)
    if best_params:
        lines += ["```json", json.dumps(best_params, indent=2, default=str), "```"]
    else:
        lines.append("_No successful trials._")
    lines.append("")

    # Curl command
    lines += [
        "## Queue This Run on the Server",
        "",
        "POST to `/jobs/queue` (no auth required):",
        "",
        "```bash",
        curl_cmd,
        "```",
        "",
        "Then launch the queued job:",
        "```bash",
        "curl -X POST \\",
        f"  '{cfg.start_date}' \\",
        "  # Replace with: POST /jobs/run/<queue_id>  -H 'X-Password: <your-password>'",
        "```",
        "",
        f"> `n_tickers` is set to `null` (full universe). "
        f"Tuning was done on {cfg.n_tickers} tickers.",
        "",
    ]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────


def _print_summary(study: optuna.Study, rows: list[dict], curl_cmd: str) -> None:
    print()
    print("=" * 100)
    print("Hyperparameter search — results")
    print("=" * 100)

    valid = [r for r in rows if not r.get("error")]
    if not valid:
        print("No completed trials.")
        return

    n_total = len(valid)
    cols = ["trial", "hold_sharpe", "hold_ci_lo", "hold_ci_hi", "hold_dd",
            "dev_sharpe", "position_sizing", "horizons", "use_meta_labelling",
            "ranks_only", "elapsed_s"]
    df = pd.DataFrame(valid)
    disp_cols = [c for c in cols if c in df.columns]
    fmt = {}
    for c in ["hold_sharpe", "hold_ci_lo", "hold_ci_hi", "dev_sharpe"]:
        if c in df.columns:
            fmt[c] = lambda x: f"{x:+.3f}" if pd.notna(x) else "  nan"
    if "hold_dd" in df.columns:
        fmt["hold_dd"] = lambda x: f"{x:+.2%}" if pd.notna(x) else "  nan"

    print(f"\nTop {min(10, len(df))} configs (by holdout Sharpe):\n")
    print(df.head(10)[disp_cols].to_string(index=False, formatters=fmt))

    best = valid[0]
    s, lo, hi = (best.get("hold_sharpe", float("nan")),
                 best.get("hold_ci_lo", float("nan")),
                 best.get("hold_ci_hi", float("nan")))
    n_pos_ci = sum(1 for r in valid if (r.get("hold_ci_lo") or float("nan")) > 0)
    sig = [r for r in valid if (r.get("hold_ci_lo") or float("nan")) > 0]

    print()
    print("Honest interpretation:")
    print(f"  Best estimate   : Sharpe={s:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]")
    print(f"  CI > 0 (real)   : {n_pos_ci} / {n_total}")
    if sig:
        print(f"  Best w/ CI > 0  : trial {sig[0].get('trial', '?')}, Sharpe={sig[0].get('hold_sharpe', float('nan')):+.3f}")
    else:
        print("  Best w/ CI > 0  : NONE")
    print()

    best_params = best_trial_params(study)
    if best_params:
        print("Best config parameters:")
        print(json.dumps(best_params, indent=2, default=str))
        print()

    print("Queue on server:")
    print(curl_cmd)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    optuna.logging.set_verbosity(optuna.logging.DEBUG if args.verbose else optuna.logging.WARNING)

    import datetime

    ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    study_name = args.study_name or f"hypersearch_{ts}"

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = REPORTS_DIR / f"{study_name}_n{args.n_tickers}_{args.start[:4]}.csv"
    out_md = REPORTS_DIR / f"{study_name}_n{args.n_tickers}_{args.start[:4]}.md"

    cfg = HypersearchConfig(
        n_trials=args.n_trials,
        n_tickers=args.n_tickers,
        start_date=args.start,
        end_date=args.end,
        holdout_years=args.holdout_years,
        bootstrap_n=args.bootstrap_n,
        universe_sampling=args.universe_sampling,
        seed=args.seed,
    )

    # Create or load Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    print(f"Hyperparameter search — {args.n_trials} trials")
    print(f"  universe : {args.n_tickers} tickers ({args.universe_sampling}), "
          f"{args.start} → {args.end or 'today'}")
    print(f"  holdout  : {args.holdout_years} years   bootstrap_n={args.bootstrap_n}")
    print(f"  study    : {study_name}")
    if args.storage:
        print(f"  storage  : {args.storage}")
    print(f"  CSV      : {out_csv}")
    print(f"  Report   : {out_md}")
    print()

    on_trial_done = _make_on_trial_done(args.n_trials)

    try:
        run_hypersearch(cfg, on_trial_done=on_trial_done, study=study)
    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results.")

    rows = trials_to_records(study)

    best_params = best_trial_params(study)
    if best_params:
        best_body = _to_refresh_request_body(best_params, cfg)
        curl_cmd = _build_curl(best_body, args.server_url)
    else:
        best_body = None
        curl_cmd = "# no successful trials"

    _print_summary(study, rows, curl_cmd)

    if rows:
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"CSV saved : {out_csv}")

    _write_md_report(study, rows, out_md, cfg, curl_cmd)
    print(f"Report    : {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
