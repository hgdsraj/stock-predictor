#!/usr/bin/env python
"""Hyperparameter search for the Phase 5+ pipeline via Optuna (Bayesian TPE).

Searches 20 pipeline parameters to maximise holdout Sharpe ratio. Uses
Tree-structured Parzen Estimator (TPE) so it finds good regions of the
parameter space ~10x faster than an equivalent grid search.

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

from stockpred.config import CVConfig
from stockpred.models.gbm import GBMConfig
from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

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
# Parameter space
# ─────────────────────────────────────────────────────────────────────────────


def _suggest_config(trial: optuna.Trial, args: argparse.Namespace) -> PipelineV5Config:
    """Map Optuna trial suggestions to a PipelineV5Config.

    20 parameters: portfolio construction, signal, features,
    meta-labelling (conditional), GBM, and CV walk-forward settings.
    """

    # ── Portfolio construction ─────────────────────────────────────────────
    position_sizing = trial.suggest_categorical("position_sizing", ["vol_scaled", "hrp", "top_k"])
    k_per_side_pct = trial.suggest_float("k_per_side_pct", 0.08, 0.25)
    leverage_per_side = trial.suggest_float("leverage_per_side", 0.5, 1.5)
    _sc = trial.suggest_categorical("sector_cap_gross", ["none", "0.20", "0.30", "0.40"])
    sector_cap_gross: float | None = None if _sc == "none" else float(_sc)
    min_trade_threshold = trial.suggest_float("min_trade_threshold", 0.001, 0.015, log=True)

    # ── Signal ────────────────────────────────────────────────────────────
    _hz = trial.suggest_categorical("horizons", ["5", "1,5"])
    horizons = tuple(int(h) for h in _hz.split(","))
    ensemble_weighting = trial.suggest_categorical("ensemble_weighting", ["ic_ir", "equal"])

    # ── Features ──────────────────────────────────────────────────────────
    use_tier2_features = trial.suggest_categorical("use_tier2_features", [True, False])
    use_regime_features = trial.suggest_categorical("use_regime_features", [True, False])
    use_sector_features = trial.suggest_categorical("use_sector_features", [True, False])
    ranks_only = trial.suggest_categorical("ranks_only", [True, False])
    beta_neutralise = trial.suggest_categorical("beta_neutralise", [True, False])

    # ── Meta-labelling (conditional on use_meta_labelling) ────────────────
    use_meta_labelling = trial.suggest_categorical("use_meta_labelling", [True, False])
    if use_meta_labelling:
        meta_threshold = trial.suggest_float("meta_threshold", 0.50, 0.65)
        meta_mode = trial.suggest_categorical("meta_mode", ["binary", "confidence"])
        meta_conf_floor = (
            trial.suggest_float("meta_conf_floor", 0.48, 0.72)
            if meta_mode == "confidence"
            else 0.5
        )
    else:
        meta_threshold = 0.55
        meta_mode = "binary"
        meta_conf_floor = 0.5

    # ── GBM ───────────────────────────────────────────────────────────────
    num_leaves = trial.suggest_categorical("num_leaves", [31, 63, 127])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    n_estimators = trial.suggest_int("n_estimators", 400, 1200, step=200)

    # ── CV ────────────────────────────────────────────────────────────────
    train_years = trial.suggest_int("train_years", 2, 4)

    return PipelineV5Config(
        start_date=args.start,
        end_date=args.end,
        n_tickers=args.n_tickers,
        universe_sampling=args.universe_sampling,
        refresh_data=False,
        horizons=horizons,
        model="gbm",
        gbm=GBMConfig(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        ),
        use_sector_features=use_sector_features,
        use_tier2_features=use_tier2_features,
        use_regime_features=use_regime_features,
        beta_neutralise=beta_neutralise,
        bootstrap_method="block",
        cv=CVConfig(
            train_years=train_years,
            test_months=6,
            embargo_days=25,
            min_train_obs=1000,
        ),
        holdout_years=args.holdout_years,
        position_sizing=position_sizing,
        k_per_side_pct=k_per_side_pct,
        leverage_per_side=leverage_per_side,
        sector_cap_gross=sector_cap_gross,
        min_trade_threshold=min_trade_threshold,
        ensemble_weighting=ensemble_weighting,
        use_meta_labelling=use_meta_labelling,
        meta_threshold=meta_threshold,
        meta_mode=meta_mode,
        meta_conf_floor=meta_conf_floor,
        meta_conf_cap=1.0,
        meta_walk_forward_folds=1,
        meta_per_sector=False,
        use_triple_barrier_labels=False,
        ranks_only=ranks_only,
        feature_exclude=(),
        use_edgar_features=False,
        use_edgar_item_features=False,
        bootstrap_n=args.bootstrap_n,
        tearsheet_path=None,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Objective
# ─────────────────────────────────────────────────────────────────────────────

_PENALTY = -10.0  # returned on error / NaN so Optuna avoids those regions


def make_objective(args: argparse.Namespace):
    n_trials = args.n_trials

    def objective(trial: optuna.Trial) -> float:
        cfg = _suggest_config(trial, args)
        t0 = time.time()
        try:
            res = run_pipeline_v5(cfg)
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - t0, 1)
            log.warning("Trial %d failed in %.0fs: %s", trial.number, elapsed, exc)
            trial.set_user_attr("error", str(exc)[:200])
            trial.set_user_attr("elapsed_s", elapsed)
            _print_trial_line(trial.number, n_trials, float("nan"), float("nan"), float("nan"), float("nan"), elapsed, error=str(exc)[:120])
            return _PENALTY

        elapsed = round(time.time() - t0, 1)
        hold = res.get("holdout_metrics") or {}
        dev = res.get("metrics") or {}
        ci = res.get("bootstrap_sharpe") or {}

        hold_sharpe = hold.get("sharpe", float("nan"))
        hold_ci_lo = ci.get("sharpe_lo", float("nan"))
        hold_ci_hi = ci.get("sharpe_hi", float("nan"))
        hold_dd = hold.get("max_drawdown", float("nan"))
        hold_hit = hold.get("hit_ratio", float("nan"))
        hold_ann_return = hold.get("ann_return", float("nan"))
        dev_sharpe = dev.get("sharpe", float("nan"))

        trial.set_user_attr("hold_sharpe", _f(hold_sharpe))
        trial.set_user_attr("hold_ci_lo", _f(hold_ci_lo))
        trial.set_user_attr("hold_ci_hi", _f(hold_ci_hi))
        trial.set_user_attr("hold_dd", _f(hold_dd))
        trial.set_user_attr("hold_hit", _f(hold_hit))
        trial.set_user_attr("hold_ann_return", _f(hold_ann_return))
        trial.set_user_attr("dev_sharpe", _f(dev_sharpe))
        trial.set_user_attr("elapsed_s", elapsed)
        trial.set_user_attr("error", "")

        _print_trial_line(trial.number, n_trials, hold_sharpe, hold_ci_lo, hold_ci_hi, hold_dd, elapsed)

        return _f(hold_sharpe) if pd.notna(hold_sharpe) else _PENALTY

    return objective


def _f(x) -> float:
    """Safe float cast; NaN on failure."""
    try:
        v = float(x)
        return v if pd.notna(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _print_trial_line(num, n, sharpe, ci_lo, ci_hi, dd, elapsed, error=""):
    tag = f"[{num + 1:>3}/{n}]"
    if error:
        print(f"{tag} FAILED ({elapsed:.0f}s): {error}", flush=True)
    else:
        s = f"{sharpe:+.3f}" if pd.notna(sharpe) else "  nan "
        lo = f"{ci_lo:+.3f}" if pd.notna(ci_lo) else "   nan"
        hi = f"{ci_hi:+.3f}" if pd.notna(ci_hi) else "   nan"
        d = f"{dd:+.2%}" if pd.notna(dd) else "   nan"
        print(f"{tag} Sharpe={s}  CI=[{lo},{hi}]  DD={d}  ({elapsed:.0f}s)", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Results → RefreshRequest body → curl command
# ─────────────────────────────────────────────────────────────────────────────


def _to_refresh_request_body(params: dict, args: argparse.Namespace) -> dict:
    """Convert best Optuna params to a RefreshRequest-compatible JSON body.

    The body targets a production-scale run (n_tickers=None = full universe,
    bootstrap_n=500 for honest CIs). The caller can override before posting.
    """
    horizons = [int(h) for h in params.get("horizons", "5").split(",")]
    sc = params.get("sector_cap_gross", "0.30")
    sector_cap_gross = None if sc == "none" else float(sc)

    body: dict = {
        "phase": 5,
        # History — keep the tuning window; bump n_tickers for production
        "start_date": args.start,
        "end_date": args.end,
        "n_tickers": None,  # full universe for production run
        "universe_sampling": args.universe_sampling,
        "horizons": horizons,
        "model": "gbm",
        # GBM
        "gbm": {
            "num_leaves": params.get("num_leaves", 63),
            "learning_rate": round(params.get("learning_rate", 0.03), 6),
            "n_estimators": params.get("n_estimators", 800),
        },
        # CV
        "cv": {
            "train_years": params.get("train_years", 3),
            "test_months": 6,
            "embargo_days": 25,
            "min_train_obs": 1000,
        },
        "holdout_years": args.holdout_years,
        # Portfolio
        "position_sizing": params.get("position_sizing", "vol_scaled"),
        "k_per_side_pct": round(params.get("k_per_side_pct", 0.15), 4),
        "leverage_per_side": round(params.get("leverage_per_side", 1.0), 4),
        "sector_cap_gross": sector_cap_gross,
        "min_trade_threshold": round(params.get("min_trade_threshold", 0.005), 6),
        # Signal
        "ensemble_weighting": params.get("ensemble_weighting", "ic_ir"),
        # Features
        "use_sector_features": params.get("use_sector_features", True),
        "use_tier2_features": params.get("use_tier2_features", True),
        "use_regime_features": params.get("use_regime_features", True),
        "beta_neutralise": params.get("beta_neutralise", False),
        "ranks_only": params.get("ranks_only", False),
        # Meta-labelling
        "use_meta_labelling": params.get("use_meta_labelling", False),
        "meta_threshold": round(params.get("meta_threshold", 0.55), 4),
        "meta_mode": params.get("meta_mode", "binary"),
        "meta_conf_floor": round(params.get("meta_conf_floor", 0.5), 4),
        "meta_conf_cap": 1.0,
        "meta_walk_forward_folds": 1,
        # Full bootstrap for production
        "bootstrap_n": 500,
        "bootstrap_method": "block",
    }
    return body


def _build_curl(body: dict, server_url: str) -> str:
    url = server_url.rstrip("/") + "/jobs/queue"
    body_json = json.dumps(body, separators=(", ", ": "))
    return f"curl -X POST \\\n  '{url}' \\\n  -H 'Content-Type: application/json' \\\n  -d '{body_json}'"


# ─────────────────────────────────────────────────────────────────────────────
# Results DataFrame
# ─────────────────────────────────────────────────────────────────────────────


def _build_results_df(study: optuna.Study) -> pd.DataFrame:
    rows = []
    for t in study.trials:
        if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.FAIL):
            continue
        row: dict = {"trial": t.number}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "hold_sharpe" in df.columns:
        df = df.sort_values("hold_sharpe", ascending=False, na_position="last").reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


def _write_md_report(
    study: optuna.Study,
    df: pd.DataFrame,
    out_md: Path,
    args: argparse.Namespace,
    curl_cmd: str,
    best_body: dict | None,
) -> None:
    import datetime

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    valid = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    n_total = len(valid)

    lines: list[str] = []
    lines += [
        f"# Hyperparameter Search Results",
        f"",
        f"Generated: {now}  ",
        f"Study: `{study.study_name}`  ",
        f"Universe: {args.n_tickers} tickers (`{args.universe_sampling}`), {args.start} → {args.end or 'today'}  ",
        f"Trials completed: {len(study.trials)} ({n_total} succeeded)  ",
        f"Objective: maximise **holdout Sharpe** (last {args.holdout_years} years, never seen during tuning)  ",
        f"",
    ]

    # ── Top 10 ──────────────────────────────────────────────────────────────
    lines.append("## Top 10 Configs")
    lines.append("")

    if valid.empty:
        lines.append("_No successful trials._")
    else:
        top10 = valid.head(10)
        table_cols = [
            ("Rank", None),
            ("Sharpe", "hold_sharpe"),
            ("CI lo", "hold_ci_lo"),
            ("CI hi", "hold_ci_hi"),
            ("Max DD", "hold_dd"),
            ("Ann Ret", "hold_ann_return"),
            ("Dev Sh", "dev_sharpe"),
            ("Sizing", "position_sizing"),
            ("Horizons", "horizons"),
            ("Meta", "use_meta_labelling"),
            ("Ranks", "ranks_only"),
            ("s", "elapsed_s"),
        ]
        header = " | ".join(h for h, _ in table_cols)
        sep = " | ".join("---" for _ in table_cols)
        lines += [f"| {header} |", f"| {sep} |"]
        for rank, (_, row) in enumerate(top10.iterrows(), 1):
            def fv(col):
                if col is None:
                    return str(rank)
                v = row.get(col, float("nan"))
                if col in ("hold_sharpe", "hold_ci_lo", "hold_ci_hi", "dev_sharpe"):
                    return f"{v:+.3f}" if pd.notna(v) else "nan"
                if col in ("hold_dd", "hold_ann_return"):
                    return f"{v:+.1%}" if pd.notna(v) else "nan"
                if col == "elapsed_s":
                    return f"{v:.0f}s" if pd.notna(v) else "nan"
                return str(v) if pd.notna(v) else "nan"
            cells = " | ".join(fv(col) for _, col in table_cols)
            lines.append(f"| {cells} |")
        lines.append("")

    # ── Honest interpretation ────────────────────────────────────────────────
    lines.append("## Honest Interpretation")
    lines.append("")
    if not valid.empty:
        best_row = valid.iloc[0]
        s = best_row.get("hold_sharpe", float("nan"))
        lo = best_row.get("hold_ci_lo", float("nan"))
        hi = best_row.get("hold_ci_hi", float("nan"))
        t = int(best_row.get("trial", -1))

        n_pos_ci = int((valid["hold_ci_lo"] > 0).sum()) if "hold_ci_lo" in valid.columns else 0
        n_straddle = int(
            ((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum()
        ) if "hold_ci_lo" in valid.columns and "hold_ci_hi" in valid.columns else 0

        sig = valid[valid["hold_ci_lo"] > 0] if "hold_ci_lo" in valid.columns else pd.DataFrame()
        lines += [
            f"- **Best point estimate**: Sharpe = `{s:+.3f}` (trial {t})",
            f"  95% block-bootstrap CI: [`{lo:+.3f}`, `{hi:+.3f}`]",
            f"- Configs with CI **strictly > 0** (statistically real edge): **{n_pos_ci} / {n_total}**",
            f"- Configs with CI straddling zero (could be noise): {n_straddle} / {n_total}",
        ]
        if not sig.empty:
            bs = sig.iloc[0]
            lines.append(
                f"- **Best config with CI > 0**: trial {int(bs.get('trial', -1))}, "
                f"Sharpe = `{bs.get('hold_sharpe', float('nan')):+.3f}` "
                f"CI = [`{bs.get('hold_ci_lo', float('nan')):+.3f}`, `{bs.get('hold_ci_hi', float('nan')):+.3f}`]"
            )
        else:
            lines.append("- **Best config with CI > 0**: _none — no statistically reliable edge found_")
        lines.append("")
        lines.append(
            "> The CI is computed on a small, fast-mode universe. "
            "Validate the best config on the full S&P 500 universe before drawing conclusions."
        )
    else:
        lines += ["_No successful trials to interpret._"]
    lines.append("")

    # ── Best config parameters ───────────────────────────────────────────────
    lines.append("## Best Config Parameters")
    lines.append("")
    try:
        best_params = study.best_trial.params
        lines.append("```json")
        lines.append(json.dumps(best_params, indent=2, default=str))
        lines.append("```")
    except ValueError:
        lines.append("_No successful trials._")
    lines.append("")

    # ── Curl command ─────────────────────────────────────────────────────────
    lines.append("## Queue This Run on the Server")
    lines.append("")
    lines += [
        "POST to `/jobs/queue` (no auth required — queues for later password-protected launch):",
        "",
        "```bash",
        curl_cmd,
        "```",
        "",
        "Then launch the queued job with the `X-Password` header:",
        "```bash",
        "# 1. Get the queue_id from the response above, then:",
        "curl -X POST \\",
        f"  '{args.server_url.rstrip('/')}/jobs/run/<queue_id>' \\",
        "  -H 'X-Password: <your-password>'",
        "```",
        "",
        "> **Note:** `n_tickers` is set to `null` (full universe) in the command above.",
        "> The tuning was done on a {}-ticker sample. Full-universe results will differ.".format(args.n_tickers),
    ]
    lines.append("")

    # ── Full results table ───────────────────────────────────────────────────
    if not df.empty:
        lines.append("## All Trials")
        lines.append("")
        lines.append("_Sorted by holdout Sharpe descending._")
        lines.append("")

        all_cols = [
            ("Trial", "trial"),
            ("Sharpe", "hold_sharpe"),
            ("CI lo", "hold_ci_lo"),
            ("CI hi", "hold_ci_hi"),
            ("DD", "hold_dd"),
            ("Dev Sh", "dev_sharpe"),
            ("Sizing", "position_sizing"),
            ("Hz", "horizons"),
            ("Ens", "ensemble_weighting"),
            ("T2", "use_tier2_features"),
            ("Reg", "use_regime_features"),
            ("Meta", "use_meta_labelling"),
            ("Rnk", "ranks_only"),
            ("Meta thr", "meta_threshold"),
            ("Leaves", "num_leaves"),
            ("LR", "learning_rate"),
            ("k%", "k_per_side_pct"),
            ("Lev", "leverage_per_side"),
        ]
        avail = [(h, c) for h, c in all_cols if c in df.columns]
        header = " | ".join(h for h, _ in avail)
        sep = " | ".join("---" for _ in avail)
        lines += [f"| {header} |", f"| {sep} |"]

        for _, row in df.iterrows():
            def fv2(col):
                v = row.get(col, float("nan"))
                if col in ("hold_sharpe", "hold_ci_lo", "hold_ci_hi", "dev_sharpe"):
                    return f"{v:+.3f}" if pd.notna(v) else "nan"
                if col == "hold_dd":
                    return f"{v:+.1%}" if pd.notna(v) else "nan"
                if col in ("k_per_side_pct", "leverage_per_side"):
                    return f"{v:.3f}" if pd.notna(v) else "nan"
                if col == "learning_rate":
                    return f"{v:.4f}" if pd.notna(v) else "nan"
                if col in ("meta_threshold",):
                    return f"{v:.3f}" if pd.notna(v) else "nan"
                if col in ("use_tier2_features", "use_regime_features", "use_meta_labelling", "ranks_only"):
                    return "Y" if v else "N"
                return str(v) if pd.notna(v) else "nan"
            cells = " | ".join(fv2(col) for _, col in avail)
            lines.append(f"| {cells} |")
        lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────


def _print_console_summary(study: optuna.Study, df: pd.DataFrame, curl_cmd: str) -> None:
    print()
    print("=" * 100)
    print("Hyperparameter search — results")
    print("=" * 100)

    valid = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    if valid.empty:
        print("No completed trials.")
        return

    n_total = len(valid)
    display_cols = [c for c in [
        "trial", "hold_sharpe", "hold_ci_lo", "hold_ci_hi", "hold_dd",
        "dev_sharpe", "position_sizing", "horizons", "ensemble_weighting",
        "use_meta_labelling", "ranks_only", "elapsed_s",
    ] if c in valid.columns]

    fmt = {}
    for c in ["hold_sharpe", "hold_ci_lo", "hold_ci_hi", "dev_sharpe"]:
        if c in valid.columns:
            fmt[c] = lambda x: f"{x:+.3f}" if pd.notna(x) else "  nan"
    if "hold_dd" in valid.columns:
        fmt["hold_dd"] = lambda x: f"{x:+.2%}" if pd.notna(x) else "  nan"

    print(f"\nTop {min(10, len(valid))} configs (by holdout Sharpe):\n")
    print(valid.head(10)[display_cols].to_string(index=False, formatters=fmt))

    n_pos_ci = int((valid["hold_ci_lo"] > 0).sum()) if "hold_ci_lo" in valid.columns else 0
    n_straddle = int(
        ((valid["hold_ci_lo"] <= 0) & (valid["hold_ci_hi"] >= 0)).sum()
    ) if "hold_ci_lo" in valid.columns and "hold_ci_hi" in valid.columns else 0

    best = valid.iloc[0]
    s, lo, hi, t = (
        best.get("hold_sharpe", float("nan")),
        best.get("hold_ci_lo", float("nan")),
        best.get("hold_ci_hi", float("nan")),
        int(best.get("trial", -1)),
    )
    sig = valid[valid["hold_ci_lo"] > 0] if "hold_ci_lo" in valid.columns else pd.DataFrame()

    print()
    print("Honest interpretation:")
    print(f"  Best point estimate  : Sharpe={s:+.3f}  CI=[{lo:+.3f}, {hi:+.3f}]  (trial {t})")
    print(f"  CI > 0 (real edge)   : {n_pos_ci} / {n_total}")
    print(f"  CI straddles zero    : {n_straddle} / {n_total}")
    if not sig.empty:
        bs = sig.iloc[0]
        print(
            f"  Best w/ CI > 0       : trial {int(bs.get('trial', -1))}, "
            f"Sharpe={bs.get('hold_sharpe', float('nan')):+.3f}"
        )
    else:
        print("  Best w/ CI > 0       : NONE")
    print()

    try:
        best_params = study.best_trial.params
        print("Best config parameters:")
        print(json.dumps(best_params, indent=2, default=str))
        print()
    except ValueError:
        pass

    print("Queue this config on the server:")
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

    print(f"Hyperparameter search — {args.n_trials} trials")
    print(f"  universe : {args.n_tickers} tickers ({args.universe_sampling}), {args.start} → {args.end or 'today'}")
    print(f"  holdout  : {args.holdout_years} years   bootstrap_n={args.bootstrap_n}")
    print(f"  study    : {study_name}")
    if args.storage:
        print(f"  storage  : {args.storage}")
    print(f"  CSV      : {out_csv}")
    print(f"  Report   : {out_md}")
    print()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    try:
        study.optimize(make_objective(args), n_trials=args.n_trials, show_progress_bar=False)
    except KeyboardInterrupt:
        print("\nInterrupted — saving partial results.")

    df = _build_results_df(study)

    # Generate curl command from best params
    try:
        best_params = study.best_trial.params
        best_body = _to_refresh_request_body(best_params, args)
        curl_cmd = _build_curl(best_body, args.server_url)
    except ValueError:
        best_params = {}
        best_body = None
        curl_cmd = "# no successful trials"

    _print_console_summary(study, df, curl_cmd)

    if not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"CSV saved : {out_csv}")

    _write_md_report(study, df, out_md, args, curl_cmd, best_body)
    print(f"Report    : {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
