#!/usr/bin/env python
"""Seed a local SQLite DB with synthetic data and (optionally) serve it.

This lets you click through the entire dashboard — Home, Screener, Ticker,
Backtest, Jobs — **without** fetching anything from yfinance or training a
model. Everything is randomly generated but shaped exactly like a real
pipeline run, so every API endpoint and every frontend page has data to show.

It writes to a *separate* database (default ``data/local_test.db``) so your
real ``data/app.db`` is never touched.

Usage
-----
    # Seed the synthetic DB and start the backend on http://127.0.0.1:8000
    uv run python scripts/seed_synthetic.py --serve

    # Just (re)seed, don't serve — e.g. to point `npm run dev` at it yourself
    uv run python scripts/seed_synthetic.py

    # Bigger universe / longer history / reproducible
    uv run python scripts/seed_synthetic.py --n-tickers 120 --days 750 --seed 7 --serve

    # Wipe and rebuild the synthetic DB from scratch
    uv run python scripts/seed_synthetic.py --reset --serve

Then, in another shell, run the frontend against it:

    cd web && npm run dev          # http://127.0.0.1:5173 (proxies to :8000)

See README.md ("Local testing with synthetic data") for the full walkthrough.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Synthetic universe metadata
# ---------------------------------------------------------------------------

_SECTORS = [
    ("Information Technology", ["Software", "Semiconductors", "IT Services"]),
    ("Health Care", ["Pharmaceuticals", "Biotechnology", "Medical Devices"]),
    ("Financials", ["Banks", "Insurance", "Capital Markets"]),
    ("Consumer Discretionary", ["Retail", "Automobiles", "Hotels & Leisure"]),
    ("Industrials", ["Aerospace & Defense", "Machinery", "Airlines"]),
    ("Energy", ["Oil & Gas", "Equipment & Services"]),
    ("Communication Services", ["Telecom", "Media", "Interactive Media"]),
    ("Consumer Staples", ["Food Products", "Beverages", "Household Products"]),
    ("Utilities", ["Electric Utilities", "Gas Utilities"]),
    ("Materials", ["Chemicals", "Metals & Mining"]),
    ("Real Estate", ["REITs", "Real Estate Mgmt"]),
]

_SUMMARY = (
    "{ticker} is a synthetic constituent generated for local testing. It does "
    "not correspond to any real company. All prices, fundamentals, predictions "
    "and backtest figures on this page are randomly generated and carry no "
    "meaning whatsoever — they exist only so the dashboard has something to render."
)


def _make_ticker(i: int) -> str:
    """Deterministic, valid, obviously-fake ticker like ZTS0, ZQX1, ..."""
    letters = "ZQXVWY"
    a = letters[i % len(letters)]
    b = letters[(i // len(letters)) % len(letters)]
    return f"{a}{b}{i % 100:02d}"


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _trading_days(end: dt.date, n: int) -> list[dt.date]:
    """The last ``n`` weekdays ending at (and including) ``end``."""
    out: list[dt.date] = []
    d = end
    while len(out) < n:
        if d.weekday() < 5:  # Mon-Fri
            out.append(d)
        d -= dt.timedelta(days=1)
    return list(reversed(out))


def _price_path(rng: np.random.Generator, n: int) -> np.ndarray:
    """Geometric random walk with a mild drift and per-ticker volatility."""
    start = float(rng.uniform(20, 400))
    daily_vol = float(rng.uniform(0.010, 0.030))
    drift = float(rng.uniform(-0.0003, 0.0006))
    shocks = rng.normal(drift, daily_vol, size=n)
    path = start * np.exp(np.cumsum(shocks))
    return path


def _fundamental_row(rng: np.random.Generator, ticker: str, last_close: float) -> dict:
    sector, industries = _SECTORS[int(rng.integers(0, len(_SECTORS)))]
    industry = industries[int(rng.integers(0, len(industries)))]
    hi = last_close * float(rng.uniform(1.05, 1.6))
    lo = last_close * float(rng.uniform(0.5, 0.95))
    return {
        "ticker": ticker,
        "sector": sector,
        "industry": industry,
        "market_cap": float(rng.uniform(2e9, 2.5e12)),
        "beta": round(float(rng.uniform(0.4, 1.9)), 2),
        "trailing_pe": round(float(rng.uniform(8, 45)), 1),
        "forward_pe": round(float(rng.uniform(7, 38)), 1),
        "dividend_yield": round(float(rng.uniform(0.0, 0.045)), 4),
        "short_ratio": round(float(rng.uniform(0.5, 8.0)), 2),
        "short_percent_of_float": round(float(rng.uniform(0.005, 0.12)), 4),
        "fifty_two_week_high": round(hi, 2),
        "fifty_two_week_low": round(lo, 2),
        "long_business_summary": _SUMMARY.format(ticker=ticker),
        "updated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None),
    }


def seed(
    db_path: Path,
    *,
    n_tickers: int,
    days: int,
    seed_val: int,
) -> dict:
    """Build the synthetic DB. Returns a small summary dict for logging."""
    # Import after argparse so a bad --help is fast and import errors are clear.
    from stockpred.backend import store
    from stockpred.backend.db import (
        create_all,
        make_engine,
        make_session_factory,
        session_scope,
    )

    rng = np.random.default_rng(seed_val)
    tickers = [_make_ticker(i) for i in range(n_tickers)]
    today = dt.date.today()
    dates = _trading_days(today, days)

    engine = make_engine(db_path)
    create_all(engine)
    SessionLocal = make_session_factory(engine)

    # ----- Prices + fundamentals ------------------------------------------
    closes: dict[str, np.ndarray] = {}
    price_rows: list[dict] = []
    fund_rows: list[dict] = []
    for t in tickers:
        path = _price_path(rng, len(dates))
        closes[t] = path
        for d, c in zip(dates, path):
            o = c * float(rng.uniform(0.99, 1.01))
            hi = max(o, c) * float(rng.uniform(1.0, 1.02))
            lo = min(o, c) * float(rng.uniform(0.98, 1.0))
            price_rows.append(
                {
                    "ticker": t,
                    "date": d,
                    "open": round(o, 2),
                    "high": round(hi, 2),
                    "low": round(lo, 2),
                    "close": round(c, 2),
                    "adj_close": round(c, 2),
                    "volume": float(int(rng.uniform(2e5, 3e7))),
                }
            )
        fund_rows.append(_fundamental_row(rng, t, float(path[-1])))

    # ----- A finished Run -------------------------------------------------
    horizons = [1, 5]
    per_horizon_diag = {
        str(h): {
            "hit_rate": round(float(rng.uniform(0.49, 0.55)), 4),
            "ic_mean": round(float(rng.uniform(-0.01, 0.03)), 4),
            "ic_std": round(float(rng.uniform(0.08, 0.14)), 4),
            "ic_ir": round(float(rng.uniform(-0.5, 2.5)), 3),
        }
        for h in horizons
    }

    # Equity curve: a slightly-negative-drift L/S strategy (honest-looking).
    eq_returns = rng.normal(-0.0002, 0.008, size=len(dates))
    cum = np.cumprod(1.0 + eq_returns)
    peak = np.maximum.accumulate(cum)
    dd = cum / peak - 1.0
    turnover = rng.uniform(0.05, 0.25, size=len(dates))

    ann = 252
    ann_return = float(cum[-1] ** (ann / len(dates)) - 1.0)
    ann_vol = float(np.std(eq_returns) * np.sqrt(ann))
    sharpe = float(np.mean(eq_returns) / (np.std(eq_returns) + 1e-12) * np.sqrt(ann))
    max_dd = float(dd.min())

    summary = {
        "metrics": {
            "ann_return": round(ann_return, 4),
            "ann_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(max_dd, 4),
        },
        "per_horizon_diagnostics": per_horizon_diag,
        "tickers_count": len(tickers),
        "feature_matrix_shape": [len(dates) * len(tickers), 18],
        "elapsed_s": round(float(rng.uniform(20, 120)), 1),
    }

    # ----- Predictions: last ~60 trading days, top/bottom-k get a side -----
    pred_dates = dates[-min(60, len(dates)) :]
    k_per_side = max(1, len(tickers) // 6)
    pred_rows: list[dict] = []
    for d in pred_dates:
        # A score per ticker; correlate loosely with that day's return so the
        # ticker page's score overlay looks plausible against the price line.
        scores = {t: float(rng.normal(0, 1)) for t in tickers}
        ordered = sorted(tickers, key=lambda t: scores[t], reverse=True)
        longs = set(ordered[:k_per_side])
        shorts = set(ordered[-k_per_side:])
        for rank, t in enumerate(ordered, start=1):
            if t in longs:
                side, weight = "long", round(0.5 / k_per_side, 5)
            elif t in shorts:
                side, weight = "short", round(-0.5 / k_per_side, 5)
            else:
                side, weight = None, 0.0
            pred_rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "score": round(scores[t], 5),
                    "rank": rank,
                    "side": side,
                    "weight": weight,
                    "per_horizon_json": {
                        str(h): round(float(rng.normal(0, 1)), 4) for h in horizons
                    },
                }
            )

    eq_rows = [
        {
            "date": d,
            "daily_return": round(float(r), 6),
            "cumulative_return": round(float(cum[i] - 1.0), 6),
            "drawdown": round(float(dd[i]), 6),
            "turnover": round(float(turnover[i]), 4),
            "benchmark_return": round(float(rng.normal(0.0003, 0.009)), 6),
        }
        for i, (d, r) in enumerate(zip(dates, eq_returns))
    ]

    # ----- Watchlist price history (so the Screener watchlist isn't blank) --
    watch_defaults = [
        ("SPY", "SPDR S&P 500 ETF", "index_etf"),
        ("HND.TO", "Synthetic Bear 2x", "leveraged_etf"),
        ("HNU.TO", "Synthetic Bull 2x", "leveraged_etf"),
        ("UNG", "Synthetic NatGas Fund", "commodity_etf"),
        ("^VIX", "Synthetic Volatility Index", "regime"),
    ]
    watch_price_rows: list[dict] = []
    for tkr, _label, _cat in watch_defaults:
        path = _price_path(rng, len(dates))
        for d, c in zip(dates, path):
            watch_price_rows.append(
                {
                    "ticker": tkr,
                    "date": d,
                    "open": round(c, 2),
                    "high": round(c * 1.01, 2),
                    "low": round(c * 0.99, 2),
                    "close": round(c, 2),
                    "adj_close": round(c, 2),
                    "volume": float(int(rng.uniform(2e5, 3e7))),
                }
            )

    # ----- Write it all ----------------------------------------------------
    with session_scope(SessionLocal) as s:
        store.upsert_prices(s, price_rows)
        store.upsert_prices(s, watch_price_rows)
        store.upsert_fundamentals(s, fund_rows)
        store.seed_default_watchlist(s)
        run = store.create_run(s, config={"synthetic": True, "horizons": horizons}, note="synthetic local-test run")
        store.upsert_predictions(s, run, pred_rows)
        store.upsert_equity(s, run, eq_rows)
        store.complete_run(s, run, summary=summary)
        run_id = run.id

    return {
        "db_path": str(db_path),
        "run_id": run_id,
        "tickers": len(tickers),
        "trading_days": len(dates),
        "predictions": len(pred_rows),
        "equity_points": len(eq_rows),
        "date_range": f"{dates[0]} → {dates[-1]}",
        "sharpe": summary["metrics"]["sharpe"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--db",
        default="data/local_test.db",
        help="Path to the synthetic SQLite DB (default: data/local_test.db)",
    )
    p.add_argument("--n-tickers", type=int, default=80, help="Universe size (default: 80)")
    p.add_argument("--days", type=int, default=500, help="Trading days of history (default: 500)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility (default: 42)")
    p.add_argument(
        "--reset", action="store_true", help="Delete the synthetic DB (and -wal/-shm) before seeding"
    )
    p.add_argument("--serve", action="store_true", help="Start the FastAPI server after seeding")
    p.add_argument("--host", default="127.0.0.1", help="Server host (with --serve)")
    p.add_argument("--port", type=int, default=8000, help="Server port (with --serve)")
    args = p.parse_args()

    db_path = Path(args.db).resolve()

    if args.reset:
        for suffix in ("", "-wal", "-shm"):
            f = Path(str(db_path) + suffix)
            if f.exists():
                f.unlink()
                print(f"removed {f}")

    print(f"Seeding synthetic data into {db_path} ...")
    info = seed(db_path, n_tickers=args.n_tickers, days=args.days, seed_val=args.seed)
    print("Done:")
    for k, v in info.items():
        print(f"  {k:14s} {v}")

    if not args.serve:
        print(
            "\nNot serving (pass --serve to launch the backend). To serve manually:\n"
            f"  STOCKPRED_DB={db_path} STOCKPRED_DISABLE_SCHEDULER=1 "
            f"uv run python scripts/serve.py --port {args.port}"
        )
        return 0

    # Point the server at the synthetic DB and disable the scheduler (no real
    # pipeline should run against fake data). These must be set before the
    # backend imports read them.
    os.environ["STOCKPRED_DB"] = str(db_path)
    os.environ.setdefault("STOCKPRED_DISABLE_SCHEDULER", "1")
    os.environ.setdefault("STOCKPRED_CORS", "http://localhost:5173,http://127.0.0.1:8000")

    import uvicorn

    print(
        f"\nServing synthetic DB at http://{args.host}:{args.port}\n"
        f"  • Open that URL for the built dashboard (if web/dist exists), or\n"
        f"  • run `cd web && npm run dev` and open http://127.0.0.1:5173\n"
        f"Press Ctrl-C to stop.\n"
    )
    uvicorn.run(
        "stockpred.backend.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
