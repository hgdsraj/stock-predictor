"""Persist pipeline artifacts to the SQLite store.

Given the dict returned by `pipeline.run_pipeline(...)`, we record:
  - a Run row with config snapshot and summary metrics
  - one Prediction row per (date, ticker) of the ensemble score
  - one EquitySample row per backtest date
  - optionally refresh PriceBar and Fundamental rows
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from stockpred.backend import store
from stockpred.backend.models import Run

log = logging.getLogger(__name__)


def _to_native(obj):
    """Coerce numpy/pandas scalars into JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, (pd.Timestamp, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return {str(i): _to_native(v) for i, v in obj.items()}
    return obj


def snapshot_run(
    s: Session,
    pipeline_result: dict,
    *,
    config: dict | None = None,
    note: str | None = None,
    persist_prices: bool = True,
    persist_fundamentals: bool = True,
) -> Run:
    """Persist a pipeline result. Returns the created Run row."""
    run = store.create_run(s, config=_to_native(config or {}), note=note)

    # --- Predictions: from the ensemble_score + weights -------------------
    score: pd.Series = pipeline_result["ensemble_score"]
    weights: pd.DataFrame = pipeline_result["weights"]
    # Convert weights wide-> long for join.
    w_long = weights.stack(future_stack=True)
    w_long.index = w_long.index.set_names(["date", "ticker"])
    # Per-horizon detail for context.
    per_h: dict[int, pd.Series] = pipeline_result.get("per_horizon_predictions", {})

    rows = []
    # rank per date by score (high -> high rank)
    score_sorted = score.dropna().to_frame("score")
    score_sorted["weight"] = w_long.reindex(score_sorted.index, fill_value=0.0)
    # Compute per-date rank.
    grouped = score_sorted.groupby(level="date")["score"].rank(method="first", ascending=False)
    score_sorted["rank"] = grouped.astype(int)

    for (date, ticker), row in score_sorted.iterrows():
        weight = float(row["weight"]) if not pd.isna(row["weight"]) else 0.0
        side: str | None
        if weight > 0:
            side = "long"
        elif weight < 0:
            side = "short"
        else:
            side = None
        per_horizon_payload = {
            str(h): float(s.loc[(date, ticker)]) if (date, ticker) in s.index else None
            for h, s in per_h.items()
        }
        rows.append(
            {
                "date": pd.Timestamp(date).date(),
                "ticker": str(ticker),
                "score": float(row["score"]),
                "rank": int(row["rank"]),
                "side": side,
                "weight": weight,
                "per_horizon_json": _to_native(per_horizon_payload),
            }
        )
    inserted = store.upsert_predictions(s, run, rows)
    log.info("snapshot: %d predictions for run %d", inserted, run.id)

    # --- Equity curve ----------------------------------------------------
    bt = pipeline_result["backtest"]
    eq_rows = []
    cum = 1.0
    daily_returns = bt.returns.fillna(0.0).clip(-5.0, 5.0)
    cum_curve = (1 + daily_returns).cumprod()
    peak = cum_curve.cummax()
    dd = cum_curve / peak - 1
    for date, r in daily_returns.items():
        d = pd.Timestamp(date).date()
        eq_rows.append(
            {
                "date": d,
                "daily_return": float(r) if not pd.isna(r) else None,
                "cumulative_return": float(cum_curve.loc[date] - 1),
                "drawdown": float(dd.loc[date]),
                "turnover": float(bt.turnover.loc[date]) if date in bt.turnover.index else None,
                "benchmark_return": None,
            }
        )
    store.upsert_equity(s, run, eq_rows)

    # --- Run summary -----------------------------------------------------
    metrics = pipeline_result.get("metrics", {})
    diag = pipeline_result.get("per_horizon_diagnostics", {})
    summary = {
        "metrics": _to_native(metrics),
        "per_horizon_diagnostics": _to_native(diag),
        "tickers_count": len(pipeline_result.get("tickers", [])),
        "feature_matrix_shape": list(pipeline_result.get("feature_matrix_shape", ())),
        "elapsed_s": pipeline_result.get("elapsed_s"),
    }
    store.complete_run(s, run, summary=summary)

    # --- Optional: persist prices and fundamentals -----------------------
    if persist_prices:
        # Bulk dump anything in the data layer's parquet cache that the pipeline
        # touched. We re-read the long_panel for the requested tickers.
        try:
            from stockpred.data import prices as prices_mod

            tickers = pipeline_result.get("tickers", [])
            long_panel = prices_mod.long_panel(tickers)
            if not long_panel.empty:
                price_rows = (
                    long_panel.reset_index()
                    .assign(date=lambda d: pd.to_datetime(d["date"]).dt.date)
                    .to_dict("records")
                )
                store.upsert_prices(s, price_rows)
        except Exception as e:  # noqa: BLE001
            log.warning("price snapshot failed: %s", e)

    if persist_fundamentals:
        try:
            from stockpred.data import fundamentals as fund_mod

            tickers = pipeline_result.get("tickers", [])
            funds = fund_mod.fetch_fundamentals(tickers)
            fund_rows = []
            for ticker, row in funds.iterrows():
                fund_rows.append(
                    {
                        "ticker": ticker,
                        "sector": row.get("sector") if pd.notna(row.get("sector")) else None,
                        "industry": row.get("industry") if pd.notna(row.get("industry")) else None,
                        "market_cap": float(row["marketCap"])
                        if pd.notna(row.get("marketCap"))
                        else None,
                        "beta": float(row["beta"]) if pd.notna(row.get("beta")) else None,
                        "trailing_pe": float(row["trailingPE"])
                        if pd.notna(row.get("trailingPE"))
                        else None,
                        "forward_pe": float(row["forwardPE"])
                        if pd.notna(row.get("forwardPE"))
                        else None,
                        "dividend_yield": float(row["dividendYield"])
                        if pd.notna(row.get("dividendYield"))
                        else None,
                        "short_ratio": float(row["shortRatio"])
                        if pd.notna(row.get("shortRatio"))
                        else None,
                        "short_percent_of_float": float(row["shortPercentOfFloat"])
                        if pd.notna(row.get("shortPercentOfFloat"))
                        else None,
                        "fifty_two_week_high": float(row["fiftyTwoWeekHigh"])
                        if pd.notna(row.get("fiftyTwoWeekHigh"))
                        else None,
                        "fifty_two_week_low": float(row["fiftyTwoWeekLow"])
                        if pd.notna(row.get("fiftyTwoWeekLow"))
                        else None,
                        "long_business_summary": (
                            str(row["longBusinessSummary"])[:8000]
                            if pd.notna(row.get("longBusinessSummary"))
                            else None
                        ),
                        "updated_at": dt.datetime.utcnow(),
                    }
                )
            store.upsert_fundamentals(s, fund_rows)
        except Exception as e:  # noqa: BLE001
            log.warning("fundamentals snapshot failed: %s", e)

    return run
