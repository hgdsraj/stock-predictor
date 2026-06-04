"""End-to-end test for the Phase 5 pipeline against synthetic data.

We patch out price/fundamentals/macro loaders so no network is needed, then
assert structural properties of the output: holdout split applied, every
horizon got a diagnostic dict, bootstrap CI populated, regime breakdown
runs (or fails gracefully), and the integrity-check 35-65% hit-rate canary
holds on noise data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5


@pytest.fixture
def synthetic_market(monkeypatch):
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2012-01-02", "2022-12-31")
    tickers = [f"SYN{i:02d}" for i in range(40)]
    rets = rng.normal(0.0002, 0.012, size=(len(dates), len(tickers)))
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    volumes = rng.integers(1_000_000, 50_000_000, size=prices.shape)

    long_rows = []
    for i, t in enumerate(tickers):
        df = pd.DataFrame(
            {
                "open": prices[:, i],
                "high": prices[:, i] * 1.005,
                "low": prices[:, i] * 0.995,
                "close": prices[:, i],
                "adj_close": prices[:, i],
                "volume": volumes[:, i],
            },
            index=dates,
        )
        df.index.name = "date"
        df["ticker"] = t
        long_rows.append(df)
    long_panel = pd.concat(long_rows).reset_index().set_index(["date", "ticker"]).sort_index()

    fake_membership = pd.DataFrame(
        {
            "ticker": tickers,
            "start_date": [pd.NaT] * len(tickers),
            "end_date": [pd.NaT] * len(tickers),
        }
    )

    # Patch the loaders the v5 pipeline reaches into via stockpred.pipeline.*
    monkeypatch.setattr(
        "stockpred.pipeline.universe_mod.fetch_sp500_membership",
        lambda *a, **kw: fake_membership,
    )
    monkeypatch.setattr(
        "stockpred.pipeline.universe_mod.all_tickers_in_range",
        lambda *a, **kw: tickers,
    )
    # pipeline_v5 imports prices_mod directly under its own namespace, not via
    # pipeline. Patch the actual module.
    monkeypatch.setattr(
        "stockpred.pipeline_v5.prices_mod.long_panel",
        lambda *a, **kw: long_panel,
    )
    # And it uses select_universe from pipeline, which uses prices via pipeline:
    monkeypatch.setattr(
        "stockpred.pipeline.prices_mod.long_panel",
        lambda *a, **kw: long_panel,
    )
    monkeypatch.setattr(
        "stockpred.pipeline_v5.fundamentals_mod.fetch_fundamentals",
        lambda *a, **kw: pd.DataFrame(),
    )
    monkeypatch.setattr(
        "stockpred.pipeline_v5.fundamentals_mod.sector_map",
        lambda *a, **kw: {},
    )
    # Macro fetch (for VIX regime) — return empty so regime breakdown is skipped.
    monkeypatch.setattr(
        "stockpred.pipeline_v5.macro_mod.fetch_macro",
        lambda *a, **kw: pd.DataFrame(),
    )
    return tickers


def test_phase5_pipeline_runs_end_to_end_on_noise(synthetic_market, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr("stockpred.pipeline_v5.REPORTS_DIR", tmp_path)

    cfg = PipelineV5Config(
        start_date="2012-01-02",
        end_date="2022-12-31",
        n_tickers=len(synthetic_market),
        horizons=(1, 5),
        model="logistic",  # faster for noise canary
        ensemble_weighting="equal",  # ic_ir would zero out h=5 by chance and produce empty
        position_sizing="vol_scaled",
        k_per_side_pct=0.2,
        sector_cap_gross=None,  # no sectors in synthetic data
        min_trade_threshold=0.0,
        holdout_years=2,
        use_sector_features=False,
        bootstrap_n=100,
    )
    result = run_pipeline_v5(cfg)

    # Structural assertions.
    assert "metrics" in result
    assert "holdout_metrics" in result
    assert "bootstrap_sharpe" in result
    assert result["per_horizon_diagnostics"]
    for h, d in result["per_horizon_diagnostics"].items():
        # DEV hit-rate canary
        if "hit_rate" in d:
            assert 0.35 < d["hit_rate"] < 0.65, f"h={h} dev hit {d['hit_rate']}"

    # Bootstrap CI populated.
    ci = result["bootstrap_sharpe"]
    assert "sharpe" in ci and "sharpe_lo" in ci and "sharpe_hi" in ci

    # Tearsheet written.
    ts = result["tearsheet_path"]
    assert ts.exists()
