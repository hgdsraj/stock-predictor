"""End-to-end pipeline integration test using *synthetic* prices.

We don't hit yfinance in CI/unit tests. Instead we monkey-patch
`stockpred.data.prices.long_panel` and `stockpred.data.universe.fetch_sp500_membership`
to return fabricated data, then run the full pipeline and assert:

  - Pipeline completes without errors.
  - Walk-forward CV produces predictions over the expected date range.
  - Backtest returns a non-empty series of finite floats.
  - Tearsheet HTML is written and is non-trivial size.
  - Hit rate is in a sane range (35%–65%); anything outside indicates a bug.

The synthetic prices are intentionally constructed to be *unpredictable*
(GBM with no exploitable structure), so the model should land near 50%. If a
future refactor accidentally introduces leakage, hit rate would shoot to 99%
and this test will catch it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stockpred.pipeline import PipelineConfig, run_phase1


@pytest.fixture
def synthetic_panel(monkeypatch):
    """Patch data loaders to return synthetic prices & a fake membership table."""
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2012-01-02", "2020-12-31")
    tickers = [f"SYN{i:02d}" for i in range(30)]
    # Pure GBM, no cross-sectional structure beyond noise.
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

    monkeypatch.setattr(
        "stockpred.pipeline.universe_mod.fetch_sp500_membership",
        lambda *a, **kw: fake_membership,
    )
    monkeypatch.setattr(
        "stockpred.pipeline.universe_mod.all_tickers_in_range",
        lambda *a, **kw: tickers,
    )
    monkeypatch.setattr(
        "stockpred.pipeline.prices_mod.long_panel",
        lambda *a, **kw: long_panel,
    )
    return tickers, long_panel


def test_pipeline_end_to_end_no_leakage(synthetic_panel, tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    # Direct tearsheet output into a temp dir for cleanliness.
    monkeypatch.setattr("stockpred.pipeline.REPORTS_DIR", tmp_path)
    tickers, _ = synthetic_panel
    cfg = PipelineConfig(
        start_date="2012-01-02",
        end_date="2020-12-31",
        n_tickers=len(tickers),
        horizon=1,
        k_per_side=5,
    )
    result = run_phase1(cfg)

    # Structural assertions.
    assert result["feature_matrix_shape"][0] > 5_000  # plenty of (date,ticker) rows
    assert result["feature_matrix_shape"][1] > 10  # plenty of features
    preds = result["predictions"]
    assert isinstance(preds, pd.Series)
    assert preds.notna().sum() > 1_000

    # Backtest.
    res = result["backtest"]
    rets = res.returns.dropna()
    assert len(rets) > 100
    assert np.isfinite(rets).all()

    # Hit rate on pure noise should sit near 50%. We allow a wide band but
    # *exclude* clearly broken values (>65% or <35%) which would indicate a
    # leakage regression.
    hit = result["hit_rate"]
    assert 0.35 < hit < 0.65, f"Suspicious hit rate {hit:.4f} on noise data"

    # IC mean on pure noise should be tiny.
    ic_mean = result["ic_summary"]["ic_mean"]
    assert abs(ic_mean) < 0.05, f"IC {ic_mean:.4f} too large for noise data"

    # Tearsheet must exist and look real.
    ts = result["tearsheet_path"]
    assert ts.exists()
    body = ts.read_text()
    assert "Equity curve" in body
    assert "Drawdown" in body
    assert "Yearly performance" in body
    assert len(body) > 5_000  # non-trivial HTML
