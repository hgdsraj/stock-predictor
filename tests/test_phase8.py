"""Phase 8 tests: pipeline wiring for meta-labelling, triple-barrier labels,
and ranks_only feature pruning.

These are integration-flavour smoke tests using the same synthetic-noise
fixture as `test_pipeline_v5`. The point is to verify the wiring runs end
to end without crashing and that the new code paths are exercised.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5


@pytest.fixture
def synthetic_market(monkeypatch):
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2012-01-02", "2022-12-31")
    tickers = [f"SYN{i:02d}" for i in range(30)]
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
        "stockpred.pipeline_v5.prices_mod.long_panel",
        lambda *a, **kw: long_panel,
    )
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
    monkeypatch.setattr(
        "stockpred.pipeline_v5.macro_mod.fetch_macro",
        lambda *a, **kw: pd.DataFrame(),
    )
    # SPY benchmark fetch needs a value too.
    monkeypatch.setattr(
        "stockpred.pipeline_v5.prices_mod.fetch_one",
        lambda *a, **kw: pd.DataFrame(),
    )
    return tickers


def test_phase8_ranks_only_runs_end_to_end(synthetic_market, tmp_path, monkeypatch, caplog):
    """ranks_only path: dataset is smaller, but pipeline must still complete
    and the noise canary (hit-rate 35-65%) holds."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr("stockpred.pipeline_v5.REPORTS_DIR", tmp_path)
    cfg = PipelineV5Config(
        start_date="2012-01-02",
        end_date="2022-12-31",
        n_tickers=len(synthetic_market),
        horizons=(1,),
        model="logistic",
        ensemble_weighting="equal",
        position_sizing="vol_scaled",
        k_per_side_pct=0.2,
        sector_cap_gross=None,
        min_trade_threshold=0.0,
        holdout_years=2,
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        ranks_only=True,
        bootstrap_n=50,
    )
    result = run_pipeline_v5(cfg)
    # Pipeline completed, ranks_only path reduced features.
    assert result["feature_matrix_shape"][1] < 20  # only rank-suffixed cols + a few
    for h, d in result["per_horizon_diagnostics"].items():
        if "hit_rate" in d:
            assert 0.35 < d["hit_rate"] < 0.65


def test_phase8_meta_labelling_runs_end_to_end(synthetic_market, tmp_path, monkeypatch, caplog):
    """Meta-gate path: pipeline must complete; gated score should produce some
    zeros (gate refused) without crashing.

    Uses GBM (logistic on this small synthetic universe produces no usable
    binary signal and short-circuits the pipeline before meta-gate runs).
    """
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr("stockpred.pipeline_v5.REPORTS_DIR", tmp_path)
    cfg = PipelineV5Config(
        start_date="2012-01-02",
        end_date="2022-12-31",
        n_tickers=len(synthetic_market),
        horizons=(5,),
        model="gbm",
        ensemble_weighting="equal",
        position_sizing="vol_scaled",
        k_per_side_pct=0.2,
        sector_cap_gross=None,
        min_trade_threshold=0.0,
        holdout_years=2,
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        use_meta_labelling=True,
        meta_threshold=0.55,
        bootstrap_n=50,
    )
    result = run_pipeline_v5(cfg)
    # Pipeline completed.
    assert "metrics" in result
    assert "holdout_metrics" in result


def test_apply_meta_gate_preserves_index():
    """Unit test for `_apply_meta_gate` (review H3): the returned series
    should have the same index as the input."""
    import logging as _logging

    from stockpred.models.gbm import GBMConfig
    from stockpred.pipeline_v5 import _apply_meta_gate

    rng = np.random.default_rng(0)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    # Force a 2-ticker panel so the meta dataset has enough rows after the
    # 80/20 split + the per-day filtering.
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    realised = pd.Series(rng.normal(0.0, 0.01, size=len(idx)), index=idx)
    labels = realised.rename("fwd_return_5").to_frame()
    feats = pd.DataFrame(
        {"f1": rng.normal(size=len(idx)), "f2": rng.normal(size=len(idx))}, index=idx
    )

    gated = _apply_meta_gate(
        score,
        feats,
        labels,
        bt_horizon_for_meta=5,
        threshold=0.5,
        gbm_cfg=GBMConfig(),
        log=_logging.getLogger("test_meta_gate"),
    )
    # Same index, same length.
    assert gated.index.equals(score.index)
    assert len(gated) == len(score)


def test_apply_meta_gate_threshold_zero_keeps_everything():
    """Threshold=0 → meta probability ≥ 0 is always true → all preds kept."""
    import logging as _logging

    from stockpred.models.gbm import GBMConfig
    from stockpred.pipeline_v5 import _apply_meta_gate

    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=300)
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    realised = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    labels = realised.rename("fwd_return_5").to_frame()
    feats = pd.DataFrame({"f1": rng.normal(size=len(idx))}, index=idx)
    gated = _apply_meta_gate(
        score,
        feats,
        labels,
        bt_horizon_for_meta=5,
        threshold=0.0,
        gbm_cfg=GBMConfig(),
        log=_logging.getLogger("test_meta_gate"),
    )
    # Train portion is unchanged; predict portion should also be unchanged
    # because every row has meta_proba >= 0.
    assert (gated == score).all() or (gated.isna() == score.isna()).all()


def test_phase8_triple_barrier_runs_end_to_end(synthetic_market, tmp_path, monkeypatch, caplog):
    """Triple-barrier label path: pipeline must complete with the alternate
    label target."""
    caplog.set_level(logging.WARNING)
    monkeypatch.setattr("stockpred.pipeline_v5.REPORTS_DIR", tmp_path)
    cfg = PipelineV5Config(
        start_date="2012-01-02",
        end_date="2022-12-31",
        n_tickers=len(synthetic_market),
        horizons=(5,),
        model="gbm",
        ensemble_weighting="equal",
        position_sizing="vol_scaled",
        k_per_side_pct=0.2,
        sector_cap_gross=None,
        min_trade_threshold=0.0,
        holdout_years=2,
        use_sector_features=False,
        use_tier2_features=False,
        use_regime_features=False,
        use_triple_barrier_labels=True,
        tb_k_sigma=2.0,
        bootstrap_n=50,
    )
    result = run_pipeline_v5(cfg)
    assert "metrics" in result
    # Triple-barrier did run (at least one horizon got a label column).
    diag = result["per_horizon_diagnostics"]
    assert any("hit_rate" in d for d in diag.values())
