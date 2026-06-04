"""Phase 9 tests: confidence-weighted sizing, walk-forward meta-CV,
sector-conditional meta classifiers."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from stockpred.models.meta import meta_confidence_weight_signal
from stockpred.pipeline_v5 import (
    PipelineV5Config,
    _apply_meta_gate,
    _apply_meta_gate_per_sector,
    run_pipeline_v5,
)


# -------- Phase 9a: confidence-weighted sizing -----------------------------


def test_confidence_weight_signal_basic():
    """P(correct)=0.5 -> 0, P=1 -> 1, P=0.75 -> 0.5 with default floor=0.5/cap=1."""
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=2), ["A", "B"]], names=["date", "ticker"]
    )
    primary = pd.Series([1.0, -1.0, 0.5, -0.5], index=idx)
    proba = pd.Series([0.50, 0.75, 1.00, 0.30], index=idx)
    out = meta_confidence_weight_signal(primary, proba)
    # weight = clip((p - 0.5) / 0.5, 0, 1)
    # row 0: p=0.5 -> w=0 -> out=0
    # row 1: p=0.75 -> w=0.5 -> out=-0.5
    # row 2: p=1.00 -> w=1.0 -> out=0.5
    # row 3: p=0.30 -> w=0 -> out=0
    np.testing.assert_allclose(out.values, [0.0, -0.5, 0.5, 0.0])


def test_confidence_weight_preserves_index_and_dtype():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=3), ["A"]], names=["date", "ticker"]
    )
    primary = pd.Series([0.5, -0.3, 0.7], index=idx)
    proba = pd.Series([0.6, 0.8, 0.4], index=idx)
    out = meta_confidence_weight_signal(primary, proba)
    assert out.index.equals(primary.index)
    assert out.dtype == primary.dtype


def test_confidence_weight_treats_nan_proba_as_zero():
    """Review H6 fix: NaN in meta_proba must not propagate to weight; the
    row should be treated as 'don't trade' (weight 0)."""
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=1), ["A", "B", "C"]],
        names=["date", "ticker"],
    )
    primary = pd.Series([1.0, -1.0, 0.5], index=idx)
    proba = pd.Series([0.8, float("nan"), 0.6], index=idx)
    out = meta_confidence_weight_signal(primary, proba)
    # Row 0: P=0.8 -> weight 0.6 -> 0.6
    # Row 1: P=NaN -> weight 0 -> 0 (the fix)
    # Row 2: P=0.6 -> weight 0.2 -> 0.1
    np.testing.assert_allclose(out.values, [0.6, 0.0, 0.1])


def test_confidence_weight_rejects_invalid_floor_cap():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=1), ["A"]], names=["date", "ticker"]
    )
    primary = pd.Series([0.5], index=idx)
    proba = pd.Series([0.6], index=idx)
    with pytest.raises(ValueError, match="cap.*must exceed.*floor"):
        meta_confidence_weight_signal(primary, proba, floor=0.7, cap=0.5)


# -------- Phase 9c: walk-forward meta-CV -----------------------------------


def test_apply_meta_gate_walk_forward_folds_runs():
    """K=3 folds should produce a series with the original index."""
    rng = np.random.default_rng(0)
    n = 600
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    realised = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    labels = realised.rename("fwd_return_5").to_frame()
    feats = pd.DataFrame({"f1": rng.normal(size=len(idx))}, index=idx)

    from stockpred.models.gbm import GBMConfig

    gated = _apply_meta_gate(
        score,
        feats,
        labels,
        bt_horizon_for_meta=5,
        threshold=0.5,
        gbm_cfg=GBMConfig(),
        log=logging.getLogger("test"),
        walk_forward_folds=3,
    )
    assert gated.index.equals(score.index)
    assert len(gated) == len(score)


# -------- Phase 9d: per-sector meta ---------------------------------------


def test_apply_meta_gate_per_sector_empty_map_falls_back():
    """Empty sector_map -> falls back to global meta-gate, doesn't crash."""
    rng = np.random.default_rng(1)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["A", "B"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    realised = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    labels = realised.rename("fwd_return_5").to_frame()
    feats = pd.DataFrame({"f1": rng.normal(size=len(idx))}, index=idx)

    from stockpred.models.gbm import GBMConfig

    out = _apply_meta_gate_per_sector(
        score,
        feats,
        labels,
        sector_map={},
        bt_horizon_for_meta=5,
        threshold=0.5,
        gbm_cfg=GBMConfig(),
        log=logging.getLogger("test"),
    )
    # Same index, same length.
    assert out.index.equals(score.index)


def test_apply_meta_gate_per_sector_with_two_sectors():
    """6 tickers, 2 sectors; each sector should get its own gated subset
    and the result should cover all tickers."""
    rng = np.random.default_rng(2)
    n = 500
    dates = pd.bdate_range("2020-01-01", periods=n)
    tickers = ["A", "B", "C", "D", "E", "F"]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx)
    realised = pd.Series(rng.normal(0, 0.01, size=len(idx)), index=idx)
    labels = realised.rename("fwd_return_5").to_frame()
    feats = pd.DataFrame({"f1": rng.normal(size=len(idx))}, index=idx)
    sector_map = {"A": "X", "B": "X", "C": "X", "D": "Y", "E": "Y", "F": "Y"}

    from stockpred.models.gbm import GBMConfig

    out = _apply_meta_gate_per_sector(
        score,
        feats,
        labels,
        sector_map=sector_map,
        bt_horizon_for_meta=5,
        threshold=0.5,
        gbm_cfg=GBMConfig(),
        log=logging.getLogger("test"),
    )
    # All tickers represented.
    out_tickers = set(out.index.get_level_values("ticker").unique())
    assert out_tickers == set(tickers)


# -------- Pipeline integration with new Phase 9 modes -------------------


@pytest.fixture
def synthetic_market(monkeypatch):
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2014-01-02", "2022-12-31")
    tickers = [f"T{i:02d}" for i in range(25)]
    prices = 100 * np.exp(
        np.cumsum(rng.normal(0.0002, 0.012, size=(len(dates), len(tickers))), axis=0)
    )
    rows = []
    for i, t in enumerate(tickers):
        df = pd.DataFrame(
            {
                "open": prices[:, i],
                "high": prices[:, i] * 1.005,
                "low": prices[:, i] * 0.995,
                "close": prices[:, i],
                "adj_close": prices[:, i],
                "volume": rng.integers(1e6, 5e7, len(dates)),
            },
            index=dates,
        )
        df.index.name = "date"
        df["ticker"] = t
        rows.append(df)
    long_panel = pd.concat(rows).reset_index().set_index(["date", "ticker"]).sort_index()
    mem = pd.DataFrame(
        {
            "ticker": tickers,
            "start_date": [pd.NaT] * len(tickers),
            "end_date": [pd.NaT] * len(tickers),
        }
    )

    monkeypatch.setattr(
        "stockpred.pipeline.universe_mod.fetch_sp500_membership", lambda *a, **kw: mem
    )
    monkeypatch.setattr(
        "stockpred.pipeline.universe_mod.all_tickers_in_range", lambda *a, **kw: tickers
    )
    monkeypatch.setattr("stockpred.pipeline_v5.prices_mod.long_panel", lambda *a, **kw: long_panel)
    monkeypatch.setattr("stockpred.pipeline.prices_mod.long_panel", lambda *a, **kw: long_panel)
    monkeypatch.setattr(
        "stockpred.pipeline_v5.fundamentals_mod.fetch_fundamentals", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr("stockpred.pipeline_v5.fundamentals_mod.sector_map", lambda *a, **kw: {})
    monkeypatch.setattr(
        "stockpred.pipeline_v5.macro_mod.fetch_macro", lambda *a, **kw: pd.DataFrame()
    )
    monkeypatch.setattr(
        "stockpred.pipeline_v5.prices_mod.fetch_one", lambda *a, **kw: pd.DataFrame()
    )
    return tickers


@pytest.mark.slow
def test_phase9_confidence_mode_runs_end_to_end(synthetic_market, tmp_path, monkeypatch):
    monkeypatch.setattr("stockpred.pipeline_v5.REPORTS_DIR", tmp_path)
    cfg = PipelineV5Config(
        start_date="2014-01-02",
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
        meta_mode="confidence",
        meta_conf_floor=0.5,
        meta_conf_cap=1.0,
        bootstrap_n=50,
    )
    result = run_pipeline_v5(cfg)
    assert "metrics" in result and "holdout_metrics" in result


@pytest.mark.slow
def test_phase9_walk_forward_meta_runs_end_to_end(synthetic_market, tmp_path, monkeypatch):
    monkeypatch.setattr("stockpred.pipeline_v5.REPORTS_DIR", tmp_path)
    cfg = PipelineV5Config(
        start_date="2014-01-02",
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
        meta_walk_forward_folds=3,
        bootstrap_n=50,
    )
    result = run_pipeline_v5(cfg)
    assert "metrics" in result and "holdout_metrics" in result
