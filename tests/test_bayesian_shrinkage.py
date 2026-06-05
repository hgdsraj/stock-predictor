"""Phase 19 tests: per-ticker Bayesian shrinkage of ensemble scores."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.portfolio.bayesian_shrinkage import (
    apply_shrinkage_to_panel,
    compute_per_ticker_sign_precision,
    compute_shrinkage_factors,
    fit_apply_bayesian_shrinkage,
)


def _make_panel(
    sign_pre_by_ticker: dict[str, float],
    n_days: int = 100,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Synthesize (pred, realised) such that each ticker has the given
    sign-precision in expectation. Both series indexed (date, ticker)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    rows = []
    for d in dates:
        for t, p in sign_pre_by_ticker.items():
            pred = rng.normal(0, 1)
            # Realised sign matches pred sign with probability p
            same = rng.random() < p
            realised = abs(rng.normal(0, 1)) * (np.sign(pred) if same else -np.sign(pred))
            rows.append((d, t, pred, realised))
    df = pd.DataFrame(rows, columns=["date", "ticker", "pred", "realised"]).set_index(
        ["date", "ticker"]
    )
    return df["pred"], df["realised"]


def test_compute_sign_precision_recovers_true_rates():
    """The empirical precision should converge to the true rate
    over enough days."""
    pred, realised = _make_panel({"GOOD": 0.7, "BAD": 0.3, "NEUTRAL": 0.5}, n_days=500, seed=1)
    sp = compute_per_ticker_sign_precision(pred, realised, min_obs=10)
    assert abs(sp["GOOD"] - 0.7) < 0.05
    assert abs(sp["BAD"] - 0.3) < 0.05
    assert abs(sp["NEUTRAL"] - 0.5) < 0.05


def test_compute_sign_precision_min_obs_drops_short_history():
    """Tickers with < min_obs paired non-NaN rows return NaN."""
    pred, realised = _make_panel({"OK": 0.6, "FEW": 0.6}, n_days=30, seed=2)
    # Manually wipe most of FEW's rows to fewer than min_obs
    pred_arr = pred.copy()
    pred_arr.loc[(slice(None), "FEW")] = np.nan
    sp = compute_per_ticker_sign_precision(pred_arr, realised, min_obs=20)
    assert "OK" in sp.index and not np.isnan(sp["OK"])
    assert "FEW" not in sp.index or np.isnan(sp.get("FEW", np.nan))


def test_compute_shrinkage_factors_formula():
    """0.5 -> 0; 0.6 -> 0.2; 0.75 -> 0.5; 1.0 -> 1.0; <0.5 -> 0."""
    sp = pd.Series({"A": 0.5, "B": 0.6, "C": 0.75, "D": 1.0, "E": 0.3})
    sf = compute_shrinkage_factors(sp, alpha=1.0)
    assert abs(sf["A"] - 0.0) < 1e-6
    assert abs(sf["B"] - 0.2) < 1e-6
    assert abs(sf["C"] - 0.5) < 1e-6
    assert abs(sf["D"] - 1.0) < 1e-6
    assert abs(sf["E"] - 0.0) < 1e-6  # worse than random -> clipped to 0


def test_compute_shrinkage_factors_alpha_scales():
    """alpha=0.5 should halve every shrink factor vs alpha=1.0."""
    sp = pd.Series({"A": 0.6, "B": 0.75})
    sf1 = compute_shrinkage_factors(sp, alpha=1.0)
    sf_half = compute_shrinkage_factors(sp, alpha=0.5)
    np.testing.assert_allclose(sf_half, sf1 * 0.5, atol=1e-6)


def test_compute_shrinkage_factors_alpha_out_of_range_raises():
    with pytest.raises(ValueError):
        compute_shrinkage_factors(pd.Series({"A": 0.6}), alpha=1.5)
    with pytest.raises(ValueError):
        compute_shrinkage_factors(pd.Series({"A": 0.6}), alpha=-0.1)


def test_compute_shrinkage_factors_nan_uses_default():
    """NaN precision -> uses default=0.5 -> shrink 0."""
    sp = pd.Series({"A": np.nan})
    sf = compute_shrinkage_factors(sp, alpha=1.0, default=0.5)
    assert sf["A"] == 0.0


def test_apply_shrinkage_zeroes_missing_tickers():
    """Tickers not in shrink_factors get factor 0 (dropped)."""
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=3), ["A", "B", "MISSING"]],
        names=["date", "ticker"],
    )
    score = pd.Series(1.0, index=idx, name="score")
    sf = pd.Series({"A": 0.5, "B": 0.2})
    sf.index.name = "ticker"
    out = apply_shrinkage_to_panel(score, sf)
    assert out.xs("A", level="ticker").iloc[0] == 0.5
    assert out.xs("B", level="ticker").iloc[0] == 0.2
    assert out.xs("MISSING", level="ticker").iloc[0] == 0.0


def test_apply_shrinkage_empty_input():
    empty = pd.Series(dtype=float, name="score")
    out = apply_shrinkage_to_panel(empty, pd.Series({"A": 0.5}))
    assert out.empty


def test_fit_apply_round_trip_zeros_bad_tickers():
    """End-to-end: a noise ticker (p=0.3) should be zeroed in the output."""
    pred, realised = _make_panel({"GOOD": 0.7, "BAD": 0.3, "NEUTRAL": 0.5}, n_days=300, seed=3)
    out = fit_apply_bayesian_shrinkage(pred, realised, pred, alpha=1.0, min_obs=10)
    # GOOD signal should be downweighted but non-zero
    assert (out.xs("GOOD", level="ticker") != 0).any()
    # BAD signal should be all zeros (worse than random -> clipped to 0)
    assert (out.xs("BAD", level="ticker") == 0).all()
    # NEUTRAL should also be zero (no edge -> shrink 0)
    assert (out.xs("NEUTRAL", level="ticker") == 0).all()


def test_fit_apply_alpha_zero_passthrough():
    """alpha=0 -> all shrink factors are 0 -> output is all zeros.
    NOT the same as 'pass through unchanged'; the formula
    shrink = (p-0.5)/0.5 * alpha is 0 when alpha=0 regardless of p.
    The use_bayesian_shrinkage gate in pipeline_v5 only invokes the
    function when alpha>0, so this is fine in practice.
    """
    pred, realised = _make_panel({"GOOD": 0.7}, n_days=100, seed=4)
    out = fit_apply_bayesian_shrinkage(pred, realised, pred, alpha=0.0, min_obs=10)
    assert (out == 0).all()
