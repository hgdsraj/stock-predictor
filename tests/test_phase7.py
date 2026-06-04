"""Phase 7 tests: HRP, triple-barrier labels, meta-labelling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockpred.backtest.hrp import HRPConfig, hrp_long_short_weights, hrp_weights
from stockpred.labels_triple_barrier import (
    TripleBarrierConfig,
    compute_triple_barrier_labels,
)
from stockpred.models.meta import (
    MetaConfig,
    build_meta_dataset,
    meta_filter_signal,
)


# -------- HRP ---------------------------------------------------------------


def test_hrp_weights_sum_to_one_and_are_positive():
    rng = np.random.default_rng(0)
    n = 8
    A = rng.normal(size=(200, n))
    cov = np.cov(A, rowvar=False)
    w = hrp_weights(cov)
    assert len(w) == n
    assert (w >= 0).all()
    np.testing.assert_allclose(w.sum(), 1.0)


def test_hrp_single_asset_returns_one():
    cov = np.array([[0.02]])
    w = hrp_weights(cov)
    assert w[0] == 1.0


def test_hrp_assigns_more_to_lower_variance():
    """Cluster of two assets where A has 4x the variance of B; HRP via the
    bisection step assigns inverse-variance, so B should get the larger
    weight."""
    cov = np.array(
        [
            [0.04, 0.0],
            [0.0, 0.01],
        ]
    )
    w = hrp_weights(cov)
    assert w[1] > w[0]


def test_hrp_long_short_weights_per_day_balance():
    """Build score + close, ask hrp_long_short_weights for a per-day balanced
    long/short basket, and verify both sides have the expected gross."""
    rng = np.random.default_rng(1)
    dates = pd.bdate_range("2020-01-01", periods=120)
    n_t = 10
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(dates), n_t)), axis=0)),
        index=dates,
        columns=[f"T{i}" for i in range(n_t)],
    )
    # Score: arbitrary, but reproducible.
    idx = pd.MultiIndex.from_product([dates[80:], close.columns], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx, name="score")

    cfg = HRPConfig(cov_window=60, top_fraction=0.3, leverage_per_side=1.0, min_names_per_side=2)
    w = hrp_long_short_weights(score, close, cfg=cfg)
    assert not w.empty
    # Each row should be dollar-neutral within rounding.
    net = w.sum(axis=1)
    assert (net.abs() < 1e-6).all(), f"non-dollar-neutral days: {net[net.abs() >= 1e-6]}"
    # And each side should have gross ~= leverage_per_side.
    long_gross = w.clip(lower=0).sum(axis=1)
    short_gross = (-w.clip(upper=0)).sum(axis=1)
    np.testing.assert_allclose(long_gross.values, 1.0, atol=1e-6)
    np.testing.assert_allclose(short_gross.values, 1.0, atol=1e-6)


# -------- Triple-barrier labels --------------------------------------------


def test_triple_barrier_label_is_one_when_upper_hit_first():
    """For label at date t, entry is close[t+1] and the path checked is
    close[t+2..t+1+H]. We bump close[t+2] strongly upward to force an
    upper-barrier hit."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(0)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=60)))
    # Bump price strongly on day t+2 (relative to t=50): index 52.
    px[52:] = px[52:] * np.exp(0.10)  # propagate so path return is large
    close = pd.DataFrame({"X": px}, index=dates)
    cfg = TripleBarrierConfig(max_horizon=5, k_up=1.5, k_dn=1.5, vol_window=21)
    out = compute_triple_barrier_labels(close, cfg)
    label_day_50 = out.loc[(dates[50], "X"), "tb_label"]
    assert label_day_50 == 1.0


def test_triple_barrier_label_is_minus_one_when_lower_hit_first():
    dates = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(2)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=60)))
    px[52:] = px[52:] * np.exp(-0.10)
    close = pd.DataFrame({"X": px}, index=dates)
    cfg = TripleBarrierConfig(max_horizon=5, k_up=1.5, k_dn=1.5, vol_window=21)
    out = compute_triple_barrier_labels(close, cfg)
    assert out.loc[(dates[50], "X"), "tb_label"] == -1.0


def test_triple_barrier_label_is_zero_when_no_barrier_hit():
    """Flat prices: neither barrier hit, vertical wins -> label 0."""
    dates = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(3)
    # Build a noisy series first 30 days, then flat for the rest.
    early = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=30)))
    px = np.concatenate([early, np.full(30, early[-1])])
    close = pd.DataFrame({"X": px}, index=dates)
    cfg = TripleBarrierConfig(max_horizon=5, k_up=2.0, k_dn=2.0, vol_window=21)
    out = compute_triple_barrier_labels(close, cfg)
    # Pick a day in the flat region.
    val = out.loc[(dates[45], "X"), "tb_label"]
    assert val == 0.0


def test_triple_barrier_labels_are_lag_safe_for_vol_denominator():
    """Mutating close[t] must not change the triple-barrier label at date t.

    The vol denominator at t uses returns strictly through close-of-(t-1),
    so any same-day mutation to close[t] should leave the label unchanged.
    """
    rng = np.random.default_rng(4)
    dates = pd.bdate_range("2020-01-01", periods=60)
    px = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=60)))
    close = pd.DataFrame({"X": px}, index=dates)
    cfg = TripleBarrierConfig(max_horizon=5, k_up=2.0, k_dn=2.0, vol_window=21)
    out = compute_triple_barrier_labels(close, cfg)
    snap = out.loc[(dates[40], "X"), "tb_label"]

    # The label at t depends on path[t+1..t+H]. Mutating close[t] DOES
    # change the entry log_p[t+1]? No — entry is log_p[t+1], not log_p[t].
    # But the vol DOES depend on returns through t-1 (so close[t-1])
    # which we don't mutate. So mutating close[t] alone must NOT change
    # the label.
    px2 = px.copy()
    px2[40] = px2[40] * 1.10  # bump close on date t
    close2 = pd.DataFrame({"X": px2}, index=dates)
    out2 = compute_triple_barrier_labels(close2, cfg)
    snap2 = out2.loc[(dates[40], "X"), "tb_label"]
    assert snap == snap2 or (pd.isna(snap) and pd.isna(snap2))


# -------- Meta-labelling ---------------------------------------------------


def test_meta_filter_zeros_low_proba_signals():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=3), ["A", "B"]], names=["date", "ticker"]
    )
    primary = pd.Series([0.5, -0.3, 0.8, -0.2, 0.1, -0.6], index=idx)
    meta = pd.Series([0.7, 0.4, 0.6, 0.3, 0.8, 0.2], index=idx)
    out = meta_filter_signal(primary, meta, p_threshold=0.5)
    # rows where meta < 0.5 should be 0; others keep their original value.
    expected = pd.Series([0.5, 0.0, 0.8, 0.0, 0.1, 0.0], index=idx)
    pd.testing.assert_series_equal(out, expected, check_names=False)


def test_build_meta_dataset_drops_realised_zero():
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=2), ["A", "B"]], names=["date", "ticker"]
    )
    # Note pandas product orders as: (date0, A), (date0, B), (date1, A), (date1, B)
    primary = pd.Series([0.5, 0.3, -0.4, -0.2], index=idx)
    realised = pd.Series([0.01, 0.0, -0.02, 0.005], index=idx)  # (date0, B) is 0
    features = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0]}, index=idx)
    X_meta, y_meta = build_meta_dataset(primary, realised, features, use_primary_score=True)
    # The realised==0 row should be dropped (one of four).
    assert len(y_meta) == 3
    assert "primary_abs" in X_meta.columns
    # Surviving rows in order:
    #   (d0,A): primary +0.5, realised +0.01 -> sign match -> 1
    #   (d1,A): primary -0.4, realised -0.02 -> sign match -> 1
    #   (d1,B): primary -0.2, realised +0.005 -> mismatch  -> 0
    assert list(y_meta.values) == [1, 1, 0]


def test_meta_config_defaults_are_sensible():
    cfg = MetaConfig()
    assert 0.5 < cfg.p_threshold < 1.0
    assert cfg.use_primary_score is True


def test_build_meta_dataset_rejects_forbidden_columns():
    """Review CRIT-3: forbid features that would leak primary or label."""
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2020-01-01", periods=2), ["A"]], names=["date", "ticker"]
    )
    primary = pd.Series([0.5, 0.3], index=idx)
    realised = pd.Series([0.01, -0.02], index=idx)
    # Forbidden name: 'primary'
    bad_feats = pd.DataFrame({"f1": [1.0, 2.0], "primary": [0.5, 0.3]}, index=idx)
    with pytest.raises(ValueError, match="forbidden columns"):
        build_meta_dataset(primary, realised, bad_feats)
    # Forbidden by 'fwd_' prefix
    bad_feats2 = pd.DataFrame({"f1": [1.0, 2.0], "fwd_return_5": [0.01, -0.02]}, index=idx)
    with pytest.raises(ValueError, match="forward-looking"):
        build_meta_dataset(primary, realised, bad_feats2)


# -------- HRP overlap bug regression --------------------------------------


def test_hrp_disjoint_cohorts_when_top_fraction_above_half():
    """Review HIGH-6: with top_fraction > 0.5, kk must be clamped to n//2
    so long and short cohorts remain disjoint and the basket stays
    dollar-neutral."""
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2020-01-01", periods=120)
    n_t = 10
    close = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(dates), n_t)), axis=0)),
        index=dates,
        columns=[f"T{i}" for i in range(n_t)],
    )
    idx = pd.MultiIndex.from_product([dates[80:], close.columns], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx, name="score")
    # top_fraction=0.6 would naively put 6 in each cohort (overlap on 2).
    cfg = HRPConfig(cov_window=60, top_fraction=0.6, leverage_per_side=1.0, min_names_per_side=2)
    w = hrp_long_short_weights(score, close, cfg=cfg)
    # Per-day a long/short must be disjoint -> dollar-neutral within rounding.
    net = w.sum(axis=1)
    assert (net.abs() < 1e-6).all(), "kk not clamped: net exposure non-zero"


def test_vol_scaled_disjoint_cohorts_when_top_fraction_above_half():
    """Same fix in vol_scaled_weights — long and short must stay disjoint
    when top_fraction > 0.5."""
    from stockpred.backtest.portfolio import vol_scaled_weights

    rng = np.random.default_rng(13)
    dates = pd.bdate_range("2020-01-01", periods=5)
    tickers = [f"T{i}" for i in range(10)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    score = pd.Series(rng.normal(size=len(idx)), index=idx, name="score")
    vol = pd.DataFrame(
        rng.uniform(0.005, 0.05, size=(len(dates), len(tickers))), index=dates, columns=tickers
    )
    w = vol_scaled_weights(score, vol, top_fraction=0.6, leverage_per_side=1.0)
    net = w.sum(axis=1)
    assert (net.abs() < 1e-6).all(), "kk not clamped in vol_scaled_weights"
