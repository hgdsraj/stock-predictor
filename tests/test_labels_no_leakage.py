"""Guard tests: forward returns never use information at time t to compute label at time t.

If the label at date t uses prices known on date t, then any feature on date t
would be trivially "predictive" -> the model would look great but be useless live.
We assert: label_t depends only on prices strictly after t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpred.labels import compute_forward_returns


def _toy_prices(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=60)
    cols = ["AAA", "BBB", "CCC"]
    px = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(idx), len(cols))), axis=0)),
        index=idx,
        columns=cols,
    )
    return px


def test_label_does_not_depend_on_current_close():
    """Mutating close[t] must NOT change label[t] when trade_next_open=True."""
    px = _toy_prices()
    labels = compute_forward_returns(px, horizons=(1,), trade_next_open=True)[1]
    label_at_t = labels.iloc[10].copy()

    px2 = px.copy()
    px2.iloc[10] = px2.iloc[10] * 2.0  # arbitrary perturbation to close[t]
    labels2 = compute_forward_returns(px2, horizons=(1,), trade_next_open=True)[1]

    # Label at t should be unchanged because it uses close[t+1] and close[t+2].
    pd.testing.assert_series_equal(labels2.iloc[10], label_at_t, check_names=False)


def test_label_does_not_depend_on_past_prices():
    """Mutating close[t-1] must NOT change label[t]."""
    px = _toy_prices()
    labels = compute_forward_returns(px, horizons=(5,), trade_next_open=True)[5]
    label_at_t = labels.iloc[20].copy()

    px2 = px.copy()
    px2.iloc[5] = px2.iloc[5] * 0.5  # mutate the distant past
    labels2 = compute_forward_returns(px2, horizons=(5,), trade_next_open=True)[5]

    pd.testing.assert_series_equal(labels2.iloc[20], label_at_t, check_names=False)


def test_label_depends_on_future_prices():
    """Sanity: mutating close[t+2] DOES change label[t] for horizon 1."""
    px = _toy_prices()
    labels = compute_forward_returns(px, horizons=(1,), trade_next_open=True)[1]
    label_at_t = labels.iloc[10].copy()

    px2 = px.copy()
    px2.iloc[12] = px2.iloc[12] * 1.5
    labels2 = compute_forward_returns(px2, horizons=(1,), trade_next_open=True)[1]

    diff = (labels2.iloc[10] - label_at_t).abs().sum()
    assert diff > 0, "Label should change when future prices change"
