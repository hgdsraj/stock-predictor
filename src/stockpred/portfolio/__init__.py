"""Portfolio-level signal transforms applied AFTER per-horizon model
predictions and BEFORE the existing portfolio constructors in
`stockpred.backtest.portfolio` / `stockpred.backtest.hrp`.

Modules:
  bayesian_shrinkage  Phase 19: per-ticker shrinkage of scores by
                      historical sign-precision.
"""

from stockpred.portfolio.bayesian_shrinkage import (
    apply_shrinkage_to_panel,
    compute_per_ticker_sign_precision,
    compute_shrinkage_factors,
    fit_apply_bayesian_shrinkage,
)

__all__ = [
    "apply_shrinkage_to_panel",
    "compute_per_ticker_sign_precision",
    "compute_shrinkage_factors",
    "fit_apply_bayesian_shrinkage",
]
