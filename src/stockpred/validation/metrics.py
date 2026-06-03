"""Evaluation metrics.

Two layers:

1. **Statistical metrics** computed per (date, prediction, realised) tuple:
     - hit_rate: fraction of correctly signed predictions
     - information_coefficient (IC): Spearman correlation of preds vs realised
       returns per date, then averaged
     - rank_ic: Pearson on ranks (equivalent to Spearman)

2. **Trading metrics** computed on a strategy return series:
     - annualised_return, annualised_vol, sharpe, sortino, max_drawdown,
       calmar, hit_ratio, turnover
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ---------------------------- statistical ---------------------------- #


def hit_rate(preds: pd.Series, realised: pd.Series) -> float:
    """Fraction of predictions where sign(pred) == sign(realised), ignoring zeros."""
    df = pd.concat([preds.rename("p"), realised.rename("r")], axis=1).dropna()
    df = df[(df["p"] != 0) & (df["r"] != 0)]
    if df.empty:
        return float("nan")
    return float((np.sign(df["p"]) == np.sign(df["r"])).mean())


def information_coefficient(
    preds: pd.Series, realised: pd.Series, *, by: str = "date"
) -> pd.Series:
    """Per-date Spearman correlation of preds vs realised returns.

    `preds` and `realised` must share a multiindex containing `by`.
    """
    df = pd.concat([preds.rename("p"), realised.rename("r")], axis=1).dropna()

    def _ic(group: pd.DataFrame) -> float:
        if len(group) < 5:
            return float("nan")
        rho, _ = spearmanr(group["p"], group["r"])
        return float(rho)

    return df.groupby(level=by).apply(_ic)


def ic_summary(ic_series: pd.Series) -> dict[str, float]:
    """Mean IC, IC std, IC IR (mean/std * sqrt(N))."""
    ic = ic_series.dropna()
    if ic.empty:
        return {"ic_mean": float("nan"), "ic_std": float("nan"), "ic_ir": float("nan")}
    return {
        "ic_mean": float(ic.mean()),
        "ic_std": float(ic.std()),
        "ic_ir": float(ic.mean() / ic.std() * np.sqrt(252)) if ic.std() else float("nan"),
    }


# ----------------------------- trading ------------------------------- #


def annualised_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    return float((1 + r).prod() ** (periods_per_year / len(r)) - 1)


def annualised_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / periods_per_year
    if r.std() == 0 or r.empty:
        return float("nan")
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino(returns: pd.Series, periods_per_year: int = 252, rf: float = 0.0) -> float:
    r = returns.dropna() - rf / periods_per_year
    downside = r[r < 0].std()
    if downside == 0 or r.empty:
        return float("nan")
    return float(r.mean() / downside * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    cum = (1 + returns.fillna(0)).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1
    return float(dd.min())


def calmar(returns: pd.Series, periods_per_year: int = 252) -> float:
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("nan")
    return float(annualised_return(returns, periods_per_year) / abs(mdd))


def tearsheet_metrics(returns: pd.Series) -> dict[str, float]:
    """One-shot summary suitable for printing."""
    return {
        "ann_return": annualised_return(returns),
        "ann_vol": annualised_vol(returns),
        "sharpe": sharpe(returns),
        "sortino": sortino(returns),
        "max_drawdown": max_drawdown(returns),
        "calmar": calmar(returns),
        "hit_ratio": float((returns > 0).mean()),
        "n_days": int(returns.dropna().shape[0]),
    }
