"""Core hyperparameter search logic for the Phase 5+ pipeline.

Factored out of scripts/run_hypersearch.py so it can be called from:
  - The CLI (scripts/run_hypersearch.py)
  - The web server job runner (backend/jobs.py)

The search space spans 20 pipeline parameters and optimises holdout Sharpe
via Optuna's TPE sampler.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable

import optuna
import pandas as pd

from stockpred.config import CVConfig
from stockpred.models.gbm import GBMConfig
from stockpred.pipeline_v5 import PipelineV5Config, run_pipeline_v5

log = logging.getLogger(__name__)

PENALTY = -10.0  # returned on error/NaN so Optuna avoids those regions


@dataclass
class HypersearchConfig:
    """All knobs for a hyperparameter search run."""

    n_trials: int = 50
    n_tickers: int = 25
    start_date: str = "2015-01-01"
    end_date: str | None = None
    holdout_years: int = 2
    bootstrap_n: int = 50
    universe_sampling: str = "current"
    seed: int = 42


def suggest_pipeline_config(trial: optuna.Trial, cfg: HypersearchConfig) -> PipelineV5Config:
    """Map a single Optuna trial to a PipelineV5Config.

    20 parameters across: portfolio construction, signal, features,
    meta-labelling (conditional), GBM, and CV walk-forward settings.
    """
    # ── Portfolio construction ─────────────────────────────────────────────
    position_sizing = trial.suggest_categorical("position_sizing", ["vol_scaled", "hrp", "top_k"])
    k_per_side_pct = trial.suggest_float("k_per_side_pct", 0.08, 0.25)
    leverage_per_side = trial.suggest_float("leverage_per_side", 0.5, 1.5)
    _sc = trial.suggest_categorical("sector_cap_gross", ["none", "0.20", "0.30", "0.40"])
    sector_cap_gross: float | None = None if _sc == "none" else float(_sc)
    min_trade_threshold = trial.suggest_float("min_trade_threshold", 0.001, 0.015, log=True)

    # ── Signal ────────────────────────────────────────────────────────────
    _hz = trial.suggest_categorical("horizons", ["5", "1,5"])
    horizons = tuple(int(h) for h in _hz.split(","))
    ensemble_weighting = trial.suggest_categorical("ensemble_weighting", ["ic_ir", "equal"])

    # ── Features ──────────────────────────────────────────────────────────
    use_tier2_features = trial.suggest_categorical("use_tier2_features", [True, False])
    use_regime_features = trial.suggest_categorical("use_regime_features", [True, False])
    use_sector_features = trial.suggest_categorical("use_sector_features", [True, False])
    ranks_only = trial.suggest_categorical("ranks_only", [True, False])
    beta_neutralise = trial.suggest_categorical("beta_neutralise", [True, False])

    # ── Meta-labelling (conditional on use_meta_labelling) ────────────────
    use_meta_labelling = trial.suggest_categorical("use_meta_labelling", [True, False])
    if use_meta_labelling:
        meta_threshold = trial.suggest_float("meta_threshold", 0.50, 0.65)
        meta_mode = trial.suggest_categorical("meta_mode", ["binary", "confidence"])
        meta_conf_floor = (
            trial.suggest_float("meta_conf_floor", 0.48, 0.72)
            if meta_mode == "confidence"
            else 0.5
        )
    else:
        meta_threshold = 0.55
        meta_mode = "binary"
        meta_conf_floor = 0.5

    # ── GBM ───────────────────────────────────────────────────────────────
    num_leaves = trial.suggest_categorical("num_leaves", [31, 63, 127])
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.08, log=True)
    n_estimators = trial.suggest_int("n_estimators", 400, 1200, step=200)

    # ── CV ────────────────────────────────────────────────────────────────
    train_years = trial.suggest_int("train_years", 2, 4)

    return PipelineV5Config(
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        n_tickers=cfg.n_tickers,
        universe_sampling=cfg.universe_sampling,
        refresh_data=False,
        horizons=horizons,
        model="gbm",
        gbm=GBMConfig(
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
        ),
        use_sector_features=use_sector_features,
        use_tier2_features=use_tier2_features,
        use_regime_features=use_regime_features,
        beta_neutralise=beta_neutralise,
        bootstrap_method="block",
        cv=CVConfig(
            train_years=train_years,
            test_months=6,
            embargo_days=25,
            min_train_obs=1000,
        ),
        holdout_years=cfg.holdout_years,
        position_sizing=position_sizing,
        k_per_side_pct=k_per_side_pct,
        leverage_per_side=leverage_per_side,
        sector_cap_gross=sector_cap_gross,
        min_trade_threshold=min_trade_threshold,
        ensemble_weighting=ensemble_weighting,
        use_meta_labelling=use_meta_labelling,
        meta_threshold=meta_threshold,
        meta_mode=meta_mode,
        meta_conf_floor=meta_conf_floor,
        meta_conf_cap=1.0,
        meta_walk_forward_folds=1,
        meta_per_sector=False,
        use_triple_barrier_labels=False,
        ranks_only=ranks_only,
        feature_exclude=(),
        use_edgar_features=False,
        use_edgar_item_features=False,
        bootstrap_n=cfg.bootstrap_n,
        tearsheet_path=None,
    )


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v if pd.notna(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def run_hypersearch(
    cfg: HypersearchConfig,
    *,
    on_trial_done: Callable[[dict], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    study: optuna.Study | None = None,
) -> optuna.Study:
    """Run the hyperparameter search.

    Args:
        cfg:           Search configuration.
        on_trial_done: Called after each trial with a flat dict of trial data.
                       Used by the server job to persist progress to the DB.
        should_stop:   Called before each trial; return True to stop early.
                       Used by the job runner to handle cancellation.
        study:         Existing Optuna study to resume. Created fresh if None.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if study is None:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        if should_stop is not None and should_stop():
            trial.study.stop()
            return PENALTY

        pipeline_cfg = suggest_pipeline_config(trial, cfg)
        t0 = time.time()
        try:
            res = run_pipeline_v5(pipeline_cfg)
        except Exception as exc:  # noqa: BLE001
            elapsed = round(time.time() - t0, 1)
            log.warning("trial %d failed (%.0fs): %s", trial.number, elapsed, exc)
            trial.set_user_attr("error", str(exc)[:200])
            trial.set_user_attr("elapsed_s", elapsed)
            if on_trial_done:
                on_trial_done(_trial_row(trial, PENALTY))
            return PENALTY

        elapsed = round(time.time() - t0, 1)
        hold = res.get("holdout_metrics") or {}
        dev = res.get("metrics") or {}
        ci = res.get("bootstrap_sharpe") or {}

        hold_sharpe = hold.get("sharpe", float("nan"))

        trial.set_user_attr("hold_sharpe", _safe_float(hold_sharpe))
        trial.set_user_attr("hold_ci_lo", _safe_float(ci.get("sharpe_lo", float("nan"))))
        trial.set_user_attr("hold_ci_hi", _safe_float(ci.get("sharpe_hi", float("nan"))))
        trial.set_user_attr("hold_dd", _safe_float(hold.get("max_drawdown", float("nan"))))
        trial.set_user_attr("hold_hit", _safe_float(hold.get("hit_ratio", float("nan"))))
        trial.set_user_attr("hold_ann_return", _safe_float(hold.get("ann_return", float("nan"))))
        trial.set_user_attr("dev_sharpe", _safe_float(dev.get("sharpe", float("nan"))))
        trial.set_user_attr("elapsed_s", elapsed)
        trial.set_user_attr("error", "")

        value = _safe_float(hold_sharpe) if pd.notna(hold_sharpe) else PENALTY

        if on_trial_done:
            on_trial_done(_trial_row(trial, value))

        return value

    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)
    return study


def _trial_row(trial: optuna.Trial, value: float) -> dict:
    """Flatten a trial into a JSON-serialisable dict for storage."""
    row: dict = {"trial": trial.number, "value": value}
    row.update(trial.params)
    row.update(trial.user_attrs)
    return row


def best_trial_params(study: optuna.Study) -> dict | None:
    try:
        return dict(study.best_trial.params)
    except ValueError:
        return None


def best_trial_sharpe(study: optuna.Study) -> float | None:
    try:
        v = study.best_value
        return float(v) if pd.notna(v) and v != PENALTY else None
    except ValueError:
        return None


def trials_to_records(study: optuna.Study) -> list[dict]:
    """Return all completed trial rows, sorted by hold_sharpe descending."""
    rows = []
    for t in study.trials:
        if t.state not in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.FAIL,
        ):
            continue
        row: dict = {"trial": t.number, "value": t.value if t.value is not None else PENALTY}
        row.update(t.params)
        row.update(t.user_attrs)
        rows.append(row)
    rows.sort(key=lambda r: r.get("hold_sharpe", float("-inf")), reverse=True)
    return rows
