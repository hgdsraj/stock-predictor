# stock-predictor

Cross-sectional directional forecaster for S&P 500 equities. Local Python, free data,
honest validation. Built to **learn what works and what doesn't**, not to promise alpha.

## Honest expectations (read first)

- **Baseline:** random = 50% accuracy. A non-leaking model on daily horizons typically
  achieves **51–54% directional accuracy out-of-sample**. Anything above ~55% on
  walk-forward validation is almost always a bug (lookahead, target leakage, survivorship,
  or train/test contamination).
- **Most "successful" backtests on the internet are broken.** This project applies the
  defenses from Marcos López de Prado's *Advances in Financial Machine Learning*:
  purged + embargoed walk-forward CV, realistic transaction costs, point-in-time labels.
- **Free data has real limits.** No point-in-time fundamentals; survivorship in current
  ticker lists; no tick data. We mitigate by reconstructing S&P 500 historical
  constituents from Wikipedia change logs, but it's still imperfect.
- **The benchmark to beat is "long SPY".** Most strategies lose to it after costs.

## Architecture

```
src/stockpred/
├── data/         # universe, prices, macro loaders (parquet-cached)
├── features/     # technical + cross-sectional features (lag-safe)
├── labels.py     # forward returns + binary labels for multiple horizons
├── models/       # logistic baseline, LightGBM
├── validation/   # purged walk-forward CV, IC / hit-rate / Sharpe
├── backtest/     # vectorized engine with costs, long/short top-k portfolio
└── reports/      # HTML tearsheet
```

## Quickstart

```bash
# install (one-time)
uv sync --extra dev

# Phase 1 end-to-end: pull data, train baseline, walk-forward backtest
uv run python scripts/run_phase1.py
```

## Anti-patterns this project actively prevents

- ❌ Train on all data, "test" on a subset → enforced via `WalkForwardSplit`.
- ❌ Features using same-day close to predict same-day return → labels are forward,
  features use only data up to `t`, prediction is for `[t+1, t+1+h]`.
- ❌ Hyperparameter tuning on the test set → CV is nested.
- ❌ Backtest without costs → costs default to **5 bps per side**, configurable.
- ❌ Survivorship → universe is point-in-time constituents.
- ❌ Cherry-picked window → reports break out yearly performance + holdout.

## Roadmap

- **Phase 1** — Foundation: universe, prices, baseline, walk-forward, costs ← *here*
- **Phase 2** — Features + LightGBM
- **Phase 3** — Portfolio construction + tearsheet
- **Phase 4** — Stress tests, holdout, sensitivity
- **Phase 5** — (Optional) sequence models, news sentiment, regime overlay

## License

Personal/research use. No warranty. **Not investment advice.**
