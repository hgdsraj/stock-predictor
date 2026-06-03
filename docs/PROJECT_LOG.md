# Project log — stock-predictor

A detailed log of decisions, what was built, why, and what the first real
backtest produced. This file lives in-repo so it travels with the code.

---

## 1. Original ask

> "build the best stock predictor in history"

This was scoped down to something honest and achievable through a short Q&A:

| Decision        | Choice                                         |
| --------------- | ---------------------------------------------- |
| Goal            | Directional price forecaster (cross-sectional) |
| Scope           | Serious system, weeks+                         |
| Constraints     | Free, local, fully portable                    |
| Universe        | S&P 500 historical constituents                |
| Horizons        | 1d / 5d / 21d                                  |
| History         | 2005–present (default in config)               |
| Env             | Local Python project (uv-managed)              |
| Webapp          | Plan a real hosted webapp later (deferred)     |
| Project log     | `docs/PROJECT_LOG.md` (this file)              |

## 2. Honest baseline expectations (stated up front)

- Random = 50% directional accuracy.
- A non-leaking model on daily horizon, liquid US equities, typically achieves
  **51–54% out-of-sample**.
- Anything > 55% on walk-forward is almost always a bug.
- Most "successful" backtests on the internet are broken (lookahead,
  survivorship, no costs, optimised on test set).
- Benchmark to beat is **long SPY** — most strategies lose to it after costs.

## 3. Architecture

```
src/stockpred/
├── config.py              # paths + dataclass configs (Backtest/Horizon/CV/Universe)
├── pipeline.py            # end-to-end Phase 1 driver, importable
├── data/
│   ├── universe.py        # S&P 500 historical constituents via Wikipedia change log
│   ├── prices.py          # yfinance loader with per-ticker parquet cache
│   └── macro.py           # FRED loader (VIX, yields, USD, oil) — not used in Phase 1
├── features/
│   ├── technical.py       # lag-safe technicals: returns, vol, RSI, MACD, BB, etc.
│   └── cross_sectional.py # per-date ranks, centered exactly at 0
├── labels.py              # forward returns + binary direction, multi-horizon
├── validation/
│   ├── walk_forward.py    # expanding-window CV with purge + embargo
│   └── metrics.py         # IC / IC IR / Sharpe / Sortino / max-DD / Calmar
├── models/
│   ├── baseline.py        # impute → scale → logistic regression
│   └── gbm.py             # LightGBM regressor on forward returns
├── backtest/
│   ├── portfolio.py       # top-k long / bottom-k short, dollar-neutral
│   └── engine.py          # vectorised, costs, configurable trade_lag
└── reports/
    └── tearsheet.py       # self-contained HTML report (equity, DD, yearly table)

scripts/run_phase1.py      # CLI entrypoint: data → features → CV → backtest → report
tests/                     # 16 unit + integration tests
```

## 4. What was built (chronological)

1. **Project scaffold** — `pyproject.toml`, `.gitignore`, `uv` venv. Pinned
   PyPI directly via `[[tool.uv.index]]` so the project resolves on any
   machine, independent of any corp-internal package mirror.
2. **Universe loader** — Wikipedia change log parsing for S&P 500 historical
   membership. Caches a parquet snapshot to `data/cache/`.
3. **Prices loader** — yfinance with per-ticker parquet cache and ThreadPool
   parallelism. Tolerates partial failures (some tickers just won't download).
4. **Macro loader** — FRED via pandas-datareader. Plumbed but not wired into
   Phase 1 features yet.
5. **Labels** — Forward log returns over horizons 1d/5d/21d. Crucially
   `trade_next_open=True` by default: label for date `t` uses `close[t+1]` and
   `close[t+1+h]`, never the close on date `t` itself.
6. **Features** — 15+ lag-safe technical features per ticker per date, then
   cross-sectional rank versions of every numeric column (centered at 0 by
   subtracting per-day mean of pct-rank).
7. **Walk-forward CV** — Expanding window with explicit purge + embargo to
   prevent label leakage across the train/test boundary.
8. **Metrics** — IC, IC IR, hit rate, Sharpe, Sortino, max drawdown, Calmar.
9. **Models** — Logistic regression baseline (transparent leakage canary)
   and LightGBM (for Phase 2 work).
10. **Portfolio & backtest** — Top-k long / bottom-k short, equal-weighted,
    dollar-neutral. Vectorised backtester with realistic costs.
11. **Tearsheet** — Self-contained HTML report with embedded PNG charts.
12. **Tests** — 16 tests (4 categories) covering leakage, walk-forward
    correctness, backtest engine semantics, feature lag-safety, full
    end-to-end pipeline on synthetic data.

## 5. Bugs surfaced *by tests* and fixed

| # | Bug | How found | Fix |
| - | --- | --------- | --- |
| 1 | `pd.read_html(resp.text)` interpreted body as file path | Real-data run | Wrap in `io.StringIO(resp.text)`. Added `test_universe_html_parse.py` regression test. |
| 2 | Cross-sectional ranks not centered at zero (pct_rank mean = 0.625, not 0.5, with k=4) | `test_cross_sectional_ranks_centered_at_zero` | Subtract per-day mean of pct_rank instead of fixed 0.5. |
| 3 | Backtest gross_return was 0 on day 0 (no held position), polluting metrics | `test_constant_long_position_earns_underlying_return` | Mark gross NaN when no held position; net still charges day-0 cost. |
| 4 | Off-by-one between label horizon and backtest realisation: label was `[t+1, t+1+h]`, backtest realised `[t, t+1]` | Diagnostic of "positive IC, negative strategy" | Added `trade_lag` parameter to engine, default `trade_lag=2` for the pipeline (matches `trade_next_open=True`). Added `test_trade_lag_2_matches_label_alignment`. |
| 5 | Daily rebalancing with horizon-5 prediction destroyed signal via turnover | Backtest still negative after lag fix | Added `rebalance_every` knob in `PipelineConfig`. Default = horizon. |

## 6. Test suite

```
$ uv run pytest tests/ -v
…
============================= 16 passed in ~30s =============================
```

| File                              | Tests | Purpose                                      |
| --------------------------------- | ----- | -------------------------------------------- |
| `test_labels_no_leakage.py`       | 3     | Forward labels do not depend on past/current prices |
| `test_walk_forward.py`            | 3     | CV has no train/test overlap, respects embargo, expanding window grows |
| `test_features.py`                | 3     | Features lag-safe; cross-sectional ranks centered + bounded |
| `test_backtest_engine.py`         | 4     | Constant long earns underlying, costs charged correctly, dollar-neutral offsets, trade_lag=2 alignment |
| `test_universe_html_parse.py`     | 2     | Wikipedia HTML parses to membership; `read_html` regression |
| `test_pipeline_integration.py`    | 1     | Full end-to-end pipeline on synthetic noise: hit rate must be 35–65%, IC < 0.05 (leakage canary) |

## 7. Phase 1 results on real data

Run: `uv run python scripts/run_phase1.py --start 2018-01-01 --end 2024-12-31 --n-tickers 100 --k 10 --horizon 5`

| Metric             | Value           | Honest reading |
| ------------------ | --------------- | -------------- |
| Universe           | 100 names       | Some yfinance failures dropped a few |
| Feature matrix     | 167,822 × 36    | Full panel |
| Walk-forward folds | 6               | Train years=3, test months=6, embargo=10d |
| OOS hit rate       | **52.9%**       | Within honest 51–54% range |
| OOS IC mean        | **+0.012**      | Tiny but positive |
| OOS IC IR          | **+1.05**       | Stable signal across folds |
| Annualised return  | **-6.6%**       | Loses to cash |
| Sharpe (net)       | **-0.52**       | Bad |
| Max drawdown       | **-42%**        | Brutal |

**Reading:** the predictive signal is real (IC IR > 1) but the cohort the
model flags as "most likely to go up" experiences asymmetric tail losses that
overwhelm the modest edge. Inverting the sign of the score brings the strategy
to roughly flat (Sharpe ≈ 0), not positive — so it's not just a sign bug; the
quantile portfolios are not behaving like the means.

This is the kind of result that takes weeks of follow-up research: feature
ablation, sector neutralisation, regime split, dollar-neutral vs beta-neutral,
better risk model, etc. That's Phase 2+.

**Crucially:** the infrastructure correctly produced an honest disappointing
result. Pretending otherwise would defeat the purpose.

## 8. What's intentionally NOT here

- **No webapp / hosting / accounts.** That was deferred; the project is a
  local research codebase.
- **No corp / internal dependencies.** Pure PyPI, free APIs (yfinance, FRED,
  Wikipedia). Runs identically on any machine.
- **No point-in-time fundamentals.** Free data doesn't provide them honestly.
- **No tick data, no intraday.** Daily bars only.

## 9. How to run

```bash
# One-time setup
uv sync --extra dev

# All tests (~30s, includes one integration test with synthetic data)
uv run pytest tests/ -v

# End-to-end Phase 1 on real data (downloads to data/cache/ on first run)
uv run python scripts/run_phase1.py --start 2018-01-01 --end 2024-12-31 \
    --n-tickers 100 --k 10 --horizon 5

# Inspect the HTML tearsheet
xdg-open reports/phase1_h5_k10.html   # or open with any browser
```

## 10. Roadmap

- **Phase 1 — Foundation.** ✅ Done.
- **Phase 2 — GBM + sector neutralisation + more features.** Wire LightGBM
  through the pipeline; add sector dummies / sector-relative ranks; try
  alternate label definitions (vol-scaled, triple-barrier).
- **Phase 3 — Portfolio construction.** Risk model, beta neutralisation,
  turnover constraint via optimisation rather than coarse-rebalance hack.
- **Phase 4 — Stress tests.** Held-out 2-year window never touched; bootstrap
  Sharpe CIs; sensitivity to costs, hyperparameters, universe slices.
- **Phase 5 — Optional.** Sequence models (Transformer panel), LLM-based
  sentiment, regime overlay; later: hosted dashboard.

## 11. Files of interest

- `README.md` — quickstart + honest expectations
- `src/stockpred/pipeline.py` — the spine; read this first
- `src/stockpred/backtest/engine.py` — note the `trade_lag` docstring
- `tests/test_pipeline_integration.py` — leakage canary
- `reports/*.html` — generated tearsheets
