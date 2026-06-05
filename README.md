# stock-predictor

A cross-sectional directional forecaster for S&P 500 equities, with a real
backend, real frontend, and honest backtests. Free public data only.
Single-container deploy. Built to surface what works and what doesn't —
*not* to promise alpha.

> **First time reading this?** Start with [`docs/CONCEPTS.md`](docs/CONCEPTS.md).
> It explains every term used here (return, hit rate, IC, IC IR, Sharpe,
> drawdown, leakage, walk-forward CV, …) for someone with no quant
> background. Then [`docs/USAGE.md`](docs/USAGE.md) shows how to install,
> run, and interpret the dashboard end-to-end.

## What it does in one paragraph

Every trading day, the system predicts which S&P 500 stocks will go up the
most and which will go down the most over the next 1, 5, and (formerly) 21
trading days. It buys the predicted winners, sells short the predicted
losers, and tracks the resulting portfolio's performance. A web dashboard
shows the predictions, a screener of every ticker, per-ticker deep-dive
pages with charts and news, and a backtest tearsheet. The pipeline can be
re-run on demand or on a daily cron.

A few non-S&P "watchlist" instruments (HND.TO, HNU.TO, UNG, SPY, ^VIX) are
also tracked for context — charted on the screener page but **not** fed into
the model (they behave differently from regular stocks).

## Honest expectations (read first)

- **Baseline.** Random = 50% directional accuracy. A non-leaking model on
  daily horizons typically achieves **51–54% out-of-sample**. Anything above
  ~55% on walk-forward is almost always a bug (lookahead, target leakage,
  contamination).
- **Most "successful" backtests on the internet are broken.** This project
  applies defenses from Marcos López de Prado's *Advances in Financial
  Machine Learning*: purged + embargoed walk-forward CV (in **trading days**),
  realistic transaction costs, point-in-time labels, horizon-aware backtest,
  exact-bounded cross-sectional ranks, leakage canary in CI.
- **Free data has real limits.** No real-time short interest, no point-in-time
  fundamentals (yfinance `.info` is current-as-of-fetch), survivorship
  mitigated but not eliminated.
- **The benchmark is "long SPY."** Most strategies lose to it after costs.
  The Phase 1+2 results on real data (see [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md))
  do too. The dashboard shows that honestly.

## What's in the box

```
src/stockpred/
├── config.py              # paths, dataclass configs
├── pipeline.py            # end-to-end driver
├── data/                  # universe, prices, macro, fundamentals (all cached)
├── features/              # lag-safe technicals, cross-sectional ranks, sector neutralisation
├── labels.py              # forward returns + vol-scaled targets per horizon
├── validation/            # walk-forward CV (trading-day embargo), metrics, stress tests
├── models/                # logistic baseline, LightGBM
├── backtest/              # horizon-aware engine, top-K & vol-scaled portfolios, sector caps
├── reports/               # standalone HTML tearsheet
└── backend/               # SQLite + SQLAlchemy + FastAPI + APScheduler

web/                       # React + Vite + TS + Tailwind + shadcn-style UI
├── src/pages/             # Home, Screener, Ticker, Backtest
├── src/components/        # Layout, ThemeProvider, ui/ primitives
└── ...

scripts/
├── run_phase1.py          # end-to-end CLI: data → backtest → tearsheet
└── serve.py               # uvicorn server

tests/                     # 45 tests; leakage canary + endpoint round-trip
docs/                      # PROJECT_LOG, DEPLOYMENT, HANDOFF
Dockerfile + docker-compose.yml
```

## Quick start

### Local dev (Python + Node)

```bash
# Install
uv sync --extra dev
cd web && npm ci && cd ..

# Run the pipeline once, write a tearsheet
uv run python scripts/run_phase1.py --start 2018-01-01 --n-tickers 60 \
    --horizons 1 5 21 --k 10 --model gbm

# Serve the backend
uv run python scripts/serve.py

# In another shell, run the frontend in dev mode (Vite + HMR)
cd web && npm run dev

# Or build the frontend and let FastAPI serve it
cd web && npm run build
# now visit http://127.0.0.1:8000
```

### Docker

```bash
docker compose up --build
open http://localhost:8000
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for Fly.io, Render, and VM
deployment walkthroughs.

### Local testing with synthetic data (no yfinance, no model training)

To click through the whole dashboard without fetching real data or running
the pipeline, seed a throwaway SQLite DB with randomly-generated data and
serve it:

```bash
# Seed data/local_test.db with fake prices/predictions/backtest and serve it
uv run python scripts/seed_synthetic.py --serve
# → backend on http://127.0.0.1:8000 (open it for the built dashboard)
```

This writes to `data/local_test.db`, so your real `data/app.db` is never
touched. Useful flags: `--n-tickers`, `--days`, `--seed`, `--reset`,
`--port`. The generated tickers (e.g. `ZQ07`) are obviously fake and every
number on the page is random — it exists only to exercise the UI.

To work on the **React frontend** against that synthetic data with hot
reload, start the backend (above) and in another shell:

```bash
cd web
npm install          # first time only
npm run dev          # Vite dev server on http://127.0.0.1:5173
```

Open <http://127.0.0.1:5173>. Vite proxies API calls to the backend on
`:8000`, so edits to the React code hot-reload instantly while still showing
the seeded data. See [`docs/USAGE.md`](docs/USAGE.md#local-testing-with-synthetic-data)
for the full walkthrough.

## Frontend pages

- **Home** — today's top long and short cohorts, KPI tiles, equity curve.
- **Screener** — the full S&P 500 with sector / industry / market-cap
  filters, sortable columns, live search by ticker or industry.
- **Ticker** — per-name page: 2y price chart with prediction-score overlay,
  fundamentals card (sector, beta, P/E, short ratio, 52w range), business
  summary.
- **Backtest** — KPI tiles (Sharpe, return, drawdown), equity curve,
  drawdown chart, per-horizon IC diagnostics, yearly performance.

Theme toggle, mobile nav, dark/light, all built with shadcn-style primitives
+ TanStack Query/Table + Recharts. Builds to ~205 KB gzipped.

## API endpoints

All under the FastAPI app (`http://localhost:8000`). Swagger UI at `/docs`.

| Method | Path                          | Returns                                      |
| ------ | ----------------------------- | -------------------------------------------- |
| GET    | `/healthz`                    | DB + scheduler status                        |
| GET    | `/tickers`                    | Universe summary                             |
| GET    | `/tickers/{t}/details`        | Fundamentals + last N days prices + preds    |
| GET    | `/predictions/latest`         | Top long / bottom short for the latest run   |
| GET    | `/runs`                       | Recent runs metadata                         |
| GET    | `/runs/{id}/equity`           | Equity curve for a specific run              |
| GET    | `/backtest/summary`           | Tearsheet payload                            |
| POST   | `/jobs/refresh`               | Trigger an on-demand pipeline run. Optional JSON body selects phase (1 or 5) and all config knobs — see [DEPLOYMENT.md](docs/DEPLOYMENT.md#trigger-a-refresh-from-cli) |
| GET    | `/jobs/{id}`                  | Job status                                   |

## Anti-patterns this project actively prevents

| What | How |
| ---- | --- |
| Train on test data | `WalkForwardSplit` with purge + embargo in **trading days** (default 25) |
| Same-day lookahead | Labels use `close[t+1] → close[t+1+h]`; engine realises the SAME window |
| Cost-free backtest | Default 6 bps/side; turnover charged on the day the trade clears |
| Survivorship | `members_on(date)` strict-boundary; pipeline `universe_sampling='random'` |
| Predicting on noise | Test rows with all-NaN features → NaN prediction (no base-rate fallback) |
| Cherry-picked window | Per-horizon diagnostics + yearly breakdown surfaced on the Backtest page |
| Optimising on test | Hyperparameters are static; tuning would require a held-out window — see `validation/stress.py` for the scaffold |

## Honest result on real data

100 randomly-sampled S&P names, 2018–2024, GBM ensemble across horizons
{1, 5, 21}:

| Horizon | Hit rate | IC mean | IC IR    |
| ------- | -------- | ------- | -------- |
| 1d      | 51.4%    | +0.002  | +0.24    |
| 5d      | 53.7%    | +0.024  | **+2.45** |
| 21d     | 55.6%    | -0.001  | −0.13    |

Strategy net: Sharpe **−1.3**, ann return **−10%**, max DD **−57%**.
The 5d horizon shows a real signal (IC IR > 2). The 21d horizon has no
signal at all. The equal-weight ensemble washes the 5d edge out, and
transaction costs do the rest. The infrastructure is honest; turning that
modest IC IR into a profitable strategy is the next phase of *research*,
not engineering.

## Tests

```
$ uv run pytest tests/ -v
============================= 45 passed in ~25s =============================
```

- `test_labels_no_leakage.py` — labels at t do not depend on prices at or before t
- `test_walk_forward.py` — embargo in trading days, no train/test overlap, default ≥ max-horizon
- `test_features.py` — features lag-safe, cross-sectional ranks bounded exactly to [-0.5, 0.5]
- `test_backtest_engine.py` — horizon-aware accumulation, cost timing, dollar neutrality
- `test_portfolio_construction.py` — vol-scaling, sector caps, turnover threshold, IC-IR weighting
- `test_stress.py` — holdout split, bootstrap Sharpe CI, sensitivity grid, regime breakdown
- `test_universe_html_parse.py` — Wikipedia parser + strict membership boundary
- `test_baseline_nan.py` — predictions are NaN when inputs are all-NaN
- `test_pipeline_integration.py` — end-to-end on synthetic noise with hit-rate canary
- `test_backend_api.py` — FastAPI endpoint contracts + snapshot round-trip

## Phase 5 + 6: improved pipeline + leakage audit (latest)

`scripts/run_phase5.py` plugs in: **IC-IR-weighted ensemble**, **vol-scaled
position sizing**, **30% sector caps**, **min trade threshold**, **untouched
2-year holdout**, **block-bootstrap Sharpe CI**, **Tier-2 features** (12-1
momentum, IVOL, β, max return, Amihud), **cross-asset regime features**
(VIX, term spread, USD, cross-sectional dispersion), and optional
**portfolio-level beta neutralisation vs SPY**.

`scripts/leakage_audit.py` runs the pipeline twice (as-is and with a hard
t-1 cutoff on features) to surface any same-day leakage in the label or
feature plumbing.

### Phase 6 finding: a real label leak was uncovered and fixed
The Phase 5 vol-scaled-return target was sharing `close[t]` with features
like `ret_1d` via its trailing-vol denominator (computed through close-of-t).
The model was getting most of its "predictive" power from the denominator,
not the forward return. **Fix:** the denominator is now strictly t-1 lagged.

### Honest real-data result (60 names, 2018–2024, h ∈ {1, 5})

| Metric                     | Phase 2 baseline | Phase 5 (leaky) | Phase 6 (leak-fixed + Tier-2 + regime) |
| -------------------------- | ---------------- | --------------- | ---------------------------------------- |
| DEV Sharpe (net)           | −1.30            | −0.04           | **−0.40**                                |
| HOLDOUT Sharpe (net)       | *not computed*   | −0.84           | **−0.95**                                |
| HOLDOUT 95% block-CI       | —                | [−1.60, −0.15]  | **[−1.70, −0.23]** — entirely negative   |

### Sensitivity grid result (8 combinations across k_per_side × cost × beta_neutralise)
- best HOLDOUT Sharpe: **−0.63**
- combos with HOLDOUT Sharpe > 0: **0 / 8**
- combos with bootstrap CI lower > 0: **0 / 8**

### Phase 7: HRP + triple-barrier + meta-labelling + per-feature audit + big-universe run

Added: Hierarchical Risk Parity portfolio construction (López de Prado 2016),
triple-barrier labels (Ch. 3), meta-labelling (Ch. 3.6), per-feature leakage
audit, and a defensive ±50% clip on daily returns to handle yfinance
data-quality glitches.

**Big universe (822 historical S&P 500 tickers, 2008-2024) result:**

| Configuration                    | HOLDOUT Sharpe | HOLDOUT 95% block-CI | HOLDOUT max DD |
| -------------------------------- | -------------- | -------------------- | -------------- |
| Phase 6 vol_scaled               | −0.78          | [−1.29, −0.31]       | −32%           |
| Phase 7 HRP                      | **−0.69**      | **[−1.12, −0.24]**   | **−29.6%**     |

Per-horizon HOLDOUT IC IR is now **positive on both horizons** (h=1d +0.69,
h=5d +0.49) on the big universe — the signal is real and survives the
honest out-of-sample test. But the strategy still loses money after costs.
HRP gave a small but real improvement over vol-scaled top-K.

**The bottom line after four phases of careful work: the strategy class
(free-data daily-bar cross-sectional L/S) does not have meaningful retail-
accessible edge in this period.** The infrastructure remains valuable for
honest experimentation. See [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md)
for the full Phase 7 write-up and Phase 8+ roadmap.

### Phase 8: meta-labelling + triple-barrier + per-feature audit + ranks_only

Added: meta-labelling gate (López de Prado Ch. 3.6) — secondary binary
classifier that gates trades on P(correct); triple-barrier labels as an
optional regression target; `ranks_only` flag to keep only cross-sectional
rank features; per-feature leakage audit which confirmed raw features
degrade ~100% under hard-cutoff while their ranked versions degrade only
15-50%.

**Phase 8 corrected real-data result** (150 current S&P names, 2014-2024,
h=5, HRP + meta-gating + ranks-only):

| Metric                | Phase 7 | Phase 8 (corrected) |
| --------------------- | ------- | ------------------- |
| HOLDOUT Sharpe        | −0.69   | **−0.16**           |
| HOLDOUT 95% block-CI  | [−1.12, −0.24] | **[−0.67, +0.29]** |
| HOLDOUT max DD        | −29.6%  | **−16.0%**          |

**For the first time in the project, the holdout 95% CI straddles zero**
rather than sitting entirely below. The point estimate is still negative
but no longer statistically distinguishable from random. Max drawdown was
halved. A sub-agent reviewer caught two critical bugs in the initial
Phase 8 wiring (double z-scoring of gated zeros, holdout meta trained on
gated dev); the corrected −0.16 above is the honest number.

**The bottom line after six phases: the strategy class does not produce
statistically significant *positive* risk-adjusted return on unseen data,
but it now produces results indistinguishable from zero rather than
significantly negative — which is genuine progress.**

### Phase 9: confidence-weighted sizing, walk-forward meta-CV, sector-conditional meta

Added: confidence-weighted signal sizing (`clip((P − floor) / (cap − floor),
0, 1)` instead of binary threshold), expanding-window walk-forward CV inside
the meta-gate, and per-sector meta classifiers (with cross-sectional columns
dropped to actually isolate sector-specific learning).

**Phase 9 honest real-data result** (150 names, 2014-2024, h=5, HRP, ranks-
only, **confidence mode + walk-forward 3 folds**):

| Metric                | Phase 8 (binary) | Phase 9 (confidence + WF-CV) |
| --------------------- | ---------------- | ---------------------------- |
| HOLDOUT Sharpe        | **−0.16**        | −0.57                        |
| HOLDOUT 95% block-CI  | **[−0.67, +0.29]** | [−1.03, −0.15]             |

**Phase 9 made things worse.** This is an honest finding: the binary gate's
hard refusal to trade was protecting against losses on a holdout where the
signal flipped sign. Confidence-weighted sizing trades more often (smaller
size each), and on a backwards signal more trades = more losses.

A sub-agent reviewer caught 4 critical/high bugs in the initial Phase 9
wiring (per-sector meta seeing cross-sectional features defeating isolation;
walk-forward concat without integrity check; NaN proba propagating in
confidence mode; HRP silent equal-weight fallback on bad covariance). All
fixed with regression tests. The corrected −0.57 above reflects the bug-
fixed pipeline.

**The bottom line after seven phases: more code rigor on top of an absent
signal does not manufacture signal. Phase 8 (binary meta + ranks-only)
remains the best honest result.**

```bash
uv run python scripts/run_phase5.py \
    --start 2018-01-01 \
    --n-tickers 100 \
    --horizons 1 5 \
    --weighting ic_ir \
    --position-sizing vol_scaled \
    --sector-cap 0.30 \
    --min-trade-threshold 0.005 \
    --holdout-years 2
```

## Watchlist

A small watchlist is seeded on first boot (HND.TO, HNU.TO, UNG, SPY, ^VIX)
and surfaced on the Screener page. You can add/remove arbitrary tickers via
`POST/DELETE /watchlist/{ticker}` (API key required). The model never trains
on or predicts for these — they're for context only, because leveraged ETFs,
crypto, and indices behave too differently from S&P 500 stocks for the
current model class to handle responsibly.

## News

Per-ticker headlines are pulled from yfinance (`Ticker.news`) and shown on
the ticker detail page. **The model does not consume these as features** —
free headline-level news is too sparse and ambiguous to score sentiment from
without a proper LLM API, and rolling sentiment scoring into features is the
single fastest way to introduce look-ahead bias. We surface them for human
context; if you later want event flags (earnings windows, 8-K filings) as
features, the research sub-agent's report in [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md)
explains how to do it without leaking the future.

## Out of scope (and why)

These are not in the project and won't be added without a corresponding
change to the constraints:

- **Intraday / real-time trading.** yfinance data is 15+ minutes delayed
  and rate-limited; honest intraday work requires paid feeds
  ($50–$500/month) and a broker API. The architecture would also change
  substantially (latency monitoring, different model class, real exec
  infra). See [`docs/CONCEPTS.md`](docs/CONCEPTS.md) §12.
- **Buying or selling actual stocks.** No broker integration. The backtest
  is a research output, not a trading system.
- **Point-in-time fundamentals.** yfinance `.info` is current-as-of-fetch.
  Using it historically leaks the future. We display fundamentals on the
  ticker page but never feed them to the model.
- **News sentiment.** Headlines alone are too noisy for a useful daily
  signal without paid sentiment APIs or a substantial NLP pipeline.

## Project documents

- [`docs/CONCEPTS.md`](docs/CONCEPTS.md) — **terminology & ideas for
  beginners.** Read first.
- [`docs/USAGE.md`](docs/USAGE.md) — install, run, interpret, operate,
  extend.
- [`docs/PROJECT_LOG.md`](docs/PROJECT_LOG.md) — full chronological log
  including the strategy-research sub-agent's report and Phase 5 results.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — Docker, Fly, Render, VM.
- [`docs/HANDOFF.md`](docs/HANDOFF.md) — resume protocol for new sessions.

## License

Personal / research use. **Not investment advice.**
