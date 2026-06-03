# Project log — stock-predictor

Chronological log of decisions, what was built, and what the real backtest
produced. Lives in-repo so it travels with the code.

---

## Session 1 — Phase 1: foundation

### Ask
> "build the best stock predictor in history"

Scoped down to: directional cross-sectional forecaster, serious scope,
fully portable, free data only.

### Built
- `pyproject.toml` with public PyPI pinned (no corp mirrors).
- `data/` — Wikipedia S&P 500 historical constituents, yfinance prices with
  parquet caching, FRED macro loader.
- `features/` — lag-safe technicals (returns, vol, RSI, MACD, BB, etc.) +
  cross-sectional ranks.
- `labels.py` — forward log returns + binary direction per horizon.
- `validation/walk_forward.py` — expanding window with purge + embargo.
- `validation/metrics.py` — IC / IC IR / Sharpe / Sortino / max DD / Calmar.
- `models/baseline.py` — impute → scale → logistic regression.
- `models/gbm.py` — LightGBM scaffold.
- `backtest/portfolio.py` — top-K long/short.
- `backtest/engine.py` — vectorised backtester.
- `reports/tearsheet.py` — self-contained HTML report.
- `pipeline.py` — end-to-end driver.
- `scripts/run_phase1.py` — CLI.
- 16 tests covering leakage canary, walk-forward correctness, backtest
  semantics, features, end-to-end on synthetic data.

### Phase 1 result on real data
- 100 names, 2018-2024, horizon=1: hit 51.2%, IC IR +0.44, Sharpe −0.81.
- Honest, infrastructure-correct, signal too weak to overcome costs.

---

## Session 2 — Code review + bug fixes + Phase 2/3/4 + backend + frontend + deployment

### Sub-agents dispatched in parallel
1. **Research** — wrote a detailed report on free fundamentals data sources
   (yfinance reliability fields, FINRA short interest flat files, SEC EDGAR
   endpoints with rate limits, NASDAQ earnings calendar). Inlined into the
   handoff doc.
2. **Stress-test reviewer** — comprehensive audit of all Phase 1 modules.
   Surfaced 3 CRITICAL bugs, 6 HIGH/MEDIUM bugs, 8 LOW/NIT issues.

### CRITICAL bugs surfaced by review and FIXED with regression tests
- **C1** — `run_backtest` was not horizon-aware. For horizons > 1, only one
  day of the h-day forward window was realised, while the model was trained
  on h-day cumulative returns. Engine rewritten with `horizon`, `trade_lag`,
  and cadence enforcement. New parametrised test asserts cumulative return
  over the held window equals the expected h-day return exactly.
- **C2** — `WalkForwardSplit` embargo was in **calendar** days, but horizons
  are in **trading** days. For h=21 with embargo=10, ~14 days of label
  leakage actually occurred. Switched to positional offsets into the sorted
  trading-day index. Default embargo bumped to 25.
- **C3** — `add_cross_sectional_ranks` was approximately bounded, not
  exactly. Switched to `(rank - 1) / (n - 1) - 0.5` so min/max are exactly
  ±0.5 with no floating-point slop. Test asserts the exact bounds.

### HIGH/MEDIUM bugs also FIXED with regression tests
- **H2** — costs now charged on the day the trade clears (signal day +
  trade_lag), not on the signal day.
- **H3** — `members_on` uses strict `>` on end_date (ticker removed on D is
  not in the index at close of D).
- **H5** — ADV feature renamed to `adv_proxy_21` with docstring explaining
  that `adj_close × volume` is not true dollar volume.
- **M3** — baseline returns NaN for test rows that were entirely NaN in the
  original `X_test` (no silent base-rate fallback).
- **M4** — `select_universe` no longer silently picks current-only
  constituents. New `universe_sampling` knob: `random` (default, unbiased,
  deterministic seed), `current` (loud SURVIVORSHIP warning), `first`
  (alphabetical, mildly biased, transparent).
- **M6** — tearsheet yearly table uses per-column formatters; Sharpe no
  longer rendered as a percentage.
- **L6** — equity/drawdown charts use `returns.dropna()` so NaN stretches
  don't render as flat zero-drawdown.
- **L7** — fundamentals loader rate-limits between submissions (the previous
  sleep was after-the-fact and ineffective).

### Phase 2 — model improvements
- `data/fundamentals.py` — yfinance `.info` per-ticker caching, parquet store
  at `data/cache/fundamentals.parquet`.
- `features/cross_sectional.py` — `neutralise_by_sector` (sector-relative
  cross-section) and `add_sector_dummies` (one-hot sector membership).
- `labels.py` — `compute_vol_scaled_forward_returns`: forward log return
  divided by trailing realised vol (lag-safe). `long_labels` emits
  `fwd_vs_{h}` per horizon by default.
- `pipeline.py` rewrite: `PipelineConfig.horizons` (plural), `model`
  ({'gbm' | 'logistic'}), `use_sector_features`. Per-horizon walk-forward
  training + cross-sectional z-scored ensemble.
- `scripts/run_phase1.py` rewritten: `--horizons 1 5 21 --model gbm
  --no-sector --universe-sampling random`.

### Phase 2 real-data result (60 names, 2018-2024, GBM h=1/5/21)
| Horizon | Hit rate | IC mean | IC IR    |
| ------- | -------- | ------- | -------- |
| 1d      | 51.5%    | +0.002  | +0.24    |
| 5d      | 53.7%    | +0.024  | **+2.45** |
| 21d     | 55.6%    | −0.001  | −0.13    |

Strategy Sharpe **−1.3**. The 5d signal is real; the 1d and 21d horizons
wash it out at equal ensemble weights; transaction costs do the rest.

### Phase 3 — portfolio construction
- `backtest/portfolio.py`:
  - `vol_scaled_weights` — signal × inverse-vol, normalised per side.
  - `apply_sector_caps` — shrink any sector's gross to a cap.
  - `apply_min_trade_threshold` — suppress small day-to-day rebalances.
  - `ic_ir_weighted_ensemble` — weight horizons by their IC IR; horizons with
    IC IR ≤ 0 get zero weight; loud fallback to equal weights when all are
    negative.

### Phase 4 — stress tests
- `validation/stress.py`:
  - `holdout_split_dates` — chronological dev/holdout partition.
  - `bootstrap_sharpe` — i.i.d. resampling CI for annualised Sharpe.
  - `sensitivity_grid` — run a callable across a grid of params.
  - `vix_regime`, `spy_regime` — regime labels for breakdown.
  - `regime_breakdown` — per-regime mean/std/Sharpe/hit/ann_return.

### Backend
- `backend/db.py` — SQLite engine with WAL + FK pragmas, `session_scope`.
- `backend/models.py` — ORM: `Run`, `Prediction`, `PriceBar`, `Fundamental`,
  `EquitySample`.
- `backend/store.py` — repository pattern with SQLite `ON CONFLICT` upserts.
- `backend/snapshot.py` — bridges pipeline output to DB (predictions,
  equity, run summary; optionally refreshes prices + fundamentals).
- `backend/jobs.py` — APScheduler with daily cron + weekly cleanup;
  on-demand jobs tracked in memory.
- `backend/schemas.py` — Pydantic v2 response models.
- `backend/api.py` — FastAPI: 10 routes covering health, tickers, ticker
  details, predictions, runs, equity curves, backtest summary, job control.
  Static-mounts a built SPA from `web/dist/` at `/`.
- `scripts/serve.py` — uvicorn entrypoint.
- Backend tests round-trip predictions/prices/fundamentals through the API.

### Frontend
React 18 + Vite + TypeScript + Tailwind + shadcn-style components +
TanStack Query/Table + Recharts + Lucide icons. 4 pages:
- `Home` — KPI tiles, equity curve, top longs/shorts.
- `Screener` — filterable, sortable table over the universe.
- `Ticker/:t` — price chart with prediction overlay, fundamentals card,
  business summary.
- `Backtest` — KPI tiles, equity, drawdown, per-horizon diagnostics, yearly
  table.

Built and tested locally (Node 20):
```
dist/index.html                   0.71 kB │ gzip:   0.42 kB
dist/assets/index-…css           13.00 kB │ gzip:   3.55 kB
dist/assets/index-…js           716.65 kB │ gzip: 204.80 kB
✓ built in 7.23s
```

### Deployment
- `Dockerfile` — multi-stage: Node 20 builds frontend, Python 3.12 slim runs
  uvicorn and serves both API and SPA from one process.
- `.dockerignore` — keeps the build context small.
- `docker-compose.yml` — single service, persistent `./data` volume.
- `docs/DEPLOYMENT.md` — local Docker, manual, Fly.io with `fly.toml`,
  Render with Disk + Cron Job, generic VM with Caddy + Let's Encrypt.
  Environment variable matrix, backup recipes, security notes.

### Tests at end of session
```
$ uv run pytest tests/
45 passed in ~25s
```
- 7 backtest engine (horizon-aware, cost timing, cadence, dollar-neutral)
- 1 baseline NaN prediction
- 5 features (lag-safe, exact-bounded ranks, single-obs day)
- 3 labels no-leakage
- 1 pipeline integration (synthetic noise canary)
- 6 portfolio construction (top-K, vol-scaled, sector caps, threshold, IC IR)
- 6 stress (holdout, bootstrap CI, sensitivity grid, regime breakdown)
- 3 universe HTML + strict boundary
- 6 walk-forward (trading-day embargo, default ≥ max horizon, contract)
- 7 backend API (health, list, details, snapshot round-trip, etc.)

### What is intentionally NOT in this session
- A real trading link to a broker.
- Real-time data. yfinance + Wikipedia + FRED on a daily cadence is the
  free-data ceiling.
- A "winning" strategy. The current GBM ensemble loses to cash on the
  real-data backtest. The infra honestly reports it. Improving the model
  is Phase 5+ research.

### Project layout (final)
```
stock-predictor/
├── src/stockpred/
│   ├── backend/        # db, models, store, snapshot, jobs, schemas, api
│   ├── backtest/       # engine.py (horizon-aware), portfolio.py
│   ├── data/           # universe, prices, macro, fundamentals
│   ├── features/       # technical.py, cross_sectional.py
│   ├── models/         # baseline.py, gbm.py
│   ├── reports/        # tearsheet.py
│   ├── validation/     # walk_forward.py, metrics.py, stress.py
│   ├── config.py
│   ├── labels.py
│   └── pipeline.py
├── web/                # Vite + React + TS frontend
│   ├── src/api/        # client + types
│   ├── src/components/ # Layout, ThemeProvider, ui/
│   ├── src/lib/        # cn, format
│   ├── src/pages/      # Home, Screener, Ticker, Backtest
│   ├── package.json
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── index.html
├── scripts/            # run_phase1.py, serve.py
├── tests/              # 45 tests
├── docs/               # PROJECT_LOG, DEPLOYMENT, HANDOFF
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Next steps (Phase 5+)

Honest signal research, not engineering:
1. Use `validation/stress.py` to bootstrap-CI the Phase 2 backtest. If the
   point-estimate Sharpe sits comfortably inside a CI that includes 0, stop
   pretending there's a strategy.
2. Drop the 21d horizon (IC IR ≈ 0) from the ensemble. Re-run with just
   {1, 5} weighted by IC IR.
3. Try sector neutralisation flag turned ON; current real-data run had it
   off because the fundamentals fetch on cold-cache is slow.
4. Run a sensitivity grid (`scripts/run_phase1.py` × grid of `cost_bps`,
   `k_per_side`). If results swing wildly with a 2 → 6 bps cost change, the
   "edge" is overfitting.
5. (Long shot) news/sentiment via free LLM API + EDGAR 8-K filings.
