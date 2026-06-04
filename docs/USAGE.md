# Usage guide

Read [`CONCEPTS.md`](CONCEPTS.md) first if you've never built or evaluated a
stock prediction model before; it defines every term used here.

This document is your end-to-end manual: install → run → interpret →
operate → extend.

---

## 1. What you get when you run this

A self-contained app with three things:

| Surface          | What it does                                                                 |
| ---------------- | ----------------------------------------------------------------------------- |
| **CLI script**   | `scripts/run_phase1.py` runs the whole pipeline once and writes an HTML tearsheet to `reports/`. |
| **Web dashboard**| `scripts/serve.py` starts a FastAPI server with the SPA at `http://localhost:8000` showing the latest run, a screener, per-ticker pages, and a backtest tearsheet. |
| **API**          | The same FastAPI server exposes JSON endpoints (Swagger UI at `/docs`). Anyone can pull data; only holders of an API key can trigger a refresh. |

---

## 2. Installation

### Option A — Docker (simplest)

```bash
git clone https://github.com/hgdsraj/stock-predictor.git
cd stock-predictor

# Generate an API key so you can trigger refreshes via the dashboard
echo "STOCKPRED_API_KEY=$(openssl rand -hex 16)" > .env
# (docker-compose.yml does NOT auto-read .env by default; if you want the env
#  applied without copy-paste, add `env_file: .env` to the compose service.)

docker compose up --build
# open http://localhost:8000
```

### Option B — Local dev

Requires Python 3.11+, Node 20+, and `uv` (https://docs.astral.sh/uv/).

```bash
# Backend deps
uv sync --extra dev

# Frontend deps + build
cd web && npm ci && npm run build && cd ..

# Generate a development API key (only needed if you want POST /jobs/refresh)
export STOCKPRED_API_KEY="dev-only-do-not-reuse"

# Start the server
uv run python scripts/serve.py --host 127.0.0.1 --port 8000

# In another terminal, kick off a refresh (Phase 1, all defaults)
curl -X POST -H "X-API-Key: $STOCKPRED_API_KEY" http://127.0.0.1:8000/jobs/refresh

# Or run Phase 5 with a smaller universe
curl -X POST -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"phase": 5, "n_tickers": 50}' \
     http://127.0.0.1:8000/jobs/refresh
```

Open <http://127.0.0.1:8000> in a browser. The dashboard will be empty until
the refresh job completes (1–5 minutes depending on your universe size and
yfinance throttling).

### Option C — Frontend dev mode (HMR)

If you're working on the React code, run the backend (any of the above)
and *then*:

```bash
cd web && npm run dev
# open http://localhost:5173
```

Vite proxies `/api/*` paths to `http://127.0.0.1:8000`. Hot module reload
gives you instant UI feedback.

---

## 3. Your first run

Easiest path: skip the web stack and run the CLI directly. This proves the
pipeline works before you debug anything else.

```bash
uv run python scripts/run_phase1.py \
    --start 2018-01-01 \
    --end   2024-12-31 \
    --n-tickers 50 \
    --universe-sampling random \
    --model gbm \
    --horizons 1 5 21 \
    --k 10
```

Flags explained:

| Flag                 | Meaning                                                                    |
| -------------------- | -------------------------------------------------------------------------- |
| `--start`            | First trading date the model has access to.                                |
| `--end`              | Last trading date considered. Omit for "today".                            |
| `--n-tickers`        | How many of the historical S&P 500 to use. Smaller = faster, less robust.   |
| `--universe-sampling`| `random` (default; unbiased) / `current` (survivorship-biased; loud warning) / `first` (alphabetical; mildly biased). |
| `--model`            | `gbm` (LightGBM regressor; default) or `logistic` (transparent baseline).  |
| `--horizons 1 5 21`  | Which forecast horizons (trading days) to train and ensemble.              |
| `--k`                | Top/bottom-K per side for the long/short portfolio.                        |
| `--no-sector`        | Skip yfinance `.info` (faster cold-start, no sector features).             |
| `--refresh`          | Force re-download even if parquet caches exist.                            |
| `-v / --verbose`     | DEBUG logs.                                                                |

Expected runtime:
- 30 tickers, 5 years: ~30 seconds
- 100 tickers, 5 years: ~2 minutes
- 100 tickers, 15 years: ~6 minutes (the LightGBM training dominates)

The script:
1. Downloads and caches prices under `data/cache/prices/*.parquet`.
2. Trains a model per horizon with proper walk-forward CV.
3. Ensembles the predictions, builds a long/short portfolio.
4. Runs the backtest with realistic costs.
5. Writes an HTML tearsheet to `reports/run_gbm_h1-5-21_k10.html`.
6. Prints a summary to stdout.

Open the HTML in any browser to see the equity curve, drawdown chart, and
yearly metrics.

---

## 4. Interpreting the output

Sample stdout from a real run:

```
================================================================
 Pipeline complete
================================================================
  Universe size      : 60
  Feature matrix     : (105600, 36)

  Per-horizon OOS:
    h= 1d   hit=0.5146   ic_mean=+0.00233   ic_ir=+0.244
    h= 5d   hit=0.5368   ic_mean=+0.02382   ic_ir=+2.446
    h=21d   hit=0.5561   ic_mean=-0.00125   ic_ir=-0.131

  Backtest (ensemble):
    Ann return (net)   : -10.47%
    Ann vol            :  8.21%
    Sharpe (net)       : -1.305
    Max drawdown       : -57.01%
```

How to read this:

- **Per-horizon OOS** — out-of-sample diagnostics, one row per forecast
  horizon. The 5d horizon has IC IR +2.45, suggesting a real per-horizon
  signal (see CONCEPTS.md §3 on what IC IR means). The 21d horizon's
  IC IR ≈ 0, suggesting the model has no edge there.
- **Backtest (ensemble)** — what happens when you actually trade the
  ensemble score. The strategy lost money on this period and configuration.
- Note the gap between "per-horizon edge looks real" and "strategy lost
  money": equal-weighting horizons with mixed signal, daily-rebalancing on
  multi-day predictions, and not vol-scaling the positions destroys the
  edge. Phase 5 fixes these (see §6 below).

---

## 5. Using the dashboard

Once you've started the server and a refresh has completed:

### Home page

- **KPI tiles**: Sharpe, ann return, max DD, ann vol from the most recent
  backtest. Tiny percentages = small numbers, not "1.2%" being good or bad.
- **Equity curve**: cumulative growth of $1 over time. A flat or downward
  curve means the strategy lost or didn't make money.
- **Top movers**: today's highest-scored longs and lowest-scored shorts.
  Click any ticker to drill in.

### Screener

- Sortable, filterable table of every ticker in your latest run's universe.
- Search by ticker symbol or industry name; filter by GICS sector.
- Click any row to go to the ticker detail page.

### Ticker page

- Header: ticker, sector, industry, market cap, beta, P/E (TTM), dividend yield.
- Price chart: 2 years of adjusted close, with our model's prediction score
  overlaid as bars (positive = bullish, negative = bearish at that point in
  time). The score axis is on the right.
- Fundamentals card: 52-week high/low, short ratio, short % of float, P/E
  ratios, beta.
- About: business description.

**Important caveat**: the fundamentals card data comes from yfinance `.info`
which returns **current values**, not historical ones. If you're looking at
historical price data alongside "trailing P/E of 22", that P/E is from
today, not from the date you're looking at. We do not feed these values
into the model — they're for context only.

### Backtest page

- Eight KPI tiles for every standard metric.
- Equity curve (same as home, larger).
- Drawdown chart: how far below the running peak the strategy has fallen.
- Per-horizon diagnostics: hit rate, IC mean, IC IR for each horizon.
- Yearly table: annualised return, Sharpe, and trading days per year.

When you see negative numbers in green or positive numbers in red: that's
the colour-coded sign indicator. Green = good for that field (higher Sharpe,
lower drawdown). Red = bad.

---

## 6. Phase 5 improvements (the new pipeline mode)

The default `scripts/run_phase1.py` runs the **Phase 2** pipeline (basic
equal-weight ensemble). To get the improved Phase 5 pipeline:

```bash
uv run python scripts/run_phase5.py \
    --start 2018-01-01 \
    --n-tickers 100 \
    --horizons 1 5 \
    --k 10 \
    --weighting ic_ir \
    --position-sizing vol_scaled \
    --sector-cap 0.30 \
    --min-trade-threshold 0.005
```

What this changes vs Phase 1/2:

- `--weighting ic_ir`: weight horizons by their out-of-sample IC IR; horizons
  with IR ≤ 0 get zero weight (they were noise).
- `--position-sizing vol_scaled`: weight ∝ |score| / volatility, normalised
  per side.
- `--sector-cap 0.30`: no single sector can exceed 30% gross exposure.
- `--min-trade-threshold 0.005`: skip rebalances smaller than 0.5%.

These four changes together: dilute the bad horizons, control risk per
position, force diversification, suppress noise-trading. The result is
typically a much smaller drawdown and a less-negative (or sometimes
positive!) Sharpe.

The script also runs:
- A **bootstrap Sharpe confidence interval** so you know whether the
  result is statistically distinguishable from zero.
- An **out-of-sample holdout window** (the last 2 years are never seen
  during CV training).
- A **regime breakdown** of returns by VIX quintile so you can see whether
  the strategy works equally well in calm and stressed markets.

---

## 7. Running on production data

A few things to be aware of for a real run:

### Universe choice

- Default: 100 randomly-sampled historical S&P 500 names. Good for tutorial
  use; not enough breadth for a robust signal.
- For honest research: `--n-tickers None` to use *all* historical
  constituents (~700 names). Takes ~15 minutes per run.
- For "what's in the index right now": `--universe-sampling current` — but
  this is **survivorship biased**, the warning is printed loudly, and any
  positive backtest result should be discounted heavily.

### Cost assumptions

In `src/stockpred/config.py` you can adjust:

- `commission_bps`: default 1 bp.
- `spread_bps`: default 4 bps (half-spread per side; conservative for
  liquid S&P names).
- `slippage_bps`: default 1 bp.

Total round-trip: 12 bps. Realistic retail with a good broker. For
institutional add ~50% margin; for unrestricted retail (Robinhood) the
spread cost is higher.

### Refreshing

In production you have two options:

1. **APScheduler** (default): the backend runs a cron job every weekday at
   22:00 ET. No setup needed.
2. **External cron**: disable APScheduler via `STOCKPRED_DISABLE_SCHEDULER=1`,
   then have your hosting platform's cron hit `POST /jobs/refresh` once a
   day. Safer for multi-replica deployments.

---

## 8. API reference

All endpoints are documented interactively at `http://localhost:8000/docs`.
A quick cheat sheet:

```bash
# Health check
curl http://localhost:8000/healthz

# List all known tickers with last-price + sector
curl http://localhost:8000/tickers

# Per-ticker details (price history + predictions + fundamentals)
curl 'http://localhost:8000/tickers/AAPL/details?days=365'

# Today's top movers
curl 'http://localhost:8000/predictions/latest?top_k=10'

# Recent runs
curl 'http://localhost:8000/runs?limit=5'

# Equity curve for a specific run
curl http://localhost:8000/runs/3/equity

# Backtest summary
curl http://localhost:8000/backtest/summary

# --- Authenticated endpoints ---

# Trigger Phase 1 (basic GBM, top-k portfolio) — body is optional
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     http://localhost:8000/jobs/refresh

# Trigger Phase 5 (vol-scaled, regime-aware, sector-capped)
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"phase": 5}' \
     http://localhost:8000/jobs/refresh

# Custom run: Phase 5, smaller universe, force-refresh cached data
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
           "phase": 5,
           "n_tickers": 50,
           "start_date": "2015-01-01",
           "refresh_data": true,
           "horizons": [1, 5],
           "position_sizing": "vol_scaled",
           "use_regime_features": true
         }' \
     http://localhost:8000/jobs/refresh

# Tune GBM hyper-params
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "gbm",
           "gbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31}
         }' \
     http://localhost:8000/jobs/refresh

# All body fields and their defaults — see DEPLOYMENT.md for the full reference

# Poll job status
curl http://localhost:8000/jobs/<job-id>
```

Responses are always JSON. NaN and Infinity values are serialised as `null`
(RFC-7159 compliant).

---

## 9. Common operations

### Reset everything

```bash
# Wipes the database, parquet cache, generated reports
rm -rf data reports

# On next refresh, everything regenerates from scratch
```

### Backup the database

```bash
# Local
cp data/app.db ~/backups/app.db.$(date +%F)

# Docker
docker compose exec app sh -c "cp /app/data/app.db /app/data/app.db.bak"
docker cp stock-predictor:/app/data/app.db.bak ./
```

### Move to a bigger universe

Increase `--n-tickers`, the run takes longer but the signal is more robust.
The parquet cache means subsequent runs reuse downloaded prices.

### Debug a specific ticker

```bash
# Python REPL
uv run python
>>> from stockpred.data import prices
>>> df = prices.fetch_one("AAPL", start="2020-01-01")
>>> df.head()
```

### Inspect the database directly

```bash
sqlite3 data/app.db

sqlite> .tables
sqlite> .schema runs
sqlite> SELECT id, status, started_at, completed_at,
...            json_extract(summary_json, '$.metrics.sharpe') AS sharpe
...     FROM runs ORDER BY id DESC LIMIT 5;
```

---

## 10. Extending the project

### Add a new feature

1. Edit `src/stockpred/features/technical.py` (or create a new module under
   `features/`).
2. The function should return a long-form DataFrame indexed by `[date,
   ticker]` with feature columns. **Critical**: any rolling window must
   only look backward; mutating future prices must not change earlier
   feature values. Use the leakage test in `tests/test_features.py` as a
   template.
3. The pipeline auto-picks up all numeric columns, so no other wiring
   needed.

### Add a new model

1. Create a module under `src/stockpred/models/` with a function that
   takes `(X_train, y_train, X_valid, y_valid)` and returns a fitted
   estimator with a `.predict(X)` method.
2. Add the option to `pipeline._fit_and_predict_fold`.
3. Add the option to `PipelineConfig.model` and to the CLI `--model` flag
   in `scripts/run_phase1.py`.

### Add a new endpoint

1. Add a function under `register_routes` in
   `src/stockpred/backend/api.py`.
2. Add a Pydantic response model in `src/stockpred/backend/schemas.py`.
3. Add a fetch function in `src/stockpred/backend/store.py` if it needs
   DB access.
4. Add an entry in `web/src/api/client.ts` and a type in
   `web/src/api/types.ts`.

### Add a new page to the dashboard

1. Create `web/src/pages/Foo.tsx`.
2. Add the route in `web/src/main.tsx`.
3. Add a nav entry in `web/src/components/Layout.tsx`.

---

## 11. Troubleshooting

### "No tickers in the universe"

Wikipedia probably blocked your request. Wait 30 seconds and retry. If
persistent, edit the User-Agent in `src/stockpred/data/universe.py`.

### yfinance fails for half my tickers

This happens. yfinance is unofficial scraping and silently 429s. The
pipeline tolerates partial failures — names with no data are just dropped
from the universe for that run. If most fail, your IP is rate-limited;
wait an hour.

### The pipeline is very slow

- Profile: `time uv run python scripts/run_phase1.py --n-tickers 30 ...`.
- LightGBM dominates with many features; set `--model logistic` for
  10× faster runs while debugging.
- Parquet caches are read once and reused — first run downloads everything,
  subsequent runs are much faster.

### The dashboard says "loading" forever

- Check the browser console for errors.
- Hit `/healthz` directly: `curl http://localhost:8000/healthz`. Should
  return `{"status":"ok","db":"ok","scheduler":...}`.
- Check that a refresh has actually completed:
  `curl http://localhost:8000/runs`. If empty, kick one off.

### "X-API-Key required" when I POST /jobs/refresh

You haven't set `STOCKPRED_API_KEY`. Either set it and re-launch the
server, or accept that writes are intentionally disabled.

### Tests fail after I changed the engine

Re-run with `-v`: `uv run pytest tests/test_backtest_engine.py -v`. Then
read the assertion message — the tests are designed to fail loudly on
exactly the leakage / cost / alignment bugs the project is trying to
prevent.

### The frontend build fails

```bash
cd web
rm -rf node_modules dist
npm ci
npm run build
```

If it still fails, check Node version: `node --version` should be 20+.

---

## 12. Performance benchmarks

Rough timings on a 2024-vintage laptop (Apple M2 / 16 GB):

| Operation                                  | Time     |
| ------------------------------------------ | -------- |
| `uv sync --extra dev`                      | 60 s     |
| `npm ci && npm run build`                  | 90 s     |
| First pipeline run, 30 tickers × 5 years   | 30 s     |
| Subsequent run (cached prices)             | 8 s      |
| First pipeline run, 100 tickers × 5 years  | 2 m      |
| First pipeline run, 500 tickers × 15 years | 15 m     |
| Container build (cold)                     | 5 m      |
| Container build (warm)                     | 45 s     |
| Cold start of `serve.py`                   | 2 s      |
| HTTP latency (any read endpoint)           | < 50 ms  |

---

## 13. Where to read more

- [`CONCEPTS.md`](CONCEPTS.md) — every metric, term, and design decision
  explained.
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker, Fly.io, Render, VM with Caddy.
- [`PROJECT_LOG.md`](PROJECT_LOG.md) — chronological history of every
  change and why.
- [`HANDOFF.md`](HANDOFF.md) — protocol for a future agent picking the
  project up mid-stream.
