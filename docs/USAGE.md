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

Vite proxies API calls (`/healthz`, `/tickers`, `/predictions`, `/runs`,
`/backtest`, `/jobs`) to `http://127.0.0.1:8000` (see
[`web/vite.config.ts`](../web/vite.config.ts)). Hot module reload gives you
instant UI feedback. The backend it proxies to can hold real data *or*
synthetic data — see the next section.

### Local testing with synthetic data

You usually don't want to wait for a real pipeline run (and hammer yfinance)
just to see the dashboard or work on the frontend. `scripts/seed_synthetic.py`
fills a **separate** SQLite DB (`data/local_test.db`, so your real
`data/app.db` is untouched) with randomly-generated prices, fundamentals,
predictions, an equity curve, and a finished run — shaped exactly like a real
pipeline result, so every page and every endpoint has data to show.

```bash
# Seed the synthetic DB and start the backend on http://127.0.0.1:8000
uv run python scripts/seed_synthetic.py --serve
```

Open <http://127.0.0.1:8000> for the built dashboard (if you've run
`npm run build`), or for live frontend work start Vite against it:

```bash
# Terminal 1 — backend serving the synthetic DB
uv run python scripts/seed_synthetic.py --serve

# Terminal 2 — Vite dev server with hot reload, proxying to :8000
cd web && npm install && npm run dev
# open http://127.0.0.1:5173
```

Useful flags:

| Flag           | Default               | Meaning                                            |
| -------------- | --------------------- | -------------------------------------------------- |
| `--db`         | `data/local_test.db`  | Where to write the synthetic SQLite DB.            |
| `--n-tickers`  | `80`                  | Number of synthetic tickers to generate.           |
| `--days`       | `500`                 | Trading days of price history.                     |
| `--seed`       | `42`                  | RNG seed (same seed → identical data).             |
| `--reset`      | off                   | Delete the synthetic DB (and `-wal`/`-shm`) first. |
| `--serve`      | off                   | Launch the backend after seeding.                  |
| `--host`/`--port` | `127.0.0.1`/`8000` | Server bind address (with `--serve`).              |

The generated tickers (e.g. `ZQ07`) are obviously fake and every number is
random — nothing here implies any real-world prediction. To re-seed cleanly,
add `--reset`. To seed without serving (e.g. to point your own server at it):

```bash
uv run python scripts/seed_synthetic.py --reset --n-tickers 120 --days 750
STOCKPRED_DB=data/local_test.db STOCKPRED_DISABLE_SCHEDULER=1 \
    uv run python scripts/serve.py
```

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

## 6b. Phase 6–11 improvements (advanced flags)

Phases 6–11 layer in research-grade portfolio tooling. None of them is
guaranteed to flip HOLDOUT Sharpe above zero (see `docs/continue.md`
phase ledger for the honest numbers), but each adds a tool you can
combine with the Phase 5 baseline.

### Phase 6 — Tier-2 features, regime features, beta neutralisation, block bootstrap

```bash
uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 \
    --beta-neutralise \
    --bootstrap-method block
```

- `--beta-neutralise`: subtract SPY exposure from the daily long-short
  PnL so the strategy is a pure-alpha bet.
- `--bootstrap-method block` (default): the bootstrap re-samples in
  blocks of size = horizon to preserve autocorrelation from overlapping
  positions. The IID alternative (`--bootstrap-method iid`) over-states
  significance for overlapping-horizon strategies.
- `--no-tier2 / --no-regime / --no-sector`: turn OFF the Tier-2 features
  (12-1 momentum, IVOL, β, Amihud illiquidity), regime broadcasts (VIX
  quintile, term-spread quintile), and sector dummies respectively. The
  Phase 8 best config has all three OFF.

### Phase 7 — HRP portfolio + triple-barrier label scaffold

```bash
uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 \
    --position-sizing hrp
```

- `--position-sizing hrp`: Hierarchical Risk Parity weights (López de
  Prado Ch. 16). Splits the long sleeve and short sleeve into clusters
  by correlation and allocates inverse-variance within each cluster.
- The backtest engine clips per-name `pct_change` at ±50% to defend
  against delisted-ticker price corruptions from yfinance (cosmetic
  warning spam in the log when this fires).

### Phase 8 — meta-labelling + ranks_only + triple-barrier labels

```bash
uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 \
    --meta-labelling --meta-threshold 0.55 \
    --ranks-only \
    [--triple-barrier --tb-k-sigma 2.0]
```

- `--meta-labelling`: train a secondary binary classifier per fold
  predicting `P(primary score has correct sign)`. Gate the primary
  score: zero-out rows where `P < --meta-threshold`. Reduces turnover
  and improves precision at the cost of recall.
- `--meta-threshold 0.55`: gate threshold (default 0.55).
- `--ranks-only`: drop the raw feature columns and keep only the
  cross-sectional `_rank` columns (plus sector dummies / regime
  broadcasts if enabled). Per-feature audit on the 150-name × 11-yr
  universe shows raw columns degrade ~100% under hard-cutoff while
  the ranked versions degrade only 15–50%.
- `--triple-barrier`: switch from the simple forward-return label to
  the López de Prado triple-barrier signed return per horizon. Bounded
  by construction (clipped at ±`tb-k-sigma` units of trailing vol).

### Phase 9 — confidence-weighted sizing + walk-forward meta-CV

```bash
uv run python scripts/run_phase5.py [...same as Phase 8...] \
    --meta-mode confidence --meta-conf-floor 0.60 --meta-conf-cap 1.0 \
    --meta-walk-forward-folds 3 \
    [--meta-per-sector]
```

- `--meta-mode confidence`: scale signal by
  `clip((P-floor)/(cap-floor), 0, 1)` instead of a hard binary gate.
- `--meta-conf-floor 0.60`: the Phase 10 sweet spot. **Do NOT use the
  default `0.5`** — Phase 10 swept floors {0.50…0.75} and 0.50
  reproducibly underperforms binary by ~0.4 Sharpe; 0.60 has the best
  point estimate.
- `--meta-walk-forward-folds 3`: K folds of expanding-window
  meta-classifier CV. K=1 reproduces the Phase 8 single-pass behaviour.
- `--meta-per-sector`: one meta classifier per sector (requires
  fundamentals successfully loaded; falls back to global meta if not).

### Phase 10 — confidence-floor sweep

```bash
uv run python scripts/phase10_conf_floor_sweep.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 \
    --floors 0.50 0.55 0.60 0.65 0.70 0.75
```

Runs the Phase 8 best config 7 times: once with `--meta-mode binary`
(baseline) and once per floor with `--meta-mode confidence`. Reports
holdout Sharpe + 95% block-bootstrap CI per run. Output:
`reports/phase10_conf_floor_sweep_<start>_<end>_n<N>.csv`.

### Phase 11 — feature pruning by per-feature audit

```bash
# Step 1: re-build the per-feature audit on your universe (~30 min)
uv run python scripts/per_feature_audit.py \
    --start 2014-01-01 --end 2024-12-31 \
    --n-tickers 150 --horizon 5 --top-n 20

# Step 2: run the 3-config sweep (~10 min)
uv run python scripts/phase11_feature_pruning.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150
```

The driver reads `reports/per_feature_audit.csv` and runs three configs
on top of the Phase 8 best config:
- baseline (no pruning)
- drop bottom 25% by `pct_drop` (least same-day work — noise candidates)
- drop top 25% by `pct_drop` (most same-day work — sanity check; should
  be worse than baseline)

The pipeline now exposes a `PipelineV5Config.feature_exclude:
tuple[str, ...]` field so future phases can chain pruning with new
features.

---

## 6c. News features (Phase 12–14)

### Phase 12 — SEC EDGAR 8-K event flags  *(IMPLEMENTED)*

8-K is the SEC form a public company files when a "material event"
happens (CEO change, earnings pre-release, M&A, etc.). Free, full
historical coverage from 2001, no API key.

```bash
# Quick smoke test (40 tickers x 3 years) — completes in ~50 sec
uv run python scripts/run_phase5.py \
    --start 2022-01-01 --end 2024-12-31 --n-tickers 40 --horizons 5 \
    --weighting equal --position-sizing hrp \
    --k-pct 0.15 --sector-cap 0.30 --min-trade-threshold 0.005 \
    --holdout-years 1 --no-sector --no-regime --no-tier2 \
    --universe-sampling current --bootstrap-method block \
    --meta-labelling --meta-threshold 0.55 --ranks-only \
    --edgar-events --bootstrap-n 50

# Production (150 tickers x 11 years)
uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 \
    [...same flags...] \
    --edgar-events
```

Adds 4 features per (date, ticker):
- `edgar_has_8k`        (int8)  1 if 8-K filed on this trading day
- `edgar_count_8k_5d`   (int16) rolling 5-trading-day count
- `edgar_count_8k_21d`  (int16) rolling 21-trading-day count
- `edgar_count_8k_63d`  (int16) rolling 63-trading-day count

The `edgar_` prefix is recognised by `--ranks-only` so these
discrete event-count features are kept (they have no `_rank`
counterpart by design — counts don't need cross-sectional ranking).

**SEC compliance is mandatory.** SEC requires a User-Agent header
identifying you and rate-limits at 10 req/sec. We default to:

```bash
export EDGAR_USER_AGENT="Your Name your-email@example.com"
```

If you don't set it, requests use a generic UA `stock-predictor/0.2
(raj.axisos@gmail.com)` which SEC may rate-limit harder if many
unrelated users share it.

**Equivalent `curl` recipes** (if you want to fetch caches manually
before running the pipeline):

```bash
UA="Your Name your-email@example.com"
mkdir -p data/cache/edgar

# Ticker -> CIK mapping (one-shot, ~3 MB)
curl -A "$UA" -o data/cache/edgar/ticker_to_cik.json \
    https://www.sec.gov/files/company_tickers.json

# Per-quarter form index (~2 MB each, one per qtr from 2014 onward)
for YEAR in 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
  for QTR in 1 2 3 4; do
    curl -A "$UA" \
        -o data/cache/edgar/form_${YEAR}Q${QTR}.idx \
        "https://www.sec.gov/Archives/edgar/full-index/${YEAR}/QTR${QTR}/form.idx"
    sleep 0.2   # be nice to SEC
  done
done
```

(The pipeline will re-parse these into per-quarter parquet caches at
`data/cache/edgar/8k_<YYYY>Q<n>.parquet` on the first `--edgar-events`
run.)

**Storage**: ~30 KB per quarter parquet × 44 quarters ≈ 1.3 MB on
disk. RAM impact at backtest: <100 MB peak for 150 tickers × 11 yr.

**Honest expectations**: 8-K events are sparse (most companies file
2-5 per year). Sharpe lift, if any, will be small. Run the full
production sweep + bootstrap CI before claiming a result.

**Phase 12 production result** (150 tickers × 11yr): hold Sharpe **−0.376**
vs baseline −0.158. EDGAR raw counts HURT the strategy. Use
`--edgar-items` (Phase 13) instead.

### Phase 13 — SEC EDGAR 8-K item-code flags  *(IMPLEMENTED)*

Same SEC data as Phase 12, but extracts the per-filing **item codes**
(item 2.02 = earnings, item 5.02 = CEO change, item 8.01 = M&A, etc.)
which carry directional information that raw counts don't.

```bash
# Quick smoke test (3 min)
uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 --horizons 5 \
    --weighting equal --position-sizing hrp \
    --k-pct 0.15 --sector-cap 0.30 --min-trade-threshold 0.005 \
    --holdout-years 2 --no-sector --no-regime --no-tier2 \
    --universe-sampling current --bootstrap-method block \
    --meta-labelling --meta-threshold 0.55 --ranks-only \
    --edgar-items
```

Adds 15 features per (date, ticker) — 5 item families × 3 windows:

- `edgaritem_earnings_today` (int8); `edgaritem_earnings_21d`, `_63d` (int16)
- `edgaritem_ceo_change_today`, `_21d`, `_63d`
- `edgaritem_ma_today`, `_21d`, `_63d` (covers item 1.01 + 2.01 + 8.01)
- `edgaritem_guidance_today`, `_21d`, `_63d`
- `edgaritem_going_concern_today`, `_21d`, `_63d`

Uses SEC's per-company submissions JSON endpoint:

```bash
# One-off curl per ticker (Python pipeline does this automatically):
curl -A "Your Name your-email@example.com" \
    "https://data.sec.gov/submissions/CIK0000320193.json" \
    | jq '.filings.recent | {form, filingDate, items}' \
    | head -20
```

**Honest result on production smoke** (150 tickers × 11 yr): hold Sharpe
**+0.173** with 95% CI **[−0.32, +0.58]**. The point estimate is
positive AND the holdout drawdown is the smallest across all phases
(−8.2% vs −16% baseline). CI still straddles zero so this is not
yet a statistically significant edge, but it's the closest the project
has come.

**Default recommendation**: enable `--edgar-items`, do NOT enable
`--edgar-events` (Phase 12 raw counts hurt the strategy).

### Phase 13 — GDELT 2.0 tone + theme counts

GDELT 2.0 publishes a global event/sentiment dataset every 15 minutes,
free, no API key. Coverage from 2015-02-18. We aggregate per-ticker
daily features: average tone, mention count, and theme counts for a
small set of finance themes (EARNINGS, REGULATION, ACQUISITION).

**Required `curl` for the master file list**:

```bash
# Master list of every 15-min slice (~5 MB text file, updated daily)
curl -o data/cache/gdelt/masterfilelist.txt \
    http://data.gdeltproject.org/gdeltv2/masterfilelist.txt
```

Memory caution: each daily aggregate is ~50 MB compressed; the full
2015–2024 raw dump is ~18 GB. The agent will stream + filter via
`pd.read_csv(url, chunksize=200_000)` and keep ONLY rows mentioning
S&P 500 tickers, persisting the result as gzipped parquet (~1 GB
final size).

Planned CLI flags:

- `--gdelt-tone`: enable GDELT daily tone + count features.
- `--gdelt-cache-dir data/cache/gdelt`: where the parquet caches live.

### Phase 14 — FinBERT live-mode sentiment (dashboard ONLY)

ProsusAI/FinBERT is a 110-M-parameter financial-domain BERT classifier
(positive / neutral / negative) fine-tuned on Financial PhraseBank.
We use it ONLY for the dashboard's per-ticker news panel — never as a
backtest feature, because the underlying yfinance news source only
has ~30 days of history (using it as a feature would create
catastrophic walk-forward bias).

**Required model download (one-shot, ~440 MB)**:

```bash
# Hugging Face CLI (preferred)
pip install huggingface_hub
huggingface-cli download ProsusAI/finbert --local-dir models/finbert

# OR direct curl (skip CLI)
mkdir -p models/finbert
for f in config.json vocab.txt pytorch_model.bin tokenizer.json; do
    curl -L -o models/finbert/$f \
        https://huggingface.co/ProsusAI/finbert/resolve/main/$f
done
```

Then `pip install transformers torch` (~1.5 GB). The agent will gate
the import inside `news.py` so the rest of the pipeline doesn't fail
if these heavy deps aren't installed.

Planned env vars:

- `FINBERT_MODEL_DIR=models/finbert` (path to the local model)
- `FINBERT_BATCH_SIZE=32` (CPU inference; lower if OOMing)
- `FINBERT_ENABLED=true` (off by default to keep the dashboard
  lightweight when the model isn't downloaded)

---

## 6d. Hyperparameter search (find the best config automatically)

The search is available in **two ways**:

| Mode | Use when |
| ---- | -------- |
| **Web UI** (`/hypersearch` page) | Server is running; want to queue, launch, and monitor via the dashboard |
| **CLI** (`scripts/run_hypersearch.py`) | Headless, local dev, or scheduled batch run |

Both use the same `stockpred.hypersearch` core logic and Optuna TPE sampler.
Results from the web UI are stored in the database and visible under
`GET /hypersearch/runs`.

### Web UI

1. Start the server (`uv run python scripts/serve.py`)
2. Open **http://localhost:8000/hypersearch**
3. Click **New Search** → configure trials, tickers, date range
4. The job is queued — click **Launch** and enter `STOCKPRED_PW`
5. Logs stream live; the trial table updates after every completed trial
6. When done, expand any row to see the top-10 table, best config JSON, and
   the exact curl command to promote those parameters to a production pipeline run

### CLI quick start

```bash
# Smoke test: 20 trials, ~25 tickers, ~30-80 min total
uv run python scripts/run_hypersearch.py --n-trials 20

# Recommended: 50 trials on 25 tickers (~2-4 hours)
uv run python scripts/run_hypersearch.py

# Longer overnight run for higher confidence
uv run python scripts/run_hypersearch.py --n-trials 100 --n-tickers 40
```

### What it searches

| Group | Parameters |
| ----- | ---------- |
| Portfolio | `position_sizing`, `k_per_side_pct`, `leverage_per_side`, `sector_cap_gross`, `min_trade_threshold` |
| Signal | `horizons`, `ensemble_weighting` |
| Features | `use_tier2_features`, `use_regime_features`, `use_sector_features`, `ranks_only`, `beta_neutralise` |
| Meta-labelling | `use_meta_labelling`, `meta_threshold`, `meta_mode`, `meta_conf_floor` (conditional) |
| GBM | `num_leaves`, `learning_rate`, `n_estimators` |
| CV | `train_years` |

### CLI flags

| Flag | Default | Meaning |
| ---- | ------- | ------- |
| `--n-trials` | `50` | Number of Optuna trials |
| `--n-tickers` | `25` | Universe size per trial (25 ≈ 2-4 min/trial) |
| `--start` | `2015-01-01` | History start date |
| `--holdout-years` | `2` | Years held out (never seen during tuning) |
| `--bootstrap-n` | `50` | Bootstrap samples (50=fast; 500=honest) |
| `--server-url` | `http://localhost:8000` | Used to generate the curl command in the report |
| `--storage` | none | SQLite/PostgreSQL URL for persistence/parallel workers |
| `--study-name` | auto | Name for the Optuna study |

### Outputs

After every run the script writes two files to `reports/`:

| File | Contents |
| ---- | -------- |
| `hypersearch_<name>_n<N>_<year>.csv` | All trials sorted by holdout Sharpe |
| `hypersearch_<name>_n<N>_<year>.md` | Human-readable report with top-10 table, honest CI interpretation, best config JSON, and a ready-to-paste `curl` command |

### Reading the results honestly

The markdown report includes a **95% block-bootstrap Sharpe CI** for every
trial. A Sharpe CI that straddles zero means the result is statistically
indistinguishable from noise. Only configs where the **CI lower bound > 0**
have a statistically detectable edge.

After tuning, run the best config at full scale to validate:

```bash
uv run python scripts/run_phase5.py \
    --n-tickers 150 --start 2015-01-01 \
    <best params from the report> \
    --bootstrap-n 500
```

### Parallel / resumed studies

Pass `--storage` to persist the study to SQLite or PostgreSQL. Multiple
terminals sharing the same storage and study name run trials independently:

```bash
# Terminal 1
uv run python scripts/run_hypersearch.py \
    --storage sqlite:///reports/hypersearch.db --study-name prod --n-trials 50

# Terminal 2 (same storage → shared study, no duplicate work)
uv run python scripts/run_hypersearch.py \
    --storage sqlite:///reports/hypersearch.db --study-name prod --n-trials 50
```

---

## 6e. Memory budget (8 GB / 8 vCPU target)

User direction 2026-06-04: production deploy target is an 8 GB / 8
vCPU box. Each pipeline run now logs peak RSS at the end:

```
Phase 5 complete in 184.3s (peak RSS: 2.71 GB)
```

If a run exceeds 6 GB (1 GB headroom for OS + 1 GB for other
processes), a warning fires:

```
WARNING: Peak RSS 6.47 GB exceeds 6 GB budget (8 GB box, 2 GB headroom).
```

For pre-flight checks before pushing data-heavy phases (12, 13):

```bash
# Tiny smoke (5-10 min) — confirms wiring + RSS sanity
uv run python scripts/run_phase5.py \
    --start 2022-01-01 --end 2024-12-31 --n-tickers 40 \
    --bootstrap-n 50 [your new flag]

# Production smoke (30-60 min) — confirms scale + RAM stays under 6 GB
nohup uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 --n-tickers 150 \
    [your new flag] > logs/smoke.log 2>&1 &

# Then verify
grep -E "Peak RSS|HOLDOUT" logs/smoke.log | tail
```

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

# --- Hypersearch endpoints (no auth to queue; X-Password to launch) ---

# Queue a hypersearch job (no auth required)
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{
           "n_trials": 50,
           "n_tickers": 25,
           "start_date": "2015-01-01",
           "holdout_years": 2,
           "bootstrap_n": 50,
           "universe_sampling": "current",
           "seed": 42
         }' \
     http://localhost:8000/jobs/queue

# Launch the queued job (requires STOCKPRED_PW)
curl -X POST \
     -H "X-Password: $STOCKPRED_PW" \
     http://localhost:8000/jobs/run/<queue_id>

# List all hypersearch runs (metadata, no trial rows)
curl http://localhost:8000/hypersearch/runs

# Full detail for one run including all trial results
curl http://localhost:8000/hypersearch/runs/<run_id>

# Get the hypersearch run linked to a specific job
curl http://localhost:8000/hypersearch/runs/by-job/<job_id>
```

Responses are always JSON. NaN and Infinity values are serialised as `null`
(RFC-7159 compliant).

---

## 9. Recommended parameters for production (Railway / 8 GB)

> **Updated for Phase 13** — the config below reflects the highest documented
> holdout Sharpe (+0.173, DD −8.2%) across all research phases. It supersedes
> the pre-Phase-8 config that was here previously.
>
> **Honest context**: the 95% bootstrap CI is [−0.32, +0.58] — it straddles
> zero, so this is not a statistically proven edge. It is the best result
> the research has produced so far. A simple S&P 500 buy-and-hold has a
> long-run Sharpe of ~0.5–0.6, so do not treat +0.17 as a finished product.

### Queue the job (no auth required)

```bash
curl -X POST \
     -H "Content-Type: application/json" \
     -d '{
       "phase": 5,
       "start_date": "2014-01-01",
       "n_tickers": null,
       "universe_sampling": "current",
       "horizons": [5],

       "use_sector_features": false,
       "use_tier2_features": false,
       "use_regime_features": false,
       "beta_neutralise": false,

       "ensemble_weighting": "equal",
       "position_sizing": "hrp",
       "k_per_side_pct": 0.15,
       "leverage_per_side": 1.0,
       "sector_cap_gross": 0.30,
       "min_trade_threshold": 0.005,
       "holdout_years": 2,

       "ranks_only": true,
       "use_meta_labelling": true,
       "meta_threshold": 0.55,
       "meta_mode": "binary",
       "meta_walk_forward_folds": 1,

       "use_edgar_features": false,
       "use_edgar_item_features": true,

       "bootstrap_n": 200,
       "cv": {
         "train_years": 5,
         "test_months": 6,
         "embargo_days": 25,
         "min_train_obs": 1000
       },
       "gbm": {
         "num_leaves": 63,
         "learning_rate": 0.05,
         "n_estimators": 600,
         "min_data_in_leaf": 500,
         "feature_fraction": 0.7,
         "bagging_fraction": 0.8,
         "bagging_freq": 5,
         "reg_lambda": 2.0,
         "early_stopping_rounds": 30
       }
     }' \
     https://stock-predictor-production-d4d4.up.railway.app/jobs/queue
```

This returns a `{"id": "<queue_id>", ...}`. Then launch it with your password:

```bash
curl -X POST \
     -H "X-Password: $STOCKPRED_PW" \
     https://stock-predictor-production-d4d4.up.railway.app/jobs/run/<queue_id>
```

### Why each choice (Phase 13 research findings)

| Parameter | Value | Reason |
|---|---|---|
| `horizons` | `[5]` | h=5d has IC IR 1.18 in live data; h=1d (IR 0.61) adds noise and turnover; h=21d adds little when running walk-forward cadence at 5d |
| `position_sizing` | `hrp` | Hierarchical Risk Parity (Phase 7) beat vol_scaled in holdout |
| `ensemble_weighting` | `equal` | Single horizon → weighting is moot; matches documented Phase 13 config |
| `use_*_features` | all `false` | **Counter-intuitive**: Phase 13 found that stripping sector/regime/tier2 features and keeping only price-derived cross-sectional ranks improved holdout Sharpe. The additional features were adding noise and overfitting surface for the walk-forward folds |
| `ranks_only` | `true` | Phase 8 finding: cross-sectional rank columns carry the durable signal; raw feature values add regime-level dependence that doesn't generalise |
| `use_meta_labelling` | `true` (binary @ 0.55) | Phase 8/13 best config; binary gate outperformed confidence mode (Phase 10 confidence gave +0.08 vs +0.17 binary) |
| `use_edgar_item_features` | `true` | Phase 13: per-item 8-K codes (CEO change 5.02, earnings 2.02, M&A 1.01/2.01/8.01) carry directional signal raw counts don't |
| `use_edgar_features` | `false` | **Do NOT enable** — Phase 12 raw 8-K event counts hurt the strategy |
| `start_date` | `2014-01-01` | Walk-forward burn-in means 2010–2013 produce zero positions anyway; dropping them cuts memory ~25% |
| `min_data_in_leaf` | `500` | Scaled up from 200 (research used 150 tickers); with ~500 tickers you have ~3× more rows per fold — higher floor prevents overfitting |
| `learning_rate` | `0.05` | Slightly higher for larger dataset; early stopping at 30 rounds handles convergence |
| `bootstrap_n` | `200` | Reduced from 500 — stress test is informative but not the bottleneck |

### Memory and runtime (8 vCPU / 8 GB RAM)

| Component | 150 tickers (research) | ~500 tickers (this run) |
|---|---|---|
| Prices + features | ~0.4 GB | ~1.3 GB |
| EDGAR item fetch | ~0.1 GB | ~0.3 GB |
| LightGBM training | ~0.3 GB | ~0.9 GB |
| Python + OS | ~0.3 GB | ~0.3 GB |
| **Peak total** | **~1.0 GB** | **~2.5–3 GB** |

8 GB RAM: ✅ comfortable headroom.

LightGBM uses all available threads — 8 vCPUs makes training ~5× faster than
the 1–2 vCPU baseline. Expected runtime: **1–2 hours** (warm cache) or
**1.5–2.5 hours** on first run including EDGAR and price fetches for 500 tickers.

---

## 10. Common operations

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

## 11. Extending the project

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

## 12. Troubleshooting

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

## 13. Performance benchmarks

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

## 14. Where to read more

- [`CONCEPTS.md`](CONCEPTS.md) — every metric, term, and design decision
  explained.
- [`DEPLOYMENT.md`](DEPLOYMENT.md) — Docker, Fly.io, Render, VM with Caddy.
- [`PROJECT_LOG.md`](PROJECT_LOG.md) — chronological history of every
  change and why.
- [`HANDOFF.md`](HANDOFF.md) — protocol for a future agent picking the
  project up mid-stream.
