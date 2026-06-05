# Deployment guide

This project ships as a **single container** that runs:

- the FastAPI backend (`/healthz`, `/tickers`, `/predictions/latest`, …),
- the APScheduler daily refresh job (in-process),
- the React SPA, served from `/`.

State is one SQLite file plus a parquet cache. No external infrastructure
required. To run, you need:

- Docker (with `compose` plugin) on the host, OR
- A platform that builds from a `Dockerfile` (Fly.io, Render, Railway, Koyeb, …), OR
- Python 3.11+ and Node 20+ on the host (manual).
## Local quick-start (Docker)

```bash
git clone https://github.com/hgdsraj/stock-predictor.git
cd stock-predictor
docker compose up --build
# open http://localhost:8000
```

First boot will create `./data/app.db` and `./data/cache/*.parquet`. The
dashboard will be empty until the first pipeline run completes. Click
**Refresh** in the header to trigger one, or wait for the daily cron (default
22:00 America/New_York, weekdays).

To inspect the DB:

```bash
sqlite3 ./data/app.db ".tables"
```

To stop:

```bash
docker compose down            # keeps data/ volume
docker compose down -v         # also wipes data/
```

## Local manual (no Docker)

Requires Python 3.11+, Node 20+, and `uv` (https://docs.astral.sh/uv/).

```bash
# Backend
uv sync --extra dev
uv run python scripts/serve.py --host 0.0.0.0 --port 8000

# Frontend (in another terminal)
cd web
npm ci
npm run build            # for production: served by FastAPI from web/dist
# or
npm run dev              # for development: Vite dev server with HMR
```

The Vite dev server (`npm run dev`) proxies API calls to `127.0.0.1:8000`.
For the production build, the same FastAPI process serves both `/api/*` and
the SPA from one origin — no CORS, no separate hosting.

## Production: Fly.io

Fly.io has the lowest friction for this kind of single-container app, and
the free tier comfortably handles it.

```bash
brew install flyctl          # or follow https://fly.io/docs/hands-on/install-flyctl/
fly auth login               # browser flow

# In the repo root:
fly launch                   # creates fly.toml; pick "Yes" for Dockerfile detection
                             # decline Postgres/Redis when prompted
fly volumes create stockpred_data --size 1 --region <your-region>
fly deploy
```

Create a `fly.toml` like the one below (or edit the one `fly launch`
generated):

```toml
app = "stock-predictor"
primary_region = "iad"      # change to your nearest region

[build]
dockerfile = "Dockerfile"

[env]
STOCKPRED_DB = "/data/app.db"
STOCKPRED_CORS = "*"        # tighten in production

[http_service]
internal_port = 8000
force_https = true
auto_stop_machines = "stop"  # save money when idle
auto_start_machines = true

[[mounts]]
source = "stockpred_data"
destination = "/data"

[checks.healthz]
type = "http"
port = 8000
path = "/healthz"
interval = "30s"
timeout = "5s"
```

Then:

```bash
fly deploy
fly open                     # opens https://stock-predictor.fly.dev
fly logs                     # tail logs
fly ssh console              # shell into the running machine
```

Backups: `fly ssh console`, then `cp /data/app.db /data/app.db.$(date +%Y%m%d)`
or pull it down with `fly ssh sftp get /data/app.db`.

## Production: Render

1. Push the repo to GitHub.
2. Create a new **Web Service** at <https://dashboard.render.com>.
3. Choose **Docker** as the runtime; point at your repo.
4. Add a **Disk** of 1 GB mounted at `/app/data` so SQLite persists.
5. Environment variables: `STOCKPRED_DB=/app/data/app.db`.
6. Health check path: `/healthz`.

Render runs each build as a fresh container, so the daily cron inside the
container *will* keep working — but a Render restart cancels in-flight jobs.
If you want bulletproof scheduling, use a separate Render **Cron Job** that
hits `POST /jobs/refresh` once a day, and disable the in-process scheduler
with `STOCKPRED_DISABLE_SCHEDULER=1`.

## Production: VM (DigitalOcean, Hetzner, AWS Lightsail, your basement)

Minimal: one 1 GB / 1 vCPU box runs the full stack.

```bash
# On a fresh Ubuntu 24.04:
sudo apt update && sudo apt install -y docker.io docker-compose-plugin
sudo usermod -aG docker $USER
# log out, log back in

git clone https://github.com/hgdsraj/stock-predictor.git
cd stock-predictor
docker compose up -d --build

# Reverse proxy + automatic HTTPS via Caddy
sudo apt install -y caddy
echo 'your.domain.com {
  reverse_proxy 127.0.0.1:8000
}' | sudo tee /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

`systemctl status caddy` to confirm. DNS A record pointing to your VM, and
Caddy will request a Let's Encrypt cert automatically.

Backups: a daily cron line on the host:

```cron
30 4 * * *  cp /home/youruser/stock-predictor/data/app.db /home/youruser/backups/app.db.$(date +\%Y\%m\%d) && find /home/youruser/backups -mtime +30 -delete
```

## Environment variables (all optional)

| Var                          | Default                                           | Meaning                                                                                       |
| ---------------------------- | ------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `STOCKPRED_DB`               | `data/app.db`                                     | SQLite file path                                                                              |
| `STOCKPRED_WEB_DIST`         | `web/dist`                                        | Directory holding the built SPA                                                               |
| `STOCKPRED_DISABLE_SCHEDULER`| `0`                                               | Set `1` to skip APScheduler at startup                                                        |
| `STOCKPRED_CORS`             | `http://localhost:5173,http://127.0.0.1:8000`     | Comma-separated allowed origins. Use `"*"` to widen (no credentials allowed in that mode).    |
| `STOCKPRED_API_KEY`          | *(unset → write endpoints disabled)*              | Secret value. `POST /jobs/refresh` requires header `X-API-Key: <value>`.                       |

## Updating

```bash
git pull
docker compose build
docker compose up -d
```

A schema change ships as additive ORM columns; the `create_all()` call on
startup will add new tables. Destructive changes will be released with an
Alembic migration script and noted in the commit message.

## Common ops

### Trigger a refresh from CLI

`POST /jobs/refresh` accepts an optional JSON body. Omit it entirely to run
Phase 1 with stock defaults.

```bash
# Minimal — Phase 1, all defaults
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     http://localhost:8000/jobs/refresh
# returns: {"job_id": "<uuid>", "status": "queued"}

# Phase 5 (vol-scaled, regime-aware) with a smaller universe
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"phase": 5, "n_tickers": 50}' \
     http://localhost:8000/jobs/refresh

# Force-refresh cached data and use a specific date window
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"refresh_data": true, "start_date": "2015-01-01"}' \
     http://localhost:8000/jobs/refresh

# Poll until done
curl http://localhost:8000/jobs/<uuid>
```

#### Phase 13 BEST honest config — single curl

This is the documented best-known config (HOLDOUT Sharpe +0.173, CI
[-0.32, +0.58], DD -8.2%). Designed for an 8 GB / 8 vCPU box; peak
RSS ~1 GB, runtime ~3-7 min (first cold-cache run includes a ~30 sec
SEC EDGAR submissions fetch + ~4 min EDGAR per-quarter form-index
fetch if those caches don't exist yet).

```bash
curl -X POST \
     -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     --max-time 1800 \
     -d '{
       "phase": 5,
       "start_date": "2014-01-01",
       "n_tickers": 150,
       "universe_sampling": "current",
       "horizons": [5],
       "model": "gbm",
       "use_sector_features": false,
       "use_tier2_features": false,
       "use_regime_features": false,
       "beta_neutralise": false,
       "bootstrap_method": "block",
       "holdout_years": 2,
       "position_sizing": "hrp",
       "k_per_side_pct": 0.15,
       "sector_cap_gross": 0.30,
       "min_trade_threshold": 0.005,
       "ensemble_weighting": "equal",
       "bootstrap_n": 500,
       "use_meta_labelling": true,
       "meta_threshold": 0.55,
       "ranks_only": true,
       "meta_mode": "binary",
       "use_edgar_item_features": true
     }' \
     http://localhost:8000/jobs/refresh
```

Notes:
- **Do NOT add `"use_edgar_features": true`** (Phase 12 raw 8-K counts
  hurt the strategy: Sharpe -0.158 → -0.376).
- Set env `EDGAR_USER_AGENT="Your Name your-email@example.com"` on the
  server so SEC has someone to contact.
- Returns `409` if a pipeline run is already in flight.

#### Full body reference

All fields are optional. Unset fields use the pipeline defaults shown below.

```jsonc
{
  // Which pipeline to run.
  // 1 = Phase 1 (basic GBM, top-k portfolio)
  // 5 = Phase 5 (vol-scaled, regime-aware, sector-capped)
  "phase": 1,

  // Universe / history
  "start_date": "2010-01-01",  // ISO date
  "end_date": null,            // null = today
  "n_tickers": 100,            // null = all S&P 500 members
  "universe_sampling": "random",  // "random" | "current" | "first"
  "refresh_data": false,       // force-refetch cached price/fundamental data

  // Model
  "horizons": [1, 5, 21],     // forecast horizons in trading days
  "model": "gbm",             // "gbm" | "logistic"
  "use_sector_features": true,

  // Cross-validation
  "cv": {
    "train_years": 3,
    "test_months": 6,
    "embargo_days": 25,
    "min_train_obs": 1000
  },

  // LightGBM hyper-params (ignored when model = "logistic")
  "gbm": {
    "num_leaves": 63,
    "learning_rate": 0.03,
    "n_estimators": 800,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50
  },

  // ── Phase 1 only ──────────────────────────────────────────────
  "k_per_side": 20,           // number of longs and shorts
  "feature_cols": null,       // explicit feature list; null = use all

  // ── Phase 5 only ──────────────────────────────────────────────
  "use_tier2_features": true,    // 12-1 momentum, IVOL, beta, max-ret, Amihud
  "use_regime_features": true,   // VIX, term spread, USD, cross-sec dispersion
  "beta_neutralise": false,      // portfolio-level beta-vs-SPY neutralisation
  "bootstrap_method": "block",   // "block" | "iid"
  "holdout_years": 2,            // years held out from CV
  "position_sizing": "vol_scaled", // "vol_scaled" | "top_k"
  "k_per_side_pct": 0.15,        // fraction of universe per side (vol_scaled)
  "leverage_per_side": 1.0,
  "sector_cap_gross": 0.30,      // max gross exposure per GICS sector; null = uncapped
  "min_trade_threshold": 0.005,
  "ensemble_weighting": "ic_ir", // "ic_ir" | "equal"
  "bootstrap_n": 500
}
```

### Inspect the latest run

```bash
curl http://localhost:8000/runs?limit=5 | jq .
```

### Force-reset the DB

```bash
docker compose down
rm -rf data/app.db data/cache
docker compose up -d
```

### See what the cron will do next

The default cron is `0 22 * * 1-5 America/New_York`. To override, set
the environment variable `STOCKPRED_DAILY_CRON="0 23 * * *"` (adding this
hookup is a 3-line code change in `jobs.py`; left as an exercise for now).

## Tuning

- **CPU**: LightGBM is CPU-bound; more cores → faster pipeline. Default is
  unlimited.
- **Memory**: the pipeline holds ~100MB per 100 tickers x 5y. 1 GB RAM is fine
  for the default 100-name universe; bump to 2 GB for the full S&P 500.
- **Disk**: ~50 MB per 100 tickers x 10y price history. SQLite grows ~30 MB
  per year of daily snapshots.

## Security notes

- **Read endpoints** (`/healthz`, `/tickers`, `/predictions/latest`,
  `/runs`, `/backtest/summary`, …) are public. Treat the data as
  research output, not anything sensitive.
- **Write endpoints** (`POST /jobs/refresh`) are gated by `STOCKPRED_API_KEY`.
  If the env var is unset, they return 403; if set, every request must carry
  `X-API-Key: <value>`. This protects against drive-by CSRF — anyone can hit
  the read endpoints, but only holders of the key can trigger an expensive
  pipeline run.
- **CORS** defaults to `localhost:5173,127.0.0.1:8000`. Widen via
  `STOCKPRED_CORS` only when you know what you're doing. The wildcard `*`
  silently disables `allow_credentials` (browsers reject `*` + credentials
  per spec).
- **The container runs as a non-root user** (UID 1001) by default. The
  `./data` volume must be writable by this user. On most hosts `chown -R
  1001:1001 ./data` once at install does the trick.
- **No transport encryption** is configured inside the container; put a
  reverse proxy (Caddy / nginx / Cloudflare Tunnel) in front for HTTPS.
- **Path traversal in the SPA fallback** is explicitly defended against
  (test: `tests/test_backend_api.py::test_spa_fallback_rejects_path_traversal`).
- For sensitive deployments, add an upstream auth layer (Cloudflare Access,
  Tailscale, OIDC via Caddy `forward_auth`). This project's API key is a
  drive-by deterrent, not a substitute for proper auth.

## What this project will NOT do for you

- Place real orders. There is no broker integration. This is *backtest only.*
- Beat the market. Phase 1+2 honest results are documented in
  [`PROJECT_LOG.md`](./PROJECT_LOG.md).
- Replace your judgement. Treat all numbers shown on the dashboard as
  research output, not advice.

---

## Future: zero-downtime deploys + managed infra

Currently `railway up` kills the container immediately, which terminates any
in-flight pipeline thread. This is acceptable for now — just check `/jobs`
before deploying. Options when this becomes a real problem:

### Option A — SIGTERM graceful drain (easy, no infra change)

Catch `SIGTERM` in the lifespan and stall shutdown until the running job
finishes or a timeout expires. Set `stopTimeoutSeconds = 7200` in
`railway.toml`. Railway queues the new deploy behind the drain. Works for
planned deploys; does not protect against OOM kills or `railway down`.

Estimated effort: ~30 min. No new infrastructure.

### Option B — Separate Railway worker service (proper fix)

Split into two Railway services sharing a database:

- **`web`** — FastAPI + SPA. Deploys freely, just reads DB and enqueues jobs.
- **`worker`** — Long-running process that polls `queued_jobs` and runs
  pipelines. Only deploy when idle (or implement its own graceful drain).

Jobs survive web server deploys entirely. Requires a shared DB that both
services can reach — which means dropping SQLite.

Estimated effort: 1–2 days including DB migration.

### On switching to managed Postgres

Postgres removes the SQLite write-serialisation limit and enables the web +
worker split, but it complicates local development:

- Local runs currently need zero infra (just `uv run` or `docker compose up`).
- With Postgres, local dev needs a running Postgres instance.

**Mitigation**: keep SQLite for local dev, use Postgres in production. The
SQLAlchemy layer already abstracts the difference — only the connection URL
changes. One `if "sqlite" in DB_URL` branch in `db.py` handles the
`PRAGMA journal_mode=WAL` that SQLite needs but Postgres doesn't.

`railway.toml` would set `DATABASE_URL` to Railway's managed Postgres;
local `.env` keeps `STOCKPRED_DB=data/app.db`. No code changes needed in
most places; just make sure to test both dialects in CI.
