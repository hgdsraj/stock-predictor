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

| Var                          | Default                     | Meaning                                 |
| ---------------------------- | --------------------------- | --------------------------------------- |
| `STOCKPRED_DB`               | `data/app.db`               | SQLite file path                        |
| `STOCKPRED_WEB_DIST`         | `web/dist`                  | Directory holding the built SPA         |
| `STOCKPRED_DISABLE_SCHEDULER`| `0`                         | Set `1` to skip APScheduler at startup  |
| `STOCKPRED_CORS`             | `*`                         | Comma-separated allowed origins         |

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

```bash
curl -X POST http://localhost:8000/jobs/refresh
# returns: {"job_id": "<uuid>", "status": "queued"}

curl http://localhost:8000/jobs/<uuid>
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

- The API has **no authentication**. Do not expose it to the public internet
  without putting an auth layer (Caddy Basic Auth, Cloudflare Access,
  Tailscale, a reverse-proxy with OIDC) in front.
- The container runs as root by default — switch to a dedicated user with a
  `USER` directive in your private fork before production exposure.
- `STOCKPRED_CORS=*` is convenient locally but should be locked down in
  production.

## What this project will NOT do for you

- Place real orders. There is no broker integration. This is *backtest only.*
- Beat the market. Phase 1+2 honest results are documented in
  [`PROJECT_LOG.md`](./PROJECT_LOG.md).
- Replace your judgement. Treat all numbers shown on the dashboard as
  research output, not advice.
