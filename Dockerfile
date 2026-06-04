# Multi-stage build:
#   1. node:20 — install frontend deps, run `npm run build` -> web/dist
#   2. python:3.12-slim — install backend deps via uv, copy code + built SPA,
#      uvicorn serves both /api and the SPA from one process.
#
# Image size target: ~600-700 MB (mostly pandas/numpy/lightgbm). The frontend
# build stage is discarded.

# ----- 1. Frontend build ----------------------------------------------------
FROM node:20-alpine AS web-build
WORKDIR /web

# Cache dep install: copy lockfile + package.json first, then sources.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build

# ----- 2. Python runtime ----------------------------------------------------
FROM python:3.12-slim AS runtime

# Install uv (fast Python package manager) into /usr/local/bin.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && cp /root/.local/bin/uv /usr/local/bin/uv \
    && cp /root/.local/bin/uvx /usr/local/bin/uvx 2>/dev/null || true

WORKDIR /app

# Cache deps: copy pyproject + uv.lock first.
COPY pyproject.toml uv.lock* ./
COPY README.md ./
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy the rest of the project source.
COPY src/ ./src/
COPY scripts/ ./scripts/

# Copy the built frontend from the previous stage.
COPY --from=web-build /web/dist /app/web/dist

# Create the data directory inside the image so the bind-mount target exists.
RUN mkdir -p /app/data /app/logs /app/reports /app/models

# Environment defaults.
ENV STOCKPRED_DB=/app/data/app.db \
    STOCKPRED_WEB_DIST=/app/web/dist \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/app/.venv/bin:${PATH}"

EXPOSE 8000

# Drop privileges to a non-root user (review finding M3). The data directory
# is owned by this user so volume mounts work.
RUN useradd -u 1001 -m -s /bin/false app \
    && chown -R app /app
USER app

# Healthcheck: container is "healthy" only after the API responds.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" \
      || exit 1

CMD ["uvicorn", "stockpred.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
