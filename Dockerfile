# Multi-stage build:
# 1. node:20 — install frontend deps, run `npm run build` -> web/dist
# 2. python:3.12-slim — install backend deps via uv, copy code + built SPA,
# uvicorn serves both /api and the SPA from one process.

# ----- 1. Frontend build ----------------------------------------------------
    FROM node:20-alpine AS web-build

    WORKDIR /web
    
    COPY web/package.json web/package-lock.json* ./
    RUN npm ci --no-audit --no-fund
    
    COPY web/ ./
    RUN npm run build
    
    # ----- 2. Python runtime ----------------------------------------------------
    FROM python:3.12-slim AS runtime
    
    RUN apt-get update \
        && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 \
        && rm -rf /var/lib/apt/lists/* \
        && curl -LsSf https://astral.sh/uv/install.sh | sh \
        && cp /root/.local/bin/uv /usr/local/bin/uv \
        && cp /root/.local/bin/uvx /usr/local/bin/uvx 2>/dev/null || true
    
    WORKDIR /app
    
    # Copy everything hatchling needs before uv sync
    COPY pyproject.toml uv.lock* README.md ./
    COPY src/ ./src/
    COPY scripts/ ./scripts/
    
    RUN uv sync --frozen --no-dev || uv sync --no-dev
    
    COPY --from=web-build /web/dist /app/web/dist
    
    # Create runtime directories
    RUN mkdir -p /app/data /app/logs /app/reports /app/models
    
    ENV STOCKPRED_DB=/app/data/app.db \
        STOCKPRED_WEB_DIST=/app/web/dist \
        PYTHONUNBUFFERED=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PATH="/app/.venv/bin:${PATH}"
    
    EXPOSE 8000
    
    # NOTE: Running as root for Railway volume compatibility.
    # Railway mounts volumes as root, so a non-root user cannot write to them.
    
    HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=5 \
        CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)" \
        || exit 1
    
    CMD ["uvicorn", "stockpred.backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
    