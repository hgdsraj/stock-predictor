#!/usr/bin/env python
"""Run the stock-predictor FastAPI server with uvicorn.

Usage:
    uv run python scripts/serve.py [--host 0.0.0.0] [--port 8000] [--reload]

Environment variables honoured:
    STOCKPRED_DB                 — path to SQLite file (default data/app.db)
    STOCKPRED_WEB_DIST           — path to built frontend (default web/dist)
    STOCKPRED_DISABLE_SCHEDULER  — "1" to disable APScheduler (useful for tests)
    STOCKPRED_CORS               — comma-separated origins, default "*"
"""

from __future__ import annotations

import argparse
import logging

import uvicorn


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--reload", action="store_true")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    uvicorn.run(
        "stockpred.backend.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
