// Thin fetch wrapper. In dev, Vite proxies API calls to the FastAPI server
// running on http://127.0.0.1:8000 (see vite.config.ts). In production, the
// FastAPI app serves the built SPA from the same origin, so relative URLs
// "just work".

import type {
  BacktestSummary,
  HealthResponse,
  JobResponse,
  RunSummary,
  TickerDetail,
  TickerSummary,
  TopMovers,
} from "./types";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

async function get<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`GET ${path} -> ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`POST ${path} -> ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => get<HealthResponse>("/healthz"),
  tickers: () => get<TickerSummary[]>("/tickers"),
  tickerDetail: (ticker: string, days = 365) =>
    get<TickerDetail>(`/tickers/${encodeURIComponent(ticker)}/details?days=${days}`),
  latestPredictions: (top_k = 10) =>
    get<TopMovers>(`/predictions/latest?top_k=${top_k}`),
  runs: (limit = 20) => get<RunSummary[]>(`/runs?limit=${limit}`),
  runEquity: (run_id: number) =>
    get<{ date: string; cumulative_return: number | null; drawdown: number | null }[]>(
      `/runs/${run_id}/equity`,
    ),
  backtestSummary: () => get<BacktestSummary>("/backtest/summary"),
  refresh: () => post<JobResponse>("/jobs/refresh"),
  jobStatus: (job_id: string) => get<JobResponse>(`/jobs/${encodeURIComponent(job_id)}`),
};
