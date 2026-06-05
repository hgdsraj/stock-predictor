// Thin fetch wrapper. In dev, Vite proxies API calls to the FastAPI server
// running on http://127.0.0.1:8000 (see vite.config.ts). In production, the
// FastAPI app serves the built SPA from the same origin, so relative URLs
// "just work".

import type {
  BacktestSummary,
  HealthResponse,
  JobDetail,
  JobResponse,
  NewsHeadline,
  Quote,
  QueuedJob,
  RunSummary,
  TickerDetail,
  TickerSummary,
  TopMovers,
  WatchedItem,
} from "./types";

const API_BASE = (import.meta as any).env?.VITE_API_BASE || "";

async function get<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "GET",
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as any).detail || `GET ${path} → ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function post<T>(path: string, body?: unknown, headers?: Record<string, string>): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      Accept: "application/json",
      "Content-Type": "application/json",
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error((errBody as any).detail || `POST ${path} → ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function del<T>(path: string, headers?: Record<string, string>): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
    method: "DELETE",
    headers: { Accept: "application/json", ...headers },
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error((errBody as any).detail || `DELETE ${path} → ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

export const api = {
  // ── Existing read endpoints ────────────────────────────────────────────
  health: () => get<HealthResponse>("/healthz"),
  tickers: () => get<TickerSummary[]>("/tickers"),
  tickerDetail: (ticker: string, days = 365) =>
    get<TickerDetail>(`/tickers/${encodeURIComponent(ticker)}/details?days=${days}`),
  tickerNews: (ticker: string, limit = 20) =>
    get<NewsHeadline[]>(`/tickers/${encodeURIComponent(ticker)}/news?limit=${limit}`),
  /** Latest delayed (~15min) quote. Server-cached a few seconds. */
  quote: (ticker: string) => get<Quote>(`/quote/${encodeURIComponent(ticker)}`),
  latestPredictions: (top_k = 10) =>
    get<TopMovers>(`/predictions/latest?top_k=${top_k}`),
  runs: (limit = 20) => get<RunSummary[]>(`/runs?limit=${limit}`),
  runEquity: (run_id: number) =>
    get<{ date: string; cumulative_return: number | null; drawdown: number | null }[]>(
      `/runs/${run_id}/equity`,
    ),
  backtestSummary: () => get<BacktestSummary>("/backtest/summary"),
  watchlist: () => get<WatchedItem[]>("/watchlist"),

  // ── Jobs (read) ────────────────────────────────────────────────────────
  /** List recent in-flight / completed jobs (no logs in list). */
  listJobs: (limit = 25) => get<JobDetail[]>(`/jobs?limit=${limit}`),
  /** Full detail for one job including logs. Polls for running jobs. */
  jobDetail: (job_id: string) => get<JobDetail>(`/jobs/${encodeURIComponent(job_id)}`),

  // ── Queue (read) ───────────────────────────────────────────────────────
  /** List all queued jobs (pending + launched + cancelled). */
  listQueued: () => get<QueuedJob[]>("/jobs/queue"),

  // ── Existing privileged trigger (X-API-Key) ────────────────────────────
  /** Immediate trigger — requires STOCKPRED_API_KEY via X-API-Key header. */
  refresh: (apiKey?: string) =>
    post<JobResponse>("/jobs/refresh", undefined, apiKey ? { "X-API-Key": apiKey } : {}),

  // ── Queue management (no auth needed to queue; password needed to run) ─
  /** Submit a job to the queue. No auth required. Max 5 pending at once. */
  queueJob: (config: Record<string, unknown>) =>
    post<QueuedJob>("/jobs/queue", config),

  /** Launch a pending queued job. Requires STOCKPRED_PW via X-Password. */
  launchQueued: (queue_id: string, password: string) =>
    post<JobResponse>(`/jobs/run/${encodeURIComponent(queue_id)}`, undefined, {
      "X-Password": password,
    }),

  /** Delete a pending queued job. Requires X-Password. */
  deleteQueued: (queue_id: string, password: string) =>
    del<{ ok: boolean }>(`/jobs/queue/${encodeURIComponent(queue_id)}`, {
      "X-Password": password,
    }),

  /** Soft-cancel a running job. Requires X-Password. */
  cancelJob: (job_id: string, password: string) =>
    del<{ ok: boolean }>(`/jobs/${encodeURIComponent(job_id)}/cancel`, {
      "X-Password": password,
    }),

  /** Job status (legacy single-field response). */
  jobStatus: (job_id: string) => get<JobDetail>(`/jobs/${encodeURIComponent(job_id)}`),
};
