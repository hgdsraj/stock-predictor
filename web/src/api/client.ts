// Thin fetch wrapper. In dev, Vite proxies API calls to the FastAPI server
// running on http://127.0.0.1:8000 (see vite.config.ts). In production, the
// FastAPI app serves the built SPA from the same origin, so relative URLs
// "just work".

import type {
  ActivateRunResponse,
  BacktestSummary,
  HealthResponse,
  HypersearchRequest,
  HypersearchRun,
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

// Helper: build a querystring from an object, omitting null/undefined values.
function qs(params: Record<string, string | number | null | undefined>): string {
  const entries = Object.entries(params).filter(([, v]) => v !== null && v !== undefined);
  if (entries.length === 0) return "";
  return "?" + entries.map(([k, v]) => `${k}=${encodeURIComponent(String(v))}`).join("&");
}

export const api = {
  // ── Existing read endpoints ────────────────────────────────────────────
  health: () => get<HealthResponse>("/healthz"),
  tickers: () => get<TickerSummary[]>("/tickers"),
  /** Ticker detail. `runId` selects the model run whose predictions block to embed
   *  (defaults to the active/latest run on the server). */
  tickerDetail: (ticker: string, days = 365, runId?: number | null) =>
    get<TickerDetail>(
      `/tickers/${encodeURIComponent(ticker)}/details${qs({ days, run_id: runId })}`,
    ),
  tickerNews: (ticker: string, limit = 20) =>
    get<NewsHeadline[]>(`/tickers/${encodeURIComponent(ticker)}/news?limit=${limit}`),
  /** Latest delayed (~15min) quote. Server-cached a few seconds. */
  quote: (ticker: string) => get<Quote>(`/quote/${encodeURIComponent(ticker)}`),
  /** Top long/short movers for the active or specified run. */
  latestPredictions: (top_k = 10, runId?: number | null) =>
    get<TopMovers>(`/predictions/latest${qs({ top_k, run_id: runId })}`),
  /** List of recent runs (newest first). */
  runs: (limit = 20) => get<RunSummary[]>(`/runs?limit=${limit}`),
  /** Single run by id. */
  run: (run_id: number) => get<RunSummary>(`/runs/${run_id}`),
  /** Equity curve for a single run. */
  runEquity: (run_id: number) =>
    get<{ date: string; cumulative_return: number | null; drawdown: number | null }[]>(
      `/runs/${run_id}/equity`,
    ),
  /** Full BacktestSummary for an arbitrary run id. */
  runBacktest: (run_id: number) => get<BacktestSummary>(`/runs/${run_id}/backtest`),
  /** Backtest summary for the active/latest run, or the explicitly-requested run. */
  backtestSummary: (runId?: number | null) =>
    get<BacktestSummary>(`/backtest/summary${qs({ run_id: runId })}`),
  /** Pin a run as the server-side default data source. Requires X-Password. */
  activateRun: (run_id: number, password: string) =>
    post<ActivateRunResponse>(`/runs/${run_id}/activate`, undefined, {
      "X-Password": password,
    }),
  /** Clear the active-run pin. Requires X-Password. */
  deactivateRun: (password: string) =>
    post<ActivateRunResponse>("/runs/deactivate", undefined, {
      "X-Password": password,
    }),
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

  // ── Hypersearch ────────────────────────────────────────────────────────
  /** Queue a hypersearch job. No auth required. */
  queueHypersearch: (cfg: HypersearchRequest) =>
    post<QueuedJob>("/jobs/queue", cfg),

  /** List all hypersearch runs (metadata, no trials). */
  listHypersearchRuns: (limit = 25) =>
    get<HypersearchRun[]>(`/hypersearch/runs?limit=${limit}`),

  /** Full detail for one hypersearch run including all trial rows. */
  hypersearchRun: (run_id: number) =>
    get<HypersearchRun>(`/hypersearch/runs/${run_id}`),

  /** Get hypersearch run linked to a job_id (polls while running). */
  hypersearchRunByJob: (job_id: string) =>
    get<HypersearchRun>(`/hypersearch/runs/by-job/${encodeURIComponent(job_id)}`),
};
