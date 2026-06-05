// Types matching backend Pydantic schemas (src/stockpred/backend/schemas.py).
// Kept in sync manually — small enough that this is fine.

export interface TickerSummary {
  ticker: string;
  sector: string | null;
  industry: string | null;
  last_price: number | null;
  market_cap: number | null;
  last_updated: string | null;
}

export interface PriceBar {
  date: string;
  open: number | null;
  high: number | null;
  low: number | null;
  close: number | null;
  adj_close: number | null;
  volume: number | null;
}

export interface Prediction {
  date: string;
  ticker: string;
  score: number;
  rank: number | null;
  side: "long" | "short" | null;
  weight: number | null;
  per_horizon: Record<string, number | null>;
}

export interface TickerDetail {
  ticker: string;
  sector: string | null;
  industry: string | null;
  market_cap: number | null;
  beta: number | null;
  trailing_pe: number | null;
  forward_pe: number | null;
  dividend_yield: number | null;
  short_ratio: number | null;
  short_percent_of_float: number | null;
  fifty_two_week_high: number | null;
  fifty_two_week_low: number | null;
  long_business_summary: string | null;
  prices: PriceBar[];
  predictions: Prediction[];
}

export interface TopMovers {
  date: string | null;
  long: Prediction[];
  short: Prediction[];
}

export interface EquityPoint {
  date: string;
  daily_return: number | null;
  cumulative_return: number | null;
  drawdown: number | null;
  turnover: number | null;
  benchmark_return: number | null;
}

export interface RunSummary {
  id: number;
  started_at: string;
  completed_at: string | null;
  status: string;
  metrics: Record<string, number>;
  per_horizon_diagnostics: Record<string, Record<string, number>>;
  tickers_count: number;
  note: string | null;
  /** Full PipelineConfig snapshot. Defaults to {} for old runs that
   *  predate the API change. */
  config: Record<string, unknown>;
  /** UUID of the JobRecord that produced this run, if any. */
  job_id: string | null;
  /** True if this run is the server-side default data source. */
  is_active: boolean;
}

export interface BacktestSummary {
  run: RunSummary;
  equity_curve: EquityPoint[];
}

export interface ActivateRunResponse {
  active_run_id: number | null;
  message: string;
}

export interface JobResponse {
  job_id: string;
  status: string;
  detail: string | null;
}

export interface JobDetail {
  job_id: string;
  status: string;
  job_type: "pipeline" | "hypersearch";
  started_at: string | null;
  updated_at: string | null;
  config: Record<string, unknown>;
  logs: string[];
  run_id: number | null;
  elapsed_s: number | null;
  error: string | null;
}

export interface HypersearchRequest {
  n_trials?: number;
  n_tickers?: number;
  start_date?: string;
  end_date?: string | null;
  holdout_years?: number;
  bootstrap_n?: number;
  universe_sampling?: "current" | "first" | "random";
  seed?: number;
}

export interface HypersearchTrial {
  trial: number;
  value: number | null;
  hold_sharpe: number | null;
  hold_ci_lo: number | null;
  hold_ci_hi: number | null;
  hold_dd: number | null;
  hold_hit: number | null;
  hold_ann_return: number | null;
  dev_sharpe: number | null;
  elapsed_s: number | null;
  error: string | null;
  params: Record<string, unknown>;
}

export interface HypersearchRun {
  id: number;
  job_id: string | null;
  started_at: string;
  completed_at: string | null;
  status: string;
  config: Record<string, unknown>;
  n_trials_requested: number;
  n_trials_done: number;
  best_sharpe: number | null;
  best_params: Record<string, unknown> | null;
  trials: HypersearchTrial[];
}

export interface QueuedJob {
  id: string;
  created_at: string;
  config: Record<string, unknown>;
  label: string | null;
  status: "pending" | "launched" | "cancelled";
  launched_at: string | null;
  job_id: string | null;
}

export interface HealthResponse {
  status: string;
  db: string;
  scheduler: string;
}

export interface Quote {
  ticker: string;
  price: number | null;
  previous_close: number | null;
  open: number | null;
  day_high: number | null;
  day_low: number | null;
  volume: number | null;
  market_cap: number | null;
  change: number | null;
  change_pct: number | null;
  as_of: string;
  delayed: boolean;
}

export interface WatchedItem {
  ticker: string;
  label: string | null;
  category: string | null;
  note: string | null;
  last_price: number | null;
  last_updated: string | null;
}

export interface NewsHeadline {
  uuid: string;
  title: string | null;
  publisher: string | null;
  link: string | null;
  type: string | null;
  published_at: string | null;
}
