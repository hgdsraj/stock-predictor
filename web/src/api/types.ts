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
}

export interface BacktestSummary {
  run: RunSummary;
  equity_curve: EquityPoint[];
}

export interface JobResponse {
  job_id: string;
  status: string;
  detail: string | null;
}

export interface HealthResponse {
  status: string;
  db: string;
  scheduler: string;
}
