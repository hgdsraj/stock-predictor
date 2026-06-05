// Plain-English definitions surfaced as hover popovers across the app.
// Sourced from docs/CONCEPTS.md so the UI and docs stay consistent.

export interface GlossaryEntry {
  term: string;
  short: string; // one-liner shown in the tooltip header
  body: string; // fuller explanation
}

export const glossary = {
  // ── Model output ───────────────────────────────────────────────────────
  score: {
    term: "Model score",
    short: "A unit-less number ranking stocks within each day.",
    body:
      "For each stock each day the model outputs a single score. Positive = we expect it to outperform; negative = underperform; bigger magnitude = more confident. " +
      "The score has no natural units — it is only used to RANK stocks within a day. We buy the highest-scored names and short the lowest-scored. " +
      "A score is not a price target and not a probability.",
  },
  signal: {
    term: "Signal (Buy / Sell)",
    short: "The sign of the latest score → long or short.",
    body:
      "Buy = the latest score is positive (a long candidate). Sell = the latest score is negative (a short candidate). " +
      "Because the model predicts forward returns, the most recent score is effectively the call for the next session. " +
      "This is research output for a dollar-neutral long/short book — not personalised advice.",
  },

  // ── Fundamentals (ticker page) ─────────────────────────────────────────
  market_cap: {
    term: "Market cap",
    short: "Share price × shares outstanding — the company's total equity value.",
    body:
      "The total market value of a company's shares. Large-cap (>$10B) names are generally more liquid and less volatile than small-caps. " +
      "Shown for context only — it is NOT fed into the model.",
  },
  beta: {
    term: "Beta",
    short: "Sensitivity of the stock to the overall market (SPY).",
    body:
      "Beta measures how much the stock moves relative to the market. Beta 1.0 = moves with the market; >1 = amplifies market moves; " +
      "<1 = dampens them; negative = moves opposite. Our Phase 5 strategy can beta-neutralise the book so returns reflect stock-picking, not market exposure.",
  },
  trailing_pe: {
    term: "P/E (TTM)",
    short: "Price ÷ trailing-twelve-month earnings per share.",
    body:
      "How many dollars investors pay per dollar of the last year's earnings. High P/E = priced for growth; low P/E = cheap or troubled. " +
      "This is a CURRENT value from yfinance, not point-in-time, so it is shown for context only and never fed to the model.",
  },
  forward_pe: {
    term: "Forward P/E",
    short: "Price ÷ analysts' expected next-year earnings.",
    body:
      "Like P/E but using forecast earnings instead of trailing. Lower than trailing P/E implies analysts expect earnings to grow. " +
      "Context only — not a model input.",
  },
  dividend_yield: {
    term: "Dividend yield",
    short: "Annual dividend ÷ price.",
    body:
      "The cash dividend a shareholder receives per year as a percent of the share price. Higher yields are common in mature, slower-growth companies. " +
      "Context only.",
  },
  short_ratio: {
    term: "Short ratio (days to cover)",
    short: "Shares sold short ÷ average daily volume.",
    body:
      "Roughly how many days of normal trading it would take short-sellers to buy back (cover) their positions. " +
      "A high short ratio can signal bearish sentiment, or set up a 'short squeeze' if the price rises and shorts rush to cover.",
  },
  short_percent_of_float: {
    term: "Short % of float",
    short: "Shares sold short ÷ freely-tradable shares.",
    body:
      "The fraction of the tradable share count that is currently sold short. Elevated values (>10–20%) indicate heavy bearish positioning and squeeze potential.",
  },
  fifty_two_week_high: {
    term: "52-week high",
    short: "Highest price over the past year.",
    body:
      "The highest traded price in the trailing 52 weeks. Often watched as a resistance level — price can stall there, or breaking above can signal momentum.",
  },
  fifty_two_week_low: {
    term: "52-week low",
    short: "Lowest price over the past year.",
    body:
      "The lowest traded price in the trailing 52 weeks. Often watched as a support level — price can bounce there, or breaking below can signal weakness.",
  },

  // ── Strategy metrics ───────────────────────────────────────────────────
  sharpe: {
    term: "Sharpe ratio",
    short: "Annualised return per unit of volatility.",
    body:
      "mean_daily_return / std_daily_return × √252. Out-of-sample, after costs: <0 loses money; 0–0.5 mediocre; 0.5–1.0 is where real strategies live; " +
      ">2.0 — suspect a leak before celebrating.",
  },
  ann_return: {
    term: "Annualised return",
    short: "Compounded return scaled to a yearly rate.",
    body: "The strategy's average growth rate expressed per year. Net of the configured trading costs (~12 bps round-trip by default).",
  },
  ann_vol: {
    term: "Annualised volatility",
    short: "Standard deviation of returns, scaled to a year.",
    body: "How much the strategy's returns swing around their average, annualised. Higher = bumpier ride.",
  },
  max_drawdown: {
    term: "Max drawdown",
    short: "Largest peak-to-trough loss in the equity curve.",
    body:
      "The worst decline from a previous high. -50% means an investor who entered at the peak watched their account halve before any recovery. " +
      "<10% tiny; 25–50% requires real stomach; 50%+ and most investors would have quit.",
  },
  ic: {
    term: "IC (Information Coefficient)",
    short: "Daily rank correlation between scores and realised returns.",
    body:
      "Per day, did the stocks ranked higher actually go up more? Averaged across days. A real daily signal usually has mean IC of 0.01–0.05; higher should make you suspicious.",
  },
  ic_ir: {
    term: "IC IR",
    short: "Consistency of the IC signal (annualised t-stat).",
    body:
      "mean IC / std IC × √252. >0.5 noteworthy; >1.0 institutionally interesting; >2.0 very rare (investigate for leakage); <0 the signal works in reverse.",
  },
  hit_rate: {
    term: "Hit rate",
    short: "Fraction of predictions with the correct sign.",
    body:
      "Random = 50%. A real but small edge = 51–54%. Above 55% out-of-sample almost certainly means a bug. A high hit rate can still lose money if the losers are bigger than the winners — which is why we don't optimise for it.",
  },
  hit_ratio: {
    term: "Hit ratio (daily)",
    short: "Fraction of days the portfolio made money.",
    body:
      "Different from hit rate: this is the share of DAYS the book was up. Random walk ~50%; real edge 50–55%. A portfolio can have a 45% daily hit ratio and still be positive if the up days are bigger than the down days — so don't optimise for it.",
  },
  sortino: {
    term: "Sortino ratio",
    short: "Like Sharpe, but only penalises downside volatility.",
    body:
      "Return per unit of DOWNSIDE risk (upside swings aren't treated as risk). Always at least as large as Sharpe; the gap tells you how asymmetric the return distribution is.",
  },
  calmar: {
    term: "Calmar ratio",
    short: "Annualised return ÷ |max drawdown|.",
    body:
      "Return earned per unit of worst-case loss. Higher is better. Useful when you care more about the deepest hole than about day-to-day volatility.",
  },
  turnover: {
    term: "Turnover",
    short: "Sum of |Δweight| across positions per day.",
    body:
      "How much of the book is traded each day. Turnover of 2.0 means both legs fully rebalanced. Higher turnover = higher transaction costs, which eat the edge.",
  },
  inverse: {
    term: "Inverse strategy",
    short: "What you'd get doing the exact opposite of every position.",
    body:
      "Negate every weight: long becomes short and vice-versa. If the real strategy were systematically anti-predictive, its inverse would make money. " +
      "Plotting both is an honesty check — for a genuine signal the inverse should lose; if the inverse looks much better, the signal is backwards or noise.",
  },
  per_stock_strategy: {
    term: "Score strategy ($1 growth)",
    short: "Growth of $1 trading this one stock on the model score.",
    body:
      "Each day: go long when the score is positive, short when negative, then realise the next day's return. Compound those daily results from $1. " +
      "Compared against simply buying and holding the stock. This is a single-name illustration of whether the score has timed THIS stock — not the portfolio strategy.",
  },
} satisfies Record<string, GlossaryEntry>;

export type GlossaryKey = keyof typeof glossary;
