# Concepts & terminology

Reading order: this file → `USAGE.md` → `PROJECT_LOG.md`.

If you've never built or evaluated a stock-prediction system before, read this
end-to-end. Every term used elsewhere in the project is defined here, with the
mental model behind it and the gotchas the maths textbooks gloss over.

> **One sentence the whole project is built around:** *most published backtests
> are wrong, almost always because of subtle ways the future leaks into the
> past, and the discipline of avoiding that leakage is harder than building the
> model itself.*

---

## 1. The big picture

We try to predict, for every stock in the S&P 500, **what its return will be
over the next h trading days**. We do this every day, for many stocks at
once. Then we build a portfolio that:

- **Buys (goes long)** the stocks we predict will go up the most.
- **Sells (goes short)** the stocks we predict will go down the most.

If our predictions have any edge, this portfolio makes money on average. If
they don't, it loses to "just buy SPY and hold it" (the S&P 500 ETF). Most
strategies lose to SPY after costs.

### Why long *and* short ("dollar-neutral")?

Going long *and* short the same dollar amount cancels out general market
movement. If SPY goes up 1%, both my longs and my shorts go up roughly 1%,
which roughly cancels in the P&L. What's left is the *cross-sectional* signal:
did the stocks I picked outperform the stocks I shorted? That's a much cleaner
test of whether the model knows anything.

---

## 2. Returns

### Simple return vs log return

If a stock goes from $100 to $102:
- **Simple return** = `(102 - 100) / 100 = 0.02 = +2%`
- **Log return** = `ln(102/100) ≈ 0.0198 ≈ +1.98%`

For small changes they're nearly identical. We use **log returns** for
modelling because:

- They're additive: `log(P3/P1) = log(P3/P2) + log(P2/P1)`. Simple returns
  compound, which is mathematically annoying.
- They're approximately normally distributed for short windows, which most
  statistical models prefer.

We use **simple returns** for portfolio P&L because that's literally what
your account balance does.

### Forward return

For each stock and each day `t`, the **h-day forward return** is the return
*from `t` going forward `h` days*. This is what we try to predict.

- For h=1: return from close of day `t+1` to close of day `t+2`.
- For h=5: return from close of day `t+1` to close of day `t+6` (~one week).
- For h=21: return from close of day `t+1` to close of day `t+22` (~one month).

Notice that we never use the close of day `t` itself in the forward return
window. If we did, it would be **same-day lookahead**: the model would have
seen the very price it's trying to predict. (See "leakage" below.)

---

## 3. Predictions: what the model produces

For each (date, ticker) the model outputs a single number — a **score** —
which can be:
- positive (we think this stock will go up),
- negative (we think it will go down),
- bigger in magnitude = more confident.

The score has no natural units. We use it to **rank** stocks within each day:
buy the top-K names, short the bottom-K names.

### Hit rate

**Hit rate** = the fraction of predictions where the sign was right.

- Random guessing: 50%.
- A model with a real (small) edge: 51–54%.
- Anything above 55% on out-of-sample data with proper validation is almost
  certainly a bug. Hedge funds with massive resources celebrate 53%.

Hit rate is **easy to misinterpret**. A model can have 65% hit rate but lose
money, if the 35% of trades that go wrong lose 4x as much as the 65% that
go right. That's why we don't optimise for hit rate.

### IC (Information Coefficient)

IC is the **Spearman rank correlation between predicted scores and realised
returns**, computed *per day across the cross-section of stocks*. Then we
average across all days.

In plain English: on each day, did the stocks you ranked higher actually go
up more? IC says how monotonic that relationship is.

- IC of 0 = no relationship between your scores and what happened.
- IC of 1 = perfect ranking.
- IC of -1 = perfectly wrong ranking (which is information — just flip the
  sign).
- A real trading signal usually has **mean IC of 0.01 to 0.05** on daily
  cross-sectional data. Anything higher should make you suspicious.

### IC IR (Information Ratio of IC)

Mean IC alone doesn't tell us how *consistent* the signal is. A model with
mean IC = 0.02 ± 0.20 every day is noisy garbage that happens to be right
on average. A model with mean IC = 0.02 ± 0.01 every day is a real signal.

**IC IR = mean IC / std IC × sqrt(252)**

This is an annualised t-statistic on the IC series. Numbers to remember:

- **IC IR > 0.5**: noteworthy.
- **IC IR > 1.0**: institutionally interesting.
- **IC IR > 2.0**: very rare; investigate for leakage before celebrating.
- **IC IR < 0**: the signal works in reverse (or is noise).

In Phase 2 of this project we get IC IR +2.45 on the 5-day horizon. That's
suspicious enough that we don't actually trust it without more out-of-sample
validation (it could be a fluke of the 2018–2024 window).

---

## 4. Portfolio construction

A score per stock is not a strategy. We have to convert scores into
**weights** (how much of each stock to hold).

### Top-K equal-weight

Simplest method: on each day, take the top-K stocks by score, give each one
`1/K` long; take the bottom-K and give each `-1/K`. Done.

- Pros: extremely transparent, easy to reason about, no parameters.
- Cons: ignores conviction (a name with score 10x another gets the same
  weight) and ignores risk (a high-vol stock gets the same exposure as a
  low-vol one).

### Vol-scaled top-K

Same selection, but weight ∝ |score| / volatility. A high-conviction,
low-vol stock gets more capital than a low-conviction, high-vol one. Then
normalise so each side sums to a target gross exposure.

This *usually* improves Sharpe because it shrinks risky positions. Built in
`backtest/portfolio.py::vol_scaled_weights`.

### Sector caps

If 8 of your top-10 longs are all in Tech, your "stock-picking signal" is
really a "Tech beta" signal in disguise. We cap each sector's gross weight
to a fraction (default 30%) to force diversification.

Built in `backtest/portfolio.py::apply_sector_caps`.

### Minimum trade threshold

A noisy signal will produce tiny day-to-day weight changes (-0.123 → -0.122
→ -0.124). Each tiny change costs transaction fees, which eat the edge. We
ignore weight changes below a threshold (default 0.5%) — if today's signal
isn't materially different from yesterday's, don't trade.

Built in `backtest/portfolio.py::apply_min_trade_threshold`.

### Why these matter

The Phase 2 ensemble produces an OK signal but a **terrible** backtest
because:
1. Equal-weight 1/5/21d ensemble drowns the 5d edge in the 21d noise.
2. No vol scaling: a 5% move on a wild biotech is treated the same as 5%
   on Coca-Cola.
3. Daily rebalancing on noisy signals burns 10s of bps per day in costs.
4. No sector diversification = the strategy is implicitly betting on whichever
   sector the model happens to overweight.

Phase 5 fixes all four. See `PROJECT_LOG.md` for actual results.

---

## 5. Costs

A backtest without costs is fiction. Real costs in retail-grade US equities:

- **Commission**: typically 0–1 bp per side (a "bp" = 0.01% = 1/10,000).
- **Bid-ask spread**: 1–10 bps per side depending on the stock's liquidity.
  Crossing the spread (taking the offer instead of waiting) costs you half
  the spread. Our default assumes 4 bps.
- **Slippage / market impact**: in retail size, ~1 bp. Big institutions move
  the market when they trade and pay much more.

Total: **~6 bps per side, 12 bps round trip** by default. A strategy that
turns over 100% of its book every day pays ~30% of NAV per year in costs.
Most signals never recover this.

When you see the backtest's "ann_return: -10%" think: "if my pre-cost return
were +20%/yr, the post-cost would be -10%." A small change in cost assumption
flips the result.

---

## 6. Metrics for evaluating a strategy

### Sharpe ratio

`Sharpe = mean_daily_return / std_daily_return × sqrt(252)`

A unit-less number measuring return per unit of volatility, annualised. Rough
guide for *out-of-sample, after costs*:

- **< 0**: loses money.
- **0–0.5**: mediocre, beats cash by less than its swings.
- **0.5–1.0**: real strategies live here.
- **1.0–2.0**: very good, you have something institutional.
- **> 2.0**: suspect leakage before celebrating.

### Sortino ratio

Like Sharpe but only penalises *downside* volatility (because upside swings
aren't a "risk"). Always at least as big as Sharpe.

### Max drawdown

The biggest peak-to-trough loss in the strategy's equity curve. A -50%
drawdown means an investor who started at the peak watched their account
halve before recovering (if it ever did).

- < 10%: tiny.
- 10–25%: tolerable for many investors.
- 25–50%: requires real stomach.
- 50%+: most investors would have closed the strategy long ago, missing the
  recovery.

### Calmar ratio

`Calmar = annualised_return / |max_drawdown|`

A risk-adjusted return that focuses on the worst loss. Roughly: "how many
units of drawdown am I taking per unit of annual return?"

### Turnover

Sum of |Δweight| across all positions per day. A turnover of 2.0 means the
portfolio fully rebalanced both legs that day. Higher turnover = higher
costs.

### Hit ratio (different from "hit rate")

Sometimes "hit ratio" appears in the backtest summary, meaning the fraction
of **days** where the portfolio made money. Random walk: ~50%. Strategies
with real edge: 50–55%. **Note**: a portfolio can have hit ratio 45% and
positive ann_return if the 55% loser days are small and the 45% winner days
are big. Don't optimise for it.

---

## 7. Leakage: how backtests lie

This is the *one thing* that separates a real backtest from a fantasy. Below
are the failure modes the project actively defends against. Every one of
them has tripped up academic papers and hedge fund pitches.

### a) Same-day lookahead

If you use today's close to "predict" today's close, your model will be
99.99% accurate. The fix: predict only what you couldn't have observed at
prediction time.

### b) Label window overlap (the trickiest)

Suppose horizon = 21 days, and you train on data through end of June and
test on July. Your last training label uses prices through late July — but
those are in the test window. The model has effectively peeked at test
prices through its labels.

**Fix**: in cross-validation, drop training labels whose window overlaps
the test window. Then add a buffer ("embargo") so even nearby autocorrelation
can't leak. *Crucially* the buffer must be measured in **trading days**, not
calendar days, otherwise a 10-day buffer is only ~7 trading days and a 21-day
horizon still leaks. We had this exact bug in v0.1 and fixed it.

### c) Survivorship bias

If you backtest on "stocks that are in the S&P 500 today", you've implicitly
excluded every company that went bankrupt or got delisted in the test
period. Those bankrupt companies would have hammered your performance. The
biased backtest looks ~3–5% better per year than reality.

**Fix**: use the *point-in-time* index membership — only consider stocks
that were in the index *on each historical date*. This project reconstructs
S&P 500 membership from Wikipedia's change log.

### d) Training on the test set (the cardinal sin)

Tuning a hyperparameter, looking at the test-set result, then tuning more,
and iterating until the test result looks good — is just training on the
test set in slow motion. Anyone reporting a great backtest like this is
either lying or fooling themselves.

**Defence**: a single "holdout" period that is *never touched* by anyone
during development. The final number reported is the holdout number. The
project ships `validation/stress.py::holdout_split_dates` for this.

### e) Look-ahead in features

If you compute a feature using a "current" earnings surprise on Monday,
that number was probably released on Friday afternoon — but a real trader
who only sees Monday's news at 9:30 AM Monday doesn't know it yet. Point-in-
time features matter.

We don't use point-in-time fundamentals in this project (yfinance `.info`
returns *current* data, not historical). The dashboard's "Short ratio" and
"P/E" panels are *current as of fetch*, which means they're forward-leaking
when displayed against historical prices. We surface this caveat in the UI
and in `USAGE.md`. We do not feed any of these fields into the model.

### f) Cost-free backtest

Costs eat alpha. A noisy signal that turns over 50% per day will burn 30
bps/day in costs (default) = 75%/year. Any "successful" backtest without
realistic costs is fiction.

### g) Cherry-picking the period

A backtest from 2017 to 2021 looks very different from one from 2007 to
2011. The project always shows yearly breakdown so you can see which
periods carried the result.

---

## 8. Walk-forward CV (the right way to validate)

You can't shuffle your data and do random k-fold cross-validation on time
series — that would let the model train on the future and test on the past.

**Walk-forward** instead:

```
Train period 1 → Embargo → Test period 1
                                   ↓
              Train period 2 (includes period-1 dates) → Embargo → Test period 2
                                                                      ↓
                                                       Train period 3 → ...
```

- Each test period uses only data from before it.
- The embargo prevents label-window overlap (see §7b).
- The training set grows over time ("expanding window") so later folds get
  more data, mimicking the real-world case where you'd retrain periodically.

We use a 3-year initial training window, 6-month test windows, 25-trading-day
embargo (covers our maximum 21-day label horizon plus a safety margin).

---

## 9. The ensemble: why multiple horizons?

We train *three separate models*, one each for the 1d, 5d, and 21d
forward-return prediction. Why not just one?

- **Different horizons capture different signals.** 1d is often
  microstructure / mean-reversion. 5d is often momentum. 21d is often
  longer-term factors. A model forced to pick one is leaving information on
  the table.
- **Cross-checks for leakage.** If 1d looks great but 21d looks terrible on
  the same data, something is fishy. (And in fact our real-data run shows
  exactly this pattern, suggesting the 21d horizon has no usable signal —
  or that the cost drag at h=21 with our rebalance cadence is killing it.)
- **Combining predictions reduces noise.** Averaging cross-sectional
  z-scores from multiple horizons gives a smoother score than any single
  horizon.

The Phase 5 ensemble drops horizons whose out-of-sample IC IR is ≤ 0 and
weights the surviving ones by their IC IR. This avoids letting bad
horizons drag down good ones.

---

## 10. Why our backtest still loses money

A summary, because this is important and unintuitive:

1. The signal *is* real on the 5-day horizon (IC IR +2.45 is strong).
2. The 1-day signal is marginal; the 21-day signal is negative.
3. **Equal-weighting them** in the ensemble dilutes the 5d edge with noise
   from the others. Fix: IC-IR-weighted ensemble (Phase 5a).
4. **Daily rebalancing on a 5-day signal** rebalances every day for a
   prediction that only refreshes every 5 days. Cost drag dominates the
   edge. Fix: rebalance cadence matched to horizon (already done in engine,
   but needs the right horizon to dominate the ensemble; Phase 5a).
5. **No risk control**: a few outlier-bad days swamp many small good days.
   Fix: vol-scaled sizing (Phase 5b).
6. **No sector diversification**: the "model" sometimes becomes a sector
   bet. Fix: sector caps (Phase 5b).

Phase 5 implements 3–6. Even with all of them, the strategy might still lose
to SPY net of costs — that's the honest answer. Going beyond it would
require either much better data (which costs money) or a multi-month
research effort to find and validate a new feature family.

---

## 11. What "high win rate" actually means

When a non-quant says "high win rate" they usually mean one of three things:

- **Hit rate**: % of predictions with the right sign. Optimising this leads
  to strategies that scalp tiny consistent wins and then take catastrophic
  losses (e.g. "I sold puts naked for 5 years, made 8% a year, then lost
  everything in March 2020").
- **Daily hit ratio**: % of days the portfolio is up. Same trap.
- **Calmar / Sharpe**: actually risk-adjusted return. **This is what you
  want to maximise**, not raw win rate.

If you ever see a product promising "85% win rate", run.

---

## 12. Things this project explicitly cannot do

- **Predict the level of an index** (S&P 500 going up tomorrow as a whole).
  We forecast cross-sectional differences between stocks, not market
  direction.
- **Day-trade or intraday-trade.** Daily bars only.
- **Account for tax**, dividends differently, fees beyond the configured
  bps, capital constraints. The backtest assumes you can put on a $1 long
  and $1 short for every $1 of capital, infinitely. Real margin requirements
  are stricter.
- **Beat the market**. Said three times because it matters.

---

## 13. Further reading

If this introduction made you want more:

- **Marcos López de Prado**, *Advances in Financial Machine Learning*
  (the source for our purged + embargoed CV, triple-barrier labels, and
  many other anti-leakage defences).
- **Ernest Chan**, *Algorithmic Trading: Winning Strategies and Their
  Rationale* (practical, retail-friendly).
- **Andrew Lo et al.**, *A Non-Random Walk Down Wall Street* (gentle
  introduction to whether stock prices are predictable at all).
- **AQR Capital "10 Practical Investment Insights"** white papers (free,
  honest, written by practitioners).
- Avoid 95% of "algorithmic trading" YouTube and Medium content. The
  signal-to-noise is terrible and most of it falls afoul of §7.
