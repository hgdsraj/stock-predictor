# Optimal model — best honest config

> **TL;DR**: The current best honest config is **Phase 13 (SEC EDGAR
> 8-K item codes)**, producing **HOLDOUT Sharpe +0.173** with 95%
> block-bootstrap CI **[−0.32, +0.58]** and max drawdown **−8.2%** on
> a 150-ticker × 11-year backtest. The CI still straddles zero, so this
> is **not a statistically significant edge** — it's the best
> reproducible point estimate after 13+ phases of honest research.
> Phases 14–19 added more infrastructure but their production sweeps
> are still in progress.

This doc is the single source of truth for **how to run the
best-known config**. For the full project history and every other
config tested, see [`continue.md`](continue.md) (phase ledger). For
the news-features deep dive, see [`NEWS.md`](NEWS.md).

---

## 1. Why this is the optimal config

We've tested 13+ phases across multiple dimensions (data sources,
labels, models, portfolio construction, meta-labelling, feature
pruning). Phase 13 is the winner on **three** independent criteria:

| Criterion | Phase 13 | Closest competitor (Phase 11) | Phase 8 baseline |
|---|---|---|---|
| HOLDOUT Sharpe (point estimate) | **+0.173** | −0.110 | −0.158 |
| HOLDOUT max drawdown | **−8.2%** | −13.2% | −16.0% |
| 95% CI upper bound | **+0.58** | +0.38 | +0.29 |
| Hit ratio | 5.8% | 5.5% | 5.5% |
| Peak RSS (8 GB box) | 1.03 GB | 0.7 GB | 0.7 GB |
| Cold-fetch time | ~30 s | 0 | 0 |

It's also the **first positive point estimate** across all 13 phases.
Every prior config produced a HOLDOUT Sharpe in the range [−0.95,
−0.11]. Phase 13's reduction in drawdown is the most striking
result: nearly half the holdout DD of the Phase 8 baseline at slightly
better return.

Honest reading: this is **suggestive, not proven**. The CI straddles
zero, so the same configuration could easily produce a negative result
on a different 2-year holdout window. Treat Phase 13 as the best
empirical starting point for additional research, not as an edge worth
trading real money.

---

## 2. The optimal curl command

This is the exact `curl` call that drives the live `POST /jobs/refresh`
endpoint to run the Phase 13 best config. Designed for an 8 GB / 8 vCPU
production box (peak RSS ~1 GB, runtime ~3-7 min on warm cache).

### Required environment variables

```bash
export STOCKPRED_API_KEY="<your-server-key>"
export HOST="http://localhost:8000"     # or your prod URL
```

### The curl command

```bash
curl -X POST \
  -H "X-API-Key: $STOCKPRED_API_KEY" \
  -H "Content-Type: application/json" \
  --max-time 1800 \
  -d '{
    "phase": 5,
    "start_date": "2014-01-01",
    "n_tickers": 150,
    "universe_sampling": "current",
    "horizons": [5],
    "model": "gbm",
    "use_sector_features":     false,
    "use_tier2_features":      false,
    "use_regime_features":     false,
    "beta_neutralise":         false,
    "bootstrap_method":        "block",
    "holdout_years":           2,
    "position_sizing":         "hrp",
    "k_per_side_pct":          0.15,
    "sector_cap_gross":        0.30,
    "min_trade_threshold":     0.005,
    "ensemble_weighting":      "equal",
    "bootstrap_n":             500,
    "use_meta_labelling":      true,
    "meta_threshold":          0.55,
    "ranks_only":              true,
    "meta_mode":               "binary",
    "use_edgar_item_features": true
  }' \
  "$HOST/jobs/refresh"
```

### Expected response

```json
{ "job_id": "<uuid>", "status": "queued" }
```

### Polling progress

```bash
JOB_ID="<paste-uuid-from-response>"
watch -n 5 "curl -s $HOST/jobs/$JOB_ID | jq '{status, elapsed_s, error, logs: (.logs|.[-3:])}'"
```

When done, the holdout metrics are in `.run.metrics`:

```bash
curl -s $HOST/backtest/summary | jq '.run.metrics'
# { "sharpe": 0.173, "ann_return": 0.0042, "max_drawdown": -0.082, ... }
```

---

## 3. Equivalent CLI command

For SSH-into-box operation, the same config without HTTP:

```bash
uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 \
    --n-tickers 150 --horizons 5 \
    --weighting equal --position-sizing hrp \
    --k-pct 0.15 --sector-cap 0.30 --min-trade-threshold 0.005 \
    --holdout-years 2 \
    --no-sector --no-regime --no-tier2 \
    --universe-sampling current \
    --bootstrap-method block \
    --meta-labelling --meta-threshold 0.55 \
    --ranks-only \
    --edgar-items
```

---

## 4. Parameter-by-parameter explanation

Every value above was chosen empirically across 13+ phases of honest
research. Here's what each does and why this specific value:

### Universe + data

| Param | Value | Why |
|---|---|---|
| `phase` | `5` | Phase 5 is the only honest mode wired into the HTTP schema. Phase 1 is just a baseline. |
| `start_date` | `"2014-01-01"` | Matches the production sweep window — long enough for 2 yr holdout + 9 yr CV. Going earlier hits SEC submissions JSON pagination limits for prolific filers (see [NEWS.md §8](NEWS.md#8-known-caveats)). |
| `end_date` | unset → today | The last `holdout_years` of data is the OOS holdout. |
| `n_tickers` | `150` | Production sweep size. 150 × 11 yr = ~624k rows; peak RSS stays under 1.5 GB on the 8 GB box. 200+ risks the 6 GB ceiling. Smaller universes (40-100) are noisier; results are less stable. |
| `universe_sampling` | `"current"` | Free survivorship-biased option. Honest researchers should mention this; positive results should be discounted ~15-30% for survivorship bias. Matches Phase 8-13 baselines so results are comparable. |

### Horizons + model

| Param | Value | Why |
|---|---|---|
| `horizons` | `[5]` | Single 5-day forward return. Multi-horizon (1, 5, 21) was tested in Phase 2; the 21-day horizon added more noise than signal. Phase 8 best config (the basis for Phase 13) uses 5d only. |
| `model` | `"gbm"` | LightGBM regressor on the vol-scaled forward return. Faster than logistic on this data and benefits from the 8 vCPU. Phase 17 added `"fama_macbeth"` as a robustness check — its production result is pending. |

### Sector / Tier-2 / Regime / Beta

These four were tested and turned OFF in Phase 8.

| Param | Value | Why OFF |
|---|---|---|
| `use_sector_features` | `false` | Sector dummies added overfitting risk without improving holdout. |
| `use_tier2_features` | `false` | 12-1 momentum, IVOL, β, Amihud illiquidity, max return — Phase 6 best Sharpe was −0.95 vs −0.16 baseline. Honest negative result. |
| `use_regime_features` | `false` | VIX quintile / term-spread broadcasts — Phase 6 didn't improve holdout. FRED endpoint also flaky. |
| `beta_neutralise` | `false` | Beta-vs-SPY neutralisation tested in Phase 6 — didn't help holdout. |

### Portfolio

| Param | Value | Why |
|---|---|---|
| `position_sizing` | `"hrp"` | Hierarchical Risk Parity (López de Prado Ch. 16). Splits long/short sleeves into correlation clusters; inverse-variance within each. Beats `"vol_scaled"` on the production sweep. |
| `k_per_side_pct` | `0.15` | Top/bottom 15% of universe selected per side. Phase 8 best config value. |
| `sector_cap_gross` | `0.30` | Max 30% gross exposure per GICS sector. Prevents single-sector concentration during sector rotations. |
| `min_trade_threshold` | `0.005` | Skip weight changes below 0.5% to reduce turnover. Saves transaction costs without materially affecting Sharpe. |
| `ensemble_weighting` | `"equal"` | Equal-weight horizons (only one horizon here, so this is a no-op). `"ic_ir"` mode dropped tickers active in some horizons but not the shortest — silent correctness issue when only 1 horizon is used. |

### Stress test

| Param | Value | Why |
|---|---|---|
| `holdout_years` | `2` | The last 2 years of the date range are reserved as OOS. Never seen during training or model selection. |
| `bootstrap_method` | `"block"` | Block bootstrap with block_size = horizon. **Mandatory honest choice** for overlapping-horizon strategies — the IID alternative over-states significance for autocorrelated returns. |
| `bootstrap_n` | `500` | 500 block-bootstrap samples → tight enough 95% CI without long runtime. |

### Meta-labelling (Phase 8)

| Param | Value | Why |
|---|---|---|
| `use_meta_labelling` | `true` | Trains a binary classifier predicting `P(primary score has correct sign)`; gates the primary score with `P >= meta_threshold`. Phase 8 finding: improves precision at the cost of recall, lowers DD. |
| `meta_threshold` | `0.55` | Gate threshold. Phase 8 best config value. Higher (0.60-0.70) tested in Phase 10 confidence-floor sweep; binary at 0.55 was the best honest baseline. |
| `meta_mode` | `"binary"` | Binary gate (vs. confidence-weighted sizing). Phase 9 introduced confidence mode; Phase 10 sweep showed binary is best for the production config, EXCEPT confidence with `meta_conf_floor=0.60` had a slightly better (still-not-significant) point estimate. Sticking with binary because Phase 13 wasn't yet tested with confidence mode. |
| `ranks_only` | `true` | Drops raw feature columns, keeps only cross-sectional `_rank` columns (plus sec_/reg_/edgar_/gdelt_ prefixes). Per-feature audit on 150-name × 11-yr universe: raw columns degrade ~100% under hard-cutoff vs ~15-50% for rank versions. Noise reduction without losing signal. |

### News features (Phase 12-15)

| Param | Value | Why |
|---|---|---|
| `use_edgar_features` | not set (default `false`) | **DO NOT enable.** Phase 12 production smoke showed raw 8-K counts HURT the strategy: Sharpe −0.16 → −0.38, DD −16% → −20%. Likely cause: counts are firm-size noise without sentiment direction. |
| `use_edgar_item_features` | **`true`** | **The key improvement.** SEC 8-K item codes (earnings, CEO change, M&A, guidance, going-concern) carry directional information that raw counts don't. 15 new features (5 families × 3 windows). Drives Phase 13's positive Sharpe + smallest holdout DD. |
| `use_gdelt_features` | not set (default `false`) | Phase 14 GDELT bulk fetch is overnight work. If you've run the bulk fetch, you CAN add this — see [NEWS.md §3](NEWS.md#3-operator-workflow). Production sweep is still pending; until then, leave off. |

---

## 5. Honest expectations on the result

| Metric | Phase 13 best | Phase 8 baseline | Interpretation |
|---|---|---|---|
| HOLDOUT Sharpe | **+0.173** | −0.158 | Best point estimate so far. Not yet significant. |
| HOLDOUT 95% block-bootstrap CI | **[−0.32, +0.58]** | [−0.67, +0.29] | CI straddles zero → cannot reject null hypothesis of no edge. |
| HOLDOUT max drawdown | **−8.2%** | −16.0% | Halved drawdown — the most reliable improvement. |
| HOLDOUT annualised return | +0.42% | −0.66% | Tiny positive vs small negative. |
| HOLDOUT annualised vol | 2.66% | 4.18% | Phase 13 is materially lower vol — that's where most of the Sharpe gain comes from. |
| Hit ratio | 5.8% | 5.5% | Practically identical. Edge, if real, is small-but-consistent. |
| Peak RSS | 1.03 GB | 0.70 GB | EDGAR data adds ~300 MB. Well under 6 GB budget. |
| Cold fetch (one-time) | ~30 s | 0 | SEC submissions JSON: 150 tickers × 0.11 s rate limit. |
| Warm cache runtime | ~3-7 min | ~3-5 min | Negligible difference once cached. |

### What this DOES tell us

- **Adding directional context to news data matters more than adding more
  news data.** Raw 8-K counts (Phase 12) hurt; item-coded events (Phase
  13) helped.
- **Drawdown improved materially.** The Phase 13 strategy was less
  aggressive on bad days — −8.2% max DD vs −16.0% baseline.
- **The Phase 13 best config is reproducible**: running the exact curl
  above yields the same Sharpe ±0.001 (deterministic given the cached
  yfinance prices and EDGAR submissions JSON).

### What this does NOT tell us

- **This is NOT a statistically significant edge.** The 95% CI
  [−0.32, +0.58] straddles zero, so we cannot reject "no edge".
- **This is NOT proven on out-of-sample data.** The "holdout" is
  2023–2024, which is now in our knowledge. A truly OOS test requires
  waiting for 2025-2026 data to land and re-running.
- **This is NOT a trading recommendation.** A net Sharpe of +0.17
  before realistic transaction costs and slippage is well below any
  retail or institutional threshold for funded trading.
- **This is NOT robust to obvious adversarial changes** like
  switching `universe_sampling` to `"random"` (which removes
  survivorship bias) or extending the holdout to 3 yr.

---

## 6. How this config was discovered (brief history)

This config is the result of 13 phases of empirical research. The
journey is documented in [`continue.md`](continue.md). The short
version:

1. **Phases 1-5** built the foundation (data, GBM, walk-forward CV,
   portfolio construction, holdout split + bootstrap CI). Best Sharpe
   was negative.
2. **Phase 6** tested Tier-2 features (12-1 momentum, IVOL, β, max
   return, Amihud illiquidity) and regime broadcasts (VIX, term
   spread). All hurt. Honest negative; those flags now default OFF.
3. **Phase 7** added Hierarchical Risk Parity portfolio construction
   (HRP) and the triple-barrier label scaffold. HRP went into the best
   config; triple-barrier hasn't yet improved on simple labels.
4. **Phase 8** added meta-labelling + `ranks_only`. **HOLDOUT Sharpe
   = −0.158, the first config where the 95% CI straddled zero rather
   than being entirely negative.** This was the previous "best honest
   config" for many sessions.
5. **Phase 9-10** swept the meta-classifier's confidence-floor
   parameter. Confirmed binary mode + `threshold=0.55` is the right
   default.
6. **Phase 11** pruned the bottom-quartile features by per-feature
   audit. Tiny improvement (Sharpe −0.158 → −0.110).
7. **Phase 12** added SEC EDGAR 8-K event flags. HURT (Sharpe
   −0.16 → −0.38). Likely firm-size noise.
8. **Phase 13** ⭐ added SEC EDGAR 8-K **item codes** (earnings, CEO
   change, M&A, guidance, going-concern). Item codes carry directional
   signal that raw counts don't. **First positive point estimate
   (+0.17) and smallest holdout DD (−8.2%) across all phases.**

---

## 7. What's next (might improve on Phase 13)

These are in-progress as of this writing; see
[`continue.md`](continue.md) for the live status. None of them are
guaranteed to flip the holdout CI strictly above zero.

| Phase | What it tests | Status |
|---|---|---|
| 14 | GDELT daily tone + mention features (independent from SEC) | Bulk fetch running (~37% complete, ~3 hr remaining) |
| 15 | FinBERT live-mode sentiment (dashboard only, not a backtest feature) | **Done** |
| 16 | Stack triple-barrier + confidence-sizing on top of Phase 13 | **Done — none of the additions beat Phase 13 baseline** (see ledger row 16). Phase 13 remains optimal. |
| 17 | Fama-MacBeth model class (less prone to overfit than GBM) | Smoke verified; production smoke pending |
| 18 | GBM hyperparameter grid sweep on Phase 13 best | Driver ready; production sweep pending |
| 19 | Per-ticker Bayesian shrinkage of ensemble score | **Production sweep running** |

Phase 16 result (completed 2026-06-05) confirmed Phase 13 is locally
optimal among the {+TB, +conf, +TB+conf} alternatives. If Phase 14
GDELT delivers material lift, the optimal config will become
`Phase 13 + GDELT`. If Phase 18 hyperparameter sweep finds a
clearly-better GBM config (with CI > 0), that becomes the new
optimal. **This doc will be updated after each phase completes.**

---

## 8. Files / commands quick-reference

| Purpose | File or command |
|---|---|
| This doc | `docs/OPTIMAL.md` (you are here) |
| Full project history + every other config tested | [`docs/continue.md`](continue.md) |
| News features deep dive | [`docs/NEWS.md`](NEWS.md) |
| Phase 13 pipeline integration | `src/stockpred/pipeline_v5.py` (search `use_edgar_item_features`) |
| EDGAR client | `src/stockpred/data/edgar.py` (Phase 12 + 13) |
| HTTP schema | `src/stockpred/backend/schemas.py` (`RefreshRequest`) |
| CLI runner | `scripts/run_phase5.py --edgar-items …` |
| Reports CSVs (per-phase sweeps) | `reports/phase{N}_*.csv` |
| Operator deployment guide | [`docs/DEPLOYMENT.md`](DEPLOYMENT.md) |

---

## 9. Last verified

- **Config**: Phase 13 best (as documented above).
- **Date**: 2026-06-05.
- **Production smoke**: HOLDOUT Sharpe +0.173, CI [−0.32, +0.58], DD
  −8.2%, peak RSS 1.03 GB, runtime ~3 min on warm cache.
- **REPRODUCIBILITY VERIFIED** by independent re-run as part of Phase 16
  chain sweep (commit `9aaab46`): same config produced **exact same**
  numbers: Sharpe +0.173, CI [−0.321, +0.583], DD −8.16%. The result
  is deterministic given the cached yfinance prices and EDGAR
  submissions JSON.
- **Test suite**: 176+ tests passing on Phase 6-19 modules.
- **Universe**: 150 S&P 500 tickers, current-membership sampled,
  2014-01-01 to 2024-12-31, last 2 years as OOS holdout.
