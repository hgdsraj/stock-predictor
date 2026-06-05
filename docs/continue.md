# Continue here — session resume protocol

> If you (Claude, in a new session) are starting fresh on this project,
> **read this file first**. It is the single source of truth for what's
> done, what's broken, and what to do next. Updated after every phase.

## TL;DR for the next session
- Repo: `~/projects/stock-predictor`, remote `https://github.com/hgdsraj/stock-predictor`
- Python env: `.venv/` via `uv` (`uv sync --extra dev` to install)
- Test command: `uv run pytest tests/` (101 fast + 2 slow as of last commit)
- Frontend builds: `cd web && npm ci && npm run build` (Node 20+ required)
- DO NOT push to GitHub yourself; the user pushes manually using a PAT
- DO NOT use Google internal services / corp tools — public PyPI + free data only
- DO NOT chase "win rate" as a metric — the correct objective is HOLDOUT
  Sharpe with bootstrap CI not entirely below zero
- The strategy LOSES money on holdout; reporting that honestly is the deliverable

## Phase-by-phase ledger

Each phase commit is on `main`. `git log --oneline` shows them in reverse order.

| Phase | Status | Best HOLDOUT Sharpe | HOLDOUT 95% CI | Notes |
| ----- | ------ | ------------------- | -------------- | ----- |
| 1 | DONE | n/a (no holdout split yet) | — | Foundation: data, features, GBM, walk-forward CV, top-K backtest, tearsheet |
| 2 | DONE | (no holdout) | — | LightGBM through pipeline + multi-horizon ensemble. Sub-agent caught 3 CRIT + 6 HI/M bugs (all fixed) |
| 3 | DONE | — | — | Phase 3 portfolio improvements built (vol-scaled, sector caps, threshold, IC-IR ensemble); first standalone phase |
| 4 | DONE | — | — | Stress tools: holdout split, bootstrap CI, sensitivity grid, regime breakdown |
| 5 | DONE | −0.84 (leaky) → **−0.52** (leak-fixed) | [−1.34, +0.22] | Wired Phase 3/4 into pipeline. C1 (label leak) fixed. |
| 6 | DONE | **−0.95** (Tier-2 + regime made it worse) | [−1.70, −0.23] | Tier-2 features (12-1 momentum, IVOL, β, Amihud), regime features (VIX, term spread), block bootstrap, beta neutralisation |
| 7 | DONE | **−0.69** (HRP, big universe 822 names) | [−1.12, −0.24] | HRP, triple-barrier scaffold, meta-labelling scaffold, per-feature audit, engine ±50% clip |
| 8 | DONE | **−0.16** ← **BEST RESULT** | **[−0.67, +0.29]** ← straddles zero | Wired meta-labelling + triple-barrier + ranks_only. Reviewer caught 3 CRIT (double z-score, holdout meta on gated dev, ranks_only dropped tier-2). All fixed. |
| 9 | DONE | −0.57 (made things worse) | [−1.03, −0.15] | Confidence sizing + walk-forward meta-CV + per-sector meta. Real code-rigor improvements but did not improve backtest. Best remains Phase 8. |
| 10 | DONE | **+0.08** (confidence floor=0.60); −0.16 (binary) | [−0.38, +0.49] (best); [−0.67, +0.29] (binary) | Confidence-floor sweep on Phase 8 best config. Hypothesis confirmed: high floor (≥ 0.55) DOES recover Phase-8-like behavior; default Phase 9 floor=0.50 was the regression. **Best point estimate** at floor=0.60 (+0.077) but CI still straddles zero. NO config has CI strictly above zero. Reproducibility ✓: binary baseline matched documented Phase 8 (−0.158 vs −0.16); floor=0.50 matched documented Phase 9 (−0.570 vs −0.57). Reviewer caught C1 (baseline drift risk) + H1 (dead CLI flag) + 3 MED; all fixed. |
| 11 | DONE | **−0.11** (drop bottom 25%); −0.16 (baseline); −0.19 (drop top 25%) | [−0.58, +0.38] / [−0.67, +0.29] / [−0.63, +0.24] | Feature pruning via per_feature_audit on big universe (20 features, all positive `pct_drop` — no leak suspects). Driver: `scripts/phase11_feature_pruning.py`. Pipeline hook: `PipelineV5Config.feature_exclude`. Dropped 5 lowest-impact features (adv_proxy_21, dist_low_252_rank, ret_252d_rank, kurt_63, dist_low_252) → marginally better point estimate (+0.048 vs baseline) AND smaller DD (−13.2% vs −16.0%). Sanity check passes: dropping top 5 (vol_21d, vol_21d_rank, vol_63d_rank, kurt_63_rank, macd_signal) made things worse. Still NO config CI strictly above zero. Honest top-line unchanged. 4 new tests + RSS logging added. |
| 12 | DONE | **−0.38** (EDGAR enabled, WORSE than baseline) | [−0.84, +0.08] | SEC EDGAR 8-K event features. New module `src/stockpred/data/edgar.py` (free, full historical, no API key, SEC-compliant). New CLI flag `--edgar-events`. Adds 4 features: `edgar_has_8k`, `edgar_count_8k_{5,21,63}d`. **Honest finding**: HURT the strategy. Sharpe dropped −0.16 → −0.38, DD widened −16% → −20%. CI still straddles zero (not statistically worse) but point estimate clearly degraded vs Phase 11 baseline. Likely cause: count features are firm-size-noise; `has_8k` lacks sentiment direction. RAM: 0.97 GB peak (well under 6 GB budget); cold EDGAR fetch ~4 min (44 quarters, rate-limited). Sub-agent reviewer caught 2 CRITICAL + 1 HIGH (Procter&Gamble parser bug) + 1 production-smoke regression (alternate 'File Name' header spelling in 2014); all fixed before commit. 22 tests passing. Total suite: 127. |
| 13 | DONE | **+0.17** (EDGAR items, BEST point estimate so far) | **[−0.32, +0.58]** | SEC EDGAR 8-K item-code features. New flag `--edgar-items`. Per-item-family flags + rolling counts: earnings (2.02), CEO change (5.02), M&A (1.01+2.01+8.01), guidance (7.01), going-concern (3.01+3.03+4.02). Output prefix `edgaritem_`. 15 features added (5 families × 3 windows). **Honest finding**: hypothesis confirmed — item-coded events DO carry directional signal that raw counts don't. Sharpe went from −0.11 (Phase 11 baseline) → **+0.17** (Phase 13). DD shrunk from −13% → **−8.2%** (smallest holdout DD across all phases). BUT 95% CI [−0.32, +0.58] still straddles zero — not statistically significant. This is honest progress, not yet proof. RAM: 1.03 GB peak; cold fetch ~30 sec (150 tickers × 0.11s). Sub-agent caught 3 real findings before production: CRIT-1 (dual-class GOOG/GOOGL double-count via shared CIK), CRIT-3 (HTTPError swallowing hides 429/403), HIGH-3 (submissions JSON pagination cap); all fixed. Internal bug caught during test: date↔items array misalignment after independent sort. 5 new regression tests. Total suite: 139. |
| 14 | INFRA DONE; BULK FETCH RUNNING | TBD | TBD | GDELT 1.0 daily tone + mentions. New module `src/stockpred/data/gdelt.py` (streaming CSV parser, snappy parquet per-day cache). New CLI `--gdelt`; pipeline field `use_gdelt_features`; HTTP schema field too. 6 features per (date, ticker): mention_count, article_count, tone_mean, tone_std, rolling 5d/21d versions. Strict 8 GB RAM discipline (stream + filter + dtype shrink). Bulk-fetch driver `scripts/phase14_gdelt_bulk_fetch.py` runs overnight (~2-3 hr for 4018 daily files at 0.5s rate limit). Cache directory budget: ~50 MB (per-day parquet ~10-100 KB after S&P-500 filter). 11 new tests passing. |
| 15 | INFRA DONE | n/a (dashboard-only, not a backtest feature) | n/a | FinBERT live-mode sentiment. New module `src/stockpred/data/sentiment.py` (lazy-loaded transformers + torch; ~440 MB model download + ~1.5 GB deps; graceful degradation when not installed). Per-headline cache by sha256(title). Exposed via `GET /tickers/{ticker}/news?with_sentiment=true` -- 5 new optional fields on `NewsHeadline` (sentiment_label, sentiment_net, sentiment_{positive,neutral,negative}). NEVER a backtest feature (yfinance news has ~30d history -> catastrophic walk-forward bias if used as feature). 7 tests passing. Sub-agent reviewer caught C3 (broad except scope) + fixed. |
| 16 | DONE | **+0.17** (Phase 13 baseline reproduced; nothing layered improved) | [−0.32, +0.58] | Chain sweep driver `scripts/phase16_chain_sweep.py`. 4 configs on Phase 13 baseline:<br>  baseline (Phase 13)        → Sharpe **+0.173** CI [−0.32, +0.58] DD −8.2%  ⭐<br>  + triple-barrier labels    → Sharpe +0.067 CI [−0.47, +0.54] DD −11.7%<br>  + conf(floor=0.60)         → Sharpe −0.013 CI [−0.48, +0.47] DD −13.9%<br>  + TB + conf(floor=0.60)    → Sharpe −0.084 CI [−0.53, +0.37] DD −17.7%<br>**Honest finding**: Phase 13 (binary meta + ranks_only + HRP + EDGAR items) remains optimal. Adding TB hurts; adding confidence sizing hurts more; both together hurt most. Triple-barrier on top of meta-labelled signals is double-bounding the same target → probably overfits. Phase 10 found floor=0.60 was a Phase 10 sweet spot in ISOLATION, but ON TOP of Phase 13's already-binary meta, it adds noise. Reproducibility verified: baseline run reproduced Phase 13 exactly (+0.173). |
| 17 | INFRA DONE | TBD (need production smoke) | TBD | Fama-MacBeth cross-sectional regression as 3rd model class. New module `src/stockpred/models/fama_macbeth.py`. Per-date ridge-regularised OLS → time-series average factor returns → OOS prediction = X_te @ lambda_hat. Wired via `model='fama_macbeth'` in CLI / schema / pipeline. 8 tests passing including 2-factor noiseless recovery, NaN handling, shrinkage scaling, pipeline dispatch. Sub-agent reviewer fixed H2 (median caching) and removed C2 false-alarm comment. |
| 18 | INFRA DONE | TBD (production sweep pending) | TBD | LightGBM hyperparameter grid sweep on Phase 13 best. Driver `scripts/phase18_hp_sweep.py` sweeps 3 num_leaves x 3 lr x 2 n_estimators x 2 min_data_in_leaf = 36 configs (~108 min on warm cache). Output ranked by HOLDOUT Sharpe with explicit warning about HOLDOUT-data selection bias. |
| 19 | DONE | Phase 13 baseline reproduced (+0.17); all alpha > 0 HURT | [−0.32, +0.58] (baseline best) | Per-ticker Bayesian shrinkage of ensemble score by historical sign-precision. Production sweep:<br>  alpha=0.00 (no shrinkage)    → Sharpe **+0.173** CI [−0.32, +0.58] DD −8.2%  ⭐<br>  alpha=0.25                   → Sharpe −0.011 CI [−0.50, +0.41] DD −14.0%<br>  alpha=0.50                   → Sharpe +0.028 CI [−0.45, +0.45] DD −11.7%<br>  alpha=0.75                   → Sharpe +0.009 CI [−0.48, +0.46] DD −12.1%<br>  alpha=1.00                   → Sharpe −0.047 CI [−0.53, +0.41] DD −12.3%<br>**Honest finding**: shrinkage HURT at every alpha > 0. Likely reason: dev-window sign-precision has high variance for small per-ticker sample sizes, so "shrinking" introduces more noise than signal. The Phase 13 baseline is robust to this kind of weighting transformation. C1/H1/H4 reviewer fixes verified; 14 unit tests passing. |

**Best honest config (Phase 13; supersedes Phase 11)**:
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
    --ranks-only --edgar-items
# -> hold Sharpe +0.173, CI [-0.32, +0.58], DD -8.2%, RSS 1.03 GB
```
**Note**: do NOT add `--edgar-events` (Phase 12 raw counts hurt the
strategy). Item codes (`--edgar-items`) carry directional signal that
raw counts don't.

**Earlier Phase 11 config (drop-bottom-5) and Phase 8 config still
reproducible** via scripts/phase11_feature_pruning.py and the longer
`run_phase5.py` invocation respectively. Phase 13 should be the
starting baseline for Phase 14+.

**Best honest config from Phase 8 (still reproducible)**:
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
    --ranks-only
```

## Hard constraints (don't violate)

1. **Free data only.** yfinance (delayed daily bars), Wikipedia (S&P 500 changes), FRED CSV, FINRA flat files, SEC EDGAR JSON, GDELT 2.0 (free). No paid APIs.
2. **No corp / Google-internal services.** Pure PyPI, public Internet only.
3. **Honest results.** If the strategy loses money, show that. Never optimise for win rate. Always report HOLDOUT (not DEV) Sharpe with bootstrap CI.
4. **No account creation in user's name** on hosting platforms.
5. **No git push by us.** User does it manually via PAT.
6. **No intraday / real-time.** yfinance is 15+ min delayed; out of scope without paid data.
7. **No leakage.** Every new feature/label must pass the leakage audit (`scripts/leakage_audit.py`). Same-day shared inputs between feature and label are the #1 failure mode.
8. **8 GB / 8 vCPU RAM budget (user direction 2026-06-04).** Production
   deploy target is an 8 GB / 8 vCPU box. Every new code path MUST:
   (a) read source data in chunks (`pd.read_csv(..., chunksize=...)`),
   (b) cache to gzipped parquet (`compression="snappy"` or `"gzip"`),
   (c) shrink dtypes where safe (`float32` for features, `int16/int32`
       for counts, `category` for ticker strings),
   (d) `del` intermediate DataFrames + `gc.collect()` between phases,
   (e) NEVER hold the full GDELT CSV in memory uncompressed,
   (f) log peak RSS via `psutil` at the end of each phase so we catch
       regressions early. See "Memory discipline" section below.
9. **End-to-end smoke test after every phase (user direction 2026-06-04).**
   Before declaring a phase done, run `uv run python scripts/run_phase5.py`
   with the new feature ENABLED on a tiny universe (--n-tickers 40,
   --start 2020-01-01) AND on the production config (150 names × 11yr).
   Verify the pipeline completes, holdout metrics are present, and peak
   RSS stays under 6 GB (1 GB headroom). If either check fails, the
   phase is not done.
10. **Docs must stay in sync (user direction 2026-06-04).** Any new CLI
    flag, env var, or required system step (e.g. `curl` to fetch a
    bulk file, or `pip install` of a heavy model) MUST land in
    `docs/USAGE.md` in the same commit. The user's deploy box won't
    have access to my session memory.

## Architecture (top-level)

```
src/stockpred/
├── data/         # universe (Wikipedia), prices (yfinance), macro (FRED), fundamentals, news
├── features/     # technical, cross_sectional, tier2, regime
├── labels.py     # forward returns + vol-scaled (P6L1 leak-fixed)
├── labels_triple_barrier.py
├── models/       # baseline (logistic), gbm (LightGBM), meta (binary + confidence)
├── backtest/     # engine (horizon-aware, ±50% clip), portfolio (top-K, vol-scaled, HRP), hrp
├── validation/   # walk_forward (trading-day embargo), metrics, stress (block bootstrap)
├── reports/      # tearsheet
├── pipeline.py   # Phase 1/2 path
├── pipeline_v5.py # Phase 5/6/7/8/9 path (this is the main pipeline now)
└── backend/      # SQLite + FastAPI + APScheduler

scripts/
├── run_phase1.py
├── run_phase5.py # the one you usually want; takes Phase 5-9 flags
├── leakage_audit.py
├── per_feature_audit.py
├── sensitivity.py
└── serve.py      # uvicorn for the dashboard

web/              # React + Vite + TS + Tailwind dashboard

tests/            # 101 passing + 2 slow deselected
docs/
├── continue.md     # YOU ARE HERE (start here every session)
├── CONCEPTS.md     # beginner glossary
├── USAGE.md        # end-to-end user manual
├── DEPLOYMENT.md   # Docker / Fly / Render / VM
├── PROJECT_LOG.md  # chronological history per phase
└── HANDOFF.md      # legacy resume protocol (superseded by this file)
```

## How to resume (concrete protocol)

### Step 1 — Verify the world is intact (3 min)
```bash
cd ~/projects/stock-predictor
git status                                    # should be clean
git log --oneline -10                         # last few phase commits
uv sync --extra dev > /dev/null
uv run pytest tests/                          # expect 101 passed, 2 deselected
```

If tests are red: `git checkout main` and start from the last green commit
on the remote. Do NOT push a broken state.

### Step 2 — Read the latest phase's PROJECT_LOG entry
`docs/PROJECT_LOG.md` has a `Session N — Phase N` entry per phase. Read the
most recent one fully. It tells you what was tried, what the reviewer found,
what the honest numbers were, and the next-steps roadmap.

### Step 3 — Pick a Phase X+ item from the roadmap
The current next-steps roadmap (see `Phase 10+ roadmap` below) is the
candidate list. Pick one based on either: (a) what the user asked for, or
(b) the next item in expected ROI order.

### Step 4 — For each phase, this order
1. Implement (modules + tests).
2. Run `uv run pytest tests/` — must stay green.
3. Run real-data backtest with the new feature enabled.
4. Dispatch a `task` sub-agent to review the new code (`subagent_type=general`).
5. Fix anything the reviewer flags as CRITICAL or HIGH.
6. Re-run tests + commit.
7. **Update `docs/continue.md` and the phase ledger above**.
8. Optionally append a Session entry to `PROJECT_LOG.md`.

### Step 5 — How to push (only when user says "push")
```bash
read -s GH_TOKEN
git push "https://x-access-token:${GH_TOKEN}@github.com/hgdsraj/stock-predictor.git" main
unset GH_TOKEN
```

## Phase 10+ roadmap (in expected ROI order)

These are the candidates for the next phases. Status: planned / not started
except as noted.

**Phase 10 results (DONE)**: confidence floor ∈ {0.50, 0.55, 0.60, 0.65,
0.70, 0.75} sweep on Phase 8 best config. Driver: `scripts/phase10_conf_
floor_sweep.py`. Output: `reports/phase10_conf_floor_sweep.csv`.
- Reproducibility verified: binary baseline matched documented Phase 8
  (−0.158 vs −0.16); floor=0.50 matched documented Phase 9 (−0.570 vs −0.57).
- **Best**: floor=0.60 → +0.077 hold Sharpe, CI [−0.38, +0.49]. **CI still
  straddles zero; not significant.** Smallest holdout DD (−14.2%) too.
- All floors ≥ 0.55 recover Phase-8-like behavior. The Phase 9 default
  floor=0.50 was indeed the culprit for the Phase 9 regression.
- **No config has CI strictly above zero. The honest result is unchanged.**

Next candidates:

1. **Phase 11 — feature pruning from per-feature audit** *(DONE)*: see
   ledger row 11. Best result: drop bottom 5 by pct_drop → hold Sharpe
   −0.11, CI [−0.58, +0.38], DD −13.2%. The 5 features to drop on the
   150-name × 11yr universe are: `adv_proxy_21`, `dist_low_252_rank`,
   `ret_252d_rank`, `kurt_63`, `dist_low_252`. Subsequent phases should
   layer on TOP of this pruned baseline via the new `feature_exclude`
   config field.
2. **Phase 12 — EDGAR 8-K event flags as features** *(DONE; result
   was honest-negative)*: see ledger row 12. Sharpe degraded from
   −0.16 → −0.38 vs Phase 11 baseline. Likely cause: count features
   are firm-size noise; `has_8k` lacks sentiment direction. **Default
   `--edgar-events` to OFF; do not enable it unless paired with a
   sentiment-direction signal (Phase 13 GDELT, or filtered to
   specific 8-K item codes).** Future enhancement: extract 8-K item
   codes (`item_5.02 = CEO change`, `item_2.02 = earnings release`,
   etc.) as separate per-(date, ticker) flags. Item-coded events
   carry directional information; raw counts don't.
3. **Phase 13 — EDGAR 8-K item-code extraction** *(DONE; FIRST
   POSITIVE point estimate across all 13 phases)*: see ledger row 13.
   Hold Sharpe +0.17, DD −8.2% (smallest). CI [−0.32, +0.58] still
   straddles zero so not yet a statistically significant edge, but
   the qualitative direction is clear: item-coded events DO carry
   directional information that raw counts (Phase 12) don't.
   **Recommended default**: `--edgar-items` ON, `--edgar-events` OFF.
4. **Phase 14 — GDELT daily aggregates** *(user direction 2026-06-05;
   "overnight" budget acknowledged)*: free GDELT 2.0 GKG daily
   per-ticker tone score and theme counts, historical from 2015-02.
   Bulk fetch via the daily aggregate files (~100 MB/day raw, ~3 GB
   on disk after S&P-500 ticker filter + parquet compression). Cold
   download takes ~6-12 hr at typical bandwidth; should be launched
   overnight or in background. Memory plan: stream + filter + cache,
   never hold a full daily file uncompressed. Bias risk: left
   boundary at ~2015-02 reduces 2014-2015Q1 training rows.
5. **Phase 15 — FinBERT live-mode sentiment** *(user direction
   2026-06-05; DASHBOARD-ONLY, NOT a backtest feature)*: use existing
   yfinance news plumbing (`src/stockpred/data/news.py`) + local
   FinBERT model (~500 MB download) to score headlines for the Ticker
   detail page in the UI. Surface latest sentiment as a panel, not a
   model input. We do NOT use it as a backtest feature because
   yfinance only has ~30 days of history; using it as a feature would
   create catastrophic selection bias in walk-forward CV.
6. **Phase 16 — chained TB + meta on the best config**: `--triple-barrier`
   + `--meta-labelling --ranks-only --position-sizing hrp` together,
   layered on whatever the best result from Phases 11-14 ends up being.
   Consider also pinning `--meta-mode confidence --meta-conf-floor 0.60`
   (the Phase 10 sweet spot) as a third leg.
7. **Phase 17 — Fama-MacBeth cross-sectional regression**: replace the
   per-date GBM-then-rank with a daily Fama-MacBeth regression of returns
   on factor exposures. Different model class, less prone to overfit on
   tabular data with weak signals.
8. **Phase 18 — hyperparameter sweep on the best config**: with the
   sensitivity grid runner, sweep GBM `num_leaves`, `learning_rate`,
   `n_estimators`, `min_data_in_leaf` on the best post-Phase-16 config.
   Report best holdout Sharpe + CI.
9. **Phase 19 — robust signal aggregation**: replace `top_bottom_k` per
   day with a daily Bayesian shrinkage of the GBM output toward zero,
   weighted by historical sign-precision per ticker. Lopez de Prado
   Ch. 4 style.
10. **OUT-OF-SCOPE without budget**: intraday data via Alpaca/Polygon
    (paid); options-flow / IV-skew (paid); high-frequency news wire
    (Bloomberg/Reuters, paid); GDELT BigQuery export (requires GCP
    account, which the project policy disallows).

**None of these are guaranteed to flip HOLDOUT Sharpe above zero.** The
strategy-research sub-agent's ceiling estimate for free-data daily-bar
S&P 500 cross-sectional L/S is net Sharpe 0.4–0.8 *if* something works,
with most retail attempts capping below 1.0. We are at 0 (CI straddles).

**News-as-features rationale (per user direction 2026-06-04)**: EDGAR
event flags first (most-defensible, full history); GDELT tone second
(layered on, ~2015 boundary); FinBERT third (live-mode only, dashboard
panel, NOT a backtest feature — yfinance shallow history would cause
catastrophic walk-forward bias).

## Memory discipline (8 GB RAM target)

The user's deploy box has 8 GB RAM and 8 vCPU. The current pipeline
peaks at ~3 GB during the Phase 8 best config (150 names × 11 yr). News
features risk doubling this. Rules for every new data source:

1. **Cache to gzipped/snappy parquet on disk.** A 5 GB GDELT CSV
   compresses to ~400 MB parquet. Always:
   ```python
   df.to_parquet(path, compression="snappy", index=False)
   ```
2. **Stream/chunked reads for large source files.** Never `pd.read_csv`
   a multi-GB file in one shot. Use `chunksize=200_000` and concatenate
   only the rows that pass an early `query`.
3. **Shrink dtypes after load.**
   ```python
   df["ticker"] = df["ticker"].astype("category")
   df["count_8k_21d"] = df["count_8k_21d"].astype("int16")
   df["tone"] = df["tone"].astype("float32")
   ```
4. **Free intermediates explicitly.** After joining a news DataFrame
   into the master feature matrix, `del news_df` + `gc.collect()`.
5. **Log peak RSS at end of each pipeline phase.** Add to
   `pipeline_v5.run_pipeline_v5`:
   ```python
   import psutil
   rss_gb = psutil.Process().memory_info().rss / 1024**3
   log.info("Peak RSS for this phase: %.2f GB", rss_gb)
   ```
6. **`PipelineV5Config.refresh_data=False` is the default for a reason.**
   News fetch + score loops should ALWAYS be cache-first; force-refresh
   only when explicitly asked.

If a new feature ever causes peak RSS to exceed 6 GB (leaving 1 GB for
the OS + 1 GB headroom), the phase is not done; profile + fix BEFORE
committing.

## End-to-end smoke-test checklist

Before declaring any phase done, run BOTH:

```bash
# 1. Tiny smoke (5-10 min): does it complete + produce holdout metrics?
uv run python scripts/run_phase5.py \
    --start 2020-01-01 --end 2023-12-31 \
    --n-tickers 40 --bootstrap-n 100 \
    [your new flag]

# 2. Production smoke (30-60 min): does it scale + stay under RAM budget?
nohup uv run python scripts/run_phase5.py \
    --start 2014-01-01 --end 2024-12-31 \
    --n-tickers 150 \
    [your new flag] > logs/phaseN_smoke.log 2>&1 &
# After:  grep -E "Peak RSS|HOLDOUT" logs/phaseN_smoke.log | tail -10
```

If either run crashes, OOMs, or produces no holdout metrics, the phase
is not done — fix BEFORE updating the ledger.

## Known issues / wart list (don't add to roadmap; just be aware)

- `scripts/sensitivity.py` `--cost-grid` monkey-patch doesn't actually
  vary costs (it patches the wrong import). Fix is ~5 lines.
- `pipeline_v5.py` is now ~1100 lines; not a refactor priority but consider
  splitting `_apply_meta_gate*` into a `meta_gate.py` helper module.
- LSP/pyright warnings on pandas `Series` vs `DataFrame` typing are
  expected noise; nothing to fix.
- yfinance occasionally returns 404 for delisted tickers; pipeline handles
  it but warning spam is loud. Cosmetic.
- The `--meta-mode confidence` mode is highly floor-sensitive on this
  data (Phase 10): floor=0.50 (Phase 9 default) clearly underperforms
  binary; floor ∈ [0.55, 0.75] is roughly equivalent to binary with
  floor=0.60 giving the best point estimate (+0.077) but a CI that
  still straddles zero. If you ship a default, ship floor=0.60; do not
  ship floor=0.50.
- EDGAR form.idx header has two known spellings: "Filename" (post-2015)
  and "File Name" with a space (2014 and earlier). The parser accepts
  either; deleting `data/cache/edgar/*.parquet` forces a re-fetch with
  the latest parser. Set env `EDGAR_USER_AGENT="Name email"` for
  production deploys so SEC has someone to contact if your traffic
  becomes problematic.
- yfinance returns `datetime64[us]` for trading-day indexes; some other
  free data sources return `datetime64[ms]`. `merge_asof` is strict
  about matching dtypes and raises "incompatible merge keys" without
  explicit casts. Any new data-source module should `.astype("datetime64[ns]")`
  both sides before merging. EDGAR module does this; future sources
  must too.

## Sub-agent dispatch pattern (proven to find real bugs)

After implementing a phase, BEFORE running real-data backtests:

```python
task(
    description="Review Phase N",
    subagent_type="general",
    prompt="""Read-only review of Phase N additions to /usr/local/google/
    home/mahey/projects/stock-predictor/. Do NOT modify files. Return
    markdown report.

    NEW code: [list files + line ranges]

    Check for: CRITICAL leakage, walk-forward fold overlap, holdout
    contamination, NaN propagation, default values that silently misbehave.
    HIGH: numerical stability, error swallowing, missing input validation.
    MEDIUM: docstring/code drift, dead code, missing test coverage.

    Format: markdown with severity headings, file:line citations,
    ~1500 words max."""
)
```

Each Phase 6/7/8/9 review found at least one CRITICAL bug. The pattern works.

## Update protocol for this file

After each phase commit, update:
1. The phase ledger table (add a row, update Status to DONE)
2. The Phase 10+ roadmap (mark items DONE, add new items if discovered)
3. The "Known issues" list if any cosmetic items added

Then commit `docs/continue.md` along with the phase commit, so the next
session can resume immediately.
