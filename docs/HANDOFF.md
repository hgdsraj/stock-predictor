# Handoff to the next session

If you're a future me (or another agent) picking this project up because the
previous session ran out of context, **read this file first**, then
`docs/PROJECT_LOG.md`, then the open todo list at the end of this file.

## Project at a glance

- Repo: `~/projects/stock-predictor`, remote `https://github.com/hgdsraj/stock-predictor`
- User: raj.axisos@gmail.com (GitHub user `hgdsraj`)
- Python env: `.venv/` via `uv` (run `uv sync --extra dev` to install)
- Tests: `uv run pytest tests/ -v` (must stay green)
- Hard constraints:
  - **Portable**: no Google-internal services, no corp tools. Public PyPI only.
  - **Free data**: yfinance, FRED, Wikipedia, FINRA flat files, SEC EDGAR. No paid APIs.
  - **Honest results**: a beautiful dashboard that displays negative Sharpe is correct. Never sugarcoat.
  - **No account creation** in the user's name on third-party services.

## Original ask of this big work block

User said:
> do all the phases and add all the details to the readme and update the project log.
> I'd like you to have it ready for deployment too and have a doc explaining how to deploy it on the web.
> Create a full server with a nice dashboard and everything and have it run the models as a cron job or on demand via the server
> and have everything displayed beautifully with beautiful charts etc, filter by individual stock tickers, have past data and
> prediction data for past year ready on the server and beautiful layouts wiht great fast searches and filters, also beautiful
> details like short interest etc like finviz quality but more basic ofcourse.

And critically:
> nothing has to give to ship it in one go. you can take a month if needed but i don't think you need it.
> make it great and take your time, have sub agents work as additional team members if you need to.
> have a sub agent stress test and review your code.

So: phases 2-4, FastAPI backend, React+Vite+TS+Tailwind+shadcn frontend, Dockerfile, deployment doc. Use sub-agents for parallel work. No quality trade-offs.

## What is DONE in this session before this handoff

### Phase 1 (already done in prior session; still good)
- Universe, prices, macro, baseline (logistic), GBM, technical/cross-sectional features, walk-forward CV, metrics, vectorised backtest, tearsheet, 16 unit tests, end-to-end pipeline script.

### Sub-agent reports received
1. **Research sub-agent** — produced a detailed free-fundamentals report covering yfinance reliability, FINRA short-interest flat files (`https://cdn.finra.org/equity/regsho/monthly/shrt<YYYYMMDD>.txt`), SEC EDGAR endpoints, earnings calendars, sector classification. **Action**: this report should be inlined verbatim or summarised into `docs/PROJECT_LOG.md` ("Data sources" section) and used as the spec for `src/stockpred/data/short_interest.py` and `src/stockpred/data/edgar.py` you will write in Phase 2/backend.
2. **Stress-test sub-agent** — found **3 CRITICAL bugs in Phase 1 code** that must be fixed before Phase 2 runs, plus HIGH/MEDIUM findings. Bugs are documented in `docs/PROJECT_LOG.md` as well as below.

### Bugs found by stress-test (status — all FIXED + regression-tested)
- **C1** horizon-aware engine ✅ (commit 4a70af0)
- **C2** trading-day embargo, default 25 ✅
- **C3** exact-bounded cross-sectional ranks ✅
- **H2** turnover/cost timing on clearing day ✅
- **H3** members_on strict end_date ✅
- **H5** ADV renamed to adv_proxy_21 ✅
- **M3** baseline returns NaN on all-NaN test rows ✅
- **M4** universe sampling: random default; current/first opt-in with loud warning ✅
- **M6** tearsheet per-column formatters ✅
- **L6** equity/dd charts use returns.dropna() ✅
- **L7** fundamentals rate-limit before submit ✅

### Phase 2 — DONE (commit de4b478)
- `src/stockpred/data/fundamentals.py` — yfinance .info caching, parquet store.
- `src/stockpred/features/cross_sectional.py` — `neutralise_by_sector` + `add_sector_dummies`.
- `src/stockpred/labels.py` — `compute_vol_scaled_forward_returns`; `long_labels` emits `fwd_vs_{h}` by default.
- `src/stockpred/pipeline.py` — rewritten: `PipelineConfig.horizons` (plural), `model={'gbm'|'logistic'}`, `use_sector_features`. Per-horizon walk-forward training, cross-sectional z-scored ensemble.
- `scripts/run_phase1.py` — supports `--horizons 1 5 21 --model gbm --no-sector --universe-sampling random`.

### Real-data Phase 2 result (60 names, 2018-2024, h={1,5,21} GBM ensemble)
- h=1d  IC IR +0.24  hit 51.5%
- h=5d  IC IR **+2.45**  hit 53.7%   ← actual signal
- h=21d IC IR -0.13  hit 55.6%       ← no signal / sign-flipped
- Strategy: Sharpe -1.3, ann_return -10.5%, max DD -57%.
- Conclusion: 5d signal is real; 1d and 21d wash it out at equal ensemble weights. Cost drag is meaningful. Phase 3 should: weight horizons by their IC IR or just drop 21d, add vol-scaled position sizing, cap sector exposure.

### Stuff still to do
- Phase 3: position sizing (signal × inv-vol), sector caps, turnover threshold, IC-IR-weighted ensemble.
- Phase 4: held-out window, bootstrap Sharpe CI, sensitivity grid, regime breakdown.
- Backend (SQLite + FastAPI + APScheduler + snapshot writer).
- Frontend (Vite + React + TS + Tailwind + shadcn + Recharts + TanStack — 4 pages).
- Dockerfile / docker-compose / `docs/DEPLOYMENT.md`.
- README + PROJECT_LOG update.

## Resume protocol

Do these in order:

### Step 1: Verify the current state (5 min)
```bash
cd ~/projects/stock-predictor
git status
git log --oneline -5
uv sync --extra dev > /dev/null
uv run pytest tests/ -v
```

If tests are red, find why (likely the engine rewrite + new horizon-aware tests need polish — see Open Items below).

### Step 2: Finish Critical/High bug fixes from the review
Tackle in this order; each gets its own commit. After each fix, **add a regression test** before moving on (the review listed which test to add per finding):

1. Make engine tests green (Step 1 fallout).
2. C2 — embargo in trading days.
3. C3 — exact-bounded cross-sectional ranks.
4. H3 — members_on strict boundary.
5. H5 — ADV mis-naming or fix.
6. M3 — NaN-row mask in baseline predictor.
7. M4 — survivorship in `select_universe`.
8. M6, L6, L7 — cosmetic / minor.

Run full tests after every fix. **Do not** start Phase 2 model work with any CRITICAL or HIGH unresolved.

### Step 3: Phase 2 — model improvements
- `src/stockpred/labels.py`: add `compute_vol_scaled_returns(prices, horizon, vol_window=21)` returning `fwd_return / rolling_std` so different horizons are unit-comparable.
- `src/stockpred/pipeline.py`:
  - Wire `train_gbm` / `predict_gbm` as the default model (keep baseline as fallback).
  - Multi-horizon ensemble: train one GBM per horizon in {1, 5, 21}, average the predicted vol-scaled returns (or rank-average); construct portfolio from the ensemble score.
  - Plumb sector data: call `fundamentals.fetch_fundamentals`, build `sector_map`, pass to `neutralise_by_sector` and `add_sector_dummies`.
- Tests: GBM smoke test, ensemble integration test on synthetic data (hit-rate 35-65%, no leakage), sector-neutral feature test.

### Step 4: Phase 3 — portfolio
- `src/stockpred/backtest/portfolio.py`: add `vol_scaled_weights(score, vol, leverage)` and `apply_sector_caps(weights, sector_map, max_per_sector)`.
- Add a `min_trade_threshold` knob: only rebalance a name if |Δw| exceeds threshold (reduces turnover-driven cost drag).

### Step 5: Phase 4 — stress tests
- `src/stockpred/validation/stress.py`:
  - `holdout_split(dates, holdout_years=2)` — last N years never touched in CV.
  - `bootstrap_sharpe(returns, n=1000)` — CI on Sharpe.
  - `sensitivity_grid(pipeline_fn, param_grid)` — runs the pipeline across a dict of param ranges.
  - `regime_breakdown(returns, vix)` — split returns by VIX quintile, by SPY bull/bear.

### Step 6: Backend
Suggested layout:
```
src/stockpred/backend/
├── __init__.py
├── db.py              # SQLAlchemy engine + Base
├── models.py          # ORM: Run, Prediction, PriceBar, Fundamental
├── store.py           # repository pattern: write_run(), get_predictions(t), etc.
├── snapshot.py        # called at end of pipeline: persists artifacts to DB
├── jobs.py            # APScheduler setup, daily refresh job
├── api.py             # FastAPI app with all routes
└── schemas.py         # Pydantic request/response models
scripts/serve.py       # uvicorn entrypoint
```

Endpoints (decided in earlier message):
- `GET  /healthz`
- `GET  /tickers` — universe + sector + last-updated
- `GET  /tickers/{t}` — last year OHLCV + predictions
- `GET  /tickers/{t}/details` — fundamentals + short interest if available
- `GET  /predictions/latest` — top-k long / bottom-k short for the latest run
- `GET  /backtest/summary` — equity curve, key metrics
- `POST /jobs/refresh` — trigger refresh; returns job id, idempotent within window

Use SQLite (`data/app.db`) — portable, no infra. Use SQLAlchemy 2.0 typed style. Background jobs via APScheduler in-process (single-node deploy assumption); for multi-node, switch to a queue later.

Tests: pytest + httpx.AsyncClient against the FastAPI app, in-memory SQLite per test.

### Step 7: Frontend
```
web/
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.js
├── index.html
├── src/
│   ├── main.tsx
│   ├── App.tsx
│   ├── api/client.ts          # fetch + TanStack Query setup
│   ├── components/
│   │   ├── ui/                # shadcn components (button, card, table, etc.)
│   │   ├── ChartEquity.tsx
│   │   ├── ChartDrawdown.tsx
│   │   ├── ChartCandles.tsx
│   │   ├── ScreenerTable.tsx
│   │   ├── TopMoversCard.tsx
│   │   └── ThemeToggle.tsx
│   ├── pages/
│   │   ├── Home.tsx
│   │   ├── Screener.tsx
│   │   ├── Ticker.tsx
│   │   └── Backtest.tsx
│   └── lib/
│       ├── format.ts          # number / percent / date formatters
│       └── theme.ts
```

Stack: Vite + React 18 + TypeScript + Tailwind + shadcn/ui + TanStack Query + TanStack Table + Recharts + Lucide icons. Dark/light theme. Server pagination for screener. Client-side search/filter. ETag-aware API client.

### Step 8: Deployment
- `Dockerfile` (multi-stage):
  1. Stage `web-build`: `node:20-alpine` → `npm ci && npm run build` produces `web/dist/`.
  2. Stage `runtime`: `python:3.12-slim` → install uv → copy project + built `web/dist/` → uvicorn `stockpred.backend.api:app` mounts `web/dist/` for the SPA.
- `docker-compose.yml`: single service, mount `./data` as a volume so the SQLite + parquet cache persist.
- `docs/DEPLOYMENT.md`: step-by-step for Fly.io, Render, and a generic VM (systemd unit + Caddy reverse proxy + Let's Encrypt).

### Step 9: Final wrap
- Spawn one more sub-agent for a full code review pass. Address findings.
- Update `README.md` (quickstart, screenshots, deployment summary).
- Update `docs/PROJECT_LOG.md` (everything done in this session).
- Run all tests one more time. Run `scripts/run_phase1.py` for an end-to-end real-data sanity check.
- `git add -A && git commit -m "..."` and tell the user to push.

## Sub-agent dispatch tips

- Use `task` tool with `subagent_type=general` for research / review / non-code work.
- Use `task` tool with `subagent_type=explore` for quick file-finding searches.
- Sub-agents return a single message; capture it and feed it into the next step.
- Run multiple sub-agents in parallel in a single message when their work is disjoint (e.g. research + review of existing code).
- Reasonable parallel sub-agents during this project:
  - Frontend reviewer (a11y + perf) while you write Docker.
  - Backend stress tester (endpoint contracts) after API is up.
  - Final pre-commit reviewer right before push.

## Things to NOT do

- Don't push to GitHub yourself. The user runs the push (we have the protocol: `read -s GH_TOKEN && git push https://x-access-token:${GH_TOKEN}@github.com/...`).
- Don't create accounts on hosting providers in the user's name.
- Don't use Google-internal tools (`g4`, `blaze`, `gpaste`, `borgcfg`, etc.). This project is portable.
- Don't use the `data/` directory contents in git — `.gitignore` already excludes price/parquet caches.
- Don't fake or smooth backtest numbers. If the strategy is bad, the dashboard says so.

## Open todo list at handoff

(Updated by the previous session's `todowrite`; the current list is the source of truth in-session — check it first.)

1. FIX C1: backtest accumulates h-day window correctly (horizon-aware engine) — **in progress; rewrite done, test verification pending**
2. FIX C2: embargo in trading days, default >= max horizon + buffer
3. FIX C3: cross-sectional ranks bounded exactly to [-0.5, 0.5]
4. FIX H3: members_on uses strict end_date > d
5. FIX H2: turnover/cost timing on held (shifted), not signal day — **done in engine rewrite, needs test confirmation**
6. FIX M4: select_universe no longer biases to current constituents silently
7. FIX M3: fit_predict_proba returns NaN for all-NaN test rows
8. FIX H5: ADV uses raw close * volume, not adj_close * volume
9. FIX M6: tearsheet yearly table column-specific formatting
10. FIX L6: tearsheet equity curve uses returns.dropna()
11. FIX L7: fundamentals rate-limit sleep before submit
12. Add regression tests for all CRITICAL + HIGH fixes (≈8 new tests)
13. Phase 2: sector loader + sector-neutral features (already started; finish wiring)
14. Phase 2: LightGBM through pipeline + multi-horizon ensemble
15. Phase 2: vol-scaled label option
16. Phase 3: position sizing, turnover threshold, sector caps
17. Phase 4: held-out, bootstrap Sharpe CI, sensitivity grid, regime breakdown
18. Backend: SQLite + SQLAlchemy + repository
19. Backend: snapshot writer
20. Backend: FastAPI app
21. Backend: APScheduler
22. Backend tests
23. Frontend: Vite + React + TS + Tailwind + shadcn scaffold
24. Frontend: layout + API client + theme
25. Frontend: 4 pages (Home, Screener, Ticker, Backtest)
26. Dockerfile + docker-compose
27. docs/DEPLOYMENT.md
28. Update README + PROJECT_LOG
29. Final code-review sub-agent + final E2E + commit + push instructions

## Last-known-good test snapshot (Session 7 end)

```
$ uv run pytest tests/ -q
101 passed, 2 deselected in ~185s
```

(2 slow tests deselected by default; run with `pytest -m slow` for those.)

Session 7 added:
- `meta_confidence_weight_signal` in `models/meta.py`
- `_apply_meta_gate_per_sector` helper in `pipeline_v5.py`
- `_apply_meta_gate` extended with mode/conf_floor/conf_cap/walk_forward_folds
- CLI: `--meta-mode`, `--meta-conf-floor`, `--meta-conf-cap`,
  `--meta-walk-forward-folds`, `--meta-per-sector`
- pytest slow marker (deselected by default)
- 7 new tests in `test_phase9.py` (5 unit + 2 slow integration)

Phase 9 reviewer fixes applied (4 critical/high):
- C1: per-sector meta drops cross-sectional columns to actually isolate sector
- C2: `pd.concat(..., verify_integrity=True)` on walk-forward folds
- C3: removed duplicate HOLDOUT log statement
- H4: warn when meta-per-sector flag set (holdout still uses global)
- H5: `_cluster_var` guards against ill-conditioned diagonal
- H6: confidence mode treats NaN proba as zero (matches binary mode)

Phase 9 honest result (150 names, 2014-2024, h=5, HRP, confidence mode, WF=3):
- HOLDOUT Sharpe: **−0.57** (worse than Phase 8's −0.16)
- HOLDOUT 95% CI: [−1.03, −0.15] (back to entirely negative)

The Phase 9 "improvements" are real code-rigor improvements but did NOT
improve the actual backtest. Binary gate's hard refusal remains the best
config. This is an honest finding: more sophistication on top of an absent
signal doesn't manufacture signal.

**Best result remains Phase 8.** Phase X+ would need either paid data or
fundamentally different model class.

(The 101 in Session 5 included some tests that were redundant; Session 6
consolidated and added Phase 8 tests, netting 95.)

Session 6 added:
- `_apply_meta_gate` in pipeline_v5.py — gates ensemble score on meta P(correct)
- Triple-barrier label injection in pipeline_v5.py
- `ranks_only` config option to drop raw cols
- CLI: `--meta-labelling`, `--meta-threshold`, `--triple-barrier`,
  `--tb-k-sigma`, `--ranks-only`
- `_build_weights` now accepts `precomputed_score` (review C1 fix)
- 5 new tests in test_phase8.py

Sub-agent reviewer fixes applied:
- C1: precomputed_score bypass for gated scores (no double z-score)
- C2: ungated dev score retained for holdout-meta training
- H1: re-rank Tier-2 features so ranks_only doesn't drop them
- M7: CLI input validation for `--meta-threshold` and `--k-pct`

Phase 8 honest result (best config: 150 names, h=5, HRP, meta-gating,
ranks_only):
- HOLDOUT Sharpe: **−0.16** (was −0.69 pre-Phase 8)
- HOLDOUT 95% block-CI: **[−0.67, +0.29]** — *straddles zero* for the
  first time in the project
- HOLDOUT max DD: **−16.0%** (was −29.6%)

The earlier "Phase 8 = +0.09 Sharpe" claim was an artefact of the C1
double-z-score bug. Corrected number is honest.

Session 5 added:
- `src/stockpred/backtest/hrp.py` — Hierarchical Risk Parity
- `src/stockpred/labels_triple_barrier.py` — Triple-barrier labels
- `src/stockpred/models/meta.py` — Meta-labelling
- `scripts/per_feature_audit.py` — Per-feature leakage attribution
- Engine: `pct_change` now clipped to ±50% (data-glitch defence)
- 14 new tests in `test_phase7.py`, 1 new in `test_backtest_engine.py`

Phase 7 reviewer fixes applied:
- HRP/vol_scaled `kk <= n//2` clamp
- `_cov_estimate` drops `bfill`
- HRP numerical stability hardening
- `build_meta_dataset` forbidden-column guard

Big-universe Phase 7 result:
- 822 historical S&P 500 tickers, 2008-2024, h={1,5}
- HOLDOUT Sharpe: vol_scaled −0.78, HRP **−0.69** (HRP slightly better)
- HOLDOUT 95% block-CI for HRP: [−1.12, −0.24] (still entirely negative)
- Per-horizon HOLDOUT IC IR: h=1 +0.69, h=5 +0.49 (real signal!)
- The signal is real; portfolio construction is reasonable; but the
  strategy still loses money on holdout after costs.

Phase 8+ research items (in PROJECT_LOG):
- Wire meta-labelling as a primary-signal gate
- Wire triple-barrier as an alternate training target
- Run per_feature_audit on the big universe
- Try longer horizons (h=21+) on the big universe

Phase 6 additions (this session):
- `src/stockpred/features/tier2.py` — momentum_12_1, st_reversal_5,
  max_daily_return, amihud_illiquidity, beta_vs_bench, idio_vol_vs_bench
- `src/stockpred/features/regime.py` — VIX + FRED macro + cross-sectional
  dispersion, with `broadcast_to_panel()` helper
- `src/stockpred/backtest/portfolio.py::neutralise_portfolio_beta`
- `src/stockpred/validation/stress.py::bootstrap_sharpe` now accepts
  `method='block'` (default for Phase 5 pipeline)
- `src/stockpred/pipeline_v5.py` config knobs: `use_tier2_features`,
  `use_regime_features`, `beta_neutralise`, `bootstrap_method`
- `scripts/leakage_audit.py` (THIS IS THE MOST IMPORTANT FILE — caught
  a real bug)
- `scripts/sensitivity.py` (works but cost-grid monkey-patch is buggy)
- `tests/test_phase6.py` — 13 tests covering all of the above

THE CRITICAL FIX in this session: `compute_vol_scaled_forward_returns`
was leaking via shared `close[t]` between feature `ret_1d` and the
trailing-vol denominator. Fixed by shifting the denominator +1 day. All
previous Phase 5 numbers were partly artefactual; PROJECT_LOG documents
the honest post-fix numbers.

The honest result: across a sensitivity grid of 8 combinations, **zero**
combos produced positive holdout Sharpe with a CI excluding zero. The
strategy as currently configured does not generalise out-of-sample.

Test files (10):
- test_backtest_engine.py (7)
- test_baseline_nan.py (1)
- test_features.py (5)
- test_labels_no_leakage.py (3)
- test_pipeline_integration.py (1) — Phase 2 synthetic E2E
- test_pipeline_v5.py (1) — Phase 5 synthetic E2E
- test_portfolio_construction.py (6)
- test_stress.py (6)
- test_universe_html_parse.py (3)
- test_walk_forward.py (6)
- test_backend_api.py (12)
- test_watchlist_and_news.py (11)

## Session 3 additions

- **`docs/CONCEPTS.md`** + **`docs/USAGE.md`** — beginner-friendly docs.
- **`src/stockpred/pipeline_v5.py`** + **`scripts/run_phase5.py`** — Phase 5
  pipeline with IC-IR ensemble + vol-scaled sizing + sector caps + min trade
  threshold + holdout + bootstrap CI + VIX regime breakdown.
- **`src/stockpred/data/news.py`** + `NewsItem` table + `/tickers/{t}/news`
  endpoint + Ticker page news card.
- **`WatchedTicker`** table + `/watchlist` endpoints + Screener page panel.
  Seeded with HND.TO, HNU.TO, UNG, SPY, ^VIX on first boot.
- **`macro.py` rewritten** to drop `pandas-datareader` and pull FRED CSVs
  via plain `requests`.
- **Strict ticker validation** on every `{ticker}` path parameter and
  `WatchedAdd.ticker` Pydantic field (review C1).
- **Chronological train/valid split** in Phase 5 holdout scoring (review C2).
- **News link scheme allowlist** (only `http(s)://` persisted).

## Phase 5 real-data result

```
DEV   Sharpe -0.04  ann_return -0.6%  max_dd -23%
HOLD  Sharpe -0.84  bootstrap 95% CI [-1.60, -0.15]  ← entirely negative
```

The dev metrics improved massively from Phase 2 (Sharpe went from -1.30 to
~breakeven). The holdout is statistically significantly negative. We do NOT
claim a working strategy.

## Open items for the next session

In rough priority order:

### From the Phase 5 reviewer's open findings
- **H1**: bootstrap CI is i.i.d. on autocorrelated returns. Move to block
  bootstrap with block length = `bt_horizon`.
- **H2**: `_trailing_vol` double-lag with `trade_lag=1` in engine. Remove
  the `.shift(1)` in `_trailing_vol` OR document explicitly.
- **H3**: holdout IC-IR weights are calibrated on a different model than
  the one producing holdout preds. Either compute IC-IR on the same final
  fit, or use the last walk-forward fold's model for holdout.
- **M1–M6**: see review report (search PROJECT_LOG for "Phase 5 reviewer").
- **L2 fixed**: CORS now allows DELETE.
- **Test gaps**: holdout-leakage canary, `_split_holdout` boundary test,
  `_trailing_vol` lag unit test, SSRF test on POST /watchlist (added),
  XSS test on news (added), bootstrap CI sanity test, ic_ir ensemble path
  E2E (currently only `equal` path is end-to-end tested).

### From the strategy-research report (Phase 6+ — real signal work)
1. Audit the +2.45 dev IC IR (recompute features with strict t-1 cutoff;
   Newey-West-adjusted IC IR t-stat).
2. Beta-neutralisation vs SPY at the portfolio level.
3. Add Tier-2 features: 12-1 momentum, IVOL, β, max return, Amihud, 52-week
   high, sector-relative momentum.
4. Add cross-asset regime features: VIX level/delta, term structure, USD,
   credit spreads.
5. Triple-barrier labels + meta-labelling (López de Prado Ch. 3.6).
6. Sensitivity grid across cost assumptions.

### NOT to be done in this project without scope change
- Intraday or real-time trading (free data is delayed; needs paid feeds).
- Broker integration (this is a research tool, not an exec system).
- Point-in-time fundamentals (yfinance `.info` is current; leaks if used historically).
- Sentiment scoring of news (free headline-only data + no paid LLM = noise).

If a future session can't reach test green, `git checkout` to the most recent commit on `main` and start from there.

---

*Written by Claude (CloudCode) during the long-running build-out session, in case I run out of context.*
