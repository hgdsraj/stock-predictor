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

**Best honest config (Phase 8)**:
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

1. **Free data only.** yfinance (delayed daily bars), Wikipedia (S&P 500 changes), FRED CSV, FINRA flat files, SEC EDGAR JSON. No paid APIs.
2. **No corp / Google-internal services.** Pure PyPI, public Internet only.
3. **Honest results.** If the strategy loses money, show that. Never optimise for win rate. Always report HOLDOUT (not DEV) Sharpe with bootstrap CI.
4. **No account creation in user's name** on hosting platforms.
5. **No git push by us.** User does it manually via PAT.
6. **No intraday / real-time.** yfinance is 15+ min delayed; out of scope without paid data.
7. **No leakage.** Every new feature/label must pass the leakage audit (`scripts/leakage_audit.py`). Same-day shared inputs between feature and label are the #1 failure mode.

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

These are the candidates for the next phases. Status: planned / not started.

1. **Phase 10 — confidence floor sweep**: Phase 9 confidence mode lost
   money because the default floor=0.5 was too permissive. Try sweeping
   floor ∈ {0.55, 0.60, 0.65, 0.70} on the best Phase 8 config. If a
   higher floor recovers Phase 8 behaviour smoothly, that's a useful
   default; if not, deprecate confidence mode in favour of binary.
2. **Phase 11 — feature pruning from per-feature audit**: re-run
   `per_feature_audit.py` on the big universe (currently it's 100-name),
   then drop the bottom-quartile by `pct_drop`. Test whether removing
   noisy features improves holdout.
3. **Phase 12 — chained TB + meta on the best config**: `--triple-barrier`
   + `--meta-labelling --ranks-only --position-sizing hrp` together.
   Currently neither has been tested in combination on the big universe.
4. **Phase 13 — Fama-MacBeth cross-sectional regression**: replace the
   per-date GBM-then-rank with a daily Fama-MacBeth regression of returns
   on factor exposures. Different model class, less prone to overfit on
   tabular data with weak signals.
5. **Phase 14 — hyperparameter sweep on the best config**: with the
   sensitivity grid runner, sweep GBM `num_leaves`, `learning_rate`,
   `n_estimators`, `min_data_in_leaf` on Phase 8 config. Report best
   holdout Sharpe + CI.
6. **Phase 15 — robust signal aggregation**: replace `top_bottom_k` per
   day with a daily Bayesian shrinkage of the GBM output toward zero,
   weighted by historical sign-precision per ticker. Lopez de Prado
   Ch. 4 style.
7. **Phase 16 (out of scope without budget)**: news + sentiment via
   local FinBERT (~500 MB model), event flags from EDGAR 8-K, intraday
   data via Alpaca/Polygon (paid).

**None of these are guaranteed to flip HOLDOUT Sharpe above zero.** The
strategy-research sub-agent's ceiling estimate for free-data daily-bar
S&P 500 cross-sectional L/S is net Sharpe 0.4–0.8 *if* something works,
with most retail attempts capping below 1.0. We are at 0 (CI straddles).

## Known issues / wart list (don't add to roadmap; just be aware)

- `scripts/sensitivity.py` `--cost-grid` monkey-patch doesn't actually
  vary costs (it patches the wrong import). Fix is ~5 lines.
- `pipeline_v5.py` is now ~1100 lines; not a refactor priority but consider
  splitting `_apply_meta_gate*` into a `meta_gate.py` helper module.
- LSP/pyright warnings on pandas `Series` vs `DataFrame` typing are
  expected noise; nothing to fix.
- yfinance occasionally returns 404 for delisted tickers; pipeline handles
  it but warning spam is loud. Cosmetic.
- The `psf-mode: Confidence` mode is documented but in practice produces
  worse results than binary on this data. Document but don't remove.

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
