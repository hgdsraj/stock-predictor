# Future Plan: Daily Inference Without Full Retraining

## Problem

Running a full Phase 5 pipeline on the S&P 500 takes 30–80 minutes and uses 4–8 GB of RAM because
it re-trains the LightGBM model from scratch on 10–15 years of data every time.  For a daily
signal update we don't need to retrain — the model weights from the last training run are already
good.  We only need to:

1. Fetch today's price data (and the last ~300 days as lookback for features).
2. Compute features for today using the *already-trained* model.
3. Run prediction.
4. Write the new prediction rows to the DB.

This document outlines how to add a fast "inference-only" mode alongside the existing full training
pipeline.

---

## Architecture Overview

```
Full training run (weekly / monthly)          Daily inference run (every trading day)
────────────────────────────────────          ──────────────────────────────────────
• Walk-forward CV on full history             • Load persisted boosters from disk
• Trains N boosters (1 per horizon)           • Fetch last 300 trading days of prices
• Saves boosters → model store                • Compute features (same code path)
• Writes Run + Predictions + EquitySamples    • Run predict_gbm for each horizon
• Estimated time: 30–80 min                   • Ensemble → weights → Predictions only
                                              • Estimated time: 1–3 min
```

---

## Phase A — Model Persistence (no new endpoints)

**Goal:** after every training run, serialize the trained boosters to disk.

### Storage layout

```
data/models/
  latest/                  ← symlink → YYYY-MM-DD_<run_id>/
  YYYY-MM-DD_<run_id>/
    meta.json              ← {phase, horizons, feature_cols, n_tickers, trained_at, run_id}
    booster_h1.lgb
    booster_h5.lgb
    booster_h21.lgb        ← (Phase 1 only if 21d was used)
    sector_map.json
    universe.json          ← list of tickers used
```

### Code changes

- `src/stockpred/models/gbm.py` — add `save_booster(booster, path)` and `load_booster(path)`.
- `src/stockpred/pipeline.py` / `pipeline_v5.py` — after training each fold's final model,
  persist to the model store.  The *last* trained booster per horizon (trained on all dev data)
  is the one saved for inference.
- `src/stockpred/backend/snapshot.py` — record `model_store_path` in `Run.summary_json`.
- New `src/stockpred/models/registry.py` — helpers: `latest_model_path()`,
  `load_model_meta(path)`, `save_model_bundle(path, boosters, meta)`.

**Risk:** the model store directory must be on a persistent volume (same as `data/app.db`).

---

## Phase B — Inference Pipeline

**Goal:** a lightweight pipeline that runs prediction only.

### New file: `src/stockpred/pipeline_infer.py`

```python
@dataclass
class InferenceConfig:
    model_path: str | None = None   # None = use latest
    lookback_days: int = 300        # enough for 12-1 momentum + vol features
    end_date: str | None = None     # None = today
    refresh_data: bool = False

def run_inference(cfg: InferenceConfig | None = None) -> dict:
    """
    1. Load model bundle from disk (meta + boosters).
    2. Fetch last cfg.lookback_days of prices for the universe in meta.json.
    3. Compute features using the same feature pipeline.
    4. For each horizon, predict with the saved booster.
    5. Ensemble + vol-scale weights using meta config.
    6. Return predictions dict (no backtest, no equity curve).
    """
```

### Key constraints
- Features must be computed exactly as during training (same feature functions, same parameters).
  The `meta.json` stores which feature flags were active.
- No labels are computed — inference is forward-looking by definition.
- The returned dict is compatible with `snapshot_run` but only writes `Prediction` rows, not
  `EquitySample` rows (no returns to compute yet).

### Memory profile
- 500 tickers × 300 days × 8 columns = ~1.2M rows → ~100 MB
- Feature matrix: 500 tickers × 1 date × ~50 features = tiny
- Loading 2 boosters (h=1, h=5): ~50–200 MB depending on tree depth
- **Total: ~300–500 MB** vs 4–8 GB for full training

---

## Phase C — New API Endpoint + Scheduler

### New endpoint

```
POST /jobs/predict
```

Body (all optional):

```jsonc
{
  "model_path": null,       // null = use latest trained model
  "lookback_days": 300,
  "refresh_data": false
}
```

Returns `{job_id, status: "queued"}` — same pattern as `/jobs/refresh`.

Requires `X-Password` header (same `STOCKPRED_PW` guard as launch/cancel).

### Scheduler changes

```
Existing daily cron (midnight UTC):  full training  →  change to weekly (e.g. every Sunday)
New daily cron (06:00 UTC weekdays): inference only  →  POST /jobs/predict internally
```

The weekly training run keeps the model current; the daily inference run updates predictions
cheaply.  If no trained model exists yet, the daily cron falls back to a full training run.

### UI changes
- Jobs list: add a "type" column — `training` vs `inference`.
- Inference runs are shown with a "⚡ Fast" badge.
- The "New Job" form gets a third option: "Daily Inference (fast)".

---

## Phase D — Model Staleness + Monitoring

The silent failure mode is: market regime shifts, but the stored model is months old.

### Staleness warnings
- `GET /healthz` returns `model_age_days` — days since last training run.
- If `model_age_days > 30`, return a `model_stale: true` flag.
- Dashboard shows a warning banner when model is stale.

### Automatic fallback
- If the daily inference cron fires and `model_age_days > 60`, run a full training job instead
  (or queue one if memory/time is a concern).

---

## Implementation Order

| Phase | Effort | Unlock |
|-------|--------|--------|
| A — Model persistence | ~4h | Enables B and C |
| B — Inference pipeline | ~6h | Core feature |
| C — Endpoint + scheduler | ~3h | Deployable |
| D — Staleness monitoring | ~2h | Production safety |

**Total estimated effort:** ~15 hours of implementation + testing.

---

## Open Questions

1. **Model store location on Railway**: must be on the same persistent volume as `data/`.
   Currently `data/` is the only mount point.  The model store goes under `data/models/`.

2. **Feature parity check**: if the codebase is updated (new features added), the saved model
   may be incompatible.  A version hash of the feature pipeline code should be stored in
   `meta.json` and checked at inference time.

3. **Sector/fundamental features**: fundamentals (sector, beta, P/E) are needed for some
   features.  The inference pipeline should use the cached fundamentals from the last training
   run, not re-fetch them (yfinance is slow and unreliable for production daily jobs).

4. **Walk-forward vs single model**: Phase 5 trains one model per fold.  The "final" model
   trained on all dev data is used for inference.  This is reasonable but slightly different
   from the ensemble of fold-specific models used in backtesting.

5. **Holdout window**: inference always predicts on "future" data (today), so holdout is
   N/A.  But the training pipeline's holdout window becomes important as a validation signal
   for model quality before deploying to inference.
