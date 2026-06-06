# News-features route: end-to-end guide

This guide documents how the project ingests news / event data and feeds
it into the prediction pipeline (Phases 12–15). It covers:

- The four news sources we use and **why we have four**
- Pipeline configuration: CLI flags, `PipelineV5Config` fields, HTTP body
- Operator workflow: one-time setup, daily refresh, monitoring
- Honest expectations: which sources help, which hurt
- Leakage-safety arguments for each source
- Memory profile on the 8 GB / 8 vCPU production target

If you just want a one-liner: enable **Phase 13 (EDGAR items)** for the
backtest (the only news source that improved holdout Sharpe); enable
**Phase 15 (FinBERT)** for the dashboard sentiment panel; **leave
Phase 12 (raw 8-K counts) off** (it hurt the strategy); **Phase 14
(GDELT)** is optional and requires a one-time overnight bulk fetch.

---

## 1. The four news sources

| Phase | Source | URL | Coverage | Free? | API key? | Honest contribution to HOLDOUT Sharpe |
|---|---|---|---|---|---|---|
| 12 | SEC EDGAR 8-K event flags + counts | data.sec.gov | 2001-present | ✅ | ❌ | **−0.22 vs baseline (hurt)** |
| 13 | SEC EDGAR 8-K item codes | data.sec.gov | 2001-present | ✅ | ❌ | **+0.29 vs baseline (best)** |
| 14 | GDELT 1.0 daily tone + mentions | data.gdeltproject.org | 2013-04-present | ✅ | ❌ | TBD (pending production sweep) |
| 15 | FinBERT live-mode sentiment | huggingface.co/ProsusAI/finbert | ~30d (yfinance news) | ✅ | ❌ | **Dashboard-only, NOT a feature** |

### Why four?

Each source answers a different question:

- **Phase 12 (raw 8-K counts):** "Did this company file ANYTHING material
  today?" → null-direction signal; just a presence flag. Honestly tested,
  this HURT the strategy (firm-size noise: big companies file more
  regardless of return direction).
- **Phase 13 (8-K item codes):** "What KIND of material event?" Item 2.02
  = earnings, 5.02 = CEO change, 8.01 = M&A, etc. → carries the
  directional signal Phase 12 lacks. Best honest result across 13 phases.
- **Phase 14 (GDELT):** "Independent of SEC, what is the news tone for
  this company today?" → adds a non-SEC view; news mentions in non-US
  publications, social discussions, etc.
- **Phase 15 (FinBERT live):** "What is the sentiment of TODAY's
  headlines?" → live-only because yfinance's news goes back ~30 days;
  using it as a backtest feature would create catastrophic walk-forward
  bias. **Always dashboard-only.**

---

## 2. Pipeline configuration

### A. CLI flags on `scripts/run_phase5.py`

```bash
--edgar-events       # Phase 12: raw 8-K event count features
--edgar-items        # Phase 13: 8-K item-code features  (RECOMMENDED)
--gdelt              # Phase 14: GDELT daily tone+mentions
                     # Phase 15: no CLI flag; only used by the news API endpoint
```

### B. `PipelineV5Config` fields

```python
PipelineV5Config(
    ...
    use_edgar_features=False,        # Phase 12 — leave off
    use_edgar_item_features=True,    # Phase 13 — recommended
    use_gdelt_features=False,        # Phase 14 — opt-in
)
```

### C. HTTP `POST /jobs/refresh` body

```json
{
  "use_edgar_features":      false,
  "use_edgar_item_features": true,
  "use_gdelt_features":      false
}
```

All three default to `false`. The Phase 15 sentiment scorer has no
`/jobs/refresh` flag — it activates automatically when the
`/tickers/{ticker}/news` endpoint is hit.

---

## 3. Operator workflow

### One-time setup

#### Step 1 — Identify yourself to SEC

SEC requires every HTTP request to have a `User-Agent` header
identifying you (their rule, enforced at the CDN). Set it once on your
deploy box:

```bash
export EDGAR_USER_AGENT="Your Name your-email@example.com"
```

Without it, requests use the generic `stock-predictor/0.2
(raj.axisos@gmail.com)` default; SEC may rate-limit harder when many
users share that string.

#### Step 2 — Warm the EDGAR ticker → CIK map (needed for Phases 12, 13, 14)

```bash
uv run python -c \
  "from stockpred.data import edgar; edgar.fetch_ticker_to_cik(refresh=True)"
```

This caches `data/cache/edgar/ticker_to_cik.json` (~3 MB) AND
`data/cache/edgar/company_tickers.json` (~900 KB; raw SEC titles,
needed by GDELT name→ticker matching).

#### Step 3 — One-time EDGAR form-index pre-fetch (Phase 12 / 13)

The first Phase 12 or 13 backtest run will download 44 quarterly form
indexes (2014Q1 → 2024Q4), at ~5 sec/qtr = ~4 min total cold fetch.
Subsequent runs hit the per-quarter parquet caches at
`data/cache/edgar/8k_<YYYY>Q<n>.parquet`. Manual `curl` equivalent:

```bash
UA="$EDGAR_USER_AGENT"
mkdir -p data/cache/edgar
for YEAR in 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024; do
  for QTR in 1 2 3 4; do
    curl -A "$UA" \
        -o data/cache/edgar/form_${YEAR}Q${QTR}.idx \
        "https://www.sec.gov/Archives/edgar/full-index/${YEAR}/QTR${QTR}/form.idx"
    sleep 0.2
  done
done
```

#### Step 4 — Overnight GDELT bulk fetch (Phase 14 only)

```bash
nohup uv run python scripts/phase14_gdelt_bulk_fetch.py \
    --tickers-from-edgar \
    > logs/phase14_bulk.log 2>&1 &
tail -f logs/phase14_bulk.log
```

This downloads 4018 daily GKG files (~5–15 MB compressed each) at
0.5 s/file rate-limit → **~2–3 hours total**. Per-day parquet caches at
`data/cache/gdelt/gkg_YYYYMMDD.parquet`, ~10–100 KB each after filtering
to S&P-500 mentions. Idempotent; safe to interrupt and resume.

#### Step 5 — One-time FinBERT model download (Phase 15 only)

```bash
# Heavy deps (~1.5 GB)
pip install transformers torch huggingface_hub

# Model (~440 MB)
export FINBERT_MODEL_DIR="$PWD/models/finbert"
huggingface-cli download ProsusAI/finbert --local-dir "$FINBERT_MODEL_DIR"
```

If you skip this step, the news endpoint still returns headlines (without
sentiment fields) and a `WARNING` is logged.

### Daily runs

Once setup is done, trigger the best honest config with one HTTP call.
The complete curl command with all parameters is in
[`docs/OPTIMAL.md §2`](OPTIMAL.md) — copy it from there to avoid
parameter drift. Runtime: ~3–7 min on warm cache.

```bash
# Short form — see OPTIMAL.md for the full body
curl -X POST -H "X-API-Key: $STOCKPRED_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"phase": 5, "use_edgar_item_features": true, ...}' \
     http://localhost:8000/jobs/refresh

# Poll progress
curl http://localhost:8000/jobs/<uuid> | jq '{status, elapsed_s, error}'
```

### Live dashboard sentiment (Phase 15)

The frontend automatically requests sentiment-scored headlines:

```bash
curl http://localhost:8000/tickers/AAPL/news?limit=20&with_sentiment=true
```

Each headline payload includes:

```json
{
  "uuid": "...",
  "title": "Apple beats earnings",
  "publisher": "Reuters",
  "link": "https://...",
  "published_at": "2026-01-25T16:30:00",
  "sentiment_label":    "positive",
  "sentiment_net":      0.74,
  "sentiment_positive": 0.85,
  "sentiment_neutral":  0.11,
  "sentiment_negative": 0.04
}
```

If FinBERT isn't installed, the five `sentiment_*` fields are `null`.
Per-headline scores are cached on disk by `sha256(title)` at
`data/cache/sentiment/{xx}/{full_hash}.json`, so repeat lookups are
free.

---

## 4. What feature columns each phase adds

### Phase 12 (`--edgar-events` / `use_edgar_features: true`)

Per (date, ticker), 4 columns prefixed `edgar_`:

| Column | dtype | Meaning |
|---|---|---|
| `edgar_has_8k` | int8 | 1 if ANY 8-K was filed for this ticker on this trading day |
| `edgar_count_8k_5d` | int16 | Rolling count over last 5 trading days |
| `edgar_count_8k_21d` | int16 | Rolling count over last 21 trading days |
| `edgar_count_8k_63d` | int16 | Rolling count over last 63 trading days |

### Phase 13 (`--edgar-items` / `use_edgar_item_features: true`)

Per (date, ticker), 15 columns prefixed `edgaritem_`: 5 item families ×
3 time windows (today + 21d + 63d):

| Family | SEC item codes | Window flags |
|---|---|---|
| `edgaritem_earnings_*` | 2.02 | `_today`, `_21d`, `_63d` |
| `edgaritem_ceo_change_*` | 5.02 | `_today`, `_21d`, `_63d` |
| `edgaritem_ma_*` | 1.01 + 2.01 + 8.01 | `_today`, `_21d`, `_63d` |
| `edgaritem_guidance_*` | 7.01 | `_today`, `_21d`, `_63d` |
| `edgaritem_going_concern_*` | 3.01 + 3.03 + 4.02 | `_today`, `_21d`, `_63d` |

`_today` is int8 (binary flag), `_21d`/`_63d` are int16 (counts).

### Phase 14 (`--gdelt` / `use_gdelt_features: true`)

Per (date, ticker), 6 columns prefixed `gdelt_`:

| Column | dtype | Meaning |
|---|---|---|
| `gdelt_mention_count` | int16 | Distinct articles mentioning the ticker that day |
| `gdelt_article_count` | int32 | Sum of NUMARTS across rows (total articles attributed) |
| `gdelt_tone_mean` | float32 | Average article tone in [−100, +100] |
| `gdelt_tone_std` | float32 | Tone dispersion within the day |
| `gdelt_mention_{5,21}d` | int16 | Rolling mention count |
| `gdelt_tone_{5,21}d` | float32 | Rolling tone mean |

### Phase 15 (no feature columns)

By design. Phase 15 returns sentiment via the `/tickers/{ticker}/news`
endpoint only; never as a backtest feature.

### All columns survive `ranks_only`

The `ranks_only=true` filter (Phase 8's noise-reduction trick) keeps any
column matching:

- `*_rank` (the canonical rank features)
- `sec_*` (sector dummies)
- `reg_*` (regime broadcasts)
- `edgar*` (Phase 12 `edgar_` AND Phase 13 `edgaritem_`)
- `gdelt_*` (Phase 14)

So you can stack `--edgar-items --gdelt --ranks-only` without losing
the news columns.

---

## 5. Honest expectations (results table)

Production smoke results (150 tickers × 11 years, holdout 2 yr):

| Config | hold Sharpe | 95% block-bootstrap CI | DD | n_features |
|---|---|---|---|---|
| Phase 8 baseline | −0.158 | [−0.67, +0.29] | −16.0% | 18 |
| Phase 11 (drop bottom 5 features) | −0.110 | [−0.58, +0.38] | −13.2% | 13 |
| **Phase 12 (EDGAR raw counts)** | **−0.376** | [−0.84, +0.08] | −20.2% | 22 |
| **Phase 13 (EDGAR item codes)** | **+0.173** | [−0.32, +0.58] | **−8.2%** | 33 |
| Phase 14 (GDELT) | TBD | TBD | TBD | 39 |

**Honest reading:**

- **Phase 12 (raw 8-K counts) hurt the strategy.** Default it OFF.
  Likely cause: counts are firm-size correlated noise (big companies
  file more 8-Ks regardless of return direction); `has_8k` has no
  sentiment direction.
- **Phase 13 (item codes) is the best honest result across all phases.**
  Sharpe +0.17, smallest holdout drawdown (−8.2%). 95% CI still straddles
  zero so NOT statistically significant, but the qualitative story is
  consistent: item codes carry directional signal that raw counts don't.
- **No combination has produced a CI strictly above zero.** The honest
  top-line for the strategy class remains: this is not a proven edge.

---

## 6. Leakage safety

The pipeline enforces `trade_lag=1` in `backtest.engine.run_backtest`:
weights set on day t affect positions on day t+1. Each news source
must be (a) joined onto the t-indexed feature matrix and (b) consist of
data publicly visible by EOD t. Per-source analysis:

| Source | What date is the feature indexed by? | Public visibility | Safe? |
|---|---|---|---|
| Phase 12 | SEC filing date | Within minutes (SEC EDGAR pushes filings live) | ✅ |
| Phase 13 | SEC filing date | Within minutes | ✅ |
| Phase 14 | GDELT GKG date | Day-D file published ~24h after day D; we honor the t-1 rule | ✅ |
| Phase 15 | yfinance article timestamp | Live (~minutes); dashboard-only | N/A (not a feature) |

**Weekend/holiday filings**: For Phases 12, 13, and 14, filings dated on
a non-trading day are forward-shifted to the NEXT trading day via
`pd.merge_asof(direction='forward')`. Never retroactively applied to a
prior Friday. Tested by:

- `tests/test_edgar.py::test_fetch_quarter_8k_handles_weekend_filing`
- `tests/test_gdelt.py::test_build_gdelt_features_weekend_filing_forwards_to_monday`

---

## 7. Memory budget (8 GB / 8 vCPU target)

| Source | Cold-fetch peak RSS | Cold-fetch time | Cache size | Backtest-time RSS adder |
|---|---|---|---|---|
| Phase 12 | ~50 MB | ~4 min (44 qtr files at 5 sec each) | ~1.3 MB total | <20 MB |
| Phase 13 | ~30 MB | ~30 sec (150 ticker submissions JSONs) | ~50 MB total | <20 MB |
| Phase 14 | ~80 MB peak (streamed) | **2–3 hours** for 11 years | ~50 MB total | ~50 MB |
| Phase 15 | ~600 MB (model load) | ~10 sec model load + ~50 headlines/sec inference | per-headline JSON | N/A |

Production smoke with Phase 13 enabled: peak RSS **1.03 GB** (well
under the 6 GB ceiling). Add Phase 14: estimated peak **~1.1 GB**. Add
Phase 15 in the same process: **~1.7 GB** (FinBERT model alone is 600 MB).

If you OOM:
- Lower `FINBERT_BATCH_SIZE` from default 32 to 8 or 4.
- Run the bulk-fetch (`scripts/phase14_gdelt_bulk_fetch.py`) on a beefier
  one-off box, then rsync the parquet cache to the deploy box.
- Set `FINBERT_ENABLED=off` if dashboard sentiment isn't worth the RAM.

---

## 8. Known caveats

- **Dual-class ticker dedup (Phase 13 reviewer CRIT-1).** GOOG and
  GOOGL share CIK 0001652044. The pipeline dedups by CIK and mirrors
  the result into both ticker columns — verified by
  `tests/test_edgar.py::test_build_8k_item_features_dedupes_dual_class_tickers`.
- **Procter & Gamble parser bug (Phase 12 reviewer HIGH-3).** Real SEC
  company names can contain multiple consecutive spaces (e.g. "PROCTER
  &  GAMBLE CO"). The Phase 12 form.idx parser uses fixed-width
  column slicing (not whitespace splitting) to handle these. Regression
  test:
  `tests/test_edgar.py::test_fetch_quarter_8k_handles_procter_and_gamble_multi_space_name`.
- **2014 form.idx schema drift.** SEC's 2014 form.idx files use the
  header label `"File Name"` (with a space), not `"Filename"`. The
  parser accepts both. Regression test:
  `tests/test_edgar.py::test_parse_idx_header_accepts_alternate_filename_spelling`.
- **SEC submissions JSON pagination cap (Phase 13 reviewer HIGH-3).**
  `data.sec.gov/submissions/CIK*.json` returns only the most recent
  ~1000 filings under `filings.recent`; older filings spill into
  `filings.files[]` which we don't fetch. For mega-cap names that file
  >150 8-Ks per year, the OLDEST years of an 11-yr backtest may be
  under-counted. The pipeline logs a `WARNING` when this happens.
  Regression test:
  `tests/test_edgar.py::test_fetch_8k_items_warns_on_paginated_files`.
- **GDELT name matching is fuzzy.** GDELT mentions companies by NAME,
  not ticker. We strip common legal suffixes (INC, CORP, COMPANY,
  HOLDINGS, etc.) and reject names < 4 characters to avoid false
  positives (e.g. "CAT" matching the word "cat"). Some signal lost;
  this is the free-data ceiling.
- **HTTPError swallowing narrowed.** Earlier code silently turned every
  HTTPError into an empty DataFrame, hiding 429 (rate limit) and 403
  (User-Agent rejected) errors. Fixed: 404 still swallowed, 429 retries
  once with sleep, 403/5xx re-raised so operators see them. Regression
  tests:
  `tests/test_edgar.py::test_fetch_8k_items_{404_returns_empty,403_reraises,429_retries_once}`.

---

## 9. Files / commands reference

| Purpose | File or command |
|---|---|
| EDGAR client (Phase 12 + 13) | `src/stockpred/data/edgar.py` |
| GDELT client (Phase 14) | `src/stockpred/data/gdelt.py` |
| FinBERT live scorer (Phase 15) | `src/stockpred/data/sentiment.py` |
| yfinance news (Phase 15 input) | `src/stockpred/data/news.py` |
| Pipeline integration | `src/stockpred/pipeline_v5.py` (search `use_edgar_features`, `use_edgar_item_features`, `use_gdelt_features`) |
| CLI runner | `scripts/run_phase5.py --edgar-items --gdelt …` |
| GDELT bulk fetch | `scripts/phase14_gdelt_bulk_fetch.py --tickers-from-edgar` |
| HTTP refresh schema | `src/stockpred/backend/schemas.py` (`RefreshRequest`) |
| HTTP news endpoint | `src/stockpred/backend/api.py` (`/tickers/{ticker}/news`) |
| Tests (EDGAR) | `tests/test_edgar.py` (33 tests) |
| Tests (GDELT) | `tests/test_gdelt.py` (11 tests) |
| Tests (FinBERT) | `tests/test_sentiment.py` (7 tests) |
| Tests (HTTP schema) | `tests/test_backend_api.py` (14 tests) |

---

## 10. Quick-reference cheatsheet

| Goal | Command |
|---|---|
| One-time SEC ticker map warm | `uv run python -c "from stockpred.data import edgar; edgar.fetch_ticker_to_cik(refresh=True)"` |
| Run BEST honest config via CLI | `uv run python scripts/run_phase5.py --start 2014-01-01 --end 2024-12-31 --n-tickers 150 --horizons 5 --weighting equal --position-sizing hrp --k-pct 0.15 --sector-cap 0.30 --min-trade-threshold 0.005 --holdout-years 2 --no-sector --no-regime --no-tier2 --universe-sampling current --bootstrap-method block --meta-labelling --meta-threshold 0.55 --ranks-only --edgar-items` |
| Run BEST honest config via HTTP | See §3 (`Daily runs`) for full curl. |
| Start GDELT overnight bulk fetch | `nohup uv run python scripts/phase14_gdelt_bulk_fetch.py --tickers-from-edgar > logs/phase14.log 2>&1 &` |
| Download FinBERT model | `huggingface-cli download ProsusAI/finbert --local-dir models/finbert` |
| Check FinBERT availability | `uv run python -c "from stockpred.data.sentiment import is_available; print(is_available())"` |
| Get news + sentiment for a ticker | `curl http://localhost:8000/tickers/AAPL/news?limit=20` |
| Disable FinBERT globally | `export FINBERT_ENABLED=off` |
