"""Phase 15: FinBERT-based news sentiment scoring (live-mode, DASHBOARD ONLY).

This module wraps ProsusAI/FinBERT (a 110-M-param BERT classifier
fine-tuned on Financial PhraseBank) to score news headlines as
positive / neutral / negative with confidence.

Critical constraints (please honor them):
  - DASHBOARD-ONLY: never use FinBERT scores as a backtest feature.
    yfinance news only goes back ~30 days, so using sentiment-scored
    headlines as a feature would create catastrophic walk-forward
    selection bias.
  - LAZY-LOAD: `transformers` + `torch` are ~1.5 GB. We import them
    ONLY on first use of the scorer, and gracefully return "model not
    available" if the imports fail (deploy boxes without the heavy
    deps stay lightweight).
  - 8 GB RAM target: peak RSS during inference is ~600 MB for the
    model itself + ~50 MB working memory. Set FINBERT_BATCH_SIZE low
    (default 32) on memory-constrained boxes.

Operator setup:
  pip install transformers torch  # ~1.5 GB
  # Then either env (preferred):
  export FINBERT_MODEL_DIR="/path/to/local/finbert"
  huggingface-cli download ProsusAI/finbert --local-dir "$FINBERT_MODEL_DIR"
  # ... or let the first call download to ~/.cache/huggingface/...

Per-headline scores are cached to disk by sha256(title) so repeated
backend calls don't re-run inference.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import TypedDict

from stockpred.config import CACHE_DIR

log = logging.getLogger(__name__)

CACHE_DIR_SENTIMENT = CACHE_DIR / "sentiment"
CACHE_DIR_SENTIMENT.mkdir(parents=True, exist_ok=True)

_FINBERT_MODEL_DIR = os.environ.get("FINBERT_MODEL_DIR") or "ProsusAI/finbert"
_FINBERT_BATCH_SIZE = int(os.environ.get("FINBERT_BATCH_SIZE", "32"))
_FINBERT_ENABLED = os.environ.get("FINBERT_ENABLED", "auto").lower()


class SentimentScore(TypedDict):
    """One headline's score."""

    label: str  # 'positive' | 'neutral' | 'negative' | 'unavailable'
    positive: float
    neutral: float
    negative: float
    # Net = positive - negative, in [-1, +1]. Convenient single-number summary.
    net: float


_UNAVAILABLE_SCORE: SentimentScore = {
    "label": "unavailable",
    "positive": 0.0,
    "neutral": 0.0,
    "negative": 0.0,
    "net": 0.0,
}


def _headline_hash(text: str) -> str:
    """Cache key for a headline; sha256 of normalised (lowercase) text."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _cache_path(headline_hash: str) -> Path:
    # Bucket by first 2 hex chars to avoid one huge directory.
    sub = CACHE_DIR_SENTIMENT / headline_hash[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{headline_hash}.json"


def _read_cache(headline_hash: str) -> SentimentScore | None:
    p = _cache_path(headline_hash)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        log.warning("Sentiment cache %s unreadable (%s); ignoring.", p, e)
        return None


def _write_cache(headline_hash: str, score: SentimentScore) -> None:
    try:
        _cache_path(headline_hash).write_text(json.dumps(score))
    except Exception as e:  # noqa: BLE001
        log.warning("Could not write sentiment cache for %s: %s", headline_hash, e)


class _LazyPipeline:
    """Holds the FinBERT pipeline; loaded on first use.

    `__call__` returns either a list of SentimentScore or None if the
    model isn't available. We intentionally swallow ALL import + model-
    load errors and degrade to None so the backend stays alive when
    transformers/torch aren't installed.
    """

    _instance: "_LazyPipeline | None" = None

    def __init__(self) -> None:
        self._pipe = None
        self._tried_load = False
        self._load_error: str | None = None

    def _load(self) -> None:
        if self._tried_load:
            return
        self._tried_load = True
        if _FINBERT_ENABLED == "off":
            self._load_error = "FINBERT_ENABLED=off"
            log.info("FinBERT disabled via env (FINBERT_ENABLED=off).")
            return
        try:
            # Heavy imports happen ONLY here.
            from transformers import (  # type: ignore[import-not-found]
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline as hf_pipeline,
            )
        except ImportError as e:
            self._load_error = f"transformers not installed: {e}"
            log.warning(
                "FinBERT not available: %s. Run `pip install transformers torch` "
                "(~1.5 GB) and optionally `huggingface-cli download "
                "ProsusAI/finbert --local-dir $FINBERT_MODEL_DIR`.",
                e,
            )
            return
        try:
            log.info(
                "Loading FinBERT model from %s (may download ~440 MB on first run)...",
                _FINBERT_MODEL_DIR,
            )
            tok = AutoTokenizer.from_pretrained(_FINBERT_MODEL_DIR)
            mdl = AutoModelForSequenceClassification.from_pretrained(_FINBERT_MODEL_DIR)
            self._pipe = hf_pipeline(
                "text-classification",
                model=mdl,
                tokenizer=tok,
                top_k=None,  # return all class scores
            )
            log.info("FinBERT model ready (batch_size=%d).", _FINBERT_BATCH_SIZE)
        except Exception as e:  # noqa: BLE001
            self._load_error = f"model load failed: {e}"
            log.warning("FinBERT model load failed (%s); scoring will be unavailable.", e)
            self._pipe = None

    def available(self) -> bool:
        self._load()
        return self._pipe is not None

    def __call__(self, texts: list[str]) -> list[SentimentScore] | None:
        self._load()
        if self._pipe is None:
            return None
        # Truncate to FinBERT's 512-token limit at character level
        # (rough but safe; a 4-char-per-token average).
        clean = [(t or "").strip()[:2000] for t in texts]
        non_empty = [t for t in clean if t]
        if not non_empty:
            return [_UNAVAILABLE_SCORE.copy() for _ in texts]
        try:
            raw = self._pipe(
                non_empty,
                batch_size=_FINBERT_BATCH_SIZE,
                truncation=True,
                padding=True,
            )
        except Exception as e:  # noqa: BLE001
            log.warning("FinBERT inference failed: %s", e)
            return None
        # `raw` is list[list[{label, score}]] when top_k=None
        scored: list[SentimentScore] = []
        for entry in raw:
            score_map = {item["label"].lower(): float(item["score"]) for item in entry}
            pos = score_map.get("positive", 0.0)
            neu = score_map.get("neutral", 0.0)
            neg = score_map.get("negative", 0.0)
            label = max(("positive", pos), ("neutral", neu), ("negative", neg), key=lambda x: x[1])[
                0
            ]
            scored.append(
                {
                    "label": label,
                    "positive": pos,
                    "neutral": neu,
                    "negative": neg,
                    "net": pos - neg,
                }
            )
        # Re-interleave with the empty-text inputs.
        out: list[SentimentScore] = []
        idx = 0
        for t in clean:
            if t:
                out.append(scored[idx])
                idx += 1
            else:
                out.append(_UNAVAILABLE_SCORE.copy())
        return out


def _get_pipeline() -> _LazyPipeline:
    if _LazyPipeline._instance is None:
        _LazyPipeline._instance = _LazyPipeline()
    return _LazyPipeline._instance


def score_headlines(titles: list[str]) -> list[SentimentScore]:
    """Score a list of headlines. Returns one SentimentScore per input
    (in the same order). Cache-first; only un-cached titles hit FinBERT.

    Honors the lazy import + graceful degradation policy: if FinBERT
    isn't available, returns 'unavailable' for everything.
    """
    if not titles:
        return []

    hashes = [_headline_hash(t) for t in titles]
    cached: list[SentimentScore | None] = [_read_cache(h) for h in hashes]
    # Find indices that need scoring.
    missing_idx = [i for i, c in enumerate(cached) if c is None]
    if not missing_idx:
        return [c for c in cached if c is not None]  # type: ignore[misc]

    pipe = _get_pipeline()
    if not pipe.available():
        # Degrade: return cached where present, 'unavailable' for rest.
        return [c if c is not None else _UNAVAILABLE_SCORE.copy() for c in cached]

    to_score = [titles[i] for i in missing_idx]
    scored = pipe(to_score)
    if scored is None:
        return [c if c is not None else _UNAVAILABLE_SCORE.copy() for c in cached]
    # Cache new scores
    for j, i in enumerate(missing_idx):
        cached[i] = scored[j]
        _write_cache(hashes[i], scored[j])
    return [c if c is not None else _UNAVAILABLE_SCORE.copy() for c in cached]


def is_available() -> bool:
    """Cheap probe for /healthz to report whether FinBERT is wired up."""
    return _get_pipeline().available()
