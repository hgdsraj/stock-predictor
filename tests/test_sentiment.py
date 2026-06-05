"""Phase 15 tests: FinBERT sentiment scorer (live-mode, dashboard only).

These tests do NOT load real FinBERT. They verify:
  - Lazy import: module imports cleanly even when transformers is missing
  - Graceful degradation when the pipeline can't load (returns 'unavailable')
  - Cache round-trip per headline hash
  - Empty / whitespace-only inputs handled
  - Cached scores re-used (no re-inference on second call)
  - is_available() returns False when transformers absent (test env)
"""

from __future__ import annotations

import pytest

from stockpred.data import sentiment


@pytest.fixture(autouse=True)
def isolate_cache(tmp_path, monkeypatch):
    """Per-test sentiment cache dir + reset singleton pipeline."""
    monkeypatch.setattr(sentiment, "CACHE_DIR_SENTIMENT", tmp_path)
    monkeypatch.setattr(
        sentiment,
        "_cache_path",
        lambda h: tmp_path / f"{h}.json",
    )
    # Reset the singleton between tests so we can test multiple
    # availability scenarios in one run.
    sentiment._LazyPipeline._instance = None
    return tmp_path


def test_module_imports_without_transformers():
    """The module MUST import cleanly even when transformers/torch
    aren't installed; the heavy imports happen lazily inside the
    pipeline class."""
    from stockpred.data import sentiment as s

    assert hasattr(s, "score_headlines")
    assert hasattr(s, "is_available")


def test_score_headlines_empty_list_returns_empty():
    assert sentiment.score_headlines([]) == []


def test_score_headlines_returns_unavailable_when_no_model(monkeypatch):
    """When FinBERT isn't loadable, every input should get the
    'unavailable' marker score and the backend must still be able to
    serialize it."""

    # Force the lazy pipeline to fail loading
    def _fake_load(self):
        self._tried_load = True
        self._load_error = "test: transformers not installed"
        self._pipe = None

    monkeypatch.setattr(sentiment._LazyPipeline, "_load", _fake_load)

    out = sentiment.score_headlines(["Apple beats earnings", "Tesla misses"])
    assert len(out) == 2
    for s in out:
        assert s["label"] == "unavailable"
        assert s["positive"] == 0.0
        assert s["net"] == 0.0


def test_score_headlines_caches_results(monkeypatch):
    """Once a headline is scored, a second call must not re-run inference."""
    inference_calls = {"n": 0}

    def _fake_load(self):
        self._tried_load = True

        # Mock the HF pipeline's __call__ signature: takes a list of
        # strings, returns list[list[{label, score}]] when top_k=None.
        def _fake_call(texts, **kwargs):
            inference_calls["n"] += 1
            return [
                [
                    {"label": "positive", "score": 0.8},
                    {"label": "neutral", "score": 0.1},
                    {"label": "negative", "score": 0.1},
                ]
                for _ in texts
            ]

        self._pipe = _fake_call

    monkeypatch.setattr(sentiment._LazyPipeline, "_load", _fake_load)
    # First call: hits inference
    out1 = sentiment.score_headlines(["Apple beats earnings"])
    assert out1[0]["label"] == "positive"
    assert inference_calls["n"] == 1

    # Second call with the same headline: cache hit, no inference.
    out2 = sentiment.score_headlines(["Apple beats earnings"])
    assert out2[0]["label"] == "positive"
    assert inference_calls["n"] == 1  # unchanged


def test_score_headlines_mixed_cached_and_new(monkeypatch):
    """A batch with some cached and some new headlines should only
    score the new ones."""
    # Pre-seed the cache for one headline
    h1 = sentiment._headline_hash("Already scored")
    sentiment._write_cache(
        h1,
        {
            "label": "positive",
            "positive": 0.9,
            "neutral": 0.05,
            "negative": 0.05,
            "net": 0.85,
        },
    )

    new_calls = {"texts": []}

    def _fake_load(self):
        self._tried_load = True

        def _fake_call(texts, **kwargs):
            new_calls["texts"].extend(texts)
            return [
                [
                    {"label": "positive", "score": 0.1},
                    {"label": "neutral", "score": 0.2},
                    {"label": "negative", "score": 0.7},
                ]
                for _ in texts
            ]

        self._pipe = _fake_call

    monkeypatch.setattr(sentiment._LazyPipeline, "_load", _fake_load)

    out = sentiment.score_headlines(["Already scored", "Brand new"])
    # Old one comes from cache
    assert out[0]["net"] == 0.85
    # New one comes from inference
    assert out[1]["label"] == "negative"
    # Only the new headline went to FinBERT.
    assert new_calls["texts"] == ["Brand new"]


def test_score_headlines_handles_empty_strings(monkeypatch):
    """Empty / whitespace-only titles get the 'unavailable' marker
    without ever invoking the model."""
    invoked = {"n": 0}

    def _fake_load(self):
        self._tried_load = True

        def _fake_call(texts, **kwargs):
            invoked["n"] += 1
            return [
                [
                    {"label": "positive", "score": 0.9},
                    {"label": "neutral", "score": 0.05},
                    {"label": "negative", "score": 0.05},
                ]
                for _ in texts
            ]

        self._pipe = _fake_call

    monkeypatch.setattr(sentiment._LazyPipeline, "_load", _fake_load)

    out = sentiment.score_headlines(["", "   ", ""])
    assert all(s["label"] == "unavailable" for s in out)
    assert invoked["n"] == 0  # never called for empty inputs


def test_is_available_returns_false_when_transformers_missing():
    """In the test env (no transformers), is_available() must be False."""
    # Don't monkeypatch _load; let it try the real import and fail.
    assert sentiment.is_available() is False
