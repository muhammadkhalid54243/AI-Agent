"""ResilientEmbeddings — deterministic, no API key, no network."""

import pytest
from langchain_core.embeddings import Embeddings

from framework.rag.embeddings import ResilientEmbeddings, EmbeddingCapReached


class FakeInner(Embeddings):
    """Counts API calls; can be told to raise a rate-limit error the first N times."""
    def __init__(self, fail_times=0):
        self.calls = 0
        self.fail_times = fail_times

    def embed_documents(self, texts):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [[float(len(t)), 0.0] for t in texts]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


def _wrap(inner, tmp_path, **kw):
    kw.setdefault("throttle_s", 0)      # no real sleeping in tests
    kw.setdefault("max_backoff_s", 0)
    return ResilientEmbeddings(inner, model_tag="t", cache_dir=str(tmp_path), **kw)


def test_cache_makes_reruns_free(tmp_path):
    inner = FakeInner()
    emb = _wrap(inner, tmp_path, batch_size=2)
    emb.embed_documents(["aa", "bbb", "cccc"])
    first_calls = inner.calls
    assert first_calls > 0
    # Second identical run hits cache only — zero new API calls.
    emb2 = _wrap(FakeInner(), tmp_path, batch_size=2)
    emb2.embed_documents(["aa", "bbb", "cccc"])
    assert emb2._inner.calls == 0


def test_normalized_unit_length(tmp_path):
    emb = _wrap(FakeInner(), tmp_path, normalize=True)
    v = emb.embed_query("hello")
    assert abs(sum(x * x for x in v) - 1.0) < 1e-9


def test_backoff_then_success(tmp_path):
    inner = FakeInner(fail_times=2)          # 429 twice, then succeeds
    emb = _wrap(FakeInner(fail_times=2), tmp_path, max_retries=5)
    out = emb.embed_documents(["x"])
    assert out and emb._inner.calls == 3     # 2 failures + 1 success


def test_call_cap_halts_and_is_resumable(tmp_path):
    inner = FakeInner()
    emb = _wrap(inner, tmp_path, batch_size=1, max_calls=2)
    with pytest.raises(EmbeddingCapReached):
        emb.embed_documents(["a", "b", "c", "d"])   # 3rd batch trips the cap
    # What did fit is cached, so a resumed run does less work.
    assert inner.calls == 2


def test_make_embeddings_rejects_unknown_provider():
    # Name is resolved BEFORE any heavy model import, so a bad provider fails fast.
    from framework.rag.embeddings import make_embeddings
    with pytest.raises(ValueError):
        make_embeddings("not-a-real-embedder")
