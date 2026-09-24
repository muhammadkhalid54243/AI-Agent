"""Resilient, cached embeddings — makes a rate-limited / free-tier provider usable
for offline ingestion.

Why this exists and not `resilience()`: that middleware retries chat-model calls
inside the agent loop. Embeddings never touch the agent loop — they go straight
through the vector store to the provider SDK — so the retry has to live here, at
the embeddings layer.

Wraps ANY LangChain Embeddings and adds, without changing the interface:
  - exponential backoff on 429 / RESOURCE_EXHAUSTED (rate limits are transient);
  - a throttle between sub-batches so a burst of large batches doesn't blow the
    tokens-per-minute ceiling (the actual failure we diagnosed);
  - a hard cap on API calls (CLAUDE.md: halt and report, never silently spend);
  - an on-disk cache keyed by (model_tag, text) hash, so a re-run is free and an
    interrupted run resumes from what it already embedded.

Errors that survive backoff are raised once, with the partial cache already on
disk — the caller decides whether to resume.
"""

import hashlib
import json
import math
import os
import time

from langchain_core.embeddings import Embeddings


class EmbeddingCapReached(RuntimeError):
    """Raised when the API-call cap trips. The cache is written; the run is resumable."""


def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "resource_exhausted" in s or "rate limit" in s


def _l2_normalize(vec: list) -> list:
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


class ResilientEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, *, model_tag: str,
                 cache_dir: str = ".rag_cache/emb", batch_size: int = 50,
                 throttle_s: float = 2.0, max_retries: int = 8,
                 max_backoff_s: float = 60.0, max_calls: int | None = 400,
                 normalize: bool = True):
        self._inner = inner
        self._tag = model_tag
        self._cache_dir = cache_dir
        self._batch = batch_size
        self._throttle = throttle_s
        self._max_retries = max_retries
        self._max_backoff = max_backoff_s
        self._max_calls = max_calls
        self._normalize = normalize
        self._calls = 0
        os.makedirs(cache_dir, exist_ok=True)

    # --- cache -------------------------------------------------------------
    def _key(self, text: str) -> str:
        h = hashlib.sha256(f"{self._tag}\x00{text}".encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir, f"{h}.json")

    def _cache_get(self, text: str):
        p = self._key(text)
        if os.path.exists(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None
        return None

    def _cache_put(self, text: str, vec: list) -> None:
        try:
            with open(self._key(text), "w", encoding="utf-8") as f:
                json.dump(vec, f)
        except Exception:
            pass  # a cache write failure must not abort ingestion

    # --- one resilient batch call -----------------------------------------
    def _embed_batch(self, batch: list) -> list:
        if self._max_calls is not None and self._calls >= self._max_calls:
            raise EmbeddingCapReached(
                f"embedding API-call cap reached ({self._max_calls}); "
                "partial cache written, re-run to resume.")
        for attempt in range(self._max_retries + 1):
            try:
                self._calls += 1
                vecs = self._inner.embed_documents(batch)
                return [_l2_normalize(v) if self._normalize else v for v in vecs]
            except Exception as exc:
                if _is_rate_limit(exc) and attempt < self._max_retries:
                    time.sleep(min(2 ** attempt, self._max_backoff))
                    continue
                raise

    # --- Embeddings interface ---------------------------------------------
    def embed_documents(self, texts: list) -> list:
        out: list = [None] * len(texts)
        misses: list = []       # indices needing an API call
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                out[i] = cached
            else:
                misses.append(i)

        for start in range(0, len(misses), self._batch):
            group = misses[start:start + self._batch]
            vecs = self._embed_batch([texts[i] for i in group])
            for i, v in zip(group, vecs):
                out[i] = v
                self._cache_put(texts[i], v)
            if start + self._batch < len(misses):
                time.sleep(self._throttle)   # pace to stay under TPM
        return out

    def embed_query(self, text: str) -> list:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vec = self._embed_batch([text])[0]
        self._cache_put(text, vec)
        return vec


# --- provider factory: one place to swap the embedder ---------------------
# PRIMARY = bge-m3 (local, Apache-2.0, no quota, matched/beat Gemini on the SHR
# gold set). SECONDARY = Gemini (frontier, needs GOOGLE_API_KEY). Swap via the
# `provider` arg or the EMBED_PROVIDER env var. Heavy deps import lazily so the
# module stays cheap to import and unit tests (fake embeddings) never load torch.

def _bge_m3():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True})  # bge-m3 takes NO query prefix


def _gemini_768():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", google_api_key=key,
        output_dimensionality=768)


def make_embeddings(provider: str | None = None, *, cache: bool = True, **cache_kw):
    """Build the configured embedder, wrapped in the resilient cache by default.

    provider: "bge-m3" (default/primary) or "gemini" (secondary); falls back to the
    EMBED_PROVIDER env var, then bge-m3. Extra kwargs pass to ResilientEmbeddings.
    """
    provider = (provider or os.getenv("EMBED_PROVIDER") or "bge-m3").lower()
    if provider in ("bge-m3", "bge", "local"):
        tag, normalize, defaults = "bge-m3", False, {"throttle_s": 0.0, "max_calls": None}
        build = _bge_m3
    elif provider in ("gemini", "google"):
        tag, normalize, defaults = "gemini-embedding-001@768", True, {"throttle_s": 3.0, "batch_size": 32}
        build = _gemini_768
    else:
        raise ValueError(f"unknown embedder {provider!r}; use 'bge-m3' or 'gemini'")
    inner = build()
    if not cache:
        return inner
    defaults.update(cache_kw)
    return ResilientEmbeddings(inner, model_tag=tag, normalize=normalize, **defaults)
