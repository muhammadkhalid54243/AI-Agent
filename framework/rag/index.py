"""Hybrid retrieval — dense + lexical, fused. The two channels fail on DISJOINT
inputs: dense embeddings smear exact identifiers (error codes, clause numbers,
surnames, acronyms) into "things that look similar", while BM25 nails them and
fails on paraphrase. Neither dominates; the ensemble does. BM25 also brings IDF for
free — it down-weights ubiquitous terms (the HRCP problem we hand-gated in Phase 3).

Reciprocal Rank Fusion (RRF) combines the two rankings without needing to reconcile
their incomparable score scales — it uses only rank position, so it needs no
per-corpus tuning.
"""

import re

from rank_bm25 import BM25Okapi

from framework.rag.chunk_index import ChunkIndex
from framework.rag.chunker import leaf_chunks


def _tok(text: str) -> list:
    return re.findall(r"\w+", text.lower())


def reciprocal_rank_fusion(ranked_lists: list, k: int = 60) -> list:
    """Fuse ranked id-lists into one. score(id) = sum 1/(k + rank). Higher = better."""
    scores = {}
    for lst in ranked_lists:
        for rank, id_ in enumerate(lst):
            scores[id_] = scores.get(id_, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class HybridIndex:
    def __init__(self, chunks: list, embeddings, *, rrf_k: int = 60):
        self._leaves = leaf_chunks(chunks)
        self._by_id = {c.id: c for c in self._leaves}
        self._ids = [c.id for c in self._leaves]
        self._dense = ChunkIndex(chunks, embeddings)
        self._bm25 = BM25Okapi([_tok(c.embed_text) for c in self._leaves])
        self._rrf_k = rrf_k

    def retrieve(self, query: str, k: int, *, candidates: int = 100) -> list:
        """Retrieve `candidates` from each channel, RRF-fuse, return the top `k` hits."""
        dense_ids = [h["chunk_id"] for h in self._dense.retrieve(query, candidates)]
        scores = self._bm25.get_scores(_tok(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidates]
        bm25_ids = [self._ids[i] for i in ranked]
        fused = reciprocal_rank_fusion([dense_ids, bm25_ids], self._rrf_k)[:k]
        return [self._hit(i) for i in fused]

    def _hit(self, cid: str) -> dict:
        c = self._by_id[cid]
        return {"text": c.text, "page": c.page_start, "section_path": list(c.section_path),
                "chunk_id": c.id, "kind": c.kind, "doc_id": c.doc_id}
