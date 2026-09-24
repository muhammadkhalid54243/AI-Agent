"""Phase 4 hybrid + rerank + expand — offline (fake dense embeddings, fake scorer)."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from framework.rag.chunker import Chunk
from framework.rag.index import HybridIndex, reciprocal_rank_fusion
from framework.rag.rerank import rerank
from framework.rag.expand import expand_to_parents


def _c(cid, text, parent=None, kind="prose"):
    return Chunk(id=cid, text=text, embed_text=text, doc_id="d", page_start=0, page_end=0,
                 section_path=("S",), kind=kind, parent_id=parent)


def test_rrf_ranks_agreement_highest():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert fused[0] == "a"                    # top of both lists


def test_hybrid_finds_exact_term_via_bm25():
    # An exact rare identifier the fake dense embedder can't rank -> BM25 must carry it.
    chunks = [_c(f"d:ck{i}", f"generic filler text number {i}") for i in range(8)]
    chunks.append(_c("d:ckX", "the incident report references clause XZ-4471 specifically"))
    idx = HybridIndex(chunks, DeterministicFakeEmbedding(size=32))
    hits = idx.retrieve("XZ-4471", k=3, candidates=20)
    assert any(h["chunk_id"] == "d:ckX" for h in hits)


def test_rerank_reorders_by_scorer():
    hits = [{"text": "irrelevant", "chunk_id": "1"}, {"text": "the answer is here", "chunk_id": "2"}]
    def scorer(pairs):
        return [1.0 if "answer" in t else 0.0 for _, t in pairs]
    out = rerank("q", hits, k=1, scorer=scorer)
    assert out[0]["chunk_id"] == "2"


def test_rerank_degrades_on_scorer_error():
    hits = [{"text": "a", "chunk_id": "1"}, {"text": "b", "chunk_id": "2"}]
    def boom(pairs):
        raise RuntimeError("model down")
    out = rerank("q", hits, k=2, scorer=boom)
    assert [h["chunk_id"] for h in out] == ["1", "2"]     # input order preserved


def test_expand_swaps_children_for_parent_and_dedupes():
    chunks = [
        _c("d:ckP", "the whole parent section text"),        # parent
        _c("d:ck1", "child one", parent="d:ckP"),
        _c("d:ck2", "child two", parent="d:ckP"),
        _c("d:ckL", "a lone leaf with no parent"),
    ]
    hits = [{"chunk_id": "d:ck1", "text": "child one"},
            {"chunk_id": "d:ck2", "text": "child two"},
            {"chunk_id": "d:ckL", "text": "a lone leaf with no parent"}]
    out = expand_to_parents(hits, chunks)
    ids = [h["chunk_id"] for h in out]
    assert ids == ["d:ckP", "d:ckL"]                          # two children -> one parent; leaf kept
    assert out[0]["text"] == "the whole parent section text"  # parent context returned
