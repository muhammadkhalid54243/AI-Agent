"""Deterministic abstention gate — offline (fake embeddings, no model).

The gate moves answer-vs-abstain OUT of the stochastic model: below a retrieval-confidence
threshold the served path returns NOT_COVERED with no model call. These check the signal is
surfaced and the threshold logic is sound; the actual threshold VALUE is calibrated against
gold in examples/abstention_eval.py, not asserted here (fake embeddings are noise by design).
"""

from langchain_core.embeddings import DeterministicFakeEmbedding

from framework.ingest.ir import Block, Coverage, GlossaryEntry, OutlineNode, ParsedDoc
from framework.rag.corpus import CorpusRetriever


def _doc(doc_id: str, year: str, clause: str) -> ParsedDoc:
    blocks = [
        Block(id=f"{doc_id}:p0:b0", kind="heading", text="Key issues", page=0, section_path=()),
        Block(id=f"{doc_id}:p0:b1", kind="paragraph",
              text=f"In {year} the commission reviewed enforced-disappearance cases.",
              page=0, section_path=("Key issues",)),
        Block(id=f"{doc_id}:p0:b2", kind="paragraph",
              text=f"The tribunal logged clause {clause} for review in {year}.",
              page=0, section_path=("Key issues",)),
    ]
    return ParsedDoc(doc_id=doc_id, source_path=f"{doc_id}.pdf", checksum="c", profile="report",
                     blocks=blocks, coverage=Coverage(blocks_extracted=len(blocks)),
                     outline=[OutlineNode(1, "Key issues", 0)],
                     glossary={"HRCP": GlossaryEntry(expansion="Human Rights Commission of Pakistan")})


def _corpus() -> CorpusRetriever:
    docs = [_doc("2023_SHR_English", "2023", "ALPHA-2301"),
            _doc("2024_SHR_English", "2024", "BRAVO-2402")]
    return CorpusRetriever(docs, embeddings=DeterministicFakeEmbedding(size=32))


def test_hits_carry_channel_scores():
    hits = _corpus()._search("enforced disappearance cases", k=5)
    assert hits
    assert all("dense_score" in h and "bm25_score" in h for h in hits)


def test_confidence_is_the_max_signal():
    c = _corpus()
    hits = c._search("enforced disappearance cases", k=5)
    assert isinstance(c.confidence(hits), float)
    assert c.confidence(hits) == max(h["dense_score"] for h in hits if h["dense_score"] is not None)
    assert c.confidence([]) is None


def test_confident_is_monotonic_in_the_dense_threshold():
    # override disabled (min_bm25 huge) so only cosine decides; cosine is in [-1, 1].
    c = _corpus()
    hits = c._search("enforced disappearance cases", k=5)
    assert c.confident(hits, min_dense=-1.0, min_bm25=1e9)        # everything clears -1
    assert not c.confident(hits, min_dense=1.01, min_bm25=1e9)    # nothing exceeds cosine 1


def test_bm25_override_answers_a_strong_exact_match_at_low_cosine():
    # An exact rare clause id: dense cosine is unreliable, BM25 is certain -> must answer.
    c = _corpus()
    hits = c._search("ALPHA-2301", k=4)
    bmax = max((h["bm25_score"] or 0.0) for h in hits)
    assert bmax > 0
    # cosine gate impossible to pass (1.01) + no reranker -> only the override can answer.
    assert c.confident(hits, min_dense=1.01, min_rerank=1e9, min_bm25=bmax)
    assert not c.confident(hits, min_dense=1.01, min_rerank=1e9, min_bm25=bmax + 1.0)


def test_rerank_score_takes_over_the_gate_when_present():
    # When the reranker ran, the gate judges on its scale and ignores dense cosine entirely.
    c = _corpus()
    rhits = [{"text": "x", "dense_score": -0.9, "bm25_score": 0.0, "rerank_score": 3.0}]
    assert c.confident(rhits, min_rerank=2.0, min_dense=1.01, min_bm25=1e9)
    assert not c.confident(rhits, min_rerank=5.0, min_dense=1.01, min_bm25=1e9)


def test_no_hits_never_confident():
    assert _corpus().confident([]) is False
