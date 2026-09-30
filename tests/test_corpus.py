"""Multi-doc corpus retrieval — offline (fake embeddings).

Two near-identical reports that differ only by year + a unique clause id. The point of
multi-doc: every hit must name the report it came from (doc_id), and an exact term that
lives in one report must be attributed to THAT report — the cross-year confusion the
gold-set doc-attribution metric guards against.
"""

from langchain_core.embeddings import DeterministicFakeEmbedding

from framework.ingest.ir import Block, Coverage, GlossaryEntry, OutlineNode, ParsedDoc
from framework.rag.corpus import CorpusRetriever, _label


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


def test_hits_carry_doc_id():
    hits = _corpus()._search("enforced disappearance cases", k=5)
    assert hits and all("doc_id" in h for h in hits)


def test_corpus_spans_both_reports():
    # A term both reports share must surface both — the index isn't siloed per doc.
    hits = _corpus()._search("enforced disappearance", k=4)
    assert {h["doc_id"] for h in hits} == {"2023_SHR_English", "2024_SHR_English"}


def test_exact_term_retrieves_owning_report():
    # BM25 guarantees an exact rare id lands in the top-k of the report that holds it
    # (offline: fake dense embeddings are noise, so we assert presence, not rank-0 —
    # attribution *quality* under real semantics is measured in examples/multidoc_eval.py).
    c = _corpus()
    assert any(h["doc_id"] == "2023_SHR_English" and "ALPHA-2301" in h["text"]
               for h in c._search("ALPHA-2301", k=4))
    assert any(h["doc_id"] == "2024_SHR_English" and "BRAVO-2402" in h["text"]
               for h in c._search("BRAVO-2402", k=4))


def test_doc_label_reads_as_report_year():
    assert _label(_doc("2024_SHR_English", "2024", "X")) == "SHR 2024"
    assert _corpus().doc_label("2023_SHR_English") == "SHR 2023"


def test_merged_glossary_available():
    assert _corpus().glossary["HRCP"].expansion == "Human Rights Commission of Pakistan"
