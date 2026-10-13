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


# --- Phase D: query-shape dispatch (get_section / get_table / retrieve) ---

def _list_doc() -> ParsedDoc:
    """A doc whose 'Rights' section has 5 items long enough to split into several leaf
    chunks (so returning the whole section beats a small top-k), plus one table."""
    filler = "Everyone is entitled to it and its full protection under the law. " * 20
    blocks = [Block(id="D:p0:b0", kind="heading", text="Rights", page=0, section_path=())]
    for i, name in enumerate(["alpha", "bravo", "charlie", "delta", "echo"]):
        blocks.append(Block(id=f"D:p0:b{i+1}", kind="paragraph",
                            text=f"The right to {name}. {filler}",
                            page=0, section_path=("Rights",)))
    blocks.append(Block(id="D:p1:bt0", kind="table",
                        text="|City|Count|\n|---|---|\n|Lahore|5|\n|Karachi|9|",
                        page=1, section_path=("Crime data",)))
    return ParsedDoc(doc_id="D", source_path="d.pdf", checksum="c", profile="report",
                     blocks=blocks, coverage=Coverage(blocks_extracted=len(blocks)),
                     outline=[OutlineNode(1, "Rights", 0), OutlineNode(1, "Crime data", 1)])


def _list_corpus() -> CorpusRetriever:
    return CorpusRetriever([_list_doc()], embeddings=DeterministicFakeEmbedding(size=32))


def test_get_section_returns_whole_section_as_multiple_leaves():
    hits = _list_corpus().get_section("D", "Rights")
    blob = " ".join(h["text"] for h in hits)
    assert all(name in blob for name in ["alpha", "bravo", "charlie", "delta", "echo"])
    assert len(hits) >= 3                       # the section really did span several leaves


def test_retrieve_dispatches_enumerate_to_the_whole_section():
    # k=1 would give ONE chunk via _search; the enumerate dispatch returns the whole section.
    hits = _list_corpus().retrieve("list every right entitled protection", 1)
    blob = " ".join(h["text"] for h in hits)
    assert all(name in blob for name in ["alpha", "bravo", "charlie", "delta", "echo"])


def test_retrieve_fact_query_honors_k():
    hits = _list_corpus().retrieve("how many rights entitled protection", 2)
    assert len(hits) <= 2                        # fact path is the ordinary top-k search


def test_get_table_returns_verbatim():
    hits = _list_corpus().get_table("D", "City")
    assert hits and "|City|Count|" in hits[0]["text"] and "Karachi|9" in hits[0]["text"]
