"""Phase 5 agentic retrieval — the 5 structure-aware tools. Offline (fake embeddings)."""

from langchain_core.embeddings import DeterministicFakeEmbedding

from framework.ingest.ir import Block, Coverage, ParsedDoc, OutlineNode, GlossaryEntry
from framework.rag.retriever import DocumentRetriever


def _doc():
    blocks = [
        Block(id="d:p0:b0", kind="heading", text="Chapter 1", page=0, section_path=()),
        Block(id="d:p0:b1", kind="paragraph", text="The ReAct pattern loops think act observe.",
              page=0, section_path=("Chapter 1",)),
        Block(id="d:p0:b2", kind="paragraph", text="Charges were filed under PECA in 2024.",
              page=0, section_path=("Chapter 1",)),
        Block(id="d:p0:b3", kind="table", text="|Metric|Value|\n|---|---|\n|cases|42|",
              page=0, section_path=("Chapter 1",)),
    ]
    return ParsedDoc(doc_id="d", source_path="x.pdf", checksum="c", profile="report",
                     blocks=blocks, coverage=Coverage(blocks_extracted=len(blocks)),
                     outline=[OutlineNode(1, "Chapter 1", 0)],
                     glossary={"PECA": GlossaryEntry(expansion="Prevention of Electronic Crimes Act")})


def _tools():
    dr = DocumentRetriever(_doc(), embeddings=DeterministicFakeEmbedding(size=32))
    return {t.name: t for t in dr.as_tools()}


def test_exposes_five_structure_aware_tools():
    names = set(_tools())
    assert names == {"search_documents", "list_outline", "get_section", "lookup_term", "get_table"}


def test_list_outline_returns_toc():
    assert "Chapter 1" in _tools()["list_outline"].invoke({})


def test_lookup_term_hits_glossary():
    tools = _tools()
    assert "Prevention of Electronic Crimes Act" in tools["lookup_term"].invoke({"term": "PECA"})
    assert "not in the glossary" in tools["lookup_term"].invoke({"term": "ZZZ"})


def test_get_section_returns_section_text():
    out = _tools()["get_section"].invoke({"section_title": "chapter 1"})
    assert "ReAct pattern" in out


def test_get_table_returns_verbatim_markdown():
    out = _tools()["get_table"].invoke({"query": "Metric"})
    assert "|Metric|Value|" in out and "cases|42" in out


def test_search_documents_returns_hits():
    out = _tools()["search_documents"].invoke({"query": "ReAct pattern"})
    assert isinstance(out, str) and len(out) > 0
