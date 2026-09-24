from langchain_core.embeddings import DeterministicFakeEmbedding
from framework.rag import Retriever
from framework.rag.chunker import chunk_text


def test_chunk_text_splits_with_overlap():
    words = " ".join(str(i) for i in range(700))
    chunks = chunk_text(words, chunk_size=300, overlap=50)
    assert len(chunks) >= 3
    assert all(isinstance(c, str) for c in chunks)


def _retriever(k=1):
    emb = DeterministicFakeEmbedding(size=64)
    return Retriever(
        ["The ReAct pattern loops think act observe.",
         "RAG grounds answers in retrieved documents.",
         "MCP standardizes tool access."],
        embeddings=emb, k=k, chunk_size=50, overlap=5,
    )


def test_retriever_as_tool_retrieves():
    tool = _retriever().as_tool()
    assert tool.name == "search_documents"
    result = tool.invoke({"query": "documents"})
    assert isinstance(result, str) and len(result) > 0


def test_as_tools_returns_navigable_set():
    tools = _retriever().as_tools()
    names = {t.name for t in tools}
    assert {"search_documents", "lookup_term", "get_section"} <= names


def test_search_exposes_addressable_indices():
    # The loop depends on hits being addressable: search must surface an [idx N]
    # that get_section can then expand. Without this, chaining is impossible.
    tool = {t.name: t for t in _retriever(k=3).as_tools()}["search_documents"]
    out = tool.invoke({"query": "documents"})
    assert "[idx" in out


def test_lookup_term_is_lexical_not_semantic():
    # The vocabulary-mismatch fallback: an exact term the agent read in a chunk.
    tool = {t.name: t for t in _retriever().as_tools()}["lookup_term"]
    out = tool.invoke({"term": "MCP"})
    assert "MCP" in out


def test_get_section_expands_neighbors_and_guards_bounds():
    tools = {t.name: t for t in _retriever().as_tools()}
    got = tools["get_section"].invoke({"index": 0, "window": 1})
    assert isinstance(got, str) and len(got) > 0
    # Out-of-range index degrades gracefully, never raises.
    oob = tools["get_section"].invoke({"index": 9999, "window": 1})
    assert isinstance(oob, str)
