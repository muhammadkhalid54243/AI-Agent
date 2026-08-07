from langchain_core.embeddings import DeterministicFakeEmbedding
from framework.rag import Retriever
from framework.rag.chunker import chunk_text


def test_chunk_text_splits_with_overlap():
    words = " ".join(str(i) for i in range(700))
    chunks = chunk_text(words, chunk_size=300, overlap=50)
    assert len(chunks) >= 3
    assert all(isinstance(c, str) for c in chunks)


def test_retriever_as_tool_retrieves():
    emb = DeterministicFakeEmbedding(size=64)
    r = Retriever(
        ["The ReAct pattern loops think act observe.",
         "RAG grounds answers in retrieved documents.",
         "MCP standardizes tool access."],
        embeddings=emb, k=1, chunk_size=50, overlap=5,
    )
    tool = r.as_tool()
    assert tool.name == "search_documents"
    result = tool.invoke({"query": "documents"})
    assert isinstance(result, str) and len(result) > 0
