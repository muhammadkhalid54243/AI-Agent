"""RAG retriever — chunk + embed texts, expose semantic search as a tool.

Wraps an in-memory vector store. Default embeddings are Google's (needs
GOOGLE_API_KEY); inject any Embeddings for tests or other providers. `.as_tool()`
returns a standard LangChain tool the agent can call to ground its answers.
"""

import os

from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import InMemoryVectorStore

from framework.rag.chunker import chunk_text


class Retriever:
    def __init__(self, texts, *, embeddings=None, k=3, chunk_size=300, overlap=50):
        self._embeddings = embeddings or self._default_embeddings()
        self._k = k
        docs = self._load(texts)
        chunks = []
        for d in docs:
            chunks.extend(chunk_text(d, chunk_size, overlap))
        self._store = InMemoryVectorStore.from_texts(chunks, embedding=self._embeddings)

    @staticmethod
    def _default_embeddings():
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    @staticmethod
    def _load(texts) -> list[str]:
        if isinstance(texts, (list, tuple)):
            return list(texts)
        # a directory or file path
        if os.path.isdir(texts):
            out = []
            for name in os.listdir(texts):
                path = os.path.join(texts, name)
                if os.path.isfile(path):
                    with open(path, "r", encoding="utf-8") as f:
                        out.append(f.read())
            return out
        with open(texts, "r", encoding="utf-8") as f:
            return [f.read()]

    def as_tool(self, name: str = "search_documents",
                description: str = "Search the knowledge base for relevant passages."):
        retriever = self._store.as_retriever(search_kwargs={"k": self._k})
        return create_retriever_tool(retriever, name, description)
