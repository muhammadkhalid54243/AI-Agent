# LangGraph Framework — Phase 4 (RAG + MCP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add two opt-in tool sources — a RAG retriever (`.as_tool()`) and an MCP loader (`load_mcp_tools()`) — that produce LangChain tools the `Agent` consumes via `add_tools`.

**Architecture:** `Retriever` chunks texts, embeds them into an `InMemoryVectorStore`, and exposes `.as_tool()` via `create_retriever_tool`. `load_mcp_tools(config)` wraps the async `MultiServerMCPClient.get_tools()` in a sync call. Both return standard LangChain tools — no Agent changes needed.

**Tech Stack:** `langchain_core` vector store + `create_retriever_tool`, embeddings (Google / injectable fake), `langchain-mcp-adapters`, numpy, pytest.

## Global Constraints

- Python `>=3.12`; secrets from env via `.env`.
- Verified API facts (probed live):
  - `InMemoryVectorStore.from_texts(texts, embedding=emb)`; `.as_retriever(search_kwargs={"k": k})`.
  - `create_retriever_tool(retriever, name, description)` (from `langchain_core.tools`) → a tool; `tool.invoke({"query": ...})` returns joined chunk text.
  - Cosine similarity requires `numpy` (added as a dep).
  - `MultiServerMCPClient({name: {"command","args","transport":"stdio"}})`; `await client.get_tools()` → LangChain tools (async).
  - Existing `mcp_server/server.py` exposes `lookup_employee`, `list_department`, `count_employees`.
- Build on Phases 1–3: `Agent.add_tools`, `framework.tools`.

---

## File structure (Phase 4)

- Create `framework/rag/__init__.py` — re-exports `Retriever`.
- Create `framework/rag/chunker.py` — `chunk_text()`.
- Create `framework/rag/retriever.py` — `Retriever`.
- Create `framework/mcp/__init__.py` — re-exports `load_mcp_tools`.
- Create `framework/mcp/loader.py` — `load_mcp_tools(config)`.
- Test: `tests/test_rag.py`, `tests/test_mcp.py`.

---

### Task 1: RAG — chunker + Retriever + .as_tool()

**Files:**
- Create: `framework/rag/chunker.py`, `framework/rag/retriever.py`, `framework/rag/__init__.py`
- Test: `tests/test_rag.py`

**Interfaces:**
- Produces:
  - `chunk_text(text, chunk_size=300, overlap=50) -> list[str]`.
  - `Retriever(texts, *, embeddings=None, k=3, chunk_size=300, overlap=50)` where `texts` is a list of strings or a directory path; `embeddings` defaults to `GoogleGenerativeAIEmbeddings` (needs `GOOGLE_API_KEY`) but is injectable.
  - `Retriever.as_tool(name="search_documents", description=...) -> BaseTool`.

- [ ] **Step 1: Write the failing test (no network — inject fake embeddings)**

`tests/test_rag.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rag.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.rag`.

- [ ] **Step 3: Write the chunker**

`framework/rag/chunker.py`:
```python
"""Split text into overlapping word-window chunks.

Overlap keeps meaning intact at chunk boundaries — a sentence spanning two chunks
is captured whole in at least one.
"""


def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start:start + chunk_size]))
        start += max(1, chunk_size - overlap)
    return chunks
```

- [ ] **Step 4: Write the Retriever**

`framework/rag/retriever.py`:
```python
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
```

`framework/rag/__init__.py`:
```python
from framework.rag.retriever import Retriever

__all__ = ["Retriever"]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_rag.py -q`
Expected: PASS (2 passed).

---

### Task 2: MCP — load_mcp_tools()

**Files:**
- Create: `framework/mcp/loader.py`, `framework/mcp/__init__.py`
- Test: `tests/test_mcp.py`

**Interfaces:**
- Produces: `load_mcp_tools(config: dict) -> list` — sync; `config` is a `MultiServerMCPClient` dict (`{name: {"command","args","transport"}}`). Returns LangChain tools.

- [ ] **Step 1: Write the failing test (uses the local stdio server; no API key)**

`tests/test_mcp.py`:
```python
import sys
from framework.mcp import load_mcp_tools


def test_load_mcp_tools_from_stdio_server():
    tools = load_mcp_tools({
        "directory": {
            "command": sys.executable,
            "args": ["mcp_server/server.py"],
            "transport": "stdio",
        }
    })
    names = [t.name for t in tools]
    assert "lookup_employee" in names
    assert "count_employees" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_mcp.py -q`
Expected: FAIL with `ModuleNotFoundError: framework.mcp`.

- [ ] **Step 3: Write the loader**

`framework/mcp/loader.py`:
```python
"""MCP tool loader — discover tools from MCP servers over the protocol.

The agent doesn't hand-wire these tools; it discovers them at runtime from any
MCP server. get_tools() is async, so this wraps it in a synchronous call for the
framework's sync API. `config` follows MultiServerMCPClient's shape:
    {"name": {"command": "python", "args": ["server.py"], "transport": "stdio"}}
"""

import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient


def load_mcp_tools(config: dict) -> list:
    """Connect to the configured MCP server(s) and return their tools (synchronously)."""
    async def _get():
        client = MultiServerMCPClient(config)
        return await client.get_tools()

    return asyncio.run(_get())
```

`framework/mcp/__init__.py`:
```python
from framework.mcp.loader import load_mcp_tools

__all__ = ["load_mcp_tools"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_mcp.py -q`
Expected: PASS (1 passed) — the loader spawns the stdio server and discovers its tools.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: all Phase 1–4 tests pass.

---

## Self-review

**Spec coverage (Phase 4):** RAG as a retriever tool → Task 1; MCP via `langchain-mcp-adapters`
→ Task 2. Both return standard LangChain tools, so they compose through the existing
`Agent.add_tools` with no Agent changes — consistent with the modular "add as needed" model.

**Placeholder scan:** all code shown; RAG unit test injects `DeterministicFakeEmbedding` (no
network); MCP test uses the local stdio server (no API key) — both deterministic. No TBD/TODO.

**Type consistency:** `chunk_text -> list[str]`, `Retriever(texts, embeddings=, k=)`,
`.as_tool(name, description)`, and `load_mcp_tools(config) -> list` are consistent with the probed
APIs (`InMemoryVectorStore.from_texts`, `create_retriever_tool`, `MultiServerMCPClient.get_tools`).
